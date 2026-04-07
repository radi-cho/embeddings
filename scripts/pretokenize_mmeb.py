#!/usr/bin/env python3
"""
Parallel offline pre-tokenization for MMEB-train.

Memory-safe design:
- pixel_values stored as bfloat16 (halves blob size: 7MB → 3.5MB per image)
- Flushes to disk every 200 rows (~0.4GB buffer per worker)
- Uses multiprocessing 'spawn' with strict single-thread pinning
- gc.collect() + explicit del after every shard flush
- 8 workers on 24-CPU/167GB: ~12GB total, leaves 150GB+ free
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_MAX_THREADS"):
    os.environ[v] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_hf_token_path = Path(__file__).resolve().parents[1] / ".hf_token_local"
if _hf_token_path.is_file():
    _tok = _hf_token_path.read_text().strip()
    if _tok:
        os.environ.setdefault("HF_TOKEN", _tok)

import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.dataset import ALL_MMEB_SUBSETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

SHARD_SIZE = 200


def _encode_role(embedder, role: dict) -> bytes:
    """Preprocess one role → serialized bytes. pixel_values stored as bf16."""
    conv = embedder.format_model_input(
        text=role.get("text"), image=role.get("image"),
        video=role.get("video"), instruction=role.get("instruction"))
    try:
        proc = embedder._preprocess_inputs([conv])
    except Exception:
        conv = embedder.format_model_input(
            text=role.get("text") or "NULL", instruction=role.get("instruction"))
        proc = embedder._preprocess_inputs([conv])

    cpu = {}
    for k, v in proc.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                cpu[k] = v.detach().to(dtype=torch.bfloat16).cpu()
            else:
                cpu[k] = v.detach().cpu()
        else:
            cpu[k] = v
    buf = io.BytesIO()
    torch.save(cpu, buf)
    return buf.getvalue()


def _flush_shard(q, p, n, tt, sn, shard_dir: Path, shard_idx: int):
    from datasets import Dataset, Features, Value
    features = Features({
        "q_bytes": Value("binary"), "p_bytes": Value("binary"),
        "n_bytes": Value("binary"), "task_type": Value("string"),
        "subset_name": Value("string"),
    })
    ds = Dataset.from_dict({
        "q_bytes": q, "p_bytes": p, "n_bytes": n,
        "task_type": tt, "subset_name": sn,
    }, features=features)
    dest = shard_dir / f"shard_{shard_idx:04d}"
    ds.save_to_disk(str(dest))
    del ds
    q.clear(); p.clear(); n.clear(); tt.clear(); sn.clear()
    gc.collect()


def _write_status(status_dir: str, subset_name: str, done: int, total: int,
                  rate: float, elapsed: float, finished: bool = False):
    Path(status_dir).mkdir(parents=True, exist_ok=True)
    (Path(status_dir) / f"{subset_name}.json").write_text(json.dumps({
        "subset": subset_name, "done": done, "total": total,
        "pct": round(100.0 * done / max(total, 1), 1),
        "rate": round(rate, 1), "elapsed_s": round(elapsed, 1),
        "finished": finished, "pid": os.getpid(),
    }))


def pretokenize_one_subset(args_tuple) -> str:
    (subset_name, model_path, max_length, max_pixels, split,
     image_dir, output_root_str, max_samples, cache_dir, resume,
     status_dir_str, shard_size) = args_tuple

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_MAX_THREADS"):
        os.environ[v] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch as _torch
    _torch.set_num_threads(1)

    output_root = Path(output_root_str)
    out_dir = output_root / subset_name

    if resume and out_dir.is_dir() and (out_dir / "dataset_info.json").is_file():
        logger.info("[%s] SKIP (already on disk)", subset_name)
        return f"{subset_name}: skipped"

    from src.models.qwen35_embedding import Qwen35Embedder
    from src.train.dataset import MMEBDataset, DEFAULT_EMBED_INSTRUCTION

    embedder = Qwen35Embedder(
        model_name_or_path=model_path, max_length=max_length,
        max_pixels=max_pixels, load_model=False,
    )
    ds = MMEBDataset(
        subset_name=subset_name, split=split,
        image_dir=image_dir, max_samples=max_samples, cache_dir=cache_dir,
    )
    n = len(ds)
    t0 = time.time()
    inst = DEFAULT_EMBED_INSTRUCTION

    shard_dir = output_root / f"_shards_{subset_name}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    q_buf, p_buf, n_buf, tt_buf, sn_buf = [], [], [], [], []
    shard_idx = 0

    for idx in range(n):
        item = ds[idx]
        q_role = {**item["query"], "instruction": inst}
        p_role = {**item["positive"], "instruction": inst}
        neg = item["negative"]
        n_role = ({**neg, "instruction": inst} if neg is not None
                  else {"text": item["positive"].get("text") or "NULL",
                        "image": None, "instruction": inst})

        q_buf.append(_encode_role(embedder, q_role))
        p_buf.append(_encode_role(embedder, p_role))
        n_buf.append(_encode_role(embedder, n_role))
        tt_buf.append(item["task_type"])
        sn_buf.append(subset_name)

        del item, q_role, p_role, n_role, neg

        if len(q_buf) >= shard_size:
            _flush_shard(q_buf, p_buf, n_buf, tt_buf, sn_buf, shard_dir, shard_idx)
            shard_idx += 1

        done = idx + 1
        if done % 200 == 0 or done == n:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta_m = (n - done) / max(rate, 1e-6) / 60.0
            logger.info("[%s] %d/%d (%.1f%%) | %.1f ex/s | ETA %.0fm",
                        subset_name, done, n, 100.0 * done / n, rate, eta_m)
            _write_status(status_dir_str, subset_name, done, n, rate, elapsed)
            sys.stdout.flush(); sys.stderr.flush()

    if q_buf:
        _flush_shard(q_buf, p_buf, n_buf, tt_buf, sn_buf, shard_dir, shard_idx)

    from datasets import concatenate_datasets, load_from_disk
    shard_paths = sorted(shard_dir.glob("shard_*"))
    if shard_paths:
        shards = [load_from_disk(str(sp)) for sp in shard_paths]
        combined = concatenate_datasets(shards)
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.save_to_disk(str(out_dir))
        del shards, combined
        gc.collect()
    shutil.rmtree(shard_dir, ignore_errors=True)

    elapsed = time.time() - t0
    with open(out_dir / "pretokenize_meta.json", "w") as f:
        json.dump({"subset_name": subset_name, "num_examples": n,
                    "split": split, "seconds": elapsed}, f, indent=2)

    _write_status(status_dir_str, subset_name, n, n,
                  n / max(elapsed, 1e-6), elapsed, finished=True)
    msg = f"{subset_name}: {n} rows in {elapsed:.0f}s ({n/max(elapsed,1e-6):.1f} ex/s)"
    logger.info("[%s] DONE — %s", subset_name, msg)
    return msg


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_path", default=str(PROJECT_ROOT / "models/Qwen3.5-0.8B"))
    pa.add_argument("--output_dir", default=str(PROJECT_ROOT / "datasets/mmeb_pretokenized"))
    pa.add_argument("--split", default="diverse_instruction")
    pa.add_argument("--image_dir", default=None)
    pa.add_argument("--max_length", type=int, default=512)
    pa.add_argument("--max_pixels", type=int, default=401408)
    pa.add_argument("--cache_dir", default=None)
    pa.add_argument("--subsets", default=None)
    pa.add_argument("--max_samples_per_subset", type=int, default=None)
    pa.add_argument("--resume", action="store_true")
    pa.add_argument("--workers", type=int, default=20)
    pa.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    args = pa.parse_args()

    output_root = Path(args.output_dir).resolve()
    status_dir = output_root / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    subsets = ([s.strip() for s in args.subsets.split(",") if s.strip()]
               if args.subsets else list(ALL_MMEB_SUBSETS))
    subsets = [s for s in subsets if s in ALL_MMEB_SUBSETS]

    w = min(args.workers, len(subsets))
    logger.info("Launching %d spawn-workers for %d subsets "
                "(shard_size=%d, bf16 pixel_values, single-threaded)",
                w, len(subsets), args.shard_size)

    task_args = [
        (name, args.model_path, args.max_length, args.max_pixels, args.split,
         args.image_dir, str(output_root), args.max_samples_per_subset,
         args.cache_dir, args.resume, str(status_dir), args.shard_size)
        for name in subsets
    ]

    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(processes=w) as pool:
        results = pool.map(pretokenize_one_subset, task_args)

    elapsed = time.time() - t0
    total_rows = 0
    for r in results:
        if "rows" in r:
            total_rows += int(r.split(":")[1].strip().split()[0])
    logger.info("=" * 60)
    logger.info("All done in %.1f minutes. %d rows at %.0f aggregate ex/s.",
                elapsed / 60.0, total_rows, total_rows / max(elapsed, 1))
    for r in results:
        logger.info("  %s", r)
    logger.info("Output: %s", output_root)


if __name__ == "__main__":
    main()
