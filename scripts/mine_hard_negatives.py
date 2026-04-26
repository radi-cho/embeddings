#!/usr/bin/env python3
"""Hard-negative mining for Stage-2 (Qwen3-VL paper).

SINGLE-PROCESS (cuda:0 only). Safe to run under nohup.
Logs go to FILES via the stdlib logger; the console only gets one
concise line per dataset start/complete.

For each dataset:
  1. Load (query, positive) rows.
  2. Dedup corpus = unique positive texts/images.
  3. Embed corpus on cuda:0 in batches of 64 (OOM fallback: 32, 16, 8, 4).
  4. Embed queries similarly.
  5. faiss.IndexFlatIP on CPU, top-K=50 inner-product search.
  6. Filter: keep candidates where cos(q,c) < cos(q,pos) + 0.1 and c != positive,
     take first K=15; if <15 pass the margin, pad with next-best up to 30.
  7. Write JSONL line-by-line to /data/training_data_mined/<name>.jsonl.
  8. Emit ONE "[mine] ..." summary line.

When all requested datasets are done: touch /tmp/mining_complete.sentinel.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_MODEL = "/data/outputs/qwen35-0.8b-10M-pretrain-lr1e4-a64/merged-15000"
BASE_MODEL = "/home/shared/embeddings/models/checkpoints/Qwen3.5-0.8B"
STAGE1_CKPT = "/data/outputs/qwen35-0.8b-10M-pretrain-lr1e4-a64/checkpoint-15000"
OUT_DIR = Path("/data/training_data_mined")
DATA_DIR = Path("/data/training_data")
IMAGE_DIR = Path("/home/shared/embeddings/datasets/mmeb_train_images/images")
SENTINEL = Path("/tmp/mining_complete.sentinel")

FALSE_NEG_MARGIN = 0.1
TOP_K = 50
NUM_HN = 15
PAD_MAX = 30
IMAGE_PLACEHOLDER = "<|image_1|>"
DEFAULT_INSTRUCTION = "Represent the user's input."

MMEB_RETRIEVAL = ["CIRR", "MSCOCO", "MSCOCO_i2t", "MSCOCO_t2i", "NIGHTS",
                  "VisualNews_i2t", "VisualNews_t2i", "VisDial", "WebQA"]
MMEB_VQA = ["OK-VQA", "A-OKVQA", "DocVQA", "ChartQA", "Visual7W", "InfographicsVQA"]

# File logger only — console stays terse.
LOG_PATH = Path("/tmp/mining_detail.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a")],
)
logger = logging.getLogger("mine")


def _stamp(msg: str) -> None:
    """One concise console line (used for dataset start / complete / summary)."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Sample loaders — each returns list of dicts:
#   {"query": {"text":.., "image_path":..},
#    "positive": {"text":.., "image_path":..},
#    "task_type": .., "subset_name": ..}
# ---------------------------------------------------------------------------

def _clean_mmeb(t: str) -> str:
    return (t or "").replace(IMAGE_PLACEHOLDER, "").strip()


def load_mmeb_subset(name: str, task_type: str) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMEB-train", name,
                      split="diverse_instruction", streaming=False)
    out: List[Dict] = []
    for r in ds:
        q_text = _clean_mmeb(r.get("qry", ""))
        q_img = r.get("qry_image_path") or ""
        p_text = _clean_mmeb(r.get("pos_text", ""))
        p_img = r.get("pos_image_path") or ""
        out.append({
            "query": {"text": q_text or None, "image_path": q_img or None},
            "positive": {"text": p_text or None, "image_path": p_img or None},
            "task_type": task_type, "subset_name": name,
        })
    return out


def load_text_triplet(jsonl_path: Path, subset_name: str,
                      max_samples: Optional[int] = None) -> List[Dict]:
    rows: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            q, p = r.get("query"), r.get("positive")
            if not q or not p:
                continue
            rows.append({
                "query": {"text": q, "image_path": None},
                "positive": {"text": p, "image_path": None},
                "task_type": r.get("task_type", "retrieval"),
                "subset_name": subset_name,
            })
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def load_colpali(data_dir: Path) -> Tuple[List[Dict], Any]:
    from datasets import load_from_disk
    ds = load_from_disk(str(data_dir))
    out: List[Dict] = []
    for i, r in enumerate(ds):
        q = r.get("query")
        if not q:
            continue
        pid = r.get("image_filename") or f"colpali_{i}"
        out.append({
            "query": {"text": q, "image_path": None},
            "positive": {"text": None, "image_path": pid, "_colpali_idx": i},
            "task_type": "retrieval", "subset_name": "ColPali",
            "_colpali_idx": i,
        })
    return out, ds


def positive_key(pos: Dict) -> str:
    img = pos.get("image_path")
    if img:
        return f"IMG::{img}"
    return f"TXT::{(pos.get('text') or '').strip()}"


# ---------------------------------------------------------------------------
# Embedding — single-process, cuda:0
# ---------------------------------------------------------------------------

def _load_pil(path: str, image_dir: Path):
    rel = path[len("images/"):] if path.startswith("images/") else path
    full = image_dir / rel
    if not full.is_file():
        return None
    from PIL import Image
    try:
        return Image.open(full).convert("RGB")
    except Exception:
        return None


def _to_embed_item(side: Dict, image_dir: Path, colpali_ds=None) -> Dict:
    text = side.get("text")
    img = None
    img_path = side.get("image_path")
    if img_path:
        if colpali_ds is not None and isinstance(side.get("_colpali_idx"), int):
            try:
                raw = colpali_ds[side["_colpali_idx"]].get("image")
            except Exception:
                raw = None
            if raw is not None:
                if hasattr(raw, "convert"):
                    img = raw.convert("RGB")
                else:
                    from io import BytesIO
                    from PIL import Image
                    try:
                        img = Image.open(BytesIO(raw)).convert("RGB")
                    except Exception:
                        img = None
        else:
            img = _load_pil(img_path, image_dir)
    item = {"instruction": DEFAULT_INSTRUCTION}
    if text:
        item["text"] = text
    if img is not None:
        item["image"] = img
    if "text" not in item and "image" not in item:
        item["text"] = "NULL"
    return item


def embed_items(embedder, items: List[Dict], image_dir: Path, dim: int,
                colpali_ds=None, batch_size: int = 64,
                tag: str = "") -> np.ndarray:
    n = len(items)
    out = np.zeros((n, dim), dtype=np.float32)
    if n == 0:
        return out
    bs = batch_size
    i = 0
    t0 = time.time()
    last_log = t0
    while i < n:
        j = min(i + bs, n)
        try:
            embed = [_to_embed_item(s, image_dir, colpali_ds) for s in items[i:j]]
            with torch.inference_mode():
                emb = embedder.process(embed, normalize=True)
            out[i:j] = emb.detach().float().cpu().numpy()
            i = j
            now = time.time()
            if now - last_log > 30:
                logger.info("%s embed %d/%d (%.1f%%) bs=%d  elapsed=%.1fmin",
                            tag, i, n, 100 * i / n, bs, (now - t0) / 60)
                last_log = now
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                logger.error("%s OOM at bs=1 for item %d; zeroing row", tag, i)
                out[i] = 0.0
                i += 1
                bs = 4
            else:
                bs = max(1, bs // 2)
                logger.warning("%s OOM, reducing bs -> %d", tag, bs)
        except Exception as e:
            logger.error("%s error batch %d:%d  %s", tag, i, j, e)
            out[i:j] = 0.0
            i = j
    logger.info("%s done %d items in %.1fmin", tag, n, (time.time() - t0) / 60)
    return out


def faiss_topk(corpus_emb: np.ndarray, query_emb: np.ndarray,
               k: int) -> Tuple[np.ndarray, np.ndarray]:
    import faiss
    try:
        faiss.omp_set_num_threads(max(1, (os.cpu_count() or 1) // 2))
    except Exception:
        pass
    d = corpus_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(corpus_emb))
    bs = 4096
    all_s: List[np.ndarray] = []
    all_i: List[np.ndarray] = []
    for s in range(0, len(query_emb), bs):
        sc, ix = index.search(np.ascontiguousarray(query_emb[s:s + bs]), k)
        all_s.append(sc)
        all_i.append(ix)
    return np.concatenate(all_s), np.concatenate(all_i)


# ---------------------------------------------------------------------------
# Mine one dataset
# ---------------------------------------------------------------------------

def mine_dataset(name: str, rows: List[Dict], embedder, image_dir: Path,
                 dim: int, colpali_ds=None, corpus_bs: int = 64,
                 query_bs: int = 64) -> Tuple[Path, int, float]:
    out_path = OUT_DIR / f"{name}.jsonl"
    if out_path.is_file() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            n = sum(1 for _ in f)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        avg = 0.0
        _stamp(f"[mine] {name}: SKIP (already exists) rows={n} file={out_path}")
        return out_path, n, avg

    n = len(rows)
    logger.info("=== %s: %d queries ===", name, n)

    # Dedup corpus = unique positive sides
    corpus_items: List[Dict] = []
    key_to_idx: Dict[str, int] = {}
    q_pos_idx = np.empty(n, dtype=np.int64)
    for i, r in enumerate(rows):
        k = positive_key(r["positive"])
        j = key_to_idx.get(k)
        if j is None:
            j = len(corpus_items)
            key_to_idx[k] = j
            corpus_items.append(r["positive"])
        q_pos_idx[i] = j
    logger.info("%s: %d unique positives", name, len(corpus_items))

    # Embed corpus then queries (corpus is usually smaller, embed first)
    c_emb = embed_items(embedder, corpus_items, image_dir, dim,
                        colpali_ds=colpali_ds, batch_size=corpus_bs,
                        tag=f"{name}.c")
    gc.collect()
    torch.cuda.empty_cache()

    queries = [r["query"] for r in rows]
    q_emb = embed_items(embedder, queries, image_dir, dim,
                        colpali_ds=colpali_ds, batch_size=query_bs,
                        tag=f"{name}.q")
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("%s: FAISS top-%d search", name, TOP_K)
    t0 = time.time()
    scores, idxs = faiss_topk(c_emb, q_emb, TOP_K)
    logger.info("%s: search done in %.1fmin", name, (time.time() - t0) / 60)

    pos_scores = (q_emb * c_emb[q_pos_idx]).sum(-1)

    full_count = 0
    partial_count = 0
    zero_count = 0
    padded_count = 0
    total_hn = 0

    with open(out_path, "w") as fout:
        for i, r in enumerate(rows):
            pi = int(q_pos_idx[i])
            ps = float(pos_scores[i])
            passed: List[int] = []
            extras: List[int] = []
            seen = {pi}
            for rank_k in range(TOP_K):
                ci = int(idxs[i, rank_k])
                if ci < 0 or ci in seen:
                    continue
                seen.add(ci)
                sc = float(scores[i, rank_k])
                if sc > ps + FALSE_NEG_MARGIN:
                    extras.append(ci)
                else:
                    passed.append(ci)
                if len(passed) >= NUM_HN:
                    break

            hn_idx = list(passed)
            if len(hn_idx) < NUM_HN:
                target = min(PAD_MAX, NUM_HN)
                for rank_k in range(TOP_K):
                    if len(hn_idx) >= target:
                        break
                    ci = int(idxs[i, rank_k])
                    if ci < 0 or ci == pi or ci in hn_idx:
                        continue
                    hn_idx.append(ci)
                if hn_idx and len(hn_idx) > len(passed):
                    padded_count += 1

            hns: List[Dict] = []
            for ci in hn_idx[:NUM_HN]:
                hn = corpus_items[ci]
                hns.append({"text": hn.get("text"),
                            "image_path": hn.get("image_path")})

            if len(hns) == NUM_HN:
                full_count += 1
            elif len(hns) > 0:
                partial_count += 1
            else:
                zero_count += 1
            total_hn += len(hns)

            rec = {
                "query": {"text": r["query"].get("text"),
                          "image_path": r["query"].get("image_path")},
                "positive": {"text": r["positive"].get("text"),
                             "image_path": r["positive"].get("image_path")},
                "hard_negatives": hns,
                "task_type": r["task_type"],
                "subset_name": r["subset_name"],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    avg_hn = total_hn / max(n, 1)
    logger.info("%s: wrote %d rows (%.1f MB) full=%d partial=%d zero=%d padded=%d",
                name, n, size_mb, full_count, partial_count, zero_count, padded_count)
    _stamp(f"[mine] {name}: {n} queries, avg_hn={avg_hn:.2f}, "
           f"size={size_mb:.1f}MB, file={out_path}")
    return out_path, n, avg_hn


# ---------------------------------------------------------------------------
# Merge LoRA helper
# ---------------------------------------------------------------------------

def ensure_merged(merged_path: str) -> str:
    if Path(merged_path, "config.json").is_file():
        return merged_path
    logger.info("Merging %s into %s ...", STAGE1_CKPT, merged_path)
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16)
    peft = PeftModel.from_pretrained(base, STAGE1_CKPT)
    merged = peft.merge_and_unload()
    merged.save_pretrained(merged_path)
    tok.save_pretrained(merged_path)
    del merged, peft, base
    gc.collect()
    torch.cuda.empty_cache()
    return merged_path


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=1310720)
    parser.add_argument("--datasets",
                        default="quora,msmarco,gooaq,colpali,mmeb_retrieval,mmeb_vqa")
    parser.add_argument("--gooaq-max", type=int, default=500_000)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(args.datasets.split(","))

    # -- Load embedder ONCE on cuda:0 ------------------------------------
    model_path = ensure_merged(args.model)
    logger.info("Loading embedder from %s on cuda:0", model_path)
    from models.qwen35_embedding import Qwen35Embedder
    embedder = Qwen35Embedder(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        max_pixels=args.max_pixels,
    )
    # Ensure on cuda:0
    if embedder.model is not None:
        embedder.model.to("cuda:0")
        embedder.model.eval()

    summary: List[Tuple[str, int, float, float]] = []  # name, n, avg_hn, size_mb

    def _record(out_path: Path, n: int, avg_hn: float):
        if out_path.is_file():
            mb = out_path.stat().st_size / (1024 * 1024)
            summary.append((out_path.name, n, avg_hn, mb))

    # -- quora -----------------------------------------------------------
    if "quora" in wanted:
        _stamp("[mine] quora: start")
        rows = load_text_triplet(DATA_DIR / "quora" / "train.jsonl", "quora")
        _record(*mine_dataset("quora", rows, embedder, IMAGE_DIR, args.dim,
                              corpus_bs=args.batch_size, query_bs=args.batch_size))
        del rows
        gc.collect()

    # -- msmarco ---------------------------------------------------------
    if "msmarco" in wanted:
        _stamp("[mine] msmarco: start")
        rows = load_text_triplet(DATA_DIR / "msmarco" / "train.jsonl", "msmarco")
        _record(*mine_dataset("msmarco", rows, embedder, IMAGE_DIR, args.dim,
                              corpus_bs=args.batch_size, query_bs=args.batch_size))
        del rows
        gc.collect()

    # -- gooaq -----------------------------------------------------------
    if "gooaq" in wanted:
        _stamp("[mine] gooaq: start")
        rows = load_text_triplet(DATA_DIR / "gooaq" / "train.jsonl", "gooaq",
                                  max_samples=args.gooaq_max)
        _record(*mine_dataset("gooaq", rows, embedder, IMAGE_DIR, args.dim,
                              corpus_bs=args.batch_size, query_bs=args.batch_size))
        del rows
        gc.collect()

    # -- colpali ---------------------------------------------------------
    if "colpali" in wanted:
        _stamp("[mine] colpali: start")
        try:
            rows, ds = load_colpali(DATA_DIR / "colpali" / "data")
            _record(*mine_dataset("colpali", rows, embedder, IMAGE_DIR, args.dim,
                                  colpali_ds=ds,
                                  corpus_bs=max(8, args.batch_size // 4),
                                  query_bs=args.batch_size))
            del rows, ds
            gc.collect()
        except Exception as e:
            logger.error("colpali failed: %s", e)
            _stamp(f"[mine] colpali: FAILED {e}")

    # -- MMEB retrieval --------------------------------------------------
    if "mmeb_retrieval" in wanted:
        for name in MMEB_RETRIEVAL:
            _stamp(f"[mine] mmeb_{name}: start")
            try:
                rows = load_mmeb_subset(name, "retrieval")
            except Exception as e:
                logger.error("Failed loading %s: %s", name, e)
                _stamp(f"[mine] mmeb_{name}: FAILED load {e}")
                continue
            if not rows:
                _stamp(f"[mine] mmeb_{name}: empty, skipping")
                continue
            _record(*mine_dataset(f"mmeb_{name}", rows, embedder, IMAGE_DIR, args.dim,
                                  corpus_bs=max(8, args.batch_size // 4),
                                  query_bs=max(8, args.batch_size // 4)))
            del rows
            gc.collect()

    # -- MMEB VQA --------------------------------------------------------
    if "mmeb_vqa" in wanted:
        for name in MMEB_VQA:
            _stamp(f"[mine] mmeb_{name}: start")
            try:
                rows = load_mmeb_subset(name, "vqa")
            except Exception as e:
                logger.error("Failed loading %s: %s", name, e)
                _stamp(f"[mine] mmeb_{name}: FAILED load {e}")
                continue
            if not rows:
                _stamp(f"[mine] mmeb_{name}: empty, skipping")
                continue
            _record(*mine_dataset(f"mmeb_{name}", rows, embedder, IMAGE_DIR, args.dim,
                                  corpus_bs=max(8, args.batch_size // 4),
                                  query_bs=max(8, args.batch_size // 4)))
            del rows
            gc.collect()

    # -- sentinel + final summary ---------------------------------------
    SENTINEL.write_text("done\n")
    total_rows = 0
    total_mb = 0.0
    _stamp("================ MINING SUMMARY ================")
    for fname, n, avg_hn, mb in summary:
        total_rows += n
        total_mb += mb
        _stamp(f"  {fname:40s}  rows={n:>8d}  avg_hn={avg_hn:5.2f}  size={mb:8.1f}MB")
    _stamp(f"  TOTAL: rows={total_rows}  size={total_mb:.1f}MB")
    _stamp("================================================")


if __name__ == "__main__":
    main()
