#!/usr/bin/env python3
"""
Download ~10M training examples from multiple sources for mixed multimodal
embedding training, plus optional MMEB training images.

Text-only sources  -> JSONL files
Multimodal sources -> HF Arrow datasets (save_to_disk) or JSONL
Video sources      -> JSONL manifests (frames downloaded separately)
MMEB images        -> Extracted per-subset image directories

All output goes to --output_dir (default /data/training_data/).
Each source gets its own subdirectory with a manifest.json.

Uses HF_TOKEN from .hf_token_local for gated datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_hf_token_path = PROJECT_ROOT / ".hf_token_local"
HF_TOKEN: str | None = None
if _hf_token_path.is_file():
    tok = _hf_token_path.read_text().strip()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        HF_TOKEN = tok

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def write_jsonl(path: Path, samples: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Wrote %d samples to %s", len(samples), path)


def write_manifest(out_dir: Path, name: str, count: int, task_type: str,
                   data_format: str, elapsed: float, **extra):
    m = {"name": name, "count": count, "task_type": task_type,
         "format": data_format, "elapsed_s": round(elapsed, 1), **extra}
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=2))


@contextmanager
def source_dir(out_dir: Path, name: str):
    """Yield (subdir, should_skip). Caller should return early when skip is True."""
    d = out_dir / name
    if (d / "manifest.json").is_file():
        logger.info("[%s] SKIP (manifest exists)", name)
        yield d, True
    else:
        d.mkdir(parents=True, exist_ok=True)
        yield d, False


def stream_to_jsonl(name: str, hf_id: str, split: str, out_dir: Path,
                    row_fn, max_samples: int, task_type: str = "retrieval",
                    hf_subset: str | None = None, streaming: bool = True,
                    log_every: int = 500_000):
    """Stream an HF dataset, transform rows via row_fn, write JSONL + manifest."""
    with source_dir(out_dir, name) as (d, skip):
        if skip:
            return

    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    logger.info("[%s] Streaming up to %d samples...", name, max_samples)
    kw: dict = dict(split=split, streaming=streaming)
    if hf_subset:
        kw["name"] = hf_subset
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    ds = load_dataset(hf_id, **kw)

    t0 = time.time()
    samples = []
    for row in ds:
        rec = row_fn(row)
        if rec is not None:
            samples.append(rec)
        if len(samples) >= max_samples:
            break
        if log_every and len(samples) % log_every == 0 and len(samples) > 0:
            logger.info("[%s] %d / %d ...", name, len(samples), max_samples)
    write_jsonl(d / "train.jsonl", samples)
    write_manifest(d, name, len(samples), task_type, "jsonl", time.time() - t0)


# ---- Text-only sources ----

def download_msmarco(out_dir: Path, max_samples: int = 3_000_000):
    stream_to_jsonl(
        "msmarco",
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "train", out_dir,
        row_fn=lambda r: {"query": r["query"], "positive": r["positive"],
                          "negative": r["negative"], "task_type": "retrieval"},
        max_samples=max_samples,
        hf_subset="triplet",
    )


def download_allnli(out_dir: Path, max_samples: int = 300_000):
    stream_to_jsonl(
        "allnli", "sentence-transformers/all-nli", "train", out_dir,
        row_fn=lambda r: {"query": r["anchor"], "positive": r["positive"],
                          "negative": r["negative"], "task_type": "retrieval"},
        max_samples=max_samples,
        hf_subset="triplet",
    )


def download_gooaq(out_dir: Path, max_samples: int = 3_000_000):
    stream_to_jsonl(
        "gooaq", "sentence-transformers/gooaq", "train", out_dir,
        row_fn=lambda r: {"query": r["question"], "positive": r["answer"],
                          "negative": None, "task_type": "retrieval"},
        max_samples=max_samples,
    )


def download_quora(out_dir: Path, max_samples: int = 500_000):
    stream_to_jsonl(
        "quora", "sentence-transformers/quora-duplicates", "train", out_dir,
        row_fn=lambda r: {"query": r["anchor"], "positive": r["positive"],
                          "negative": r.get("negative"), "task_type": "retrieval"},
        max_samples=max_samples,
        hf_subset="triplet",
        log_every=100_000,
    )


def download_stsb(out_dir: Path):
    stream_to_jsonl(
        "stsb", "sentence-transformers/stsb", "train", out_dir,
        row_fn=lambda r: {"sentence1": r["sentence1"], "sentence2": r["sentence2"],
                          "score": float(r["score"]), "task_type": "sts"},
        max_samples=999_999_999,
        task_type="sts",
        streaming=False,
        log_every=0,
    )


# ---- Multimodal image sources ----

def download_megapairs(out_dir: Path, max_samples: int = 5_000_000):
    """Download MegaPairs annotations JSONL (text metadata only, no images)."""
    with source_dir(out_dir, "megapairs") as (d, skip):
        if skip:
            return

    d = out_dir / "megapairs"
    d.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    logger.info("[megapairs] Downloading annotation JSONL (gated, needs HF_TOKEN)...")
    t0 = time.time()

    local_path = hf_hub_download(
        repo_id="JUNJIE99/MegaPairs",
        filename="annotation/megapairs.jsonl",
        repo_type="dataset",
        cache_dir=str(d / "_hf_cache"),
        token=HF_TOKEN,
    )
    logger.info("[megapairs] JSONL downloaded to %s, sampling %d rows...", local_path, max_samples)

    count = 0
    out_file = d / "train.jsonl"
    with open(local_path) as fin, open(out_file, "w") as fout:
        for line in fin:
            fout.write(line)
            count += 1
            if count >= max_samples:
                break
            if count % 500_000 == 0:
                logger.info("[megapairs] %d / %d ...", count, max_samples)

    write_manifest(d, "megapairs", count, "retrieval", "jsonl",
                   time.time() - t0,
                   note="Images not included. Download images tar separately for visual training.")


def download_colpali(out_dir: Path):
    """Download ColPali train set (query + document page image) as Arrow shards."""
    with source_dir(out_dir, "colpali") as (d, skip):
        if skip:
            return

    d = out_dir / "colpali"
    d.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    logger.info("[colpali] Downloading full train set (127k, ~50GB images)...")
    t0 = time.time()
    retries = 3
    for attempt in range(retries):
        try:
            ds = load_dataset("vidore/colpali_train_set", split="train",
                             token=HF_TOKEN)
            ds.save_to_disk(str(d / "data"))
            write_manifest(d, "colpali", len(ds), "retrieval", "arrow",
                           time.time() - t0)
            logger.info("[colpali] Done: %d examples in %.0fs", len(ds), time.time() - t0)
            return
        except Exception as e:
            logger.warning("[colpali] attempt %d/%d failed: %s", attempt + 1, retries, e)
            if attempt == retries - 1:
                raise


# ---- Video sources ----

def download_llava_hound(out_dir: Path, max_samples: int = 300_000):
    """Download LLaVA-Hound video QA data as JSONL manifest."""
    def _row_fn(row):
        convs = row.get("conversations", [])
        question = answer = ""
        for turn in convs:
            if turn.get("from") == "human":
                question = turn.get("value", "")
            elif turn.get("from") == "gpt":
                answer = turn.get("value", "")
        video_path = row.get("video", "")
        if question and answer and video_path:
            return {"query_text": question, "positive_text": answer,
                    "video_path": video_path, "task_type": "vqa"}
        return None

    stream_to_jsonl(
        "llava_hound", "lmms-lab/LLaVA-Video-178K", "open_ended", out_dir,
        row_fn=_row_fn,
        max_samples=max_samples,
        task_type="vqa",
        hf_subset="llava_hound",
        log_every=50_000,
    )


def download_video_retrieval(out_dir: Path):
    """Download MSR-VTT + VATEX train splits as JSONL."""
    with source_dir(out_dir, "video_retrieval") as (d, skip):
        if skip:
            return

    d = out_dir / "video_retrieval"
    d.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    t0 = time.time()
    total = 0

    for name, hf_id, split, q_col, vid_col in [
        ("msrvtt", "AlexZigma/msr-vtt", "train", "caption", "video_id"),
        ("vatex", "lmms-lab/VATEX", "train", "enCap", "videoID"),
    ]:
        logger.info("[video_retrieval] Loading %s ...", name)
        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
            samples = []
            for row in ds:
                q = row.get(q_col, "")
                if isinstance(q, list):
                    q = q[0] if q else ""
                vid = row.get(vid_col, "")
                if q and vid:
                    samples.append({
                        "query_text": q, "positive_text": None,
                        "video_id": vid, "source": name, "task_type": "retrieval",
                    })
                if len(samples) >= 50_000:
                    break
            write_jsonl(d / f"{name}.jsonl", samples)
            total += len(samples)
        except Exception as e:
            logger.warning("[video_retrieval] Failed to load %s: %s", name, e)

    write_manifest(d, "video_retrieval", total, "retrieval", "jsonl_video",
                   time.time() - t0,
                   note="Video frames must be downloaded separately")


# ---- MMEB training images ----

MMEB_REPO_ID = "TIGER-Lab/MMEB-train"
MMEB_ALL_SUBSETS = [
    "A-OKVQA", "CIRR", "ChartQA", "DocVQA", "HatefulMemes",
    "ImageNet_1K", "InfographicsVQA", "MSCOCO", "MSCOCO_i2t",
    "MSCOCO_t2i", "N24News", "NIGHTS", "OK-VQA", "SUN397",
    "VOC2007", "VisDial", "Visual7W", "VisualNews_i2t",
    "VisualNews_t2i", "WebQA",
]


def download_mmeb_images(out_dir: Path, subsets: list[str] | None = None,
                         cache_dir: str | None = None):
    """Download and extract MMEB training images from TIGER-Lab/MMEB-train.

    Extracts to: {out_dir}/mmeb_images/images/{subset}/Train/*.jpg
    """
    from huggingface_hub import hf_hub_download

    subsets = subsets or MMEB_ALL_SUBSETS
    images_dir = out_dir / "mmeb_images" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for i, subset in enumerate(subsets):
        subset_dir = images_dir / subset / "Train"
        if subset_dir.is_dir() and any(subset_dir.iterdir()):
            logger.info("[mmeb %d/%d] %s -- already extracted, skipping",
                        i + 1, len(subsets), subset)
            continue

        zip_path_in_repo = f"images_zip/{subset}.zip"
        logger.info("[mmeb %d/%d] Downloading %s.zip ...", i + 1, len(subsets), subset)
        try:
            local_zip = hf_hub_download(
                MMEB_REPO_ID, zip_path_in_repo,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        except Exception as e:
            logger.warning("[mmeb] FAILED to download %s: %s", subset, e)
            continue

        logger.info("[mmeb] Extracting to %s/ ...", images_dir)
        with zipfile.ZipFile(local_zip) as zf:
            zf.extractall(images_dir)

        if subset_dir.is_dir():
            n = len(list(subset_dir.iterdir()))
            logger.info("[mmeb] %s -- %d images", subset, n)
        else:
            logger.warning("[mmeb] expected %s but not found after extraction", subset_dir)

    logger.info("[mmeb] Done. Use --image_dir %s when training.", images_dir)


# ---- Main ----

def main():
    p = argparse.ArgumentParser(description="Download ~10M mixed training data")
    p.add_argument("--output_dir", default="/data/training_data")
    p.add_argument("--msmarco_max", type=int, default=3_000_000)
    p.add_argument("--gooaq_max", type=int, default=3_000_000)
    p.add_argument("--megapairs_max", type=int, default=5_000_000)
    p.add_argument("--llava_hound_max", type=int, default=300_000)
    p.add_argument("--skip", nargs="*", default=[],
                   help="Source names to skip (e.g. megapairs colpali mmeb_images)")
    p.add_argument("--mmeb_subsets", type=str, default=None,
                   help="Comma-separated MMEB subset names (default: all)")
    p.add_argument("--mmeb_cache_dir", type=str, default=None,
                   help="HF download cache for MMEB image zips")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    skip = set(s.lower() for s in args.skip)
    t0 = time.time()

    mmeb_subsets = args.mmeb_subsets.split(",") if args.mmeb_subsets else None

    sources = [
        ("msmarco", lambda: download_msmarco(out, args.msmarco_max)),
        ("allnli", lambda: download_allnli(out)),
        ("gooaq", lambda: download_gooaq(out, args.gooaq_max)),
        ("quora", lambda: download_quora(out)),
        ("stsb", lambda: download_stsb(out)),
        ("megapairs", lambda: download_megapairs(out, args.megapairs_max)),
        ("colpali", lambda: download_colpali(out)),
        ("llava_hound", lambda: download_llava_hound(out, args.llava_hound_max)),
        ("video_retrieval", lambda: download_video_retrieval(out)),
        ("mmeb_images", lambda: download_mmeb_images(out, mmeb_subsets, args.mmeb_cache_dir)),
    ]

    for name, fn in sources:
        if name in skip:
            logger.info("SKIPPING %s (--skip)", name)
            continue
        try:
            fn()
        except Exception as e:
            logger.error("FAILED %s: %s", name, e, exc_info=True)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All downloads done in %.1f minutes.", elapsed / 60)
    logger.info("Output: %s", out)
    for sub in sorted(out.iterdir()):
        mf = sub / "manifest.json"
        if mf.is_file():
            m = json.loads(mf.read_text())
            logger.info("  %s: %s examples (%s)", m["name"], m["count"], m["format"])


if __name__ == "__main__":
    main()
