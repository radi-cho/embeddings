#!/usr/bin/env python3
"""
MMEB (Multimodal Embedding Benchmark) evaluation.

MMEB-V2 (paper / leaderboard) uses the same Hugging Face *instruction* configs as VLM2Vec
(`ziyjiang/MMEB_Test_Instruct`, one config per task, split `test`) while hosting frozen
media under `TIGER-Lab/MMEB-V2` (`image-tasks/mmeb_v1.tar.gz`). This script follows that split:
annotations from the instruct dataset, pixels from the MMEB-V2 image release.

Paper Sec. 6.1 settings mirrored here: context length 16,384 tokens; image-side budget
matches `Qwen3-VL-Embedding`’s `scripts/qwen3_vl_embedding.py` (1800 vision tokens →
`max_pixels = 1800 * 32 * 32`, not 28×28).

Usage:
    # First: download MMEB-V2 image tarball (one-time, ~7 GB)
    python src/eval/run_mmeb.py --download_images --cache_dir datasets/mmeb_cache

    # Quick eval: 4 tasks, one per category (~15 min)
    python src/eval/run_mmeb.py --model_path models/Qwen3-VL-Embedding-2B \
        --quick

    # Specific tasks
    python src/eval/run_mmeb.py --model_path models/Qwen3-VL-Embedding-2B \
        --tasks N24News OK-VQA

    # All 36 image tasks
    python src/eval/run_mmeb.py --model_path models/Qwen3-VL-Embedding-2B \
        --full

    # List available tasks
    python src/eval/run_mmeb.py --list_tasks
"""

import argparse
import json
import os
import sys
import logging
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def attach_run_log(output_dir: Path) -> None:
    """Append MMTEB-style run.log under output_dir (full timestamp lines)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path.resolve())

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test queries, candidates, instructions (per-task configs). Same HF id as VLM2Vec MMEB-V2 eval.
MMEB_TEST_INSTRUCTIONS = "ziyjiang/MMEB_Test_Instruct"
# Official MMEB-V2 image release (parquet-free; media only).
MMEB_V2_MEDIA_REPO = "TIGER-Lab/MMEB-V2"
MMEB_V2_IMAGE_TAR = "image-tasks/mmeb_v1.tar.gz"
# Qwen3-VL-Embedding vision cap: 1800 tokens with IMAGE_FACTOR=32 in upstream script.
MMEB_IMAGE_MIN_PIXELS = 4 * 32 * 32
MMEB_IMAGE_MAX_PIXELS = 1800 * 32 * 32

TASK_CATEGORIES = {
    "classification": [
        "ImageNet-1K", "ImageNet-A", "ImageNet-R", "ObjectNet",
        "Country211", "SUN397", "Place365",
        "VOC2007", "N24News", "HatefulMemes",
    ],
    "vqa": [
        "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA",
        "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA",
    ],
    "retrieval": [
        "MSCOCO_i2t", "MSCOCO_t2i", "VisualNews_i2t", "VisualNews_t2i",
        "VisDial", "CIRR", "NIGHTS", "WebQA", "FashionIQ",
        "Wiki-SS-NQ", "OVEN", "EDIS",
    ],
    "grounding": [
        "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing",
    ],
}

QUICK_TASKS = ["N24News", "OK-VQA", "MSCOCO_i2t", "RefCOCO"]
ALL_TASKS = [t for tasks in TASK_CATEGORIES.values() for t in tasks]


def get_category(task_name):
    for cat, tasks in TASK_CATEGORIES.items():
        if task_name in tasks:
            return cat
    return "unknown"


def strip_image_placeholder(text):
    """Remove VLM2Vec <|image_1|> tokens from MMEB text fields."""
    if not text:
        return ""
    return text.replace("<|image_1|>\n", "").replace("<|image_1|>", "").strip()


def resolve_image(img_path, image_dir):
    """Resolve a relative MMEB image path to an absolute path, or None."""
    if not img_path:
        return None
    full = os.path.join(image_dir, img_path)
    if os.path.exists(full):
        return full
    return None


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def download_images(cache_dir):
    """Download and extract MMEB-V2 image tarball from TIGER-Lab/MMEB-V2 (mmeb_v1)."""
    from huggingface_hub import hf_hub_download
    import tarfile

    cache_dir = Path(cache_dir)
    extract_root = cache_dir / "mmeb_v2_image_tasks"

    def _resolved_image_root(base: Path) -> Optional[Path]:
        """HF tarball currently unpacks to `MMEB/<task>/...`; older docs mention `mmeb_v1/`."""
        for name in ("mmeb_v1", "MMEB"):
            p = base / name
            if p.is_dir() and any(p.iterdir()):
                return p
        return None

    existing = _resolved_image_root(extract_root)
    if existing is not None:
        n = len([d for d in existing.iterdir() if d.is_dir()])
        logger.info("MMEB-V2 images already extracted at %s (%d folders)", existing, n)
        return existing

    logger.info("Downloading MMEB-V2 image-tasks/mmeb_v1.tar.gz (~7 GB) from %s ...", MMEB_V2_MEDIA_REPO)
    tar_path = hf_hub_download(
        repo_id=MMEB_V2_MEDIA_REPO,
        filename=MMEB_V2_IMAGE_TAR,
        repo_type="dataset",
        cache_dir=str(cache_dir / "hf_download"),
    )

    logger.info("Extracting to %s ...", extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(extract_root)

    resolved = _resolved_image_root(extract_root)
    if resolved is not None:
        logger.info("Done. Image root: %s", resolved)
        return resolved

    subdirs = [d for d in extract_root.iterdir() if d.is_dir()]
    if subdirs:
        logger.info("Done (using extract root as image dir).")
        return extract_root

    raise RuntimeError(f"Unexpected archive layout after extracting {tar_path}; expected MMEB/ or mmeb_v1/")


def find_image_dir(cache_dir):
    """Locate the extracted image directory (MMEB-V2 mmeb_v1 or legacy MMEB-v1 zip layout)."""
    cache_dir = Path(cache_dir)
    for candidate in [
        cache_dir / "mmeb_v2_image_tasks" / "mmeb_v1",
        cache_dir / "mmeb_v2_image_tasks" / "MMEB",
        cache_dir / "mmeb_v2_image_tasks",
        cache_dir / "images" / "images",
        cache_dir / "images",
        cache_dir,
    ]:
        if candidate.exists() and candidate.is_dir():
            for task in ["N24News", "OK-VQA", "MSCOCO_i2t", "ImageNet-1K"]:
                if (candidate / task).exists():
                    return candidate
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path,
    *,
    max_length: int = 16384,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
):
    """Load embedding model, auto-detecting Qwen3-VL vs Qwen3.5.

    `max_length` defaults to 16,384 per Qwen3-VL paper Sec. 6.1. Vision caps are enforced on
    Qwen3-VL via `min_pixels` / `max_pixels` (default matches upstream Qwen3-VL-Embedding).
    """
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        with open(config_path) as f:
            model_type = json.load(f).get("model_type", "")

    mip = MMEB_IMAGE_MIN_PIXELS if image_min_pixels is None else image_min_pixels
    mp = MMEB_IMAGE_MAX_PIXELS if image_max_pixels is None else image_max_pixels

    if "qwen3_vl" in model_type:
        scripts_dir = Path(model_path) / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        from qwen3_vl_embedding import Qwen3VLEmbedder

        class _MMEBQwen3VLEmbedder(Qwen3VLEmbedder):
            """Applies optional per-item min/max_pixels from MMEB `make_item` (upstream ignores these keys)."""

            def process(self, inputs, normalize=True):
                conversations = []
                for ele in inputs:
                    conv = self.format_model_input(
                        text=ele.get("text"),
                        image=ele.get("image"),
                        video=ele.get("video"),
                        instruction=ele.get("instruction"),
                        fps=ele.get("fps"),
                        max_frames=ele.get("max_frames"),
                    )
                    for msg in conv:
                        for part in msg.get("content", []):
                            if isinstance(part, dict) and part.get("type") == "image":
                                if ele.get("max_pixels") is not None:
                                    part["max_pixels"] = ele["max_pixels"]
                                if ele.get("min_pixels") is not None:
                                    part["min_pixels"] = ele["min_pixels"]
                    conversations.append(conv)

                processed_inputs = self._preprocess_inputs(conversations)
                processed_inputs = {k: v.to(self.model.device) for k, v in processed_inputs.items()}
                outputs = self.forward(processed_inputs)
                embeddings = self._pooling_last(
                    outputs["last_hidden_state"], outputs["attention_mask"]
                )
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=-1)
                return embeddings

        logger.info(
            "Loading Qwen3-VL from %s (max_length=%s, image max_pixels=%s)",
            model_path,
            max_length,
            mp,
        )
        model = _MMEBQwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            max_length=max_length,
            min_pixels=mip,
            max_pixels=mp,
        )
        return model, "qwen3vl"
    else:
        from src.models.qwen35_embedding import Qwen35Embedder

        logger.info(
            "Loading Qwen3.5 from %s (max_length=%s, image max_pixels=%s)",
            model_path,
            max_length,
            mp,
        )
        model = Qwen35Embedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            max_length=max_length,
            min_pixels=mip,
            max_pixels=mp,
        )
        return model, "qwen35"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_batch(model, items, batch_size=32):
    """Embed a list of dicts (text/image/instruction) in batches.

    On CUDA OOM, halves the micro-batch down to 1 (same strategy as run_mmteb.py).
    """
    all_embs = []
    i, n = 0, len(items)
    while i < n:
        chunk = min(batch_size, n - i)
        while chunk >= 1:
            batch = items[i : i + chunk]
            try:
                with torch.no_grad():
                    embs = model.process(batch)
                all_embs.append(embs.cpu().float())
                i += chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                if chunk <= 1:
                    raise
                chunk = max(1, chunk // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(
                    "CUDA OOM during MMEB embed; retrying with chunk size %d (start index %d)",
                    chunk,
                    i,
                )
    return torch.cat(all_embs, dim=0)


def make_item(
    text,
    img_path,
    image_dir,
    instruction=None,
    *,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
):
    """Build a dict suitable for model.process() from MMEB fields.

    Uses the separated instruction field from the MMEB test instruct dataset.
    When an image path is present, attaches `min_pixels` / `max_pixels` for Qwen3-VL
    so vision resolution matches paper / upstream embedding code (dynamic-resize cap).
    """
    item = {}
    clean_text = strip_image_placeholder(text) if text else ""
    clean_inst = strip_image_placeholder(instruction) if instruction else ""
    img = resolve_image(img_path, image_dir)
    if img:
        item["image"] = img
        item["min_pixels"] = (
            MMEB_IMAGE_MIN_PIXELS if image_min_pixels is None else image_min_pixels
        )
        item["max_pixels"] = (
            MMEB_IMAGE_MAX_PIXELS if image_max_pixels is None else image_max_pixels
        )
    if clean_text:
        item["text"] = clean_text
    if clean_inst:
        item["instruction"] = clean_inst
    return item if item else {"text": ""}


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def evaluate_task(
    model,
    task_name,
    image_dir,
    batch_size,
    *,
    test_instructions_repo: str,
    image_min_pixels: Optional[int],
    image_max_pixels: Optional[int],
):
    """Load one MMEB task and evaluate.

    Each example has its own candidate set with the correct answer at index 0.
    Pre-embeds all queries and candidates in batched passes, then scores on CPU.
    """
    from datasets import load_dataset

    logger.info(f"--- {task_name} ---")
    ds = load_dataset(test_instructions_repo, task_name, split="test")
    n = len(ds)
    n_cands = len(ds[0]["tgt_text"])
    category = get_category(task_name)

    logger.info(f"  category={category}  examples={n}  candidates={n_cands}")

    # Build all query items
    queries = [
        make_item(
            ex["qry_text"],
            ex["qry_img_path"],
            image_dir,
            instruction=ex.get("qry_inst"),
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )
        for ex in ds
    ]

    # Deduplicate candidates: build a unique set keyed by (text, img_path).
    # Official Qwen3-VL-Embedding eval does NOT pass tgt_inst to candidates;
    # candidates fall back to the embedder's default instruction.
    unique_cands = {}  # key -> index in unique list
    unique_cand_items = []
    cand_indices = []  # [n, n_cands]
    for ex in ds:
        ex_indices = []
        for t, p in zip(ex["tgt_text"], ex["tgt_img_path"]):
            key = (t, p)
            if key not in unique_cands:
                unique_cands[key] = len(unique_cand_items)
                unique_cand_items.append(
                    make_item(
                        t,
                        p,
                        image_dir,
                        instruction=None,
                        image_min_pixels=image_min_pixels,
                        image_max_pixels=image_max_pixels,
                    )
                )
            ex_indices.append(unique_cands[key])
        cand_indices.append(ex_indices)

    # Batched embedding passes
    logger.info(f"  Embedding {n} queries ...")
    qry_embs = embed_batch(model, queries, batch_size)

    logger.info(f"  Embedding {len(unique_cand_items)} unique candidates (from {n * n_cands} total) ...")
    unique_cand_embs = embed_batch(model, unique_cand_items, batch_size)

    # Gather per-example candidate embeddings via index lookup
    idx_tensor = torch.tensor(cand_indices, dtype=torch.long)  # [n, n_cands]
    all_cand_embs = unique_cand_embs[idx_tensor]  # [n, n_cands, dim]

    # Score: for each query, check if index 0 (ground truth) ranks highest
    sims = torch.bmm(qry_embs.unsqueeze(1), all_cand_embs.transpose(1, 2)).squeeze(1)
    correct = (sims.argmax(dim=1) == 0).sum().item()
    hit1 = correct / n * 100

    logger.info(f"  => hit@1 = {hit1:.2f}%")
    return {
        "task": task_name,
        "category": category,
        "hit_at_1": round(hit1, 2),
        "num_examples": n,
        "num_candidates": n_cands,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MMEB multimodal embedding eval")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir (default: results/<model>/mmeb/<run>/")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to extracted MMEB images (auto-detected if omitted)")
    parser.add_argument("--cache_dir", type=str, default="datasets/mmeb_cache")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true",
                        help=f"4-task subset: {QUICK_TASKS}")
    parser.add_argument("--full", action="store_true", help="All 36 tasks")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Token context cap for the embedder (Qwen3-VL paper Sec. 6.1: 16384).",
    )
    parser.add_argument(
        "--image_min_pixels",
        type=int,
        default=None,
        help="Override vision min_pixels (default: Qwen3-VL-Embedding MMEB preset).",
    )
    parser.add_argument(
        "--image_max_pixels",
        type=int,
        default=None,
        help=(
            "Override vision max_pixels (default: 1800 * 32^2 from Qwen3-VL-Embedding, "
            "paper 1800 vision tokens)."
        ),
    )
    parser.add_argument(
        "--test_instructions_repo",
        type=str,
        default=MMEB_TEST_INSTRUCTIONS,
        help=(
            "HF dataset for MMEB test rows (MMEB-V2 eval code uses ziyjiang/MMEB_Test_Instruct; "
            "images come from TIGER-Lab/MMEB-V2 via --download_images)."
        ),
    )
    parser.add_argument("--list_tasks", action="store_true")
    parser.add_argument("--download_images", action="store_true",
                        help="Download MMEB-V2 mmeb_v1 images from TIGER-Lab/MMEB-V2 and exit")
    args = parser.parse_args()

    # --- list tasks ---
    if args.list_tasks:
        for cat, tasks in TASK_CATEGORIES.items():
            print(f"\n{cat.upper()} ({len(tasks)} tasks):")
            for t in tasks:
                print(f"  {t}")
        print(f"\nTotal: {len(ALL_TASKS)} tasks")
        print(f"Quick subset: {QUICK_TASKS}")
        return

    # --- download only ---
    if args.download_images:
        download_images(args.cache_dir)
        return

    # --- eval ---
    if not args.model_path:
        parser.error("--model_path is required for evaluation")

    # Auto-generate output dir: results/<model>/mmeb/<run_number>/
    if not args.output_dir:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        base = Path("results") / model_name / "mmeb"
        base.mkdir(parents=True, exist_ok=True)
        existing = [int(d.name) for d in base.iterdir() if d.is_dir() and d.name.isdigit()]
        run_num = max(existing, default=0) + 1
        args.output_dir = str(base / str(run_num))

    attach_run_log(Path(args.output_dir))
    logger.info("MMEB image eval: batch_size=%s, max_length=%s", args.batch_size, args.max_length)
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    if torch.cuda.is_available():
        logger.info("CUDA device in process: %s", torch.cuda.get_device_name(0))

    # Determine task list
    if args.tasks:
        tasks = args.tasks
    elif args.quick:
        tasks = QUICK_TASKS
    elif args.full:
        tasks = ALL_TASKS
    else:
        tasks = QUICK_TASKS
        logger.info("No --tasks/--quick/--full specified, defaulting to --quick")

    logger.info(
        "Running %d task(s); embed micro-batch halves on CUDA OOM (see run_mmteb.py pattern)",
        len(tasks),
    )

    # Resolve image directory
    if args.image_dir:
        image_dir = args.image_dir
    else:
        image_dir = find_image_dir(args.cache_dir)
        if image_dir is None:
            logger.info("Images not found locally, downloading...")
            image_dir = download_images(args.cache_dir)
    image_dir = str(image_dir)
    logger.info(f"Image dir: {image_dir}")

    # Load model
    model, model_type = load_model(
        args.model_path,
        max_length=args.max_length,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )

    # Evaluate
    results = []
    for task_name in tasks:
        try:
            r = evaluate_task(
                model,
                task_name,
                image_dir,
                args.batch_size,
                test_instructions_repo=args.test_instructions_repo,
                image_min_pixels=args.image_min_pixels,
                image_max_pixels=args.image_max_pixels,
            )
            results.append(r)
        except Exception as e:
            logger.error(f"FAILED {task_name}: {e}", exc_info=True)
            results.append({
                "task": task_name,
                "category": get_category(task_name),
                "hit_at_1": None,
                "error": str(e),
            })

    # Summarise
    valid = [r for r in results if r.get("hit_at_1") is not None]
    per_cat = {}
    for r in valid:
        per_cat.setdefault(r["category"], []).append(r["hit_at_1"])

    summary = {
        "model_path": args.model_path,
        "model_type": model_type,
        "num_tasks": len(valid),
        "mean_hit_at_1": round(float(np.mean([r["hit_at_1"] for r in valid])), 2) if valid else None,
        "per_category": {
            cat: {"mean": round(float(np.mean(scores)), 2), "num_tasks": len(scores)}
            for cat, scores in per_cat.items()
        },
        "tasks": results,
        "eval_settings": {
            "max_length": args.max_length,
            "image_min_pixels": args.image_min_pixels or MMEB_IMAGE_MIN_PIXELS,
            "image_max_pixels": args.image_max_pixels or MMEB_IMAGE_MAX_PIXELS,
            "test_instructions_repo": args.test_instructions_repo,
            "mmeb_v2_media_repo": MMEB_V2_MEDIA_REPO,
            "mmeb_v2_image_tar": MMEB_V2_IMAGE_TAR,
            "note": (
                "36 image tasks: annotations from test_instructions_repo; "
                "pixels from MMEB-V2 mmeb_v1 tarball when using --download_images."
            ),
        },
    }

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table grouped by category, matching MMEB-V2 leaderboard style
    print()
    print("=" * 65)
    print(f"  MMEB Results — {os.path.basename(args.model_path)}")
    print("=" * 65)

    # Group results by category (preserve category ordering)
    cat_order = ["classification", "vqa", "retrieval", "grounding"]
    results_by_cat = {}
    for r in results:
        results_by_cat.setdefault(r["category"], []).append(r)

    for cat in cat_order:
        cat_results = results_by_cat.get(cat, [])
        if not cat_results:
            continue
        cat_label = {
            "classification": "Image CLS",
            "vqa": "Image QA",
            "retrieval": "Image RET",
            "grounding": "Image GD",
        }.get(cat, cat.upper())
        print(f"\n  {cat_label}")
        print(f"  {'-' * 50}")
        cat_scores = []
        for r in cat_results:
            score = f"{r['hit_at_1']:5.2f}" if r.get("hit_at_1") is not None else "ERROR"
            print(f"    {r['task']:30s}  {score}")
            if r.get("hit_at_1") is not None:
                cat_scores.append(r["hit_at_1"])
        if cat_scores:
            print(f"    {'':30s}  -----")
            print(f"    {cat_label + ' Mean':30s}  {np.mean(cat_scores):5.2f}")

    # Print any uncategorized tasks
    for cat, cat_results in results_by_cat.items():
        if cat not in cat_order:
            print(f"\n  {cat.upper()}")
            print(f"  {'-' * 50}")
            for r in cat_results:
                score = f"{r['hit_at_1']:5.2f}" if r.get("hit_at_1") is not None else "ERROR"
                print(f"    {r['task']:30s}  {score}")

    # Overall summary
    print()
    print("=" * 65)
    if valid:
        # Per-category means in one line (like the leaderboard)
        cat_means = {}
        for cat in cat_order:
            scores = [r["hit_at_1"] for r in results_by_cat.get(cat, [])
                      if r.get("hit_at_1") is not None]
            if scores:
                cat_means[cat] = np.mean(scores)

        cat_abbrev = {
            "classification": "CLS",
            "vqa": "QA",
            "retrieval": "RET",
            "grounding": "GD",
        }
        parts = [f"{cat_abbrev.get(c, c)}: {v:.1f}" for c, v in cat_means.items()]
        print(f"  {' | '.join(parts)}")
        print(f"  Image Overall: {summary['mean_hit_at_1']:.2f}")
    print("=" * 65)
    print(f"\n  Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
