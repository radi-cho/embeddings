#!/usr/bin/env python3
"""
MMEB (Multimodal Embedding Benchmark) evaluation.

Usage:
    # First: download images (one-time, ~7 GB)
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
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MMEB_DATASET = "ziyjiang/MMEB_Test_Instruct"

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
    """Download and extract MMEB-eval images.zip from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import zipfile

    cache_dir = Path(cache_dir)
    images_dir = cache_dir / "images"

    # Check if already extracted
    if images_dir.exists():
        # Verify by checking for at least one task folder
        subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
        if subdirs:
            logger.info(f"Images already extracted at {images_dir} ({len(subdirs)} folders)")
            return images_dir
        # Might be nested: images/images/
        nested = images_dir / "images"
        if nested.exists():
            subdirs = [d for d in nested.iterdir() if d.is_dir()]
            if subdirs:
                logger.info(f"Images found at {nested} ({len(subdirs)} folders)")
                return nested

    logger.info("Downloading MMEB-eval images.zip (~7.1 GB)...")
    zip_path = hf_hub_download(
        repo_id=MMEB_DATASET,
        filename="images.zip",
        repo_type="dataset",
        cache_dir=str(cache_dir / "hf_download"),
    )

    logger.info(f"Extracting to {images_dir}...")
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(images_dir)

    # Handle nested extraction
    nested = images_dir / "images"
    if nested.exists() and nested.is_dir():
        images_dir = nested

    logger.info("Done.")
    return images_dir


def find_image_dir(cache_dir):
    """Locate the extracted image directory."""
    cache_dir = Path(cache_dir)
    for candidate in [
        cache_dir / "images" / "images",
        cache_dir / "images",
        cache_dir,
    ]:
        if candidate.exists() and candidate.is_dir():
            # Check for a task subfolder
            for task in ["N24News", "OK-VQA", "MSCOCO_i2t", "ImageNet-1K"]:
                if (candidate / task).exists():
                    return candidate
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path):
    """Load embedding model, auto-detecting Qwen3-VL vs Qwen3.5."""
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        with open(config_path) as f:
            model_type = json.load(f).get("model_type", "")

    if "qwen3_vl" in model_type:
        scripts_dir = Path(model_path) / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        from qwen3_vl_embedding import Qwen3VLEmbedder

        logger.info(f"Loading Qwen3-VL from {model_path}")
        model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
        )
        return model, "qwen3vl"
    else:
        from src.models.qwen35_embedding import Qwen35Embedder

        logger.info(f"Loading Qwen3.5 from {model_path}")
        model = Qwen35Embedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
        )
        return model, "qwen35"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_batch(model, items, batch_size=32):
    """Embed a list of dicts (text/image/instruction) in batches."""
    all_embs = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        with torch.no_grad():
            embs = model.process(batch)
        all_embs.append(embs.cpu().float())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)


def make_item(text, img_path, image_dir, instruction=None):
    """Build a dict suitable for model.process() from MMEB fields.

    Uses the separated instruction field from MMEB_Test_Instruct.
    """
    item = {}
    clean_text = strip_image_placeholder(text) if text else ""
    clean_inst = strip_image_placeholder(instruction) if instruction else ""
    img = resolve_image(img_path, image_dir)
    if img:
        item["image"] = img
    if clean_text:
        item["text"] = clean_text
    if clean_inst:
        item["instruction"] = clean_inst
    return item if item else {"text": ""}


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def evaluate_task(model, task_name, image_dir, batch_size):
    """Load one MMEB task and evaluate.

    Each example has its own candidate set with the correct answer at index 0.
    Pre-embeds all queries and candidates in batched passes, then scores on CPU.
    """
    from datasets import load_dataset

    logger.info(f"--- {task_name} ---")
    ds = load_dataset(MMEB_DATASET, task_name, split="test")
    n = len(ds)
    n_cands = len(ds[0]["tgt_text"])
    category = get_category(task_name)

    logger.info(f"  category={category}  examples={n}  candidates={n_cands}")

    # Build all query items
    queries = [
        make_item(ex["qry_text"], ex["qry_img_path"], image_dir, instruction=ex.get("qry_inst"))
        for ex in ds
    ]

    # Deduplicate candidates: build a unique set keyed by (text, img_path, instruction)
    unique_cands = {}  # key -> index in unique list
    unique_cand_items = []
    # Per-example mapping: for each example, the indices into unique_cands
    cand_indices = []  # [n, n_cands]
    for ex in ds:
        tgt_inst = ex.get("tgt_inst", "")
        ex_indices = []
        for t, p in zip(ex["tgt_text"], ex["tgt_img_path"]):
            key = (t, p, tgt_inst)
            if key not in unique_cands:
                unique_cands[key] = len(unique_cand_items)
                unique_cand_items.append(make_item(t, p, image_dir, instruction=tgt_inst))
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
    parser.add_argument("--list_tasks", action="store_true")
    parser.add_argument("--download_images", action="store_true",
                        help="Download images and exit")
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
    model, model_type = load_model(args.model_path)

    # Evaluate
    results = []
    for task_name in tasks:
        try:
            r = evaluate_task(model, task_name, image_dir, args.batch_size)
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
