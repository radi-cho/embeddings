#!/usr/bin/env python3
"""
MMEB-V2 VisDoc (Visual Document) evaluation.

Evaluates multimodal embedding models on MMEB-V2 document retrieval tasks:
text query -> document page image.  Three sub-benchmarks: ViDoRe (v1+v2),
VisRAG, and ViDoSeek/MMLongBench.

All tasks use BEIR format (queries / corpus / qrels splits on HuggingFace) and
global retrieval evaluation.  Corpus images are PIL objects inside the HF
dataset and are saved to disk as PNG on first use.

Consistent with run_mmeb.py (image) and run_mmeb_video.py (video) -- same
model loading, embedding strategy, output format, and CLI interface.

Directory structure (under --visdoc_dir, default: datasets/mmeb_cache/visdoc-tasks):
    images/  (auto-populated from HF corpus on first eval)
        ViDoRe_arxivqa/       {corpus_id}.png
        VisRAG_ArxivQA/       {short_hashed_name}.png
        ViDoSeek-page/        {corpus_id}.png
        ...

Usage:
    # Quick eval: 1 task per sub-benchmark
    python src/eval/run_mmeb_visdoc.py --model_path models/Qwen3-VL-Embedding-2B --quick

    # Specific tasks
    python src/eval/run_mmeb_visdoc.py --model_path models/Qwen3-VL-Embedding-2B \\
        --tasks ViDoRe_arxivqa VisRAG_ChartQA

    # All 24 active tasks (matches Qwen visdoc.yaml)
    python src/eval/run_mmeb_visdoc.py --model_path models/Qwen3-VL-Embedding-2B --full

    # List available tasks
    python src/eval/run_mmeb_visdoc.py --list_tasks
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.eval.eval_utils import attach_run_log, load_model, embed_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MMEB_IMAGE_MIN_PIXELS = 4 * 32 * 32
MMEB_IMAGE_MAX_PIXELS = 1800 * 32 * 32

TASK_INST_QRY = "Find a document image that matches the given query."

# HuggingFace dataset sources  (repo, language_filter_or_None, split)
# Language filter is applied to the queries subset only (ViDoRe v2 multilingual).
# BEIR subsets ("queries", "corpus", "qrels") are loaded separately.
HF_DATASETS = {
    # ViDoRe v1
    "ViDoRe_arxivqa":       ("vidore/arxivqa_test_subsampled_beir",                         None,      "test"),
    "ViDoRe_docvqa":        ("vidore/docvqa_test_subsampled_beir",                          None,      "test"),
    "ViDoRe_infovqa":       ("vidore/infovqa_test_subsampled_beir",                         None,      "test"),
    "ViDoRe_tabfquad":      ("vidore/tabfquad_test_subsampled_beir",                        None,      "test"),
    "ViDoRe_tatdqa":        ("vidore/tatdqa_test_beir",                                     None,      "test"),
    "ViDoRe_shiftproject":  ("vidore/shiftproject_test_beir",                                None,      "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence":  ("vidore/syntheticDocQA_artificial_intelligence_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_energy":                   ("vidore/syntheticDocQA_energy_test_beir",                  None, "test"),
    "ViDoRe_syntheticDocQA_government_reports":       ("vidore/syntheticDocQA_government_reports_test_beir",      None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry":      ("vidore/syntheticDocQA_healthcare_industry_test_beir",     None, "test"),
    # ViDoRe v2
    "ViDoRe_esg_reports_human_labeled_v2":            ("vidore/esg_reports_human_labeled_v2",         None,      "test"),
    "ViDoRe_biomedical_lectures_v2":                  ("vidore/biomedical_lectures_v2",               "english", "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual":     ("vidore/biomedical_lectures_v2",               None,      "test"),
    "ViDoRe_economics_reports_v2":                    ("vidore/economics_reports_v2",                 "english", "test"),
    "ViDoRe_economics_reports_v2_multilingual":       ("vidore/economics_reports_v2",                 None,      "test"),
    "ViDoRe_esg_reports_v2":                          ("vidore/esg_reports_v2",                       "english", "test"),
    "ViDoRe_esg_reports_v2_multilingual":             ("vidore/esg_reports_v2",                       None,      "test"),
    # VisRAG
    "VisRAG_ArxivQA":   ("openbmb/VisRAG-Ret-Test-ArxivQA",   None, "train"),
    "VisRAG_ChartQA":   ("openbmb/VisRAG-Ret-Test-ChartQA",   None, "train"),
    "VisRAG_MP-DocVQA": ("openbmb/VisRAG-Ret-Test-MP-DocVQA", None, "train"),
    "VisRAG_SlideVQA":  ("openbmb/VisRAG-Ret-Test-SlideVQA",  None, "train"),
    "VisRAG_InfoVQA":   ("openbmb/VisRAG-Ret-Test-InfoVQA",   None, "train"),
    "VisRAG_PlotQA":    ("openbmb/VisRAG-Ret-Test-PlotQA",    None, "train"),
    # ViDoSeek / MMLongBench
    "ViDoSeek-page":    ("VLM2Vec/ViDoSeek-page-fixed", None, "test"),
    "ViDoSeek-doc":     ("VLM2Vec/ViDoSeek",            None, "test"),
    "MMLongBench-doc":  ("VLM2Vec/MMLongBench-doc",     None, "test"),
    "MMLongBench-page": ("VLM2Vec/MMLongBench-page-fixed", None, "test"),
}

# Parser type controls how corpus images are named on disk.
# "vidore": {corpus_id}.png   "visrag": MD5-hashed short name
VISDOC_TASKS = {}
for name in HF_DATASETS:
    parser = "visrag" if name.startswith("VisRAG_") else "vidore"
    if name.startswith("ViDoRe_") and "v2" not in name:
        cat = "vidore_v1"
    elif name.startswith("ViDoRe_"):
        cat = "vidore_v2"
    elif name.startswith("VisRAG_"):
        cat = "visrag"
    else:
        cat = "vidoseek"
    VISDOC_TASKS[name] = {"category": cat, "parser": parser}

TASK_CATEGORIES = {
    "vidore_v1": [t for t, v in VISDOC_TASKS.items() if v["category"] == "vidore_v1"],
    "vidore_v2": [t for t, v in VISDOC_TASKS.items() if v["category"] == "vidore_v2"],
    "visrag":    [t for t, v in VISDOC_TASKS.items() if v["category"] == "visrag"],
    "vidoseek":  [t for t, v in VISDOC_TASKS.items() if v["category"] == "vidoseek"],
}

# Active tasks match Qwen visdoc.yaml (English-only ViDoRe v2 variants commented out).
_ENGLISH_ONLY_V2 = {
    "ViDoRe_biomedical_lectures_v2",
    "ViDoRe_economics_reports_v2",
    "ViDoRe_esg_reports_v2",
}
ALL_TASKS = [t for t in HF_DATASETS if t not in _ENGLISH_ONLY_V2]
QUICK_TASKS = ["ViDoRe_arxivqa", "VisRAG_ChartQA", "ViDoSeek-page"]


def get_category(task_name):
    return VISDOC_TASKS.get(task_name, {}).get("category", "unknown")


# ---------------------------------------------------------------------------
# VisRAG filename hashing  (from Qwen visrag_dataset.py)
# ---------------------------------------------------------------------------

def get_short_imagename(image_name):
    """Truncate + MD5 hash for path-safe filenames (VisRAG convention)."""
    base, ext = os.path.splitext(image_name)
    short_base = base[:50] + "_" + hashlib.md5(image_name.encode("utf-8")).hexdigest()[:8]
    return short_base + ext


def _corpus_image_path(task_name, image_root, corpus_id):
    """Return the on-disk path for a corpus image."""
    if VISDOC_TASKS[task_name]["parser"] == "visrag":
        return os.path.join(image_root, get_short_imagename(corpus_id))
    return os.path.join(image_root, f"{corpus_id}.png")


# ---------------------------------------------------------------------------
# Qrels helpers
# ---------------------------------------------------------------------------

def load_qrels_mapping(qrels_dataset):
    """Build {query_id: {corpus_id: relevance_score}} from a BEIR qrels split.

    Only entries with score > 0 are kept.  Duplicates keep the max score.
    Matches Qwen dataset_utils.load_qrels_mapping exactly.
    """
    mapping = {}
    for row in qrels_dataset:
        qid = row["query-id"]
        docid = row["corpus-id"]
        score = row["score"]
        if score > 0:
            mapping.setdefault(qid, {})
            mapping[qid][docid] = max(mapping[qid].get(docid, 0), score)
    return mapping


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _hit_at_k(ranked_ids, qrel_dict, k=1):
    for rid in ranked_ids[:k]:
        if qrel_dict.get(rid, 0) > 0:
            return 1.0
    return 0.0


def _ndcg_at_k(ranked_ids, qrel_dict, k=5):
    dcg = 0.0
    for i in range(min(k, len(ranked_ids))):
        rel = qrel_dict.get(ranked_ids[i], 0)
        dcg += rel / np.log2(i + 2)
    ideal_rels = sorted(qrel_dict.values(), reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_rels))):
        idcg += ideal_rels[i] / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_ids, qrel_dict, k=5):
    n_rel = sum(1 for s in qrel_dict.values() if s > 0)
    if n_rel == 0:
        return 0.0
    found = sum(1 for rid in ranked_ids[:k] if qrel_dict.get(rid, 0) > 0)
    return found / n_rel


# load_model and embed_batch imported from eval_utils.
def load_visdoc_model(model_path, *, max_length=16384,
                      image_min_pixels=None, image_max_pixels=None):
    return load_model(
        model_path, max_length=max_length,
        default_min_pixels=MMEB_IMAGE_MIN_PIXELS,
        default_max_pixels=MMEB_IMAGE_MAX_PIXELS,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels)


# ---------------------------------------------------------------------------
# BEIR data loading
# ---------------------------------------------------------------------------

def _load_beir_splits(task_name):
    """Load queries, corpus, and qrels from a BEIR-format HuggingFace dataset."""
    from datasets import load_dataset

    repo, lang, split = HF_DATASETS[task_name]
    queries = load_dataset(repo, "queries", split=split)
    corpus = load_dataset(repo, "corpus", split=split)
    qrels = load_dataset(repo, "qrels", split=split)

    # Language filter (ViDoRe v2 multilingual datasets)
    if lang is not None:
        queries = queries.filter(lambda ex: ex.get("language") == lang)

    return queries, corpus, qrels


def _save_corpus_images(task_name, corpus_dataset, image_root):
    """Save PIL images from the HF corpus to disk (idempotent)."""
    os.makedirs(image_root, exist_ok=True)
    saved, skipped = 0, 0
    for row in corpus_dataset:
        corpus_id = row["corpus-id"]
        img_path = _corpus_image_path(task_name, image_root, corpus_id)
        if not os.path.exists(img_path):
            pil_img = row.get("image")
            if pil_img is not None:
                pil_img.save(img_path)
                saved += 1
            else:
                skipped += 1
    if saved:
        logger.info("  Saved %d corpus images to %s", saved, image_root)
    if skipped:
        logger.warning("  %d corpus rows had no image data", skipped)


def load_visdoc_data(task_name, visdoc_dir):
    """Load one VisDoc task.

    Returns
    -------
    query_items : list[dict]
        Items for model.process()  (text + instruction).
    corpus_items : list[dict]
        Items for model.process()  (image path).
    corpus_ids : list[str]
        Corpus ID parallel to corpus_items.
    query_qrels : list[dict]
        Per-query relevance dict  {corpus_id: score}.
    """
    queries_ds, corpus_ds, qrels_ds = _load_beir_splits(task_name)
    qrels_mapping = load_qrels_mapping(qrels_ds)

    # Save corpus images to disk
    image_root = os.path.join(visdoc_dir, "images", task_name)
    _save_corpus_images(task_name, corpus_ds, image_root)

    # Build corpus items (one per document page)
    corpus_items = []
    corpus_ids = []
    for row in corpus_ds:
        cid = row["corpus-id"]
        img_path = _corpus_image_path(task_name, image_root, cid)
        if os.path.exists(img_path):
            corpus_items.append({
                "image": img_path,
                "min_pixels": MMEB_IMAGE_MIN_PIXELS,
                "max_pixels": MMEB_IMAGE_MAX_PIXELS,
            })
            corpus_ids.append(cid)

    # Build query items (only queries that have qrels entries)
    query_items = []
    query_qrels = []
    skipped = 0
    for row in queries_ds:
        qid = row["query-id"]
        if qid not in qrels_mapping:
            skipped += 1
            continue
        query_items.append({
            "text": row["query"],
            "instruction": TASK_INST_QRY,
        })
        query_qrels.append(qrels_mapping[qid])

    if skipped:
        logger.info("  Skipped %d queries without qrels", skipped)

    logger.info(
        "  %d queries, %d corpus images, %d qrels entries",
        len(query_items), len(corpus_items), len(qrels_mapping),
    )
    return query_items, corpus_items, corpus_ids, query_qrels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_task(model, task_name, visdoc_dir, batch_size):
    """Load data and evaluate one VisDoc task (global retrieval)."""
    logger.info("--- %s ---", task_name)

    query_items, corpus_items, corpus_ids, query_qrels = load_visdoc_data(
        task_name, visdoc_dir,
    )
    n_q = len(query_items)
    n_c = len(corpus_items)
    if not n_q or not n_c:
        return None

    logger.info("  Embedding %d queries ...", n_q)
    qry_embs = embed_batch(model, query_items, batch_size)
    logger.info("  Embedding %d corpus images ...", n_c)
    corpus_embs = embed_batch(model, corpus_items, batch_size)

    # Full similarity matrix  [N_q, N_c]
    sims = qry_embs @ corpus_embs.T
    ranked_indices = sims.argsort(dim=1, descending=True).cpu().numpy()

    # Compute per-query metrics
    hits1, ndcgs5, recalls5 = [], [], []
    for i in range(n_q):
        ranked = [corpus_ids[j] for j in ranked_indices[i]]
        qrel = query_qrels[i]
        hits1.append(_hit_at_k(ranked, qrel, k=1))
        ndcgs5.append(_ndcg_at_k(ranked, qrel, k=5))
        recalls5.append(_recall_at_k(ranked, qrel, k=5))

    hit1 = float(np.mean(hits1)) * 100
    ndcg5 = float(np.mean(ndcgs5)) * 100
    recall5 = float(np.mean(recalls5)) * 100

    logger.info("  => hit@1=%.2f%%  ndcg@5=%.2f%%  recall@5=%.2f%%", hit1, ndcg5, recall5)
    return {
        "task": task_name,
        "category": get_category(task_name),
        "hit_at_1": round(hit1, 2),
        "ndcg_at_5": round(ndcg5, 2),
        "recall_at_5": round(recall5, 2),
        "num_queries": n_q,
        "num_corpus": n_c,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MMEB-V2 VisDoc embedding eval")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir (default: results/<model>/mmeb_visdoc/<run>/)")
    parser.add_argument("--visdoc_dir", type=str, default="datasets/mmeb_cache/visdoc-tasks",
                        help="Base directory for document images (auto-populated from HF)")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true",
                        help=f"Quick subset: {QUICK_TASKS}")
    parser.add_argument("--full", action="store_true",
                        help=f"All {len(ALL_TASKS)} active tasks (matches Qwen visdoc.yaml)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Embed micro-batch size (default 16)")
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--image_min_pixels", type=int, default=None)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--list_tasks", action="store_true")
    args = parser.parse_args()

    # --- list tasks ---
    if args.list_tasks:
        cat_labels = {
            "vidore_v1": "ViDoRe v1", "vidore_v2": "ViDoRe v2",
            "visrag": "VisRAG", "vidoseek": "ViDoSeek / MMLongBench",
        }
        for cat in ["vidore_v1", "vidore_v2", "visrag", "vidoseek"]:
            tasks = TASK_CATEGORIES[cat]
            print(f"\n{cat_labels[cat]} ({len(tasks)} tasks):")
            for t in tasks:
                active = " " if t in ALL_TASKS else "*"
                repo = HF_DATASETS[t][0]
                print(f"  {active} {t:52s}  HF={repo}")
        print(f"\nActive: {len(ALL_TASKS)} tasks   (* = English-only, not in --full)")
        print(f"Quick subset: {QUICK_TASKS}")
        return

    # --- eval ---
    if not args.model_path:
        parser.error("--model_path is required for evaluation")

    if not args.output_dir:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        base = Path("results") / model_name / "mmeb_visdoc"
        base.mkdir(parents=True, exist_ok=True)
        existing = [int(d.name) for d in base.iterdir() if d.is_dir() and d.name.isdigit()]
        run_num = max(existing, default=0) + 1
        args.output_dir = str(base / str(run_num))

    attach_run_log(Path(args.output_dir))
    logger.info("MMEB VisDoc eval: batch_size=%s, max_length=%s", args.batch_size, args.max_length)
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    if args.tasks:
        tasks = args.tasks
    elif args.quick:
        tasks = QUICK_TASKS
    elif args.full:
        tasks = ALL_TASKS
    else:
        tasks = QUICK_TASKS
        logger.info("No --tasks/--quick/--full specified, defaulting to --quick")

    logger.info("Running %d task(s)", len(tasks))
    logger.info("VisDoc dir: %s", args.visdoc_dir)

    model, model_type = load_visdoc_model(
        args.model_path,
        max_length=args.max_length,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )

    # Evaluate
    results = []
    for task_name in tasks:
        try:
            r = evaluate_task(model, task_name, args.visdoc_dir, args.batch_size)
            if r:
                results.append(r)
            else:
                results.append({
                    "task": task_name, "category": get_category(task_name),
                    "hit_at_1": None, "ndcg_at_5": None, "recall_at_5": None,
                    "error": "No valid examples",
                })
        except Exception as e:
            logger.error("FAILED %s: %s", task_name, e, exc_info=True)
            results.append({
                "task": task_name, "category": get_category(task_name),
                "hit_at_1": None, "ndcg_at_5": None, "recall_at_5": None,
                "error": str(e),
            })

    # --- Summarise ---
    valid = [r for r in results if r.get("hit_at_1") is not None]
    per_cat = {}
    for r in valid:
        per_cat.setdefault(r["category"], []).append(r)

    def _cat_means(key):
        return {
            cat: round(float(np.mean([r[key] for r in rs])), 2)
            for cat, rs in per_cat.items()
        }

    summary = {
        "model_path": args.model_path,
        "model_type": model_type,
        "num_tasks": len(valid),
        "mean_hit_at_1": round(float(np.mean([r["hit_at_1"] for r in valid])), 2) if valid else None,
        "mean_ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in valid])), 2) if valid else None,
        "mean_recall_at_5": round(float(np.mean([r["recall_at_5"] for r in valid])), 2) if valid else None,
        "per_category": {
            cat: {
                "mean_hit_at_1": round(float(np.mean([r["hit_at_1"] for r in rs])), 2),
                "mean_ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in rs])), 2),
                "num_tasks": len(rs),
            }
            for cat, rs in per_cat.items()
        },
        "tasks": results,
        "eval_settings": {
            "max_length": args.max_length,
            "image_min_pixels": args.image_min_pixels or MMEB_IMAGE_MIN_PIXELS,
            "image_max_pixels": args.image_max_pixels or MMEB_IMAGE_MAX_PIXELS,
            "visdoc_dir": args.visdoc_dir,
        },
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print table ---
    print()
    print("=" * 78)
    print(f"  MMEB VisDoc Results \u2014 {os.path.basename(args.model_path)}")
    print("=" * 78)

    cat_order = ["vidore_v1", "vidore_v2", "visrag", "vidoseek"]
    cat_labels = {
        "vidore_v1": "ViDoRe v1",
        "vidore_v2": "ViDoRe v2",
        "visrag": "VisRAG",
        "vidoseek": "ViDoSeek / MMLongBench",
    }
    results_by_cat = {}
    for r in results:
        results_by_cat.setdefault(r["category"], []).append(r)

    print(f"\n  {'Task':44s} {'hit@1':>7s} {'ndcg@5':>8s} {'R@5':>7s}")
    for cat in cat_order:
        cat_results = results_by_cat.get(cat, [])
        if not cat_results:
            continue
        label = cat_labels.get(cat, cat)
        print(f"\n  {label}")
        print(f"  {'-' * 68}")
        scores_h, scores_n = [], []
        for r in cat_results:
            h = f"{r['hit_at_1']:5.2f}" if r.get("hit_at_1") is not None else "ERROR"
            n = f"{r['ndcg_at_5']:6.2f}" if r.get("ndcg_at_5") is not None else " ERROR"
            rc = f"{r['recall_at_5']:5.2f}" if r.get("recall_at_5") is not None else "ERROR"
            print(f"    {r['task']:42s} {h:>7s} {n:>8s} {rc:>7s}")
            if r.get("hit_at_1") is not None:
                scores_h.append(r["hit_at_1"])
                scores_n.append(r["ndcg_at_5"])
        if scores_h:
            print(f"    {'':42s} {'-----':>7s} {'------':>8s} {'-----':>7s}")
            print(
                f"    {label + ' Mean':42s} "
                f"{np.mean(scores_h):5.2f}   "
                f"{np.mean(scores_n):6.2f}"
            )

    # Uncategorised
    for cat, cat_results in results_by_cat.items():
        if cat not in cat_order:
            print(f"\n  {cat.upper()}")
            for r in cat_results:
                h = f"{r['hit_at_1']:5.2f}" if r.get("hit_at_1") is not None else "ERROR"
                print(f"    {r['task']:42s} {h:>7s}")

    print()
    print("=" * 78)
    if valid:
        cat_means = {}
        for cat in cat_order:
            rs = [r for r in results_by_cat.get(cat, []) if r.get("ndcg_at_5") is not None]
            if rs:
                cat_means[cat] = np.mean([r["ndcg_at_5"] for r in rs])
        abbrev = {
            "vidore_v1": "ViDoRe", "vidore_v2": "ViDoRe-v2",
            "visrag": "VisRAG", "vidoseek": "ViDoSeek",
        }
        parts = [f"{abbrev.get(c, c)}: {v:.1f}" for c, v in cat_means.items()]
        print(f"  NDCG@5  {' | '.join(parts)}")
        print(f"  VisDoc Overall  hit@1={summary['mean_hit_at_1']:.2f}  ndcg@5={summary['mean_ndcg_at_5']:.2f}")
    print("=" * 78)
    print(f"\n  Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
