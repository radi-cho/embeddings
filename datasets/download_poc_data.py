#!/usr/bin/env python3
"""Download a small PoC training subset from public HuggingFace datasets.

Produces a single JSONL file in the unified format:
  {instruction, query: {text, [image]}, positive: {text, [image]}, negatives: [{text, [image]}]}
"""
import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset

random.seed(42)


def download_nli_triplets(out_path: Path, max_samples: int = 50000):
    """AllNLI triplets from sentence-transformers – already has anchor/pos/neg."""
    print(f"Downloading AllNLI (max {max_samples})...")
    ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train", streaming=True)
    samples = []
    for row in ds:
        samples.append({
            "instruction": "Given a premise, retrieve a semantically similar hypothesis.",
            "query": {"text": row["anchor"]},
            "positive": {"text": row["positive"]},
            "negatives": [{"text": row["negative"]}],
        })
        if len(samples) >= max_samples:
            break
    write_jsonl(out_path / "nli_triplets.jsonl", samples)
    return len(samples)


def download_msmarco(out_path: Path, max_samples: int = 50000):
    """MS-MARCO passage ranking triplets."""
    print(f"Downloading MS-MARCO (max {max_samples})...")
    ds = load_dataset("sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1", "triplet", split="train", streaming=True)
    samples = []
    for row in ds:
        samples.append({
            "instruction": "Given a web search query, retrieve a relevant passage that answers the query.",
            "query": {"text": row["query"]},
            "positive": {"text": row["positive"]},
            "negatives": [{"text": row["negative"]}],
        })
        if len(samples) >= max_samples:
            break
    write_jsonl(out_path / "msmarco_triplets.jsonl", samples)
    return len(samples)


def download_gooaq(out_path: Path, max_samples: int = 50000):
    """Google auto-complete QA pairs."""
    print(f"Downloading GooAQ (max {max_samples})...")
    ds = load_dataset("sentence-transformers/gooaq", split="train", streaming=True)
    samples = []
    for row in ds:
        samples.append({
            "instruction": "Given a question, retrieve a passage that answers it.",
            "query": {"text": row["question"]},
            "positive": {"text": row["answer"]},
            "negatives": [],
        })
        if len(samples) >= max_samples:
            break
    write_jsonl(out_path / "gooaq_pairs.jsonl", samples)
    return len(samples)


def download_simplewiki(out_path: Path, max_samples: int = 50000):
    """Simple English Wikipedia paraphrase pairs."""
    print(f"Downloading SimpleWiki (max {max_samples})...")
    ds = load_dataset("sentence-transformers/simple-wiki", split="train", streaming=True)
    samples = []
    for row in ds:
        cols = list(row.keys())
        if len(cols) >= 2:
            samples.append({
                "instruction": "Retrieve a paraphrase of the given text.",
                "query": {"text": row[cols[0]]},
                "positive": {"text": row[cols[1]]},
                "negatives": [],
            })
        if len(samples) >= max_samples:
            break
    write_jsonl(out_path / "simplewiki_pairs.jsonl", samples)
    return len(samples)


def merge_all(out_path: Path, output_file: str = "poc_train.jsonl"):
    """Merge all downloaded JSONL files into one shuffled training file."""
    all_samples = []
    for f in sorted(out_path.glob("*.jsonl")):
        if f.name == output_file:
            continue
        with open(f) as fh:
            for line in fh:
                all_samples.append(json.loads(line))
    random.shuffle(all_samples)
    merged = out_path / output_file
    write_jsonl(merged, all_samples)
    print(f"Merged {len(all_samples)} samples -> {merged}")
    return len(all_samples)


def write_jsonl(path: Path, samples: list):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/poc_data")
    parser.add_argument("--max_per_source", type=int, default=50000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    total += download_nli_triplets(out, args.max_per_source)
    total += download_msmarco(out, args.max_per_source)
    total += download_gooaq(out, args.max_per_source)
    total += download_simplewiki(out, args.max_per_source)

    merged = merge_all(out)
    print(f"\nDone. Total PoC samples: {merged}")


if __name__ == "__main__":
    main()
