#!/usr/bin/env python3
"""Compare MMTEB results between two models side-by-side.

Usage:
  python src/eval/compare_models.py --results_a results/qwen3vl --results_b results/qwen35
  python src/eval/compare_models.py --model_a Qwen/Qwen3-VL-Embedding-2B --model_b models/Qwen3.5-0.8B
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_summary(results_dir: str) -> dict:
    path = Path(results_dir) / "summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def run_eval_if_needed(model_path: str, output_dir: str, fast_tasks_only: bool = True):
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        print(f"Results already exist at {output_dir}, skipping eval")
        return
    cmd = [
        sys.executable, "src/eval/run_mmteb.py",
        "--model_path", model_path,
        "--output_dir", output_dir,
    ]
    if not fast_tasks_only:
        cmd.append("--full")
    print(f"Running eval: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def print_comparison(summary_a: dict, summary_b: dict):
    name_a = summary_a.get("model_path", "Model A")
    name_b = summary_b.get("model_path", "Model B")
    types_a = summary_a.get("per_type", {})
    types_b = summary_b.get("per_type", {})
    all_types = sorted(set(list(types_a.keys()) + list(types_b.keys())))

    col_w = max(len(name_a), len(name_b), 12) + 2
    print(f"\n{'Task Type':<35s} {name_a:>{col_w}s} {name_b:>{col_w}s} {'Delta':>8s}")
    print("-" * (35 + col_w * 2 + 10))

    for tt in all_types:
        sa = types_a.get(tt, 0)
        sb = types_b.get(tt, 0)
        delta = sb - sa
        sign = "+" if delta > 0 else ""
        print(f"  {tt:<33s} {sa:>{col_w}.2f} {sb:>{col_w}.2f} {sign}{delta:>7.2f}")

    mean_task_a = summary_a.get("mean_task", 0)
    mean_task_b = summary_b.get("mean_task", 0)
    mean_type_a = summary_a.get("mean_type", 0)
    mean_type_b = summary_b.get("mean_type", 0)

    print("-" * (35 + col_w * 2 + 10))
    d1 = mean_task_b - mean_task_a
    d2 = mean_type_b - mean_type_a
    print(f"  {'Mean (Task)':<33s} {mean_task_a:>{col_w}.2f} {mean_task_b:>{col_w}.2f} {'+'if d1>0 else ''}{d1:>7.2f}")
    print(f"  {'Mean (Type)':<33s} {mean_type_a:>{col_w}.2f} {mean_type_b:>{col_w}.2f} {'+'if d2>0 else ''}{d2:>7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, help="First model path (will run eval if results missing)")
    parser.add_argument("--model_b", type=str, help="Second model path (will run eval if results missing)")
    parser.add_argument("--results_a", type=str, help="Pre-computed results dir for model A")
    parser.add_argument("--results_b", type=str, help="Pre-computed results dir for model B")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    dir_a = args.results_a or f"results/{Path(args.model_a).name}" if args.model_a else args.results_a
    dir_b = args.results_b or f"results/{Path(args.model_b).name}" if args.model_b else args.results_b

    if not dir_a or not dir_b:
        parser.error("Provide either --results_a/--results_b or --model_a/--model_b")

    if args.model_a and not (Path(dir_a) / "summary.json").exists():
        run_eval_if_needed(args.model_a, dir_a, not args.full)
    if args.model_b and not (Path(dir_b) / "summary.json").exists():
        run_eval_if_needed(args.model_b, dir_b, not args.full)

    summary_a = load_summary(dir_a)
    summary_b = load_summary(dir_b)

    if not summary_a:
        print(f"No results found at {dir_a}")
        return
    if not summary_b:
        print(f"No results found at {dir_b}")
        return

    print_comparison(summary_a, summary_b)


if __name__ == "__main__":
    main()
