#!/usr/bin/env python3
"""
Quick cosine similarity between two inputs.

Each input can have text, an image path, and/or a video (list of frame paths).
Uses the same model loading as the eval harnesses.

Usage:
    # Text vs text
    python src/manual-tests/similarity.py --model_path models/Qwen3-VL-Embedding-2B \
        --a "a cat sitting on a couch" --b "a dog playing in the yard"

    # Text vs image
    python src/manual-tests/similarity.py --model_path models/Qwen3-VL-Embedding-2B \
        --a "a cat" --b_image path/to/cat.jpg

    # Image vs image
    python src/manual-tests/similarity.py --model_path models/Qwen3-VL-Embedding-2B \
        --a_image img1.png --b_image img2.png

    # With instructions
    python src/manual-tests/similarity.py --model_path models/Qwen3-VL-Embedding-2B \
        --a "what breed is this cat?" --a_instruction "Represent the query." \
        --b_image cat.jpg --b_instruction "Represent the document."
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model(model_path, max_length=16384):
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        with open(config_path) as f:
            model_type = json.load(f).get("model_type", "")

    min_pixels = 4 * 32 * 32
    max_pixels = 1800 * 32 * 32

    if "qwen3_vl" in model_type:
        scripts_dir = Path(model_path) / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        from qwen3_vl_embedding import Qwen3VLEmbedder

        model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        from src.models.qwen35_embedding import Qwen35Embedder

        model = Qwen35Embedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    return model


def build_item(text=None, image=None, instruction=None):
    item = {}
    if text:
        item["text"] = text
    if image:
        item["image"] = image
    if instruction:
        item["instruction"] = instruction
    return item if item else {"text": ""}


def main():
    p = argparse.ArgumentParser(description="Cosine similarity between two inputs")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--max_length", type=int, default=16384)

    p.add_argument("--a", type=str, default=None, help="Text for input A")
    p.add_argument("--a_image", type=str, default=None, help="Image path for input A")
    p.add_argument("--a_instruction", type=str, default=None)

    p.add_argument("--b", type=str, default=None, help="Text for input B")
    p.add_argument("--b_image", type=str, default=None, help="Image path for input B")
    p.add_argument("--b_instruction", type=str, default=None)
    args = p.parse_args()

    model = load_model(args.model_path, args.max_length)

    item_a = build_item(args.a, args.a_image, args.a_instruction)
    item_b = build_item(args.b, args.b_image, args.b_instruction)

    print(f"A: {item_a}")
    print(f"B: {item_b}")

    with torch.no_grad():
        embs = model.process([item_a, item_b])

    emb_a = embs[0]
    emb_b = embs[1]
    cos_sim = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()

    print(f"\nCosine similarity: {cos_sim:.6f}")


if __name__ == "__main__":
    main()
