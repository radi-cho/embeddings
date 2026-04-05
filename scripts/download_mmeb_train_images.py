#!/usr/bin/env python3
"""
Download and extract MMEB training images from TIGER-Lab/MMEB-train.

Extracts to: {output_dir}/images/{subset}/Train/*.jpg
This matches the HF dataset paths (images/{subset}/Train/{filename}.jpg)
so --image_dir should point to {output_dir}/images.

Usage:
    python scripts/download_mmeb_train_images.py

    # Custom output directory
    python scripts/download_mmeb_train_images.py --output_dir datasets/mmeb_train_images

    # Download specific subsets only
    python scripts/download_mmeb_train_images.py --subsets N24News,OK-VQA

Then train with:
    python src/train/train.py --model_path models/Qwen3-VL-2B \
        --image_dir datasets/mmeb_train_images/images ...
"""

import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download

REPO_ID = "TIGER-Lab/MMEB-train"

ALL_SUBSETS = [
    "A-OKVQA", "CIRR", "ChartQA", "DocVQA", "HatefulMemes",
    "ImageNet_1K", "InfographicsVQA", "MSCOCO", "MSCOCO_i2t",
    "MSCOCO_t2i", "N24News", "NIGHTS", "OK-VQA", "SUN397",
    "VOC2007", "VisDial", "Visual7W", "VisualNews_i2t",
    "VisualNews_t2i", "WebQA",
]


def main():
    parser = argparse.ArgumentParser(description="Download MMEB training images")
    parser.add_argument("--output_dir", type=str, default="datasets/mmeb_train_images",
                        help="Where to extract images")
    parser.add_argument("--subsets", type=str, default=None,
                        help="Comma-separated subset names (default: all)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF download cache directory")
    args = parser.parse_args()

    subsets = args.subsets.split(",") if args.subsets else ALL_SUBSETS
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(images_dir, subset, "Train")
        if os.path.isdir(subset_dir) and os.listdir(subset_dir):
            print(f"[{i+1}/{len(subsets)}] {subset} — already extracted, skipping")
            continue

        zip_path_in_repo = f"images_zip/{subset}.zip"
        print(f"[{i+1}/{len(subsets)}] Downloading {subset}.zip ...")
        try:
            local_zip = hf_hub_download(
                REPO_ID, zip_path_in_repo,
                repo_type="dataset",
                cache_dir=args.cache_dir,
            )
        except Exception as e:
            print(f"  FAILED to download: {e}")
            continue

        print(f"  Extracting to {images_dir}/ ...")
        with zipfile.ZipFile(local_zip) as zf:
            zf.extractall(images_dir)

        if os.path.isdir(subset_dir):
            n = len(os.listdir(subset_dir))
            print(f"  Done — {n} images")
        else:
            print(f"  Warning: expected {subset_dir} but not found after extraction")

    print(f"\nAll done. Use --image_dir {images_dir} when training.")


if __name__ == "__main__":
    main()
