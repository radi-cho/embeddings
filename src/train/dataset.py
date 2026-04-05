"""
Dataset module for MMEB multimodal embedding training.

Loads the TIGER-Lab/MMEB-train dataset from Hugging Face, supporting:
- All 20 subsets by default, or user-selected subsets via CLI
- Hard negatives (neg_text / neg_image_path) when available
- Progress logging for download/loading
- Text-only, image-only, and multimodal query/document formats
- STS-style datasets (future) with real-valued similarity scores
"""

import logging
import os
import random
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

logger = logging.getLogger(__name__)

ALL_MMEB_SUBSETS = [
    "A-OKVQA", "CIRR", "ChartQA", "DocVQA", "HatefulMemes",
    "ImageNet_1K", "InfographicsVQA", "MSCOCO", "MSCOCO_i2t",
    "MSCOCO_t2i", "N24News", "NIGHTS", "OK-VQA", "SUN397",
    "VOC2007", "VisDial", "Visual7W", "VisualNews_i2t",
    "VisualNews_t2i", "WebQA",
]

TASK_TYPE_MAP = {
    "classification": ["N24News", "HatefulMemes", "VOC2007", "SUN397", "ImageNet_1K"],
    "vqa": ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
             "Visual7W", "VisDial", "WebQA"],
    "retrieval": ["MSCOCO", "MSCOCO_i2t", "MSCOCO_t2i",
                  "VisualNews_i2t", "VisualNews_t2i", "CIRR", "NIGHTS"],
}

HF_MMEB_REPO = "TIGER-Lab/MMEB-train"
IMAGE_PLACEHOLDER = "<|image_1|>"


def _build_hf_image_url(relative_path: str) -> str:
    return f"https://huggingface.co/datasets/{HF_MMEB_REPO}/resolve/main/{relative_path}"


def _load_image_from_path(path: str, image_dir: Optional[str] = None) -> Optional[Image.Image]:
    """Load an image from a local path or HF URL."""
    if not path:
        return None

    if image_dir:
        # HF paths are "images/{subset}/Train/{filename}.jpg".
        # With --image_dir pointing to the extracted images/ directory,
        # join directly: "{image_dir}/{subset}/Train/{filename}.jpg"
        full = os.path.join(image_dir, path.replace("images/", "", 1) if path.startswith("images/") else path)
        if os.path.exists(full):
            try:
                return Image.open(full).convert("RGB")
            except Exception as e:
                logger.warning("Failed to load local image %s: %s", full, e)
                return None

    try:
        import requests
        url = _build_hf_image_url(path)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to load image from HF %s: %s", path, e)
        return None


class MMEBDataset(Dataset):
    """
    Wraps a single MMEB-train subset. Each item yields a dict with:
      - query: dict with keys text, image (PIL or None), instruction
      - positive: dict with keys text, image (PIL or None)
      - negative: dict with keys text, image (PIL or None) or None
      - task_type: str (classification, vqa, retrieval)
      - subset_name: str
    """

    def __init__(
        self,
        subset_name: str,
        split: str = "diverse_instruction",
        image_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset

        self.subset_name = subset_name
        self.image_dir = image_dir
        self.task_type = self._infer_task_type(subset_name)

        logger.info("Loading MMEB subset '%s' (split=%s) ...", subset_name, split)
        ds = load_dataset(
            HF_MMEB_REPO, subset_name,
            split=split, streaming=False,
            cache_dir=cache_dir,
        )
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        self.data = ds
        logger.info("  '%s': %d examples loaded (task_type=%s)", subset_name, len(self.data), self.task_type)

    @staticmethod
    def _infer_task_type(name: str) -> str:
        for task_type, subsets in TASK_TYPE_MAP.items():
            if name in subsets:
                return task_type
        return "retrieval"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data[idx]

        qry_text = row["qry"]
        qry_image_path = row.get("qry_image_path", "")
        pos_text = row.get("pos_text", "")
        pos_image_path = row.get("pos_image_path", "")
        neg_text = row.get("neg_text", "")
        neg_image_path = row.get("neg_image_path", "")

        qry_image = _load_image_from_path(qry_image_path, self.image_dir) if qry_image_path else None
        pos_image = _load_image_from_path(pos_image_path, self.image_dir) if pos_image_path else None

        has_neg = bool(neg_text or neg_image_path)
        neg_image = None
        if has_neg and neg_image_path:
            neg_image = _load_image_from_path(neg_image_path, self.image_dir)

        # Strip image placeholder from text when image was successfully loaded
        # (the model's processor injects vision tokens itself).
        # When image failed to load, also strip the placeholder to avoid confusing the tokenizer.
        def _clean(text: str, has_image: bool) -> str:
            if IMAGE_PLACEHOLDER in text:
                text = text.replace(IMAGE_PLACEHOLDER, "").strip()
            return text

        clean_qry_text = _clean(qry_text, qry_image is not None)
        clean_pos_text = _clean(pos_text, pos_image is not None)
        clean_neg_text = _clean(neg_text, neg_image is not None) if has_neg else ""

        query = {"text": clean_qry_text or None, "image": qry_image}
        positive = {"text": clean_pos_text or None, "image": pos_image}
        negative = {"text": clean_neg_text or None, "image": neg_image} if has_neg else None

        return {
            "query": query,
            "positive": positive,
            "negative": negative,
            "task_type": self.task_type,
            "subset_name": self.subset_name,
        }


class STSDataset(Dataset):
    """
    For STS-style datasets with real-valued similarity scores.
    Placeholder for future use; can wrap any dataset yielding
    (sentence_a, sentence_b, score) triples.
    """

    def __init__(self, data: List[Tuple[str, str, float]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        a, b, score = self.data[idx]
        return {
            "query": {"text": a, "image": None},
            "positive": {"text": b, "image": None},
            "negative": None,
            "score": score,
            "task_type": "sts",
            "subset_name": "sts",
        }


def collate_embedding_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a batch of dataset items into training-ready format.

    Returns dict with:
      - queries: list of dicts (text, image, instruction)
      - positives: list of dicts (text, image, instruction)
      - negatives: list of dicts or empty list
      - task_types: list of str
      - scores: tensor of floats (for STS) or None
    """
    queries = []
    positives = []
    negatives = []
    task_types = []
    scores = []
    has_scores = "score" in batch[0]

    default_instruction = "Represent the user's input."

    for item in batch:
        q = item["query"]
        p = item["positive"]

        queries.append({
            "text": q.get("text"),
            "image": q.get("image"),
            "instruction": default_instruction,
        })
        positives.append({
            "text": p.get("text"),
            "image": p.get("image"),
            "instruction": default_instruction,
        })

        if item.get("negative") is not None:
            n = item["negative"]
            negatives.append({
                "text": n.get("text"),
                "image": n.get("image"),
                "instruction": default_instruction,
            })

        task_types.append(item["task_type"])
        if has_scores:
            scores.append(item["score"])

    return {
        "queries": queries,
        "positives": positives,
        "negatives": negatives,
        "task_types": task_types,
        "scores": torch.tensor(scores, dtype=torch.float32) if scores else None,
    }


def build_mmeb_dataset(
    subsets: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    split: str = "diverse_instruction",
    image_dir: Optional[str] = None,
    max_samples_per_subset: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> ConcatDataset:
    """
    Build a concatenated dataset from MMEB-train subsets.

    Args:
        subsets: Specific subset names, or None for all.
        task_types: Filter by task type (classification, vqa, retrieval).
        split: HF dataset split name.
        image_dir: Local directory with pre-downloaded images.
        max_samples_per_subset: Cap per subset for debugging.
        cache_dir: HF datasets cache directory.
    """
    target_subsets = list(ALL_MMEB_SUBSETS)

    if subsets:
        target_subsets = [s for s in target_subsets if s in subsets]
        unknown = set(subsets) - set(ALL_MMEB_SUBSETS)
        if unknown:
            logger.warning("Unknown subsets ignored: %s", unknown)

    if task_types:
        allowed = set()
        for tt in task_types:
            allowed.update(TASK_TYPE_MAP.get(tt, []))
        target_subsets = [s for s in target_subsets if s in allowed]

    logger.info("Will load %d MMEB subsets: %s", len(target_subsets), target_subsets)
    datasets = []
    for i, name in enumerate(target_subsets):
        logger.info("[%d/%d] Loading subset '%s' ...", i + 1, len(target_subsets), name)
        try:
            ds = MMEBDataset(
                subset_name=name,
                split=split,
                image_dir=image_dir,
                max_samples=max_samples_per_subset,
                cache_dir=cache_dir,
            )
            datasets.append(ds)
        except Exception as e:
            logger.error("Failed to load subset '%s': %s", name, e)
            continue

    if not datasets:
        raise RuntimeError("No datasets loaded successfully.")

    combined = ConcatDataset(datasets)
    logger.info("Combined dataset: %d total examples from %d subsets", len(combined), len(datasets))
    return combined


def build_dataloader(
    subsets: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    split: str = "diverse_instruction",
    image_dir: Optional[str] = None,
    max_samples_per_subset: Optional[int] = None,
    cache_dir: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_mmeb_dataset(
        subsets=subsets,
        task_types=task_types,
        split=split,
        image_dir=image_dir,
        max_samples_per_subset=max_samples_per_subset,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_embedding_batch,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
    )
