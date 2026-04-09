"""Dataset module for multimodal embedding training.

Data sources:
- MMEB-train (TIGER-Lab/MMEB-train): 20 subsets, VQA/classification/retrieval
- Text triplets: MS MARCO, AllNLI, GooAQ, Quora (JSONL)
- STS-B: sentence pairs with float scores (JSONL)
- MegaPairs: image-text triplets with hard negatives (JSONL annotations)
- ColPali: visual document retrieval (Arrow with embedded images)
- Video: LLaVA-Hound, MSR-VTT (JSONL manifests)
"""

import io
import json
import logging
import os
from pathlib import Path
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
DEFAULT_EMBED_INSTRUCTION = "Represent the user's input."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: str, image_dir: Optional[str] = None) -> Optional[Image.Image]:
    if not path:
        return None
    if image_dir:
        rel = path.replace("images/", "", 1) if path.startswith("images/") else path
        full = os.path.join(image_dir, rel)
        if os.path.exists(full):
            try:
                return Image.open(full).convert("RGB")
            except Exception:
                return None
    try:
        import requests
        url = f"https://huggingface.co/datasets/{HF_MMEB_REPO}/resolve/main/{path}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _clean_mmeb_text(text: str) -> str:
    return text.replace(IMAGE_PLACEHOLDER, "").strip() if IMAGE_PLACEHOLDER in text else text


def _infer_task_type(name: str) -> str:
    for tt, subsets in TASK_TYPE_MAP.items():
        if name in subsets:
            return tt
    return "retrieval"


def _filter_subsets(subsets, task_types):
    target = list(ALL_MMEB_SUBSETS)
    if subsets:
        target = [s for s in target if s in subsets]
    if task_types:
        allowed = {s for tt in task_types for s in TASK_TYPE_MAP.get(tt, [])}
        target = [s for s in target if s in allowed]
    return target


def _load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class MMEBDataset(Dataset):
    """Single MMEB-train subset."""

    def __init__(self, subset_name: str, split="diverse_instruction",
                 image_dir=None, max_samples=None, cache_dir=None):
        from datasets import load_dataset
        self.subset_name = subset_name
        self.image_dir = image_dir
        self.task_type = _infer_task_type(subset_name)
        ds = load_dataset(HF_MMEB_REPO, subset_name, split=split,
                          streaming=False, cache_dir=cache_dir)
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        self.data = ds
        logger.info("MMEB '%s': %d examples (task=%s)", subset_name, len(ds), self.task_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        qimg = _load_image(r.get("qry_image_path", ""), self.image_dir)
        pimg = _load_image(r.get("pos_image_path", ""), self.image_dir)
        has_neg = bool(r.get("neg_text") or r.get("neg_image_path"))
        nimg = _load_image(r.get("neg_image_path", ""), self.image_dir) if has_neg else None
        return {
            "query": {"text": _clean_mmeb_text(r["qry"]) or None, "image": qimg},
            "positive": {"text": _clean_mmeb_text(r.get("pos_text", "")) or None, "image": pimg},
            "negative": {"text": _clean_mmeb_text(r.get("neg_text", "")) or None,
                         "image": nimg} if has_neg else None,
            "task_type": self.task_type,
            "subset_name": self.subset_name,
        }


class STSDataset(Dataset):
    """STS pairs with float scores from JSONL ({sentence1, sentence2, score})."""

    def __init__(self, jsonl_path: str, subset_name="stsb"):
        self.subset_name = subset_name
        self.data = [(r["sentence1"], r["sentence2"], float(r["score"]))
                     for r in _load_jsonl(jsonl_path)]
        logger.info("STSDataset '%s': %d rows", subset_name, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b, score = self.data[idx]
        return {"query": {"text": a, "image": None},
                "positive": {"text": b, "image": None},
                "negative": None, "score": score,
                "task_type": "sts", "subset_name": self.subset_name}


class TextTripletDataset(Dataset):
    """Text-only triplets from JSONL ({query, positive, negative?, task_type?})."""

    def __init__(self, jsonl_path: str, subset_name=None, max_samples=None):
        self.subset_name = subset_name or Path(jsonl_path).parent.name
        self.data = _load_jsonl(jsonl_path, max_samples)
        logger.info("TextTriplet '%s': %d rows", self.subset_name, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        neg = r.get("negative")
        return {"query": {"text": r["query"], "image": None},
                "positive": {"text": r["positive"], "image": None},
                "negative": {"text": neg, "image": None} if neg else None,
                "task_type": r.get("task_type", "retrieval"),
                "subset_name": self.subset_name}


class MegaPairsDataset(Dataset):
    """MegaPairs image-text triplets (JSONL: {q_img, q_texts, t_img, hns})."""

    def __init__(self, jsonl_path: str, image_dir=None, max_samples=None):
        self.image_dir = image_dir
        self.data = _load_jsonl(jsonl_path, max_samples)
        logger.info("MegaPairs: %d rows", len(self.data))

    def __len__(self):
        return len(self.data)

    def _img(self, path):
        if not path or not self.image_dir:
            return None
        full = os.path.join(self.image_dir, path)
        try:
            return Image.open(full).convert("RGB") if os.path.exists(full) else None
        except Exception:
            return None

    def __getitem__(self, idx):
        r = self.data[idx]
        texts = r.get("q_texts") or r.get("q_text") or []
        hns = r.get("hns", [])
        neg_path = hns[1] if len(hns) > 1 else (hns[0] if hns else "")
        neg_img = self._img(neg_path)
        t_img = self._img(r.get("t_img", ""))
        q_img = self._img(r.get("q_img", ""))
        return {"query": {"text": texts[0] if texts else None, "image": q_img},
                "positive": {"text": None, "image": t_img or q_img},
                "negative": {"text": None, "image": neg_img} if neg_img else None,
                "task_type": "retrieval", "subset_name": "MegaPairs"}


class ColPaliDataset(Dataset):
    """ColPali visual document retrieval (query text + page image, Arrow)."""

    def __init__(self, data_dir: str, max_samples=None):
        from datasets import load_from_disk
        self.data = load_from_disk(data_dir)
        if max_samples and max_samples < len(self.data):
            self.data = self.data.select(range(max_samples))
        logger.info("ColPali: %d rows", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        img = r.get("image")
        if img is not None and not isinstance(img, Image.Image):
            try:
                img = Image.open(io.BytesIO(img)).convert("RGB")
            except Exception:
                img = None
        return {"query": {"text": r.get("query", ""), "image": None},
                "positive": {"text": None, "image": img},
                "negative": None,
                "task_type": "retrieval", "subset_name": "ColPali"}


class VideoTripletDataset(Dataset):
    """Video-text pairs from JSONL ({query_text, positive_text, video_path})."""

    def __init__(self, jsonl_path: str, video_root=None, subset_name=None, max_samples=None):
        self.subset_name = subset_name or Path(jsonl_path).parent.name
        self.video_root = video_root
        self.data = _load_jsonl(jsonl_path, max_samples)
        logger.info("VideoTriplet '%s': %d rows", self.subset_name, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        vp = r.get("video_path", "")
        if self.video_root and vp:
            vp = os.path.join(self.video_root, vp)
        return {"query": {"text": r.get("query_text"), "image": None,
                          "video": vp or None},
                "positive": {"text": r.get("positive_text"), "image": None},
                "negative": None,
                "task_type": r.get("task_type", "retrieval"),
                "subset_name": self.subset_name}


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_embedding_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate items into {queries, positives, negatives, task_types, scores}.

    Items without hard negatives get None (no fake placeholders).
    """
    queries, positives, negatives, task_types, scores = [], [], [], [], []
    has_scores = any("score" in item for item in batch)
    has_neg = False
    inst = DEFAULT_EMBED_INSTRUCTION

    for item in batch:
        for role, src in [("queries", "query"), ("positives", "positive")]:
            d = item[src]
            (queries if role == "queries" else positives).append({
                "text": d.get("text"), "image": d.get("image"),
                "video": d.get("video"), "instruction": inst})

        neg = item.get("negative")
        if neg is not None:
            negatives.append({"text": neg.get("text"), "image": neg.get("image"),
                              "video": neg.get("video"), "instruction": inst})
            has_neg = True
        else:
            negatives.append(None)

        task_types.append(item["task_type"])
        if has_scores:
            scores.append(item.get("score", 0.0))

    return {"queries": queries, "positives": positives,
            "negatives": negatives if has_neg else None,
            "task_types": task_types,
            "scores": torch.tensor(scores, dtype=torch.float32) if scores else None}


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------

def build_mmeb_dataset(subsets=None, task_types=None, split="diverse_instruction",
                       image_dir=None, max_samples_per_subset=None, cache_dir=None):
    target = _filter_subsets(subsets, task_types)
    logger.info("Loading %d MMEB subsets", len(target))
    parts = []
    for i, name in enumerate(target):
        logger.info("[%d/%d] %s", i + 1, len(target), name)
        try:
            parts.append(MMEBDataset(name, split=split, image_dir=image_dir,
                                     max_samples=max_samples_per_subset, cache_dir=cache_dir))
        except Exception as e:
            logger.error("Failed '%s': %s", name, e)
    if not parts:
        raise RuntimeError("No MMEB subsets loaded")
    combined = ConcatDataset(parts)
    logger.info("MMEB total: %d examples from %d subsets", len(combined), len(parts))
    return combined


def _make_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_embedding_batch,
                      num_workers=num_workers, drop_last=True, pin_memory=False)


def build_dataloader(subsets=None, task_types=None, split="diverse_instruction",
                     image_dir=None, max_samples_per_subset=None, cache_dir=None,
                     batch_size=4, num_workers=0, shuffle=True):
    ds = build_mmeb_dataset(subsets, task_types, split, image_dir,
                            max_samples_per_subset, cache_dir)
    return _make_dataloader(ds, batch_size, num_workers, shuffle)


def build_mixed_dataloader(data_dir: str, image_dir=None,
                           mmeb_split="diverse_instruction",
                           max_samples_per_subset=None, cache_dir=None,
                           batch_size=4, num_workers=0, shuffle=True):
    """ConcatDataset from all sources under data_dir + MMEB from HF."""
    base = Path(data_dir)
    parts: List[Dataset] = []

    try:
        parts.append(build_mmeb_dataset(split=mmeb_split, image_dir=image_dir,
                     max_samples_per_subset=max_samples_per_subset, cache_dir=cache_dir))
    except Exception as e:
        logger.error("MMEB-train: %s", e)

    for name in ["msmarco", "allnli", "gooaq", "quora"]:
        p = base / name / "train.jsonl"
        if p.is_file():
            parts.append(TextTripletDataset(str(p), name, max_samples_per_subset))

    p = base / "stsb" / "train.jsonl"
    if p.is_file():
        parts.append(STSDataset(str(p)))

    p = base / "megapairs" / "train.jsonl"
    if p.is_file():
        try:
            parts.append(MegaPairsDataset(str(p), image_dir, max_samples_per_subset))
        except Exception as e:
            logger.error("MegaPairs: %s", e)

    p = base / "colpali" / "data"
    if p.is_dir():
        try:
            parts.append(ColPaliDataset(str(p), max_samples_per_subset))
        except Exception as e:
            logger.error("ColPali: %s", e)

    p = base / "llava_hound" / "train.jsonl"
    if p.is_file():
        parts.append(VideoTripletDataset(str(p), subset_name="llava_hound"))

    vr = base / "video_retrieval"
    if vr.is_dir():
        for jf in sorted(vr.glob("*.jsonl")):
            parts.append(VideoTripletDataset(str(jf), subset_name=jf.stem))

    if not parts:
        raise RuntimeError(f"No data under {data_dir}")

    combined = ConcatDataset(parts)
    logger.info("Mixed total: %d examples from %d sources", len(combined), len(parts))
    return _make_dataloader(combined, batch_size, num_workers, shuffle)
