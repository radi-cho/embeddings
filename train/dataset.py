"""Dataset module for multimodal embedding training.

Two dataset builders:
  - `build_mixed_dataset`:  Stage 1 (large, weakly supervised; in-batch negs)
  - `build_stage2_dataset`: Stage 2 (curated; mined K hard-negatives per query +
                             MMEB classification wrong-class labels + AllNLI
                             contradictions + STS-B for CoSENT)

Data sources:
- MMEB-train (TIGER-Lab/MMEB-train): 20 subsets, VQA/classification/retrieval
- Text triplets: MS MARCO, AllNLI, GooAQ, Quora (JSONL)
- STS-B: sentence pairs with float scores (JSONL)
- MegaPairs: image-text pairs (JSONL + local images)    [Stage 1 only]
- ColPali: visual document retrieval (Arrow + embedded images)
- Video: LLaVA-Hound / MSRVTT (JSONL + frame dirs)      [Stage 1 only]
- Mined hard-negatives: scripts/mine_hard_negatives.py output (JSONL)
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

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
DEFAULT_INSTRUCTION = "Represent the user's input."
FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Query-side instructions (positives / hard negatives keep DEFAULT_INSTRUCTION in collate).
CLASSIFICATION_INSTRUCTIONS = (
    "Represent the given image for classification.",
    "Classify this image into one of the categories.",
)
INSTRUCTION_VQA = "Represent the given image with the following question."
INSTRUCTION_T2I = "Find an image that matches the given caption."
INSTRUCTION_I2T = "Represent the given image for retrieval."
INSTRUCTION_MSCOCO_GROUND = "Locate the described region in the image."
INSTRUCTION_IMAGE_PAIR = "Find a similar image."
INSTRUCTION_TEXT_RETRIEVAL = "Given a query, retrieve a relevant passage."
INSTRUCTION_SEMANTIC_TEXT = "Retrieve semantically similar text."
INSTRUCTION_COLPALI = "Retrieve a document page relevant to the query."
INSTRUCTION_VIDEO = "Represent the video with the following text for retrieval."

# Stage-1 optimized mix: MMEB subsets and per-source caps (~5.128M total).
STAGE1_RETRIEVAL_SUBSETS = [
    "MSCOCO", "MSCOCO_i2t", "MSCOCO_t2i",
    "VisualNews_i2t", "VisualNews_t2i",
    "CIRR", "NIGHTS", "VisDial", "WebQA",
]
STAGE1_VQA_SUBSETS = [
    "OK-VQA", "DocVQA", "ChartQA", "Visual7W",
    "InfographicsVQA", "A-OKVQA",
]
STAGE1_CLASSIFICATION_SUBSETS = list(TASK_TYPE_MAP["classification"])

STAGE1_CAP_MEGAPAIRS = 1_500_000
STAGE1_CAP_MMEB_RETRIEVAL_TOTAL = 550_000
STAGE1_CAP_MSMARCO = 500_000
STAGE1_CAP_GOOAQ = 500_000
STAGE1_CAP_MMEB_CLASS_TOTAL = 300_000
STAGE1_CAP_MMEB_VQA_TOTAL = 250_000
STAGE1_CAP_COLPALI = 118_000
STAGE1_CAP_ALLNLI = 300_000
STAGE1_CAP_QUORA = 100_000
STAGE1_CAP_LAVA_HOUND = 255_000
STAGE1_CAP_MINED_RETRIEVAL_TOTAL = 650_000
STAGE1_CAP_MINED_CLASSIFICATION = 100_000


def _split_budget(total: int, n: int) -> List[int]:
    if n <= 0:
        return []
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def mmeb_query_instruction(
    subset_name: str,
    row_index: int,
    qry_raw: str,
    qry_image_path: Optional[str],
    pos_text: str,
    pos_image_path: Optional[str],
) -> str:
    """Instruction for the query side of an MMEB row (paper-aligned templates)."""
    tt = _infer_task_type(subset_name)
    q_text = _clean_mmeb_text(qry_raw) if qry_raw else ""
    p_text = _clean_mmeb_text(pos_text) if pos_text else ""
    q_has_path = bool(qry_image_path and str(qry_image_path).strip())
    p_has_path = bool(pos_image_path and str(pos_image_path).strip())

    if tt == "classification":
        return CLASSIFICATION_INSTRUCTIONS[row_index % len(CLASSIFICATION_INSTRUCTIONS)]
    if tt == "vqa":
        return INSTRUCTION_VQA

    if subset_name in ("MSCOCO_t2i", "VisualNews_t2i"):
        return INSTRUCTION_T2I
    if subset_name in ("MSCOCO_i2t", "VisualNews_i2t"):
        return INSTRUCTION_I2T
    if subset_name == "MSCOCO":
        return INSTRUCTION_MSCOCO_GROUND

    # Generic retrieval: infer from modalities.
    if q_has_path and p_has_path:
        return INSTRUCTION_IMAGE_PAIR
    if q_has_path and p_text and not p_has_path:
        return INSTRUCTION_I2T
    if p_has_path and q_text and not q_has_path:
        return INSTRUCTION_T2I
    if q_has_path:
        return INSTRUCTION_I2T
    return INSTRUCTION_T2I


def text_triplet_query_instruction(subset_name: str) -> str:
    s = (subset_name or "").lower()
    if s in ("msmarco", "gooaq"):
        return INSTRUCTION_TEXT_RETRIEVAL
    if s in ("allnli", "quora"):
        return INSTRUCTION_SEMANTIC_TEXT
    return INSTRUCTION_TEXT_RETRIEVAL


def mined_row_query_instruction(task_type: str, subset_name: str, row_index: int) -> str:
    """Template for mined JSONL rows (subset names may be 'mmeb_OK-VQA', etc.)."""
    tt = task_type or "retrieval"
    # Normalize mmeb_* stem
    name = subset_name or ""
    if name.startswith("mmeb_"):
        name = name[len("mmeb_") :]
    if name in ALL_MMEB_SUBSETS:
        # Use classification rotation for cls; else reuse MMEB rules with text-only hints.
        if tt == "classification":
            return CLASSIFICATION_INSTRUCTIONS[row_index % len(CLASSIFICATION_INSTRUCTIONS)]
        if tt == "vqa" or name in TASK_TYPE_MAP.get("vqa", []):
            return INSTRUCTION_VQA
        return mmeb_query_instruction(
            name, row_index, "", None, "", None,
        )
    if tt == "classification":
        return CLASSIFICATION_INSTRUCTIONS[row_index % len(CLASSIFICATION_INSTRUCTIONS)]
    if name.lower() == "colpali":
        return INSTRUCTION_COLPALI
    if name.lower() in ("msmarco", "gooaq"):
        return INSTRUCTION_TEXT_RETRIEVAL
    if name.lower() in ("quora", "allnli"):
        return INSTRUCTION_SEMANTIC_TEXT
    return INSTRUCTION_TEXT_RETRIEVAL


def fallback_query_instruction(item: Dict[str, Any], row_index: int) -> str:
    """If a dataset did not set query['instruction'], infer from metadata."""
    tt = item.get("task_type", "retrieval")
    sn = item.get("subset_name", "")
    if tt == "sts":
        return INSTRUCTION_SEMANTIC_TEXT
    if sn == "MegaPairs":
        return INSTRUCTION_IMAGE_PAIR
    if sn == "ColPali":
        return INSTRUCTION_COLPALI
    if sn in ("llava_hound", "msrvtt", "video_retrieval"):
        return INSTRUCTION_VIDEO
    if sn in ("msmarco", "gooaq", "allnli", "quora"):
        return text_triplet_query_instruction(sn)
    if sn in ALL_MMEB_SUBSETS:
        return mmeb_query_instruction(sn, row_index, "", None, "", None)
    return DEFAULT_INSTRUCTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: str, image_dir: Optional[str] = None) -> Optional[Image.Image]:
    if not path or not image_dir:
        return None
    rel = path.replace("images/", "", 1) if path.startswith("images/") else path
    full = os.path.join(image_dir, rel)
    try:
        return Image.open(full).convert("RGB") if os.path.isfile(full) else None
    except Exception:
        return None


def _resolve_video(path: str) -> Any:
    if not path or not os.path.isdir(path):
        return path or None
    frames = sorted(
        f.path for f in os.scandir(path)
        if f.is_file() and os.path.splitext(f.name)[1].lower() in FRAME_EXTS
    )
    return frames if frames else None


def _clean_mmeb_text(text: str) -> str:
    return text.replace(IMAGE_PLACEHOLDER, "").strip() if IMAGE_PLACEHOLDER in text else text


def _infer_task_type(name: str) -> str:
    for tt, subsets in TASK_TYPE_MAP.items():
        if name in subsets:
            return tt
    return "retrieval"


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
    def __init__(self, subset_name, split="diverse_instruction",
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
        q_img_path = r.get("qry_image_path", "")
        p_img_path = r.get("pos_image_path", "")
        q_inst = mmeb_query_instruction(
            self.subset_name, idx, r.get("qry", ""), q_img_path,
            r.get("pos_text", ""), p_img_path,
        )
        out = {
            "query": {"text": _clean_mmeb_text(r["qry"]) or None,
                      "image": _load_image(q_img_path, self.image_dir),
                      "instruction": q_inst},
            "positive": {"text": _clean_mmeb_text(r.get("pos_text", "")) or None,
                         "image": _load_image(p_img_path, self.image_dir)},
            "task_type": self.task_type, "subset_name": self.subset_name,
        }
        if self.task_type == "classification":
            neg_text = r.get("neg_text")
            if neg_text and isinstance(neg_text, str) and neg_text.strip():
                out["negative_texts"] = [_clean_mmeb_text(neg_text)]
            elif neg_text and isinstance(neg_text, list) and len(neg_text) > 0:
                out["negative_texts"] = [_clean_mmeb_text(t) for t in neg_text if t]
        return out


class STSDataset(Dataset):
    def __init__(self, jsonl_path, subset_name="stsb"):
        self.subset_name = subset_name
        self.data = [(r["sentence1"], r["sentence2"], float(r["score"]))
                     for r in _load_jsonl(jsonl_path)]
        logger.info("STS '%s': %d rows", subset_name, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b, score = self.data[idx]
        return {"query": {"text": a, "instruction": INSTRUCTION_SEMANTIC_TEXT},
                "positive": {"text": b},
                "score": score, "task_type": "sts", "subset_name": self.subset_name}


class TextTripletDataset(Dataset):
    def __init__(self, jsonl_path, subset_name=None, max_samples=None):
        self.subset_name = subset_name or Path(jsonl_path).parent.name
        self.data = _load_jsonl(jsonl_path, max_samples)
        logger.info("TextTriplet '%s': %d rows", self.subset_name, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        out = {"query": {"text": r["query"],
                         "instruction": text_triplet_query_instruction(self.subset_name)},
               "positive": {"text": r["positive"]},
               "task_type": r.get("task_type", "retrieval"),
               "subset_name": self.subset_name}
        neg = r.get("negative") or r.get("hard_negative")
        if isinstance(neg, str) and neg.strip():
            out["hard_negatives"] = [{"text": neg, "image": None, "video": None}]
        elif isinstance(neg, list) and neg:
            out["hard_negatives"] = [{"text": t, "image": None, "video": None}
                                      for t in neg if isinstance(t, str) and t.strip()]
        return out


class MinedNegativesDataset(Dataset):
    """Rows of mined hard-negative JSONL produced by scripts/mine_hard_negatives.py.

    Each line:
      {"query":{"text":..,"image_path":..},
       "positive":{"text":..,"image_path":..},
       "hard_negatives":[{"text":..,"image_path":..}, ...],
       "task_type": "...", "subset_name": "..."}
    """

    def __init__(self, jsonl_path, image_dir=None, max_samples=None,
                 max_hard_negatives=15, subset_name=None):
        self.jsonl_path = str(jsonl_path)
        self.image_dir = image_dir
        self.max_hn = max_hard_negatives
        self.subset_name = subset_name or Path(jsonl_path).stem
        self.data = _load_jsonl(self.jsonl_path, max_samples)
        logger.info("MinedNegatives '%s': %d rows", self.subset_name, len(self.data))

    def __len__(self):
        return len(self.data)

    def _side(self, side):
        text = side.get("text")
        img = None
        p = side.get("image_path")
        if p and self.image_dir:
            img = _load_image(p, self.image_dir)
        return {"text": text, "image": img, "video": None}

    def __getitem__(self, idx):
        r = self.data[idx]
        hns_raw = r.get("hard_negatives") or []
        hns = [self._side(h) for h in hns_raw[: self.max_hn]]
        q = self._side(r["query"])
        q["instruction"] = mined_row_query_instruction(
            r.get("task_type", "retrieval"),
            r.get("subset_name", self.subset_name),
            idx,
        )
        return {
            "query": q,
            "positive": self._side(r["positive"]),
            "hard_negatives": hns,
            "task_type": r.get("task_type", "retrieval"),
            "subset_name": r.get("subset_name", self.subset_name),
        }


class MegaPairsDataset(Dataset):
    def __init__(self, jsonl_path, image_dir=None, max_samples=None):
        self.image_dir = image_dir
        self.data = _load_jsonl(jsonl_path, max_samples)
        logger.info("MegaPairs: %d rows", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        texts = r.get("q_texts") or r.get("q_text") or []
        q_img = _load_image(r.get("q_img", ""), self.image_dir)
        t_img = _load_image(r.get("t_img", ""), self.image_dir)
        return {"query": {"text": texts[0] if texts else None, "image": q_img,
                         "instruction": INSTRUCTION_IMAGE_PAIR},
                "positive": {"image": t_img or q_img},
                "task_type": "retrieval", "subset_name": "MegaPairs"}


class ColPaliDataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
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
        return {"query": {"text": r.get("query", ""),
                         "instruction": INSTRUCTION_COLPALI},
                "positive": {"image": img},
                "task_type": "retrieval", "subset_name": "ColPali"}


class VideoTripletDataset(Dataset):
    def __init__(self, jsonl_path, video_root=None, subset_name=None, max_samples=None):
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
        return {"query": {"text": r.get("query_text"), "video": _resolve_video(vp),
                         "instruction": INSTRUCTION_VIDEO},
                "positive": {"text": r.get("positive_text")},
                "task_type": r.get("task_type", "retrieval"),
                "subset_name": self.subset_name}


# ---------------------------------------------------------------------------
# Task-stratified sampler
# ---------------------------------------------------------------------------

class TaskStratifiedSampler(Sampler):
    """Yields batches where all items share the same task_type.

    The dataset must be a ConcatDataset of sub-datasets that each expose a
    `.task_type` attribute (MMEBDataset, MinedNegativesDataset, etc.) or a
    constant via the item dict.  We precompute a task_type index at init by
    scanning each sub-dataset.

    Each epoch: shuffle within each task group, round-robin across groups
    proportional to their size, drop remainder per group. Supports DDP via
    num_replicas / rank.
    """

    def __init__(self, dataset: ConcatDataset, batch_size: int,
                 num_replicas: int = 1, rank: int = 0,
                 seed: int = 42, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Build index: global_idx -> task_type
        self.task_indices: Dict[str, List[int]] = {}
        offset = 0
        for sub_ds in dataset.datasets:
            tt = getattr(sub_ds, "task_type", None)
            if tt is None:
                # MinedNegativesDataset doesn't have a single task_type;
                # peek at the first item's task_type.
                try:
                    tt = sub_ds[0].get("task_type", "retrieval")
                except Exception:
                    tt = "retrieval"
            n = len(sub_ds)
            self.task_indices.setdefault(tt, []).extend(range(offset, offset + n))
            offset += n

        logger.info("TaskStratifiedSampler: %s",
                     {k: len(v) for k, v in self.task_indices.items()})

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        import random as _rng
        g = _rng.Random(self.seed + self.epoch)

        # Shuffle each task group
        groups = {}
        for tt, indices in self.task_indices.items():
            shuffled = list(indices)
            g.shuffle(shuffled)
            groups[tt] = shuffled

        # Shard per rank
        for tt in groups:
            idx = groups[tt]
            per_rank = len(idx) // self.num_replicas
            start = self.rank * per_rank
            groups[tt] = idx[start : start + per_rank]

        # Build batches within each group
        all_batches = []
        bs = self.batch_size
        for tt, idx in groups.items():
            for i in range(0, len(idx) - (bs - 1), bs):
                all_batches.append(idx[i : i + bs])
            if not self.drop_last and len(idx) % bs != 0:
                all_batches.append(idx[-(len(idx) % bs):])

        # Shuffle batches across groups so task order is random
        g.shuffle(all_batches)

        for batch in all_batches:
            yield from batch

    def __len__(self):
        total = 0
        bs = self.batch_size
        for idx in self.task_indices.values():
            per_rank = len(idx) // self.num_replicas
            total += per_rank // bs
        return total * bs


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_embedding_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries, positives, task_types, scores = [], [], [], []
    has_scores = any("score" in item for item in batch)
    negative_texts: List[Optional[List[str]]] = []
    hard_negatives_per_item: List[List[Dict[str, Any]]] = []

    for i, item in enumerate(batch):
        d = item["query"]
        q_inst = d.get("instruction") or fallback_query_instruction(item, i)
        queries.append({"text": d.get("text"), "image": d.get("image"),
                        "video": d.get("video"), "instruction": q_inst})
        d = item["positive"]
        positives.append({"text": d.get("text"), "image": d.get("image"),
                          "video": d.get("video"), "instruction": DEFAULT_INSTRUCTION})
        task_types.append(item["task_type"])
        if has_scores:
            scores.append(item.get("score", 0.0))
        negative_texts.append(item.get("negative_texts"))
        hns = item.get("hard_negatives") or []
        hard_negatives_per_item.append([
            {"text": h.get("text"), "image": h.get("image"),
             "video": h.get("video"), "instruction": DEFAULT_INSTRUCTION}
            for h in hns
        ])

    out = {"queries": queries, "positives": positives,
           "task_types": task_types,
           "scores": torch.tensor(scores, dtype=torch.float32) if scores else None}

    if any(nt is not None for nt in negative_texts):
        out["negative_texts"] = negative_texts

    # Pad hard-negative lists to the batch-max K so the downstream loss sees
    # a uniform (B, K, D) tensor. Pad with a sentinel (text="NULL") which gets
    # masked out by the false-negative margin filter in masked_infonce_loss.
    max_k = max((len(hns) for hns in hard_negatives_per_item), default=0)
    if max_k > 0:
        sentinel = {"text": "NULL", "image": None, "video": None,
                    "instruction": DEFAULT_INSTRUCTION}
        padded = []
        for hns in hard_negatives_per_item:
            if len(hns) == 0:
                padded.append([dict(sentinel) for _ in range(max_k)])
            elif len(hns) < max_k:
                # repeat the last one to reach max_k (still a hard neg, just dupe)
                padded.append(list(hns) + [hns[-1]] * (max_k - len(hns)))
            else:
                padded.append(hns[:max_k])
        out["hard_negatives"] = padded
    return out


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------

def build_mmeb_dataset(subsets=None, task_types=None, split="diverse_instruction",
                       image_dir=None, max_samples_per_subset=None, cache_dir=None):
    target = list(ALL_MMEB_SUBSETS)
    if subsets:
        target = [s for s in target if s in subsets]
    if task_types:
        allowed = {s for tt in task_types for s in TASK_TYPE_MAP.get(tt, [])}
        target = [s for s in target if s in allowed]
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


def build_dataloader(subsets=None, task_types=None, split="diverse_instruction",
                     image_dir=None, max_samples_per_subset=None, cache_dir=None,
                     batch_size=4, num_workers=0, shuffle=True):
    ds = build_mmeb_dataset(subsets, task_types, split, image_dir,
                            max_samples_per_subset, cache_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_embedding_batch,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=True, persistent_workers=num_workers > 0)


def build_mixed_dataset(data_dir, image_dir=None, megapairs_image_dir=None,
                        mmeb_split="diverse_instruction",
                        max_samples_per_subset=None, cache_dir=None):
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
            parts.append(MegaPairsDataset(str(p), megapairs_image_dir or image_dir,
                                          max_samples_per_subset))
        except Exception as e:
            logger.error("MegaPairs: %s", e)

    p = base / "colpali" / "data"
    if p.is_dir():
        try:
            parts.append(ColPaliDataset(str(p), max_samples_per_subset))
        except Exception as e:
            logger.error("ColPali: %s", e)

    for name, jpath in [("llava_hound", base / "llava_hound" / "train.jsonl")]:
        if jpath.is_file():
            parts.append(VideoTripletDataset(str(jpath), video_root=str(base),
                                             subset_name=name))

    vr = base / "video_retrieval"
    if vr.is_dir():
        for jf in sorted(vr.glob("*.jsonl")):
            parts.append(VideoTripletDataset(str(jf), video_root=str(base),
                                             subset_name=jf.stem))

    if not parts:
        raise RuntimeError(f"No data under {data_dir}")

    combined = ConcatDataset(parts)
    logger.info("Mixed total: %d examples from %d sources", len(combined), len(parts))
    return combined


def build_stage1_optimized_dataset(
    data_dir,
    image_dir=None,
    megapairs_image_dir=None,
    mined_dir=None,
    mmeb_split="diverse_instruction",
    cache_dir=None,
):
    """Balanced ~5.128M-sample Stage-1 mix (instruction-aware; mined HNs optional).

    If ``mined_dir`` is missing or empty, mined chunks are skipped (logged).
    """
    base = Path(data_dir)
    mined_root = Path(mined_dir) if mined_dir else None
    parts: List[Dataset] = []

    # --- MegaPairs ---------------------------------------------------------
    p = base / "megapairs" / "train.jsonl"
    if p.is_file():
        try:
            parts.append(MegaPairsDataset(
                str(p), megapairs_image_dir or image_dir, STAGE1_CAP_MEGAPAIRS))
        except Exception as e:
            logger.error("MegaPairs (optimized): %s", e)

    # --- MMEB retrieval (subset budget) ------------------------------------
    r_caps = _split_budget(STAGE1_CAP_MMEB_RETRIEVAL_TOTAL, len(STAGE1_RETRIEVAL_SUBSETS))
    for name, cap in zip(STAGE1_RETRIEVAL_SUBSETS, r_caps):
        try:
            parts.append(MMEBDataset(
                name, split=mmeb_split, image_dir=image_dir,
                max_samples=cap, cache_dir=cache_dir))
        except Exception as e:
            logger.error("MMEB retrieval '%s': %s", name, e)

    # --- MMEB VQA ----------------------------------------------------------
    v_caps = _split_budget(STAGE1_CAP_MMEB_VQA_TOTAL, len(STAGE1_VQA_SUBSETS))
    for name, cap in zip(STAGE1_VQA_SUBSETS, v_caps):
        try:
            parts.append(MMEBDataset(
                name, split=mmeb_split, image_dir=image_dir,
                max_samples=cap, cache_dir=cache_dir))
        except Exception as e:
            logger.error("MMEB VQA '%s': %s", name, e)

    # --- MMEB classification -----------------------------------------------
    c_caps = _split_budget(STAGE1_CAP_MMEB_CLASS_TOTAL, len(STAGE1_CLASSIFICATION_SUBSETS))
    for name, cap in zip(STAGE1_CLASSIFICATION_SUBSETS, c_caps):
        try:
            parts.append(MMEBDataset(
                name, split=mmeb_split, image_dir=image_dir,
                max_samples=cap, cache_dir=cache_dir))
        except Exception as e:
            logger.error("MMEB cls '%s': %s", name, e)

    # --- Text triplets -----------------------------------------------------
    p = base / "msmarco" / "train.jsonl"
    if p.is_file():
        parts.append(TextTripletDataset(str(p), "msmarco", STAGE1_CAP_MSMARCO))
    p = base / "gooaq" / "train.jsonl"
    if p.is_file():
        parts.append(TextTripletDataset(str(p), "gooaq", STAGE1_CAP_GOOAQ))
    p = base / "allnli" / "train.jsonl"
    if p.is_file():
        parts.append(TextTripletDataset(str(p), "allnli", STAGE1_CAP_ALLNLI))
    p = base / "quora" / "train.jsonl"
    if p.is_file():
        parts.append(TextTripletDataset(str(p), "quora", STAGE1_CAP_QUORA))

    # --- STS-B (full file; typically ~5.7k) ---------------------------------
    p = base / "stsb" / "train.jsonl"
    if p.is_file():
        parts.append(STSDataset(str(p)))

    # --- ColPali -----------------------------------------------------------
    p = base / "colpali" / "data"
    if p.is_dir():
        try:
            parts.append(ColPaliDataset(str(p), STAGE1_CAP_COLPALI))
        except Exception as e:
            logger.error("ColPali (optimized): %s", e)

    # --- Video (LLaVA-Hound) ----------------------------------------------
    p = base / "llava_hound" / "train.jsonl"
    if p.is_file():
        parts.append(VideoTripletDataset(
            str(p), video_root=str(base), subset_name="llava_hound",
            max_samples=STAGE1_CAP_LAVA_HOUND))

    # NOTE: Mined hard negatives are NOT included in Stage 1 (paper §4.1).
    # Stage 1 relies on in-batch negatives only. Mined HNs are for Stage 2.

    if not parts:
        raise RuntimeError(f"No data for optimized Stage-1 mix under {data_dir}")

    combined = ConcatDataset(parts)
    logger.info("Stage1 optimized total: %d examples from %d sources",
                len(combined), len(parts))
    return combined


# ---------------------------------------------------------------------------
# Stage-2 mixed dataset: mined hard-negatives + unmined originals
# ---------------------------------------------------------------------------

def build_stage2_dataset(
    data_dir, mined_dir="/data/training_data_mined",
    image_dir=None, mmeb_split="diverse_instruction",
    max_samples_per_subset=None, cache_dir=None,
    max_hard_negatives=15,
):
    """Stage-2 dataset (paper §4.3): mined hard-negatives + curated originals.

    Sources:
      - MinedNegativesDataset for each <name>.jsonl under mined_dir  (K hard negs)
      - MMEBDataset for classification subsets (use neg_cand as wrong-class neg)
      - TextTripletDataset for AllNLI (uses its built-in `negative`, 1 HN)
      - STSDataset for STS-B  (CoSENT loss path)

    EXCLUDES (already in Stage-1, no new signal from in-batch-only):
      MegaPairs, LLaVA-Hound, MSRVTT / other video retrieval datasets.
    """
    base = Path(data_dir)
    mined = Path(mined_dir)
    parts: List[Dataset] = []

    # --- Mined hard-neg datasets (MMEB retrieval/VQA + text + ColPali) -----
    # MMEB mined files may be named mmeb_<subset>.jsonl; the row's own
    # subset_name / task_type field still takes precedence at __getitem__,
    # the MinedNegativesDataset.subset_name is just for logging.
    if mined.is_dir():
        for fp in sorted(mined.glob("*.jsonl")):
            if fp.stat().st_size == 0:
                continue
            sname = fp.stem[len("mmeb_"):] if fp.stem.startswith("mmeb_") else fp.stem
            try:
                parts.append(MinedNegativesDataset(
                    str(fp), image_dir=image_dir,
                    max_samples=max_samples_per_subset,
                    max_hard_negatives=max_hard_negatives,
                    subset_name=sname))
            except Exception as e:
                logger.error("MinedNegatives '%s': %s", fp.name, e)
    logger.info("Stage2: loaded %d mined source(s)", len(parts))

    # --- MMEB classification (wrong-class negatives via neg_cand) ----------
    for name in TASK_TYPE_MAP["classification"]:
        try:
            parts.append(MMEBDataset(
                name, split=mmeb_split, image_dir=image_dir,
                max_samples=max_samples_per_subset, cache_dir=cache_dir))
        except Exception as e:
            logger.error("MMEB cls '%s': %s", name, e)

    # --- AllNLI triplets (native `negative` -> 1 HN per row) --------------
    p = base / "allnli" / "train.jsonl"
    if p.is_file():
        parts.append(TextTripletDataset(str(p), "allnli", max_samples_per_subset))

    # --- STS-B (CoSENT) ---------------------------------------------------
    p = base / "stsb" / "train.jsonl"
    if p.is_file():
        parts.append(STSDataset(str(p)))

    if not parts:
        raise RuntimeError(f"No data under {data_dir} or {mined_dir}")

    combined = ConcatDataset(parts)
    logger.info("Stage2 total: %d examples from %d sources",
                len(combined), len(parts))
    return combined
