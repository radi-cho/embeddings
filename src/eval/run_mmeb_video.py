#!/usr/bin/env python3
"""
MMEB-V2 Video evaluation.

Evaluates multimodal embedding models on MMEB-V2 video tasks across four categories:
Video Classification (5), Video Retrieval (5), Video QA (5), Video Moment Retrieval (3).

Consistent with run_mmeb.py for image tasks -- same model loading, embedding strategy,
output format, and CLI interface.

Requirements:
    - Video files must be pre-downloaded to the expected directory structure.
    - Frame extraction is handled automatically (cv2) and cached on disk.
    - Classification labels are loaded from Qwen3-VL-Embedding/src/.../video_classification_utils.py.

Directory structure (under --video_dir, default: datasets/mmeb_cache/video-tasks):
    videos/
        video_cls/{HMDB51,UCF101,K700,Breakfast,SSv2}/              *.mp4
        video_ret/{MSR-VTT,MSVD,DiDeMo,YouCook2,VATEX}/            *.mp4
        video_qa/{NExTQA,Video-MME,EgoSchema,ActivityNetQA}/        *.mp4
        video_qa/MVBench/{tvqa/frames_fps3_hq/,...}/                *.avi,*.mp4
    frames/  (auto-generated cache -- do not pre-populate)
        ...

    Moment retrieval tasks (QVHighlight, Charades-STA, MomentSeeker) require
    pre-extracted frames at frames/video_mret/{task}/ in the directory layout
    produced by the Qwen3-VL-Embedding extraction scripts.

Usage:
    # Quick eval: 1 task per category (~HMDB51, MSR-VTT, NExTQA, QVHighlight)
    python src/eval/run_mmeb_video.py --model_path models/Qwen3-VL-Embedding-2B \\
        --video_dir datasets/mmeb_cache/video-tasks --quick

    # Specific tasks
    python src/eval/run_mmeb_video.py --model_path models/Qwen3-VL-Embedding-2B \\
        --tasks HMDB51 MSR-VTT NExTQA

    # All 18 video tasks
    python src/eval/run_mmeb_video.py --model_path models/Qwen3-VL-Embedding-2B --full

    # List available tasks
    python src/eval/run_mmeb_video.py --list_tasks
"""

import argparse
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def attach_run_log(output_dir: Path) -> None:
    """Append run.log under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", (output_dir / "run.log").resolve())


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MMEB_IMAGE_MIN_PIXELS = 4 * 32 * 32
MMEB_IMAGE_MAX_PIXELS = 1800 * 32 * 32
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_NUM_FRAMES = 64
DEFAULT_MAX_FRAMES_SAVED = 64

# HuggingFace dataset sources (from Qwen3-VL-Embedding constant.py)
HF_DATASETS = {
    # Video-CLS
    "HMDB51":       ("VLM2Vec/HMDB51",       None,      "test"),
    "UCF101":       ("VLM2Vec/UCF101",       None,      "test"),
    "Breakfast":    ("VLM2Vec/Breakfast",     None,      "test"),
    "Kinetics-700": ("VLM2Vec/Kinetics-700", None,      "test"),
    "SmthSmthV2":   ("VLM2Vec/SmthSmthV2",   None,      "test"),
    # Video-RET
    "MSR-VTT":      ("VLM2Vec/MSR-VTT",      "test_1k", "test"),
    "MSVD":         ("VLM2Vec/MSVD",         None,      "test"),
    "DiDeMo":       ("VLM2Vec/DiDeMo",       None,      "test"),
    "YouCook2":     ("lmms-lab/YouCook2",    None,      "val"),
    "VATEX":        ("VLM2Vec/VATEX",        None,      "test"),
    # Video-QA
    "NExTQA":       ("VLM2Vec/NExTQA",       "MC",      "test"),
    "EgoSchema":    ("VLM2Vec/EgoSchema",    "Subset",  "test"),
    "MVBench":      ("VLM2Vec/MVBench",      None,      "train"),
    "Video-MME":    ("VLM2Vec/Video-MME",    None,      "test"),
    "ActivityNetQA":("VLM2Vec/ActivityNetQA", None,     "test"),
    # Video-MRET
    "QVHighlight":  ("VLM2Vec/QVHighlight",  None,      "test"),
    "Charades-STA": ("VLM2Vec/Charades-STA", None,      "test"),
    "MomentSeeker": ("VLM2Vec/MomentSeeker", None,      "test"),
}

# Task configurations (matching Qwen3-VL-Embedding video.yaml)
VIDEO_TASKS = {
    # --- Video-CLS ---
    "HMDB51": {
        "category": "video_cls", "eval_type": "global",
        "video_subdir": "video_cls/HMDB51",
        "instruction": "What actions or objects interactions are the person in the video doing?",
    },
    "UCF101": {
        "category": "video_cls", "eval_type": "global",
        "video_subdir": "video_cls/UCF101",
        "instruction": "What activities or sports are being performed by the person in the video?",
    },
    "Breakfast": {
        "category": "video_cls", "eval_type": "global",
        "video_subdir": "video_cls/Breakfast",
        "instruction": "Recognize the breakfast type that the person is cooking in the video.",
    },
    "Kinetics-700": {
        "category": "video_cls", "eval_type": "global",
        "video_subdir": "video_cls/K700",
        "instruction": "Recognize the category of the video content.",
    },
    "SmthSmthV2": {
        "category": "video_cls", "eval_type": "global",
        "video_subdir": "video_cls/SSv2",
        "instruction": "What actions or object interactions are being performed by the person in the video?",
    },
    # --- Video-RET ---
    "MSR-VTT": {
        "category": "video_ret", "eval_type": "global",
        "video_subdir": "video_ret/MSR-VTT",
        "instruction": "Find a video that contains the following visual content.",
    },
    "MSVD": {
        "category": "video_ret", "eval_type": "global",
        "video_subdir": "video_ret/MSVD",
        "instruction": "Find the video snippet that corresponds to the given summary.",
    },
    "DiDeMo": {
        "category": "video_ret", "eval_type": "global",
        "video_subdir": "video_ret/DiDeMo",
        "instruction": "Find a video that includes the following described scenes.",
    },
    "YouCook2": {
        "category": "video_ret", "eval_type": "global",
        "video_subdir": "video_ret/YouCook2",
        "instruction": "Find a video that demonstrates the following action while making a recipe.",
    },
    "VATEX": {
        "category": "video_ret", "eval_type": "global",
        "video_subdir": "video_ret/VATEX",
        "instruction": "Select a video that fits the description provided.",
    },
    # --- Video-QA ---
    "NExTQA": {
        "category": "video_qa", "eval_type": "local",
        "video_subdir": "video_qa/NExTQA",
        "instruction": (
            "Given a video and a question, select the most accurate answer from "
            "the provided candidates. Return only the exact text of your chosen answer."
        ),
    },
    "EgoSchema": {
        "category": "video_qa", "eval_type": "local",
        "video_subdir": "video_qa/EgoSchema",
        "instruction": (
            "Given a video and a question, select the most accurate answer from "
            "the provided candidates. Return only the exact text of your chosen answer."
        ),
    },
    "MVBench": {
        "category": "video_qa", "eval_type": "local",
        "video_subdir": "video_qa/MVBench",
        "instruction": (
            "Given a video and a question, select the most accurate answer from "
            "the provided candidates. Return only the exact text of your chosen answer."
        ),
    },
    "Video-MME": {
        "category": "video_qa", "eval_type": "local",
        "video_subdir": "video_qa/Video-MME",
        "instruction": (
            "Given a video and a question, select the most accurate answer from "
            "the provided candidates. Return only the exact text of your chosen answer."
        ),
    },
    "ActivityNetQA": {
        "category": "video_qa", "eval_type": "local",
        "video_subdir": "video_qa/ActivityNetQA",
        "instruction": (
            "Given a video and a question, select the most accurate answer from "
            "the provided candidates. Return only the exact text of your chosen answer."
        ),
    },
    # --- Video-MRET ---
    "QVHighlight": {
        "category": "video_mret", "eval_type": "local",
        "video_subdir": "video_mret/QVHighlight",
        "instruction": "Find the clip that corresponds to the described scene in the given video.",
        "num_video_frames": 64, "num_clip_frames": 8,
        "max_video_frames_saved": 64, "max_clip_frames_saved": 8,
    },
    "Charades-STA": {
        "category": "video_mret", "eval_type": "local",
        "video_subdir": "video_mret/Charades-STA",
        "instruction": "Find the clip that corresponds to the described scene in the given video.",
        "num_video_frames": 64, "num_clip_frames": 8,
        "max_video_frames_saved": 64, "max_clip_frames_saved": 8,
    },
    "MomentSeeker": {
        "category": "video_mret", "eval_type": "local",
        "video_subdir": "video_mret/MomentSeeker",
        "instruction": "Find the clip that corresponds to the given text.",
        "num_video_frames": 64,
    },
}

TASK_CATEGORIES = {
    "video_cls":  ["HMDB51", "UCF101", "Breakfast", "Kinetics-700", "SmthSmthV2"],
    "video_ret":  ["MSR-VTT", "MSVD", "DiDeMo", "YouCook2", "VATEX"],
    "video_qa":   ["NExTQA", "EgoSchema", "MVBench", "Video-MME", "ActivityNetQA"],
    "video_mret": ["QVHighlight", "Charades-STA", "MomentSeeker"],
}

QUICK_TASKS = ["HMDB51", "MSR-VTT", "NExTQA", "QVHighlight"]
ALL_TASKS = [t for tasks in TASK_CATEGORIES.values() for t in tasks]

# MVBench subset metadata (from Qwen3-VL-Embedding mvbench_dataset.py)
MVBENCH_SUBSET_META = {
    "episodic_reasoning":       {"video_path": "tvqa/frames_fps3_hq/",          "data_type": "frame"},
    "action_sequence":          {"video_path": "star/Charades_v1_480/",          "data_type": "video"},
    "action_prediction":        {"video_path": "star/Charades_v1_480/",          "data_type": "video"},
    "action_antonym":           {"video_path": "ssv2_video/",                    "data_type": "video"},
    "fine_grained_action":      {"video_path": "Moments_in_Time_Raw/videos/",   "data_type": "video"},
    "unexpected_action":        {"video_path": "FunQA_test/test/",               "data_type": "video"},
    "object_existence":         {"video_path": "clevrer/video_validation/",      "data_type": "video"},
    "object_interaction":       {"video_path": "star/Charades_v1_480/",          "data_type": "video"},
    "object_shuffle":           {"video_path": "perception/videos/",             "data_type": "video"},
    "moving_direction":         {"video_path": "clevrer/video_validation/",      "data_type": "video"},
    "action_localization":      {"video_path": "sta/sta_video/",                 "data_type": "video"},
    "scene_transition":         {"video_path": "scene_qa/video/",                "data_type": "video"},
    "action_count":             {"video_path": "perception/videos/",             "data_type": "video"},
    "moving_count":             {"video_path": "clevrer/video_validation/",      "data_type": "video"},
    "moving_attribute":         {"video_path": "clevrer/video_validation/",      "data_type": "video"},
    "state_change":             {"video_path": "perception/videos/",             "data_type": "video"},
    "fine_grained_pose":        {"video_path": "nturgbd/",                       "data_type": "video"},
    "character_order":          {"video_path": "perception/videos/",             "data_type": "video"},
    "egocentric_navigation":    {"video_path": "vlnqa/",                         "data_type": "video"},
    "counterfactual_inference": {"video_path": "clevrer/video_validation/",      "data_type": "video"},
}


def get_category(task_name):
    for cat, tasks in TASK_CATEGORIES.items():
        if task_name in tasks:
            return cat
    return "unknown"


# ---------------------------------------------------------------------------
# Classification labels (loaded from Qwen3-VL-Embedding reference)
# ---------------------------------------------------------------------------

def _load_cls_labels():
    """Load video classification label mappings from Qwen3-VL-Embedding."""
    utils_path = (
        PROJECT_ROOT / "Qwen3-VL-Embedding" / "src" / "evaluation" / "mmeb_v2"
        / "data" / "datasets" / "video_classification_utils.py"
    )
    if not utils_path.exists():
        raise FileNotFoundError(
            f"Classification labels not found at {utils_path}. "
            "Ensure the Qwen3-VL-Embedding directory is present in the project root."
        )
    spec = importlib.util.spec_from_file_location("_vcls_utils", utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VIDEOCLS_LABEL_MAPPING, mod.DATASET_INSTRUCTION


# ---------------------------------------------------------------------------
# Video processing utilities (ported from Qwen3-VL-Embedding vision_utils.py)
# ---------------------------------------------------------------------------

def load_frames(frames_dir):
    """Load image frame paths from a directory with natural sort."""
    def _key(filename):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", filename)]

    if not os.path.isdir(frames_dir):
        return []
    return [
        os.path.join(frames_dir, f)
        for f in sorted(os.listdir(frames_dir), key=_key)
        if os.path.splitext(f)[-1].lower() in IMAGE_EXTENSIONS
    ]


def sample_frames(frames, num_segments):
    """Uniform temporal sampling via linspace."""
    n = len(frames)
    if n <= num_segments:
        result = list(frames)
        while len(result) < num_segments:
            result.append(frames[-1])
        return result
    indices = np.linspace(0, n - 1, num_segments, dtype=int).tolist()
    return [frames[i] for i in indices]


def process_video_frames(frame_dir, num_frames=None):
    """Load frames from directory and optionally subsample to num_frames."""
    if num_frames == 0:
        return []
    frames = load_frames(frame_dir)
    if not frames:
        return []
    if num_frames is not None and num_frames <= len(frames):
        frames = sample_frames(frames, num_frames)
    return frames


def extract_frames_cv2(video_path, frame_dir, max_frames_saved):
    """Extract up to max_frames_saved uniformly-spaced frames using cv2.

    Same strategy as inline extraction in the Qwen reference dataset loaders
    (nextqa, videomme, egoschema, activitynetqa, mvbench).
    """
    import cv2

    if os.path.exists(frame_dir) and any(
        f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(frame_dir)
    ):
        return
    if not os.path.exists(video_path):
        logger.warning("Video not found: %s", video_path)
        return
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        logger.warning("Zero frames in video: %s", video_path)
        return
    step = max(1, total // max_frames_saved)
    saved = 0
    for i in range(max_frames_saved):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_dir, f"{saved:04d}.jpeg"), frame)
        saved += 1
    cap.release()


def ensure_frames(video_path, frame_dir, max_frames_saved):
    """Extract frames if the cache directory is empty."""
    if load_frames(frame_dir):
        return
    extract_frames_cv2(video_path, frame_dir, max_frames_saved)


def qa_template(question, candidates, answer):
    """Format QA with lettered options.  From Qwen vision_utils.py."""
    q = f"{question}\nOptions:\n"
    answer_idx = -1
    options = []
    for idx, c in enumerate(candidates):
        q += f"({chr(ord('A') + idx)}) {c}\n"
        options.append(f"({chr(ord('A') + idx)}) {c}")
        if c == answer:
            answer_idx = idx
    q = q.rstrip()
    ans_str = f"({chr(ord('A') + answer_idx)}) {answer}" if answer_idx >= 0 else answer
    return q, options, ans_str, answer_idx


# ---------------------------------------------------------------------------
# Model loading  (same pattern as run_mmeb.py)
# ---------------------------------------------------------------------------

def load_model(
    model_path,
    *,
    max_length: int = 16384,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
):
    """Load embedding model, auto-detecting Qwen3-VL vs Qwen3.5."""
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

        class _Embedder(Qwen3VLEmbedder):
            """Pass per-item min/max_pixels for images (upstream ignores these keys)."""

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
                processed = self._preprocess_inputs(conversations)
                processed = {k: v.to(self.model.device) for k, v in processed.items()}
                out = self.forward(processed)
                embs = self._pooling_last(out["last_hidden_state"], out["attention_mask"])
                if normalize:
                    embs = F.normalize(embs, p=2, dim=-1)
                return embs

        logger.info("Loading Qwen3-VL from %s (max_length=%s)", model_path, max_length)
        model = _Embedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            max_length=max_length,
            min_pixels=mip,
            max_pixels=mp,
        )
        return model, "qwen3vl"
    else:
        from src.models.qwen35_embedding import Qwen35Embedder

        logger.info("Loading Qwen3.5 from %s (max_length=%s)", model_path, max_length)
        model = Qwen35Embedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            max_length=max_length,
            min_pixels=mip,
            max_pixels=mp,
        )
        return model, "qwen35"


# ---------------------------------------------------------------------------
# Embedding helpers  (same as run_mmeb.py)
# ---------------------------------------------------------------------------

def embed_batch(model, items, batch_size=8):
    """Embed a list of dicts in micro-batches with CUDA OOM recovery.

    Default batch_size is 8 (lower than image eval) because video inputs are
    much larger in token count.
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
                    "CUDA OOM; retrying chunk=%d (index %d)", chunk, i,
                )
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# HuggingFace dataset helpers
# ---------------------------------------------------------------------------

def _load_hf(task_name):
    from datasets import load_dataset

    repo, subset, split = HF_DATASETS[task_name]
    return load_dataset(repo, subset, split=split)


def _load_hf_mvbench():
    from datasets import load_dataset, concatenate_datasets

    repo, _, split = HF_DATASETS["MVBench"]
    parts = []
    for subset_name in MVBENCH_SUBSET_META:
        ds = load_dataset(repo, subset_name, split=split)
        ds = ds.add_column("subset", [subset_name] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts)


# ---------------------------------------------------------------------------
# Data loading -- Video Classification (global eval)
# ---------------------------------------------------------------------------

def load_video_cls_data(task_name, video_dir, num_frames, max_frames_saved):
    """Returns (queries, corpus, gt_indices) for global hit@1 classification."""
    label_map, instr_map = _load_cls_labels()
    ds = _load_hf(task_name)
    info = VIDEO_TASKS[task_name]
    subdir = info["video_subdir"]
    instruction = info.get("instruction") or instr_map.get(task_name, "Classify the video.")
    if instruction.endswith(":"):
        instruction = instruction[:-1] + "."

    labels = label_map[task_name]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    queries, gt_indices = [], []
    skipped = 0
    for ex in ds:
        vid_id = ex["video_id"]
        vpath = os.path.join(video_dir, "videos", subdir, vid_id + ".mp4")
        fdir = os.path.join(video_dir, "frames", subdir, vid_id)
        ensure_frames(vpath, fdir, max_frames_saved)
        fps = process_video_frames(fdir, num_frames)
        if not fps:
            skipped += 1
            continue
        queries.append({"video": fps, "instruction": instruction})
        gt_indices.append(label_to_idx.get(ex["pos_text"], -1))

    if skipped:
        logger.warning("%s: skipped %d examples (missing frames)", task_name, skipped)

    corpus = [{"text": lab} for lab in labels]
    return queries, corpus, gt_indices


# ---------------------------------------------------------------------------
# Data loading -- Video Retrieval (global eval)
# ---------------------------------------------------------------------------

def _normalize_ret_row(task_name, ex):
    """Return (video_id, video_filename, caption) for a retrieval task row."""
    if task_name == "MSR-VTT":
        return ex["video_id"], ex["video"], ex["caption"]
    if task_name == "MSVD":
        return ex["video_id"], ex["video"], ex["captions"][0]
    if task_name == "DiDeMo":
        vfile = ex["video_path"]
        return os.path.splitext(os.path.basename(vfile))[0], os.path.basename(vfile), ex["caption"]
    if task_name == "YouCook2":
        return ex["id"], os.path.basename(ex["video_path"]), ex["sentence"]
    if task_name == "VATEX":
        return ex["videoID"], ex["videoID"] + ".mp4", ex["enCap"][0]
    raise ValueError(f"Unknown retrieval task: {task_name}")


def load_video_ret_data(task_name, video_dir, num_frames, max_frames_saved):
    """Returns (queries, corpus, gt_indices) for text-to-video retrieval."""
    ds = _load_hf(task_name)
    info = VIDEO_TASKS[task_name]
    subdir = info["video_subdir"]
    instruction = info["instruction"]

    queries, gt_video_ids = [], []
    unique_videos = {}
    skipped = 0

    for ex in ds:
        vid_id, vid_file, caption = _normalize_ret_row(task_name, ex)
        queries.append({"text": caption, "instruction": instruction})
        gt_video_ids.append(vid_id)

        if vid_id not in unique_videos:
            vpath = os.path.join(video_dir, "videos", subdir, vid_file)
            fdir = os.path.join(video_dir, "frames", subdir, vid_id)
            ensure_frames(vpath, fdir, max_frames_saved)
            fps = process_video_frames(fdir, num_frames)
            if fps:
                unique_videos[vid_id] = {"video": fps}
            else:
                skipped += 1
                unique_videos[vid_id] = {"text": ""}  # fallback placeholder

    if skipped:
        logger.warning("%s: %d videos with missing frames", task_name, skipped)

    corpus_ids = list(unique_videos.keys())
    corpus = list(unique_videos.values())
    id_to_idx = {v: i for i, v in enumerate(corpus_ids)}
    gt_indices = [id_to_idx[v] for v in gt_video_ids]
    return queries, corpus, gt_indices


# ---------------------------------------------------------------------------
# Data loading -- Video QA (local eval)
# ---------------------------------------------------------------------------

def load_video_qa_data(task_name, video_dir, num_frames, max_frames_saved):
    """Returns (queries, per_query_cands, answer_indices) for video QA."""
    if task_name == "MVBench":
        return _load_mvbench_data(video_dir, num_frames, max_frames_saved)
    ds = _load_hf(task_name)
    info = VIDEO_TASKS[task_name]
    subdir = info["video_subdir"]
    instruction = info["instruction"]

    queries, per_query_cands, answer_indices = [], [], []
    skipped = 0

    for ex in ds:
        # --- per-task column normalisation ---
        if task_name == "NExTQA":
            vid_id = str(ex["video"])
            vid_prefix = ""
            options = [ex["a0"], ex["a1"], ex["a2"], ex["a3"], ex["a4"]]
            fmt_q, fmt_opts, _, _ = qa_template(ex["question"], options, options[ex["answer"]])
            cands = [{"text": o} for o in fmt_opts]
            ans_idx = ex["answer"]

        elif task_name == "Video-MME":
            vid_id = ex["videoID"]
            vid_prefix = ""
            options = ex["options"]
            fmt_q = ex["question"] + "\n" + "\n".join(options)
            cands = [
                {"text": o[o.find(". "):].strip(". ") if ". " in o else o}
                for o in options
            ]
            ans_idx = ["A", "B", "C", "D"].index(ex["answer"])

        elif task_name == "EgoSchema":
            vid_id = str(ex["video_idx"])
            vid_prefix = ""
            options = ex["option"]
            fmt_q = ex["question"] + " " + " ".join(options)
            cands = [{"text": o[o.find(". "):].strip(". ")} for o in options]
            ans_idx = int(ex["answer"])

        elif task_name == "ActivityNetQA":
            vid_id = "v_" + ex["video_name"]
            vid_prefix = ""
            fmt_q = ex["question"] + "? (A) yes; (B) no."
            cands = [{"text": "yes"}, {"text": "no"}]
            ans_idx = ["yes", "no"].index(ex["answer"])

        else:
            raise ValueError(f"Unknown QA task: {task_name}")

        vpath = os.path.join(video_dir, "videos", subdir, vid_prefix + vid_id + ".mp4")
        fdir = os.path.join(video_dir, "frames", subdir, vid_prefix + vid_id)
        ensure_frames(vpath, fdir, max_frames_saved)
        fps = process_video_frames(fdir, num_frames)
        if not fps:
            skipped += 1
            continue

        queries.append({"video": fps, "text": fmt_q, "instruction": instruction})
        per_query_cands.append(cands)
        answer_indices.append(ans_idx)

    if skipped:
        logger.warning("%s: skipped %d examples (missing frames)", task_name, skipped)
    return queries, per_query_cands, answer_indices


def _load_mvbench_data(video_dir, num_frames, max_frames_saved):
    """MVBench: 20 subsets, mixed video/frame sources."""
    ds = _load_hf_mvbench()
    info = VIDEO_TASKS["MVBench"]
    vid_subdir = info["video_subdir"]
    instruction = info["instruction"]
    video_root = os.path.join(video_dir, "videos", vid_subdir)
    frame_root = os.path.join(video_dir, "frames", vid_subdir)

    queries, per_query_cands, answer_indices = [], [], []
    skipped = 0

    for ex in ds:
        subset = ex["subset"]
        meta = MVBENCH_SUBSET_META.get(subset)
        if meta is None:
            skipped += 1
            continue

        video_filename = ex["video"]
        vpath = os.path.join(video_root, meta["video_path"], video_filename)
        fdir = os.path.join(frame_root, subset, video_filename)

        if meta["data_type"] == "video":
            ensure_frames(vpath, fdir, max_frames_saved)
        elif meta["data_type"] == "frame":
            if not os.path.exists(fdir) and os.path.exists(vpath):
                shutil.copytree(vpath, fdir, dirs_exist_ok=True)

        fps = process_video_frames(fdir, num_frames)
        if not fps:
            skipped += 1
            continue

        fmt_q, fmt_opts, _, ans_idx = qa_template(
            ex["question"], ex["candidates"], ex["answer"],
        )
        queries.append({"video": fps, "text": fmt_q, "instruction": instruction})
        per_query_cands.append([{"text": o} for o in fmt_opts])
        answer_indices.append(ans_idx)

    if skipped:
        logger.warning("MVBench: skipped %d examples", skipped)
    return queries, per_query_cands, answer_indices


# ---------------------------------------------------------------------------
# Data loading -- Video Moment Retrieval (local eval)
# ---------------------------------------------------------------------------

def load_video_mret_data(task_name, video_dir, num_frames, max_frames_saved):
    """Dispatcher for moment retrieval tasks."""
    if task_name == "MomentSeeker":
        return _load_momentseeker_data(video_dir, num_frames)
    return _load_moment_retrieval_data(task_name, video_dir, num_frames, max_frames_saved)


def _load_moment_retrieval_data(task_name, video_dir, num_frames, max_frames_saved):
    """QVHighlight / Charades-STA.

    Expects pre-extracted frames at frames/{subdir}/{video_name}/ with
    subdirectories: query/ (full video), positive*/ (correct clip), and other
    clip directories as negatives.
    """
    ds = _load_hf(task_name)
    info = VIDEO_TASKS[task_name]
    frame_root = os.path.join(video_dir, "frames", info["video_subdir"])
    video_root = os.path.join(video_dir, "videos", info["video_subdir"])
    n_vid_frames = info.get("num_video_frames", num_frames)
    n_clip_frames = info.get("num_clip_frames", 8)
    max_vid_saved = info.get("max_video_frames_saved", max_frames_saved)
    instruction = info["instruction"]

    queries, per_query_cands, answer_indices = [], [], []
    skipped = 0

    for ex in ds:
        video_name = os.path.splitext(os.path.basename(ex["video_path"]))[0]
        frames_dir = os.path.join(frame_root, video_name)

        # Query video frames
        qry_dir = os.path.join(frames_dir, "query")
        if not load_frames(qry_dir):
            vp = os.path.join(video_root, os.path.basename(ex["video_path"]))
            if os.path.exists(vp):
                extract_frames_cv2(vp, qry_dir, max_vid_saved)
        qry_fps = process_video_frames(qry_dir, n_vid_frames)
        if not qry_fps:
            skipped += 1
            continue

        # Clip candidates from subdirectories
        cands = []
        pos_idx = -1
        if os.path.isdir(frames_dir):
            clip_dirs = sorted(
                d for d in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, d)) and d != "query"
            )
            for ci, cname in enumerate(clip_dirs):
                cfps = process_video_frames(os.path.join(frames_dir, cname), n_clip_frames)
                if cfps:
                    cands.append({"video": cfps})
                    if cname.startswith("positive"):
                        pos_idx = ci

        if not cands or pos_idx < 0:
            skipped += 1
            continue

        queries.append({"text": ex["query"], "video": qry_fps, "instruction": instruction})
        per_query_cands.append(cands)
        answer_indices.append(pos_idx)

    if skipped:
        logger.warning("%s: skipped %d examples", task_name, skipped)
    return queries, per_query_cands, answer_indices


def _load_momentseeker_data(video_dir, num_frames):
    """MomentSeeker: multi-modal queries (text / text+image / text+video) and clip candidates."""
    ds = _load_hf("MomentSeeker")
    info = VIDEO_TASKS["MomentSeeker"]
    frame_root = os.path.join(video_dir, "frames", info["video_subdir"])
    video_root = os.path.join(video_dir, "videos", info["video_subdir"])
    n_vid = info.get("num_video_frames", num_frames)

    queries, per_query_cands, answer_indices = [], [], []
    skipped = 0

    for ex in ds:
        inp = ex["input_frames"]
        item = {"text": ex["query"]}

        # Determine query modality
        if inp.endswith(".mp4"):
            vname = inp.split(".mp4")[0].replace("/", "_")
            q_fdir = os.path.join(frame_root, "video_frames", vname)
            if not load_frames(q_fdir):
                vp = os.path.join(video_root, inp)
                if os.path.exists(vp):
                    extract_frames_cv2(vp, q_fdir, n_vid)
            qfps = load_frames(q_fdir)
            if qfps:
                item["video"] = qfps
            item["instruction"] = "Find the clip that corresponds to the given sentence and video segment."
        elif inp.endswith(".jpg"):
            img_path = os.path.join(frame_root, f"query_{inp}")
            if os.path.exists(img_path):
                item["image"] = img_path
            item["instruction"] = "Select the video clip that aligns with the given text and image."
        else:
            item["instruction"] = info["instruction"]

        # Build candidates: positives first, then negatives
        cands = []
        for clip in ex["positive_frames"]:
            cname = clip["output_path"].replace("/", "_").split(".mp4")[0]
            cfdir = os.path.join(frame_root, "video_frames", cname)
            if not load_frames(cfdir):
                cp = os.path.join(video_root, clip["output_path"])
                if os.path.exists(cp):
                    extract_frames_cv2(cp, cfdir, n_vid)
            cfps = load_frames(cfdir)
            if cfps:
                cands.append({"video": cfps})
        n_pos = len(cands)

        for clip in ex["negative_frames"]:
            cname = clip["output_path"].replace("/", "_").split(".mp4")[0]
            cfdir = os.path.join(frame_root, "video_frames", cname)
            if not load_frames(cfdir):
                cp = os.path.join(video_root, clip["output_path"])
                if os.path.exists(cp):
                    extract_frames_cv2(cp, cfdir, n_vid)
            cfps = load_frames(cfdir)
            if cfps:
                cands.append({"video": cfps})

        if not cands or n_pos == 0:
            skipped += 1
            continue

        queries.append(item)
        per_query_cands.append(cands)
        answer_indices.append(0)  # positives are first

    if skipped:
        logger.warning("MomentSeeker: skipped %d examples", skipped)
    return queries, per_query_cands, answer_indices


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_global(model, task_name, queries, corpus, gt_indices, batch_size):
    """Embed all queries & corpus, full sim matrix, hit@1."""
    n = len(queries)
    if not n:
        return None
    logger.info("  Embedding %d queries ...", n)
    qry_embs = embed_batch(model, queries, batch_size)
    logger.info("  Embedding %d corpus items ...", len(corpus))
    corpus_embs = embed_batch(model, corpus, batch_size)

    sims = qry_embs @ corpus_embs.T
    gt = torch.tensor(gt_indices, dtype=torch.long)
    correct = (sims.argmax(dim=1) == gt).sum().item()
    hit1 = correct / n * 100
    logger.info("  => hit@1 = %.2f%%", hit1)
    return {
        "task": task_name,
        "category": get_category(task_name),
        "hit_at_1": round(hit1, 2),
        "num_examples": n,
        "num_candidates": len(corpus),
    }


def evaluate_local(model, task_name, queries, per_query_cands, answer_indices, batch_size):
    """Per-query candidate ranking with deduplication, hit@1.

    Same strategy as run_mmeb.py evaluate_task: deduplicate candidates across
    all queries, embed unique set once, then look up per-query.
    """
    n = len(queries)
    if not n:
        return None

    def _key(c):
        parts = []
        if c.get("text"):
            parts.append(("t", c["text"]))
        if c.get("video"):
            parts.append(("v", tuple(c["video"]) if isinstance(c["video"], list) else c["video"]))
        if c.get("image"):
            parts.append(("i", c["image"]))
        return tuple(parts)

    unique, unique_items = {}, []
    cand_idx_map = []  # per-query list of indices into unique_items
    for cands in per_query_cands:
        ex_idx = []
        for c in cands:
            k = _key(c)
            if k not in unique:
                unique[k] = len(unique_items)
                unique_items.append(c)
            ex_idx.append(unique[k])
        cand_idx_map.append(ex_idx)

    logger.info("  Embedding %d queries ...", n)
    qry_embs = embed_batch(model, queries, batch_size)
    logger.info(
        "  Embedding %d unique candidates (from %d total) ...",
        len(unique_items), sum(len(c) for c in per_query_cands),
    )
    cand_embs = embed_batch(model, unique_items, batch_size)

    correct = 0
    for i in range(n):
        idx = torch.tensor(cand_idx_map[i], dtype=torch.long)
        ex_embs = cand_embs[idx]
        sims = qry_embs[i] @ ex_embs.T
        if sims.argmax().item() == answer_indices[i]:
            correct += 1

    hit1 = correct / n * 100
    logger.info("  => hit@1 = %.2f%%", hit1)
    avg_cands = sum(len(c) for c in per_query_cands) / n
    return {
        "task": task_name,
        "category": get_category(task_name),
        "hit_at_1": round(hit1, 2),
        "num_examples": n,
        "num_candidates": round(avg_cands, 1),
    }


# ---------------------------------------------------------------------------
# Task dispatcher
# ---------------------------------------------------------------------------

def evaluate_task(model, task_name, video_dir, batch_size, num_frames, max_frames_saved):
    """Load data and evaluate one video task."""
    cat = VIDEO_TASKS[task_name]["category"]
    logger.info("--- %s (category=%s) ---", task_name, cat)

    if cat == "video_cls":
        q, c, gt = load_video_cls_data(task_name, video_dir, num_frames, max_frames_saved)
        return evaluate_global(model, task_name, q, c, gt, batch_size)

    if cat == "video_ret":
        q, c, gt = load_video_ret_data(task_name, video_dir, num_frames, max_frames_saved)
        return evaluate_global(model, task_name, q, c, gt, batch_size)

    if cat == "video_qa":
        q, pqc, ai = load_video_qa_data(task_name, video_dir, num_frames, max_frames_saved)
        return evaluate_local(model, task_name, q, pqc, ai, batch_size)

    if cat == "video_mret":
        q, pqc, ai = load_video_mret_data(task_name, video_dir, num_frames, max_frames_saved)
        return evaluate_local(model, task_name, q, pqc, ai, batch_size)

    raise ValueError(f"Unknown category: {cat}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MMEB-V2 video embedding eval")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir (default: results/<model>/mmeb_video/<run>/)")
    parser.add_argument("--video_dir", type=str, default="datasets/mmeb_cache/video-tasks",
                        help="Base directory for video files and frame cache")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true",
                        help=f"Quick subset: {QUICK_TASKS}")
    parser.add_argument("--full", action="store_true", help="All 18 video tasks")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Embed micro-batch size (default 8, lower than image eval)")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Token context cap (Qwen3-VL paper Sec. 6.1: 16384)")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Frames per video for model input (default 64, matching Qwen video.yaml)")
    parser.add_argument("--max_frames_saved", type=int, default=DEFAULT_MAX_FRAMES_SAVED,
                        help="Max frames to extract per video (default 64)")
    parser.add_argument("--image_min_pixels", type=int, default=None)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--list_tasks", action="store_true")
    args = parser.parse_args()

    # --- list tasks ---
    if args.list_tasks:
        for cat, tasks in TASK_CATEGORIES.items():
            label = {
                "video_cls": "Video CLS", "video_ret": "Video RET",
                "video_qa": "Video QA", "video_mret": "Video MRET",
            }.get(cat, cat)
            print(f"\n{label} ({len(tasks)} tasks):")
            for t in tasks:
                ti = VIDEO_TASKS[t]
                print(f"  {t:20s}  eval={ti['eval_type']:6s}  HF={HF_DATASETS[t][0]}")
        print(f"\nTotal: {len(ALL_TASKS)} video tasks")
        print(f"Quick subset: {QUICK_TASKS}")
        return

    # --- eval ---
    if not args.model_path:
        parser.error("--model_path is required for evaluation")

    # Auto output dir
    if not args.output_dir:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        base = Path("results") / model_name / "mmeb_video"
        base.mkdir(parents=True, exist_ok=True)
        existing = [int(d.name) for d in base.iterdir() if d.is_dir() and d.name.isdigit()]
        run_num = max(existing, default=0) + 1
        args.output_dir = str(base / str(run_num))

    attach_run_log(Path(args.output_dir))
    logger.info(
        "MMEB video eval: batch_size=%s, num_frames=%s, max_length=%s",
        args.batch_size, args.num_frames, args.max_length,
    )
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    # Task list
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
    logger.info("Video dir: %s", args.video_dir)

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
                model, task_name, args.video_dir,
                args.batch_size, args.num_frames, args.max_frames_saved,
            )
            if r:
                results.append(r)
            else:
                results.append({
                    "task": task_name, "category": get_category(task_name),
                    "hit_at_1": None, "error": "No valid examples",
                })
        except Exception as e:
            logger.error("FAILED %s: %s", task_name, e, exc_info=True)
            results.append({
                "task": task_name, "category": get_category(task_name),
                "hit_at_1": None, "error": str(e),
            })

    # --- Summarise ---
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
            "num_frames": args.num_frames,
            "max_frames_saved": args.max_frames_saved,
            "image_min_pixels": args.image_min_pixels or MMEB_IMAGE_MIN_PIXELS,
            "image_max_pixels": args.image_max_pixels or MMEB_IMAGE_MAX_PIXELS,
            "video_dir": args.video_dir,
        },
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print table ---
    print()
    print("=" * 65)
    print(f"  MMEB Video Results \u2014 {os.path.basename(args.model_path)}")
    print("=" * 65)

    cat_order = ["video_cls", "video_ret", "video_qa", "video_mret"]
    cat_labels = {
        "video_cls": "Video CLS", "video_ret": "Video RET",
        "video_qa": "Video QA", "video_mret": "Video MRET",
    }
    results_by_cat = {}
    for r in results:
        results_by_cat.setdefault(r["category"], []).append(r)

    for cat in cat_order:
        cat_results = results_by_cat.get(cat, [])
        if not cat_results:
            continue
        label = cat_labels.get(cat, cat.upper())
        print(f"\n  {label}")
        print(f"  {'-' * 50}")
        cat_scores = []
        for r in cat_results:
            score = f"{r['hit_at_1']:5.2f}" if r.get("hit_at_1") is not None else "ERROR"
            print(f"    {r['task']:30s}  {score}")
            if r.get("hit_at_1") is not None:
                cat_scores.append(r["hit_at_1"])
        if cat_scores:
            print(f"    {'':30s}  -----")
            print(f"    {label + ' Mean':30s}  {np.mean(cat_scores):5.2f}")

    # Uncategorised (should not happen)
    for cat, cat_results in results_by_cat.items():
        if cat not in cat_order:
            print(f"\n  {cat.upper()}")
            print(f"  {'-' * 50}")
            for r in cat_results:
                score = f"{r['hit_at_1']:5.2f}" if r.get("hit_at_1") is not None else "ERROR"
                print(f"    {r['task']:30s}  {score}")

    print()
    print("=" * 65)
    if valid:
        cat_means = {}
        for cat in cat_order:
            scores = [
                r["hit_at_1"] for r in results_by_cat.get(cat, [])
                if r.get("hit_at_1") is not None
            ]
            if scores:
                cat_means[cat] = np.mean(scores)
        abbrev = {"video_cls": "CLS", "video_ret": "RET", "video_qa": "QA", "video_mret": "MRET"}
        parts = [f"{abbrev.get(c, c)}: {v:.1f}" for c, v in cat_means.items()]
        print(f"  {' | '.join(parts)}")
        print(f"  Video Overall: {summary['mean_hit_at_1']:.2f}")
    print("=" * 65)
    print(f"\n  Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
