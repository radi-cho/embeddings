"""Shared utilities for MMEB / MMTEB evaluation scripts.

Extracted from run_mmeb.py, run_mmeb_video.py, run_mmeb_visdoc.py to
eliminate duplication. These functions are semantically frozen — do not
modify their behavior without re-validating all benchmark results.
"""

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def attach_run_log(output_dir: Path) -> None:
    """Append timestamped run.log under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", (output_dir / "run.log").resolve())


def detect_model_type(model_path) -> str:
    """Return 'qwen3vl' or 'qwen35' based on config.json or path heuristics."""
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        with open(config_path) as f:
            model_type = json.load(f).get("model_type", "")
    if not model_type:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
            model_type = getattr(cfg, "model_type", "") or ""
        except Exception:
            pass

    path_str = str(model_path)
    if "qwen3_vl" in model_type.lower():
        return "qwen3vl"
    if "Qwen3-VL" in path_str or "qwen3-vl" in path_str.lower():
        return "qwen3vl"
    return "qwen35"


def load_qwen3vl_embedder_class():
    """Import Qwen3VLEmbedder from local scripts or vendor submodule."""
    vendor = PROJECT_ROOT / "Qwen3-VL-Embedding" / "src" / "models" / "qwen3_vl_embedding.py"
    if vendor.is_file():
        name = "_qwen3_vl_embedding_vendor"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, vendor)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        return sys.modules[name].Qwen3VLEmbedder
    raise ImportError(f"Cannot find Qwen3VLEmbedder (tried {vendor})")


def load_model(
    model_path,
    *,
    max_length: int = 16384,
    default_min_pixels: int = 1,
    default_max_pixels: int = 1843200,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
):
    """Load embedding model, auto-detecting Qwen3-VL vs Qwen3.5.

    Returns (model, model_type_str). The model has a .process(items) method.
    For Qwen3-VL, wraps in an inner class that respects per-item pixel overrides.
    """
    mt = detect_model_type(model_path)
    mip = default_min_pixels if image_min_pixels is None else image_min_pixels
    mp = default_max_pixels if image_max_pixels is None else image_max_pixels

    if mt == "qwen3vl":
        scripts_dir = Path(model_path) / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        try:
            from qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError:
            Qwen3VLEmbedder = load_qwen3vl_embedder_class()

        class _Embedder(Qwen3VLEmbedder):
            """Applies per-item min/max_pixels overrides for MMEB evaluation."""
            def process(self, inputs, normalize=True):
                conversations = []
                for ele in inputs:
                    conv = self.format_model_input(
                        text=ele.get("text"), image=ele.get("image"),
                        video=ele.get("video"), instruction=ele.get("instruction"),
                        fps=ele.get("fps"), max_frames=ele.get("max_frames"))
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
        return _Embedder(model_name_or_path=model_path, torch_dtype=torch.bfloat16,
                         max_length=max_length, min_pixels=mip, max_pixels=mp), "qwen3vl"
    else:
        sys.path.insert(0, str(PROJECT_ROOT))
        from models.qwen35_embedding import Qwen35Embedder
        logger.info("Loading Qwen3.5 from %s (max_length=%s)", model_path, max_length)
        return Qwen35Embedder(model_name_or_path=model_path, torch_dtype=torch.bfloat16,
                              max_length=max_length, min_pixels=mip, max_pixels=mp), "qwen35"


def embed_batch(model, items, batch_size=32):
    """Embed a list of dicts in micro-batches with CUDA OOM recovery.

    Halves chunk size on OOM down to 1. Returns (N, D) float32 tensor on CPU.
    """
    all_embs = []
    i, n = 0, len(items)
    while i < n:
        chunk = min(batch_size, n - i)
        while chunk >= 1:
            try:
                with torch.no_grad():
                    embs = model.process(items[i : i + chunk])
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
                logger.warning("CUDA OOM; retrying chunk=%d (index %d)", chunk, i)
    return torch.cat(all_embs, dim=0)
