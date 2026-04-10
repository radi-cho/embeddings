#!/usr/bin/env python3
"""MMTEB evaluation for Qwen3.5 embedding models and Qwen3-VL-Embedding baselines.

Usage:
  python eval/run_mmteb.py --model_path models/checkpoints/Qwen3.5-0.8B --output_dir results/qwen35
  python eval/run_mmteb.py --model_path Qwen/Qwen3-VL-Embedding-2B --output_dir results/qwen3vl
  python eval/run_mmteb.py --model_path models/checkpoints/Qwen3.5-0.8B --full --output_dir results/full
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def attach_run_log(output_dir: Path) -> None:
    """Write a run.log under output_dir mirroring all root-logger output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path.resolve())

QUICK_TASK = "STSBenchmark"

# Paper-style STS: `--sts` uses `get_tasks(task_types=["STS"])` (full multilingual MMTEB STS suite).
# For English-only or harness-specific subsets, use `--benchmark MTEB(eng, v2)` or `--tasks BIOSSES ...`.

FAST_TASKS = [
    "BUCC.v2",
    "AmazonCounterfactualClassification",
    "ArXivHierarchicalClusteringP2P",
    "IFIRAila",
    "MultiEURLEXMultilabelClassification",
    "LegalBenchPC",
    "AskUbuntuDupQuestions",
    "AILACasedocs",
    "STSBenchmark",
]


def detect_model_type(model_path: str) -> str:
    """Classify checkpoint for MTEB embedder choice (Hub id or local path).

    Uses local config.json when present; otherwise `AutoConfig.from_pretrained` so Hub
    models are detected without relying on name substrings alone.
    """
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.is_file():
        with open(config_path) as f:
            model_type = json.load(f).get("model_type", "")
    else:
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", "") or ""
        except Exception:
            model_type = ""

    mt = model_type.lower()
    if "qwen3_5" in mt or "qwen3.5" in mt:
        return "qwen3.5"
    if "qwen3_vl" in mt:
        return "qwen3vl"

    if "Qwen3-VL" in model_path or "qwen3-vl" in model_path.lower():
        return "qwen3vl"
    if "Qwen3.5" in model_path or "qwen3_5" in model_path.lower() or "qwen3.5" in model_path.lower():
        return "qwen3.5"
    return "qwen3.5"


def load_qwen35_embedder(model_path: str, **kwargs):
    from models.qwen35_embedding import Qwen35Embedder
    lora_path = kwargs.get("lora_path", None)
    base_path = kwargs.get("base_model_path", model_path)

    if lora_path:
        embedder = Qwen35Embedder(
            model_name_or_path=base_path,
            torch_dtype=torch.bfloat16,
            max_length=kwargs.get("max_length", 8192),
        )
        from peft import PeftModel
        embedder.model = PeftModel.from_pretrained(embedder.model, lora_path)
        embedder.model.eval()
        return embedder

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            acfg = json.load(f)
        real_base = acfg.get("base_model_name_or_path", "models/checkpoints/Qwen3.5-0.8B")
        embedder = Qwen35Embedder(
            model_name_or_path=real_base,
            torch_dtype=torch.bfloat16,
            max_length=kwargs.get("max_length", 8192),
        )
        from peft import PeftModel
        embedder.model = PeftModel.from_pretrained(embedder.model, model_path)
        embedder.model.eval()
        return embedder

    return Qwen35Embedder(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        max_length=kwargs.get("max_length", 8192),
    )


def load_qwen3vl_embedder(model_path: str, **kwargs):
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    except ImportError:
        raise ImportError("Qwen3-VL model classes not available in this transformers version")

    try:
        from qwen_vl_utils.vision_process import process_vision_info
    except ImportError:
        process_vision_info = None

    from dataclasses import dataclass
    from transformers.modeling_outputs import ModelOutput
    from transformers.cache_utils import Cache
    from typing import Optional, Union

    @dataclass
    class _EmbOutput(ModelOutput):
        last_hidden_state: Optional[torch.FloatTensor] = None
        attention_mask: Optional[torch.Tensor] = None

    class _Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.model.set_input_embeddings(value)

        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, pixel_values=None,
                    pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
                    cache_position=None, **kwargs):
            outputs = self.model(
                input_ids=input_ids, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                position_ids=position_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                cache_position=cache_position, **kwargs,
            )
            return _EmbOutput(last_hidden_state=outputs.last_hidden_state, attention_mask=attention_mask)

    import torch.nn.functional as F
    import unicodedata

    class _Qwen3VLEmbedder:
        PAD_TOKEN = "<|endoftext|>"

        def __init__(self, model_name_or_path, max_length=8192, **kw):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.max_length = max_length
            self.default_instruction = "Represent the user's input."
            self.model = _Qwen3VLForEmbedding.from_pretrained(
                model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ).to(device)
            self.processor = Qwen3VLProcessor.from_pretrained(model_name_or_path, padding_side="right")
            self.model.eval()

        @staticmethod
        def _pool_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            flipped = attention_mask.flip(dims=[1])
            last_pos = flipped.argmax(dim=1)
            col = attention_mask.shape[1] - last_pos - 1
            row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
            return hidden_state[row, col]

        def _format_turn(self, ele) -> list:
            instruction = ele.get("instruction", self.default_instruction)
            if instruction:
                instruction = instruction.strip()
                if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
                    instruction += "."
            content = []
            text = ele.get("text")
            if text:
                content.append({"type": "text", "text": text})
            if not content:
                content.append({"type": "text", "text": "NULL"})
            return [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content},
            ]

        def process(self, inputs, normalize=True):
            # Match HF `scripts/qwen3_vl_embedding.py`: chat template + optional vision preprocessing.
            conversations = [self._format_turn(ele) for ele in inputs]
            text = self.processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=False,
            )
            if isinstance(text, str):
                text = [text]

            if process_vision_info is not None:
                try:
                    images, video_inputs, video_kwargs = process_vision_info(
                        conversations, image_patch_size=16,
                        return_video_metadata=True, return_video_kwargs=True,
                    )
                except Exception as e:
                    logger.warning("process_vision_info failed (%s); falling back to text-only processor path", e)
                    images, video_inputs, video_kwargs = None, None, {"do_sample_frames": False}
                if video_inputs is not None:
                    videos, video_metadata = zip(*video_inputs)
                    videos, video_metadata = list(videos), list(video_metadata)
                else:
                    videos, video_metadata = None, None
                processed = self.processor(
                    text=text,
                    images=images,
                    videos=videos,
                    video_metadata=video_metadata,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    do_resize=False,
                    return_tensors="pt",
                    **video_kwargs,
                )
            else:
                processed = self.processor(
                    text=text, truncation=True, max_length=self.max_length,
                    padding=True, return_tensors="pt",
                )
            processed = {k: v.to(self.model.device) for k, v in processed.items()}
            with torch.no_grad():
                outputs = self.model(**processed)

            hidden = outputs.last_hidden_state
            mask = processed.get("attention_mask")
            if mask is None:
                raise RuntimeError("Qwen3-VL embedding forward missing attention_mask")
            embs = self._pool_last(hidden, mask)
            if normalize:
                embs = F.normalize(embs, p=2, dim=-1)
            return embs

    return _Qwen3VLEmbedder(model_path, max_length=kwargs.get("max_length", 8192))


_TASK_PROMPTS: dict = None  # lazy-loaded from task_prompts.json


def _load_task_prompts() -> dict:
    global _TASK_PROMPTS
    if _TASK_PROMPTS is None:
        prompts_path = Path(__file__).resolve().parent / "task_prompts.json"
        if prompts_path.exists():
            with open(prompts_path) as f:
                _TASK_PROMPTS = json.load(f)
            logger.info("Loaded %d task prompts from %s", len(_TASK_PROMPTS), prompts_path)
        else:
            logger.warning("task_prompts.json not found at %s; using MTEB metadata only", prompts_path)
            _TASK_PROMPTS = {}
    return _TASK_PROMPTS


def _encode_batch_size_for_task(task_name: str, default: int) -> int:
    """Per-task encode micro-batch; keep defaults except known OOM-heavy tasks."""
    if task_name in ("STS22", "STS22.v2"):
        return 16
    return default


class QwenEmbeddingEncoder:
    """MTEB-compatible encoder wrapping either Qwen3.5 or Qwen3-VL embedders."""

    def __init__(self, embedder, batch_size: int = 32, model_name: str = "qwen-embedding"):
        self.embedder = embedder
        self.batch_size = batch_size
        from mteb.models.model_meta import ModelMeta
        meta = ModelMeta.create_empty()
        try:
            self.mteb_model_meta = meta.model_copy(update={"name": model_name})
        except AttributeError:
            meta.name = model_name
            self.mteb_model_meta = meta

    @staticmethod
    def _instruction_for_task(task_metadata, prompt_type) -> str:
        """Mirror Qwen3-Embedding evaluation/qwen3_embedding_model.py:get_instruction.

        Priority (matching upstream exactly):
          1. task_prompts.json lookup by task name (instruction_dict in Qwen's code).
             Entries can be a string (used for both sides) or a dict with
             "query"/"passage" keys (symmetric task).
          2. MTEB task metadata prompt (super().get_instruction in Qwen's code).
          3. Type-based hard overrides (always applied after step 1/2):
             - Retrieval doc side  → "" (unless symmetric task from step 1)
             - STS / PairClassification → "Retrieve semantically similar text"
             - BitextMining        → "Retrieve parallel sentences"
          4. Retrieval query fallback when nothing else matched →
             "Retrieval relevant passage for the given query."
          5. Whatever instruction was resolved (may be None/"" → embedder default).
        """
        from mteb.types import PromptType
        import mteb as _mteb

        task_name = getattr(task_metadata, "name", "")
        task_type = getattr(task_metadata, "type", "") or ""
        prompts = _load_task_prompts()

        # Step 1: task_prompts.json (Qwen instruction_dict)
        instruction = None
        sym_task = False
        if task_name in prompts:
            entry = prompts[task_name]
            if isinstance(entry, dict):
                pt_key = "query" if prompt_type == PromptType.query else "passage"
                instruction = entry.get(pt_key, "")
                sym_task = True
            else:
                instruction = str(entry)

        # Step 2: fallback to MTEB task metadata (super().get_instruction)
        if instruction is None:
            prompt_field = getattr(task_metadata, "prompt", None)
            if isinstance(prompt_field, str) and prompt_field.strip():
                instruction = prompt_field.strip()
            elif isinstance(prompt_field, dict):
                pt_key = "query" if prompt_type == PromptType.query else "passage"
                val = prompt_field.get(pt_key) or prompt_field.get("document")
                if val:
                    instruction = str(val).strip()

        # Step 3: type-based overrides (always win, matching upstream)
        if "Retrieval" in task_type and not sym_task and prompt_type != PromptType.query:
            return ""

        if task_type in ("STS", "PairClassification"):
            return "Retrieve semantically similar text"

        if task_type == "BitextMining":
            return "Retrieve parallel sentences"

        # Step 4: retrieval query fallback
        if "Retrieval" in task_type and prompt_type == PromptType.query and instruction is None:
            return "Retrieval relevant passage for the given query."

        return instruction or ""

    def encode(self, inputs: DataLoader, *, task_metadata, hf_split, hf_subset,
               prompt_type=None, **kwargs) -> np.ndarray:
        show_progress_bar = kwargs.get("show_progress_bar", True)
        micro_batch = int(kwargs.get("batch_size", self.batch_size))

        all_texts = [text for batch in inputs for text in batch["text"]]

        instruction = self._instruction_for_task(task_metadata, prompt_type)

        all_embeddings = []
        i, n = 0, len(all_texts)
        task_label = getattr(task_metadata, "name", "?")
        pbar = tqdm(
            total=n,
            desc=f"Encode {task_label}",
            unit="seq",
            disable=not show_progress_bar,
            leave=False,
            dynamic_ncols=True,
        )
        try:
            while i < n:
                chunk = min(micro_batch, n - i)
                while chunk >= 1:
                    batch_texts = all_texts[i : i + chunk]
                    items = [{"text": t, "instruction": instruction} for t in batch_texts]
                    try:
                        with torch.no_grad():
                            embs = self.embedder.process(items, normalize=True)
                        all_embeddings.append(embs.cpu().float().numpy())
                        i += chunk
                        pbar.update(chunk)
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
                            "CUDA OOM encoding %s (%s); retrying with chunk size %d (start index %d)",
                            task_label,
                            hf_subset,
                            chunk,
                            i,
                        )
        finally:
            pbar.close()

        return np.concatenate(all_embeddings, axis=0)

    def similarity(self, emb1, emb2):
        if isinstance(emb1, np.ndarray):
            emb1 = torch.from_numpy(emb1)
        if isinstance(emb2, np.ndarray):
            emb2 = torch.from_numpy(emb2)
        return (emb1 @ emb2.T).numpy()

    def similarity_pairwise(self, emb1, emb2):
        if isinstance(emb1, np.ndarray):
            emb1 = torch.from_numpy(emb1)
        if isinstance(emb2, np.ndarray):
            emb2 = torch.from_numpy(emb2)
        return (emb1 * emb2).sum(dim=-1).numpy()


def run_eval(args):
    output_dir = Path(args.output_dir)
    attach_run_log(output_dir)

    model_type = detect_model_type(args.model_path)
    logger.info(f"Detected model type: {model_type} for {args.model_path}")

    if args.batch_size is None:
        batch_size = 32
    else:
        batch_size = args.batch_size
    logger.info("Encode batch_size=%s (CUDA OOM halves micro-batch until 1)", batch_size)

    if model_type == "qwen3vl":
        embedder = load_qwen3vl_embedder(args.model_path, max_length=args.max_length)
    else:
        embedder = load_qwen35_embedder(args.model_path, max_length=args.max_length)

    encoder = QwenEmbeddingEncoder(embedder, batch_size=batch_size)

    import mteb
    if args.quick:
        tasks = mteb.get_tasks(tasks=[QUICK_TASK])
    elif args.sts:
        tasks = mteb.get_tasks(task_types=["STS"])
    elif args.benchmark:
        bench = mteb.get_benchmark(args.benchmark)
        tasks = list(bench.tasks)
    elif args.full:
        text_types = ["BitextMining", "Classification", "Clustering", "InstructionRetrieval",
                       "MultilabelClassification", "PairClassification", "Reranking", "Retrieval", "STS"]
        tasks = mteb.get_tasks(task_types=text_types, languages=["eng"])
    elif args.tasks:
        tasks = mteb.get_tasks(tasks=args.tasks)
    else:
        tasks = mteb.get_tasks(tasks=FAST_TASKS)

    tasks = list(tasks)
    logger.info(f"Running {len(tasks)} tasks")

    output_dir.mkdir(parents=True, exist_ok=True)

    from mteb.results import ModelResult

    task_results_acc: list = []
    exceptions_acc: list | None = None
    model_name: str | None = None
    model_revision: str | None = None

    tasks_bar = tqdm(tasks, desc="MMTEB tasks", dynamic_ncols=True)
    for ti, task in enumerate(tasks_bar):
        tname = task.metadata.name
        task_bs = _encode_batch_size_for_task(tname, batch_size)
        tasks_bar.set_postfix_str(tname[:40] + ("…" if len(tname) > 40 else ""))
        logger.info(
            "MMTEB task %d/%d: %s (encode batch_size=%s)",
            ti + 1,
            len(tasks),
            tname,
            task_bs,
        )
        encode_kwargs = {"batch_size": task_bs, "show_progress_bar": True}
        _res = mteb.evaluate(
            encoder,
            task,
            overwrite_strategy="always",
            encode_kwargs=encode_kwargs,
            show_progress_bar=False,
        )
        task_results_acc.extend(_res.task_results)
        if _res.exceptions:
            exceptions_acc = (exceptions_acc or []) + list(_res.exceptions)
        if model_name is None:
            model_name = _res.model_name
            model_revision = _res.model_revision

    results = ModelResult(
        model_name=model_name or "unknown",
        model_revision=model_revision,
        task_results=task_results_acc,
        exceptions=exceptions_acc,
    )

    scores_by_type = {}
    per_task_scores = {}
    all_scores = []
    for task_result in results.task_results:
        task_name = task_result.task.metadata.name
        task_type = task_result.task.metadata.type
        task_means = []
        for split_results in task_result.scores.values():
            for score_dict in split_results:
                main_score = score_dict.get("main_score", None)
                if main_score is not None:
                    scores_by_type.setdefault(task_type, []).append(main_score)
                    all_scores.append(main_score)
                    task_means.append(main_score)
        if task_means:
            per_task_scores[task_name] = {
                "type": task_type,
                "score": float(np.mean(task_means) * 100),
                "num_subsets": len(task_means),
            }

    print("\n" + "=" * 60)
    print(f"MMTEB Results for: {args.model_path}")
    print("=" * 60)
    if per_task_scores:
        print("\n  Per-task (mean over hf subsets run)")
        for tn in sorted(per_task_scores):
            row = per_task_scores[tn]
            print(f"    {tn:40s}: {row['score']:6.2f}  ({row['num_subsets']} subsets, {row['type']})")
        print()
    for tt in sorted(scores_by_type):
        scores = scores_by_type[tt]
        print(f"  {tt:35s}: {np.mean(scores)*100:.2f} ({len(scores)} subset scores)")
    if all_scores:
        print(f"  {'Mean (all subset scores)':35s}: {np.mean(all_scores)*100:.2f}")
    if scores_by_type:
        type_means = [np.mean(v) for v in scores_by_type.values()]
        print(f"  {'Mean (Type)':35s}: {np.mean(type_means)*100:.2f}")
    print("=" * 60)

    summary = {
        "model_path": args.model_path,
        "model_type": model_type,
        "batch_size": batch_size,
        "encode_batch_size_overrides": {"STS22": 16, "STS22.v2": 16},
        "num_tasks": len(per_task_scores),
        "mean_task": float(np.mean(all_scores) * 100) if all_scores else 0,
        "mean_type": float(np.mean([np.mean(v) for v in scores_by_type.values()]) * 100) if scores_by_type else 0,
        "per_type": {k: float(np.mean(v) * 100) for k, v in scores_by_type.items()},
        "per_task": per_task_scores,
        "eval_note": (
            "STS: AnySTSEvaluator (main_score typically cosine_spearman). "
            "Paper-style: STS/PairClassification use symmetric 'Retrieve semantically similar text.' instruction; "
            "Retrieval doc side uses empty instruction; BitextMining uses 'Retrieve parallel sentences.'. "
            "--sts runs all registered MMTEB task_types=STS (multilingual)."
        ),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MMTEB evaluation for Qwen embedding models")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/mmteb")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Encoder micro-batch size (default: 32; halves on CUDA OOM until chunk size 1)",
    )
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--quick", action="store_true", help="Run only STSBenchmark (fastest comparison)")
    parser.add_argument(
        "--sts",
        action="store_true",
        help="Run all MMTEB STS tasks (multilingual; paper-style instruction, not English-only list)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Official MTEB benchmark name, e.g. MTEB(eng, v2) (applies subset filters like hf_subsets on tasks)",
    )
    parser.add_argument("--full", action="store_true", help="Run full MMTEB (all English tasks)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific task names to run")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
