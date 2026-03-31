#!/usr/bin/env python3
"""MMTEB evaluation for Qwen3.5 embedding models and Qwen3-VL-Embedding baselines.

Usage:
  python src/eval/run_mmteb.py --model_path models/Qwen3.5-0.8B --output_dir results/qwen35
  python src/eval/run_mmteb.py --model_path Qwen/Qwen3-VL-Embedding-2B --output_dir results/qwen3vl
  python src/eval/run_mmteb.py --model_path models/Qwen3.5-0.8B --full --output_dir results/full
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUICK_TASK = "STSBenchmark"

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
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        if "qwen3_5" in model_type:
            return "qwen3.5"
        if "qwen3_vl" in model_type:
            return "qwen3vl"
    if "Qwen3-VL" in model_path or "qwen3-vl" in model_path.lower():
        return "qwen3vl"
    if "Qwen3.5" in model_path or "qwen3_5" in model_path.lower() or "qwen3.5" in model_path.lower():
        return "qwen3.5"
    return "qwen3.5"


def load_qwen35_embedder(model_path: str, **kwargs):
    from src.models.qwen35_embedding import Qwen35Embedder
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
        real_base = acfg.get("base_model_name_or_path", "models/Qwen3.5-0.8B")
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

        def process(self, inputs, normalize=True):
            conversations = []
            for ele in inputs:
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
                conv = [
                    {"role": "system", "content": [{"type": "text", "text": instruction}]},
                    {"role": "user", "content": content},
                ]
                conversations.append(conv)

            texts = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
            if isinstance(texts, str):
                texts = [texts]
            processed = self.processor(
                text=texts, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            )
            processed = {k: v.to(self.model.device) for k, v in processed.items()}
            with torch.no_grad():
                outputs = self.model(**processed)

            hidden = outputs.last_hidden_state
            mask = outputs.attention_mask
            flipped = mask.flip(dims=[1])
            last_pos = flipped.argmax(dim=1)
            col = mask.shape[1] - last_pos - 1
            row = torch.arange(hidden.shape[0], device=hidden.device)
            embs = hidden[row, col]
            if normalize:
                embs = F.normalize(embs, p=2, dim=-1)
            return embs

    return _Qwen3VLEmbedder(model_path, max_length=kwargs.get("max_length", 8192))


class QwenEmbeddingEncoder:
    """MTEB-compatible encoder wrapping either Qwen3.5 or Qwen3-VL embedders."""

    def __init__(self, embedder, batch_size: int = 32, model_name: str = "qwen-embedding"):
        self.embedder = embedder
        self.batch_size = batch_size
        from mteb.models.model_meta import ModelMeta
        self.mteb_model_meta = ModelMeta.create_empty()
        self.mteb_model_meta.name = model_name

    def encode(self, inputs: DataLoader, *, task_metadata, hf_split, hf_subset,
               prompt_type=None, **kwargs) -> np.ndarray:
        all_texts = [text for batch in inputs for text in batch["text"]]

        instruction = "Represent the user's input."
        from mteb.types import PromptType
        if prompt_type == PromptType.query:
            task_type = task_metadata.type if hasattr(task_metadata, "type") else ""
            if "Retrieval" in task_type or "Retrieval" in task_metadata.name:
                instruction = "Given a query, retrieve a relevant document that answers the query."
            elif "Classification" in task_type:
                instruction = "Classify the given text."
            elif "Clustering" in task_type:
                instruction = "Identify the topic or theme of the given text."
            elif "STS" in task_type or "STS" in task_metadata.name:
                instruction = "Retrieve a semantically similar text."
            elif "PairClassification" in task_type:
                instruction = "Retrieve a text that is semantically similar or related."
            elif "Reranking" in task_type:
                instruction = "Given a query, retrieve a relevant document."
            elif "BitextMining" in task_type:
                instruction = "Retrieve a translation of the given text."

        all_embeddings = []
        for i in range(0, len(all_texts), self.batch_size):
            batch_texts = all_texts[i:i + self.batch_size]
            items = [{"text": t, "instruction": instruction} for t in batch_texts]
            with torch.no_grad():
                embs = self.embedder.process(items, normalize=True)
            all_embeddings.append(embs.cpu().float().numpy())

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
    model_type = detect_model_type(args.model_path)
    logger.info(f"Detected model type: {model_type} for {args.model_path}")

    if model_type == "qwen3vl":
        embedder = load_qwen3vl_embedder(args.model_path, max_length=args.max_length)
    else:
        embedder = load_qwen35_embedder(args.model_path, max_length=args.max_length)

    encoder = QwenEmbeddingEncoder(embedder, batch_size=args.batch_size)

    import mteb
    if args.quick:
        tasks = mteb.get_tasks(tasks=[QUICK_TASK])
    elif args.full:
        text_types = ["BitextMining", "Classification", "Clustering", "InstructionRetrieval",
                       "MultilabelClassification", "PairClassification", "Reranking", "Retrieval", "STS"]
        tasks = mteb.get_tasks(task_types=text_types, languages=["eng"])
    elif args.tasks:
        tasks = mteb.get_tasks(tasks=args.tasks)
    else:
        tasks = mteb.get_tasks(tasks=FAST_TASKS)

    logger.info(f"Running {len(tasks)} tasks")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = mteb.evaluate(
        encoder,
        tasks,
        overwrite_strategy="always",
    )

    scores_by_type = {}
    all_scores = []
    for task_result in results:
        for split_results in task_result.scores.values():
            for score_dict in split_results:
                main_score = score_dict.get("main_score", None)
                if main_score is not None:
                    task_type = task_result.task.metadata.type
                    scores_by_type.setdefault(task_type, []).append(main_score)
                    all_scores.append(main_score)

    print("\n" + "=" * 60)
    print(f"MMTEB Results for: {args.model_path}")
    print("=" * 60)
    for tt in sorted(scores_by_type):
        scores = scores_by_type[tt]
        print(f"  {tt:35s}: {np.mean(scores)*100:.2f} ({len(scores)} tasks)")
    if all_scores:
        print(f"  {'Mean (Task)':35s}: {np.mean(all_scores)*100:.2f}")
    if scores_by_type:
        type_means = [np.mean(v) for v in scores_by_type.values()]
        print(f"  {'Mean (Type)':35s}: {np.mean(type_means)*100:.2f}")
    print("=" * 60)

    summary = {
        "model_path": args.model_path,
        "model_type": model_type,
        "num_tasks": len(all_scores),
        "mean_task": float(np.mean(all_scores) * 100) if all_scores else 0,
        "mean_type": float(np.mean([np.mean(v) for v in scores_by_type.values()]) * 100) if scores_by_type else 0,
        "per_type": {k: float(np.mean(v) * 100) for k, v in scores_by_type.items()},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MMTEB evaluation for Qwen embedding models")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/mmteb")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--quick", action="store_true", help="Run only STSBenchmark (fastest comparison)")
    parser.add_argument("--full", action="store_true", help="Run full MMTEB (all English tasks)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific task names to run")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
