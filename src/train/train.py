#!/usr/bin/env python3
"""
Multimodal contrastive training for Qwen3.5-0.8B embedding model.

Following the Qwen3-VL-Embedding paper:
- LoRA adaptation (rank=32, alpha=32, targets: q/k/v/up/down/gate_proj)
- InfoNCE with false-negative masking + hard negatives
- CoSent loss for STS data
- Matryoshka Representation Learning (MRL) at dims [1024,768,512,256,128,64]
- Multi-GPU via accelerate

Usage (single GPU):
    python src/train/train.py --model_path models/Qwen3.5-0.8B --subsets N24News

Usage (multi-GPU via accelerate, see scripts/run_training.sh):
    accelerate launch --num_processes 2 src/train/train.py ...
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train.dataset import build_dataloader
from src.train.losses import (
    DEFAULT_MRL_DIMS,
    classification_contrastive_loss,
    cosent_loss,
    masked_infonce_loss,
    mrl_cosent_loss,
    mrl_infonce_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paper Table 1: LoRA config for Qwen3-VL-Embedding
LORA_TARGET_MODULES_QWEN35 = [
    "q_proj", "k_proj", "v_proj",
    "up_proj", "down_proj", "gate_proj",
]

LORA_TARGET_MODULES_QWEN3VL = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # Vision encoder projections
    "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj",
]


def _detect_model_type(model_path: str) -> str:
    """Auto-detect Qwen3-VL vs Qwen3.5 from config.json."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            if "qwen3_vl" in json.load(f).get("model_type", ""):
                return "qwen3vl"
    return "qwen35"


def _load_embedder(model_path: str, max_length: int, model_type: str):
    """Load the right embedder class based on model type."""
    if model_type == "qwen3vl":
        scripts_dir = Path(model_path) / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        from qwen3_vl_embedding import Qwen3VLEmbedder

        logger.info("Loading Qwen3-VL embedder from %s", model_path)
        return Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            max_length=max_length,
        )
    else:
        from src.models.qwen35_embedding import Qwen35Embedder

        logger.info("Loading Qwen3.5 embedder from %s", model_path)
        return Qwen35Embedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            max_length=max_length,
        )


def encode_batch(embedder, items: list) -> torch.Tensor:
    """
    Encode a batch of items through the embedding model (with gradients).

    Each item should have keys: text, image, instruction.
    Returns raw (un-normalized) embeddings for MRL truncation downstream.
    """
    if not items:
        return None

    conversations = [
        embedder.format_model_input(
            text=ele.get("text"),
            image=ele.get("image"),
            video=ele.get("video"),
            instruction=ele.get("instruction"),
        )
        for ele in items
    ]
    processed = embedder._preprocess_inputs(conversations)
    processed = {k: v.to(embedder.model.device) for k, v in processed.items()}
    outputs = embedder.model(**processed)
    hidden = outputs.last_hidden_state
    mask = processed["attention_mask"]
    return embedder._pooling_last(hidden, mask)


def compute_loss(
    query_embs: torch.Tensor,
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    task_types: list,
    scores: torch.Tensor,
    args,
) -> torch.Tensor:
    """Dispatch to the appropriate loss function based on task type mix."""
    mrl_dims = args.mrl_dims if args.use_mrl else None

    has_sts = scores is not None and any(t == "sts" for t in task_types)

    if has_sts:
        if mrl_dims:
            return mrl_cosent_loss(
                query_embs, pos_embs, scores,
                mrl_dims=mrl_dims, temperature=args.temperature,
            )
        else:
            q_n = F.normalize(query_embs, dim=-1)
            p_n = F.normalize(pos_embs, dim=-1)
            return cosent_loss(q_n, p_n, scores, temperature=args.temperature)

    hard_neg = neg_embs if (neg_embs is not None and neg_embs.shape[0] > 0) else None

    if mrl_dims:
        return mrl_infonce_loss(
            query_embs, pos_embs, hard_neg,
            mrl_dims=mrl_dims, temperature=args.temperature,
            stage=args.training_stage,
        )
    else:
        q_n = F.normalize(query_embs, dim=-1)
        p_n = F.normalize(pos_embs, dim=-1)
        hn_n = F.normalize(hard_neg, dim=-1) if hard_neg is not None else None
        return masked_infonce_loss(
            q_n, p_n, hn_n,
            temperature=args.temperature, stage=args.training_stage,
        )


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        mixed_precision="bf16",
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "training_args.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    if args.use_wandb and accelerator.is_main_process:
        wandb_kwargs = {"name": args.wandb_run_name or None}
        wandb_project = args.wandb_project
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        elif "/" in wandb_project:
            entity, wandb_project = wandb_project.split("/", 1)
            wandb_kwargs["entity"] = entity
        accelerator.init_trackers(
            project_name=wandb_project,
            config=vars(args),
            init_kwargs={"wandb": wandb_kwargs},
        )

    model_type = _detect_model_type(args.model_path)
    accelerator.print(f"Loading model from {args.model_path} (type={model_type})")
    embedder = _load_embedder(args.model_path, args.max_length, model_type)

    lora_targets = LORA_TARGET_MODULES_QWEN3VL if model_type == "qwen3vl" else LORA_TARGET_MODULES_QWEN35
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
    )
    embedder.model = get_peft_model(embedder.model, lora_config)
    if accelerator.is_main_process:
        embedder.model.print_trainable_parameters()

    if args.gradient_checkpointing:
        embedder.model.enable_input_require_grads()
        embedder.model.gradient_checkpointing_enable()

    subsets = args.subsets.split(",") if args.subsets else None
    task_types_filter = args.task_types.split(",") if args.task_types else None

    dataloader = build_dataloader(
        subsets=subsets,
        task_types=task_types_filter,
        split=args.dataset_split,
        image_dir=args.image_dir,
        max_samples_per_subset=args.max_samples_per_subset,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    optimizer = AdamW(
        embedder.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = (len(dataloader) * args.epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    embedder.model, optimizer, dataloader, scheduler = accelerator.prepare(
        embedder.model, optimizer, dataloader, scheduler
    )

    accelerator.print(
        f"Training: {len(dataloader)} batches/epoch, "
        f"{total_steps} total optimizer steps, {args.epochs} epochs, "
        f"warmup={warmup_steps} steps"
    )

    global_step = 0
    embedder.model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(embedder.model):
                queries = batch["queries"]
                positives = batch["positives"]
                negatives = batch["negatives"]
                batch_task_types = batch["task_types"]
                batch_scores = batch["scores"]
                if batch_scores is not None:
                    batch_scores = batch_scores.to(accelerator.device)

                q_embs = encode_batch(embedder, queries)
                p_embs = encode_batch(embedder, positives)
                n_embs = encode_batch(embedder, negatives) if negatives else None

                loss = compute_loss(
                    q_embs, p_embs, n_embs,
                    batch_task_types, batch_scores, args,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(embedder.model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.detach().float().item()
            epoch_steps += 1

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.log_interval == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / epoch_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    samples_per_sec = (epoch_steps * args.batch_size) / elapsed
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/samples_per_sec": samples_per_sec,
                    }
                    logger.info(
                        "Epoch %d | Step %d/%d | Loss %.4f | LR %.2e | %.1f samples/s",
                        epoch, global_step, total_steps, avg_loss, lr, samples_per_sec,
                    )
                    if args.use_wandb:
                        accelerator.log(log_dict, step=global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_checkpoint(accelerator, embedder, args, global_step)

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        accelerator.print(f"Epoch {epoch} finished. Avg loss: {avg_epoch_loss:.4f}")

    _save_checkpoint(accelerator, embedder, args, global_step, final=True)

    if args.use_wandb and accelerator.is_main_process:
        accelerator.end_training()

    accelerator.print("Training complete.")


def _save_checkpoint(accelerator, embedder, args, global_step, final=False):
    if not accelerator.is_main_process:
        return
    output_dir = Path(args.output_dir)
    if final:
        ckpt_dir = output_dir / "final"
    else:
        ckpt_dir = output_dir / f"checkpoint-{global_step}"

    accelerator.print(f"Saving checkpoint to {ckpt_dir}")
    unwrapped = accelerator.unwrap_model(embedder.model)
    unwrapped.save_pretrained(ckpt_dir)
    if hasattr(embedder, "processor"):
        embedder.processor.save_pretrained(ckpt_dir)
    elif hasattr(embedder, "tokenizer"):
        embedder.tokenizer.save_pretrained(ckpt_dir)
    elif hasattr(embedder, "_has_processor") and embedder._has_processor:
        embedder.processor.tokenizer.save_pretrained(ckpt_dir)
    else:
        accelerator.print("Warning: could not find tokenizer/processor to save")


def parse_mrl_dims(s: str) -> list:
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Train Qwen embedding model (auto-detects Qwen3.5 or Qwen3-VL)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen35-embedding-train")

    # LoRA (paper: rank=32, alpha=32)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Dataset
    parser.add_argument("--subsets", type=str, default=None,
                        help="Comma-separated subset names, or None for all")
    parser.add_argument("--task_types", type=str, default=None,
                        help="Comma-separated: classification,vqa,retrieval")
    parser.add_argument("--dataset_split", type=str, default="diverse_instruction")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Local dir with pre-downloaded MMEB images")
    parser.add_argument("--max_samples_per_subset", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Loss
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--training_stage", type=int, default=1, choices=[1, 2],
                        help="Stage 1: full InfoNCE (q-q, d-d terms). Stage 2: simplified.")
    parser.add_argument("--use_mrl", action="store_true", default=True)
    parser.add_argument("--no_mrl", action="store_true", help="Disable MRL")
    parser.add_argument("--mrl_dims", type=str, default="1024,768,512,256,128,64")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="embeddings")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    if args.no_mrl:
        args.use_mrl = False
    args.mrl_dims = parse_mrl_dims(args.mrl_dims)

    train(args)


if __name__ == "__main__":
    main()
