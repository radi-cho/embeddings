#!/usr/bin/env python3
"""
Multimodal contrastive training for Qwen3.5 / Qwen3-VL embedding models.

Following the Qwen3-VL-Embedding paper:
- LoRA adaptation (rank=32, alpha=32, targets: q/k/v/up/down/gate_proj)
- InfoNCE with false-negative masking + hard negatives
- CoSent loss for STS data
- Matryoshka Representation Learning (MRL) at dims [1024,768,512,256,128,64]
- Multi-GPU via accelerate DDP
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train.dataset import build_dataloader
from src.train.losses import (
    cosent_loss, masked_infonce_loss, mrl_cosent_loss, mrl_infonce_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_TARGET_MODULES_QWEN35 = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
LORA_TARGET_MODULES_QWEN3VL = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj",
]


def _detect_model_type(model_path):
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            if "qwen3_vl" in json.load(f).get("model_type", ""):
                return "qwen3vl"
    return "qwen35"


def _load_embedder(model_path, max_length, model_type, max_pixels=None):
    px = {"max_pixels": max_pixels} if max_pixels else {}
    if model_type == "qwen3vl":
        scripts_dir = Path(model_path) / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        from qwen3_vl_embedding import Qwen3VLEmbedder
        return Qwen3VLEmbedder(model_name_or_path=model_path, torch_dtype=torch.bfloat16,
                               max_length=max_length, **px)
    from src.models.qwen35_embedding import Qwen35Embedder
    return Qwen35Embedder(model_name_or_path=model_path, torch_dtype=torch.bfloat16,
                          max_length=max_length, **px)


def encode_batch(embedder, items, device):
    """Encode a batch in a single forward pass. DDP-compatible since each
    rank calls this exactly once per role (queries, positives, negatives)."""
    if not items:
        return None
    conversations = [
        embedder.format_model_input(
            text=ele.get("text"), image=ele.get("image"),
            video=ele.get("video"), instruction=ele.get("instruction"))
        for ele in items
    ]
    try:
        processed = embedder._preprocess_inputs(conversations)
    except Exception as e:
        logger.warning("Batch preprocessing failed (%s), falling back to text-only", e)
        conversations = [
            embedder.format_model_input(text=ele.get("text") or "NULL",
                                        instruction=ele.get("instruction"))
            for ele in items
        ]
        processed = embedder._preprocess_inputs(conversations)
    processed = {k: v.to(device) for k, v in processed.items()}
    out = embedder.model(**processed)
    return embedder._pooling_last(out.last_hidden_state, processed["attention_mask"])


def compute_loss(query_embs, pos_embs, neg_embs, task_types, scores, args):
    mrl_dims = args.mrl_dims if args.use_mrl else None
    has_sts = scores is not None and any(t == "sts" for t in task_types)
    if has_sts:
        if mrl_dims:
            return mrl_cosent_loss(query_embs, pos_embs, scores,
                                   mrl_dims=mrl_dims, temperature=args.temperature)
        return cosent_loss(F.normalize(query_embs, dim=-1),
                           F.normalize(pos_embs, dim=-1), scores, temperature=args.temperature)
    hard_neg = neg_embs if (neg_embs is not None and neg_embs.shape[0] > 0) else None
    if mrl_dims:
        return mrl_infonce_loss(query_embs, pos_embs, hard_neg, mrl_dims=mrl_dims,
                                temperature=args.temperature, stage=args.training_stage)
    q_n, p_n = F.normalize(query_embs, dim=-1), F.normalize(pos_embs, dim=-1)
    hn_n = F.normalize(hard_neg, dim=-1) if hard_neg is not None else None
    return masked_infonce_loss(q_n, p_n, hn_n, temperature=args.temperature, stage=args.training_stage)


def train(args):
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=10))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        mixed_precision="bf16",
        kwargs_handlers=[pg_kwargs])

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "training_args.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    if args.use_wandb and accelerator.is_main_process:
        wkw = {"name": args.wandb_run_name or None}
        wp = args.wandb_project
        if args.wandb_entity:
            wkw["entity"] = args.wandb_entity
        elif "/" in wp:
            ent, wp = wp.split("/", 1)
            wkw["entity"] = ent
        accelerator.init_trackers(project_name=wp, config=vars(args), init_kwargs={"wandb": wkw})

    model_type = _detect_model_type(args.model_path)
    accelerator.print(f"Loading model from {args.model_path} (type={model_type})")
    embedder = _load_embedder(args.model_path, args.max_length, model_type, max_pixels=args.max_pixels)
    lora_targets = LORA_TARGET_MODULES_QWEN3VL if model_type == "qwen3vl" else LORA_TARGET_MODULES_QWEN35
    embedder.model = get_peft_model(embedder.model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_targets))
    if accelerator.is_main_process:
        embedder.model.print_trainable_parameters()
    if args.gradient_checkpointing:
        embedder.model.enable_input_require_grads()
        embedder.model.gradient_checkpointing_enable()

    subsets = args.subsets.split(",") if args.subsets else None
    tt_filter = args.task_types.split(",") if args.task_types else None
    dataloader = build_dataloader(
        subsets=subsets, task_types=tt_filter, split=args.dataset_split,
        image_dir=args.image_dir, max_samples_per_subset=args.max_samples_per_subset,
        cache_dir=args.cache_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True)
    optimizer = AdamW(embedder.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(dataloader) * args.epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)
    embedder.model, optimizer, dataloader, scheduler = accelerator.prepare(
        embedder.model, optimizer, dataloader, scheduler)

    accelerator.print(
        f"Training: {len(dataloader)} batches/epoch, {total_steps} optimizer steps, "
        f"{args.epochs} epochs, warmup={warmup_steps}, gpus={accelerator.num_processes}")

    global_step = 0
    embedder.model.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_steps, t0 = 0.0, 0, time.time()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(embedder.model):
                scores = batch["scores"]
                if scores is not None:
                    scores = scores.to(accelerator.device)

                q = encode_batch(embedder, batch["queries"], accelerator.device)
                p = encode_batch(embedder, batch["positives"], accelerator.device)
                n = encode_batch(embedder, batch["negatives"], accelerator.device)
                loss = compute_loss(q, p, n, batch["task_types"], scores, args)

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
                    avg = epoch_loss / epoch_steps
                    lr = scheduler.get_last_lr()[0]
                    sps = (epoch_steps * args.batch_size * accelerator.num_processes) / (time.time() - t0)
                    logger.info("Epoch %d | Step %d/%d | Loss %.4f | LR %.2e | %.1f samples/s",
                                epoch, global_step, total_steps, avg, lr, sps)
                    if args.use_wandb:
                        accelerator.log({"train/loss": avg, "train/lr": lr, "train/epoch": epoch,
                                         "train/global_step": global_step, "train/samples_per_sec": sps},
                                        step=global_step)
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_ckpt(accelerator, embedder, args, global_step)
        accelerator.print(f"Epoch {epoch} done. Avg loss: {epoch_loss / max(epoch_steps, 1):.4f}")
    _save_ckpt(accelerator, embedder, args, global_step, final=True)
    if args.use_wandb and accelerator.is_main_process:
        accelerator.end_training()
    accelerator.print("Training complete.")


def _save_ckpt(accelerator, embedder, args, global_step, final=False):
    if not accelerator.is_main_process:
        return
    d = Path(args.output_dir) / ("final" if final else f"checkpoint-{global_step}")
    accelerator.print(f"Saving checkpoint to {d}")
    accelerator.unwrap_model(embedder.model).save_pretrained(d)
    tok = getattr(embedder, "processor", None) or getattr(embedder, "tokenizer", None)
    if tok:
        tok.save_pretrained(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_dir", default="outputs/qwen35-embedding-train")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--subsets", default=None)
    p.add_argument("--task_types", default=None)
    p.add_argument("--dataset_split", default="diverse_instruction")
    p.add_argument("--image_dir", default=None)
    p.add_argument("--max_pixels", type=int, default=401408)
    p.add_argument("--max_samples_per_subset", type=int, default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--training_stage", type=int, default=1, choices=[1, 2])
    p.add_argument("--use_mrl", action="store_true", default=True)
    p.add_argument("--no_mrl", action="store_true")
    p.add_argument("--mrl_dims", default="1024,768,512,256,128,64")
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="embeddings")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    args = p.parse_args()
    if args.no_mrl:
        args.use_mrl = False
    args.mrl_dims = [int(x) for x in args.mrl_dims.split(",")]
    train(args)

if __name__ == "__main__":
    main()
