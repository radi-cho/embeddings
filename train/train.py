#!/usr/bin/env python3
"""Multimodal contrastive pretraining for Qwen3.5 / Qwen3-VL embedding models.

Simple DDP training with cross-GPU embedding gathering for contrastive loss.
Each GPU encodes its local batch, embeddings are gathered across GPUs via
GatherWithGrad so the loss sees (batch_size * num_gpus) in-batch negatives,
then loss.backward() flows gradients back through DDP normally.

No GradCache -- batch_size must fit in GPU memory. Scale effective batch
by adding GPUs (batch_size * num_gpus) or gradient_accumulation_steps.
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
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.dataset import build_dataloader, build_mixed_dataloader, collate_embedding_batch
from train.losses import (
    cosent_loss, masked_infonce_loss, mrl_cosent_loss, mrl_infonce_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_TARGETS = {
    "qwen35": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    "qwen3vl": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj",
                 "o_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"],
}


def _world():
    return dist.get_world_size() if dist.is_initialized() else 1

def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


class GatherWithGrad(torch.autograd.Function):
    """all_gather that lets gradients flow back through the local shard."""
    @staticmethod
    def forward(ctx, x):
        if _world() == 1:
            return x
        gathered = [torch.zeros_like(x) for _ in range(_world())]
        dist.all_gather(gathered, x.contiguous())
        gathered[_rank()] = x
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad):
        if _world() == 1:
            return grad
        chunk = grad.shape[0] // _world()
        return grad[_rank() * chunk : (_rank() + 1) * chunk]


def _gather_metadata(task_types, scores):
    if _world() == 1:
        return task_types, scores
    all_tt = [None] * _world()
    dist.all_gather_object(all_tt, list(task_types))
    g_tt = [t for sub in all_tt for t in sub]
    if scores is not None:
        parts = [torch.zeros_like(scores) for _ in range(_world())]
        dist.all_gather(parts, scores.contiguous())
        return g_tt, torch.cat(parts, dim=0)
    return g_tt, None


def _format_item(embedder, e):
    return embedder.format_model_input(
        text=e.get("text"), image=e.get("image"),
        video=e.get("video"), instruction=e.get("instruction"))


def _format_text_only(embedder, e):
    return embedder.format_model_input(
        text=e.get("text") or "NULL", instruction=e.get("instruction"))


def _encode(model, embedder, items, device):
    """Encode a batch. On vision preprocessing failure, retry as text-only.
    OOM and timeouts are not caught — they propagate to fail the step."""
    if not items:
        return None
    convs = [_format_item(embedder, e) for e in items]
    try:
        proc = embedder._preprocess_inputs(convs)
    except Exception as exc:
        logger.warning("Vision preprocess failed (%s), text-only fallback for batch", exc)
        convs = [_format_text_only(embedder, e) for e in items]
        proc = embedder._preprocess_inputs(convs)
    proc = {k: v.to(device) for k, v in proc.items() if torch.is_tensor(v)}
    out = model(**proc)
    return embedder._pooling_last(out.last_hidden_state, proc["attention_mask"])


def _compute_loss(q_emb, p_emb, task_types, scores, args):
    mrl = args.mrl_dims if args.use_mrl else None
    device = q_emb.device
    sts_idx = [i for i, t in enumerate(task_types) if t == "sts"]
    con_idx = [i for i, t in enumerate(task_types) if t != "sts"]
    losses = []
    if con_idx:
        ci = torch.tensor(con_idx, device=device)
        q, p = q_emb[ci], p_emb[ci]
        if mrl:
            losses.append(mrl_infonce_loss(q, p, None, mrl_dims=mrl,
                          temperature=args.temperature, stage=args.training_stage))
        else:
            losses.append(masked_infonce_loss(
                F.normalize(q, dim=-1), F.normalize(p, dim=-1), None,
                temperature=args.temperature, stage=args.training_stage))
    if sts_idx and scores is not None:
        si = torch.tensor(sts_idx, device=device)
        q, p, s = q_emb[si], p_emb[si], scores[si]
        if mrl:
            losses.append(mrl_cosent_loss(q, p, s, mrl_dims=mrl,
                          temperature=args.temperature))
        else:
            losses.append(cosent_loss(F.normalize(q, dim=-1),
                          F.normalize(p, dim=-1), s, temperature=args.temperature))
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return sum(losses) / len(losses)


def _train_step(embedder, raw_model, device, batch, args):
    q_local = _encode(raw_model, embedder, batch["queries"], device)
    p_local = _encode(raw_model, embedder, batch["positives"], device)
    if _world() > 1:
        dist.barrier()
    q = GatherWithGrad.apply(q_local) if _world() > 1 else q_local
    p = GatherWithGrad.apply(p_local) if _world() > 1 else p_local
    scores = batch["scores"].to(device) if batch["scores"] is not None else None
    g_tt, g_scores = _gather_metadata(batch["task_types"], scores)
    loss = _compute_loss(q, p, g_tt, g_scores, args)
    loss.backward()
    return loss.detach().float().item()


def _detect_model_type(model_path):
    cfg = Path(model_path) / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            if "qwen3_vl" in json.load(f).get("model_type", ""):
                return "qwen3vl"
    return "qwen35"


def _load_embedder(model_path, max_length, model_type, max_pixels=None):
    kw = {"max_pixels": max_pixels} if max_pixels else {}
    if model_type == "qwen3vl":
        sd = Path(model_path) / "scripts"
        if sd.exists():
            sys.path.insert(0, str(sd))
        from qwen3_vl_embedding import Qwen3VLEmbedder
        return Qwen3VLEmbedder(model_name_or_path=model_path,
                               torch_dtype=torch.bfloat16, max_length=max_length, **kw)
    from models.qwen35_embedding import Qwen35Embedder
    return Qwen35Embedder(model_name_or_path=model_path,
                          torch_dtype=torch.bfloat16, max_length=max_length, **kw)


def _save_ckpt(accelerator, embedder, args, step, final=False):
    if not accelerator.is_main_process:
        return
    d = Path(args.output_dir) / ("final" if final else f"checkpoint-{step}")
    accelerator.print(f"Saving to {d}")
    accelerator.unwrap_model(embedder.model).save_pretrained(d)
    tok = getattr(embedder, "processor", None) or getattr(embedder, "tokenizer", None)
    if tok:
        tok.save_pretrained(d)


def _build_dataloader(args, embedder, batch_size):
    subsets = args.subsets.split(",") if args.subsets else None
    tt_filter = args.task_types.split(",") if args.task_types else None
    kw = dict(batch_size=batch_size, num_workers=args.num_workers, shuffle=True)
    if args.data_dir:
        return build_mixed_dataloader(
            data_dir=args.data_dir, image_dir=args.image_dir,
            mmeb_split=args.dataset_split,
            max_samples_per_subset=args.max_samples_per_subset,
            cache_dir=args.cache_dir, **kw)
    return build_dataloader(
        subsets=subsets, task_types=tt_filter, split=args.dataset_split,
        image_dir=args.image_dir, max_samples_per_subset=args.max_samples_per_subset,
        cache_dir=args.cache_dir, **kw)


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        mixed_precision="bf16",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))])

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
            wkw["entity"], wp = wp.split("/", 1)
        accelerator.init_trackers(project_name=wp, config=vars(args),
                                  init_kwargs={"wandb": wkw})

    mt = _detect_model_type(args.model_path)
    accelerator.print(f"Loading {args.model_path} (type={mt})")
    embedder = _load_embedder(args.model_path, args.max_length, mt, args.max_pixels)
    embedder.model = get_peft_model(embedder.model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS[mt]))
    if accelerator.is_main_process:
        embedder.model.print_trainable_parameters()
    if args.gradient_checkpointing:
        embedder.model.enable_input_require_grads()
        embedder.model.gradient_checkpointing_enable()

    world = accelerator.num_processes
    bs = args.batch_size
    effective_bs = bs * world * args.gradient_accumulation_steps
    accelerator.print(
        f"\n  Batch size: {bs}/GPU x {world} GPUs"
        f" x {args.gradient_accumulation_steps} accum = {effective_bs} effective\n"
        f"  LR: {args.lr}  |  Epochs: {args.epochs}  |  Seed: {args.seed}")

    dataloader = _build_dataloader(args, embedder, bs)
    optimizer = AdamW(embedder.model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    embedder.model, optimizer = accelerator.prepare(embedder.model, optimizer)

    if world > 1:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataloader.dataset, num_replicas=world,
                                     rank=_rank(), shuffle=True, seed=args.seed)
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset, batch_size=bs, sampler=sampler,
            collate_fn=collate_embedding_batch, num_workers=args.num_workers,
            drop_last=True, pin_memory=False)

    total_steps = len(dataloader) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = accelerator.prepare(get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps))
    accelerator.print(f"  Steps: {total_steps} ({len(dataloader)}/epoch), warmup={warmup}")

    raw_model = accelerator.unwrap_model(embedder.model)
    global_step = 0
    embedder.model.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_steps, t0 = 0.0, 0, time.time()
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            optimizer.zero_grad()
            with accelerator.no_sync(embedder.model):
                step_loss = _train_step(
                    embedder, raw_model, accelerator.device, batch, args)
            if world > 1:
                dist.barrier()
                for p in embedder.model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(
                embedder.model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += step_loss
            epoch_steps += 1
            global_step += 1

            if global_step % args.log_interval == 0 and accelerator.is_main_process:
                avg = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                sps = (epoch_steps * effective_bs) / (time.time() - t0)
                logger.info("E%d S%d/%d | loss=%.4f lr=%.2e | %.1f samp/s bs=%d",
                            epoch, global_step, total_steps, avg, lr, sps, effective_bs)
                if args.use_wandb:
                    accelerator.log({"train/loss": avg, "train/lr": lr,
                                     "train/epoch": epoch, "train/step": global_step,
                                     "train/samples_per_sec": sps}, step=global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_ckpt(accelerator, embedder, args, global_step)

        accelerator.print(f"Epoch {epoch} done. Loss: {epoch_loss / max(epoch_steps, 1):.4f}")

    _save_ckpt(accelerator, embedder, args, global_step, final=True)
    if args.use_wandb and accelerator.is_main_process:
        accelerator.end_training()
    accelerator.print("Done.")


def main():
    p = argparse.ArgumentParser(description="Multimodal contrastive pretraining")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="outputs/qwen35-embedding-train")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_pixels", type=int, default=1310720)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--image_dir", default=None)
    p.add_argument("--dataset_split", default="diverse_instruction")
    p.add_argument("--subsets", default=None)
    p.add_argument("--task_types", default=None)
    p.add_argument("--max_samples_per_subset", type=int, default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--training_stage", type=int, default=1, choices=[1, 2])
    p.add_argument("--use_mrl", action="store_true", default=True)
    p.add_argument("--no_mrl", action="store_true")
    p.add_argument("--mrl_dims", default="1024,256,64")
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="embeddings")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)

    a = p.parse_args()
    if a.no_mrl:
        a.use_mrl = False
    a.mrl_dims = [int(x) for x in a.mrl_dims.split(",")]
    train(a)


if __name__ == "__main__":
    main()
