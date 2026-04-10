#!/usr/bin/env python3
"""Multimodal contrastive pretraining for Qwen3.5 / Qwen3-VL embedding models.

Cross-GPU contrastive training with GradCache:
  contrastive_batch_size = total in-batch negatives (gathered across GPUs)
  per_device_batch       = contrastive_batch_size / num_gpus  (auto)
  micro_batch_size       = forward pass chunk size (GPU memory limit)
  GradCache enables when per_device_batch > micro_batch_size (auto)
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

from train.dataset import build_dataloader, build_mixed_dataloader
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


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _world():
    return dist.get_world_size() if dist.is_initialized() else 1

def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


class GatherWithGrad(torch.autograd.Function):
    """all_gather with gradient passthrough for the local shard."""
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
        c = grad.shape[0] // _world()
        return grad[_rank() * c : (_rank() + 1) * c]


def _gather(emb):
    if emb is None or _world() == 1:
        return emb
    return GatherWithGrad.apply(emb)


def _gather_detached(t):
    if t is None or _world() == 1:
        return t
    gathered = [torch.zeros_like(t) for _ in range(_world())]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0)


def _gather_metadata(task_types, scores):
    """Gather task_types (list[str]) and scores (tensor|None) across GPUs."""
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


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _format_item(embedder, e):
    return embedder.format_model_input(
        text=e.get("text"), image=e.get("image"),
        video=e.get("video"), instruction=e.get("instruction"))


def _encode(model, embedder, items, device):
    """Encode a list of items → (B, D) pooled embeddings. OOM-safe."""
    if not items:
        return None
    convs = [_format_item(embedder, e) for e in items]
    try:
        proc = embedder._preprocess_inputs(convs)
    except Exception:
        convs = [embedder.format_model_input(
            text=e.get("text") or "NULL", instruction=e.get("instruction"))
            for e in items]
        proc = embedder._preprocess_inputs(convs)
    proc = {k: v.to(device) for k, v in proc.items() if torch.is_tensor(v)}
    try:
        out = model(**proc)
        return embedder._pooling_last(out.last_hidden_state, proc["attention_mask"])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        embs = []
        for e in items:
            try:
                p = embedder._preprocess_inputs([_format_item(embedder, e)])
            except Exception:
                p = embedder._preprocess_inputs([embedder.format_model_input(
                    text=e.get("text") or "NULL", instruction=e.get("instruction"))])
            p = {k: v.to(device) for k, v in p.items() if torch.is_tensor(v)}
            out = model(**p)
            embs.append(embedder._pooling_last(out.last_hidden_state, p["attention_mask"]))
        return torch.cat(embs, dim=0)


def _filter_negatives(negatives):
    """Extract genuine hard negatives, dropping None placeholders."""
    if negatives is None:
        return []
    return [n for n in negatives if n is not None]


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _compute_loss(q_emb, p_emb, n_emb, task_types, scores, args):
    """Route by task_type: contrastive items → InfoNCE, STS items → CoSent."""
    mrl = args.mrl_dims if args.use_mrl else None
    device = q_emb.device

    sts_idx = [i for i, t in enumerate(task_types) if t == "sts"]
    con_idx = [i for i, t in enumerate(task_types) if t != "sts"]
    losses = []

    if con_idx:
        ci = torch.tensor(con_idx, device=device)
        q, p = q_emb[ci], p_emb[ci]
        n = n_emb[ci] if n_emb is not None and n_emb.shape[0] == q_emb.shape[0] else None
        if mrl:
            losses.append(mrl_infonce_loss(q, p, n, mrl_dims=mrl,
                          temperature=args.temperature, stage=args.training_stage))
        else:
            hn = n if (n is not None and n.shape[0] > 0) else None
            losses.append(masked_infonce_loss(
                F.normalize(q, dim=-1), F.normalize(p, dim=-1),
                F.normalize(hn, dim=-1) if hn is not None else None,
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


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def _simple_step(embedder, model, device, batch, args):
    """Forward+backward with cross-GPU embedding gathering."""
    q = _encode(model, embedder, batch["queries"], device)
    p = _encode(model, embedder, batch["positives"], device)
    negs = _filter_negatives(batch["negatives"])
    n = _encode(model, embedder, negs, device) if negs else None

    q, p, n = _gather(q), _gather(p), _gather(n)
    scores = batch["scores"].to(device) if batch["scores"] is not None else None
    g_tt, g_scores = _gather_metadata(batch["task_types"], scores)

    loss = _compute_loss(q, p, n, g_tt, g_scores, args)
    loss.backward()
    return loss.detach().float().item()


def _gradcache_step(embedder, model, raw_model, device, batch, micro_bs, args):
    """GradCache: encode locally → gather globally → loss → replay grads."""

    def _cache_role(items):
        if not items:
            return None
        parts = []
        for chunk in _chunked(items, micro_bs):
            with torch.no_grad():
                parts.append(_encode(raw_model, embedder, chunk, device))
        return torch.cat(parts, dim=0)

    local_q = _cache_role(batch["queries"])
    local_p = _cache_role(batch["positives"])
    negs = _filter_negatives(batch["negatives"])
    local_n = _cache_role(negs) if negs else None
    local_bs = local_q.shape[0]

    global_q = _gather_detached(local_q).detach().requires_grad_(True)
    global_p = _gather_detached(local_p).detach().requires_grad_(True)
    global_n = None
    if local_n is not None:
        global_n = _gather_detached(local_n).detach().requires_grad_(True)

    scores = batch["scores"].to(device) if batch["scores"] is not None else None
    g_tt, g_scores = _gather_metadata(batch["task_types"], scores)

    loss = _compute_loss(global_q, global_p, global_n, g_tt, g_scores, args)

    targets = [global_q, global_p] + ([global_n] if global_n is not None else [])
    grads = torch.autograd.grad(loss, targets)

    rank = _rank()
    q_grad = grads[0][rank * local_bs : (rank + 1) * local_bs]
    p_grad = grads[1][rank * local_bs : (rank + 1) * local_bs]
    n_grad = None
    if local_n is not None:
        ln = local_n.shape[0]
        n_grad = grads[2][rank * ln : (rank + 1) * ln]

    roles = [(batch["queries"], q_grad), (batch["positives"], p_grad)]
    if negs:
        roles.append((negs, n_grad))

    for items, grad in roles:
        if items is None or grad is None:
            continue
        offset = 0
        for chunk in _chunked(items, micro_bs):
            cg = grad[offset:offset + len(chunk)]
            emb = _encode(model, embedder, chunk, device)
            emb.backward(cg)
            offset += len(chunk)

    return loss.detach().float().item()


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

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


def _build_dataloader(args, embedder, per_device_batch):
    subsets = args.subsets.split(",") if args.subsets else None
    tt_filter = args.task_types.split(",") if args.task_types else None
    kw = dict(batch_size=per_device_batch, num_workers=args.num_workers, shuffle=True)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        mixed_precision="bf16",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=10))])

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

    # Model + LoRA
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

    # Batch math
    world = accelerator.num_processes
    cbs = args.contrastive_batch_size
    mbs = args.micro_batch_size
    assert cbs % world == 0, f"contrastive_batch_size ({cbs}) must be divisible by num_gpus ({world})"
    per_dev = cbs // world
    use_gc = per_dev > mbs

    accelerator.print(
        f"\n  Contrastive BS: {cbs} ({per_dev}/GPU x {world})\n"
        f"  Micro BS: {mbs}  |  GradCache: {'ON' if use_gc else 'OFF'}\n"
        f"  Grad accum: {args.gradient_accumulation_steps}  |  "
        f"Gather: {'YES' if world > 1 else 'N/A'}\n"
        f"  Seed: {args.seed}")

    # Data + optimizer + scheduler
    dataloader = _build_dataloader(args, embedder, per_dev)
    optimizer = AdamW(embedder.model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    embedder.model, optimizer, dataloader = accelerator.prepare(
        embedder.model, optimizer, dataloader)

    total_steps = len(dataloader) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = accelerator.prepare(get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps))
    raw_model = accelerator.unwrap_model(embedder.model)

    accelerator.print(f"  Steps: {total_steps} ({len(dataloader)}/epoch), warmup={warmup}")

    # Training loop
    global_step = 0
    embedder.model.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_steps, t0 = 0.0, 0, time.time()
        for batch in dataloader:
            with accelerator.accumulate(embedder.model):
                optimizer.zero_grad()
                if use_gc:
                    with accelerator.no_sync(embedder.model):
                        step_loss = _gradcache_step(
                            embedder, embedder.model, raw_model,
                            accelerator.device, batch, mbs, args)
                    if world > 1:
                        for p in embedder.model.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                else:
                    step_loss = _simple_step(
                        embedder, embedder.model, accelerator.device, batch, args)
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
                sps = (epoch_steps * cbs) / (time.time() - t0)
                logger.info("E%d S%d/%d | loss=%.4f lr=%.2e | %.1f samp/s cbs=%d",
                            epoch, global_step, total_steps, avg, lr, sps, cbs)
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
    # Model
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="outputs/qwen35-embedding-train")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_pixels", type=int, default=1310720)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true")
    # Data
    p.add_argument("--data_dir", default=None)
    p.add_argument("--image_dir", default=None)
    p.add_argument("--dataset_split", default="diverse_instruction")
    p.add_argument("--subsets", default=None)
    p.add_argument("--task_types", default=None)
    p.add_argument("--max_samples_per_subset", type=int, default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--num_workers", type=int, default=4)
    # Training
    p.add_argument("--contrastive_batch_size", type=int, default=1024)
    p.add_argument("--micro_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # Loss
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--training_stage", type=int, default=1, choices=[1, 2])
    p.add_argument("--use_mrl", action="store_true", default=True)
    p.add_argument("--no_mrl", action="store_true")
    p.add_argument("--mrl_dims", default="1024,256,64")
    # Logging
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
