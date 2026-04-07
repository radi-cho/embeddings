#!/usr/bin/env python3
"""
Multimodal contrastive training for Qwen3.5 / Qwen3-VL embedding models.

Uses GradCache to decouple contrastive batch size from GPU memory:
- Encode many micro-batches with no_grad, cache embeddings
- Compute contrastive loss over the full cache (large effective batch)
- Replay gradients through the model in micro-batches

Multi-GPU via accelerate DDP.
"""

import argparse
import json
import logging
import math
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

from src.train.dataset import build_dataloader, build_pretokenized_dataloader
from src.train.losses import (
    cosent_loss, masked_infonce_loss, mrl_cosent_loss, mrl_infonce_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_QWEN35 = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
LORA_QWEN3VL = LORA_QWEN35 + [
    "o_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj",
]


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
        return Qwen3VLEmbedder(
            model_name_or_path=model_path, torch_dtype=torch.bfloat16,
            max_length=max_length, **kw)
    from src.models.qwen35_embedding import Qwen35Embedder
    return Qwen35Embedder(
        model_name_or_path=model_path, torch_dtype=torch.bfloat16,
        max_length=max_length, **kw)


# ---------------------------------------------------------------------------
# GradCache helpers
# ---------------------------------------------------------------------------

def _preprocess_single(embedder, item):
    """Preprocess one item into model-ready tensors (on CPU)."""
    conv = embedder.format_model_input(
        text=item.get("text"), image=item.get("image"),
        video=item.get("video"), instruction=item.get("instruction"))
    try:
        return embedder._preprocess_inputs([conv])
    except Exception:
        conv = embedder.format_model_input(
            text=item.get("text") or "NULL", instruction=item.get("instruction"))
        return embedder._preprocess_inputs([conv])


def _encode_micro(model, embedder, items, device):
    """Forward a micro-batch, return pooled embeddings (B_micro, D).
    Falls back to per-item encoding on CUDA OOM."""
    if not items:
        return None
    convs = [
        embedder.format_model_input(
            text=e.get("text"), image=e.get("image"),
            video=e.get("video"), instruction=e.get("instruction"))
        for e in items
    ]
    try:
        proc = embedder._preprocess_inputs(convs)
    except Exception:
        convs = [
            embedder.format_model_input(
                text=e.get("text") or "NULL", instruction=e.get("instruction"))
            for e in items
        ]
        proc = embedder._preprocess_inputs(convs)
    proc = {k: v.to(device) for k, v in proc.items() if torch.is_tensor(v)}
    try:
        out = model(**proc)
        return embedder._pooling_last(out.last_hidden_state, proc["attention_mask"])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        embs = []
        for e in items:
            conv = embedder.format_model_input(
                text=e.get("text"), image=e.get("image"),
                video=e.get("video"), instruction=e.get("instruction"))
            try:
                p = embedder._preprocess_inputs([conv])
            except Exception:
                conv = embedder.format_model_input(
                    text=e.get("text") or "NULL", instruction=e.get("instruction"))
                p = embedder._preprocess_inputs([conv])
            p = {k: v.to(device) for k, v in p.items() if torch.is_tensor(v)}
            out = model(**p)
            embs.append(embedder._pooling_last(out.last_hidden_state, p["attention_mask"]))
        return torch.cat(embs, dim=0)


def _encode_micro_pretok_list(model, embedder, tensor_dicts, device):
    """Forward each pretokenized example (vision-safe); returns (N, D)."""
    if not tensor_dicts:
        return None
    embs = []
    for d in tensor_dicts:
        proc = {k: v.to(device) for k, v in d.items() if torch.is_tensor(v)}
        out = model(**proc)
        embs.append(embedder._pooling_last(out.last_hidden_state, proc["attention_mask"]))
    return torch.cat(embs, dim=0)


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def gradcache_step(
    embedder, model, raw_model, device,
    queries, positives, negatives, task_types, scores,
    micro_batch_size, args, pretokenized: bool = False,
):
    """
    GradCache training step:
    1. Encode all items in micro-batches with no_grad, cache embeddings
    2. Compute contrastive loss over the full cached embeddings
    3. Replay: for each micro-batch, re-encode with grad and backprop
       the portion of the loss gradient that corresponds to that chunk
    """
    mrl_dims = args.mrl_dims if args.use_mrl else None

    # --- Phase 1: Cache embeddings (no grad) ---
    all_roles = [queries, positives, negatives]
    cached = []
    for role_items in all_roles:
        if not role_items:
            cached.append(None)
            continue
        if pretokenized:
            tb = role_items
            n = len(tb)
            role_embs = []
            for s in range(0, n, micro_batch_size):
                e = min(s + micro_batch_size, n)
                chunk = tb[s:e]
                with torch.no_grad():
                    emb = _encode_micro_pretok_list(raw_model, embedder, chunk, device)
                role_embs.append(emb)
            cached.append(torch.cat(role_embs, dim=0))
        else:
            chunks = list(_chunked(role_items, micro_batch_size))
            role_embs = []
            for chunk in chunks:
                with torch.no_grad():
                    emb = _encode_micro(raw_model, embedder, chunk, device)
                    role_embs.append(emb)
            cached.append(torch.cat(role_embs, dim=0))

    q_cache, p_cache, n_cache = cached

    # --- Phase 2: Compute loss on full cached embeddings ---
    # Detach and require grad so we can get d(loss)/d(embedding)
    q_cache = q_cache.detach().requires_grad_(True)
    p_cache = p_cache.detach().requires_grad_(True)
    if n_cache is not None:
        n_cache = n_cache.detach().requires_grad_(True)

    has_sts = scores is not None and any(t == "sts" for t in task_types)
    if has_sts:
        if mrl_dims:
            loss = mrl_cosent_loss(q_cache, p_cache, scores,
                                   mrl_dims=mrl_dims, temperature=args.temperature)
        else:
            loss = cosent_loss(F.normalize(q_cache, dim=-1),
                               F.normalize(p_cache, dim=-1),
                               scores, temperature=args.temperature)
    else:
        hn = n_cache if (n_cache is not None and n_cache.shape[0] > 0) else None
        if mrl_dims:
            loss = mrl_infonce_loss(q_cache, p_cache, hn, mrl_dims=mrl_dims,
                                    temperature=args.temperature,
                                    stage=args.training_stage)
        else:
            loss = masked_infonce_loss(
                F.normalize(q_cache, dim=-1),
                F.normalize(p_cache, dim=-1),
                F.normalize(hn, dim=-1) if hn is not None else None,
                temperature=args.temperature, stage=args.training_stage)

    # Get embedding gradients
    grad_targets = [q_cache, p_cache]
    if n_cache is not None:
        grad_targets.append(n_cache)
    emb_grads = torch.autograd.grad(loss, grad_targets)
    q_grad, p_grad = emb_grads[0], emb_grads[1]
    n_grad = emb_grads[2] if n_cache is not None else None

    # --- Phase 3: Replay gradients through the model ---
    # Re-encode each micro-batch WITH grad and backprop using cached gradients
    all_role_items = [queries, positives, negatives]
    all_grads = [q_grad, p_grad, n_grad]

    for role_items, role_grad in zip(all_role_items, all_grads):
        if not role_items or role_grad is None:
            continue
        if pretokenized:
            tb = role_items
            n = len(tb)
            offset = 0
            for s in range(0, n, micro_batch_size):
                e = min(s + micro_batch_size, n)
                chunk = tb[s:e]
                chunk_size = len(chunk)
                chunk_grad = role_grad[offset:offset + chunk_size]
                for j, d in enumerate(chunk):
                    proc = {k: v.to(device) for k, v in d.items() if torch.is_tensor(v)}
                    out = model(**proc)
                    emb = embedder._pooling_last(
                        out.last_hidden_state, proc["attention_mask"])
                    emb.backward(chunk_grad[j: j + 1])
                offset += chunk_size
        else:
            chunks = list(_chunked(role_items, micro_batch_size))
            offset = 0
            for chunk in chunks:
                chunk_size = len(chunk)
                chunk_grad = role_grad[offset:offset + chunk_size]
                emb = _encode_micro(model, embedder, chunk, device)
                emb.backward(chunk_grad)
                offset += chunk_size

    return loss.detach().float().item()


def simple_step(embedder, model, device, queries, positives, negatives,
                task_types, scores, args):
    """Simple forward+backward — no GradCache. Contrastive batch = micro-batch."""
    mrl_dims = args.mrl_dims if args.use_mrl else None

    q_emb = _encode_micro(model, embedder, queries, device)
    p_emb = _encode_micro(model, embedder, positives, device)
    n_emb = _encode_micro(model, embedder, negatives, device) if negatives else None

    has_sts = scores is not None and any(t == "sts" for t in task_types)
    if has_sts:
        if mrl_dims:
            loss = mrl_cosent_loss(q_emb, p_emb, scores,
                                   mrl_dims=mrl_dims, temperature=args.temperature)
        else:
            loss = cosent_loss(F.normalize(q_emb, dim=-1),
                               F.normalize(p_emb, dim=-1),
                               scores, temperature=args.temperature)
    else:
        hn = n_emb if (n_emb is not None and n_emb.shape[0] > 0) else None
        if mrl_dims:
            loss = mrl_infonce_loss(q_emb, p_emb, hn, mrl_dims=mrl_dims,
                                    temperature=args.temperature,
                                    stage=args.training_stage)
        else:
            loss = masked_infonce_loss(
                F.normalize(q_emb, dim=-1),
                F.normalize(p_emb, dim=-1),
                F.normalize(hn, dim=-1) if hn is not None else None,
                temperature=args.temperature, stage=args.training_stage)

    loss.backward()
    return loss.detach().float().item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    pg = InitProcessGroupKwargs(timeout=timedelta(minutes=10))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        mixed_precision="bf16",
        kwargs_handlers=[pg])

    if args.seed:
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
        accelerator.init_trackers(
            project_name=wp, config=vars(args), init_kwargs={"wandb": wkw})

    mt = _detect_model_type(args.model_path)
    accelerator.print(f"Loading model from {args.model_path} (type={mt})")
    embedder = _load_embedder(
        args.model_path, args.max_length, mt, max_pixels=args.max_pixels)
    targets = LORA_QWEN3VL if mt == "qwen3vl" else LORA_QWEN35
    embedder.model = get_peft_model(embedder.model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets))
    if accelerator.is_main_process:
        embedder.model.print_trainable_parameters()
    if args.gradient_checkpointing:
        embedder.model.enable_input_require_grads()
        embedder.model.gradient_checkpointing_enable()

    use_gc = not args.no_grad_cache
    micro_bs = args.batch_size

    if use_gc:
        world = accelerator.num_processes
        if args.effective_batch_size % world != 0:
            raise ValueError(
                f"effective_batch_size ({args.effective_batch_size}) must be divisible "
                f"by num_processes ({world})."
            )
        dl_batch = args.effective_batch_size // world
    else:
        dl_batch = micro_bs

    subsets = args.subsets.split(",") if args.subsets else None
    tt_filter = args.task_types.split(",") if args.task_types else None
    pretok = bool(args.pretokenized_dir)
    if pretok:
        pad_id = embedder.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = embedder.tokenizer.eos_token_id or 0
        dataloader = build_pretokenized_dataloader(
            pretokenized_dir=args.pretokenized_dir,
            subsets=subsets,
            task_types=tt_filter,
            batch_size=dl_batch,
            num_workers=args.num_workers,
            shuffle=True,
            pad_token_id=int(pad_id),
        )
    else:
        dataloader = build_dataloader(
            subsets=subsets, task_types=tt_filter, split=args.dataset_split,
            image_dir=args.image_dir,
            max_samples_per_subset=args.max_samples_per_subset,
            cache_dir=args.cache_dir, batch_size=dl_batch,
            num_workers=args.num_workers, shuffle=True)

    optimizer = AdamW(embedder.model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)

    embedder.model, optimizer, dataloader = accelerator.prepare(
        embedder.model, optimizer, dataloader)

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    scheduler = accelerator.prepare(scheduler)

    raw_model = accelerator.unwrap_model(embedder.model)

    eff_batch = dl_batch * accelerator.num_processes * args.gradient_accumulation_steps
    mode = "GradCache" if use_gc else "simple"
    accelerator.print(
        f"Training: {len(dataloader)} steps/epoch, {total_steps} total steps, "
        f"{args.epochs} epochs, warmup={warmup_steps}\n"
        f"  mode={mode}, batch/rank={dl_batch}, micro_bs={micro_bs}, "
        f"grad_accum={args.gradient_accumulation_steps}, "
        f"effective_batch={eff_batch}, pretokenized={pretok}, "
        f"gpus={accelerator.num_processes}")

    global_step = 0
    embedder.model.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_steps, t0 = 0.0, 0, time.time()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(embedder.model):
                optimizer.zero_grad()

                scores = batch["scores"]
                if scores is not None:
                    scores = scores.to(accelerator.device)

                if use_gc:
                    step_loss = gradcache_step(
                        embedder, embedder.model, raw_model, accelerator.device,
                        batch["queries"], batch["positives"], batch["negatives"],
                        batch["task_types"], scores,
                        micro_batch_size=micro_bs, args=args,
                        pretokenized=pretok,
                    )
                    if accelerator.num_processes > 1:
                        for p in embedder.model.parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(
                                    p.grad, op=torch.distributed.ReduceOp.AVG)
                else:
                    step_loss = simple_step(
                        embedder, embedder.model, accelerator.device,
                        batch["queries"], batch["positives"], batch["negatives"],
                        batch["task_types"], scores, args,
                    )

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
                sps = (epoch_steps * eff_batch) / (time.time() - t0)
                logger.info(
                    "Epoch %d | Step %d/%d | Loss %.4f | LR %.2e | %.1f samples/s",
                    epoch, global_step, total_steps, avg, lr, sps)
                if args.use_wandb:
                    accelerator.log({
                        "train/loss": avg, "train/lr": lr,
                        "train/epoch": epoch, "train/global_step": global_step,
                        "train/samples_per_sec": sps,
                    }, step=global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_ckpt(accelerator, embedder, args, global_step)

        accelerator.print(
            f"Epoch {epoch} done. Avg loss: {epoch_loss / max(epoch_steps, 1):.4f}")

    _save_ckpt(accelerator, embedder, args, global_step, final=True)
    if args.use_wandb and accelerator.is_main_process:
        accelerator.end_training()
    accelerator.print("Training complete.")


def _save_ckpt(accelerator, embedder, args, global_step, final=False):
    if not accelerator.is_main_process:
        return
    d = Path(args.output_dir) / ("final" if final else f"checkpoint-{global_step}")
    accelerator.print(f"Saving to {d}")
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
    p.add_argument("--no_grad_cache", action="store_true",
                   help="Disable GradCache; use simple forward+backward. "
                        "Contrastive batch = batch_size per GPU.")
    p.add_argument("--subsets", default=None)
    p.add_argument("--task_types", default=None)
    p.add_argument("--dataset_split", default="diverse_instruction")
    p.add_argument("--image_dir", default=None)
    p.add_argument("--max_pixels", type=int, default=401408)
    p.add_argument("--max_samples_per_subset", type=int, default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument(
        "--pretokenized_dir",
        default=None,
        help="Load MMEB from pretokenized shards (scripts/pretokenize_mmeb.py).",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Micro-batch size per forward pass (fits in GPU mem)")
    p.add_argument(
        "--effective_batch_size",
        type=int,
        default=2048,
        help="Global contrastive batch (fixed across GPUs). Per-GPU batch = this / num_gpus.",
    )
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
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
