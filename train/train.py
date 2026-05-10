#!/usr/bin/env python3
"""Multimodal contrastive pretraining for Qwen3.5 / Qwen3-VL embedding models.

DDP training with cross-GPU embedding gathering for contrastive loss:
  - Queries and positives are encoded on parallel CUDA streams.
  - Optional hard negatives (classification wrong-class labels and/or mined
    K negatives for retrieval/VQA) are encoded in the same no_sync context.
  - Embeddings are gathered across ranks via GatherWithGrad, then routed to
    per-task losses (InfoNCE stage 1/2, CoSENT, classification InfoNCE).
  - OOM on any rank → all ranks skip the step in sync (no DDP deadlock).

Stage toggle:
  --training_stage 2 + --mined_dir  =>  build_stage2_dataset (paper §4.3).
  --use_optimized_mix               =>  build_stage1_optimized_dataset (~5.1M mix).
  else (Stage 1, no optimized flag) =>  build_mixed_dataset (full corpus).
"""

import argparse
import contextlib
import json
import logging
import os
import random as _random
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.dataset import (
    build_dataloader, build_mixed_dataset, build_stage1_optimized_dataset,
    build_stage2_dataset,
    collate_embedding_batch, TaskStratifiedSampler,
)
from train.losses import (
    classification_infonce_loss, cosent_loss,
    hardness_weighted_infonce_loss, masked_infonce_loss,
    mrl_classification_loss, mrl_cosent_loss,
    mrl_hardness_weighted_infonce_loss, mrl_infonce_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch GatedDeltaNet to keep gating in bf16 (avoids fp32 cast that slows fla)
# ---------------------------------------------------------------------------

def _patch_gdn_bf16():
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    except ImportError:
        return False

    def _bf16_forward(self, hidden_states, cache_params=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        conv_state = None
        recurrent_state = None
        if cache_params is not None:
            layer_cache = cache_params.layers[self.layer_idx]
            conv_state = getattr(layer_cache, "conv_states", None)
            recurrent_state = getattr(layer_cache, "recurrent_states", None)
        # HF cache can report has_previous_state while conv/recurrent tensors are still
        # unset — causal_conv1d_update requires non-None conv_state (see instruction_pipeline crash).
        use_precomputed_states = (
            cache_params is not None
            and getattr(cache_params, "has_previous_state", False)
            and conv_state is not None
            and recurrent_state is not None
        )

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv, conv_state,
                self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv, weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias, activation=self.activation, seq_idx=None)
            else:
                _INT32_MAX = 2**31 - 1
                if mixed_qkv.numel() > _INT32_MAX:
                    chunks = []
                    cs = max(1, batch_size // 2)
                    for s in range(0, batch_size, cs):
                        c = mixed_qkv[s:s + cs]
                        chunks.append(F.silu(self.conv1d(c)[:, :, :seq_len]))
                    mixed_qkv = torch.cat(chunks, dim=0)
                else:
                    mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        target_dtype = query.dtype
        g = -self.A_log.to(target_dtype).exp() * F.softplus(a.to(target_dtype) + self.dt_bias.to(target_dtype))

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None, output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True)
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True)

        if cache_params is not None and last_recurrent_state is not None:
            cache_params.layers[self.layer_idx].recurrent_states = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    Qwen3_5GatedDeltaNet.forward = _bf16_forward
    return True


if _patch_gdn_bf16():
    logger.info("Patched Qwen3_5GatedDeltaNet for bf16 gating (fla fast path)")

LORA_TARGETS = {
    "qwen35": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    "qwen3vl": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj",
                 "o_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"],
}


# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------

def _world():
    return dist.get_world_size() if dist.is_initialized() else 1

def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


class GatherWithGrad(torch.autograd.Function):
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


# Task-type <-> integer codec used by _gather_metadata's tensor all_gather
# (avoids pickle overhead of all_gather_object).
_TT_ENCODE = {"sts": 0, "classification": 2}
_TT_DECODE = {0: "sts", 1: "other", 2: "classification"}


def _gather_metadata(task_types, scores, device):
    """Gather task_types and scores across ranks via tensor ops (no pickle)."""
    if _world() == 1:
        return task_types, scores
    bs = len(task_types)
    tt_tensor = torch.tensor(
        [_TT_ENCODE.get(t, 1) for t in task_types],
        device=device, dtype=torch.int32)
    tt_parts = [torch.zeros_like(tt_tensor) for _ in range(_world())]
    dist.all_gather(tt_parts, tt_tensor.contiguous())
    g_tt = [_TT_DECODE.get(v, "other")
            for v in torch.cat(tt_parts, dim=0).tolist()]

    local_scores = scores if scores is not None else torch.zeros(bs, device=device)
    parts = [torch.zeros_like(local_scores) for _ in range(_world())]
    dist.all_gather(parts, local_scores.contiguous())
    g_scores = torch.cat(parts, dim=0)
    has_scores = scores is not None
    has_flag = torch.tensor([1.0 if has_scores else 0.0], device=device)
    dist.all_reduce(has_flag)
    if has_flag.item() > 0:
        return g_tt, g_scores
    return g_tt, None


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_batch(model, embedder, items, device):
    convs = [embedder.format_model_input(
        text=e.get("text"), image=e.get("image"),
        video=e.get("video"), instruction=e.get("instruction"))
        for e in items]
    try:
        proc = embedder._preprocess_inputs(convs)
    except Exception as exc:
        logger.warning("Vision preprocess failed (%s), text-only fallback", exc)
        convs = [embedder.format_model_input(
            text=e.get("text") or "NULL", instruction=e.get("instruction"))
            for e in items]
        proc = embedder._preprocess_inputs(convs)
    proc = {k: v.to(device, non_blocking=True) for k, v in proc.items()
            if torch.is_tensor(v)}
    out = model(**proc)
    return embedder._pooling_last(out.last_hidden_state, proc["attention_mask"])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _compute_loss(q_emb, p_emb, task_types, scores, args,
                   q_local=None, p_local=None, local_task_types=None, hn_emb=None,
                   mined_hn_gathered=None):
    """Compute training loss with stage-dependent task routing.

    Stage 1 (paper §4.1): plain InfoNCE (Eq. 1) for ALL data, including
    classification. STS rows use CoSENT. No task-specific loss routing.

    Stage 2 (paper §4.2): per-task-type routing:
      - classification → classification_infonce_loss (wrong-class labels only)
      - retrieval/VQA → InfoNCE stage=2 + gathered mined hard negatives
      - STS → CoSENT

    The final loss is the mean of all contributing terms.
    """
    mrl = args.mrl_dims if args.use_mrl else None
    device = q_emb.device
    stage = args.training_stage

    sts_idx = [i for i, t in enumerate(task_types) if t == "sts"]
    losses = []

    # =======================================================================
    # Stage 1: everything (except STS) gets InfoNCE on gathered q/p
    # If hardness_alpha > 0, use hardness-weighted variant (LLaVE).
    # =======================================================================
    if stage == 1:
        non_sts_idx = [i for i, t in enumerate(task_types) if t != "sts"]
        if non_sts_idx:
            ci = torch.tensor(non_sts_idx, device=device)
            q, p = q_emb[ci], p_emb[ci]
            hn_ret = None
            if mined_hn_gathered is not None and mined_hn_gathered.shape[0] > 0:
                Bg = q_emb.shape[0]
                K = mined_hn_gathered.shape[0] // Bg if Bg > 0 else 0
                if K > 0 and mined_hn_gathered.shape[0] == Bg * K:
                    hn_reshaped = mined_hn_gathered.view(Bg, K, -1)
                    hn_ret = hn_reshaped[ci].reshape(-1, hn_reshaped.shape[-1])

            use_hw = getattr(args, "hardness_alpha", 0.0) > 0.0
            if mrl:
                if use_hw:
                    losses.append(mrl_hardness_weighted_infonce_loss(
                        q, p, hn_ret, mrl_dims=mrl,
                        temperature=args.temperature, alpha=args.hardness_alpha,
                        stage=1))
                else:
                    losses.append(mrl_infonce_loss(
                        q, p, hn_ret, mrl_dims=mrl,
                        temperature=args.temperature, stage=1))
            else:
                qn = F.normalize(q, dim=-1)
                pn = F.normalize(p, dim=-1)
                hnn = F.normalize(hn_ret, dim=-1) if hn_ret is not None else None
                if use_hw:
                    losses.append(hardness_weighted_infonce_loss(
                        qn, pn, hnn,
                        temperature=args.temperature, alpha=args.hardness_alpha,
                        stage=1))
                else:
                    losses.append(masked_infonce_loss(
                        qn, pn, hnn,
                        temperature=args.temperature, stage=1))

    # =======================================================================
    # Stage 2: per-task-type routing with hard negatives
    # =======================================================================
    else:
        con_idx = [i for i, t in enumerate(task_types)
                   if t not in ("sts", "classification")]

        # Classification: rank-local + optional wrong-class hard negatives
        lcls_idx = [i for i, t in enumerate(local_task_types)
                    if t == "classification"]
        if lcls_idx:
            lci = torch.tensor(lcls_idx, device=device)
            q, p = q_local[lci], p_local[lci]
            if hn_emb is not None and hn_emb.shape[0] > 0:
                if mrl:
                    losses.append(mrl_classification_loss(
                        q, p, hn_emb, mrl_dims=mrl, temperature=args.temperature))
                else:
                    losses.append(classification_infonce_loss(
                        F.normalize(q, dim=-1), F.normalize(p, dim=-1),
                        F.normalize(hn_emb, dim=-1), temperature=args.temperature))
            else:
                if mrl:
                    losses.append(mrl_infonce_loss(q, p, None, mrl_dims=mrl,
                                  temperature=args.temperature, stage=stage))
                else:
                    losses.append(masked_infonce_loss(
                        F.normalize(q, dim=-1), F.normalize(p, dim=-1), None,
                        temperature=args.temperature, stage=stage))

        # Retrieval / VQA: gathered in-batch + gathered mined HNs
        if con_idx:
            ci = torch.tensor(con_idx, device=device)
            q_ret = q_emb[ci]
            p_ret = p_emb[ci]

            hn_ret = None
            if mined_hn_gathered is not None and mined_hn_gathered.shape[0] > 0:
                B_gathered = q_emb.shape[0]
                K = mined_hn_gathered.shape[0] // B_gathered if B_gathered > 0 else 0
                if K > 0 and mined_hn_gathered.shape[0] == B_gathered * K:
                    hn_reshaped = mined_hn_gathered.view(B_gathered, K, -1)
                    hn_ret = hn_reshaped[ci].reshape(-1, hn_reshaped.shape[-1])

            if mrl:
                losses.append(mrl_infonce_loss(
                    q_ret, p_ret, hn_ret,
                    mrl_dims=mrl, temperature=args.temperature, stage=stage))
            else:
                losses.append(masked_infonce_loss(
                    F.normalize(q_ret, dim=-1),
                    F.normalize(p_ret, dim=-1),
                    F.normalize(hn_ret, dim=-1) if hn_ret is not None else None,
                    temperature=args.temperature, stage=stage))

    # STS: CoSENT loss (both stages)
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
# Training step — parallel q/p encoding via CUDA streams
# ---------------------------------------------------------------------------

_stream_q = None
_stream_p = None


def _get_streams(device):
    global _stream_q, _stream_p
    if _stream_q is None:
        _stream_q = torch.cuda.Stream(device=device)
        _stream_p = torch.cuda.Stream(device=device)
    return _stream_q, _stream_p


def _hidden_size_from_model(model):
    """Return hidden_size from a HF config, checking nested text_config for VLMs."""
    cfg = model.module.config if hasattr(model, "module") else model.config
    for attr in ("hidden_size", "d_model"):
        v = getattr(cfg, attr, None)
        if v:
            return v
    if hasattr(cfg, "text_config"):
        v = getattr(cfg.text_config, "hidden_size", None)
        if v:
            return v
    raise AttributeError(f"Cannot determine hidden_size from config: {type(cfg)}")


def _train_step(embedder, model, device, batch, args):
    """DDP-safe training step with full OOM protection.

    Uses model.no_sync() so DDP gradient hooks don't fire during backward
    (preventing deadlock if one rank OOMs mid-backward). Gradients are
    manually all-reduced after a successful backward on all ranks.
    Returns loss float, or None if any rank OOMed at any phase.

    With gradient checkpointing enabled, OOMs during backward can manifest as
    RuntimeError or even non-Python CUDA errors that bypass the normal
    try/except. We catch *all* exceptions during backward, and also proactively
    skip steps when free GPU memory is dangerously low before backward begins.
    """
    sq, sp = _get_streams(device)
    bs = len(batch["queries"])
    hidden_dim = _hidden_size_from_model(model)
    oom = False

    # --- Phase 1: forward encode (inside no_sync so no DDP hooks) ----------
    no_sync = model.no_sync if hasattr(model, "no_sync") else contextlib.nullcontext
    hn_local = None
    mined_hn_local = None
    with no_sync():
        try:
            with torch.cuda.stream(sq):
                q_local = _encode_batch(model, embedder, batch["queries"], device)
            with torch.cuda.stream(sp):
                p_local = _encode_batch(model, embedder, batch["positives"], device)
            sq.synchronize()
            sp.synchronize()

            # Classification wrong-class labels (Stage 2 only).
            if args.training_stage >= 2:
                neg_texts = batch.get("negative_texts")
                if neg_texts is not None:
                    cls_mask = [t == "classification" for t in batch["task_types"]]
                    neg_items = []
                    for i, is_cls in enumerate(cls_mask):
                        if is_cls and neg_texts[i]:
                            for t in neg_texts[i]:
                                neg_items.append({"text": t, "instruction": "Represent the label."})
                    if neg_items:
                        hn_local = _encode_batch(model, embedder, neg_items, device)

            # Mined / JSONL hard negatives (Stage 1 optimized mix + Stage 2 retrieval).
            mined_hns = batch.get("hard_negatives")
            if mined_hns and max(len(h) for h in mined_hns) > 0:
                flat = [h for row in mined_hns for h in row]
                if flat:
                    _hn_bs = bs
                    hn_parts = []
                    for _i in range(0, len(flat), _hn_bs):
                        hn_parts.append(
                            _encode_batch(model, embedder, flat[_i:_i + _hn_bs], device))
                    mined_hn_local = torch.cat(hn_parts, dim=0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom = True
            q_local = torch.zeros(bs, hidden_dim, device=device, dtype=torch.bfloat16)
            p_local = torch.zeros(bs, hidden_dim, device=device, dtype=torch.bfloat16)
            hn_local = None
            mined_hn_local = None
            logger.warning("OOM during forward on rank %d", _rank())

    q = GatherWithGrad.apply(q_local) if _world() > 1 else q_local
    p = GatherWithGrad.apply(p_local) if _world() > 1 else p_local
    mined_hn_gathered = None
    if mined_hn_local is not None and _world() > 1:
        mined_hn_gathered = GatherWithGrad.apply(mined_hn_local)
    elif mined_hn_local is not None:
        mined_hn_gathered = mined_hn_local

    oom_flag = torch.tensor([1.0 if oom else 0.0], device=device)
    if _world() > 1:
        dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
    if oom_flag.item() > 0:
        return None

    # --- Pre-backward memory check: skip if recomputation will likely OOM ---
    # Gradient checkpointing roughly doubles peak memory during backward vs.
    # forward-only because activations are recomputed. If free memory after
    # forward is below a safety threshold, skip proactively.
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        used_frac = 1.0 - (free_mem / total_mem)
        if used_frac > 0.92:
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            used_frac = 1.0 - (free_mem / total_mem)
            if used_frac > 0.90:
                logger.warning(
                    "Proactive skip: %.1f%% GPU memory used after forward on rank %d "
                    "(free=%.0fMB); skipping backward to avoid OOM during recomputation.",
                    used_frac * 100, _rank(), free_mem / 1e6)
                oom = True
                oom_flag.fill_(1.0)
                if _world() > 1:
                    dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                return None

    # --- Phase 2: loss + backward (still no DDP sync — manual reduce after) -
    scores = batch["scores"].to(device) if batch["scores"] is not None else None
    g_tt, g_scores = _gather_metadata(batch["task_types"], scores, device)
    loss = _compute_loss(q, p, g_tt, g_scores, args,
                         q_local=q_local, p_local=p_local,
                         local_task_types=batch["task_types"], hn_emb=hn_local,
                         mined_hn_gathered=mined_hn_gathered)
    loss_val = loss.detach().float().item()

    with no_sync():
        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom = True
            logger.warning("OOM during backward on rank %d", _rank())
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        except RuntimeError as e:
            # Gradient checkpointing recomputation can raise RuntimeError wrapping
            # a CUDA OOM or other CUDA error instead of OutOfMemoryError directly.
            torch.cuda.empty_cache()
            oom = True
            logger.warning("RuntimeError during backward on rank %d: %s", _rank(), str(e)[:200])
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        except Exception as e:
            # Catch-all for any other unexpected exception during backward to
            # prevent one rank from dying and causing NCCL timeout on others.
            torch.cuda.empty_cache()
            oom = True
            logger.error("Unexpected error during backward on rank %d: %s", _rank(), str(e)[:200])
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

    oom_flag.fill_(1.0 if oom else 0.0)
    if _world() > 1:
        dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
    if oom_flag.item() > 0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        return None

    # --- Phase 3: manual gradient all-reduce (all ranks healthy) -----------
    if _world() > 1:
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(_world())

    return loss_val


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _detect_model_type(model_path):
    cfg = Path(model_path) / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            if "qwen3_vl" in json.load(f).get("model_type", ""):
                return "qwen3vl"
    return "qwen35"


def _load_embedder(model_path, max_length, model_type, max_pixels=None,
                    video_total_pixels=None):
    kw = {}
    if max_pixels:
        kw["max_pixels"] = max_pixels
    if video_total_pixels:
        kw["total_pixels"] = video_total_pixels
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


def _save_ckpt(accelerator, embedder, args, step, optimizer=None,
               scheduler=None, epoch=0, epoch_loss=0.0, epoch_steps=0, final=False):
    d = Path(args.output_dir) / ("final" if final else f"checkpoint-{step}")
    accelerator.print(f"Saving to {d}")
    if accelerator.is_main_process:
        d.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.unwrap_model(embedder.model).save_pretrained(d)
    if accelerator.is_main_process:
        tok = getattr(embedder, "processor", None) or getattr(embedder, "tokenizer", None)
        if tok:
            tok.save_pretrained(d)

    train_state = {
        "global_step": step,
        "epoch": epoch,
        "epoch_loss": epoch_loss,
        "epoch_steps": epoch_steps,
    }
    state_dir = d / "train_state"
    accelerator.save_state(state_dir)
    if accelerator.is_main_process:
        torch.save(train_state, state_dir / "extra_state.pt")
        rng = {
            "python": _random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng, state_dir / "rng_states.pt")
    accelerator.wait_for_everyone()


def _load_ckpt(accelerator, args, resume_dir):
    """Restore optimizer, scheduler, RNG, and training counters from checkpoint."""
    state_dir = Path(resume_dir) / "train_state"
    if not state_dir.exists():
        raise FileNotFoundError(f"No train_state in {resume_dir}")
    accelerator.load_state(state_dir)
    extra = torch.load(state_dir / "extra_state.pt", map_location="cpu", weights_only=True)
    rng_path = state_dir / "rng_states.pt"
    if rng_path.exists():
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        _random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.random.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
    accelerator.print(f"Resumed from {resume_dir} at step {extra['global_step']}")
    return extra


def _build_dataloader(args, embedder, batch_size):
    subsets = args.subsets.split(",") if args.subsets else None
    tt_filter = args.task_types.split(",") if args.task_types else None

    # Without a data_dir root we fall back to the legacy MMEB-only builder.
    if not args.data_dir:
        return build_dataloader(
            subsets=subsets, task_types=tt_filter, split=args.dataset_split,
            image_dir=args.image_dir, max_samples_per_subset=args.max_samples_per_subset,
            cache_dir=args.cache_dir,
            batch_size=batch_size, num_workers=args.num_workers, shuffle=True)

    from torch.utils.data import DistributedSampler

    # Stage 2: curated mined mix + stratified batches.
    if args.training_stage == 2 and args.mined_dir:
        dataset = build_stage2_dataset(
            data_dir=args.data_dir, mined_dir=args.mined_dir,
            image_dir=args.image_dir,
            mmeb_split=args.dataset_split,
            max_samples_per_subset=args.max_samples_per_subset,
            cache_dir=args.cache_dir)
        sampler = TaskStratifiedSampler(
            dataset, batch_size=batch_size,
            num_replicas=_world(), rank=_rank(), seed=args.seed)
    elif args.use_optimized_mix:
        dataset = build_stage1_optimized_dataset(
            data_dir=args.data_dir,
            image_dir=args.image_dir,
            megapairs_image_dir=args.megapairs_image_dir,
            mined_dir=args.mined_dir,
            mmeb_split=args.dataset_split,
            cache_dir=args.cache_dir,
        )
        sampler = DistributedSampler(
            dataset, num_replicas=_world(), rank=_rank(),
            shuffle=True, seed=args.seed) if _world() > 1 else None
    else:
        dataset = build_mixed_dataset(
            data_dir=args.data_dir, image_dir=args.image_dir,
            megapairs_image_dir=args.megapairs_image_dir,
            mmeb_split=args.dataset_split,
            max_samples_per_subset=args.max_samples_per_subset,
            cache_dir=args.cache_dir)
        sampler = DistributedSampler(dataset, num_replicas=_world(),
                                     rank=_rank(), shuffle=True, seed=args.seed) if _world() > 1 else None

    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, shuffle=(sampler is None),
                      collate_fn=collate_embedding_batch,
                      num_workers=args.num_workers, drop_last=True,
                      pin_memory=True, persistent_workers=args.num_workers > 0,
                      prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        mixed_precision="bf16",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=5))])

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
    is_lora_checkpoint = (Path(args.model_path) / "adapter_config.json").exists()
    if is_lora_checkpoint:
        with open(Path(args.model_path) / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path")
        if not base_model_path or not Path(base_model_path).exists():
            raise RuntimeError(
                f"Cannot resolve base_model_name_or_path={base_model_path!r} "
                f"from {args.model_path}/adapter_config.json. "
                f"Merge the LoRA adapter first or fix the path.")
        accelerator.print(f"  LoRA adapter detected; loading base model from {base_model_path}")
        embedder = _load_embedder(base_model_path, args.max_length, mt, args.max_pixels,
                                  args.video_total_pixels)
        from peft import PeftModel
        embedder.model = PeftModel.from_pretrained(
            embedder.model, args.model_path, is_trainable=True)
        accelerator.print("  Loaded LoRA adapter weights from checkpoint (trainable)")
    else:
        embedder = _load_embedder(args.model_path, args.max_length, mt, args.max_pixels,
                                  args.video_total_pixels)
        embedder.model = get_peft_model(embedder.model, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS[mt]))
    if accelerator.is_main_process:
        embedder.model.print_trainable_parameters()
    if args.gradient_checkpointing:
        embedder.model.enable_input_require_grads()
        embedder.model.gradient_checkpointing_enable()

    if args.compile:
        accelerator.print("Compiling model with torch.compile ...")
        embedder.model = torch.compile(embedder.model)

    world = accelerator.num_processes
    bs = args.batch_size
    effective_bs = bs * world * args.gradient_accumulation_steps
    accelerator.print(
        f"\n  Batch size: {bs}/GPU x {world} GPUs"
        f" x {args.gradient_accumulation_steps} accum = {effective_bs} effective\n"
        f"  LR: {args.lr}  |  Epochs: {args.epochs}  |  Seed: {args.seed}")

    dataloader = _build_dataloader(args, embedder, bs)
    optimizer = AdamW(embedder.model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, fused=True)
    embedder.model, optimizer = accelerator.prepare(embedder.model, optimizer)

    total_steps = len(dataloader) * args.epochs
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = accelerator.prepare(get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps))
    accelerator.print(f"  Steps: {total_steps} ({len(dataloader)}/epoch), warmup={warmup}")

    global_step = 0
    start_epoch = 0
    epoch_loss_init, epoch_steps_init = 0.0, 0
    if args.resume_from:
        extra = _load_ckpt(accelerator, args, args.resume_from)
        global_step = extra["global_step"]
        start_epoch = extra["epoch"]
        epoch_loss_init = extra.get("epoch_loss", 0.0)
        epoch_steps_init = extra.get("epoch_steps", 0)

    embedder.model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = epoch_loss_init if epoch == start_epoch else 0.0
        epoch_steps = epoch_steps_init if epoch == start_epoch else 0
        t0 = time.time()
        for attr in ('batch_sampler', 'sampler'):
            s = getattr(dataloader, attr, None)
            if s and hasattr(s, 'set_epoch'):
                s.set_epoch(epoch)
                break
        steps_to_skip = epoch_steps_init if epoch == start_epoch else 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < steps_to_skip:
                continue
            optimizer.zero_grad()
            step_loss = _train_step(
                embedder, embedder.model, accelerator.device, batch, args)

            if step_loss is None:
                optimizer.zero_grad()
                scheduler.step()
                epoch_steps += 1
                global_step += 1
                if accelerator.is_main_process:
                    logger.warning("Skipping step %d (OOM on >= 1 rank)", global_step)
                continue

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
                _save_ckpt(accelerator, embedder, args, global_step,
                           optimizer=optimizer, scheduler=scheduler,
                           epoch=epoch, epoch_loss=epoch_loss, epoch_steps=epoch_steps)

            if args.max_steps and global_step >= args.max_steps:
                accelerator.print(f"Reached max_steps={args.max_steps}, stopping.")
                break

        accelerator.print(f"Epoch {epoch} done. Loss: {epoch_loss / max(epoch_steps, 1):.4f}")

        if args.max_steps and global_step >= args.max_steps:
            break

    _save_ckpt(accelerator, embedder, args, global_step,
               optimizer=optimizer, scheduler=scheduler,
               epoch=epoch, epoch_loss=epoch_loss, epoch_steps=epoch_steps, final=True)
    if args.use_wandb and accelerator.is_main_process:
        accelerator.end_training()
    accelerator.print("Done.")


def main():
    p = argparse.ArgumentParser(description="Multimodal contrastive pretraining")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="outputs/qwen35-embedding-train")
    p.add_argument("--resume_from", default=None,
                   help="Checkpoint dir to resume from (restores optimizer, scheduler, RNG, step)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_pixels", type=int, default=1310720)
    p.add_argument("--video_total_pixels", type=int, default=9216000)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--use_optimized_mix", action="store_true",
                   help="Stage 1: use build_stage1_optimized_dataset (~5.1M instruction-aware mix)")
    p.add_argument("--mined_dir", default=None,
                   help="Stage 2: required for build_stage2_dataset. "
                        "Stage 1 optimized mix: optional dir of mined *.jsonl (incl. classification_hn.jsonl).")
    p.add_argument("--image_dir", default=None)
    p.add_argument("--megapairs_image_dir", default=None)
    p.add_argument("--dataset_split", default="diverse_instruction")
    p.add_argument("--subsets", default=None)
    p.add_argument("--task_types", default=None)
    p.add_argument("--max_samples_per_subset", type=int, default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Stop after this many steps (default: full epoch)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--hardness_alpha", type=float, default=0.0,
                   help="LLaVE hardness-weighted loss alpha (0=disabled, 9=paper default)")
    p.add_argument("--training_stage", type=int, default=1, choices=[1, 2])
    p.add_argument("--use_mrl", action="store_true", default=True)
    p.add_argument("--no_mrl", action="store_true")
    p.add_argument("--mrl_dims", default="1024,256,64")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model before training")
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
