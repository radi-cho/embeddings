#!/usr/bin/env python3
"""Contrastive training for Qwen3.5-0.8B embedding model with LoRA."""
import argparse
import json
import logging
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.qwen35_embedding import Qwen35Embedder
from src.train.data_loader import build_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj",
]


def infonce_loss(query_embs: torch.Tensor, doc_embs: torch.Tensor, temperature: float = 0.02) -> torch.Tensor:
    sim = query_embs @ doc_embs.T / temperature
    labels = torch.arange(len(query_embs), device=sim.device)
    return F.cross_entropy(sim, labels)


def mrl_infonce_loss(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    dims: list = [64, 128, 256, 512, 1024],
    temperature: float = 0.02,
) -> torch.Tensor:
    total = 0.0
    for d in dims:
        q = F.normalize(query_embs[:, :d], dim=-1)
        d_ = F.normalize(doc_embs[:, :d], dim=-1)
        total += infonce_loss(q, d_, temperature)
    return total / len(dims)


def encode_batch(embedder: Qwen35Embedder, items: list) -> torch.Tensor:
    """Encode a batch with gradients enabled (for training)."""
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
    embs = embedder._pooling_last(hidden, mask)
    return F.normalize(embs, p=2, dim=-1)


def train(args):
    logger.info(f"Loading model from {args.model_path}")
    embedder = Qwen35Embedder(
        model_name_or_path=args.model_path,
        torch_dtype=torch.bfloat16,
        max_length=args.max_length,
    )

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
    )
    embedder.model = get_peft_model(embedder.model, lora_config)
    embedder.model.print_trainable_parameters()

    if args.gradient_checkpointing:
        embedder.model.enable_input_require_grads()
        embedder.model.gradient_checkpointing_enable()

    dataloader = build_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(
        embedder.model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    logger.info(f"Training: {len(dataloader)} batches/epoch, {total_steps} total steps, {args.epochs} epochs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    embedder.model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            queries = batch["queries"]
            positives = batch["positives"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                q_embs = encode_batch(embedder, queries)
                p_embs = encode_batch(embedder, positives)

                if args.use_mrl:
                    loss = mrl_infonce_loss(q_embs, p_embs, temperature=args.temperature)
                else:
                    loss = infonce_loss(q_embs, p_embs, temperature=args.temperature)

                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(embedder.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(f"Epoch {epoch} Step {global_step} Loss {avg_loss:.4f} LR {lr:.2e}")

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    embedder.model.save_pretrained(ckpt_dir)
                    embedder.tokenizer.save_pretrained(ckpt_dir)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch} finished. Avg loss: {avg_epoch_loss:.4f}")

    final_dir = output_dir / "final"
    embedder.model.save_pretrained(final_dir)
    embedder.tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete. Final model saved to {final_dir}")

    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/Qwen3.5-0.8B")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/embedding-poc")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--use_mrl", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
