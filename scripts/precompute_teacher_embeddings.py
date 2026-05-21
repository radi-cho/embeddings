#!/usr/bin/env python3
"""Pre-compute teacher embeddings from Qwen3-VL-Embedding-8B.

Supports:
  - MMEB subsets (diverse_instruction or original splits)
  - Text triplet datasets (msmarco, quora, allnli)
  - VisRAG document retrieval datasets (synthetic, indomain)

Outputs 1024-dim normalized embeddings as numpy arrays (float16) to data/teacher_embeddings/.

Usage:
    # MMEB subsets
    python scripts/precompute_teacher_embeddings.py --subsets "ImageNet_1K,MSCOCO"
    python scripts/precompute_teacher_embeddings.py --split original --prefix orig_

    # Text datasets
    python scripts/precompute_teacher_embeddings.py --text-dataset msmarco

    # VisRAG (sharded for parallelism)
    python scripts/precompute_teacher_embeddings.py --visrag synthetic --shard 0 --num-shards 6
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TEACHER_MODEL = str(ROOT / "models/checkpoints/Qwen3-VL-Embedding-8B")
OUTPUT_DIR = ROOT / "data/teacher_embeddings"
EMBED_DIM = 1024

MMEB_ALL_SUBSETS = [
    "A-OKVQA", "CIRR", "ChartQA", "DocVQA", "HatefulMemes",
    "ImageNet_1K", "InfographicsVQA", "MSCOCO", "MSCOCO_i2t",
    "MSCOCO_t2i", "N24News", "NIGHTS", "OK-VQA", "SUN397",
    "VOC2007", "VisDial", "Visual7W", "VisualNews_i2t",
    "VisualNews_t2i", "WebQA",
]


def load_teacher(model_path, device="cuda:0"):
    """Load Qwen3-VL-Embedding-8B and return an embedder object."""
    from transformers import AutoModel, AutoProcessor
    logger.info("Loading teacher model from %s on %s", model_path, device)
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    return model, processor


def format_input(text=None, image=None, instruction="Represent the user's input."):
    """Format a single sample for the teacher model."""
    content = []
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    if image is not None:
        if isinstance(image, str):
            content.append({"type": "image", "image": "file://" + image if not image.startswith(("http", "file://")) else image})
        else:
            content.append({"type": "image", "image": image})
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": "NULL"})
    return conversation


def embed_batch(model, processor, conversations, device, dim=EMBED_DIM):
    """Embed a batch of conversations, return (N, dim) normalized float16 tensor."""
    from qwen_vl_utils.vision_process import process_vision_info

    texts = processor.tokenizer.apply_chat_template(
        conversations, add_generation_prompt=False, tokenize=False)
    if isinstance(texts, str):
        texts = [texts]
    pad_token = "<|endoftext|>"
    texts = [t.rstrip() + pad_token for t in texts]

    has_vision = any(
        any(item.get("type") in ("image", "video")
            for msg in conv for item in (msg.get("content") if isinstance(msg.get("content"), list) else []))
        for conv in conversations
    )

    if has_vision:
        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations, return_video_metadata=True, return_video_kwargs=True)
        except Exception:
            images, video_inputs, video_kwargs = None, None, {"do_sample_frames": False}
        if video_inputs is not None:
            videos_list, video_metadata = zip(*video_inputs)
            videos_list, video_metadata = list(videos_list), list(video_metadata)
        else:
            videos_list, video_metadata = None, None
        inputs = processor(
            text=texts, images=images, videos=videos_list, video_metadata=video_metadata,
            padding=True, do_resize=True, return_tensors="pt", **video_kwargs)
    else:
        inputs = processor.tokenizer(
            texts, truncation=True, max_length=4096, padding=True, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        flipped = attention_mask.flip(dims=[1])
        last_pos = attention_mask.shape[1] - flipped.argmax(dim=1) - 1
        row_idx = torch.arange(last_hidden.shape[0], device=device)
        embeddings = last_hidden[row_idx, last_pos]
        embeddings = embeddings[:, :dim]
        embeddings = F.normalize(embeddings.float(), p=2, dim=-1)

    return embeddings.cpu().half().numpy()


def process_subset(model, processor, subset_name, device, image_dir, batch_size, output_dir,
                   split="diverse_instruction", prefix=""):
    """Process one MMEB subset and save query/positive embeddings."""
    from datasets import load_dataset
    from train.dataset import mmeb_query_instruction, _clean_mmeb_text, _load_image

    q_path = output_dir / f"{prefix}{subset_name}_queries.npy"
    p_path = output_dir / f"{prefix}{subset_name}_positives.npy"
    if q_path.exists() and p_path.exists():
        logger.info("SKIP %s%s (already exists)", prefix, subset_name)
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    ds = load_dataset("TIGER-Lab/MMEB-train", subset_name, split=split,
                      token=hf_token if hf_token else None)
    n = len(ds)
    logger.info("Processing %s: %d samples", subset_name, n)

    q_embs = np.zeros((n, EMBED_DIM), dtype=np.float16)
    p_embs = np.zeros((n, EMBED_DIM), dtype=np.float16)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_rows = ds[start:end]

        q_convs = []
        p_convs = []
        for i in range(end - start):
            idx = start + i
            qry = batch_rows["qry"][i] if "qry" in batch_rows else ""
            qry_img_path = batch_rows.get("qry_image_path", [""])[i]
            pos_text = batch_rows.get("pos_text", [""])[i]
            pos_img_path = batch_rows.get("pos_image_path", [""])[i]

            q_text = _clean_mmeb_text(qry) if qry else None
            q_img = _load_image(qry_img_path, image_dir) if qry_img_path else None
            q_inst = mmeb_query_instruction(subset_name, idx, qry, qry_img_path, pos_text, pos_img_path)

            p_text = _clean_mmeb_text(pos_text) if pos_text else None
            p_img = _load_image(pos_img_path, image_dir) if pos_img_path else None

            q_convs.append(format_input(text=q_text, image=q_img, instruction=q_inst))
            p_convs.append(format_input(text=p_text, image=p_img))

        try:
            q_batch_emb = embed_batch(model, processor, q_convs, device)
            p_batch_emb = embed_batch(model, processor, p_convs, device)
            q_embs[start:end] = q_batch_emb
            p_embs[start:end] = p_batch_emb
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("OOM at %s batch %d-%d, retrying with bs=1", subset_name, start, end)
            for i in range(end - start):
                try:
                    q_embs[start + i] = embed_batch(model, processor, [q_convs[i]], device)
                    p_embs[start + i] = embed_batch(model, processor, [p_convs[i]], device)
                except Exception as e:
                    logger.error("Failed sample %d in %s: %s", start + i, subset_name, str(e)[:100])
                    torch.cuda.empty_cache()

        if (start // batch_size) % 50 == 0 and start > 0:
            logger.info("  %s: %d/%d done", subset_name, start, n)

    np.save(str(q_path), q_embs)
    np.save(str(p_path), p_embs)
    logger.info("Saved %s: queries=%s positives=%s", subset_name, q_path.name, p_path.name)


def process_text_dataset(model, processor, name, jsonl_path, output_dir, device,
                         max_samples=None, batch_size=128):
    """Process a text triplet dataset (msmarco, quora, allnli)."""
    import json

    q_path = output_dir / f"text_{name}_queries.npy"
    p_path = output_dir / f"text_{name}_positives.npy"
    if q_path.exists() and p_path.exists():
        logger.info("SKIP text_%s (already exists)", name)
        return

    logger.info("Loading text dataset '%s' from %s", name, jsonl_path)
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break

    queries = [r.get("query", "") for r in rows]
    positives = [r.get("positive", "") for r in rows]
    n = len(queries)

    q_embs = np.zeros((n, EMBED_DIM), dtype=np.float16)
    p_embs = np.zeros((n, EMBED_DIM), dtype=np.float16)

    logger.info("Embedding %d queries...", n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        convs = [format_input(text=t, instruction="Represent the user's input.") for t in queries[start:end]]
        q_embs[start:end] = embed_batch(model, processor, convs, device)
        if start > 0 and (start // batch_size) % 20 == 0:
            logger.info("  queries: %d/%d", start, n)

    logger.info("Embedding %d positives...", n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        convs = [format_input(text=t) for t in positives[start:end]]
        p_embs[start:end] = embed_batch(model, processor, convs, device)
        if start > 0 and (start // batch_size) % 20 == 0:
            logger.info("  positives: %d/%d", start, n)

    np.save(str(q_path), q_embs)
    np.save(str(p_path), p_embs)
    logger.info("Saved text_%s: %d samples", name, n)


def process_visrag(model, processor, name, data_dir, output_dir, device,
                   text_bs=128, img_bs=8, shard=0, num_shards=1):
    """Process a VisRAG dataset (query=text, positive=document image).

    Supports sharding for parallel processing across GPUs.
    """
    from datasets import load_dataset

    suffix = f"_shard{shard}" if num_shards > 1 else ""
    q_path = output_dir / f"visrag_{name}_queries{suffix}.npy"
    p_path = output_dir / f"visrag_{name}_positives{suffix}.npy"
    if q_path.exists() and p_path.exists():
        logger.info("SKIP visrag_%s%s (already exists)", name, suffix)
        return

    logger.info("Loading visrag_%s from %s", name, data_dir)
    ds = load_dataset("parquet", data_dir=data_dir, split="train")
    n = len(ds)

    shard_size = (n + num_shards - 1) // num_shards
    start_idx = shard * shard_size
    end_idx = min(start_idx + shard_size, n)
    shard_n = end_idx - start_idx
    logger.info("visrag_%s shard %d/%d: samples %d-%d (%d)",
                name, shard, num_shards, start_idx, end_idx, shard_n)

    q_embs = np.zeros((shard_n, EMBED_DIM), dtype=np.float16)
    p_embs = np.zeros((shard_n, EMBED_DIM), dtype=np.float16)

    # Queries are text-only (fast)
    logger.info("Embedding queries (text-only)...")
    for i in range(0, shard_n, text_bs):
        j = min(i + text_bs, shard_n)
        queries = ds[start_idx + i:start_idx + j]["query"]
        convs = [format_input(text=t, instruction="Retrieve a document page relevant to the query.")
                 for t in queries]
        q_embs[i:j] = embed_batch(model, processor, convs, device)
        if i > 0 and (i // text_bs) % 20 == 0:
            logger.info("  queries: %d/%d", i, shard_n)

    # Positives are document images (slow)
    logger.info("Embedding positives (document images)...")
    for i in range(0, shard_n, img_bs):
        j = min(i + img_bs, shard_n)
        images = ds[start_idx + i:start_idx + j]["image"]
        convs = [format_input(image=img) for img in images]
        try:
            p_embs[i:j] = embed_batch(model, processor, convs, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for k in range(j - i):
                try:
                    p_embs[i + k] = embed_batch(model, processor, [format_input(image=images[k])], device)
                except Exception:
                    torch.cuda.empty_cache()
        if i > 0 and (i // img_bs) % 100 == 0:
            logger.info("  positives: %d/%d", i, shard_n)

    np.save(str(q_path), q_embs)
    np.save(str(p_path), p_embs)
    logger.info("DONE visrag_%s shard %d", name, shard)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=TEACHER_MODEL)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--image-dir", default=str(ROOT / "datasets/mmeb_train_images/images"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subsets", default=None, help="Comma-separated MMEB subset list (default: all)")
    parser.add_argument("--split", default="diverse_instruction", help="HF dataset split")
    parser.add_argument("--prefix", default="", help="Prefix for output filenames (e.g. 'orig_')")
    # Text datasets
    parser.add_argument("--text-dataset", default=None,
                        help="Text dataset to process: msmarco, quora, or allnli")
    parser.add_argument("--text-max-samples", type=int, default=None)
    # VisRAG
    parser.add_argument("--visrag", default=None, choices=["synthetic", "indomain"],
                        help="VisRAG dataset to process")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("HF_TOKEN", open(ROOT / ".hf_token_local").read().strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.gpu}"
    model, processor = load_teacher(args.model, device=device)

    if args.text_dataset:
        text_configs = {
            "msmarco": (ROOT / "data/training_data/msmarco/train.jsonl", 500000),
            "quora": (ROOT / "data/training_data/quora/train.jsonl", None),
            "allnli": (ROOT / "data/training_data/allnli/train.jsonl", 300000),
        }
        path, default_max = text_configs[args.text_dataset]
        process_text_dataset(model, processor, args.text_dataset, str(path),
                           output_dir, device, max_samples=args.text_max_samples or default_max)
    elif args.visrag:
        visrag_dirs = {
            "synthetic": str(ROOT / "data/training_data/visrag_synthetic/data"),
            "indomain": str(ROOT / "data/training_data/visrag_indomain/data"),
        }
        name = "syn" if args.visrag == "synthetic" else "ind"
        process_visrag(model, processor, name, visrag_dirs[args.visrag],
                      output_dir, device, shard=args.shard, num_shards=args.num_shards)
    else:
        subsets = args.subsets.split(",") if args.subsets else MMEB_ALL_SUBSETS
        for subset_name in subsets:
            process_subset(model, processor, subset_name, device,
                           args.image_dir, args.batch_size, output_dir,
                           split=args.split, prefix=args.prefix)

    logger.info("All done!")


if __name__ == "__main__":
    main()
