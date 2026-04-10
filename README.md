# cho-embedding-0.8b

## Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Match torchaudio to your CUDA version:
pip install torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install wandb  # optional, for experiment tracking
```

Verify Qwen3.5 support (requires transformers from source):
```bash
python -c "from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel; print('OK')"
```

## Models

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/checkpoints/Qwen3.5-0.8B
```

The base model is Qwen3.5-0.8B: a native multimodal (vision+language) model with hidden dim 1024, 24 layers (hybrid DeltaNet + GQA), and 262k context. It uses `Qwen3VLProcessor` and causal attention with last-token pooling for embeddings.

For evaluation baselines, also download `Qwen/Qwen3-VL-Embedding-2B`.

## HuggingFace Token

Some datasets (MegaPairs) and models are gated. Create `.hf_token_local` (gitignored):
```bash
echo "hf_YOUR_TOKEN" > .hf_token_local
```

All scripts auto-load this file.

## Training Data

Download ~9.2M examples across 9 sources (~72 GB on disk):
```bash
python scripts/download_training_data.py --output_dir /data/training_data
```

This downloads text triplets (MS MARCO, AllNLI, GooAQ, Quora), STS-B scores, MegaPairs annotations (5M, images not included due to size), ColPali document images (118k), and video manifests (LLaVA-Hound, MSR-VTT). Each source gets a `manifest.json` with row counts.

MMEB-train (1.07M multimodal examples from `TIGER-Lab/MMEB-train`) is loaded from HuggingFace at runtime. Its images should be pre-downloaded:
```bash
python eval/run_mmeb.py --download_images --cache_dir datasets/mmeb_cache
```

## Training

```bash
# Default: 2 GPUs, contrastive batch 1024, lr=2e-5, 1 epoch
nohup bash scripts/run_pretrain.sh > logs/pretrain.log 2>&1 & disown

# Override any parameter via environment:
CUDA_DEVICES=0,1,2,3,4,5,6,7 CONTRASTIVE_BATCH=2048 bash scripts/run_pretrain.sh
```

The training implements Stage 1 (contrastive pretraining) from the Qwen3-VL-Embedding paper (`docs/qwen-3-vl-emb-paper/colm2024_conference.tex`):
- **Loss**: MRL-wrapped masked InfoNCE (Eq. 1) with all 5 Z_i terms (positive, hard negatives, q-q, d-d, q-d in-batch) and false-negative masking (margin 0.1). STS-B data uses CoSent loss (Eq. 2). Batches with mixed task types are split and routed to the correct loss.
- **Distributed**: DDP with cross-GPU embedding all_gather before loss computation. GradCache auto-enables when per-device batch exceeds micro-batch size, decoupling GPU memory from contrastive batch size.
- **Model**: LoRA (rank 32, alpha 32) on all linear projections. Gradient checkpointing enabled.
- **Image budget**: 1,280 tokens (~1.31M pixels), matching paper Section 5.2.

## Evaluation

### MMEB (multimodal: image, video, document)

```bash
# Image tasks (36 tasks, 4 categories)
python eval/run_mmeb.py --model_path <path> --full --output_dir results/mmeb/<name>

# Video tasks (18 tasks)
python eval/run_mmeb_video.py --model_path <path> --full

# Visual document tasks (24 tasks)
python eval/run_mmeb_visdoc.py --model_path <path> --full
```

All MMEB scripts auto-detect Qwen3-VL vs Qwen3.5 models. Image budget for evaluation is 1,800 tokens (per paper Section 6.1). Results are saved as JSON with per-task hit@1 scores.

### MMTEB (text-only)

Two evaluation paths:

1. **Direct** via `eval/run_mmteb.py` — wraps models in an MTEB-compatible encoder using per-task instructions from `eval/task_prompts.json`:
   ```bash
   python eval/run_mmteb.py --model_path <path> --sts --output_dir results/mmteb/<name>
   ```

2. **CSV-based** via the Qwen3-Embedding submodule (`Qwen3-Embedding/evaluation/run_csv_repro.py`) — reproduces exact Qwen paper results using their task CSV:
   ```bash
   python Qwen3-Embedding/evaluation/run_csv_repro.py \
     --tasks_csv Qwen3-Embedding/tmp_sts_only.csv \
     --model_path <path> --backend mmteb_chat
   ```

### Scheduled benchmarks

```bash
bash scripts/schedule_benchmarks.sh
```

Runs MTEB on GPU 0 and MMEB on GPU 1 in parallel, detached for SSH disconnect. Logs to `scheduler_logs/`.

## Reference Material

- `docs/qwen-3-vl-emb-paper/colm2024_conference.tex` — the paper. Section 5.1 for loss formulations, 5.1.1 for MRL, 5.2 for training setup, Section 4 for multi-stage pipeline.
- `Qwen3-VL-Embedding/` — reference 2B model implementation and evaluation scripts (git submodule).
- `Qwen3-Embedding/` — MTEB evaluation scripts and task CSVs (git submodule).
- `docs/qwen-3-vl-emb.md`, `docs/qwen-3.5-0.8b.md` — HuggingFace model cards.
