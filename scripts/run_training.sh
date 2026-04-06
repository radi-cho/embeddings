#!/usr/bin/env bash
#
# Train Qwen3.5-0.8B embedding model with LoRA on MMEB-train.
#
# Hyperparameters follow the Qwen3-VL-Embedding paper:
#   - LoRA rank=32, alpha=32, targets: q/k/v/up/down/gate_proj
#   - Temperature 0.02, InfoNCE Stage 1 with false-negative masking
#   - MRL dims: 1024,768,512,256,128,64
#   - Cosine LR schedule with 10% warmup
#
# Usage:
#   bash scripts/run_training.sh              # defaults: GPUs 0,1, all subsets
#   CUDA_DEVICES=0 bash scripts/run_training.sh  # single GPU
#   CUDA_DEVICES=0,1,2,3 bash scripts/run_training.sh  # 4 GPUs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="${PROJECT_ROOT}/.venv"
PYTHON="${VENV}/bin/python"
TRAIN_SCRIPT="${PROJECT_ROOT}/src/train/train.py"

# ---------- Configurable via environment ----------
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-0.8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen35-emb-mmeb}"
IMAGE_DIR="${IMAGE_DIR:-}"
CACHE_DIR="${CACHE_DIR:-}"
SUBSETS="${SUBSETS:-}"
TASK_TYPES="${TASK_TYPES:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# ---------- Hyperparameters ----------
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
TEMPERATURE="${TEMPERATURE:-0.02}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_PIXELS="${MAX_PIXELS:-401408}"
TRAINING_STAGE="${TRAINING_STAGE:-1}"
MRL_DIMS="${MRL_DIMS:-1024,768,512,256,128,64}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SEED="${SEED:-42}"

# Weights & Biases (disabled by default)
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-embeddings}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# ---------- Resolve GPU count ----------
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_DEVICES}"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=============================================="
echo " Qwen3.5-0.8B Embedding Training"
echo "=============================================="
echo "  GPUs:           ${CUDA_DEVICES} (${NUM_GPUS} devices)"
echo "  Model:          ${MODEL_PATH}"
echo "  Output:         ${OUTPUT_DIR}"
echo "  Batch/GPU:      ${BATCH_SIZE}"
echo "  Grad accum:     ${GRAD_ACCUM}"
echo "  Effective batch: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  Epochs:         ${EPOCHS}"
echo "  LR:             ${LR}"
echo "  Temperature:    ${TEMPERATURE}"
echo "  Max pixels:     ${MAX_PIXELS}"
echo "  MRL dims:       ${MRL_DIMS}"
echo "  LoRA:           rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Stage:          ${TRAINING_STAGE}"
echo "  Max length:     ${MAX_LENGTH}"
echo "  WandB:          ${USE_WANDB}"
echo "=============================================="

# Build CLI args
EXTRA_ARGS=""
if [ -n "${SUBSETS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --subsets ${SUBSETS}"
fi
if [ -n "${TASK_TYPES}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --task_types ${TASK_TYPES}"
fi
if [ -n "${IMAGE_DIR}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --image_dir ${IMAGE_DIR}"
fi
if [ -n "${CACHE_DIR}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --cache_dir ${CACHE_DIR}"
fi
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max_samples_per_subset ${MAX_SAMPLES}"
fi
if [ -n "${NUM_WORKERS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --num_workers ${NUM_WORKERS}"
fi
        EXTRA_ARGS="${EXTRA_ARGS} --wandb_entity ${WANDB_ENTITY}"
    fi
    if [ -n "${WANDB_RUN_NAME}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --wandb_run_name ${WANDB_RUN_NAME}"
    fi
fi

# Increase NCCL timeout for vision batches with variable sequence lengths
export NCCL_TIMEOUT=1800000

ACCELERATE="${PYTHON} -m accelerate.commands.accelerate_cli"

if [ "${NUM_GPUS}" -gt 1 ]; then
    ${ACCELERATE} launch \
        --num_processes "${NUM_GPUS}" \
        --mixed_precision bf16 \
        "${TRAIN_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --temperature "${TEMPERATURE}" \
        --max_length "${MAX_LENGTH}" \
        --max_pixels "${MAX_PIXELS}" \
        --training_stage "${TRAINING_STAGE}" \
        --use_mrl \
        --mrl_dims "${MRL_DIMS}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --gradient_checkpointing \
        --save_steps "${SAVE_STEPS}" \
        --log_interval "${LOG_INTERVAL}" \
        --seed "${SEED}" \
        ${EXTRA_ARGS}
else
    ${PYTHON} "${TRAIN_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --temperature "${TEMPERATURE}" \
        --max_length "${MAX_LENGTH}" \
        --max_pixels "${MAX_PIXELS}" \
        --training_stage "${TRAINING_STAGE}" \
        --use_mrl \
        --mrl_dims "${MRL_DIMS}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --gradient_checkpointing \
        --save_steps "${SAVE_STEPS}" \
        --log_interval "${LOG_INTERVAL}" \
        --seed "${SEED}" \
        ${EXTRA_ARGS}
fi
