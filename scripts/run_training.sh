#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${PROJECT_ROOT}/src/train/train.py"

CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-0.8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen35-emb-mmeb}"
IMAGE_DIR="${IMAGE_DIR:-}"
PRETOKENIZED_DIR="${PRETOKENIZED_DIR:-}"
SUBSETS="${SUBSETS:-}"
TASK_TYPES="${TASK_TYPES:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NO_GRAD_CACHE="${NO_GRAD_CACHE:-false}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
TEMPERATURE="${TEMPERATURE:-0.02}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_PIXELS="${MAX_PIXELS:-401408}"
TRAINING_STAGE="${TRAINING_STAGE:-1}"
MRL_DIMS="${MRL_DIMS:-1024,256,64}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SEED="${SEED:-42}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"

USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-embeddings}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:-}"
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_DEVICES}"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=============================================="
echo " Qwen3.5-0.8B Embedding Training (GradCache)"
echo "=============================================="
echo "  GPUs:              ${CUDA_DEVICES} (${NUM_GPUS} devices)"
echo "  Model:             ${MODEL_PATH}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Pretokenized dir:  ${PRETOKENIZED_DIR:-<live MMEB>}"
echo "  Micro-batch/GPU:   ${BATCH_SIZE}"
echo "  Effective batch:   ${EFFECTIVE_BATCH} (global)"
echo "  Batch per rank:    $((EFFECTIVE_BATCH / NUM_GPUS))"
echo "  Epochs:            ${EPOCHS}"
echo "  LR:                ${LR}"
echo "  Temperature:       ${TEMPERATURE}"
echo "  Max pixels:        ${MAX_PIXELS}"
echo "  MRL dims:          ${MRL_DIMS}"
echo "  LoRA:              rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Workers:           ${NUM_WORKERS}"
echo "  WandB:             ${USE_WANDB}"
echo "=============================================="

EXTRA_ARGS=""
[ -n "${SUBSETS}" ] && EXTRA_ARGS="${EXTRA_ARGS} --subsets ${SUBSETS}"
[ -n "${TASK_TYPES}" ] && EXTRA_ARGS="${EXTRA_ARGS} --task_types ${TASK_TYPES}"
[ -n "${IMAGE_DIR}" ] && EXTRA_ARGS="${EXTRA_ARGS} --image_dir ${IMAGE_DIR}"
[ -n "${MAX_SAMPLES}" ] && EXTRA_ARGS="${EXTRA_ARGS} --max_samples_per_subset ${MAX_SAMPLES}"
[ -n "${PRETOKENIZED_DIR}" ] && EXTRA_ARGS="${EXTRA_ARGS} --pretokenized_dir ${PRETOKENIZED_DIR}"
[ "${NO_GRAD_CACHE}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --no_grad_cache"
[ "${GRADIENT_CHECKPOINTING}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --gradient_checkpointing"
if [ "${USE_WANDB}" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_wandb --wandb_project ${WANDB_PROJECT}"
    [ -n "${WANDB_ENTITY}" ] && EXTRA_ARGS="${EXTRA_ARGS} --wandb_entity ${WANDB_ENTITY}"
    [ -n "${WANDB_RUN_NAME}" ] && EXTRA_ARGS="${EXTRA_ARGS} --wandb_run_name ${WANDB_RUN_NAME}"
fi

ACCELERATE="${PYTHON} -m accelerate.commands.accelerate_cli"

COMMON_ARGS="--model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} --effective_batch_size ${EFFECTIVE_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --epochs ${EPOCHS} --lr ${LR} --temperature ${TEMPERATURE} \
    --max_length ${MAX_LENGTH} --max_pixels ${MAX_PIXELS} \
    --training_stage ${TRAINING_STAGE} --use_mrl --mrl_dims ${MRL_DIMS} \
    --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
    --num_workers ${NUM_WORKERS} \
    --save_steps ${SAVE_STEPS} --log_interval ${LOG_INTERVAL} \
    --seed ${SEED} ${EXTRA_ARGS}"

if [ "${NUM_GPUS}" -gt 1 ]; then
    ${ACCELERATE} launch --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
        "${TRAIN_SCRIPT}" ${COMMON_ARGS}
else
    ${PYTHON} "${TRAIN_SCRIPT}" ${COMMON_ARGS}
fi
