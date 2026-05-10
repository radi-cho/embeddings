#!/usr/bin/env bash
# LLaVE-style training: MMEB-only (662K), hardness-weighted loss, 1 epoch.
# Matches LLaVE-0.5B setup as closely as possible on our Qwen3.5-0.8B base.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PY="${ROOT}/.venv/bin/python"
TRAIN="${ROOT}/train/train.py"

if [[ -z "${HF_TOKEN:-}" && -f "${ROOT}/.hf_token_local" ]]; then
    export HF_TOKEN="$(tr -d '
' < "${ROOT}/.hf_token_local")"
fi

: "${CUDA_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MODEL_PATH:=${ROOT}/models/checkpoints/Qwen3.5-0.8B}"
: "${OUTPUT_DIR:=${ROOT}/data/outputs/llave-style-mmeb-only}"
: "${IMAGE_DIR:=${ROOT}/datasets/mmeb_train_images/images}"

# Standard training params (same as run_pretrain.sh) — only data source + loss differ.
: "${BATCH_SIZE:=64}"
: "${GRAD_ACCUM:=1}"
: "${EPOCHS:=1}"
: "${LR:=1e-4}"
: "${TEMPERATURE:=0.02}"
: "${HARDNESS_ALPHA:=9.0}"
: "${MAX_LENGTH:=512}"
: "${MAX_PIXELS:=1843200}"
: "${TRAINING_STAGE:=1}"
: "${LORA_RANK:=32}"
: "${LORA_ALPHA:=64}"
: "${SAVE_STEPS:=500}"
: "${LOG_INTERVAL:=1}"
: "${NUM_WORKERS:=8}"
: "${PREFETCH_FACTOR:=4}"
: "${SEED:=42}"
: "${GRADIENT_CHECKPOINTING:=true}"
: "${COMPILE:=false}"

: "${USE_WANDB:=true}"
: "${WANDB_PROJECT:=embeddings}"
: "${WANDB_ENTITY:=radi-and-people}"
: "${WANDB_RUN_NAME:=llave-hw9-mmeb662k-bs512-lr1e4-8xH100}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IFS=',' read -ra GPUS <<< "${CUDA_DEVICES}"
NUM_GPUS=${#GPUS[@]}
EFFECTIVE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))

echo "==============================================="
echo " LLaVE-style MMEB-only Training"
echo "==============================================="
echo "  GPUs:              ${CUDA_DEVICES} (${NUM_GPUS})"
echo "  Model:             ${MODEL_PATH}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Batch size:        ${BATCH_SIZE}/GPU x ${NUM_GPUS} GPUs x ${GRAD_ACCUM} accum = ${EFFECTIVE} effective"
echo "  LR:                ${LR}  |  Epochs: ${EPOCHS}"
echo "  Hardness alpha:    ${HARDNESS_ALPHA}"
echo "  Max pixels:        ${MAX_PIXELS} (matches eval)"
echo "  LoRA:              rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  MRL:               enabled (1024,256,64)"
echo "  Gradient ckpt:     ${GRADIENT_CHECKPOINTING}"
echo "==============================================="

# MMEB-only: use legacy MMEB builder (no --data_dir, no --use_optimized_mix).
# Cap each subset at 50K to match LLaVE's balanced approach.
ARGS="--model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --epochs ${EPOCHS} --lr ${LR} --temperature ${TEMPERATURE} \
    --hardness_alpha ${HARDNESS_ALPHA} \
    --max_length ${MAX_LENGTH} --max_pixels ${MAX_PIXELS} \
    --training_stage ${TRAINING_STAGE} \
    --use_mrl --mrl_dims 1024,256,64 \
    --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
    --num_workers ${NUM_WORKERS} --prefetch_factor ${PREFETCH_FACTOR} --seed ${SEED} \
    --save_steps ${SAVE_STEPS} --log_interval ${LOG_INTERVAL} \
    --warmup_ratio 0.1 \
    --max_samples_per_subset 50000"

[ -n "${IMAGE_DIR:-}" ] && ARGS="${ARGS} --image_dir ${IMAGE_DIR}"
[ "${GRADIENT_CHECKPOINTING}" = "true" ] && ARGS="${ARGS} --gradient_checkpointing"
[ "${COMPILE}" = "true" ] && ARGS="${ARGS} --compile"
if [ "${USE_WANDB}" = "true" ]; then
    ARGS="${ARGS} --use_wandb --wandb_project ${WANDB_PROJECT}"
    [ -n "${WANDB_ENTITY:-}" ]   && ARGS="${ARGS} --wandb_entity ${WANDB_ENTITY}"
    [ -n "${WANDB_RUN_NAME:-}" ] && ARGS="${ARGS} --wandb_run_name ${WANDB_RUN_NAME}"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    ${PY} -m accelerate.commands.accelerate_cli launch \
        --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
        "${TRAIN}" ${ARGS}
else
    ${PY} "${TRAIN}" ${ARGS}
fi
