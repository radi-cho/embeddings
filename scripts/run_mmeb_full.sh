#!/usr/bin/env bash
# Stage 1 pretraining on full MMEB dataset (all 20 subsets, ~1.07M samples).
# Balanced across classification, VQA, retrieval, and grounding.
# 1 epoch, low LR, cross-device negative gathering via GatherWithGrad.
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
: "${OUTPUT_DIR:=${ROOT}/data/outputs/stage1-mmeb-full}"
: "${IMAGE_DIR:=${ROOT}/datasets/mmeb_train_images/images}"

: "${BATCH_SIZE:=32}"
: "${GRAD_ACCUM:=1}"
: "${EPOCHS:=1}"
: "${MAX_STEPS:=0}"
: "${LR:=1e-5}"
: "${TEMPERATURE:=0.02}"
: "${MAX_LENGTH:=4096}"
: "${MAX_PIXELS:=1310720}"
: "${TRAINING_STAGE:=1}"
: "${MRL_DIMS:=1024,256,64}"
: "${LORA_RANK:=32}"
: "${LORA_ALPHA:=64}"
: "${SAVE_STEPS:=500}"
: "${LOG_INTERVAL:=1}"
: "${NUM_WORKERS:=8}"
: "${PREFETCH_FACTOR:=4}"
: "${SEED:=42}"
: "${GRADIENT_CHECKPOINTING:=true}"
: "${USE_OPTIMIZED_MIX:=true}"
: "${DATA_MIX_VERSION:=mmeb_full}"

: "${USE_WANDB:=true}"
: "${WANDB_PROJECT:=embeddings}"
: "${WANDB_ENTITY:=radi-and-people}"
: "${WANDB_RUN_NAME:=stage1-mmeb-full-1e5-8xH100}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
IFS=',' read -ra GPUS <<< "${CUDA_DEVICES}"
NUM_GPUS=${#GPUS[@]}
EFFECTIVE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))

echo "==============================================="
echo " Stage 1: MMEB-Full Pretraining"
echo "==============================================="
echo "  GPUs:              ${CUDA_DEVICES} (${NUM_GPUS})"
echo "  Model:             ${MODEL_PATH}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Batch size:        ${BATCH_SIZE}/GPU x ${NUM_GPUS} GPUs x ${GRAD_ACCUM} accum = ${EFFECTIVE} effective"
echo "  LR:                ${LR}  |  Epochs: ${EPOCHS}"
echo "  Max pixels:        ${MAX_PIXELS}  |  MRL: ${MRL_DIMS}"
echo "  LoRA:              rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Data mix:          ${DATA_MIX_VERSION} (~1.07M MMEB training samples)"
echo "  Seed:              ${SEED}"
echo "  WandB:             ${USE_WANDB} (${WANDB_PROJECT})"
echo "  Gradient ckpt:     ${GRADIENT_CHECKPOINTING}"
echo "==============================================="

ARGS="--model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --epochs ${EPOCHS} --lr ${LR} --temperature ${TEMPERATURE} \
    --max_length ${MAX_LENGTH} --max_pixels ${MAX_PIXELS} \
    --training_stage ${TRAINING_STAGE} --use_mrl --mrl_dims ${MRL_DIMS} \
    --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
    --num_workers ${NUM_WORKERS} --prefetch_factor ${PREFETCH_FACTOR} --seed ${SEED} \
    --save_steps ${SAVE_STEPS} --log_interval ${LOG_INTERVAL}"

[ -n "${IMAGE_DIR:-}" ]    && ARGS="${ARGS} --image_dir ${IMAGE_DIR}"
[ "${USE_OPTIMIZED_MIX}" = "true" ] && ARGS="${ARGS} --use_optimized_mix --data_mix_version ${DATA_MIX_VERSION}"
[ "${GRADIENT_CHECKPOINTING}" = "true" ] && ARGS="${ARGS} --gradient_checkpointing"
[ -n "${RESUME_FROM:-}" ] && ARGS="${ARGS} --resume_from ${RESUME_FROM}"
if [ -n "${MAX_STEPS:-}" ] && [ "${MAX_STEPS}" -gt 0 ]; then
    ARGS="${ARGS} --max_steps ${MAX_STEPS}"
fi
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
