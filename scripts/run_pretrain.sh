#!/usr/bin/env bash
# Contrastive pretraining for Qwen3.5-0.8B multimodal embedding model.
#
# Simple DDP -- each GPU encodes batch_size items, embeddings are gathered
# across GPUs for contrastive loss.  Effective batch = batch_size * num_gpus.
#
# For 8 GPUs:  128/GPU -> 1024 effective (target)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PY="${ROOT}/.venv/bin/python"
TRAIN="${ROOT}/train/train.py"

if [[ -z "${HF_TOKEN:-}" && -f "${ROOT}/.hf_token_local" ]]; then
    export HF_TOKEN="$(tr -d '
' < "${ROOT}/.hf_token_local")"
fi

: "${CUDA_DEVICES:=0,1}"
: "${MODEL_PATH:=${ROOT}/models/checkpoints/Qwen3.5-0.8B}"
: "${OUTPUT_DIR:=/data/outputs/qwen35-0.8b-10M-pretrain}"
: "${IMAGE_DIR:=${ROOT}/datasets/mmeb_train_images/images}"
: "${DATA_DIR:=/data/training_data}"

: "${BATCH_SIZE:=64}"
: "${GRAD_ACCUM:=1}"
: "${EPOCHS:=1}"
: "${LR:=2e-5}"
: "${TEMPERATURE:=0.02}"
: "${MAX_LENGTH:=512}"
: "${MAX_PIXELS:=1310720}"
: "${TRAINING_STAGE:=1}"
: "${MRL_DIMS:=1024,256,64}"
: "${LORA_RANK:=32}"
: "${LORA_ALPHA:=32}"
: "${SAVE_STEPS:=500}"
: "${LOG_INTERVAL:=1}"
: "${NUM_WORKERS:=4}"
: "${SEED:=42}"
: "${GRADIENT_CHECKPOINTING:=true}"

: "${USE_WANDB:=true}"
: "${WANDB_PROJECT:=embeddings}"
: "${WANDB_ENTITY:=radi-and-people}"
: "${WANDB_RUN_NAME:=qwen35-0.8b-10M-pretrain-bs64}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IFS=',' read -ra GPUS <<< "${CUDA_DEVICES}"
NUM_GPUS=${#GPUS[@]}
EFFECTIVE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))

echo "==============================================="
echo " Qwen3.5-0.8B Contrastive Pretraining"
echo "==============================================="
echo "  GPUs:              ${CUDA_DEVICES} (${NUM_GPUS})"
echo "  Model:             ${MODEL_PATH}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Batch size:        ${BATCH_SIZE}/GPU x ${NUM_GPUS} GPUs x ${GRAD_ACCUM} accum = ${EFFECTIVE} effective"
echo "  LR:                ${LR}  |  Epochs: ${EPOCHS}"
echo "  Max pixels:        ${MAX_PIXELS}  |  MRL: ${MRL_DIMS}"
echo "  LoRA:              rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Seed:              ${SEED}"
echo "  WandB:             ${USE_WANDB} (${WANDB_PROJECT})"
echo "==============================================="

ARGS="--model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --epochs ${EPOCHS} --lr ${LR} --temperature ${TEMPERATURE} \
    --max_length ${MAX_LENGTH} --max_pixels ${MAX_PIXELS} \
    --training_stage ${TRAINING_STAGE} --use_mrl --mrl_dims ${MRL_DIMS} \
    --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
    --num_workers ${NUM_WORKERS} --seed ${SEED} \
    --save_steps ${SAVE_STEPS} --log_interval ${LOG_INTERVAL}"

[ -n "${IMAGE_DIR:-}" ]    && ARGS="${ARGS} --image_dir ${IMAGE_DIR}"
[ -n "${DATA_DIR:-}" ]     && ARGS="${ARGS} --data_dir ${DATA_DIR}"
[ -n "${SUBSETS:-}" ]      && ARGS="${ARGS} --subsets ${SUBSETS}"
[ -n "${TASK_TYPES:-}" ]   && ARGS="${ARGS} --task_types ${TASK_TYPES}"
[ -n "${MAX_SAMPLES:-}" ]  && ARGS="${ARGS} --max_samples_per_subset ${MAX_SAMPLES}"
[ "${GRADIENT_CHECKPOINTING}" = "true" ] && ARGS="${ARGS} --gradient_checkpointing"
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
