#!/usr/bin/env bash
# Fresh distillation: full fine-tune from base Qwen3.5-0.8B with teacher embeddings.
# No LoRA, no prior stages. Single epoch, direct teacher distillation.
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
: "${OUTPUT_DIR:=${ROOT}/data/outputs/distill-fresh}"
: "${IMAGE_DIR:=${ROOT}/datasets/mmeb_train_images/images}"
: "${TEACHER_EMBED_DIR:=${ROOT}/data/teacher_embeddings}"

: "${BATCH_SIZE:=48}"
: "${GRAD_ACCUM:=1}"
: "${EPOCHS:=1}"
: "${MAX_STEPS:=0}"
: "${LR:=2e-5}"
: "${TEMPERATURE:=0.02}"
: "${MAX_LENGTH:=4096}"
: "${MAX_PIXELS:=1310720}"
: "${TRAINING_STAGE:=3}"
: "${MRL_DIMS:=1024,256,64}"
: "${SAVE_STEPS:=500}"
: "${LOG_INTERVAL:=1}"
: "${NUM_WORKERS:=8}"
: "${PREFETCH_FACTOR:=4}"
: "${SEED:=42}"
: "${GRADIENT_CHECKPOINTING:=true}"

: "${DISTILL_ALPHA_CON:=0.3}"
: "${DISTILL_ALPHA_KL:=0.5}"
: "${DISTILL_ALPHA_MSE:=0.2}"

: "${USE_WANDB:=true}"
: "${WANDB_PROJECT:=embeddings}"
: "${WANDB_ENTITY:=radi-and-people}"
: "${WANDB_RUN_NAME:=distill-fresh-fulltune-8xH100}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
IFS=',' read -ra GPUS <<< "${CUDA_DEVICES}"
NUM_GPUS=${#GPUS[@]}
EFFECTIVE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))

echo "==============================================="
echo " Fresh Distillation: Full Fine-Tune"
echo "==============================================="
echo "  GPUs:              ${CUDA_DEVICES} (${NUM_GPUS})"
echo "  Model:             ${MODEL_PATH}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Teacher embeds:    ${TEACHER_EMBED_DIR}"
echo "  Batch size:        ${BATCH_SIZE}/GPU x ${NUM_GPUS} GPUs = ${EFFECTIVE} effective"
echo "  LR:                ${LR}  |  Epochs: ${EPOCHS}"
echo "  Loss weights:      con=${DISTILL_ALPHA_CON} kl=${DISTILL_ALPHA_KL} mse=${DISTILL_ALPHA_MSE}"
echo "  MRL:               ${MRL_DIMS}"
echo "  Full fine-tune:    YES (no LoRA)"
echo "  Gradient ckpt:     ${GRADIENT_CHECKPOINTING}"
echo "==============================================="

ARGS="--model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --epochs ${EPOCHS} --lr ${LR} --temperature ${TEMPERATURE} \
    --max_length ${MAX_LENGTH} --max_pixels ${MAX_PIXELS} \
    --training_stage ${TRAINING_STAGE} --use_mrl --mrl_dims ${MRL_DIMS} \
    --no_lora \
    --num_workers ${NUM_WORKERS} --prefetch_factor ${PREFETCH_FACTOR} --seed ${SEED} \
    --save_steps ${SAVE_STEPS} --log_interval ${LOG_INTERVAL} \
    --teacher_embed_dir ${TEACHER_EMBED_DIR} \
    --distill_alpha_con ${DISTILL_ALPHA_CON} \
    --distill_alpha_kl ${DISTILL_ALPHA_KL} \
    --distill_alpha_mse ${DISTILL_ALPHA_MSE}"

[ -n "${IMAGE_DIR:-}" ] && ARGS="${ARGS} --image_dir ${IMAGE_DIR}"
[ "${GRADIENT_CHECKPOINTING}" = "true" ] && ARGS="${ARGS} --gradient_checkpointing"
[ -n "${RESUME_FROM:-}" ] && ARGS="${ARGS} --resume_from ${RESUME_FROM}"
if [ -n "${MAX_STEPS:-}" ] && [ "${MAX_STEPS}" -gt 0 ]; then
    ARGS="${ARGS} --max_steps ${MAX_STEPS}"
fi
if [ "${USE_WANDB}" = "true" ]; then
    ARGS="${ARGS} --use_wandb --wandb_project ${WANDB_PROJECT}"
    [ -n "${WANDB_ENTITY:-}" ] && ARGS="${ARGS} --wandb_entity ${WANDB_ENTITY}"
    [ -n "${WANDB_RUN_NAME:-}" ] && ARGS="${ARGS} --wandb_run_name ${WANDB_RUN_NAME}"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    ${PY} -m accelerate.commands.accelerate_cli launch \
        --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
        "${TRAIN}" ${ARGS}
else
    ${PY} "${TRAIN}" ${ARGS}
fi
