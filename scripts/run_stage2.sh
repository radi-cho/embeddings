#!/usr/bin/env bash
# Stage 2 training with K=15 mined hard negatives (Qwen3-VL paper methodology).
#
# Prerequisites:
#   - Stage 1 LoRA checkpoint at ${ROOT}/data/outputs/stage1-lr1e4-a64/final
#   - Mined JSONL files in ${ROOT}/data/training_data_mined/ (produced by scripts/mine_hard_negatives.py)
#
# Data routing (per task_type):
#   sts            -> CoSENT loss (STS-B, symmetric)
#   classification -> classification_infonce_loss (MMEB neg_cand wrong-class labels)
#   other          -> masked_infonce_loss stage=2 (no q-q/d-d cross terms) with mined HNs
#
# Launch (recommended, survives SSH disconnect):
#   nohup setsid bash scripts/run_stage2.sh > results/stage2-k15.log 2>&1 < /dev/null &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PY="${ROOT}/.venv/bin/python"
TRAIN="${ROOT}/train/train.py"

if [[ -z "${HF_TOKEN:-}" && -f "${ROOT}/.hf_token_local" ]]; then
    export HF_TOKEN="$(tr -d '
' < "${ROOT}/.hf_token_local")"
fi

# ---- Paths ----
: "${CUDA_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MODEL_PATH:=${ROOT}/data/outputs/stage1-mmeb-full/final}"
: "${OUTPUT_DIR:=${ROOT}/data/outputs/stage2-mmeb-full-hn}"
: "${IMAGE_DIR:=${ROOT}/datasets/mmeb_train_images/images}"
: "${DATA_DIR:=${ROOT}/data/training_data}"
: "${MINED_DIR:=${ROOT}/data/training_data_mined}"

# ---- Optimizer / schedule ----
: "${BATCH_SIZE:=32}"          # Halved from Stage 1 (K=15 HNs triple forward cost)
: "${GRAD_ACCUM:=1}"
: "${EPOCHS:=1}"
: "${LR:=1e-5}"                # Same as Stage 1 (fine-tuning from checkpoint)
: "${TEMPERATURE:=0.02}"
: "${MAX_LENGTH:=512}"
: "${MAX_PIXELS:=1310720}"
: "${TRAINING_STAGE:=2}"       # Drops q-q and d-d cross terms in InfoNCE
: "${MRL_DIMS:=1024,256,64}"
: "${LORA_RANK:=32}"
: "${LORA_ALPHA:=64}"
: "${SAVE_STEPS:=500}"
: "${LOG_INTERVAL:=1}"
: "${NUM_WORKERS:=4}"
: "${PREFETCH_FACTOR:=2}"
: "${SEED:=42}"
: "${GRADIENT_CHECKPOINTING:=true}"
: "${COMPILE:=false}"

# ---- WandB ----
: "${USE_WANDB:=true}"
: "${WANDB_PROJECT:=embeddings}"
: "${WANDB_ENTITY:=radi-and-people}"
: "${WANDB_RUN_NAME:=stage2-mmeb-full-hn-8xH100}"

# ---- Safety / perf env ----
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

IFS=',' read -ra GPUS <<< "${CUDA_DEVICES}"
NUM_GPUS=${#GPUS[@]}
EFFECTIVE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))

echo "==============================================="
echo " Qwen3.5-0.8B Stage 2 — Mined Hard Negatives (K=15)"
echo "==============================================="
echo "  GPUs:              ${CUDA_DEVICES} (${NUM_GPUS})"
echo "  Model:             ${MODEL_PATH}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Mined dir:         ${MINED_DIR}"
echo "  Data dir:          ${DATA_DIR}"
echo "  Batch size:        ${BATCH_SIZE}/GPU x ${NUM_GPUS} GPUs x ${GRAD_ACCUM} accum = ${EFFECTIVE} effective"
echo "  LR:                ${LR}  |  Epochs: ${EPOCHS}"
echo "  Stage:             ${TRAINING_STAGE} (InfoNCE drops q-q, d-d terms)"
echo "  Max pixels:        ${MAX_PIXELS}  |  MRL: ${MRL_DIMS}"
echo "  LoRA:              rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Seed:              ${SEED}"
echo "  WandB:             ${USE_WANDB} (${WANDB_PROJECT})"
echo "==============================================="

# ---- Pre-flight checks ----
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${MODEL_PATH}"
    exit 1
fi
if [ ! -d "${MINED_DIR}" ] || [ -z "$(ls -A "${MINED_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: ${MINED_DIR} is empty. Run scripts/mine_hard_negatives.py first."
    exit 1
fi
echo "Mined files:"
ls -lh "${MINED_DIR}" | tail -n +2 | awk '{printf "  %-35s %s\n", $NF, $5}'
echo "==============================================="

ARGS="--model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --epochs ${EPOCHS} --lr ${LR} --temperature ${TEMPERATURE} \
    --max_length ${MAX_LENGTH} --max_pixels ${MAX_PIXELS} \
    --training_stage ${TRAINING_STAGE} --use_mrl --mrl_dims ${MRL_DIMS} \
    --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
    --num_workers ${NUM_WORKERS} --prefetch_factor ${PREFETCH_FACTOR} --seed ${SEED} \
    --save_steps ${SAVE_STEPS} --log_interval ${LOG_INTERVAL} \
    --mined_dir ${MINED_DIR}"

[ -n "${IMAGE_DIR:-}" ]    && ARGS="${ARGS} --image_dir ${IMAGE_DIR}"
[ -n "${DATA_DIR:-}" ]     && ARGS="${ARGS} --data_dir ${DATA_DIR}"
[ -n "${MAX_SAMPLES:-}" ]  && ARGS="${ARGS} --max_samples_per_subset ${MAX_SAMPLES}"
[ "${GRADIENT_CHECKPOINTING}" = "true" ] && ARGS="${ARGS} --gradient_checkpointing"
[ "${COMPILE}" = "true" ]                && ARGS="${ARGS} --compile"
[ -n "${RESUME_FROM:-}" ]  && ARGS="${ARGS} --resume_from ${RESUME_FROM}"

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
