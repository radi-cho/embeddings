#!/usr/bin/env bash
# Run classification-label mining, then standard retrieval/VQA/text mining (GPU 0),
# then Stage-1 instruction-aware pretraining on all GPUs.
# Safe to detach: nohup bash scripts/run_instruction_mining_then_train.sh </dev/null &>/tmp/instr_pipeline.log &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
MINER="${ROOT}/scripts/mine_hard_negatives.py"

: "${MINING_GPU:=0}"
: "${MODEL_FOR_MINING:=}"

export CUDA_VISIBLE_DEVICES="${MINING_GPU}"
MIN_ARGS=()
[[ -n "${MODEL_FOR_MINING}" ]] && MIN_ARGS+=(--model "${MODEL_FOR_MINING}")

echo "[pipeline] Classification label mining on GPU ${MINING_GPU} ..."
"${PY}" "${MINER}" --mine-classification-labels "${MIN_ARGS[@]}"

echo "[pipeline] Retrieval / VQA / text mining (multimodal, images loaded) ..."
"${PY}" "${MINER}" "${MIN_ARGS[@]}"

unset CUDA_VISIBLE_DEVICES
echo "[pipeline] Starting Stage-1 training (see scripts/run_pretrain.sh for env) ..."
exec bash "${ROOT}/scripts/run_pretrain.sh"
