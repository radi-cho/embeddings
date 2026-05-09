#!/usr/bin/env bash
# Merge LoRA checkpoints and run MMEB eval for instruction-aware Stage-1 output.
# Usage: CKPT_DIR=... CHECKPOINTS=(1000 2000 ... 10000) bash scripts/eval_instruction_aware_stage1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BASE_MODEL="${ROOT}/models/checkpoints/Qwen3.5-0.8B"
CKPT_DIR="${CKPT_DIR:-${ROOT}/data/outputs/stage1-instruction-aware-10k}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/stage1-instruction-aware-10k}"

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    CHECKPOINTS=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
fi
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p "${RESULTS_BASE}"

for i in "${!CHECKPOINTS[@]}"; do
    STEP="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
    ADAPTER="${CKPT_DIR}/checkpoint-${STEP}"
    MERGED="${CKPT_DIR}/merged-${STEP}"
    OUT_DIR="${RESULTS_BASE}/step-${STEP}"
    LOG="${RESULTS_BASE}/step-${STEP}.log"

    if [[ ! -d "${ADAPTER}" ]]; then
        echo "SKIP: ${ADAPTER} not found"
        continue
    fi

    echo "GPU ${GPU}: step ${STEP} → ${OUT_DIR}"

    nohup bash -c "
        set -e
        export CUDA_VISIBLE_DEVICES=${GPU}
        if [[ ! -f '${MERGED}/config.json' ]]; then
            echo '[step ${STEP}] Merging adapter...'
            ${PY} -c \"
import torch
from peft import PeftModel
import sys; sys.path.insert(0, '${ROOT}')
from models.qwen35_embedding import Qwen35Embedder
emb = Qwen35Embedder(model_name_or_path='${BASE_MODEL}', torch_dtype=torch.bfloat16)
emb.model = PeftModel.from_pretrained(emb.model, '${ADAPTER}')
emb.model = emb.model.merge_and_unload()
emb.model.save_pretrained('${MERGED}')
emb.processor.save_pretrained('${MERGED}')
print('[step ${STEP}] Merge done')
\"
        fi
        ${PY} '${ROOT}/eval/run_mmeb.py' \
            --model_path '${MERGED}' \
            --output_dir '${OUT_DIR}' \
            --full \
            --cache_dir '${ROOT}/datasets/mmeb_cache' \
            --batch_size 16
    " >"${LOG}" 2>&1 &

    echo "  log: ${LOG}"
done

echo "Launched ${#CHECKPOINTS[@]} eval jobs; tail logs under ${RESULTS_BASE}/"
