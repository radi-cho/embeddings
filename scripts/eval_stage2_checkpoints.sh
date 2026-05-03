#!/usr/bin/env bash
# Merge LoRA checkpoints and run full MMEB eval on 8 GPUs, one checkpoint per GPU.
# Survives SSH disconnect via nohup + setsid.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BASE_MODEL="${ROOT}/models/checkpoints/Qwen3.5-0.8B"
CKPT_DIR="/data/outputs/qwen35-0.8b-stage2-k15"
RESULTS_BASE="${ROOT}/results/stage2-mmeb-checkpoints"

# 8 checkpoints: first, final, and 6 evenly spaced in between
CHECKPOINTS=(500 2000 3500 5000 7000 8500 10000 final)
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p "${RESULTS_BASE}"

for i in "${!CHECKPOINTS[@]}"; do
    STEP="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"

    if [[ "${STEP}" == "final" ]]; then
        ADAPTER="${CKPT_DIR}/final"
        TAG="final"
    else
        ADAPTER="${CKPT_DIR}/checkpoint-${STEP}"
        TAG="step-${STEP}"
    fi

    MERGED="${CKPT_DIR}/merged-${STEP}"
    OUT_DIR="${RESULTS_BASE}/${TAG}"
    LOG="${RESULTS_BASE}/${TAG}.log"

    if [[ ! -d "${ADAPTER}" ]]; then
        echo "SKIP: ${ADAPTER} not found"
        continue
    fi

    echo "GPU ${GPU}: ${TAG} → ${OUT_DIR}"

    nohup setsid bash -c "
        set -e
        export CUDA_VISIBLE_DEVICES=${GPU}
        export OMP_NUM_THREADS=4
        export TOKENIZERS_PARALLELISM=false

        if [[ ! -f '${MERGED}/config.json' ]]; then
            echo '[${TAG}] Merging adapter...'
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
print('[${TAG}] Merge done')
\"
        else
            echo '[${TAG}] Merged model exists, skipping merge'
        fi

        echo '[${TAG}] Starting eval on GPU ${GPU}...'
        ${PY} ${ROOT}/eval/run_mmeb.py \
            --model_path '${MERGED}' \
            --output_dir '${OUT_DIR}' \
            --full \
            --cache_dir '${ROOT}/datasets/mmeb_cache' \
            --batch_size 16

        echo '[${TAG}] DONE'
    " > "${LOG}" 2>&1 < /dev/null &

    echo "  PID=$! → ${LOG}"
done

echo ""
echo "All 8 jobs launched. Monitor with:"
echo "  tail -f ${RESULTS_BASE}/*.log | grep -E 'hit@1|Merging|DONE|Error'"
