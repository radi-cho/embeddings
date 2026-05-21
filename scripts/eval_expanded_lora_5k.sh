#!/usr/bin/env bash
# Evaluate stage1-expanded-lora-5k checkpoints on MMEB, 8 in parallel on 8 GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BASE_MODEL="${ROOT}/models/checkpoints/Qwen3.5-0.8B"
CKPT_DIR="${ROOT}/data/outputs/stage1-expanded-lora-5k"
RESULTS_BASE="${ROOT}/results/stage1-expanded-lora-mmeb"

CHECKPOINTS=(1000 1500 2000 2500 3000 3500 4500 final)
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p "${RESULTS_BASE}"

for i in "${!CHECKPOINTS[@]}"; do
    STEP="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"

    if [[ "${STEP}" == "final" ]]; then
        ADAPTER="${CKPT_DIR}/final"
        MERGED="${CKPT_DIR}/merged-final"
    else
        ADAPTER="${CKPT_DIR}/checkpoint-${STEP}"
        MERGED="${CKPT_DIR}/merged-${STEP}"
    fi

    OUT_DIR="${RESULTS_BASE}/step-${STEP}"
    LOG="${RESULTS_BASE}/step-${STEP}.log"

    if [[ -f "${OUT_DIR}/summary.json" ]]; then
        echo "SKIP: ${OUT_DIR}/summary.json already exists"
        continue
    fi

    if [[ ! -d "${ADAPTER}" ]]; then
        echo "SKIP: ${ADAPTER} not found"
        continue
    fi

    echo "GPU ${GPU}: step ${STEP} → ${OUT_DIR}"

    nohup bash -c "
        set -e
        export CUDA_VISIBLE_DEVICES=${GPU}

        # Merge LoRA adapter
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
        else
            echo '[step ${STEP}] Merged model exists, skipping merge'
        fi

        # Run full MMEB eval
        echo '[step ${STEP}] Starting eval on GPU ${GPU}...'
        ${PY} ${ROOT}/eval/run_mmeb.py \
            --model_path '${MERGED}' \
            --output_dir '${OUT_DIR}' \
            --full \
            --cache_dir '${ROOT}/datasets/mmeb_cache' \
            --batch_size 16

        echo '[step ${STEP}] DONE'
    " > "${LOG}" 2>&1 &

    echo "  PID=$! → ${LOG}"
done

echo ""
echo "All jobs launched. Monitor with:"
echo "  tail -f ${RESULTS_BASE}/step-*.log | grep -E 'hit@1|Merging|DONE|Error'"
