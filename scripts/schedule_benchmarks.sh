#!/usr/bin/env bash
# Schedule MTEB (CSV tasks) on GPU 0 and MMEB full on GPU 1, detached for SSH logout.
# GPU 1 waits until no other run_mmeb.py process is active, then runs VL-2B-Instruct then Qwen3.5.
# Logs: scheduler_logs/
#
# Gated Hub models: set HF_TOKEN in the environment, or create a single-line file:
#   .hf_token_local   (gitignored; not committed)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${HF_TOKEN:-}" ]]; then
  :
elif [[ -f "${ROOT}/.hf_token_local" ]]; then
  export HF_TOKEN="$(tr -d '\r\n' <"${ROOT}/.hf_token_local")"
  echo "Loaded HF_TOKEN from ${ROOT}/.hf_token_local"
else
  echo "Warning: no HF_TOKEN and no .hf_token_local — gated models may fail (401)." >&2
fi
CSV="${ROOT}/Qwen3-Embedding/tmp_sts_only.csv"
PY="${ROOT}/.venv/bin/python"
EVAL="${ROOT}/Qwen3-Embedding/evaluation"
MMEB="${ROOT}/src/eval/run_mmeb.py"
LOG="${ROOT}/scheduler_logs"
mkdir -p "${LOG}"
TS="$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "${CSV}" ]]; then
  echo "Missing CSV: ${CSV}" >&2
  exit 1
fi
if [[ ! -x "${PY}" && ! -f "${PY}" ]]; then
  echo "Missing venv python: ${PY}" >&2
  exit 1
fi

GPU0_LOG="${LOG}/gpu0_mteb_csv_${TS}.log"
GPU1_LOG="${LOG}/gpu1_mmeb_${TS}.log"

# GPU 0: three sequential MTEB jobs (same 131 tasks as CSV columns; vl_embedding = Qwen repo style)
if command -v setsid >/dev/null 2>&1; then
  LAUNCH="setsid"
else
  LAUNCH=""
fi

nohup ${LAUNCH} bash <<GPU0EOF >>"${GPU0_LOG}" 2>&1 &
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
cd "${EVAL}"
echo "=== GPU0 MTEB CSV chain start $(date -Is) ==="
"${PY}" run_csv_repro.py \
  --tasks_csv "${CSV}" \
  --model_path "Qwen/Qwen3-VL-Embedding-2B" \
  --backend vl_embedding \
  --output_dir "${ROOT}/results/mteb-csv-sts/Qwen3-VL-Embedding-2B" \
  --batch_size 8 \
  --max_length 8192 \
  --precision fp16 \
  || echo "ERROR: job 1/3 (VL-Embedding-2B MTEB CSV) failed — see log"
echo "=== Job 1/3 done $(date -Is) ==="
"${PY}" run_csv_repro.py \
  --tasks_csv "${CSV}" \
  --model_path "Qwen/Qwen3-VL-2B-Instruct" \
  --backend mmteb_chat \
  --output_dir "${ROOT}/results/mteb-csv-sts/Qwen3-VL-2B-Instruct" \
  --max_length 8192 \
  || echo "ERROR: job 2/3 (Qwen3-VL-2B MTEB CSV) failed — check HF_TOKEN / model access"
echo "=== Job 2/3 done $(date -Is) ==="
"${PY}" run_csv_repro.py \
  --tasks_csv "${CSV}" \
  --model_path "Qwen/Qwen3.5-0.8B" \
  --backend mmteb_chat \
  --output_dir "${ROOT}/results/mteb-csv-sts/Qwen3.5-0.8B" \
  --max_length 8192 \
  || echo "ERROR: job 3/3 (Qwen3.5-0.8B MTEB CSV) failed — see log"
echo "=== GPU0 MTEB CSV chain finished $(date -Is) ==="
GPU0EOF

GPU0_PID=$!
echo "${GPU0_PID}" >"${LOG}/gpu0_mteb_csv_${TS}.pid"
disown "${GPU0_PID}" 2>/dev/null || true

# GPU 1: after idle (no run_mmeb.py elsewhere), MMEB full for VL-2B-Instruct then Qwen3.5
nohup ${LAUNCH} bash <<GPU1EOF >>"${GPU1_LOG}" 2>&1 &
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd "${ROOT}"
# pgrep -f 'run_mmeb.py' matches bash -c launchers whose argv contains that string; only treat real Python as busy.
mmeb_python_busy() {
  local pid exe
  for pid in \$(pgrep -f "run_mmeb\\.py" 2>/dev/null || true); do
    exe=\$(readlink -f "/proc/\${pid}/exe" 2>/dev/null || readlink "/proc/\${pid}/exe" 2>/dev/null || true)
    case "\$exe" in */python*|*/python3*) return 0 ;; esac
  done
  return 1
}
echo "=== GPU1 MMEB chain start $(date -Is) ==="
echo "=== Waiting for other run_mmeb.py to finish $(date -Is) ==="
while mmeb_python_busy; do
  sleep 45
done
sleep 5
echo "=== Starting MMEB Qwen3-VL-2B-Instruct $(date -Is) ==="
"${PY}" "${MMEB}" \
  --model_path "Qwen/Qwen3-VL-2B-Instruct" \
  --full \
  --output_dir "${ROOT}/results/mmeb-scheduled/Qwen3-VL-2B-Instruct" \
  --max_length 16384 \
  || echo "ERROR: MMEB Qwen3-VL-2B failed — check HF_TOKEN / local model path"
echo "=== MMEB Qwen3-VL-2B step ended $(date -Is) ==="
"${PY}" "${MMEB}" \
  --model_path "Qwen/Qwen3.5-0.8B" \
  --full \
  --output_dir "${ROOT}/results/mmeb-scheduled/Qwen3.5-0.8B" \
  --max_length 16384 \
  || echo "ERROR: MMEB Qwen3.5-0.8B failed — see log"
echo "=== GPU1 MMEB chain finished $(date -Is) ==="
GPU1EOF

GPU1_PID=$!
echo "${GPU1_PID}" >"${LOG}/gpu1_mmeb_${TS}.pid"
disown "${GPU1_PID}" 2>/dev/null || true

echo "Started detached chains:"
echo "  GPU0 MTEB (CSV tasks) PID ${GPU0_PID}  log ${GPU0_LOG}"
# echo "  GPU1 MMEB (full)      PID ${GPU1_PID}  log ${GPU1_LOG}"
echo "PIDs also written under ${LOG}/"
