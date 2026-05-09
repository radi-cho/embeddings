#!/usr/bin/env bash
# Wait for mining to finish, then launch Stage 2.
set -euo pipefail
cd /home/shared/embeddings
source .venv/bin/activate

echo "[$(date)] Waiting for mining to finish..."
while pgrep -f "mine_hard_negatives" > /dev/null 2>&1; do
    sleep 60
    echo "[$(date)] Mining still running... $(ls data/training_data_mined/*.jsonl 2>/dev/null | wc -l) files mined"
done

echo "[$(date)] Mining complete. Files:"
ls -lh data/training_data_mined/

echo "[$(date)] Launching Stage 2 training..."
exec bash scripts/run_stage2.sh
