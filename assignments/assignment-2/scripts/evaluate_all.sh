#!/bin/bash
# Score every finished run on both held-out splits: val (model selection) and test (reporting).
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PYTHON="../.venv/bin/python"
LOG="results/evaluate.log"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

for run in results/runs/*/; do
  run="$(basename "$run")"
  for split in val test; do
    if [ -f "results/evaluation/$run/$split/report.json" ]; then
      log "$run $split | already done"; continue
    fi
    log "$run $split | started"
    if out=$(PYTHONUNBUFFERED=1 "$PYTHON" -u -m src.evaluate "$run" --split "$split" 2>&1); then
      log "$run $split | $out"
    else
      log "$run $split | FAILED: $out"
    fi
  done
done
log "=== evaluation finished ==="
