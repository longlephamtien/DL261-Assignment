#!/bin/bash
# The Assignment 2 experiment matrix of EXPERIMENTS.md, run sequentially on one device.
#
#   scripts/experiments.sh [arm ...]
#
# With no argument it runs every arm. Arms already marked done are skipped, so the script can
# be interrupted and restarted. Progress goes to the terminal and to results/experiments.log.
#
# Arms are ordered so the headline comparison finishes first and the extension is last:
#   e1-deeplabv3  pretrained arm, also the reference E2 and E3 are measured against
#   e1-unet       baseline trained from scratch
#   e2-frozen     the controlled experiment
#   e3-dice       the ablation
#   e4-resnet50   extension, seed 0 only, droppable
set -u
set -o pipefail  # without this, `tee` would report success for a run that crashed

cd "$(dirname "$0")/.." || exit 1
PYTHON="../.venv/bin/python"
MARKERS="results/.completed"
LOG="results/experiments.log"
mkdir -p "$MARKERS" "$(dirname "$LOG")"

[ -x "$PYTHON" ] || { echo "no interpreter at $PYTHON; create the virtual environment first"; exit 1; }

declare -a ARMS=()
for seed in 0 1 2; do ARMS+=("e1-deeplabv3:$seed:"); done
for seed in 0 1 2; do ARMS+=("e1-unet:$seed:"); done
for seed in 0 1 2; do ARMS+=("e2-frozen:$seed:model.args.freeze_backbone=true"); done
for seed in 0 1 2; do ARMS+=("e3-dice:$seed:train.loss=dice"); done
ARMS+=("e4-resnet50:0:model.args.backbone=resnet50")

config_for() { case "$1" in e1-unet) echo configs/unet.yaml;; *) echo configs/deeplabv3.yaml;; esac; }

selected=()
for entry in "${ARMS[@]}"; do
  arm="${entry%%:*}"
  if [ "$#" -eq 0 ]; then selected+=("$entry"); else
    for choice in "$@"; do [ "$choice" = "$arm" ] && selected+=("$entry"); done
  fi
done

total=${#selected[@]}
[ "$total" -eq 0 ] && { echo "no arm matched: $*"; exit 1; }

echo "=== $total runs queued, $(date '+%F %T') ===" | tee -a "$LOG"
started_all=$(date +%s)
index=0

for entry in "${selected[@]}"; do
  index=$((index + 1))
  arm="${entry%%:*}"
  rest="${entry#*:}"
  seed="${rest%%:*}"
  override="${rest#*:}"
  marker="$MARKERS/$arm-seed$seed"

  if [ -f "$marker" ]; then
    echo "[$index/$total] $arm seed $seed | already done, skipping" | tee -a "$LOG"
    continue
  fi

  args=(--config "$(config_for "$arm")" --set "seed=$seed")
  [ -n "$override" ] && args+=(--set "$override")

  echo "" | tee -a "$LOG"
  echo "[$index/$total] $arm seed $seed | started $(date '+%F %T')" | tee -a "$LOG"
  started=$(date +%s)

  if "$PYTHON" -m src.train "${args[@]}" 2>&1 | tee -a "$LOG"; then
    elapsed=$(( $(date +%s) - started ))
    date +%s > "$marker"
    printf '[%d/%d] %s seed %s | done in %dh%02dm, %dh%02dm elapsed overall\n' \
      "$index" "$total" "$arm" "$seed" $((elapsed / 3600)) $(((elapsed % 3600) / 60)) \
      $((($(date +%s) - started_all) / 3600)) $(((($(date +%s) - started_all) % 3600) / 60)) | tee -a "$LOG"
  else
    echo "[$index/$total] $arm seed $seed | FAILED, continuing with the next arm" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== queue finished $(date '+%F %T') ===" | tee -a "$LOG"
