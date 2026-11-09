#!/bin/bash
# The Assignment 2 experiment matrix of EXPERIMENTS.md, run sequentially on one device.
#
#   scripts/experiments.sh [arm ...]
#
# With no argument it runs every arm. Arms already marked done are skipped, so the script can
# be interrupted and restarted. Everything goes to results/experiments.log, one timestamp per
# line, and nothing to the terminal: thirteen runs of forty epochs would bury the shell.
#
# Arms are ordered so the headline comparison finishes first and the extension is last:
#   e1-deeplabv3  pretrained arm, also the reference E2 and E3 are measured against
#   e1-unet       baseline trained from scratch
#   e2-frozen     the controlled experiment
#   e3-dice       the ablation
#   e4-resnet50   extension, seed 0 only, droppable
set -u
set -o pipefail  # without this, the log filter would report success for a run that crashed

cd "$(dirname "$0")/.." || exit 1
PYTHON="../.venv/bin/python"
MARKERS="results/.completed"
LOG="results/experiments.log"
mkdir -p "$MARKERS" "$(dirname "$LOG")"

log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

# Stamps each line a run prints. Reading line by line keeps the log current while the run is
# still going, which a plain redirect would not.
stamp() { while IFS= read -r line; do printf '%s  %s\n' "$(date '+%F %T')" "$line"; done >> "$LOG"; }

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

echo "$total runs queued; follow results/experiments.log"
log "=== $total runs queued ==="
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
    log "[$index/$total] $arm seed $seed | already done, skipping"
    continue
  fi

  args=(--config "$(config_for "$arm")" --set "seed=$seed")
  [ -n "$override" ] && args+=(--set "$override")

  log ""
  log "[$index/$total] $arm seed $seed | started"
  started=$(date +%s)

  # -u keeps the per-epoch lines coming: piped onward, Python would block-buffer stdout and
  # the progress would only reach the log once a run had already finished.
  if PYTHONUNBUFFERED=1 "$PYTHON" -u -m src.train "${args[@]}" 2>&1 | stamp; then
    elapsed=$(( $(date +%s) - started ))
    overall=$(( $(date +%s) - started_all ))
    date +%s > "$marker"
    log "$(printf '[%d/%d] %s seed %s | done in %dh%02dm, %dh%02dm elapsed overall' \
      "$index" "$total" "$arm" "$seed" $((elapsed / 3600)) $(((elapsed % 3600) / 60)) \
      $((overall / 3600)) $(((overall % 3600) / 60)))"
  else
    log "[$index/$total] $arm seed $seed | FAILED, continuing with the next arm"
  fi
done

log ""
log "=== queue finished ==="
