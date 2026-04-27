#!/bin/bash
# B3: window-size sensitivity sweep — W ∈ {5, 10, 30} × 3 seeds × 2 conditions.
# Tests whether the σ₂/σ₃ lock-in detector is robust to choice of window size.
set -e
cd "$(dirname "$0")"
LOG=runs/window_sweep.log
echo "=== window_sweep started $(date) ===" >> "$LOG"

WINDOWS=(5 10 30)
SEEDS=(0 1 2)

for W in "${WINDOWS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for eta in 0.0002 0; do
      tag="winW${W}_eta${eta}_seed${seed}"
      if [[ -f "runs/$tag/log.jsonl" ]] && [[ $(wc -l < "runs/$tag/log.jsonl") -ge 401 ]]; then
        echo "[skip] $tag" | tee -a "$LOG"; continue
      fi
      echo "[run] $tag at $(date +%T)" | tee -a "$LOG"
      python -u tian_eigengap.py \
        --M 71 --K 2048 --n-train 2016 \
        --eta "$eta" --lr 1e-3 \
        --epochs 400 --eval-every 200 \
        --window "$W" --seed "$seed" \
        --tag "$tag" >> "$LOG" 2>&1
    done
  done
done

echo "=== window_sweep done $(date) ===" >> "$LOG"
