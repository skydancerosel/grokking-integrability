#!/bin/bash
# Priority 1: η sweep -- 5 η values × 5 seeds × 600 epochs.
# Tests whether the 84-epoch lead time of fft_dist_from_ideal generalizes
# across η or is an artifact of the (M=71, η=2e-4) point.
set -e
cd "$(dirname "$0")"

LOG=runs/eta_sweep.log
echo "=== eta_sweep started $(date) ===" >> "$LOG"

ETAS=(0.00001 0.00005 0.0001 0.0002 0.0005)
SEEDS=(0 1 2 3 4)

for eta in "${ETAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="eta${eta}_seed${seed}"
    if [[ -f "runs/$tag/log.jsonl" ]] && [[ $(wc -l < "runs/$tag/log.jsonl") -ge 601 ]]; then
      echo "[skip] $tag" | tee -a "$LOG"
      continue
    fi
    echo "[run] $tag at $(date +%T)" | tee -a "$LOG"
    python -u tian_eigengap.py \
      --M 71 --K 2048 --n-train 2016 \
      --eta "$eta" --lr 1e-3 \
      --epochs 600 --eval-every 200 \
      --window 20 --seed "$seed" \
      --tag "$tag" >> "$LOG" 2>&1
  done
done

echo "=== eta_sweep done $(date) ===" >> "$LOG"
