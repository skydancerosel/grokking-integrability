#!/bin/bash
# 15 seeds x 2 conditions (eta=0.0002 grok, eta=0 control), 400 epochs each.
# Sequential to avoid MPS contention. ~35 min on M4 Max.
set -e
cd "$(dirname "$0")"

LOG=runs/sweep.log
echo "=== sweep started $(date) ===" >> "$LOG"

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  for eta in 0.0002 0; do
    tag="sweep_eta${eta}_seed${seed}"
    if [[ -f "runs/$tag/log.jsonl" ]] && [[ $(wc -l < "runs/$tag/log.jsonl") -ge 401 ]]; then
      echo "[skip] $tag already complete" | tee -a "$LOG"
      continue
    fi
    echo "[run]  $tag  $(date +%T)" | tee -a "$LOG"
    python -u tian_eigengap.py \
      --M 71 --K 2048 --n-train 2016 \
      --eta "$eta" --lr 1e-3 \
      --epochs 400 --eval-every 100 \
      --window 20 --seed "$seed" \
      --tag "$tag" >> "$LOG" 2>&1
  done
done

echo "=== sweep done $(date) ===" >> "$LOG"
