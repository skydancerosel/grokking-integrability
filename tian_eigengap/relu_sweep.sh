#!/bin/bash
# B1: ReLU activation sweep — same setup as headline sweep but σ(x)=ReLU(x).
# Tests whether spectral signatures generalize beyond the σ=x² regime where
# Tian's Theorem 6 was proven.
# 800 epochs (vs 400 for sqr) because ReLU groks on a longer timescale on this setup.
set -e
cd "$(dirname "$0")"
LOG=runs/relu_sweep.log
echo "=== relu_sweep started $(date) ===" >> "$LOG"

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  for eta in 0.0002 0; do
    tag="relu_eta${eta}_seed${seed}"
    if [[ -f "runs/$tag/log.jsonl" ]] && [[ $(wc -l < "runs/$tag/log.jsonl") -ge 801 ]]; then
      echo "[skip] $tag" | tee -a "$LOG"; continue
    fi
    echo "[run] $tag at $(date +%T)" | tee -a "$LOG"
    python -u tian_eigengap.py \
      --M 71 --K 2048 --n-train 2016 \
      --eta "$eta" --lr 1e-3 \
      --epochs 800 --eval-every 200 \
      --window 20 --seed "$seed" \
      --activation relu \
      --tag "$tag" >> "$LOG" 2>&1
  done
done

echo "=== relu_sweep done $(date) ===" >> "$LOG"
