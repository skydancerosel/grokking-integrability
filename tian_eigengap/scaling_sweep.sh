#!/bin/bash
# Priority 3: M × p × seed scaling sweep.
# M ∈ {41, 71, 127}, p ∈ {0.1, 0.2, 0.3, 0.5}, 5 seeds, η=2e-4, K=2048.
# Train length scales with M. Tests whether fft_dist fire correlates with
# Tian's Theorem 4 generalization boundary (n_critical ~ M log M).
set -e
cd "$(dirname "$0")"

LOG=runs/scaling_sweep.log
echo "=== scaling_sweep started $(date) ===" >> "$LOG"

# (M, epochs)
declare -a MS=("41:400" "71:600" "127:1000")
PS=(0.1 0.2 0.3 0.5)
SEEDS=(0 1 2 3 4)

for ms in "${MS[@]}"; do
  M="${ms%:*}"
  EPOCHS="${ms#*:}"
  for p in "${PS[@]}"; do
    for s in "${SEEDS[@]}"; do
      tag="scal_M${M}_p${p}_seed${s}"
      total_examples=$((M * M))
      n_train=$(python -c "print(int(round($p * $total_examples)))")
      expected_rows=$((EPOCHS + 1))
      if [[ -f "runs/$tag/log.jsonl" ]] && [[ $(wc -l < "runs/$tag/log.jsonl") -ge $expected_rows ]]; then
        echo "[skip] $tag" | tee -a "$LOG"
        continue
      fi
      echo "[run] $tag M=$M p=$p n_train=$n_train epochs=$EPOCHS at $(date +%T)" | tee -a "$LOG"
      python -u tian_eigengap.py \
        --M "$M" --K 2048 --n-train "$n_train" \
        --eta 2e-4 --lr 1e-3 \
        --epochs "$EPOCHS" --eval-every 200 \
        --window 20 --seed "$s" \
        --tag "$tag" >> "$LOG" 2>&1
    done
  done
done

echo "=== scaling_sweep done $(date) ===" >> "$LOG"
