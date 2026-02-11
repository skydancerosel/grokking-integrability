#!/usr/bin/env python3
"""
Run add with the ORIGINAL (slow) hyperparameters: lr=5e-5, wd=0.1, 3 layers.
This produces a direct comparison with the Power et al. sweep (lr=1e-3, wd=1.0, 2 layers).

Only runs (a+b) mod 97 — the one operation we know groks with these settings.
3 seeds × 2 wd settings = 6 runs.
"""

import sys
sys.path.insert(0, ".")
from grok_sweep import SweepConfig, OPERATIONS, run_single, OUT_DIR
import torch
from pathlib import Path

SLOW_DIR = Path(__file__).parent / "grok_sweep_results_slow"

def main():
    SLOW_DIR.mkdir(exist_ok=True)

    seeds = [42, 137, 2024]
    weight_decays = [0.1, 0.0]
    total = len(seeds) * len(weight_decays)
    run_idx = 0

    print(f"Slow sweep: add only, lr=5e-5, wd=0.1/0.0, 3 layers, 3 seeds = {total} runs")
    print(f"Output: {SLOW_DIR}/\n")

    for wd in weight_decays:
        for seed in seeds:
            run_idx += 1
            tag = f"add_wd{wd}_s{seed}_slow"
            out_path = SLOW_DIR / f"{tag}.pt"

            if out_path.exists():
                print(f"[{run_idx}/{total}] {tag} — already exists, skipping")
                continue

            print(f"\n[{run_idx}/{total}] {tag}")

            # No-wd runs: cap at 200k like before
            steps = 200_000 if wd == 0.0 else 800_000

            cfg = SweepConfig(
                OP_NAME="add",
                WEIGHT_DECAY=wd,
                SEED=seed,
                # Override to original hyperparameters
                LR=5e-5,
                N_LAYERS=3,
                STEPS=steps,
                EVAL_EVERY=1000,
                MODEL_LOG_EVERY=2000,   # every 2k steps like original
                ADAM_BETA1=0.9,
                ADAM_BETA2=0.999,       # default Adam β₂
            )
            result = run_single(cfg)

            torch.save(result, out_path)
            print(f"  saved → {out_path.name} "
                  f"({len(result['attn_logs'])} snaps, "
                  f"grokked={result['grokked']})")

    print("\nDone.")

if __name__ == "__main__":
    main()
