"""Plot σ₂/σ₃ trajectory for different window sizes W.

For each (W, eta) pair, overlay the σ₂/σ₃ curves across seeds 0,1,2.
Shows whether the signal is window-size invariant.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
RUNS = HERE / "runs"
WINDOWS = [10, 20, 50, 100]
SEEDS = [0, 1, 2]


def load(tag):
    p = RUNS / tag / "log.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(line) for line in open(p)]
    keys = sorted({k for r in rows for k in r.keys()})
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def main():
    fig, axes = plt.subplots(2, len(WINDOWS), figsize=(3.4 * len(WINDOWS), 6),
                             sharex=True, sharey="row")
    for col, W in enumerate(WINDOWS):
        for row, eta in enumerate(["0.0002", "0"]):
            ax = axes[row, col]
            for seed in SEEDS:
                tag = f"win_W{W}_eta{eta}_seed{seed}"
                d = load(tag)
                if d is None:
                    continue
                ax.plot(d["epoch"], d["W_gap23"], lw=1.0, alpha=0.7, label=f"seed {seed}")
            ax.set_yscale("log")
            ax.set_title(f"W={W}, η={eta}")
            ax.grid(alpha=0.3, which="both")
            if col == 0:
                ax.set_ylabel("σ₂/σ₃")
            if row == 1:
                ax.set_xlabel("epoch")
            if col == 0 and row == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    out = RUNS / "window_sensitivity.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
