"""Plot Tian Fig.3 replication + eigengap overlay for the single-seed runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(tag: str):
    p = Path("runs") / tag / "log.jsonl"
    rows = [json.loads(line) for line in open(p)]
    keys = sorted({k for r in rows for k in r.keys()})
    out = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}
    return out


def main():
    grok = load("fig3_eta0.0002_seed0")
    nogrok = load("fig3_eta0_seed0")

    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)

    for col, (data, title) in enumerate([(grok, "η = 0.0002 (grok)"), (nogrok, "η = 0 (no grok)")]):
        ep = data["epoch"]
        # row 0: accuracy
        ax = axes[0, col]
        ax.plot(ep, data["train_acc"], label="train", lw=1.5)
        ax.plot(ep, data["test_acc"], label="test", lw=1.5)
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="center right", fontsize=8)
        ax.grid(alpha=0.3)

        # row 1: Tian metric -- F̃^T F̃ off/diag and FF^T dist from ideal
        ax = axes[1, col]
        ax.plot(ep, data["ftf_ratio"], label="~F^T~F off/diag", lw=1.5)
        ax.plot(ep, data["fft_dist_from_ideal"], label="P1⊥ FF^T dist-from-ideal", lw=1.5, color="C2")
        ax.axhline(0.08, ls="--", color="gray", lw=0.7)
        ax.set_ylabel("Tian metric")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # row 2: |G_F|
        ax = axes[2, col]
        ax.plot(ep, data["gF_norm"], color="C3", lw=1.5)
        ax.set_ylabel("‖G_F‖")
        ax.grid(alpha=0.3)

        # row 3: rolling-window eigengap on W (sigma_2 / sigma_3)
        ax = axes[3, col]
        if "W_gap23" in data:
            ax.plot(ep, data["W_gap23"], label="σ₂/σ₃ (W)", lw=1.2, color="C4")
        if "V_gap23" in data:
            ax.plot(ep, data["V_gap23"], label="σ₂/σ₃ (V)", lw=1.2, color="C5", alpha=0.7)
        ax.set_ylabel("eigengap σ₂/σ₃")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    out = Path("runs") / "fig3_overlay.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
