#!/usr/bin/env python3
"""Compare SED-intervention conditions A/B/C/D/E."""

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "intervention_results"
MODES = ["A", "B", "C", "D", "E"]
LABELS = {
    "A": "A: control (no projection)",
    "B": "B: remove SED v₁,₂,₃",
    "C": "C: keep only SED v₁,₂,₃",
    "D": "D: remove random 3D",
    "E": "E: keep only random 3D",
}
COLORS = {"A": "black", "B": "tab:red", "C": "tab:blue",
          "D": "tab:orange", "E": "tab:green"}


def load(m, seed=42):
    return torch.load(ROOT / f"intervention_{m}_s{seed}.pt", map_location="cpu",
                      weights_only=False)


def grok_step(r):
    ta = np.array([m["test_acc"] for m in r["metrics"]])
    ms = np.array([m["step"] for m in r["metrics"]])
    return int(ms[int(np.argmax(ta >= 0.5))]) if (ta >= 0.5).any() else -1


def main():
    runs = {m: load(m) for m in MODES if (ROOT / f"intervention_{m}_s42.pt").exists()}

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    print(f"{'mode':>5}  {'grok @':>8}  {'final step':>11}  {'final cfg':>30}")
    for m, r in runs.items():
        ms = np.array([d["step"] for d in r["metrics"]])
        tr = np.array([d["train_acc"] for d in r["metrics"]])
        te = np.array([d["test_acc"] for d in r["metrics"]])
        c = COLORS[m]
        gs = grok_step(r)
        ax[0].plot(ms, tr, c=c, alpha=0.6, ls="--", lw=1.0)
        ax[0].plot(ms, te, c=c, alpha=1.0, lw=1.5,
                   label=f"{LABELS[m]}  (grok @ {gs if gs>0 else 'never'})")
        ax[1].plot(ms, te - tr, c=c, alpha=1.0, lw=1.5)
        print(f"{m:>5}  {gs:>8}  {r['cfg']['final_step']:>11}  "
              f"grokked={r['cfg']['grokked']}, n_proj={r['cfg'].get('n_projected','?')}")

    ax[0].set_ylabel("accuracy")
    ax[0].set_ylim(-0.02, 1.05)
    ax[0].axhline(0.5, c="gray", ls=":", lw=0.5)
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].set_title("Train (dashed) and test (solid) accuracy under SED intervention")

    ax[1].set_ylabel("test - train acc  (≈ -generalization gap)")
    ax[1].axhline(0, c="gray", ls=":", lw=0.5)
    ax[1].set_xlabel("training step")
    ax[1].set_title("Closing of train/test gap")

    fig.suptitle("SED-LCH causal intervention on attention updates (seed=42, K=3)")
    fig.tight_layout()
    out = ROOT / "intervention_compare.png"
    fig.savefig(out, dpi=130)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
