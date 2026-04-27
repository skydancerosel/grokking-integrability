"""B1 ReLU comparison plot: σ=x² vs σ=ReLU spectral signatures and Theorem 6."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
RUNS = HERE / "runs"
FIGS = HERE / "paper" / "figures"


def load_set(prefix, n_seeds=15, min_rows=801):
    runs = []
    for s in range(n_seeds):
        p = RUNS / f"{prefix}{s}" / "log.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(line) for line in open(p)]
        if len(rows) < min_rows:
            continue
        keys = {k for r in rows for k in r.keys()}
        runs.append({k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys})
    return runs


def main():
    # σ=x² runs (use 400-epoch sweep, pad NaN)
    sqr_grok = []
    sqr_ctrl = []
    for s in range(15):
        for path, dst in [(f"sweep_eta0.0002_seed{s}", sqr_grok),
                          (f"sweep_eta0_seed{s}", sqr_ctrl)]:
            p = RUNS / path / "log.jsonl"
            if not p.exists():
                continue
            rows = [json.loads(line) for line in open(p)]
            if len(rows) < 401:
                continue
            keys = {k for r in rows for k in r.keys()}
            dst.append({k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys})

    relu_grok = load_set("relu_eta0.0002_seed")
    relu_ctrl = load_set("relu_eta0_seed")
    print(f"σ=x²: {len(sqr_grok)} grok, {len(sqr_ctrl)} ctrl")
    print(f"σ=ReLU: {len(relu_grok)} grok, {len(relu_ctrl)} ctrl")

    # Plot: 4 panels — test acc, σ₂/σ₃ on ΔW, σ₁/σ₂ on ΔW, fft_dist
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel A: test acc
    ax = axes[0, 0]
    for runs, color, lbl, n_ep in [(sqr_grok, "C0", "x² grok", 400),
                                    (sqr_ctrl, "C0", "x² ctrl", 400),
                                    (relu_grok, "C2", "ReLU grok", 800),
                                    (relu_ctrl, "C2", "ReLU ctrl", 800)]:
        if not runs:
            continue
        arr = np.stack([r["test_acc"][:n_ep+1] for r in runs], 0)
        med = np.median(arr, 0)
        ep = np.arange(n_ep + 1)
        ls = "-" if "grok" in lbl else "--"
        ax.plot(ep, med, color=color, lw=1.5, ls=ls, label=lbl)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
    ax.set_title("(A) Test accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.05)

    # Panel B: σ₂/σ₃ on ΔW
    ax = axes[0, 1]
    for runs, color, lbl, n_ep in [(sqr_grok, "C0", "x² grok", 400),
                                    (sqr_ctrl, "C0", "x² ctrl", 400),
                                    (relu_grok, "C2", "ReLU grok", 800),
                                    (relu_ctrl, "C2", "ReLU ctrl", 800)]:
        if not runs:
            continue
        arr = np.stack([r["W_gap23"][:n_ep+1] for r in runs], 0)
        med = np.nanmedian(arr, 0)
        ep = np.arange(n_ep + 1)
        ls = "-" if "grok" in lbl else "--"
        ax.plot(ep, med, color=color, lw=1.5, ls=ls, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("σ₂/σ₃ on ΔW")
    ax.set_title("(B) σ₂/σ₃ — rank-2 lock-in detector\nworks for x², fails for ReLU")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # Panel C: σ₁/σ₂ on ΔW
    ax = axes[1, 0]
    for runs, color, lbl, n_ep in [(sqr_grok, "C0", "x² grok", 400),
                                    (sqr_ctrl, "C0", "x² ctrl", 400),
                                    (relu_grok, "C2", "ReLU grok", 800),
                                    (relu_ctrl, "C2", "ReLU ctrl", 800)]:
        if not runs:
            continue
        arr = np.stack([r["W_sigma1"][:n_ep+1] / np.maximum(r["W_sigma2"][:n_ep+1], 1e-30) for r in runs], 0)
        med = np.nanmedian(arr, 0)
        ep = np.arange(n_ep + 1)
        ls = "-" if "grok" in lbl else "--"
        ax.plot(ep, med, color=color, lw=1.5, ls=ls, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("σ₁/σ₂ on ΔW")
    ax.set_title("(C) σ₁/σ₂ — would be the rank-1 analogue\n(modest separation on ReLU)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # Panel D: fft_dist_from_ideal
    ax = axes[1, 1]
    for runs, color, lbl, n_ep in [(sqr_grok, "C0", "x² grok", 400),
                                    (sqr_ctrl, "C0", "x² ctrl", 400),
                                    (relu_grok, "C2", "ReLU grok", 800),
                                    (relu_ctrl, "C2", "ReLU ctrl", 800)]:
        if not runs:
            continue
        arr = np.stack([r["fft_dist_from_ideal"][:n_ep+1] for r in runs], 0)
        med = np.median(arr, 0)
        ep = np.arange(n_ep + 1)
        ls = "-" if "grok" in lbl else "--"
        ax.plot(ep, med, color=color, lw=1.5, ls=ls, label=lbl)
    ax.axhline(0.075, color="k", ls=":", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\rho_{\mathrm{tian}}$")
    ax.set_title("(D) Tian level metric — fires immediately at ep 0 for ReLU\n(initialization is already non-lazy)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Reviewer #2 — σ=x² vs σ=ReLU: spectral signatures are activation-specific\n"
                 "(Theorem 6 sign rule itself is general; rolling-ΔW manifestation is not)",
                 fontsize=11)
    plt.tight_layout()
    out = FIGS / "relu_comparison.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
