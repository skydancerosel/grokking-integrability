"""No-MPS reanalysis to address reviewer criticisms #5 (level-metric robustness),
#8 (rank-2 evidence partial), #9 (multi-seed Theorem 6), #11 (baseline comparison).

Uses only data already in runs/sweep_eta*_seed*/log.jsonl
plus the multi-seed theorem6 summaries written by theorem6_verify.py.

Outputs:
  paper/figures/baseline_sigmas.png      -- σ₂/σ₃ vs simpler ΔW spectral baselines
  paper/figures/rank2_evidence.png       -- σ₁,σ₂,σ₃ trajectories for W AND V
  paper/figures/level_metric_decomp.png  -- ρ_tian + components fft_diag, fft_off
  paper/figures/multi_seed_thm6.png      -- multi-seed Theorem 6 sign-match
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
RUNS = HERE / "runs"
FIGS = HERE / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def load(eta_str, prefix="sweep_eta", min_rows=401):
    runs = []
    for d in sorted(RUNS.glob(f"{prefix}{eta_str}_seed*")):
        if not d.is_dir():
            continue
        p = d / "log.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            n = sum(1 for _ in f)
        if n < min_rows:
            continue
        rows = [json.loads(line) for line in open(p)]
        keys = {k for r in rows for k in r.keys()}
        runs.append({k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
                     for k in keys})
    return runs


def stack_med_iqr(runs, key):
    arr = np.stack([r[key] for r in runs], 0)
    med = np.nanmedian(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    return arr, med, q1, q3


def plot_baseline_sigmas(grok, ctrl, ep, outpath):
    """#11: σ₂/σ₃ vs simpler spectral baselines.
    Reviewer asks: do simpler signals — σ₁ alone, σ₃ alone — predict the same
    things as σ₂/σ₃? If yes, the ratio is redundant. If no, it has unique content.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    panels = [
        ("W_sigma1", "σ₁ on ΔW (top eigval, magnitude proxy)", axes[0, 0]),
        ("W_sigma2", "σ₂ on ΔW (the persistent mode)", axes[0, 1]),
        ("W_sigma3", "σ₃ on ΔW (collapses in grok)", axes[1, 0]),
        ("W_gap23", "σ₂/σ₃ on ΔW (our detector)", axes[1, 1]),
    ]
    for key, title, ax in panels:
        for runs, color, lbl in [(grok, "C0", "η=2e-4"), (ctrl, "C1", "η=0")]:
            _, med, q1, q3 = stack_med_iqr(runs, key)
            ax.plot(ep, med, color=color, lw=1.6, label=lbl)
            ax.fill_between(ep, q1, q3, color=color, alpha=0.18)
        ax.axvline(174, color="gray", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(key.replace("_", " "))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("epoch")

    fig.suptitle("Reviewer #11 — alternative spectral baselines on ΔW Gram\n"
                 "(do simpler signals discriminate as cleanly as σ₂/σ₃?)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close()


def plot_rank2_evidence(grok, ctrl, ep, outpath):
    """#8 partial: show σ₁, σ₂, σ₃ separately on log scale for both W and V.
    A genuinely rank-2 spectrum has σ₁ and σ₂ stable while σ₃ collapses (and
    σ₃ joining σ₄, σ₅ at the noise floor). Existing logs have only top-3.
    Showing W and V trajectories side-by-side triangulates: if both spectra
    show the same collapse structure, rank-2 is consistent with both layers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    for col, layer in enumerate(["W", "V"]):
        for sigma_idx, sigma_name in [(1, "σ₁"), (2, "σ₂"), (3, "σ₃")]:
            key = f"{layer}_sigma{sigma_idx}"
            _, gmed, _, _ = stack_med_iqr(grok, key)
            _, cmed, _, _ = stack_med_iqr(ctrl, key)
            axes[0, col].plot(ep, gmed, lw=1.4, label=sigma_name)
            axes[1, col].plot(ep, cmed, lw=1.4, label=sigma_name)

        axes[0, col].set_yscale("log")
        axes[0, col].set_title(f"{layer} updates — η=2e-4 (grok)")
        axes[0, col].set_ylabel("eigenvalue (median, n=15)")
        axes[0, col].axvline(174, color="gray", ls="--", lw=0.8)
        axes[0, col].legend(); axes[0, col].grid(alpha=0.3, which="both")

        axes[1, col].set_yscale("log")
        axes[1, col].set_title(f"{layer} updates — η=0 (control)")
        axes[1, col].set_xlabel("epoch")
        axes[1, col].set_ylabel("eigenvalue (median)")
        axes[1, col].axvline(174, color="gray", ls="--", lw=0.8)
        axes[1, col].legend(); axes[1, col].grid(alpha=0.3, which="both")

    fig.suptitle("Reviewer #8 partial — top-3 eigvals of rolling ΔW and ΔV "
                 "(σ₃ collapses while σ₁, σ₂ stabilize in grok; all three "
                 "collapse to noise in control)", fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close()


def plot_level_metric_decomp(grok, ctrl, ep, outpath):
    """#5 partial: decompose ρ_tian into a(t) and b(t).
    Reviewer asks ρ_tian with a, b frozen at epoch-0. That requires fresh
    training. As a no-MPS proxy: how much do a(t) and b(t) themselves drift
    in grok vs control? If they don't drift much but ρ_tian rises substantially,
    the metric is reading deviation from the (a I + b 11ᵀ) structure beyond
    what re-fitting (a, b) per epoch can explain.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for ax, key, title in [
        (axes[0], "fft_diag", r"(A) $a(t)$ = mean $|$diag$(P_1^\perp F F^\top)|$"),
        (axes[1], "fft_off",  r"(B) $b(t)$ = mean $|$off-diag$(P_1^\perp F F^\top)|$"),
        (axes[2], "fft_dist_from_ideal", r"(C) $\rho_{\mathrm{tian}}(t)$"),
    ]:
        for runs, color, lbl in [(grok, "C0", "η=2e-4"), (ctrl, "C1", "η=0")]:
            _, med, q1, q3 = stack_med_iqr(runs, key)
            ax.plot(ep, med, color=color, lw=1.6, label=lbl)
            ax.fill_between(ep, q1, q3, color=color, alpha=0.18)
        ax.set_title(title)
        if key != "fft_dist_from_ideal":
            ax.set_yscale("log")
        ax.legend(); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("a(t)")
    axes[1].set_ylabel("b(t)")
    axes[2].set_ylabel(r"$\rho_{\mathrm{tian}}$")
    axes[2].axhline(0.075, color="k", ls="--", lw=0.8)
    axes[2].set_xlabel("epoch")

    fig.suptitle("Reviewer #5 partial — components of $\\rho_{\\mathrm{tian}}$ "
                 "(do (a,b) drift, or is the residual driving the rise?)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close()


def plot_multi_seed_thm6(outpath, n_seeds=5):
    """#9: multi-seed Theorem 6 verification."""
    epochs = [50, 100, 175, 250, 300]
    sign_match = {ep: [] for ep in epochs}
    frac_neg = {ep: [] for ep in epochs}
    for s in range(n_seeds):
        p = RUNS / f"sweep_eta0.0002_seed{s}_theorem6_summary.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        for ep in epochs:
            sign_match[ep].append(d[str(ep)]["frac_sign_match"])
            frac_neg[ep].append(d[str(ep)]["frac_neg_B_on_pos_sim"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = epochs
    sm_mat = np.array([sign_match[ep] for ep in epochs])  # [5, n_seeds]
    sm_med = np.median(sm_mat, axis=1)
    sm_q1 = np.percentile(sm_mat, 25, axis=1)
    sm_q3 = np.percentile(sm_mat, 75, axis=1)
    ax.plot(xs, sm_med, "o-", color="C3", lw=2, ms=10, label="sign-match (median)")
    ax.fill_between(xs, sm_q1, sm_q3, color="C3", alpha=0.25, label="IQR")
    # individual seed traces
    for j in range(sm_mat.shape[1]):
        ax.plot(xs, sm_mat[:, j], "-", color="C3", alpha=0.25, lw=0.7)
    ax.axhline(0.95, color="k", ls="--", lw=0.8, label="0.95 saturation")
    ax.axvline(174, color="gray", ls="--", lw=0.8, label="σ₂/σ₃ slope-fire (ep 174)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\Pr[\mathrm{sgn}(B_{j\ell}) = -\mathrm{sgn}(\widetilde f_j^\top P_\eta \widetilde f_\ell)]$")
    ax.set_title(f"Reviewer #9 — Theorem 6 sign-match across n={sm_mat.shape[1]} seeds\n"
                 f"(top-200 most-similar feature pairs, deterministic-replay checkpoints)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.78, 1.005)

    # annotate with median + IQR per epoch
    for x, m, q1, q3 in zip(xs, sm_med, sm_q1, sm_q3):
        ax.annotate(f"{m:.3f}\n[{q1:.3f},{q3:.3f}]", xy=(x, m), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="C3", fontweight="bold")

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close()

    print("\n--- Multi-seed Theorem 6 sign-match summary ---")
    for ep in epochs:
        vals = np.array(sign_match[ep])
        print(f"  epoch {ep:>3d}: median={np.median(vals):.3f}  "
              f"IQR=[{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]  "
              f"min={vals.min():.3f}  max={vals.max():.3f}  n={len(vals)}")


def main():
    grok = load("0.0002")
    ctrl = load("0")
    print(f"loaded {len(grok)} grok, {len(ctrl)} control runs")
    if not grok or not ctrl:
        return
    ep = grok[0]["epoch"]

    plot_baseline_sigmas(grok, ctrl, ep, FIGS / "baseline_sigmas.png")
    plot_rank2_evidence(grok, ctrl, ep, FIGS / "rank2_evidence.png")
    plot_level_metric_decomp(grok, ctrl, ep, FIGS / "level_metric_decomp.png")
    plot_multi_seed_thm6(FIGS / "multi_seed_thm6.png", n_seeds=5)


if __name__ == "__main__":
    main()
