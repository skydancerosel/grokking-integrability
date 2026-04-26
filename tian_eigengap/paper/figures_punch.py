"""Custom figure for the punch note: σ₂/σ₃ curve with Theorem 6 sign-match overlay."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
RUNS = HERE.parent / "runs"


def load_runs(eta_str, min_rows=401):
    runs = []
    for d in sorted(RUNS.glob(f"sweep_eta{eta_str}_seed*")):
        if not d.is_dir():
            continue
        p = d / "log.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            n = sum(1 for _ in f)
        if n < min_rows:
            continue
        rows = [json.loads(l) for l in open(p)]
        runs.append({k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
                     for k in {kk for r in rows for kk in r.keys()}})
    return runs


def main():
    grok = load_runs("0.0002")
    ctrl = load_runs("0")
    ep = grok[0]["epoch"]

    # Theorem 6 sign-match values from theorem6_verify.py output
    thm6_epochs = np.array([50, 100, 175, 250, 300])
    thm6_match = np.array([0.83, 0.91, 0.975, 0.975, 0.995])

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))

    # σ₂/σ₃ curves
    g_arr = np.stack([r["W_gap23"] for r in grok], axis=0)
    c_arr = np.stack([r["W_gap23"] for r in ctrl], axis=0)
    g_med = np.nanmedian(g_arr, axis=0)
    g_q1 = np.nanpercentile(g_arr, 25, axis=0)
    g_q3 = np.nanpercentile(g_arr, 75, axis=0)
    c_med = np.nanmedian(c_arr, axis=0)

    line_g = ax1.plot(ep, g_med, color="C0", lw=2.0, label=r"$\sigma_2/\sigma_3$ on $\Delta W$, $\eta=2\!\times\!10^{-4}$ (grok)")[0]
    ax1.fill_between(ep, g_q1, g_q3, color="C0", alpha=0.18)
    line_c = ax1.plot(ep, c_med, color="C1", lw=2.0, label=r"$\sigma_2/\sigma_3$, $\eta = 0$ (control)")[0]
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(r"$\sigma_2/\sigma_3$ on rolling $\Delta W$ Gram (log)")
    ax1.grid(alpha=0.3, which="both")

    # mark slope-fire epoch
    ax1.axvline(174, color="C0", ls="--", lw=1.2, alpha=0.7)
    ax1.text(174, ax1.get_ylim()[1] * 0.4,
             "slope-fire\nep 174",
             color="C0", ha="center", va="top", fontsize=9)

    # Theorem 6 sign-match on twin axis
    ax2 = ax1.twinx()
    line_t = ax2.plot(thm6_epochs, thm6_match, "o-", color="C3",
                       lw=1.8, ms=10, mec="black", mew=0.7,
                       label="Theorem 6 sign-match (top-200 similar pairs)")[0]
    for ep_t, m in zip(thm6_epochs, thm6_match):
        ax2.annotate(f"{m:.3f}",
                     (ep_t, m), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color="C3", fontweight="bold")
    ax2.set_ylim(0.5, 1.05)
    ax2.set_ylabel("sign-match  $\\Pr[\\,\\mathrm{sgn}(B_{j\\ell}) = -\\mathrm{sgn}(\\widetilde f_j^\\top P_\\eta \\widetilde f_\\ell)\\,]$",
                   color="C3")
    ax2.tick_params(axis="y", colors="C3")

    # combined legend
    lines = [line_g, line_c, line_t]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=9, framealpha=0.95)

    ax1.set_title(r"Stage III lock-in: $\sigma_2/\sigma_3$ rises as Theorem 6 saturates")

    plt.tight_layout()
    out = HERE / "figures" / "punch_lockin.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
