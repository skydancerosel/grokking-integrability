"""B3 window-size sensitivity analysis.

For each window W ∈ {5, 10, 20, 30}, check:
  1. Does the slope detector still give clean grok-vs-control specificity?
  2. Is the fire epoch stable across W?
  3. Do σ₃, σ₄, σ₅ collapse together (rank-2 evidence) for the new runs that
     log them?

W=20 data is taken from the existing headline sweep (sweep_eta*_seed{0,1,2}).
W=5,10,30 data is from the new winW{W}_eta*_seed{0,1,2} runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
RUNS = HERE / "runs"
FIGS = HERE / "paper" / "figures"


def load_run(tag, min_rows=401):
    p = RUNS / tag / "log.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(line) for line in open(p)]
    if len(rows) < min_rows:
        return None
    keys = {k for r in rows for k in r.keys()}
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def slope(arr, win=25):
    x = np.log(np.clip(arr, 1e-6, None))
    s = np.full_like(x, np.nan)
    s[win:] = (x[win:] - x[:-win]) / win
    return s


def fire_epoch(r, key, slope_thr=0.04, ep_min=100):
    """First epoch ≥ ep_min where slope(d log key / dt) > slope_thr."""
    x = r[key]
    valid = ~np.isnan(x)
    if valid.sum() < 30:
        return float("nan")
    sl = slope(x)
    mask = (sl > slope_thr) & ~np.isnan(sl) & (np.arange(len(sl)) >= ep_min)
    return int(np.where(mask)[0][0]) if mask.any() else float("nan")


def main():
    WINDOWS = [5, 10, 20, 30]
    SEEDS = [0, 1, 2]

    # Load W=20 from existing headline sweep
    data = {W: {"grok": [], "ctrl": []} for W in WINDOWS}
    for s in SEEDS:
        r = load_run(f"sweep_eta0.0002_seed{s}")
        if r is not None:
            data[20]["grok"].append(r)
        r = load_run(f"sweep_eta0_seed{s}")
        if r is not None:
            data[20]["ctrl"].append(r)
    for W in [5, 10, 30]:
        for s in SEEDS:
            r = load_run(f"winW{W}_eta0.0002_seed{s}")
            if r is not None:
                data[W]["grok"].append(r)
            r = load_run(f"winW{W}_eta0_seed{s}")
            if r is not None:
                data[W]["ctrl"].append(r)

    print(f"{'W':>4s}  grok n  ctrl n")
    for W in WINDOWS:
        print(f"{W:>4d}  {len(data[W]['grok']):>6d}  {len(data[W]['ctrl']):>6d}")

    # Fire-time table
    print("\n--- σ₂/σ₃ slope-fire epoch (slope > 0.04 at ep ≥ 100) ---")
    print(f"{'W':>4s}  {'grok fire eps':>20s}  {'ctrl fire eps':>20s}  late σ₂/σ₃ grok  late σ₂/σ₃ ctrl")
    rows_summary = []
    for W in WINDOWS:
        gf = [fire_epoch(r, "W_gap23") for r in data[W]["grok"]]
        cf = [fire_epoch(r, "W_gap23") for r in data[W]["ctrl"]]
        # late stage σ₂/σ₃ in epoch 200-400
        late_g = [np.nanmedian(r["W_gap23"][200:401]) for r in data[W]["grok"]]
        late_c = [np.nanmedian(r["W_gap23"][200:401]) for r in data[W]["ctrl"]]
        rows_summary.append((W, gf, cf, late_g, late_c))
        gf_str = "[" + ", ".join(f"{f:.0f}" if not np.isnan(f) else "—" for f in gf) + "]"
        cf_str = "[" + ", ".join(f"{f:.0f}" if not np.isnan(f) else "—" for f in cf) + "]"
        print(f"{W:>4d}  {gf_str:>20s}  {cf_str:>20s}  {np.median(late_g):>14.2f}  {np.median(late_c):>14.2f}")

    # Plot: σ₂/σ₃ trajectories at each W
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, W in zip(axes.flat, WINDOWS):
        for r in data[W]["grok"]:
            ax.plot(r["epoch"], r["W_gap23"], color="C0", lw=1.0, alpha=0.7)
        for r in data[W]["ctrl"]:
            ax.plot(r["epoch"], r["W_gap23"], color="C1", lw=1.0, alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"W = {W}")
        ax.set_ylabel("σ₂/σ₃ on ΔW")
        ax.axvline(174, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.3, which="both")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    fig.suptitle("Reviewer #3 — window-size sensitivity\n"
                 "blue = grok (η=2e-4), orange = control (η=0); 3 seeds each\n"
                 "vertical line = canonical slope-fire epoch 174 (W=20)", fontsize=11)
    plt.tight_layout()
    out = FIGS / "window_sweep.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")
    plt.close()

    # Plot 2: rank-2 evidence with σ₁..σ₅ at each W (only the new W=5,10,30 have σ₄,σ₅)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, W in zip(axes, [5, 10, 30]):
        gr = data[W]["grok"]
        if not gr or "W_sigma4" not in gr[0]:
            continue
        for sigma_idx in [1, 2, 3, 4, 5]:
            arr = np.stack([r[f"W_sigma{sigma_idx}"] for r in gr], 0)
            med = np.nanmedian(arr, axis=0)
            ax.plot(gr[0]["epoch"], med, lw=1.4, label=f"σ_{sigma_idx}")
        ax.set_yscale("log")
        ax.set_title(f"W = {W}, η=2e-4")
        ax.set_xlabel("epoch")
        ax.axvline(174, color="gray", ls="--", lw=0.8)
        ax.legend(loc="lower left", ncol=2, fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("eigenvalue (median, n=3)")
    fig.suptitle("Reviewer #8 — top-5 eigvals of rolling ΔW Gram (rank-2 evidence)\n"
                 "σ₃, σ₄, σ₅ should collapse together at noise floor while σ₁, σ₂ persist",
                 fontsize=11)
    plt.tight_layout()
    out = FIGS / "rank2_top5.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
    plt.close()

    # Print σ₄ vs σ₃ vs σ₅ at the lock-in epoch for each W
    print("\n--- σ₃, σ₄, σ₅ at epoch 200 (W=5,10,30 only), grok median across 3 seeds ---")
    print(f"{'W':>4s}  {'σ₃':>12s}  {'σ₄':>12s}  {'σ₅':>12s}  {'σ₃/σ₅':>8s}")
    for W in [5, 10, 30]:
        gr = data[W]["grok"]
        if not gr:
            continue
        s3 = np.median([r["W_sigma3"][200] for r in gr])
        s4 = np.median([r["W_sigma4"][200] for r in gr])
        s5 = np.median([r["W_sigma5"][200] for r in gr])
        ratio = s3/s5 if s5 > 0 else float("nan")
        print(f"{W:>4d}  {s3:>12.3e}  {s4:>12.3e}  {s5:>12.3e}  {ratio:>8.2f}")


if __name__ == "__main__":
    main()
