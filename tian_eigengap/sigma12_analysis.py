"""σ₁/σ₂ reanalysis on existing 30-run sweep logs.

Tests the hypothesis: σ₁/σ₂ on rolling-window ΔW Gram opens at Stage I→II,
leading test accuracy by ~50 epochs in grok and never opening in control.

Outputs:
  runs/sigma12_trajectories.png  -- per-seed σ₁/σ₂ overlays (linear + log)
  runs/sigma12_fire_times.png    -- fire-time distributions for several thresholds
  runs/sigma12_headline.png      -- 3-row overlay (acc / σ₁/σ₂ / σ₂/σ₃)
  prints fire-time stats for thresholds {10, 25, 50, 100, 200} and slope-based
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = Path(__file__).parent / "runs"


def load_runs(eta_str: str, prefix: str = "sweep_eta", min_rows: int = 401):
    runs = []
    for d in sorted(RUNS.glob(f"{prefix}{eta_str}_seed*")):
        p = d / "log.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            n = sum(1 for _ in f)
        if n < min_rows:
            continue
        rows = [json.loads(line) for line in open(p)]
        keys = sorted({k for r in rows for k in r.keys()})
        runs.append({k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
                     for k in keys})
    return runs


def fire_time_threshold(x: np.ndarray, threshold: float,
                        sustained: int = 1,
                        ep_min: int = 0) -> float:
    """First epoch >= ep_min where x sustains >= threshold for `sustained` consecutive
    epochs."""
    if sustained <= 1:
        crossings = np.where((x >= threshold) & (np.arange(len(x)) >= ep_min))[0]
        return int(crossings[0]) if len(crossings) else float("nan")
    # check for sustained crossings
    above = (x >= threshold).astype(int)
    for i in range(ep_min, len(x) - sustained + 1):
        if above[i:i + sustained].all():
            return int(i)
    return float("nan")


def main():
    grok = load_runs("0.0002")
    ctrl = load_runs("0")
    print(f"loaded {len(grok)} grok, {len(ctrl)} control runs")
    if not grok or not ctrl:
        print("no runs!")
        return

    ep = grok[0]["epoch"]

    # ---- 1. σ₁/σ₂ trajectories (per-seed overlay)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (runs, title) in enumerate([(grok, "η = 0.0002 (grok)"),
                                         (ctrl, "η = 0 (control)")]):
        for r in runs:
            s12 = np.where(r["W_sigma2"] > 0,
                           r["W_sigma1"] / np.maximum(r["W_sigma2"], 1e-30),
                           np.nan)
            axes[0, col].plot(ep, s12, lw=0.7, alpha=0.5)
            axes[1, col].plot(ep, s12, lw=0.7, alpha=0.5)
        axes[0, col].set_title(f"{title} — σ₁/σ₂ on ΔW (linear)")
        axes[0, col].set_ylabel("σ₁/σ₂")
        axes[0, col].grid(alpha=0.3)
        axes[1, col].set_title(f"{title} — σ₁/σ₂ on ΔW (log)")
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylabel("σ₁/σ₂ (log)")
        axes[1, col].set_xlabel("epoch")
        axes[1, col].grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(RUNS / "sigma12_trajectories.png", dpi=120)
    plt.close()
    print("saved sigma12_trajectories.png")

    # ---- 2. Fire-time analysis at multiple thresholds
    THRESHOLDS = [10, 25, 50, 100, 200]
    SUSTAINED = [1, 5, 10]
    print()
    print(f"{'threshold':>10s}  {'sustained':>10s}  {'grok median':>14s}  {'IQR':>14s}  {'grok n_fired':>14s}  {'ctrl n_fired':>14s}")
    for thr in THRESHOLDS:
        for sus in SUSTAINED:
            fg = []
            fc = []
            for r in grok:
                s12 = r["W_sigma1"] / np.maximum(r["W_sigma2"], 1e-30)
                # NaN out before window fills
                valid = ~np.isnan(r["W_sigma1"])
                s12_fill = np.where(valid, s12, -np.inf)
                fg.append(fire_time_threshold(s12_fill, thr, sus))
            for r in ctrl:
                s12 = r["W_sigma1"] / np.maximum(r["W_sigma2"], 1e-30)
                valid = ~np.isnan(r["W_sigma1"])
                s12_fill = np.where(valid, s12, -np.inf)
                fc.append(fire_time_threshold(s12_fill, thr, sus))
            fg = np.array(fg, dtype=float)
            fc = np.array(fc, dtype=float)
            med = np.nanmedian(fg)
            iqr = (np.nanpercentile(fg, 25), np.nanpercentile(fg, 75))
            print(f"{thr:>10.0f}  {sus:>10d}  {med:>14.0f}  {f'[{iqr[0]:.0f},{iqr[1]:.0f}]':>14s}"
                  f"  {f'{(~np.isnan(fg)).sum()}/{len(fg)}':>14s}  {f'{(~np.isnan(fc)).sum()}/{len(fc)}':>14s}")

    # ---- 3. slope-based fire (analog of what worked for σ₂/σ₃)
    print()
    def slope(arr, win=25):
        x = np.log(np.clip(arr, 1e-6, None))
        s = np.full_like(x, np.nan)
        s[win:] = (x[win:] - x[:-win]) / win
        return s

    SLOPE_THRS = [0.02, 0.04, 0.08]
    EP_MINS = [0, 50, 100]
    print(f"{'slope_thr':>10s}  {'ep_min':>8s}  {'grok median':>14s}  {'IQR':>14s}  {'grok n_fired':>14s}  {'ctrl n_fired':>14s}")
    for thr in SLOPE_THRS:
        for em in EP_MINS:
            fg = []
            fc = []
            for r in grok:
                s12 = r["W_sigma1"] / np.maximum(r["W_sigma2"], 1e-30)
                sl = slope(s12)
                ft = fire_time_threshold(np.where(~np.isnan(sl), sl, -np.inf), thr, 1, em)
                fg.append(ft)
            for r in ctrl:
                s12 = r["W_sigma1"] / np.maximum(r["W_sigma2"], 1e-30)
                sl = slope(s12)
                ft = fire_time_threshold(np.where(~np.isnan(sl), sl, -np.inf), thr, 1, em)
                fc.append(ft)
            fg = np.array(fg, dtype=float)
            fc = np.array(fc, dtype=float)
            med = np.nanmedian(fg)
            iqr = (np.nanpercentile(fg, 25), np.nanpercentile(fg, 75))
            print(f"{thr:>10.2f}  {em:>8d}  {med:>14.0f}  {f'[{iqr[0]:.0f},{iqr[1]:.0f}]':>14s}"
                  f"  {f'{(~np.isnan(fg)).sum()}/{len(fg)}':>14s}  {f'{(~np.isnan(fc)).sum()}/{len(fc)}':>14s}")

    # ---- 4. Lag analysis vs test_acc=0.5  (the leading-indicator question)
    print()
    print("--- LAG vs test_acc = 0.5 (negative = leads, positive = lags) ---")
    test_05_grok = []
    for r in grok:
        m = r["test_acc"] >= 0.5
        test_05_grok.append(int(ep[m][0]) if m.any() else float("nan"))
    test_05_grok = np.array(test_05_grok, dtype=float)
    print(f"  test_acc=0.5 median ep: {np.median(test_05_grok):.0f}, IQR=[{np.percentile(test_05_grok,25):.0f}, {np.percentile(test_05_grok,75):.0f}]")

    # σ₁/σ₂ best signal: simple threshold at 25 with sustained=5
    fg25 = []
    fc25 = []
    for r in grok:
        s12 = r["W_sigma1"] / np.maximum(r["W_sigma2"], 1e-30)
        valid = ~np.isnan(r["W_sigma1"])
        fg25.append(fire_time_threshold(np.where(valid, s12, -np.inf), 25.0, 5))
    for r in ctrl:
        s12 = r["W_sigma1"] / np.maximum(r["W_sigma2"], 1e-30)
        valid = ~np.isnan(r["W_sigma1"])
        fc25.append(fire_time_threshold(np.where(valid, s12, -np.inf), 25.0, 5))
    fg25 = np.array(fg25, dtype=float)
    fc25 = np.array(fc25, dtype=float)
    lag = fg25 - test_05_grok
    print(f"  σ₁/σ₂ ≥ 25 sustained 5 — lag vs test=0.5: median={np.nanmedian(lag):.0f} ep, IQR=[{np.nanpercentile(lag,25):.0f}, {np.nanpercentile(lag,75):.0f}]")
    print(f"  ctrl crossings at same threshold: {(~np.isnan(fc25)).sum()}/{len(fc25)}")

    # ---- 5. Headline 3-row overlay
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for runs, color, lbl in [(grok, "C0", "η=0.0002"), (ctrl, "C1", "η=0")]:
        ta = np.stack([r["test_acc"] for r in runs], 0)
        mu = ta.mean(0); sd = ta.std(0)
        axes[0].plot(ep, mu, color=color, lw=1.8, label=lbl)
        axes[0].fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18)
        # σ₁/σ₂
        s12 = np.stack([r["W_sigma1"] / np.maximum(r["W_sigma2"], 1e-30) for r in runs], 0)
        med = np.nanmedian(s12, axis=0)
        q1 = np.nanpercentile(s12, 25, axis=0)
        q3 = np.nanpercentile(s12, 75, axis=0)
        axes[1].plot(ep, med, color=color, lw=1.8, label=lbl)
        axes[1].fill_between(ep, q1, q3, color=color, alpha=0.18)
        # σ₂/σ₃
        s23 = np.stack([r["W_gap23"] for r in runs], 0)
        med = np.nanmedian(s23, axis=0)
        q1 = np.nanpercentile(s23, 25, axis=0)
        q3 = np.nanpercentile(s23, 75, axis=0)
        axes[2].plot(ep, med, color=color, lw=1.8, label=lbl)
        axes[2].fill_between(ep, q1, q3, color=color, alpha=0.18)
    axes[0].set_ylabel("test accuracy"); axes[0].set_title("(A) Test accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel("σ₁/σ₂ (W)"); axes[1].set_title("(B) σ₁/σ₂ on rolling-window ΔW — Stage I→II candidate")
    axes[1].set_yscale("log"); axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    axes[2].set_ylabel("σ₂/σ₃ (W)"); axes[2].set_title("(C) σ₂/σ₃ on rolling-window ΔW — Stage III lock-in")
    axes[2].set_yscale("log"); axes[2].set_xlabel("epoch"); axes[2].legend()
    axes[2].grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(RUNS / "sigma12_headline.png", dpi=120)
    plt.close()
    print("\nsaved sigma12_headline.png")


if __name__ == "__main__":
    main()
