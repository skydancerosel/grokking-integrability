"""Cross-seed analysis of the Tian eigengap sweep.

Loads runs/sweep_eta*_seed*/log.jsonl, computes:
  - Fig 3 replication with seed-mean ± std
  - 2-panel overlay (Tian metric + σ₂/σ₃) for both conditions
  - Fire-time lag distribution: σ₂/σ₃ vs Tian metric vs test-acc transition
  - Late-stage σ₂/σ₃ ratio between grok and no-grok (the strongest discriminator
    we saw on the single seed)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = Path(__file__).parent / "runs"


def load_run(tag: str) -> dict[str, np.ndarray]:
    p = RUNS_DIR / tag / "log.jsonl"
    rows = [json.loads(line) for line in open(p)]
    keys = sorted({k for r in rows for k in r.keys()})
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def load_condition(eta_str: str, seeds: Iterable[int], min_rows: int = 401):
    runs = []
    for s in seeds:
        tag = f"sweep_eta{eta_str}_seed{s}"
        p = RUNS_DIR / tag / "log.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            n_rows = sum(1 for _ in f)
        if n_rows < min_rows:
            continue
        runs.append(load_run(tag))
    return runs


def stack(runs, key):
    return np.stack([r[key] for r in runs], axis=0)


def fire_time(x: np.ndarray, baseline_window: slice = slice(None, 10),
              search_window: slice | None = None,
              fraction: float = 0.5,
              abs_threshold: float | None = None) -> int | float:
    """Find first epoch where x crosses a threshold.

    If abs_threshold is given, use it directly; otherwise threshold =
    baseline + fraction * (peak - baseline) computed within search_window.
    Returns NaN if never crosses."""
    s = x if search_window is None else x[search_window]
    valid = ~np.isnan(x)
    if abs_threshold is not None:
        thr = abs_threshold
    else:
        base = np.nanmedian(x[baseline_window]) if np.any(valid[baseline_window]) else 0.0
        peak = np.nanmax(s)
        thr = base + fraction * (peak - base)
    crossings = np.where((x >= thr) & valid)[0]
    if search_window is not None:
        lo = search_window.start or 0
        hi = search_window.stop or len(x)
        crossings = crossings[(crossings >= lo) & (crossings < hi)]
    return int(crossings[0]) if len(crossings) else float("nan")


def main():
    seeds = list(range(15))
    grok = load_condition("0.0002", seeds)
    nogrok = load_condition("0", seeds)
    print(f"loaded grok={len(grok)} runs, nogrok={len(nogrok)} runs")
    if not grok:
        print("no runs yet; exiting")
        return

    # ------ Fig 3 mean+std replication ------
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for col, (runs, title) in enumerate([(grok, "η = 0.0002 (grok)"),
                                         (nogrok, "η = 0 (control)")]):
        if not runs:
            continue
        ep = runs[0]["epoch"]
        # accuracy
        ax = axes[0, col]
        for key, color, lbl in [("train_acc", "C0", "train"), ("test_acc", "C1", "test")]:
            arr = stack(runs, key)
            mu, sd = arr.mean(0), arr.std(0)
            ax.plot(ep, mu, color=color, label=lbl, lw=1.5)
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.set_ylabel("accuracy")
        ax.set_title(f"{title}  (n={len(runs)})")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="center right", fontsize=8)
        ax.grid(alpha=0.3)

        # Tian metric (~F^T~F off/diag and FF^T dist from ideal)
        ax = axes[1, col]
        for key, color, lbl in [("ftf_ratio", "C0", "~F^T~F off/diag"),
                                ("fft_dist_from_ideal", "C2", "P_1⊥ FF^T dist")]:
            arr = stack(runs, key)
            mu, sd = arr.mean(0), arr.std(0)
            ax.plot(ep, mu, color=color, label=lbl, lw=1.5)
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.axhline(0.08, ls="--", color="gray", lw=0.7, label="8% bound")
        ax.set_ylabel("Tian metric")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # |G_F|
        ax = axes[2, col]
        arr = stack(runs, "gF_norm")
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(ep, mu, color="C3", lw=1.5)
        ax.fill_between(ep, mu - sd, mu + sd, color="C3", alpha=0.2)
        ax.set_ylabel("‖G_F‖")
        ax.grid(alpha=0.3)

        # σ₂/σ₃ rolling-window eigengap
        ax = axes[3, col]
        if "W_gap23" in runs[0]:
            arr = stack(runs, "W_gap23")
            mu = np.nanmedian(arr, axis=0)
            q1 = np.nanpercentile(arr, 25, axis=0)
            q3 = np.nanpercentile(arr, 75, axis=0)
            ax.plot(ep, mu, color="C4", lw=1.5, label="σ₂/σ₃ (W)")
            ax.fill_between(ep, q1, q3, color="C4", alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("σ₂/σ₃")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RUNS_DIR / "fig3_replication_seedmean.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
    plt.close()

    # ------ fire-time lag distribution ------
    if grok:
        # Threshold for σ₂/σ₃: we want a level the η=0 control never reaches.
        # The control's σ₂/σ₃ rarely exceeds ~35, so use 50 as a hard threshold.
        gap_threshold = 50.0
        rows = []
        for r in grok:
            ep = r["epoch"]
            n = len(ep)
            t_test_05 = fire_time(r["test_acc"], slice(0, 5), slice(0, n), 0.5)
            # σ₂/σ₃: fire = first time it exceeds gap_threshold (above noise floor)
            t_gap = fire_time(r["W_gap23"], abs_threshold=gap_threshold)
            # fft_dist: 50% of peak in epochs 0..200 (the early rise)
            t_dist = fire_time(r["fft_dist_from_ideal"], slice(0, 5), slice(0, 200), 0.5)
            t_gF = fire_time(r["gF_norm"], slice(0, 5), slice(0, 200), 0.5)
            rows.append((t_gap, t_dist, t_gF, t_test_05))
        rows = np.array(rows, dtype=float)
        labels = [f"σ₂/σ₃ ≥ {gap_threshold:g}", "fft_dist_from_ideal (50% peak)",
                  "‖G_F‖ (50% peak)", "test_acc ≥ 0.5"]
        plt.figure(figsize=(8, 4))
        for i, lbl in enumerate(labels):
            data = rows[:, i]
            data = data[~np.isnan(data)]
            plt.scatter([i] * len(data), data, alpha=0.6)
        plt.xticks(range(len(labels)), labels, rotation=15)
        plt.ylabel("epoch")
        plt.title("Fire times across seeds (η=0.0002)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out = RUNS_DIR / "fire_time_distribution.png"
        plt.savefig(out, dpi=120)
        print(f"saved {out}")
        plt.close()

        # lag statistics
        print("\n--- fire-time stats (η=0.0002) ---")
        for i, lbl in enumerate(labels):
            data = rows[:, i]
            data = data[~np.isnan(data)]
            print(f"  {lbl:32s}: median={np.median(data):.0f}  IQR=[{np.percentile(data,25):.0f}, {np.percentile(data,75):.0f}]  n={len(data)}")
        print()
        # σ₂/σ₃ vs other signals -- lag = t(test) - t(gap)
        gap_to_test = rows[:, 3] - rows[:, 0]
        gap_to_test = gap_to_test[~np.isnan(gap_to_test)]
        print(f"  test_acc=0.5 - σ₂/σ₃ fire: median={np.median(gap_to_test):.0f}, IQR=[{np.percentile(gap_to_test,25):.0f}, {np.percentile(gap_to_test,75):.0f}]")

    # ------ slope-based σ₂/σ₃ detector (alternate signal) ------
    if grok and nogrok:
        # Compute log(σ₂/σ₃) slope via 25-epoch backward difference, look for first
        # sustained large positive slope (the "leap" event).
        def slope_signal(r, win=25):
            x = np.log(np.clip(r["W_gap23"], 1e-6, None))
            slope = np.full_like(x, np.nan)
            slope[win:] = (x[win:] - x[:-win]) / win
            return slope

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
        for ax, runs, title in [(axes[0], grok, "η=0.0002 (grok)"),
                                (axes[1], nogrok, "η=0 (control)")]:
            ep = runs[0]["epoch"]
            slopes = np.stack([slope_signal(r) for r in runs], 0)
            mu = np.nanmedian(slopes, axis=0)
            q1 = np.nanpercentile(slopes, 25, axis=0)
            q3 = np.nanpercentile(slopes, 75, axis=0)
            ax.plot(ep, mu, lw=1.5, color="C0")
            ax.fill_between(ep, q1, q3, color="C0", alpha=0.2)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("epoch")
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("d log(σ₂/σ₃) / dt   (25-epoch window)")
        plt.tight_layout()
        out = RUNS_DIR / "eigengap_slope.png"
        plt.savefig(out, dpi=120)
        print(f"saved {out}")
        plt.close()

        # late-window slope detector: positive slope after epoch 100 = "Stage III leap"
        fire_slope_thr = 0.04
        slope_fires_grok = []
        slope_fires_nogrok = []
        late = lambda ep: ep >= 100
        for r in grok:
            sl = slope_signal(r)
            ep = r["epoch"]
            mask = (sl > fire_slope_thr) & (~np.isnan(sl)) & late(ep)
            slope_fires_grok.append(int(ep[mask][0]) if mask.any() else float("nan"))
        for r in nogrok:
            sl = slope_signal(r)
            ep = r["epoch"]
            mask = (sl > fire_slope_thr) & (~np.isnan(sl)) & late(ep)
            slope_fires_nogrok.append(int(ep[mask][0]) if mask.any() else float("nan"))
        sf_g = np.array(slope_fires_grok, dtype=float)
        sf_n = np.array(slope_fires_nogrok, dtype=float)
        print(f"\n--- late-window slope-based σ₂/σ₃ fire (slope > {fire_slope_thr} for ep≥100) ---")
        print(f"  η=0.0002: median={np.nanmedian(sf_g):.0f}, IQR=[{np.nanpercentile(sf_g,25):.0f}, {np.nanpercentile(sf_g,75):.0f}], n_fired={(~np.isnan(sf_g)).sum()}/{len(sf_g)}")
        print(f"  η=0     : n_fired={(~np.isnan(sf_n)).sum()}/{len(sf_n)} (median={np.nanmedian(sf_n):.0f} if any)")
        # lag of late-slope fire vs test_acc=0.99 (full grokking)
        if (~np.isnan(sf_g)).all() and len(sf_g) > 0:
            test_99_grok = []
            for r in grok:
                ep = r["epoch"]
                m = (r["test_acc"] >= 0.99)
                test_99_grok.append(int(ep[m][0]) if m.any() else float("nan"))
            test_99_grok = np.array(test_99_grok, dtype=float)
            lag = sf_g - test_99_grok
            print(f"  σ₂/σ₃ slope-fire − test_acc=0.99: median lag={np.nanmedian(lag):.0f} epochs, IQR=[{np.nanpercentile(lag,25):.0f}, {np.nanpercentile(lag,75):.0f}]")

    # ------ late-stage σ₂/σ₃ ratio (the cleanest discriminator) ------
    if grok and nogrok:
        late_grok = []
        late_no = []
        for r in grok:
            mask = (r["epoch"] >= 200) & (r["epoch"] <= 400)
            v = r["W_gap23"][mask]
            late_grok.append(np.nanmedian(v))
        for r in nogrok:
            mask = (r["epoch"] >= 200) & (r["epoch"] <= 400)
            v = r["W_gap23"][mask]
            late_no.append(np.nanmedian(v))
        late_grok = np.array(late_grok)
        late_no = np.array(late_no)
        plt.figure(figsize=(6, 4))
        plt.boxplot([late_no, late_grok], labels=["η=0", "η=0.0002"], showfliers=False)
        plt.scatter(np.ones_like(late_no), late_no, alpha=0.6, color="C1")
        plt.scatter(np.ones_like(late_grok) * 2, late_grok, alpha=0.6, color="C0")
        plt.ylabel("median σ₂/σ₃ over epochs 200–400 (W)")
        plt.yscale("log")
        plt.title("Late-stage eigengap separation")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        out = RUNS_DIR / "late_eigengap_separation.png"
        plt.savefig(out, dpi=120)
        print(f"saved {out}")
        plt.close()
        print(f"\n--- late σ₂/σ₃ medians (epochs 200-400) ---")
        print(f"  η=0     : median={np.median(late_no):.2f}, range=[{late_no.min():.2f}, {late_no.max():.2f}]")
        print(f"  η=0.0002: median={np.median(late_grok):.2f}, range=[{late_grok.min():.2f}, {late_grok.max():.2f}]")
        print(f"  ratio (grok/control medians): {np.median(late_grok)/np.median(late_no):.1f}x")


if __name__ == "__main__":
    main()
