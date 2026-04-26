"""Analyze the η sweep — does fft_dist_from_ideal lead test_acc consistently?

For each (η, seed) run, compute:
  - test_acc fire time (first ep ≥ 0.5)
  - fft_dist_from_ideal fire time (threshold 0.075 — the value from the
    primary M=71, η=2e-4 finding)
  - σ₂/σ₃ slope-fire ep (Stage III lock-in)
  - Lead/lag relative to test_acc

Headline plot: scatter (test_acc fire ep) vs (fft fire ep), colored by η.
  Outcome A: fft fires near constant ep regardless of η (horizontal scatter).
  Outcome B: slope < 1 (fft tracks fraction of grok time).
  Outcome C: slope = 1 with negative intercept (consistent ~80 ep lead).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = Path(__file__).parent / "runs"
ETA_VALS = ["0.00001", "0.00005", "0.0001", "0.0002", "0.0005"]
SEEDS = [0, 1, 2, 3, 4]
FFT_THRESHOLD = 0.075
SIGMA23_SLOPE_THR = 0.04
SIGMA23_EP_MIN = 100


def load_run(eta_str: str, seed: int, min_rows: int = 601) -> dict | None:
    p = RUNS / f"eta{eta_str}_seed{seed}" / "log.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    if len(rows) < min_rows:
        return None
    keys = sorted({k for r in rows for k in r.keys()})
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def fire_threshold(x: np.ndarray, thr: float) -> float:
    c = np.where(x >= thr)[0]
    return int(c[0]) if len(c) else float("nan")


def slope_fire(x: np.ndarray, win: int, slope_thr: float, ep_min: int) -> float:
    lx = np.log(np.clip(x, 1e-6, None))
    sl = np.full_like(lx, np.nan)
    sl[win:] = (lx[win:] - lx[:-win]) / win
    valid = (~np.isnan(sl)) & (np.arange(len(sl)) >= ep_min)
    c = np.where((sl > slope_thr) & valid)[0]
    return int(c[0]) if len(c) else float("nan")


def main():
    rows = []
    for eta in ETA_VALS:
        for s in SEEDS:
            d = load_run(eta, s)
            if d is None:
                continue
            ep = d["epoch"]
            t_test = float(np.where(d["test_acc"] >= 0.5)[0][0]) if (d["test_acc"] >= 0.5).any() else float("nan")
            t_test99 = float(np.where(d["test_acc"] >= 0.99)[0][0]) if (d["test_acc"] >= 0.99).any() else float("nan")
            t_fft = fire_threshold(d["fft_dist_from_ideal"], FFT_THRESHOLD)
            t_lock = slope_fire(d["W_gap23"], 25, SIGMA23_SLOPE_THR, SIGMA23_EP_MIN)
            rows.append({
                "eta": float(eta), "eta_str": eta, "seed": s,
                "t_test_05": t_test, "t_test_099": t_test99,
                "t_fft": t_fft, "t_lock": t_lock,
                "groks": (~np.isnan(t_test)) and (~np.isnan(t_test99)),
                "late_sigma23": float(np.nanmedian(d["W_gap23"][300:])),
                "fft_max": float(np.nanmax(d["fft_dist_from_ideal"])),
            })
    print(f"loaded {len(rows)} runs")

    # ---- text summary ----
    print(f"\n{'eta':>10s}  {'n':>3s}  {'n_grok':>7s}  {'fft_fire':>10s}  {'test_05':>10s}  {'lead':>8s}  "
          f"{'lock':>6s}  {'lag':>6s}  {'late σ₂/σ₃':>14s}  {'fft_max':>9s}")
    for eta in ETA_VALS:
        eta_rows = [r for r in rows if r["eta_str"] == eta]
        if not eta_rows:
            continue
        n_grok = sum(r["groks"] for r in eta_rows)
        fft_med = np.nanmedian([r["t_fft"] for r in eta_rows])
        test_med = np.nanmedian([r["t_test_05"] for r in eta_rows])
        lead = (test_med - fft_med) if not (np.isnan(test_med) or np.isnan(fft_med)) else float("nan")
        lock_med = np.nanmedian([r["t_lock"] for r in eta_rows])
        lag = (lock_med - test_med) if not (np.isnan(test_med) or np.isnan(lock_med)) else float("nan")
        sig_med = np.nanmedian([r["late_sigma23"] for r in eta_rows])
        fft_max_med = np.nanmedian([r["fft_max"] for r in eta_rows])
        print(f"{eta:>10s}  {len(eta_rows):>3d}  {n_grok:>7d}  {fft_med:>10.0f}  {test_med:>10.0f}  "
              f"{lead:>+8.0f}  {lock_med:>6.0f}  {lag:>+6.0f}  {sig_med:>14.1f}  {fft_max_med:>9.4f}")

    # ---- main scatter plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis
    eta_floats = [float(e) for e in ETA_VALS]
    log_eta_min, log_eta_max = np.log10(min(eta_floats)), np.log10(max(eta_floats))

    ax = axes[0]
    for r in rows:
        if not r["groks"]:
            ax.scatter(r["t_test_05"], r["t_fft"], color="gray", marker="x", s=40, alpha=0.5)
            continue
        c = cmap((np.log10(r["eta"]) - log_eta_min) / (log_eta_max - log_eta_min + 1e-9))
        ax.scatter(r["t_test_05"], r["t_fft"], color=c, s=70, alpha=0.85, edgecolor="k", lw=0.5)
    # reference lines
    diag = np.array([0, 700])
    ax.plot(diag, diag, "k--", lw=0.6, label="y = x (slope 1, no lead)")
    ax.plot(diag, diag - 84, "g--", lw=0.6, label="y = x − 84 (constant 84-ep lead)")
    ax.set_xlabel("test_acc=0.5 fire epoch")
    ax.set_ylabel("fft_dist≥0.075 fire epoch")
    ax.set_title("(A) Lead-time scatter — colored by log η")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in rows:
        if not r["groks"]:
            continue
        c = cmap((np.log10(r["eta"]) - log_eta_min) / (log_eta_max - log_eta_min + 1e-9))
        ax.scatter(r["t_test_05"], r["t_test_05"] - r["t_fft"], color=c, s=70,
                   alpha=0.85, edgecolor="k", lw=0.5)
    ax.axhline(84, ls="--", color="g", lw=0.6, label="84-ep lead (M=71, η=2e-4 baseline)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("test_acc=0.5 fire epoch")
    ax.set_ylabel("lead time (test − fft) [ep]")
    ax.set_title("(B) Lead time vs grok time — colored by log η")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=log_eta_min, vmax=log_eta_max))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, location="right", shrink=0.6)
    cb.set_label("log₁₀ η")
    cb.set_ticks(np.log10(eta_floats))
    cb.set_ticklabels([f"{e:g}" for e in eta_floats])

    out = RUNS / "eta_sweep_scatter.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")
    plt.close()

    # ---- regression to classify outcome ----
    grok_rows = [r for r in rows if r["groks"] and not np.isnan(r["t_fft"])]
    if len(grok_rows) >= 5:
        x = np.array([r["t_test_05"] for r in grok_rows], dtype=float)
        y = np.array([r["t_fft"] for r in grok_rows], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-30)
        print(f"\nlinear regression: t_fft = {slope:.3f} × t_test + {intercept:.1f}  R²={r2:.3f}")
        if abs(slope) < 0.1:
            outcome = "A — fft fires at near-constant epoch independent of η (Stage I detector, NOT predictive)"
        elif slope > 0.7 and slope < 1.3:
            outcome = "C — slope ≈ 1 with intercept ≈ {:.0f} (consistent lead time, predictive)".format(intercept)
        else:
            outcome = "B — slope between A and C (fft tracks training-time fraction)"
        print(f"OUTCOME: {outcome}")


if __name__ == "__main__":
    main()
