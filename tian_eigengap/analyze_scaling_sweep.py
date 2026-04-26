"""Priority 3 analysis: M × p scaling sweep.

Tests whether fft_dist_from_ideal ≥ 0.075 fire times correlate with
Tian's Theorem 4 boundary (test_acc reaches 1 iff n ≳ M log M).

Outputs:
  runs/scaling_grok_boundary.png   -- (M, p) grokking boundary, color = grok rate
  runs/scaling_fft_overlay.png     -- (M, p) plane colored by median lead time
  text summary across cells
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = Path(__file__).parent / "runs"
MS = [41, 71, 127]
PS = [0.1, 0.2, 0.3, 0.5]
SEEDS = [0, 1, 2, 3, 4]
FFT_THRESHOLD = 0.075


def expected_epochs(M: int) -> int:
    return {41: 400, 71: 600, 127: 1000}[M]


def load_run(M: int, p: float, seed: int) -> dict | None:
    tag = f"scal_M{M}_p{p}_seed{seed}"
    p_path = RUNS / tag / "log.jsonl"
    if not p_path.exists():
        return None
    rows = [json.loads(l) for l in open(p_path)]
    if len(rows) < expected_epochs(M):
        return None
    keys = sorted({k for r in rows for k in r.keys()})
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def fire(x: np.ndarray, thr: float) -> float:
    c = np.where(x >= thr)[0]
    return int(c[0]) if len(c) else float("nan")


def main():
    cells = []
    for M in MS:
        for p in PS:
            seeds = []
            for s in SEEDS:
                d = load_run(M, p, s)
                if d is None:
                    continue
                test = d["test_acc"]
                t_test05 = fire(test, 0.5)
                t_test99 = fire(test, 0.99)
                t_fft = fire(d["fft_dist_from_ideal"], FFT_THRESHOLD)
                fft_max = float(np.nanmax(d["fft_dist_from_ideal"]))
                seeds.append({
                    "seed": s, "groks": (~np.isnan(t_test99)),
                    "t_test05": t_test05, "t_test99": t_test99,
                    "t_fft": t_fft, "fft_max": fft_max,
                    "lead": (t_test05 - t_fft) if not (np.isnan(t_test05) or np.isnan(t_fft)) else float("nan"),
                })
            if seeds:
                grok_rate = sum(s["groks"] for s in seeds) / len(seeds)
                fft_fire_rate = sum(not np.isnan(s["t_fft"]) for s in seeds) / len(seeds)
                grok_t99 = [s["t_test99"] for s in seeds if s["groks"]]
                fft_lead = [s["lead"] for s in seeds if s["groks"] and not np.isnan(s["lead"])]
                cells.append({
                    "M": M, "p": p, "n_seeds": len(seeds),
                    "grok_rate": grok_rate, "fft_fire_rate": fft_fire_rate,
                    "median_t_test99": np.median(grok_t99) if grok_t99 else float("nan"),
                    "median_t_fft": np.nanmedian([s["t_fft"] for s in seeds]),
                    "median_lead": np.median(fft_lead) if fft_lead else float("nan"),
                    "seeds": seeds,
                })
    print(f"loaded {len(cells)} cells with at least one completed run")

    # Text summary
    print(f"\n{'M':>4s}  {'p':>5s}  {'n':>3s}  {'grok_rate':>10s}  {'fft_fire':>10s}  {'med_test99':>12s}  {'med_fft_fire':>14s}  {'med_lead':>10s}")
    for c in cells:
        print(f"{c['M']:>4d}  {c['p']:>5.2f}  {c['n_seeds']:>3d}  {c['grok_rate']:>10.2f}  "
              f"{c['fft_fire_rate']:>10.2f}  {c['median_t_test99']:>12.0f}  "
              f"{c['median_t_fft']:>14.0f}  {c['median_lead']:>+10.0f}")

    # Also print Tian's predicted critical p_crit ~ log(M)/M
    print("\nTian Theorem 4 predicted p_crit ~ log(M)/M (training ratio above = grok):")
    for M in MS:
        print(f"  M={M}: p_crit ≈ {math.log(M)/M:.3f}")

    # ---- plots ----
    if not cells:
        return
    if len(cells) < len(MS) * len(PS):
        print(f"\n(partial sweep: {len(cells)}/{len(MS)*len(PS)} cells. Plotting what we have.)")

    # Build matrices
    grok_mat = np.full((len(MS), len(PS)), np.nan)
    lead_mat = np.full((len(MS), len(PS)), np.nan)
    fft_fire_mat = np.full((len(MS), len(PS)), np.nan)
    test_mat = np.full((len(MS), len(PS)), np.nan)
    for c in cells:
        i = MS.index(c["M"])
        j = PS.index(c["p"])
        grok_mat[i, j] = c["grok_rate"]
        lead_mat[i, j] = c["median_lead"]
        fft_fire_mat[i, j] = c["fft_fire_rate"]
        test_mat[i, j] = c["median_t_test99"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: grok rate heatmap
    ax = axes[0]
    im = ax.imshow(grok_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(PS))); ax.set_xticklabels([f"{p:g}" for p in PS])
    ax.set_yticks(range(len(MS))); ax.set_yticklabels(MS)
    ax.set_xlabel("p (training fraction)")
    ax.set_ylabel("M")
    ax.set_title("(A) Grok rate (fraction of seeds reaching test=0.99)")
    for i in range(len(MS)):
        for j in range(len(PS)):
            if not np.isnan(grok_mat[i, j]):
                ax.text(j, i, f"{grok_mat[i, j]:.2f}", ha="center", va="center",
                        color="black" if grok_mat[i, j] > 0.4 else "white", fontsize=10)
    plt.colorbar(im, ax=ax, label="grok rate")
    # Tian predicted boundary
    for i, M in enumerate(MS):
        p_crit = math.log(M) / M
        if p_crit < PS[0]:
            x_pos = -0.5
        elif p_crit > PS[-1]:
            x_pos = len(PS) - 0.5
        else:
            # interpolate position in PS index space
            for jj in range(len(PS) - 1):
                if PS[jj] <= p_crit <= PS[jj + 1]:
                    x_pos = jj + (p_crit - PS[jj]) / (PS[jj + 1] - PS[jj])
                    break
        ax.axvline(x_pos, ymin=(i - 0.5 + 0.5) / len(MS) - 0.3 / len(MS),
                   ymax=(i + 0.5 + 0.5) / len(MS) + 0.3 / len(MS),
                   color="blue", lw=2, alpha=0.5)
    # actually a simpler boundary line: plot p_crit per M as scatter
    for i, M in enumerate(MS):
        p_crit = math.log(M) / M
        if PS[0] <= p_crit <= PS[-1]:
            for jj in range(len(PS) - 1):
                if PS[jj] <= p_crit <= PS[jj + 1]:
                    x = jj + (p_crit - PS[jj]) / (PS[jj + 1] - PS[jj])
                    ax.scatter(x, i, marker="*", s=160, color="blue",
                               edgecolor="white", lw=1, zorder=10,
                               label="Tian Thm 4 p_crit" if i == 0 else None)
                    break
    ax.legend(loc="upper left")

    # Panel 2: fft fire rate heatmap
    ax = axes[1]
    im = ax.imshow(fft_fire_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(PS))); ax.set_xticklabels([f"{p:g}" for p in PS])
    ax.set_yticks(range(len(MS))); ax.set_yticklabels(MS)
    ax.set_xlabel("p"); ax.set_ylabel("M")
    ax.set_title("(B) fft fire rate (fraction of seeds with fft ≥ 0.075)")
    for i in range(len(MS)):
        for j in range(len(PS)):
            if not np.isnan(fft_fire_mat[i, j]):
                ax.text(j, i, f"{fft_fire_mat[i, j]:.2f}", ha="center", va="center",
                        color="black" if fft_fire_mat[i, j] > 0.4 else "white", fontsize=10)
    plt.colorbar(im, ax=ax, label="fft fire rate")

    # Panel 3: median lead time
    ax = axes[2]
    im = ax.imshow(lead_mat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(PS))); ax.set_xticklabels([f"{p:g}" for p in PS])
    ax.set_yticks(range(len(MS))); ax.set_yticklabels(MS)
    ax.set_xlabel("p"); ax.set_ylabel("M")
    ax.set_title("(C) Median lead time (test=0.5 − fft fire) [ep]")
    for i in range(len(MS)):
        for j in range(len(PS)):
            if not np.isnan(lead_mat[i, j]):
                ax.text(j, i, f"{lead_mat[i, j]:.0f}", ha="center", va="center",
                        color="white", fontsize=10)
    plt.colorbar(im, ax=ax, label="lead [ep]")

    plt.tight_layout()
    out = RUNS / "scaling_boundary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
