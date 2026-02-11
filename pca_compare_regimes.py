#!/usr/bin/env python3
"""
Compare PC1% between two hyperparameter regimes for (a+b) mod p:

  Regime A (slow):  lr=5e-5, wd=0.1, 3 layers, ~300 snapshots, grok at ~570k steps
  Regime B (Power): lr=1e-3, wd=1.0, 2 layers, ~30 snapshots,  grok at ~3k steps

Produces:
  - Side-by-side bar chart of PC1% per weight matrix
  - Summary table
  - Subsampled comparison (regime A downsampled to match B's snapshot count)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SLOW_DIR = Path(__file__).parent / "grok_sweep_results_slow"
FAST_DIR = Path(__file__).parent / "grok_sweep_results"
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
COLORS_WK = {"WQ": "#1f77b4", "WK": "#ff7f0e", "WV": "#2ca02c", "WO": "#d62728"}
SEEDS = [42, 137, 2024]


def collect_trajectory(attn_logs, layer_idx, key):
    steps, mats = [], []
    for snap in attn_logs:
        steps.append(snap["step"])
        mats.append(snap["layers"][layer_idx][key].float())
    return np.array(steps), mats


def pca_pc1(mats):
    """Return PC1 explained variance ratio."""
    if len(mats) < 3:
        return None
    W0 = mats[0].reshape(-1).numpy()
    X = np.stack([m.reshape(-1).numpy() - W0 for m in mats])
    X -= X.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    eig = (S ** 2) / (X.shape[0] - 1)
    total = eig.sum()
    if total < 1e-15:
        return None
    return float(eig[0] / total) * 100


def subsample_mats(mats, target_n):
    """Subsample a list of matrices to target_n evenly-spaced entries."""
    T = len(mats)
    if T <= target_n:
        return mats
    idx = np.linspace(0, T - 1, target_n, dtype=int)
    return [mats[i] for i in idx]


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Load all runs
    slow_runs = {}  # (wd, seed) -> data
    fast_runs = {}

    for pt in sorted(SLOW_DIR.glob("add_wd*_slow.pt")):
        data = torch.load(pt, map_location="cpu", weights_only=False)
        cfg = data["cfg"]
        slow_runs[(cfg["WEIGHT_DECAY"], cfg["SEED"])] = data

    for pt in sorted(FAST_DIR.glob("add_wd*_s*.pt")):
        data = torch.load(pt, map_location="cpu", weights_only=False)
        cfg = data["cfg"]
        fast_runs[(cfg["WEIGHT_DECAY"], cfg["SEED"])] = data

    print(f"Loaded {len(slow_runs)} slow runs, {len(fast_runs)} fast runs")

    # ── Compute PC1% for all configurations ────────────────────────────────
    # For each regime: full PCA, and subsampled PCA (slow → 30 snaps)

    results = {}  # (regime, wd, seed, layer, wkey, variant) -> pc1%
    # variant: "full", "sub30"

    for (wd, seed), data in slow_runs.items():
        n_layers = data["cfg"]["N_LAYERS"]
        logs = data["attn_logs"]
        for li in range(n_layers):
            for wkey in WEIGHT_KEYS:
                _, mats = collect_trajectory(logs, li, wkey)
                pc1 = pca_pc1(mats)
                if pc1 is not None:
                    results[("slow", wd, seed, li, wkey, "full")] = pc1
                # subsampled to ~30
                sub_mats = subsample_mats(mats, 33)
                pc1_sub = pca_pc1(sub_mats)
                if pc1_sub is not None:
                    results[("slow", wd, seed, li, wkey, "sub30")] = pc1_sub

    for (wd, seed), data in fast_runs.items():
        if data["cfg"]["OP_NAME"] != "add":
            continue
        n_layers = data["cfg"]["N_LAYERS"]
        logs = data["attn_logs"]
        for li in range(n_layers):
            for wkey in WEIGHT_KEYS:
                _, mats = collect_trajectory(logs, li, wkey)
                pc1 = pca_pc1(mats)
                if pc1 is not None:
                    results[("fast", wd, seed, li, wkey, "full")] = pc1

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  REGIME COMPARISON: (a+b) mod 97")
    print("=" * 100)

    # Last layer for each regime
    slow_nlayers = 3
    fast_nlayers = 2

    for regime, nlayers, wd_grok, label in [
        ("slow", slow_nlayers, 0.1, "Slow (lr=5e-5, wd=0.1, 3L)"),
        ("fast", fast_nlayers, 1.0, "Fast (lr=1e-3, wd=1.0, 2L)"),
    ]:
        print(f"\n  {label}:")
        print(f"  {'wd':>4s}  {'seed':>5s}  {'layer':>5s}  {'var':>5s}  ", end="")
        for wkey in WEIGHT_KEYS:
            print(f"{wkey:>8s}", end="  ")
        print("  mean")

        for wd in [wd_grok, 0.0]:
            for seed in SEEDS:
                for li in range(nlayers):
                    for variant in (["full", "sub30"] if regime == "slow" else ["full"]):
                        pc1s = []
                        tag = f"{'grok' if wd == wd_grok else 'nowd'}"
                        print(f"  {wd:4.1f}  {seed:5d}  L{li:4d}  {variant:>5s}  ", end="")
                        for wkey in WEIGHT_KEYS:
                            k = (regime, wd, seed, li, wkey, variant)
                            if k in results:
                                v = results[k]
                                pc1s.append(v)
                                print(f"{v:7.1f}%", end="  ")
                            else:
                                print(f"{'N/A':>8s}", end="  ")
                        mean = np.mean(pc1s) if pc1s else 0
                        print(f"  {mean:5.1f}%")

    # ── Focused comparison: grok runs, last layer ──────────────────────────
    print("\n" + "=" * 100)
    print("  KEY COMPARISON — Grokking runs, last layer of each regime")
    print("=" * 100)
    print(f"\n  {'Config':>40s}  ", end="")
    for wkey in WEIGHT_KEYS:
        print(f"{wkey:>8s}", end="  ")
    print("  mean")
    print(f"  {'─'*40}  ", end="")
    for _ in WEIGHT_KEYS:
        print(f"{'─'*8}", end="  ")
    print(f"  {'─'*6}")

    configs = [
        ("Slow full (lr=5e-5,wd=0.1,3L,~290sn)", "slow", 0.1, slow_nlayers - 1, "full"),
        ("Slow sub30 (same, 30 snapshots)", "slow", 0.1, slow_nlayers - 1, "sub30"),
        ("Fast full (lr=1e-3,wd=1.0,2L,~30sn)", "fast", 1.0, fast_nlayers - 1, "full"),
        ("Slow no-wd (lr=5e-5,wd=0,3L)", "slow", 0.0, slow_nlayers - 1, "full"),
        ("Fast no-wd (lr=1e-3,wd=0,2L)", "fast", 0.0, fast_nlayers - 1, "full"),
    ]

    bar_data = {}  # config_label -> {wkey: (mean, std)}

    for label, regime, wd, li, variant in configs:
        print(f"  {label:>40s}  ", end="")
        wkey_means = []
        for wkey in WEIGHT_KEYS:
            vals = []
            for seed in SEEDS:
                k = (regime, wd, seed, li, wkey, variant)
                if k in results:
                    vals.append(results[k])
            mean = np.mean(vals) if vals else 0
            std = np.std(vals) if len(vals) > 1 else 0
            wkey_means.append(mean)
            bar_data.setdefault(label, {})[wkey] = (mean, std)
            print(f"{mean:7.1f}%", end="  ")
        overall = np.mean(wkey_means) if wkey_means else 0
        print(f"  {overall:5.1f}%")

    # ── Figure: side-by-side comparison ────────────────────────────────────
    fig, axes = plt.subplots(1, len(WEIGHT_KEYS), figsize=(5 * len(WEIGHT_KEYS), 5.5), squeeze=False)

    config_labels = [c[0] for c in configs[:3]]  # grok configs only
    colors = ["#1a5276", "#2e86c1", "#85c1e9"]
    short_labels = [
        "lr=5e-5, wd=0.1\n3L, ~290 snaps",
        "lr=5e-5, wd=0.1\n3L, 30 snaps (sub)",
        "lr=1e-3, wd=1.0\n2L, ~30 snaps",
    ]

    for wi, wkey in enumerate(WEIGHT_KEYS):
        ax = axes[0, wi]
        x = np.arange(len(config_labels))
        means = [bar_data[c][wkey][0] for c in config_labels]
        stds = [bar_data[c][wkey][1] for c in config_labels]
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, width=0.6)
        ax.set_ylabel("PC1 (%)")
        ax.set_title(wkey, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8, ha="center")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        # annotate
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{m:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("PC1% Comparison: Slow vs Fast Hyperparameters — (a+b) mod 97, Grokking Runs\n(last layer, mean over 3 seeds)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    out_path = OUT_DIR / "figH_regime_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {out_path.name}")

    # ── Figure: decomposition of the PC1% drop ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    wkey_order = WEIGHT_KEYS
    x = np.arange(len(wkey_order))
    width = 0.22

    for ci, (label, color, short) in enumerate([
        (config_labels[0], "#1a5276", "Full (wd=0.1, 3L, 290sn)"),
        (config_labels[1], "#2e86c1", "Subsampled (wd=0.1, 3L, 30sn)"),
        (config_labels[2], "#85c1e9", "Power (wd=1.0, 2L, 30sn)"),
    ]):
        means = [bar_data[label][wk][0] for wk in wkey_order]
        stds = [bar_data[label][wk][1] for wk in wkey_order]
        ax.bar(x + ci * width, means, width, yerr=stds, label=short,
               color=color, alpha=0.85, capsize=3)

    ax.set_ylabel("PC1 explained variance (%)", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(wkey_order, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Decomposing the PC1% Drop: Snapshot Count vs Hyperparameters\n(a+b) mod 97, grokking runs, last layer",
                 fontsize=13)

    # Add annotations for the drops
    for wi, wk in enumerate(wkey_order):
        full = bar_data[config_labels[0]][wk][0]
        sub = bar_data[config_labels[1]][wk][0]
        power = bar_data[config_labels[2]][wk][0]
        snap_drop = full - sub
        hp_drop = sub - power
        total_drop = full - power

        ax.annotate(f"snap: −{snap_drop:.0f}pp",
                    xy=(wi + 0.5 * width, (full + sub) / 2),
                    fontsize=7, ha="center", color="#666")
        ax.annotate(f"hp: −{hp_drop:.0f}pp",
                    xy=(wi + 1.5 * width, (sub + power) / 2),
                    fontsize=7, ha="center", color="#666")

    fig.tight_layout()
    out_path2 = OUT_DIR / "figI_pc1_drop_decomposition.png"
    fig.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path2.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
