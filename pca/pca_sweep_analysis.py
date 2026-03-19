#!/usr/bin/env python3
"""
Cross-operation PCA eigenanalysis of grokking sweep results.

Loads all .pt files from grok_sweep_results/ and produces:
  1. Summary table: PC1% for every (op, wd, seed, layer, weight)
  2. Figure A: PC1% bar chart — grok vs no-wd, averaged over seeds, per operation
  3. Figure B: PC1% heatmap — operations × weight matrices (grok runs only)
  4. Figure C: Eigenspectrum (top-5) per operation (averaged over seeds)
  5. Figure D: Grokking step vs PC1% scatter — does earlier grokking = higher PC1?
  6. Figure E: Random-walk null model — cross-operation sigma above null
  7. Figure F: Temporal PC1% evolution (expanding-window) for each operation
  8. Summary JSON with all computed statistics
"""

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ──────────────────────────────────────────────────────────────────
SWEEP_DIR = Path(__file__).parent / "grok_sweep_results"
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
TOP_K = 10
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
COLORS_WK = {"WQ": "#1f77b4", "WK": "#ff7f0e", "WV": "#2ca02c", "WO": "#d62728"}

OP_ORDER = ["add", "sub", "mul", "x2_y2", "x2_xy_y2", "x3_xy"]
OP_LABELS = {
    "add": "(a+b)",
    "sub": "(a−b)",
    "mul": "(a×b)",
    "x2_y2": "(a²+b²)",
    "x2_xy_y2": "(a²+ab+b²)",
    "x3_xy": "(a³+ab)",
}
SEEDS = [42, 137, 2024]
WDS = [1.0, 0.0]
# ────────────────────────────────────────────────────────────────────────────


def load_run(path):
    """Load a single sweep run .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def collect_trajectory(attn_logs, layer_idx, key):
    """Return (steps, mats) for a single weight across all snapshots."""
    steps, mats = [], []
    for snap in attn_logs:
        steps.append(snap["step"])
        mats.append(snap["layers"][layer_idx][key].float())
    return np.array(steps), mats


def pca_on_trajectory(mats, top_k):
    """PCA on flattened weight deltas from initialisation."""
    W0 = mats[0].reshape(-1).numpy()
    X = np.stack([m.reshape(-1).numpy() - W0 for m in mats])
    X -= X.mean(axis=0, keepdims=True)
    T, D = X.shape
    k = min(top_k, T, D)
    if T < 3:
        return None

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigenvalues = (S ** 2) / (T - 1)
    total_var = eigenvalues.sum()
    if total_var < 1e-15:
        return None
    explained = eigenvalues / total_var
    return {
        "eigenvalues": eigenvalues[:k],
        "explained_ratio": explained[:k],
        "components": Vt[:k],
        "total_var": float(total_var),
        "scores": (U[:, :k] * S[:k]),
    }


def expanding_window_pca(mats, top_k, n_checkpoints=15):
    """PCA on trajectory[:t] for growing t."""
    W0 = mats[0].reshape(-1).numpy()
    flat = np.stack([m.reshape(-1).numpy() - W0 for m in mats])
    T = len(flat)
    min_t = max(5, T // n_checkpoints)
    sizes = np.unique(np.linspace(min_t, T, n_checkpoints, dtype=int))

    records = []
    for t in sizes:
        chunk = flat[:t]
        chunk = chunk - chunk.mean(axis=0, keepdims=True)
        if chunk.shape[0] < 3:
            continue
        _, S, _ = np.linalg.svd(chunk, full_matrices=False)
        eig = (S ** 2) / (chunk.shape[0] - 1)
        total = eig.sum()
        if total < 1e-15:
            continue
        k = min(top_k, len(eig))
        records.append({
            "n_snaps": int(t),
            "pc1_pct": float(eig[0] / total) * 100,
            "top3_pct": float(eig[:min(3, k)].sum() / total) * 100,
            "top5_pct": float(eig[:min(5, k)].sum() / total) * 100,
        })
    return records


def random_walk_null(mats, n_trials=10, seed=123):
    """
    Random-walk null model — matched step norms, random directions.
    Uses gram matrix trick for efficiency.
    """
    flat = np.stack([m.reshape(-1).numpy() for m in mats])
    deltas = np.diff(flat, axis=0)
    step_norms = np.linalg.norm(deltas, axis=1)
    T, D = flat.shape
    rng = np.random.RandomState(seed)

    null_pc1 = []
    for _ in range(n_trials):
        directions = rng.randn(T - 1, D)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
        syn_deltas = directions * step_norms[:, None]
        syn_traj = np.zeros((T, D))
        syn_traj[1:] = np.cumsum(syn_deltas, axis=0)
        syn_traj -= syn_traj.mean(axis=0, keepdims=True)
        G = syn_traj @ syn_traj.T
        eigvals = np.linalg.eigvalsh(G)[::-1]
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        null_pc1.append(eigvals[0] / total if total > 0 else 0)
    return np.array(null_pc1)


# ═══════════════════════════════════════════════════════════════════════════
# Load all sweep results
# ═══════════════════════════════════════════════════════════════════════════

def load_all_runs():
    """Load all sweep .pt files. Return dict keyed by (op, wd, seed)."""
    runs = {}
    for pt_file in sorted(SWEEP_DIR.glob("*.pt")):
        if pt_file.name == "sweep_summary.json":
            continue
        data = load_run(pt_file)
        cfg = data["cfg"]
        key = (cfg["OP_NAME"], cfg["WEIGHT_DECAY"], cfg["SEED"])
        runs[key] = data
    return runs


def compute_all_pca(runs):
    """Compute PCA for all runs. Returns nested dict of results."""
    all_pca = {}   # (op, wd, seed, layer, wkey) -> pca_result
    for (op, wd, seed), data in runs.items():
        n_layers = data["cfg"]["N_LAYERS"]
        logs = data["attn_logs"]
        if len(logs) < 5:
            print(f"  [skip] {op} wd={wd} s={seed}: only {len(logs)} snapshots")
            continue
        for li in range(n_layers):
            for wkey in WEIGHT_KEYS:
                _, mats = collect_trajectory(logs, li, wkey)
                res = pca_on_trajectory(mats, TOP_K)
                if res is not None:
                    all_pca[(op, wd, seed, li, wkey)] = res
    return all_pca


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_a_grok_vs_nowd(all_pca, n_layers):
    """
    Figure A: PC1% bar chart — grokking vs no-wd, averaged over seeds.
    One subplot per layer, grouped by operation.
    """
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)

    for col, li in enumerate(range(n_layers)):
        ax = axes[0, col]
        x = np.arange(len(OP_ORDER))
        width = 0.35

        grok_means, grok_stds = [], []
        nowd_means, nowd_stds = [], []

        for op in OP_ORDER:
            # grok (wd=0.1)
            vals = []
            for seed in SEEDS:
                # Average over all weight keys for this layer
                pc1s = []
                for wkey in WEIGHT_KEYS:
                    k = (op, 1.0, seed, li, wkey)
                    if k in all_pca:
                        pc1s.append(all_pca[k]["explained_ratio"][0] * 100)
                if pc1s:
                    vals.append(np.mean(pc1s))
            grok_means.append(np.mean(vals) if vals else 0)
            grok_stds.append(np.std(vals) if len(vals) > 1 else 0)

            # no-wd
            vals = []
            for seed in SEEDS:
                pc1s = []
                for wkey in WEIGHT_KEYS:
                    k = (op, 0.0, seed, li, wkey)
                    if k in all_pca:
                        pc1s.append(all_pca[k]["explained_ratio"][0] * 100)
                if pc1s:
                    vals.append(np.mean(pc1s))
            nowd_means.append(np.mean(vals) if vals else 0)
            nowd_stds.append(np.std(vals) if len(vals) > 1 else 0)

        bars1 = ax.bar(x - width/2, grok_means, width, yerr=grok_stds,
                        label="grok (wd=1.0)", color="#2ca02c", alpha=0.85, capsize=3)
        bars2 = ax.bar(x + width/2, nowd_means, width, yerr=nowd_stds,
                        label="no-wd (wd=0.0)", color="#d62728", alpha=0.85, capsize=3)

        ax.set_ylabel("Mean PC1 explained var (%)")
        ax.set_title(f"Layer {li}")
        ax.set_xticks(x)
        ax.set_xticklabels([OP_LABELS[op] for op in OP_ORDER], rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-Operation: PC1% — Grokking vs No-Weight-Decay\n(averaged over seeds & weight matrices)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figA_grok_vs_nowd_crossop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figA_grok_vs_nowd_crossop.png")


def fig_b_heatmap(all_pca, n_layers):
    """
    Figure B: PC1% heatmap — operations × weight matrices.
    Grokking runs only, averaged over seeds.
    """
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4), squeeze=False)

    for col, li in enumerate(range(n_layers)):
        ax = axes[0, col]
        data = np.zeros((len(OP_ORDER), len(WEIGHT_KEYS)))

        for i, op in enumerate(OP_ORDER):
            for j, wkey in enumerate(WEIGHT_KEYS):
                vals = []
                for seed in SEEDS:
                    k = (op, 1.0, seed, li, wkey)
                    if k in all_pca:
                        vals.append(all_pca[k]["explained_ratio"][0] * 100)
                data[i, j] = np.mean(vals) if vals else 0

        im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        ax.set_yticks(range(len(OP_ORDER)))
        ax.set_yticklabels([OP_LABELS[op] for op in OP_ORDER], fontsize=9)
        ax.set_xticks(range(len(WEIGHT_KEYS)))
        ax.set_xticklabels(WEIGHT_KEYS)
        ax.set_title(f"Layer {li}")

        # annotate cells
        for i in range(len(OP_ORDER)):
            for j in range(len(WEIGHT_KEYS)):
                color = "white" if data[i, j] > 60 else "black"
                ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                        fontsize=8, color=color)

        fig.colorbar(im, ax=ax, pad=0.02).set_label("PC1 %")

    fig.suptitle("PC1% Heatmap — Grokking Runs (averaged over seeds)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figB_pc1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figB_pc1_heatmap.png")


def fig_c_eigenspectrum(all_pca, n_layers):
    """
    Figure C: Top-5 eigenspectrum per operation (grok runs, last layer, averaged over seeds).
    """
    li = n_layers - 1  # last layer

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for idx, op in enumerate(OP_ORDER):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        x = np.arange(5)
        width = 0.18

        for wi, wkey in enumerate(WEIGHT_KEYS):
            vals_per_pc = []
            for pc_i in range(5):
                seed_vals = []
                for seed in SEEDS:
                    k = (op, 1.0, seed, li, wkey)
                    if k in all_pca and len(all_pca[k]["explained_ratio"]) > pc_i:
                        seed_vals.append(all_pca[k]["explained_ratio"][pc_i] * 100)
                vals_per_pc.append(np.mean(seed_vals) if seed_vals else 0)

            ax.bar(x + wi * width, vals_per_pc, width,
                   label=wkey, color=COLORS_WK[wkey], alpha=0.85)

        ax.set_xlabel("PC index")
        ax.set_ylabel("Explained var (%)")
        ax.set_title(f"{OP_LABELS[op]} mod p")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([f"PC{i+1}" for i in range(5)])
        if idx == 0:
            ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Top-5 Eigenspectrum per Operation (Layer {li}, grok runs, mean over seeds)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC_eigenspectrum_crossop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figC_eigenspectrum_crossop.png")


def fig_d_grok_step_vs_pc1(all_pca, runs, n_layers):
    """
    Figure D: Scatter plot — grokking step vs PC1%.
    Each point is one grokking run (wd=0.1).
    """
    li = n_layers - 1  # last layer

    fig, axes = plt.subplots(1, len(WEIGHT_KEYS), figsize=(4.5 * len(WEIGHT_KEYS), 4), squeeze=False)

    op_colors = plt.cm.Set2(np.linspace(0, 1, len(OP_ORDER)))

    for wi, wkey in enumerate(WEIGHT_KEYS):
        ax = axes[0, wi]
        for oi, op in enumerate(OP_ORDER):
            for seed in SEEDS:
                run_key = (op, 1.0, seed)
                pca_key = (op, 1.0, seed, li, wkey)
                if run_key not in runs or pca_key not in all_pca:
                    continue
                run = runs[run_key]
                grokked = run["grokked"]
                final_step = run["final_step"]
                pc1 = all_pca[pca_key]["explained_ratio"][0] * 100

                marker = "o" if grokked else "x"
                ax.scatter(final_step / 1000, pc1, c=[op_colors[oi]],
                           marker=marker, s=50, edgecolors="black" if grokked else "none",
                           linewidths=0.5, label=OP_LABELS[op] if seed == SEEDS[0] else "")

        ax.set_xlabel("Final step (k)")
        ax.set_ylabel("PC1 (%)")
        ax.set_title(wkey)
        ax.grid(alpha=0.3)
        if wi == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Grokking Step vs PC1% (Layer {li})\n○ = grokked, × = did not grok",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figD_grok_step_vs_pc1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figD_grok_step_vs_pc1.png")


def fig_e_null_model(runs, n_layers, n_null_trials=10):
    """
    Figure E: Cross-operation sigma above random-walk null.
    Bar chart of z-scores per (operation, weight key) for last layer.
    """
    li = n_layers - 1
    print(f"\n  Computing random-walk null for {len(OP_ORDER)} ops × {len(WEIGHT_KEYS)} weights...")

    z_scores = {}  # (op, wkey) -> mean z-score over seeds

    for op in OP_ORDER:
        for wkey in WEIGHT_KEYS:
            zs = []
            for seed in SEEDS:
                run_key = (op, 1.0, seed)
                if run_key not in runs:
                    continue
                data = runs[run_key]
                logs = data["attn_logs"]
                if len(logs) < 5:
                    continue
                _, mats = collect_trajectory(logs, li, wkey)

                # real PC1%
                res = pca_on_trajectory(mats, TOP_K)
                if res is None:
                    continue
                real_pc1 = res["explained_ratio"][0] * 100

                # null distribution
                null_dist = random_walk_null(mats, n_trials=n_null_trials, seed=seed) * 100
                null_mean = null_dist.mean()
                null_std = null_dist.std()
                z = (real_pc1 - null_mean) / (null_std + 1e-8)
                zs.append(z)
                print(f"    {op} {wkey} s{seed}: real={real_pc1:.1f}% null={null_mean:.1f}±{null_std:.1f}% → z={z:.1f}σ")

            z_scores[(op, wkey)] = np.mean(zs) if zs else 0

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(OP_ORDER))
    width = 0.18
    for wi, wkey in enumerate(WEIGHT_KEYS):
        vals = [z_scores.get((op, wkey), 0) for op in OP_ORDER]
        ax.bar(x + wi * width, vals, width, label=wkey, color=COLORS_WK[wkey], alpha=0.85)

    ax.axhline(y=3, color="gray", ls="--", alpha=0.5, label="3σ threshold")
    ax.set_ylabel("Z-score (σ above null)")
    ax.set_xlabel("Operation")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([OP_LABELS[op] for op in OP_ORDER], fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Cross-Operation: Sigma Above Random-Walk Null (Layer {li}, mean over seeds)",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figE_null_zscores_crossop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figE_null_zscores_crossop.png")

    return z_scores


def fig_f_temporal_evolution(runs, n_layers):
    """
    Figure F: Expanding-window PC1% over training for each operation.
    One subplot per operation, grok runs only (wd=0.1), last layer, WV.
    """
    li = n_layers - 1
    wkey = "WV"  # typically strongest signal

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, op in enumerate(OP_ORDER):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        for si, seed in enumerate(SEEDS):
            run_key = (op, 1.0, seed)
            if run_key not in runs:
                continue
            data = runs[run_key]
            logs = data["attn_logs"]
            if len(logs) < 5:
                continue

            steps_arr, mats = collect_trajectory(logs, li, wkey)
            recs = expanding_window_pca(mats, TOP_K, n_checkpoints=20)
            if not recs:
                continue

            # x-axis: fraction of snapshots used
            fracs = [r["n_snaps"] / len(mats) * 100 for r in recs]
            pc1s = [r["pc1_pct"] for r in recs]

            label_str = f"s{seed}"
            if data["grokked"]:
                label_str += f" ✓@{data['final_step']//1000}k"
            else:
                label_str += " ✗"

            ax.plot(fracs, pc1s, color=seed_colors[si], linewidth=1.5,
                    label=label_str, marker=".", markersize=4)

        ax.set_xlabel("% of trajectory used")
        ax.set_ylabel("PC1 (%)")
        ax.set_title(f"{OP_LABELS[op]} mod p")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)

    fig.suptitle(f"Expanding-Window PC1% Over Training (Layer {li}, {wkey}, grok runs)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figF_temporal_crossop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figF_temporal_crossop.png")


def fig_g_per_weight_crossop(all_pca, n_layers):
    """
    Figure G: Per-weight-matrix PC1% comparison across operations.
    One subplot per weight key, bars grouped by operation, grok vs nowd.
    Last layer only.
    """
    li = n_layers - 1
    fig, axes = plt.subplots(1, len(WEIGHT_KEYS), figsize=(5 * len(WEIGHT_KEYS), 5), squeeze=False)

    for wi, wkey in enumerate(WEIGHT_KEYS):
        ax = axes[0, wi]
        x = np.arange(len(OP_ORDER))
        width = 0.35

        grok_means, grok_stds = [], []
        nowd_means, nowd_stds = [], []

        for op in OP_ORDER:
            gvals = [all_pca[(op, 1.0, s, li, wkey)]["explained_ratio"][0] * 100
                     for s in SEEDS if (op, 1.0, s, li, wkey) in all_pca]
            nvals = [all_pca[(op, 0.0, s, li, wkey)]["explained_ratio"][0] * 100
                     for s in SEEDS if (op, 0.0, s, li, wkey) in all_pca]
            grok_means.append(np.mean(gvals) if gvals else 0)
            grok_stds.append(np.std(gvals) if len(gvals) > 1 else 0)
            nowd_means.append(np.mean(nvals) if nvals else 0)
            nowd_stds.append(np.std(nvals) if len(nvals) > 1 else 0)

        ax.bar(x - width/2, grok_means, width, yerr=grok_stds,
               label="grok", color="#2ca02c", alpha=0.85, capsize=3)
        ax.bar(x + width/2, nowd_means, width, yerr=nowd_stds,
               label="no-wd", color="#d62728", alpha=0.85, capsize=3)
        ax.set_ylabel("PC1 (%)")
        ax.set_title(f"{wkey} (Layer {li})")
        ax.set_xticks(x)
        ax.set_xticklabels([OP_LABELS[op] for op in OP_ORDER], rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Weight PC1%: Grokking vs No-WD Across Operations",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figG_per_weight_crossop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figG_per_weight_crossop.png")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(all_pca, runs, n_layers):
    """Print and return a summary table of all results."""
    summary = []

    print(f"\n{'='*100}")
    print("  CROSS-OPERATION PCA SUMMARY")
    print(f"{'='*100}")

    for li in range(n_layers):
        print(f"\n  Layer {li}:")
        print(f"  {'op':>12s}  {'wd':>4s}  {'seed':>5s}  {'grok':>5s}  {'step':>6s}  ", end="")
        for wkey in WEIGHT_KEYS:
            print(f"{wkey:>8s}", end="  ")
        print("  mean")
        print(f"  {'─'*12}  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*6}  ", end="")
        for _ in WEIGHT_KEYS:
            print(f"{'─'*8}", end="  ")
        print(f"  {'─'*6}")

        for op in OP_ORDER:
            for wd in WDS:
                for seed in SEEDS:
                    run_key = (op, wd, seed)
                    if run_key not in runs:
                        continue
                    run = runs[run_key]
                    grokked = "YES" if run["grokked"] else "no"
                    final_step = run["final_step"]

                    pc1s = []
                    print(f"  {op:>12s}  {wd:4.1f}  {seed:5d}  {grokked:>5s}  {final_step:6d}  ", end="")
                    for wkey in WEIGHT_KEYS:
                        k = (op, wd, seed, li, wkey)
                        if k in all_pca:
                            pc1 = all_pca[k]["explained_ratio"][0] * 100
                            pc1s.append(pc1)
                            print(f"{pc1:7.1f}%", end="  ")
                        else:
                            print(f"{'N/A':>8s}", end="  ")
                    mean_pc1 = np.mean(pc1s) if pc1s else 0
                    print(f"  {mean_pc1:5.1f}%")

                    summary.append({
                        "op": op, "wd": wd, "seed": seed,
                        "layer": li,
                        "grokked": run["grokked"],
                        "final_step": final_step,
                        **{f"pc1_{wkey}": (all_pca[(op, wd, seed, li, wkey)]["explained_ratio"][0] * 100
                           if (op, wd, seed, li, wkey) in all_pca else None)
                           for wkey in WEIGHT_KEYS},
                        "mean_pc1": mean_pc1,
                    })

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # check what's available
    pt_files = list(SWEEP_DIR.glob("*.pt"))
    pt_files = [f for f in pt_files if f.stem != "sweep_summary"]
    print(f"Found {len(pt_files)} .pt files in {SWEEP_DIR}/")

    if len(pt_files) == 0:
        print("No sweep results found. Run grok_sweep.py first.")
        sys.exit(1)

    # load all
    print("Loading all runs...")
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs")

    # show what we have
    ops_found = set()
    for (op, wd, seed) in runs:
        ops_found.add(op)
    print(f"Operations found: {sorted(ops_found)}")

    # determine n_layers from first run
    first_run = next(iter(runs.values()))
    n_layers = first_run["cfg"]["N_LAYERS"]

    # compute PCA
    print("\nComputing PCA for all runs...")
    all_pca = compute_all_pca(runs)
    print(f"Computed {len(all_pca)} PCA results")

    # summary table
    summary = print_summary_table(all_pca, runs, n_layers)

    # save summary JSON
    summary_path = OUT_DIR / "pca_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")

    # generate figures
    print("\n" + "=" * 70)
    print("  GENERATING FIGURES")
    print("=" * 70)

    fig_a_grok_vs_nowd(all_pca, n_layers)
    fig_b_heatmap(all_pca, n_layers)
    fig_c_eigenspectrum(all_pca, n_layers)
    fig_d_grok_step_vs_pc1(all_pca, runs, n_layers)
    fig_e_null_model(runs, n_layers, n_null_trials=10)
    fig_f_temporal_evolution(runs, n_layers)
    fig_g_per_weight_crossop(all_pca, n_layers)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
