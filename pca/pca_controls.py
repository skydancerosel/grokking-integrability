#!/usr/bin/env python3
"""
PCA control experiments for grokking eigenanalysis.

Control 1 — No-weight-decay baseline:
    Compare PCA eigenstructure of the grokking run (wd=0.1) vs a
    memorise-only run (wd=0).  If the rank-1 concentration is a
    grokking-specific phenomenon, the no-wd run should be *less*
    concentrated.

Control 2 — Fourier basis alignment (semantic check):
    Nanda et al. showed grokking on (a+b) mod p learns Fourier
    components: cos(2πk·n/p), sin(2πk·n/p).  We build the full
    Fourier basis for the weight matrices and measure the cosine
    alignment between the PCA principal components and each Fourier
    direction.  If PC1 is semantically meaningful, it should align
    with a small number of Fourier frequencies.

    The Fourier basis for a d×d weight matrix is constructed as
    outer products of the 1-D DFT basis vectors for Z/pZ.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
GROK_FILE = Path(__file__).parent / "grok_runs.pt"
NOWD_FILE = Path(__file__).parent / "grok_runs_nowd.pt"
OUT_DIR = Path(__file__).parent / "pca_plots"
TOP_K = 10
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
COLORS = {"WQ": "#1f77b4", "WK": "#ff7f0e", "WV": "#2ca02c", "WO": "#d62728"}
# ────────────────────────────────────────────────────────────────────────────


def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["cfg"], data["attn_logs"]


def collect_trajectory(attn_logs, layer_idx, key):
    steps, mats = [], []
    for snap in attn_logs:
        steps.append(snap["step"])
        mats.append(snap["layers"][layer_idx][key].float())
    return np.array(steps), mats


def pca_on_trajectory(mats, top_k):
    W0 = mats[0].reshape(-1).numpy()
    X = np.stack([m.reshape(-1).numpy() - W0 for m in mats])
    X -= X.mean(axis=0, keepdims=True)
    T, D = X.shape
    k = min(top_k, T, D)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigenvalues = (S ** 2) / (T - 1)
    total_var = eigenvalues.sum()
    explained = eigenvalues / total_var
    Z = U[:, :k] * S[:k]
    return {
        "eigenvalues": eigenvalues[:k],
        "explained_ratio": explained[:k],
        "scores": Z,
        "components": Vt[:k],   # (k, D) — principal directions
        "total_var": float(total_var),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 1 — grokking vs no-weight-decay comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_control_1():
    """Compare PCA eigenstructure: grokking (wd=0.1) vs memorise-only (wd=0)."""
    if not NOWD_FILE.exists():
        print(f"[control 1] {NOWD_FILE} not found — skipping. Run groking_v3_nowd.py first.")
        return

    cfg_grok, logs_grok = load_data(GROK_FILE)
    cfg_nowd, logs_nowd = load_data(NOWD_FILE)
    n_layers = cfg_grok["N_LAYERS"]

    print(f"[control 1] grokking: {len(logs_grok)} snaps, no-wd: {len(logs_nowd)} snaps")

    # ── collect global PCA for both runs ────────────────────────────────
    results = {}   # (run, layer, key) -> pca dict
    for label, logs in [("grok", logs_grok), ("nowd", logs_nowd)]:
        for li in range(n_layers):
            for key in WEIGHT_KEYS:
                _, mats = collect_trajectory(logs, li, key)
                results[(label, li, key)] = pca_on_trajectory(mats, TOP_K)

    # ── Figure 6: PC1% bar comparison across layers & weights ───────────
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5), squeeze=False)
    x = np.arange(len(WEIGHT_KEYS))
    width = 0.35
    for col, li in enumerate(n_layers * [None] if False else range(n_layers)):
        ax = axes[0, col]
        grok_vals = [results[("grok", li, k)]["explained_ratio"][0] * 100 for k in WEIGHT_KEYS]
        nowd_vals = [results[("nowd", li, k)]["explained_ratio"][0] * 100 for k in WEIGHT_KEYS]
        bars1 = ax.bar(x - width/2, grok_vals, width, label="grok (wd=0.1)", color="#2ca02c", alpha=0.85)
        bars2 = ax.bar(x + width/2, nowd_vals, width, label="no-wd (wd=0)", color="#d62728", alpha=0.85)
        ax.set_ylabel("PC1 explained var (%)")
        ax.set_title(f"Layer {li}")
        ax.set_xticks(x)
        ax.set_xticklabels(WEIGHT_KEYS)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        # annotate values
        for bar, val in zip(list(bars1) + list(bars2), grok_vals + nowd_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Control 1: PC1 Concentration — Grokking vs No-Weight-Decay", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_control_grok_vs_nowd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig6_control_grok_vs_nowd.png")

    # ── Figure 7: full top-5 eigenspectrum comparison ───────────────────
    fig, axes = plt.subplots(len(WEIGHT_KEYS), n_layers,
                             figsize=(5 * n_layers, 3.5 * len(WEIGHT_KEYS)), squeeze=False)
    for col, li in enumerate(range(n_layers)):
        for row, key in enumerate(WEIGHT_KEYS):
            ax = axes[row, col]
            g = results[("grok", li, key)]["explained_ratio"][:5] * 100
            n = results[("nowd", li, key)]["explained_ratio"][:5] * 100
            xx = np.arange(len(g))
            ax.bar(xx - 0.18, g, 0.35, label="grok", color="#2ca02c", alpha=0.85)
            ax.bar(xx + 0.18, n, 0.35, label="no-wd", color="#d62728", alpha=0.85)
            ax.set_xticks(xx)
            ax.set_xticklabels([f"PC{i+1}" for i in range(len(g))])
            ax.set_ylabel("Expl. var (%)")
            ax.set_title(f"Layer {li} — {key}")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Control 1: Top-5 Eigenspectrum — Grokking vs No-Weight-Decay", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_eigenspectrum_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig7_eigenspectrum_comparison.png")

    # ── Figure 8: PC1-PC2 trajectory comparison ────────────────────────
    fig, axes = plt.subplots(len(WEIGHT_KEYS), n_layers * 2,
                             figsize=(4 * n_layers * 2, 3.5 * len(WEIGHT_KEYS)),
                             squeeze=False)
    for li in range(n_layers):
        for row, key in enumerate(WEIGHT_KEYS):
            for run_idx, (label, logs) in enumerate([("grok", logs_grok), ("nowd", logs_nowd)]):
                ax = axes[row, li * 2 + run_idx]
                res = results[(label, li, key)]
                steps_arr, _ = collect_trajectory(logs, li, key)
                sc = ax.scatter(res["scores"][:, 0], res["scores"][:, 1],
                                c=steps_arr / 1000, cmap="viridis", s=6, edgecolors="none")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                tag = "GROK" if label == "grok" else "NO-WD"
                ax.set_title(f"L{li} {key} [{tag}]", fontsize=9)
                fig.colorbar(sc, ax=ax, pad=0.02).set_label("step(k)", fontsize=7)
    fig.suptitle("Control 1: PC1–PC2 Trajectories — Grokking (left) vs No-WD (right)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_trajectory_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig8_trajectory_comparison.png")

    # ── print summary table ─────────────────────────────────────────────
    print("\n  PC1 explained variance (%) — grokking vs no-weight-decay:")
    print(f"  {'':8s}", end="")
    for li in range(n_layers):
        print(f"  Layer {li} grok  Layer {li} nowd", end="")
    print()
    for key in WEIGHT_KEYS:
        print(f"  {key:8s}", end="")
        for li in range(n_layers):
            g = results[("grok", li, key)]["explained_ratio"][0] * 100
            n = results[("nowd", li, key)]["explained_ratio"][0] * 100
            diff = g - n
            print(f"     {g:5.1f}%       {n:5.1f}%", end="")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 2 — Fourier basis alignment
# ═══════════════════════════════════════════════════════════════════════════

def build_fourier_basis_1d(p):
    """
    Build normalised Fourier basis vectors for Z/pZ.

    Returns:
        freqs: list of (freq_index, 'cos'/'sin') labels
        basis: (n_basis, p) array — each row is a normalised basis vector

    For freq k = 0:        DC component (constant)
    For freq k = 1..(p-1): cos(2π k n / p) and sin(2π k n / p)
    """
    n = np.arange(p)
    basis = []
    freqs = []

    # DC
    dc = np.ones(p) / np.sqrt(p)
    basis.append(dc)
    freqs.append((0, "dc"))

    for k in range(1, (p + 1) // 2 + 1):
        c = np.cos(2 * np.pi * k * n / p)
        c /= np.linalg.norm(c)
        basis.append(c)
        freqs.append((k, "cos"))

        s = np.sin(2 * np.pi * k * n / p)
        norm = np.linalg.norm(s)
        if norm > 1e-10:
            s /= norm
            basis.append(s)
            freqs.append((k, "sin"))

    return freqs, np.stack(basis)


def build_fourier_basis_matrix(p, d_model):
    """
    Build Fourier basis for a (d_model × d_model) weight matrix.

    The key insight: the weight matrix W acts on d_model-dimensional vectors.
    The first p dimensions correspond to token embeddings for 0..p-1.
    So we project the *flattened* weight delta onto outer products of
    1-D Fourier vectors.

    But for a simpler, more robust approach: we measure alignment by
    projecting the PC direction (a d_model² vector) onto individual
    Fourier modes acting on the ROW space and COLUMN space separately.

    Specifically, for each Fourier freq k, we compute:
        alignment_k = sum over (row_mode, col_mode) of
                      |<PC, vec(row_mode ⊗ col_mode)>|²

    This captures how much of the PC direction lives in the k-th
    frequency subspace.

    For tractability, we use the 1-D basis on each axis independently
    and report the total energy per frequency.
    """
    freqs, basis_1d = build_fourier_basis_1d(p)  # (n_basis, p)
    n_basis = len(freqs)

    # We'll embed each 1-D Fourier vector into d_model dimensions
    # (pad with zeros beyond index p)
    basis_full = np.zeros((n_basis, d_model))
    basis_full[:, :p] = basis_1d[:, :min(p, d_model)]

    return freqs, basis_full


def fourier_alignment(pc_vec, freqs, basis_full, d_model):
    """
    Measure how much a PC direction (flattened d×d weight) aligns with
    each Fourier frequency.

    Strategy: reshape PC to (d, d), project rows and columns onto
    Fourier basis, aggregate energy per frequency.
    """
    W = pc_vec.reshape(d_model, d_model)

    # Project W onto Fourier basis: B @ W @ B.T gives the
    # representation in the Fourier-row × Fourier-col basis
    # Energy at frequency pair (k_row, k_col):
    #   |B[k_row] @ W @ B[k_col]|²
    F = basis_full @ W @ basis_full.T   # (n_basis, n_basis)

    # Aggregate per-frequency energy (sum over all pairs involving freq k)
    n_basis = len(freqs)
    freq_energy = {}
    for i, (k, _) in enumerate(freqs):
        if k not in freq_energy:
            freq_energy[k] = 0.0
        # row-frequency = k, any column
        freq_energy[k] += np.sum(F[i, :] ** 2)
        # column-frequency = k, any row (avoid double-counting diagonal)
        freq_energy[k] += np.sum(F[:, i] ** 2) - F[i, i] ** 2

    total = sum(freq_energy.values())
    if total > 0:
        freq_energy = {k: v / total for k, v in freq_energy.items()}

    return freq_energy


def run_control_2():
    """Fourier basis alignment of PCA components."""
    cfg, logs = load_data(GROK_FILE)
    p = cfg["P"]
    d_model = cfg["D_MODEL"]
    n_layers = cfg["N_LAYERS"]

    freqs, basis_full = build_fourier_basis_matrix(p, d_model)

    print(f"\n[control 2] Fourier basis: {len(freqs)} modes for p={p}")

    # analyse last layer first, then all
    all_alignments = {}   # (layer, key, pc_idx) -> {freq: energy}

    for li in range(n_layers):
        for key in WEIGHT_KEYS:
            _, mats = collect_trajectory(logs, li, key)
            res = pca_on_trajectory(mats, TOP_K)

            for pc_idx in range(min(3, len(res["components"]))):
                pc_vec = res["components"][pc_idx]
                align = fourier_alignment(pc_vec, freqs, basis_full, d_model)
                all_alignments[(li, key, pc_idx)] = align

    # ── Figure 9: Fourier energy spectrum of PC1 per (layer, weight) ──
    max_freq = (p + 1) // 2
    freq_range = np.arange(0, max_freq + 1)

    fig, axes = plt.subplots(len(WEIGHT_KEYS), n_layers,
                             figsize=(5 * n_layers, 3.5 * len(WEIGHT_KEYS)),
                             squeeze=False)
    for col, li in enumerate(range(n_layers)):
        for row, key in enumerate(WEIGHT_KEYS):
            ax = axes[row, col]
            for pc_idx, (color, ls) in enumerate(
                    [("#1f77b4", "-"), ("#ff7f0e", "--"), ("#2ca02c", ":")]):
                align = all_alignments[(li, key, pc_idx)]
                energies = [align.get(k, 0) * 100 for k in freq_range]
                ax.plot(freq_range, energies, color=color, ls=ls,
                        linewidth=1.2, label=f"PC{pc_idx+1}")
            ax.set_xlabel("Fourier freq k")
            ax.set_ylabel("Energy (%)")
            ax.set_title(f"Layer {li} — {key}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("Control 2: Fourier Frequency Alignment of Top PCs", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_fourier_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig9_fourier_alignment.png")

    # ── Figure 10: top-5 Fourier freqs for PC1, heatmap ────────────────
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4), squeeze=False)
    for col, li in enumerate(range(n_layers)):
        ax = axes[0, col]
        data = []
        for key in WEIGHT_KEYS:
            align = all_alignments[(li, key, 0)]
            sorted_freqs = sorted(align.items(), key=lambda x: -x[1])
            row = [v * 100 for _, v in sorted_freqs[:10]]
            data.append(row)
            # Print top-5 for log
            top5 = sorted_freqs[:5]
            print(f"  Layer {li} {key} PC1 top freqs: " +
                  "  ".join(f"k={k}:{v*100:.1f}%" for k, v in top5))
        data = np.array(data)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(WEIGHT_KEYS)))
        ax.set_yticklabels(WEIGHT_KEYS)
        ax.set_xlabel("Rank (sorted by energy)")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels([f"#{i+1}" for i in range(data.shape[1])])
        ax.set_title(f"Layer {li}")
        fig.colorbar(im, ax=ax, pad=0.02).set_label("Energy %")
    fig.suptitle("Control 2: Top-10 Fourier Modes of PC1 (ranked by energy)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig10_fourier_top_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig10_fourier_top_modes.png")

    # ── Figure 11: concentration — how many freqs capture 80% of PC1? ──
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), squeeze=False)
    for col, li in enumerate(range(n_layers)):
        ax = axes[0, col]
        for key in WEIGHT_KEYS:
            align = all_alignments[(li, key, 0)]
            sorted_e = sorted(align.values(), reverse=True)
            cumsum = np.cumsum(sorted_e)
            k80 = int(np.searchsorted(cumsum, 0.8)) + 1
            ax.bar(key, k80, color=COLORS[key], alpha=0.85)
            ax.text(key, k80 + 0.3, str(k80), ha="center", fontsize=9)
        ax.set_ylabel("# Fourier freqs for 80% energy")
        ax.set_title(f"Layer {li}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Control 2: Fourier Concentration of PC1 (freqs needed for 80%)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig11_fourier_concentration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig11_fourier_concentration.png")


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 3 — Random-walk null model
# ═══════════════════════════════════════════════════════════════════════════

def random_walk_null(mats, top_k, n_trials=10, seed=123):
    """
    Generate a random-walk null distribution for PC1%.

    Strategy: measure the per-step deltas from the real trajectory,
    then generate synthetic trajectories with the same step sizes
    but random (isotropic) directions.  Compute PC1% via the covariance
    trick (T×T gram matrix instead of D×D) since T≪D.

    This controls for the T≪D sampling bias — if PC1% in the real
    data significantly exceeds the null, it's not just an artefact
    of low T/D ratio.
    """
    W0 = mats[0].reshape(-1).numpy()
    flat = np.stack([m.reshape(-1).numpy() for m in mats])  # (T, D)
    deltas = np.diff(flat, axis=0)                           # (T-1, D)
    step_norms = np.linalg.norm(deltas, axis=1)              # (T-1,)

    T, D = flat.shape
    rng = np.random.RandomState(seed)

    null_pc1 = []
    for trial in range(n_trials):
        # random directions with matched norms — work in low-dim T space
        # Instead of generating (T-1, D) random vectors, use the covariance trick:
        # The eigenvalues of X X^T / (T-1) give the same nonzero eigenvalues as X^T X / (T-1)
        directions = rng.randn(T - 1, D)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
        syn_deltas = directions * step_norms[:, None]

        syn_traj = np.zeros((T, D))
        syn_traj[1:] = np.cumsum(syn_deltas, axis=0)
        syn_traj -= syn_traj.mean(axis=0, keepdims=True)

        # Covariance trick: eigenvalues of (T×T) gram matrix = nonzero eigenvalues of cov
        G = syn_traj @ syn_traj.T  # (T, T) — MUCH cheaper than full SVD
        eigvals = np.linalg.eigvalsh(G)[::-1]  # descending
        eigvals = np.maximum(eigvals, 0)  # numerical cleanup
        total = eigvals.sum()
        null_pc1.append(eigvals[0] / total if total > 0 else 0)

    return np.array(null_pc1)


def run_control_3():
    """Compare real PC1% against random-walk null."""
    cfg, logs = load_data(GROK_FILE)
    n_layers = cfg["N_LAYERS"]
    print(f"\n[control 3] Random-walk null model (20 trials per weight matrix)")

    real_pc1 = {}
    null_pc1 = {}

    for li in range(n_layers):
        for key in WEIGHT_KEYS:
            _, mats = collect_trajectory(logs, li, key)
            res = pca_on_trajectory(mats, TOP_K)
            real_pc1[(li, key)] = res["explained_ratio"][0] * 100
            null_dist = random_walk_null(mats, TOP_K) * 100
            null_pc1[(li, key)] = null_dist

    # ── Figure 12: real vs null PC1% ────────────────────────────────────
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5), squeeze=False)
    for col, li in enumerate(range(n_layers)):
        ax = axes[0, col]
        x = np.arange(len(WEIGHT_KEYS))
        real_vals = [real_pc1[(li, k)] for k in WEIGHT_KEYS]
        null_means = [null_pc1[(li, k)].mean() for k in WEIGHT_KEYS]
        null_stds = [null_pc1[(li, k)].std() for k in WEIGHT_KEYS]

        ax.bar(x - 0.18, real_vals, 0.35, label="Real (grokking)",
               color="#2ca02c", alpha=0.85)
        ax.bar(x + 0.18, null_means, 0.35, label="Null (random walk)",
               color="#999999", alpha=0.85, yerr=null_stds, capsize=3)
        ax.set_ylabel("PC1 explained var (%)")
        ax.set_title(f"Layer {li}")
        ax.set_xticks(x)
        ax.set_xticklabels(WEIGHT_KEYS)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # annotate
        for xi, (rv, nm) in enumerate(zip(real_vals, null_means)):
            ax.text(xi - 0.18, rv + 1, f"{rv:.1f}", ha="center", fontsize=7)
            ax.text(xi + 0.18, nm + 2, f"{nm:.1f}", ha="center", fontsize=7)

    fig.suptitle("Control 3: Real PC1% vs Random-Walk Null Model", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig12_null_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig12_null_model.png")

    # ── print summary ───────────────────────────────────────────────────
    print("\n  Real vs Null PC1% (mean ± std):")
    print(f"  {'':8s}", end="")
    for li in range(n_layers):
        print(f"  L{li} real   L{li} null", end="")
    print()
    for key in WEIGHT_KEYS:
        print(f"  {key:8s}", end="")
        for li in range(n_layers):
            r = real_pc1[(li, key)]
            nm = null_pc1[(li, key)].mean()
            ns = null_pc1[(li, key)].std()
            sigma_away = (r - nm) / (ns + 1e-8)
            print(f"  {r:5.1f}%  {nm:4.1f}±{ns:.1f}%", end="")
        print()
    print()
    # sigma-level summary
    print("  Sigma above null:")
    for key in WEIGHT_KEYS:
        print(f"  {key:8s}", end="")
        for li in range(n_layers):
            r = real_pc1[(li, key)]
            nm = null_pc1[(li, key)].mean()
            ns = null_pc1[(li, key)].std()
            sigma = (r - nm) / (ns + 1e-8)
            print(f"  L{li}: {sigma:5.1f}σ", end="")
        print()


# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("  CONTROL 1: Grokking vs No-Weight-Decay baseline")
    print("=" * 70)
    run_control_1()

    print("\n" + "=" * 70)
    print("  CONTROL 2: Fourier basis alignment")
    print("=" * 70)
    run_control_2()

    print("\n" + "=" * 70)
    print("  CONTROL 3: Random-walk null model")
    print("=" * 70)
    run_control_3()

    print("\nAll controls complete.")


if __name__ == "__main__":
    main()
