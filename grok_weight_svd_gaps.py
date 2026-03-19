#!/usr/bin/env python3
"""
Compute WEIGHT MATRIX SVD eigenvalue gaps (σ₁-σ₂, σ₂-σ₃) from the actual
W_Q and W_K matrices at each checkpoint, and plot against:
  1. Matrix commutator ||[W_Q, W_K]||_F
  2. SGD commutator defect D
  3. Generalization (train/test accuracy)

This tests the narrative:
  g₂₃↓ → SGD spike → modes compete (σ₁≈σ₂) → comm peak →
  one mode dominates (σ₁>>σ₂) → comm collapse → grokking

Uses existing grok_sweep_results/ and commutator_results.pt.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SWEEP_DIR = Path(__file__).parent / "grok_sweep_results"
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
OUT_DIR.mkdir(exist_ok=True)

TEST_OPS = ["add", "sub", "mul", "x2_y2"]
NON_GROK_OPS = ["x2_xy_y2", "x3_xy"]
SEEDS = [42, 137, 2024]
N_SV = 5  # track top-5 singular values


def load_run(op, wd, seed):
    path = SWEEP_DIR / f"{op}_wd{wd}_s{seed}.pt"
    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None


def extract_metrics(data):
    m = data["metrics"]
    return (np.array([e["step"] for e in m]),
            np.array([e["train_acc"] for e in m]),
            np.array([e["test_acc"] for e in m]))


def find_grok_step(data):
    for e in data["metrics"]:
        if e["test_acc"] >= 0.9:
            return e["step"]
    return None


# ── Core: SVD of weight matrices at each checkpoint ────────────────────

def compute_weight_svd(data, layer_idx=0, n_sv=N_SV):
    """SVD of W_Q and W_K at each checkpoint. Returns singular values and gaps.

    For each snapshot:
      - Compute SVD of W_Q and W_K (128x128 matrices)
      - Record top-n singular values
      - Compute gaps σ₁-σ₂, σ₂-σ₃
      - Also compute per-head (32x32 blocks) SVD
    """
    logs = data["attn_logs"]
    d_head = 32
    n_heads = 4

    steps = []
    # Full matrix SVD
    sv_Q, sv_K = [], []              # [T, n_sv] singular values
    gap_Q, gap_K = [], []            # [T, 2] gaps (σ1-σ2, σ2-σ3)
    ratio_Q, ratio_K = [], []        # [T, n_sv] normalized σ_i / σ_1

    # Per-head SVD
    head_gap_Q = []                  # [T, n_heads, 2]
    head_gap_K = []

    # Matrix commutator
    comm_norms = []
    head_comm_norms = []             # [T, n_heads]

    for snap in logs:
        WQ = snap["layers"][layer_idx]["WQ"].float().numpy()
        WK = snap["layers"][layer_idx]["WK"].float().numpy()
        steps.append(snap["step"])

        # Full matrix SVD
        UQ, SQ, VtQ = np.linalg.svd(WQ, full_matrices=False)
        UK, SK, VtK = np.linalg.svd(WK, full_matrices=False)

        k = min(n_sv, len(SQ))
        svq = np.zeros(n_sv); svq[:k] = SQ[:k]
        svk = np.zeros(n_sv); svk[:k] = SK[:k]
        sv_Q.append(svq)
        sv_K.append(svk)

        gap_Q.append([SQ[0] - SQ[1], SQ[1] - SQ[2]])
        gap_K.append([SK[0] - SK[1], SK[1] - SK[2]])

        ratio_Q.append(svq / max(SQ[0], 1e-30))
        ratio_K.append(svk / max(SK[0], 1e-30))

        # Matrix commutator
        comm = WQ @ WK - WK @ WQ
        comm_norms.append(np.linalg.norm(comm, "fro"))

        # Per-head
        hgq, hgk, hcn = [], [], []
        for h in range(n_heads):
            s, e = h * d_head, (h + 1) * d_head
            q_block = WQ[s:e, s:e]
            k_block = WK[s:e, s:e]

            sq = np.linalg.svd(q_block, compute_uv=False)
            sk = np.linalg.svd(k_block, compute_uv=False)
            hgq.append([sq[0] - sq[1], sq[1] - sq[2]])
            hgk.append([sk[0] - sk[1], sk[1] - sk[2]])
            hcn.append(np.linalg.norm(q_block @ k_block - k_block @ q_block, "fro"))

        head_gap_Q.append(hgq)
        head_gap_K.append(hgk)
        head_comm_norms.append(hcn)

    return dict(
        steps=np.array(steps),
        sv_Q=np.array(sv_Q), sv_K=np.array(sv_K),
        gap_Q=np.array(gap_Q), gap_K=np.array(gap_K),
        ratio_Q=np.array(ratio_Q), ratio_K=np.array(ratio_K),
        comm_norms=np.array(comm_norms),
        head_gap_Q=np.array(head_gap_Q),  # [T, 4, 2]
        head_gap_K=np.array(head_gap_K),
        head_comm_norms=np.array(head_comm_norms),  # [T, 4]
    )


def load_sgd_defect():
    path = OUT_DIR / "commutator_results.pt"
    if not path.exists():
        return {}
    cr = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for key, d in cr.items():
        op, wd = key
        if wd != 1.0:
            continue
        comm = d["comm"]
        steps = np.array([c["step"] for c in comm])
        defect = np.array([c["defect_median"] for c in comm])
        out[op] = (steps, defect)
    return out


def interp_to_common(steps_a, vals_a, steps_b, vals_b):
    lo = max(steps_a[0], steps_b[0])
    hi = min(steps_a[-1], steps_b[-1])
    common = np.arange(lo, hi + 1, 100)
    va = np.interp(common, steps_a, vals_a)
    vb = np.interp(common, steps_b, vals_b)
    return common, va, vb


# ── Figure 1: Master time-series panel ─────────────────────────────────

def plot_master_timeseries(all_data, sgd_all, save_path):
    """Per operation: 5-panel showing SVD gaps, commutators, accuracy."""
    sc = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}

    fig, axes = plt.subplots(len(TEST_OPS), 5, figsize=(26, 3.5 * len(TEST_OPS)),
                             sharex=False)
    if len(TEST_OPS) == 1:
        axes = axes.reshape(1, 5)

    for row, op in enumerate(TEST_OPS):
        seeds_d = all_data.get(op, {})

        # P1: σ₁-σ₂ for W_Q
        ax = axes[row, 0]
        for seed, d in sorted(seeds_d.items()):
            c = sc[seed]
            ax.plot(d["svd"]["steps"], d["svd"]["gap_Q"][:, 0], color=c, lw=1.5,
                    label=f"$W_Q$ s{seed}")
            ax.plot(d["svd"]["steps"], d["svd"]["gap_K"][:, 0], color=c, lw=1,
                    ls="--", alpha=0.6, label=f"$W_K$ s{seed}")
            if d["grok"]:
                ax.axvline(d["grok"], color=c, ls=":", alpha=0.3)
        ax.set_ylabel(f"{op}\n$\\sigma_1 - \\sigma_2$")
        if row == 0:
            ax.set_title("Weight SVD gap $\\sigma_1 - \\sigma_2$", fontsize=10)
        ax.legend(fontsize=5, ncol=2)

        # P2: σ₂-σ₃ for W_Q
        ax = axes[row, 1]
        for seed, d in sorted(seeds_d.items()):
            c = sc[seed]
            ax.plot(d["svd"]["steps"], d["svd"]["gap_Q"][:, 1], color=c, lw=1.5,
                    label=f"$W_Q$ s{seed}")
            ax.plot(d["svd"]["steps"], d["svd"]["gap_K"][:, 1], color=c, lw=1,
                    ls="--", alpha=0.6, label=f"$W_K$ s{seed}")
            if d["grok"]:
                ax.axvline(d["grok"], color=c, ls=":", alpha=0.3)
        ax.set_ylabel("$\\sigma_2 - \\sigma_3$")
        if row == 0:
            ax.set_title("Weight SVD gap $\\sigma_2 - \\sigma_3$", fontsize=10)
        ax.legend(fontsize=5, ncol=2)

        # P3: Matrix commutator
        ax = axes[row, 2]
        for seed, d in sorted(seeds_d.items()):
            c = sc[seed]
            ax.plot(d["svd"]["steps"], d["svd"]["comm_norms"], color=c, lw=1.5,
                    label=f"s{seed}")
            if d["grok"]:
                ax.axvline(d["grok"], color=c, ls=":", alpha=0.3)
        ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$")
        if row == 0:
            ax.set_title("Matrix commutator", fontsize=10)
        ax.legend(fontsize=6)

        # P4: SGD defect
        ax = axes[row, 3]
        sgd = sgd_all.get(op)
        if sgd:
            ax.semilogy(sgd[0], sgd[1], color="#9467bd", lw=2)
        for seed, d in sorted(seeds_d.items()):
            if d["grok"]:
                ax.axvline(d["grok"], color=sc[seed], ls=":", alpha=0.3)
        ax.set_ylabel("SGD defect $D$")
        if row == 0:
            ax.set_title("SGD commutator defect", fontsize=10)

        # P5: Accuracy
        ax = axes[row, 4]
        for seed, d in sorted(seeds_d.items()):
            c = sc[seed]
            ms, tr, te = d["metrics"]
            ax.plot(ms, tr, color=c, lw=0.8, alpha=0.3)
            ax.plot(ms, te, color=c, lw=1.5, label=f"s{seed}")
            if d["grok"]:
                ax.axvline(d["grok"], color=c, ls=":", alpha=0.3)
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(-0.05, 1.1)
        if row == 0:
            ax.set_title("Generalization", fontsize=10)
        ax.legend(fontsize=6)

    for j in range(5):
        axes[-1, j].set_xlabel("Step")

    fig.suptitle("Weight matrix SVD gaps vs commutators — testing the causal chain",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 2: Scatter – SVD gaps vs commutators ────────────────────────

def plot_scatter_svd(all_data, sgd_all, save_path):
    """Scatter of weight SVD gaps vs both commutators."""
    op_colors = {"add": "#e41a1c", "sub": "#377eb8", "mul": "#4daf4a", "x2_y2": "#984ea3"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Collect data
    g12Q_mc, g23Q_mc = [], []
    g12Q_sgd, g23Q_sgd = [], []
    g12K_mc, g23K_mc = [], []

    for op in TEST_OPS:
        seeds_d = all_data.get(op, {})
        sgd = sgd_all.get(op)
        for seed, d in seeds_d.items():
            sv = d["svd"]
            steps = sv["steps"]
            gapQ = sv["gap_Q"]
            gapK = sv["gap_K"]
            comm = sv["comm_norms"]

            for i in range(len(steps)):
                g12Q_mc.append((gapQ[i, 0], comm[i], op, seed))
                g23Q_mc.append((gapQ[i, 1], comm[i], op, seed))
                g12K_mc.append((gapK[i, 0], comm[i], op, seed))
                g23K_mc.append((gapK[i, 1], comm[i], op, seed))

            if sgd and seed == 42:
                sgd_s, sgd_d = sgd
                if len(sgd_s) > 1 and len(steps) > 1:
                    common, g12, sd = interp_to_common(steps, gapQ[:, 0], sgd_s, sgd_d)
                    _, g23, sd2 = interp_to_common(steps, gapQ[:, 1], sgd_s, sgd_d)
                    for i in range(len(common)):
                        g12Q_sgd.append((g12[i], sd[i], op, seed))
                        g23Q_sgd.append((g23[i], sd2[i], op, seed))

    def do_scatter(ax, data, xlabel, ylabel, title, logy=False):
        if not data:
            ax.set_title(title + " (no data)")
            return
        for op in TEST_OPS:
            pts = [(x, y) for x, y, o, s in data if o == op]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=10, alpha=0.35, color=op_colors[op], label=op)
                if len(xs) > 2:
                    r = np.corrcoef(xs, np.log10(ys) if logy else ys)[0, 1]
                    ax.annotate(f"{op}: r={r:.2f}",
                                xy=(0.02, 0.95 - 0.06 * TEST_OPS.index(op)),
                                xycoords="axes fraction", fontsize=7, color=op_colors[op])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, loc="lower right")

    do_scatter(axes[0, 0], g12Q_mc,
               "$\\sigma_1 - \\sigma_2$ ($W_Q$)", "$\\|[W_Q, W_K]\\|_F$",
               "$W_Q$ gap $\\sigma_1-\\sigma_2$ vs matrix comm")
    do_scatter(axes[0, 1], g23Q_mc,
               "$\\sigma_2 - \\sigma_3$ ($W_Q$)", "$\\|[W_Q, W_K]\\|_F$",
               "$W_Q$ gap $\\sigma_2-\\sigma_3$ vs matrix comm")
    do_scatter(axes[0, 2], g12K_mc,
               "$\\sigma_1 - \\sigma_2$ ($W_K$)", "$\\|[W_Q, W_K]\\|_F$",
               "$W_K$ gap $\\sigma_1-\\sigma_2$ vs matrix comm")

    do_scatter(axes[1, 0], g12Q_sgd,
               "$\\sigma_1 - \\sigma_2$ ($W_Q$)", "SGD defect $D$",
               "$W_Q$ gap $\\sigma_1-\\sigma_2$ vs SGD defect", logy=True)
    do_scatter(axes[1, 1], g23Q_sgd,
               "$\\sigma_2 - \\sigma_3$ ($W_Q$)", "SGD defect $D$",
               "$W_Q$ gap $\\sigma_2-\\sigma_3$ vs SGD defect", logy=True)
    do_scatter(axes[1, 2], g23K_mc,
               "$\\sigma_2 - \\sigma_3$ ($W_K$)", "$\\|[W_Q, W_K]\\|_F$",
               "$W_K$ gap $\\sigma_2-\\sigma_3$ vs matrix comm")

    fig.suptitle("Weight SVD gaps vs commutators (all ops × seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 3: Phase-separated – pre vs post grok ──────────────────────

def plot_phase_scatter(all_data, save_path):
    """Scatter colored by training phase: pre-grok / transition / post-grok."""
    op_colors = {"add": "#e41a1c", "sub": "#377eb8", "mul": "#4daf4a", "x2_y2": "#984ea3"}
    phase_markers = {"pre": "o", "trans": "D", "post": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    pre_12, trans_12, post_12 = [], [], []
    pre_23, trans_23, post_23 = [], [], []

    for op in TEST_OPS:
        for seed, d in all_data.get(op, {}).items():
            sv = d["svd"]
            grok = d["grok"]
            if grok is None:
                continue

            # Define phases: pre = before grok-500, trans = grok±500, post = after grok+500
            for i in range(len(sv["steps"])):
                step = sv["steps"][i]
                g12 = sv["gap_Q"][i, 0]
                g23 = sv["gap_Q"][i, 1]
                mc = sv["comm_norms"][i]
                entry = (g12, g23, mc, op)

                if step < grok - 500:
                    pre_12.append((g12, mc, op))
                    pre_23.append((g23, mc, op))
                elif step < grok + 500:
                    trans_12.append((g12, mc, op))
                    trans_23.append((g23, mc, op))
                else:
                    post_12.append((g12, mc, op))
                    post_23.append((g23, mc, op))

    def do_phase(ax, pre, trans, post, xlabel, ylabel, title):
        for phase, data, marker, alpha, label in [
            ("pre", pre, "o", 0.3, "pre-grok"),
            ("trans", trans, "D", 0.6, "transition"),
            ("post", post, "s", 0.3, "post-grok"),
        ]:
            for op in TEST_OPS:
                pts = [(x, y) for x, y, o in data if o == op]
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, s=12, alpha=alpha, color=op_colors[op],
                               marker=marker, label=f"{op} {label}" if op == TEST_OPS[0] else "")
            # Single legend entry per phase
            if data:
                xs, ys = zip(*[(x, y) for x, y, o in data])
                ax.scatter([], [], s=20, color="gray", marker=marker, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, ncol=2, loc="best")

    do_phase(axes[0, 0], pre_12, trans_12, post_12,
             "$\\sigma_1-\\sigma_2$ ($W_Q$)", "$\\|[W_Q,W_K]\\|_F$",
             "$W_Q$ gap₁₂ vs matrix comm (by phase)")
    do_phase(axes[0, 1], pre_23, trans_23, post_23,
             "$\\sigma_2-\\sigma_3$ ($W_Q$)", "$\\|[W_Q,W_K]\\|_F$",
             "$W_Q$ gap₂₃ vs matrix comm (by phase)")

    # Also: gap12 vs gap23 colored by phase
    pre_gg, trans_gg, post_gg = [], [], []
    for op in TEST_OPS:
        for seed, d in all_data.get(op, {}).items():
            sv = d["svd"]
            grok = d["grok"]
            if grok is None:
                continue
            for i in range(len(sv["steps"])):
                step = sv["steps"][i]
                g12 = sv["gap_Q"][i, 0]
                g23 = sv["gap_Q"][i, 1]
                entry = (g12, g23, op)
                if step < grok - 500:
                    pre_gg.append(entry)
                elif step < grok + 500:
                    trans_gg.append(entry)
                else:
                    post_gg.append(entry)

    do_phase(axes[1, 0], pre_gg, trans_gg, post_gg,
             "$\\sigma_1-\\sigma_2$", "$\\sigma_2-\\sigma_3$",
             "Gap₁₂ vs Gap₂₃ (phase trajectory)")

    # gap12 / gap23 ratio over time
    ax = axes[1, 1]
    sc = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}
    for op in TEST_OPS:
        for seed, d in sorted(all_data.get(op, {}).items()):
            sv = d["svd"]
            g12 = sv["gap_Q"][:, 0]
            g23 = sv["gap_Q"][:, 1]
            ratio = g12 / np.maximum(g23, 1e-10)
            c = op_colors[op]
            ax.plot(sv["steps"], ratio, color=c, lw=1, alpha=0.5,
                    label=f"{op}" if seed == 42 else "")
            if d["grok"]:
                ax.axvline(d["grok"], color=c, ls=":", alpha=0.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("$(\\sigma_1-\\sigma_2)/(\\sigma_2-\\sigma_3)$")
    ax.set_title("Gap ratio over training", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("Weight SVD gaps: phase structure", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 4: Per-head SVD gaps vs per-head commutator ─────────────────

def plot_perhead(all_data, save_path):
    """Per-head analysis: do individual head SVD gaps predict that head's commutator?"""
    op_colors = {"add": "#e41a1c", "sub": "#377eb8", "mul": "#4daf4a", "x2_y2": "#984ea3"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Collect per-head scatter data
    head_g12_vs_comm = []  # (gap, comm, op, head_idx)

    for op in TEST_OPS:
        for seed, d in all_data.get(op, {}).items():
            sv = d["svd"]
            for i in range(len(sv["steps"])):
                for h in range(4):
                    g12 = sv["head_gap_Q"][i, h, 0]
                    cn = sv["head_comm_norms"][i, h]
                    head_g12_vs_comm.append((g12, cn, op, h))

    for h in range(4):
        ax = axes[h // 2, h % 2]
        for op in TEST_OPS:
            pts = [(x, y) for x, y, o, hi in head_g12_vs_comm if o == op and hi == h]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=8, alpha=0.3, color=op_colors[op], label=op)
                if len(xs) > 2:
                    r = np.corrcoef(xs, ys)[0, 1]
                    ax.annotate(f"{op}: r={r:.2f}",
                                xy=(0.02, 0.95 - 0.06 * TEST_OPS.index(op)),
                                xycoords="axes fraction", fontsize=7, color=op_colors[op])
        ax.set_xlabel(f"$\\sigma_1-\\sigma_2$ (head {h})")
        ax.set_ylabel(f"$\\|[W_Q^h, W_K^h]\\|_F$")
        ax.set_title(f"Head {h}: SVD gap vs head commutator", fontsize=10)
        ax.legend(fontsize=6)

    fig.suptitle("Per-head: SVD gap $\\sigma_1-\\sigma_2$ vs head commutator", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 5: Narrative test – overlay all quantities on common timeline ─

def plot_narrative_test(all_data, sgd_all, save_path):
    """All quantities normalized to [0,1] on the same axis, testing the
    conjectured temporal ordering."""
    sc = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}

    fig, axes = plt.subplots(len(TEST_OPS), 1, figsize=(14, 4 * len(TEST_OPS)),
                             sharex=False)
    if len(TEST_OPS) == 1:
        axes = [axes]

    for row, op in enumerate(TEST_OPS):
        ax = axes[row]
        seeds_d = all_data.get(op, {})
        sgd = sgd_all.get(op)

        # Use seed=42 as the representative (has SGD defect)
        d42 = seeds_d.get(42)
        if d42 is None:
            continue

        sv = d42["svd"]
        steps = sv["steps"]
        grok = d42["grok"]

        def norm01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / max(mx - mn, 1e-30)

        # 1. σ₁-σ₂ (W_Q) – inverted so "modes compete" = high
        g12 = sv["gap_Q"][:, 0]
        ax.plot(steps, norm01(g12), color="#e41a1c", lw=2,
                label="$\\sigma_1-\\sigma_2$ ($W_Q$)")

        # 2. σ₂-σ₃ (W_Q)
        g23 = sv["gap_Q"][:, 1]
        ax.plot(steps, norm01(g23), color="#ff7f0e", lw=2,
                label="$\\sigma_2-\\sigma_3$ ($W_Q$)")

        # 3. Matrix commutator
        mc = sv["comm_norms"]
        ax.plot(steps, norm01(mc), color="#9467bd", lw=2, ls="--",
                label="$\\|[W_Q,W_K]\\|_F$")

        # 4. SGD defect (log-scale then normalize)
        if sgd:
            sgd_s, sgd_d = sgd
            sgd_log = np.log10(np.maximum(sgd_d, 1e-3))
            ax.plot(sgd_s, norm01(sgd_log), color="#2ca02c", lw=2, ls="-.",
                    label="SGD defect (log)")

        # 5. Test accuracy
        ms, tr, te = d42["metrics"]
        ax.plot(ms, te, color="#17becf", lw=1.5, ls=":",
                label="Test acc")

        if grok:
            ax.axvline(grok, color="black", ls="--", alpha=0.5, lw=1.5,
                       label=f"grok @{grok}")

        ax.set_ylabel(f"{op}\nnormalized [0,1]")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7, ncol=3, loc="upper right")

        if row == 0:
            ax.set_title("Narrative test: all quantities on common [0,1] scale (seed=42)",
                         fontsize=12)

    axes[-1].set_xlabel("Training step")
    fig.suptitle("Testing: g₂₃↓ → SGD spike → modes compete → comm peak → "
                 "one mode wins → comm collapse → grok",
                 fontsize=11, y=1.01, style="italic")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 6: Grokking vs non-grokking control ────────────────────────

def plot_grok_vs_control(all_data, save_path):
    """Compare weight SVD gap dynamics for grokking (wd=1) vs non-grokking (wd=0)."""
    op_colors_grok = {"add": "#e41a1c", "sub": "#377eb8", "mul": "#4daf4a", "x2_y2": "#984ea3"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Load wd=0 controls
    for ax_idx, (gap_idx, gap_label) in enumerate([(0, "$\\sigma_1-\\sigma_2$"),
                                                     (1, "$\\sigma_2-\\sigma_3$")]):
        ax_grok = axes[0, ax_idx]
        ax_ctrl = axes[1, ax_idx]

        for op in TEST_OPS:
            c = op_colors_grok[op]

            # Grokking (wd=1)
            d = all_data.get(op, {}).get(42)
            if d:
                sv = d["svd"]
                ax_grok.plot(sv["steps"], sv["gap_Q"][:, gap_idx], color=c, lw=1.5,
                             label=f"{op}")
                if d["grok"]:
                    ax_grok.axvline(d["grok"], color=c, ls=":", alpha=0.3)

            # Non-grokking control (wd=0)
            ctrl_data = load_run(op, 0.0, 42)
            if ctrl_data:
                sv_ctrl = compute_weight_svd(ctrl_data)
                ax_ctrl.plot(sv_ctrl["steps"], sv_ctrl["gap_Q"][:, gap_idx], color=c,
                             lw=1.5, label=f"{op}")

        ax_grok.set_title(f"Grokking (wd=1): {gap_label} ($W_Q$)", fontsize=10)
        ax_grok.set_ylabel(gap_label)
        ax_grok.legend(fontsize=7)
        ax_ctrl.set_title(f"Control (wd=0): {gap_label} ($W_Q$)", fontsize=10)
        ax_ctrl.set_ylabel(gap_label)
        ax_ctrl.set_xlabel("Step")
        ax_ctrl.legend(fontsize=7)

    fig.suptitle("SVD gap dynamics: grokking vs non-grokking control", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("WEIGHT SVD EIGENVALUE GAPS vs COMMUTATORS")
    print("=" * 72)

    sgd_all = load_sgd_defect()
    print(f"SGD defect data: {list(sgd_all.keys())}")

    all_data = {}
    for op in TEST_OPS:
        all_data[op] = {}
        for seed in SEEDS:
            data = load_run(op, 1.0, seed)
            if data is None or not data.get("grokked", False):
                print(f"  {op} s{seed}: skipped")
                continue

            svd = compute_weight_svd(data)
            met = extract_metrics(data)
            grok = find_grok_step(data)

            all_data[op][seed] = dict(svd=svd, metrics=met, grok=grok)

            # Print key moments
            g12 = svd["gap_Q"][:, 0]
            g23 = svd["gap_Q"][:, 1]
            mc = svd["comm_norms"]
            peak_mc_idx = np.argmax(mc[5:]) + 5  # skip first 5 for stability
            peak_g12_idx = np.argmax(g12[5:]) + 5
            min_g12_idx = np.argmin(g12[5:]) + 5

            print(f"  {op} s{seed}: grok@{grok}  "
                  f"σ₁-σ₂ peak@{svd['steps'][peak_g12_idx]}({g12[peak_g12_idx]:.3f})  "
                  f"σ₁-σ₂ min@{svd['steps'][min_g12_idx]}({g12[min_g12_idx]:.3f})  "
                  f"comm peak@{svd['steps'][peak_mc_idx]}({mc[peak_mc_idx]:.2f})")

    print("\nGenerating figures...")
    plot_master_timeseries(all_data, sgd_all,
                           OUT_DIR / "figSVD1_master_timeseries.png")
    plot_scatter_svd(all_data, sgd_all,
                     OUT_DIR / "figSVD2_scatter_gaps_vs_comm.png")
    plot_phase_scatter(all_data,
                       OUT_DIR / "figSVD3_phase_scatter.png")
    plot_perhead(all_data,
                 OUT_DIR / "figSVD4_perhead.png")
    plot_narrative_test(all_data, sgd_all,
                        OUT_DIR / "figSVD5_narrative_test.png")
    plot_grok_vs_control(all_data,
                         OUT_DIR / "figSVD6_grok_vs_control.png")

    # Save results
    save_data = {}
    for op in TEST_OPS:
        for seed, d in all_data[op].items():
            save_data[(op, seed)] = d["svd"]
    torch.save(save_data, OUT_DIR / "weight_svd_gap_results.pt")

    # ── Print narrative verdict ──
    print(f"\n{'=' * 72}")
    print("NARRATIVE TIMING TEST (seed=42)")
    print("Proposed: g₂₃↓ → SGD spike → modes compete → comm peak → "
          "mode dominates → comm collapse → grok")
    print(f"{'=' * 72}")

    for op in TEST_OPS:
        d = all_data[op].get(42)
        if d is None:
            continue
        sv = d["svd"]
        grok = d["grok"]
        steps = sv["steps"]
        g12 = sv["gap_Q"][:, 0]
        g23 = sv["gap_Q"][:, 1]
        mc = sv["comm_norms"]

        # Find key events
        # g23 decline start: first step where g23 starts monotonically decreasing
        g23_peak_idx = np.argmax(g23[:min(15, len(g23))])  # peak in early training

        # comm peak
        mc_peak_idx = np.argmax(mc[3:]) + 3
        mc_peak_step = steps[mc_peak_idx]
        mc_peak_val = mc[mc_peak_idx]

        # comm collapse: where mc drops below 50% of peak
        mc_collapse_step = None
        mid = mc[-1] + 0.5 * (mc_peak_val - mc[-1])
        for j in range(mc_peak_idx + 1, len(mc)):
            if mc[j] < mid:
                mc_collapse_step = steps[j]
                break

        # g12 minimum (modes most similar)
        g12_min_idx = np.argmin(g12)
        g12_min_step = steps[g12_min_idx]

        # g12 maximum (one mode dominates)
        g12_max_idx = np.argmax(g12[3:]) + 3
        g12_max_step = steps[g12_max_idx]

        # SGD spike
        sgd = sgd_all.get(op)
        sgd_spike = None
        if sgd:
            sgd_s, sgd_d = sgd
            baseline = max(np.median(sgd_d[:3]), 0.1)
            for i in range(2, len(sgd_s)):
                if sgd_d[i] > 10 * baseline and sgd_d[i] > 20:
                    sgd_spike = int(sgd_s[i])
                    break

        print(f"\n  {op}:")
        print(f"    g₂₃ peak (starts declining): step {steps[g23_peak_idx]}")
        print(f"    σ₁-σ₂ max (one mode ahead):  step {g12_max_step} (σ₁-σ₂={g12[g12_max_idx]:.3f})")
        print(f"    comm peak:                    step {mc_peak_step}")
        print(f"    SGD spike:                    step {sgd_spike}")
        print(f"    comm collapse:                step {mc_collapse_step}")
        print(f"    σ₁-σ₂ min (modes closest):   step {g12_min_step} (σ₁-σ₂={g12[g12_min_idx]:.3f})")
        print(f"    grok:                         step {grok}")

        # Check ordering
        events = []
        events.append(("g₂₃↓", steps[g23_peak_idx]))
        events.append(("σ₁₂ max", g12_max_step))
        events.append(("comm peak", mc_peak_step))
        if sgd_spike:
            events.append(("SGD spike", sgd_spike))
        if mc_collapse_step:
            events.append(("comm collapse", mc_collapse_step))
        events.append(("σ₁₂ min", g12_min_step))
        if grok:
            events.append(("grok", grok))
        events.sort(key=lambda x: x[1])
        print(f"    Actual order: {' → '.join(f'{e[0]}@{e[1]}' for e in events)}")

    print(f"\nSaved: weight_svd_gap_results.pt")


if __name__ == "__main__":
    main()
