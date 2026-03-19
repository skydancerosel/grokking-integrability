#!/usr/bin/env python3
"""
Compute representation eigenvalue gaps (λ₁-λ₂, λ₂-λ₃) from expanding-window
PCA on QK weight update deltas, and plot against:
  1. Matrix commutator ||[W_Q, W_K]||_F
  2. SGD commutator defect D

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
SEEDS = [42, 137, 2024]
N_TOP_PC = 5


def load_run(op, wd, seed):
    path = SWEEP_DIR / f"{op}_wd{wd}_s{seed}.pt"
    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None


def extract_metrics(data):
    m = data["metrics"]
    return (np.array([e["step"] for e in m]),
            np.array([e["train_acc"] for e in m]),
            np.array([e["test_acc"] for e in m]))


def compute_eigenvalue_gaps(attn_logs, layer_idx=0, top_k=N_TOP_PC):
    """Expanding-window PCA: return raw eigenvalues and gaps at each window."""
    deltas, delta_steps = [], []
    for i in range(1, len(attn_logs)):
        WQ0 = attn_logs[i-1]["layers"][layer_idx]["WQ"].float().numpy().flatten()
        WK0 = attn_logs[i-1]["layers"][layer_idx]["WK"].float().numpy().flatten()
        WQ1 = attn_logs[i]["layers"][layer_idx]["WQ"].float().numpy().flatten()
        WK1 = attn_logs[i]["layers"][layer_idx]["WK"].float().numpy().flatten()
        deltas.append(np.concatenate([WQ1 - WQ0, WK1 - WK0]))
        delta_steps.append(attn_logs[i]["step"])

    steps_out = []
    raw_eigvals_list = []  # raw eigenvalues (top-k)
    ratio_list = []        # explained variance ratios (top-k)
    gap_raw_list = []      # [λ1-λ2, λ2-λ3] raw
    gap_ratio_list = []    # [r1-r2, r2-r3] ratio gaps

    for t in range(3, len(deltas) + 1):
        step = delta_steps[t - 1]
        X = np.stack(deltas[:t])
        X -= X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        eigvals = (S ** 2) / max(X.shape[0] - 1, 1)
        total = eigvals.sum()
        if total < 1e-30:
            continue

        k = min(top_k, len(eigvals))
        ev = np.zeros(top_k)
        ev[:k] = eigvals[:k]
        ratios = np.zeros(top_k)
        ratios[:k] = eigvals[:k] / total

        steps_out.append(step)
        raw_eigvals_list.append(ev.copy())
        ratio_list.append(ratios.copy())

        # Gaps
        gap_raw_list.append([ev[0] - ev[1], ev[1] - ev[2]])
        gap_ratio_list.append([ratios[0] - ratios[1], ratios[1] - ratios[2]])

    return (np.array(steps_out), np.array(raw_eigvals_list),
            np.array(ratio_list), np.array(gap_raw_list), np.array(gap_ratio_list))


def compute_matrix_commutator(data, layer_idx=0):
    logs = data["attn_logs"]
    steps, norms = [], []
    for snap in logs:
        WQ = snap["layers"][layer_idx]["WQ"].float().numpy()
        WK = snap["layers"][layer_idx]["WK"].float().numpy()
        steps.append(snap["step"])
        norms.append(np.linalg.norm(WQ @ WK - WK @ WQ, "fro"))
    return np.array(steps), np.array(norms)


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


def find_grok_step(data):
    m = data["metrics"]
    for e in m:
        if e["test_acc"] >= 0.9:
            return e["step"]
    return None


def interp_to_common(steps_a, vals_a, steps_b, vals_b):
    """Interpolate both series to common step grid (intersection of ranges)."""
    lo = max(steps_a[0], steps_b[0])
    hi = min(steps_a[-1], steps_b[-1])
    common = np.arange(lo, hi + 1, 100)
    va = np.interp(common, steps_a, vals_a)
    vb = np.interp(common, steps_b, vals_b)
    return common, va, vb


# ── Figure 1: Time series – eigenvalue gaps + commutators (per-seed) ────

def plot_timeseries(all_data, sgd_all, save_path):
    """6-panel grid: eigenvalue gaps overlaid with both commutators."""
    fig, axes = plt.subplots(len(TEST_OPS), 2, figsize=(16, 3.5 * len(TEST_OPS)),
                             sharex=False)
    if len(TEST_OPS) == 1:
        axes = axes.reshape(1, 2)

    sc = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}

    for row, op in enumerate(TEST_OPS):
        seeds_d = all_data.get(op, {})

        # Left: λ₁-λ₂ gap (ratio) + matrix comm
        ax = axes[row, 0]
        ax_r = ax.twinx()
        for seed, d in sorted(seeds_d.items()):
            c = sc.get(seed, "gray")
            gap_steps, gap_ratio = d["gap_steps"], d["gap_ratio"]
            comm_steps, comm_norms = d["comm_steps"], d["comm_norms"]
            grok = d["grok_step"]

            ax.plot(gap_steps, gap_ratio[:, 0], color=c, lw=1.5,
                    label=f"$\\lambda_1-\\lambda_2$ s{seed}")
            ax_r.plot(comm_steps, comm_norms, color=c, ls="--", lw=1.2, alpha=0.6)
            if grok:
                ax.axvline(grok, color=c, ls=":", alpha=0.3, lw=0.8)

        ax.set_ylabel(f"{op}\n$\\lambda_1 - \\lambda_2$ (ratio)", fontsize=9)
        ax_r.set_ylabel("$\\|[W_Q, W_K]\\|_F$", fontsize=8, color="gray")
        ax_r.tick_params(axis="y", labelcolor="gray")
        if row == 0:
            ax.set_title("Eigenvalue gap $\\lambda_1 - \\lambda_2$  (solid)\nvs  matrix comm (dashed)", fontsize=10)
        ax.legend(fontsize=6, loc="upper left")

        # Right: λ₂-λ₃ gap (ratio) + SGD defect
        ax = axes[row, 1]
        for seed, d in sorted(seeds_d.items()):
            c = sc.get(seed, "gray")
            gap_steps, gap_ratio = d["gap_steps"], d["gap_ratio"]
            grok = d["grok_step"]
            ax.plot(gap_steps, gap_ratio[:, 1], color=c, lw=1.5,
                    label=f"$\\lambda_2-\\lambda_3$ s{seed}")
            if grok:
                ax.axvline(grok, color=c, ls=":", alpha=0.3, lw=0.8)

        sgd = sgd_all.get(op)
        if sgd:
            ax_r = ax.twinx()
            ax_r.semilogy(sgd[0], sgd[1], color="#9467bd", lw=1.5, alpha=0.5,
                          label="SGD defect")
            ax_r.set_ylabel("SGD defect $D$", fontsize=8, color="#9467bd")
            ax_r.tick_params(axis="y", labelcolor="#9467bd")

        ax.set_ylabel(f"$\\lambda_2 - \\lambda_3$ (ratio)", fontsize=9)
        if row == 0:
            ax.set_title("Eigenvalue gap $\\lambda_2 - \\lambda_3$  (solid)\nvs  SGD defect (purple)", fontsize=10)
        ax.legend(fontsize=6, loc="upper left")

    axes[-1, 0].set_xlabel("Training step")
    axes[-1, 1].set_xlabel("Training step")
    fig.suptitle("Representation eigenvalue gaps vs commutators", fontsize=13, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 2: Scatter plots – gaps vs commutators ──────────────────────

def plot_scatter(all_data, sgd_all, save_path):
    """Scatter: eigenvalue gaps vs commutator values at matched steps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sc = {42: "o", 137: "s", 2024: "^"}
    op_colors = {"add": "#e41a1c", "sub": "#377eb8", "mul": "#4daf4a", "x2_y2": "#984ea3"}

    # Collect scatter data
    gap12_vs_matcomm = []  # (gap, comm, op, seed)
    gap23_vs_matcomm = []
    gap12_vs_sgd = []
    gap23_vs_sgd = []

    for op in TEST_OPS:
        seeds_d = all_data.get(op, {})
        sgd = sgd_all.get(op)

        for seed, d in seeds_d.items():
            gap_steps = d["gap_steps"]
            gap_ratio = d["gap_ratio"]
            comm_steps = d["comm_steps"]
            comm_norms = d["comm_norms"]

            # Interpolate matrix comm to gap steps
            if len(comm_steps) > 1 and len(gap_steps) > 1:
                common, g12, mc = interp_to_common(
                    gap_steps, gap_ratio[:, 0], comm_steps, comm_norms)
                _, g23, mc2 = interp_to_common(
                    gap_steps, gap_ratio[:, 1], comm_steps, comm_norms)
                for i in range(len(common)):
                    gap12_vs_matcomm.append((g12[i], mc[i], op, seed))
                    gap23_vs_matcomm.append((g23[i], mc2[i], op, seed))

            # Interpolate SGD defect to gap steps
            if sgd and seed == 42:  # SGD defect only for seed 42
                sgd_s, sgd_d = sgd
                if len(sgd_s) > 1 and len(gap_steps) > 1:
                    common, g12, sd = interp_to_common(
                        gap_steps, gap_ratio[:, 0], sgd_s, sgd_d)
                    _, g23, sd2 = interp_to_common(
                        gap_steps, gap_ratio[:, 1], sgd_s, sgd_d)
                    for i in range(len(common)):
                        gap12_vs_sgd.append((g12[i], sd[i], op, seed))
                        gap23_vs_sgd.append((g23[i], sd2[i], op, seed))

    def do_scatter(ax, data, xlabel, ylabel, title):
        if not data:
            ax.set_title(title + "\n(no data)")
            return
        for op in TEST_OPS:
            pts = [(x, y) for x, y, o, s in data if o == op]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=8, alpha=0.35, color=op_colors[op], label=op)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    do_scatter(axes[0, 0], gap12_vs_matcomm,
               "$\\lambda_1 - \\lambda_2$", "$\\|[W_Q, W_K]\\|_F$",
               "Gap $\\lambda_1-\\lambda_2$ vs matrix commutator")
    do_scatter(axes[0, 1], gap23_vs_matcomm,
               "$\\lambda_2 - \\lambda_3$", "$\\|[W_Q, W_K]\\|_F$",
               "Gap $\\lambda_2-\\lambda_3$ vs matrix commutator")

    # SGD scatter with log-y
    for idx, (data, gap_label) in enumerate([(gap12_vs_sgd, "$\\lambda_1-\\lambda_2$"),
                                              (gap23_vs_sgd, "$\\lambda_2-\\lambda_3$")]):
        ax = axes[1, idx]
        if data:
            for op in TEST_OPS:
                pts = [(x, y) for x, y, o, s in data if o == op]
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, s=8, alpha=0.35, color=op_colors[op], label=op)
            ax.set_yscale("log")
        ax.set_xlabel(gap_label)
        ax.set_ylabel("SGD defect $D$")
        ax.set_title(f"Gap {gap_label} vs SGD defect", fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle("Eigenvalue gaps vs commutators (all ops, all seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 3: Multi-seed summary per operation ─────────────────────────

def plot_multiseed_summary(all_data, sgd_all, save_path):
    """Per-op: 3-panel (gap12, gap23, both commutators) with all seeds."""
    fig = plt.figure(figsize=(18, 4.5 * len(TEST_OPS)))
    gs = GridSpec(len(TEST_OPS), 3, figure=fig, hspace=0.35, wspace=0.35)
    sc = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}

    for row, op in enumerate(TEST_OPS):
        seeds_d = all_data.get(op, {})

        # Panel 1: λ₁-λ₂ over time
        ax = fig.add_subplot(gs[row, 0])
        for seed, d in sorted(seeds_d.items()):
            c = sc[seed]
            ax.plot(d["gap_steps"], d["gap_ratio"][:, 0], color=c, lw=1.5,
                    label=f"s{seed}")
            if d["grok_step"]:
                ax.axvline(d["grok_step"], color=c, ls=":", alpha=0.4)
        ax.set_ylabel(f"{op}\n$\\lambda_1-\\lambda_2$")
        ax.legend(fontsize=6)
        if row == 0:
            ax.set_title("$\\lambda_1 - \\lambda_2$ (variance ratio gap)", fontsize=10)
        if row == len(TEST_OPS) - 1:
            ax.set_xlabel("Step")

        # Panel 2: λ₂-λ₃ over time
        ax = fig.add_subplot(gs[row, 1])
        for seed, d in sorted(seeds_d.items()):
            c = sc[seed]
            ax.plot(d["gap_steps"], d["gap_ratio"][:, 1], color=c, lw=1.5,
                    label=f"s{seed}")
            if d["grok_step"]:
                ax.axvline(d["grok_step"], color=c, ls=":", alpha=0.4)
        ax.set_ylabel("$\\lambda_2-\\lambda_3$")
        ax.legend(fontsize=6)
        if row == 0:
            ax.set_title("$\\lambda_2 - \\lambda_3$ (variance ratio gap)", fontsize=10)
        if row == len(TEST_OPS) - 1:
            ax.set_xlabel("Step")

        # Panel 3: Both commutators (normalized for comparison)
        ax = fig.add_subplot(gs[row, 2])
        for seed, d in sorted(seeds_d.items()):
            c = sc[seed]
            cn = d["comm_norms"]
            if cn.max() > 0:
                cn_norm = cn / cn.max()
            else:
                cn_norm = cn
            ax.plot(d["comm_steps"], cn_norm, color=c, lw=1.2,
                    label=f"mat s{seed}")
            if d["grok_step"]:
                ax.axvline(d["grok_step"], color=c, ls=":", alpha=0.4)

        sgd = sgd_all.get(op)
        if sgd:
            sgd_s, sgd_d = sgd
            sgd_norm = sgd_d / sgd_d.max() if sgd_d.max() > 0 else sgd_d
            ax.plot(sgd_s, sgd_norm, color="#9467bd", lw=2, ls="--",
                    label="SGD (s42)", alpha=0.7)

        ax.set_ylabel("Normalized")
        ax.legend(fontsize=6)
        if row == 0:
            ax.set_title("Matrix comm (solid) + SGD defect (dashed)", fontsize=10)
        if row == len(TEST_OPS) - 1:
            ax.set_xlabel("Step")

    fig.suptitle("Eigenvalue gaps and commutators across operations and seeds",
                 fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 4: Phase-aligned correlation ──────────────────────────────

def plot_phase_correlation(all_data, sgd_all, save_path):
    """Correlation between gap dynamics and commutator dynamics in pre-grok
    vs post-grok phases."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    op_colors = {"add": "#e41a1c", "sub": "#377eb8", "mul": "#4daf4a", "x2_y2": "#984ea3"}

    pre_g12_mc, post_g12_mc = [], []  # (gap, comm, op) pre/post grok
    pre_g23_mc, post_g23_mc = [], []

    for op in TEST_OPS:
        seeds_d = all_data.get(op, {})
        for seed, d in seeds_d.items():
            grok = d["grok_step"]
            if grok is None:
                continue
            gap_steps = d["gap_steps"]
            gap_ratio = d["gap_ratio"]
            comm_steps = d["comm_steps"]
            comm_norms = d["comm_norms"]

            if len(comm_steps) < 2 or len(gap_steps) < 2:
                continue

            common, g12, mc = interp_to_common(
                gap_steps, gap_ratio[:, 0], comm_steps, comm_norms)
            _, g23, mc2 = interp_to_common(
                gap_steps, gap_ratio[:, 1], comm_steps, comm_norms)

            for i in range(len(common)):
                entry12 = (g12[i], mc[i], op)
                entry23 = (g23[i], mc2[i], op)
                if common[i] < grok:
                    pre_g12_mc.append(entry12)
                    pre_g23_mc.append(entry23)
                else:
                    post_g12_mc.append(entry12)
                    post_g23_mc.append(entry23)

    def do_phase_scatter(ax, data, xlabel, ylabel, title):
        if not data:
            ax.set_title(title + " (no data)")
            return
        for op in TEST_OPS:
            pts = [(x, y) for x, y, o in data if o == op]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=10, alpha=0.4, color=op_colors[op], label=op)
                # Compute Pearson r
                if len(xs) > 2:
                    r = np.corrcoef(xs, ys)[0, 1]
                    ax.annotate(f"{op}: r={r:.2f}", xy=(0.02, 0.98 - 0.06 * TEST_OPS.index(op)),
                                xycoords="axes fraction", fontsize=7, color=op_colors[op])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, loc="lower right")

    do_phase_scatter(axes[0, 0], pre_g12_mc,
                     "$\\lambda_1-\\lambda_2$", "$\\|[W_Q,W_K]\\|_F$",
                     "PRE-grok: gap₁₂ vs matrix comm")
    do_phase_scatter(axes[0, 1], post_g12_mc,
                     "$\\lambda_1-\\lambda_2$", "$\\|[W_Q,W_K]\\|_F$",
                     "POST-grok: gap₁₂ vs matrix comm")
    do_phase_scatter(axes[1, 0], pre_g23_mc,
                     "$\\lambda_2-\\lambda_3$", "$\\|[W_Q,W_K]\\|_F$",
                     "PRE-grok: gap₂₃ vs matrix comm")
    do_phase_scatter(axes[1, 1], post_g23_mc,
                     "$\\lambda_2-\\lambda_3$", "$\\|[W_Q,W_K]\\|_F$",
                     "POST-grok: gap₂₃ vs matrix comm")

    fig.suptitle("Phase-separated correlations: eigenvalue gaps vs matrix commutator",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 5: Layer comparison ─────────────────────────────────────────

def plot_layer_comparison(save_path):
    """Compare eigenvalue gaps from layer 0 vs layer 1."""
    sgd_all = load_sgd_defect()
    sc = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}

    fig, axes = plt.subplots(len(TEST_OPS), 2, figsize=(14, 3.5 * len(TEST_OPS)))
    if len(TEST_OPS) == 1:
        axes = axes.reshape(1, 2)

    for row, op in enumerate(TEST_OPS):
        for li, layer_idx in enumerate([0, 1]):
            ax = axes[row, li]
            for seed in SEEDS:
                data = load_run(op, 1.0, seed)
                if data is None or not data.get("grokked", False):
                    continue
                logs = data["attn_logs"]
                steps, raw_ev, ratios, gap_raw, gap_ratio = compute_eigenvalue_gaps(
                    logs, layer_idx=layer_idx)
                c = sc[seed]
                ax.plot(steps, gap_ratio[:, 0], color=c, lw=1.5,
                        label=f"$\\lambda_1-\\lambda_2$ s{seed}")
                ax.plot(steps, gap_ratio[:, 1], color=c, lw=1, ls="--",
                        label=f"$\\lambda_2-\\lambda_3$ s{seed}", alpha=0.6)
                grok = find_grok_step(data)
                if grok:
                    ax.axvline(grok, color=c, ls=":", alpha=0.3, lw=0.8)

            ax.set_ylabel(f"{op}" if li == 0 else "")
            if row == 0:
                ax.set_title(f"Layer {layer_idx} eigenvalue gaps", fontsize=10)
            if row == len(TEST_OPS) - 1:
                ax.set_xlabel("Step")
            ax.legend(fontsize=5, ncol=2)

    fig.suptitle("Eigenvalue gaps: Layer 0 vs Layer 1", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("EIGENVALUE GAPS vs COMMUTATORS")
    print("=" * 72)

    sgd_all = load_sgd_defect()
    print(f"SGD defect data: {list(sgd_all.keys())}")

    # Compute all eigenvalue gaps + matrix commutators
    all_data = {}  # {op: {seed: {gap_steps, gap_raw, gap_ratio, comm_steps, comm_norms, grok_step}}}

    for op in TEST_OPS:
        all_data[op] = {}
        for seed in SEEDS:
            data = load_run(op, 1.0, seed)
            if data is None or not data.get("grokked", False):
                print(f"  {op} s{seed}: skipped")
                continue

            logs = data["attn_logs"]
            gap_steps, raw_ev, ratios, gap_raw, gap_ratio = compute_eigenvalue_gaps(logs)
            comm_steps, comm_norms = compute_matrix_commutator(data)
            grok = find_grok_step(data)

            all_data[op][seed] = dict(
                gap_steps=gap_steps, raw_eigvals=raw_ev, ratios=ratios,
                gap_raw=gap_raw, gap_ratio=gap_ratio,
                comm_steps=comm_steps, comm_norms=comm_norms,
                grok_step=grok,
            )
            print(f"  {op} s{seed}: {len(gap_steps)} PCA windows, "
                  f"gap12={gap_ratio[-1, 0]:.3f}, gap23={gap_ratio[-1, 1]:.3f}, "
                  f"grok@{grok}")

    # ── Generate figures ──
    print("\nGenerating figures...")

    plot_timeseries(all_data, sgd_all,
                    OUT_DIR / "figEG1_eiggap_vs_commutators_timeseries.png")

    plot_scatter(all_data, sgd_all,
                 OUT_DIR / "figEG2_eiggap_vs_commutators_scatter.png")

    plot_multiseed_summary(all_data, sgd_all,
                           OUT_DIR / "figEG3_eiggap_multiseed_summary.png")

    plot_phase_correlation(all_data, sgd_all,
                           OUT_DIR / "figEG4_eiggap_phase_correlation.png")

    plot_layer_comparison(OUT_DIR / "figEG5_eiggap_layer_comparison.png")

    # ── Save results ──
    save_data = {}
    for op in TEST_OPS:
        for seed, d in all_data[op].items():
            save_data[(op, seed)] = {
                "gap_steps": d["gap_steps"],
                "gap_raw": d["gap_raw"],
                "gap_ratio": d["gap_ratio"],
                "raw_eigvals": d["raw_eigvals"],
                "ratios": d["ratios"],
                "comm_steps": d["comm_steps"],
                "comm_norms": d["comm_norms"],
                "grok_step": d["grok_step"],
            }
    torch.save(save_data, OUT_DIR / "eigenvalue_gap_results.pt")
    print(f"\nSaved: eigenvalue_gap_results.pt")

    # ── Print summary statistics ──
    print(f"\n{'=' * 72}")
    print("SUMMARY: Eigenvalue gap statistics at final window")
    print(f"{'=' * 72}")
    print(f"{'Op':10s} {'Seed':>6s} {'λ₁-λ₂':>10s} {'λ₂-λ₃':>10s} {'λ₁/Σ':>8s} {'λ₂/Σ':>8s} {'λ₃/Σ':>8s}")
    for op in TEST_OPS:
        for seed in sorted(all_data[op].keys()):
            d = all_data[op][seed]
            r = d["ratios"][-1]
            g = d["gap_ratio"][-1]
            print(f"  {op:8s} s{seed:>4d}  {g[0]:10.4f} {g[1]:10.4f} "
                  f"{r[0]:8.4f} {r[1]:8.4f} {r[2]:8.4f}")


if __name__ == "__main__":
    main()
