#!/usr/bin/env python3
"""
Phase portrait: (σ₁-σ₂) vs ||[W_Q, W_K]||_F

The single figure that shows grokking as a trajectory through a geometric
phase space, not just curves over time.

x-axis: σ₁-σ₂ of W_Q (spectral gap = mode competition vs dominance)
y-axis: ||[W_Q, W_K]||_F (algebraic non-commutativity)
color:  training step (or test accuracy)
markers: start, SGD spike, comm peak, grok

For grokking operations this should reveal a characteristic loop:
  init → rise in commutator → near-degenerate region →
  rightward (one mode wins) → downward (comm collapse) → grok
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

SWEEP_DIR = Path(__file__).parent / "grok_sweep_results"
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
OUT_DIR.mkdir(exist_ok=True)

TEST_OPS = ["add", "sub", "mul", "x2_y2"]
SEEDS = [42, 137, 2024]


def load_run(op, wd, seed):
    path = SWEEP_DIR / f"{op}_wd{wd}_s{seed}.pt"
    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None


def compute_trajectory(data, layer_idx=0):
    """Extract (step, σ₁-σ₂, ||[WQ,WK]||, test_acc) at each checkpoint."""
    logs = data["attn_logs"]
    metrics = data["metrics"]

    # Build step→test_acc lookup
    acc_steps = np.array([e["step"] for e in metrics])
    acc_vals = np.array([e["test_acc"] for e in metrics])

    steps, gaps, comms, test_accs = [], [], [], []
    for snap in logs:
        WQ = snap["layers"][layer_idx]["WQ"].float().numpy()
        WK = snap["layers"][layer_idx]["WK"].float().numpy()
        step = snap["step"]

        SQ = np.linalg.svd(WQ, compute_uv=False)
        gap = SQ[0] - SQ[1]
        comm = np.linalg.norm(WQ @ WK - WK @ WQ, "fro")
        ta = np.interp(step, acc_steps, acc_vals)

        steps.append(step)
        gaps.append(gap)
        comms.append(comm)
        test_accs.append(ta)

    return (np.array(steps), np.array(gaps), np.array(comms),
            np.array(test_accs))


def find_grok_step(data):
    for e in data["metrics"]:
        if e["test_acc"] >= 0.9:
            return e["step"]
    return None


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
        sgd_steps = np.array([c["step"] for c in comm])
        sgd_defect = np.array([c["defect_median"] for c in comm])
        # Find spike
        if len(sgd_steps) >= 3:
            baseline = max(np.median(sgd_defect[:3]), 0.1)
            for i in range(2, len(sgd_steps)):
                if sgd_defect[i] > 10 * baseline and sgd_defect[i] > 20:
                    out[op] = int(sgd_steps[i])
                    break
    return out


def smooth(x, window=3):
    """Simple rolling mean to reduce checkpoint noise."""
    if len(x) <= window:
        return x.copy()
    kernel = np.ones(window) / window
    # Pad to preserve endpoints
    padded = np.concatenate([[x[0]] * (window // 2), x, [x[-1]] * (window // 2)])
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def make_arrow_trajectory(ax, x, y, colors, cmap, norm, lw=2.5, arrow_every=5,
                          smooth_window=0):
    """Draw a trajectory as a colored line with arrow heads showing direction."""
    if smooth_window > 1:
        x = smooth(x, smooth_window)
        y = smooth(y, smooth_window)

    # Colored line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw,
                        capstyle="round", joinstyle="round")
    lc.set_array(colors[:-1])
    ax.add_collection(lc)

    # Direction arrows at regular intervals
    for i in range(arrow_every, len(x) - 1, arrow_every):
        dx = x[min(i + 1, len(x) - 1)] - x[max(i - 1, 0)]
        dy = y[min(i + 1, len(y) - 1)] - y[max(i - 1, 0)]
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-10:
            continue
        # Normalize arrow length for visual clarity
        scale = min(0.3 * length, 0.02)
        dx_n, dy_n = dx / length * scale, dy / length * scale
        color = cmap(norm(colors[i]))
        ax.annotate("", xy=(x[i] + dx_n, y[i] + dy_n),
                     xytext=(x[i], y[i]),
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=1.8, mutation_scale=14))


def annotate_event(ax, x, y, label, marker, color, offset=(10, 10), fontsize=8):
    """Place an event marker with label."""
    ax.plot(x, y, marker=marker, color=color, markersize=10, zorder=10,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=offset,
                fontsize=fontsize, fontweight="bold", color=color,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                zorder=11)


# ── HERO: single operation, big and clear ──────────────────────────────

def add_phase_regions(ax, xlim, ylim):
    """Shade three phase regions in the (σ₁-σ₂, ||[WQ,WK]||) plane."""
    from matplotlib.patches import FancyBboxPatch
    xlo, xhi = xlim
    ylo, yhi = ylim

    # Phase boundaries (data-driven from add s42):
    #   competition:  gap < 0.04  (modes nearly degenerate)
    #   instability:  comm > 10.2 AND gap > 0.04  (high non-commutativity, gap opening)
    #   alignment:    comm < 10.2 AND gap > 0.02  (post-peak collapse toward grok)
    gap_thresh = 0.04
    comm_thresh = 10.2

    # I. Competition (left strip) — purple
    ax.axvspan(xlo, gap_thresh, alpha=0.045, color="#9467bd", zorder=0)
    ax.text(gap_thresh * 0.45, yhi - 0.15 * (yhi - ylo),
            "competition\n$\\sigma_1 \\approx \\sigma_2$",
            fontsize=9, color="#7b5ea7", ha="center", va="top",
            style="italic", alpha=0.8,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # II. Instability (top-right) — red
    ax.fill_between([gap_thresh, xhi], comm_thresh, yhi,
                    alpha=0.045, color="#d62728", zorder=0)
    ax.text(gap_thresh + 0.55 * (xhi - gap_thresh), yhi - 0.06 * (yhi - ylo),
            "instability",
            fontsize=9, color="#c44e52", ha="center", va="top",
            style="italic", alpha=0.8,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # III. Alignment (bottom-right) — green
    ax.fill_between([gap_thresh, xhi], ylo, comm_thresh,
                    alpha=0.045, color="#2ca02c", zorder=0)
    ax.text(gap_thresh + 0.55 * (xhi - gap_thresh), ylo + 0.06 * (yhi - ylo),
            "alignment",
            fontsize=9, color="#2e8b57", ha="center", va="bottom",
            style="italic", alpha=0.8,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Dashed boundary lines
    ax.axvline(gap_thresh, color="#666666", ls="--", lw=0.9, alpha=0.5, zorder=0)
    ax.axhline(comm_thresh, xmin=(gap_thresh - xlo) / (xhi - xlo), xmax=1.0,
               color="#666666", ls="--", lw=0.9, alpha=0.5, zorder=0)


def plot_hero_portrait(save_path):
    """The hero figure: add, seed=42 only. Phase regions + arrows every ~200 steps."""
    sgd_spikes = load_sgd_defect()
    sgd_spike = sgd_spikes.get("add")
    SW = 3

    data = load_run("add", 1.0, 42)
    steps, gaps, comms, test_accs = compute_trajectory(data)
    grok = find_grok_step(data)

    gaps_s = smooth(gaps, SW)
    comms_s = smooth(comms, SW)

    # Checkpoint cadence is ~100 steps, so arrow every 2 checkpoints ≈ 200 steps
    step_cadence = int(np.median(np.diff(steps)))
    arrow_every = max(1, round(200 / step_cadence))

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for panel, (cval, clabel, cmap_name) in enumerate([
        (steps, "Training step", "viridis"),
        (test_accs, "Test accuracy", "RdYlGn"),
    ]):
        ax = axes[panel]
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=cval.min(), vmax=cval.max())

        # Ghost raw
        ax.plot(gaps, comms, color="#cccccc", alpha=0.3, lw=0.5, zorder=1)

        make_arrow_trajectory(ax, gaps_s, comms_s, cval, cmap, norm,
                              lw=3.5, arrow_every=arrow_every)

        # Events
        annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                       offset=(-20, -18))

        if sgd_spike:
            idx = np.argmin(np.abs(steps - sgd_spike))
            annotate_event(ax, gaps_s[idx], comms_s[idx],
                           f"SGD spike\n(step {sgd_spike})", "^",
                           "#2ca02c", offset=(-55, 18))

        mc_peak = np.argmax(comms_s[3:]) + 3
        annotate_event(ax, gaps_s[mc_peak], comms_s[mc_peak],
                       f"comm peak\n(step {steps[mc_peak]})", "D",
                       "#d62728", offset=(12, 12))

        g_min = np.argmin(gaps_s[3:]) + 3
        annotate_event(ax, gaps_s[g_min], comms_s[g_min],
                       f"σ₁≈σ₂\n(step {steps[g_min]})", "v",
                       "#9467bd", offset=(-50, -22))

        if grok:
            idx = np.argmin(np.abs(steps - grok))
            annotate_event(ax, gaps_s[idx], comms_s[idx],
                           f"GROK\n(step {grok})", "*", "#ff7f0e",
                           offset=(12, -22))

        # Set limits before adding phase regions
        ax.autoscale_view()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Add a little padding
        xpad = 0.05 * (xlim[1] - xlim[0])
        ypad = 0.05 * (ylim[1] - ylim[0])
        xlim = (xlim[0] - xpad, xlim[1] + xpad)
        ylim = (ylim[0] - ypad, ylim[1] + ypad)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        add_phase_regions(ax, xlim, ylim)

        ax.set_xlabel("$\\sigma_1 - \\sigma_2$ ($W_Q$)    [spectral gap]",
                       fontsize=13)
        ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$    [non-commutativity]",
                       fontsize=13)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(clabel, fontsize=11)

        ax.set_title(f"colored by {clabel.lower()}", fontsize=12)

    fig.suptitle("Phase portrait of grokking:  add  (seed 42, layer 0)\n"
                 "$x = \\sigma_1 - \\sigma_2$  (spectral gap)  vs  "
                 "$y = \\|[W_Q, W_K]\\|_F$  (non-commutativity)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── All 4 ops × 3 seeds grid ──────────────────────────────────────────

def plot_grid_portrait(save_path):
    """4 ops × 3 seeds, colored by step, with event markers. Smoothed."""
    sgd_spikes = load_sgd_defect()
    SW = 3

    fig, axes = plt.subplots(len(TEST_OPS), 3, figsize=(18, 5 * len(TEST_OPS)))
    if len(TEST_OPS) == 1:
        axes = axes.reshape(1, 3)

    cmap = plt.get_cmap("viridis")

    for row, op in enumerate(TEST_OPS):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            data = load_run(op, 1.0, seed)
            if data is None or not data.get("grokked", False):
                ax.set_title(f"{op} s{seed} (skip)")
                continue

            steps, gaps, comms, test_accs = compute_trajectory(data)
            grok = find_grok_step(data)
            norm = Normalize(vmin=steps.min(), vmax=steps.max())

            gaps_s = smooth(gaps, SW)
            comms_s = smooth(comms, SW)

            # Ghost raw
            ax.plot(gaps, comms, color="gray", alpha=0.1, lw=0.4)

            make_arrow_trajectory(ax, gaps_s, comms_s, steps, cmap, norm,
                                  lw=2.5, arrow_every=4)

            # Events on smoothed trajectory
            annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                           offset=(-12, -12), fontsize=7)

            sgd_spike = sgd_spikes.get(op) if seed == 42 else None
            if sgd_spike:
                idx = np.argmin(np.abs(steps - sgd_spike))
                annotate_event(ax, gaps_s[idx], comms_s[idx], "SGD", "^",
                               "#2ca02c", offset=(-20, 10), fontsize=7)

            mc_peak = np.argmax(comms_s[3:]) + 3
            annotate_event(ax, gaps_s[mc_peak], comms_s[mc_peak], "peak", "D",
                           "#d62728", offset=(8, 8), fontsize=7)

            g_min = np.argmin(gaps_s[3:]) + 3
            annotate_event(ax, gaps_s[g_min], comms_s[g_min], "σ₁≈σ₂", "v",
                           "#9467bd", offset=(-25, -15), fontsize=7)

            if grok:
                idx = np.argmin(np.abs(steps - grok))
                annotate_event(ax, gaps_s[idx], comms_s[idx], "grok", "*",
                               "#ff7f0e", offset=(8, -12), fontsize=7)

            ax.set_xlabel("$\\sigma_1 - \\sigma_2$", fontsize=9)
            ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$", fontsize=9)
            ax.set_title(f"{op}  seed={seed}", fontsize=10)

    fig.suptitle("Phase portraits: spectral gap vs non-commutativity  "
                 "(4 ops × 3 seeds, smoothed)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Grok vs non-grok control ──────────────────────────────────────────

def plot_grok_vs_control(save_path):
    """Same phase space, grokking (wd=1) vs memorizing (wd=0)."""
    SW = 3
    fig, axes = plt.subplots(2, len(TEST_OPS), figsize=(5.5 * len(TEST_OPS), 10))

    cmap = plt.get_cmap("viridis")

    for col, op in enumerate(TEST_OPS):
        for row, (wd, label) in enumerate([(1.0, "grokking (wd=1)"),
                                             (0.0, "memorizing (wd=0)")]):
            ax = axes[row, col]
            data = load_run(op, wd, 42)
            if data is None:
                ax.set_title(f"{op} {label} (no data)")
                continue

            steps, gaps, comms, test_accs = compute_trajectory(data)
            norm = Normalize(vmin=steps.min(), vmax=steps.max())

            gaps_s = smooth(gaps, SW)
            comms_s = smooth(comms, SW)
            ax.plot(gaps, comms, color="gray", alpha=0.1, lw=0.4)

            make_arrow_trajectory(ax, gaps_s, comms_s, steps, cmap, norm,
                                  lw=2, arrow_every=max(1, len(steps) // 15))

            annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                           offset=(-12, -12), fontsize=7)

            grok = None
            if data.get("grokked", False):
                grok = find_grok_step(data)
            if grok:
                idx = np.argmin(np.abs(steps - grok))
                annotate_event(ax, gaps_s[idx], comms_s[idx], "grok", "*",
                               "#ff7f0e", offset=(8, -12), fontsize=7)

            ax.set_xlabel("$\\sigma_1 - \\sigma_2$", fontsize=9)
            ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$", fontsize=9)
            ax.set_title(f"{op}  {label}", fontsize=10)

    fig.suptitle("Phase portrait: grokking vs memorizing\n"
                 "Same phase space ($\\sigma_1-\\sigma_2$ vs $\\|[W_Q,W_K]\\|_F$), "
                 "seed=42, smoothed",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── 3D trajectory (σ₁-σ₂, σ₂-σ₃, ||comm||) ──────────────────────────

def plot_3d_portrait(save_path):
    """3D phase portrait adding σ₂-σ₃ as the third axis."""
    fig = plt.figure(figsize=(18, 7))

    cmap = plt.get_cmap("viridis")

    for col, op in enumerate(TEST_OPS):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        data = load_run(op, 1.0, 42)
        if data is None or not data.get("grokked", False):
            continue

        logs = data["attn_logs"]
        metrics = data["metrics"]
        acc_steps = np.array([e["step"] for e in metrics])
        acc_vals = np.array([e["test_acc"] for e in metrics])

        steps, g12, g23, comms = [], [], [], []
        for snap in logs:
            WQ = snap["layers"][0]["WQ"].float().numpy()
            WK = snap["layers"][0]["WK"].float().numpy()
            SQ = np.linalg.svd(WQ, compute_uv=False)
            steps.append(snap["step"])
            g12.append(SQ[0] - SQ[1])
            g23.append(SQ[1] - SQ[2])
            comms.append(np.linalg.norm(WQ @ WK - WK @ WQ, "fro"))

        steps = np.array(steps)
        g12 = np.array(g12)
        g23 = np.array(g23)
        comms = np.array(comms)
        norm = Normalize(vmin=steps.min(), vmax=steps.max())

        for i in range(len(steps) - 1):
            ax.plot(g12[i:i+2], g23[i:i+2], comms[i:i+2],
                    color=cmap(norm(steps[i])), lw=1.5)

        # Mark grok
        grok = find_grok_step(data)
        if grok:
            idx = np.argmin(np.abs(steps - grok))
            ax.scatter(g12[idx], g23[idx], comms[idx], color="#ff7f0e",
                       s=100, marker="*", zorder=10, edgecolors="white")

        ax.scatter(g12[0], g23[0], comms[0], color="#555555", s=60,
                   marker="o", zorder=10, edgecolors="white")

        ax.set_xlabel("$\\sigma_1-\\sigma_2$", fontsize=8)
        ax.set_ylabel("$\\sigma_2-\\sigma_3$", fontsize=8)
        ax.set_zlabel("$\\|[W_Q,W_K]\\|$", fontsize=8)
        ax.set_title(f"{op}", fontsize=11)
        ax.view_init(elev=25, azim=-45)

    fig.suptitle("3D phase portrait: ($\\sigma_1-\\sigma_2$, "
                 "$\\sigma_2-\\sigma_3$, $\\|[W_Q,W_K]\\|_F$)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("PHASE PORTRAITS")
    print("=" * 72)

    plot_hero_portrait(OUT_DIR / "figPP1_hero_phase_portrait.png")
    plot_grid_portrait(OUT_DIR / "figPP2_grid_phase_portrait.png")
    plot_grok_vs_control(OUT_DIR / "figPP3_grok_vs_control_portrait.png")
    plot_3d_portrait(OUT_DIR / "figPP4_3d_phase_portrait.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
