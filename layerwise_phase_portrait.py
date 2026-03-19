#!/usr/bin/env python3
"""
Layer-wise phase portrait cascade.

For each layer l and each head h, plot:
  x = σ₁(W_Q) - σ₂(W_Q)
  y = ||[W_Q, W_K]||_F

Stacked vertically by layer (and optionally by head within layer).
Generates figures for single-task, modadd_modmul, and tri-task.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

BASE = Path(__file__).parent
OUT_DIR = BASE / "layerwise_phase_portraits"
OUT_DIR.mkdir(exist_ok=True)

TASK_COLORS = {"add": "#e41a1c", "mul": "#377eb8", "sq": "#4daf4a",
               "x2_y2": "#4daf4a", "sub": "#984ea3"}


def smooth(x, window=3):
    if len(x) <= window:
        return x.copy()
    kernel = np.ones(window) / window
    padded = np.concatenate([[x[0]] * (window // 2), x, [x[-1]] * (window // 2)])
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def compute_layer_trajectories(data):
    """For each layer, compute (steps, gap, comm) trajectory.
    Also computes per-head trajectories."""
    logs = data["attn_logs"]
    n_layers = len(logs[0]["layers"])
    d_head = 32
    n_heads = 4

    # Layer-level
    layer_trajs = []
    for li in range(n_layers):
        steps, gaps, comms = [], [], []
        for snap in logs:
            WQ = snap["layers"][li]["WQ"].float().numpy()
            WK = snap["layers"][li]["WK"].float().numpy()
            SQ = np.linalg.svd(WQ, compute_uv=False)
            comm = np.linalg.norm(WQ @ WK - WK @ WQ, "fro")
            steps.append(snap["step"])
            gaps.append(SQ[0] - SQ[1])
            comms.append(comm)
        layer_trajs.append((np.array(steps), np.array(gaps), np.array(comms)))

    # Head-level
    head_trajs = []
    for li in range(n_layers):
        for h in range(n_heads):
            steps, gaps, comms = [], [], []
            for snap in logs:
                WQ = snap["layers"][li]["WQ"].float().numpy()
                WK = snap["layers"][li]["WK"].float().numpy()
                s, e = h * d_head, (h + 1) * d_head
                q = WQ[s:e, s:e]
                k = WK[s:e, s:e]
                sq = np.linalg.svd(q, compute_uv=False)
                comm = np.linalg.norm(q @ k - k @ q, "fro")
                steps.append(snap["step"])
                gaps.append(sq[0] - sq[1])
                comms.append(comm)
            head_trajs.append((li, h, np.array(steps), np.array(gaps), np.array(comms)))

    return layer_trajs, head_trajs


def make_arrow_trajectory(ax, x, y, colors, cmap, norm, lw=2.5, arrow_every=5):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw,
                        capstyle="round", joinstyle="round")
    lc.set_array(colors[:-1])
    ax.add_collection(lc)

    for i in range(arrow_every, len(x) - 1, arrow_every):
        dx = x[min(i + 1, len(x) - 1)] - x[max(i - 1, 0)]
        dy = y[min(i + 1, len(y) - 1)] - y[max(i - 1, 0)]
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-10:
            continue
        scale = min(0.3 * length, 0.02)
        dx_n, dy_n = dx / length * scale, dy / length * scale
        color = cmap(norm(colors[i]))
        ax.annotate("", xy=(x[i] + dx_n, y[i] + dy_n),
                     xytext=(x[i], y[i]),
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=1.5, mutation_scale=12))


def annotate_event(ax, x, y, label, marker, color, offset=(10, 10), fontsize=7):
    ax.plot(x, y, marker=marker, color=color, markersize=8, zorder=10,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=offset,
                fontsize=fontsize, fontweight="bold", color=color,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
                zorder=11)


# ═══════════════════════════════════════════════════════════════════════════
# Layer-level cascade (one row per layer)
# ═══════════════════════════════════════════════════════════════════════════

def plot_layer_cascade(layer_trajs, grok_steps, title, save_path):
    """Stack phase portraits by layer, colored by training step."""
    n_layers = len(layer_trajs)
    SW = 3
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 6 * n_layers),
                             sharex=False, sharey=False)
    if n_layers == 1:
        axes = [axes]
    # Plot top layer on top (reverse order so Layer 1 is top panel)
    axes_ordered = list(reversed(axes))

    for li in range(n_layers):
        ax = axes_ordered[li]
        steps, gaps, comms = layer_trajs[li]
        gaps_s = smooth(gaps, SW)
        comms_s = smooth(comms, SW)
        norm = Normalize(vmin=steps.min(), vmax=steps.max())

        step_cadence = int(np.median(np.diff(steps))) if len(steps) > 1 else 100
        arrow_every = max(1, round(200 / step_cadence))

        # Ghost raw
        ax.plot(gaps, comms, color="#cccccc", alpha=0.25, lw=0.5, zorder=1)

        make_arrow_trajectory(ax, gaps_s, comms_s, steps, cmap, norm,
                              lw=3, arrow_every=arrow_every)

        # Init marker
        annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                       offset=(-15, -12))

        # Comm peak
        mc_peak = np.argmax(comms_s[3:]) + 3
        annotate_event(ax, gaps_s[mc_peak], comms_s[mc_peak],
                       f"peak @{steps[mc_peak]}", "D", "#d62728",
                       offset=(10, 8))

        # Grok markers
        for task, gs in sorted(grok_steps.items(), key=lambda x: x[1]):
            if gs <= steps[-1]:
                idx = np.argmin(np.abs(steps - gs))
                annotate_event(ax, gaps_s[idx], comms_s[idx],
                               f"grok:{task}", "*",
                               TASK_COLORS.get(task, "#ff7f0e"),
                               offset=(10, -14))

        ax.set_ylabel(f"Layer {li}\n$\\|[W_Q, W_K]\\|_F$", fontsize=11)
        ax.set_xlabel("$\\sigma_1 - \\sigma_2$ ($W_Q$)", fontsize=10)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02, label="Step")

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Head-level cascade (one row per layer×head)
# ═══════════════════════════════════════════════════════════════════════════

def plot_head_cascade(head_trajs, grok_steps, title, save_path):
    """Stack phase portraits by layer×head."""
    n_rows = len(head_trajs)
    SW = 3
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows),
                             sharex=False, sharey=False)
    if n_rows == 1:
        axes = [axes]
    # Reverse so top layer is at top of figure
    axes_ordered = list(reversed(axes))

    for idx, (li, h, steps, gaps, comms) in enumerate(head_trajs):
        ax = axes_ordered[idx]
        gaps_s = smooth(gaps, SW)
        comms_s = smooth(comms, SW)
        norm = Normalize(vmin=steps.min(), vmax=steps.max())

        step_cadence = int(np.median(np.diff(steps))) if len(steps) > 1 else 100
        arrow_every = max(1, round(200 / step_cadence))

        ax.plot(gaps, comms, color="#cccccc", alpha=0.25, lw=0.5, zorder=1)
        make_arrow_trajectory(ax, gaps_s, comms_s, steps, cmap, norm,
                              lw=2.5, arrow_every=arrow_every)

        annotate_event(ax, gaps_s[0], comms_s[0], "init", "o", "#555555",
                       offset=(-12, -10), fontsize=6)

        mc_peak = np.argmax(comms_s[3:]) + 3
        annotate_event(ax, gaps_s[mc_peak], comms_s[mc_peak],
                       f"pk@{steps[mc_peak]}", "D", "#d62728",
                       offset=(8, 6), fontsize=6)

        for task, gs in sorted(grok_steps.items(), key=lambda x: x[1]):
            if gs <= steps[-1]:
                i = np.argmin(np.abs(steps - gs))
                annotate_event(ax, gaps_s[i], comms_s[i],
                               f"{task}", "*",
                               TASK_COLORS.get(task, "#ff7f0e"),
                               offset=(8, -10), fontsize=6)

        ax.set_ylabel(f"L{li}·H{h}\n$\\|[W_Q^h,W_K^h]\\|$", fontsize=9)
        ax.set_xlabel("$\\sigma_1 - \\sigma_2$", fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Combined: all layers on one axis (overlay, not stacked)
# ═══════════════════════════════════════════════════════════════════════════

def plot_layer_overlay(layer_trajs, grok_steps, title, save_path):
    """All layers on one plot, different colors per layer."""
    SW = 3
    layer_cmaps = ["Blues", "Oranges"]
    layer_line_colors = ["#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(14, 10))

    for li, (steps, gaps, comms) in enumerate(layer_trajs):
        gaps_s = smooth(gaps, SW)
        comms_s = smooth(comms, SW)
        cmap = plt.get_cmap(layer_cmaps[li % len(layer_cmaps)])
        norm = Normalize(vmin=steps.min(), vmax=steps.max())

        step_cadence = int(np.median(np.diff(steps))) if len(steps) > 1 else 100
        arrow_every = max(1, round(200 / step_cadence))

        # Ghost
        ax.plot(gaps, comms, color=layer_line_colors[li % len(layer_line_colors)],
                alpha=0.08, lw=0.5, zorder=1)

        make_arrow_trajectory(ax, gaps_s, comms_s, steps, cmap, norm,
                              lw=3, arrow_every=arrow_every)

        # Init
        annotate_event(ax, gaps_s[0], comms_s[0], f"L{li} init", "o",
                       layer_line_colors[li % len(layer_line_colors)],
                       offset=(-20, -15 + li * 25))

        # Peak
        mc_peak = np.argmax(comms_s[3:]) + 3
        annotate_event(ax, gaps_s[mc_peak], comms_s[mc_peak],
                       f"L{li} peak\n@{steps[mc_peak]}", "D",
                       layer_line_colors[li % len(layer_line_colors)],
                       offset=(12, 10 - li * 25))

    # Grok markers (on Layer 0 trajectory)
    steps0, gaps0, comms0 = layer_trajs[0]
    gaps0_s = smooth(gaps0, SW)
    comms0_s = smooth(comms0, SW)
    for task, gs in sorted(grok_steps.items(), key=lambda x: x[1]):
        if gs <= steps0[-1]:
            idx = np.argmin(np.abs(steps0 - gs))
            annotate_event(ax, gaps0_s[idx], comms0_s[idx],
                           f"grok:{task}\n@{gs}", "*",
                           TASK_COLORS.get(task, "#ff7f0e"),
                           offset=(12, -20))

    ax.set_xlabel("$\\sigma_1 - \\sigma_2$ ($W_Q$)    [spectral gap]", fontsize=13)
    ax.set_ylabel("$\\|[W_Q, W_K]\\|_F$    [non-commutativity]", fontsize=13)
    ax.set_title(title, fontsize=13)

    # Manual legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=layer_line_colors[i], lw=3,
                       label=f"Layer {i}") for i in range(len(layer_trajs))]
    ax.legend(handles=handles, fontsize=11, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════════

def run_singletask():
    print("=== SINGLE-TASK ===")
    for op in ["add", "mul", "x2_y2"]:
        path = BASE / f"grok_sweep_results/{op}_wd1.0_s42.pt"
        if not path.exists():
            continue
        print(f"  Loading {op}...")
        data = torch.load(path, map_location="cpu", weights_only=False)
        grok_step = None
        for e in data["metrics"]:
            if e["test_acc"] >= 0.9:
                grok_step = e["step"]
                break
        grok_steps = {op: grok_step} if grok_step else {}

        layer_trajs, head_trajs = compute_layer_trajectories(data)
        del data

        plot_layer_cascade(layer_trajs, grok_steps,
                           f"Single-task ({op}): Layer-wise phase portrait cascade  (s42)",
                           OUT_DIR / f"single_{op}_layer_cascade_s42.png")
        plot_head_cascade(head_trajs, grok_steps,
                          f"Single-task ({op}): Head-wise phase portrait cascade  (s42)",
                          OUT_DIR / f"single_{op}_head_cascade_s42.png")
        plot_layer_overlay(layer_trajs, grok_steps,
                           f"Single-task ({op}): Layer overlay  (s42)",
                           OUT_DIR / f"single_{op}_layer_overlay_s42.png")


def run_modadd():
    print("\n=== MODADD+MODMUL ===")
    for seed in [42]:
        path = BASE / f"modadd_modmul/results/multitask_s{seed}.pt"
        if not path.exists():
            continue
        print(f"  Loading seed={seed}...")
        data = torch.load(path, map_location="cpu", weights_only=False)
        grok_steps = {}
        if data.get("grok_step_add"):
            grok_steps["add"] = data["grok_step_add"]
        if data.get("grok_step_mul"):
            grok_steps["mul"] = data["grok_step_mul"]

        layer_trajs, head_trajs = compute_layer_trajectories(data)
        del data

        plot_layer_cascade(layer_trajs, grok_steps,
                           f"ModAdd+ModMul: Layer-wise phase portrait cascade  (s{seed})",
                           OUT_DIR / f"modadd_layer_cascade_s{seed}.png")
        plot_head_cascade(head_trajs, grok_steps,
                          f"ModAdd+ModMul: Head-wise phase portrait cascade  (s{seed})",
                          OUT_DIR / f"modadd_head_cascade_s{seed}.png")
        plot_layer_overlay(layer_trajs, grok_steps,
                           f"ModAdd+ModMul: Layer overlay  (s{seed})",
                           OUT_DIR / f"modadd_layer_overlay_s{seed}.png")


def run_tritask():
    print("\n=== TRI-TASK ===")
    for seed in [42]:
        path = BASE / f"tri_task/results/tritask_wd1_s{seed}.pt"
        if not path.exists():
            continue
        print(f"  Loading seed={seed}...")
        data = torch.load(path, map_location="cpu", weights_only=False)
        grok_steps = data.get("grok_step", {})

        layer_trajs, head_trajs = compute_layer_trajectories(data)
        del data

        plot_layer_cascade(layer_trajs, grok_steps,
                           f"Tri-task: Layer-wise phase portrait cascade  (s{seed})",
                           OUT_DIR / f"tritask_layer_cascade_s{seed}.png")
        plot_head_cascade(head_trajs, grok_steps,
                          f"Tri-task: Head-wise phase portrait cascade  (s{seed})",
                          OUT_DIR / f"tritask_head_cascade_s{seed}.png")
        plot_layer_overlay(layer_trajs, grok_steps,
                           f"Tri-task: Layer overlay  (s{seed})",
                           OUT_DIR / f"tritask_layer_overlay_s{seed}.png")


def main():
    run_singletask()
    run_modadd()
    run_tritask()
    print(f"\nAll → {OUT_DIR}/")


if __name__ == "__main__":
    main()
