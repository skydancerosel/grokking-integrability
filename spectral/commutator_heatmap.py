#!/usr/bin/env python3
"""
Layer-wise commutator collapse heatmap:
  y-axis: layer (or layer×head)
  x-axis: training step
  color:  ||[W_Q, W_K]||_F

Generates heatmaps for both tri-task and modadd_modmul datasets.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

BASE = Path(__file__).parent
OUT_DIR = BASE / "commutator_heatmaps"
OUT_DIR.mkdir(exist_ok=True)


def compute_heatmap_data(data):
    """Extract layer×step and (layer,head)×step commutator norms."""
    logs = data["attn_logs"]
    n_layers = len(logs[0]["layers"])
    d_head = 32
    n_heads = 4

    steps = []
    layer_comm = []       # [T, n_layers]
    head_comm = []        # [T, n_layers * n_heads]

    for snap in logs:
        steps.append(snap["step"])
        lc, hc = [], []

        for li in range(n_layers):
            WQ = snap["layers"][li]["WQ"].float().numpy()
            WK = snap["layers"][li]["WK"].float().numpy()

            # Full-layer commutator
            comm = WQ @ WK - WK @ WQ
            lc.append(np.linalg.norm(comm, "fro"))

            # Per-head commutator
            for h in range(n_heads):
                s, e = h * d_head, (h + 1) * d_head
                q = WQ[s:e, s:e]
                k = WK[s:e, s:e]
                hc.append(np.linalg.norm(q @ k - k @ q, "fro"))

        layer_comm.append(lc)
        head_comm.append(hc)

    return (np.array(steps),
            np.array(layer_comm).T,    # [n_layers, T]
            np.array(head_comm).T)     # [n_layers*n_heads, T]


def make_heatmap(steps, matrix, y_labels, title, grok_steps, save_path,
                 cmap="magma", figsize=None):
    """
    matrix: [n_rows, T]
    grok_steps: dict {task_name: step}
    """
    n_rows, T = matrix.shape

    if figsize is None:
        figsize = (max(14, T * 0.04), max(3, n_rows * 0.6 + 2))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, origin="lower",
                   interpolation="nearest",
                   extent=[steps[0], steps[-1], -0.5, n_rows - 0.5])

    # Grok lines
    task_colors = {"add": "#00ff88", "mul": "#00ccff", "sq": "#ffff00"}
    task_ls = {"add": "-", "mul": "--", "sq": ":"}
    for task, gs in sorted(grok_steps.items(), key=lambda x: x[1]):
        ax.axvline(gs, color=task_colors.get(task, "white"),
                   ls=task_ls.get(task, "-"),
                   lw=2, alpha=0.9, label=f"grok:{task} @ {gs}")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13, pad=12)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("$\\|[W_Q, W_K]\\|_F$", fontsize=11)

    ax.legend(loc="upper right", fontsize=9,
              facecolor="black", edgecolor="white",
              labelcolor="white", framealpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Tri-task
# ═══════════════════════════════════════════════════════════════════════════

def run_tritask():
    print("=== TRI-TASK HEATMAPS ===")
    seeds = [42, 137, 2024]

    for seed in seeds:
        path = BASE / f"tri_task/results/tritask_wd1_s{seed}.pt"
        if not path.exists():
            print(f"  [SKIP] {path.name}")
            continue

        print(f"  Loading seed={seed}...")
        data = torch.load(path, map_location="cpu", weights_only=False)
        grok_steps = data.get("grok_step", {})

        steps, layer_mat, head_mat = compute_heatmap_data(data)
        del data  # free memory

        n_layers = layer_mat.shape[0]
        n_heads = 4

        # Layer-level heatmap
        layer_labels = [f"Layer {i}" for i in range(n_layers)]
        make_heatmap(
            steps, layer_mat, layer_labels,
            f"Tri-task: Layer-wise commutator collapse  (seed={seed})",
            grok_steps,
            OUT_DIR / f"tritask_layer_heatmap_s{seed}.png",
            figsize=(16, 3.5),
        )

        # Layer×Head heatmap (the one with cascade effect)
        head_labels = [f"L{li}·H{h}" for li in range(n_layers) for h in range(n_heads)]
        make_heatmap(
            steps, head_mat, head_labels,
            f"Tri-task: Layer×Head commutator collapse  (seed={seed})",
            grok_steps,
            OUT_DIR / f"tritask_head_heatmap_s{seed}.png",
            figsize=(16, 6),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ModAdd+ModMul
# ═══════════════════════════════════════════════════════════════════════════

def run_modadd_modmul():
    print("\n=== MODADD+MODMUL HEATMAPS ===")
    seeds = [42, 137, 2024]

    for seed in seeds:
        path = BASE / f"modadd_modmul/results/multitask_s{seed}.pt"
        if not path.exists():
            print(f"  [SKIP] {path.name}")
            continue

        print(f"  Loading seed={seed}...")
        data = torch.load(path, map_location="cpu", weights_only=False)

        grok_steps = {}
        if data.get("grok_step_add"):
            grok_steps["add"] = data["grok_step_add"]
        if data.get("grok_step_mul"):
            grok_steps["mul"] = data["grok_step_mul"]

        steps, layer_mat, head_mat = compute_heatmap_data(data)
        del data

        n_layers = layer_mat.shape[0]

        layer_labels = [f"Layer {i}" for i in range(n_layers)]
        make_heatmap(
            steps, layer_mat, layer_labels,
            f"ModAdd+ModMul: Layer-wise commutator collapse  (seed={seed})",
            grok_steps,
            OUT_DIR / f"modadd_layer_heatmap_s{seed}.png",
            figsize=(16, 3.5),
        )

        head_labels = [f"L{li}·H{h}" for li in range(n_layers) for h in range(4)]
        make_heatmap(
            steps, head_mat, head_labels,
            f"ModAdd+ModMul: Layer×Head commutator collapse  (seed={seed})",
            grok_steps,
            OUT_DIR / f"modadd_head_heatmap_s{seed}.png",
            figsize=(16, 6),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Single-task (for comparison)
# ═══════════════════════════════════════════════════════════════════════════

def run_singletask():
    print("\n=== SINGLE-TASK HEATMAPS ===")
    for op in ["add", "mul", "x2_y2"]:
        path = BASE / f"grok_sweep_results/{op}_wd1.0_s42.pt"
        if not path.exists():
            print(f"  [SKIP] {path.name}")
            continue

        print(f"  Loading {op} s42...")
        data = torch.load(path, map_location="cpu", weights_only=False)

        grok_step = None
        for e in data["metrics"]:
            if e["test_acc"] >= 0.9:
                grok_step = e["step"]
                break
        grok_steps = {op: grok_step} if grok_step else {}

        steps, layer_mat, head_mat = compute_heatmap_data(data)
        del data

        n_layers = layer_mat.shape[0]

        layer_labels = [f"Layer {i}" for i in range(n_layers)]
        make_heatmap(
            steps, layer_mat, layer_labels,
            f"Single-task ({op}): Layer-wise commutator  (seed=42)",
            grok_steps,
            OUT_DIR / f"single_{op}_layer_heatmap_s42.png",
            figsize=(16, 3.5),
        )

        head_labels = [f"L{li}·H{h}" for li in range(n_layers) for h in range(4)]
        make_heatmap(
            steps, head_mat, head_labels,
            f"Single-task ({op}): Layer×Head commutator  (seed=42)",
            grok_steps,
            OUT_DIR / f"single_{op}_head_heatmap_s42.png",
            figsize=(16, 6),
        )


def main():
    run_singletask()
    run_modadd_modmul()
    run_tritask()
    print(f"\nAll heatmaps → {OUT_DIR}/")


if __name__ == "__main__":
    main()
