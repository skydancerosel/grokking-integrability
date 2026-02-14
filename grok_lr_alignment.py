#!/usr/bin/env python3
"""
LR-dependent commutator alignment analysis.

For each of 3 learning rates, trains a model and measures trajectory-curvature
alignment at 4 strategic checkpoints (early, memorization, defect spike, post-grok).

Produces:
  figPD3 — Alignment vs training phase, faceted by LR (shows sign flip / isotropization)
"""

import math, time, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from grok_sweep import (
    SweepConfig, ModOpTransformer, build_dataset, sample_batch,
    OPERATIONS, get_device, eval_accuracy,
)
from grok_commutator_analysis import (
    commutator_defect, flatten_model_params, train_with_checkpoints,
)
from grok_converse_commutator import (
    mean_alignment_cosine, build_commutator_subspace,
    project_onto_subspace, random_alignment_baseline,
)
from grok_generalization_dynamics import find_spike_step, find_grok_step_from_records

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"

LRS = [1e-4, 1e-3, 1e-2]
OPS = ["add", "mul"]       # hero op + one replication
SEED = 42                   # single seed is sufficient for this analysis
WD = 1.0

N_COMM = 12                # commutator samples per checkpoint (match converse analysis)
COMM_ETA = 1e-3

# LR-dependent training limits
MAX_STEPS_BY_LR = {1e-4: 500_000, 1e-3: 200_000, 1e-2: 200_000}

# Checkpoint density: save every N steps (need enough to pick 4 strategic ones)
CKPT_EVERY_BY_LR = {1e-4: 1000, 1e-3: 100, 1e-2: 50}

# Phase labels
PHASES = ["early", "memorization", "spike", "post-grok"]


# ═══════════════════════════════════════════════════════════════════════════
# Select 4 strategic checkpoints from a set of available checkpoints
# ═══════════════════════════════════════════════════════════════════════════

def select_strategic_checkpoints(checkpoints, metrics, grokked, grok_step_approx=None):
    """
    Pick 4 checkpoint indices corresponding to:
      0. early (5% into training)
      1. memorization (train_acc > 0.95 but test_acc < 0.2)
      2. defect spike region (just before grokking, or 60% through if no grok)
      3. post-grok (after generalization, or end if no grok)

    Returns list of 4 (step, state_dict) tuples.
    """
    steps = [s for s, _ in checkpoints]
    n = len(steps)

    # Build a step → metrics lookup
    met_by_step = {}
    for m in metrics:
        met_by_step[m["step"]] = m

    # Find memorization step: first step where train_acc > 0.95
    mem_step = None
    for m in metrics:
        if m["train_acc"] >= 0.95:
            mem_step = m["step"]
            break

    # Find grok step from metrics
    grok_step = None
    for m in metrics:
        if m["test_acc"] >= 0.90:
            grok_step = m["step"]
            break

    # If we have a known grok step, use it
    if grok_step_approx is not None and grok_step is None:
        grok_step = grok_step_approx

    final_step = steps[-1]

    # Pick the 4 target steps
    targets = []

    # 0. Early: ~5-10% into training, but at least step 200
    early_target = max(200, int(final_step * 0.05))
    targets.append(early_target)

    # 1. Memorization: when train_acc hits 0.95, or 30% if not found
    if mem_step is not None:
        targets.append(mem_step)
    else:
        targets.append(int(final_step * 0.3))

    # 2. Spike: ~80% of way to grok, or 70% through training
    if grok_step is not None:
        targets.append(int(grok_step * 0.85))
    else:
        targets.append(int(final_step * 0.7))

    # 3. Post-grok: grok_step + 500, or end
    if grok_step is not None:
        targets.append(min(grok_step + 500, final_step))
    else:
        targets.append(final_step)

    # Find nearest checkpoint for each target
    selected = []
    for t in targets:
        best_idx = min(range(n), key=lambda i: abs(steps[i] - t))
        selected.append(checkpoints[best_idx])

    # Ensure no duplicates (if checkpoints are sparse)
    seen = set()
    deduped = []
    for s, sd in selected:
        if s not in seen:
            seen.add(s)
            deduped.append((s, sd))

    return deduped


# ═══════════════════════════════════════════════════════════════════════════
# Measure alignment at strategic checkpoints
# ═══════════════════════════════════════════════════════════════════════════

def measure_alignment_at_checkpoints(op_name, lr, wd, seed):
    """
    Train a model at given LR, select 4 strategic checkpoints,
    and measure commutator alignment at each.
    """
    device = get_device()
    max_steps = MAX_STEPS_BY_LR[lr]
    ckpt_every = CKPT_EVERY_BY_LR[lr]

    cfg = SweepConfig(
        OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed,
        STEPS=max_steps, LR=lr,
    )

    print(f"  Training {op_name} lr={lr:.0e} ...")
    model, checkpoints, attn_logs, metrics, grokked, train_pairs, test_pairs = \
        train_with_checkpoints(cfg, checkpoint_every=ckpt_every)
    print(f"    grokked={grokked}, {len(checkpoints)} checkpoints")

    # Select 4 strategic checkpoints
    strategic = select_strategic_checkpoints(checkpoints, metrics, grokked)
    print(f"    Selected {len(strategic)} strategic checkpoints: "
          f"steps={[s for s, _ in strategic]}")

    op_info = OPERATIONS[op_name]
    op_fn = op_info["fn"]

    # Rebuild model for loading
    model = ModOpTransformer(cfg).to(device)

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    results = []
    prev_theta = None

    for ci, (step, sd) in enumerate(strategic):
        model.load_state_dict(sd)
        model.to(device)
        model.eval()

        theta = flatten_model_params(model).cpu()

        # Trajectory step
        if prev_theta is not None:
            dtheta = theta - prev_theta
        else:
            dtheta = None

        # Collect K commutator deltas
        comm_deltas = []
        defects = []
        for _ in range(N_COMM):
            D, delta, gcos, normA, normB = commutator_defect(
                model, batch_fn, device, eta=COMM_ETA
            )
            comm_deltas.append(delta.cpu())
            defects.append(D)

        defect_med = float(np.median(defects))

        # Alignment metrics
        if dtheta is not None and dtheta.norm() > 1e-30:
            mean_cos = mean_alignment_cosine(dtheta, comm_deltas)
            C = build_commutator_subspace(comm_deltas)
            _, _, _, sub_frac = project_onto_subspace(dtheta, C)
        else:
            mean_cos = float("nan")
            sub_frac = float("nan")

        # Get accuracy at this step
        train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
        test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)

        results.append({
            "step": step,
            "defect_median": defect_med,
            "mean_cos": mean_cos,
            "sub_frac": sub_frac,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "dtheta_norm": dtheta.norm().item() if dtheta is not None else 0.0,
            "phase": PHASES[ci] if ci < len(PHASES) else "extra",
        })

        prev_theta = theta

        print(f"      [{PHASES[ci] if ci < len(PHASES) else '?':>14s}] "
              f"step={step:>7d} | defect={defect_med:>8.1f} | "
              f"mean|cos|={mean_cos:.6f} | "
              f"train={train_acc:.3f} test={test_acc:.3f}")

    return {
        "results": results,
        "grokked": grokked,
        "op": op_name,
        "lr": lr,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════

def run_alignment_sweep():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # Get random baseline
    # Total params ~290k
    test_cfg = SweepConfig()
    test_model = ModOpTransformer(test_cfg)
    total_params = sum(p.numel() for p in test_model.parameters())
    rand_cos, rand_frac = random_alignment_baseline(total_params, n_comm=N_COMM)
    print(f"  Random baseline: cos={rand_cos:.6f}, frac={rand_frac:.8f}")
    print(f"  Total params: {total_params}")
    del test_model

    # Check for cached results
    cache_path = OUT_DIR / "lr_alignment_results.pt"
    all_runs = {}
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if "all_runs" in cached:
            all_runs = cached["all_runs"]
            print(f"  Loaded {len(all_runs)} cached alignment runs")

    total = len(LRS) * len(OPS)
    run_i = 0

    for op_name in OPS:
        for lr in LRS:
            run_i += 1
            key = (op_name, lr)

            if key in all_runs:
                print(f"\n  [{run_i}/{total}] {op_name} lr={lr:.0e} — cached")
                continue

            print(f"\n  [{run_i}/{total}] {op_name} lr={lr:.0e}")
            data = measure_alignment_at_checkpoints(op_name, lr, WD, SEED)
            all_runs[key] = data

            # Save incrementally
            torch.save({
                "all_runs": all_runs,
                "rand_cos": rand_cos,
                "rand_frac": rand_frac,
                "total_params": total_params,
            }, cache_path)

    return all_runs, rand_cos, rand_frac


# ═══════════════════════════════════════════════════════════════════════════
# Figure: alignment across LRs and phases
# ═══════════════════════════════════════════════════════════════════════════

def plot_alignment_figure(all_runs, rand_cos):
    """
    Grouped bar chart: mean|cos| at each phase, grouped by LR.
    One panel per operation.
    """
    n_ops = len(OPS)
    fig, axes = plt.subplots(1, n_ops, figsize=(7 * n_ops, 5), squeeze=False)

    LR_COLORS = {1e-4: "#1f77b4", 1e-3: "#ff7f0e", 1e-2: "#2ca02c"}
    bar_width = 0.25

    for oi, op_name in enumerate(OPS):
        ax = axes[0, oi]

        for li, lr in enumerate(LRS):
            key = (op_name, lr)
            if key not in all_runs:
                continue
            data = all_runs[key]
            results = data["results"]

            phases = [r["phase"] for r in results if not np.isnan(r["mean_cos"])]
            cosines = [r["mean_cos"] for r in results if not np.isnan(r["mean_cos"])]
            steps_labels = [f"{r['step']//1000}k" for r in results if not np.isnan(r["mean_cos"])]

            x = np.arange(len(phases))
            offset = (li - 1) * bar_width
            bars = ax.bar(x + offset, cosines, bar_width, label=f"lr={lr:.0e}",
                         color=LR_COLORS[lr], alpha=0.8, edgecolor="white")

            # Add step labels on bars
            for bi, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       steps_labels[bi], ha="center", va="bottom", fontsize=7,
                       rotation=45)

        # Random baseline
        ax.axhline(y=rand_cos, color="black", linestyle=":", linewidth=1.5,
                   alpha=0.7, label="random baseline")

        ax.set_xticks(np.arange(len(PHASES) - 1))  # skip "early" since dtheta=None
        phase_labels = PHASES[1:]  # first phase has no dtheta
        ax.set_xticklabels(phase_labels, fontsize=10)
        ax.set_ylabel("Mean |cos(Δθ, δ)|", fontsize=11)
        ax.set_yscale("log")
        ax.set_title(f"{OPERATIONS[op_name]['label']} mod 97", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Trajectory–Curvature Alignment across Learning Rates\n"
                 "(4 training phases, wd=1.0, seed=42)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPD3_lr_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPD3_lr_alignment.png")


def plot_alignment_trajectory(all_runs, rand_cos):
    """
    Line plot: alignment at each checkpoint step, one line per LR.
    Shows the trajectory of alignment over training.
    """
    n_ops = len(OPS)
    fig, axes = plt.subplots(1, n_ops, figsize=(7 * n_ops, 5), squeeze=False)

    LR_COLORS = {1e-4: "#1f77b4", 1e-3: "#ff7f0e", 1e-2: "#2ca02c"}

    for oi, op_name in enumerate(OPS):
        ax = axes[0, oi]

        for lr in LRS:
            key = (op_name, lr)
            if key not in all_runs:
                continue
            data = all_runs[key]
            results = data["results"]

            # Filter out NaN (first checkpoint has no dtheta)
            valid = [(r["step"], r["mean_cos"], r["phase"])
                     for r in results if not np.isnan(r["mean_cos"])]
            if not valid:
                continue

            steps = [v[0] for v in valid]
            cosines = [v[1] for v in valid]
            phases = [v[2] for v in valid]

            color = LR_COLORS[lr]
            ax.plot(range(len(steps)), cosines, 'o-', color=color,
                    linewidth=2, markersize=8, alpha=0.85,
                    label=f"lr={lr:.0e} ({'grok' if data['grokked'] else 'no grok'})")

            # Annotate with phase names
            for i, (s, c, p) in enumerate(valid):
                ax.annotate(f"{p}\n(step {s//1000}k)" if s >= 1000 else f"{p}\n(step {s})",
                           xy=(i, c), xytext=(0, 12),
                           textcoords="offset points", fontsize=7,
                           ha="center", color=color, alpha=0.7)

        # Random baseline
        ax.axhline(y=rand_cos, color="black", linestyle=":", linewidth=1.5,
                   alpha=0.7, label="random (isotropic)")

        ax.set_xticks(range(3))
        ax.set_xticklabels(PHASES[1:], fontsize=10)
        ax.set_ylabel("Mean |cos(Δθ, δ)|", fontsize=11)
        ax.set_yscale("log")
        ax.set_title(f"{OPERATIONS[op_name]['label']} mod 97", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Trajectory–Curvature Alignment: LR-Dependent Dynamics\n"
                 "(wd=1.0, seed=42, 12 commutator samples per checkpoint)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPD3_lr_alignment_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPD3_lr_alignment_trajectory.png")


def print_alignment_table(all_runs, rand_cos):
    """Print summary table."""
    print(f"\n{'='*90}")
    print("  LR-DEPENDENT ALIGNMENT ANALYSIS")
    print(f"{'='*90}")
    print(f"  Random baseline: mean|cos| = {rand_cos:.6f}")
    print(f"\n  {'Op':>6s}  {'LR':>8s}  {'Phase':>14s}  {'Step':>7s}  "
          f"{'mean|cos|':>10s}  {'ratio':>6s}  {'defect':>8s}  {'test_acc':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*7}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}")

    for op_name in OPS:
        for lr in LRS:
            key = (op_name, lr)
            if key not in all_runs:
                continue
            data = all_runs[key]
            for r in data["results"]:
                cos = r["mean_cos"]
                ratio = cos / rand_cos if not np.isnan(cos) and rand_cos > 0 else float("nan")
                print(f"  {op_name:>6s}  {lr:>8.0e}  {r['phase']:>14s}  {r['step']:>7d}  "
                      f"{cos:>10.6f}  {ratio:>6.1f}x  {r['defect_median']:>8.1f}  "
                      f"{r['test_acc']:>8.3f}")
            print()


def plot_alignment_vs_defect(all_runs, rand_cos):
    """
    Phase portrait: defect (x) vs alignment ratio (y), colored by LR, shaped by op.
    Connected trajectories with arrows show flow mem → spike → post.
    Formal region labels, grok-time star markers.
    """
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe

    LR_COLORS = {1e-4: "#1f77b4", 1e-3: "#ff7f0e", 1e-2: "#2ca02c"}
    LR_LABELS = {1e-4: r"$\eta=10^{-4}$", 1e-3: r"$\eta=10^{-3}$", 1e-2: r"$\eta=10^{-2}$"}
    OP_MARKERS = {"add": "o", "mul": "s"}
    PHASE_LABELS = {"memorization": "mem", "spike": "spike", "post-grok": "post"}

    fig, ax = plt.subplots(figsize=(8, 6))

    # ── Region shading with formal labels ──────────────────────────────
    ax.axhspan(0, 1.0, alpha=0.06, color="#1f77b4", zorder=0)
    ax.axhspan(1.0, 2.8, alpha=0.06, color="#d62728", zorder=0)

    # Random baseline
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.5,
              zorder=1, label="isotropic baseline")

    # ── Plot connected trajectories per (op, lr) ──────────────────────
    for op_name in OPS:
        for lr in LRS:
            key = (op_name, lr)
            if key not in all_runs:
                continue
            data = all_runs[key]

            valid = [(r["defect_median"], r["mean_cos"] / rand_cos, r["phase"],
                      r.get("test_acc", 0.0))
                     for r in data["results"]
                     if not np.isnan(r["mean_cos"]) and r["defect_median"] > 0]
            if not valid:
                continue

            defects = [v[0] for v in valid]
            ratios = [v[1] for v in valid]
            phases = [v[2] for v in valid]
            test_accs = [v[3] for v in valid]

            color = LR_COLORS[lr]
            marker = OP_MARKERS.get(op_name, "^")

            # (A) Connecting line with arrows showing flow direction
            # Draw line segments with arrowheads
            for seg_i in range(len(defects) - 1):
                x0, y0 = defects[seg_i], ratios[seg_i]
                x1, y1 = defects[seg_i + 1], ratios[seg_i + 1]
                # Use log-space midpoint for arrow placement on log x-axis
                xmid = np.exp((np.log(x0) + np.log(x1)) / 2)
                ymid = (y0 + y1) / 2
                ax.annotate("",
                           xy=(x1, y1), xytext=(x0, y0),
                           arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.15",
                                          color=color, lw=1.8, alpha=0.6,
                                          connectionstyle="arc3,rad=0.0"),
                           zorder=2)

            # Scatter points
            for i, (d, r, p, ta) in enumerate(valid):
                # (D) Star marker at post-grok (≥90% acc)
                if p == "post-grok" and ta >= 0.90:
                    # Star behind the main marker
                    ax.scatter(d, r, c=color, marker="*", s=350,
                              alpha=0.5, edgecolors=color, linewidth=0.8, zorder=3)

                ax.scatter(d, r, c=color, marker=marker, s=120,
                          alpha=0.9, edgecolors="k", linewidth=0.6, zorder=4)

                # Phase annotation with smart offset to avoid overlap
                offx = 10 if i == 0 else -10
                offy = 8
                ha = "left" if i == 0 else "right"
                # Adjust for crowded regions
                if lr == 1e-3 and p == "spike":
                    offy = -12
                elif lr == 1e-2 and p == "memorization":
                    offy = -12

                label_text = PHASE_LABELS.get(p, p[:3])
                ax.annotate(label_text,
                           xy=(d, r), xytext=(offx, offy), textcoords="offset points",
                           fontsize=7.5, color=color, fontweight="bold", ha=ha,
                           bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                     ec=color, alpha=0.7, linewidth=0.5),
                           zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Commutator defect (curvature magnitude)", fontsize=12)
    ax.set_ylabel(r"Alignment ratio  (mean$|\cos(\Delta\theta,\delta)|$ / random)", fontsize=11)
    ax.set_ylim(0, 2.8)

    # ── (C) Formal region labels ──────────────────────────────────────
    ax.text(0.97, 0.04, r"$\mathbf{Region\;I}$: Overdamped",
           transform=ax.transAxes, fontsize=9.5, color="#1f77b4", alpha=0.6,
           ha="right", fontstyle="italic",
           path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.text(0.97, 0.97, r"$\mathbf{Region\;II}$: Underdamped",
           transform=ax.transAxes, fontsize=9.5, color="#d62728", alpha=0.6,
           ha="right", va="top", fontstyle="italic",
           path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # ── Legend ────────────────────────────────────────────────────────
    handles = []
    for lr in LRS:
        handles.append(Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=LR_COLORS[lr], markersize=9,
                             markeredgecolor="k", markeredgewidth=0.5,
                             label=LR_LABELS[lr]))
    handles.append(Line2D([], [], marker="o", color="gray", markersize=7,
                         linestyle="None", label=r"$(a\!+\!b)$"))
    handles.append(Line2D([], [], marker="s", color="gray", markersize=7,
                         linestyle="None", label=r"$(a\!\times\!b)$"))
    handles.append(Line2D([], [], marker="*", color="gold", markersize=12,
                         linestyle="None", markeredgecolor="k",
                         markeredgewidth=0.3, label=r"$\geq 90\%$ test acc"))
    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                         alpha=0.5, label="isotropic baseline"))
    ax.legend(handles=handles, fontsize=8.5, loc="upper left",
             framealpha=0.92, edgecolor="gray", ncol=1)
    ax.grid(alpha=0.15, which="both")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPD4_alignment_vs_defect.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPD4_alignment_vs_defect.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_runs, rand_cos, rand_frac = run_alignment_sweep()
    print_alignment_table(all_runs, rand_cos)

    print("\n  Generating figures...")
    plot_alignment_figure(all_runs, rand_cos)
    plot_alignment_trajectory(all_runs, rand_cos)
    plot_alignment_vs_defect(all_runs, rand_cos)
    print("\n  Done!")
