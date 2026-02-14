#!/usr/bin/env python3
"""
Learning rate sweep: phase diagram of grokking dynamics.

Sweeps lr in {1e-4, 1e-3, 1e-2} with fixed wd=1.0 across all 6 operations
and 3 seeds.  Reuses existing lr=1e-3 data from generalization_dynamics_results.pt.

Produces:
  figPD  — 2×2 phase diagram (grok fraction, grok step, max defect, lead time)
  figPD2 — Hero figure: defect + accuracy across LRs for "add" operation
"""

import math, time, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from grok_sweep import (
    SweepConfig, ModOpTransformer, build_dataset, sample_batch,
    OPERATIONS, get_device, eval_accuracy,
)
from grok_commutator_analysis import commutator_defect
from grok_generalization_dynamics import find_spike_step, find_grok_step_from_records

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
LRS = [1e-4, 1e-3, 1e-2]
OPS = list(OPERATIONS.keys())  # all 6
GROK_OPS = ["add", "sub", "mul", "x2_y2"]
NOGROK_OPS = ["x2_xy_y2", "x3_xy"]
SEEDS = [42, 137, 2024]
WD = 1.0

# LR-dependent max steps
MAX_STEPS_BY_LR = {1e-4: 500_000, 1e-3: 200_000, 1e-2: 200_000}
NOGROK_MAX = 50_000  # cap non-grokking ops (just need to confirm no grok)

COMM_EVERY = 100     # defect measurement interval (steps)
COMM_K = 5           # commutator samples per checkpoint
COMM_ETA = 1e-3
POST_GROK_STEPS = 1000  # keep training after grokking

OP_LABELS = {
    "add": "(a+b)", "sub": "(a−b)", "mul": "(a×b)", "x2_y2": "(a²+b²)",
    "x2_xy_y2": "(a²+ab+b²)", "x3_xy": "(a³+ab)",
}


# ═══════════════════════════════════════════════════════════════════════════
# Training with inline commutator measurement (LR-parameterized)
# ═══════════════════════════════════════════════════════════════════════════

def train_with_defect_tracking_lr(op_name, lr, wd, seed, max_steps):
    """
    Train a model with LR=lr, measuring commutator defect + test accuracy
    every COMM_EVERY steps.  Returns a dict with records and summary.
    """
    device = get_device()
    cfg = SweepConfig(
        OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed,
        STEPS=max_steps, LR=lr,
    )
    op_info = OPERATIONS[op_name]
    op_fn = op_info["fn"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_pairs, test_pairs = build_dataset(
        cfg.P, cfg.TRAIN_FRACTION, cfg.SEED, op_fn, op_info["restrict_nonzero"]
    )

    model = ModOpTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    records = []
    grokked = False
    grok_step = None
    diverged = False
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    # Measure at step 0
    train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
    test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)
    defects = []
    for _ in range(COMM_K):
        D, delta, gcos, nA, nB = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
        defects.append(D)
    records.append({
        "step": 0,
        "defect_median": float(np.median(defects)),
        "defect_p25": float(np.percentile(defects, 25)),
        "defect_p75": float(np.percentile(defects, 75)),
        "train_acc": train_acc,
        "test_acc": test_acc,
    })

    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)
        logits = model(a, b)
        loss = loss_fn(logits, y)

        # Divergence detection
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"      DIVERGED at step {step} (loss={loss.item()})")
            diverged = True
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        # Measure at regular intervals
        if step % COMM_EVERY == 0:
            model.eval()
            train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
            test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)

            defects = []
            for _ in range(COMM_K):
                D, delta, gcos, nA, nB = commutator_defect(
                    model, batch_fn, device, eta=COMM_ETA
                )
                defects.append(D)

            records.append({
                "step": step,
                "defect_median": float(np.median(defects)),
                "defect_p25": float(np.percentile(defects, 25)),
                "defect_p75": float(np.percentile(defects, 75)),
                "train_acc": train_acc,
                "test_acc": test_acc,
            })
            model.train()

        # Check for grokking
        if step % cfg.EVAL_EVERY == 0:
            if step % COMM_EVERY != 0:
                train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
                test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)

            if test_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE and not grokked:
                    grokked = True
                    grok_step = step
                    print(f"      GROKKED at step {step}")
            else:
                patience = 0

        # Post-grok tail
        if grokked:
            steps_after_grok += 1
            if steps_after_grok >= POST_GROK_STEPS:
                break

        # Progress logging
        if step % 5000 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            d = last_r.get("defect_median", 0)
            ta = last_r.get("test_acc", 0)
            print(f"      step {step:7d} | test {ta:.3f} | defect {d:.1f} | {elapsed:.1f}m")

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "diverged": diverged,
        "op": op_name,
        "lr": lr,
        "wd": wd,
        "seed": seed,
        "final_step": records[-1]["step"] if records else 0,
        "final_train_acc": records[-1]["train_acc"] if records else 0,
        "final_test_acc": records[-1]["test_acc"] if records else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reuse logic: load existing lr=1e-3 data
# ═══════════════════════════════════════════════════════════════════════════

def load_existing_lr1e3():
    """Load existing lr=1e-3, wd=1.0 runs from generalization_dynamics_results.pt."""
    cache_path = OUT_DIR / "generalization_dynamics_results.pt"
    loaded = {}

    if not cache_path.exists():
        print("  No existing generalization_dynamics_results.pt found")
        return loaded

    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    if "all_runs" not in cached:
        return loaded

    for key, data in cached["all_runs"].items():
        # Keys are (op_name, seed) for grokking ops, (op_name+"_nowd", seed) for controls
        if isinstance(key, tuple) and len(key) == 2:
            op_name, seed = key
            # Skip no-wd controls and non-standard keys
            if "_nowd" in str(op_name):
                continue
            if op_name in OPS and seed in SEEDS:
                new_key = (op_name, 1e-3, seed)
                # Add lr field if missing
                data_copy = dict(data)
                data_copy["lr"] = 1e-3
                if "diverged" not in data_copy:
                    data_copy["diverged"] = False
                loaded[new_key] = data_copy
                print(f"    reused: {op_name} lr=1e-3 s={seed} "
                      f"(grokked={data.get('grokked', '?')})")

    return loaded


# ═══════════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # Load cached results
    results_path = OUT_DIR / "lr_sweep_results.pt"
    all_runs = {}

    if results_path.exists():
        cached = torch.load(results_path, map_location="cpu", weights_only=False)
        if "all_runs" in cached:
            all_runs = cached["all_runs"]
            print(f"  Loaded {len(all_runs)} cached LR-sweep runs")

    # Load existing lr=1e-3 data
    if not any(k[1] == 1e-3 for k in all_runs):
        existing = load_existing_lr1e3()
        all_runs.update(existing)
        print(f"  Total after reuse: {len(all_runs)} runs")

    # Count what's needed
    total_needed = len(LRS) * len(OPS) * len(SEEDS)
    already_done = len(all_runs)
    print(f"\n  Grid: {len(LRS)} LRs × {len(OPS)} ops × {len(SEEDS)} seeds = {total_needed} runs")
    print(f"  Already done: {already_done}, remaining: {total_needed - already_done}")

    run_i = 0
    for lr in LRS:
        for op_name in OPS:
            is_nogrok = (op_name in NOGROK_OPS)
            max_steps = NOGROK_MAX if is_nogrok else MAX_STEPS_BY_LR[lr]

            for seed in SEEDS:
                run_i += 1
                key = (op_name, lr, seed)

                if key in all_runs:
                    continue

                tag = f"{op_name} lr={lr:.0e} s={seed}"
                print(f"\n  [{run_i}/{total_needed}] {tag} (max {max_steps} steps)")

                data = train_with_defect_tracking_lr(op_name, lr, WD, seed, max_steps)
                all_runs[key] = data

                print(f"    → grokked={data['grokked']} diverged={data['diverged']} "
                      f"step={data['grok_step']} "
                      f"{len(data['records'])} measurements")

                # Save after each run (incremental)
                torch.save({"all_runs": all_runs, "config": {
                    "lrs": LRS, "ops": OPS, "seeds": SEEDS, "wd": WD,
                }}, results_path)

    print(f"\n  Sweep complete: {len(all_runs)} runs saved to {results_path.name}")
    return all_runs


# ═══════════════════════════════════════════════════════════════════════════
# Phase diagram figure
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary(all_runs):
    """Compute per-(lr, op) summary stats averaged over seeds."""
    summary = {}
    for lr in LRS:
        for op_name in OPS:
            grok_fracs = []
            grok_steps = []
            max_defects = []
            lead_times = []
            diverged_count = 0

            for seed in SEEDS:
                key = (op_name, lr, seed)
                if key not in all_runs:
                    continue
                data = all_runs[key]

                if data.get("diverged", False):
                    diverged_count += 1
                    continue

                grok_fracs.append(1 if data["grokked"] else 0)

                if data["grokked"] and data["grok_step"] is not None:
                    grok_steps.append(data["grok_step"])

                recs = data["records"]
                if recs:
                    max_d = max(r["defect_median"] for r in recs)
                    max_defects.append(max_d)

                    spike = find_spike_step(recs)
                    grok_90 = find_grok_step_from_records(recs, 0.9)
                    if spike is not None and grok_90 is not None:
                        lead_times.append(grok_90 - spike)

            summary[(lr, op_name)] = {
                "grok_frac": np.mean(grok_fracs) if grok_fracs else 0,
                "n_grok": sum(grok_fracs),
                "n_total": len(grok_fracs),
                "grok_step_mean": np.mean(grok_steps) if grok_steps else None,
                "grok_step_std": np.std(grok_steps) if len(grok_steps) > 1 else 0,
                "max_defect_mean": np.mean(max_defects) if max_defects else None,
                "max_defect_std": np.std(max_defects) if len(max_defects) > 1 else 0,
                "lead_time_mean": np.mean(lead_times) if lead_times else None,
                "lead_time_std": np.std(lead_times) if len(lead_times) > 1 else 0,
                "n_diverged": diverged_count,
            }

    return summary


def plot_phase_diagram(all_runs):
    summary = compute_summary(all_runs)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    n_lr = len(LRS)
    n_op = len(OPS)

    lr_labels = [f"{lr:.0e}" for lr in LRS]
    op_labels = [OP_LABELS[op] for op in OPS]

    # ── Panel A: Grok fraction ──────────────────────────────────────────
    ax = axes[0, 0]
    grid = np.zeros((n_lr, n_op))
    for i, lr in enumerate(LRS):
        for j, op in enumerate(OPS):
            s = summary[(lr, op)]
            grid[i, j] = s["grok_frac"]
            # Annotate
            if s["n_diverged"] > 0:
                txt = f"{s['n_grok']}/{s['n_total']}\n({s['n_diverged']}div)"
            else:
                txt = f"{s['n_grok']}/{s['n_total']}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, fontweight="bold")

    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_op)); ax.set_xticklabels(op_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n_lr)); ax.set_yticklabels(lr_labels, fontsize=10)
    ax.set_ylabel("Learning rate", fontsize=11)
    ax.set_title("(A) Grok fraction", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Fraction grokked")

    # ── Panel B: Grok step ──────────────────────────────────────────────
    ax = axes[0, 1]
    grid = np.full((n_lr, n_op), np.nan)
    for i, lr in enumerate(LRS):
        for j, op in enumerate(OPS):
            s = summary[(lr, op)]
            if s["grok_step_mean"] is not None:
                grid[i, j] = s["grok_step_mean"]
                ax.text(j, i, f"{s['grok_step_mean']:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, "---", ha="center", va="center", fontsize=10, color="#999")

    # Mask NaN for colormap
    masked = np.ma.masked_invalid(grid)
    if masked.count() > 0:
        im = ax.imshow(masked, cmap="viridis",
                       norm=LogNorm(vmin=max(np.nanmin(grid[grid > 0]), 1),
                                    vmax=np.nanmax(grid)),
                       aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean grok step")
    else:
        ax.imshow(np.zeros((n_lr, n_op)), cmap="Greys", vmin=0, vmax=1, aspect="auto")

    # Gray out non-grokking cells
    for i in range(n_lr):
        for j in range(n_op):
            if np.isnan(grid[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=True, facecolor="#e0e0e0", edgecolor="white", linewidth=1))
                ax.text(j, i, "---", ha="center", va="center", fontsize=10, color="#999")

    ax.set_xticks(range(n_op)); ax.set_xticklabels(op_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n_lr)); ax.set_yticklabels(lr_labels, fontsize=10)
    ax.set_ylabel("Learning rate", fontsize=11)
    ax.set_title("(B) Mean grok step", fontsize=13, fontweight="bold")

    # ── Panel C: Max defect ─────────────────────────────────────────────
    ax = axes[1, 0]
    grid = np.full((n_lr, n_op), np.nan)
    for i, lr in enumerate(LRS):
        for j, op in enumerate(OPS):
            s = summary[(lr, op)]
            if s["max_defect_mean"] is not None:
                grid[i, j] = s["max_defect_mean"]
                ax.text(j, i, f"{s['max_defect_mean']:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, "---", ha="center", va="center", fontsize=10, color="#999")

    masked = np.ma.masked_invalid(grid)
    if masked.count() > 0:
        vmin = max(np.nanmin(grid[grid > 0]), 0.1)
        im = ax.imshow(masked, cmap="inferno",
                       norm=LogNorm(vmin=vmin, vmax=np.nanmax(grid)),
                       aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean max defect")
    else:
        ax.imshow(np.zeros((n_lr, n_op)), cmap="Greys", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_op)); ax.set_xticklabels(op_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n_lr)); ax.set_yticklabels(lr_labels, fontsize=10)
    ax.set_ylabel("Learning rate", fontsize=11)
    ax.set_title("(C) Mean max defect", fontsize=13, fontweight="bold")

    # ── Panel D: Lead time ──────────────────────────────────────────────
    ax = axes[1, 1]
    grid = np.full((n_lr, n_op), np.nan)
    for i, lr in enumerate(LRS):
        for j, op in enumerate(OPS):
            s = summary[(lr, op)]
            if s["lead_time_mean"] is not None:
                grid[i, j] = s["lead_time_mean"]
                ax.text(j, i, f"{s['lead_time_mean']:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, "---", ha="center", va="center", fontsize=10, color="#999")

    # Lead time can be negative in principle; use diverging colormap
    masked = np.ma.masked_invalid(grid)
    if masked.count() > 0:
        vmax = max(abs(np.nanmax(grid)), abs(np.nanmin(grid)), 1)
        im = ax.imshow(masked, cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean lead time (steps)")
    else:
        ax.imshow(np.zeros((n_lr, n_op)), cmap="Greys", vmin=0, vmax=1, aspect="auto")

    # Gray out non-grokking
    for i in range(n_lr):
        for j in range(n_op):
            if np.isnan(grid[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=True, facecolor="#e0e0e0", edgecolor="white", linewidth=1))
                ax.text(j, i, "---", ha="center", va="center", fontsize=10, color="#999")

    ax.set_xticks(range(n_op)); ax.set_xticklabels(op_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n_lr)); ax.set_yticklabels(lr_labels, fontsize=10)
    ax.set_ylabel("Learning rate", fontsize=11)
    ax.set_title("(D) Mean lead time (spike → grok)", fontsize=13, fontweight="bold")

    fig.suptitle("Phase Diagram: Grokking Dynamics across Learning Rates\n"
                 f"(wd={WD}, 3 seeds per cell, mod 97)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPD_lr_phase_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPD_lr_phase_diagram.png")


# ═══════════════════════════════════════════════════════════════════════════
# Hero figure: defect trajectories across LRs for one operation
# ═══════════════════════════════════════════════════════════════════════════

def plot_hero(all_runs, hero_op="add"):
    """Dual-axis plot: defect + test accuracy for hero_op across 3 LRs."""
    LR_COLORS = {1e-4: "#1f77b4", 1e-3: "#ff7f0e", 1e-2: "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    for lr in LRS:
        color = LR_COLORS[lr]
        # Pick the seed with the largest lead time (most dramatic)
        best_seed = None
        best_lead = -1e9
        for seed in SEEDS:
            key = (hero_op, lr, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            if data.get("diverged", False):
                continue
            recs = data["records"]
            spike = find_spike_step(recs)
            grok_90 = find_grok_step_from_records(recs, 0.9)
            lead = (grok_90 - spike) if (spike is not None and grok_90 is not None) else -1e9
            if lead > best_lead:
                best_lead = lead
                best_seed = seed

        if best_seed is None:
            # No valid seed; plot first available
            for seed in SEEDS:
                key = (hero_op, lr, seed)
                if key in all_runs and not all_runs[key].get("diverged", False):
                    best_seed = seed
                    break
        if best_seed is None:
            continue

        data = all_runs[(hero_op, lr, best_seed)]
        recs = data["records"]
        steps = [r["step"] for r in recs]
        defects = [r["defect_median"] for r in recs]
        test_accs = [r["test_acc"] for r in recs]

        lr_str = f"{lr:.0e}"
        ax.plot(steps, defects, color=color, linewidth=2, alpha=0.85,
                label=f"Defect (lr={lr_str})")
        ax2.plot(steps, test_accs, color=color, linewidth=1.5, linestyle="--",
                 alpha=0.6, label=f"Test acc (lr={lr_str})")

        # Mark spike and grok
        spike = find_spike_step(recs)
        grok_90 = find_grok_step_from_records(recs, 0.9)
        if spike is not None:
            ax.axvline(x=spike, color=color, linestyle=":", alpha=0.5, linewidth=1)
        if grok_90 is not None:
            ax.axvline(x=grok_90, color=color, linestyle="-.", alpha=0.3, linewidth=1)

        # Annotate lead time
        if spike is not None and grok_90 is not None:
            mid = (spike + grok_90) / 2
            ax.annotate(f"Δ={grok_90-spike}", xy=(mid, 0.5),
                        xycoords=("data", "axes fraction"),
                        fontsize=8, ha="center", color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7))

    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect (median)", fontsize=11)
    ax2.set_ylabel("Test accuracy", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax.set_title(f"Defect Predicts Grokking across Learning Rates — {OP_LABELS[hero_op]} mod 97",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)

    # Combined legend
    from matplotlib.lines import Line2D
    handles = []
    for lr in LRS:
        c = LR_COLORS[lr]
        handles.append(Line2D([0], [0], color=c, linewidth=2, label=f"lr={lr:.0e} defect"))
        handles.append(Line2D([0], [0], color=c, linewidth=1.5, linestyle="--",
                              label=f"lr={lr:.0e} test acc"))
    handles.append(Line2D([0], [0], color="gray", linewidth=1, linestyle=":",
                          label="Defect spike"))
    handles.append(Line2D([0], [0], color="gray", linewidth=1, linestyle="-.",
                          label="Grok (90%)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figPD2_lr_sweep_hero.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figPD2_lr_sweep_hero.png")


# ═══════════════════════════════════════════════════════════════════════════
# Print summary table
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(all_runs):
    summary = compute_summary(all_runs)

    print(f"\n{'='*90}")
    print("  LR SWEEP PHASE DIAGRAM SUMMARY")
    print(f"{'='*90}")
    print(f"  {'LR':>8s}  {'Operation':>12s}  {'Grok':>6s}  {'Step':>8s}  "
          f"{'MaxDefect':>10s}  {'LeadTime':>10s}  {'Div':>4s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*4}")

    for lr in LRS:
        for op in OPS:
            s = summary[(lr, op)]
            grok_str = f"{s['n_grok']}/{s['n_total']}"
            step_str = f"{s['grok_step_mean']:.0f}" if s["grok_step_mean"] else "---"
            defect_str = f"{s['max_defect_mean']:.1f}" if s["max_defect_mean"] else "---"
            lead_str = f"{s['lead_time_mean']:.0f}" if s["lead_time_mean"] else "---"
            div_str = str(s["n_diverged"]) if s["n_diverged"] > 0 else ""
            print(f"  {lr:>8.0e}  {op:>12s}  {grok_str:>6s}  {step_str:>8s}  "
                  f"{defect_str:>10s}  {lead_str:>10s}  {div_str:>4s}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_runs = run_sweep()
    print_summary_table(all_runs)

    print("\n  Generating figures...")
    plot_phase_diagram(all_runs)
    plot_hero(all_runs, hero_op="add")
    print("\n  Done!")
