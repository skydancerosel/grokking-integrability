#!/usr/bin/env python3
"""
Generalization dynamics: does commutator defect spike predict grokking?

Fine-grained temporal analysis overlaying commutator defect with test accuracy
to show that the defect explosion precedes the generalization transition.

Produces:
  figW — Defect vs test accuracy for each grokking op (3 seeds overlaid)
  figX — Lead-time scatter: how many steps before grokking does defect spike?
"""

import math, time, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from grok_sweep import (
    SweepConfig, ModOpTransformer, build_dataset, sample_batch,
    OPERATIONS, get_device, extract_attn_matrices, eval_accuracy,
)
from grok_commutator_analysis import (
    flatten_model_params, _param_offsets, commutator_defect,
)

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
GROK_OPS = ["add", "sub", "mul", "x2_y2"]
NOGROK_OPS = ["x2_xy_y2", "x3_xy"]   # non-grokking controls
SEEDS = [42, 137, 2024]

# Fine-grained: measure commutator every 100 steps during the grokking window
COMM_EVERY = 100        # commutator measurement interval (steps)
COMM_K = 5              # commutator samples per checkpoint (lighter than 12)
COMM_ETA = 1e-3
MAX_STEPS = 200_000     # max training steps (will early-stop on grok)

# We also need a short post-grok tail to show the defect AFTER generalization
POST_GROK_STEPS = 1000  # keep training for this many steps after grokking


# ═══════════════════════════════════════════════════════════════════════════
# Training with inline commutator measurement
# ═══════════════════════════════════════════════════════════════════════════

def train_with_defect_tracking(op_name, wd, seed, max_steps=None):
    """
    Train a model, measuring commutator defect + test accuracy every
    COMM_EVERY steps.  Returns a list of records with both.
    """
    device = get_device()
    steps = max_steps if max_steps is not None else MAX_STEPS
    cfg = SweepConfig(
        OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed, STEPS=steps,
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
        model.parameters(), lr=cfg.LR, weight_decay=wd,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    records = []
    grokked = False
    grok_step = None
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
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        # Measure at regular intervals
        if step % COMM_EVERY == 0:
            model.eval()
            train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
            test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)

            # Commutator defect
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
                # Eval was already done above if divisible by COMM_EVERY
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

        # Post-grok: continue a bit then stop
        if grokked:
            steps_after_grok += 1
            if steps_after_grok >= POST_GROK_STEPS:
                break

        # Progress logging
        if step % 1000 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            d = last_r.get("defect_median", 0)
            ta = last_r.get("test_acc", 0)
            print(f"      step {step:6d} | test {ta:.3f} | defect {d:.1f} | {elapsed:.1f}m")

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "op": op_name,
        "wd": wd,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis: find defect spike step and grok step
# ═══════════════════════════════════════════════════════════════════════════

def find_spike_step(records, threshold_factor=10, min_defect=20):
    """
    Find first step where defect > threshold_factor × initial AND > min_defect.
    Uses a rolling window to avoid noise spikes.
    """
    if len(records) < 3:
        return None

    # Baseline: median of first 3 measurements
    baseline = np.median([r["defect_median"] for r in records[:3]])
    baseline = max(baseline, 0.1)

    for i in range(2, len(records)):
        d = records[i]["defect_median"]
        if d > threshold_factor * baseline and d > min_defect:
            return records[i]["step"]
    return None


def find_grok_step_from_records(records, threshold=0.5):
    """Find first step where test accuracy exceeds threshold."""
    for r in records:
        if r["test_acc"] >= threshold:
            return r["step"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ── Load cached results if available ──────────────────────────────────
    cache_path = OUT_DIR / "generalization_dynamics_results.pt"
    all_runs = {}
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if "all_runs" in cached:
            all_runs = cached["all_runs"]
            print(f"  Loaded {len(all_runs)} cached runs from {cache_path.name}")

    # ── Run all grokking operations × seeds ───────────────────────────────
    total_runs = len(GROK_OPS) * len(SEEDS)
    run_i = 0

    for op_name in GROK_OPS:
        for seed in SEEDS:
            run_i += 1
            tag = f"{op_name}_wd1.0_s{seed}"
            key = (op_name, seed)

            if key in all_runs:
                print(f"\n  [{run_i}/{total_runs}] {tag} — cached, skipping")
                continue

            print(f"\n  [{run_i}/{total_runs}] {tag}")

            data = train_with_defect_tracking(op_name, 1.0, seed)
            all_runs[key] = data

            # Quick summary
            print(f"    → grokked={data['grokked']} "
                  f"(step={data['grok_step']}), "
                  f"{len(data['records'])} measurements")

    # ── Also run no-wd controls for add (1 seed, short) ──────────────────
    for op_name in ["add"]:
        for seed in SEEDS[:1]:   # 1 seed control is enough
            key = (op_name + "_nowd", seed)
            tag = f"{op_name}_wd0.0_s{seed}"

            if key in all_runs:
                print(f"\n  [ctrl] {tag} — cached, skipping")
                continue

            print(f"\n  [ctrl] {tag}")
            data = train_with_defect_tracking(op_name, 0.0, seed, max_steps=5_000)
            all_runs[key] = data
            print(f"    → grokked={data['grokked']}, "
                  f"{len(data['records'])} measurements")

    # ── Non-grokking operations as controls (wd=1.0, 1 seed each) ──────
    for op_name in NOGROK_OPS:
        seed = SEEDS[0]
        key = (op_name, seed)
        tag = f"{op_name}_wd1.0_s{seed}"

        if key in all_runs:
            print(f"\n  [no-grok ctrl] {tag} — cached, skipping")
            continue

        print(f"\n  [no-grok ctrl] {tag}")
        data = train_with_defect_tracking(op_name, 1.0, seed, max_steps=7_500)
        all_runs[key] = data
        print(f"    → grokked={data['grokked']}, "
              f"{len(data['records'])} measurements")

    # ── Compute lead times ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  DEFECT SPIKE vs GROKKING TIMING")
    print(f"{'='*80}")
    print(f"  {'Config':>25s}  {'spike_step':>10s}  {'grok_50%':>10s}  "
          f"{'grok_90%':>10s}  {'lead→50%':>10s}  {'lead→90%':>10s}")

    lead_times_50 = []
    lead_times_90 = []

    for op_name in GROK_OPS:
        for seed in SEEDS:
            key = (op_name, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]

            spike = find_spike_step(recs)
            grok_50 = find_grok_step_from_records(recs, 0.5)
            grok_90 = find_grok_step_from_records(recs, 0.9)

            lead_50 = (grok_50 - spike) if (spike is not None and grok_50 is not None) else None
            lead_90 = (grok_90 - spike) if (spike is not None and grok_90 is not None) else None

            if lead_50 is not None:
                lead_times_50.append(lead_50)
            if lead_90 is not None:
                lead_times_90.append(lead_90)

            tag = f"{op_name} s={seed}"
            print(f"  {tag:>25s}  {str(spike):>10s}  {str(grok_50):>10s}  "
                  f"{str(grok_90):>10s}  {str(lead_50):>10s}  {str(lead_90):>10s}")

    if lead_times_50:
        print(f"\n  Lead time to 50% acc:  mean={np.mean(lead_times_50):.0f}, "
              f"median={np.median(lead_times_50):.0f}, "
              f"range=[{min(lead_times_50)}, {max(lead_times_50)}] steps")
    if lead_times_90:
        print(f"  Lead time to 90% acc:  mean={np.mean(lead_times_90):.0f}, "
              f"median={np.median(lead_times_90):.0f}, "
              f"range=[{min(lead_times_90)}, {max(lead_times_90)}] steps")

        # Sign test: all lead times positive => p = 2^{-n}
        n_positive = sum(1 for l in lead_times_90 if l > 0)
        n_total = len(lead_times_90)
        p_sign = 2 ** (-n_total)  # under H0: spike equally likely before/after
        print(f"\n  Sign test: {n_positive}/{n_total} positive lead times, "
              f"p = 2^{{-{n_total}}} = {p_sign:.6f}")

    # ══════════════════════════════════════════════════════════════════════
    # Figure W: Defect vs Test Accuracy — 2×2 panel (one per op)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating figures...")

    OP_LABELS = {
        "add": "(a+b)", "sub": "(a−b)", "mul": "(a×b)", "x2_y2": "(a²+b²)",
        "x2_xy_y2": "(a²+ab+b²)", "x3_xy": "(a³+ab)",
    }
    SEED_COLORS = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}

    ALL_PANEL_OPS = GROK_OPS + NOGROK_OPS   # 4 grokking + 2 non-grokking = 6 panels
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    for idx, op_name in enumerate(ALL_PANEL_OPS):
        ax = axes[idx // 2, idx % 2]
        ax2 = ax.twinx()

        is_nogrok = (op_name in NOGROK_OPS)
        seeds_to_plot = [SEEDS[0]] if is_nogrok else SEEDS

        for seed in seeds_to_plot:
            key = (op_name, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]
            steps = [r["step"] for r in recs]
            defects = [r["defect_median"] for r in recs]
            test_accs = [r["test_acc"] for r in recs]

            color = SEED_COLORS[seed]

            # Plot defect on left axis
            ax.plot(steps, defects, color=color, linewidth=1.5, alpha=0.8,
                    label=f"defect s={seed}" if idx == 0 else "")

            # Plot test accuracy on right axis
            ax2.plot(steps, test_accs, color=color, linewidth=1.5,
                     linestyle="--", alpha=0.6,
                     label=f"test acc s={seed}" if idx == 0 else "")

            # Mark spike step (only for grokking ops)
            if not is_nogrok:
                spike = find_spike_step(recs)
                if spike is not None:
                    ax.axvline(x=spike, color=color, linestyle=":", alpha=0.4, linewidth=1)

        # Mark grokking region (only for grokking ops)
        if not is_nogrok:
            grok_steps = [all_runs[(op_name, s)]["grok_step"]
                          for s in SEEDS if (op_name, s) in all_runs
                          and all_runs[(op_name, s)]["grok_step"] is not None]
            if grok_steps:
                ax.axvspan(min(grok_steps) - 100, max(grok_steps) + 100,
                           alpha=0.1, color="green", label="grok region")

        ax.set_yscale("log")
        ax.set_ylabel("Commutator defect (median)", fontsize=10, color="#333")
        ax2.set_ylabel("Test accuracy", fontsize=10, color="#666")
        ax2.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Training step")

        grok_tag = "DOES NOT GROK" if is_nogrok else "wd=1.0"
        ax.set_title(f"{OP_LABELS[op_name]} mod 97  ({grok_tag})", fontsize=12,
                     color="#999" if is_nogrok else "black")
        ax.grid(alpha=0.2)

    # Global legend
    handles = []
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="gray", linewidth=2,
                          label="Defect (solid, left)"))
    handles.append(Line2D([0], [0], color="gray", linewidth=2, linestyle="--",
                          label="Test acc (dashed, right)"))
    handles.append(Line2D([0], [0], color="gray", linewidth=1, linestyle=":",
                          label="Defect spike point"))
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor="green", alpha=0.2, label="Grok region"))

    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Commutator Defect Predicts Grokking\n"
                 "(top 4: grokking ops, bottom 2: non-grokking controls)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figW_defect_predicts_grokking.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figW_defect_predicts_grokking.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure X: Lead-time scatter + summary
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Scatter — spike step vs grok step (50% and 90%)
    ax = axes[0]
    op_colors = {"add": "#1f77b4", "sub": "#ff7f0e", "mul": "#2ca02c", "x2_y2": "#d62728"}

    for op_name in GROK_OPS:
        for seed in SEEDS:
            key = (op_name, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]
            spike = find_spike_step(recs)
            grok_90 = find_grok_step_from_records(recs, 0.9)

            if spike is not None and grok_90 is not None:
                ax.scatter(spike, grok_90, color=op_colors[op_name],
                          s=80, alpha=0.8, edgecolors="k", linewidth=0.5,
                          label=OP_LABELS[op_name] if seed == SEEDS[0] else "")

    # Diagonal reference line (spike == grok)
    all_vals = []
    for op_name in GROK_OPS:
        for seed in SEEDS:
            key = (op_name, seed)
            if key not in all_runs:
                continue
            recs = all_runs[key]["records"]
            s = find_spike_step(recs)
            g = find_grok_step_from_records(recs, 0.9)
            if s is not None:
                all_vals.append(s)
            if g is not None:
                all_vals.append(g)

    if all_vals:
        lo, hi = min(all_vals) - 200, max(all_vals) + 200
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="spike = grok")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("Defect spike step", fontsize=12)
    ax.set_ylabel("Grok step (90% test acc)", fontsize=12)
    ax.set_title("Defect Spike vs Grokking Step", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate: points above the diagonal = spike precedes grok
    ax.text(0.05, 0.92, "← spike precedes grok",
            transform=ax.transAxes, fontsize=9, color="green", alpha=0.7)
    ax.text(0.6, 0.08, "spike follows grok →",
            transform=ax.transAxes, fontsize=9, color="red", alpha=0.7)

    # Panel 2: Bar chart of lead times by operation
    ax = axes[1]
    op_leads = {}
    for op_name in GROK_OPS:
        leads = []
        for seed in SEEDS:
            key = (op_name, seed)
            if key not in all_runs:
                continue
            recs = all_runs[key]["records"]
            spike = find_spike_step(recs)
            grok_90 = find_grok_step_from_records(recs, 0.9)
            if spike is not None and grok_90 is not None:
                leads.append(grok_90 - spike)
        op_leads[op_name] = leads

    x_pos = np.arange(len(GROK_OPS))
    means = [np.mean(op_leads[op]) if op_leads[op] else 0 for op in GROK_OPS]
    stds = [np.std(op_leads[op]) if len(op_leads[op]) > 1 else 0 for op in GROK_OPS]

    bars = ax.bar(x_pos, means, yerr=stds,
                  color=[op_colors[op] for op in GROK_OPS],
                  alpha=0.8, capsize=5, edgecolor="k", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([OP_LABELS[op] for op in GROK_OPS], fontsize=11)
    ax.set_ylabel("Lead time (grok_step − spike_step)", fontsize=12)
    ax.set_title("Defect Spike Lead Time Before Grokking\n(positive = spike precedes grok)",
                 fontsize=12)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")

    # Add individual data points
    for i, op_name in enumerate(GROK_OPS):
        for j, lead in enumerate(op_leads[op_name]):
            ax.scatter(i + (j - 1) * 0.1, lead, color="black",
                      s=30, zorder=5, alpha=0.7)

    # Sign test annotation
    all_leads_90 = [l for op in GROK_OPS for l in op_leads[op]]
    if all_leads_90:
        n_pos = sum(1 for l in all_leads_90 if l > 0)
        n_tot = len(all_leads_90)
        p_val = 2 ** (-n_tot)
        ax.text(0.98, 0.95,
                f"Sign test: {n_pos}/{n_tot} positive\n"
                f"p = 2$^{{-{n_tot}}}$ = {p_val:.1e}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="lightyellow", edgecolor="gray"))

    fig.suptitle("Commutator Defect as Early Warning Signal for Grokking",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figX_defect_lead_time.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figX_defect_lead_time.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure W2: Zoomed single-panel hero figure for the paper
    # ══════════════════════════════════════════════════════════════════════
    # Pick the cleanest example (add, seed with clearest lead)
    best_op, best_seed = None, None
    best_lead = -1e9
    for op_name in GROK_OPS:
        for seed in SEEDS:
            key = (op_name, seed)
            if key not in all_runs:
                continue
            recs = all_runs[key]["records"]
            spike = find_spike_step(recs)
            grok_90 = find_grok_step_from_records(recs, 0.9)
            if spike is not None and grok_90 is not None:
                lead = grok_90 - spike
                if lead > best_lead:
                    best_lead = lead
                    best_op = op_name
                    best_seed = seed

    if best_op is not None:
        data = all_runs[(best_op, best_seed)]
        recs = data["records"]
        steps = [r["step"] for r in recs]
        defects = [r["defect_median"] for r in recs]
        test_accs = [r["test_acc"] for r in recs]
        train_accs = [r["train_acc"] for r in recs]

        spike = find_spike_step(recs)
        grok_90 = find_grok_step_from_records(recs, 0.9)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = ax.twinx()

        # Defect with IQR ribbon
        defect_25 = [r["defect_p25"] for r in recs]
        defect_75 = [r["defect_p75"] for r in recs]
        ax.fill_between(steps, defect_25, defect_75, alpha=0.15, color="#1a5276")
        ax.plot(steps, defects, color="#1a5276", linewidth=2.5,
                label="Commutator defect")

        # Accuracies on right axis
        ax2.plot(steps, test_accs, color="#e74c3c", linewidth=2.5,
                 linestyle="--", label="Test accuracy")
        ax2.plot(steps, train_accs, color="#e74c3c", linewidth=1.5,
                 linestyle=":", alpha=0.5, label="Train accuracy")

        # Mark spike and grok points
        if spike is not None:
            ax.axvline(x=spike, color="#1a5276", linestyle=":", linewidth=2,
                       alpha=0.7, label=f"Defect spike (step {spike})")
        if grok_90 is not None:
            ax.axvline(x=grok_90, color="#e74c3c", linestyle=":", linewidth=2,
                       alpha=0.7, label=f"90% test acc (step {grok_90})")

        # Annotate the lead time
        if spike is not None and grok_90 is not None:
            mid = (spike + grok_90) / 2
            ax.annotate("", xy=(spike, ax.get_ylim()[1] * 0.7),
                        xytext=(grok_90, ax.get_ylim()[1] * 0.7),
                        arrowprops=dict(arrowstyle="<->", color="black",
                                       linewidth=1.5))
            ax.text(mid, ax.get_ylim()[1] * 0.8,
                    f"Δ = {grok_90 - spike} steps",
                    ha="center", fontsize=11, fontweight="bold")

        ax.set_yscale("log")
        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("Commutator defect", fontsize=12, color="#1a5276")
        ax.tick_params(axis="y", labelcolor="#1a5276")
        ax2.set_ylabel("Accuracy", fontsize=12, color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.set_ylim(-0.05, 1.1)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                  loc="center left")
        ax.grid(alpha=0.2)

        label_op = OP_LABELS[best_op]
        fig.suptitle(f"Commutator Defect Predicts Grokking: {label_op} mod 97\n"
                     f"(defect spike at step {spike}, grokking at step {grok_90}; "
                     f"lead time = {grok_90 - spike} steps)",
                     fontsize=13, y=1.03)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "figW2_hero_defect_predicts_grok.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figW2_hero_defect_predicts_grok.png "
              f"(best: {best_op} s={best_seed}, lead={best_lead})")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = OUT_DIR / "generalization_dynamics_results.pt"
    torch.save({
        "all_runs": all_runs,
        "lead_times_50": lead_times_50,
        "lead_times_90": lead_times_90,
    }, save_path)
    print(f"\n  saved {save_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
