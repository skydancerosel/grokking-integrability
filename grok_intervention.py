#!/usr/bin/env python3
"""
Causal intervention experiment: does commutator defect CAUSE grokking?

Two experimental arms:
  1A (Induce):  Artificially boost defect → does grokking accelerate?
  1B (Suppress): Suppress defect → does grokking delay/fail?

Two-phase design:
  Phase 1: Train baseline, extract PCA basis B from trajectory
  Phase 2: Re-train with interventions using fixed B from Phase 1

Conditions:
  baseline   — Normal training
  1A-kick    — One-time weight perturbation along defect eigenvector
  1A-noise   — Repeated orthogonal noise injection
  1B-project — Project out orthogonal gradient component
  1B-penalty — Scale down orthogonal gradient component

Produces:
  figI1 — Defect trajectories per condition (3×2 panel)
  figI2 — Test accuracy overlay (hero figure)
  figI3 — Grok step comparison (bar chart)
  figI4 — Summary table (rendered as figure)
  figI5 — Hyperparameter sensitivity (4-panel supplementary)
"""

import math, time, random, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── imports from existing scripts ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from grok_sweep import (
    SweepConfig, ModOpTransformer, build_dataset, sample_batch,
    OPERATIONS, get_device, extract_attn_matrices, eval_accuracy,
)
from grok_commutator_analysis import (
    flatten_model_params, _param_offsets,
    commutator_defect, commutator_defect_median, build_pca_basis,
    train_with_checkpoints,
)
from grok_generalization_dynamics import (
    find_spike_step, find_grok_step_from_records,
)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

OUT_DIR = Path(__file__).parent / "pca_sweep_plots"

OP_NAME = "add"
SEEDS = [42, 137, 2024]

COMM_EVERY = 100          # measure commutator every N steps
COMM_K = 5                # commutator samples per measurement
COMM_ETA = 1e-3
MAX_STEPS = 200_000
POST_GROK_STEPS = 1000    # keep training after grokking

# Intervention timing
T_START = 500              # start interventions after memorization
SWEEP_MAX_STEPS = 15_000   # cap sweep runs (5× typical grok step)

# Primary hyperparameters (one per condition)
PRIMARY_HPARAMS = {
    "baseline":    {},
    "1A-kick":     {"alpha": 10, "t_start": T_START},
    "1A-noise":    {"epsilon": 0.1, "noise_interval": 50, "t_start": T_START},
    "1B-project":  {"strength": 0.5, "t_start": T_START},
    "1B-penalty":  {"lambda_penalty": 0.5, "penalty_interval": 10, "t_start": T_START},
}

CONDITIONS = list(PRIMARY_HPARAMS.keys())

# Hyperparameter sweep grids (supplementary)
SWEEP_GRIDS = {
    "1A-kick":     [{"alpha": a, "t_start": T_START} for a in [1, 5, 10, 20]],
    "1A-noise":    [{"epsilon": e, "noise_interval": 50, "t_start": T_START}
                    for e in [0.01, 0.05, 0.1, 0.5]],
    "1B-project":  [{"strength": s, "t_start": T_START} for s in [0.25, 0.5, 0.75, 1.0]],
    "1B-penalty":  [{"lambda_penalty": l, "penalty_interval": 10, "t_start": T_START}
                    for l in [0.1, 0.3, 0.5, 0.9]],
}

CONDITION_COLORS = {
    "baseline":    "#333333",
    "1A-kick":     "#e74c3c",
    "1A-noise":    "#e67e22",
    "1B-project":  "#2980b9",
    "1B-penalty":  "#8e44ad",
}
CONDITION_LABELS = {
    "baseline":    "Baseline",
    "1A-kick":     "1A: Defect kick",
    "1A-noise":    "1A: Orthogonal noise",
    "1B-project":  "1B: Gradient projection",
    "1B-penalty":  "1B: Gradient penalty",
}
SEED_MARKERS = {42: "o", 137: "s", 2024: "^"}


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def write_params_to_model(model, theta):
    """Write flat parameter vector theta back into model parameters."""
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(theta[offset:offset + n].view_as(p))
            offset += n


def apply_defect_kick(model, batch_fn, device, B, hparams):
    """
    One-time perturbation along the orthogonal defect eigenvector.
    Magnitude: alpha × ||single gradient step||.
    """
    alpha = hparams.get("alpha", 10)

    # Compute commutator delta (K=9 for robust median)
    out = commutator_defect_median(model, batch_fn, device, K=9, eta=COMM_ETA)
    delta = out["median_delta"].to(device)

    # Also get gradient step norm for scale calibration
    _, _, _, normA, normB = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
    grad_step_norm = normA.item()  # ||eta * gA||

    # Extract orthogonal component
    proj_coeffs = B.T @ delta
    delta_proj = B @ proj_coeffs
    delta_perp = delta - delta_proj

    perp_norm = delta_perp.norm()
    if perp_norm < 1e-15:
        print("    WARNING: delta_perp is near-zero, skipping kick")
        return

    direction = delta_perp / perp_norm
    epsilon = alpha * grad_step_norm

    theta = flatten_model_params(model).to(device)
    theta_new = theta + epsilon * direction
    write_params_to_model(model, theta_new)

    print(f"    KICK applied: alpha={alpha}, epsilon={epsilon:.4f}, "
          f"||delta_perp||={perp_norm.item():.4f}, "
          f"grad_step_norm={grad_step_norm:.4f}")


def inject_orthogonal_noise(model, B, hparams, rng):
    """
    Inject Gaussian noise into the subspace orthogonal to the PCA manifold.
    Uses a separate RNG to avoid corrupting training batch sampling.
    """
    epsilon = hparams.get("epsilon", 0.1)
    device = next(model.parameters()).device

    theta = flatten_model_params(model).to(device)

    # Generate random noise using separate RNG
    noise = torch.randn(theta.shape[0], generator=rng, device=device)

    # Project out PCA-parallel component
    noise_proj = B @ (B.T @ noise)
    noise_perp = noise - noise_proj

    # Normalize to desired magnitude
    n_norm = noise_perp.norm()
    if n_norm < 1e-15:
        return
    noise_perp = noise_perp / n_norm * epsilon

    theta_new = theta + noise_perp
    write_params_to_model(model, theta_new)


def project_gradient_to_pca(model, B, strength=1.0):
    """
    After loss.backward(), remove (strength fraction of) the orthogonal
    gradient component, constraining updates toward the PCA manifold.
    """
    grads = []
    params_with_grad = []
    for p in model.parameters():
        if not p.requires_grad or p.grad is None:
            continue
        grads.append(p.grad.flatten())
        params_with_grad.append(p)

    if not grads:
        return

    grad_flat = torch.cat(grads)
    device = grad_flat.device
    B_dev = B.to(device)

    # Decompose
    grad_parallel = B_dev @ (B_dev.T @ grad_flat)
    grad_perp = grad_flat - grad_parallel

    # Suppress orthogonal component
    grad_new = grad_flat - strength * grad_perp

    # Write back
    offset = 0
    for p in params_with_grad:
        n = p.grad.numel()
        p.grad.copy_(grad_new[offset:offset + n].view_as(p.grad))
        offset += n


def scale_orthogonal_gradient(model, B, hparams):
    """
    Scale down the orthogonal gradient component by (1 - lambda_penalty).
    Equivalent to project_gradient_to_pca with strength=lambda_penalty.
    """
    lam = hparams.get("lambda_penalty", 0.5)
    project_gradient_to_pca(model, B, strength=lam)


def hparams_key(hparams):
    """Convert hparams dict to a hashable tuple for caching."""
    return tuple(sorted(hparams.items()))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Baseline training + PCA basis extraction
# ═══════════════════════════════════════════════════════════════════════════

def run_baseline(op_name, seed, checkpoint_every=200):
    """
    Train a baseline model, extract PCA basis and defect trajectory.
    Returns (B, baseline_data) where B is the PCA basis [P, 16].
    """
    device = get_device()
    cfg = SweepConfig(OP_NAME=op_name, WEIGHT_DECAY=1.0, SEED=seed)
    op_info = OPERATIONS[op_name]
    op_fn = op_info["fn"]

    print(f"  Phase 1: Training baseline {op_name} seed={seed}...")

    model, checkpoints, attn_logs, metrics, grokked, train_pairs, test_pairs = \
        train_with_checkpoints(cfg, checkpoint_every=checkpoint_every)

    print(f"    grokked={grokked}, {len(checkpoints)} checkpoints, "
          f"{len(attn_logs)} attn snapshots")

    # Build PCA basis
    print(f"    Building PCA basis...")
    B = build_pca_basis(model, attn_logs, n_components=2, device="cpu")
    if B is not None:
        print(f"    Basis shape: {B.shape}")
        # Verify orthonormality
        BtB = B.T @ B
        ortho_err = (BtB - torch.eye(B.shape[1])).abs().max().item()
        print(f"    Orthonormality check: max|B^T B - I| = {ortho_err:.2e}")
    else:
        print(f"    WARNING: Could not build PCA basis!")

    # Also run defect tracking on the baseline for comparison
    print(f"    Measuring baseline defect trajectory...")
    baseline_records = _measure_defect_trajectory(
        model, cfg, op_fn, op_info, train_pairs, test_pairs, device,
        max_steps=None, grok_step_from_training=checkpoints[-1][0] if grokked else None,
    )

    # Find grok step and spike step
    grok_step = None
    for m in metrics:
        if m["test_acc"] >= cfg.STOP_ACC:
            grok_step = m["step"]
            break
    spike_step = find_spike_step(baseline_records) if baseline_records else None

    baseline_data = {
        "records": baseline_records,
        "grokked": grokked,
        "grok_step": grok_step,
        "spike_step": spike_step,
        "metrics": metrics,
        "n_checkpoints": len(checkpoints),
    }

    return B, baseline_data


def _measure_defect_trajectory(model, cfg, op_fn, op_info, train_pairs,
                                test_pairs, device, max_steps=None,
                                grok_step_from_training=None):
    """
    Re-train a model from scratch measuring defect + accuracy every COMM_EVERY
    steps. This is the baseline's Phase-2 equivalent (no intervention).
    Returns list of records.
    """
    # We actually just run train_with_intervention with condition="baseline"
    # But since we need model state from Phase 1 for PCA, we skip this
    # and let Phase 2 handle the baseline measurement.
    # Return empty — Phase 2 baseline will provide the trajectory.
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Intervention training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_with_intervention(op_name, wd, seed, condition, B, hparams,
                            max_steps=None):
    """
    Train a model with causal intervention, measuring defect + accuracy
    every COMM_EVERY steps.

    Args:
        condition: str in CONDITIONS
        B: PCA basis tensor [P, 16] from Phase 1 baseline (can be None for baseline)
        hparams: dict with intervention-specific hyperparameters
    """
    device = get_device()
    steps = max_steps if max_steps is not None else MAX_STEPS
    cfg = SweepConfig(OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed, STEPS=steps)
    op_info = OPERATIONS[op_name]
    op_fn = op_info["fn"]

    # CRITICAL: Same seed as baseline → same initial weights and data split
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

    B_dev = B.to(device) if B is not None else None

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    # Separate RNG for intervention noise (doesn't corrupt training batches)
    intervention_rng = torch.Generator(device=device)
    intervention_rng.manual_seed(seed + 99999)

    t_start = hparams.get("t_start", T_START)
    kick_applied = False
    records = []
    grokked = False
    grok_step = None
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    # ── Measurement at step 0 ─────────────────────────────────────────
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

    # ── Training loop ─────────────────────────────────────────────────
    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)
        logits = model(a, b)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        # ── INTERVENTION HOOKS (before opt.step()) ────────────────
        if step >= t_start and B_dev is not None:
            if condition == "1A-kick" and not kick_applied:
                apply_defect_kick(model, batch_fn, device, B_dev, hparams)
                kick_applied = True

            elif condition == "1A-noise":
                noise_interval = hparams.get("noise_interval", 50)
                if step % noise_interval == 0:
                    inject_orthogonal_noise(model, B_dev, hparams, intervention_rng)

            elif condition == "1B-project":
                project_gradient_to_pca(model, B_dev, hparams.get("strength", 0.5))

            elif condition == "1B-penalty":
                penalty_interval = hparams.get("penalty_interval", 10)
                if step % penalty_interval == 0:
                    scale_orthogonal_gradient(model, B_dev, hparams)

        opt.step()

        # ── Measurement ───────────────────────────────────────────
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

        # ── Grokking detection ────────────────────────────────────
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

        # ── Post-grok tail ────────────────────────────────────────
        if grokked:
            steps_after_grok += 1
            if steps_after_grok >= POST_GROK_STEPS:
                break

        # ── Progress logging ──────────────────────────────────────
        if step % 1000 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            d = last_r.get("defect_median", 0)
            ta = last_r.get("test_acc", 0)
            print(f"      step {step:6d} | test {ta:.3f} | defect {d:.1f} | "
                  f"{elapsed:.1f}m")

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "condition": condition,
        "hparams": hparams,
        "op": op_name,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def make_figI1(all_runs, out_dir):
    """figI1: Defect trajectories per condition (3×2 panel)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx // 2, idx % 2]
        ax2 = ax.twinx()

        for seed in SEEDS:
            key = (cond, OP_NAME, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]
            if not recs:
                continue

            steps = [r["step"] for r in recs]
            defects = [r["defect_median"] for r in recs]
            test_accs = [r["test_acc"] for r in recs]

            color = CONDITION_COLORS[cond]
            alpha_val = 0.5 + 0.2 * SEEDS.index(seed)

            ax.plot(steps, defects, color=color, linewidth=1.5, alpha=alpha_val,
                    label=f"s={seed}" if idx == 0 else "")
            ax2.plot(steps, test_accs, color=color, linewidth=1.2,
                     linestyle="--", alpha=alpha_val * 0.7)

            # Mark grok step
            if data["grok_step"] is not None:
                ax.axvline(x=data["grok_step"], color=color, linestyle=":",
                           alpha=0.3, linewidth=1)

        t_start = PRIMARY_HPARAMS[cond].get("t_start", T_START)
        if cond != "baseline":
            ax.axvline(x=t_start, color="gray", linestyle="-.", alpha=0.5,
                       linewidth=1.5, label="Intervention start" if idx == 1 else "")

        ax.set_yscale("log")
        ax.set_ylabel("Defect (median)", fontsize=10)
        ax2.set_ylabel("Test acc", fontsize=10, color="#666")
        ax2.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Training step")
        ax.set_title(f"{CONDITION_LABELS[cond]}", fontsize=12,
                     color=CONDITION_COLORS[cond])
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8)

    # Remove empty 6th panel
    if len(CONDITIONS) < 6:
        axes[2, 1].set_visible(False)

    fig.suptitle("Commutator Defect Under Causal Interventions\n"
                 f"(op={OP_NAME}, wd=1.0, solid=defect, dashed=test acc)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "figI1_intervention_defect_trajectories.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI1_intervention_defect_trajectories.png")


def make_figI2(all_runs, out_dir):
    """figI2: Test accuracy overlay — all conditions on one hero plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for cond in CONDITIONS:
        all_steps = set()
        seed_data = {}
        for seed in SEEDS:
            key = (cond, OP_NAME, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            recs = all_runs[key]["records"]
            if not recs:
                continue
            sd = {r["step"]: r["test_acc"] for r in recs}
            seed_data[seed] = sd
            all_steps.update(sd.keys())

        if not seed_data:
            continue

        steps_sorted = sorted(all_steps)
        means = []
        lows = []
        highs = []
        for s in steps_sorted:
            vals = [sd[s] for sd in seed_data.values() if s in sd]
            if vals:
                means.append(np.mean(vals))
                lows.append(np.min(vals))
                highs.append(np.max(vals))
            else:
                means.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))

        color = CONDITION_COLORS[cond]
        ax.plot(steps_sorted, means, color=color, linewidth=2.5,
                label=CONDITION_LABELS[cond])
        ax.fill_between(steps_sorted, lows, highs, color=color, alpha=0.15)

    ax.axvline(x=T_START, color="gray", linestyle="-.", alpha=0.6,
               linewidth=1.5, label=f"Intervention start (step {T_START})")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=10, loc="center left")
    ax.grid(alpha=0.2)
    ax.set_title(f"Grokking Under Causal Interventions: {OP_NAME} mod 97\n"
                 f"(mean ± range over 3 seeds, wd=1.0)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "figI2_intervention_accuracy_overlay.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI2_intervention_accuracy_overlay.png")


def make_figI3(all_runs, out_dir):
    """figI3: Grok step comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Grok step per condition
    ax = axes[0]
    x_pos = np.arange(len(CONDITIONS))
    grok_steps_by_cond = {}

    for cond in CONDITIONS:
        steps_list = []
        for seed in SEEDS:
            key = (cond, OP_NAME, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            data = all_runs[key]
            if data["grokked"] and data["grok_step"] is not None:
                steps_list.append(data["grok_step"])
        grok_steps_by_cond[cond] = steps_list

    means = []
    stds = []
    colors = []
    for cond in CONDITIONS:
        gs = grok_steps_by_cond[cond]
        if gs:
            means.append(np.mean(gs))
            stds.append(np.std(gs) if len(gs) > 1 else 0)
        else:
            means.append(MAX_STEPS)
            stds.append(0)
        colors.append(CONDITION_COLORS[cond])

    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=5, edgecolor="k", linewidth=0.5)

    # Mark "did not grok" bars
    for i, cond in enumerate(CONDITIONS):
        gs = grok_steps_by_cond[cond]
        n_grok = len(gs)
        n_total = len(SEEDS)
        if n_grok < n_total:
            ax.text(i, means[i] + stds[i] + 100,
                    f"{n_grok}/{n_total} grokked",
                    ha="center", fontsize=8, color="red")

        # Individual data points
        for j, g in enumerate(gs):
            ax.scatter(i + (j - 1) * 0.1, g, color="black", s=25,
                       zorder=5, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Grok step (test acc ≥ 98%)", fontsize=11)
    ax.set_title("Grokking Timing Under Intervention", fontsize=12)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Defect spike step comparison
    ax = axes[1]
    spike_steps_by_cond = {}

    for cond in CONDITIONS:
        spikes = []
        for seed in SEEDS:
            key = (cond, OP_NAME, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            recs = all_runs[key]["records"]
            spike = find_spike_step(recs)
            if spike is not None:
                spikes.append(spike)
        spike_steps_by_cond[cond] = spikes

    means_s = []
    stds_s = []
    for cond in CONDITIONS:
        ss = spike_steps_by_cond[cond]
        if ss:
            means_s.append(np.mean(ss))
            stds_s.append(np.std(ss) if len(ss) > 1 else 0)
        else:
            means_s.append(0)
            stds_s.append(0)

    bars = ax.bar(x_pos, means_s, yerr=stds_s, color=colors, alpha=0.8,
                  capsize=5, edgecolor="k", linewidth=0.5)

    for i, cond in enumerate(CONDITIONS):
        for j, s in enumerate(spike_steps_by_cond[cond]):
            ax.scatter(i + (j - 1) * 0.1, s, color="black", s=25,
                       zorder=5, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Defect spike step", fontsize=11)
    ax.set_title("Defect Spike Timing Under Intervention", fontsize=12)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Causal Intervention Results: {OP_NAME} mod 97",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "figI3_intervention_grok_timing.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI3_intervention_grok_timing.png")


def make_figI4(all_runs, out_dir):
    """figI4: Summary table as figure."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    headers = ["Condition", "Grok Rate", "Mean Grok Step", "Mean Spike Step",
               "Mean Lead Time"]
    rows = []

    for cond in CONDITIONS:
        grok_steps = []
        spike_steps = []
        lead_times = []
        n_grok = 0

        for seed in SEEDS:
            key = (cond, OP_NAME, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]

            if data["grokked"] and data["grok_step"] is not None:
                n_grok += 1
                grok_steps.append(data["grok_step"])

            spike = find_spike_step(recs)
            if spike is not None:
                spike_steps.append(spike)

            grok_90 = find_grok_step_from_records(recs, 0.9)
            if spike is not None and grok_90 is not None:
                lead_times.append(grok_90 - spike)

        row = [
            CONDITION_LABELS[cond],
            f"{n_grok}/{len(SEEDS)}",
            f"{np.mean(grok_steps):.0f}" if grok_steps else "—",
            f"{np.mean(spike_steps):.0f}" if spike_steps else "—",
            f"{np.mean(lead_times):.0f}" if lead_times else "—",
        ]
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by condition
    for i, cond in enumerate(CONDITIONS):
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(
                CONDITION_COLORS[cond] + "22")  # light tint

    fig.suptitle(f"Intervention Summary: {OP_NAME} mod 97",
                 fontsize=14, y=0.95)
    fig.tight_layout()
    fig.savefig(out_dir / "figI4_intervention_summary_table.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI4_intervention_summary_table.png")


def make_figI5(all_runs, sweep_runs, out_dir):
    """figI5: Hyperparameter sensitivity (4-panel supplementary)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sweep_configs = [
        ("1A-kick",    "alpha",          SWEEP_GRIDS["1A-kick"]),
        ("1A-noise",   "epsilon",        SWEEP_GRIDS["1A-noise"]),
        ("1B-project", "strength",       SWEEP_GRIDS["1B-project"]),
        ("1B-penalty", "lambda_penalty", SWEEP_GRIDS["1B-penalty"]),
    ]

    for idx, (cond, param_name, grid) in enumerate(sweep_configs):
        ax = axes[idx // 2, idx % 2]

        param_vals = []
        grok_steps_mean = []
        grok_steps_all = []

        for hp in grid:
            val = hp[param_name]
            param_vals.append(val)
            steps_for_val = []
            for seed in SEEDS:
                key = (cond, OP_NAME, seed, hparams_key(hp))
                if key not in sweep_runs:
                    continue
                data = sweep_runs[key]
                if data["grokked"] and data["grok_step"] is not None:
                    steps_for_val.append(data["grok_step"])
            grok_steps_all.append(steps_for_val)
            grok_steps_mean.append(np.mean(steps_for_val) if steps_for_val else MAX_STEPS)

        # Also add baseline reference
        baseline_steps = []
        for seed in SEEDS:
            key = ("baseline", OP_NAME, seed, hparams_key(PRIMARY_HPARAMS["baseline"]))
            if key in all_runs:
                data = all_runs[key]
                if data["grokked"] and data["grok_step"] is not None:
                    baseline_steps.append(data["grok_step"])
        baseline_mean = np.mean(baseline_steps) if baseline_steps else None

        color = CONDITION_COLORS[cond]
        ax.plot(param_vals, grok_steps_mean, "o-", color=color, linewidth=2,
                markersize=8, label=CONDITION_LABELS[cond])

        # Individual seed points
        for i, steps_for_val in enumerate(grok_steps_all):
            for s in steps_for_val:
                ax.scatter(param_vals[i], s, color=color, s=20, alpha=0.5)

        # Baseline reference line
        if baseline_mean is not None:
            ax.axhline(y=baseline_mean, color="#333", linestyle="--",
                       alpha=0.5, linewidth=1.5, label="Baseline")

        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel("Grok step", fontsize=11)
        ax.set_title(f"{CONDITION_LABELS[cond]}: sensitivity to {param_name}",
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Hyperparameter Sensitivity of Causal Interventions\n"
                 f"(op={OP_NAME}, wd=1.0, 3 seeds)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "figI5_intervention_hparam_sensitivity.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI5_intervention_hparam_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    print(f"Operation: {OP_NAME}, Seeds: {SEEDS}")
    print(f"Intervention start: step {T_START}")

    # ── Load cache ────────────────────────────────────────────────────
    cache_path = OUT_DIR / "intervention_results.pt"
    all_runs = {}
    sweep_runs_cached = {}
    pca_bases = {}
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_runs = cached.get("all_runs", {})
        sweep_runs_cached = cached.get("sweep_runs", {})
        pca_bases = cached.get("pca_bases", {})
        print(f"  Loaded {len(all_runs)} primary + {len(sweep_runs_cached)} sweep cached runs")

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Baseline training + PCA basis extraction
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PHASE 1: Baseline Training + PCA Basis")
    print(f"{'='*70}")

    for seed in SEEDS:
        basis_key = (OP_NAME, seed)
        if basis_key in pca_bases:
            print(f"\n  seed={seed}: PCA basis cached, skipping Phase 1")
            continue

        print(f"\n  seed={seed}:")
        B, baseline_data = run_baseline(OP_NAME, seed)
        pca_bases[basis_key] = B

        # Save incrementally
        torch.save({"all_runs": all_runs, "pca_bases": pca_bases}, cache_path)

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Primary intervention runs
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PHASE 2: Primary Intervention Runs")
    print(f"{'='*70}")

    total_primary = len(CONDITIONS) * len(SEEDS)
    run_i = 0

    for cond in CONDITIONS:
        for seed in SEEDS:
            run_i += 1
            hp = PRIMARY_HPARAMS[cond]
            key = (cond, OP_NAME, seed, hparams_key(hp))

            if key in all_runs:
                data = all_runs[key]
                print(f"\n  [{run_i}/{total_primary}] {cond} s={seed} — "
                      f"cached (grokked={data['grokked']}, "
                      f"step={data.get('grok_step')})")
                continue

            print(f"\n  [{run_i}/{total_primary}] {cond} s={seed}")

            B = pca_bases.get((OP_NAME, seed))
            if B is None and cond != "baseline":
                print(f"    WARNING: No PCA basis for seed={seed}, skipping")
                continue

            data = train_with_intervention(OP_NAME, 1.0, seed, cond, B, hp)
            all_runs[key] = data

            print(f"    → grokked={data['grokked']} "
                  f"(step={data['grok_step']}), "
                  f"{len(data['records'])} measurements")

            # Save incrementally
            torch.save({"all_runs": all_runs, "pca_bases": pca_bases}, cache_path)

    # ══════════════════════════════════════════════════════════════════
    # Phase 2b: Hyperparameter sweep (supplementary)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PHASE 2b: Hyperparameter Sweep")
    print(f"{'='*70}")

    sweep_runs = dict(all_runs)  # include primary runs
    sweep_runs.update(sweep_runs_cached)  # restore cached sweep runs
    total_sweep = sum(len(grid) * len(SEEDS)
                      for grid in SWEEP_GRIDS.values())
    run_i = 0

    for cond, grid in SWEEP_GRIDS.items():
        for hp in grid:
            for seed in SEEDS:
                run_i += 1
                key = (cond, OP_NAME, seed, hparams_key(hp))

                if key in sweep_runs:
                    print(f"  [{run_i}/{total_sweep}] {cond} {hp} s={seed} — cached")
                    continue

                print(f"  [{run_i}/{total_sweep}] {cond} {hp} s={seed}")

                B = pca_bases.get((OP_NAME, seed))
                if B is None:
                    print(f"    WARNING: No PCA basis, skipping")
                    continue

                data = train_with_intervention(
                    OP_NAME, 1.0, seed, cond, B, hp,
                    max_steps=SWEEP_MAX_STEPS,
                )
                sweep_runs[key] = data

                print(f"    → grokked={data['grokked']} "
                      f"(step={data['grok_step']})")

                # Save incrementally
                torch.save({
                    "all_runs": all_runs,
                    "sweep_runs": sweep_runs,
                    "pca_bases": pca_bases,
                }, cache_path)

    # ══════════════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  INTERVENTION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Condition':>20s}  {'Grok':>6s}  {'Grok Step':>10s}  "
          f"{'Spike Step':>10s}  {'Lead Time':>10s}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")

    for cond in CONDITIONS:
        for seed in SEEDS:
            key = (cond, OP_NAME, seed, hparams_key(PRIMARY_HPARAMS[cond]))
            if key not in all_runs:
                continue
            data = all_runs[key]
            recs = data["records"]

            spike = find_spike_step(recs)
            grok_90 = find_grok_step_from_records(recs, 0.9)
            lead = (grok_90 - spike) if (spike is not None and grok_90 is not None) else None

            tag = f"{cond} s={seed}"
            gs = str(data["grok_step"]) if data["grok_step"] else "—"
            ss = str(spike) if spike else "—"
            lt = str(lead) if lead else "—"
            gr = "YES" if data["grokked"] else "no"

            print(f"  {tag:>20s}  {gr:>6s}  {gs:>10s}  {ss:>10s}  {lt:>10s}")

    # ══════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  Generating figures...")
    make_figI1(all_runs, OUT_DIR)
    make_figI2(all_runs, OUT_DIR)
    make_figI3(all_runs, OUT_DIR)
    make_figI4(all_runs, OUT_DIR)
    make_figI5(all_runs, sweep_runs, OUT_DIR)

    # ── Final save ────────────────────────────────────────────────────
    torch.save({
        "all_runs": all_runs,
        "sweep_runs": sweep_runs,
        "pca_bases": pca_bases,
    }, cache_path)
    print(f"\n  saved {cache_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
