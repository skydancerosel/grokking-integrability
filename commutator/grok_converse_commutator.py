#!/usr/bin/env python3
"""
Converse commutator analysis: project the weight TRAJECTORY onto
commutator (curvature) directions.

Previous result (grok_commutator_analysis.py):
    Commutators are ⊥ to PCA manifold → PCA manifold is integrable.

This script asks the converse:
    Does the weight trajectory AVOID high-curvature directions?

For each checkpoint t:
  1. Compute the weight-trajectory step: dtheta(t) = theta(t) - theta(t-1)
  2. Collect K commutator delta vectors at theta(t)
  3. Orthonormalise the commutator deltas → commutator subspace C(t)
  4. Project dtheta(t) onto C(t)

If grokking finds a flat channel, the trajectory should have LOW alignment
with commutator directions — even though both are nonzero vectors in the
same 290k-dim space.

Produces:
  figO — Alignment cosine: |<dtheta, delta_comm>| / (||dtheta|| * ||delta||)
  figP — Fraction of trajectory step living in commutator subspace
  figQ — Comparison: grok vs no-wd alignment
  figR — Combined: defect magnitude × trajectory-commutator alignment
"""

import math, time, random, sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── imports from existing scripts ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from grok_sweep import (
    SweepConfig, ModOpTransformer, build_dataset, sample_batch,
    OPERATIONS, get_device, extract_attn_matrices, eval_accuracy,
)
from grok_commutator_analysis import (
    flatten_model_params, _param_offsets,
    commutator_defect, train_with_checkpoints,
)

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
GROK_OPS = ["add", "sub", "mul", "x2_y2"]
SEED = 42
CHECKPOINT_EVERY = 200
N_COMM_SAMPLES = 12       # commutator deltas per checkpoint for subspace
COMM_ETA = 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# Core analysis
# ═══════════════════════════════════════════════════════════════════════════

def build_commutator_subspace(deltas, min_sv_ratio=1e-6):
    """
    Given a list of commutator delta vectors [each shape (P,)],
    build an orthonormal basis for the subspace they span.

    Returns C: [P, k] orthonormal columns, or None if degenerate.
    """
    if not deltas:
        return None

    D = torch.stack(deltas, dim=1)  # [P, N]
    # Remove near-zero columns
    norms = D.norm(dim=0)
    good = norms > norms.max() * min_sv_ratio
    if good.sum() < 1:
        return None
    D = D[:, good]

    # QR orthonormalise
    Q, R = torch.linalg.qr(D.float(), mode="reduced")
    # Keep only columns with non-tiny R diagonal
    diag = R.diag().abs()
    keep = diag > diag.max() * min_sv_ratio
    Q = Q[:, keep]

    if Q.shape[1] == 0:
        return None
    return Q


def project_onto_subspace(v, Q):
    """
    Project vector v onto the column-space of Q.
    Returns (proj_norm, resid_norm, v_norm, frac).
    frac = proj_norm / v_norm.
    """
    v = v.float()
    v_norm = v.norm().item()
    if v_norm < 1e-30 or Q is None:
        return 0.0, 0.0, v_norm, 0.0

    coeffs = Q.T @ v            # [k]
    proj = Q @ coeffs           # [P]
    proj_norm = proj.norm().item()
    resid_norm = (v - proj).norm().item()
    frac = proj_norm / v_norm
    return proj_norm, resid_norm, v_norm, frac


def mean_alignment_cosine(v, deltas, eps=1e-30):
    """
    Mean |cos(v, delta_i)| over a list of commutator deltas.
    Measures how aligned the trajectory step is with individual
    curvature directions.
    """
    v = v.float()
    vn = v.norm()
    if vn < eps or not deltas:
        return 0.0

    cosines = []
    for d in deltas:
        d = d.float()
        dn = d.norm()
        if dn < eps:
            continue
        cos = (v @ d).abs() / (vn * dn)
        cosines.append(cos.item())

    if not cosines:
        return 0.0
    return float(np.mean(cosines))


def max_alignment_cosine(v, deltas, eps=1e-30):
    """Max |cos(v, delta_i)| — worst-case alignment."""
    v = v.float()
    vn = v.norm()
    if vn < eps or not deltas:
        return 0.0
    cosines = []
    for d in deltas:
        d = d.float()
        dn = d.norm()
        if dn < eps:
            continue
        cos = (v @ d).abs() / (vn * dn)
        cosines.append(cos.item())
    return float(max(cosines)) if cosines else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Random baseline: expected alignment for random vectors
# ═══════════════════════════════════════════════════════════════════════════

def random_alignment_baseline(dim, n_comm=12, n_trials=200):
    """
    Expected |cos| between a random unit vector and n_comm random unit vectors
    in R^dim.  Analytical: E[|cos|] = sqrt(2/(pi*dim)) for large dim.
    Also returns the expected subspace projection fraction for k-dim subspace.
    """
    # Analytical for isotropic Gaussian
    expected_cos = np.sqrt(2.0 / (np.pi * dim))
    # Expected fraction of norm in k-dim random subspace: k/dim
    expected_frac = n_comm / dim
    return expected_cos, expected_frac


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyse_one_run(op_name, wd, seed=42):
    """
    Train, then at each checkpoint:
      - Compute weight trajectory step dtheta
      - Sample N commutator deltas
      - Measure alignment + subspace projection
    """
    device = get_device()
    steps = 10_000 if wd == 0.0 else 200_000
    ckpt_every = CHECKPOINT_EVERY if wd > 0 else 500

    cfg = SweepConfig(
        OP_NAME=op_name,
        WEIGHT_DECAY=wd,
        SEED=seed,
        STEPS=steps,
    )

    print(f"  Training {op_name} wd={wd}...")
    model, checkpoints, attn_logs, metrics, grokked, train_pairs, test_pairs = \
        train_with_checkpoints(cfg, checkpoint_every=ckpt_every)
    print(f"  grokked={grokked}, {len(checkpoints)} checkpoints")

    op_info = OPERATIONS[cfg.OP_NAME]
    op_fn = op_info["fn"]

    # Rebuild model for loading checkpoints
    model = ModOpTransformer(cfg).to(device)

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    results = []
    prev_theta = None
    total = len(checkpoints)

    # Subsample if too many checkpoints
    if wd == 0.0 and total > 25:
        idx = np.linspace(0, total - 1, 25, dtype=int)
        checkpoints = [checkpoints[i] for i in idx]
        total = len(checkpoints)
        print(f"  Subsampled to {total} checkpoints")

    for ci, (step, sd) in enumerate(checkpoints):
        model.load_state_dict(sd)
        model.to(device)

        # Current flat params
        theta = flatten_model_params(model).cpu()

        # Weight trajectory step
        if prev_theta is not None:
            dtheta = theta - prev_theta
        else:
            dtheta = None

        # Collect commutator deltas
        comm_deltas = []
        defects = []
        for _ in range(N_COMM_SAMPLES):
            D, delta, gcos, normA, normB = commutator_defect(
                model, batch_fn, device, eta=COMM_ETA
            )
            comm_deltas.append(delta.cpu())
            defects.append(D)

        defect_med = float(np.median(defects))

        if dtheta is not None and dtheta.norm() > 1e-30:
            # Build commutator subspace
            C = build_commutator_subspace(comm_deltas)

            # Subspace projection
            _, _, _, sub_frac = project_onto_subspace(dtheta, C)

            # Pairwise alignment
            mean_cos = mean_alignment_cosine(dtheta, comm_deltas)
            max_cos = max_alignment_cosine(dtheta, comm_deltas)
        else:
            sub_frac = float("nan")
            mean_cos = float("nan")
            max_cos = float("nan")

        results.append({
            "step": step,
            "defect_median": defect_med,
            "mean_cos": mean_cos,
            "max_cos": max_cos,
            "sub_frac": sub_frac,
            "dtheta_norm": dtheta.norm().item() if dtheta is not None else 0.0,
        })

        prev_theta = theta

        if (ci + 1) % 5 == 0 or ci == total - 1:
            print(f"      ckpt {ci+1}/{total}: step={step}, "
                  f"defect={defect_med:.2f}, "
                  f"mean|cos|={mean_cos:.6f}, "
                  f"sub_frac={sub_frac:.4f}")

    return {
        "results": results,
        "grokked": grokked,
        "metrics": metrics,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # Get parameter dimension for random baseline
    cfg_tmp = SweepConfig()
    model_tmp = ModOpTransformer(cfg_tmp)
    _, total_params = _param_offsets(model_tmp)
    rand_cos, rand_frac = random_alignment_baseline(total_params, N_COMM_SAMPLES)
    print(f"Parameter space dim: {total_params}")
    print(f"Random baseline: E[|cos|]={rand_cos:.6f}, "
          f"E[subspace frac]={rand_frac:.6f} "
          f"({N_COMM_SAMPLES}/{total_params})")
    del model_tmp

    all_data = {}

    for op_name in GROK_OPS:
        for wd in [1.0, 0.0]:
            tag = f"{op_name}_wd{wd}"
            print(f"\n{'='*70}")
            print(f"  {tag}")
            print(f"{'='*70}")

            data = analyse_one_run(op_name, wd, seed=SEED)
            all_data[(op_name, wd)] = data

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  CONVERSE COMMUTATOR ANALYSIS — TRAJECTORY vs CURVATURE ALIGNMENT")
    print(f"{'='*80}")
    print(f"  Random baseline: E[|cos|]={rand_cos:.6f}, "
          f"E[sub_frac]={rand_frac:.6f}")
    print()
    print(f"  {'Config':>25s}  {'grok':>5s}  {'defect':>8s}  "
          f"{'mean|cos|':>10s}  {'max|cos|':>10s}  {'sub_frac':>10s}  "
          f"{'vs_rand':>8s}")

    for (op, wd), data in sorted(all_data.items()):
        res = data["results"]
        # Use last non-nan checkpoint
        valid = [r for r in res if not np.isnan(r["mean_cos"])]
        if not valid:
            continue
        last = valid[-1]
        ratio = last["mean_cos"] / rand_cos if rand_cos > 0 else float("nan")
        tag = f"{op} wd={wd}"
        print(f"  {tag:>25s}  {'yes' if data['grokked'] else 'no':>5s}  "
              f"{last['defect_median']:8.2f}  "
              f"{last['mean_cos']:10.6f}  {last['max_cos']:10.6f}  "
              f"{last['sub_frac']:10.6f}  {ratio:8.1f}×")

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating figures...")

    colors_op = {
        "add": "#1f77b4", "sub": "#ff7f0e",
        "mul": "#2ca02c", "x2_y2": "#d62728",
    }

    # ── Figure O: Mean alignment cosine over training ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        label_op = OPERATIONS[op_name]["label"]

        for wd, ls, lbl in [(1.0, "-", "wd=1.0 (grok)"),
                             (0.0, "--", "wd=0 (no-wd)")]:
            key = (op_name, wd)
            if key not in all_data:
                continue
            res = all_data[key]["results"]
            steps = [r["step"] for r in res]
            cosines = [r["mean_cos"] for r in res]
            ax.plot(steps, cosines, label=lbl, linewidth=2, linestyle=ls,
                    color=colors_op[op_name] if wd > 0 else "gray")

        # Random baseline
        ax.axhline(y=rand_cos, color="black", linestyle=":",
                    alpha=0.5, linewidth=1.5, label=f"random ({rand_cos:.1e})")

        ax.set_title(f"{label_op} mod 97", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean |cos(dθ, δ_comm)|")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle(
        "Trajectory–Curvature Alignment\n"
        "(low alignment = trajectory avoids high-curvature directions)",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figO_trajectory_alignment.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figO_trajectory_alignment.png")

    # ── Figure P: Subspace projection fraction ───────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        label_op = OPERATIONS[op_name]["label"]

        for wd, ls, lbl in [(1.0, "-", "wd=1.0 (grok)"),
                             (0.0, "--", "wd=0 (no-wd)")]:
            key = (op_name, wd)
            if key not in all_data:
                continue
            res = all_data[key]["results"]
            steps = [r["step"] for r in res]
            fracs = [r["sub_frac"] for r in res]
            ax.plot(steps, fracs, label=lbl, linewidth=2, linestyle=ls,
                    color=colors_op[op_name] if wd > 0 else "gray")

        ax.axhline(y=rand_frac, color="black", linestyle=":",
                    alpha=0.5, linewidth=1.5,
                    label=f"random ({rand_frac:.1e})")

        ax.set_title(f"{label_op} mod 97", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Fraction of ||dθ|| in commutator subspace")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle(
        "Trajectory Projection onto Commutator Subspace\n"
        f"({N_COMM_SAMPLES}-dim commutator subspace in {total_params}-dim param space)",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figP_trajectory_in_comm_subspace.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figP_trajectory_in_comm_subspace.png")

    # ── Figure Q: Alignment ratio (actual / random) ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for op_name in GROK_OPS:
        for wd, ls, alpha in [(1.0, "-", 1.0), (0.0, "--", 0.5)]:
            key = (op_name, wd)
            if key not in all_data:
                continue
            res = all_data[key]["results"]
            steps = [r["step"] for r in res]
            ratios = [r["mean_cos"] / rand_cos if not np.isnan(r["mean_cos"])
                      else float("nan") for r in res]
            label_op = OPERATIONS[op_name]["label"]
            ax.plot(steps, ratios,
                    label=f"{label_op} (wd={wd})",
                    linewidth=2 if wd > 0 else 1.5,
                    linestyle=ls, alpha=alpha,
                    color=colors_op[op_name])

    ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.5,
               linewidth=1.5, label="random baseline (1.0×)")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Alignment ratio (actual / random)", fontsize=12)
    ax.set_title(
        "Trajectory–Curvature Alignment Relative to Random Baseline\n"
        "(<1× = avoids curvature, >1× = attracted to curvature)",
        fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figQ_alignment_ratio.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figQ_alignment_ratio.png")

    # ── Figure R: Defect × alignment combined ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        label_op = OPERATIONS[op_name]["label"]

        key = (op_name, 1.0)
        if key not in all_data:
            continue
        res = all_data[key]["results"]
        steps = [r["step"] for r in res]
        defs = [r["defect_median"] for r in res]
        cosines = [r["mean_cos"] for r in res]

        color1, color2 = "#1a5276", "#e74c3c"
        ax.plot(steps, defs, label="Defect (left)", linewidth=2, color=color1)
        ax.set_ylabel("Commutator defect", color=color1, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color1)
        ax.set_yscale("log")

        ax2 = ax.twinx()
        ax2.plot(steps, cosines, label="mean|cos| (right)", linewidth=2,
                 color=color2, linestyle="--")
        ax2.axhline(y=rand_cos, color="gray", linestyle=":", alpha=0.5)
        ax2.set_ylabel("Mean |cos(dθ, δ)|", color=color2, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_yscale("log")

        ax.set_title(f"{label_op} mod 97 (wd=1.0)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                  loc="upper left")

    fig.suptitle(
        "Curvature Grows But Trajectory Stays Orthogonal\n"
        "(defect ↑ while alignment stays near random baseline)",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figR_defect_vs_alignment.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figR_defect_vs_alignment.png")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = OUT_DIR / "converse_commutator_results.pt"
    torch.save({
        "all_data": all_data,
        "rand_cos": rand_cos,
        "rand_frac": rand_frac,
        "total_params": total_params,
        "n_comm_samples": N_COMM_SAMPLES,
    }, save_path)
    print(f"\n  saved {save_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
