#!/usr/bin/env python3
"""
Multi-seed commutator analysis for publication.

Combines forward (commutator → PCA projection) and converse (trajectory →
commutator alignment) analyses across 3 seeds × 6 ops × 2 wd.

Key measurements per checkpoint:
  Forward:   resid/full  (should be ≈100% → PCA manifold integrable)
  Converse:  mean|cos(dθ, δ)|  (should be ≈ random baseline)

Produces summary figures with error bars (mean ± std over seeds).
"""

import math, time, random, sys
from dataclasses import asdict
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
from pca_sweep_analysis import pca_on_trajectory, collect_trajectory
from grok_commutator_analysis import (
    flatten_model_params, _param_offsets, commutator_defect,
    projected_commutator, build_pca_basis, train_with_checkpoints,
    attn_weight_mask,
)
from grok_converse_commutator import (
    build_commutator_subspace, project_onto_subspace,
    mean_alignment_cosine, random_alignment_baseline,
)

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
ALL_OPS = ["add", "sub", "mul", "x2_y2", "x2_xy_y2", "x3_xy"]
GROK_OPS = ["add", "sub", "mul", "x2_y2"]
NOGROK_OPS = ["x2_xy_y2", "x3_xy"]
SEEDS = [42, 137, 2024]
CHECKPOINT_EVERY = 200
N_COMM = 12          # commutator samples per checkpoint
COMM_ETA = 1e-3
N_PCA_COMP = 2


# ═══════════════════════════════════════════════════════════════════════════
# Per-run analysis: forward + converse in one pass
# ═══════════════════════════════════════════════════════════════════════════

def analyse_run(op_name, wd, seed):
    """
    Train one model, then at each checkpoint measure:
      - Forward:  commutator → PCA basis projection  (resid/full)
      - Converse: trajectory step → commutator alignment  (mean|cos|)
    """
    device = get_device()

    # Non-grokking ops get shorter runs; no-wd also shorter
    if wd == 0.0:
        max_steps = 10_000
        ckpt_every = 500
    elif op_name in NOGROK_OPS:
        max_steps = 10_000     # they won't grok, just need baseline
        ckpt_every = 500
    else:
        max_steps = 200_000
        ckpt_every = CHECKPOINT_EVERY

    cfg = SweepConfig(
        OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed, STEPS=max_steps,
    )

    model, checkpoints, attn_logs, metrics, grokked, train_pairs, test_pairs = \
        train_with_checkpoints(cfg, checkpoint_every=ckpt_every)

    op_fn = OPERATIONS[op_name]["fn"]

    # Build PCA basis from full trajectory (forward direction)
    model_fresh = ModOpTransformer(cfg).to(device)
    B = build_pca_basis(model_fresh, attn_logs, n_components=N_PCA_COMP, device="cpu")
    amask = attn_weight_mask(model_fresh)

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    # Subsample checkpoints if needed
    total = len(checkpoints)
    if total > 30:
        idx = np.linspace(0, total - 1, 30, dtype=int)
        checkpoints = [checkpoints[i] for i in idx]

    prev_theta = None
    records = []

    for ci, (step, sd) in enumerate(checkpoints):
        model_fresh.load_state_dict(sd)
        model_fresh.to(device)
        theta = flatten_model_params(model_fresh).cpu()

        dtheta = (theta - prev_theta) if prev_theta is not None else None

        # Collect commutator deltas
        comm_deltas = []
        defects = []
        normAs, normBs = [], []
        for _ in range(N_COMM):
            D, delta, gcos, nA, nB = commutator_defect(
                model_fresh, batch_fn, device, eta=COMM_ETA
            )
            comm_deltas.append(delta.cpu())
            defects.append(D)
            normAs.append(nA.cpu())
            normBs.append(nB.cpu())

        defect_med = float(np.median(defects))

        # ── Forward: project commutator onto PCA basis ──
        delta0 = comm_deltas[0]
        nA0, nB0 = normAs[0], normBs[0]
        pc = projected_commutator(
            delta0, B.cpu() if B is not None else None, nA0, nB0
        )
        resid_frac = pc["resid"] / pc["full"] if pc["full"] > 1e-15 else float("nan")

        # Attention weight fraction
        delta_full_norm = delta0.norm().item()
        attn_frac = (delta0[amask].norm().item() / (delta_full_norm + 1e-15))

        # ── Converse: project trajectory onto commutator directions ──
        if dtheta is not None and dtheta.norm() > 1e-30:
            C = build_commutator_subspace(comm_deltas)
            _, _, _, sub_frac = project_onto_subspace(dtheta, C)
            m_cos = mean_alignment_cosine(dtheta, comm_deltas)
        else:
            sub_frac = float("nan")
            m_cos = float("nan")

        records.append({
            "step": step,
            "defect_median": defect_med,
            "resid_frac": resid_frac,
            "attn_frac": attn_frac,
            "mean_cos": m_cos,
            "sub_frac": sub_frac,
        })

        prev_theta = theta

    grok_step = None
    if grokked:
        for m in metrics:
            if m["test_acc"] >= 0.98:
                grok_step = m["step"]
                break

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cfg_tmp = SweepConfig()
    model_tmp = ModOpTransformer(cfg_tmp)
    _, total_params = _param_offsets(model_tmp)
    rand_cos, rand_frac = random_alignment_baseline(total_params, N_COMM)
    print(f"Param dim: {total_params}, rand |cos|={rand_cos:.6f}, "
          f"rand sub_frac={rand_frac:.6f}")
    del model_tmp

    # ── Run all conditions ───────────────────────────────────────────────
    all_data = {}   # (op, wd, seed) -> dict
    total_runs = len(ALL_OPS) * 2 * len(SEEDS)
    run_i = 0

    for op_name in ALL_OPS:
        for wd in [1.0, 0.0]:
            for seed in SEEDS:
                run_i += 1
                tag = f"{op_name}_wd{wd}_s{seed}"
                print(f"\n[{run_i}/{total_runs}] {tag}")

                data = analyse_run(op_name, wd, seed)
                all_data[(op_name, wd, seed)] = data

                g = data["grokked"]
                gs = data.get("grok_step")
                recs = data["records"]
                valid = [r for r in recs if not np.isnan(r["mean_cos"])]
                if valid:
                    last = valid[-1]
                    print(f"  grokked={g} (step={gs}), "
                          f"defect={last['defect_median']:.1f}, "
                          f"resid_frac={last['resid_frac']:.3f}, "
                          f"mean|cos|={last['mean_cos']:.6f} "
                          f"({last['mean_cos']/rand_cos:.1f}× rand)")

    # ── Aggregate across seeds ───────────────────────────────────────────
    # For each (op, wd), compute mean ± std of final-checkpoint values
    print(f"\n{'='*90}")
    print("  MULTI-SEED SUMMARY  (mean ± std over 3 seeds)")
    print(f"{'='*90}")
    print(f"  Random baseline: E[|cos|]={rand_cos:.6f}")
    print()
    print(f"  {'Config':>28s}  {'grok':>8s}  {'defect':>10s}  "
          f"{'resid/full':>12s}  {'|cos|/rand':>12s}  {'attn%':>8s}")

    agg = {}  # (op, wd) -> dict of arrays

    for op_name in ALL_OPS:
        for wd in [1.0, 0.0]:
            defects, resids, cosines, attns, grok_steps = [], [], [], [], []
            n_grok = 0

            for seed in SEEDS:
                d = all_data[(op_name, wd, seed)]
                if d["grokked"]:
                    n_grok += 1
                if d["grok_step"] is not None:
                    grok_steps.append(d["grok_step"])

                valid = [r for r in d["records"] if not np.isnan(r["mean_cos"])]
                if valid:
                    last = valid[-1]
                    defects.append(last["defect_median"])
                    resids.append(last["resid_frac"])
                    cosines.append(last["mean_cos"])
                    attns.append(last["attn_frac"])

            if not defects:
                continue

            def fmt_mean_std(vals):
                m, s = np.mean(vals), np.std(vals)
                return f"{m:.3f}±{s:.3f}"

            cos_ratio = np.array(cosines) / rand_cos
            tag = f"{op_name} wd={wd}"
            gs_str = f"{n_grok}/3"
            if grok_steps:
                gs_str += f" @{int(np.mean(grok_steps))}"

            print(f"  {tag:>28s}  {gs_str:>8s}  "
                  f"{fmt_mean_std(defects):>10s}  "
                  f"{fmt_mean_std(resids):>12s}  "
                  f"{fmt_mean_std(cos_ratio.tolist()):>12s}  "
                  f"{fmt_mean_std(attns):>8s}")

            agg[(op_name, wd)] = {
                "defects": defects,
                "resids": resids,
                "cosines": cosines,
                "cos_ratios": cos_ratio.tolist(),
                "attns": attns,
                "n_grok": n_grok,
                "grok_steps": grok_steps,
            }

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating figures...")

    OP_LABELS = {
        "add": "(a+b)", "sub": "(a−b)", "mul": "(a×b)",
        "x2_y2": "(a²+b²)", "x2_xy_y2": "(a²+ab+b²)", "x3_xy": "(a³+ab)",
    }

    # ── Figure S: Bar chart — resid/full by operation ────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(ALL_OPS))
    width = 0.35

    for wi, (wd, color, label) in enumerate([
        (1.0, "#2980b9", "wd=1.0 (grok)"),
        (0.0, "#e74c3c", "wd=0.0 (no-wd)"),
    ]):
        means, stds = [], []
        for op in ALL_OPS:
            key = (op, wd)
            if key in agg:
                vals = agg[key]["resids"]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x_pos + wi * width, means, width, yerr=stds,
               label=label, color=color, alpha=0.8, capsize=3)

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([OP_LABELS[op] for op in ALL_OPS], fontsize=11)
    ax.set_ylabel("Residual fraction (resid / full)", fontsize=12)
    ax.set_title("PCA Manifold Integrability Across Operations\n"
                 "(resid/full ≈ 1.0 → commutator ⊥ PCA manifold, 3 seeds)",
                 fontsize=13)
    ax.set_ylim(0.9, 1.02)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figS_multiseed_integrability.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figS_multiseed_integrability.png")

    # ── Figure T: Bar chart — alignment ratio by operation ───────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    for wi, (wd, color, label) in enumerate([
        (1.0, "#2980b9", "wd=1.0"),
        (0.0, "#e74c3c", "wd=0.0"),
    ]):
        means, stds = [], []
        for op in ALL_OPS:
            key = (op, wd)
            if key in agg:
                vals = agg[key]["cos_ratios"]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x_pos + wi * width, means, width, yerr=stds,
               label=label, color=color, alpha=0.8, capsize=3)

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([OP_LABELS[op] for op in ALL_OPS], fontsize=11)
    ax.set_ylabel("Alignment ratio (actual / random)", fontsize=12)
    ax.set_title("Trajectory–Curvature Alignment vs Random Baseline\n"
                 "(≈1.0 = no structured alignment, 3 seeds)", fontsize=13)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5,
               alpha=0.7, label="random baseline")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figT_multiseed_alignment.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figT_multiseed_alignment.png")

    # ── Figure U: Defect bar chart (grok vs no-grok ops) ────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    for wi, (wd, color, label) in enumerate([
        (1.0, "#2980b9", "wd=1.0"),
        (0.0, "#e74c3c", "wd=0.0"),
    ]):
        means, stds = [], []
        for op in ALL_OPS:
            key = (op, wd)
            if key in agg:
                vals = agg[key]["defects"]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x_pos + wi * width, means, width, yerr=stds,
               label=label, color=color, alpha=0.8, capsize=3)

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([OP_LABELS[op] for op in ALL_OPS], fontsize=11)
    ax.set_ylabel("Commutator defect (median)", fontsize=12)
    ax.set_title("Commutator Defect by Operation\n"
                 "(grokking ops show 10-1000× higher defect, 3 seeds)",
                 fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figU_multiseed_defect.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figU_multiseed_defect.png")

    # ── Figure V: Temporal traces with seed ribbons (add only) ───────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, ylabel, title in [
        (axes[0], "defect_median", "Commutator defect", "Defect explosion"),
        (axes[1], "resid_frac", "Residual fraction", "Integrability (resid/full)"),
        (axes[2], "mean_cos", "Mean |cos(dθ,δ)|", "Trajectory–curvature alignment"),
    ]:
        for wd, color, ls, label in [
            (1.0, "#2980b9", "-", "wd=1.0 (grok)"),
            (0.0, "#e74c3c", "--", "wd=0"),
        ]:
            all_traces = []
            for seed in SEEDS:
                key = (GROK_OPS[0], wd, seed)  # add
                if key not in all_data:
                    continue
                recs = all_data[key]["records"]
                valid = [(r["step"], r[metric]) for r in recs
                         if not np.isnan(r[metric])]
                if valid:
                    steps, vals = zip(*valid)
                    ax.plot(steps, vals, color=color, alpha=0.3, linewidth=1,
                            linestyle=ls)
                    all_traces.append((list(steps), list(vals)))

            # Plot mean as thick line
            if all_traces:
                # Interpolate to common steps
                all_steps = sorted(set(s for tr in all_traces for s in tr[0]))
                # Simple: just plot individual seeds with transparency

            ax.plot([], [], color=color, linewidth=2, linestyle=ls, label=label)

        if metric == "defect_median":
            ax.set_yscale("log")
        if metric == "mean_cos":
            ax.axhline(y=rand_cos, color="black", linestyle=":",
                        alpha=0.5, label=f"random ({rand_cos:.1e})")
        if metric == "resid_frac":
            ax.set_ylim(0.9, 1.02)

        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("(a+b) mod 97 — 3 seeds", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figV_temporal_add_seeds.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figV_temporal_add_seeds.png")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = OUT_DIR / "multiseed_commutator_results.pt"
    torch.save({
        "all_data": all_data,
        "agg": agg,
        "rand_cos": rand_cos,
        "rand_frac": rand_frac,
        "total_params": total_params,
        "seeds": SEEDS,
    }, save_path)
    print(f"\n  saved {save_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
