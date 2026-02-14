#!/usr/bin/env python3
"""
Multi-operation replication of the 1B-project dose-response.

The original intervention experiment (grok_intervention.py) found that projecting
gradients onto the PCA manifold delays/kills grokking with a dose-response curve:
  strength 0.25 → +34 steps
  strength 0.50 → +300 steps
  strength 0.75 → +800 steps
  strength 1.00 → DNF (0/3 seeds grok)

This was only tested on `add`. This script replicates the finding across all 4
grokking operations (sub, mul, x2_y2) to confirm it's a universal property of
the grokking manifold, not operation-specific.

For each operation:
  Phase 1: Train baseline → build PCA basis B
  Phase 2: Re-train with gradient projection at strengths [0.25, 0.5, 0.75, 1.0]

Produces:
  figI10 — Multi-op dose-response (2×2 panel, one per op)
  figI11 — Combined dose-response overlay (all ops on one plot)
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
    flatten_model_params, _param_offsets,
    commutator_defect, build_pca_basis, train_with_checkpoints,
)
from grok_generalization_dynamics import (
    find_spike_step, find_grok_step_from_records,
)
from grok_intervention import (
    write_params_to_model, project_gradient_to_pca, hparams_key,
    COMM_EVERY, COMM_K, COMM_ETA, POST_GROK_STEPS, T_START,
)

OUT_DIR = Path(__file__).parent / "pca_sweep_plots"

# All 4 grokking operations (add included for comparison with original results)
GROK_OPS = ["add", "sub", "mul", "x2_y2"]
SEEDS = [42, 137, 2024]
STRENGTHS = [0.25, 0.5, 0.75, 1.0]
MAX_STEPS = 200_000
SWEEP_MAX_STEPS = 15_000


def train_with_projection(op_name, wd, seed, B, strength, max_steps=None,
                           label="proj"):
    """Train with gradient projection onto PCA basis B."""
    device = get_device()
    steps = max_steps if max_steps is not None else MAX_STEPS
    cfg = SweepConfig(OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed, STEPS=steps)
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
    B_dev = B.to(device) if B is not None else None

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    records = []
    grokked = False
    grok_step = None
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    # Step 0 measurement
    train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
    test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)
    defects = []
    for _ in range(COMM_K):
        D, delta, gcos, nA, nB = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
        defects.append(D)
    records.append({
        "step": 0,
        "defect_median": float(np.median(defects)),
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

        # Projection intervention (after t_start)
        if step >= T_START and B_dev is not None:
            project_gradient_to_pca(model, B_dev, strength=strength)

        opt.step()

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
                "train_acc": train_acc,
                "test_acc": test_acc,
            })
            model.train()

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

        if grokked:
            steps_after_grok += 1
            if steps_after_grok >= POST_GROK_STEPS:
                break

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
        "label": label,
        "strength": strength,
        "seed": seed,
        "op_name": op_name,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = OUT_DIR / "multiop_1b_results.pt"
    all_runs = {}
    pca_bases = {}

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_runs = cached.get("all_runs", {})
        pca_bases = cached.get("pca_bases", {})
        print(f"  Loaded {len(all_runs)} cached runs, {len(pca_bases)} bases")

    # Also try to load PCA bases from the original intervention cache
    interv_cache = OUT_DIR / "intervention_results.pt"
    if interv_cache.exists():
        interv = torch.load(interv_cache, map_location="cpu", weights_only=False)
        existing_bases = interv.get("pca_bases", {})
        for k, v in existing_bases.items():
            if k not in pca_bases:
                pca_bases[k] = v
        if existing_bases:
            print(f"  Loaded {len(existing_bases)} PCA bases from intervention cache")

    # ── Phase 1: Build PCA bases for all ops ──────────────────────────
    print(f"\n{'='*70}")
    print("  Phase 1: Build PCA Bases (all operations)")
    print(f"{'='*70}")

    for op_name in GROK_OPS:
        for seed in SEEDS:
            basis_key = (op_name, seed)
            if basis_key in pca_bases:
                print(f"  PCA basis for {op_name} seed={seed}: cached "
                      f"(shape {pca_bases[basis_key].shape})")
                continue

            print(f"\n  Training baseline: {op_name} seed={seed}...")
            cfg = SweepConfig(OP_NAME=op_name, WEIGHT_DECAY=1.0, SEED=seed)
            model, checkpoints, attn_logs, metrics, grokked, _, _ = \
                train_with_checkpoints(cfg, checkpoint_every=200)
            B = build_pca_basis(model, attn_logs, n_components=2, device="cpu")
            pca_bases[basis_key] = B
            print(f"    PCA basis shape: {B.shape}, grokked={grokked}")

            # Save after each basis (in case of interruption)
            torch.save({
                "all_runs": all_runs, "pca_bases": pca_bases,
            }, cache_path)

    # ── Phase 2: 1B-project dose-response for all ops ─────────────────
    print(f"\n{'='*70}")
    print("  Phase 2: 1B-Project Dose-Response (all operations)")
    print(f"{'='*70}")

    # Conditions: baseline + 4 strengths per operation
    conditions = [("baseline", 0.0)]
    for s in STRENGTHS:
        conditions.append(("1B-project", s))

    total = len(GROK_OPS) * len(conditions) * len(SEEDS)
    run_i = 0

    for op_name in GROK_OPS:
        print(f"\n  --- Operation: {op_name} ---")
        for cond_name, strength in conditions:
            for seed in SEEDS:
                run_i += 1
                key = (op_name, cond_name, strength, seed)

                if key in all_runs:
                    data = all_runs[key]
                    gs = data.get("grok_step", "—")
                    print(f"  [{run_i}/{total}] {op_name} {cond_name} "
                          f"str={strength} s={seed} — cached "
                          f"(grok={data['grokked']}, step={gs})")
                    continue

                print(f"\n  [{run_i}/{total}] {op_name} {cond_name} "
                      f"str={strength} s={seed}")

                B = pca_bases[(op_name, seed)]

                # Cap max_steps for strength=1.0 (known to DNF on add)
                max_steps = SWEEP_MAX_STEPS if strength >= 1.0 else None

                if cond_name == "baseline":
                    # Baseline: no projection (strength=0 means B doesn't matter)
                    data = train_with_projection(
                        op_name, 1.0, seed, B, strength=0.0,
                        max_steps=max_steps, label="baseline",
                    )
                else:
                    data = train_with_projection(
                        op_name, 1.0, seed, B, strength,
                        max_steps=max_steps, label=cond_name,
                    )
                all_runs[key] = data

                gs = data["grok_step"] if data["grok_step"] else "DNF"
                print(f"    → grokked={data['grokked']} (step={gs})")

                # Save after each run
                torch.save({
                    "all_runs": all_runs, "pca_bases": pca_bases,
                }, cache_path)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  MULTI-OPERATION 1B-PROJECT DOSE-RESPONSE RESULTS")
    print(f"{'='*80}")

    for op_name in GROK_OPS:
        print(f"\n  Operation: {op_name}")
        print(f"  {'Condition':>15s}  {'Strength':>8s}  "
              f"{'Grok Rate':>10s}  {'Mean Step':>10s}  {'Steps (per seed)':>30s}")
        print(f"  {'─'*15}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*30}")

        for cond_name, strength in conditions:
            grok_steps = []
            n_grok = 0
            per_seed = []
            for seed in SEEDS:
                key = (op_name, cond_name, strength, seed)
                if key not in all_runs:
                    per_seed.append("?")
                    continue
                data = all_runs[key]
                if data["grokked"] and data["grok_step"] is not None:
                    grok_steps.append(data["grok_step"])
                    n_grok += 1
                    per_seed.append(str(data["grok_step"]))
                else:
                    per_seed.append("DNF")

            rate = f"{n_grok}/{len(SEEDS)}"
            mean = f"{np.mean(grok_steps):.0f}" if grok_steps else "—"
            seeds_str = ", ".join(per_seed)
            tag = cond_name
            print(f"  {tag:>15s}  {strength:8.2f}  {rate:>10s}  "
                  f"{mean:>10s}  {seeds_str:>30s}")

    # ── Figure I10: Multi-op dose-response (2×2 panel) ─────────────────
    print(f"\n  Generating figures...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    op_colors = {
        "add": "#e74c3c",
        "sub": "#2980b9",
        "mul": "#27ae60",
        "x2_y2": "#8e44ad",
    }

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2][idx % 2]

        # Baseline reference
        bl_steps = []
        for seed in SEEDS:
            key = (op_name, "baseline", 0.0, seed)
            if key in all_runs and all_runs[key]["grokked"]:
                bl_steps.append(all_runs[key]["grok_step"])
        if bl_steps:
            ax.axhline(y=np.mean(bl_steps), color="#333", linestyle="--",
                        alpha=0.5, linewidth=1.5, label="Baseline")

        # Dose-response curve
        means = []
        for strength in STRENGTHS:
            steps_for_str = []
            for seed in SEEDS:
                key = (op_name, "1B-project", strength, seed)
                if key in all_runs and all_runs[key]["grokked"]:
                    steps_for_str.append(all_runs[key]["grok_step"])
            if steps_for_str:
                means.append(np.mean(steps_for_str))
            else:
                means.append(SWEEP_MAX_STEPS)

            # Individual seed points
            for s in steps_for_str:
                ax.scatter(strength, s, color=op_colors[op_name], s=25, alpha=0.5)

        ax.plot(STRENGTHS, means, "o-", color=op_colors[op_name], linewidth=2.5,
                markersize=10, label="1B-project")

        # Mark DNF
        for i, strength in enumerate(STRENGTHS):
            n_dnf = 0
            for seed in SEEDS:
                key = (op_name, "1B-project", strength, seed)
                if key in all_runs and not all_runs[key]["grokked"]:
                    n_dnf += 1
            if n_dnf > 0:
                ax.annotate(f"{n_dnf}/3 DNF", (strength, means[i]),
                            textcoords="offset points", xytext=(0, 12),
                            fontsize=9, ha="center", color="#c0392b",
                            fontweight="bold")

        op_label = op_name.replace("_", r"\_") if "_" in op_name else op_name
        ax.set_title(f"{op_label} mod 97", fontsize=12, fontweight="bold")
        ax.set_xlabel("Projection strength", fontsize=11)
        ax.set_ylabel("Grok step", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("1B-Project Dose-Response Across Operations\n"
                 f"(wd=1.0, 3 seeds, projection from step {T_START})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figI10_multiop_dose_response.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI10_multiop_dose_response.png")

    # ── Figure I11: Combined overlay ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Grok step vs strength, all ops on one plot
    ax = axes[0]
    for op_name in GROK_OPS:
        bl_steps = []
        for seed in SEEDS:
            key = (op_name, "baseline", 0.0, seed)
            if key in all_runs and all_runs[key]["grokked"]:
                bl_steps.append(all_runs[key]["grok_step"])
        bl_mean = np.mean(bl_steps) if bl_steps else None

        means = []
        for strength in STRENGTHS:
            steps_for_str = []
            for seed in SEEDS:
                key = (op_name, "1B-project", strength, seed)
                if key in all_runs and all_runs[key]["grokked"]:
                    steps_for_str.append(all_runs[key]["grok_step"])
            if steps_for_str:
                means.append(np.mean(steps_for_str))
            else:
                means.append(SWEEP_MAX_STEPS)

        # Normalize to baseline for cross-op comparison
        if bl_mean and bl_mean > 0:
            norm_means = [m / bl_mean for m in means]
        else:
            norm_means = means

        op_label = op_name.replace("_", r"\_") if "_" in op_name else op_name
        ax.plot(STRENGTHS, norm_means, "o-", color=op_colors[op_name],
                linewidth=2.5, markersize=10, label=op_label)

    ax.axhline(y=1.0, color="#333", linestyle="--", alpha=0.5, linewidth=1.5,
               label="Baseline (1.0×)")
    ax.set_xlabel("Projection strength", fontsize=12)
    ax.set_ylabel("Grok step (normalized to baseline)", fontsize=12)
    ax.set_title("Dose-Response: All Operations", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: Grok rate
    ax = axes[1]
    for op_name in GROK_OPS:
        rates = []
        for strength in STRENGTHS:
            n_grok = sum(1 for s in SEEDS
                         if (op_name, "1B-project", strength, s) in all_runs
                         and all_runs[(op_name, "1B-project", strength, s)]["grokked"])
            rates.append(n_grok / len(SEEDS))

        op_label = op_name.replace("_", r"\_") if "_" in op_name else op_name
        ax.plot(STRENGTHS, rates, "o-", color=op_colors[op_name],
                linewidth=2.5, markersize=10, label=op_label)

    ax.set_xlabel("Projection strength", fontsize=12)
    ax.set_ylabel("Grok rate (fraction)", fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Grok Success Rate: All Operations", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("1B-Project Replication Across Grokking Operations\n"
                 f"(wd=1.0, 3 seeds, projection from step {T_START})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figI11_multiop_combined.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI11_multiop_combined.png")

    # Final save
    torch.save({
        "all_runs": all_runs, "pca_bases": pca_bases,
    }, cache_path)
    print(f"\n  saved {cache_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
