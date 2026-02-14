#!/usr/bin/env python3
"""
Ablation: Random subspace projection control for 1B-project.

The 1B-project experiment showed that projecting gradients onto the PCA
manifold (removing orthogonal components) kills grokking at strength=1.0.

But is this specific to the PCA manifold, or would ANY 16-dimensional
projection kill grokking?

This script projects gradients onto a RANDOM 16-dim subspace as a control.
If grokking still fails → the effect is non-specific (any constraint kills it).
If grokking survives → the PCA manifold is special; the orthogonal directions
specifically matter for grokking.

Produces:
  figI6 — Random vs PCA projection comparison (dose-response)
  figI7 — Accuracy overlay: baseline vs PCA-project vs random-project
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
OP_NAME = "add"
SEEDS = [42, 137, 2024]
MAX_STEPS = 200_000
SWEEP_MAX_STEPS = 15_000
STRENGTHS = [0.25, 0.5, 0.75, 1.0]


def build_random_basis(model, n_dirs=16, seed=0):
    """
    Build a random orthonormal basis with the same shape as the PCA basis.
    This serves as the control: same dimensionality, but random directions.
    """
    _, total_params = _param_offsets(model)
    rng = torch.Generator()
    rng.manual_seed(seed)
    R = torch.randn(total_params, n_dirs, generator=rng)
    Q, _ = torch.linalg.qr(R, mode="reduced")
    return Q


def train_with_projection(op_name, wd, seed, B, strength, max_steps=None,
                           label="proj"):
    """Train with gradient projection onto basis B (PCA or random)."""
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
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = OUT_DIR / "ablation_random_projection_results.pt"
    all_runs = {}
    pca_bases = {}
    random_bases = {}

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_runs = cached.get("all_runs", {})
        pca_bases = cached.get("pca_bases", {})
        random_bases = cached.get("random_bases", {})
        print(f"  Loaded {len(all_runs)} cached runs")

    # ── Phase 1: Get PCA bases + build random bases ───────────────────
    print(f"\n{'='*70}")
    print("  Phase 1: Build PCA and Random Bases")
    print(f"{'='*70}")

    # Load PCA bases from intervention results if available
    interv_cache = OUT_DIR / "intervention_results.pt"
    if interv_cache.exists() and not pca_bases:
        interv = torch.load(interv_cache, map_location="cpu", weights_only=False)
        pca_bases = interv.get("pca_bases", {})
        print(f"  Loaded {len(pca_bases)} PCA bases from intervention cache")

    for seed in SEEDS:
        basis_key = (OP_NAME, seed)

        # PCA basis
        if basis_key not in pca_bases:
            print(f"\n  Training baseline for PCA basis (seed={seed})...")
            cfg = SweepConfig(OP_NAME=OP_NAME, WEIGHT_DECAY=1.0, SEED=seed)
            model, checkpoints, attn_logs, metrics, grokked, _, _ = \
                train_with_checkpoints(cfg, checkpoint_every=200)
            B = build_pca_basis(model, attn_logs, n_components=2, device="cpu")
            pca_bases[basis_key] = B
            print(f"    PCA basis shape: {B.shape}")

        # Random basis (same dimensionality: 16 directions)
        if basis_key not in random_bases:
            cfg = SweepConfig(OP_NAME=OP_NAME, WEIGHT_DECAY=1.0, SEED=seed)
            torch.manual_seed(seed)
            model_tmp = ModOpTransformer(cfg)
            n_dirs = pca_bases[basis_key].shape[1]  # match PCA dimensionality
            R = build_random_basis(model_tmp, n_dirs=n_dirs, seed=seed + 77777)
            random_bases[basis_key] = R
            print(f"  Random basis for seed={seed}: shape {R.shape}")

    # ── Phase 2: Run projection experiments ───────────────────────────
    print(f"\n{'='*70}")
    print("  Phase 2: PCA vs Random Projection Sweep")
    print(f"{'='*70}")

    conditions = []
    for strength in STRENGTHS:
        conditions.append(("pca-project", strength))
        conditions.append(("random-project", strength))
    # Also baseline (no projection)
    conditions.insert(0, ("baseline", 0.0))

    total = len(conditions) * len(SEEDS)
    run_i = 0

    for cond_name, strength in conditions:
        for seed in SEEDS:
            run_i += 1
            key = (cond_name, strength, seed)

            if key in all_runs:
                data = all_runs[key]
                print(f"  [{run_i}/{total}] {cond_name} str={strength} s={seed} — "
                      f"cached (grok={data['grokked']}, step={data.get('grok_step')})")
                continue

            print(f"\n  [{run_i}/{total}] {cond_name} str={strength} s={seed}")

            basis_key = (OP_NAME, seed)
            if cond_name == "baseline":
                B = pca_bases[basis_key]  # doesn't matter, strength=0
                max_steps = None
            elif cond_name == "pca-project":
                B = pca_bases[basis_key]
                max_steps = SWEEP_MAX_STEPS if strength >= 1.0 else None
            else:
                B = random_bases[basis_key]
                max_steps = SWEEP_MAX_STEPS if strength >= 1.0 else None

            data = train_with_projection(
                OP_NAME, 1.0, seed, B, strength,
                max_steps=max_steps, label=cond_name,
            )
            all_runs[key] = data

            print(f"    → grokked={data['grokked']} (step={data['grok_step']})")

            torch.save({
                "all_runs": all_runs,
                "pca_bases": pca_bases,
                "random_bases": random_bases,
            }, cache_path)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  ABLATION RESULTS: PCA vs RANDOM PROJECTION")
    print(f"{'='*80}")
    print(f"  {'Condition':>20s}  {'Strength':>8s}  {'Grok':>5s}  {'Grok Step':>10s}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*5}  {'─'*10}")

    for cond_name, strength in conditions:
        for seed in SEEDS:
            key = (cond_name, strength, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            gs = str(data["grok_step"]) if data["grok_step"] else "—"
            gr = "YES" if data["grokked"] else "no"
            tag = f"{cond_name} s={seed}"
            print(f"  {tag:>20s}  {strength:8.2f}  {gr:>5s}  {gs:>10s}")

    # ── Figure I6: Dose-response comparison ───────────────────────────
    print(f"\n  Generating figures...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (cond_name, color, marker) in enumerate([
        ("pca-project", "#2980b9", "o"),
        ("random-project", "#27ae60", "s"),
    ]):
        ax = axes[0]
        means = []
        all_pts = []
        for strength in STRENGTHS:
            steps_for_str = []
            for seed in SEEDS:
                key = (cond_name, strength, seed)
                if key in all_runs:
                    data = all_runs[key]
                    if data["grokked"] and data["grok_step"] is not None:
                        steps_for_str.append(data["grok_step"])
            all_pts.append(steps_for_str)
            means.append(np.mean(steps_for_str) if steps_for_str else MAX_STEPS)

        label_name = "PCA projection" if "pca" in cond_name else "Random projection"
        ax.plot(STRENGTHS, means, f"{marker}-", color=color, linewidth=2.5,
                markersize=10, label=label_name)
        for i, pts in enumerate(all_pts):
            for p in pts:
                ax.scatter(STRENGTHS[i], p, color=color, s=25, alpha=0.5)

    # Baseline reference
    baseline_steps = []
    for seed in SEEDS:
        key = ("baseline", 0.0, seed)
        if key in all_runs and all_runs[key]["grokked"]:
            baseline_steps.append(all_runs[key]["grok_step"])
    if baseline_steps:
        axes[0].axhline(y=np.mean(baseline_steps), color="#333", linestyle="--",
                         alpha=0.5, linewidth=1.5, label="Baseline (no projection)")

    axes[0].set_xlabel("Projection strength", fontsize=12)
    axes[0].set_ylabel("Grok step", fontsize=12)
    axes[0].set_title("Dose-Response: PCA vs Random Projection", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Panel 2: Grok rate
    ax = axes[1]
    for cond_name, color, marker in [
        ("pca-project", "#2980b9", "o"),
        ("random-project", "#27ae60", "s"),
    ]:
        rates = []
        for strength in STRENGTHS:
            n_grok = 0
            for seed in SEEDS:
                key = (cond_name, strength, seed)
                if key in all_runs and all_runs[key]["grokked"]:
                    n_grok += 1
            rates.append(n_grok / len(SEEDS))

        label_name = "PCA projection" if "pca" in cond_name else "Random projection"
        ax.plot(STRENGTHS, rates, f"{marker}-", color=color, linewidth=2.5,
                markersize=10, label=label_name)

    ax.set_xlabel("Projection strength", fontsize=12)
    ax.set_ylabel("Grok rate (fraction)", fontsize=12)
    ax.set_title("Grok Success Rate: PCA vs Random", fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Ablation: Is the PCA Manifold Special?\n"
                 f"(op={OP_NAME}, wd=1.0, 3 seeds, intervention at step {T_START})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figI6_ablation_random_vs_pca.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI6_ablation_random_vs_pca.png")

    # ── Figure I7: Accuracy overlay at strength=1.0 ───────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_configs = [
        ("baseline", 0.0, "#333333", "Baseline"),
        ("pca-project", 1.0, "#2980b9", "PCA projection (str=1.0)"),
        ("random-project", 1.0, "#27ae60", "Random projection (str=1.0)"),
    ]

    for cond_name, strength, color, label in plot_configs:
        all_steps_set = set()
        seed_data = {}
        for seed in SEEDS:
            key = (cond_name, strength, seed)
            if key not in all_runs:
                continue
            recs = all_runs[key]["records"]
            if not recs:
                continue
            sd = {r["step"]: r["test_acc"] for r in recs}
            seed_data[seed] = sd
            all_steps_set.update(sd.keys())

        if not seed_data:
            continue

        steps_sorted = sorted(all_steps_set)
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

        ax.plot(steps_sorted, means, color=color, linewidth=2.5, label=label)
        ax.fill_between(steps_sorted, lows, highs, color=color, alpha=0.15)

    ax.axvline(x=T_START, color="gray", linestyle="-.", alpha=0.6, linewidth=1.5)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_title(f"PCA vs Random Projection at Full Strength: {OP_NAME} mod 97\n"
                 f"(mean ± range, 3 seeds)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figI7_ablation_accuracy_overlay.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI7_ablation_accuracy_overlay.png")

    # Final save
    torch.save({
        "all_runs": all_runs,
        "pca_bases": pca_bases,
        "random_bases": random_bases,
    }, cache_path)
    print(f"\n  saved {cache_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
