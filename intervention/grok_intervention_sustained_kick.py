#!/usr/bin/env python3
"""
Stronger 1A intervention: sustained directional kicks along the actual
commutator defect direction.

The original 1A-kick was a single perturbation at step t_start with
epsilon ~0.015 (10× gradient step norm). It had no effect because:
  - One-time perturbation is absorbed by Adam's momentum within a few steps
  - The magnitude was tiny relative to ~290k-dim parameter space

This script tests sustained, repeated kicks along the *actual* commutator
direction (recomputed every kick_interval steps), with much larger magnitudes.

Conditions:
  baseline           — No intervention
  1A-sustained-comm  — Kick along commutator direction every kick_interval steps
  1A-sustained-rand  — Kick along random orthogonal direction (control)

The key comparison: does kicking along the *specific* commutator direction
accelerate grokking more than kicking along a random orthogonal direction?

Produces:
  figI8 — Grok step vs kick magnitude (comm vs random)
  figI9 — Accuracy + defect overlay for best condition
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
    commutator_defect, commutator_defect_median, build_pca_basis,
    train_with_checkpoints,
)
from grok_generalization_dynamics import (
    find_spike_step, find_grok_step_from_records,
)
from grok_intervention import (
    write_params_to_model, hparams_key,
    COMM_EVERY, COMM_K, COMM_ETA, POST_GROK_STEPS, T_START,
)

OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
OP_NAME = "add"
SEEDS = [42, 137, 2024]
MAX_STEPS = 200_000

# Sustained kick parameters
KICK_INTERVAL = 50         # kick every N steps
KICK_ALPHAS = [50, 100, 200, 500]  # multiples of gradient step norm


def sustained_kick_along_commutator(model, batch_fn, device, B, alpha):
    """
    Compute fresh commutator delta, extract orthogonal component,
    kick weights along that specific direction.
    """
    out = commutator_defect_median(model, batch_fn, device, K=5, eta=COMM_ETA)
    delta = out["median_delta"].to(device)

    # Get gradient step norm for scale calibration
    _, _, _, normA, _ = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
    grad_step_norm = normA.item()

    # Extract orthogonal component
    B_dev = B.to(device)
    proj_coeffs = B_dev.T @ delta
    delta_proj = B_dev @ proj_coeffs
    delta_perp = delta - delta_proj

    perp_norm = delta_perp.norm()
    if perp_norm < 1e-15:
        return 0.0

    direction = delta_perp / perp_norm
    epsilon = alpha * grad_step_norm

    theta = flatten_model_params(model).to(device)
    theta_new = theta + epsilon * direction
    write_params_to_model(model, theta_new)
    return epsilon


def sustained_kick_random_orthogonal(model, device, B, alpha, rng):
    """
    Kick along a random direction orthogonal to the PCA manifold.
    Same magnitude as the commutator kick, but random direction.
    """
    theta = flatten_model_params(model).to(device)

    # Need grad step norm — approximate from model parameter scale
    # Use a fixed estimate based on typical values from the experiment
    grad_step_norm = 1.5e-3  # typical ||eta * gA|| from logs

    noise = torch.randn(theta.shape[0], generator=rng, device=device)
    B_dev = B.to(device)
    noise_proj = B_dev @ (B_dev.T @ noise)
    noise_perp = noise - noise_proj
    n_norm = noise_perp.norm()
    if n_norm < 1e-15:
        return 0.0

    direction = noise_perp / n_norm
    epsilon = alpha * grad_step_norm

    theta_new = theta + epsilon * direction
    write_params_to_model(model, theta_new)
    return epsilon


def train_with_sustained_kick(op_name, wd, seed, B, condition, alpha,
                               kick_interval=KICK_INTERVAL, max_steps=None):
    """
    Train with sustained kicks every kick_interval steps.
    condition: "baseline", "1A-sustained-comm", "1A-sustained-rand"
    """
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

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    intervention_rng = torch.Generator(device=device)
    intervention_rng.manual_seed(seed + 88888)

    records = []
    grokked = False
    grok_step = None
    patience = 0
    steps_after_grok = 0
    n_kicks = 0
    t0 = time.time()

    # Step 0 measurement
    train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
    test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)
    defects = []
    for _ in range(COMM_K):
        D, delta, gcos, nA, nB = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
        defects.append(D)
    records.append({
        "step": 0, "defect_median": float(np.median(defects)),
        "train_acc": train_acc, "test_acc": test_acc,
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

        # Sustained kick (AFTER optimizer step, to inject perturbation)
        if step >= T_START and step % kick_interval == 0:
            if condition == "1A-sustained-comm":
                sustained_kick_along_commutator(model, batch_fn, device, B, alpha)
                n_kicks += 1
            elif condition == "1A-sustained-rand":
                sustained_kick_random_orthogonal(model, device, B, alpha,
                                                  intervention_rng)
                n_kicks += 1

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
                "step": step, "defect_median": float(np.median(defects)),
                "train_acc": train_acc, "test_acc": test_acc,
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
                    print(f"      GROKKED at step {step} (kicks={n_kicks})")
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
                  f"kicks={n_kicks} | {elapsed:.1f}m")

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "condition": condition,
        "alpha": alpha,
        "n_kicks": n_kicks,
        "seed": seed,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = OUT_DIR / "sustained_kick_results.pt"
    all_runs = {}
    pca_bases = {}

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_runs = cached.get("all_runs", {})
        pca_bases = cached.get("pca_bases", {})
        print(f"  Loaded {len(all_runs)} cached runs")

    # Load PCA bases
    interv_cache = OUT_DIR / "intervention_results.pt"
    if interv_cache.exists() and not pca_bases:
        interv = torch.load(interv_cache, map_location="cpu", weights_only=False)
        pca_bases = interv.get("pca_bases", {})
        print(f"  Loaded {len(pca_bases)} PCA bases from intervention cache")

    # Build PCA bases if needed
    for seed in SEEDS:
        basis_key = (OP_NAME, seed)
        if basis_key not in pca_bases:
            print(f"\n  Training baseline for PCA basis (seed={seed})...")
            cfg = SweepConfig(OP_NAME=OP_NAME, WEIGHT_DECAY=1.0, SEED=seed)
            model, checkpoints, attn_logs, metrics, grokked, _, _ = \
                train_with_checkpoints(cfg, checkpoint_every=200)
            B = build_pca_basis(model, attn_logs, n_components=2, device="cpu")
            pca_bases[basis_key] = B
            print(f"    PCA basis shape: {B.shape}")

    # ── Run experiments ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Sustained Kick Experiments")
    print(f"{'='*70}")

    conditions = [("baseline", 0)]
    for alpha in KICK_ALPHAS:
        conditions.append(("1A-sustained-comm", alpha))
        conditions.append(("1A-sustained-rand", alpha))

    total = len(conditions) * len(SEEDS)
    run_i = 0

    for cond_name, alpha in conditions:
        for seed in SEEDS:
            run_i += 1
            key = (cond_name, alpha, seed)

            if key in all_runs:
                data = all_runs[key]
                print(f"  [{run_i}/{total}] {cond_name} alpha={alpha} s={seed} — "
                      f"cached (grok={data['grokked']}, step={data.get('grok_step')})")
                continue

            print(f"\n  [{run_i}/{total}] {cond_name} alpha={alpha} s={seed}")

            B = pca_bases[(OP_NAME, seed)]
            data = train_with_sustained_kick(
                OP_NAME, 1.0, seed, B, cond_name, alpha,
            )
            all_runs[key] = data
            print(f"    → grokked={data['grokked']} (step={data['grok_step']}, "
                  f"kicks={data['n_kicks']})")

            torch.save({
                "all_runs": all_runs, "pca_bases": pca_bases,
            }, cache_path)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUSTAINED KICK RESULTS")
    print(f"{'='*80}")
    print(f"  {'Condition':>25s}  {'Alpha':>6s}  {'Grok':>5s}  "
          f"{'Grok Step':>10s}  {'Kicks':>6s}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*5}  {'─'*10}  {'─'*6}")

    for cond_name, alpha in conditions:
        for seed in SEEDS:
            key = (cond_name, alpha, seed)
            if key not in all_runs:
                continue
            data = all_runs[key]
            gs = str(data["grok_step"]) if data["grok_step"] else "—"
            gr = "YES" if data["grokked"] else "no"
            tag = f"{cond_name} s={seed}"
            print(f"  {tag:>25s}  {alpha:6d}  {gr:>5s}  {gs:>10s}  "
                  f"{data.get('n_kicks', 0):6d}")

    # ── Figure I8: Dose-response ──────────────────────────────────────
    print(f"\n  Generating figures...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Grok step vs alpha
    ax = axes[0]
    for cond_name, color, marker, label in [
        ("1A-sustained-comm", "#e74c3c", "o", "Commutator direction"),
        ("1A-sustained-rand", "#95a5a6", "s", "Random direction"),
    ]:
        means = []
        for alpha in KICK_ALPHAS:
            steps_list = []
            for seed in SEEDS:
                key = (cond_name, alpha, seed)
                if key in all_runs and all_runs[key]["grokked"]:
                    steps_list.append(all_runs[key]["grok_step"])
            means.append(np.mean(steps_list) if steps_list else MAX_STEPS)

        ax.plot(KICK_ALPHAS, means, f"{marker}-", color=color, linewidth=2.5,
                markersize=10, label=label)

        for i, alpha in enumerate(KICK_ALPHAS):
            for seed in SEEDS:
                key = (cond_name, alpha, seed)
                if key in all_runs and all_runs[key]["grokked"]:
                    ax.scatter(alpha, all_runs[key]["grok_step"],
                              color=color, s=25, alpha=0.5)

    # Baseline reference
    baseline_steps = []
    for seed in SEEDS:
        key = ("baseline", 0, seed)
        if key in all_runs and all_runs[key]["grokked"]:
            baseline_steps.append(all_runs[key]["grok_step"])
    if baseline_steps:
        ax.axhline(y=np.mean(baseline_steps), color="#333", linestyle="--",
                    alpha=0.5, linewidth=1.5, label="Baseline")

    ax.set_xlabel("Kick magnitude (α × grad step norm)", fontsize=12)
    ax.set_ylabel("Grok step", fontsize=12)
    ax.set_title("Sustained Kicks: Commutator vs Random Direction", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: Grok rate
    ax = axes[1]
    for cond_name, color, marker, label in [
        ("1A-sustained-comm", "#e74c3c", "o", "Commutator direction"),
        ("1A-sustained-rand", "#95a5a6", "s", "Random direction"),
    ]:
        rates = []
        for alpha in KICK_ALPHAS:
            n_grok = sum(1 for s in SEEDS
                         if (cond_name, alpha, s) in all_runs
                         and all_runs[(cond_name, alpha, s)]["grokked"])
            rates.append(n_grok / len(SEEDS))
        ax.plot(KICK_ALPHAS, rates, f"{marker}-", color=color, linewidth=2.5,
                markersize=10, label=label)

    ax.set_xlabel("Kick magnitude (α × grad step norm)", fontsize=12)
    ax.set_ylabel("Grok rate", fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Grok Success Rate vs Kick Magnitude", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Sustained Directional Kicks Along Commutator\n"
                 f"(op={OP_NAME}, kick every {KICK_INTERVAL} steps from step {T_START})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figI8_sustained_kick_dose_response.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI8_sustained_kick_dose_response.png")

    # ── Figure I9: Best condition accuracy overlay ────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    # Pick middle alpha for the overlay
    mid_alpha = KICK_ALPHAS[len(KICK_ALPHAS) // 2]

    plot_configs = [
        ("baseline", 0, "#333333", "Baseline"),
        ("1A-sustained-comm", mid_alpha, "#e74c3c",
         f"Comm kicks (α={mid_alpha})"),
        ("1A-sustained-rand", mid_alpha, "#95a5a6",
         f"Random kicks (α={mid_alpha})"),
    ]

    for cond_name, alpha, color, label in plot_configs:
        all_steps_set = set()
        seed_data_acc = {}
        seed_data_def = {}
        for seed in SEEDS:
            key = (cond_name, alpha, seed)
            if key not in all_runs:
                continue
            recs = all_runs[key]["records"]
            if not recs:
                continue
            sd_a = {r["step"]: r["test_acc"] for r in recs}
            sd_d = {r["step"]: r["defect_median"] for r in recs}
            seed_data_acc[seed] = sd_a
            seed_data_def[seed] = sd_d
            all_steps_set.update(sd_a.keys())

        if not seed_data_acc:
            continue

        steps_sorted = sorted(all_steps_set)
        means_a = [np.mean([sd[s] for sd in seed_data_acc.values() if s in sd])
                    for s in steps_sorted]
        means_d = [np.mean([sd[s] for sd in seed_data_def.values() if s in sd])
                    for s in steps_sorted]

        ax.plot(steps_sorted, means_d, color=color, linewidth=2, alpha=0.8)
        ax2.plot(steps_sorted, means_a, color=color, linewidth=2,
                 linestyle="--", alpha=0.8, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect", fontsize=12)
    ax2.set_ylabel("Test accuracy", fontsize=12)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(fontsize=10, loc="center left")
    ax.grid(alpha=0.2)
    ax.axvline(x=T_START, color="gray", linestyle="-.", alpha=0.5, linewidth=1.5)
    ax.set_title(f"Sustained Kicks: Defect + Accuracy ({OP_NAME} mod 97)\n"
                 f"(solid=defect, dashed=test acc, mean over 3 seeds)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figI9_sustained_kick_overlay.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figI9_sustained_kick_overlay.png")

    torch.save({
        "all_runs": all_runs, "pca_bases": pca_bases,
    }, cache_path)
    print(f"\n  saved {cache_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
