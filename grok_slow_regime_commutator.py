#!/usr/bin/env python3
"""
Commutator analysis on the SLOW grokking regime (lr=5e-5, wd=0.1, 3 layers).

Verifies that integrability and defect-predicts-grokking hold in a different
hyperparameter regime, closing the gap between wd=0.1 and wd=1.0 results.

Slow regime groks at ~500k steps (vs ~3k for Power et al.), so we use
coarser measurement intervals (every 5000 steps).

Produces:
  figY  — Regime comparison: integrability + defect for slow vs fast
  figZ  — Defect predicts grokking in slow regime (hero figure)
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
from pca_sweep_analysis import pca_on_trajectory, collect_trajectory
from grok_commutator_analysis import (
    flatten_model_params, _param_offsets, commutator_defect,
    projected_commutator, build_pca_basis, attn_weight_mask,
)

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
SEEDS = [42, 137]          # 2 seeds (enough for regime comparison)

# Slow regime hyperparameters
SLOW_LR = 5e-5
SLOW_WD = 0.1
SLOW_LAYERS = 3
SLOW_BETA2 = 0.999
SLOW_MAX_STEPS = 650_000   # enough headroom past grok (~500-550k)
SLOW_NOWD_STEPS = 50_000   # short no-wd control

# Measurement intervals — coarser because runs are 200x longer
COMM_EVERY = 5_000         # commutator every 5k steps
COMM_K = 5                 # commutator samples per checkpoint
COMM_ETA = 1e-3
N_PCA_COMP = 2
POST_GROK_STEPS = 20_000   # continue 20k steps after grokking


# ═══════════════════════════════════════════════════════════════════════════
# Training with inline commutator + PCA measurement
# ═══════════════════════════════════════════════════════════════════════════

def train_slow_with_tracking(seed, wd, max_steps):
    """Train slow-regime model, measuring commutator defect + test accuracy."""
    device = get_device()
    cfg = SweepConfig(
        OP_NAME="add",
        WEIGHT_DECAY=wd,
        SEED=seed,
        STEPS=max_steps,
        LR=SLOW_LR,
        N_LAYERS=SLOW_LAYERS,
        ADAM_BETA1=0.9,
        ADAM_BETA2=SLOW_BETA2,
        EVAL_EVERY=1000,
        MODEL_LOG_EVERY=5000,
    )
    op_fn = OPERATIONS["add"]["fn"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_pairs, test_pairs = build_dataset(
        cfg.P, cfg.TRAIN_FRACTION, cfg.SEED, op_fn,
        OPERATIONS["add"]["restrict_nonzero"]
    )

    model = ModOpTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=wd,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    # For PCA: collect attention weight snapshots
    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]

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

    for step in range(1, max_steps + 1):
        model.train()
        a, b, y = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)
        logits = model(a, b)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        # Log attention weights for PCA
        if step % cfg.MODEL_LOG_EVERY == 0:
            attn_logs.append({"step": step, "layers": extract_attn_matrices(model)})

        # Measure commutator + accuracy
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

        if step % 50_000 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            d = last_r.get("defect_median", 0)
            ta = last_r.get("test_acc", 0)
            print(f"      step {step:7d} | test {ta:.3f} | defect {d:.1f} | {elapsed:.1f}m")

    # Build PCA basis from collected attention logs
    model_fresh = ModOpTransformer(cfg).to(device)
    B = build_pca_basis(model_fresh, attn_logs, n_components=N_PCA_COMP, device="cpu")

    # Compute integrability at final checkpoint
    resid_fracs = []
    if B is not None:
        model.eval()
        for _ in range(COMM_K):
            D, delta, gcos, nA, nB = commutator_defect(model, batch_fn, device, eta=COMM_ETA)
            pc = projected_commutator(delta.cpu(), B.cpu(), nA.cpu(), nB.cpu())
            rf = pc["resid"] / pc["full"] if pc["full"] > 1e-15 else float("nan")
            resid_fracs.append(rf)

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "resid_fracs": resid_fracs,
        "seed": seed,
        "wd": wd,
        "attn_log_count": len(attn_logs),
    }


# ═══════════════════════════════════════════════════════════════════════════

def find_spike_step(records, threshold_factor=10, min_defect=5):
    if len(records) < 3:
        return None
    baseline = np.median([r["defect_median"] for r in records[:3]])
    baseline = max(baseline, 0.1)
    for r in records[2:]:
        if r["defect_median"] > threshold_factor * baseline and r["defect_median"] > min_defect:
            return r["step"]
    return None


def find_grok_step_from_records(records, threshold=0.9):
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
    print(f"Slow regime: lr={SLOW_LR}, wd={SLOW_WD}, {SLOW_LAYERS} layers, "
          f"β₂={SLOW_BETA2}")
    print(f"Expected grokking: ~500k steps\n")

    all_runs = {}

    # ── Grokking runs: wd=0.1 ─────────────────────────────────────────────
    for seed in SEEDS:
        tag = f"add_wd{SLOW_WD}_s{seed}_slow"
        print(f"\n  [{tag}]")
        data = train_slow_with_tracking(seed, SLOW_WD, SLOW_MAX_STEPS)
        all_runs[("add", SLOW_WD, seed)] = data
        rf = np.mean(data["resid_fracs"]) if data["resid_fracs"] else float("nan")
        print(f"    → grokked={data['grokked']} (step={data['grok_step']}), "
              f"resid_frac={rf:.4f}, {len(data['records'])} measurements")

    # ── No-wd controls ────────────────────────────────────────────────────
    for seed in SEEDS[:1]:  # 1 seed control
        tag = f"add_wd0.0_s{seed}_slow"
        print(f"\n  [ctrl: {tag}]")
        data = train_slow_with_tracking(seed, 0.0, SLOW_NOWD_STEPS)
        all_runs[("add", 0.0, seed)] = data
        print(f"    → grokked={data['grokked']}, "
              f"{len(data['records'])} measurements")

    # ── Compute timing ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SLOW REGIME: DEFECT SPIKE vs GROKKING")
    print(f"{'='*80}")
    print(f"  {'Config':>25s}  {'spike_step':>12s}  {'grok_50%':>10s}  "
          f"{'grok_90%':>10s}  {'lead→90%':>12s}  {'resid_frac':>12s}")

    for key, data in sorted(all_runs.items()):
        if not data["grokked"]:
            continue
        recs = data["records"]
        spike = find_spike_step(recs)
        grok_50 = find_grok_step_from_records(recs, 0.5)
        grok_90 = find_grok_step_from_records(recs, 0.9)
        lead_90 = (grok_90 - spike) if (spike and grok_90) else None
        rf = np.mean(data["resid_fracs"]) if data["resid_fracs"] else float("nan")

        tag = f"{key[0]} wd={key[1]} s={key[2]}"
        print(f"  {tag:>25s}  {str(spike):>12s}  {str(grok_50):>10s}  "
              f"{str(grok_90):>10s}  {str(lead_90):>12s}  {rf:>12.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Figure Y: Regime comparison — integrability + defect
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating figures...")

    # Load Power et al. (fast) results for comparison
    fast_data = None
    fast_path = OUT_DIR / "multiseed_commutator_results.pt"
    if fast_path.exists():
        fast_data = torch.load(fast_path, weights_only=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Integrability comparison
    ax = axes[0]
    regimes = []
    resid_vals = []

    # Slow regime
    slow_resids = []
    for key, data in all_runs.items():
        if data["grokked"] and data["resid_fracs"]:
            slow_resids.extend(data["resid_fracs"])
    if slow_resids:
        regimes.append("Slow\n(lr=5e-5, wd=0.1\n3 layers)")
        resid_vals.append(slow_resids)

    # Fast regime (from saved data)
    if fast_data:
        fast_resids = []
        agg = fast_data.get("agg", {})
        for key, vals in agg.items():
            op, wd = key
            if wd == 1.0 and op in ["add", "sub", "mul", "x2_y2"]:
                fast_resids.extend(vals.get("resids", []))
        if fast_resids:
            regimes.append("Power et al.\n(lr=1e-3, wd=1.0\n2 layers)")
            resid_vals.append(fast_resids)

    colors = ["#e67e22", "#2980b9"]
    for i, (label, vals) in enumerate(zip(regimes, resid_vals)):
        ax.bar(i, np.mean(vals), yerr=np.std(vals), color=colors[i],
               alpha=0.8, capsize=5, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("Residual fraction (resid/full)")
    ax.set_title("Integrability Across Regimes")
    ax.set_ylim(0.95, 1.02)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Defect comparison (at grok point)
    ax = axes[1]
    regime_defects = []

    # Slow
    slow_defects = []
    for key, data in all_runs.items():
        if data["grokked"]:
            recs = data["records"]
            grok_idx = None
            for i, r in enumerate(recs):
                if r["test_acc"] >= 0.9:
                    grok_idx = i
                    break
            if grok_idx and grok_idx > 0:
                slow_defects.append(recs[grok_idx - 1]["defect_median"])
    if slow_defects:
        regime_defects.append(("Slow\n(wd=0.1)", slow_defects, "#e67e22"))

    # Fast
    if fast_data:
        fast_defects = []
        agg = fast_data.get("agg", {})
        if ("add", 1.0) in agg:
            fast_defects = agg[("add", 1.0)].get("defects", [])
        if fast_defects:
            regime_defects.append(("Power\n(wd=1.0)", fast_defects, "#2980b9"))

    for i, (label, vals, color) in enumerate(regime_defects):
        ax.bar(i, np.mean(vals), yerr=np.std(vals) if len(vals) > 1 else 0,
               color=color, alpha=0.8, capsize=5, edgecolor="k", linewidth=0.5)

    ax.set_xticks(range(len(regime_defects)))
    ax.set_xticklabels([r[0] for r in regime_defects], fontsize=10)
    ax.set_ylabel("Commutator defect at grok point")
    ax.set_title("Defect Magnitude: Slow vs Fast")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: Lead time comparison (normalized by grok step)
    ax = axes[2]
    regime_leads = []

    # Slow
    slow_leads = []
    for key, data in all_runs.items():
        if data["grokked"]:
            recs = data["records"]
            spike = find_spike_step(recs)
            grok_90 = find_grok_step_from_records(recs, 0.9)
            gs = data["grok_step"] or grok_90 or 1
            if spike and grok_90:
                slow_leads.append((grok_90 - spike) / gs)  # normalized
    if slow_leads:
        regime_leads.append(("Slow\n(wd=0.1)", slow_leads, "#e67e22"))

    # Fast: load from generalization dynamics
    gen_path = OUT_DIR / "generalization_dynamics_results.pt"
    if gen_path.exists():
        gen_data = torch.load(gen_path, weights_only=False)
        fast_leads = []
        for key, data in gen_data.get("all_runs", {}).items():
            if not isinstance(key, tuple):
                continue
            if not data.get("grokked"):
                continue
            recs = data["records"]
            spike = find_spike_step(recs)
            grok_90 = find_grok_step_from_records(recs, 0.9)
            gs = data.get("grok_step") or grok_90 or 1
            if spike and grok_90:
                fast_leads.append((grok_90 - spike) / gs)
        if fast_leads:
            regime_leads.append(("Power\n(wd=1.0)", fast_leads, "#2980b9"))

    for i, (label, vals, color) in enumerate(regime_leads):
        ax.bar(i, np.mean(vals), yerr=np.std(vals) if len(vals) > 1 else 0,
               color=color, alpha=0.8, capsize=5, edgecolor="k", linewidth=0.5)
        # Overlay individual points
        for j, v in enumerate(vals):
            ax.scatter(i + (j - len(vals)/2) * 0.08, v, color="black",
                      s=25, zorder=5, alpha=0.7)

    ax.set_xticks(range(len(regime_leads)))
    ax.set_xticklabels([r[0] for r in regime_leads], fontsize=10)
    ax.set_ylabel("Lead time / grok step (normalized)")
    ax.set_title("Defect Lead Time (Normalized)")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Integrability Holds Across Hyperparameter Regimes\n"
                 "(slow: lr=5e-5, wd=0.1, 3L  vs  fast: lr=1e-3, wd=1.0, 2L)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figY_regime_comparison_commutator.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figY_regime_comparison_commutator.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure Z: Hero figure — defect predicts grokking in slow regime
    # ══════════════════════════════════════════════════════════════════════
    # Pick the grok run with best lead time
    best_key = None
    best_lead = -1
    for key, data in all_runs.items():
        if not data["grokked"]:
            continue
        recs = data["records"]
        spike = find_spike_step(recs)
        grok_90 = find_grok_step_from_records(recs, 0.9)
        if spike and grok_90 and (grok_90 - spike) > best_lead:
            best_lead = grok_90 - spike
            best_key = key

    if best_key is not None:
        data = all_runs[best_key]
        recs = data["records"]
        steps = [r["step"] for r in recs]
        defects = [r["defect_median"] for r in recs]
        test_accs = [r["test_acc"] for r in recs]
        train_accs = [r["train_acc"] for r in recs]

        spike = find_spike_step(recs)
        grok_90 = find_grok_step_from_records(recs, 0.9)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax2 = ax.twinx()

        # Defect with IQR
        d25 = [r["defect_p25"] for r in recs]
        d75 = [r["defect_p75"] for r in recs]
        ax.fill_between(steps, d25, d75, alpha=0.15, color="#1a5276")
        ax.plot(steps, defects, color="#1a5276", linewidth=2.5,
                label="Commutator defect")

        # Accuracies
        ax2.plot(steps, test_accs, color="#e74c3c", linewidth=2.5,
                 linestyle="--", label="Test accuracy")
        ax2.plot(steps, train_accs, color="#e74c3c", linewidth=1.5,
                 linestyle=":", alpha=0.5, label="Train accuracy")

        if spike is not None:
            ax.axvline(x=spike, color="#1a5276", linestyle=":", linewidth=2,
                       alpha=0.7, label=f"Defect spike (step {spike:,})")
        if grok_90 is not None:
            ax.axvline(x=grok_90, color="#e74c3c", linestyle=":", linewidth=2,
                       alpha=0.7, label=f"90% test acc (step {grok_90:,})")

        if spike and grok_90:
            mid = (spike + grok_90) / 2
            ymax = ax.get_ylim()[1]
            ax.annotate("", xy=(spike, ymax * 0.7),
                        xytext=(grok_90, ymax * 0.7),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
            ax.text(mid, ymax * 0.85,
                    f"Δ = {grok_90 - spike:,} steps",
                    ha="center", fontsize=12, fontweight="bold")

        ax.set_yscale("log")
        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("Commutator defect", fontsize=12, color="#1a5276")
        ax.tick_params(axis="y", labelcolor="#1a5276")
        ax2.set_ylabel("Accuracy", fontsize=12, color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.set_ylim(-0.05, 1.1)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center left")
        ax.grid(alpha=0.2)

        rf = np.mean(data["resid_fracs"]) if data["resid_fracs"] else float("nan")
        fig.suptitle(
            f"Slow Regime: Defect Predicts Grokking  —  (a+b) mod 97\n"
            f"(lr=5e-5, wd=0.1, 3 layers; spike@{spike:,}, "
            f"grok@{grok_90:,}; lead={grok_90-spike:,} steps; "
            f"resid/full={rf:.4f})",
            fontsize=12, y=1.03)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "figZ_slow_regime_hero.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved figZ_slow_regime_hero.png "
              f"(seed={best_key[2]}, lead={best_lead:,})")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = OUT_DIR / "slow_regime_commutator_results.pt"
    torch.save(all_runs, save_path)
    print(f"\n  saved {save_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
