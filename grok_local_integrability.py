#!/usr/bin/env python3
"""
Local integrability test: exactly replicating bubble_exp_comm.py approach.

At each measurement step:
  1. Build a joint basis B [P, K] from top-k SVD of CURRENT weight matrices
     (WQ, WK, WV, WO per layer), embedded in full parameter space, QR'd.
     This is extract_parameter_subspace() from bubble_exp_comm.py.
  2. Compute commutator delta [P] at that point.
  3. Project delta onto B: proj = B @ (B.T @ delta), resid = delta - proj.
  4. Report proj/full, resid/full.

This differs from grok_commutator_analysis.py (which used global PCA over
the trajectory) in that the basis is LOCAL — recomputed from the current
weight matrices at each step.

Produces:
  figL1 — proj/full and resid/full over training (4 ops, seed=42)
  figL2 — Multi-seed replication
  figL3 — proj/full vs random baseline √(K/P)
  figL4 — Multi-op comparison bar chart
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
    OPERATIONS, get_device, eval_accuracy,
)
from grok_commutator_analysis import (
    flatten_model_params, _param_offsets, commutator_defect,
)

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
GROK_OPS = ["add", "sub", "mul", "x2_y2"]
NOGROK_OPS = ["x2_xy_y2", "x3_xy"]
SEEDS = [42, 137, 2024]

COMM_EVERY = 100        # measurement interval (steps)
COMM_K = 5              # commutator samples per measurement
COMM_ETA = 1e-3
MAX_STEPS = 200_000
POST_GROK_STEPS = 1000

SVD_TOPK = 3            # top-k singular directions per weight block


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Local basis — exact copy of bubble_exp_comm.py approach
# ═══════════════════════════════════════════════════════════════════════════

def _block_basis(block, topk):
    """Top-k singular directions (flattened) for a weight block.
    Exactly as in bubble_exp_comm.py line 356."""
    U, S, Vh = torch.linalg.svd(block, full_matrices=False)
    vecs = []
    r = min(topk, S.numel())
    for i in range(r):
        comp = S[i] * (U[:, i].unsqueeze(1) @ Vh[i].unsqueeze(0))
        flat = comp.reshape(-1)
        norm = flat.norm()
        if norm > 0:
            vecs.append(flat / norm)
    return vecs


def extract_parameter_subspace(model, k=3, device="cpu"):
    """
    Build a joint basis from top-k SVD of WQ/WK/WV/WO per layer,
    embedded in full parameter space.

    Adapted from bubble_exp_comm.py line 369 for ModOpTransformer
    (which uses nn.TransformerEncoder with in_proj_weight / out_proj.weight
    instead of the custom qkv/out layers in BubbleTransformer).

    Returns:
        B: [P, K] orthonormal basis
    """
    offsets, total_params = _param_offsets(model)
    basis_vecs = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim  # 128

        # in_proj_weight: [3*d, d] = [WQ; WK; WV] stacked
        ip_w = attn.in_proj_weight.detach()
        ip_id = id(attn.in_proj_weight)
        ip_offset = offsets.get(ip_id, None)

        if ip_offset is not None:
            for wkey, row_start in [("WQ", 0), ("WK", d), ("WV", 2*d)]:
                block = ip_w[row_start:row_start+d, :]  # [d, d]
                local_start = row_start * d  # offset within in_proj_weight
                for vec in _block_basis(block, k):
                    gv = torch.zeros(total_params, device=device)
                    start = ip_offset + local_start
                    gv[start:start + block.numel()] = vec.to(device)
                    basis_vecs.append(gv)

        # out_proj.weight: [d, d]
        out_w = attn.out_proj.weight.detach()
        out_id = id(attn.out_proj.weight)
        out_offset = offsets.get(out_id, None)

        if out_offset is not None:
            for vec in _block_basis(out_w, k):
                gv = torch.zeros(total_params, device=device)
                gv[out_offset:out_offset + out_w.numel()] = vec.to(device)
                basis_vecs.append(gv)

    if not basis_vecs:
        return None

    B = torch.stack(basis_vecs, dim=1)  # [P, K]
    # QR on CPU (MPS doesn't support linalg_qr)
    B_cpu = B.cpu() if B.device.type != "cpu" else B
    B_ortho, _ = torch.linalg.qr(B_cpu, mode="reduced")
    return B_ortho.to(device)


def projected_commutator(delta, B, normA, normB, eps=1e-12):
    """
    Project commutator delta onto basis B.
    Exactly as in bubble_exp_comm.py line 422.

    Returns: {proj, resid, full} all scale-normalized.
    """
    delta = delta.reshape(-1)

    if B is None or delta.numel() != B.shape[0]:
        full_val = (delta.norm() / (normA * normB + eps)).item()
        return {"proj": float("nan"), "resid": float("nan"), "full": full_val}

    coeffs = B.T @ delta
    proj = B @ coeffs
    resid = delta - proj

    scale = normA * normB + eps
    return {
        "proj": (proj.norm() / scale).item(),
        "resid": (resid.norm() / scale).item(),
        "full": (delta.norm() / scale).item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Training with local integrability measurement
# ═══════════════════════════════════════════════════════════════════════════

def train_with_local_integrability(op_name, wd, seed, max_steps=None):
    """
    Train a model, measuring local integrability at regular intervals.
    Exactly follows the bubble_exp_comm.py pattern:
      - extract_parameter_subspace(model, k=3) recomputed at each step
      - projected_commutator(delta, B, normA, normB)
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

    offsets, total_params = _param_offsets(model)
    print(f"    Total params: {total_params}")

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    records = []
    grokked = False
    grok_step = None
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    def measure_step(step):
        """Take one local integrability measurement."""
        model.eval()

        # 1. Accuracy
        train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
        test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)

        # 2. Build LOCAL basis from current weight SVD
        B = extract_parameter_subspace(model, k=SVD_TOPK, device="cpu")
        K = B.shape[1] if B is not None else 0
        random_baseline = math.sqrt(K / total_params) if total_params > 0 else 0.0

        # 3. Compute commutators and project onto local basis
        model.train()
        proj_vals = []
        resid_vals = []
        full_vals = []
        defect_vals = []

        for _ in range(COMM_K):
            D_val, delta, gcos, nA, nB = commutator_defect(
                model, batch_fn, device, eta=COMM_ETA
            )
            defect_vals.append(D_val)

            delta_cpu = delta.detach().cpu()
            nA_cpu = nA.cpu() if hasattr(nA, 'cpu') else torch.tensor(nA)
            nB_cpu = nB.cpu() if hasattr(nB, 'cpu') else torch.tensor(nB)

            pc = projected_commutator(delta_cpu, B, nA_cpu, nB_cpu)
            proj_vals.append(pc["proj"])
            resid_vals.append(pc["resid"])
            full_vals.append(pc["full"])

        # Median across K samples
        proj_med = float(np.median(proj_vals))
        resid_med = float(np.median(resid_vals))
        full_med = float(np.median(full_vals))
        defect_med = float(np.median(defect_vals))

        proj_frac = proj_med / (full_med + 1e-15)
        resid_frac = resid_med / (full_med + 1e-15)
        ratio_to_random = proj_frac / (random_baseline + 1e-15)

        rec = {
            "step": step,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "defect_median": defect_med,
            "proj": proj_med,
            "resid": resid_med,
            "full": full_med,
            "proj_frac": proj_frac,
            "resid_frac": resid_frac,
            "K": K,
            "random_baseline": random_baseline,
            "ratio_to_random": ratio_to_random,
        }
        return rec

    # ── Measure at step 0 ────────────────────────────────────────────────
    rec0 = measure_step(0)
    records.append(rec0)
    print(f"      step 0 | test {rec0['test_acc']:.3f} | defect {rec0['defect_median']:.1f} "
          f"| proj/full={rec0['proj_frac']:.3f} resid/full={rec0['resid_frac']:.3f} "
          f"| ratio={rec0['ratio_to_random']:.2f}x")

    # ── Training loop ────────────────────────────────────────────────────
    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)
        logits = model(a, b)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        # Measure at intervals
        if step % COMM_EVERY == 0:
            rec = measure_step(step)
            records.append(rec)

        # Check for grokking
        if step % cfg.EVAL_EVERY == 0:
            if step % COMM_EVERY == 0:
                test_acc = records[-1]["test_acc"]
            else:
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
                if step % COMM_EVERY != 0:
                    rec = measure_step(step)
                    records.append(rec)
                break

        # Progress
        if step % 500 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            print(f"      step {step:6d} | test {last_r.get('test_acc',0):.3f} | "
                  f"defect {last_r.get('defect_median',0):.1f} | "
                  f"proj/full={last_r.get('proj_frac',0):.3f} "
                  f"resid/full={last_r.get('resid_frac',0):.3f} | "
                  f"ratio={last_r.get('ratio_to_random',0):.2f}x | {elapsed:.1f}m")

    return {
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "op": op_name,
        "wd": wd,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_L1_proj_resid(all_results):
    """
    figL1: proj/full and resid/full over training for each grokking op (seed=42).
    Exactly the plot bubble_exp_comm.py would produce.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        key = (op_name, 1.0, 42)
        if key not in all_results:
            ax.set_title(f"{op_name} — no data")
            continue

        data = all_results[key]
        records = data["records"]
        steps = [r["step"] for r in records]
        proj_fracs = [r["proj_frac"] for r in records]
        resid_fracs = [r["resid_frac"] for r in records]
        random_bl = records[0]["random_baseline"] if records else 0

        ax.plot(steps, proj_fracs, label="proj/full (∥ local basis)",
                linewidth=2.5, color="#27ae60")
        ax.plot(steps, resid_fracs, label="resid/full (⊥ local basis)",
                linewidth=2.5, color="#e74c3c")
        ax.axhline(y=random_bl, color="gray", linestyle=":",
                   linewidth=1.5, alpha=0.7,
                   label=f"Random baseline √(K/P)={random_bl:.4f}")

        # Grok step
        if data["grokked"] and data["grok_step"]:
            ax.axvline(x=data["grok_step"], color="blue", linestyle="--",
                      linewidth=2, alpha=0.5, label=f"Grok @ {data['grok_step']}")

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} (seed=42, wd=1.0)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Fraction of ||commutator||")
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.2)
        ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.2)
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)

    fig.suptitle("Local Integrability (bubble_exp_comm.py method)\n"
                 "Basis = top-3 weight SVD per block, recomputed at each step",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figL1_local_proj_resid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figL1_local_proj_resid.png")


def fig_L2_multiseed(all_results):
    """
    figL2: proj/full with 3 seeds overlaid per op, plus test accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    seed_colors = {42: "#1f77b4", 137: "#ff7f0e", 2024: "#2ca02c"}

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]

        for seed in SEEDS:
            key = (op_name, 1.0, seed)
            if key not in all_results:
                continue
            data = all_results[key]
            records = data["records"]
            steps = [r["step"] for r in records]
            proj_fracs = [r["proj_frac"] for r in records]

            ax.plot(steps, proj_fracs, label=f"proj/full (seed={seed})",
                    linewidth=1.5, color=seed_colors[seed], alpha=0.8)

            # Also plot test_acc on secondary axis
            if seed == 42:
                ax2 = ax.twinx()
                test_accs = [r["test_acc"] for r in records]
                ax2.plot(steps, test_accs, linewidth=1.5, color="gray",
                         linestyle="--", alpha=0.4, label="test acc (s=42)")
                ax2.set_ylabel("Test accuracy", color="gray", fontsize=9)
                ax2.set_ylim(-0.05, 1.05)

        # Random baseline
        any_key = (op_name, 1.0, 42)
        if any_key in all_results:
            recs = all_results[any_key]["records"]
            if recs:
                rb = recs[0]["random_baseline"]
                ax.axhline(y=rb, color="red", linestyle=":", linewidth=1.5,
                           alpha=0.5, label=f"Random √(K/P)={rb:.4f}")

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} (wd=1.0)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("proj / full")
        ax.set_ylim(-0.05, max(0.5, ax.get_ylim()[1]))
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

    fig.suptitle("Local Integrability — Multi-Seed Replication\n"
                 "(proj/full: fraction of commutator in local weight SVD basis)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figL2_local_multiseed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figL2_local_multiseed.png")


def fig_L3_ratio_to_random(all_results):
    """
    figL3: ratio = (proj/full) / √(K/P) over training.
    ratio > 1 means commutator aligns with local weight structure beyond random.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        key = (op_name, 1.0, 42)
        if key not in all_results:
            ax.set_title(f"{op_name} — no data")
            continue

        data = all_results[key]
        records = data["records"]
        steps = [r["step"] for r in records]
        ratios = [r["ratio_to_random"] for r in records]
        defects = [r["defect_median"] for r in records]

        # Primary: ratio
        color1 = "#1a5276"
        ax.plot(steps, ratios, linewidth=2, color=color1, label="ratio to random")
        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2,
                   alpha=0.7, label="Random baseline")
        ax.set_ylabel("proj/full ÷ √(K/P)", color=color1, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color1)

        # Secondary: defect
        ax2 = ax.twinx()
        ax2.plot(steps, defects, linewidth=1.5, color="#e67e22",
                 linestyle="--", alpha=0.6, label="defect")
        ax2.set_ylabel("Commutator defect", color="#e67e22", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#e67e22")
        ax2.set_yscale("log")

        # Grok step
        if data["grokked"] and data["grok_step"]:
            ax.axvline(x=data["grok_step"], color="green", linestyle="--",
                      linewidth=2, alpha=0.4)

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} (seed=42, wd=1.0)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("Local Integrability Ratio Over Training\n"
                 "(ratio > 1 = commutator aligns with local weight structure)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figL3_local_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figL3_local_ratio.png")


def fig_L4_multiop_bars(all_results):
    """
    figL4: Bar chart comparing local integrability across ops and phases.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Grokking ops, 3 phases
    ax = axes[0]
    x_positions = np.arange(len(GROK_OPS))
    width = 0.25
    phases = ["early", "pre-grok", "post-grok"]
    phase_colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for pi, phase in enumerate(phases):
        means = []
        stds = []
        for op_name in GROK_OPS:
            seed_vals = []
            for seed in SEEDS:
                key = (op_name, 1.0, seed)
                if key not in all_results:
                    continue
                data = all_results[key]
                records = data["records"]
                gs = data.get("grok_step", None)

                if phase == "early":
                    candidates = [r for r in records if 100 <= r["step"] <= 500]
                elif phase == "pre-grok":
                    if gs:
                        candidates = [r for r in records
                                      if gs - 1000 <= r["step"] <= gs - 200]
                    else:
                        n = len(records)
                        candidates = records[n*2//3:n*5//6]
                elif phase == "post-grok":
                    if gs:
                        candidates = [r for r in records if r["step"] > gs + 200]
                    else:
                        candidates = records[-3:]

                for r in candidates:
                    seed_vals.append(r["proj_frac"])

            means.append(np.mean(seed_vals) if seed_vals else 0)
            stds.append(np.std(seed_vals) if seed_vals else 0)

        ax.bar(x_positions + pi * width, means, width, yerr=stds,
               label=phase, color=phase_colors[pi], alpha=0.8, capsize=3)

    # Random baseline
    any_key = (GROK_OPS[0], 1.0, 42)
    if any_key in all_results:
        rb = all_results[any_key]["records"][0]["random_baseline"]
        ax.axhline(y=rb, color="red", linestyle=":", linewidth=2, alpha=0.5,
                   label=f"Random √(K/P)={rb:.4f}")

    ax.set_xticks(x_positions + width)
    ax.set_xticklabels([OPERATIONS[op]["label"] for op in GROK_OPS], fontsize=9)
    ax.set_ylabel("proj / full")
    ax.set_title("Grokking Ops: proj/full by Phase")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: All ops, late training
    ax = axes[1]
    all_ops = GROK_OPS + NOGROK_OPS
    x_positions = np.arange(len(all_ops))

    means = []
    stds = []
    for op_name in all_ops:
        seed_vals = []
        for seed in SEEDS:
            key = (op_name, 1.0, seed)
            if key not in all_results:
                continue
            data = all_results[key]
            records = data["records"]
            for r in records[-5:]:
                seed_vals.append(r["proj_frac"])

        means.append(np.mean(seed_vals) if seed_vals else 0)
        stds.append(np.std(seed_vals) if seed_vals else 0)

    colors = ["#2ecc71" if op in GROK_OPS else "#e74c3c" for op in all_ops]
    ax.bar(x_positions, means, 0.6, yerr=stds, color=colors, alpha=0.8, capsize=3)

    if any_key in all_results:
        rb = all_results[any_key]["records"][0]["random_baseline"]
        ax.axhline(y=rb, color="red", linestyle=":", linewidth=2, alpha=0.5,
                   label=f"Random √(K/P)={rb:.4f}")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([OPERATIONS[op]["label"] for op in all_ops],
                       fontsize=8, rotation=15)
    ax.set_ylabel("proj / full (late training)")
    ax.set_title("Grokking vs Non-Grokking")
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ecc71", label="Groks"),
                       Patch(facecolor="#e74c3c", label="Does not grok")]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.suptitle("Local Integrability: proj/full Across Operations\n"
                 "(bubble_exp_comm.py method: joint basis, recomputed each step)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figL4_local_multiop_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figL4_local_multiop_bars.png")


def fig_L5_combined_hero(all_results):
    """
    figL5: Hero figure — defect, proj/full, resid/full, test_acc all on one plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        key = (op_name, 1.0, 42)
        if key not in all_results:
            ax.set_title(f"{op_name} — no data")
            continue

        data = all_results[key]
        records = data["records"]
        steps = [r["step"] for r in records]

        # Left axis: defect (log scale)
        defects = [r["defect_median"] for r in records]
        ax.plot(steps, defects, linewidth=2, color="#e74c3c", label="Defect")
        ax.set_yscale("log")
        ax.set_ylabel("Commutator defect", color="#e74c3c", fontsize=10)
        ax.tick_params(axis="y", labelcolor="#e74c3c")

        # Right axis: proj/full + test_acc
        ax2 = ax.twinx()
        proj_fracs = [r["proj_frac"] for r in records]
        test_accs = [r["test_acc"] for r in records]

        ax2.plot(steps, proj_fracs, linewidth=2, color="#2ecc71",
                 label="proj/full")
        ax2.plot(steps, test_accs, linewidth=1.5, color="#3498db",
                 linestyle="--", alpha=0.6, label="test acc")
        ax2.set_ylabel("proj/full & test acc", fontsize=10)
        ax2.set_ylim(-0.05, 1.05)

        # Random baseline
        rb = records[0]["random_baseline"]
        ax2.axhline(y=rb, color="gray", linestyle=":", linewidth=1.5,
                    alpha=0.5, label=f"Random={rb:.4f}")

        if data["grokked"] and data["grok_step"]:
            ax.axvline(x=data["grok_step"], color="blue", linestyle="--",
                      linewidth=2, alpha=0.3)

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} (seed=42, wd=1.0)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left")

    fig.suptitle("Local Integrability: Defect × Projection × Test Accuracy\n"
                 "(basis = weight SVD recomputed each step)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figL5_local_hero.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figL5_local_hero.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = OUT_DIR / "local_integrability_v2.pt"
    if cache_path.exists():
        print(f"Loading cached results from {cache_path.name}...")
        all_results = torch.load(cache_path, weights_only=False)
    else:
        all_results = {}

    # ── Phase 1: Grokking ops, 3 seeds ──────────────────────────────────
    for op_name in GROK_OPS:
        for seed in SEEDS:
            key = (op_name, 1.0, seed)
            if key in all_results:
                print(f"\n  CACHED: {op_name} wd=1.0 seed={seed}")
                continue

            print(f"\n{'='*70}")
            print(f"  {op_name} wd=1.0 seed={seed}")
            print(f"{'='*70}")

            result = train_with_local_integrability(op_name, wd=1.0, seed=seed)
            all_results[key] = result

            torch.save(all_results, cache_path)
            print(f"  saved checkpoint ({len(all_results)} total runs)")

    # ── Phase 2: Non-grokking controls ──────────────────────────────────
    for op_name in NOGROK_OPS:
        key = (op_name, 1.0, 42)
        if key in all_results:
            print(f"\n  CACHED: {op_name} wd=1.0 seed=42")
            continue

        print(f"\n{'='*70}")
        print(f"  {op_name} wd=1.0 seed=42 (non-grokking control)")
        print(f"{'='*70}")

        result = train_with_local_integrability(
            op_name, wd=1.0, seed=42, max_steps=10_000,
        )
        all_results[key] = result
        torch.save(all_results, cache_path)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  LOCAL INTEGRABILITY SUMMARY (bubble_exp_comm.py method)")
    print(f"{'='*90}")
    print(f"  {'Config':>25s}  {'grok':>5s}  {'grok_step':>10s}  "
          f"{'K':>4s}  {'√(K/P)':>7s}  "
          f"{'proj/full':>10s}  {'resid/full':>10s}  "
          f"{'ratio':>7s}  {'defect':>8s}")

    for key in sorted(all_results.keys()):
        data = all_results[key]
        op, wd, seed = key
        records = data["records"]
        gs = data.get("grok_step", None)

        # Late training values
        late = records[-5:] if len(records) >= 5 else records
        pf = np.mean([r["proj_frac"] for r in late])
        rf = np.mean([r["resid_frac"] for r in late])
        ratio = np.mean([r["ratio_to_random"] for r in late])
        defect = np.mean([r["defect_median"] for r in late])
        K = late[0]["K"] if late else 0
        rb = late[0]["random_baseline"] if late else 0

        tag = f"{op} wd={wd} s={seed}"
        print(f"  {tag:>25s}  {'yes' if data['grokked'] else 'no':>5s}  "
              f"{str(gs) if gs else '—':>10s}  "
              f"{K:4d}  {rb:7.4f}  "
              f"{pf:10.4f}  {rf:10.4f}  "
              f"{ratio:7.1f}x  {defect:8.1f}")

    # ── Figures ──────────────────────────────────────────────────────────
    print("\n  Generating figures...")
    fig_L1_proj_resid(all_results)
    fig_L2_multiseed(all_results)
    fig_L3_ratio_to_random(all_results)
    fig_L4_multiop_bars(all_results)
    fig_L5_combined_hero(all_results)

    # ── Save final ───────────────────────────────────────────────────────
    torch.save(all_results, cache_path)
    print(f"\n  Final results saved to {cache_path.name}")
    print(f"  Total runs: {len(all_results)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
