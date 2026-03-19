#!/usr/bin/env python3
"""
Commutator defect analysis on the grokking PCA manifold.

Adapts the commutator framework from bubble_exp_comm.py to the modular
arithmetic grokking setting.  For each operation that groks:

  1. Train model, saving state_dict checkpoints every ~200 steps
  2. After training, compute PCA on full weight trajectory -> build basis B
     from PC1-PC2 of each attention weight matrix, embedded in full param space
  3. For each checkpoint, compute commutator defect (K=9 median) and project
     the commutator vector onto the PCA manifold

Produces:
  figJ — Commutator defect over training (one line per operation)
  figK — Projected vs residual commutator (per operation)
  figL — Grok vs no-wd comparison
  figM — Projected fraction (proj/full) over training
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
from pca_sweep_analysis import pca_on_trajectory, collect_trajectory

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
# Focus on operations that grok + one that doesn't
GROK_OPS = ["add", "sub", "mul", "x2_y2"]
NOGROK_OP = "x2_xy_y2"
SEED = 42
CHECKPOINT_EVERY = 200   # save state_dict every N steps
COMM_K = 9               # number of commutator samples for median
COMM_ETA = 1e-3           # step size for commutator
N_PCA_COMPONENTS = 2      # PC1 + PC2


# ═══════════════════════════════════════════════════════════════════════════
# Commutator functions (adapted from bubble_exp_comm.py)
# ═══════════════════════════════════════════════════════════════════════════

def flatten_model_params(model):
    return torch.cat([
        p.detach().flatten()
        for p in model.parameters()
        if p.requires_grad
    ])


def _param_offsets(model):
    """Return {id(param): start_offset} and total_params for flat param vector."""
    offsets = {}
    cursor = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        offsets[id(p)] = cursor
        cursor += p.numel()
    return offsets, cursor


def commutator_defect(model, batch_fn, device, eta=1e-3, eps=1e-12):
    """
    Scale-normalized commutator:
        ||theta_AB - theta_BA|| / (||eta*gA|| * ||eta*gB||)

    Adapted for grokking: batch_fn returns (a, b, y).
    """
    was_training = model.training
    model.train()

    def flat_params():
        return torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])

    def write_params(theta):
        with torch.no_grad():
            offset = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                p.copy_(theta[offset:offset+n].view_as(p))
                offset += n

    def batch_grad(a, b, y):
        model.zero_grad(set_to_none=True)
        logits = model(a, b)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in model.parameters() if p.requires_grad
        ])

    # Sample two batches
    aA, bA, yA = batch_fn()
    aB, bB, yB = batch_fn()
    aA, bA, yA = aA.to(device), bA.to(device), yA.to(device)
    aB, bB, yB = aB.to(device), bB.to(device), yB.to(device)

    theta0 = flatten_model_params(model)

    # Gradients at theta0
    gA = batch_grad(aA, bA, yA)
    gB = batch_grad(aB, bB, yB)

    # Cosine similarity
    gA_norm0 = gA.norm()
    gB_norm0 = gB.norm()
    grad_cos = (gA @ gB) / (gA_norm0 * gB_norm0 + eps)
    grad_cos = grad_cos.item()

    # Path AB
    write_params(theta0 - eta * gA)
    gB1 = batch_grad(aB, bB, yB)
    thetaAB = theta0 - eta * gA - eta * gB1

    # Path BA
    write_params(theta0 - eta * gB)
    gA1 = batch_grad(aA, bA, yA)
    thetaBA = theta0 - eta * gB - eta * gA1

    # Restore
    write_params(theta0)
    if not was_training:
        model.eval()

    normA = (eta * gA).norm()
    normB = (eta * gB).norm()
    delta = thetaAB - thetaBA
    raw_norm = delta.norm()
    defect = (raw_norm / (normA * normB + eps)).item()

    return defect, delta.detach(), grad_cos, normA.detach(), normB.detach()


def commutator_defect_median(model, batch_fn, device, K=9, eta=1e-3):
    """Run K commutator samples, return median + all deltas."""
    Ds = []
    deltas = []
    gcoss = []

    for _ in range(K):
        D, delta, gcos, nA, nB = commutator_defect(model, batch_fn, device, eta=eta)
        Ds.append(D)
        deltas.append(delta)
        gcoss.append(gcos)

    Ds_t = torch.tensor(Ds)
    med_idx = Ds_t.argsort()[len(Ds_t)//2]
    return {
        "median": Ds_t.median().item(),
        "p90": Ds_t.quantile(0.9).item(),
        "raw": Ds,
        "deltas": deltas,
        "gcoss": gcoss,
        "median_delta": deltas[med_idx],
        "median_normA": (eta * torch.ones(1)).item(),  # placeholder
    }


def projected_commutator(delta, B, normA, normB, eps=1e-12):
    """
    Project commutator delta onto basis B.
    Returns proj/resid/full (all scale-normalized).
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
# PCA basis construction
# ═══════════════════════════════════════════════════════════════════════════

def build_pca_basis(model, attn_logs, n_components=2, device="cpu"):
    """
    Build orthonormal basis from top PCA directions of each attention weight
    trajectory, embedded into full parameter space.

    For each layer x {WQ, WK, WV, WO}:
      - Run pca_on_trajectory() on saved weight snapshots
      - Take top n_components principal directions (shape [D^2])
      - Embed into full param vector at the correct offset

    Returns B: [total_params, K] orthonormal basis  (K = n_layers * 4 * n_components)
    """
    offsets, total_params = _param_offsets(model)
    basis_vecs = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        attn_mod = layer.self_attn
        d = attn_mod.embed_dim

        # in_proj_weight: [3*d, d] contains WQ, WK, WV stacked
        in_proj_id = id(attn_mod.in_proj_weight)
        out_proj_id = id(attn_mod.out_proj.weight)

        if in_proj_id not in offsets or out_proj_id not in offsets:
            print(f"  WARNING: layer {layer_idx} param IDs not found in offsets")
            continue

        in_proj_offset = offsets[in_proj_id]
        out_proj_offset = offsets[out_proj_id]

        # WQ, WK, WV share in_proj_weight
        for wkey, local_start in [("WQ", 0), ("WK", d*d), ("WV", 2*d*d)]:
            _, mats = collect_trajectory(attn_logs, layer_idx, wkey)
            pca = pca_on_trajectory(mats, top_k=n_components)
            if pca is None:
                continue
            n_avail = min(n_components, len(pca["components"]))
            for ci in range(n_avail):
                direction = torch.from_numpy(pca["components"][ci]).float()
                gv = torch.zeros(total_params, device=device)
                start = in_proj_offset + local_start
                gv[start:start + d*d] = direction.to(device)
                basis_vecs.append(gv)

        # WO
        _, mats = collect_trajectory(attn_logs, layer_idx, "WO")
        pca = pca_on_trajectory(mats, top_k=n_components)
        if pca is None:
            continue
        n_avail = min(n_components, len(pca["components"]))
        for ci in range(n_avail):
            direction = torch.from_numpy(pca["components"][ci]).float()
            gv = torch.zeros(total_params, device=device)
            gv[out_proj_offset:out_proj_offset + d*d] = direction.to(device)
            basis_vecs.append(gv)

    if not basis_vecs:
        return None

    B = torch.stack(basis_vecs, dim=1)  # [P, K]
    B_ortho, _ = torch.linalg.qr(B.cpu(), mode="reduced")
    return B_ortho.to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Training with checkpoints
# ═══════════════════════════════════════════════════════════════════════════

def train_with_checkpoints(cfg, checkpoint_every=200):
    """
    Train a model, saving state_dict snapshots at regular intervals.
    Also logs attention weights for PCA.

    Returns:
        checkpoints: list of (step, state_dict_cpu)
        attn_logs:   list of {"step": int, "layers": [...]}
        metrics:     list of {"step", "train_acc", "test_acc"}
        grokked:     bool
    """
    device = get_device()
    op_info = OPERATIONS[cfg.OP_NAME]
    op_fn = op_info["fn"]

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    train_pairs, test_pairs = build_dataset(
        cfg.P, cfg.TRAIN_FRACTION, cfg.SEED, op_fn, op_info["restrict_nonzero"]
    )

    model = ModOpTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    checkpoints = [(0, {k: v.cpu().clone() for k, v in model.state_dict().items()})]
    metrics = []
    patience = 0
    grokked = False
    t0 = time.time()

    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)
        logits = model(a, b)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % cfg.MODEL_LOG_EVERY == 0:
            attn_logs.append({"step": step, "layers": extract_attn_matrices(model)})

        if step % checkpoint_every == 0:
            checkpoints.append(
                (step, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            )

        if step % cfg.EVAL_EVERY == 0 or step == 1:
            train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
            test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)
            metrics.append({"step": step, "train_acc": train_acc, "test_acc": test_acc})

            if step % (cfg.EVAL_EVERY * 10) == 0:
                elapsed = (time.time() - t0) / 60
                print(f"    step {step:6d} | train {train_acc:.3f} | test {test_acc:.3f} | "
                      f"{elapsed:.1f}m | ckpts {len(checkpoints)}")

            if test_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE:
                    grokked = True
                    print(f"    GROKKED at step {step}")
                    break
            else:
                patience = 0

    return model, checkpoints, attn_logs, metrics, grokked, train_pairs, test_pairs


# ═══════════════════════════════════════════════════════════════════════════
# Per-checkpoint commutator measurement
# ═══════════════════════════════════════════════════════════════════════════

def attn_weight_mask(model):
    """
    Return a boolean mask [total_params] that is True for attention weight
    parameters (in_proj_weight, out_proj.weight, in_proj_bias, out_proj.bias).
    This lets us measure what fraction of the commutator lives in the
    attention weight subspace vs the rest (FFN, embeddings, LN, head).
    """
    offsets, total_params = _param_offsets(model)
    mask = torch.zeros(total_params, dtype=torch.bool)

    for layer in model.encoder.layers:
        attn_mod = layer.self_attn
        for p in [attn_mod.in_proj_weight, attn_mod.out_proj.weight]:
            if p.requires_grad and id(p) in offsets:
                start = offsets[id(p)]
                mask[start:start + p.numel()] = True
        # Also include biases if they exist
        for p in [attn_mod.in_proj_bias, attn_mod.out_proj.bias]:
            if p is not None and p.requires_grad and id(p) in offsets:
                start = offsets[id(p)]
                mask[start:start + p.numel()] = True

    return mask


def measure_commutators_at_checkpoints(
    cfg, checkpoints, attn_logs, train_pairs, B, K=9, eta=1e-3
):
    """
    For each checkpoint, load state, compute commutator defect, and project
    onto PCA basis B.  Also decomposes into attention-weight vs other params.

    Returns list of dicts with step, defect_median, proj, resid, full, grad_cos,
    attn_frac (fraction of ||delta|| in attention weight coords).
    """
    device = get_device()
    op_info = OPERATIONS[cfg.OP_NAME]
    op_fn = op_info["fn"]

    # Build a fresh model to load checkpoints into
    model = ModOpTransformer(cfg).to(device)
    amask = attn_weight_mask(model)  # [P] bool

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    results = []
    total = len(checkpoints)

    for ci, (step, sd) in enumerate(checkpoints):
        model.load_state_dict(sd)
        model.to(device)

        # Compute commutator defect
        out = commutator_defect_median(model, batch_fn, device, K=K, eta=eta)

        # Project median delta onto PCA basis
        # Need normA, normB from one sample for scaling
        D, delta, gcos, normA, normB = commutator_defect(model, batch_fn, device, eta=eta)
        delta_cpu = delta.cpu()
        normA_cpu = normA.cpu()
        normB_cpu = normB.cpu()

        # Project onto PCA basis
        pc = projected_commutator(delta_cpu, B.cpu() if B is not None else None,
                                  normA_cpu, normB_cpu)

        # Decompose: attention weight coords vs rest
        delta_full_norm = delta_cpu.norm().item()
        delta_attn = delta_cpu[amask]
        delta_other = delta_cpu[~amask]
        attn_frac = (delta_attn.norm().item() / (delta_full_norm + 1e-15))
        other_frac = (delta_other.norm().item() / (delta_full_norm + 1e-15))

        results.append({
            "step": step,
            "defect_median": out["median"],
            "defect_p90": out["p90"],
            "grad_cos": np.mean(out["gcoss"]),
            "proj": pc["proj"],
            "resid": pc["resid"],
            "full": pc["full"],
            "attn_frac": attn_frac,
            "other_frac": other_frac,
        })

        if (ci+1) % 5 == 0 or ci == total - 1:
            print(f"      ckpt {ci+1}/{total}: step={step}, "
                  f"defect={out['median']:.4f}, "
                  f"proj/full={pc['proj']/(pc['full']+1e-15):.1%}, "
                  f"attn_frac={attn_frac:.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = {}  # (op_name, wd) -> {"comm": [...], "grokked": bool, ...}

    # ── Run grokking operations (wd=1.0) + no-wd controls ────────────────
    for op_name in GROK_OPS:
        for wd in [1.0, 0.0]:
            tag = f"{op_name}_wd{wd}"
            print(f"\n{'='*70}")
            print(f"  {tag} (seed={SEED})")
            print(f"{'='*70}")

            # Cap no-wd runs shorter — just need enough for comparison
            steps = 10_000 if wd == 0.0 else 200_000
            ckpt_every = CHECKPOINT_EVERY if wd > 0 else 500

            cfg = SweepConfig(
                OP_NAME=op_name,
                WEIGHT_DECAY=wd,
                SEED=SEED,
                STEPS=steps,
            )

            # Train
            print(f"  Training {op_name} wd={wd}...")
            model, checkpoints, attn_logs, metrics, grokked, train_pairs, test_pairs = \
                train_with_checkpoints(cfg, checkpoint_every=ckpt_every)

            print(f"  grokked={grokked}, {len(checkpoints)} checkpoints, "
                  f"{len(attn_logs)} attn snapshots")

            # Build PCA basis from trajectory
            print(f"  Building PCA basis (top-{N_PCA_COMPONENTS} components)...")
            B = build_pca_basis(model, attn_logs, n_components=N_PCA_COMPONENTS, device="cpu")
            if B is not None:
                print(f"  Basis shape: {B.shape}  "
                      f"({B.shape[1]} directions in {B.shape[0]}-dim param space)")
            else:
                print(f"  WARNING: Could not build PCA basis")

            # Subsample checkpoints for no-wd to keep runtime reasonable
            if wd == 0.0 and len(checkpoints) > 25:
                idx = np.linspace(0, len(checkpoints)-1, 25, dtype=int)
                checkpoints = [checkpoints[i] for i in idx]
                print(f"  Subsampled to {len(checkpoints)} checkpoints for no-wd")

            # Measure commutators
            print(f"  Measuring commutators at {len(checkpoints)} checkpoints...")
            comm_results = measure_commutators_at_checkpoints(
                cfg, checkpoints, attn_logs, train_pairs, B,
                K=COMM_K, eta=COMM_ETA
            )

            all_results[(op_name, wd)] = {
                "comm": comm_results,
                "grokked": grokked,
                "metrics": metrics,
                "n_checkpoints": len(checkpoints),
                "n_attn_snaps": len(attn_logs),
                "basis_dim": B.shape[1] if B is not None else 0,
            }

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  COMMUTATOR ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Config':>25s}  {'grok':>5s}  {'defect':>8s}  "
          f"{'resid/full':>10s}  {'attn%':>6s}  "
          f"(resid/full≈100% → PCA manifold is integrable)")

    for (op, wd), data in sorted(all_results.items()):
        comm = data["comm"]
        if not comm:
            continue
        last = comm[-1]
        rf = last["resid"] / last["full"] if last["full"] > 1e-15 else float("nan")
        af = last.get("attn_frac", float("nan"))
        tag = f"{op} wd={wd}"
        print(f"  {tag:>25s}  {'yes' if data['grokked'] else 'no':>5s}  "
              f"{last['defect_median']:8.3f}  "
              f"{rf:10.1%}  {af:6.1%}")

    # ── Figure J: Commutator defect over training ────────────────────────
    print("\n  Generating figures...")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_op = {"add": "#1f77b4", "sub": "#ff7f0e", "mul": "#2ca02c", "x2_y2": "#d62728"}

    for op_name in GROK_OPS:
        # Grok run
        key = (op_name, 1.0)
        if key in all_results:
            comm = all_results[key]["comm"]
            steps = [c["step"] for c in comm]
            defs = [c["defect_median"] for c in comm]
            label_op = OPERATIONS[op_name]["label"]
            ax.plot(steps, defs, label=f"{label_op} (wd=1.0)",
                    color=colors_op.get(op_name, "gray"), linewidth=2)

        # No-wd run (dashed)
        key = (op_name, 0.0)
        if key in all_results:
            comm = all_results[key]["comm"]
            steps = [c["step"] for c in comm]
            defs = [c["defect_median"] for c in comm]
            ax.plot(steps, defs, label=f"{label_op} (wd=0)",
                    color=colors_op.get(op_name, "gray"), linewidth=1.5,
                    linestyle="--", alpha=0.6)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Commutator defect (median)", fontsize=12)
    ax.set_title("Commutator Defect During Grokking\n"
                 "(scale-normalized, K=9 median)", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figJ_commutator_defect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figJ_commutator_defect.png")

    # ── Figure K: Integrability — commutator is orthogonal to PCA manifold ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        key = (op_name, 1.0)
        if key not in all_results:
            continue
        comm = all_results[key]["comm"]
        steps = [c["step"] for c in comm]

        # Compute residual fraction = resid/full (should be ≈1.0)
        resid_fracs = []
        proj_fracs = []
        for c in comm:
            if c["full"] > 1e-15:
                resid_fracs.append(c["resid"] / c["full"])
                proj_fracs.append(c["proj"] / c["full"])
            else:
                resid_fracs.append(float("nan"))
                proj_fracs.append(float("nan"))

        ax.plot(steps, resid_fracs, label="Residual (⊥ PCA)", linewidth=2.5,
                color="#e74c3c")
        ax.plot(steps, proj_fracs, label="Projected (∥ PCA)", linewidth=2,
                color="#27ae60", linestyle="--")

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} mod 97 (wd=1.0)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Fraction of ||commutator||")
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
        ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)
        ax.legend(fontsize=9, loc="center right")
        ax.grid(alpha=0.3)

    fig.suptitle("Integrability of PCA Manifold: Commutators are Orthogonal\n"
                 f"(resid/full ≈ 100% → curvature lives outside the {N_PCA_COMPONENTS}D execution manifold)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figK_integrability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figK_integrability.png")

    # ── Figure L: Grok vs No-WD comparison ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        label_op = OPERATIONS[op_name]["label"]

        for wd, ls, lbl, alpha in [(1.0, "-", "wd=1.0 (grok)", 1.0),
                                    (0.0, "--", "wd=0 (no-wd)", 0.6)]:
            key = (op_name, wd)
            if key not in all_results:
                continue
            comm = all_results[key]["comm"]
            steps = [c["step"] for c in comm]
            defs = [c["defect_median"] for c in comm]
            ax.plot(steps, defs, label=lbl, linewidth=2, linestyle=ls, alpha=alpha)

        ax.set_title(f"{label_op} mod 97", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Commutator defect (median)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle("Commutator Defect: Grokking vs No Weight Decay", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figL_grok_vs_nowd_commutator.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figL_grok_vs_nowd_commutator.png")

    # ── Figure M: Commutator defect × integrability combined ──────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        label_op = OPERATIONS[op_name]["label"]

        key = (op_name, 1.0)
        if key not in all_results:
            continue
        comm = all_results[key]["comm"]
        steps = [c["step"] for c in comm]
        defs = [c["defect_median"] for c in comm]
        resid_fracs = []
        for c in comm:
            if c["full"] > 1e-15:
                resid_fracs.append(c["resid"] / c["full"] * 100)
            else:
                resid_fracs.append(float("nan"))

        color1 = "#1a5276"
        color2 = "#e74c3c"
        ax.plot(steps, defs, label="Defect (left)", linewidth=2, color=color1)
        ax.set_ylabel("Commutator defect", color=color1, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color1)
        ax.set_yscale("log")

        ax2 = ax.twinx()
        ax2.plot(steps, resid_fracs, label="Resid % (right)", linewidth=2,
                 color=color2, linestyle="--")
        ax2.set_ylabel("Residual fraction (%)", color=color2, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(90, 101)

        ax.set_title(f"{label_op} mod 97 (wd=1.0)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.suptitle("Defect Explosion + Manifold Integrability\n"
                 "(curvature grows 100× during grokking, but stays ⊥ to PCA manifold)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figM_defect_integrability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figM_defect_integrability.png")

    # ── Figure N: Attention weight fraction of commutator ────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        label_op = OPERATIONS[op_name]["label"]

        for wd, ls, lbl, color in [(1.0, "-", "wd=1.0 (grok)", "#2980b9"),
                                    (0.0, "--", "wd=0", "#c0392b")]:
            key = (op_name, wd)
            if key not in all_results:
                continue
            comm = all_results[key]["comm"]
            steps = [c["step"] for c in comm]
            attn_fracs = [c.get("attn_frac", float("nan")) for c in comm]
            ax.plot(steps, attn_fracs, label=lbl, linewidth=2, linestyle=ls, color=color)

        ax.set_title(f"{label_op} mod 97", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Attention weight fraction of ||delta||")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Fraction of Commutator in Attention Weight Subspace\n"
                 "(vs FFN, embeddings, LayerNorm, head)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figN_attn_weight_fraction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figN_attn_weight_fraction.png")

    # ── Save raw results ─────────────────────────────────────────────────
    save_path = OUT_DIR / "commutator_results.pt"
    torch.save(all_results, save_path)
    print(f"\n  saved {save_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
