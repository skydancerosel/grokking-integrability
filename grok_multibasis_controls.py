#!/usr/bin/env python3
"""
Multi-basis integrability controls.

Tests whether the sign flip (exec/random > 1 during memorization,
< 1 post-grok) holds for three different "learning tangent" definitions:

  1. Weight SVD   — top-k SVD of current W  (what we already had)
  2. ΔW-SVD       — top-k SVD of (W_t - W_0)  (weight displacement since init)
  3. Gradient SVD  — top-k SVD of accumulated recent gradients

AND does per-block decomposition:
  - For each block (WQ, WK, WV, WO, MLP1, MLP2 per layer), project the
    block-portion of the commutator onto the block's own basis vs random.

If the sign flip holds across all basis types → "commutator energy leaves
the learning tangent at grokking" — a basis-independent geometric claim.
"""

import math, time, random, sys, copy
from pathlib import Path
from collections import defaultdict

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
from grok_local_integrability import (
    _block_basis, projected_commutator,
)

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
GROK_OPS = ["add", "sub", "mul", "x2_y2"]

COMM_EVERY = 100
COMM_K = 5
COMM_ETA = 1e-3
MAX_STEPS = 200_000
POST_GROK_STEPS = 1500

SVD_TOPK = 3
N_RANDOM_TRIALS = 5
GRAD_WINDOW = 50        # accumulate gradients over this many recent steps


# ═══════════════════════════════════════════════════════════════════════════
# Block registry: enumerate all weight blocks with offsets
# ═══════════════════════════════════════════════════════════════════════════

def get_block_registry(model):
    """
    Returns list of dicts describing each weight block:
      name, layer_idx, param_name, offset_in_flat, shape, numel
    """
    offsets, total_params = _param_offsets(model)
    blocks = []

    for li, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim  # 128

        # WQ, WK, WV from in_proj_weight [3d, d]
        ip_id = id(attn.in_proj_weight)
        ip_off = offsets.get(ip_id, None)
        if ip_off is not None:
            for name, row_start in [("WQ", 0), ("WK", d), ("WV", 2*d)]:
                blocks.append({
                    "name": f"L{li}_{name}",
                    "layer": li,
                    "param_name": f"encoder.layers.{li}.self_attn.in_proj_weight",
                    "offset": ip_off + row_start * d,
                    "shape": (d, d),
                    "numel": d * d,
                    "parent_param_id": ip_id,
                    "row_start": row_start,
                })

        # WO
        out_id = id(attn.out_proj.weight)
        out_off = offsets.get(out_id, None)
        if out_off is not None:
            blocks.append({
                "name": f"L{li}_WO",
                "layer": li,
                "param_name": f"encoder.layers.{li}.self_attn.out_proj.weight",
                "offset": out_off,
                "shape": attn.out_proj.weight.shape,
                "numel": attn.out_proj.weight.numel(),
                "parent_param_id": out_id,
                "row_start": None,
            })

        # MLP1 (linear1)
        l1_id = id(layer.linear1.weight)
        l1_off = offsets.get(l1_id, None)
        if l1_off is not None:
            blocks.append({
                "name": f"L{li}_MLP1",
                "layer": li,
                "param_name": f"encoder.layers.{li}.linear1.weight",
                "offset": l1_off,
                "shape": layer.linear1.weight.shape,
                "numel": layer.linear1.weight.numel(),
                "parent_param_id": l1_id,
                "row_start": None,
            })

        # MLP2 (linear2)
        l2_id = id(layer.linear2.weight)
        l2_off = offsets.get(l2_id, None)
        if l2_off is not None:
            blocks.append({
                "name": f"L{li}_MLP2",
                "layer": li,
                "param_name": f"encoder.layers.{li}.linear2.weight",
                "offset": l2_off,
                "shape": layer.linear2.weight.shape,
                "numel": layer.linear2.weight.numel(),
                "parent_param_id": l2_id,
                "row_start": None,
            })

    return blocks, total_params


def get_block_weight(model, block_info):
    """Extract current weight matrix for a block."""
    li = block_info["layer"]
    layer = model.encoder.layers[li]

    name = block_info["name"]
    if "WQ" in name or "WK" in name or "WV" in name:
        d = layer.self_attn.embed_dim
        ip_w = layer.self_attn.in_proj_weight.detach()
        row_start = block_info["row_start"]
        return ip_w[row_start:row_start+d, :]
    elif "WO" in name:
        return layer.self_attn.out_proj.weight.detach()
    elif "MLP1" in name:
        return layer.linear1.weight.detach()
    elif "MLP2" in name:
        return layer.linear2.weight.detach()
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Three basis constructors (per-block)
# ═══════════════════════════════════════════════════════════════════════════

def basis_weight_svd(model, block_info, k=3):
    """Basis 1: top-k SVD of current weight W."""
    W = get_block_weight(model, block_info)
    if W is None:
        return []
    return _block_basis(W, k)


def basis_delta_w_svd(model, block_info, init_weights, k=3):
    """Basis 2: top-k SVD of (W_current - W_init) = weight displacement."""
    W = get_block_weight(model, block_info)
    if W is None:
        return []
    W0 = init_weights[block_info["name"]]
    delta_W = W.cpu() - W0.cpu()
    # Early in training delta_W may be very small
    if delta_W.norm() < 1e-10:
        return _block_basis(W, k)  # fallback to weight SVD
    return _block_basis(delta_W, k)


def basis_grad_svd(model, block_info, grad_accum, k=3):
    """Basis 3: top-k SVD of accumulated gradient matrix."""
    bname = block_info["name"]
    if bname not in grad_accum or grad_accum[bname] is None:
        # Fallback
        W = get_block_weight(model, block_info)
        return _block_basis(W, k) if W is not None else []
    G = grad_accum[bname]
    if G.norm() < 1e-10:
        W = get_block_weight(model, block_info)
        return _block_basis(W, k) if W is not None else []
    return _block_basis(G, k)


# ═══════════════════════════════════════════════════════════════════════════
# Per-block projection
# ═══════════════════════════════════════════════════════════════════════════

def project_block(delta_flat, block_info, basis_vecs, total_params):
    """
    Project the block-portion of delta onto the block's basis.
    Returns {proj_norm, full_norm, K} for this block.
    """
    offset = block_info["offset"]
    numel = block_info["numel"]

    # Extract block portion of commutator
    delta_block = delta_flat[offset:offset + numel]
    full_norm = delta_block.norm().item()

    if not basis_vecs or full_norm < 1e-15:
        return {"proj_norm": 0.0, "full_norm": full_norm, "K": 0}

    # Stack basis into matrix [numel, K]
    B = torch.stack(basis_vecs, dim=1)  # each vec is [numel]
    B_cpu = B.cpu() if B.device.type != "cpu" else B
    if B_cpu.shape[0] != numel:
        return {"proj_norm": 0.0, "full_norm": full_norm, "K": 0}
    B_ortho, _ = torch.linalg.qr(B_cpu, mode="reduced")

    delta_block_cpu = delta_block.cpu().float()
    coeffs = B_ortho.T @ delta_block_cpu
    proj = B_ortho @ coeffs

    return {
        "proj_norm": proj.norm().item(),
        "full_norm": full_norm,
        "K": B_ortho.shape[1],
    }


def random_block_projection(delta_flat, block_info, K):
    """Project block-portion of delta onto K random directions."""
    offset = block_info["offset"]
    numel = block_info["numel"]

    delta_block = delta_flat[offset:offset + numel].cpu().float()
    full_norm = delta_block.norm().item()

    if K == 0 or full_norm < 1e-15:
        return 0.0

    projs = []
    for _ in range(N_RANDOM_TRIALS):
        G = torch.randn(numel, K)
        Q, _ = torch.linalg.qr(G, mode="reduced")
        p = Q @ (Q.T @ delta_block)
        projs.append(p.norm().item())
    return float(np.mean(projs))


# ═══════════════════════════════════════════════════════════════════════════
# Joint-basis projection (full parameter space, for the overview metric)
# ═══════════════════════════════════════════════════════════════════════════

def build_joint_basis(model, blocks, total_params, basis_fn, k=3, **kwargs):
    """Build joint basis [P, K] from per-block bases embedded in full space."""
    basis_vecs = []
    for b in blocks:
        local_vecs = basis_fn(model, b, k=k, **kwargs)
        for vec in local_vecs:
            gv = torch.zeros(total_params)
            gv[b["offset"]:b["offset"] + b["numel"]] = vec.cpu()
            basis_vecs.append(gv)

    if not basis_vecs:
        return None
    B = torch.stack(basis_vecs, dim=1)
    B_ortho, _ = torch.linalg.qr(B, mode="reduced")
    return B_ortho


# ═══════════════════════════════════════════════════════════════════════════
# Gradient accumulator
# ═══════════════════════════════════════════════════════════════════════════

class GradAccumulator:
    """Maintains a running sum of recent gradients reshaped per block."""

    def __init__(self, blocks, window=GRAD_WINDOW):
        self.blocks = blocks
        self.window = window
        self.buffer = []  # list of flat grad vectors

    def push(self, model):
        """Capture current .grad for each param, store as flat vector."""
        grads = {}
        for b in self.blocks:
            W = get_block_weight(model, b)
            if W is None:
                grads[b["name"]] = None
                continue
            # Get the actual parameter's grad
            li = b["layer"]
            layer = model.encoder.layers[li]
            name = b["name"]
            if "WQ" in name or "WK" in name or "WV" in name:
                p = layer.self_attn.in_proj_weight
                if p.grad is not None:
                    d = layer.self_attn.embed_dim
                    row_start = b["row_start"]
                    grads[name] = p.grad[row_start:row_start+d, :].detach().cpu().clone()
                else:
                    grads[name] = None
            elif "WO" in name:
                p = layer.self_attn.out_proj.weight
                grads[name] = p.grad.detach().cpu().clone() if p.grad is not None else None
            elif "MLP1" in name:
                p = layer.linear1.weight
                grads[name] = p.grad.detach().cpu().clone() if p.grad is not None else None
            elif "MLP2" in name:
                p = layer.linear2.weight
                grads[name] = p.grad.detach().cpu().clone() if p.grad is not None else None

        self.buffer.append(grads)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def get_accum(self):
        """Return accumulated gradient per block (sum over window)."""
        result = {}
        for b in self.blocks:
            bname = b["name"]
            accum = None
            for grads in self.buffer:
                g = grads.get(bname, None)
                if g is not None:
                    if accum is None:
                        accum = g.clone()
                    else:
                        accum += g
            result[bname] = accum
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Main training + measurement loop
# ═══════════════════════════════════════════════════════════════════════════

def train_multibasis(op_name, wd, seed, max_steps=None, **cfg_overrides):
    device = get_device()
    steps = max_steps if max_steps is not None else MAX_STEPS
    cfg = SweepConfig(OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed, STEPS=steps,
                      **cfg_overrides)
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

    blocks, total_params = get_block_registry(model)
    block_names = [b["name"] for b in blocks]
    print(f"    P={total_params}, {len(blocks)} weight blocks: {block_names}")

    # Save init weights for ΔW basis
    init_weights = {}
    for b in blocks:
        init_weights[b["name"]] = get_block_weight(model, b).cpu().clone()

    grad_accum = GradAccumulator(blocks, window=GRAD_WINDOW)

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    # Storage
    joint_records = []       # per-step: {step, phase, basis_type → {pf_exec, pf_rand, ratio}}
    block_records = []       # per-step per-block: same
    grokked = False
    grok_step = None
    memorized = False
    memorize_step = None
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    def classify_phase(train_acc, test_acc):
        if test_acc >= 0.95:
            return "post-grok"
        elif train_acc >= 0.95 and test_acc < 0.3:
            return "memorization"
        elif train_acc >= 0.95:
            return "pre-grok"
        else:
            return "early"

    def measure_step(step):
        model.eval()
        train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
        test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)
        phase = classify_phase(train_acc, test_acc)

        grad_acc = grad_accum.get_accum()

        # Compute commutator deltas
        model.train()
        deltas_info = []
        for _ in range(COMM_K):
            D_val, delta, gcos, nA, nB = commutator_defect(
                model, batch_fn, device, eta=COMM_ETA
            )
            deltas_info.append({
                "delta": delta.detach().cpu().float(),
                "nA": nA.cpu().float() if hasattr(nA, 'cpu') else torch.tensor(float(nA)),
                "nB": nB.cpu().float() if hasattr(nB, 'cpu') else torch.tensor(float(nB)),
                "defect": D_val,
            })

        defect_med = float(np.median([d["defect"] for d in deltas_info]))

        # ── Joint basis projections (3 basis types + random) ─────────
        basis_defs = {
            "weight_svd": lambda m, b, k=3: basis_weight_svd(m, b, k),
            "delta_w_svd": lambda m, b, k=3: basis_delta_w_svd(m, b, init_weights, k),
            "grad_svd": lambda m, b, k=3: basis_grad_svd(m, b, grad_acc, k),
        }

        joint_rec = {
            "step": step, "train_acc": train_acc, "test_acc": test_acc,
            "phase": phase, "defect": defect_med,
        }

        for btype, bfn in basis_defs.items():
            B = build_joint_basis(model, blocks, total_params, bfn, k=SVD_TOPK)
            K = B.shape[1] if B is not None else 0

            pf_exec_vals = []
            pf_rand_vals = []
            full_vals = []

            for info in deltas_info:
                delta = info["delta"]
                nA = info["nA"]
                nB = info["nB"]

                pc = projected_commutator(delta, B, nA, nB)
                pf_exec_vals.append(pc["proj"] / (pc["full"] + 1e-15))
                full_vals.append(pc["full"])

                # Random
                from grok_integrability_controls import random_projection_norm
                rand_norms = random_projection_norm(delta, K, n_trials=N_RANDOM_TRIALS)
                scale = (nA * nB + 1e-12)
                if hasattr(scale, 'item'):
                    scale = scale.item()
                rand_pf = [rn / scale / (pc["full"] + 1e-15) * pc["full"]
                           for rn in rand_norms]
                # Simpler: pf_rand = rand_norm / (full_norm_unnormalized)
                # Actually: rand_norm is raw, pc["full"] = delta.norm()/scale
                # So pf_rand = (rand_norm / scale) / (delta.norm() / scale) = rand_norm / delta.norm()
                delta_norm = delta.norm().item()
                pf_rand_vals.append(float(np.mean([rn / (delta_norm + 1e-15)
                                                    for rn in rand_norms])))

            pf_exec = float(np.median(pf_exec_vals))
            pf_rand = float(np.median(pf_rand_vals))

            joint_rec[f"{btype}_pf_exec"] = pf_exec
            joint_rec[f"{btype}_pf_rand"] = pf_rand
            joint_rec[f"{btype}_ratio"] = pf_exec / (pf_rand + 1e-15)
            joint_rec[f"{btype}_K"] = K

        # ── Per-block projections ────────────────────────────────────
        block_rec = {"step": step, "phase": phase, "blocks": {}}

        for b in blocks:
            bname = b["name"]
            block_data = {}

            for btype, bfn in basis_defs.items():
                local_vecs = bfn(model, b, k=SVD_TOPK)
                K_local = len(local_vecs)

                pf_exec_vals = []
                pf_rand_vals = []

                for info in deltas_info:
                    delta = info["delta"]
                    pb = project_block(delta, b, local_vecs, total_params)
                    pf = pb["proj_norm"] / (pb["full_norm"] + 1e-15)
                    pf_exec_vals.append(pf)

                    # Random for this block
                    rand_pn = random_block_projection(delta, b, K_local)
                    block_full = delta[b["offset"]:b["offset"]+b["numel"]].norm().item()
                    pf_rand_vals.append(rand_pn / (block_full + 1e-15))

                block_data[f"{btype}_pf_exec"] = float(np.median(pf_exec_vals))
                block_data[f"{btype}_pf_rand"] = float(np.median(pf_rand_vals))
                block_data[f"{btype}_ratio"] = (
                    block_data[f"{btype}_pf_exec"] /
                    (block_data[f"{btype}_pf_rand"] + 1e-15)
                )
                block_data[f"{btype}_K"] = K_local

            block_rec["blocks"][bname] = block_data

        return joint_rec, block_rec

    # ── Step 0 ───────────────────────────────────────────────────────
    jr0, br0 = measure_step(0)
    joint_records.append(jr0)
    block_records.append(br0)
    print(f"      step 0 | {jr0['phase']:>12s} | "
          f"W={jr0['weight_svd_ratio']:.2f}x "
          f"ΔW={jr0['delta_w_svd_ratio']:.2f}x "
          f"G={jr0['grad_svd_ratio']:.2f}x")

    # ── Training loop ────────────────────────────────────────────────
    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b_, y = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)
        logits = model(a, b_)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Accumulate gradient before stepping
        grad_accum.push(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % COMM_EVERY == 0:
            jr, br = measure_step(step)
            joint_records.append(jr)
            block_records.append(br)

            if not memorized and jr["train_acc"] >= 0.95 and jr["test_acc"] < 0.3:
                memorized = True
                memorize_step = step
                print(f"      MEMORIZED at step {step}")

        if step % cfg.EVAL_EVERY == 0:
            if step % COMM_EVERY == 0:
                test_acc = joint_records[-1]["test_acc"]
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

        if grokked:
            steps_after_grok += 1
            if steps_after_grok >= POST_GROK_STEPS:
                if step % COMM_EVERY != 0:
                    jr, br = measure_step(step)
                    joint_records.append(jr)
                    block_records.append(br)
                break

        if step % 500 == 0:
            elapsed = (time.time() - t0) / 60
            lr = joint_records[-1] if joint_records else {}
            print(f"      step {step:6d} | {lr.get('phase','?'):>12s} | "
                  f"W={lr.get('weight_svd_ratio',0):.2f}x "
                  f"ΔW={lr.get('delta_w_svd_ratio',0):.2f}x "
                  f"G={lr.get('grad_svd_ratio',0):.2f}x | "
                  f"def={lr.get('defect',0):.1f} | {elapsed:.1f}m")

    return {
        "joint_records": joint_records,
        "block_records": block_records,
        "grokked": grokked,
        "grok_step": grok_step,
        "memorize_step": memorize_step,
        "op": op_name,
        "wd": wd,
        "seed": seed,
        "total_params": total_params,
        "block_names": block_names,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

BASIS_COLORS = {
    "weight_svd": "#2ecc71",
    "delta_w_svd": "#3498db",
    "grad_svd": "#9b59b6",
}
BASIS_LABELS = {
    "weight_svd": "Weight SVD",
    "delta_w_svd": "ΔW SVD",
    "grad_svd": "Grad SVD",
}


def fig_M1_three_basis_ratios(all_results):
    """Hero: exec/random ratio over training for all 3 bases, all 4 ops."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx // 2, idx % 2]
        key = (op_name, 1.0, 42)
        if key not in all_results:
            ax.set_title(f"{op_name} — no data")
            continue

        data = all_results[key]
        jr = data["joint_records"]
        steps = [r["step"] for r in jr]

        for btype in ["weight_svd", "delta_w_svd", "grad_svd"]:
            ratios = [r.get(f"{btype}_ratio", 1.0) for r in jr]
            # Smooth with rolling median
            if len(ratios) >= 5:
                from scipy.ndimage import median_filter
                ratios_smooth = median_filter(ratios, size=5).tolist()
            else:
                ratios_smooth = ratios

            ax.plot(steps, ratios_smooth, linewidth=2.5,
                    color=BASIS_COLORS[btype],
                    label=BASIS_LABELS[btype], zorder=3)
            ax.plot(steps, ratios, linewidth=0.4,
                    color=BASIS_COLORS[btype], alpha=0.2, zorder=2)

        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2.5,
                   alpha=0.8, label="Random = 1.0")

        # Defect on twin axis
        ax2 = ax.twinx()
        defects = [r["defect"] for r in jr]
        ax2.plot(steps, defects, linewidth=1, color="#e67e22",
                 linestyle="--", alpha=0.4)
        ax2.set_yscale("log")
        ax2.set_ylabel("defect", fontsize=8, color="#e67e22")

        if data["grokked"] and data["grok_step"]:
            ax.axvline(x=data["grok_step"], color="blue", linestyle="--",
                      linewidth=2, alpha=0.3)

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op}", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("exec / random ratio")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Multi-Basis Exec/Random Ratio\n"
                 "All 3 basis types: Weight SVD, ΔW-SVD, Gradient SVD\n"
                 "Sign flip at grokking → basis-independent geometric property",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figM1_three_basis_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figM1_three_basis_ratios.png")


def fig_M2_phase_bars_multibasis(all_results):
    """Bar chart: exec/random ratio by phase, grouped by basis type."""
    phases = ["early", "memorization", "pre-grok", "post-grok"]
    basis_types = ["weight_svd", "delta_w_svd", "grad_svd"]

    fig, axes = plt.subplots(1, len(basis_types), figsize=(5 * len(basis_types), 5))

    for bi, btype in enumerate(basis_types):
        ax = axes[bi]
        x = np.arange(len(GROK_OPS))
        width = 0.2
        phase_colors = {"early": "#3498db", "memorization": "#e74c3c",
                        "pre-grok": "#f39c12", "post-grok": "#2ecc71"}

        for pi, phase in enumerate(phases):
            vals = []
            for op_name in GROK_OPS:
                key = (op_name, 1.0, 42)
                if key not in all_results:
                    vals.append(0)
                    continue
                jr = all_results[key]["joint_records"]
                phase_recs = [r for r in jr if r["phase"] == phase]
                if phase_recs:
                    vals.append(float(np.median(
                        [r.get(f"{btype}_ratio", 1.0) for r in phase_recs])))
                else:
                    vals.append(0)

            ax.bar(x + pi * width, vals, width,
                   label=phase, color=phase_colors[phase], alpha=0.8)

        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2.5, alpha=0.8)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([OPERATIONS[op]["label"] for op in GROK_OPS],
                           fontsize=8, rotation=15)
        ax.set_ylabel("exec / random")
        ax.set_title(BASIS_LABELS[btype], fontsize=12)
        if bi == 0:
            ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase Comparison Across Basis Types\n"
                 "Consistent sign flip → basis-independent",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figM2_phase_bars_multibasis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figM2_phase_bars_multibasis.png")


def fig_M3_perblock_heatmap(all_results):
    """
    Heatmap: exec/random ratio per block per phase, for each basis type.
    Rows = blocks, columns = phases. One heatmap per basis type.
    """
    phases = ["early", "memorization", "pre-grok", "post-grok"]
    basis_types = ["weight_svd", "delta_w_svd", "grad_svd"]

    # Average over all 4 ops
    fig, axes = plt.subplots(1, len(basis_types), figsize=(5 * len(basis_types), 8))

    for bi, btype in enumerate(basis_types):
        ax = axes[bi]

        # Collect block names from first result
        first_key = next(iter(all_results))
        bnames = all_results[first_key]["block_names"]

        matrix = np.zeros((len(bnames), len(phases)))

        for pi, phase in enumerate(phases):
            for bni, bname in enumerate(bnames):
                vals = []
                for op_name in GROK_OPS:
                    key = (op_name, 1.0, 42)
                    if key not in all_results:
                        continue
                    br = all_results[key]["block_records"]
                    phase_recs = [r for r in br if r["phase"] == phase]
                    for r in phase_recs:
                        bd = r["blocks"].get(bname, {})
                        ratio = bd.get(f"{btype}_ratio", 1.0)
                        if np.isfinite(ratio) and ratio < 100:
                            vals.append(ratio)
                matrix[bni, pi] = np.median(vals) if vals else 1.0

        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                       vmin=0.3, vmax=2.5, interpolation="nearest")
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=8, rotation=30)
        ax.set_yticks(range(len(bnames)))
        ax.set_yticklabels(bnames, fontsize=8)
        ax.set_title(BASIS_LABELS[btype], fontsize=11)

        # Annotate with values
        for i in range(len(bnames)):
            for j in range(len(phases)):
                ax.text(j, i, f"{matrix[i,j]:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if matrix[i,j] > 1.8 or matrix[i,j] < 0.6 else "black")

        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Per-Block Exec/Random Ratio by Phase\n"
                 "Blue > 1 = aligned with basis | Red < 1 = avoids basis\n"
                 "(Averaged over 4 grokking ops)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figM3_perblock_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figM3_perblock_heatmap.png")


def fig_M4_perblock_timeseries(all_results):
    """
    Per-block exec/random ratio over time, one panel per block type,
    using weight_svd basis. Shows which blocks drive the sign flip.
    """
    # Group blocks by type
    block_types = ["WQ", "WK", "WV", "WO", "MLP1", "MLP2"]
    bt_colors = {
        "WQ": "#e74c3c", "WK": "#3498db", "WV": "#2ecc71",
        "WO": "#9b59b6", "MLP1": "#e67e22", "MLP2": "#1abc9c",
    }

    # Use "add" as representative
    key = ("add", 1.0, 42)
    if key not in all_results:
        return

    data = all_results[key]
    br = data["block_records"]
    bnames = data["block_names"]
    steps = [r["step"] for r in br]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for bti, bt in enumerate(block_types):
        ax = axes[bti // 3, bti % 3]

        # Find blocks matching this type
        matching = [bn for bn in bnames if bt in bn]

        for bn in matching:
            ratios = []
            for r in br:
                bd = r["blocks"].get(bn, {})
                ratio = bd.get("weight_svd_ratio", 1.0)
                ratios.append(min(ratio, 10))  # clip outliers

            if len(ratios) >= 5:
                from scipy.ndimage import median_filter
                ratios_smooth = median_filter(ratios, size=5).tolist()
            else:
                ratios_smooth = ratios

            layer = bn.split("_")[0]
            ax.plot(steps, ratios_smooth, linewidth=2,
                    label=bn, alpha=0.8)

        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2, alpha=0.7)

        if data["grokked"] and data["grok_step"]:
            ax.axvline(x=data["grok_step"], color="blue", linestyle="--",
                      linewidth=2, alpha=0.3)

        ax.set_title(f"{bt} blocks", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("exec / random")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Per-Block Sign Flip — {OPERATIONS['add']['label']} (Weight SVD)\n"
                 "Which blocks drive the transition?",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figM4_perblock_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figM4_perblock_timeseries.png")


def fig_M5_perblock_allops(all_results):
    """
    Per-block ratio heatmap for each op separately (weight_svd only).
    4 heatmaps side by side.
    """
    phases = ["early", "memorization", "pre-grok", "post-grok"]

    fig, axes = plt.subplots(1, len(GROK_OPS), figsize=(4 * len(GROK_OPS), 8))

    for oi, op_name in enumerate(GROK_OPS):
        ax = axes[oi]
        key = (op_name, 1.0, 42)
        if key not in all_results:
            ax.set_title(f"{op_name} — no data")
            continue

        data = all_results[key]
        bnames = data["block_names"]
        br = data["block_records"]

        matrix = np.zeros((len(bnames), len(phases)))

        for pi, phase in enumerate(phases):
            phase_recs = [r for r in br if r["phase"] == phase]
            for bni, bname in enumerate(bnames):
                vals = []
                for r in phase_recs:
                    bd = r["blocks"].get(bname, {})
                    ratio = bd.get("weight_svd_ratio", 1.0)
                    if np.isfinite(ratio) and ratio < 100:
                        vals.append(ratio)
                matrix[bni, pi] = np.median(vals) if vals else 1.0

        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                       vmin=0.3, vmax=2.5, interpolation="nearest")
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=7, rotation=30)
        if oi == 0:
            ax.set_yticks(range(len(bnames)))
            ax.set_yticklabels(bnames, fontsize=8)
        else:
            ax.set_yticks([])

        for i in range(len(bnames)):
            for j in range(len(phases)):
                ax.text(j, i, f"{matrix[i,j]:.1f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if matrix[i,j] > 1.8 or matrix[i,j] < 0.6 else "black")

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(label_op, fontsize=10)

    fig.suptitle("Per-Block Exec/Random (Weight SVD) — All Ops\n"
                 "Consistent per-block sign flip → structural",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figM5_perblock_allops.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figM5_perblock_allops.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = OUT_DIR / "multibasis_controls.pt"
    if cache_path.exists():
        print(f"Loading cached results from {cache_path.name}...")
        all_results = torch.load(cache_path, weights_only=False)
    else:
        all_results = {}

    for op_name in GROK_OPS:
        key = (op_name, 1.0, 42)
        if key in all_results:
            print(f"\n  CACHED: {op_name}")
            continue

        print(f"\n{'='*70}")
        print(f"  {op_name} wd=1.0 seed=42")
        print(f"{'='*70}")

        result = train_multibasis(op_name, wd=1.0, seed=42)
        all_results[key] = result
        torch.save(all_results, cache_path)
        print(f"  saved checkpoint ({len(all_results)} runs)")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  MULTI-BASIS CONTROLS SUMMARY")
    print(f"{'='*90}")

    phases = ["early", "memorization", "pre-grok", "post-grok"]
    for key in sorted(all_results.keys()):
        data = all_results[key]
        label = OPERATIONS[data["op"]]["label"]
        jr = data["joint_records"]

        for phase in phases:
            phase_recs = [r for r in jr if r["phase"] == phase]
            if not phase_recs:
                continue
            n = len(phase_recs)
            w_r = np.median([r.get("weight_svd_ratio", 1) for r in phase_recs])
            d_r = np.median([r.get("delta_w_svd_ratio", 1) for r in phase_recs])
            g_r = np.median([r.get("grad_svd_ratio", 1) for r in phase_recs])
            print(f"  {label:>20s} | {phase:>14s} (n={n:3d}) | "
                  f"W={w_r:.2f}x  ΔW={d_r:.2f}x  G={g_r:.2f}x")

    # ── Figures ──────────────────────────────────────────────────────
    print("\n  Generating figures...")
    fig_M1_three_basis_ratios(all_results)
    fig_M2_phase_bars_multibasis(all_results)
    fig_M3_perblock_heatmap(all_results)
    fig_M4_perblock_timeseries(all_results)
    fig_M5_perblock_allops(all_results)

    torch.save(all_results, cache_path)
    print(f"\n  Final results saved to {cache_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
