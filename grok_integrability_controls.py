#!/usr/bin/env python3
"""
Integrability controls: three decisive tests.

(A) Random subspace control
    - For each commutator delta, project onto BOTH:
      (i)  the weight-SVD basis B_exec  [P, K]
      (ii) a random orthonormal basis B_rand [P, K]  (same K)
    - Compare proj_exec / full  vs  proj_rand / full.
    - If proj_exec ≈ proj_rand → artifact.
    - If proj_exec << proj_rand or >> proj_rand → real structure.

(B) Dimension sweep
    - Repeat projection for k = 1, 2, 4, 8, 16, 32, 64, 128
      (k = SVD directions per weight block → K = 2 layers × 4 blocks × k)
    - Plot proj/full vs K for both exec and random.
    - Random expectation: proj/full = √(K/P).
    - If exec tracks random → artifact.
    - If exec saturates early → structure.

(C) Phase comparison
    - Measure proj_exec/full in three phases:
      (i)   early (steps 100-500)
      (ii)  memorization (train_acc > 0.95, test_acc < 0.3)
      (iii) post-grokking (test_acc > 0.95)
    - If exec/random ratio changes across phases → mechanism.

All three tests run on 4 grokking ops × seed=42.
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
from grok_local_integrability import (
    _block_basis, extract_parameter_subspace, projected_commutator,
)

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
GROK_OPS = ["add", "sub", "mul", "x2_y2"]
SEEDS = [42]

COMM_EVERY = 100        # measurement interval
COMM_K = 7              # commutator samples per measurement
COMM_ETA = 1e-3
MAX_STEPS = 200_000
POST_GROK_STEPS = 1500

# Dimension sweep: k per weight block → K = 2 layers × 4 blocks × k
K_SWEEP = [1, 2, 3, 4, 8, 16, 32]
N_RANDOM_TRIALS = 5     # average over this many random bases


# ═══════════════════════════════════════════════════════════════════════════
# Random projection (efficient)
# ═══════════════════════════════════════════════════════════════════════════

def random_projection_norm(delta, K, n_trials=5):
    """
    Compute ||proj of delta onto random K-dim subspace|| efficiently.

    Instead of forming a full [P,K] orthonormal basis (expensive QR),
    we use the fact that for a random orthonormal basis Q [P,K]:
        ||Q Q^T delta||^2 = sum_i (q_i · delta)^2

    For efficiency with large K, we generate Q via QR of a [P,K] Gaussian
    matrix but only when K is small. For K > 128, we use the block approach.

    Returns: list of n_trials projection norms.
    """
    P = delta.numel()
    delta_flat = delta.reshape(-1).cpu().float()
    results = []

    for _ in range(n_trials):
        if K <= 256:
            # Direct QR — feasible for moderate K
            G = torch.randn(P, K)
            Q, _ = torch.linalg.qr(G, mode="reduced")
            proj = Q @ (Q.T @ delta_flat)
            results.append(proj.norm().item())
        else:
            # For very large K, use streaming: generate and project in blocks
            proj_sq_sum = 0.0
            remaining = K
            block = 64
            accum_Q = []
            while remaining > 0:
                bk = min(block, remaining)
                G = torch.randn(P, bk)
                Q_blk, _ = torch.linalg.qr(G, mode="reduced")
                # Orthogonalize against previous blocks
                for prev_Q in accum_Q:
                    Q_blk = Q_blk - prev_Q @ (prev_Q.T @ Q_blk)
                    Q_blk, _ = torch.linalg.qr(Q_blk, mode="reduced")
                accum_Q.append(Q_blk)
                proj_sq_sum += (Q_blk.T @ delta_flat).pow(2).sum().item()
                remaining -= bk
            results.append(math.sqrt(proj_sq_sum))

    return results


def random_orthonormal_basis(P, K, device="cpu"):
    """Generate a random K-dimensional orthonormal basis in R^P.
    Uses QR decomposition of Gaussian random matrix."""
    G = torch.randn(P, K, device="cpu")
    Q, _ = torch.linalg.qr(G, mode="reduced")
    return Q.to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Variable-k basis extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_parameter_subspace_k(model, k, device="cpu"):
    """Same as extract_parameter_subspace but with variable k.
    Returns B [P, K] where K = 2 layers × 4 blocks × k (or fewer if SVD rank < k)."""
    offsets, total_params = _param_offsets(model)
    basis_vecs = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        d = attn.embed_dim  # 128

        ip_w = attn.in_proj_weight.detach()
        ip_id = id(attn.in_proj_weight)
        ip_offset = offsets.get(ip_id, None)

        if ip_offset is not None:
            for wkey, row_start in [("WQ", 0), ("WK", d), ("WV", 2*d)]:
                block = ip_w[row_start:row_start+d, :]
                local_start = row_start * d
                for vec in _block_basis(block, k):
                    gv = torch.zeros(total_params, device=device)
                    start = ip_offset + local_start
                    gv[start:start + block.numel()] = vec.to(device)
                    basis_vecs.append(gv)

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

    B = torch.stack(basis_vecs, dim=1)
    B_cpu = B.cpu() if B.device.type != "cpu" else B
    B_ortho, _ = torch.linalg.qr(B_cpu, mode="reduced")
    return B_ortho.to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Training with all three controls measured simultaneously
# ═══════════════════════════════════════════════════════════════════════════

def train_with_controls(op_name, wd, seed, max_steps=None):
    """
    Train a model with comprehensive integrability controls:
    (A) exec vs random projection at each step
    (B) dimension sweep at selected checkpoints
    (C) phase-labeled measurements
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
    print(f"    Total params P = {total_params}")

    def batch_fn():
        return sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)

    records = []
    dim_sweep_records = []   # (B) dimension sweep checkpoints
    grokked = False
    grok_step = None
    patience = 0
    steps_after_grok = 0
    t0 = time.time()

    # Track phases
    memorized = False
    memorize_step = None

    def classify_phase(train_acc, test_acc):
        """Classify current training phase."""
        if test_acc >= 0.95:
            return "post-grok"
        elif train_acc >= 0.95 and test_acc < 0.3:
            return "memorization"
        elif train_acc >= 0.95:
            return "pre-grok"
        else:
            return "early"

    def measure_step(step):
        """(A) + (C): exec vs random projection, with phase label."""
        model.eval()
        train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
        test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)
        phase = classify_phase(train_acc, test_acc)

        # Build exec basis (k=3 → K=24)
        B_exec = extract_parameter_subspace(model, k=3, device="cpu")
        K_exec = B_exec.shape[1] if B_exec is not None else 0

        # Compute commutators and project onto BOTH exec and random bases
        model.train()
        proj_exec_vals = []
        proj_rand_vals = []
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

            # (A-i) Project onto exec basis
            pc_exec = projected_commutator(delta_cpu, B_exec, nA_cpu, nB_cpu)
            proj_exec_vals.append(pc_exec["proj"])
            full_vals.append(pc_exec["full"])

            # (A-ii) Project onto random bases — use efficient method
            rand_norms = random_projection_norm(delta_cpu, K_exec, n_trials=N_RANDOM_TRIALS)
            # Scale-normalize like projected_commutator does
            scale = (nA_cpu * nB_cpu + 1e-12).item() if hasattr(nA_cpu, 'item') else (nA_cpu * nB_cpu + 1e-12)
            rand_proj_scaled = [rn / scale for rn in rand_norms]
            proj_rand_vals.append(float(np.mean(rand_proj_scaled)))

        # Medians
        proj_exec_med = float(np.median(proj_exec_vals))
        proj_rand_med = float(np.median(proj_rand_vals))
        full_med = float(np.median(full_vals))
        defect_med = float(np.median(defect_vals))

        pf_exec = proj_exec_med / (full_med + 1e-15)
        pf_rand = proj_rand_med / (full_med + 1e-15)
        random_theory = math.sqrt(K_exec / total_params) if total_params > 0 else 0

        # Ratio: exec over random (empirical)
        exec_over_rand = pf_exec / (pf_rand + 1e-15)

        rec = {
            "step": step,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "phase": phase,
            "defect_median": defect_med,
            "K": K_exec,
            "full": full_med,
            "proj_exec": proj_exec_med,
            "proj_rand": proj_rand_med,
            "pf_exec": pf_exec,
            "pf_rand": pf_rand,
            "pf_theory": random_theory,
            "exec_over_rand": exec_over_rand,
        }
        return rec

    def dimension_sweep(step, phase):
        """(B): Sweep k and measure proj/full for both exec and random.

        Optimization: pre-compute commutator deltas once, then for random
        control, pre-generate a single large random basis and take subsets.
        """
        model.eval()

        # Pre-compute commutator deltas (reuse across k values)
        model.train()
        deltas_info = []
        for _ in range(min(COMM_K, 5)):  # use fewer samples for sweep
            D_val, delta, gcos, nA, nB = commutator_defect(
                model, batch_fn, device, eta=COMM_ETA
            )
            deltas_info.append({
                "delta": delta.detach().cpu(),
                "nA": nA.cpu() if hasattr(nA, 'cpu') else torch.tensor(nA),
                "nB": nB.cpu() if hasattr(nB, 'cpu') else torch.tensor(nB),
                "defect": D_val,
            })

        # Pre-generate random bases: one large basis, take first-K columns
        max_K = max(K_SWEEP) * 8  # 8 blocks
        max_K = min(max_K, 256)   # cap for sanity
        n_rand = 3  # fewer random trials for sweep
        rand_bases = []
        print(f"        Generating {n_rand} random bases [P={total_params}, K_max={max_K}]...")
        for _ in range(n_rand):
            G = torch.randn(total_params, max_K)
            Q, _ = torch.linalg.qr(G, mode="reduced")
            rand_bases.append(Q)

        sweep_results = []
        for k in K_SWEEP:
            # Exec basis with this k
            B_exec = extract_parameter_subspace_k(model, k=k, device="cpu")
            K_actual = B_exec.shape[1] if B_exec is not None else 0
            if K_actual == 0:
                continue

            pf_exec_vals = []
            pf_rand_vals = []

            for info in deltas_info:
                delta = info["delta"]
                nA = info["nA"]
                nB = info["nB"]

                # Exec projection
                pc_exec = projected_commutator(delta, B_exec, nA, nB)
                pf_exec_vals.append(pc_exec["proj"] / (pc_exec["full"] + 1e-15))

                # Random projection: use first K_actual columns of pre-generated bases
                rand_pfs = []
                K_use = min(K_actual, max_K)
                for Q_full in rand_bases:
                    B_rand = Q_full[:, :K_use]
                    pc_rand = projected_commutator(delta, B_rand, nA, nB)
                    rand_pfs.append(pc_rand["proj"] / (pc_rand["full"] + 1e-15))
                pf_rand_vals.append(float(np.mean(rand_pfs)))

            pf_exec_med = float(np.median(pf_exec_vals))
            pf_rand_med = float(np.median(pf_rand_vals))
            pf_theory = math.sqrt(K_actual / total_params) if total_params > 0 else 0

            sweep_results.append({
                "k": k,
                "K": K_actual,
                "pf_exec": pf_exec_med,
                "pf_rand": pf_rand_med,
                "pf_theory": pf_theory,
                "exec_over_rand": pf_exec_med / (pf_rand_med + 1e-15),
                "exec_over_theory": pf_exec_med / (pf_theory + 1e-15),
            })

        return {"step": step, "phase": phase, "sweeps": sweep_results}

    # ── Measure at step 0 ────────────────────────────────────────────────
    rec0 = measure_step(0)
    records.append(rec0)
    print(f"      step 0 | {rec0['phase']:>12s} | exec/rand={rec0['exec_over_rand']:.2f}x "
          f"| pf_exec={rec0['pf_exec']:.4f} pf_rand={rec0['pf_rand']:.4f} "
          f"| defect={rec0['defect_median']:.1f}")

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

        # Regular measurement
        if step % COMM_EVERY == 0:
            rec = measure_step(step)
            records.append(rec)

            # Track memorization
            if not memorized and rec["train_acc"] >= 0.95 and rec["test_acc"] < 0.3:
                memorized = True
                memorize_step = step
                print(f"      MEMORIZED at step {step}")

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

        # Dimension sweep at key moments
        # Do sweep at: step 200 (early), memorization, grok, post-grok+500
        do_sweep = False
        sweep_phase = None
        if step == 200:
            do_sweep = True
            sweep_phase = "early"
        elif memorized and memorize_step == step:
            do_sweep = True
            sweep_phase = "memorization"
        elif grokked and grok_step == step:
            do_sweep = True
            sweep_phase = "grok-onset"
        elif grokked and steps_after_grok == 500:
            do_sweep = True
            sweep_phase = "post-grok"

        if do_sweep:
            print(f"      Running dimension sweep at step {step} (phase: {sweep_phase})...")
            dsw = dimension_sweep(step, sweep_phase)
            dim_sweep_records.append(dsw)
            for s in dsw["sweeps"]:
                print(f"        k={s['k']:3d} K={s['K']:4d} | "
                      f"pf_exec={s['pf_exec']:.4f} pf_rand={s['pf_rand']:.4f} | "
                      f"exec/rand={s['exec_over_rand']:.2f}x exec/theory={s['exec_over_theory']:.2f}x")

        # Post-grok tail
        if grokked:
            steps_after_grok += 1
            if steps_after_grok >= POST_GROK_STEPS:
                # Final sweep
                if not any(d["phase"] == "post-grok" for d in dim_sweep_records):
                    print(f"      Running final dimension sweep (post-grok)...")
                    dsw = dimension_sweep(step, "post-grok")
                    dim_sweep_records.append(dsw)
                    for s in dsw["sweeps"]:
                        print(f"        k={s['k']:3d} K={s['K']:4d} | "
                              f"pf_exec={s['pf_exec']:.4f} pf_rand={s['pf_rand']:.4f} | "
                              f"exec/rand={s['exec_over_rand']:.2f}x")
                if step % COMM_EVERY != 0:
                    rec = measure_step(step)
                    records.append(rec)
                break

        # Progress
        if step % 500 == 0:
            elapsed = (time.time() - t0) / 60
            last_r = records[-1] if records else {}
            print(f"      step {step:6d} | {last_r.get('phase','?'):>12s} | "
                  f"exec/rand={last_r.get('exec_over_rand',0):.2f}x | "
                  f"pf_exec={last_r.get('pf_exec',0):.4f} | "
                  f"defect={last_r.get('defect_median',0):.1f} | {elapsed:.1f}m")

    return {
        "records": records,
        "dim_sweeps": dim_sweep_records,
        "grokked": grokked,
        "grok_step": grok_step,
        "memorize_step": memorize_step,
        "op": op_name,
        "wd": wd,
        "seed": seed,
        "total_params": total_params,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def fig_C1_exec_vs_random(all_results):
    """
    (A) The decisive test: proj/full for exec basis vs random basis over training.
    If the two lines overlap → artifact. If exec is different → structure.
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
        pf_exec = [r["pf_exec"] for r in records]
        pf_rand = [r["pf_rand"] for r in records]
        pf_theory = [r["pf_theory"] for r in records]

        ax.plot(steps, pf_exec, linewidth=2.5, color="#2ecc71",
                label="Execution basis (weight SVD)", zorder=3)
        ax.plot(steps, pf_rand, linewidth=2.5, color="#e74c3c",
                label=f"Random basis (avg {N_RANDOM_TRIALS} trials)", zorder=2)
        ax.axhline(y=pf_theory[0], color="gray", linestyle=":",
                   linewidth=1.5, alpha=0.7,
                   label=f"Theory √(K/P)={pf_theory[0]:.4f}")

        # Grok step
        if data["grokked"] and data["grok_step"]:
            ax.axvline(x=data["grok_step"], color="blue", linestyle="--",
                      linewidth=2, alpha=0.4, label=f"Grok @ {data['grok_step']}")

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} (seed=42)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("proj / full")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("(A) Execution vs Random Subspace Control\n"
                 "If lines overlap → dimensionality artifact. "
                 "If separated → real structure.",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC1_exec_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figC1_exec_vs_random.png")


def fig_C2_exec_over_rand_ratio(all_results):
    """
    (A continued) Ratio exec/random over training, with defect overlay.
    ratio = 1 → indistinguishable from random → artifact.
    ratio != 1 → structure.
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
        ratios = [r["exec_over_rand"] for r in records]
        defects = [r["defect_median"] for r in records]

        # Smooth ratio with rolling median (window 5)
        if len(ratios) >= 5:
            from scipy.ndimage import median_filter
            ratios_smooth = median_filter(ratios, size=5).tolist()
        else:
            ratios_smooth = ratios

        ax.plot(steps, ratios_smooth, linewidth=2.5, color="#2c3e50",
                label="exec/random (smoothed)", zorder=3)
        ax.plot(steps, ratios, linewidth=0.5, color="#2c3e50",
                alpha=0.3, zorder=2)
        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2.5,
                   alpha=0.8, label="Random = 1.0")

        # Defect on secondary axis
        ax2 = ax.twinx()
        ax2.plot(steps, defects, linewidth=1.5, color="#e67e22",
                 linestyle="--", alpha=0.5, label="defect")
        ax2.set_ylabel("Commutator defect", color="#e67e22", fontsize=9)
        ax2.set_yscale("log")

        if data["grokked"] and data["grok_step"]:
            ax.axvline(x=data["grok_step"], color="blue", linestyle="--",
                      linewidth=2, alpha=0.3)

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} (seed=42)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("exec proj / random proj", fontsize=10)
        ax.grid(alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("(A) Exec/Random Ratio Over Training\n"
                 "Ratio ≈ 1 → artifact | Ratio ≠ 1 → structure",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC2_exec_over_random_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figC2_exec_over_random_ratio.png")


def fig_C3_dimension_sweep(all_results):
    """
    (B) Dimension sweep: proj/full vs K for exec and random bases.
    Plot at each phase checkpoint.
    """
    n_ops = len(GROK_OPS)
    fig, axes = plt.subplots(n_ops, 1, figsize=(12, 4 * n_ops))
    if n_ops == 1:
        axes = [axes]

    for idx, op_name in enumerate(GROK_OPS):
        ax = axes[idx]
        key = (op_name, 1.0, 42)
        if key not in all_results:
            ax.set_title(f"{op_name} — no data")
            continue

        data = all_results[key]
        dim_sweeps = data["dim_sweeps"]

        if not dim_sweeps:
            ax.set_title(f"{op_name} — no dimension sweeps")
            continue

        phase_colors = {
            "early": "#3498db",
            "memorization": "#e74c3c",
            "grok-onset": "#f39c12",
            "post-grok": "#2ecc71",
        }
        phase_markers = {
            "early": "o",
            "memorization": "s",
            "grok-onset": "D",
            "post-grok": "^",
        }

        for dsw in dim_sweeps:
            phase = dsw["phase"]
            step = dsw["step"]
            sweeps = dsw["sweeps"]
            Ks = [s["K"] for s in sweeps]
            pf_execs = [s["pf_exec"] for s in sweeps]
            pf_rands = [s["pf_rand"] for s in sweeps]

            color = phase_colors.get(phase, "gray")
            marker = phase_markers.get(phase, "x")

            ax.plot(Ks, pf_execs, linewidth=2, color=color, marker=marker,
                    markersize=7, label=f"exec ({phase}, step {step})", zorder=3)
            ax.plot(Ks, pf_rands, linewidth=1.5, color=color, marker=marker,
                    markersize=5, linestyle="--", alpha=0.5,
                    label=f"random ({phase})", zorder=2)

        # Theory line
        P = data["total_params"]
        K_range = np.arange(1, max(K_SWEEP) * 8 + 1)
        theory = np.sqrt(K_range / P)
        ax.plot(K_range, theory, linewidth=2, color="gray", linestyle=":",
                alpha=0.6, label="Theory √(K/P)")

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} — Dimension Sweep", fontsize=12)
        ax.set_xlabel("Basis dimension K")
        ax.set_ylabel("proj / full")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("(B) Dimension Sweep: proj/full vs K\n"
                 "Solid = exec basis | Dashed = random basis | Dotted = theory √(K/P)\n"
                 "If exec ∥ random → artifact | If exec saturates → structure",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC3_dimension_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figC3_dimension_sweep.png")


def fig_C4_phase_comparison(all_results):
    """
    (C) Phase comparison: bar chart of exec/random ratio by phase.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: exec/random ratio by phase for each op
    ax = axes[0]
    phases_order = ["early", "memorization", "pre-grok", "post-grok"]
    phase_colors = {"early": "#3498db", "memorization": "#e74c3c",
                    "pre-grok": "#f39c12", "post-grok": "#2ecc71"}
    x = np.arange(len(GROK_OPS))
    width = 0.2

    for pi, phase in enumerate(phases_order):
        ratios = []
        for op_name in GROK_OPS:
            key = (op_name, 1.0, 42)
            if key not in all_results:
                ratios.append(0)
                continue
            records = all_results[key]["records"]
            phase_recs = [r for r in records if r["phase"] == phase]
            if phase_recs:
                ratios.append(float(np.median([r["exec_over_rand"] for r in phase_recs])))
            else:
                ratios.append(0)

        ax.bar(x + pi * width, ratios, width,
               label=phase, color=phase_colors[phase], alpha=0.8)

    ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2.5,
               alpha=0.8, label="Random = 1.0")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([OPERATIONS[op]["label"] for op in GROK_OPS], fontsize=9)
    ax.set_ylabel("exec / random ratio")
    ax.set_title("Exec/Random Ratio by Phase", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: raw proj/full exec vs random by phase (all ops combined)
    ax = axes[1]
    all_exec = {p: [] for p in phases_order}
    all_rand = {p: [] for p in phases_order}

    for op_name in GROK_OPS:
        key = (op_name, 1.0, 42)
        if key not in all_results:
            continue
        records = all_results[key]["records"]
        for r in records:
            if r["phase"] in phases_order:
                all_exec[r["phase"]].append(r["pf_exec"])
                all_rand[r["phase"]].append(r["pf_rand"])

    x = np.arange(len(phases_order))
    exec_means = [np.mean(all_exec[p]) if all_exec[p] else 0 for p in phases_order]
    exec_stds = [np.std(all_exec[p]) if all_exec[p] else 0 for p in phases_order]
    rand_means = [np.mean(all_rand[p]) if all_rand[p] else 0 for p in phases_order]
    rand_stds = [np.std(all_rand[p]) if all_rand[p] else 0 for p in phases_order]

    ax.bar(x - 0.15, exec_means, 0.3, yerr=exec_stds,
           label="Execution basis", color="#2ecc71", alpha=0.8, capsize=3)
    ax.bar(x + 0.15, rand_means, 0.3, yerr=rand_stds,
           label="Random basis", color="#e74c3c", alpha=0.8, capsize=3)

    # Theory line
    any_key = (GROK_OPS[0], 1.0, 42)
    if any_key in all_results:
        pf_th = all_results[any_key]["records"][0]["pf_theory"]
        ax.axhline(y=pf_th, color="gray", linestyle=":", linewidth=2, alpha=0.5,
                   label=f"Theory √(K/P)={pf_th:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(phases_order, fontsize=10)
    ax.set_ylabel("proj / full")
    ax.set_title("Exec vs Random proj/full by Phase (all ops)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("(C) Phase Comparison\n"
                 "If exec/random changes across phases → mechanism",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC4_phase_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figC4_phase_comparison.png")


def fig_C5_hero(all_results):
    """
    Combined hero figure: test_acc, defect, exec/random ratio on one plot per op.
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

        # Left axis: defect (log)
        defects = [r["defect_median"] for r in records]
        l1, = ax.plot(steps, defects, linewidth=2, color="#e74c3c", label="Defect")
        ax.set_yscale("log")
        ax.set_ylabel("Commutator defect", color="#e74c3c", fontsize=10)

        # Right axis: exec/random ratio + test_acc
        ax2 = ax.twinx()
        ratios = [r["exec_over_rand"] for r in records]
        test_accs = [r["test_acc"] for r in records]

        # Smooth ratio
        if len(ratios) >= 5:
            from scipy.ndimage import median_filter
            ratios_smooth = median_filter(ratios, size=5).tolist()
        else:
            ratios_smooth = ratios

        l2, = ax2.plot(steps, ratios_smooth, linewidth=2.5, color="#2ecc71",
                       label="exec/random ratio")
        l3, = ax2.plot(steps, test_accs, linewidth=1.5, color="#3498db",
                       linestyle="--", alpha=0.6, label="test acc")
        ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=2, alpha=0.5)
        ax2.set_ylabel("ratio & test acc", fontsize=10)

        if data["grokked"] and data["grok_step"]:
            ax.axvline(x=data["grok_step"], color="blue", linestyle="--",
                      linewidth=2, alpha=0.3)

        label_op = OPERATIONS[op_name]["label"]
        ax.set_title(f"{label_op} (seed=42)", fontsize=12)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)

        ax.legend(handles=[l1, l2, l3], fontsize=8, loc="center left")

    fig.suptitle("Hero: Defect × Exec/Random Ratio × Test Accuracy\n"
                 "Green above 1.0 = exec differs from random (structure)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC5_hero.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figC5_hero.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    cache_path = OUT_DIR / "integrability_controls.pt"
    if cache_path.exists():
        print(f"Loading cached results from {cache_path.name}...")
        all_results = torch.load(cache_path, weights_only=False)
    else:
        all_results = {}

    # Run 4 grokking ops
    for op_name in GROK_OPS:
        key = (op_name, 1.0, 42)
        if key in all_results:
            print(f"\n  CACHED: {op_name} wd=1.0 seed=42")
            continue

        print(f"\n{'='*70}")
        print(f"  {op_name} wd=1.0 seed=42")
        print(f"{'='*70}")

        result = train_with_controls(op_name, wd=1.0, seed=42)
        all_results[key] = result

        torch.save(all_results, cache_path)
        print(f"  saved checkpoint ({len(all_results)} total runs)")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  INTEGRABILITY CONTROLS SUMMARY")
    print(f"{'='*90}")

    for key in sorted(all_results.keys()):
        data = all_results[key]
        op = data["op"]
        records = data["records"]
        gs = data.get("grok_step", None)
        label = OPERATIONS[op]["label"]

        # Phase breakdown
        for phase in ["early", "memorization", "pre-grok", "post-grok"]:
            phase_recs = [r for r in records if r["phase"] == phase]
            if not phase_recs:
                continue
            pf_exec = np.median([r["pf_exec"] for r in phase_recs])
            pf_rand = np.median([r["pf_rand"] for r in phase_recs])
            ratio = np.median([r["exec_over_rand"] for r in phase_recs])
            n = len(phase_recs)
            print(f"  {label:>20s} | {phase:>14s} (n={n:3d}) | "
                  f"pf_exec={pf_exec:.4f}  pf_rand={pf_rand:.4f} | "
                  f"exec/rand={ratio:.2f}x")

        # Dimension sweeps
        for dsw in data.get("dim_sweeps", []):
            print(f"  {label:>20s} | dim sweep @ step {dsw['step']} ({dsw['phase']})")
            for s in dsw["sweeps"]:
                print(f"    {'':>20s}   k={s['k']:3d} K={s['K']:4d} | "
                      f"exec={s['pf_exec']:.4f} rand={s['pf_rand']:.4f} | "
                      f"ratio={s['exec_over_rand']:.2f}x")

    # ── Figures ──────────────────────────────────────────────────────────
    print("\n  Generating figures...")
    fig_C1_exec_vs_random(all_results)
    fig_C2_exec_over_rand_ratio(all_results)
    fig_C3_dimension_sweep(all_results)
    fig_C4_phase_comparison(all_results)
    fig_C5_hero(all_results)

    torch.save(all_results, cache_path)
    print(f"\n  Final results saved to {cache_path.name}")
    print(f"  Total runs: {len(all_results)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
