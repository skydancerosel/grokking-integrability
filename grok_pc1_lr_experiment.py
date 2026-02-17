#!/usr/bin/env python3
"""
PC1 de-concentration experiment across learning rates.

Trains models at lr={1e-4, 1e-3, 1e-2} with weight snapshot logging,
then computes expanding-window PC1 variance to test whether:
  1. PC1 de-concentration precedes grokking
  2. The lead time scales with grokking timescale

Reuses existing weight snapshots from grok_sweep_results/ for lr=1e-3.
Trains new models for lr=1e-4 and lr=1e-2.
"""

import math, time, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from grok_sweep import (
    SweepConfig, ModOpTransformer, build_dataset, sample_batch,
    OPERATIONS, get_device, eval_accuracy,
)

# ── config ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "pca_sweep_plots"
OUT_DIR.mkdir(exist_ok=True)

# Focus on add + one non-grokking control, seed 42
OPS_TO_RUN = ["add", "sub", "x2_xy_y2"]
SEED = 42
WD = 1.0
LRS = [1e-4, 1e-2]  # 1e-3 already has weight data in grok_sweep_results/

# LR-dependent config
LR_CONFIG = {
    1e-4: {"max_steps": 200_000, "log_every": 500,  "eval_every": 500},
    1e-3: {"max_steps":  10_000, "log_every": 100,  "eval_every": 100},
    1e-2: {"max_steps":   5_000, "log_every":  50,  "eval_every":  50},
}


def train_with_weight_logging(op_name, lr, wd, seed, max_steps, log_every, eval_every):
    """Train model, logging attention weights + accuracy at regular intervals."""
    device = get_device()
    cfg = SweepConfig(
        OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed,
        STEPS=max_steps, LR=lr,
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
        model.parameters(), lr=lr, weight_decay=wd,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    )
    loss_fn = nn.CrossEntropyLoss()

    attn_logs = []
    records = []
    grokked = False
    grok_step = None
    post_grok_steps = 0
    POST_GROK = 2000

    def log_weights(step):
        model.eval()
        layers_data = []
        for i, layer in enumerate(model.encoder.layers):
            attn = layer.self_attn
            d = attn.embed_dim
            if attn._qkv_same_embed_dim:
                Wq = attn.in_proj_weight[:d]
                Wk = attn.in_proj_weight[d:2*d]
                Wv = attn.in_proj_weight[2*d:]
            else:
                Wq = attn.q_proj_weight
                Wk = attn.k_proj_weight
                Wv = attn.v_proj_weight
            layers_data.append({
                "layer": i,
                "WQ": Wq.detach().cpu().clone(),
                "WK": Wk.detach().cpu().clone(),
                "WV": Wv.detach().cpu().clone(),
                "WO": attn.out_proj.weight.detach().cpu().clone(),
            })
        attn_logs.append({"step": step, "layers": layers_data})

    def log_accuracy(step):
        model.eval()
        train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
        test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)
        records.append({"step": step, "train_acc": train_acc, "test_acc": test_acc})
        return train_acc, test_acc

    # Step 0
    log_weights(0)
    train_acc, test_acc = log_accuracy(0)
    print(f"    step 0: train_acc={train_acc:.3f} test_acc={test_acc:.3f}", flush=True)

    for step in range(1, max_steps + 1):
        model.train()
        a, b, y = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, op_fn, device)
        logits = model(a, b)
        loss = loss_fn(logits, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    DIVERGED at step {step}", flush=True)
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % log_every == 0:
            log_weights(step)

        if step % eval_every == 0:
            train_acc, test_acc = log_accuracy(step)

            if test_acc >= 0.95 and not grokked:
                grokked = True
                grok_step = step
                print(f"    GROKKED at step {step}!", flush=True)

            if grokked:
                post_grok_steps += eval_every
                if post_grok_steps >= POST_GROK:
                    break

            if step % (max_steps // 10) == 0 or step <= eval_every * 5:
                print(f"    step {step}: train={train_acc:.3f} test={test_acc:.3f}", flush=True)

    return {
        "attn_logs": attn_logs,
        "records": records,
        "grokked": grokked,
        "grok_step": grok_step,
        "cfg": {"op": op_name, "lr": lr, "wd": wd, "seed": seed},
    }


def compute_pc1_from_logs(attn_logs, subsample=1):
    """Compute expanding-window PC1 variance ratio from weight logs."""
    logs = attn_logs[::subsample] if subsample > 1 else attn_logs
    T = len(logs)

    # Build W0
    W0 = []
    for layer in logs[0]['layers']:
        for key in ['WQ', 'WK', 'WV', 'WO']:
            w = layer[key]
            if isinstance(w, torch.Tensor):
                w = w.numpy()
            W0.append(w.flatten())
    w0 = np.concatenate(W0)
    D = len(w0)

    all_deltas = np.zeros((T, D))
    steps = []
    for t in range(T):
        wt = []
        for layer in logs[t]['layers']:
            for key in ['WQ', 'WK', 'WV', 'WO']:
                w = layer[key]
                if isinstance(w, torch.Tensor):
                    w = w.numpy()
                wt.append(w.flatten())
        all_deltas[t] = np.concatenate(wt) - w0
        steps.append(logs[t]['step'])

    results = []
    for t in range(3, T):
        traj = all_deltas[:t+1]
        traj_c = traj - traj.mean(axis=0)
        gram = traj_c @ traj_c.T
        eigvals = np.linalg.eigvalsh(gram)[::-1]
        total = eigvals.sum()
        pc1 = eigvals[0] / total if total > 1e-12 else 1.0
        results.append({'step': steps[t], 'pc1': pc1})

    return results


if __name__ == "__main__":
    import gc

    all_results = {}

    # ── Train new models at lr=1e-4 and lr=1e-2 ──
    for lr in LRS:
        lrc = LR_CONFIG[lr]
        for op in OPS_TO_RUN:
            print(f"\n{'='*60}")
            print(f"Training {op} @ lr={lr:.0e}, seed={SEED}")
            print(f"{'='*60}", flush=True)

            result = train_with_weight_logging(
                op, lr, WD, SEED,
                max_steps=lrc["max_steps"],
                log_every=lrc["log_every"],
                eval_every=lrc["eval_every"],
            )
            all_results[(op, lr, SEED)] = result

            print(f"  Done: {len(result['attn_logs'])} weight snapshots, "
                  f"grokked={result['grokked']}, grok_step={result['grok_step']}")
            gc.collect()

    # ── Load existing lr=1e-3 data ──
    print(f"\n{'='*60}")
    print("Loading existing lr=1e-3 weight data from grok_sweep_results/")
    print(f"{'='*60}", flush=True)

    for op in OPS_TO_RUN:
        fname = f"grok_sweep_results/{op}_wd1.0_s{SEED}.pt"
        try:
            data = torch.load(fname, weights_only=False)
            all_results[(op, 1e-3, SEED)] = {
                "attn_logs": data["attn_logs"],
                "records": data.get("metrics", []),
                "grokked": data.get("grokked", False),
                "grok_step": None,  # will extract from records
                "cfg": {"op": op, "lr": 1e-3, "wd": WD, "seed": SEED},
            }
            print(f"  Loaded {op}: {len(data['attn_logs'])} snapshots", flush=True)
            del data; gc.collect()
        except Exception as e:
            print(f"  SKIP {op}: {e}")

    # ── Compute PC1 trajectories ──
    print(f"\n{'='*60}")
    print("Computing PC1 trajectories")
    print(f"{'='*60}", flush=True)

    pc1_results = {}
    for key, result in sorted(all_results.items()):
        op, lr, seed = key
        logs = result["attn_logs"]
        n = len(logs)
        # Subsample if too many snapshots
        sub = max(1, n // 60)
        print(f"  {op} lr={lr:.0e}: {n} snapshots (subsample={sub})...", end=" ", flush=True)
        pc1 = compute_pc1_from_logs(logs, subsample=sub)
        pc1_results[key] = pc1
        grok_step = result.get("grok_step")
        print(f"done ({len(pc1)} PC1 points), grok_step={grok_step}")

    # ── Save results ──
    save_data = {}
    for key in pc1_results:
        op, lr, seed = key
        save_data[key] = {
            "pc1": pc1_results[key],
            "grokked": all_results[key]["grokked"],
            "grok_step": all_results[key]["grok_step"],
        }

    torch.save(save_data, OUT_DIR / "pc1_lr_experiment.pt")
    print(f"\nSaved to {OUT_DIR / 'pc1_lr_experiment.pt'}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("PC1 TRAJECTORY SUMMARY")
    print(f"{'='*60}")

    for key in sorted(pc1_results.keys()):
        op, lr, seed = key
        pc1 = pc1_results[key]
        grok_step = all_results[key].get("grok_step")
        grokked = all_results[key]["grokked"]
        tag = "GROK" if grokked else "NO-GK"

        # Find PC1 min (bottom of initial transient)
        if len(pc1) >= 5:
            min_pc1 = min(r['pc1'] for r in pc1)
            min_step = [r['step'] for r in pc1 if r['pc1'] == min_pc1][0]

            # Find PC1 max after min
            after_min = [r for r in pc1 if r['step'] > min_step]
            if after_min:
                max_after = max(r['pc1'] for r in after_min)
                max_step = [r['step'] for r in after_min if r['pc1'] == max_after][0]
            else:
                max_after = min_pc1
                max_step = min_step

            # Find PC1 turnover: first sustained decrease after the max
            turnover_step = None
            for i in range(2, len(pc1)):
                if pc1[i]['step'] > max_step:
                    if pc1[i]['pc1'] < pc1[i-1]['pc1'] < pc1[i-2]['pc1']:
                        turnover_step = pc1[i-2]['step']
                        break

            lead = (grok_step - turnover_step) if (grok_step and turnover_step) else None

            print(f"\n  {tag:6s} {op:12s} lr={lr:.0e}  grok={grok_step}")
            print(f"    PC1 min={min_pc1:.1%}@{min_step}  max={max_after:.1%}@{max_step}")
            print(f"    turnover={turnover_step}  lead={lead}")
        else:
            print(f"\n  {tag:6s} {op:12s} lr={lr:.0e}  TOO FEW POINTS")

    # ── Print full trajectories for inspection ──
    print(f"\n{'='*60}")
    print("FULL PC1 TRAJECTORIES")
    print(f"{'='*60}")
    for key in sorted(pc1_results.keys()):
        op, lr, seed = key
        pc1 = pc1_results[key]
        grokked = all_results[key]["grokked"]
        tag = "GROK" if grokked else "NO-GK"
        print(f"\n--- {tag} {op} lr={lr:.0e} ---")
        for r in pc1:
            print(f"  step={r['step']:8d}  PC1={r['pc1']:.1%}")
