#!/usr/bin/env python3
"""
Multi-operation grokking sweep for PCA eigenanalysis.

Trains on all 6 binary operations from Power et al. (2022):
  1. (a + b) mod p
  2. (a - b) mod p
  3. (a * b) mod p   [restricted to a,b != 0 for invertibility]
  4. (a² + b²) mod p
  5. (a² + ab + b²) mod p
  6. (a³ + ab) mod p

For each operation:
  - 2 weight-decay settings: wd=0.1 (grokking) and wd=0.0 (memorise-only)
  - 3 random seeds
  - Up to 200k training steps (early-stop at 98% test acc)
  - Attention weights logged every 2k steps

All results saved to grok_sweep_results/ as individual .pt files.
"""

import math, time, random, csv, json, sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# Operations
# ═══════════════════════════════════════════════════════════════════════════

def op_add(a, b, p):
    return (a + b) % p

def op_sub(a, b, p):
    return (a - b) % p

def op_mul(a, b, p):
    return (a * b) % p

def op_x2_y2(a, b, p):
    return (a * a + b * b) % p

def op_x2_xy_y2(a, b, p):
    return (a * a + a * b + b * b) % p

def op_x3_xy(a, b, p):
    return (a * a * a + a * b) % p

OPERATIONS = {
    "add":       {"fn": op_add,      "label": "(a+b) mod p",       "restrict_nonzero": False},
    "sub":       {"fn": op_sub,      "label": "(a−b) mod p",       "restrict_nonzero": False},
    "mul":       {"fn": op_mul,      "label": "(a×b) mod p",       "restrict_nonzero": True},
    "x2_y2":     {"fn": op_x2_y2,    "label": "(a²+b²) mod p",     "restrict_nonzero": False},
    "x2_xy_y2":  {"fn": op_x2_xy_y2, "label": "(a²+ab+b²) mod p",  "restrict_nonzero": False},
    "x3_xy":     {"fn": op_x3_xy,    "label": "(a³+ab) mod p",     "restrict_nonzero": False},
}


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SweepConfig:
    # --- match Power et al. (2022) hyperparameters ---
    P: int = 97
    TRAIN_FRACTION: float = 0.5
    D_MODEL: int = 128
    N_LAYERS: int = 2          # Power et al. used 2 layers
    N_HEADS: int = 4
    D_FF: int = 256
    DROPOUT: float = 0.0
    LR: float = 1e-3           # Power et al.: 1e-3 (was 5e-5)
    BATCH_SIZE: int = 512
    STEPS: int = 200_000       # 200k is plenty with stronger LR+WD
    EVAL_EVERY: int = 100
    MODEL_LOG_EVERY: int = 100     # log every eval for maximum PCA granularity
    GRAD_CLIP: float = 1.0
    ACC_BS: int = 2048
    STOP_ACC: float = 0.98
    STOP_PATIENCE: int = 3
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.98   # Power et al.: β₂=0.98

    # sweep params (set per run)
    OP_NAME: str = "add"
    WEIGHT_DECAY: float = 1.0  # Power et al.: 1.0 (was 0.1)
    SEED: int = 42

OUT_DIR = Path(__file__).parent / "grok_sweep_results"


# ═══════════════════════════════════════════════════════════════════════════
# Device
# ═══════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def build_dataset(p, frac, seed, op_fn, restrict_nonzero=False):
    if restrict_nonzero:
        pairs = [(a, b) for a in range(1, p) for b in range(1, p)]
    else:
        pairs = [(a, b) for a in range(p) for b in range(p)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = int(frac * len(pairs))
    return pairs[:n], pairs[n:]


def sample_batch(pairs, bs, p, op_fn, device):
    idx = np.random.randint(0, len(pairs), size=bs)
    ab = np.array([pairs[i] for i in idx], dtype=np.int64)
    a = torch.tensor(ab[:, 0], device=device)
    b = torch.tensor(ab[:, 1], device=device)
    y = op_fn(a, b, p)
    return a, b, y


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class ModOpTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.P, cfg.D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(2, cfg.D_MODEL) / math.sqrt(cfg.D_MODEL))
        enc = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL, nhead=cfg.N_HEADS,
            dim_feedforward=cfg.D_FF, dropout=cfg.DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg.N_LAYERS)
        self.ln = nn.LayerNorm(cfg.D_MODEL)
        self.head = nn.Linear(cfg.D_MODEL, cfg.P)

    def forward(self, a, b):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        return self.head(self.ln(h[:, 0, :]))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_accuracy(model, pairs, cfg, op_fn, device):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(pairs), cfg.ACC_BS):
        chunk = pairs[i:i+cfg.ACC_BS]
        ab = torch.tensor(chunk, device=device)
        a, b = ab[:, 0], ab[:, 1]
        y = op_fn(a, b, cfg.P)
        pred = model(a, b).argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def extract_attn_matrices(model):
    logs = []
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
        logs.append({
            "layer": i,
            "WQ": Wq.detach().cpu().clone(),
            "WK": Wk.detach().cpu().clone(),
            "WV": Wv.detach().cpu().clone(),
            "WO": attn.out_proj.weight.detach().cpu().clone(),
        })
    return logs


# ═══════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════

def run_single(cfg: SweepConfig):
    device = get_device()
    op_info = OPERATIONS[cfg.OP_NAME]
    op_fn = op_info["fn"]

    # seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # data
    train_pairs, test_pairs = build_dataset(
        cfg.P, cfg.TRAIN_FRACTION, cfg.SEED, op_fn, op_info["restrict_nonzero"]
    )

    # model
    model = ModOpTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
                            betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2))
    loss_fn = nn.CrossEntropyLoss()

    attn_logs = [{"step": 0, "layers": extract_attn_matrices(model)}]
    metrics = []
    patience = 0
    t0 = time.time()
    grokked = False

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

        if step % cfg.EVAL_EVERY == 0 or step == 1:
            train_acc = eval_accuracy(model, train_pairs, cfg, op_fn, device)
            test_acc = eval_accuracy(model, test_pairs, cfg, op_fn, device)
            elapsed = (time.time() - t0) / 60
            metrics.append({"step": step, "train_acc": train_acc, "test_acc": test_acc})

            if step % (cfg.EVAL_EVERY * 10) == 0 or step == 1:
                wd_tag = f"wd={cfg.WEIGHT_DECAY}"
                print(f"  [{cfg.OP_NAME} s{cfg.SEED} {wd_tag}] "
                      f"step {step:6d} | train {train_acc:.3f} | test {test_acc:.3f} | "
                      f"{elapsed:.1f}m | snaps {len(attn_logs)}")

            if test_acc >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE:
                    grokked = True
                    print(f"  → GROKKED at step {step} (test_acc={test_acc:.3f})")
                    break
            else:
                patience = 0

    result = {
        "attn_logs": attn_logs,
        "cfg": asdict(cfg),
        "metrics": metrics,
        "grokked": grokked,
        "final_step": step,
        "final_train_acc": metrics[-1]["train_acc"] if metrics else 0,
        "final_test_acc": metrics[-1]["test_acc"] if metrics else 0,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    seeds = [42, 137, 2024]
    weight_decays = [1.0, 0.0]   # Power et al. used wd=1.0 for grokking
    op_names = list(OPERATIONS.keys())

    total = len(op_names) * len(weight_decays) * len(seeds)
    print(f"Sweep: {len(op_names)} ops × {len(weight_decays)} wd × {len(seeds)} seeds = {total} runs")
    print(f"Output: {OUT_DIR}/")
    print()

    summary = []
    run_idx = 0

    for op_name in op_names:
        for wd in weight_decays:
            for seed in seeds:
                run_idx += 1
                tag = f"{op_name}_wd{wd}_s{seed}"
                out_path = OUT_DIR / f"{tag}.pt"

                if out_path.exists():
                    print(f"[{run_idx}/{total}] {tag} — already exists, skipping")
                    data = torch.load(out_path, map_location="cpu", weights_only=False)
                    summary.append({
                        "op": op_name, "wd": wd, "seed": seed,
                        "grokked": data["grokked"],
                        "final_step": data["final_step"],
                        "final_test_acc": data["final_test_acc"],
                        "n_snapshots": len(data["attn_logs"]),
                    })
                    continue

                print(f"\n[{run_idx}/{total}] {tag}")
                # No-wd runs: shorter + coarser logging (just need PCA baseline)
                if wd == 0.0:
                    cfg = SweepConfig(OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed,
                                      STEPS=50_000, MODEL_LOG_EVERY=500)
                else:
                    cfg = SweepConfig(OP_NAME=op_name, WEIGHT_DECAY=wd, SEED=seed)
                result = run_single(cfg)

                torch.save(result, out_path)
                print(f"  saved → {out_path.name} "
                      f"({len(result['attn_logs'])} snaps, "
                      f"grokked={result['grokked']})")

                summary.append({
                    "op": op_name, "wd": wd, "seed": seed,
                    "grokked": result["grokked"],
                    "final_step": result["final_step"],
                    "final_test_acc": result["final_test_acc"],
                    "n_snapshots": len(result["attn_logs"]),
                })

    # save summary
    summary_path = OUT_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'op':>12s}  {'wd':>4s}  {'seed':>5s}  {'grok':>5s}  {'step':>6s}  {'test_acc':>8s}  {'snaps':>5s}")
    print(f"{'─'*12}  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*5}")
    for s in summary:
        print(f"{s['op']:>12s}  {s['wd']:4.1f}  {s['seed']:5d}  "
              f"{'YES' if s['grokked'] else 'no':>5s}  {s['final_step']:6d}  "
              f"{s['final_test_acc']:8.3f}  {s['n_snapshots']:5d}")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
