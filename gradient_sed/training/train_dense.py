#!/usr/bin/env python3
"""Train mod-add transformer with dense checkpointing (one seed per call).

Matches the exact protocol of training_cache.pt:
  p=97, 2L Transformer, d_model=128, 4 heads, d_ff=256, GELU pre-norm,
  AdamW(lr=1e-3, wd=1.0, betas=(0.9, 0.98)), batch 512, grad clip 1.0,
  full state_dicts every 25 steps, eval every 25 steps,
  early stop after STOP_PATIENCE=20 consecutive evals at test_acc >= 0.98.

Output: <out_dir>/training_cache_s<seed>.pt
"""

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    P: int = 97
    TRAIN_FRACTION: float = 0.5
    D_MODEL: int = 128
    N_LAYERS: int = 2
    N_HEADS: int = 4
    D_FF: int = 256
    DROPOUT: float = 0.0
    LR: float = 1e-3
    BATCH_SIZE: int = 512
    STEPS: int = 8000
    EVAL_EVERY: int = 25
    CHECKPOINT_EVERY: int = 25
    GRAD_CLIP: float = 1.0
    ACC_BS: int = 2048
    STOP_ACC: float = 0.98
    STOP_PATIENCE: int = 20
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.98
    WEIGHT_DECAY: float = 1.0
    SEED: int = 42


class ModAddTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.P, cfg.D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(2, cfg.D_MODEL) / math.sqrt(cfg.D_MODEL))
        enc = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL, nhead=cfg.N_HEADS, dim_feedforward=cfg.D_FF,
            dropout=cfg.DROPOUT, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg.N_LAYERS)
        self.ln = nn.LayerNorm(cfg.D_MODEL)
        self.head = nn.Linear(cfg.D_MODEL, cfg.P)

    def forward(self, a, b):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        return self.head(self.ln(h[:, 0, :]))


OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    "x2_y2": lambda a, b, p: (a * a + b * b) % p,
    "x2_xy_y2": lambda a, b, p: (a * a + a * b + b * b) % p,
    "x3_xy": lambda a, b, p: (a * a * a + a * b) % p,
}
NONZERO_OPS = {"mul"}


def op_add(a, b, p):
    return (a + b) % p


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_dataset(p, frac, seed, op_name="add"):
    nonzero = op_name in NONZERO_OPS
    lo = 1 if nonzero else 0
    pairs = [(a, b) for a in range(lo, p) for b in range(lo, p)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = int(frac * len(pairs))
    return pairs[:n], pairs[n:]


def sample_batch(pairs, bs, p, device, op_fn=op_add):
    idx = np.random.randint(0, len(pairs), size=bs)
    ab = np.array([pairs[i] for i in idx], dtype=np.int64)
    a = torch.tensor(ab[:, 0], device=device)
    b = torch.tensor(ab[:, 1], device=device)
    y = op_fn(a, b, p)
    return a, b, y


@torch.no_grad()
def eval_acc(model, pairs, cfg, device, op_fn=op_add):
    model.eval()
    correct = total = 0
    for i in range(0, len(pairs), cfg.ACC_BS):
        chunk = pairs[i:i + cfg.ACC_BS]
        ab = torch.tensor(chunk, device=device)
        a, b = ab[:, 0], ab[:, 1]
        y = op_fn(a, b, cfg.P)
        pred = model(a, b).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--op", choices=list(OPS.keys()), default="add")
    ap.add_argument("--out-dir", type=str,
                    default=str(Path(__file__).parent / "coherence_edge_results"))
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = Config(SEED=args.seed, STEPS=args.steps)
    device = args.device or get_device()
    op_fn = OPS[args.op]
    print(f"[info] seed={cfg.SEED} op={args.op} device={device} steps={cfg.STEPS}")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    train_pairs, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED, args.op)
    print(f"[info] train={len(train_pairs)}, test={len(test_pairs)}")

    model = ModAddTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
    )

    checkpoints = [(0, {k: v.cpu().clone() for k, v in model.state_dict().items()})]
    metrics = []
    patience = 0
    grokked = False
    final_step = cfg.STEPS
    t0 = time.time()

    for step in range(1, cfg.STEPS + 1):
        model.train()
        a, b, y = sample_batch(train_pairs, cfg.BATCH_SIZE, cfg.P, device, op_fn)
        loss = F.cross_entropy(model(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if step % cfg.CHECKPOINT_EVERY == 0:
            checkpoints.append(
                (step, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            )

        if step % cfg.EVAL_EVERY == 0:
            tr = eval_acc(model, train_pairs, cfg, device, op_fn)
            te = eval_acc(model, test_pairs, cfg, device, op_fn)
            metrics.append({"step": step, "train_acc": tr, "test_acc": te})
            if step % 1000 == 0:
                print(f"  step {step:5d} | train {tr:.3f} | test {te:.3f} | "
                      f"{(time.time()-t0)/60:.1f}m | ckpts {len(checkpoints)}")
            if te >= cfg.STOP_ACC:
                patience += 1
                if patience >= cfg.STOP_PATIENCE:
                    grokked = True
                    final_step = step
                    print(f"  GROKKED at step {step}, test={te:.3f}")
                    break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    op_suffix = "" if args.op == "add" else f"_{args.op}"
    out_path = out_dir / f"training_cache_s{cfg.SEED}{op_suffix}.pt"
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    cfg_dict["op"] = args.op
    cfg_dict["grokked"] = grokked
    cfg_dict["final_step"] = final_step
    torch.save({
        "checkpoints": checkpoints,
        "metrics": metrics,
        "test_pairs": test_pairs,
        "cfg": cfg_dict,
    }, out_path)
    print(f"[saved] {out_path}  ({len(checkpoints)} ckpts, "
          f"{(time.time()-t0)/60:.1f}m total, grokked={grokked} at step {final_step})")


if __name__ == "__main__":
    main()
