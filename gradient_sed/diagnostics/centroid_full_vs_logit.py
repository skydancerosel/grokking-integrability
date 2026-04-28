#!/usr/bin/env python3
"""Compare single-logit centroid vs full J^T·1 centroid at three checkpoints.

Single-logit:    μ_x = ∇_emb ℓ_{y(x)}(x)            (used in the paper)
Full J·1:        μ_x = ∇_emb (Σ_c ℓ_c(x))           (LCH paper convention)

For three checkpoints (init, mid-training, post-grok), compute both, and
report:
  - cosine similarity per probe (then averaged), top-K alignment of PCs,
  - rank-90 of each centroid matrix,
  - R_k (gradient-SED) under each centroid definition.

Reads training_cache.pt for add seed 42.
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    P: int = 97
    D_MODEL: int = 128
    N_LAYERS: int = 2
    N_HEADS: int = 4
    D_FF: int = 256
    DROPOUT: float = 0.0


class ModAddTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.P, cfg.D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(2, cfg.D_MODEL) / math.sqrt(cfg.D_MODEL))
        enc = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL, nhead=cfg.N_HEADS, dim_feedforward=cfg.D_FF,
            dropout=cfg.DROPOUT, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg.N_LAYERS)
        self.ln = nn.LayerNorm(cfg.D_MODEL)
        self.head = nn.Linear(cfg.D_MODEL, cfg.P)

    def embed(self, a, b):
        x = torch.stack([a, b], dim=1)
        return self.tok_emb(x) + self.pos_emb.unsqueeze(0)

    def forward_from_emb(self, emb):
        h = self.encoder(emb)
        return self.head(self.ln(h[:, 0, :]))


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def centroid_single_logit(model, probes, device, batch=256):
    a, b, y = probes
    model.eval()
    out = []
    for i in range(0, a.size(0), batch):
        ai, bi, yi = (t[i:i + batch].to(device) for t in (a, b, y))
        emb = model.embed(ai, bi).detach().requires_grad_(True)
        logits = model.forward_from_emb(emb)
        scalar = logits.gather(1, yi.unsqueeze(1)).squeeze(1).sum()
        grad = torch.autograd.grad(scalar, emb, retain_graph=False)[0]
        out.append(grad.detach().cpu().reshape(grad.size(0), -1).numpy())
    return np.concatenate(out, axis=0)


def centroid_full_j1(model, probes, device, batch=256):
    """μ_x = ∇_emb (Σ_c ℓ_c(x)) — sum over all output classes."""
    a, b, _ = probes
    model.eval()
    out = []
    for i in range(0, a.size(0), batch):
        ai, bi = (t[i:i + batch].to(device) for t in (a, b))
        emb = model.embed(ai, bi).detach().requires_grad_(True)
        logits = model.forward_from_emb(emb)
        scalar = logits.sum()  # sum over batch AND classes
        grad = torch.autograd.grad(scalar, emb, retain_graph=False)[0]
        out.append(grad.detach().cpu().reshape(grad.size(0), -1).numpy())
    return np.concatenate(out, axis=0)


def cos_sim_rows(A, B):
    """Per-row cosine similarity between A and B (both (N, D))."""
    nA = np.linalg.norm(A, axis=1, keepdims=True)
    nB = np.linalg.norm(B, axis=1, keepdims=True)
    return (A * B).sum(axis=1) / np.maximum(nA[:, 0] * nB[:, 0], 1e-30)


def rank_90(mu):
    muc = mu - mu.mean(axis=0, keepdims=True)
    s = np.linalg.svd(muc, compute_uv=False)
    ev = (s ** 2)
    cum = np.cumsum(ev / max(ev.sum(), 1e-30))
    return int(np.searchsorted(cum, 0.9) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="/Users/tara-mini/bubble/spectral/coherence_edge_results/training_cache.pt")
    ap.add_argument("--n-probes", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    cfg_d = cache["cfg"]
    cfg = Config(P=cfg_d["P"], D_MODEL=cfg_d["D_MODEL"],
                 N_LAYERS=cfg_d["N_LAYERS"], N_HEADS=cfg_d["N_HEADS"],
                 D_FF=cfg_d["D_FF"], DROPOUT=cfg_d["DROPOUT"])
    ckpts = cache["checkpoints"]
    n = len(ckpts)
    # pick init, mid-training (50%), post-grok (last)
    idxs = [0, n // 2, n - 1]
    print(f"[info] device={device}, ckpts={n}, comparing at idx={idxs}, "
          f"steps={[ckpts[i][0] for i in idxs]}")

    rng = np.random.RandomState(args.seed)
    pa = rng.randint(0, cfg.P, size=args.n_probes).astype(np.int64)
    pb = rng.randint(0, cfg.P, size=args.n_probes).astype(np.int64)
    py = (pa + pb) % cfg.P
    probes = (torch.from_numpy(pa), torch.from_numpy(pb), torch.from_numpy(py))

    model = ModAddTransformer(cfg).to(device)

    print(f"\n{'phase':>12} {'step':>6} | {'rank90 SL':>11} {'rank90 J1':>11} "
          f"{'mean cos':>9} {'top1 cos':>9} {'top3 cos':>9}")
    print("-" * 75)
    for phase, idx in zip(["init", "mid", "post-grok"], idxs):
        sd = ckpts[idx][1]
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        mu_sl = centroid_single_logit(model, probes, device)
        mu_j1 = centroid_full_j1(model, probes, device)

        # row-wise cosine similarity
        cs = cos_sim_rows(mu_sl, mu_j1)
        # top PC alignment: SVD of each, top-1 cosine of right singular vec
        for mu, name in [(mu_sl, "SL"), (mu_j1, "J1")]:
            pass
        muc_sl = mu_sl - mu_sl.mean(axis=0, keepdims=True)
        muc_j1 = mu_j1 - mu_j1.mean(axis=0, keepdims=True)
        _, _, Vt_sl = np.linalg.svd(muc_sl, full_matrices=False)
        _, _, Vt_j1 = np.linalg.svd(muc_j1, full_matrices=False)
        # top-1 PC cosine
        top1_cos = abs(float(Vt_sl[0] @ Vt_j1[0]))
        # top-3 subspace alignment: average of |V_sl V_j1^T| singular values
        M = np.abs(Vt_sl[:3] @ Vt_j1[:3].T)
        top3_cos = float(np.linalg.svd(M, compute_uv=False).mean())

        r_sl = rank_90(mu_sl)
        r_j1 = rank_90(mu_j1)
        print(f"{phase:>12} {ckpts[idx][0]:>6} | {r_sl:>11d} {r_j1:>11d} "
              f"{cs.mean():>9.3f} {top1_cos:>9.3f} {top3_cos:>9.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
