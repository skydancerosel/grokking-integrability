#!/usr/bin/env python3
"""Ablations for the gradient-SED diagnostic: vary W, K, ε, probe-seed.

For each axis, sweep three values around the canonical setting and report
R_k peak and final at each. The point is a defensive check that the
~200× single-task and ~30× multitask R_k effects survive perturbation
of the diagnostic's hyperparameters.

Reads the same training_cache.pt as sed_lch_gradient.py.
Output: a single .pt with all sweep results + console table.
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

    def forward(self, a, b):
        return self.forward_from_emb(self.embed(a, b))


OPS = {"add": lambda a, b, p: (a + b) % p}


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def is_attn(name):
    return ("self_attn" in name) and ("weight" in name) and ("bias" not in name)


def get_full_spec(sd):
    return [(k, tuple(v.shape), v.numel()) for k, v in sd.items()]


def flatten_full(sd):
    return torch.cat([v.detach().float().reshape(-1) for v in sd.values()]).cpu().numpy()


def flatten_attn_sd(sd):
    parts = [v.detach().float().reshape(-1) for k, v in sd.items() if is_attn(k)]
    return torch.cat(parts).cpu().numpy()


def unflatten_full(flat, full_spec):
    sd, i = {}, 0
    for k, shape, n in full_spec:
        sd[k] = torch.from_numpy(flat[i:i + n]).float().reshape(shape)
        i += n
    return sd


def add_attn_delta(direction_attn, base_full, eps, full_spec):
    out = base_full.copy()
    i, j = 0, 0
    for k, _, n in full_spec:
        if is_attn(k):
            out[i:i + n] = base_full[i:i + n] + eps * direction_attn[j:j + n]
            j += n
        i += n
    return out


def grad_attn_at(model, batch):
    a, b, y = batch
    model.zero_grad(set_to_none=True)
    logits = model(a, b)
    loss = F.cross_entropy(logits, y)
    attn_pms = [p for n, p in model.named_parameters() if is_attn(n)]
    grads = torch.autograd.grad(loss, attn_pms, retain_graph=False)
    return torch.cat([g.detach().reshape(-1) for g in grads]).cpu().numpy()


def compute_centroids(model, probes, device, batch=256):
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


def perturb(model, base_full, full_spec, direction, eps, probes, device):
    flat_p = add_attn_delta(direction, base_full, +eps, full_spec)
    flat_n = add_attn_delta(direction, base_full, -eps, full_spec)
    sd_p = unflatten_full(flat_p, full_spec)
    model.load_state_dict({k: v.to(device) for k, v in sd_p.items()})
    mu_p = compute_centroids(model, probes, device)
    sd_n = unflatten_full(flat_n, full_spec)
    model.load_state_dict({k: v.to(device) for k, v in sd_n.items()})
    mu_n = compute_centroids(model, probes, device)
    diff = (mu_p - mu_n) / (2.0 * eps)
    return float(np.mean(np.sum(diff ** 2, axis=1)))


def run_one(cfg, ckpts, full_spec, grads, flats_full, P_attn,
            *, W, K, eps_rel, n_probes, probe_seed, n_random,
            n_checkpoints, op, device):
    """Compute peak and final R_k under given hyperparameters.
    R_1 peak is also returned separately (defined for all W >= 1)."""
    op_fn = OPS[op]
    K_eff = min(K, W)  # SVD of (W, P) has at most W non-zero singular values
    rng_p = np.random.RandomState(probe_seed)
    pa = rng_p.randint(0, cfg.P, size=n_probes).astype(np.int64)
    pb = rng_p.randint(0, cfg.P, size=n_probes).astype(np.int64)
    py = op_fn(pa, pb, cfg.P)
    probes = (torch.from_numpy(pa), torch.from_numpy(pb), torch.from_numpy(py))

    model = ModAddTransformer(cfg).to(device)

    n_ck = min(n_checkpoints, len(ckpts))
    ck_idx = np.linspace(0, len(ckpts) - 1, n_ck, dtype=int)
    R_k = np.zeros((n_ck, K_eff))
    rng_dir = np.random.RandomState(probe_seed + 200)

    for j, idx in enumerate(ck_idx):
        flat_full = flats_full[idx]
        flat_attn = flatten_attn_sd(ckpts[idx][1])
        eps = float(eps_rel * np.linalg.norm(flat_attn))
        t0w = max(0, min(idx, len(ckpts) - W))
        X = grads[t0w:t0w + W]
        if W == 1:
            v = X[0] / max(np.linalg.norm(X[0]), 1e-30)
            Vk = v.reshape(1, -1).astype(np.float32)
        else:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            Vk = Vt[:K_eff].astype(np.float32)
        A_k = np.zeros(K_eff)
        for k in range(K_eff):
            A_k[k] = perturb(model, flat_full, full_spec, Vk[k], eps, probes, device)
        A_rand = np.zeros(n_random)
        for r in range(n_random):
            v = rng_dir.randn(P_attn).astype(np.float32)
            v /= np.linalg.norm(v)
            A_rand[r] = perturb(model, flat_full, full_spec, v, eps, probes, device)
        med = float(np.median(A_rand))
        R_k[j] = A_k / max(med, 1e-30)

    Rmean = R_k.mean(axis=1)
    R1 = R_k[:, 0]
    return float(Rmean.max()), float(Rmean[-1]), float(R1.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--op", default="add")
    ap.add_argument("--out", default="spectral/sed_lch_results/ablation.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    print(f"[info] device={device} op={args.op}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    cfg_d = cache["cfg"]
    cfg = Config(P=cfg_d["P"], D_MODEL=cfg_d["D_MODEL"],
                 N_LAYERS=cfg_d["N_LAYERS"], N_HEADS=cfg_d["N_HEADS"],
                 D_FF=cfg_d["D_FF"], DROPOUT=cfg_d["DROPOUT"])
    ckpts = cache["checkpoints"]
    full_spec = get_full_spec(ckpts[0][1])
    P_attn = sum(n for k, _, n in full_spec if is_attn(k))

    # build fixed gradient batch and precompute gradients per checkpoint
    op_fn = OPS[args.op]
    rng = np.random.RandomState(0)
    a_g = rng.randint(0, cfg.P, size=512).astype(np.int64)
    b_g = rng.randint(0, cfg.P, size=512).astype(np.int64)
    y_g = op_fn(a_g, b_g, cfg.P)
    grad_batch = (torch.from_numpy(a_g).to(device),
                  torch.from_numpy(b_g).to(device),
                  torch.from_numpy(y_g).to(device))

    model = ModAddTransformer(cfg).to(device)
    print(f"[info] computing gradients at {len(ckpts)} ckpts...")
    grads = np.zeros((len(ckpts), P_attn), dtype=np.float32)
    t0 = time.time()
    for t_idx, (_, sd) in enumerate(ckpts):
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        grads[t_idx] = grad_attn_at(model, grad_batch)
    print(f"[info] gradient phase: {(time.time()-t0)/60:.1f}m")
    flats_full = np.stack([flatten_full(sd) for _, sd in ckpts], axis=0)

    # canonical settings
    base = dict(W=20, K=3, eps_rel=0.005, n_probes=1024,
                probe_seed=0, n_random=20, n_checkpoints=20,
                op=args.op, device=device)

    sweeps = []
    # only the missing small-W points (10/20/40 were measured previously)
    for w in [1, 2, 5]:
        sweeps.append((f"W={w}", dict(W=w)))

    print(f"\n{'sweep':>20} | {'R_k peak':>10} {'R_k final':>11} {'R_1 peak':>10}")
    print("-" * 60)
    results = []
    for name, override in sweeps:
        params = {**base, **override}
        t1 = time.time()
        peak, final, R1_peak = run_one(
            cfg, ckpts, full_spec, grads, flats_full, P_attn, **params,
        )
        results.append({"name": name, **params, "peak": peak, "final": final,
                        "R1_peak": R1_peak})
        print(f"{name:>20} | {peak:>10.2f} {final:>11.2f} {R1_peak:>10.2f}  "
              f"({(time.time()-t1)/60:.1f}m)")
        sys.stdout.flush()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"results": results, "cfg": cfg_d, "op": args.op}, out_path)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
