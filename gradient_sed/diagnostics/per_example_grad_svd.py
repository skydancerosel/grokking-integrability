#!/usr/bin/env python3
"""Per-example gradient SVD as an alternative to rolling-window gradient SED.

At a single checkpoint, compute the per-example gradient
    g_x(theta) = grad_{theta_attn} ell_x(theta)
for each x in a fixed batch of N=512 examples, stack into an (N, P_attn)
matrix, take the top-K right singular vectors. These v^pe_k are the
directions of largest example-to-example loss-gradient variance — an
instantaneous (W=1 in time) rank-K basis derived from sample diversity.

We compare R_k under v^pe_k against the existing rolling-window gradient
SED at W=1 (= avg gradient direction) and W=20 (denoised) at the same
checkpoint, and report the cosine similarity between the two bases.
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


def flatten_attn(sd):
    parts = [v.detach().float().reshape(-1) for k, v in sd.items() if is_attn(k)]
    return torch.cat(parts).cpu().numpy()


def unflatten_full(flat, spec):
    sd, i = {}, 0
    for k, shape, n in spec:
        sd[k] = torch.from_numpy(flat[i:i + n]).float().reshape(shape)
        i += n
    return sd


def add_attn_delta(direction, base_full, eps, spec):
    out = base_full.copy()
    i, j = 0, 0
    for k, _, n in spec:
        if is_attn(k):
            out[i:i + n] = base_full[i:i + n] + eps * direction[j:j + n]
            j += n
        i += n
    return out


def per_example_grads(model, batch, attn_param_names, P_attn, device):
    """Return (N, P_attn) matrix of per-example loss gradients."""
    a, b, y = batch
    N = a.shape[0]
    G = np.zeros((N, P_attn), dtype=np.float32)
    attn_pms = [p for n, p in model.named_parameters() if n in attn_param_names]
    for i in range(N):
        model.zero_grad(set_to_none=True)
        logits = model(a[i:i + 1], b[i:i + 1])
        loss = F.cross_entropy(logits, y[i:i + 1])
        grads = torch.autograd.grad(loss, attn_pms, retain_graph=False)
        G[i] = torch.cat([g.detach().reshape(-1) for g in grads]).cpu().numpy()
    return G


def rolling_grads_at(model, ckpts, ck_idx, attn_param_names, P_attn, batch, device, W):
    """Compute the W gradient vectors used by gradient-SED at checkpoint ck_idx,
    matching the existing pipeline."""
    G = np.zeros((W, P_attn), dtype=np.float32)
    a, b, y = batch
    attn_pms_list = lambda m: [p for n, p in m.named_parameters() if n in attn_param_names]
    for w in range(W):
        sd = ckpts[max(0, ck_idx - W + 1 + w)][1]
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        attn_pms = attn_pms_list(model)
        model.zero_grad(set_to_none=True)
        logits = model(a, b)
        loss = F.cross_entropy(logits, y)
        grads = torch.autograd.grad(loss, attn_pms, retain_graph=False)
        G[w] = torch.cat([g.detach().reshape(-1) for g in grads]).cpu().numpy()
    return G


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


def perturb(model, base_full, spec, direction, eps, probes, device):
    flat_p = add_attn_delta(direction, base_full, +eps, spec)
    flat_n = add_attn_delta(direction, base_full, -eps, spec)
    sd_p = unflatten_full(flat_p, spec)
    model.load_state_dict({k: v.to(device) for k, v in sd_p.items()})
    mu_p = compute_centroids(model, probes, device)
    sd_n = unflatten_full(flat_n, spec)
    model.load_state_dict({k: v.to(device) for k, v in sd_n.items()})
    mu_n = compute_centroids(model, probes, device)
    diff = (mu_p - mu_n) / (2.0 * eps)
    return float(np.mean(np.sum(diff ** 2, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="/Users/tara-mini/bubble/spectral/coherence_edge_results/training_cache.pt")
    ap.add_argument("--n-probes", type=int, default=1024)
    ap.add_argument("--n-grad-batch", type=int, default=512)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--eps-rel", type=float, default=0.005)
    ap.add_argument("--out", default="spectral/sed_lch_results/per_example_svd.pt")
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
    idxs = [0, n // 2, n - 1]
    print(f"[info] device={device}, ckpts={n}, comparing at "
          f"steps={[ckpts[i][0] for i in idxs]}")

    full_spec = get_full_spec(ckpts[0][1])
    P_attn = sum(n for k, _, n in full_spec if is_attn(k))

    # fixed gradient batch
    rng = np.random.RandomState(0)
    a_g = rng.randint(0, cfg.P, size=args.n_grad_batch).astype(np.int64)
    b_g = rng.randint(0, cfg.P, size=args.n_grad_batch).astype(np.int64)
    y_g = (a_g + b_g) % cfg.P
    grad_batch = (torch.from_numpy(a_g).to(device),
                  torch.from_numpy(b_g).to(device),
                  torch.from_numpy(y_g).to(device))

    # probes
    rng_p = np.random.RandomState(100)
    a_p = rng_p.randint(0, cfg.P, size=args.n_probes).astype(np.int64)
    b_p = rng_p.randint(0, cfg.P, size=args.n_probes).astype(np.int64)
    y_p = (a_p + b_p) % cfg.P
    probes = (torch.from_numpy(a_p), torch.from_numpy(b_p), torch.from_numpy(y_p))

    model = ModAddTransformer(cfg).to(device)
    attn_names = {n for n, _ in model.named_parameters() if is_attn(n)}

    rng_dir = np.random.RandomState(200)

    print(f"\n{'phase':>10} {'step':>5} | "
          f"{'R_k peak (per-ex)':>17} | "
          f"{'R_k peak (W=1)':>15} | "
          f"{'R_k peak (W=20)':>16} | "
          f"{'cos(v_pe_1, v_W1)':>18}")
    print("-" * 110)
    results = []
    for phase, idx in zip(["init", "mid", "post-grok"], idxs):
        sd = ckpts[idx][1]
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        flat_full = flatten_full(sd)
        flat_attn = flatten_attn(sd)
        eps = float(args.eps_rel * np.linalg.norm(flat_attn))

        # per-example gradients (option 1)
        t0 = time.time()
        G_pe = per_example_grads(model, grad_batch, attn_names, P_attn, device)
        # SVD: (N, P_attn) -> top-K right sing vec
        # use thin SVD: U S Vt
        Gc = G_pe - G_pe.mean(axis=0, keepdims=True)
        _, S_pe, Vt_pe = np.linalg.svd(Gc, full_matrices=False)
        V_pe = Vt_pe[:args.top_k].astype(np.float32)
        t_pe = time.time() - t0

        # W=1 grad direction (the average)
        t0 = time.time()
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        attn_pms = [p for n, p in model.named_parameters() if n in attn_names]
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(*grad_batch[:2]), grad_batch[2])
        grads_w1 = torch.autograd.grad(loss, attn_pms)
        g_w1 = torch.cat([g.detach().reshape(-1) for g in grads_w1]).cpu().numpy()
        v_w1 = (g_w1 / max(np.linalg.norm(g_w1), 1e-30)).astype(np.float32)
        t_w1 = time.time() - t0

        # W=20 rolling SVD: need 20 consecutive ckpts ending at idx
        t0 = time.time()
        G_w20 = rolling_grads_at(model, ckpts, idx, attn_names, P_attn,
                                 grad_batch, device, W=20)
        # restore current ckpt
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        _, S_w20, Vt_w20 = np.linalg.svd(G_w20, full_matrices=False)
        V_w20 = Vt_w20[:args.top_k].astype(np.float32)
        t_w20 = time.time() - t0

        # perturb under each basis
        # per-example
        A_pe = np.array([
            perturb(model, flat_full, full_spec, V_pe[k], eps, probes, device)
            for k in range(args.top_k)
        ])
        # W=1 (single direction)
        A_w1 = perturb(model, flat_full, full_spec, v_w1, eps, probes, device)
        # W=20
        A_w20 = np.array([
            perturb(model, flat_full, full_spec, V_w20[k], eps, probes, device)
            for k in range(args.top_k)
        ])
        # random baseline
        A_r = np.zeros(args.n_random)
        for r in range(args.n_random):
            v = rng_dir.randn(P_attn).astype(np.float32)
            v /= np.linalg.norm(v)
            A_r[r] = perturb(model, flat_full, full_spec, v, eps, probes, device)
        med = float(np.median(A_r))

        R_pe = A_pe / max(med, 1e-30)
        R_w1 = A_w1 / max(med, 1e-30)
        R_w20 = A_w20 / max(med, 1e-30)

        # cosine: top per-example PC vs W=1 direction
        cos_pe1_w1 = abs(float(V_pe[0] @ v_w1))

        print(f"{phase:>10} {ckpts[idx][0]:>5} | "
              f"{R_pe.mean():>11.1f} (max {R_pe.max():.0f}) | "
              f"{R_w1:>15.1f} | "
              f"{R_w20.mean():>10.1f} (max {R_w20.max():.0f}) | "
              f"{cos_pe1_w1:>18.3f}")
        sys.stdout.flush()

        results.append({
            "phase": phase, "step": ckpts[idx][0],
            "R_pe": R_pe, "R_w1": R_w1, "R_w20": R_w20,
            "S_pe_top10": S_pe[:10], "S_w20_top10": S_w20[:10],
            "cos_pe1_w1": cos_pe1_w1,
            "t_pe_sec": t_pe, "t_w1_sec": t_w1, "t_w20_sec": t_w20,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"results": results, "cfg": cfg_d}, out_path)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
