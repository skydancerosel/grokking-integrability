#!/usr/bin/env python3
"""Single-task per-op gradient SED.

The original SED basis came from rolling-window SVD of AdamW updates Δθ_t.
This contaminates the basis with momentum, weight-decay, and adaptive-scaling
noise. The cleaner version uses the *gradient* itself:

    g(t) = ∇_{θ_attn} L|_{θ_t}    (single fixed batch, single op)

with rolling-window SVD over {g(t-W+1), …, g(t)} → top-K right singular
vectors v_k(t).

Prediction (per referee comment): single-task R_k for sub/mul/x²+y² should
rise from the ~3.5× we measured under update-SED to closer to the multitask
per-op values (~12-45×).

Usage:
    python spectral/sed_lch_gradient.py --cache <path> --op <op> --tag <tag>
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    def embed(self, a, b):
        x = torch.stack([a, b], dim=1)
        return self.tok_emb(x) + self.pos_emb.unsqueeze(0)

    def forward_from_emb(self, emb):
        h = self.encoder(emb)
        return self.head(self.ln(h[:, 0, :]))

    def forward(self, a, b):
        return self.forward_from_emb(self.embed(a, b))


OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    "x2_y2": lambda a, b, p: (a * a + b * b) % p,
}
NONZERO_OPS = {"mul"}


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def is_attn_key(name):
    return ("self_attn" in name) and ("weight" in name) and ("bias" not in name)


def get_full_spec(sd):
    return [(k, tuple(v.shape), v.numel()) for k, v in sd.items()]


def flatten_full(sd):
    return torch.cat([v.detach().float().reshape(-1) for v in sd.values()]).cpu().numpy()


def flatten_attn(sd):
    parts = [v.detach().float().reshape(-1) for k, v in sd.items() if is_attn_key(k)]
    return torch.cat(parts).cpu().numpy()


def unflatten_full(flat, full_spec):
    sd = {}
    i = 0
    for k, shape, n in full_spec:
        sd[k] = torch.from_numpy(flat[i:i + n]).float().reshape(shape)
        i += n
    return sd


def add_attn_delta_to_full(direction_attn, base_flat_full, eps, full_spec):
    out = base_flat_full.copy()
    i = 0
    j = 0
    for k, _, n in full_spec:
        if is_attn_key(k):
            out[i:i + n] = base_flat_full[i:i + n] + eps * direction_attn[j:j + n]
            j += n
        i += n
    return out


def grad_attn_at(model, batch):
    """Return loss gradient on attention parameters, flattened to numpy."""
    a, b, y = batch
    model.zero_grad(set_to_none=True)
    logits = model(a, b)
    loss = F.cross_entropy(logits, y)
    attn_pms = [p for n, p in model.named_parameters() if is_attn_key(n)]
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


def perturb_attn(model, base_flat_full, full_spec, direction_attn, eps, probes, device):
    flat_p = add_attn_delta_to_full(direction_attn, base_flat_full, +eps, full_spec)
    flat_n = add_attn_delta_to_full(direction_attn, base_flat_full, -eps, full_spec)
    sd_p = unflatten_full(flat_p, full_spec)
    model.load_state_dict({k: v.to(device) for k, v in sd_p.items()})
    mu_p = compute_centroids(model, probes, device)
    sd_n = unflatten_full(flat_n, full_spec)
    model.load_state_dict({k: v.to(device) for k, v in sd_n.items()})
    mu_n = compute_centroids(model, probes, device)
    diff = (mu_p - mu_n) / (2.0 * eps)
    return float(np.mean(np.sum(diff ** 2, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--op", choices=list(OPS.keys()), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "sed_lch_results"))
    ap.add_argument("--n-probes", type=int, default=1024)
    ap.add_argument("--n-grad-batch", type=int, default=512)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--eps-rel", type=float, default=0.005)
    ap.add_argument("--n-checkpoints", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or get_device()
    print(f"[info] device={device} op={args.op} tag={args.tag}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    cfg_d = cache["cfg"]
    cfg = Config(P=cfg_d["P"], D_MODEL=cfg_d["D_MODEL"],
                 N_LAYERS=cfg_d["N_LAYERS"], N_HEADS=cfg_d["N_HEADS"],
                 D_FF=cfg_d["D_FF"], DROPOUT=cfg_d["DROPOUT"])
    ckpts = cache["checkpoints"]
    print(f"[info] {len(ckpts)} ckpts, steps {ckpts[0][0]}->{ckpts[-1][0]}")

    full_spec = get_full_spec(ckpts[0][1])
    Pdim_attn = sum(n for k, _, n in full_spec if is_attn_key(k))
    steps_all = np.array([s for s, _ in ckpts], dtype=np.int64)

    # fixed gradient batch
    op_fn = OPS[args.op]
    nz = args.op in NONZERO_OPS
    lo = 1 if nz else 0
    rng_b = np.random.RandomState(args.seed)
    a_g = rng_b.randint(lo, cfg.P, size=args.n_grad_batch).astype(np.int64)
    b_g = rng_b.randint(lo, cfg.P, size=args.n_grad_batch).astype(np.int64)
    y_g = op_fn(a_g, b_g, cfg.P)
    grad_batch = (
        torch.from_numpy(a_g).to(device),
        torch.from_numpy(b_g).to(device),
        torch.from_numpy(y_g).to(device),
    )

    # probes (separate sample from grad batch)
    rng_p = np.random.RandomState(args.seed + 100)
    a_p = rng_p.randint(lo, cfg.P, size=args.n_probes).astype(np.int64)
    b_p = rng_p.randint(lo, cfg.P, size=args.n_probes).astype(np.int64)
    y_p = op_fn(a_p, b_p, cfg.P)
    probes = (torch.from_numpy(a_p), torch.from_numpy(b_p), torch.from_numpy(y_p))

    model = ModAddTransformer(cfg).to(device)

    # phase 1: per-checkpoint gradient
    print(f"[info] computing gradients at {len(ckpts)} ckpts...")
    grads = np.zeros((len(ckpts), Pdim_attn), dtype=np.float32)
    t0 = time.time()
    for t_idx, (step, sd) in enumerate(ckpts):
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        grads[t_idx] = grad_attn_at(model, grad_batch)
        if t_idx % 25 == 0:
            print(f"  grad ckpt {t_idx+1}/{len(ckpts)} ({(time.time()-t0)/60:.1f}m)")
    print(f"[info] gradient phase: {(time.time()-t0)/60:.1f}m")

    # phase 2: per-window SED + perturbation
    flats_full = np.stack([flatten_full(sd) for _, sd in ckpts], axis=0)
    n_ck = min(args.n_checkpoints, len(ckpts))
    ck_idx = np.linspace(0, len(ckpts) - 1, n_ck, dtype=int)

    W = args.window
    A_k = np.zeros((n_ck, args.top_k))
    A_rand = np.zeros((n_ck, args.n_random))
    rank90 = np.zeros(n_ck, dtype=np.int64)
    sigma = np.zeros((n_ck, min(W, 10)))
    rng_dir = np.random.RandomState(args.seed + 200)

    t1 = time.time()
    for j, idx in enumerate(ck_idx):
        step = int(steps_all[idx])
        flat_full = flats_full[idx]
        flat_attn_t = flatten_attn(ckpts[idx][1])
        eps = float(args.eps_rel * np.linalg.norm(flat_attn_t))

        # SED window of GRADIENTS (not updates)
        t0w = max(0, min(idx, len(ckpts) - W))
        X = grads[t0w:t0w + W]
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        sigma[j, :min(len(S), sigma.shape[1])] = S[:sigma.shape[1]]
        Vk = Vt[:args.top_k].astype(np.float32)

        # baseline
        sd_unflat = unflatten_full(flat_full, full_spec)
        model.load_state_dict({k: v.to(device) for k, v in sd_unflat.items()})
        mu = compute_centroids(model, probes, device)
        muc = mu - mu.mean(axis=0, keepdims=True)
        s_mu = np.linalg.svd(muc, compute_uv=False)
        ev_n = (s_mu ** 2) / max((s_mu ** 2).sum(), 1e-30)
        rank90[j] = int(np.searchsorted(np.cumsum(ev_n), 0.9) + 1)

        for k in range(args.top_k):
            A_k[j, k] = perturb_attn(model, flat_full, full_spec, Vk[k],
                                     eps, probes, device)
        for r in range(args.n_random):
            v = rng_dir.randn(Pdim_attn).astype(np.float32)
            v /= np.linalg.norm(v)
            A_rand[j, r] = perturb_attn(model, flat_full, full_spec, v,
                                        eps, probes, device)

        med = float(np.median(A_rand[j]))
        Rk = A_k[j] / max(med, 1e-30)
        Rk_str = " ".join(f"R{k+1}={Rk[k]:6.2f}" for k in range(args.top_k))
        print(f"  [{j+1:3d}/{n_ck}] step={step:5d}  {Rk_str}  rank90={rank90[j]:3d}  "
              f"({(time.time()-t1)/60:.1f}m)")
        sys.stdout.flush()

    A_rand_med = np.median(A_rand, axis=1)
    R_k = A_k / np.maximum(A_rand_med[:, None], 1e-30)

    metrics = cache["metrics"]
    m_step = np.array([m["step"] for m in metrics])
    train_acc = np.array([m["train_acc"] for m in metrics])
    test_acc = np.array([m["test_acc"] for m in metrics])

    result = dict(
        cfg=cfg_d, args=vars(args),
        op=args.op,
        ckpt_steps=steps_all[ck_idx],
        sigma=sigma,
        A_k=A_k, A_rand=A_rand, R_k=R_k, rank90=rank90,
        m_step=m_step, train_acc=train_acc, test_acc=test_acc,
    )
    out_pt = out_dir / f"sed_lch_grad_{args.tag}.pt"
    torch.save(result, out_pt)
    print(f"[saved] {out_pt}")

    # plot
    sel = steps_all[ck_idx]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(m_step, train_acc, label="train", c="tab:blue")
    ax[0].plot(m_step, test_acc, label="test", c="tab:orange")
    ax[0].set_ylabel("accuracy"); ax[0].legend(loc="lower right")
    for k in range(args.top_k):
        ax[1].plot(sel, R_k[:, k], "-o", ms=3, label=fr"$R_{k+1}$")
    ax[1].axhline(1.0, c="gray", ls=":", lw=0.5)
    ax[1].set_ylabel(r"$R_k$ (gradient-SED)")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[2].plot(sel, rank90, "-o", ms=3, c="tab:green")
    ax[2].set_ylabel(r"rank-90 of $\mu$")
    ax[2].set_xlabel("training step")
    fig.suptitle(f"Per-op-gradient SED, single-task {args.op} (seed={args.seed})")
    fig.tight_layout()
    out_png = out_dir / f"sed_lch_grad_{args.tag}.png"
    fig.savefig(out_png, dpi=130)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
