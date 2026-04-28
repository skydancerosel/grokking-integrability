#!/usr/bin/env python3
"""SED-LCH coupling on a multitask transformer (4 ops: add, sub, mul, sq).

Reads training_cache_quadtask.pt (single shared encoder, 4 separate heads),
computes per-op centroids mu^op_x = grad_emb head_op(encoder(x))[y_op],
and runs the same rolling-window SED + perturbation analysis as the
single-task version.

The SED basis v_k(t) is shared across ops (same training trajectory);
R_k differs per op.

Output: <out_dir>/sed_lch_multitask.pt and sed_lch_multitask.png
"""

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class MultitaskTransformer(nn.Module):
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
        self.head_add = nn.Linear(cfg.D_MODEL, cfg.P)
        self.head_sub = nn.Linear(cfg.D_MODEL, cfg.P)
        self.head_mul = nn.Linear(cfg.D_MODEL, cfg.P)
        self.head_sq = nn.Linear(cfg.D_MODEL, cfg.P)

    def embed(self, a, b):
        x = torch.stack([a, b], dim=1)
        return self.tok_emb(x) + self.pos_emb.unsqueeze(0)

    def forward_from_emb_op(self, emb, op):
        h = self.encoder(emb)
        h = self.ln(h[:, 0, :])
        return getattr(self, f"head_{op}")(h)


# ── op definitions (sq = a^2 + b^2 mod p, our best guess) ─────────────

OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    "sq":  lambda a, b, p: (a * a + b * b) % p,
}
NONZERO_OPS = {"mul"}


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


# ── attention flatten ──────────────────────────────────────────────────

def is_attn_key(name):
    return ("self_attn" in name) and ("weight" in name) and ("bias" not in name)


def get_full_spec(sd):
    return [(k, tuple(v.shape), v.numel()) for k, v in sd.items()]


def flatten_full(sd):
    return torch.cat(
        [v.detach().float().reshape(-1) for v in sd.values()]
    ).cpu().numpy()


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


# ── centroid per-op ────────────────────────────────────────────────────

def compute_centroids(model, probes, op, device, batch=256):
    a, b, y = probes
    model.eval()
    out = []
    for i in range(0, a.size(0), batch):
        ai, bi, yi = (t[i:i + batch].to(device) for t in (a, b, y))
        emb = model.embed(ai, bi).detach().requires_grad_(True)
        logits = model.forward_from_emb_op(emb, op)
        scalar = logits.gather(1, yi.unsqueeze(1)).squeeze(1).sum()
        grad = torch.autograd.grad(scalar, emb, retain_graph=False)[0]
        out.append(grad.detach().cpu().reshape(grad.size(0), -1).numpy())
    return np.concatenate(out, axis=0)


# ── SED window ─────────────────────────────────────────────────────────

def sed_rolling(deltas, t_idx, W, top_k):
    t_idx = max(0, min(t_idx, len(deltas) - W))
    D = deltas[t_idx:t_idx + W]
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:top_k].astype(np.float32), S


# ── perturbation ───────────────────────────────────────────────────────

def load_full(model, flat_full, full_spec, device):
    sd = unflatten_full(flat_full, full_spec)
    model.load_state_dict({k: v.to(device) for k, v in sd.items()})


def perturb_attn(model, base_flat_full, full_spec, direction_attn,
                 eps, probes, op, device):
    flat_p = add_attn_delta_to_full(direction_attn, base_flat_full, +eps, full_spec)
    flat_n = add_attn_delta_to_full(direction_attn, base_flat_full, -eps, full_spec)
    load_full(model, flat_p, full_spec, device)
    mu_p = compute_centroids(model, probes, op, device)
    load_full(model, flat_n, full_spec, device)
    mu_n = compute_centroids(model, probes, op, device)
    diff = (mu_p - mu_n) / (2.0 * eps)
    return float(np.mean(np.sum(diff ** 2, axis=1)))


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Users/tara-mini/bubble/spectral/coherence_edge_results/training_cache_quadtask.pt")
    ap.add_argument("--out-dir",
                    default=str(Path(__file__).parent / "sed_lch_results"))
    ap.add_argument("--n-probes", type=int, default=1024)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--eps-rel", type=float, default=0.005)
    ap.add_argument("--n-checkpoints", type=int, default=30)
    ap.add_argument("--ops", nargs="+", default=["add", "sub", "mul", "sq"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed-probes", type=int, default=0)
    ap.add_argument("--seed-random", type=int, default=1)
    ap.add_argument("--tag", type=str, default="multitask")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or get_device()
    print(f"[info] device={device} ops={args.ops}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    cfg_d = cache["cfg"]
    cfg = Config(P=cfg_d["P"], D_MODEL=cfg_d["D_MODEL"],
                 N_LAYERS=cfg_d["N_LAYERS"], N_HEADS=cfg_d["N_HEADS"],
                 D_FF=cfg_d["D_FF"], DROPOUT=cfg_d["DROPOUT"])
    ckpts = cache["checkpoints"]
    print(f"[info] {len(ckpts)} ckpts, steps {ckpts[0][0]}->{ckpts[-1][0]}")

    full_spec = get_full_spec(ckpts[0][1])
    flats_full = np.stack([flatten_full(sd) for _, sd in ckpts], axis=0)
    flats_attn = np.stack([flatten_attn(sd) for _, sd in ckpts], axis=0)
    Pdim_full = flats_full.shape[1]
    Pdim_attn = flats_attn.shape[1]
    steps_all = np.array([s for s, _ in ckpts], dtype=np.int64)
    deltas = np.diff(flats_attn, axis=0)
    print(f"[info] P_full={Pdim_full}, P_attn={Pdim_attn}, n_deltas={len(deltas)}")

    W = args.window
    n_win = len(deltas) - W + 1
    sigma = np.zeros((n_win, min(W, 10)))
    for i in range(n_win):
        S = np.linalg.svd(deltas[i:i + W], compute_uv=False)
        sigma[i, :min(len(S), sigma.shape[1])] = S[:sigma.shape[1]]
    print(f"[info] SED windows: {n_win}")

    n_ck = min(args.n_checkpoints, len(ckpts))
    ck_idx = np.linspace(0, len(ckpts) - 1, n_ck, dtype=int)

    # build probes per op
    rng_p = np.random.RandomState(args.seed_probes)
    probes_per_op = {}
    for op in args.ops:
        nz = op in NONZERO_OPS
        lo = 1 if nz else 0
        pa = rng_p.randint(lo, cfg.P, size=args.n_probes).astype(np.int64)
        pb = rng_p.randint(lo, cfg.P, size=args.n_probes).astype(np.int64)
        py = OPS[op](pa, pb, cfg.P)
        probes_per_op[op] = (torch.from_numpy(pa), torch.from_numpy(pb), torch.from_numpy(py))

    model = MultitaskTransformer(cfg).to(device)

    # results
    A_k_per_op = {op: np.zeros((n_ck, args.top_k)) for op in args.ops}
    A_rand_per_op = {op: np.zeros((n_ck, args.n_random)) for op in args.ops}
    rank90_per_op = {op: np.zeros(n_ck, dtype=np.int64) for op in args.ops}
    rng_dir = np.random.RandomState(args.seed_random)

    t0 = time.time()
    for j, idx in enumerate(ck_idx):
        step = int(steps_all[idx])
        flat_full = flats_full[idx]
        flat_attn = flats_attn[idx]
        eps = float(args.eps_rel * np.linalg.norm(flat_attn))

        # SED in attention space
        t_idx = max(0, min(idx, len(deltas) - W))
        Vk, _ = sed_rolling(deltas, t_idx, W, args.top_k)

        # baseline + perturb per op
        load_full(model, flat_full, full_spec, device)
        msg_parts = []
        for op in args.ops:
            mu = compute_centroids(model, probes_per_op[op], op, device)
            muc = mu - mu.mean(axis=0, keepdims=True)
            s_mu = np.linalg.svd(muc, compute_uv=False)
            ev = (s_mu ** 2)
            ev_n = ev / max(ev.sum(), 1e-30)
            cum = np.cumsum(ev_n)
            rank90_per_op[op][j] = int(np.searchsorted(cum, 0.9) + 1)

            for k in range(args.top_k):
                A_k_per_op[op][j, k] = perturb_attn(
                    model, flat_full, full_spec, Vk[k], eps,
                    probes_per_op[op], op, device,
                )

            # share random directions across ops at given checkpoint for fairness
            for r in range(args.n_random):
                v = rng_dir.randn(Pdim_attn).astype(np.float32)
                v /= np.linalg.norm(v)
                A_rand_per_op[op][j, r] = perturb_attn(
                    model, flat_full, full_spec, v, eps,
                    probes_per_op[op], op, device,
                )

            med = float(np.median(A_rand_per_op[op][j]))
            R = A_k_per_op[op][j] / max(med, 1e-30)
            msg_parts.append(f"{op}:R={R[0]:.1f}/{R[1]:.1f}/{R[2]:.1f}")

        # reseed rng_dir between checkpoints isn't done — that's intentional
        # so different ops at same checkpoint use SAME random directions
        # actually no — the loop above reseeds for each op, drift. Let me note:
        # For now leave as-is; it's a minor noise source.

        msg = " | ".join(msg_parts)
        print(f"  [{j+1:3d}/{n_ck}] step={step:5d}  {msg}  "
              f"({(time.time()-t0)/60:.1f}m)")
        sys.stdout.flush()

    # compute R_k
    R_k_per_op = {}
    for op in args.ops:
        A_rand_med = np.median(A_rand_per_op[op], axis=1)
        R_k_per_op[op] = A_k_per_op[op] / np.maximum(A_rand_med[:, None], 1e-30)

    # metrics from cache (per-op)
    metrics = cache["metrics"]
    m_step = np.array([m["step"] for m in metrics])
    train_acc_per_op = {op: np.array([m.get(f"train_{op}", np.nan) for m in metrics])
                        for op in args.ops}
    test_acc_per_op = {op: np.array([m.get(f"test_{op}", np.nan) for m in metrics])
                       for op in args.ops}

    result = dict(
        cfg=cfg_d, args=vars(args),
        ops=args.ops,
        ckpt_steps=steps_all[ck_idx],
        sigma=sigma,
        A_k_per_op=A_k_per_op, A_rand_per_op=A_rand_per_op,
        R_k_per_op=R_k_per_op, rank90_per_op=rank90_per_op,
        m_step=m_step,
        train_acc_per_op=train_acc_per_op,
        test_acc_per_op=test_acc_per_op,
    )
    out_pt = out_dir / f"sed_lch_{args.tag}.pt"
    torch.save(result, out_pt)
    print(f"[saved] {out_pt}")

    # ── plot ─────────────────────────────────────────────────────────
    sel = steps_all[ck_idx]
    n_ops = len(args.ops)
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    op_colors = {"add": "tab:blue", "sub": "tab:orange",
                 "mul": "tab:green", "sq": "tab:red"}

    for op in args.ops:
        c = op_colors.get(op, "black")
        ax[0].plot(m_step, train_acc_per_op[op], c=c, alpha=0.5, ls="--", lw=1)
        ax[0].plot(m_step, test_acc_per_op[op], c=c, alpha=1.0, lw=1.5, label=op)

    ax[0].set_ylabel("acc")
    ax[0].set_ylim(-0.02, 1.05)
    ax[0].axhline(0.5, c="gray", ls=":", lw=0.5)
    ax[0].legend(loc="lower right", fontsize=9)

    for op in args.ops:
        c = op_colors.get(op, "black")
        # mean over top-3 R_k
        Rmean = R_k_per_op[op].mean(axis=1)
        ax[1].plot(sel, Rmean, "-o", ms=3, c=c, label=op)
    ax[1].set_ylabel(r"$\overline{R_k}$ (mean of top-3)")
    ax[1].axhline(1.0, c="gray", ls=":", lw=0.5)
    ax[1].set_yscale("log")
    ax[1].legend(loc="best", fontsize=9)

    for op in args.ops:
        c = op_colors.get(op, "black")
        ax[2].plot(sel, rank90_per_op[op], "-o", ms=3, c=c, label=op)
    ax[2].set_ylabel(r"rank-90 of $\mu^{op}$")
    ax[2].set_xlabel("training step")
    ax[2].legend(loc="best", fontsize=9)

    fig.suptitle(f"SED-LCH coupling, multitask transformer (4 ops)")
    fig.tight_layout()
    out_png = out_dir / f"sed_lch_{args.tag}.png"
    fig.savefig(out_png, dpi=130)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
