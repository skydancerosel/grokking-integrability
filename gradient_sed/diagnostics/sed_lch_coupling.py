#!/usr/bin/env python3
"""SED-LCH coupling on grokking modular addition (pilot, single seed).

Tests whether the spectral-edge directions (top right singular vectors of a
rolling window of parameter updates) are the parameter-space directions that
reorganize the Linear Centroid Hypothesis (LCH) centroid geometry.

Centroid (per LCH, simplified to a single output scalar as in the plan):
    mu_x(theta) = grad_emb [ logit_y(x; theta) ]
where the gradient is taken with respect to the embedded input
(tok_emb + pos_emb), so each centroid is a (T=2, d_model)-shaped vector.

Spectral-edge directions:
    rolling SVD over a window of W=20 consecutive parameter updates
    Delta theta_t = theta_t - theta_{t-1} (full-theta), top-k right singular
    vectors v_1, v_2, v_3.

Coupling score per direction v at checkpoint t:
    A(v, t)  = mean_x ||[mu_x(theta+eps v) - mu_x(theta-eps v)] / (2 eps)||_2^2
    R_k(t)   = A(v_k, t) / median_j A(r_j, t)   over 20 random Gaussian r_j

Consumes training_cache.pt produced by spectral/coherence_edge_experiment.py:
    p=97, 2L Transformer, d_model=128, 4 heads, d_ff=256, GELU pre-norm,
    WD=1.0, LR=1e-3, full state_dicts every 25 steps, seed=42.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Config / model — must match training_cache.pt
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    P: int = 97
    D_MODEL: int = 128
    N_LAYERS: int = 2
    N_HEADS: int = 4
    D_FF: int = 256
    DROPOUT: float = 0.0


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

    def embed(self, a, b):
        x = torch.stack([a, b], dim=1)
        return self.tok_emb(x) + self.pos_emb.unsqueeze(0)

    def forward_from_emb(self, emb):
        h = self.encoder(emb)
        return self.head(self.ln(h[:, 0, :]))

    def forward(self, a, b):
        return self.forward_from_emb(self.embed(a, b))


# ──────────────────────────────────────────────────────────────────────
# Param flatten / unflatten — full theta or attention-only subspace
# ──────────────────────────────────────────────────────────────────────

def is_attn_key(k: str) -> bool:
    return ("self_attn" in k) and ("weight" in k) and ("bias" not in k)


def get_full_spec(sd):
    return [(k, tuple(v.shape), v.numel()) for k, v in sd.items()]


def get_attn_spec(sd):
    return [(k, tuple(v.shape), v.numel()) for k, v in sd.items() if is_attn_key(k)]


def flatten_sd_full(sd) -> np.ndarray:
    return torch.cat(
        [v.detach().float().reshape(-1) for v in sd.values()]
    ).cpu().numpy()


def flatten_sd_attn(sd) -> np.ndarray:
    parts = [v.detach().float().reshape(-1) for k, v in sd.items() if is_attn_key(k)]
    return torch.cat(parts).cpu().numpy()


def unflatten_full(flat: np.ndarray, full_spec) -> dict:
    """Scatter a flat full-theta vector into a state_dict."""
    sd = {}
    i = 0
    for k, shape, n in full_spec:
        sd[k] = torch.from_numpy(flat[i:i + n]).float().reshape(shape)
        i += n
    return sd


def lift_attn_to_full(attn_flat: np.ndarray, base_flat_full: np.ndarray,
                      full_spec) -> np.ndarray:
    """Return new full-theta flat = base_flat_full with attention slots replaced
    by attn_flat (in the order given by full_spec)."""
    out = base_flat_full.copy()
    i = 0           # offset in full-flat
    j = 0           # offset in attn-flat
    for k, _, n in full_spec:
        if is_attn_key(k):
            out[i:i + n] = attn_flat[j:j + n]
            j += n
        i += n
    return out


def add_attn_delta_to_full(direction_attn: np.ndarray, base_flat_full: np.ndarray,
                           eps: float, full_spec) -> np.ndarray:
    """Return base_flat_full with attention slots += eps * direction_attn.
    Non-attention slots untouched."""
    out = base_flat_full.copy()
    i = 0
    j = 0
    for k, _, n in full_spec:
        if is_attn_key(k):
            out[i:i + n] = base_flat_full[i:i + n] + eps * direction_attn[j:j + n]
            j += n
        i += n
    return out


# ──────────────────────────────────────────────────────────────────────
# Centroid: mu_x = grad_emb logit_y(x)
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Perturbation
# ──────────────────────────────────────────────────────────────────────

def load_full_into(model, flat_full, full_spec, device):
    sd = unflatten_full(flat_full, full_spec)
    model.load_state_dict({k: v.to(device) for k, v in sd.items()})


def perturb_and_measure_full(model, base_flat_full, full_spec, direction_full,
                             eps, probes, device):
    """direction_full is a unit vector in full-theta space."""
    load_full_into(model, base_flat_full + eps * direction_full, full_spec, device)
    mu_p = compute_centroids(model, probes, device)
    load_full_into(model, base_flat_full - eps * direction_full, full_spec, device)
    mu_n = compute_centroids(model, probes, device)
    diff = (mu_p - mu_n) / (2.0 * eps)
    return float(np.mean(np.sum(diff ** 2, axis=1)))


def perturb_and_measure_attn(model, base_flat_full, full_spec, direction_attn,
                             eps, probes, device):
    """direction_attn is a unit vector in attention subspace; non-attn slots untouched."""
    flat_p = add_attn_delta_to_full(direction_attn, base_flat_full, +eps, full_spec)
    flat_n = add_attn_delta_to_full(direction_attn, base_flat_full, -eps, full_spec)
    load_full_into(model, flat_p, full_spec, device)
    mu_p = compute_centroids(model, probes, device)
    load_full_into(model, flat_n, full_spec, device)
    mu_n = compute_centroids(model, probes, device)
    diff = (mu_p - mu_n) / (2.0 * eps)
    return float(np.mean(np.sum(diff ** 2, axis=1)))


# ──────────────────────────────────────────────────────────────────────
# SED: rolling SVD on Delta theta, or expanding SVD on displacements
# ──────────────────────────────────────────────────────────────────────

def sed_rolling(deltas: np.ndarray, t_idx: int, W: int, top_k: int):
    """Rolling W-window SVD of consecutive parameter updates."""
    t_idx = max(0, min(t_idx, len(deltas) - W))
    D = deltas[t_idx:t_idx + W]
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:top_k].astype(np.float32), S


def sed_expanding(displacements: np.ndarray, t_idx: int, top_k: int,
                  center: bool = True):
    """Expanding-window SVD of displacements theta_t - theta_0, indices [0..t_idx].

    Matches the convention of expanding_svd in coherence_edge_experiment.py:
    centers the rows then takes SVD. Right singular vectors live in theta-space.
    """
    t_idx = max(top_k + 1, t_idx)  # need at least top_k+1 samples
    X = displacements[:t_idx + 1]
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:top_k].astype(np.float32), S


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str,
                    default="/Users/tara-mini/bubble/spectral/coherence_edge_results/training_cache.pt")
    ap.add_argument("--out-dir", type=str,
                    default=str(Path(__file__).parent / "sed_lch_results"))
    ap.add_argument("--n-probes", type=int, default=1024)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--eps-rel", type=float, default=0.005,
                    help="eps = eps_rel * ||theta_subspace||")
    ap.add_argument("--n-checkpoints", type=int, default=40)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed-probes", type=int, default=0)
    ap.add_argument("--seed-random", type=int, default=1)
    ap.add_argument("--space", choices=["full", "attn"], default="full",
                    help="parameter subspace for SED + perturbation")
    ap.add_argument("--svd-mode", choices=["rolling", "expanding"], default="rolling",
                    help="SVD over rolling W-window of Δθ vs expanding window of displacements")
    ap.add_argument("--op", choices=["add", "sub", "mul", "x2_y2",
                                     "x2_xy_y2", "x3_xy"], default="add",
                    help="binary op for centroid target y = op(a,b)")
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix for output files (default = --space[_svd-mode])")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[info] device={device}")

    # ── load cache ───────────────────────────────────────────────────
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    cfg_dict = cache["cfg"]
    cfg = Config(P=cfg_dict["P"], D_MODEL=cfg_dict["D_MODEL"],
                 N_LAYERS=cfg_dict["N_LAYERS"], N_HEADS=cfg_dict["N_HEADS"],
                 D_FF=cfg_dict["D_FF"], DROPOUT=cfg_dict["DROPOUT"])
    ckpts = cache["checkpoints"]
    print(f"[info] loaded cache: {len(ckpts)} ckpts, "
          f"steps {ckpts[0][0]}->{ckpts[-1][0]}")

    full_spec = get_full_spec(ckpts[0][1])
    flats_full = np.stack([flatten_sd_full(sd) for _, sd in ckpts], axis=0)
    if args.space == "full":
        flats_sub = flats_full
    else:
        flats_sub = np.stack([flatten_sd_attn(sd) for _, sd in ckpts], axis=0)
    Pdim_full = flats_full.shape[1]
    Pdim_sub = flats_sub.shape[1]
    steps_all = np.array([s for s, _ in ckpts], dtype=np.int64)
    deltas = np.diff(flats_sub, axis=0)
    displacements = flats_sub - flats_sub[0:1]
    print(f"[info] subspace='{args.space}', svd-mode='{args.svd_mode}', "
          f"P_full={Pdim_full}, P_sub={Pdim_sub}, n_deltas={len(deltas)}")

    # ── SED across full trajectory ──────────────────────────────────────
    W = args.window
    if args.svd_mode == "rolling":
        n_win = len(deltas) - W + 1
        sigma = np.zeros((n_win, min(W, 10)))
        g23 = np.zeros(n_win)
        for i in range(n_win):
            S = np.linalg.svd(deltas[i:i + W], compute_uv=False)
            k = min(len(S), sigma.shape[1])
            sigma[i, :k] = S[:k]
            g23[i] = S[1] / max(S[2], 1e-30)
        win_centers = (steps_all[:n_win] + steps_all[W - 1:W - 1 + n_win]) // 2
        print(f"[info] SED computed for {n_win} rolling windows")
    else:
        # expanding-window: at each time t, SVD over centered displacements [0..t]
        n_win = len(displacements) - args.top_k - 1
        sigma = np.zeros((n_win, 10))
        g23 = np.zeros(n_win)
        for i in range(n_win):
            t = i + args.top_k + 1
            X = displacements[:t + 1]
            X = X - X.mean(axis=0, keepdims=True)
            S = np.linalg.svd(X, full_matrices=False, compute_uv=False)
            k = min(len(S), sigma.shape[1])
            sigma[i, :k] = S[:k]
            g23[i] = S[1] / max(S[2], 1e-30)
        win_centers = steps_all[args.top_k + 1: args.top_k + 1 + n_win]
        print(f"[info] SED expanding-window: {n_win} time points")

    # ── pick checkpoints to perturb at ────────────────────────────────
    n_ck = min(args.n_checkpoints, len(ckpts))
    ck_idx = np.linspace(0, len(ckpts) - 1, n_ck, dtype=int)
    print(f"[info] perturbing at {n_ck} checkpoints, "
          f"steps {steps_all[ck_idx[0]]}->{steps_all[ck_idx[-1]]}")

    # ── probe set ────────────────────────────────────────────────────
    op_fns = {
        "add": lambda a, b, p: (a + b) % p,
        "sub": lambda a, b, p: (a - b) % p,
        "mul": lambda a, b, p: (a * b) % p,
        "x2_y2": lambda a, b, p: (a * a + b * b) % p,
        "x2_xy_y2": lambda a, b, p: (a * a + a * b + b * b) % p,
        "x3_xy": lambda a, b, p: (a * a * a + a * b) % p,
    }
    nonzero_ops = {"mul"}
    rng_p = np.random.RandomState(args.seed_probes)
    lo = 1 if args.op in nonzero_ops else 0
    pa = rng_p.randint(lo, cfg.P, size=args.n_probes).astype(np.int64)
    pb = rng_p.randint(lo, cfg.P, size=args.n_probes).astype(np.int64)
    py = op_fns[args.op](pa, pb, cfg.P)
    probes = (torch.from_numpy(pa), torch.from_numpy(pb), torch.from_numpy(py))
    print(f"[info] op={args.op} (probe range [{lo}, {cfg.P}))")

    # ── model ────────────────────────────────────────────────────────
    model = ModAddTransformer(cfg).to(device)

    # ── per-checkpoint perturbation loop ─────────────────────────────
    A_k = np.zeros((n_ck, args.top_k))
    A_rand = np.zeros((n_ck, args.n_random))
    rank90 = np.zeros(n_ck, dtype=np.int64)
    centroid_evs = np.zeros((n_ck, 30))
    rng_dir = np.random.RandomState(args.seed_random)

    perturb_fn = (perturb_and_measure_full if args.space == "full"
                  else perturb_and_measure_attn)

    t0 = time.time()
    for j, idx in enumerate(ck_idx):
        step = int(steps_all[idx])
        flat_full = flats_full[idx]
        flat_sub = flats_sub[idx]
        eps = float(args.eps_rel * np.linalg.norm(flat_sub))

        # SED directions: window or expanding
        if args.svd_mode == "rolling":
            t_idx = max(0, min(idx, len(deltas) - W))
            Vk, _ = sed_rolling(deltas, t_idx, W, args.top_k)
        else:
            t_idx = max(args.top_k + 1, idx)
            Vk, _ = sed_expanding(displacements, t_idx, args.top_k, center=True)

        # baseline centroids and rank-90 of centered centroid matrix
        load_full_into(model, flat_full, full_spec, device)
        mu = compute_centroids(model, probes, device)
        muc = mu - mu.mean(axis=0, keepdims=True)
        s_mu = np.linalg.svd(muc, compute_uv=False)
        ev = (s_mu ** 2)
        ev_n = ev / max(ev.sum(), 1e-30)
        cum = np.cumsum(ev_n)
        rank90[j] = int(np.searchsorted(cum, 0.9) + 1)
        centroid_evs[j, :min(30, len(ev_n))] = ev_n[:30]

        # SED perturbations
        for k in range(args.top_k):
            A_k[j, k] = perturb_fn(
                model, flat_full, full_spec, Vk[k], eps, probes, device,
            )

        # random Gaussian perturbations (unit-norm in chosen subspace)
        for r in range(args.n_random):
            v = rng_dir.randn(Pdim_sub).astype(np.float32)
            v /= np.linalg.norm(v)
            A_rand[j, r] = perturb_fn(
                model, flat_full, full_spec, v, eps, probes, device,
            )

        med = float(np.median(A_rand[j]))
        Rk_str = " ".join(f"R{k+1}={A_k[j,k]/max(med,1e-30):6.2f}"
                          for k in range(args.top_k))
        print(f"  [{j+1:3d}/{n_ck}] step={step:5d}  {Rk_str}  rank90={rank90[j]:3d}  "
              f"({(time.time()-t0)/60:.1f}m)")
        sys.stdout.flush()

    A_rand_med = np.median(A_rand, axis=1)
    R_k = A_k / np.maximum(A_rand_med[:, None], 1e-30)

    # ── train/test acc ───────────────────────────────────────────────
    metrics = cache["metrics"]
    m_step = np.array([m["step"] for m in metrics])
    train_acc = np.array([m["train_acc"] for m in metrics])
    test_acc = np.array([m["test_acc"] for m in metrics])

    # ── save ─────────────────────────────────────────────────────────
    result = dict(
        cfg=cfg_dict, args=vars(args),
        ckpt_steps=steps_all[ck_idx],
        win_centers=win_centers, sigma=sigma, g23=g23,
        A_k=A_k, A_rand=A_rand, R_k=R_k,
        rank90=rank90, centroid_ev=centroid_evs,
        m_step=m_step, train_acc=train_acc, test_acc=test_acc,
    )
    tag = args.tag
    if tag is None:
        tag = args.space if args.svd_mode == "rolling" else f"{args.space}_exp"
    out_pt = out_dir / f"sed_lch_{tag}.pt"
    torch.save(result, out_pt)
    print(f"[saved] {out_pt}")

    # ── plot ─────────────────────────────────────────────────────────
    sel = steps_all[ck_idx]
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    ax[0].plot(m_step, train_acc, label="train", c="tab:blue")
    ax[0].plot(m_step, test_acc, label="test", c="tab:orange")
    ax[0].set_ylabel("accuracy")
    ax[0].legend(loc="lower right")

    ax[1].plot(win_centers, g23, c="tab:purple")
    ax[1].axhline(1.0, c="gray", ls=":", lw=0.5)
    ax[1].set_ylabel(r"$\sigma_2/\sigma_3$ (edge gap)")
    ax[1].set_yscale("log")

    for k in range(args.top_k):
        ax[2].plot(sel, R_k[:, k], marker="o", ms=3, label=fr"$R_{k+1}$")
    ax[2].axhline(1.0, c="gray", ls=":", lw=0.5)
    ax[2].set_ylabel(r"$R_k = A_k / A_{\rm rand}$")
    ax[2].set_yscale("log")
    ax[2].legend()

    ax[3].plot(sel, rank90, c="tab:green", marker="o", ms=3)
    ax[3].set_ylabel(r"rank-90 of $\mu$ matrix")
    ax[3].set_xlabel("training step")

    fig.suptitle(f"SED-LCH coupling, grokking mod-add (p=97, seed=42, space={args.space})")
    fig.tight_layout()
    out_png = out_dir / f"sed_lch_{tag}.png"
    fig.savefig(out_png, dpi=130)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
