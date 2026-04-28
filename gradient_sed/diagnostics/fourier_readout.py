#!/usr/bin/env python3
"""Fourier readout of LCH centroid PCs over training.

For each checkpoint:
  1. Compute centroids mu_x = grad_emb logit_y(x) on a fixed probe set.
  2. SVD to get top-K centroid PCs and PC scores s_j(x).
  3. For each (j, omega) fit s_j ~ a0 + a1 cos(2 pi omega (a+b)/p) + a2 sin(...);
     also fit a-only and b-only Fourier components separately as controls.
  4. Track best R^2 and best omega per PC over training.

Predicts sharp omega-modes emerge around grokking and persist post-grokking.

Output: <out_dir>/fourier_<tag>.pt and fourier_<tag>.png
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


# ── model ──────────────────────────────────────────────────────────────

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


# ── centroid ────────────────────────────────────────────────────────────

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


# ── Fourier R^2 fit ─────────────────────────────────────────────────────

def fit_R2(scores: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Multi-output OLS R^2.
    scores: (N, K)  signals to fit
    design: (N, F)  features (include intercept column!)
    Returns R^2 per signal (K,).
    """
    N, K = scores.shape
    # solve beta = (X^T X)^-1 X^T y using lstsq
    beta, *_ = np.linalg.lstsq(design, scores, rcond=None)
    pred = design @ beta            # (N, K)
    resid = scores - pred
    ss_res = (resid ** 2).sum(axis=0)
    s_mean = scores.mean(axis=0)
    ss_tot = ((scores - s_mean) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-30)


def fourier_R2_per_omega(scores: np.ndarray, a: np.ndarray, b: np.ndarray, p: int,
                         omegas: np.ndarray):
    """For each omega and each score column, return R^2 for three single-omega bases:
    (a+b), a-only, b-only. Returns dict of arrays of shape (n_omegas, K).
    """
    N, K = scores.shape
    out = {basis: np.zeros((len(omegas), K)) for basis in ["sum", "a", "b"]}
    ones = np.ones((N, 1))
    for oi, omega in enumerate(omegas):
        for basis_name, idx in (("sum", a + b), ("a", a), ("b", b)):
            angle = 2 * np.pi * omega * idx / p
            design = np.concatenate(
                [ones, np.cos(angle).reshape(-1, 1), np.sin(angle).reshape(-1, 1)],
                axis=1,
            )
            out[basis_name][oi] = fit_R2(scores, design)
    return out


def compute_dlog_table(p: int, generator: int = 5) -> np.ndarray:
    """Discrete-log table for Z/p^*. dlog[1..p-1] = k s.t. g^k = a; dlog[0] = -1."""
    dlog = np.full(p, -1, dtype=np.int64)
    g = 1
    for k in range(p - 1):
        dlog[g] = k
        g = (g * generator) % p
    assert (dlog[1:] >= 0).all(), f"generator {generator} is not a generator of Z/{p}^*"
    return dlog


def fourier_R2_full_basis(scores: np.ndarray, a: np.ndarray, b: np.ndarray,
                          y: np.ndarray, p: int, omegas: np.ndarray,
                          dlog: np.ndarray = None):
    """R^2 of fitting scores with full Fourier basis at all omegas.

    Additive bases (always computed):
      - 'sum'  : (a+b) Fourier
      - 'a'    : a-only Fourier
      - 'b'    : b-only Fourier
      - 'a_b'  : a-only + b-only
      - 'y'    : y=op(a,b) Fourier  (the "answer" Fourier, op-specific)

    Multiplicative (log-space) bases — only when dlog provided AND a,b,y all nonzero:
      - 'log_a'   : log(a) Fourier
      - 'log_b'   : log(b) Fourier
      - 'log_a_b' : log(a) + log(b)
      - 'log_y'   : log(y) Fourier  (= log(a)+log(b) for mul)
    """
    N, K = scores.shape
    ones = np.ones((N, 1))
    ang_sum = 2 * np.pi * omegas[None, :] * (a + b)[:, None] / p
    ang_a = 2 * np.pi * omegas[None, :] * a[:, None] / p
    ang_b = 2 * np.pi * omegas[None, :] * b[:, None] / p
    ang_y = 2 * np.pi * omegas[None, :] * y[:, None] / p
    X_sum = np.concatenate([ones, np.cos(ang_sum), np.sin(ang_sum)], axis=1)
    X_a = np.concatenate([ones, np.cos(ang_a), np.sin(ang_a)], axis=1)
    X_b = np.concatenate([ones, np.cos(ang_b), np.sin(ang_b)], axis=1)
    X_ab = np.concatenate([ones, np.cos(ang_a), np.sin(ang_a),
                           np.cos(ang_b), np.sin(ang_b)], axis=1)
    X_y = np.concatenate([ones, np.cos(ang_y), np.sin(ang_y)], axis=1)
    out = {
        "sum": fit_R2(scores, X_sum),
        "a": fit_R2(scores, X_a),
        "b": fit_R2(scores, X_b),
        "a_b": fit_R2(scores, X_ab),
        "y": fit_R2(scores, X_y),
    }
    if dlog is not None and (a > 0).all() and (b > 0).all() and (y > 0).all():
        p_m1 = p - 1
        log_a = dlog[a]
        log_b = dlog[b]
        log_y = dlog[y]
        ang_la = 2 * np.pi * omegas[None, :] * log_a[:, None] / p_m1
        ang_lb = 2 * np.pi * omegas[None, :] * log_b[:, None] / p_m1
        ang_ly = 2 * np.pi * omegas[None, :] * log_y[:, None] / p_m1
        X_la = np.concatenate([ones, np.cos(ang_la), np.sin(ang_la)], axis=1)
        X_lb = np.concatenate([ones, np.cos(ang_lb), np.sin(ang_lb)], axis=1)
        X_lab = np.concatenate([ones, np.cos(ang_la), np.sin(ang_la),
                                np.cos(ang_lb), np.sin(ang_lb)], axis=1)
        X_ly = np.concatenate([ones, np.cos(ang_ly), np.sin(ang_ly)], axis=1)
        out["log_a"] = fit_R2(scores, X_la)
        out["log_b"] = fit_R2(scores, X_lb)
        out["log_a_b"] = fit_R2(scores, X_lab)
        out["log_y"] = fit_R2(scores, X_ly)
    return out


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir",
                    default=str(Path(__file__).parent / "sed_lch_results"))
    ap.add_argument("--n-probes", type=int, default=1024,
                    help="ignored if --grid is set")
    ap.add_argument("--grid", action="store_true", default=True,
                    help="use full p^2 grid of (a,b) pairs (default)")
    ap.add_argument("--no-grid", dest="grid", action="store_false")
    ap.add_argument("--op", choices=["add", "sub", "mul", "x2_y2",
                                     "x2_xy_y2", "x3_xy"], default="add",
                    help="op for centroid target")
    ap.add_argument("--n-checkpoints", type=int, default=40)
    ap.add_argument("--top-pc", type=int, default=5)
    ap.add_argument("--seed-probes", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or get_device()
    print(f"[info] device={device}, tag={args.tag}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    cfg_d = cache["cfg"]
    cfg = Config(P=cfg_d["P"], D_MODEL=cfg_d["D_MODEL"],
                 N_LAYERS=cfg_d["N_LAYERS"], N_HEADS=cfg_d["N_HEADS"],
                 D_FF=cfg_d["D_FF"], DROPOUT=cfg_d["DROPOUT"])
    ckpts = cache["checkpoints"]
    steps_all = np.array([s for s, _ in ckpts], dtype=np.int64)
    n_ck = min(args.n_checkpoints, len(ckpts))
    ck_idx = np.linspace(0, len(ckpts) - 1, n_ck, dtype=int)
    print(f"[info] {len(ckpts)} ckpts in cache, sampling {n_ck}")

    op_fns = {
        "add": lambda a, b, p: (a + b) % p,
        "sub": lambda a, b, p: (a - b) % p,
        "mul": lambda a, b, p: (a * b) % p,
        "x2_y2": lambda a, b, p: (a * a + b * b) % p,
        "x2_xy_y2": lambda a, b, p: (a * a + a * b + b * b) % p,
        "x3_xy": lambda a, b, p: (a * a * a + a * b) % p,
    }
    nonzero = args.op in {"mul"}
    lo = 1 if nonzero else 0
    if args.grid:
        rg = np.arange(lo, cfg.P)
        pa = np.repeat(rg, len(rg)).astype(np.int64)
        pb = np.tile(rg, len(rg)).astype(np.int64)
        print(f"[info] grid probes: {len(pa)} (range [{lo},{cfg.P}))")
    else:
        rng = np.random.RandomState(args.seed_probes)
        pa = rng.randint(lo, cfg.P, size=args.n_probes).astype(np.int64)
        pb = rng.randint(lo, cfg.P, size=args.n_probes).astype(np.int64)
        print(f"[info] random probes: {len(pa)}")
    py = op_fns[args.op](pa, pb, cfg.P)
    probes = (torch.from_numpy(pa), torch.from_numpy(pb), torch.from_numpy(py))
    print(f"[info] op={args.op}")

    model = ModAddTransformer(cfg).to(device)

    omegas = np.arange(1, cfg.P // 2 + 1)
    K = args.top_pc
    R2_sum_o = np.zeros((n_ck, len(omegas), K))
    R2_a_o = np.zeros((n_ck, len(omegas), K))
    R2_b_o = np.zeros((n_ck, len(omegas), K))
    R2_full_sum = np.zeros((n_ck, K))
    R2_full_a = np.zeros((n_ck, K))
    R2_full_b = np.zeros((n_ck, K))
    R2_full_a_b = np.zeros((n_ck, K))
    R2_full_y = np.zeros((n_ck, K))
    R2_full_log_a = np.zeros((n_ck, K))
    R2_full_log_b = np.zeros((n_ck, K))
    R2_full_log_a_b = np.zeros((n_ck, K))
    R2_full_log_y = np.zeros((n_ck, K))
    log_basis_used = False
    dlog_table = compute_dlog_table(cfg.P, generator=5)
    pc_var_share = np.zeros((n_ck, K))
    pc_total_var = np.zeros(n_ck)

    t0 = time.time()
    for j, idx in enumerate(ck_idx):
        step = int(steps_all[idx])
        sd = ckpts[idx][1]
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})

        mu = compute_centroids(model, probes, device)
        muc = mu - mu.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(muc, full_matrices=False)
        scores = U[:, :K] * S[:K]
        ev = (S ** 2)
        ev_n = ev / max(ev.sum(), 1e-30)
        pc_var_share[j] = ev_n[:K]
        pc_total_var[j] = ev.sum()

        r2_o = fourier_R2_per_omega(scores, pa, pb, cfg.P, omegas)
        R2_sum_o[j] = r2_o["sum"]
        R2_a_o[j] = r2_o["a"]
        R2_b_o[j] = r2_o["b"]

        r2_f = fourier_R2_full_basis(scores, pa, pb, py, cfg.P, omegas, dlog_table)
        R2_full_sum[j] = r2_f["sum"]
        R2_full_a[j] = r2_f["a"]
        R2_full_b[j] = r2_f["b"]
        R2_full_a_b[j] = r2_f["a_b"]
        R2_full_y[j] = r2_f["y"]
        if "log_y" in r2_f:
            R2_full_log_a[j] = r2_f["log_a"]
            R2_full_log_b[j] = r2_f["log_b"]
            R2_full_log_a_b[j] = r2_f["log_a_b"]
            R2_full_log_y[j] = r2_f["log_y"]
            log_basis_used = True

        best_per_pc = R2_sum_o[j].max(axis=0)
        best_omega_per_pc = omegas[R2_sum_o[j].argmax(axis=0)]
        msg = " ".join(f"PC{k+1}=R²{best_per_pc[k]:.2f}@ω{best_omega_per_pc[k]:2d}"
                       for k in range(min(3, K)))
        full_msg = (f"full[a+b]={R2_full_sum[j,:3].mean():.2f} "
                    f"full[a&b]={R2_full_a_b[j,:3].mean():.2f} "
                    f"full[y={args.op}]={R2_full_y[j,:3].mean():.2f}")
        if log_basis_used:
            full_msg += (f" log[y]={R2_full_log_y[j,:3].mean():.2f} "
                         f"log[a&b]={R2_full_log_a_b[j,:3].mean():.2f}")
        print(f"  [{j+1:3d}/{n_ck}] step={step:5d}  {msg}  "
              f"{full_msg}  ({(time.time()-t0)/60:.2f}m)")
        sys.stdout.flush()

    result = dict(
        cfg=cfg_d, args=vars(args),
        ckpt_steps=steps_all[ck_idx],
        omegas=omegas,
        R2_sum_o=R2_sum_o, R2_a_o=R2_a_o, R2_b_o=R2_b_o,
        R2_full_sum=R2_full_sum, R2_full_a=R2_full_a,
        R2_full_b=R2_full_b, R2_full_a_b=R2_full_a_b,
        R2_full_y=R2_full_y,
        R2_full_log_a=R2_full_log_a, R2_full_log_b=R2_full_log_b,
        R2_full_log_a_b=R2_full_log_a_b, R2_full_log_y=R2_full_log_y,
        log_basis_used=log_basis_used,
        pc_var_share=pc_var_share, pc_total_var=pc_total_var,
        m_step=np.array([m["step"] for m in cache["metrics"]]),
        train_acc=np.array([m["train_acc"] for m in cache["metrics"]]),
        test_acc=np.array([m["test_acc"] for m in cache["metrics"]]),
    )
    out_pt = out_dir / f"fourier_{args.tag}.pt"
    torch.save(result, out_pt)
    print(f"[saved] {out_pt}")

    # ── plot: full-basis R^2 over training (top-3 PCs averaged), plus
    #         (a+b)-R^2 heatmap for PC1 ─────────────────────────────────
    sel = steps_all[ck_idx]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax[0].plot(result["m_step"], result["train_acc"], c="tab:blue", label="train")
    ax[0].plot(result["m_step"], result["test_acc"], c="tab:orange", label="test")
    ax[0].set_ylabel("accuracy")
    ax[0].legend(loc="lower right")

    # mean over top-3 PCs
    ax[1].plot(sel, R2_full_sum[:, :3].mean(axis=1), "-o", ms=3,
               c="tab:purple", label=r"$R^2$ (a+b basis)")
    ax[1].plot(sel, R2_full_a[:, :3].mean(axis=1), "--", c="tab:red",
               label=r"$R^2$ (a-only)")
    ax[1].plot(sel, R2_full_b[:, :3].mean(axis=1), "--", c="tab:brown",
               label=r"$R^2$ (b-only)")
    ax[1].plot(sel, R2_full_a_b[:, :3].mean(axis=1), "-o", ms=3,
               c="tab:green", label=r"$R^2$ (a&b combined)")
    ax[1].plot(sel, R2_full_y[:, :3].mean(axis=1), "-s", ms=4,
               c="tab:cyan", label=fr"$R^2$ (y={args.op} additive)")
    if log_basis_used:
        ax[1].plot(sel, R2_full_log_y[:, :3].mean(axis=1), "-D", ms=4,
                   c="tab:olive", label=fr"$R^2$ (log y={args.op})")
        ax[1].plot(sel, R2_full_log_a_b[:, :3].mean(axis=1), "--",
                   c="darkmagenta", label=r"$R^2$ (log a & log b)")
    ax[1].set_ylabel(r"$R^2$ (mean over top-3 PCs)")
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].legend(loc="best", fontsize=9)

    # heatmap of single-omega (a+b) R^2 for PC1
    im = ax[2].imshow(R2_sum_o[:, :, 0].T, aspect="auto", origin="lower",
                      extent=[sel[0], sel[-1], omegas[0], omegas[-1]],
                      cmap="viridis", vmin=0, vmax=R2_sum_o[:, :, 0].max())
    ax[2].set_ylabel(r"$\omega$  (PC1, single-ω)")
    ax[2].set_xlabel("training step")
    plt.colorbar(im, ax=ax[2], label=r"$R^2$")

    fig.suptitle(f"Fourier readout on centroid PCs — {args.tag}")
    fig.tight_layout()
    out_png = out_dir / f"fourier_{args.tag}.png"
    fig.savefig(out_png, dpi=130)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
