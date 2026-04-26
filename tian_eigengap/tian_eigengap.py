"""
Replicate Tian (arXiv:2509.21519) Fig. 3 modular-addition dynamics
on the M4 Max (MPS), and (optionally) layer in a rolling-window
parameter-update eigengap signal as a candidate Stage II->III detector.

Tian's setup:
  - 2-layer MLP: Y_hat = sigma(X W) V, sigma(x) = x^2
  - Identity embedding: each token mapped to its one-hot in R^M, concatenated
  - MSE loss against zero-meaned one-hot target (Y = onehot - 1/M)
  - Adam with weight_decay = eta (the paper's eta is weight decay, NOT lr)

First-pass usage (Fig 3 sanity, no eigengap):
  python tian_eigengap.py --eta 0.0002 --seed 0 --epochs 400 --no-eigengap

With eigengap tracker:
  python tian_eigengap.py --eta 0.0002 --seed 0 --epochs 400 --window 20

Outputs per run: runs/<tag>/log.csv  (one row per epoch, all scalars)
                 runs/<tag>/config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ device

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------- dataset

def build_modadd_dataset(M: int):
    """All (a,b) pairs in [0,M)^2 with target (a+b) mod M."""
    a = torch.arange(M).repeat_interleave(M)
    b = torch.arange(M).repeat(M)
    y = (a + b) % M
    X = torch.stack([a, b], dim=1)  # [M^2, 2]
    return X, y


def split_train_test(X, y, n_train: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.size(0), generator=g)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ------------------------------------------------------------------ model

class TianMLP(nn.Module):
    """2-layer MLP with identity embedding, sigma(x)=x^2, no biases."""

    def __init__(self, M: int, num_ops: int, K: int):
        super().__init__()
        self.M = M
        self.num_ops = num_ops
        self.K = K
        # frozen identity embedding -> we implement it as a one-hot lookup at forward
        self.W = nn.Linear(num_ops * M, K, bias=False)
        self.V = nn.Linear(K, M, bias=False)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_ops] long -> [B, num_ops*M] float (concatenated one-hots)
        oh = F.one_hot(x, num_classes=self.M).float()       # [B, num_ops, M]
        return oh.view(x.size(0), self.num_ops * self.M)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        return (self.W(e)).pow(2)                           # F = sigma(X W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_act = self.hidden(x)
        return self.V(F_act)                                # F V


# ----------------------------------------------------------- loss / acc

def tian_mse_loss(out: torch.Tensor, y: torch.Tensor, M: int) -> torch.Tensor:
    """MSE with zero-meaned target Y = onehot(y) - 1/M, exactly as in Tian's code."""
    out_zm = out - out.mean(dim=1, keepdim=True)
    # ||out_zm||^2 - 2 <out_zm, e_y> + (1 - 1/M)
    sq = out_zm.pow(2).sum(dim=1).mean()
    cross = out_zm.gather(1, y.unsqueeze(1)).mean()
    return sq - 2.0 * cross + (1.0 - 1.0 / M)


@torch.no_grad()
def acc_and_loss(model: TianMLP, X, y, M: int):
    out = model(X)
    pred = out.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    loss = tian_mse_loss(out, y, M).item()
    return acc, loss


# ---------------------------------------------------- Tian Fig 3 metrics

@torch.no_grad()
def tian_metrics(model: TianMLP, X_train, Y_train_zm, M: int,
                 indep_pairs: int = 500, indep_rng: np.random.Generator | None = None,
                 do_static_eig: bool = False, do_ftf_flat: bool = False):
    """Compute the panels of Tian's Fig. 3 + the new activation-Gram quantities.

    Returns dict with:
      ftf_diag, ftf_off, ftf_ratio              (~F^T ~F)  -- "Tian metric"
      fft_diag, fft_off, fft_ratio,
      fft_dist_from_ideal                       (P_1^perp F F^T)
      gF_norm                                    ||G_F||
      ftf_eig1..5, ftf_eig_gap23, ftf_eig_gap12  static F̃^T F̃ top-5 eigenvalues
      indep_cos_med, indep_cos_p95              median / 95th-pct of |cos(g_j, g_{j'})|

    Also returns the flattened F̃^T F̃ matrix (CPU, float32) for the rolling-delta tracker.
    """
    F_act = model.hidden(X_train)                         # [n, K]
    F_zm = F_act - F_act.mean(dim=0, keepdim=True)        # P_1^perp F  (n x K)

    # ~F^T ~F  (K x K)
    ftf = F_zm.t() @ F_zm
    ftf_diag = ftf.diag().abs().mean().item()
    K = ftf.size(0)
    ftf_off = (ftf.abs().sum() - ftf.diag().abs().sum()).item() / (K * (K - 1))

    # static top-k eigvals of F̃^T F̃ — only if requested (slow, ~1s/epoch on CPU).
    # In practice (M=71 single-seed pilot) all top-3 ratios stay near 1, so we
    # default this off in sweeps.
    if do_static_eig:
        sv = torch.linalg.svdvals(F_zm.detach().to("cpu"))
        eig = (sv * sv)[:5].tolist()
        while len(eig) < 5:
            eig.append(0.0)
    else:
        eig = [float("nan")] * 5

    # F F^T (n x n) -- Tian's fit_diag_11 on the RAW kernel
    fft_raw = F_act @ F_act.t()
    n = fft_raw.size(0)
    dmean = fft_raw.diag().mean()
    omean = (fft_raw.sum() - fft_raw.diag().sum()) / (n * (n - 1))
    ideal = (dmean - omean) * torch.eye(n, device=fft_raw.device, dtype=fft_raw.dtype) + omean
    dist_from_ideal = (torch.linalg.norm(ideal - fft_raw) / torch.linalg.norm(fft_raw)).item()
    # P_1^perp F F^T
    fft = fft_raw - fft_raw.mean(dim=0, keepdim=True)
    fft_diag = fft.diag().abs().mean().item()
    fft_off = (fft.abs().sum() - fft.diag().abs().sum()).item() / (n * (n - 1))

    # G_F = P_1^perp (Y - F V) V^T   shape [n, K]
    Y_hat = model.V(F_act)
    Y_hat_zm = Y_hat - Y_hat.mean(dim=0, keepdim=True)
    Y_zm_proj = Y_train_zm - Y_train_zm.mean(dim=0, keepdim=True)
    residual = Y_zm_proj - Y_hat_zm
    gF = residual @ model.V.weight
    gF_norm = torch.linalg.norm(gF).item()

    # independence proxy: median / p95 of |cos(g_j, g_{j'})| over random pairs
    if indep_rng is not None and indep_pairs > 0:
        a_idx = indep_rng.integers(0, K, size=indep_pairs)
        b_idx = indep_rng.integers(0, K, size=indep_pairs)
        keep = a_idx != b_idx
        a_idx, b_idx = a_idx[keep], b_idx[keep]
        a = gF[:, a_idx]
        b = gF[:, b_idx]
        an = torch.linalg.norm(a, dim=0).clamp(min=1e-30)
        bn = torch.linalg.norm(b, dim=0).clamp(min=1e-30)
        cos = ((a * b).sum(dim=0) / (an * bn)).abs()
        indep_med = float(cos.median())
        indep_p95 = float(cos.quantile(0.95))
    else:
        indep_med = float("nan")
        indep_p95 = float("nan")

    out = {
        "ftf_diag": ftf_diag,
        "ftf_off": ftf_off,
        "ftf_ratio": ftf_off / max(ftf_diag, 1e-30),
        "fft_diag": fft_diag,
        "fft_off": fft_off,
        "fft_ratio": fft_off / max(fft_diag, 1e-30),
        "fft_dist_from_ideal": dist_from_ideal,
        "gF_norm": gF_norm,
        "ftf_eig1": eig[0], "ftf_eig2": eig[1], "ftf_eig3": eig[2],
        "ftf_eig4": eig[3], "ftf_eig5": eig[4],
        "ftf_eig_gap23": eig[1] / eig[2] if eig[2] > 0 else float("nan"),
        "ftf_eig_gap12": eig[0] / eig[1] if eig[1] > 0 else float("nan"),
        "indep_cos_med": indep_med,
        "indep_cos_p95": indep_p95,
    }
    # also return the flat F̃^T F̃ (CPU float32) for the rolling delta tracker
    # — only if requested (~16MB transfer per epoch + storage in deque).
    ftf_flat = ftf.detach().to("cpu", dtype=torch.float32).flatten() if do_ftf_flat else None
    return out, ftf_flat


# -------------------------------------------------- W / V update cosine

def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten()
    b = b.flatten()
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return 1.0 - (a @ b / (na * nb)).item()


# ------------------------------------------------ rolling-window eigengap

class EigengapTracker:
    """Maintain a deque of the last W flattened parameter deltas for one tensor.
    At each step: form Delta in R^{P x W}, compute Delta^T Delta in R^{W x W},
    return its top-k eigenvalues. Cheap because W << P."""

    def __init__(self, window: int, k: int = 5):
        self.window = window
        self.k = k
        self.buf: deque[torch.Tensor] = deque(maxlen=window)

    def push(self, delta: torch.Tensor):
        self.buf.append(delta.detach().flatten().to("cpu"))

    def topk(self):
        if len(self.buf) < 2:
            return None
        D = torch.stack(list(self.buf), dim=1)            # [P, len]
        G = D.t() @ D                                     # [len, len]
        evals = torch.linalg.eigvalsh(G).flip(0)          # descending
        return evals[: self.k].tolist()


class FtfDeltaTracker:
    """Same idea as EigengapTracker but on FLATTENED F̃^T F̃ matrices.
    push() takes the flat F̃^T F̃ at the current epoch; we maintain
    the previous flat and the deque of deltas (size = window)."""

    def __init__(self, window: int, k: int = 5):
        self.window = window
        self.k = k
        self.prev_flat: torch.Tensor | None = None
        self.buf: deque[torch.Tensor] = deque(maxlen=window)

    def push(self, ftf_flat: torch.Tensor):
        if self.prev_flat is not None:
            self.buf.append(ftf_flat - self.prev_flat)
        self.prev_flat = ftf_flat

    def topk(self):
        if len(self.buf) < 2:
            return None
        D = torch.stack(list(self.buf), dim=1)            # [P, len]
        G = D.t() @ D
        evals = torch.linalg.eigvalsh(G).flip(0)
        return evals[: self.k].tolist()


# ------------------------------------------------------------------- run

def run(args):
    device = pick_device()
    print(f"[device] {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data
    X, y = build_modadd_dataset(args.M)
    n_total = X.size(0)
    if args.n_train is None:
        n_train = int(round((1 - args.test_size) * n_total))
    else:
        n_train = args.n_train
    X_train, y_train, X_test, y_test = split_train_test(X, y, n_train, args.seed)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # zero-meaned target for G_F computation
    Y_oh = F.one_hot(y_train, num_classes=args.M).float()
    Y_zm = Y_oh - 1.0 / args.M

    # model + opt
    model = TianMLP(args.M, num_ops=2, K=args.K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.eta)

    # output dir
    run_tag = args.tag or f"M{args.M}_K{args.K}_eta{args.eta}_lr{args.lr}_seed{args.seed}"
    out_dir = Path(args.out) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args) | {"n_train": n_train, "n_total": n_total}, f, indent=2)

    log_path = out_dir / "log.jsonl"
    flog = open(log_path, "w")

    # eigengap state
    do_eigengap = (args.window is not None) and (args.window > 0)
    do_ftf_delta = do_eigengap and args.ftf_delta
    if do_eigengap:
        gapW = EigengapTracker(window=args.window, k=5)
        gapV = EigengapTracker(window=args.window, k=5)
        gapFTF = FtfDeltaTracker(window=args.window, k=5) if do_ftf_delta else None
    prev_W = model.W.weight.detach().clone()
    prev_V = model.V.weight.detach().clone()
    last_dW = None
    last_dV = None
    indep_rng = np.random.default_rng(args.seed)

    t0 = time.time()
    for epoch in range(args.epochs + 1):
        # eval first (Tian logs metrics BEFORE the step -- mirror that)
        train_acc, train_loss = acc_and_loss(model, X_train, y_train, args.M)
        test_acc, test_loss = acc_and_loss(model, X_test, y_test, args.M)
        tm, ftf_flat = tian_metrics(model, X_train, Y_zm, args.M,
                                    indep_pairs=args.indep_pairs, indep_rng=indep_rng,
                                    do_static_eig=args.static_eig,
                                    do_ftf_flat=do_ftf_delta)
        if do_ftf_delta and ftf_flat is not None:
            gapFTF.push(ftf_flat)

        row = {
            "epoch": epoch,
            "train_acc": train_acc, "test_acc": test_acc,
            "train_loss": train_loss, "test_loss": test_loss,
            **tm,
        }

        # eigengap log
        if do_eigengap:
            evW = gapW.topk()
            evV = gapV.topk()
            evF = gapFTF.topk() if do_ftf_delta else None
            if evW is not None:
                row.update({
                    "W_sigma1": evW[0], "W_sigma2": evW[1] if len(evW) > 1 else float("nan"),
                    "W_sigma3": evW[2] if len(evW) > 2 else float("nan"),
                    "W_gap23": (evW[1] / evW[2]) if len(evW) > 2 and evW[2] > 0 else float("nan"),
                })
            if evV is not None:
                row.update({
                    "V_sigma1": evV[0], "V_sigma2": evV[1] if len(evV) > 1 else float("nan"),
                    "V_sigma3": evV[2] if len(evV) > 2 else float("nan"),
                    "V_gap23": (evV[1] / evV[2]) if len(evV) > 2 and evV[2] > 0 else float("nan"),
                })
            if evF is not None:
                row.update({
                    "FTFd_sigma1": evF[0],
                    "FTFd_sigma2": evF[1] if len(evF) > 1 else float("nan"),
                    "FTFd_sigma3": evF[2] if len(evF) > 2 else float("nan"),
                    "FTFd_gap23": (evF[1] / evF[2]) if len(evF) > 2 and evF[2] > 0 else float("nan"),
                })

        # cosine of consecutive updates (Fig 3 right panel)
        if last_dW is not None:
            dW_now = (model.W.weight.detach() - prev_W)
            dV_now = (model.V.weight.detach() - prev_V)
            row["W_step_cos_dist"] = cosine_dist(last_dW, dW_now)
            row["V_step_cos_dist"] = cosine_dist(last_dV, dV_now)

        flog.write(json.dumps(row) + "\n")
        flog.flush()

        if epoch % args.eval_every == 0:
            print(f"epoch {epoch:4d}  train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
                  f"  ftf_eig_gap23={tm['ftf_eig_gap23']:.3f}"
                  f"  indep={tm['indep_cos_med']:.3f}"
                  f"  gF={tm['gF_norm']:.3e}  loss={train_loss:.4f}")

        if epoch == args.epochs:
            break

        # ------ optimization step ------
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(X_train)
        loss = tian_mse_loss(out, y_train, args.M)
        loss.backward()

        # capture pre-step weights, take step, then compute deltas
        prev_W = model.W.weight.detach().clone()
        prev_V = model.V.weight.detach().clone()
        opt.step()

        dW = model.W.weight.detach() - prev_W
        dV = model.V.weight.detach() - prev_V
        if do_eigengap:
            gapW.push(dW)
            gapV.push(dV)
        last_dW = dW.clone()
        last_dV = dV.clone()

    flog.close()
    dt = time.time() - t0
    print(f"[done] {dt:.1f}s  -> {log_path}")


# -------------------------------------------------------------- argparse

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=71)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--n-train", type=int, default=2016, help="Tian Fig.3 default")
    p.add_argument("--test-size", type=float, default=None,
                   help="alternative: test fraction (used iff --n-train not given)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta", type=float, default=2e-4,
                   help="weight decay (paper's eta). Use 0 for no-grok control.")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=20,
                   help="rolling-window size for eigengap tracker. 0 disables.")
    p.add_argument("--no-eigengap", action="store_true",
                   help="shortcut for --window 0")
    p.add_argument("--indep-pairs", type=int, default=500,
                   help="random pairs for the cos(g_j, g_{j'}) independence proxy")
    p.add_argument("--static-eig", action="store_true",
                   help="compute static F̃^T F̃ top-5 eigvals (slow CPU SVD; default off)")
    p.add_argument("--ftf-delta", action="store_true",
                   help="compute rolling-window eigengap on F̃^T F̃ deltas (slow; default off)")
    p.add_argument("--out", type=str, default="runs")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()
    if args.no_eigengap:
        args.window = 0
    return args


if __name__ == "__main__":
    run(parse())
