"""Tian Theorem 1 / Stage II independence proxy.

For modular addition with σ(x)=x², Theorem 1 says that during Stage II each
hidden node's gradient g_j depends only on its own w_j: w_j evolves
independently. The empirical content is that the columns of G_F should be
approximately decoupled across nodes — i.e. <g_j, g_{j'}> / (||g_j|| ||g_{j'}||)
should be small for j != j'.

This script loads a checkpoint from a saved Tian run, evaluates G_F on the full
training set, and computes the off-diagonal cosine distribution of its columns.

Usage:
  python independence_proxy.py --tag sweep_eta0.0002_seed0 --epoch 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tian_eigengap import TianMLP, build_modadd_dataset, split_train_test, pick_device


def load_config(tag: str) -> dict:
    cfg_path = Path("runs") / tag / "config.json"
    with open(cfg_path) as f:
        return json.load(f)


def reconstruct_run_state(cfg: dict, target_epoch: int):
    """We re-run training up to `target_epoch` deterministically and then
    compute G_F on the full training set. Using the same seed yields the same
    trajectory."""
    device = pick_device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    X, y = build_modadd_dataset(cfg["M"])
    Xtr, ytr, Xte, yte = split_train_test(X, y, cfg["n_train"], cfg["seed"])
    Xtr, ytr = Xtr.to(device), ytr.to(device)

    Y_oh = F.one_hot(ytr, num_classes=cfg["M"]).float()
    Y_zm = Y_oh - 1.0 / cfg["M"]

    model = TianMLP(cfg["M"], num_ops=2, K=cfg["K"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["eta"])

    from tian_eigengap import tian_mse_loss
    for epoch in range(target_epoch + 1):
        if epoch == target_epoch:
            break
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(Xtr)
        loss = tian_mse_loss(out, ytr, cfg["M"])
        loss.backward()
        opt.step()
    return model, Xtr, ytr, Y_zm


@torch.no_grad()
def compute_gF_columns(model: TianMLP, X_train, Y_zm):
    """Return G_F = P_1^perp (Y - F V) V^T   shape [n, K]."""
    F_act = model.hidden(X_train)
    Y_hat = model.V(F_act)
    Y_hat_zm = Y_hat - Y_hat.mean(dim=0, keepdim=True)
    Y_zm_p = Y_zm - Y_zm.mean(dim=0, keepdim=True)
    res = Y_zm_p - Y_hat_zm
    return res @ model.V.weight  # [n, K]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, nargs="+", default=[10, 50, 100, 150])
    p.add_argument("--n-pairs", type=int, default=2000)
    args = p.parse_args()

    cfg = load_config(args.tag)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(args.epochs), figsize=(3.5 * len(args.epochs), 3.2),
                             sharey=True)
    if len(args.epochs) == 1:
        axes = [axes]
    summary = {}
    for ax, ep in zip(axes, args.epochs):
        model, Xtr, ytr, Yzm = reconstruct_run_state(cfg, ep)
        gF = compute_gF_columns(model, Xtr, Yzm)              # [n, K]
        K = gF.size(1)
        # sample random pairs
        idx_a = rng.integers(0, K, size=args.n_pairs)
        idx_b = rng.integers(0, K, size=args.n_pairs)
        keep = idx_a != idx_b
        idx_a, idx_b = idx_a[keep], idx_b[keep]
        a = gF[:, idx_a]
        b = gF[:, idx_b]
        an = torch.linalg.norm(a, dim=0)
        bn = torch.linalg.norm(b, dim=0)
        cos = (a * b).sum(dim=0) / (an.clamp(min=1e-30) * bn.clamp(min=1e-30))
        cos = cos.cpu().numpy()
        ax.hist(cos, bins=60, alpha=0.85)
        ax.set_title(f"epoch {ep}")
        ax.set_xlabel("cos(g_j, g_{j'})")
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        summary[ep] = dict(median_abs=float(np.median(np.abs(cos))),
                           p95_abs=float(np.percentile(np.abs(cos), 95)),
                           mean=float(cos.mean()))
    axes[0].set_ylabel("count")
    fig.suptitle(f"Independence proxy: cos(g_j, g_{{j'}}) — {args.tag}")
    plt.tight_layout()
    out = Path("runs") / f"{args.tag}_independence_proxy.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
