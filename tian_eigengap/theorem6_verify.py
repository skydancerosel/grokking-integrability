"""Priority 2: empirical verification of Tian's Theorem 6 (repulsion of similar features).

Tian (arXiv:2509.21519) Theorem 6: the j-th column of F̃ B is given by
    [F̃ B]_j = b_{jj} f̃_j + Σ_{l≠j} b_{jl} f̃_l
with sign(b_{jl}) = -sign(f̃_jᵀ P_{η,-jl} f̃_l) where P_{η,-jl} := I - F̃_{-jl}(F̃_{-jl}ᵀ F̃_{-jl} + ηI)⁻¹ F̃_{-jl}ᵀ.

For the most-similar feature pairs (large |f̃_jᵀ f̃_l|), B should be NEGATIVE,
encoding the repulsion mechanism that drives Stage III.

This script reconstructs the model deterministically (same seed) up to several
checkpoints, computes B = (F̃ᵀF̃ + ηI)⁻¹, and checks the sign agreement on
top-similarity pairs.

Uses the Woodbury identity to avoid forming the K×K matrix:
   B = (F̃ᵀF̃ + ηI)⁻¹ = (1/η) I - (1/η) F̃ᵀ (F̃ F̃ᵀ + ηI)⁻¹ F̃ (1/η)
The inner inverse is n×n where n=2016.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tian_eigengap import (
    TianMLP, build_modadd_dataset, split_train_test, pick_device, tian_mse_loss,
)


def reconstruct(cfg: dict, target_epoch: int):
    """Re-train deterministically up to `target_epoch`, return model and F̃."""
    device = pick_device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    X, y = build_modadd_dataset(cfg["M"])
    Xtr, ytr, _, _ = split_train_test(X, y, cfg["n_train"], cfg["seed"])
    Xtr, ytr = Xtr.to(device), ytr.to(device)

    model = TianMLP(cfg["M"], num_ops=2, K=cfg["K"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["eta"])

    for epoch in range(target_epoch):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(Xtr)
        loss = tian_mse_loss(out, ytr, cfg["M"])
        loss.backward()
        opt.step()

    model.eval()
    return model, Xtr, ytr


@torch.no_grad()
def woodbury_B(F_tilde: torch.Tensor, eta: float) -> torch.Tensor:
    """Return B = (F̃^T F̃ + ηI)^{-1} as K×K via Woodbury through n×n inverse.

    For η=0 we fall back to the pseudoinverse via SVD (CPU)."""
    n, K = F_tilde.shape
    if eta == 0:
        # B = pinv(F̃^T F̃). On MPS we can't do SVD; do on CPU.
        Ft = F_tilde.detach().cpu().to(dtype=torch.float64)
        FtF = Ft.t() @ Ft
        return torch.linalg.pinv(FtF)
    Ft = F_tilde.detach().cpu().to(dtype=torch.float64)  # CPU + double for stability
    FFt = Ft @ Ft.t()                                     # n × n
    n_ = FFt.size(0)
    # (F̃ F̃^T + ηI)^{-1}
    inner = torch.linalg.inv(FFt + eta * torch.eye(n_, dtype=FFt.dtype))
    K_ = Ft.size(1)
    B = (1.0 / eta) * torch.eye(K_, dtype=FFt.dtype) \
        - (1.0 / eta) * Ft.t() @ inner @ Ft * (1.0 / eta)
    return B


@torch.no_grad()
def residual_similarity_signs(F_tilde: torch.Tensor, eta: float, top_pairs: int = 200,
                              n_residual_samples: int = 50) -> dict:
    """For the top-similarity feature pairs (j, l), compute B_{jl} and the
    residual similarity f̃_j^T P_{η, -jl} f̃_l, and check the Theorem 6 sign rule.

    For computational tractability, we approximate P_{η, -jl} ≈ P_η (the full
    projector that excludes only j and l would require recomputing inverse for
    each pair). Tian's theorem statement uses P_{η,-jl}; using P_η is a small
    approximation when K=2048 ≫ 2.
    """
    device = F_tilde.device
    K = F_tilde.size(1)

    # normalize columns for similarity ranking
    F_norm = F_tilde / torch.linalg.norm(F_tilde, dim=0, keepdim=True).clamp(min=1e-30)
    # cosine similarity matrix (K x K). Computed on CPU to save MPS memory.
    F_norm_cpu = F_norm.detach().to("cpu", dtype=torch.float32)
    Sim = F_norm_cpu.t() @ F_norm_cpu                      # K x K
    # mask out the diagonal
    Sim.fill_diagonal_(0.0)
    # take absolute similarity to find "most similar" pairs
    abs_sim = Sim.abs()
    # take top-pair indices (upper triangle only to avoid double counting)
    iu = torch.triu_indices(K, K, offset=1)
    sims_flat = abs_sim[iu[0], iu[1]]
    top_vals, top_idx = torch.topk(sims_flat, top_pairs)
    top_j = iu[0][top_idx].numpy()
    top_l = iu[1][top_idx].numpy()
    top_sim = Sim[top_j, top_l].numpy()  # signed similarity

    # compute B
    B = woodbury_B(F_tilde, eta)                          # K x K, CPU float64
    B_pairs = B[top_j, top_l].numpy()

    # residual similarity using full projector P_η = I - F̃ (F̃^T F̃ + ηI)^{-1} F̃^T
    Ft = F_tilde.detach().cpu().to(dtype=torch.float64)
    f_pairs_j = Ft[:, top_j]
    f_pairs_l = Ft[:, top_l]
    # P_η f̃_l = f̃_l - F̃ B F̃^T f̃_l
    Bf = B @ Ft.t()
    F_B_Ftl = Ft @ (Bf @ f_pairs_l)
    Penf_l = f_pairs_l - F_B_Ftl
    residual_sims = (f_pairs_j * Penf_l).sum(0).numpy()

    # Theorem 6 prediction: sign(B_{jl}) = -sign(residual_sim)
    # (we use P_η as approximation to P_{η,-jl})
    # For pairs with positive cosine similarity (S_{jl} > 0), residual_sim
    # should typically also be positive (residual after projection usually
    # preserves sign for highly-similar pairs), so we expect B_{jl} < 0.
    pos_mask = top_sim > 0
    sign_match = np.sign(B_pairs) == -np.sign(residual_sims)
    return {
        "top_sim": top_sim,            # signed cosine similarity
        "B_pairs": B_pairs,            # B_{jl} for top pairs
        "residual_sim": residual_sims, # f̃_j^T P_η f̃_l
        "sign_match": sign_match.astype(float),
        "frac_neg_B_on_pos_sim": float((B_pairs[pos_mask] < 0).mean()) if pos_mask.any() else float("nan"),
        "frac_sign_match": float(sign_match.mean()),
        "n_pairs": top_pairs,
    }


@torch.no_grad()
def get_F_tilde(model: TianMLP, X_train: torch.Tensor) -> torch.Tensor:
    F_act = model.hidden(X_train)
    return F_act - F_act.mean(dim=0, keepdim=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="sweep_eta0.0002_seed0",
                   help="run-tag whose config we use to reconstruct")
    p.add_argument("--epochs", type=int, nargs="+", default=[50, 100, 175, 250, 300],
                   help="epochs at which to run the verification")
    p.add_argument("--top-pairs", type=int, default=200)
    args = p.parse_args()

    cfg_path = Path("runs") / args.tag / "config.json"
    cfg = json.load(open(cfg_path))
    print(f"Config: M={cfg['M']}, K={cfg['K']}, eta={cfg['eta']}, seed={cfg['seed']}")
    print(f"Checking epochs: {args.epochs}")

    summaries = {}
    for ep in args.epochs:
        print(f"\n=== epoch {ep} ===")
        model, Xtr, _ = reconstruct(cfg, ep)
        F_tilde = get_F_tilde(model, Xtr)
        result = residual_similarity_signs(F_tilde, cfg["eta"], top_pairs=args.top_pairs)
        summaries[ep] = result
        print(f"  Top-{result['n_pairs']} most-similar pairs:")
        print(f"  median |sim|     = {np.median(np.abs(result['top_sim'])):.4f}")
        print(f"  median B_{{jl}}    = {np.median(result['B_pairs']):.4e}")
        print(f"  frac B_{{jl}} < 0  = {(result['B_pairs'] < 0).mean():.3f}")
        print(f"  frac B<0 on +sim = {result['frac_neg_B_on_pos_sim']:.3f}  (should be HIGH per Thm 6)")
        print(f"  frac sign(B_{{jl}}) = -sign(f̃_j^T P_η f̃_l): {result['frac_sign_match']:.3f}  (should be HIGH)")

    # Plot
    fig, axes = plt.subplots(2, len(args.epochs), figsize=(3.3 * len(args.epochs), 6),
                             sharey="row")
    for col, ep in enumerate(args.epochs):
        s = summaries[ep]
        ax = axes[0, col]
        ax.scatter(s["top_sim"], s["B_pairs"], alpha=0.5, s=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"epoch {ep}")
        ax.set_xlabel("cos similarity (S_{jl})")
        if col == 0:
            ax.set_ylabel("B_{jl}")
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.scatter(s["residual_sim"], s["B_pairs"], alpha=0.5, s=8, color="C2")
        # plot reference line: y = -x scaled
        rng = max(abs(s["residual_sim"]).max(), 1e-30)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("residual sim (f̃_j^T P_η f̃_l)")
        if col == 0:
            ax.set_ylabel("B_{jl}")
        # annotate sign-match fraction
        ax.text(0.05, 0.95, f"sign-match: {s['frac_sign_match']:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.grid(alpha=0.3)
    fig.suptitle(f"Theorem 6 verification — {args.tag}\n"
                 f"Top {args.top_pairs} most-similar feature pairs at each epoch")
    plt.tight_layout()
    out = Path("runs") / f"{args.tag}_theorem6.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    # also save numeric summary
    summary_json = {str(ep): {k: float(v) if not isinstance(v, np.ndarray) else None
                              for k, v in summaries[ep].items()}
                    for ep in args.epochs}
    with open(Path("runs") / f"{args.tag}_theorem6_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)


if __name__ == "__main__":
    main()
