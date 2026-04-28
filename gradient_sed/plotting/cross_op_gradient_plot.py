#!/usr/bin/env python3
"""Cross-op comparison: gradient-SED vs update-SED, single task."""

from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "sed_lch_results"

# (op, gradient-SED tag, update-SED tag, label, color)
ROWS = [
    ("add", "add_s42", "attn_s42", "add", "tab:blue"),
    ("sub", "sub_s42", "attn_sub_s42", "sub", "tab:orange"),
    ("mul", "mul_s42", "attn_mul_s42", "mul", "tab:green"),
    ("x2_y2", "x2y2_s42", "attn_x2y2_s42", "x²+y²", "tab:red"),
]


def load_grad(tag):
    return torch.load(ROOT / f"sed_lch_grad_{tag}.pt", map_location="cpu",
                      weights_only=False)


def load_upd(tag):
    return torch.load(ROOT / f"sed_lch_{tag}.pt", map_location="cpu",
                      weights_only=False)


def grok(r):
    if "test_acc" in r:
        ta = r["test_acc"]; ms = r["m_step"]
    else:
        return None
    return int(ms[int(np.argmax(ta >= 0.5))]) if (ta >= 0.5).any() else int(ms[-1])


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # Top row: R_k mean over top-3, gradient-SED vs update-SED
    for op, gtag, utag, lbl, c in ROWS:
        rg = load_grad(gtag)
        ru = load_upd(utag)
        gs_g = grok(rg) or rg["ckpt_steps"][-1]
        gs_u = grok(ru) or ru["ckpt_steps"][-1]
        Rmean_g = rg["R_k"].mean(axis=1)
        Rmean_u = ru["R_k"].mean(axis=1)
        ax[0, 0].plot(rg["ckpt_steps"] / gs_g, Rmean_g, "-o", ms=3, c=c,
                      label=f"{lbl}", alpha=0.85)
        ax[0, 1].plot(ru["ckpt_steps"] / gs_u, Rmean_u, "-o", ms=3, c=c,
                      label=f"{lbl}", alpha=0.85)

    ax[0, 0].set_title("Gradient-SED  (per-op gradient SVD)")
    ax[0, 1].set_title("Update-SED  (rolling SVD of $\\Delta\\theta$)")
    for a in ax[0]:
        a.axhline(1.0, c="gray", ls=":", lw=0.5)
        a.axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
        a.set_yscale("log")
        a.set_xlabel(r"step / step$_{\rm grok}$")
        a.set_ylabel(r"$\overline{R_k}$ (mean of top-3)")
        a.legend()

    # Bottom row: rank-90 (should be op-invariant; sanity)
    for op, gtag, utag, lbl, c in ROWS:
        rg = load_grad(gtag)
        ru = load_upd(utag)
        gs_g = grok(rg) or rg["ckpt_steps"][-1]
        gs_u = grok(ru) or ru["ckpt_steps"][-1]
        ax[1, 0].plot(rg["ckpt_steps"] / gs_g, rg["rank90"], "-o", ms=3, c=c,
                      label=f"{lbl}", alpha=0.85)
        ax[1, 1].plot(ru["ckpt_steps"] / gs_u, ru["rank90"], "-o", ms=3, c=c,
                      label=f"{lbl}", alpha=0.85)

    for a in ax[1]:
        a.set_xlabel(r"step / step$_{\rm grok}$")
        a.set_ylabel("rank-90 of centroid matrix")
        a.axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
        a.legend()
    ax[1, 0].set_title("Centroid rank (gradient-SED runs)")
    ax[1, 1].set_title("Centroid rank (update-SED runs)")

    fig.suptitle("Gradient-SED vs Update-SED, single-task (seed 42)")
    fig.tight_layout()
    out = ROOT / "cross_op_gradient_vs_update.png"
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")

    # numeric summary
    print(f"\n{'op':>6} | "
          f"{'R_k peak (grad)':>16} {'R_k final (grad)':>17} | "
          f"{'R_k peak (upd)':>15} {'R_k final (upd)':>15}")
    for op, gtag, utag, lbl, c in ROWS:
        rg = load_grad(gtag)
        ru = load_upd(utag)
        Rg = rg["R_k"].mean(axis=1)
        Ru = ru["R_k"].mean(axis=1)
        print(f"{op:>6} | "
              f"{Rg.max():>16.2f} {Rg[-1]:>17.2f} | "
              f"{Ru.max():>15.2f} {Ru[-1]:>15.2f}")


if __name__ == "__main__":
    main()
