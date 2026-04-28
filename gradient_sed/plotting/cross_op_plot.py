#!/usr/bin/env python3
"""Cross-op comparison of SED-LCH coupling and Fourier readout.

Reads sed_lch_attn_<op>_s42.pt and fourier_<op>_s42.pt for op in
{add, sub, mul, x2_y2}. Produces overlay plots aligned by step / step_grok.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "sed_lch_results"
OPS = [
    ("add", "attn_s42", "s42", "tab:blue"),
    ("sub", "attn_sub_s42", "sub_s42", "tab:orange"),
    ("mul", "attn_mul_s42", "mul_s42", "tab:green"),
    ("x2_y2", "attn_x2y2_s42", "x2y2_s42", "tab:red"),
]


def grok_step(r):
    if "test_acc" in r and "m_step" in r:
        ta = r["test_acc"]; ms = r["m_step"]
    else:
        return None
    return int(ms[int(np.argmax(ta >= 0.5))]) if (ta >= 0.5).any() else int(ms[-1])


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # SED-LCH R_k
    for op, sed_tag, fou_tag, c in OPS:
        sed_path = ROOT / f"sed_lch_{sed_tag}.pt"
        if not sed_path.exists():
            print(f"[skip] {sed_path.name} missing")
            continue
        r = torch.load(sed_path, map_location="cpu", weights_only=False)
        gs = grok_step(r) or r["ckpt_steps"][-1]
        sel = r["ckpt_steps"] / gs
        # mean of top-3 R_k
        Rmean = r["R_k"].mean(axis=1)
        ax[0, 0].plot(sel, Rmean, "-o", ms=3, c=c, label=op, alpha=0.85)
        ax[0, 1].plot(sel, r["rank90"], "-o", ms=3, c=c, label=op, alpha=0.85)

    ax[0, 0].set_xlabel(r"step / step$_{\rm grok}$")
    ax[0, 0].set_ylabel(r"$\overline{R_k}$ (mean of top-3)")
    ax[0, 0].axhline(1.0, c="gray", ls=":", lw=0.5)
    ax[0, 0].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("SED-LCH coupling per op")
    ax[0, 0].legend()

    ax[0, 1].set_xlabel(r"step / step$_{\rm grok}$")
    ax[0, 1].set_ylabel(r"rank-90 of centroid matrix")
    ax[0, 1].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
    ax[0, 1].set_title("Centroid manifold rank per op")
    ax[0, 1].legend()

    # Fourier readouts
    for op, sed_tag, fou_tag, c in OPS:
        fou_path = ROOT / f"fourier_{fou_tag}.pt"
        if not fou_path.exists():
            print(f"[skip] {fou_path.name} missing")
            continue
        r = torch.load(fou_path, map_location="cpu", weights_only=False)
        gs = grok_step(r) or r["ckpt_steps"][-1]
        sel = r["ckpt_steps"] / gs
        # full-basis y-Fourier R^2 (the "answer" basis, op-specific)
        y_R2 = r["R2_full_y"][:, :3].mean(axis=1) if "R2_full_y" in r else None
        ab_R2 = r["R2_full_a_b"][:, :3].mean(axis=1)
        if y_R2 is not None:
            ax[1, 0].plot(sel, y_R2, "-o", ms=3, c=c, label=f"{op}: y", alpha=0.85)
        ax[1, 1].plot(sel, ab_R2, "-o", ms=3, c=c, label=f"{op}: a&b", alpha=0.85)

    ax[1, 0].set_xlabel(r"step / step$_{\rm grok}$")
    ax[1, 0].set_ylabel(r"$R^2$ on y=op(a,b) Fourier")
    ax[1, 0].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
    ax[1, 0].set_ylim(-0.02, 1.02)
    ax[1, 0].set_title('"Answer" Fourier of centroid PCs')
    ax[1, 0].legend()

    ax[1, 1].set_xlabel(r"step / step$_{\rm grok}$")
    ax[1, 1].set_ylabel(r"$R^2$ on (a,b) individual Fourier")
    ax[1, 1].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
    ax[1, 1].set_ylim(-0.02, 1.02)
    ax[1, 1].set_title("Individual (a,b) Fourier of centroid PCs")
    ax[1, 1].legend()

    fig.suptitle("Cross-op SED-LCH and centroid Fourier structure (single-task, seed=42)")
    fig.tight_layout()
    out = ROOT / "cross_op_summary.png"
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")

    # Numeric table
    print("\n[table] R_k peak and final, per op:")
    print(f"{'op':>8} {'R_k peak':>10} {'R_k final':>10} {'rank90 init':>12} "
          f"{'rank90 final':>13} {'y-R² final':>11} {'a&b R² final':>13}")
    for op, sed_tag, fou_tag, c in OPS:
        sed_path = ROOT / f"sed_lch_{sed_tag}.pt"
        fou_path = ROOT / f"fourier_{fou_tag}.pt"
        if not sed_path.exists():
            continue
        r = torch.load(sed_path, map_location="cpu", weights_only=False)
        Rmean = r["R_k"].mean(axis=1)
        rank90 = r["rank90"]
        if fou_path.exists():
            f = torch.load(fou_path, map_location="cpu", weights_only=False)
            yR2 = f["R2_full_y"][-1, :3].mean() if "R2_full_y" in f else float("nan")
            abR2 = f["R2_full_a_b"][-1, :3].mean()
        else:
            yR2 = float("nan"); abR2 = float("nan")
        print(f"{op:>8} {Rmean.max():>10.2f} {Rmean[-1]:>10.2f} "
              f"{rank90[0]:>12d} {rank90[-1]:>13d} "
              f"{yR2:>11.3f} {abR2:>13.3f}")


if __name__ == "__main__":
    main()
