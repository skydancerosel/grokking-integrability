#!/usr/bin/env python3
"""Compare rolling-window vs expanding-window SED-LCH coupling across seeds."""

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "sed_lch_results"
SEEDS = [42, 137, 2024]
COLORS = {42: "tab:blue", 137: "tab:orange", 2024: "tab:green"}


def load(tag):
    return torch.load(ROOT / f"sed_lch_{tag}.pt", map_location="cpu",
                      weights_only=False)


def grok(r):
    ta = r["test_acc"]; ms = r["m_step"]
    return int(ms[int(np.argmax(ta >= 0.5))]) if (ta >= 0.5).any() else int(ms[-1])


def main():
    fig, ax = plt.subplots(3, 2, figsize=(13, 10), sharex="col")

    for col, mode in enumerate(["rolling", "expanding"]):
        suffix = "" if mode == "rolling" else "_exp"
        for s in SEEDS:
            r = load(f"attn{suffix}_s{s}")
            c = COLORS[s]
            gs = grok(r)
            sel = r["ckpt_steps"] / gs
            wc = r["win_centers"] / gs
            sigma = r["sigma"]
            g23_arr = sigma[:, 1] / np.maximum(sigma[:, 2], 1e-30)
            ax[0, col].plot(wc, g23_arr, c=c, alpha=0.85,
                            label=f"seed={s}")
            for k_idx, ax_idx in enumerate([1, 2]):
                ax[ax_idx, col].plot(sel, r["R_k"][:, k_idx], "-o", ms=3,
                                     c=c, alpha=0.85, label=f"seed={s}")

        ax[0, col].set_title(f"{mode}-window SVD")
        ax[0, col].axhline(1.0, c="gray", ls=":", lw=0.5)
        ax[0, col].axhline(2.0, c="gray", ls=":", lw=0.5)
        ax[0, col].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
        ax[0, col].set_ylabel(r"$\sigma_2/\sigma_3$")
        ax[1, col].axhline(1.0, c="gray", ls=":", lw=0.5)
        ax[1, col].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
        ax[1, col].set_ylabel(r"$R_1$")
        ax[1, col].set_yscale("log")
        ax[2, col].axhline(1.0, c="gray", ls=":", lw=0.5)
        ax[2, col].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
        ax[2, col].set_ylabel(r"$R_2$")
        ax[2, col].set_yscale("log")
        ax[2, col].set_xlabel(r"step / step$_{\rm grok}$")

    ax[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Rolling vs expanding SED-LCH coupling, attention subspace, "
                 "across 3 seeds")
    fig.tight_layout()
    out = ROOT / "sed_lch_rolling_vs_expanding.png"
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")

    # numeric summary
    print(f"\n{'tag':>22} {'g23 max':>9} {'g23 mean':>10} {'R1 max':>8} "
          f"{'R2 max':>8} {'R3 max':>8}")
    for mode in ["rolling", "expanding"]:
        suffix = "" if mode == "rolling" else "_exp"
        for s in SEEDS:
            r = load(f"attn{suffix}_s{s}")
            sigma = r["sigma"]
            g23 = sigma[:, 1] / np.maximum(sigma[:, 2], 1e-30)
            tag = f"{mode}_s{s}"
            print(f"{tag:>22} {g23.max():>9.2f} {g23.mean():>10.2f} "
                  f"{r['R_k'][:, 0].max():>8.2f} {r['R_k'][:, 1].max():>8.2f} "
                  f"{r['R_k'][:, 2].max():>8.2f}")


if __name__ == "__main__":
    main()
