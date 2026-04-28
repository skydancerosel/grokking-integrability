#!/usr/bin/env python3
"""Side-by-side comparison of full-theta vs attention-only SED-LCH coupling.

Reads sed_lch_full.pt and sed_lch_attn.pt produced by sed_lch_coupling.py and
emits a single overlay figure for direct comparison.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "sed_lch_results"
OUT = ROOT / "sed_lch_compare.png"


def load(tag):
    return torch.load(ROOT / f"sed_lch_{tag}.pt", map_location="cpu",
                      weights_only=False)


def main():
    rf = load("full")
    ra = load("attn")
    sel_f = rf["ckpt_steps"]; sel_a = ra["ckpt_steps"]
    m_step = rf["m_step"]; train = rf["train_acc"]; test = rf["test_acc"]

    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(m_step, train, label="train", c="tab:blue")
    ax[0].plot(m_step, test, label="test", c="tab:orange")
    ax[0].set_ylabel("accuracy")
    ax[0].legend(loc="lower right")

    for tag, r, ls in [("full", rf, "-"), ("attn", ra, "--")]:
        ax[1].plot(r["win_centers"], r["g23"], ls=ls,
                   label=fr"{tag}: $\sigma_2/\sigma_3$",
                   c="tab:purple" if tag == "full" else "tab:red")
    ax[1].axhline(1.0, c="gray", ls=":", lw=0.5)
    ax[1].set_ylabel(r"edge gap $\sigma_2/\sigma_3$")
    ax[1].legend()

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for k in range(3):
        ax[2].plot(sel_f, rf["R_k"][:, k], ls="-", marker="o", ms=3,
                   c=colors[k], label=fr"full $R_{k+1}$")
        ax[2].plot(sel_a, ra["R_k"][:, k], ls="--", marker="x", ms=4,
                   c=colors[k], label=fr"attn $R_{k+1}$")
    ax[2].axhline(1.0, c="gray", ls=":", lw=0.5)
    ax[2].set_ylabel(r"$R_k = A_k / A_{\rm rand}$")
    ax[2].set_yscale("log")
    ax[2].legend(ncol=3, fontsize=8)

    ax[3].plot(sel_f, rf["rank90"], "-o", ms=3, c="tab:green",
               label="rank90 (μ matrix)")
    ax[3].set_ylabel("rank-90 of centroid matrix")
    ax[3].set_xlabel("training step")
    ax[3].legend()

    fig.suptitle("SED-LCH coupling: full-θ vs attention-only "
                 "(p=97 mod-add, seed=42)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
