#!/usr/bin/env python3
"""Multi-seed overlay of SED-LCH coupling.

Reads sed_lch_attn_s{42,137,2024}.pt and produces a 4-panel overlay with
both raw-step and grokking-aligned axes.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "sed_lch_results"
SEEDS = [42, 137, 2024]
COLORS = {42: "tab:blue", 137: "tab:orange", 2024: "tab:green"}


def load(seed):
    return torch.load(ROOT / f"sed_lch_attn_s{seed}.pt", map_location="cpu",
                      weights_only=False)


def grokking_step(r) -> int:
    """First step where test_acc >= 0.5."""
    ta = r["test_acc"]; ms = r["m_step"]
    idx = int(np.argmax(ta >= 0.5))
    return int(ms[idx]) if (ta >= 0.5).any() else int(ms[-1])


def main():
    runs = {s: load(s) for s in SEEDS}
    grok = {s: grokking_step(r) for s, r in runs.items()}
    print("[info] grokking steps:", grok)

    # ── raw-step overlay ───────────────────────────────────────────────
    fig, ax = plt.subplots(4, 2, figsize=(14, 12), sharey="row")

    for s, r in runs.items():
        c = COLORS[s]
        sel = r["ckpt_steps"]
        ax[0, 0].plot(r["m_step"], r["test_acc"], c=c, alpha=0.8,
                      label=f"seed={s} (grok @ {grok[s]})")
        for k in range(3):
            ax[k + 1, 0].plot(sel, r["R_k"][:, k], "-o", ms=3, c=c, alpha=0.85,
                              label=f"seed={s}")

    for k in range(3):
        ax[k + 1, 0].axhline(1.0, c="gray", ls=":", lw=0.5)
        ax[k + 1, 0].set_yscale("log")
        ax[k + 1, 0].set_ylabel(fr"$R_{k+1}$")
    ax[0, 0].set_ylabel("test acc")
    ax[0, 0].legend(loc="lower right", fontsize=8)
    ax[3, 0].set_xlabel("training step")
    ax[0, 0].set_title("Raw step")

    # ── grokking-aligned (step / grok_step) overlay ────────────────────
    for s, r in runs.items():
        c = COLORS[s]
        gs = grok[s]
        sel = r["ckpt_steps"] / gs
        ax[0, 1].plot(r["m_step"] / gs, r["test_acc"], c=c, alpha=0.8)
        for k in range(3):
            ax[k + 1, 1].plot(sel, r["R_k"][:, k], "-o", ms=3, c=c, alpha=0.85)

    for k in range(3):
        ax[k + 1, 1].axhline(1.0, c="gray", ls=":", lw=0.5)
        ax[k + 1, 1].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.5)
        ax[k + 1, 1].set_yscale("log")
    ax[0, 1].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.5)
    ax[3, 1].set_xlabel(r"step / step$_{\rm grok}$")
    ax[0, 1].set_title("Grokking-aligned (vertical line = grok onset)")

    fig.suptitle("Multi-seed SED-LCH coupling, attention subspace, grokking mod-add (p=97)")
    fig.tight_layout()

    out = ROOT / "sed_lch_multiseed.png"
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")

    # ── rank90 overlay ─────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for s, r in runs.items():
        c = COLORS[s]
        gs = grok[s]
        ax2[0].plot(r["ckpt_steps"], r["rank90"], "-o", ms=3, c=c,
                    label=f"seed={s}")
        ax2[1].plot(r["ckpt_steps"] / gs, r["rank90"], "-o", ms=3, c=c)
    ax2[0].set_xlabel("training step"); ax2[0].set_ylabel("rank-90 of $\\mu$ matrix")
    ax2[0].legend()
    ax2[1].set_xlabel(r"step / step$_{\rm grok}$")
    ax2[1].axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.5)
    fig2.suptitle("Centroid matrix rank-90 across seeds")
    fig2.tight_layout()
    out2 = ROOT / "sed_lch_multiseed_rank90.png"
    fig2.savefig(out2, dpi=130)
    print(f"[saved] {out2}")

    # ── numeric summary ────────────────────────────────────────────────
    print("\n[summary] R_k peak values per seed (attn subspace):")
    print(f"{'seed':>6} {'grok@':>6} {'R1_pk':>7} {'R2_pk':>7} {'R3_pk':>7} "
          f"{'R1_t/g':>7} {'R2_t/g':>7} {'R3_t/g':>7} {'rank90_min':>11}")
    for s, r in runs.items():
        gs = grok[s]
        Rk = r["R_k"]; ck = r["ckpt_steps"]
        peaks = Rk.max(axis=0)
        peak_t = ck[Rk.argmax(axis=0)]
        rank_min = int(r["rank90"].min())
        print(f"{s:>6} {gs:>6} "
              f"{peaks[0]:>7.2f} {peaks[1]:>7.2f} {peaks[2]:>7.2f} "
              f"{peak_t[0]/gs:>7.2f} {peak_t[1]/gs:>7.2f} {peak_t[2]/gs:>7.2f} "
              f"{rank_min:>11}")


if __name__ == "__main__":
    main()
