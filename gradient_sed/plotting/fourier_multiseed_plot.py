#!/usr/bin/env python3
"""Multi-seed overlay of Fourier readout on centroid PCs."""

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
    return torch.load(ROOT / f"fourier_s{seed}.pt", map_location="cpu",
                      weights_only=False)


def grok(r):
    ta = r["test_acc"]; ms = r["m_step"]
    return int(ms[int(np.argmax(ta >= 0.5))]) if (ta >= 0.5).any() else int(ms[-1])


def main():
    runs = {s: load(s) for s in SEEDS}
    g = {s: grok(r) for s, r in runs.items()}
    print("[info] grokking steps:", g)

    # ── full-basis R^2 trajectories: aligned by step / grok_step ────────
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    for s, r in runs.items():
        c = COLORS[s]
        sel = r["ckpt_steps"] / g[s]
        ax[0, 0].plot(sel, r["R2_full_sum"][:, :3].mean(axis=1), "-o", ms=3,
                      c=c, label=f"seed={s}")
        ax[0, 1].plot(sel, r["R2_full_a_b"][:, :3].mean(axis=1), "-o", ms=3,
                      c=c, label=f"seed={s}")
        # PC1 best single-omega R^2(a+b)
        best_ab = r["R2_sum_o"][:, :, 0].max(axis=1)
        ax[1, 0].plot(sel, best_ab, "-o", ms=3, c=c, label=f"seed={s}")
        # rank of centroid manifold (top-3 var share)
        top3 = r["pc_var_share"][:, :3].sum(axis=1)
        ax[1, 1].plot(sel, top3, "-o", ms=3, c=c, label=f"seed={s}")

    for a in ax.flatten():
        a.axvline(1.0, c="black", ls="--", lw=0.5, alpha=0.6)
        a.legend(fontsize=9)
    ax[0, 0].set_title(r"full-basis $R^2$, (a+b)-Fourier  (mean top-3 PCs)")
    ax[0, 0].set_ylabel(r"$R^2$"); ax[0, 0].set_ylim(-0.02, 1.02)
    ax[0, 1].set_title(r"full-basis $R^2$, a&b-Fourier  (mean top-3 PCs)")
    ax[0, 1].set_ylim(-0.02, 1.02)
    ax[1, 0].set_title(r"PC1 best single-$\omega$ $R^2$ in (a+b)")
    ax[1, 0].set_ylabel(r"$R^2$"); ax[1, 0].set_ylim(-0.02, 0.55)
    ax[1, 1].set_title(r"top-3 PC variance share")
    ax[1, 1].set_ylim(0, 0.5)
    ax[1, 0].set_xlabel(r"step / step$_{\rm grok}$")
    ax[1, 1].set_xlabel(r"step / step$_{\rm grok}$")

    fig.suptitle("Multi-seed Fourier readout on LCH centroid PCs (grokking mod-add)")
    fig.tight_layout()
    out = ROOT / "fourier_multiseed.png"
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")

    # ── PC1 (step, ω) heatmaps side-by-side ──────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax_, (s, r) in zip(axes, runs.items()):
        sel = r["ckpt_steps"]
        omegas = r["omegas"]
        Z = r["R2_sum_o"][:, :, 0].T   # (omega, step)
        im = ax_.imshow(Z, aspect="auto", origin="lower",
                        extent=[sel[0], sel[-1], omegas[0], omegas[-1]],
                        cmap="viridis", vmin=0, vmax=Z.max())
        ax_.axvline(g[s], c="white", ls="--", lw=1, alpha=0.7)
        ax_.set_title(f"seed={s}, grok @ {g[s]}")
        ax_.set_xlabel("step")
        plt.colorbar(im, ax=ax_, label=r"$R^2$")
    axes[0].set_ylabel(r"$\omega$")
    fig2.suptitle(r"PC1 single-$\omega$ $R^2$ on (a+b) basis (white line = grokking)")
    fig2.tight_layout()
    out2 = ROOT / "fourier_multiseed_pc1_heatmap.png"
    fig2.savefig(out2, dpi=130)
    print(f"[saved] {out2}")

    # ── numeric summary ────────────────────────────────────────────────
    print("\n[summary] Fourier readout key checkpoints (mean over top-3 PCs):")
    print(f"{'seed':>6} {'step/grok':>10}  {'R²(a+b)':>9} {'R²(a&b)':>9} "
          f"{'PC1 best ω':>11} {'PC1 best R²':>12}")
    for s, r in runs.items():
        gs = g[s]
        sel = r["ckpt_steps"]
        for label, j in [("init", 0),
                         ("pre-half", len(sel)//4),
                         ("pre-grok", int(np.searchsorted(sel, gs * 0.85))),
                         ("at-grok", int(np.searchsorted(sel, gs * 1.0))),
                         ("post-grok", len(sel)-1)]:
            j = min(j, len(sel)-1)
            t = sel[j]
            ab = r["R2_full_sum"][j, :3].mean()
            a_b = r["R2_full_a_b"][j, :3].mean()
            best_w = r["omegas"][r["R2_sum_o"][j, :, 0].argmax()]
            best_r = r["R2_sum_o"][j, :, 0].max()
            print(f"{s:>6} {label:>10}  {ab:>9.3f} {a_b:>9.3f} "
                  f"{best_w:>11d} {best_r:>12.3f}")
        print()


if __name__ == "__main__":
    main()
