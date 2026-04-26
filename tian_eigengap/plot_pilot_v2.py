"""4-row overlay for the pilot run with new instrumentation.

Rows:
  A. train/test accuracy
  B. P_1⊥ F F^T dist-from-ideal (existing Tian metric)
  C. σ₂/σ₃ of static F̃ᵀF̃ (top-3 eigvals via SVD on F_zm)
  D. σ₂/σ₃ of rolling-window deltas of F̃ᵀF̃ (new)
  E. σ₁/σ₂ on rolling-window ΔW (the σ₁/σ₂ candidate)
  F. independence proxy: median |cos(g_j, g_{j'})|
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = Path(__file__).parent / "runs"


def load(tag):
    rows = [json.loads(l) for l in open(RUNS / tag / "log.jsonl")]
    keys = sorted({k for r in rows for k in r.keys()})
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def main():
    grok = load("pilot_v2_eta0.0002_seed0")
    ctrl = load("pilot_v2_eta0_seed0")
    print(f"grok rows: {len(grok['epoch'])}, ctrl rows: {len(ctrl['epoch'])}")

    fig, axes = plt.subplots(6, 1, figsize=(9, 14), sharex=True)
    ep_g = grok["epoch"]
    ep_c = ctrl["epoch"]
    palette = [("C0", "η=0.0002 (grok)"), ("C1", "η=0 (control)")]

    # A. accuracy
    ax = axes[0]
    for d, ep, (col, lbl) in [(grok, ep_g, palette[0]), (ctrl, ep_c, palette[1])]:
        ax.plot(ep, d["train_acc"], color=col, lw=1.0, ls="--", alpha=0.6)
        ax.plot(ep, d["test_acc"], color=col, lw=1.8, label=lbl)
    ax.set_ylabel("accuracy")
    ax.set_title("(A) Train (dashed) / test (solid) accuracy")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.05)

    # B. fft_dist_from_ideal
    ax = axes[1]
    for d, ep, (col, lbl) in [(grok, ep_g, palette[0]), (ctrl, ep_c, palette[1])]:
        ax.plot(ep, d["fft_dist_from_ideal"], color=col, lw=1.5, label=lbl)
    ax.set_ylabel("‖P₁⊥FF^T − ideal‖ / ‖FF^T‖")
    ax.set_title("(B) Tian off-diagonal metric (existing — leads in pilot)")
    ax.legend(); ax.grid(alpha=0.3)

    # C. static F̃ᵀF̃ top-3 eigvals: σ₂/σ₃
    ax = axes[2]
    for d, ep, (col, lbl) in [(grok, ep_g, palette[0]), (ctrl, ep_c, palette[1])]:
        ax.plot(ep, d["ftf_eig_gap23"], color=col, lw=1.5, label=lbl)
    ax.set_ylabel("σ₂/σ₃ of F̃ᵀF̃ (static)")
    ax.set_title("(C) Static F̃ᵀF̃ top-3 — flat (≈1) in both conditions")
    ax.set_ylim(0.95, max(2.0, np.nanmax(np.concatenate([grok["ftf_eig_gap23"], ctrl["ftf_eig_gap23"]])) * 1.1))
    ax.legend(); ax.grid(alpha=0.3)

    # D. rolling-delta F̃ᵀF̃ σ₂/σ₃
    ax = axes[3]
    for d, ep, (col, lbl) in [(grok, ep_g, palette[0]), (ctrl, ep_c, palette[1])]:
        ax.plot(ep, d["FTFd_gap23"], color=col, lw=1.5, label=lbl)
    ax.set_ylabel("σ₂/σ₃ of (Δ F̃ᵀF̃)")
    ax.set_yscale("log")
    ax.set_title("(D) Rolling-window σ₂/σ₃ on F̃ᵀF̃ deltas (NEW)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # E. σ₁/σ₂ on ΔW
    ax = axes[4]
    for d, ep, (col, lbl) in [(grok, ep_g, palette[0]), (ctrl, ep_c, palette[1])]:
        s12 = d["W_sigma1"] / np.maximum(d["W_sigma2"], 1e-30)
        ax.plot(ep, s12, color=col, lw=1.5, label=lbl)
    ax.set_ylabel("σ₁/σ₂ of ΔW")
    ax.set_yscale("log")
    ax.set_title("(E) σ₁/σ₂ on rolling ΔW (Stage I→II candidate — does not lead in pilot)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # F. independence proxy
    ax = axes[5]
    for d, ep, (col, lbl) in [(grok, ep_g, palette[0]), (ctrl, ep_c, palette[1])]:
        ax.plot(ep, d["indep_cos_med"], color=col, lw=1.5, label=lbl)
    ax.set_ylabel("median |cos(g_j, g_{j'})|")
    ax.set_xlabel("epoch")
    ax.set_title("(F) Independence proxy — stays low in grok, rises to ~1 in control")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RUNS / "pilot_v2_panels.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
