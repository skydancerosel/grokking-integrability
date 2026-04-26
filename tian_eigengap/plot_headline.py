"""Polished 2-panel headline overlay across all seeds.

Top panel: Tian metric (fft_dist_from_ideal) — rises during Stage II as features form.
Bottom panel: σ₂/σ₃ rolling-window eigengap on W parameter updates.

For both η=0.0002 (grok, blue) and η=0 (control, orange) overlaid on each panel.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = Path(__file__).parent / "runs"


def load_runs(eta_str: str, min_rows: int = 401):
    runs = []
    for d in sorted(RUNS_DIR.glob(f"sweep_eta{eta_str}_seed*")):
        p = d / "log.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            n = sum(1 for _ in f)
        if n < min_rows:
            continue
        rows = [json.loads(line) for line in open(p)]
        keys = sorted({k for r in rows for k in r.keys()})
        runs.append({k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys})
    return runs


def main():
    grok = load_runs("0.0002")
    control = load_runs("0")
    if not grok or not control:
        print("not enough runs yet")
        return
    print(f"grok: {len(grok)} seeds, control: {len(control)} seeds")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True)
    ep = grok[0]["epoch"]

    # Panel A: test accuracy (sanity)
    for runs, color, lbl in [(grok, "C0", "η=0.0002"), (control, "C1", "η=0")]:
        arr = np.stack([r["test_acc"] for r in runs], 0)
        mu, sd = arr.mean(0), arr.std(0)
        ax1.plot(ep, mu, color=color, lw=1.8, label=lbl)
        ax1.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18)
    ax1.set_ylabel("test accuracy")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="center right")
    ax1.grid(alpha=0.3)
    ax1.set_title("(A) Test accuracy")

    # Panel B: Tian metric — fft_dist_from_ideal
    for runs, color, lbl in [(grok, "C0", "η=0.0002"), (control, "C1", "η=0")]:
        arr = np.stack([r["fft_dist_from_ideal"] for r in runs], 0)
        mu, sd = arr.mean(0), arr.std(0)
        ax2.plot(ep, mu, color=color, lw=1.8, label=lbl)
        ax2.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18)
    ax2.set_ylabel("‖P₁⊥FF^T - ideal‖ / ‖FF^T‖")
    ax2.legend(loc="center right")
    ax2.grid(alpha=0.3)
    ax2.set_title("(B) Tian metric — distance of FF^T from (a·I + b·11^T) form")

    # Panel C: σ₂/σ₃ on W rolling window
    for runs, color, lbl in [(grok, "C0", "η=0.0002"), (control, "C1", "η=0")]:
        arr = np.stack([r["W_gap23"] for r in runs], 0)
        mu = np.nanmedian(arr, axis=0)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        ax3.plot(ep, mu, color=color, lw=1.8, label=lbl)
        ax3.fill_between(ep, q1, q3, color=color, alpha=0.18)
    ax3.set_yscale("log")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("σ₂/σ₃   (W rolling, W=20)")
    ax3.legend(loc="upper left")
    ax3.grid(alpha=0.3, which="both")
    ax3.set_title("(C) Rolling-window eigengap σ₂/σ₃ on parameter updates ΔW")

    plt.tight_layout()
    out = RUNS_DIR / "headline_overlay.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
