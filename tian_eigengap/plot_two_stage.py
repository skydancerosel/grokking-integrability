"""Two-stage spectral picture of grokking — final headline plot.

Top: cross-seed median + IQR for the LEADING indicator (fft_dist_from_ideal),
the LAGGING indicator (σ₂/σ₃ on rolling ΔW), and test_acc.

Bottom: fire-time distribution showing
  - fft_dist ≥ 0.075 fires at ep ~17 (LEADS test_acc by ~80 epochs)
  - test_acc ≥ 0.5 at ep ~102
  - σ₂/σ₃ slope on ΔW fires at ep ~174 (LAGS test_acc by ~70 epochs)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = Path(__file__).parent / "runs"


def load(eta_str, prefix="sweep_eta", min_rows=401):
    runs = []
    for d in sorted(RUNS.glob(f"{prefix}{eta_str}_seed*")):
        if not d.is_dir(): continue
        p = d / "log.jsonl"
        if not p.exists(): continue
        with open(p) as f:
            n = sum(1 for _ in f)
        if n < min_rows: continue
        rows = [json.loads(l) for l in open(p)]
        runs.append({k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
                     for k in {kk for r in rows for kk in r.keys()}})
    return runs


def main():
    grok = load("0.0002")
    ctrl = load("0")
    print(f"loaded {len(grok)} grok, {len(ctrl)} control runs")
    if not grok or not ctrl:
        return
    ep = grok[0]["epoch"]

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.4, 1.0], hspace=0.4, wspace=0.3)

    # Row 1: test acc + 80-ep lead annotation
    ax = fig.add_subplot(gs[0, :])
    for runs, color, lbl in [(grok, "C0", "η=0.0002"), (ctrl, "C1", "η=0")]:
        arr = np.stack([r["test_acc"] for r in runs], 0)
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(ep, mu, color=color, lw=1.8, label=lbl)
        ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18)
    ax.set_ylabel("test accuracy")
    ax.set_title("(A) Test accuracy")
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.05)

    # Row 2 left: leading (fft_dist) — linear scale
    ax = fig.add_subplot(gs[1, 0])
    for runs, color, lbl in [(grok, "C0", "η=0.0002"), (ctrl, "C1", "η=0")]:
        arr = np.stack([r["fft_dist_from_ideal"] for r in runs], 0)
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(ep, mu, color=color, lw=1.8, label=lbl)
        ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18)
    ax.axhline(0.075, color="k", ls="--", lw=0.8, label="thr = 0.075")
    ax.set_ylabel("fft_dist_from_ideal")
    ax.set_xlabel("epoch")
    ax.set_title("(B) LEADING — Tian off-diagonal magnitude")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Row 2 right: lagging (σ₂/σ₃ on ΔW) — log scale
    ax = fig.add_subplot(gs[1, 1])
    for runs, color, lbl in [(grok, "C0", "η=0.0002"), (ctrl, "C1", "η=0")]:
        arr = np.stack([r["W_gap23"] for r in runs], 0)
        mu = np.nanmedian(arr, axis=0)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        ax.plot(ep, mu, color=color, lw=1.8, label=lbl)
        ax.fill_between(ep, q1, q3, color=color, alpha=0.18)
    ax.axhline(50.0, color="k", ls="--", lw=0.8, label="σ₂/σ₃ = 50")
    ax.set_yscale("log")
    ax.set_ylabel("σ₂/σ₃ on rolling ΔW")
    ax.set_xlabel("epoch")
    ax.set_title("(C) LAGGING — rolling ΔW Gram eigengap")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # Row 3: fire-time distributions
    ax = fig.add_subplot(gs[2, :])
    fire_lead = []
    fire_test = []
    fire_lock = []
    for r in grok:
        x = r["fft_dist_from_ideal"]
        c = np.where(x >= 0.075)[0]
        fire_lead.append(int(c[0]) if len(c) else float("nan"))
        m = r["test_acc"] >= 0.5
        fire_test.append(int(ep[m][0]) if m.any() else float("nan"))
        # σ₂/σ₃ slope-fire at ep ≥ 100
        x = r["W_gap23"]
        sl = np.full_like(x, np.nan)
        sl[25:] = (np.log(np.clip(x[25:], 1e-6, None)) - np.log(np.clip(x[:-25], 1e-6, None))) / 25
        m = (sl > 0.04) & (np.arange(len(x)) >= 100) & (~np.isnan(sl))
        idx = np.where(m)[0]
        fire_lock.append(int(idx[0]) if len(idx) else float("nan"))
    fire_lead = np.array(fire_lead, dtype=float)
    fire_test = np.array(fire_test, dtype=float)
    fire_lock = np.array(fire_lock, dtype=float)

    positions = [1, 2, 3]
    data = [fire_lead, fire_test, fire_lock]
    labels = [f"fft_dist ≥ 0.075\n(LEADING, median {np.nanmedian(fire_lead):.0f})",
              f"test_acc ≥ 0.5\n(median {np.nanmedian(fire_test):.0f})",
              f"σ₂/σ₃ slope ≥ 0.04\n(LAGGING, median {np.nanmedian(fire_lock):.0f})"]
    bp = ax.boxplot(data, positions=positions, widths=0.5, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["C0", "k", "C3"]):
        patch.set_facecolor(color); patch.set_alpha(0.4)
    for x_pos, d, color in zip(positions, data, ["C0", "k", "C3"]):
        ax.scatter([x_pos] * len(d), d, color=color, alpha=0.7, s=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("epoch")
    ax.set_title(f"(D) Fire-time distribution across n={len(grok)} grok seeds — leading indicator precedes test by ~{np.nanmedian(fire_test - fire_lead):.0f} epochs")
    ax.grid(alpha=0.3, axis="y")

    plt.savefig(RUNS / "two_stage_headline.png", dpi=120, bbox_inches="tight")
    print(f"saved {RUNS / 'two_stage_headline.png'}")
    plt.close()

    print(f"\nLEADING: fft_dist ≥ 0.075 fires at ep {np.nanmedian(fire_lead):.0f} (IQR [{np.nanpercentile(fire_lead,25):.0f}, {np.nanpercentile(fire_lead,75):.0f}])")
    print(f"MIDDLE:  test_acc ≥ 0.5 at  ep {np.nanmedian(fire_test):.0f} (IQR [{np.nanpercentile(fire_test,25):.0f}, {np.nanpercentile(fire_test,75):.0f}])")
    print(f"LAGGING: σ₂/σ₃ slope at     ep {np.nanmedian(fire_lock):.0f} (IQR [{np.nanpercentile(fire_lock,25):.0f}, {np.nanpercentile(fire_lock,75):.0f}])")
    print(f"\nLeading-indicator lead time: {np.nanmedian(fire_test - fire_lead):.0f} ep (IQR [{np.nanpercentile(fire_test-fire_lead,25):.0f}, {np.nanpercentile(fire_test-fire_lead,75):.0f}])")
    print(f"Lagging-indicator lag time:  {np.nanmedian(fire_lock - fire_test):.0f} ep (IQR [{np.nanpercentile(fire_lock-fire_test,25):.0f}, {np.nanpercentile(fire_lock-fire_test,75):.0f}])")


if __name__ == "__main__":
    main()
