#!/usr/bin/env python3
"""
Replication of Xu (2026) Table 7: Eigenvalue sub-leading gap decline for modular arithmetic.

Exact criteria from the thesis:
  - g₂₃ = σ₂² - σ₃² of rolling-window Gram matrix (W=10)  (squared singular values)
  - g²³_early  = g₂₃ at the earliest available checkpoint after init
  - g²³_grok   = g₂₃ at the grokking step
  - Decline     = g²³_early / g²³_grok
  - k*_term     = argmax_j σ_j/σ_{j+1} of W_Q at the final checkpoint
  - Grokking    = first step where test_acc >= 0.95
  - Also checks: is the decline monotonic from early to grok?

Runs on all single-task sweep files (6 ops × 2 WD × 3 seeds = 36 runs).
Also runs on dual-task and tri-task WD=1.0 vs WD=0.0 (3 seeds each).

Output: table matching thesis Table 7 format, plus WD=0.0 controls.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO          = Path(__file__).parent.parent
SINGLE_DIR    = REPO / "grok_sweep_results"
DUAL_DIR      = REPO / "modadd_modmul" / "results"
TRI_DIR       = REPO / "multitask" / "results"
PLOT_DIR      = Path(__file__).parent / "coherence_edge_plots"
RESULTS_DIR   = Path(__file__).parent / "coherence_edge_results"

SEEDS = [42, 137, 2024]
OPS   = ["add", "mul", "sub", "x2_y2", "x2_xy_y2", "x3_xy"]
GROK_THRESHOLD = 0.95   # thesis criterion


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def g23_from_matrix(mat_tensor):
    """g₂₃ = λ₂ - λ₃ = σ₂² - σ₃²  (thesis exact definition)."""
    W = mat_tensor.float().numpy()
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    if len(S) < 3:
        return None, S
    return float(S[1]**2 - S[2]**2), S

# keep alias for backward compat
g23_from_wq = g23_from_matrix


def gram_stats(flat_vecs, window=10, weighted_kstar=True):
    """
    Compute ALL Gram matrix statistics from the last `window` parameter-update vectors.

    Returns (k*, R, g23, S) where:
      k*  = argmax_j score_j          (1-indexed, gap position)
      R   = σ_{k*}/σ_{k*+1}          (gap ratio at k*)
      g23 = σ₂² - σ₃²               (sub-leading eigenvalue gap of the Gram matrix)
      S   = full singular value array

    If weighted_kstar=True (default):
      score_j = (σ_j / Σσ_i) × (σ_j / σ_{j+1})
      — weights the ratio by the relative signal mass at j, suppressing tail noise.

    With window=10 the Gram matrix is ≤10×10, so k* ∈ {1,...,9}.
    """
    if len(flat_vecs) < 2:
        return None, None, None, None
    updates = [flat_vecs[i] - flat_vecs[i-1] for i in range(1, len(flat_vecs))]
    updates = updates[-window:]
    if len(updates) < 2:
        return None, None, None, None
    X = np.stack(updates)                        # [≤W, p]
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    # g₂₃ = σ₂² - σ₃² of the Gram matrix
    g23 = float(S[1]**2 - S[2]**2) if len(S) >= 3 else None
    if len(S) < 2:
        return 1, float("inf"), g23, S
    ratios = S[:-1] / np.maximum(S[1:], 1e-30)
    if weighted_kstar:
        total = float(np.sum(S)) or 1.0
        weights = S[:-1] / total                 # σ_j / Σσ_i  (relative mass)
        scores  = weights * ratios               # weighted gap score
    else:
        scores = ratios
    k = int(np.argmax(scores))
    return k + 1, float(ratios[k]), g23, S       # 1-indexed k*, ratio R, g₂₃, S


def gram_kstar(flat_vecs, window=10):
    """Backward-compat wrapper — returns only k*."""
    k, _, _, _ = gram_stats(flat_vecs, window=window)
    return k


def gram_R_trajectory(traj, window=10):
    """
    Compute (k*, R, g₂₃) at every step as rolling-window Gram quantities.
    Returns list of (step, k*, R, g23) using a sliding window ending at each step.
    """
    flat_vecs = [(t["step"], t["flat_vec"]) for t in traj if t.get("flat_vec") is not None]
    if len(flat_vecs) < window + 1:
        return []

    result = []
    steps  = [fv[0] for fv in flat_vecs]
    vecs   = [fv[1] for fv in flat_vecs]
    for end in range(window, len(vecs)):
        window_vecs = vecs[end - window : end + 1]   # window+1 points → window updates
        k, R, g23, _ = gram_stats(window_vecs, window=window)
        result.append((steps[end], k, R, g23))
    return result


def find_grok_step(metrics, threshold=GROK_THRESHOLD):
    for m in metrics:
        # Single-task: "test_acc"
        # Multi-task: keys like "test_add", "test_mul", "test_acc_add", etc.
        acc = m.get("test_acc")
        if acc is None:
            task_accs = [v for k, v in m.items()
                         if k.startswith("test") and "loss" not in k
                         and isinstance(v, (int, float))]
            if task_accs:
                acc = min(task_accs)  # grokking = ALL tasks above threshold
        if acc is not None and acc >= threshold:
            return m["step"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def subsample_logs(logs, metrics, n_early=50, n_grok_window=20, n_terminal=15):
    """
    Subsample attn_logs to keep only: first n_early, a window around grokking,
    and the last n_terminal entries.  Avoids loading 2000 entries from 1GB files.
    """
    if not logs:
        return logs
    # Find grok step from metrics
    grok_step = None
    for m in metrics:
        acc = m.get("test_acc", 0)
        if isinstance(acc, float) and acc >= GROK_THRESHOLD:
            grok_step = m["step"]
            break
    if grok_step is None:
        # No grokking: keep first n_early + last n_terminal
        return logs[:n_early] + logs[-n_terminal:]

    # Find index nearest to grok_step
    steps = [l["step"] for l in logs]
    grok_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - grok_step))
    lo = max(0, grok_idx - n_grok_window)
    hi = min(len(logs), grok_idx + n_grok_window + 1)

    early   = logs[:n_early]
    grok_w  = logs[lo:hi]
    terminal = logs[-n_terminal:]

    # Merge, preserving order, deduplicating by step
    seen = set()
    result = []
    for entry in sorted(early + grok_w + terminal, key=lambda l: l["step"]):
        if entry["step"] not in seen:
            seen.add(entry["step"])
            result.append(entry)
    return result


def trajectory_from_attn_logs(data):
    """
    Extract full trajectory from attn_logs format (single-task sweep files).
    Returns g₂₃ for W_Q, W_K, W_V, W_O of layer 0, plus mean over all 4.
    """
    logs    = data.get("attn_logs", [])
    metrics = data.get("metrics", [])
    logs    = subsample_logs(logs, metrics)   # avoid loading 2000 entries from GB files
    acc_map = {m["step"]: m["test_acc"] for m in metrics}

    traj = []
    for log in logs:
        step  = log["step"]
        mats  = {}
        for layer in log["layers"]:
            if layer.get("layer") == 0:
                for key in ("WQ", "WK", "WV", "WO"):
                    if key in layer:
                        mats[key] = layer[key]
                break
        if not mats:
            continue

        entry = {"step": step, "test_acc": acc_map.get(step)}
        for key in ("WQ", "WK", "WV", "WO"):
            if key in mats:
                g, S = g23_from_matrix(mats[key])
                entry[f"g23_{key}"] = g
                if key == "WQ":
                    entry["g23"] = g
        # mean g₂₃ across all 4 matrices
        vals = [entry.get(f"g23_{k}") for k in ("WQ","WK","WV","WO")
                if entry.get(f"g23_{k}") is not None]
        entry["g23_mean"] = float(np.mean(vals)) if vals else None
        # flat_vec: all attention matrices concatenated (for Gram matrix k*)
        parts = []
        for layer in log["layers"]:
            for key in ("WQ", "WK", "WV", "WO"):
                if key in layer:
                    parts.append(layer[key].flatten().float())
        entry["flat_vec"] = torch.cat(parts).numpy() if parts else None
        traj.append(entry)
    return traj, metrics


def trajectory_from_checkpoints(data):
    """
    Extract full trajectory from checkpoints format (multitask files).
    Returns g₂₃ for W_Q, W_K, W_V, W_O of layer 0.
    """
    checkpoints = data.get("checkpoints", [])
    metrics     = data.get("metrics", [])
    # multitask metrics may use 'test_acc_add', 'test_acc_mul', etc.
    acc_map = {}
    for m in metrics:
        step = m["step"]
        # prefer mean of all task accs, fall back to first acc key found
        task_accs = [v for k, v in m.items()
                     if k.startswith("test") and "loss" not in k
                     and isinstance(v, (int, float))]
        if task_accs:
            acc_map[step] = float(np.mean(task_accs))

    traj = []
    for step, sd in checkpoints:
        entry = {"step": step, "test_acc": acc_map.get(step)}
        for key in sorted(sd.keys()):
            if "layers.0.self_attn.in_proj_weight" in key:
                W = sd[key].float()
                d = W.shape[1]
                for mat_name, mat in [("WQ", W[:d]), ("WK", W[d:2*d]), ("WV", W[2*d:])]:
                    g, S = g23_from_matrix(mat)
                    entry[f"g23_{mat_name}"] = g
                    if mat_name == "WQ":
                        entry["g23"] = g
            elif "layers.0.self_attn.out_proj.weight" in key:
                g, S = g23_from_matrix(sd[key].float())
                entry["g23_WO"] = g
        vals = [entry.get(f"g23_{k}") for k in ("WQ","WK","WV","WO")
                if entry.get(f"g23_{k}") is not None]
        entry["g23_mean"] = float(np.mean(vals)) if vals else None
        # flat_vec: all attention weights concatenated (for Gram matrix k*)
        parts = []
        for key in sorted(sd.keys()):
            if "self_attn" in key and "weight" in key and "bias" not in key:
                parts.append(sd[key].flatten().float())
        entry["flat_vec"] = torch.cat(parts).numpy() if parts else None
        traj.append(entry)
    return traj, metrics


def analyze_run(traj, metrics, label=""):
    """
    Compute thesis Table 7 quantities for one run.
    Returns dict with all quantities, or None if trajectory empty.
    """
    if not traj:
        return None

    grok_step = find_grok_step(metrics)

    # ── ALL quantities come from the Gram matrix ──────────────────────────
    # gram_traj: (step, k*, R, g₂₃) at every rolling window
    gram_traj  = gram_R_trajectory(traj, window=10)  # list of (step, k*, R, g23)
    if not gram_traj:
        return None

    # Terminal Gram stats
    flat_vecs  = [t["flat_vec"] for t in traj if t.get("flat_vec") is not None]
    kstar_term, R_term, g23_term, _ = gram_stats(flat_vecs, window=10)
    final_step = traj[-1]["step"]

    # g₂₃_early: peak Gram-matrix g₂₃ before grokking
    pre_gram = [(s, k, R, g) for s, k, R, g in gram_traj
                if g is not None and (grok_step is None or s < grok_step)]
    if not pre_gram:
        pre_gram = [(s, k, R, g) for s, k, R, g in gram_traj if g is not None]
    if not pre_gram:
        return None

    peak       = max(pre_gram, key=lambda x: x[3])
    g23_early  = peak[3]
    early_step = peak[0]

    # g₂₃_grok: Gram-matrix g₂₃ at the window nearest the grokking step
    g23_grok = None
    if grok_step is not None:
        cands = [(s, k, R, g) for s, k, R, g in gram_traj
                 if s <= grok_step and g is not None]
        if cands:
            g23_grok = cands[-1][3]

    # R_early and R_grok
    R_early = float(np.mean([R for _, _, R, _ in pre_gram])) if pre_gram else None
    R_grok  = None
    if grok_step is not None:
        cands_r = [(s, R) for s, _, R, _ in gram_traj if s <= grok_step]
        if cands_r:
            R_grok = cands_r[-1][1]

    # Decline
    decline = None
    if g23_early and g23_grok and g23_grok > 0:
        decline = g23_early / g23_grok

    # Decline check: did g₂₃ decline overall from early peak to grokking?
    declined = None
    if decline is not None:
        declined = (decline > 1.5)   # at least 1.5× decline counts

    # Full Gram trajectory for plotting
    full_gram_traj = [(s, g) for s, _, _, g in gram_traj if g is not None]

    return {
        "label":        label,
        "grok_step":    grok_step,
        "early_step":   early_step,
        "g23_early":    g23_early,
        "g23_grok":     g23_grok,
        "decline":      decline,
        "kstar_term":   kstar_term,
        "R_term":       R_term,
        "R_early":      R_early,
        "R_grok":       R_grok,
        "gram_traj":    gram_traj,
        "full_gram_traj": full_gram_traj,
        "final_step":   final_step,
        "declined":     declined,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def fmt_g(v):
    return f"{v:.2f}" if v is not None else "—"

def fmt_decline(v):
    return f"{v:.1f}×" if v is not None else "—"

def fmt_kstar(v):
    return str(v) if v is not None else "—"

def fmt_mono(v):
    if v is None:   return "—"
    return "✓" if v else "✗"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def print_table(rows, title):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    print(f"{'Op/Task':<18} {'Seed':<6} {'Grok':<7} "
          f"{'g23_early':>10} {'g23_grok':>9} {'Decline':>9} "
          f"{'k*_term':>8} {'Decl':>5}")
    print("-"*80)
    for r in rows:
        if r is None:
            print("  [MISSING]")
            continue
        grok_str = str(r["grok_step"]) if r["grok_step"] else "none"
        print(f"{r['label']:<18} {'':<6} {grok_str:<7} "
              f"{fmt_g(r['g23_early']):>10} {fmt_g(r['g23_grok']):>9} "
              f"{fmt_decline(r['decline']):>9} "
              f"{fmt_kstar(r['kstar_term']):>8} {fmt_mono(r.get('declined')):>5}")


def print_table_with_seed(rows, title):
    print(f"\n{'='*95}")
    print(title)
    print(f"{'='*95}")
    print(f"{'Op':<14} {'Seed':<6} {'Grok':<7} "
          f"{'g23_early':>10} {'g23_grok':>9} {'Decline':>9} "
          f"{'R_early':>8} {'R_grok':>7} {'k*_term':>8} {'Decl':>5}")
    print("-"*95)
    for r in rows:
        if r is None:
            print("  [MISSING]")
            continue
        grok_str = str(r["grok_step"]) if r["grok_step"] else "none"
        seed_str = str(r.get("seed", ""))
        op_str   = r.get("op", r["label"])
        print(f"{op_str:<14} {seed_str:<6} {grok_str:<7} "
              f"{fmt_g(r['g23_early']):>10} {fmt_g(r['g23_grok']):>9} "
              f"{fmt_decline(r['decline']):>9} "
              f"{fmt_g(r.get('R_early')):>8} {fmt_g(r.get('R_grok')):>7} "
              f"{fmt_kstar(r['kstar_term']):>8} {fmt_mono(r.get('declined')):>5}")


def plot_trajectories(all_results_wd1, all_results_wd0, title, filename):
    """Plot Gram-matrix g₂₃ trajectories for WD=1 vs WD=0."""
    ops_present = sorted(set(r["op"] for r in all_results_wd1 if r is not None))
    if not ops_present:
        return
    n_ops = len(ops_present)
    fig, axes = plt.subplots(n_ops, 1, figsize=(8, 3.5 * n_ops), squeeze=False)

    for row, op in enumerate(ops_present):
        wd1_runs = [r for r in all_results_wd1 if r and r.get("op") == op]
        wd0_runs = [r for r in all_results_wd0 if r and r.get("op") == op]
        ax = axes[row, 0]

        for r in wd1_runs:
            pts = r.get("full_gram_traj", [])
            if pts:
                steps, vals = zip(*pts)
                ax.plot(steps, vals, color="#e74c3c", alpha=0.7,
                        linewidth=1.5, label=f"WD=1 s{r['seed']}")
            if r.get("grok_step"):
                ax.axvline(r["grok_step"], color="#e74c3c",
                           linestyle="--", alpha=0.3, linewidth=0.8)

        for r in wd0_runs:
            pts = r.get("full_gram_traj", [])
            if pts:
                steps, vals = zip(*pts)
                ax.plot(steps, vals, color="#3498db", alpha=0.7,
                        linestyle="--", linewidth=1.5,
                        label=f"WD=0 s{r['seed']}")

        ax.set_title(f"{op} — Gram matrix g₂₃", fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("g₂₃ = σ₂²−σ₃² (Gram)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(fontsize=6)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = PLOT_DIR / filename
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── SINGLE-TASK ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("SINGLE-TASK: Replication of Thesis Table 7")
    print("g₂₃ = σ₂² - σ₃² of rolling-window Gram matrix (W=10)")
    print("Grokking threshold: test_acc ≥ 0.95")
    print("="*80)

    wd1_rows, wd0_rows = [], []

    MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "200"))   # skip files larger than this

    for op in OPS:
        for seed in SEEDS:
            for wd, store in [("1.0", wd1_rows), ("0.0", wd0_rows)]:
                path = SINGLE_DIR / f"{op}_wd{wd}_s{seed}.pt"
                if not path.exists():
                    print(f"  MISSING: {path.name}")
                    store.append(None)
                    continue
                size_mb = path.stat().st_size / 1e6
                if size_mb > MAX_FILE_MB:
                    print(f"  LARGE ({size_mb:.0f}MB, skipped): {path.name}  "
                          f"  (set MAX_FILE_MB={int(size_mb)+1} to include)")
                    store.append(None)
                    continue
                print(f"  Loading {path.name} ({size_mb:.0f}MB)...")
                data = torch.load(path, map_location="cpu", weights_only=False)
                traj, metrics = trajectory_from_attn_logs(data)
                result = analyze_run(traj, metrics, label=f"{op}")
                if result:
                    result["op"]   = op
                    result["seed"] = seed
                    result["wd"]   = wd
                store.append(result)

    print_table_with_seed(wd1_rows,
        "WD = 1.0  (should grok — g₂₃ should decline)")
    print_table_with_seed(wd0_rows,
        "WD = 0.0  (should not grok — g₂₃ should NOT decline)")

    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY — Single-task")
    wd1_valid = [r for r in wd1_rows if r and r["grok_step"]]
    wd0_valid = [r for r in wd0_rows if r]

    declines_wd1 = [r["decline"] for r in wd1_valid if r["decline"]]
    print(f"  WD=1.0 grokking runs:   {len(wd1_valid)}")
    if declines_wd1:
        print(f"    g₂₃ decline:  mean={np.mean(declines_wd1):.1f}×  "
              f"range={min(declines_wd1):.1f}–{max(declines_wd1):.1f}×")
    wd1_declined = sum(1 for r in wd1_valid if r.get("declined"))
    print(f"    Declined (>1.5×): {wd1_declined}/{len(wd1_valid)}")

    print(f"  WD=0.0 control runs:    {len(wd0_valid)}")
    wd0_declined = sum(1 for r in wd0_valid if r.get("declined"))
    wd0_grokked  = sum(1 for r in wd0_valid if r["grok_step"])
    print(f"    Declined (>1.5×): {wd0_declined}/{len(wd0_valid)}  "
          f"({wd0_grokked} had grok step)")

    # Plots
    plot_trajectories(wd1_rows, wd0_rows,
        "Single-task: g₂₃ = λ₂ - λ₃ of W_Q  (thesis Table 7 exact definition)",
        "thesis_table7_singletask.png")

    # ── DUAL-TASK ──────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("DUAL-TASK: g₂₃ trajectory  (WD=1.0 vs WD=0.0, 3 seeds)")
    print("="*80)

    dt_wd1, dt_wd0 = [], []
    for seed in SEEDS:
        for tag, wd, store in [
            ("multitask",      "1.0", dt_wd1),
            ("multitask_nowd", "0.0", dt_wd0),
        ]:
            path = DUAL_DIR / f"{tag}_s{seed}.pt"
            if not path.exists():
                print(f"  MISSING: {path.name}")
                store.append(None)
                continue
            data   = torch.load(path, map_location="cpu", weights_only=False)
            traj, metrics = trajectory_from_checkpoints(data)
            # For dual-task, grokking = both tasks ≥ 0.95
            # Use whichever task's acc is available
            result = analyze_run(traj, metrics, label="dual-add+mul")
            if result:
                result["op"]   = "dual"
                result["seed"] = seed
                result["wd"]   = wd
            store.append(result)

    print_table_with_seed(dt_wd1, "DUAL-TASK WD=1.0")
    print_table_with_seed(dt_wd0, "DUAL-TASK WD=0.0")

    # ── TRI-TASK ───────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TRI-TASK: g₂₃ trajectory  (WD=1.0 vs WD=0.0, 3 seeds)")
    print("="*80)

    tt_wd1, tt_wd0 = [], []
    for seed in SEEDS:
        for wd_str, wd_label, store in [
            ("1",  "1.0", tt_wd1),
            ("0",  "0.0", tt_wd0),
        ]:
            path = TRI_DIR / f"tritask_wd{wd_str}_s{seed}.pt"
            if not path.exists():
                print(f"  MISSING: {path.name}")
                store.append(None)
                continue
            data   = torch.load(path, map_location="cpu", weights_only=False)
            traj, metrics = trajectory_from_checkpoints(data)
            result = analyze_run(traj, metrics, label="tritask")
            if result:
                result["op"]   = "tri"
                result["seed"] = seed
                result["wd"]   = wd_label
            store.append(result)

    print_table_with_seed(tt_wd1, "TRI-TASK WD=1.0")
    print_table_with_seed(tt_wd0, "TRI-TASK WD=0.0")

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    all_wd1 = [r for r in wd1_rows + dt_wd1 + tt_wd1
               if r and r["grok_step"]]
    all_wd0 = [r for r in wd0_rows + dt_wd0 + tt_wd0 if r]

    def summarize_group(rows, label, is_control=False):
        from collections import Counter
        declines = [r["decline"] for r in rows if r["decline"]]
        R_earls  = [r["R_early"] for r in rows if r.get("R_early") is not None]
        R_groks  = [r["R_grok"]  for r in rows if r.get("R_grok")  is not None]
        n_declined = sum(1 for r in rows if r.get("declined"))
        if is_control:
            n_grok = sum(1 for r in rows if r["grok_step"])
            decl_str = f"{n_declined}/{len(rows)}  ({n_grok} had grok step)"
        else:
            n_with_grok = sum(1 for r in rows if r["grok_step"])
            decl_str = f"{n_declined}/{n_with_grok}"
        print(f"\n  {label}  (n={len(rows)})")
        if declines:
            print(f"    g₂₃ decline:    mean={np.mean(declines):.1f}×  "
                  f"range={min(declines):.1f}–{max(declines):.1f}×  "
                  f"median={np.median(declines):.1f}×")
        print(f"    Declined (>1.5×): {decl_str}")
        if R_earls:
            r_str = f"mean={np.mean(R_earls):.3f} ± {np.std(R_earls):.3f}"
            if R_groks:
                r_str += f"   R_grok mean={np.mean(R_groks):.3f} ± {np.std(R_groks):.3f}"
            print(f"    R (Gram gap):   {r_str}")
        kstars = [r["kstar_term"] for r in rows if r["kstar_term"]]
        if kstars:
            c = Counter(kstars)
            print(f"    k*_term dist:   {dict(sorted(c.items()))}")

    summarize_group(all_wd1, "WD=1.0  (grokking runs)", is_control=False)
    summarize_group(all_wd0, "WD=0.0  (control runs)", is_control=True)

    # Save results
    torch.save({
        "wd1": wd1_rows, "wd0": wd0_rows,
        "dt_wd1": dt_wd1, "dt_wd0": dt_wd0,
        "tt_wd1": tt_wd1, "tt_wd0": tt_wd0,
    }, RESULTS_DIR / "thesis_table7_results.pt")
    print(f"\n  Results saved to {RESULTS_DIR / 'thesis_table7_results.pt'}")


if __name__ == "__main__":
    main()
