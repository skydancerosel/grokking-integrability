#!/usr/bin/env python3
"""
x2y2_multitask_composition.py

Compare composition signatures: single-task x²+y² vs tritask (add+mul+sq shared trunk).

Hypothesis: in the tritask model, the x²+y² circuit is forced to reuse
the add and mul subcircuits (shared trunk).  Therefore:
  - Additive + multiplicative Fourier features should explain MORE variance
    for the tritask model's spectral edge directions
  - The composition synergy (combined > max(add,mul)) should be LARGER

Runs the same probe as x2y2_composition_test.py on both models.
"""

import math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "coherence_edge_results"
SWEEP_DIR   = SCRIPT_DIR.parent.parent / "grok_sweep_results"
TRITASK_DIR = SCRIPT_DIR.parent.parent / "multitask" / "results"
PLOT_DIR    = SCRIPT_DIR / "x2y2_multitask_plots"
PLOT_DIR.mkdir(exist_ok=True)

P        = 97
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 2
W        = 20
N_DIR    = 8
EPS_SCALE = 0.005

# ─── Number theory ──────────────────────────────────────────────────────────

def primitive_root(p):
    for g in range(2, p):
        seen = set()
        x = 1
        for _ in range(p - 1):
            x = (x * g) % p
            seen.add(x)
        if len(seen) == p - 1:
            return g
    raise ValueError

def build_dlog_table(p):
    g = primitive_root(p)
    dlog = {}
    x = 1
    for k in range(p - 1):
        dlog[x] = k
        x = (x * g) % p
    return g, dlog

# ─── Models ─────────────────────────────────────────────────────────────────

class ModOpTransformer(nn.Module):
    """Single-task model."""
    def __init__(self, p=P, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 d_ff=256, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Parameter(torch.randn(2, d_model) / math.sqrt(d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.ln   = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)

    def forward(self, a, b, return_residual=False):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        res = h[:, 0, :]
        logits = self.head(self.ln(res))
        if return_residual:
            return logits, res
        return logits


class TritaskTransformer(nn.Module):
    """Shared-trunk model with 3 task heads."""
    def __init__(self, p=P, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 d_ff=256, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Parameter(torch.randn(2, d_model) / math.sqrt(d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.ln   = nn.LayerNorm(d_model)
        self.head_add = nn.Linear(d_model, p)
        self.head_mul = nn.Linear(d_model, p)
        self.head_sq  = nn.Linear(d_model, p)

    def forward(self, a, b, return_residual=False):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        res = h[:, 0, :]
        ln_res = self.ln(res)
        logits = {
            "add": self.head_add(ln_res),
            "mul": self.head_mul(ln_res),
            "sq":  self.head_sq(ln_res),
        }
        if return_residual:
            return logits, res
        return logits


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"

def get_attn_keys(model):
    return sorted(
        n for n, _ in model.named_parameters()
        if "self_attn" in n and "weight" in n and "bias" not in n
    )

# ─── SVD helpers ─────────────────────────────────────────────────────────────

def flatten_attn_from_logs(entry):
    parts = []
    for ld in sorted(entry["layers"], key=lambda x: x["layer"]):
        for k in ["WQ", "WK", "WV", "WO"]:
            parts.append(ld[k].flatten().float())
    return torch.cat(parts)

def compute_updates(attn_logs):
    flat = [flatten_attn_from_logs(e).numpy() for e in attn_logs]
    return [flat[i] - flat[i-1] for i in range(1, len(flat))]

def window_svd(updates, t_idx):
    start = max(0, t_idx - W + 1)
    X = np.stack(updates[start:t_idx+1])
    if X.shape[0] < 3: return None
    X -= X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S, Vt

# ─── Test pair generation ────────────────────────────────────────────────────

def make_test_pairs(p=P, train_frac=0.5, seed=42):
    """Reproduce test split from training."""
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    n_train = int(len(all_pairs) * train_frac)
    return all_pairs[n_train:]

# ─── Perturbation ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_delta_h(model, state_dict, vk_np, test_pairs, device,
                    eps_scale=EPS_SCALE, batch_size=512):
    attn_keys = get_attn_keys(model)
    flat = torch.cat([state_dict[k].float().flatten() for k in attn_keys])
    eps  = eps_scale * float(flat.norm())
    vk_t = torch.from_numpy(vk_np.copy()).float()
    vk_t = vk_t / (vk_t.norm() + 1e-30)
    pert_sd = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key in attn_keys:
        numel = pert_sd[key].numel()
        pert_sd[key] = (pert_sd[key].float() +
                        eps * vk_t[offset:offset+numel].reshape(pert_sd[key].shape))
        offset += numel
    N = len(test_pairs)
    delta_h = np.zeros((N, D_MODEL), dtype=np.float32)
    ab_all = torch.tensor(test_pairs)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ab = ab_all[start:end].to(device)
        a, b = ab[:, 0], ab[:, 1]
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        model.eval()
        _, h_base = model(a, b, return_residual=True)
        model.load_state_dict({k: v.to(device) for k, v in pert_sd.items()})
        model.eval()
        _, h_pert = model(a, b, return_residual=True)
        delta_h[start:end] = (h_pert - h_base).float().cpu().numpy()
    return delta_h

# ─── Feature construction ───────────────────────────────────────────────────

def build_composition_features(test_pairs, dlog, p=P):
    N = len(test_pairs)
    a_arr = np.array([a for a, b in test_pairs])
    b_arr = np.array([b for a, b in test_pairs])

    # Additive Fourier
    apb = (a_arr + b_arr) % p
    add_freqs = np.arange(1, p // 2 + 1)
    phases_add = np.outer(apb.astype(float), add_freqs) * (2 * np.pi / p)
    X_add = np.hstack([np.cos(phases_add), np.sin(phases_add)])

    # Multiplicative Fourier (dlog)
    dlog_sum = np.zeros(N, dtype=float)
    valid = np.ones(N, dtype=bool)
    for i, (a, b) in enumerate(test_pairs):
        if a == 0 or b == 0:
            valid[i] = False
        else:
            dlog_sum[i] = (dlog[a] + dlog[b]) % (p - 1)
    mul_freqs = np.arange(1, (p - 1) // 2 + 1)
    phases_mul = np.outer(dlog_sum, mul_freqs) * (2 * np.pi / (p - 1))
    X_mul = np.hstack([np.cos(phases_mul), np.sin(phases_mul)])

    # Cross-terms (top-10 × top-10)
    n_cross = 10
    cross_features = []
    cos_a = np.cos(phases_add[:, :n_cross])
    sin_a = np.sin(phases_add[:, :n_cross])
    cos_m = np.cos(phases_mul[:, :n_cross])
    sin_m = np.sin(phases_mul[:, :n_cross])
    for wa in range(n_cross):
        for wm in range(n_cross):
            cross_features.append(cos_a[:, wa] * cos_m[:, wm])
            cross_features.append(cos_a[:, wa] * sin_m[:, wm])
            cross_features.append(sin_a[:, wa] * cos_m[:, wm])
            cross_features.append(sin_a[:, wa] * sin_m[:, wm])
    X_cross = np.column_stack(cross_features)

    return {
        "add":              X_add,
        "mul":              X_mul,
        "combined_add_mul": np.hstack([X_add, X_mul]),
        "cross":            X_cross,
        "full":             np.hstack([X_add, X_mul, X_cross]),
        "valid":            valid,
    }

# ─── Probing ────────────────────────────────────────────────────────────────

def probe_r2(delta_h, X, valid=None, alpha=1.0):
    if valid is not None:
        delta_h = delta_h[valid]
        X = X[valid]
    y = np.sum(delta_h**2, axis=1)
    m = Ridge(alpha=alpha)
    m.fit(X, y)
    yp = m.predict(X)
    r2_s = max(1 - np.sum((y - yp)**2) / (np.sum((y - y.mean())**2) + 1e-30), 0)
    m2 = Ridge(alpha=alpha)
    m2.fit(X, delta_h)
    dhp = m2.predict(X)
    r2_m = max(1 - np.sum((delta_h - dhp)**2) /
               (np.sum((delta_h - delta_h.mean(0, keepdims=True))**2) + 1e-30), 0)
    return float(r2_s), float(r2_m)

# ─── Fourier 1D ─────────────────────────────────────────────────────────────

def fourier_concentration(delta_h, group_vals, period):
    signal = np.sum(delta_h**2, axis=1)
    means  = np.zeros(period, dtype=np.float64)
    counts = np.zeros(period, dtype=np.float64)
    for i, q in enumerate(group_vals):
        means[int(q) % period] += signal[i]
        counts[int(q) % period] += 1
    means /= np.maximum(counts, 1)
    freqs  = np.arange(1, period // 2 + 1)
    phases = np.outer(np.arange(period, dtype=float), freqs) * (2 * np.pi / period)
    cos_c  = means @ np.cos(phases) / period
    sin_c  = means @ np.sin(phases) / period
    power  = cos_c**2 + sin_c**2
    power /= (power.sum() + 1e-30)
    peak_idx = int(np.argmax(power))
    return float(power[peak_idx]), int(freqs[peak_idx])

# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_comparison(single_results, tritask_results):
    """Side-by-side: single-task vs tritask composition R²."""
    feature_sets = ["add", "mul", "combined_add_mul", "cross", "full"]
    fs_labels    = ["Add\nFourier", "Mul\n(dlog)", "Add+Mul", "Cross\n(add×mul)", "Full"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Composition Test: Single-Task vs Tritask (Shared Trunk)\n"
                 "x²+y² = (x+y)² − 2xy — does shared training force circuit reuse?",
                 fontsize=12, fontweight="bold")

    for ax_i, (results, title) in enumerate([
        (single_results, "Single-task x²+y²"),
        (tritask_results, "Tritask (add+mul+sq shared)"),
    ]):
        ax = axes[ax_i]
        n_show = min(3, len(results))
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        x = np.arange(len(feature_sets))
        w = 0.25

        for k in range(n_show):
            vals_s = [results[k].get(fs, {}).get("scalar", 0) for fs in feature_sets]
            ax.bar(x + (k-1)*w, vals_s, w, color=colors[k], alpha=0.85,
                   edgecolor="k", linewidth=0.3, label=f"v{k+1}")

        ax.set_xticks(x)
        ax.set_xticklabels(fs_labels, fontsize=9)
        ax.set_ylabel("R² (scalar probe)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, None)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figA_single_vs_tritask.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figA_single_vs_tritask.png")


def plot_synergy_comparison(single_results, tritask_results):
    """Synergy comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Single-task\nx²+y²", "Tritask\nshared trunk"]
    x = np.arange(2)
    w = 0.2
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for k in range(3):
        vals = []
        for results in [single_results, tritask_results]:
            if k in results:
                add_r2  = results[k].get("add", {}).get("scalar", 0)
                mul_r2  = results[k].get("mul", {}).get("scalar", 0)
                comb_r2 = results[k].get("combined_add_mul", {}).get("scalar", 0)
                synergy = comb_r2 - max(add_r2, mul_r2)
                vals.append(synergy)
            else:
                vals.append(0)

        ax.bar(x + (k-1)*w, vals, w, color=colors[k], alpha=0.85,
               edgecolor="k", linewidth=0.3, label=f"v{k+1}")

    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Synergy = R²(add+mul) − max(R²(add), R²(mul))")
    ax.set_title("Composition Synergy: Single-Task vs Tritask\n"
                 "Higher synergy = more circuit reuse", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figB_synergy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figB_synergy_comparison.png")


def plot_fourier_comparison(single_fourier, tritask_fourier):
    """Fourier concentration comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fourier Concentration: Single-Task vs Tritask\n"
                 "Grouping by (a²+b²) mod p  |  (a+b) mod p  |  dlog(a·b)",
                 fontsize=12, fontweight="bold")

    for ax_i, (basis_name, title) in enumerate([
        ("output", "(a²+b²) mod p"),
        ("additive", "(a+b) mod p"),
        ("dlog", "dlog(a·b)"),
    ]):
        ax = axes[ax_i]
        n_show = min(5, max(len(single_fourier), len(tritask_fourier)))
        x = np.arange(n_show)
        w = 0.35

        single_F = [single_fourier.get(k, {}).get(basis_name, {}).get("F", 0) for k in range(n_show)]
        tri_F    = [tritask_fourier.get(k, {}).get(basis_name, {}).get("F", 0) for k in range(n_show)]

        colors_s = ["#e74c3c" if k < 3 else "#d4a5a5" for k in range(n_show)]
        colors_t = ["#2980b9" if k < 3 else "#a5c8d4" for k in range(n_show)]

        ax.bar(x - w/2, single_F, w, color=colors_s, alpha=0.85,
               edgecolor="k", linewidth=0.3, label="Single-task")
        ax.bar(x + w/2, tri_F, w, color=colors_t, alpha=0.85,
               edgecolor="k", linewidth=0.3, label="Tritask")

        ax.set_xticks(x)
        ax.set_xticklabels([f"v{k+1}" for k in range(n_show)], fontsize=9)
        ax.set_ylabel("F_k")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figC_fourier_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figC_fourier_comparison.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def run_model(label, model, state_dict, attn_logs, metrics, test_pairs, dlog, device):
    """Run full composition probe on one model."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    updates = compute_updates(attn_logs)
    n = len(updates)

    # Find post-grok checkpoint for sq/x2_y2
    update_steps = [e["step"] for e in attn_logs][1:]

    # Use last checkpoint with high test accuracy
    if "test_sq" in metrics[0]:
        acc_key = "test_sq"
    elif "test_acc" in metrics[0]:
        acc_key = "test_acc"
    else:
        acc_key = list(metrics[0].keys())[0]

    post = next((t for t in range(n-1, -1, -1)
                 if min(metrics, key=lambda m: abs(m["step"] - update_steps[t]))[acc_key] > 0.95),
                n-1)
    step = update_steps[post]
    print(f"  Post-grok: t_idx={post}, step={step}")

    svd_result = window_svd(updates, post)
    if svd_result is None:
        print("  SVD failed")
        return None, None
    S, Vt = svd_result
    n_dir = min(N_DIR, Vt.shape[0])

    # Find nearest checkpoint
    # For tritask, checkpoints are stored as (step, state_dict)
    nearest_ckpt = min(state_dict if isinstance(state_dict, list) else [(0, state_dict)],
                       key=lambda cs: abs(cs[0] - step) if isinstance(cs, tuple) else 0)
    if isinstance(nearest_ckpt, tuple):
        sd = nearest_ckpt[1]
    else:
        sd = nearest_ckpt

    feats = build_composition_features(test_pairs, dlog)

    # Composition probe
    probe_results = {}
    for k in range(min(5, n_dir)):
        dh = compute_delta_h(model, sd, Vt[k], test_pairs, device)

        r = {}
        for fs_name in ["add", "mul", "combined_add_mul", "cross", "full"]:
            v = feats.get("valid") if fs_name != "add" else None
            r2_s, r2_m = probe_r2(dh, feats[fs_name], valid=v)
            r[fs_name] = {"scalar": r2_s, "multi": r2_m}

        probe_results[k] = r

        tag = "***" if k < 3 else "   "
        add_s  = r["add"]["scalar"]
        mul_s  = r["mul"]["scalar"]
        comb_s = r["combined_add_mul"]["scalar"]
        full_s = r["full"]["scalar"]
        syn    = comb_s - max(add_s, mul_s)
        print(f"  {tag} v{k+1}: add={add_s:.4f}  mul={mul_s:.4f}  "
              f"combined={comb_s:.4f}  full={full_s:.4f}  synergy={syn:+.4f}")

    # Fourier analysis
    fourier_results = {}
    for k in range(min(5, n_dir)):
        dh = compute_delta_h(model, sd, Vt[k], test_pairs, device)

        # Output basis
        out_groups = np.array([(int(a)**2 + int(b)**2) % P for a, b in test_pairs])
        F_out, freq_out = fourier_concentration(dh, out_groups, P)

        # Additive basis
        add_groups = np.array([(int(a) + int(b)) % P for a, b in test_pairs])
        F_add, freq_add = fourier_concentration(dh, add_groups, P)

        # Dlog basis
        dlog_groups = []
        valid_mask = []
        for a, b in test_pairs:
            if a == 0 or b == 0:
                dlog_groups.append(0)
                valid_mask.append(False)
            else:
                dlog_groups.append((dlog[a] + dlog[b]) % (P - 1))
                valid_mask.append(True)
        dlog_groups = np.array(dlog_groups)
        valid_mask = np.array(valid_mask)
        F_dlog, freq_dlog = fourier_concentration(
            dh[valid_mask], dlog_groups[valid_mask], P - 1)

        fourier_results[k] = {
            "output":   {"F": F_out,  "freq": freq_out},
            "additive": {"F": F_add,  "freq": freq_add},
            "dlog":     {"F": F_dlog, "freq": freq_dlog},
        }

        tag = "***" if k < 3 else "   "
        print(f"  {tag} v{k+1} Fourier: out(ω={freq_out},F={F_out:.4f})  "
              f"add(ω={freq_add},F={F_add:.4f})  "
              f"dlog(ω={freq_dlog},F={F_dlog:.4f})")

    return probe_results, fourier_results


def main():
    device = get_device()
    print(f"Device: {device}")

    gen, dlog = build_dlog_table(P)
    print(f"Primitive root mod {P}: g={gen}")

    # ── Single-task x2_y2 ────────────────────────────────────────────────
    sweep_path = SWEEP_DIR / "x2_y2_wd1.0_s42.pt"
    cache_path = RESULTS_DIR / "training_cache_x2_y2.pt"
    sweep_data = torch.load(sweep_path, map_location="cpu", weights_only=False)
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not cache_data.get("metrics"):
        cache_data["metrics"] = sweep_data.get("metrics", sweep_data.get("log", []))

    test_pairs_single = [(int(a), int(b)) for a, b in cache_data["test_pairs"]]
    model_single = ModOpTransformer().to(device)

    single_probe, single_fourier = run_model(
        "Single-task x²+y²",
        model_single, cache_data["checkpoints"],
        sweep_data["attn_logs"], cache_data["metrics"],
        test_pairs_single, dlog, device)

    # ── Tritask (shared trunk) ───────────────────────────────────────────
    tri_path = TRITASK_DIR / "tritask_wd1_s42.pt"
    tri_data = torch.load(tri_path, map_location="cpu", weights_only=False)
    test_pairs_tri = make_test_pairs(P, train_frac=tri_data["cfg"].get("TRAIN_FRACTION", 0.5),
                                      seed=42)
    print(f"\nTritask: {len(test_pairs_tri)} test pairs")
    print(f"  Grok steps: {tri_data['grok_step']}")

    model_tri = TritaskTransformer().to(device)

    tri_probe, tri_fourier = run_model(
        "Tritask (add+mul+sq) — shared trunk",
        model_tri, tri_data["checkpoints"],
        tri_data["attn_logs"], tri_data["metrics"],
        test_pairs_tri, dlog, device)

    # ── Plots ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating figures...")
    print(f"{'='*60}")

    if single_probe and tri_probe:
        plot_comparison(single_probe, tri_probe)
        plot_synergy_comparison(single_probe, tri_probe)
    if single_fourier and tri_fourier:
        plot_fourier_comparison(single_fourier, tri_fourier)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SINGLE-TASK vs TRITASK COMPOSITION SUMMARY")
    print(f"{'='*60}")

    for label, results in [("Single-task", single_probe), ("Tritask", tri_probe)]:
        if not results: continue
        print(f"\n  {label}:")
        print(f"  {'Dir':>4} {'Add':>8} {'Mul':>8} {'Combined':>10} {'Cross':>8} "
              f"{'Full':>8} {'Synergy':>9}")
        print(f"  {'-'*60}")
        for k in sorted(results):
            r = results[k]
            tag = " *" if k < 3 else "  "
            a = r["add"]["scalar"]
            m = r["mul"]["scalar"]
            c = r["combined_add_mul"]["scalar"]
            cr = r["cross"]["scalar"]
            f = r["full"]["scalar"]
            syn = c - max(a, m)
            print(f"  v{k+1}{tag} {a:8.4f} {m:8.4f} {c:10.4f} {cr:8.4f} "
                  f"{f:8.4f} {syn:+9.4f}")

    for label, results in [("Single-task", single_fourier), ("Tritask", tri_fourier)]:
        if not results: continue
        print(f"\n  {label} Fourier:")
        print(f"  {'Dir':>4} {'Out ω':>6} {'Out F':>8} {'Add ω':>6} {'Add F':>8} "
              f"{'Dlog ω':>7} {'Dlog F':>8}")
        print(f"  {'-'*55}")
        for k in sorted(results):
            r = results[k]
            tag = " *" if k < 3 else "  "
            print(f"  v{k+1}{tag} {r['output']['freq']:6d} {r['output']['F']:8.4f} "
                  f"{r['additive']['freq']:6d} {r['additive']['F']:8.4f} "
                  f"{r['dlog']['freq']:7d} {r['dlog']['F']:8.4f}")

    print(f"\nResults saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
