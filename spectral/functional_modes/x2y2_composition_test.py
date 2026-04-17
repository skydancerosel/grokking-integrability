#!/usr/bin/env python3
"""
x2y2_composition_test.py

Test the decomposition hypothesis: x²+y² = (x+y)² − 2xy

The model composes two operations it already knows:
  - Addition circuit: (a+b) mod p  →  additive Fourier ω≈25-26
  - Multiplication circuit: a·b mod p  →  dlog Fourier ω=29

If this is correct, then:
  1. Additive Fourier features alone explain SOME variance
  2. Multiplicative (dlog) Fourier features alone explain SOME variance
  3. Both COMBINED explain much MORE than either alone
  4. The improvement from combining is the signature of composition

Also test the full decomposition basis: for each pair (ω_add, ω_mul),
    cos(2π·ω_add·(a+b)/p) · cos(2π·ω_mul·dlog(ab)/(p-1))
These cross-terms capture the interaction between the two circuits.
"""

import math
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
PLOT_DIR    = SCRIPT_DIR / "x2y2_composition_plots"
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

# ─── Model ───────────────────────────────────────────────────────────────────

class ModOpTransformer(nn.Module):
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

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"

def get_attn_keys(model):
    return sorted(
        n for n, _ in model.named_parameters()
        if "self_attn" in n and "weight" in n and "bias" not in n
    )

# ─── SVD / checkpoint ───────────────────────────────────────────────────────

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

def select_postgrok(attn_logs, cache_data, updates):
    attn_steps   = [e["step"] for e in attn_logs]
    update_steps = attn_steps[1:]
    metrics      = cache_data["metrics"]
    n            = len(updates)
    post = next((t for t in range(n-1, -1, -1)
                 if min(metrics, key=lambda m: abs(m["step"] - update_steps[t]))["test_acc"] > 0.95),
                n-1)
    return post, update_steps[post]

def find_nearest_ckpt(step, cache_data):
    return min(cache_data["checkpoints"], key=lambda cs: abs(cs[0] - step))

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
    """
    Build three feature sets:
      A) Additive Fourier: cos/sin(ω·(a+b)/p) for ω = 1..p//2
      B) Multiplicative Fourier: cos/sin(ω·(dlog(a)+dlog(b))/(p-1)) for ω = 1..(p-1)//2
      C) Cross-terms: cos(ω_a·(a+b)/p)·cos(ω_m·dlog(ab)/(p-1)) etc.
         for top additive freqs × top multiplicative freqs
    """
    N = len(test_pairs)
    a_arr = np.array([a for a, b in test_pairs])
    b_arr = np.array([b for a, b in test_pairs])

    # ── A) Additive features ────────────────────────────────────────────
    apb = (a_arr + b_arr) % p
    add_freqs = np.arange(1, p // 2 + 1)
    phases_add = np.outer(apb.astype(float), add_freqs) * (2 * np.pi / p)
    add_cos = np.cos(phases_add)  # [N, n_add_freq]
    add_sin = np.sin(phases_add)
    X_add = np.hstack([add_cos, add_sin])  # [N, 2*n_add_freq]

    # ── B) Multiplicative features ──────────────────────────────────────
    # dlog(a·b) = dlog(a) + dlog(b) mod (p-1) for a,b ≠ 0
    dlog_sum = np.zeros(N, dtype=float)
    valid = np.ones(N, dtype=bool)
    for i, (a, b) in enumerate(test_pairs):
        if a == 0 or b == 0:
            valid[i] = False
        else:
            dlog_sum[i] = (dlog[a] + dlog[b]) % (p - 1)

    mul_freqs = np.arange(1, (p - 1) // 2 + 1)
    phases_mul = np.outer(dlog_sum, mul_freqs) * (2 * np.pi / (p - 1))
    mul_cos = np.cos(phases_mul)
    mul_sin = np.sin(phases_mul)
    X_mul = np.hstack([mul_cos, mul_sin])

    # ── C) Cross-terms ──────────────────────────────────────────────────
    # Use top-10 additive freqs × top-10 multiplicative freqs
    n_cross = 10
    cross_features = []
    for wa in range(n_cross):
        for wm in range(n_cross):
            cross_features.append(add_cos[:, wa] * mul_cos[:, wm])
            cross_features.append(add_cos[:, wa] * mul_sin[:, wm])
            cross_features.append(add_sin[:, wa] * mul_cos[:, wm])
            cross_features.append(add_sin[:, wa] * mul_sin[:, wm])
    X_cross = np.column_stack(cross_features)

    # ── D) Focused: just the known grokking frequencies ─────────────────
    # Add: ω=25,26 (from our results)
    # Mul (dlog): ω=29 (from our results)
    focused_add_freqs = [25, 26]
    focused_mul_freqs = [29]

    focused_features = []
    focused_names = []
    for wa in focused_add_freqs:
        phase_a = apb.astype(float) * wa * (2 * np.pi / p)
        focused_features.append(np.cos(phase_a))
        focused_names.append(f"cos({wa}·(a+b)/p)")
        focused_features.append(np.sin(phase_a))
        focused_names.append(f"sin({wa}·(a+b)/p)")

    for wm in focused_mul_freqs:
        phase_m = dlog_sum * wm * (2 * np.pi / (p - 1))
        focused_features.append(np.cos(phase_m))
        focused_names.append(f"cos({wm}·dlog(ab)/(p-1))")
        focused_features.append(np.sin(phase_m))
        focused_names.append(f"sin({wm}·dlog(ab)/(p-1))")

    # Focused cross-terms
    for wa in focused_add_freqs:
        phase_a = apb.astype(float) * wa * (2 * np.pi / p)
        for wm in focused_mul_freqs:
            phase_m = dlog_sum * wm * (2 * np.pi / (p - 1))
            focused_features.append(np.cos(phase_a) * np.cos(phase_m))
            focused_names.append(f"cos({wa}·add)·cos({wm}·mul)")
            focused_features.append(np.cos(phase_a) * np.sin(phase_m))
            focused_names.append(f"cos({wa}·add)·sin({wm}·mul)")
            focused_features.append(np.sin(phase_a) * np.cos(phase_m))
            focused_names.append(f"sin({wa}·add)·cos({wm}·mul)")
            focused_features.append(np.sin(phase_a) * np.sin(phase_m))
            focused_names.append(f"sin({wa}·add)·sin({wm}·mul)")

    X_focused = np.column_stack(focused_features)

    return {
        "add":     X_add,
        "mul":     X_mul,
        "cross":   X_cross,
        "focused": X_focused,
        "combined_add_mul": np.hstack([X_add, X_mul]),
        "full":    np.hstack([X_add, X_mul, X_cross]),
        "valid":   valid,
        "focused_names": focused_names,
    }


# ─── Probing ────────────────────────────────────────────────────────────────

def probe_r2(delta_h, X, valid=None, alpha=1.0):
    """R² of ridge probe: predict ||Δh||² and full Δh from X."""
    if valid is not None:
        delta_h = delta_h[valid]
        X = X[valid]

    y = np.sum(delta_h**2, axis=1)

    # Scalar
    m = Ridge(alpha=alpha)
    m.fit(X, y)
    yp = m.predict(X)
    r2_s = max(1 - np.sum((y - yp)**2) / (np.sum((y - y.mean())**2) + 1e-30), 0)

    # Multivariate
    m2 = Ridge(alpha=alpha)
    m2.fit(X, delta_h)
    dhp = m2.predict(X)
    r2_m = max(1 - np.sum((delta_h - dhp)**2) /
               (np.sum((delta_h - delta_h.mean(0, keepdims=True))**2) + 1e-30), 0)

    return float(r2_s), float(r2_m)


# ─── Also run on add and mul for comparison ──────────────────────────────────

def load_and_setup(op_name, device):
    cache_map = {
        "add":   RESULTS_DIR / "training_cache.pt",
        "mul":   RESULTS_DIR / "training_cache_mul.pt",
        "x2_y2": RESULTS_DIR / "training_cache_x2_y2.pt",
    }
    sweep_path = SWEEP_DIR / f"{op_name}_wd1.0_s42.pt"
    cache_path = cache_map[op_name]
    sweep_data = torch.load(sweep_path, map_location="cpu", weights_only=False)
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    attn_logs  = sweep_data["attn_logs"]
    if not cache_data.get("metrics"):
        cache_data["metrics"] = sweep_data.get("metrics", sweep_data.get("log", []))
    updates    = compute_updates(attn_logs)
    test_pairs = [(int(a), int(b)) for a, b in cache_data["test_pairs"]]
    t_idx, step = select_postgrok(attn_logs, cache_data, updates)
    S, Vt = window_svd(updates, t_idx)
    _, state_dict = find_nearest_ckpt(step, cache_data)
    model = ModOpTransformer().to(device)
    return model, state_dict, Vt, test_pairs, step


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_composition_bar(results_by_op):
    """Bar chart: R² under different feature sets, for all ops."""
    ops = list(results_by_op.keys())
    feature_sets = ["add", "mul", "combined_add_mul", "cross", "full", "focused"]
    fs_labels = ["Additive\nFourier", "Multiplicative\n(dlog) Fourier",
                 "Add + Mul\ncombined", "Cross-terms\n(add×mul)",
                 "Full\n(add+mul+cross)", "Focused\n(ω=25,26,29)"]

    n_ops = len(ops)
    n_fs  = len(feature_sets)

    fig, axes = plt.subplots(1, n_ops, figsize=(6*n_ops, 5), squeeze=False)
    fig.suptitle("Composition Test: R² Under Different Feature Sets\n"
                 "x²+y² = (x+y)² − 2xy → should need BOTH additive and multiplicative features",
                 fontsize=12, fontweight="bold")

    for col, op in enumerate(ops):
        ax = axes[0, col]
        res = results_by_op[op]

        for k in range(min(3, len(res))):
            x = np.arange(n_fs)
            vals = [res[k].get(fs, (0, 0))[0] for fs in feature_sets]  # scalar R²
            offset = (k - 1) * 0.25
            colors = ["#e74c3c", "#3498db", "#2ecc71"][k]
            ax.bar(x + offset, vals, 0.25, color=colors, alpha=0.85,
                   edgecolor="k", linewidth=0.3, label=f"v{k+1}")

        ax.set_xticks(range(n_fs))
        ax.set_xticklabels(fs_labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("R² (scalar probe)")
        ax.set_title(f"{op}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, None)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figA_composition_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figA_composition_r2.png")


def plot_synergy(results_by_op):
    """Highlight the synergy: combined R² vs max(add, mul) R²."""
    ops = list(results_by_op.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ops))
    w = 0.15

    for k, color in [(0, "#e74c3c"), (1, "#3498db"), (2, "#2ecc71")]:
        add_r2  = []
        mul_r2  = []
        comb_r2 = []
        full_r2 = []

        for op in ops:
            res = results_by_op[op]
            if k >= len(res): continue
            add_r2.append(res[k].get("add", (0, 0))[0])
            mul_r2.append(res[k].get("mul", (0, 0))[0])
            comb_r2.append(res[k].get("combined_add_mul", (0, 0))[0])
            full_r2.append(res[k].get("full", (0, 0))[0])

        max_single = [max(a, m) for a, m in zip(add_r2, mul_r2)]

        ax.bar(x + (k-1)*3*w - w, max_single, w, color=color, alpha=0.4,
               edgecolor="k", linewidth=0.3)
        ax.bar(x + (k-1)*3*w,     comb_r2,    w, color=color, alpha=0.7,
               edgecolor="k", linewidth=0.3)
        ax.bar(x + (k-1)*3*w + w, full_r2,    w, color=color, alpha=1.0,
               edgecolor="k", linewidth=0.3,
               label=f"v{k+1}" if ops.index(ops[0]) == 0 else "")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.4, label="max(add, mul)"),
        Patch(facecolor="gray", alpha=0.7, label="add + mul combined"),
        Patch(facecolor="gray", alpha=1.0, label="full (add+mul+cross)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=11)
    ax.set_ylabel("R² (scalar probe)")
    ax.set_title("Composition Synergy: Does Combining Add + Mul Help?\n"
                 "For x²+y², combined should exceed max(add, mul) = composition signature",
                 fontsize=11)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figB_synergy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figB_synergy.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    gen, dlog = build_dlog_table(P)
    print(f"Primitive root mod {P}: g={gen}")

    results_by_op = {}

    for op_name in ["add", "mul", "x2_y2"]:
        print(f"\n{'='*60}")
        print(f"  {op_name}")
        print(f"{'='*60}")

        model, state_dict, Vt, test_pairs, step = load_and_setup(op_name, device)
        n_dir = min(N_DIR, Vt.shape[0])
        print(f"  Post-grok step={step}, {len(test_pairs)} test pairs")

        feats = build_composition_features(test_pairs, dlog)

        results = {}
        for k in range(min(5, n_dir)):
            dh = compute_delta_h(model, state_dict, Vt[k], test_pairs, device)

            r2_add_s, r2_add_m     = probe_r2(dh, feats["add"])
            r2_mul_s, r2_mul_m     = probe_r2(dh, feats["mul"], valid=feats["valid"])
            r2_comb_s, r2_comb_m   = probe_r2(dh, feats["combined_add_mul"],
                                               valid=feats["valid"])
            r2_cross_s, r2_cross_m = probe_r2(dh, feats["cross"], valid=feats["valid"])
            r2_full_s, r2_full_m   = probe_r2(dh, feats["full"], valid=feats["valid"])
            r2_foc_s, r2_foc_m     = probe_r2(dh, feats["focused"], valid=feats["valid"])

            results[k] = {
                "add":              (r2_add_s, r2_add_m),
                "mul":              (r2_mul_s, r2_mul_m),
                "combined_add_mul": (r2_comb_s, r2_comb_m),
                "cross":            (r2_cross_s, r2_cross_m),
                "full":             (r2_full_s, r2_full_m),
                "focused":          (r2_foc_s, r2_foc_m),
            }

            tag = "***" if k < 3 else "   "
            synergy = r2_comb_s - max(r2_add_s, r2_mul_s)
            print(f"  {tag} v{k+1}:  add={r2_add_s:.4f}  mul={r2_mul_s:.4f}  "
                  f"combined={r2_comb_s:.4f}  cross={r2_cross_s:.4f}  "
                  f"full={r2_full_s:.4f}  focused={r2_foc_s:.4f}  "
                  f"synergy={synergy:+.4f}")

        results_by_op[op_name] = results

    # ── Plots ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating figures...")
    print(f"{'='*60}")

    plot_composition_bar(results_by_op)
    plot_synergy(results_by_op)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPOSITION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"\n  x²+y² = (x+y)² − 2xy  →  composition of addition and multiplication")
    print(f"\n  R² (scalar probe on ||Δh||²):")
    print(f"  {'Op':<8} {'Dir':>4} {'Add':>8} {'Mul':>8} {'Combined':>10} {'Cross':>8} "
          f"{'Full':>8} {'Focused':>8} {'Synergy':>9}")
    print(f"  {'-'*75}")
    for op in results_by_op:
        for k in sorted(results_by_op[op]):
            r = results_by_op[op][k]
            tag = " *" if k < 3 else "  "
            add_s  = r["add"][0]
            mul_s  = r["mul"][0]
            comb_s = r["combined_add_mul"][0]
            cross_s = r["cross"][0]
            full_s = r["full"][0]
            foc_s  = r["focused"][0]
            syn    = comb_s - max(add_s, mul_s)
            print(f"  {op:<8} v{k+1}{tag} {add_s:8.4f} {mul_s:8.4f} {comb_s:10.4f} "
                  f"{cross_s:8.4f} {full_s:8.4f} {foc_s:8.4f} {syn:+9.4f}")

    # Multivariate R²
    print(f"\n  R² (multivariate probe on full Δh):")
    print(f"  {'Op':<8} {'Dir':>4} {'Add':>8} {'Mul':>8} {'Combined':>10} {'Full':>8} {'Synergy':>9}")
    print(f"  {'-'*60}")
    for op in results_by_op:
        for k in sorted(results_by_op[op]):
            r = results_by_op[op][k]
            tag = " *" if k < 3 else "  "
            add_m  = r["add"][1]
            mul_m  = r["mul"][1]
            comb_m = r["combined_add_mul"][1]
            full_m = r["full"][1]
            syn    = comb_m - max(add_m, mul_m)
            print(f"  {op:<8} v{k+1}{tag} {add_m:8.4f} {mul_m:8.4f} {comb_m:10.4f} "
                  f"{full_m:8.4f} {syn:+9.4f}")

    print(f"\nResults saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
