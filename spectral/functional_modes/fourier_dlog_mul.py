#!/usr/bin/env python3
"""
fourier_dlog_mul.py

Discrete-log Fourier analysis for modular multiplication.

The standard Fourier basis for (a*b) mod p groups by raw output value.
But the natural characters for multiplication are discrete-log characters:
    χ_ω(a*b) = exp(2πi·ω·(dlog_g(a) + dlog_g(b)) / (p-1))

where g is a primitive root mod p.  If the grokking circuit internally
represents inputs via discrete log (well-established in the literature),
then grouping by dlog(a)+dlog(b) mod (p-1) should give much sharper
Fourier concentration than grouping by the raw product.

This script runs three analyses:

  1. Raw-output Fourier (baseline — reproduces fourier_functional_view.py results)
  2. Discrete-log Fourier (the hypothesis)
  3. Comparison figure: concentration under both bases

Also runs the same analysis for add (as control — should be unchanged or worse
under dlog basis since add doesn't use multiplicative structure).

Output
------
spectral/fourier_dlog_plots/
  figA_dlog_vs_raw_spectra.png    — side-by-side spectra for v1..v5
  figB_concentration_comparison.png — F_k under both bases for all directions
  figC_dlog_basis_test.png        — basis test under dlog Fourier
  figD_2d_fourier.png             — 2D Fourier heatmap P(ω_a, ω_b)
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "coherence_edge_results"
SWEEP_DIR   = SCRIPT_DIR.parent.parent / "grok_sweep_results"
PLOT_DIR    = SCRIPT_DIR / "fourier_dlog_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────

P        = 97
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 2
W        = 20
N_DIR    = 8
EPS_SCALE = 0.005

CACHE_PATHS = {
    "add":   RESULTS_DIR / "training_cache.pt",
    "mul":   RESULTS_DIR / "training_cache_mul.pt",
}

# ─── Discrete log table ─────────────────────────────────────────────────────

def primitive_root(p):
    """Find smallest primitive root mod p."""
    for g in range(2, p):
        seen = set()
        x = 1
        for _ in range(p - 1):
            x = (x * g) % p
            seen.add(x)
        if len(seen) == p - 1:
            return g
    raise ValueError(f"No primitive root found for p={p}")


def build_dlog_table(p):
    """Build discrete log table: dlog[a] = log_g(a) for a in 1..p-1."""
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
    if X.shape[0] < 3:
        return None
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


# ─── Residual stream perturbation ───────────────────────────────────────────

@torch.no_grad()
def compute_delta_h(model, state_dict, vk_np, test_pairs, device,
                    eps_scale=EPS_SCALE, batch_size=512):
    attn_keys = get_attn_keys(model)
    flat = torch.cat([state_dict[k].float().flatten() for k in attn_keys])
    norm = float(flat.norm())
    eps  = eps_scale * norm

    vk_t = torch.from_numpy(vk_np.copy()).float()
    vk_t = vk_t / (vk_t.norm() + 1e-30)

    pert_sd = {k: v.clone() for k, v in state_dict.items()}
    offset  = 0
    for key in attn_keys:
        numel = pert_sd[key].numel()
        pert_sd[key] = (pert_sd[key].float() +
                        eps * vk_t[offset:offset+numel].reshape(pert_sd[key].shape))
        offset += numel

    N       = len(test_pairs)
    delta_h = np.zeros((N, D_MODEL), dtype=np.float32)
    ab_all  = torch.tensor(test_pairs)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ab  = ab_all[start:end].to(device)
        a, b = ab[:, 0], ab[:, 1]
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        model.eval()
        _, h_base = model(a, b, return_residual=True)
        model.load_state_dict({k: v.to(device) for k, v in pert_sd.items()})
        model.eval()
        _, h_pert = model(a, b, return_residual=True)
        delta_h[start:end] = (h_pert - h_base).float().cpu().numpy()

    return delta_h


# ─── Fourier analysis ───────────────────────────────────────────────────────

def fourier_1d(signal_per_input, group_vals, period):
    """
    1D Fourier analysis: group signal by group_vals (mod period),
    compute mean per group, then DFT.

    Returns power_norm, freqs arrays.
    """
    means  = np.zeros(period, dtype=np.float64)
    counts = np.zeros(period, dtype=np.float64)
    for i, q in enumerate(group_vals):
        q_int = int(q) % period
        means[q_int]  += signal_per_input[i]
        counts[q_int] += 1
    means /= np.maximum(counts, 1)

    freqs  = np.arange(1, period // 2 + 1)
    phases = np.outer(np.arange(period, dtype=float), freqs) * (2 * np.pi / period)
    cos_c  = means @ np.cos(phases) / period
    sin_c  = means @ np.sin(phases) / period
    power  = cos_c**2 + sin_c**2
    power_norm = power / (power.sum() + 1e-30)

    return power_norm, freqs


def fourier_profile(delta_h, group_vals, period):
    """Fourier profile of ||Δh(x)||² grouped by group_vals."""
    signal = np.sum(delta_h**2, axis=1)
    power, freqs = fourier_1d(signal, group_vals, period)

    peak_idx   = int(np.argmax(power))
    peak_freq  = int(freqs[peak_idx])
    F_k        = float(power[peak_idx])
    top3_idx   = np.argsort(power)[::-1][:3]
    top3_freqs = [int(freqs[i]) for i in top3_idx]
    top3_power = [float(power[i]) for i in top3_idx]

    return {
        "power": power, "freqs": freqs,
        "peak_freq": peak_freq, "F_k": F_k,
        "top3_freqs": top3_freqs, "top3_power": top3_power,
    }


def fourier_2d(delta_h, test_pairs, p=P):
    """
    2D Fourier decomposition: P(ω_a, ω_b) = |Σ ||Δh||² exp(-2πi(ω_a·a+ω_b·b)/p)|²

    Returns 2D power array of shape [p//2, p//2] indexed by (ω_a-1, ω_b-1).
    """
    signal = np.sum(delta_h**2, axis=1)  # [N]
    pairs  = np.array(test_pairs, dtype=float)
    a_vals = pairs[:, 0]
    b_vals = pairs[:, 1]

    freqs  = np.arange(1, p // 2 + 1)
    n_freq = len(freqs)

    # Precompute phase factors
    phase_a = np.outer(a_vals, freqs) * (2 * np.pi / p)  # [N, n_freq]
    phase_b = np.outer(b_vals, freqs) * (2 * np.pi / p)  # [N, n_freq]

    # For each (ω_a, ω_b): Σ_n s_n exp(-i(ω_a a_n + ω_b b_n))
    # = Σ_n s_n [cos(ω_a a_n)cos(ω_b b_n) - sin(ω_a a_n)sin(ω_b b_n)]
    #   - i Σ_n s_n [cos(ω_a a_n)sin(ω_b b_n) + sin(ω_a a_n)cos(ω_b b_n)]

    cos_a = np.cos(phase_a)  # [N, n_freq]
    sin_a = np.sin(phase_a)
    cos_b = np.cos(phase_b)
    sin_b = np.sin(phase_b)

    # Weight by signal
    s = signal  # [N]

    # Real part: Σ s_n cos_a cos_b - s_n sin_a sin_b
    # [n_freq_a, n_freq_b]
    R = (s[:, None] * cos_a).T @ cos_b - (s[:, None] * sin_a).T @ sin_b
    # Imag part:
    I = -((s[:, None] * cos_a).T @ sin_b + (s[:, None] * sin_a).T @ cos_b)

    power_2d = (R**2 + I**2) / len(signal)**2
    power_2d /= (power_2d.sum() + 1e-30)

    return power_2d, freqs


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_figA_spectra(results, op_name):
    """Side-by-side: raw vs dlog spectra for v1..v5."""
    n_show = min(5, len(results["raw"]))

    fig, axes = plt.subplots(n_show, 2, figsize=(12, 2.5*n_show), squeeze=False)
    fig.suptitle(f"Raw Output vs Discrete-Log Fourier Spectra — {op_name}\n"
                 f"Left: group by (a*b) mod {P}  |  Right: group by dlog(a)+dlog(b) mod {P-1}",
                 fontsize=12, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#95a5a6", "#95a5a6", "#95a5a6"]

    for k in range(n_show):
        # Raw
        ax = axes[k, 0]
        raw = results["raw"][k]
        ax.fill_between(raw["freqs"], raw["power"], alpha=0.3, color=colors[k])
        ax.plot(raw["freqs"], raw["power"], color=colors[k], linewidth=1.2)
        peak_i = np.argmax(raw["power"])
        ax.axvline(raw["freqs"][peak_i], color=colors[k], ls="--", lw=1.5, alpha=0.7)
        ax.text(0.98, 0.92, f"v{k+1}  ω={raw['peak_freq']}  F={raw['F_k']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_xlim(0, raw["freqs"][-1])
        ax.set_ylim(0, None)
        if k == 0: ax.set_title("Raw output basis", fontsize=10)
        if k == n_show-1: ax.set_xlabel("Frequency")
        ax.set_ylabel(f"v{k+1}", fontsize=10, fontweight="bold")

        # Dlog
        ax = axes[k, 1]
        dl = results["dlog"][k]
        ax.fill_between(dl["freqs"], dl["power"], alpha=0.3, color=colors[k])
        ax.plot(dl["freqs"], dl["power"], color=colors[k], linewidth=1.2)
        peak_i = np.argmax(dl["power"])
        ax.axvline(dl["freqs"][peak_i], color=colors[k], ls="--", lw=1.5, alpha=0.7)
        ax.text(0.98, 0.92, f"v{k+1}  ω={dl['peak_freq']}  F={dl['F_k']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_xlim(0, dl["freqs"][-1])
        ax.set_ylim(0, None)
        if k == 0: ax.set_title("Discrete-log basis", fontsize=10)
        if k == n_show-1: ax.set_xlabel("Frequency")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figA_dlog_vs_raw_spectra_{op_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figA_dlog_vs_raw_spectra_{op_name}.png")


def plot_figB_concentration(results, op_name):
    """F_k under raw vs dlog basis for all directions."""
    n_dir = len(results["raw"])

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_dir)
    w = 0.35

    raw_F  = [results["raw"][k]["F_k"] for k in range(n_dir)]
    dlog_F = [results["dlog"][k]["F_k"] for k in range(n_dir)]

    colors_raw  = ["#e74c3c" if k < 3 else "#d4a5a5" for k in range(n_dir)]
    colors_dlog = ["#2980b9" if k < 3 else "#a5c8d4" for k in range(n_dir)]

    bars1 = ax.bar(x - w/2, raw_F,  w, color=colors_raw,  alpha=0.85,
                   edgecolor="k", linewidth=0.4, label="Raw output basis")
    bars2 = ax.bar(x + w/2, dlog_F, w, color=colors_dlog, alpha=0.85,
                   edgecolor="k", linewidth=0.4, label="Discrete-log basis")

    # Improvement annotations for top-3
    for k in range(min(3, n_dir)):
        ratio = dlog_F[k] / (raw_F[k] + 1e-30)
        ax.annotate(f"{ratio:.1f}×", xy=(x[k] + w/2, dlog_F[k]),
                   xytext=(0, 5), textcoords="offset points",
                   ha="center", fontsize=8, fontweight="bold",
                   color="#2980b9" if ratio > 1.5 else "#666")

    ax.axhline(1.0 / (P // 2), color="k", linestyle=":", linewidth=1,
               alpha=0.5, label="Uniform (raw, P//2)")
    ax.axhline(1.0 / ((P-1) // 2), color="gray", linestyle=":", linewidth=1,
               alpha=0.5, label="Uniform (dlog, (P-1)//2)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"v{k+1}" for k in range(n_dir)], fontsize=9)
    ax.set_ylabel("Fourier concentration F_k")
    ax.set_title(f"Fourier Concentration: Raw vs Discrete-Log — {op_name}\n"
                 "Annotations show dlog/raw improvement ratio for top-3",
                 fontsize=11)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figB_concentration_{op_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figB_concentration_{op_name}.png")


def plot_figC_basis_test(results, op_name):
    """Basis test under dlog Fourier."""
    bt = results["basis_test_dlog"]
    freqs = bt["v1"]["freqs"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Basis Test Under Discrete-Log Fourier — {op_name}\n"
                 "Does v₁+v₂ spectrum ≈ union of v₁ and v₂ peaks?",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(freqs, bt["v1"]["power"], color="#e74c3c", lw=1.5, alpha=0.8,
            label=f"v₁ (ω={bt['v1']['peak_freq']})")
    ax.plot(freqs, bt["v2"]["power"], color="#3498db", lw=1.5, alpha=0.8,
            label=f"v₂ (ω={bt['v2']['peak_freq']})")
    ax.plot(freqs, bt["v1+v2"]["power"], color="#8e44ad", lw=2.5, alpha=0.9,
            label=f"v₁+v₂ (ω={bt['v1+v2']['peak_freq']})")
    ax.axhline(1.0/len(freqs), color="k", ls=":", alpha=0.4)
    ax.set_xlabel("Frequency (dlog basis)")
    ax.set_ylabel("Normalised power")
    ax.set_title("v₁, v₂, and v₁+v₂")
    ax.legend(fontsize=8)
    ax.set_xlim(0, freqs[-1])

    ax = axes[1]
    ax.plot(freqs, bt["v1"]["power"], color="#e74c3c", lw=1, alpha=0.5, label="v₁")
    ax.plot(freqs, bt["v2"]["power"], color="#3498db", lw=1, alpha=0.5, label="v₂")
    ax.plot(freqs, bt["v3"]["power"], color="#2ecc71", lw=1, alpha=0.5, label="v₃")
    ax.plot(freqs, bt["v1+v2+v3"]["power"], color="#2c3e50", lw=2.5, alpha=0.9,
            label=f"v₁+v₂+v₃ (ω={bt['v1+v2+v3']['peak_freq']})")
    ax.axhline(1.0/len(freqs), color="k", ls=":", alpha=0.4)
    ax.set_xlabel("Frequency (dlog basis)")
    ax.set_ylabel("Normalised power")
    ax.set_title("v₁+v₂+v₃ combined")
    ax.legend(fontsize=8)
    ax.set_xlim(0, freqs[-1])

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figC_dlog_basis_test_{op_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figC_dlog_basis_test_{op_name}.png")


def plot_figD_2d_fourier(power_2d, freqs, op_name, dir_label):
    """2D Fourier heatmap P(ω_a, ω_b)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(power_2d, origin="lower", aspect="equal",
                   extent=[0.5, freqs[-1]+0.5, 0.5, freqs[-1]+0.5],
                   cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Normalised power")

    # Mark peak
    peak = np.unravel_index(np.argmax(power_2d), power_2d.shape)
    ax.plot(freqs[peak[1]], freqs[peak[0]], "c*", markersize=15,
            markeredgecolor="white", markeredgewidth=1)
    ax.set_xlabel("ω_b (frequency on input b)")
    ax.set_ylabel("ω_a (frequency on input a)")
    ax.set_title(f"2D Fourier: P(ω_a, ω_b) for {dir_label} — {op_name}\n"
                 f"Peak at ({freqs[peak[0]]}, {freqs[peak[1]]})",
                 fontsize=11)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figD_2d_fourier_{op_name}_{dir_label}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figD_2d_fourier_{op_name}_{dir_label}.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def run_op(op_name, device, dlog_table=None):
    sweep_path = SWEEP_DIR / f"{op_name}_wd1.0_s42.pt"
    cache_path = CACHE_PATHS[op_name]
    if not sweep_path.exists() or not cache_path.exists():
        print(f"  Skipping {op_name}: missing data")
        return None

    print(f"\n{'='*60}")
    print(f"  {op_name}")
    print(f"{'='*60}")

    sweep_data = torch.load(sweep_path, map_location="cpu", weights_only=False)
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    attn_logs  = sweep_data["attn_logs"]

    if not cache_data.get("metrics"):
        cache_data["metrics"] = sweep_data.get("metrics", sweep_data.get("log", []))

    updates    = compute_updates(attn_logs)
    test_pairs = [(int(a), int(b)) for a, b in cache_data["test_pairs"]]

    t_idx, step = select_postgrok(attn_logs, cache_data, updates)
    print(f"  Post-grok checkpoint: step={step}")

    svd_result = window_svd(updates, t_idx)
    if svd_result is None:
        return None
    S, Vt = svd_result
    n_dir = min(N_DIR, Vt.shape[0])

    _, state_dict = find_nearest_ckpt(step, cache_data)
    model = ModOpTransformer().to(device)

    # ── Grouping variables ──────────────────────────────────────────────────
    if op_name == "mul":
        raw_groups = np.array([(int(a)*int(b)) % P for a, b in test_pairs])
    elif op_name == "add":
        raw_groups = np.array([(int(a)+int(b)) % P for a, b in test_pairs])
    else:
        raw_groups = np.array([0]*len(test_pairs))

    # Dlog groups (only for mul, or as alternate basis for add)
    if dlog_table is not None:
        g, dlog = dlog_table
        if op_name == "mul":
            # dlog(a*b) = dlog(a) + dlog(b) mod (p-1)
            # Filter: need a != 0 and b != 0
            dlog_groups = []
            valid_mask  = []
            for a, b in test_pairs:
                if a == 0 or b == 0:
                    dlog_groups.append(0)
                    valid_mask.append(False)
                else:
                    dlog_groups.append((dlog[a] + dlog[b]) % (P - 1))
                    valid_mask.append(True)
            dlog_groups = np.array(dlog_groups)
            valid_mask  = np.array(valid_mask)
        elif op_name == "add":
            # For add, dlog basis doesn't have natural meaning — just use
            # dlog(a+b mod p) as grouping (for comparison)
            dlog_groups = []
            valid_mask  = []
            for a, b in test_pairs:
                s = (a + b) % P
                if s == 0:
                    dlog_groups.append(0)
                    valid_mask.append(False)
                else:
                    dlog_groups.append(dlog[s])
                    valid_mask.append(True)
            dlog_groups = np.array(dlog_groups)
            valid_mask  = np.array(valid_mask)
    else:
        dlog_groups = None
        valid_mask  = None

    # ── 1. Fourier profiles for all directions ─────────────────────────────
    print(f"\n  Fourier profiles (raw and dlog) for v1..v{n_dir}:")
    raw_profiles  = {}
    dlog_profiles = {}

    for k in range(n_dir):
        dh = compute_delta_h(model, state_dict, Vt[k], test_pairs, device)

        # Raw
        raw_profiles[k] = fourier_profile(dh, raw_groups, P)

        # Dlog
        if dlog_groups is not None:
            dh_valid = dh[valid_mask]
            dlog_valid = dlog_groups[valid_mask]
            dlog_profiles[k] = fourier_profile(dh_valid, dlog_valid, P - 1)

        tag = "***" if k < 3 else "   "
        raw_str  = f"raw: ω={raw_profiles[k]['peak_freq']:3d}  F={raw_profiles[k]['F_k']:.4f}"
        dlog_str = ""
        if k in dlog_profiles:
            dlog_str = f"  dlog: ω={dlog_profiles[k]['peak_freq']:3d}  F={dlog_profiles[k]['F_k']:.4f}"
        print(f"  {tag} v{k+1}: {raw_str}{dlog_str}")

    # Concentration summary
    top3_raw  = np.mean([raw_profiles[k]["F_k"] for k in range(min(3, n_dir))])
    bulk_raw  = np.mean([raw_profiles[k]["F_k"] for k in range(3, n_dir)])
    print(f"\n  Raw concentration:  top-3={top3_raw:.4f}  bulk={bulk_raw:.4f}  "
          f"ratio={top3_raw/(bulk_raw+1e-30):.2f}x")

    if dlog_profiles:
        top3_dlog = np.mean([dlog_profiles[k]["F_k"] for k in range(min(3, n_dir))])
        bulk_dlog = np.mean([dlog_profiles[k]["F_k"] for k in range(3, n_dir)])
        print(f"  Dlog concentration: top-3={top3_dlog:.4f}  bulk={bulk_dlog:.4f}  "
              f"ratio={top3_dlog/(bulk_dlog+1e-30):.2f}x")
        improve = top3_dlog / (top3_raw + 1e-30)
        print(f"  Dlog/Raw improvement for top-3: {improve:.2f}x")

    # ── 2. Basis test (dlog) ───────────────────────────────────────────────
    if dlog_groups is not None:
        print(f"\n  Basis test under dlog Fourier...")
        bt = {}
        for k in range(3):
            dh = compute_delta_h(model, state_dict, Vt[k], test_pairs, device)
            dh_v = dh[valid_mask]
            bt[f"v{k+1}"] = fourier_profile(dh_v, dlog_groups[valid_mask], P - 1)

        for label, indices in [("v1+v2", [0,1]), ("v1+v3", [0,2]),
                                ("v2+v3", [1,2]), ("v1+v2+v3", [0,1,2])]:
            v_sum = sum(Vt[idx] for idx in indices)
            v_sum /= (np.linalg.norm(v_sum) + 1e-30)
            dh = compute_delta_h(model, state_dict, v_sum, test_pairs, device)
            dh_v = dh[valid_mask]
            bt[label] = fourier_profile(dh_v, dlog_groups[valid_mask], P - 1)

        for label in ["v1", "v2", "v3", "v1+v2", "v1+v3", "v2+v3", "v1+v2+v3"]:
            r = bt[label]
            print(f"    {label:10s}: peak={r['peak_freq']:3d}  F={r['F_k']:.4f}  "
                  f"top3={r['top3_freqs']}")
    else:
        bt = None

    # ── 3. 2D Fourier for v1, v2, v3 ──────────────────────────────────────
    print(f"\n  2D Fourier for v1..v3:")
    power_2ds = {}
    for k in range(3):
        dh = compute_delta_h(model, state_dict, Vt[k], test_pairs, device)
        p2d, freqs_2d = fourier_2d(dh, test_pairs)
        power_2ds[k] = p2d

        peak = np.unravel_index(np.argmax(p2d), p2d.shape)
        peak_power = float(p2d[peak])

        # Diagonal vs off-diagonal power
        n_f = p2d.shape[0]
        diag_mask = np.eye(n_f, dtype=bool)
        diag_power = float(p2d[diag_mask].sum())
        off_power  = float(p2d[~diag_mask].sum())

        print(f"    v{k+1}: peak=({freqs_2d[peak[0]]},{freqs_2d[peak[1]]})  "
              f"peak_power={peak_power:.4f}  "
              f"diagonal={diag_power:.3f}  off-diag={off_power:.3f}")

    return {
        "raw":  raw_profiles,
        "dlog": dlog_profiles,
        "basis_test_dlog": bt,
        "power_2d": power_2ds,
        "freqs_2d": freqs_2d,
    }


def main():
    device = get_device()
    print(f"Device: {device}")

    g, dlog = build_dlog_table(P)
    print(f"Primitive root mod {P}: g={g}")
    print(f"Dlog table built: {len(dlog)} entries")

    # ── Run mul (primary) ──────────────────────────────────────────────────
    mul_results = run_op("mul", device, dlog_table=(g, dlog))

    # ── Run add (control) ──────────────────────────────────────────────────
    add_results = run_op("add", device, dlog_table=(g, dlog))

    # ── Plots ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating figures...")
    print(f"{'='*60}")

    if mul_results:
        plot_figA_spectra(mul_results, "mul")
        plot_figB_concentration(mul_results, "mul")
        if mul_results["basis_test_dlog"]:
            plot_figC_basis_test(mul_results, "mul")
        for k in range(3):
            if k in mul_results["power_2d"]:
                plot_figD_2d_fourier(mul_results["power_2d"][k],
                                    mul_results["freqs_2d"], "mul", f"v{k+1}")

    if add_results:
        plot_figA_spectra(add_results, "add")
        plot_figB_concentration(add_results, "add")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DISCRETE-LOG FOURIER SUMMARY")
    print(f"{'='*60}")

    for op_name, res in [("mul", mul_results), ("add", add_results)]:
        if res is None: continue
        print(f"\n  {op_name}:")
        print(f"  {'Dir':>4} {'Raw ω':>6} {'Raw F':>8} {'Dlog ω':>7} {'Dlog F':>8} {'Improve':>8}")
        print(f"  {'-'*45}")
        for k in range(len(res["raw"])):
            r = res["raw"][k]
            d = res["dlog"].get(k)
            tag = " *" if k < 3 else "  "
            dlog_str = ""
            improve_str = ""
            if d:
                dlog_str = f"{d['peak_freq']:7d} {d['F_k']:8.4f}"
                improve_str = f"{d['F_k']/(r['F_k']+1e-30):7.1f}×"
            print(f"  v{k+1}{tag} {r['peak_freq']:6d} {r['F_k']:8.4f} {dlog_str} {improve_str}")

    print(f"\nResults saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
