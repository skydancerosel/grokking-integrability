#!/usr/bin/env python3
"""
fourier_functional_view.py

The functional Fourier view of the {v1, v2, v3} spectral edge.

Previous experiments established:
  - v_k are NOT head-localized (purity ~1/8 uniform)
  - v_k are diffuse in activation space (rank_eff ~40)
  - SAE fails to capture the structure (Jaccard not significant vs null)

This script makes Fourier modes the primary object.  For each v_k:
  1. Peak frequency, Fourier concentration F_k, top-3 frequencies
  2. Basis test: does g_{v1+v2}(x) have spectrum ≈ union of peaks?
  3. Cross-task comparison: add → additive, mul → multiplicative, x²+y² → quadratic

Output
------
spectral/fourier_functional_plots/
  figA_fourier_profiles.png        — peak freq + concentration for v1..v8, all ops
  figB_basis_test.png              — spectrum of v1+v2 vs union of v1, v2
  figC_cross_task_frequencies.png  — which frequencies each task uses
  figD_concentration_vs_bulk.png   — top-3 vs bulk Fourier concentration
  figE_spectra_gallery.png         — raw spectra for all v_k, all ops
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
RESULTS_DIR = SCRIPT_DIR / "coherence_edge_results"
SWEEP_DIR   = SCRIPT_DIR.parent / "grok_sweep_results"
PLOT_DIR    = SCRIPT_DIR / "fourier_functional_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────

P        = 97
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 2
W        = 20       # SVD window
N_DIR    = 8        # directions to analyse
EPS_SCALE = 0.005

CACHE_PATHS = {
    "add":   RESULTS_DIR / "training_cache.pt",
    "sub":   RESULTS_DIR / "training_cache_sub.pt",
    "mul":   RESULTS_DIR / "training_cache_mul.pt",
    "x2_y2": RESULTS_DIR / "training_cache_x2_y2.pt",
}

# ─── Operations ──────────────────────────────────────────────────────────────

OPERATIONS = {
    "add":   {"label": "(a+b) mod p",    "fourier_arg": lambda a, b: (a + b) % P},
    "sub":   {"label": "(a-b) mod p",    "fourier_arg": lambda a, b: (a - b) % P},
    "mul":   {"label": "(a*b) mod p",    "fourier_arg": lambda a, b: (a * b) % P},
    "x2_y2": {"label": "(a²+b²) mod p", "fourier_arg": lambda a, b: (a*a + b*b) % P},
}

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


# ─── Checkpoint selection ────────────────────────────────────────────────────

def select_postgrok(attn_logs, cache_data, updates):
    """Select a single post-grok checkpoint."""
    attn_steps   = [e["step"] for e in attn_logs]
    update_steps = attn_steps[1:]
    metrics      = cache_data["metrics"]
    n            = len(updates)

    grok_step = next((m["step"] for m in metrics if m["test_acc"] > 0.5),
                     metrics[-1]["step"])

    # Find last step with test_acc > 0.95
    post = next((t for t in range(n-1, -1, -1)
                 if min(metrics, key=lambda m: abs(m["step"] - update_steps[t]))["test_acc"] > 0.95),
                n-1)
    return post, update_steps[post]


def find_nearest_ckpt(step, cache_data):
    return min(cache_data["checkpoints"], key=lambda cs: abs(cs[0] - step))


# ─── Core: residual stream perturbation ──────────────────────────────────────

@torch.no_grad()
def compute_delta_h(model, state_dict, vk_np, test_pairs, device,
                    eps_scale=EPS_SCALE, batch_size=512):
    """Δh(x) = h(x; θ+εv_k) − h(x; θ) for all test inputs."""
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

    return delta_h, eps


# ─── Fourier analysis ───────────────────────────────────────────────────────

def fourier_profile(delta_h, test_pairs, fourier_arg_fn, p=P):
    """
    Fourier decomposition of ||Δh(x)||² grouped by output q = fourier_arg(a,b).

    Returns
    -------
    power      : array [n_freqs]  — normalised power spectrum
    peak_freq  : int              — dominant frequency
    F_k        : float            — Fourier concentration (top-1 power / total)
    top3_freqs : list[int]        — top-3 frequencies
    top3_power : list[float]      — power at top-3 frequencies
    """
    N = len(test_pairs)
    # Scalar signal: ||Δh(x)||² for each test input
    signal_per_input = np.sum(delta_h**2, axis=1)  # [N]

    # Group by output q
    q_vals = np.array([fourier_arg_fn(int(a), int(b)) for a, b in test_pairs])
    means  = np.zeros(p, dtype=np.float64)
    counts = np.zeros(p, dtype=np.float64)
    for i in range(N):
        q = int(q_vals[i]) % p
        means[q]  += signal_per_input[i]
        counts[q] += 1
    means /= np.maximum(counts, 1)

    # DFT
    freqs  = np.arange(1, p // 2 + 1)
    phases = np.outer(np.arange(p, dtype=float), freqs) * (2 * np.pi / p)
    cos_coeffs = means @ np.cos(phases) / p
    sin_coeffs = means @ np.sin(phases) / p
    power = cos_coeffs**2 + sin_coeffs**2
    power_norm = power / (power.sum() + 1e-30)

    peak_idx   = int(np.argmax(power_norm))
    peak_freq  = int(freqs[peak_idx])
    F_k        = float(power_norm[peak_idx])

    top3_idx   = np.argsort(power_norm)[::-1][:3]
    top3_freqs = [int(freqs[i]) for i in top3_idx]
    top3_power = [float(power_norm[i]) for i in top3_idx]

    return power_norm, peak_freq, F_k, top3_freqs, top3_power


def fourier_profile_directional(delta_h, test_pairs, fourier_arg_fn, p=P):
    """
    Like fourier_profile but analyses along each of the top-3 PCA directions
    of delta_h separately.  Returns per-PC Fourier profiles.
    """
    _, S, Vt = np.linalg.svd(delta_h, full_matrices=False)
    n_pc = min(3, Vt.shape[0])
    results = []
    for pc in range(n_pc):
        # Project delta_h onto this PC
        proj = delta_h @ Vt[pc]  # [N] scalar projection
        # Use proj² as signal
        signal = proj**2
        q_vals = np.array([fourier_arg_fn(int(a), int(b)) for a, b in test_pairs])
        means  = np.zeros(p, dtype=np.float64)
        counts = np.zeros(p, dtype=np.float64)
        for i in range(len(test_pairs)):
            q = int(q_vals[i]) % p
            means[q]  += signal[i]
            counts[q] += 1
        means /= np.maximum(counts, 1)

        freqs  = np.arange(1, p // 2 + 1)
        phases = np.outer(np.arange(p, dtype=float), freqs) * (2 * np.pi / p)
        cos_c  = means @ np.cos(phases) / p
        sin_c  = means @ np.sin(phases) / p
        power  = cos_c**2 + sin_c**2
        power /= (power.sum() + 1e-30)

        peak_idx  = int(np.argmax(power))
        peak_freq = int(freqs[peak_idx])
        results.append({
            "pc": pc,
            "var_frac": float(S[pc]**2 / (S**2).sum()),
            "peak_freq": peak_freq,
            "concentration": float(power[peak_idx]),
            "power": power,
        })
    return results


# ─── Basis test ──────────────────────────────────────────────────────────────

def basis_test(model, state_dict, Vt, test_pairs, device, fourier_arg_fn):
    """
    Test: does perturbing along (v1 + v2) / ||v1 + v2|| give a spectrum
    that is the union of v1's and v2's peaks?

    Also test v1+v3, v2+v3, and v1+v2+v3.
    """
    results = {}

    # Individual spectra (re-use)
    for k in range(3):
        dh, _ = compute_delta_h(model, state_dict, Vt[k], test_pairs, device)
        pw, pf, fk, t3f, t3p = fourier_profile(dh, test_pairs, fourier_arg_fn)
        results[f"v{k+1}"] = {
            "power": pw, "peak_freq": pf, "F_k": fk,
            "top3_freqs": t3f, "top3_power": t3p,
        }

    # Combined directions
    combos = [
        ("v1+v2",     [0, 1]),
        ("v1+v3",     [0, 2]),
        ("v2+v3",     [1, 2]),
        ("v1+v2+v3",  [0, 1, 2]),
    ]

    for label, indices in combos:
        v_sum = sum(Vt[k] for k in indices)
        v_sum = v_sum / (np.linalg.norm(v_sum) + 1e-30)
        dh, _ = compute_delta_h(model, state_dict, v_sum, test_pairs, device)
        pw, pf, fk, t3f, t3p = fourier_profile(dh, test_pairs, fourier_arg_fn)
        results[label] = {
            "power": pw, "peak_freq": pf, "F_k": fk,
            "top3_freqs": t3f, "top3_power": t3p,
        }

    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_figA_profiles(all_ops):
    """Peak frequency + concentration for v1..v8, all ops side by side."""
    ops = [op for op in OPERATIONS if op in all_ops]
    n_ops = len(ops)
    if not n_ops: return

    fig, axes = plt.subplots(2, n_ops, figsize=(4.5*n_ops, 7), squeeze=False)
    fig.suptitle("Fourier Profile of Parameter-Space Directions v₁…v₈\n"
                 "Top: peak frequency  |  Bottom: Fourier concentration F_k",
                 fontsize=12, fontweight="bold")

    for col, op in enumerate(ops):
        data = all_ops[op]
        n_dir = len(data["profiles"])

        ks     = list(range(n_dir))
        peaks  = [data["profiles"][k]["peak_freq"] for k in ks]
        concs  = [data["profiles"][k]["F_k"] for k in ks]
        colors = ["#e74c3c" if k < 3 else "#95a5a6" for k in ks]

        # Top: peak frequency
        ax = axes[0, col]
        ax.bar(ks, peaks, color=colors, alpha=0.85, edgecolor="k", linewidth=0.4)
        ax.set_ylabel("Peak Fourier frequency")
        ax.set_title(OPERATIONS[op]["label"], fontsize=11, fontweight="bold")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"v{k+1}" for k in ks], fontsize=8)

        # Bottom: concentration
        ax = axes[1, col]
        ax.bar(ks, concs, color=colors, alpha=0.85, edgecolor="k", linewidth=0.4)
        ax.axhline(1.0 / (P // 2), color="k", linestyle=":", linewidth=0.8,
                   alpha=0.5, label="uniform")
        ax.set_ylabel("Fourier concentration F_k")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"v{k+1}" for k in ks], fontsize=8)
        if col == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figA_fourier_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figA_fourier_profiles.png")


def plot_figB_basis_test(all_ops):
    """Spectrum of v1+v2 vs union of v1, v2 for each op."""
    ops = [op for op in OPERATIONS if op in all_ops and "basis_test" in all_ops[op]]
    if not ops: return

    freqs = np.arange(1, P // 2 + 1)
    n_ops = len(ops)

    fig, axes = plt.subplots(2, n_ops, figsize=(5*n_ops, 7), squeeze=False)
    fig.suptitle("Basis Test: Does v₁+v₂ Have Spectrum ≈ Union of v₁ and v₂ Peaks?\n"
                 "Top: individual + combined spectra  |  Bottom: v₁+v₂+v₃ spectrum",
                 fontsize=11, fontweight="bold")

    for col, op in enumerate(ops):
        bt = all_ops[op]["basis_test"]

        # Top panel: v1, v2, and v1+v2
        ax = axes[0, col]
        ax.plot(freqs, bt["v1"]["power"], color="#e74c3c", linewidth=1.5,
                alpha=0.8, label=f"v₁ (peak={bt['v1']['peak_freq']})")
        ax.plot(freqs, bt["v2"]["power"], color="#3498db", linewidth=1.5,
                alpha=0.8, label=f"v₂ (peak={bt['v2']['peak_freq']})")
        ax.plot(freqs, bt["v1+v2"]["power"], color="#8e44ad", linewidth=2.5,
                alpha=0.9, label=f"v₁+v₂ (peak={bt['v1+v2']['peak_freq']})")
        ax.axhline(1.0/len(freqs), color="k", linestyle=":", alpha=0.4)
        ax.set_ylabel("Normalised power")
        ax.set_title(OPERATIONS[op]["label"], fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_xlim(0, freqs[-1])

        # Bottom panel: v1+v2+v3
        ax = axes[1, col]
        ax.plot(freqs, bt["v1"]["power"], color="#e74c3c", linewidth=1.0,
                alpha=0.5, label="v₁")
        ax.plot(freqs, bt["v2"]["power"], color="#3498db", linewidth=1.0,
                alpha=0.5, label="v₂")
        ax.plot(freqs, bt["v3"]["power"], color="#2ecc71", linewidth=1.0,
                alpha=0.5, label="v₃")
        ax.plot(freqs, bt["v1+v2+v3"]["power"], color="#2c3e50", linewidth=2.5,
                alpha=0.9, label=f"v₁+v₂+v₃ (peak={bt['v1+v2+v3']['peak_freq']})")
        ax.axhline(1.0/len(freqs), color="k", linestyle=":", alpha=0.4)
        ax.set_xlabel("Fourier frequency")
        ax.set_ylabel("Normalised power")
        ax.legend(fontsize=7)
        ax.set_xlim(0, freqs[-1])

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figB_basis_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figB_basis_test.png")


def plot_figC_cross_task(all_ops):
    """Which frequencies each task uses — top-3 frequencies for v1, v2, v3."""
    ops = [op for op in OPERATIONS if op in all_ops]
    if not ops: return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Cross-Task Frequency Comparison\n"
                 "Top-3 Fourier frequencies for v₁, v₂, v₃ per operation",
                 fontsize=12, fontweight="bold")

    y_pos = 0
    y_labels = []
    y_ticks  = []
    dir_colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}
    dir_labels = {0: "v₁", 1: "v₂", 2: "v₃"}

    for op in ops:
        data = all_ops[op]
        for k in range(3):
            prof = data["profiles"][k]
            top3 = prof["top3_freqs"]
            top3_pw = prof["top3_power"]

            # Size proportional to power
            sizes = [pw * 3000 for pw in top3_pw]
            ax.scatter(top3, [y_pos]*3, s=sizes, c=dir_colors[k],
                      alpha=0.7, edgecolors="k", linewidth=0.5,
                      label=f"{dir_labels[k]}" if op == ops[0] else "")

            y_labels.append(f"{OPERATIONS[op]['label']}  {dir_labels[k]}")
            y_ticks.append(y_pos)
            y_pos += 1
        y_pos += 0.5  # gap between ops

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Fourier frequency", fontsize=10)
    ax.set_xlim(0, P // 2 + 1)
    ax.axvline(P//4, color="gray", linestyle="--", alpha=0.3, label="p/4")

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figC_cross_task_frequencies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figC_cross_task_frequencies.png")


def plot_figD_concentration(all_ops):
    """Top-3 vs bulk Fourier concentration."""
    ops = [op for op in OPERATIONS if op in all_ops]
    if not ops: return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ops))
    w = 0.35

    top3_means = []
    bulk_means = []
    for op in ops:
        data = all_ops[op]
        n_dir = len(data["profiles"])
        top3_c = [data["profiles"][k]["F_k"] for k in range(min(3, n_dir))]
        bulk_c = [data["profiles"][k]["F_k"] for k in range(3, n_dir)]
        top3_means.append(np.mean(top3_c))
        bulk_means.append(np.mean(bulk_c) if bulk_c else 0)

    ax.bar(x - w/2, top3_means, w, color="#e74c3c", alpha=0.8, edgecolor="k",
           linewidth=0.5, label="Top-3 (above edge)")
    ax.bar(x + w/2, bulk_means, w, color="#95a5a6", alpha=0.8, edgecolor="k",
           linewidth=0.5, label="Bulk (below edge)")
    ax.axhline(1.0 / (P // 2), color="k", linestyle=":", linewidth=1,
               alpha=0.5, label="Uniform baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([OPERATIONS[op]["label"] for op in ops], fontsize=9)
    ax.set_ylabel("Mean Fourier concentration F_k")
    ax.set_title("Fourier Concentration: Top-3 vs Bulk Directions\n"
                 "Higher = more frequency-selective perturbation", fontsize=11)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figD_concentration_vs_bulk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figD_concentration_vs_bulk.png")


def plot_figE_spectra_gallery(all_ops):
    """Raw spectra for v1..v5, all ops."""
    ops = [op for op in OPERATIONS if op in all_ops]
    if not ops: return

    freqs = np.arange(1, P // 2 + 1)
    n_ops = len(ops)
    n_show = min(5, max(len(all_ops[op]["profiles"]) for op in ops))

    fig, axes = plt.subplots(n_show, n_ops, figsize=(4*n_ops, 2.5*n_show), squeeze=False)
    fig.suptitle("Fourier Power Spectra of ||Δh(x)||² Grouped by Output\n"
                 "Each row = one direction v_k; each column = one operation",
                 fontsize=12, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#95a5a6", "#95a5a6", "#95a5a6"]

    for col, op in enumerate(ops):
        data = all_ops[op]
        for k in range(min(n_show, len(data["profiles"]))):
            ax = axes[k, col]
            prof = data["profiles"][k]
            ax.fill_between(freqs, prof["power"], alpha=0.4, color=colors[k])
            ax.plot(freqs, prof["power"], color=colors[k], linewidth=1.2)

            # Mark peak
            peak_idx = np.argmax(prof["power"])
            ax.axvline(freqs[peak_idx], color=colors[k], linestyle="--",
                      linewidth=1.5, alpha=0.7)
            ax.text(freqs[peak_idx]+1, prof["power"].max()*0.9,
                   f"ω={prof['peak_freq']}\nF={prof['F_k']:.3f}",
                   fontsize=7, color=colors[k])

            ax.axhline(1.0/len(freqs), color="k", linestyle=":", alpha=0.3)
            ax.set_xlim(0, freqs[-1])
            ax.set_ylim(0, None)

            if col == 0:
                ax.set_ylabel(f"v{k+1}", fontsize=10, fontweight="bold")
            if k == 0:
                ax.set_title(OPERATIONS[op]["label"], fontsize=10, fontweight="bold")
            if k == n_show - 1:
                ax.set_xlabel("Frequency")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figE_spectra_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figE_spectra_gallery.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def run_op(op_name, op_cfg, device):
    sweep_path = SWEEP_DIR / f"{op_name}_wd1.0_s42.pt"
    cache_path = CACHE_PATHS[op_name]
    if not sweep_path.exists() or not cache_path.exists():
        print(f"  Skipping {op_name}: missing data")
        return None

    print(f"\n{'='*60}")
    print(f"  {op_name} — {op_cfg['label']}")
    print(f"{'='*60}")

    sweep_data = torch.load(sweep_path, map_location="cpu", weights_only=False)
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    attn_logs  = sweep_data["attn_logs"]

    if not cache_data.get("metrics"):
        cache_data["metrics"] = sweep_data.get("metrics", sweep_data.get("log", []))

    updates    = compute_updates(attn_logs)
    test_pairs = [(int(a), int(b)) for a, b in cache_data["test_pairs"]]

    t_idx, step = select_postgrok(attn_logs, cache_data, updates)
    print(f"  Post-grok checkpoint: t_idx={t_idx}, step={step}")

    svd_result = window_svd(updates, t_idx)
    if svd_result is None:
        print("  SVD failed")
        return None
    S, Vt = svd_result
    n_dir = min(N_DIR, Vt.shape[0])

    _, state_dict = find_nearest_ckpt(step, cache_data)
    model = ModOpTransformer().to(device)

    fourier_arg_fn = op_cfg["fourier_arg"]

    # ── 1. Fourier profiles for all directions ─────────────────────────────
    print(f"\n  Computing Fourier profiles for v1..v{n_dir}...")
    profiles = {}
    for k in range(n_dir):
        dh, eps = compute_delta_h(model, state_dict, Vt[k], test_pairs, device)
        pw, peak_freq, F_k, top3_f, top3_p = fourier_profile(
            dh, test_pairs, fourier_arg_fn)

        profiles[k] = {
            "power":      pw,
            "peak_freq":  peak_freq,
            "F_k":        F_k,
            "top3_freqs": top3_f,
            "top3_power": top3_p,
            "dh_norm":    float(np.linalg.norm(dh)),
        }

        tag = "***" if k < 3 else "   "
        print(f"  {tag} v{k+1}: peak_freq={peak_freq:3d}  F_k={F_k:.4f}  "
              f"top3={top3_f}  ||Δh||={profiles[k]['dh_norm']:.5f}")

    # Concentration ratio: top-3 mean / bulk mean
    top3_F = np.mean([profiles[k]["F_k"] for k in range(min(3, n_dir))])
    bulk_F = np.mean([profiles[k]["F_k"] for k in range(3, n_dir)]) if n_dir > 3 else 0
    print(f"\n  Fourier concentration: top-3 mean={top3_F:.4f}  bulk mean={bulk_F:.4f}  "
          f"ratio={top3_F/(bulk_F+1e-30):.2f}x")

    # ── 2. Basis test ──────────────────────────────────────────────────────
    print(f"\n  Running basis test (v1+v2, v1+v3, v2+v3, v1+v2+v3)...")
    bt = basis_test(model, state_dict, Vt, test_pairs, device, fourier_arg_fn)

    for label in ["v1", "v2", "v3", "v1+v2", "v1+v3", "v2+v3", "v1+v2+v3"]:
        r = bt[label]
        print(f"    {label:10s}: peak={r['peak_freq']:3d}  F={r['F_k']:.4f}  "
              f"top3={r['top3_freqs']}")

    # ── 3. Per-PC Fourier analysis for v1 ──────────────────────────────────
    print(f"\n  Per-PC Fourier analysis for v1...")
    dh_v1, _ = compute_delta_h(model, state_dict, Vt[0], test_pairs, device)
    pc_results = fourier_profile_directional(dh_v1, test_pairs, fourier_arg_fn)
    for pcr in pc_results:
        print(f"    PC{pcr['pc']+1}: var_frac={pcr['var_frac']:.3f}  "
              f"peak_freq={pcr['peak_freq']}  concentration={pcr['concentration']:.4f}")

    return {
        "profiles":   profiles,
        "basis_test": bt,
        "pc_fourier": pc_results,
        "step":       step,
        "S":          S.tolist(),
    }


def main():
    device = get_device()
    print(f"Device: {device}")

    all_ops = {}

    for op_name, op_cfg in OPERATIONS.items():
        result = run_op(op_name, op_cfg, device)
        if result is not None:
            all_ops[op_name] = result

    # ── Plots ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating figures...")
    print(f"{'='*60}")

    plot_figA_profiles(all_ops)
    plot_figB_basis_test(all_ops)
    plot_figC_cross_task(all_ops)
    plot_figD_concentration(all_ops)
    plot_figE_spectra_gallery(all_ops)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FOURIER FUNCTIONAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Op':<8} {'Dir':>4} {'Peak':>5} {'F_k':>8} {'Top-3 freqs':>25}")
    print("-" * 55)
    for op in OPERATIONS:
        if op not in all_ops: continue
        for k in range(min(N_DIR, len(all_ops[op]["profiles"]))):
            p = all_ops[op]["profiles"][k]
            tag = " *" if k < 3 else "  "
            print(f"{op:<8} v{k+1}{tag} {p['peak_freq']:5d} {p['F_k']:8.4f} "
                  f"{str(p['top3_freqs']):>25}")

    print(f"\nBasis test results:")
    print(f"{'Op':<8} {'Combo':>12} {'Peak':>5} {'F_k':>8} {'Top-3 freqs':>25}")
    print("-" * 60)
    for op in OPERATIONS:
        if op not in all_ops or "basis_test" not in all_ops[op]: continue
        bt = all_ops[op]["basis_test"]
        for label in ["v1+v2", "v1+v3", "v2+v3", "v1+v2+v3"]:
            r = bt[label]
            print(f"{op:<8} {label:>12} {r['peak_freq']:5d} {r['F_k']:8.4f} "
                  f"{str(r['top3_freqs']):>25}")

    print(f"\nResults saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
