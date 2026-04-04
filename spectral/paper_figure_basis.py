#!/usr/bin/env python3
"""
paper_figure_basis.py

Generate the key paper figure: "Spectral edge structure depends on the functional basis"

Panel A: ADD — additive Fourier → single dominant mode (ω≈25-26)
Panel B: MUL — additive Fourier → flat/noisy (wrong basis)
Panel C: MUL — discrete-log Fourier → sharp peak at ω=29 (correct basis)
Panel D left:  SUB — additive Fourier → multi-mode {6, 16, 32}
Panel D right: x²+y² — additive Fourier → diffuse, no collapse
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "coherence_edge_results"
SWEEP_DIR   = SCRIPT_DIR.parent / "grok_sweep_results"
PLOT_DIR    = SCRIPT_DIR / "paper_figures"
PLOT_DIR.mkdir(exist_ok=True)

P, D_MODEL, N_HEADS, N_LAYERS, W = 97, 128, 4, 2, 20
EPS_SCALE = 0.005

CACHE_PATHS = {
    "add":   RESULTS_DIR / "training_cache.pt",
    "sub":   RESULTS_DIR / "training_cache_sub.pt",
    "mul":   RESULTS_DIR / "training_cache_mul.pt",
    "x2_y2": RESULTS_DIR / "training_cache_x2_y2.pt",
}

# ─── Number theory ──────────────────────────────────────────────────────────

def primitive_root(p):
    for g in range(2, p):
        seen, x = set(), 1
        for _ in range(p - 1):
            x = (x * g) % p; seen.add(x)
        if len(seen) == p - 1: return g

def build_dlog_table(p):
    g = primitive_root(p)
    dlog, x = {}, 1
    for k in range(p - 1):
        dlog[x] = k; x = (x * g) % p
    return g, dlog

# ─── Model ───────────────────────────────────────────────────────────────────

class ModOpTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(P, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(2, D_MODEL) / math.sqrt(D_MODEL))
        enc = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=256, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, P)

    def forward(self, a, b, return_residual=False):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        res = h[:, 0, :]
        logits = self.head(self.ln(res))
        return (logits, res) if return_residual else logits

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def get_attn_keys(model):
    return sorted(n for n, _ in model.named_parameters()
                  if "self_attn" in n and "weight" in n and "bias" not in n)

# ─── SVD ─────────────────────────────────────────────────────────────────────

def flatten_attn(entry):
    parts = []
    for ld in sorted(entry["layers"], key=lambda x: x["layer"]):
        for k in ["WQ", "WK", "WV", "WO"]:
            parts.append(ld[k].flatten().float())
    return torch.cat(parts)

def compute_updates(attn_logs):
    flat = [flatten_attn(e).numpy() for e in attn_logs]
    return [flat[i] - flat[i-1] for i in range(1, len(flat))]

def window_svd(updates, t_idx):
    start = max(0, t_idx - W + 1)
    X = np.stack(updates[start:t_idx+1])
    if X.shape[0] < 3: return None
    X -= X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S, Vt

def select_postgrok(attn_logs, cache_data, updates):
    steps = [e["step"] for e in attn_logs][1:]
    metrics = cache_data["metrics"]
    n = len(updates)
    post = next((t for t in range(n-1, -1, -1)
                 if min(metrics, key=lambda m: abs(m["step"] - steps[t]))["test_acc"] > 0.95), n-1)
    return post, steps[post]

def find_nearest_ckpt(step, cache_data):
    return min(cache_data["checkpoints"], key=lambda cs: abs(cs[0] - step))

# ─── Perturbation ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_delta_h(model, state_dict, vk_np, test_pairs, device, batch_size=512):
    attn_keys = get_attn_keys(model)
    flat = torch.cat([state_dict[k].float().flatten() for k in attn_keys])
    eps = EPS_SCALE * float(flat.norm())
    vk_t = torch.from_numpy(vk_np.copy()).float()
    vk_t = vk_t / (vk_t.norm() + 1e-30)
    pert_sd = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key in attn_keys:
        numel = pert_sd[key].numel()
        pert_sd[key] = pert_sd[key].float() + eps * vk_t[offset:offset+numel].reshape(pert_sd[key].shape)
        offset += numel
    N = len(test_pairs)
    delta_h = np.zeros((N, D_MODEL), dtype=np.float32)
    ab_all = torch.tensor(test_pairs)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ab = ab_all[start:end].to(device)
        a, b = ab[:, 0], ab[:, 1]
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()}); model.eval()
        _, h0 = model(a, b, return_residual=True)
        model.load_state_dict({k: v.to(device) for k, v in pert_sd.items()}); model.eval()
        _, h1 = model(a, b, return_residual=True)
        delta_h[start:end] = (h1 - h0).float().cpu().numpy()
    return delta_h

# ─── Fourier ────────────────────────────────────────────────────────────────

def fourier_spectrum(delta_h, group_vals, period):
    signal = np.sum(delta_h**2, axis=1)
    means = np.zeros(period, dtype=np.float64)
    counts = np.zeros(period, dtype=np.float64)
    for i, q in enumerate(group_vals):
        means[int(q) % period] += signal[i]
        counts[int(q) % period] += 1
    means /= np.maximum(counts, 1)
    freqs = np.arange(1, period // 2 + 1)
    phases = np.outer(np.arange(period, dtype=float), freqs) * (2 * np.pi / period)
    cos_c = means @ np.cos(phases) / period
    sin_c = means @ np.sin(phases) / period
    power = cos_c**2 + sin_c**2
    power /= (power.sum() + 1e-30)
    return freqs, power

# ─── Load and compute spectra ───────────────────────────────────────────────

def load_op(op_name, device):
    sweep = torch.load(SWEEP_DIR / f"{op_name}_wd1.0_s42.pt", map_location="cpu", weights_only=False)
    cache = torch.load(CACHE_PATHS[op_name], map_location="cpu", weights_only=False)
    if not cache.get("metrics"):
        cache["metrics"] = sweep.get("metrics", sweep.get("log", []))
    updates = compute_updates(sweep["attn_logs"])
    test_pairs = [(int(a), int(b)) for a, b in cache["test_pairs"]]
    t_idx, step = select_postgrok(sweep["attn_logs"], cache, updates)
    S, Vt = window_svd(updates, t_idx)
    _, sd = find_nearest_ckpt(step, cache)
    model = ModOpTransformer().to(device)
    return model, sd, Vt, test_pairs

def get_spectra(op_name, group_fn, period, device, dlog=None):
    """Compute Fourier spectra for v1-v3."""
    model, sd, Vt, test_pairs = load_op(op_name, device)
    group_vals = np.array([group_fn(int(a), int(b)) for a, b in test_pairs])

    # Handle valid mask for dlog-based groupings
    if dlog is not None:
        valid = group_vals >= 0  # sentinel
    else:
        valid = np.ones(len(test_pairs), dtype=bool)

    spectra = {}
    for k in range(3):
        dh = compute_delta_h(model, sd, Vt[k], test_pairs, device)
        freqs, power = fourier_spectrum(dh[valid], group_vals[valid], period)
        peak_idx = np.argmax(power)
        spectra[k] = {"freqs": freqs, "power": power,
                       "peak": int(freqs[peak_idx]), "F": float(power[peak_idx])}
        print(f"  v{k+1}: peak={spectra[k]['peak']}  F={spectra[k]['F']:.4f}")
    return spectra

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    gen, dlog = build_dlog_table(P)
    print(f"Primitive root: g={gen}")

    # Panel A: ADD — additive basis
    print("\nPanel A: ADD (additive basis)")
    add_spectra = get_spectra("add", lambda a, b: (a + b) % P, P, device)

    # Panel B: MUL — additive basis (wrong)
    print("\nPanel B: MUL (additive basis — wrong)")
    mul_raw_spectra = get_spectra("mul", lambda a, b: (a * b) % P, P, device)

    # Panel C: MUL — dlog basis (correct)
    print("\nPanel C: MUL (dlog basis — correct)")
    def mul_dlog_group(a, b):
        if a == 0 or b == 0: return -1  # sentinel
        return (dlog[a] + dlog[b]) % (P - 1)
    mul_dlog_spectra = get_spectra("mul", mul_dlog_group, P - 1, device, dlog=dlog)

    # Panel D left: SUB
    print("\nPanel D left: SUB (additive basis)")
    sub_spectra = get_spectra("sub", lambda a, b: (a - b) % P, P, device)

    # Panel D right: x²+y²
    print("\nPanel D right: x²+y² (additive basis)")
    x2y2_spectra = get_spectra("x2_y2", lambda a, b: (a*a + b*b) % P, P, device)

    # ── Plot ─────────────────────────────────────────────────────────────
    print("\nGenerating figure...")

    colors = {"v1": "#e74c3c", "v2": "#3498db", "v3": "#2ecc71"}

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # Panel A: ADD
    ax_a = fig.add_subplot(gs[0, 0])
    for k in range(3):
        s = add_spectra[k]
        ax_a.plot(s["freqs"], s["power"], color=colors[f"v{k+1}"],
                  linewidth=1.8, alpha=0.85, label=f"$v_{k+1}$")
    ax_a.axhline(1.0/(P//2), color="k", ls=":", lw=0.8, alpha=0.4)
    ax_a.set_title("(a) Addition — additive basis", fontsize=10, fontweight="bold")
    ax_a.set_xlabel("Frequency $\\omega$")
    ax_a.set_ylabel("Normalised power")
    ax_a.set_xlim(0, P//2)
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.text(0.03, 0.92, "Single dominant mode\n$\\omega \\approx 25$–$26$",
              transform=ax_a.transAxes, fontsize=8, va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", alpha=0.9))

    # Panel B: MUL wrong basis
    ax_b = fig.add_subplot(gs[0, 1])
    for k in range(3):
        s = mul_raw_spectra[k]
        ax_b.plot(s["freqs"], s["power"], color=colors[f"v{k+1}"],
                  linewidth=1.8, alpha=0.85, label=f"$v_{k+1}$")
    ax_b.axhline(1.0/(P//2), color="k", ls=":", lw=0.8, alpha=0.4)
    ax_b.set_title("(b) Multiplication — additive basis", fontsize=10, fontweight="bold")
    ax_b.set_xlabel("Frequency $\\omega$")
    ax_b.set_xlim(0, P//2)
    ax_b.legend(fontsize=8, loc="upper right")
    ax_b.text(0.03, 0.92, "No structure in\nmismatched basis",
              transform=ax_b.transAxes, fontsize=8, va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", alpha=0.9))

    # Panel C: MUL correct basis
    ax_c = fig.add_subplot(gs[0, 2])
    for k in range(3):
        s = mul_dlog_spectra[k]
        ax_c.plot(s["freqs"], s["power"], color=colors[f"v{k+1}"],
                  linewidth=1.8, alpha=0.85, label=f"$v_{k+1}$")
    ax_c.axhline(1.0/((P-1)//2), color="k", ls=":", lw=0.8, alpha=0.4)
    ax_c.set_title("(c) Multiplication — discrete-log basis", fontsize=10, fontweight="bold")
    ax_c.set_xlabel("Frequency $\\omega$")
    ax_c.set_xlim(0, (P-1)//2)
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.text(0.03, 0.92, "Structure appears in\nsymmetry-adapted basis\n$\\omega = 29$",
              transform=ax_c.transAxes, fontsize=8, va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", alpha=0.9))

    # Panel D left: SUB
    ax_d1 = fig.add_subplot(gs[1, 0])
    for k in range(3):
        s = sub_spectra[k]
        ax_d1.plot(s["freqs"], s["power"], color=colors[f"v{k+1}"],
                   linewidth=1.8, alpha=0.85, label=f"$v_{k+1}$")
    ax_d1.axhline(1.0/(P//2), color="k", ls=":", lw=0.8, alpha=0.4)
    ax_d1.set_title("(d) Subtraction — additive basis", fontsize=10, fontweight="bold")
    ax_d1.set_xlabel("Frequency $\\omega$")
    ax_d1.set_ylabel("Normalised power")
    ax_d1.set_xlim(0, P//2)
    ax_d1.legend(fontsize=8, loc="upper right")
    ax_d1.text(0.03, 0.92, "Low-dimensional\nmulti-mode\n$\\omega \\in \\{6, 16, 32\\}$",
               transform=ax_d1.transAxes, fontsize=8, va="top",
               bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", alpha=0.9))

    # Panel D right: x²+y²
    ax_d2 = fig.add_subplot(gs[1, 1])
    for k in range(3):
        s = x2y2_spectra[k]
        ax_d2.plot(s["freqs"], s["power"], color=colors[f"v{k+1}"],
                   linewidth=1.8, alpha=0.85, label=f"$v_{k+1}$")
    ax_d2.axhline(1.0/(P//2), color="k", ls=":", lw=0.8, alpha=0.4)
    ax_d2.set_title("(e) $x^2+y^2$ — additive basis", fontsize=10, fontweight="bold")
    ax_d2.set_xlabel("Frequency $\\omega$")
    ax_d2.set_xlim(0, P//2)
    ax_d2.legend(fontsize=8, loc="upper right")
    ax_d2.text(0.03, 0.92, "No single-mode\ncollapse",
               transform=ax_d2.transAxes, fontsize=8, va="top",
               bbox=dict(boxstyle="round,pad=0.3", fc="#fce4ec", alpha=0.9))

    # Panel F: summary text / taxonomy
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    summary = (
        "Hierarchy of functional structure\n"
        "at the spectral edge:\n\n"
        "  Addition       → single mode\n"
        "  Multiplication → single mode\n"
        "                    (correct basis)\n"
        "  Subtraction    → multi-mode\n"
        "  $x^2+y^2$     → compositional\n"
        "                    (cross-terms)"
    )
    ax_f.text(0.1, 0.85, summary, transform=ax_f.transAxes,
              fontsize=10, va="top", family="monospace",
              bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#999",
                        alpha=0.95))

    fig.suptitle("Spectral edge structure depends on the functional basis",
                 fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(PLOT_DIR / "fig1_basis_dependence.png", dpi=200, bbox_inches="tight")
    plt.savefig(PLOT_DIR / "fig1_basis_dependence.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {PLOT_DIR}/fig1_basis_dependence.png and .pdf")


if __name__ == "__main__":
    main()
