#!/usr/bin/env python3
"""
residual_stream_alignment.py

Tests whether the {v1, v2, v3} parameter-space directions (SVD of weight
updates) correspond to interpretable features in activation space.

The parameter-space analysis found that v_k directions are NOT head-localized
(purity ~0.14 for all k, barely above uniform baseline 1/8). This script asks
the follow-up: even though v_k is diffuse in parameter space, does perturbing
along v_k produce structured, low-rank changes in the residual stream?

Two concrete tests
------------------
1. Residual stream rank structure
   For each direction v_k, compute the (N_test × D_MODEL) matrix of residual
   stream changes Δh(x) = h(x; θ+εv_k) − h(x; θ).
   SVD of Δh gives effective rank:
     - rank-1 → v_k maps to a single activation feature
     - high-rank → v_k is diffuse in activation space too
   Also compute alignment: A[k,j] = cosine(top_u_k, top_u_j) — do v1,v2,v3
   all perturb the same activation direction, or different ones?

2. Fourier alignment (modular arithmetic specific)
   For mod-p arithmetic, define Fourier basis vectors over inputs (a,b):
     F_freq_cos(a,b) = cos(2π·freq·(a+b)/p)
     F_freq_sin(a,b) = sin(2π·freq·(a+b)/p)
   Project the perturbation pattern ||Δh(a,b)||² onto this basis.
   If v_k corresponds to Fourier frequency freq*, then the perturbation will
   be concentrated on inputs where (a+b) mod p ≡ const for that freq.
   More directly: project Δh(a,b) · û_k onto the Fourier modes to get the
   Fourier spectrum of the activation perturbation.

   Expected result if v1 is the "grokking direction":
     - Post-grok: Δh shows peaked Fourier spectrum (1-3 dominant frequencies)
     - Pre-grok: Δh shows flat Fourier spectrum (no preferred frequency)

Output
------
spectral/residual_alignment_plots/
  figA_effective_rank_{op}.png      — effective rank of Δh for each v_k, per phase
  figB_alignment_matrix_{op}.png    — A[k,j] cosine alignment of top activation dirs
  figC_fourier_spectrum_{op}.png    — Fourier spectrum of perturbation for v1..v5
  figD_fourier_vs_phase_{op}.png    — how Fourier peakedness evolves across training
  figE_cross_op_fourier.png         — cross-operation comparison of v1 Fourier peak
Results saved to residual_alignment_plots/alignment_results.pt
"""

import math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "coherence_edge_results"
SWEEP_DIR   = SCRIPT_DIR.parent / "grok_sweep_results"
PLOT_DIR    = SCRIPT_DIR / "residual_alignment_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────

P        = 97
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 2
W        = 20       # SVD window
N_DIR    = 8        # directions to analyse (top-5 + 3 bulk)
EPS_SCALE = 0.005   # perturbation size (× ||θ_attn||)

PHASE_ORDER = ["early_coherent", "memorization", "grokking_transition", "stable_postgrok"]
PHASE_LABELS = {
    "early_coherent":      "Early",
    "memorization":        "Memorization",
    "grokking_transition": "Grok Trans.",
    "stable_postgrok":     "Post-Grok",
}

CACHE_PATHS = {
    "add":   RESULTS_DIR / "training_cache.pt",
    "sub":   RESULTS_DIR / "training_cache_sub.pt",
    "mul":   RESULTS_DIR / "training_cache_mul.pt",
    "x2_y2": RESULTS_DIR / "training_cache_x2_y2.pt",
}

# ─── Operations ──────────────────────────────────────────────────────────────

def op_add(a, b, p):   return (a + b) % p
def op_sub(a, b, p):   return (a - b) % p
def op_mul(a, b, p):   return (a * b) % p
def op_x2_y2(a, b, p): return (a*a + b*b) % p

OPERATIONS = {
    "add":   {"fn": op_add,   "label": "(a+b) mod p",    "fourier_fn": lambda a, b: (a + b) % P},
    "sub":   {"fn": op_sub,   "label": "(a-b) mod p",    "fourier_fn": lambda a, b: (a - b) % P},
    "mul":   {"fn": op_mul,   "label": "(a*b) mod p",    "fourier_fn": None},
    "x2_y2": {"fn": op_x2_y2, "label": "(a²+b²) mod p", "fourier_fn": lambda a, b: (a*a + b*b) % P},
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
        res = h[:, 0, :]          # residual stream at position 0, after all layers
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


def find_edge(S):
    k = min(10, len(S)-1)
    if k < 1: return 0
    mass = S[:k] / (S.sum() + 1e-15)
    gap  = S[:k] / (S[1:k+1] + 1e-15)
    return int(np.argmax(mass * gap))


# ─── Checkpoint selection ────────────────────────────────────────────────────

def select_checkpoints(attn_logs, cache_data, updates):
    attn_steps   = [e["step"] for e in attn_logs]
    update_steps = attn_steps[1:]
    metrics      = cache_data["metrics"]
    n            = len(updates)

    def acc(step):
        best = min(metrics, key=lambda m: abs(m["step"] - step))
        return best

    grok_step = next((m["step"] for m in metrics if m["test_acc"] > 0.5), metrics[-1]["step"])
    print(f"    grok_step={grok_step}")

    selected = {}
    t_early = max(W, n // 10)
    selected["early_coherent"] = (t_early, update_steps[t_early])

    mem = [t for t in range(n)
           if acc(update_steps[t])["train_acc"] > 0.8
           and acc(update_steps[t])["test_acc"] < 0.3]
    mid = mem[len(mem)//2] if mem else min(range(n), key=lambda t: abs(update_steps[t] - grok_step//2))
    selected["memorization"] = (mid, update_steps[mid])

    best = min(range(n), key=lambda t: abs(update_steps[t] - grok_step))
    selected["grokking_transition"] = (best, update_steps[best])

    post = next((t for t in range(n-1, -1, -1)
                 if acc(update_steps[t])["test_acc"] > 0.95), n-1)
    selected["stable_postgrok"] = (post, update_steps[post])

    return [(ph, *selected[ph]) for ph in PHASE_ORDER if ph in selected]


def find_nearest_ckpt(step, cache_data):
    return min(cache_data["checkpoints"], key=lambda cs: abs(cs[0] - step))


# ─── Core measurement: residual stream perturbation ─────────────────────────

@torch.no_grad()
def compute_residual_perturbation(model, state_dict, vk_np, test_pairs, device,
                                  eps_scale=EPS_SCALE, batch_size=512):
    """
    Compute Δh(x) = h(x; θ+εv_k) − h(x; θ) for all test inputs x.

    Returns
    -------
    delta_h : np.ndarray [N_test, D_MODEL]
    """
    attn_keys = get_attn_keys(model)

    # Compute perturbation magnitude
    flat = torch.cat([state_dict[k].float().flatten() for k in attn_keys])
    norm = float(flat.norm())
    eps  = eps_scale * norm

    vk_t = torch.from_numpy(vk_np.copy()).float()
    vk_t = vk_t / (vk_t.norm() + 1e-30)

    # Build perturbed state_dict
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


def effective_rank(delta_h):
    """
    Effective rank of the (N × D_MODEL) matrix:
      rank_eff = exp(H(p)) where p_k = s_k² / Σs_j²  (entropy of squared singular values)
    Also return: top-1 explained variance fraction, and top-3 fraction.
    """
    _, S, Vt = np.linalg.svd(delta_h, full_matrices=False)
    S2     = S**2
    total  = S2.sum() + 1e-30
    p      = S2 / total
    p_pos  = p[p > 1e-15]
    H      = -float(np.sum(p_pos * np.log(p_pos)))
    rank_e = float(np.exp(H))
    top1   = float(S2[0] / total)
    top3   = float(S2[:3].sum() / total)
    return rank_e, top1, top3, Vt


# ─── Fourier analysis ────────────────────────────────────────────────────────

def build_fourier_basis(test_pairs, fourier_fn, p=P):
    """
    Build Fourier basis vectors over test inputs.
    For each frequency freq in {1, ..., p//2}:
      cos_vec[i] = cos(2π·freq·fourier_fn(a_i, b_i) / p)
      sin_vec[i] = sin(2π·freq·fourier_fn(a_i, b_i) / p)

    Returns freqs array and (N_test, 2*n_freqs) basis matrix.
    """
    N      = len(test_pairs)
    freqs  = np.arange(1, p // 2 + 1)
    ab     = np.array(test_pairs)
    vals   = np.array([fourier_fn(int(a), int(b)) for a, b in ab], dtype=float)

    phases = np.outer(vals, freqs) * (2 * np.pi / p)   # [N, n_freqs]
    cos_b  = np.cos(phases)   # [N, n_freqs]
    sin_b  = np.sin(phases)   # [N, n_freqs]

    # Normalise each basis vector
    cos_b /= (np.linalg.norm(cos_b, axis=0, keepdims=True) + 1e-15)
    sin_b /= (np.linalg.norm(sin_b, axis=0, keepdims=True) + 1e-15)

    return freqs, cos_b, sin_b   # cos_b, sin_b: [N, n_freqs]


def fourier_spectrum(delta_h, top_u, cos_b, sin_b):
    """
    Project the scalar field f(x) = Δh(x)·top_u onto the Fourier basis.
    f is the projection of residual perturbation onto its dominant activation dir.

    Returns power[freq] = cos_coeff² + sin_coeff² for each frequency.
    """
    # f(x) = delta_h(x) projected onto top activation direction
    f = delta_h @ top_u   # [N]
    f = f - f.mean()

    cos_coeffs = cos_b.T @ f   # [n_freqs]
    sin_coeffs = sin_b.T @ f   # [n_freqs]
    power      = cos_coeffs**2 + sin_coeffs**2
    power     /= (power.sum() + 1e-30)
    return power


def fourier_peakedness(power):
    """
    Peakedness = max(power) / mean(power).
    Flat spectrum → 1.0; single dominant frequency → n_freqs.
    """
    return float(power.max() / (power.mean() + 1e-30))


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_figA_effective_rank(all_results, op_name, op_label):
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    n_ph   = len(phases)
    if not n_ph: return

    fig, axes = plt.subplots(1, n_ph, figsize=(4*n_ph, 4), squeeze=False)
    fig.suptitle(f"Effective Rank of Δh — {op_label}\n"
                 "rank_eff = exp(H(σ²)) — rank-1 means single activation feature",
                 fontsize=10, fontweight="bold")

    for ax, phase in zip(axes[0], phases):
        pd     = all_results[phase]
        ks     = sorted(pd["rank_eff"].keys())
        ranks  = [pd["rank_eff"][k] for k in ks]
        top1s  = [pd["top1_frac"][k] for k in ks]

        x  = np.arange(len(ks))
        ax.bar(x, ranks, color=["#e74c3c" if k < 3 else "#95a5a6" for k in ks],
               alpha=0.8, edgecolor="k", linewidth=0.4)
        ax2 = ax.twinx()
        ax2.plot(x, top1s, "D--", color="#2c3e50", markersize=5, linewidth=1.2,
                 label="top-1 var frac", zorder=5)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Top-1 variance fraction", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"v{k+1}" for k in ks], fontsize=7)
        ax.set_ylabel("Effective rank")
        ax.set_title(f"{PHASE_LABELS[phase]}\n(step {pd['step']})", fontsize=9)
        ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_ylim(0, None)

        if phase == phases[-1]:
            ax2.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figA_effective_rank_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figA_effective_rank_{op_name}.png")


def plot_figB_alignment_matrix(all_results, op_name, op_label):
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    n_ph   = len(phases)
    if not n_ph: return

    fig, axes = plt.subplots(1, n_ph, figsize=(4*n_ph, 4), squeeze=False)
    fig.suptitle(f"Alignment Matrix A[k,j] = cos(top_u_k, top_u_j) — {op_label}\n"
                 "Do v1,v2,v3 perturb the SAME activation direction? (1 = identical, 0 = orthogonal)",
                 fontsize=10, fontweight="bold")

    for ax, phase in zip(axes[0], phases):
        pd    = all_results[phase]
        A     = np.array(pd["alignment_matrix"])
        n     = A.shape[0]

        im = ax.imshow(np.abs(A), cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"v{k+1}" for k in range(n)], fontsize=7, rotation=45)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"v{k+1}" for k in range(n)], fontsize=7)
        ax.set_title(f"{PHASE_LABELS[phase]}\n(step {pd['step']})", fontsize=9)

        # box top-3
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3,
                                   fill=False, edgecolor="#e74c3c", linewidth=2))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figB_alignment_matrix_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figB_alignment_matrix_{op_name}.png")


def plot_figC_fourier_spectrum(all_results, op_name, op_label, freqs):
    """Fourier spectrum of Δh projected onto top activation direction, for each v_k."""
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    n_ph   = len(phases)
    if not n_ph: return

    # Check if Fourier data is available
    if "fourier_spectrum" not in all_results[phases[0]]:
        return

    n_dir_plot = min(5, N_DIR)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, axes = plt.subplots(1, n_ph, figsize=(5*n_ph, 4), squeeze=False)
    fig.suptitle(f"Fourier Spectrum of Residual Perturbation — {op_label}\n"
                 "Power at each frequency after projecting Δh onto its top activation direction",
                 fontsize=10, fontweight="bold")

    for ax, phase in zip(axes[0], phases):
        pd = all_results[phase]
        spectra = pd["fourier_spectrum"]   # dict k -> power array

        for k in range(n_dir_plot):
            if k not in spectra: continue
            power = spectra[k]
            lw    = 2.5 if k < 3 else 1.0
            alpha = 0.9 if k < 3 else 0.5
            ls    = "-" if k < 3 else "--"
            ax.plot(freqs, power, color=colors[k], linewidth=lw, linestyle=ls,
                    alpha=alpha, label=f"v{k+1}")

        ax.set_xlabel("Fourier frequency")
        ax.set_ylabel("Normalised power")
        ax.set_title(f"{PHASE_LABELS[phase]}\n(step {pd['step']})", fontsize=9)
        ax.set_xlim(1, freqs[-1])
        ax.set_ylim(0, None)

        # Uniform level
        ax.axhline(1.0 / len(freqs), color="k", linestyle=":", linewidth=0.8,
                   alpha=0.4, label="uniform")

        if phase == phases[0]:
            ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figC_fourier_spectrum_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figC_fourier_spectrum_{op_name}.png")


def plot_figD_fourier_peakedness(all_results, op_name, op_label):
    """How Fourier peakedness of v1..v3 evolves across training."""
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    if not phases: return
    if "fourier_peakedness" not in all_results[phases[0]]: return

    fig, ax = plt.subplots(figsize=(6, 4))

    for k, color, ls in [(0,"#e74c3c","-"), (1,"#3498db","--"), (2,"#2ecc71","-.")]:
        pks = []
        steps = []
        for ph in phases:
            pd = all_results[ph]
            if k in pd.get("fourier_peakedness", {}):
                pks.append(pd["fourier_peakedness"][k])
                steps.append(pd["step"])
        if pks:
            ax.plot(range(len(pks)), pks, marker="o", color=color, linestyle=ls,
                    linewidth=2, label=f"v{k+1}")

    # Bulk mean
    for ph_i, ph in enumerate(phases):
        pd = all_results[ph]
        bulk = [pd["fourier_peakedness"][k] for k in pd.get("fourier_peakedness", {})
                if k >= 3]
        if bulk:
            ax.scatter([ph_i], [np.mean(bulk)], color="#95a5a6",
                       marker="s", s=50, zorder=5)
    ax.scatter([], [], color="#95a5a6", marker="s", s=50, label="bulk mean")

    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8, alpha=0.4, label="flat spectrum")
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([PHASE_LABELS[ph] for ph in phases], rotation=15, ha="right")
    ax.set_ylabel("Fourier peakedness (max/mean power)")
    ax.set_title(f"Fourier Peakedness Across Training — {op_label}\n"
                 "High = perturbation concentrated on specific Fourier frequency",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, None)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figD_fourier_peakedness_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figD_fourier_peakedness_{op_name}.png")


def plot_figE_cross_op(all_op_results):
    """Cross-operation comparison: v1 Fourier peak frequency and peakedness."""
    ops = [op for op in OPERATIONS if op in all_op_results and all_op_results[op]]
    if not ops: return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Cross-Operation: v1 Residual Stream in Activation Space",
                 fontsize=11, fontweight="bold")

    op_labels = [OPERATIONS[op]["label"] for op in ops]
    x = np.arange(len(ops))

    # Panel 1: effective rank of v1 at grokking vs post-grok
    ax = axes[0]
    grok_ranks = []
    post_ranks = []
    for op in ops:
        gr = all_op_results[op].get("grokking_transition", {}).get("rank_eff", {}).get(0, np.nan)
        po = all_op_results[op].get("stable_postgrok",     {}).get("rank_eff", {}).get(0, np.nan)
        grok_ranks.append(gr)
        post_ranks.append(po)

    ax.bar(x - 0.2, grok_ranks, 0.4, label="Grok transition", color="#f39c12", alpha=0.8)
    ax.bar(x + 0.2, post_ranks, 0.4, label="Post-grok",        color="#3498db", alpha=0.8)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(op_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Effective rank of Δh(x) for v1")
    ax.set_title("v1 Effective Rank")
    ax.legend(fontsize=8)

    # Panel 2: top-1 variance fraction of v1 across phases
    ax = axes[1]
    for ph_i, (phase, color) in enumerate([
        ("memorization", "#e74c3c"),
        ("grokking_transition", "#f39c12"),
        ("stable_postgrok", "#3498db"),
    ]):
        top1s = [
            all_op_results[op].get(phase, {}).get("top1_frac", {}).get(0, np.nan)
            for op in ops
        ]
        ax.plot(x, top1s, marker="o", color=color, label=PHASE_LABELS[phase],
                linewidth=2, markersize=6)

    ax.set_xticks(x); ax.set_xticklabels(op_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 variance fraction of v1 perturbation")
    ax.set_title("How Low-Rank Is v1's Activation Perturbation?")
    ax.set_ylim(0, 1)
    ax.axhline(1.0/D_MODEL, color="k", linestyle=":", alpha=0.4, label="uniform baseline")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figE_cross_op_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figE_cross_op_summary.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_op(op_name, op_cfg, device):
    sweep_path = SWEEP_DIR / f"{op_name}_wd1.0_s42.pt"
    cache_path = CACHE_PATHS[op_name]
    if not sweep_path.exists() or not cache_path.exists():
        print(f"  Skipping {op_name}: missing data")
        return None

    print(f"\n{'='*60}\n  {op_name} — {op_cfg['label']}\n{'='*60}")

    sweep_data  = torch.load(sweep_path, map_location="cpu", weights_only=False)
    cache_data  = torch.load(cache_path, map_location="cpu", weights_only=False)
    attn_logs   = sweep_data["attn_logs"]

    if not cache_data.get("metrics"):
        cache_data["metrics"] = sweep_data.get("metrics", sweep_data.get("log", []))

    updates = compute_updates(attn_logs)
    print(f"  {len(updates)} updates, dim={len(updates[0])}")

    _, test_pairs = [], cache_data["test_pairs"]
    test_pairs = [(int(a), int(b)) for a, b in test_pairs]

    checkpoints = select_checkpoints(attn_logs, cache_data, updates)

    # Build Fourier basis if applicable
    fourier_fn = op_cfg.get("fourier_fn")
    if fourier_fn is not None:
        freqs, cos_b, sin_b = build_fourier_basis(test_pairs, fourier_fn)
        print(f"  Built Fourier basis: {len(freqs)} frequencies")
    else:
        freqs = cos_b = sin_b = None

    model = ModOpTransformer().to(device)
    results = {}

    for phase_name, t_idx, step in checkpoints:
        print(f"\n  [{phase_name}] t_idx={t_idx}, step={step}")

        svd_result = window_svd(updates, t_idx)
        if svd_result is None:
            print("    SVD failed, skipping")
            continue
        S, Vt = svd_result
        kstar = find_edge(S)
        n_dir = min(N_DIR, Vt.shape[0])

        _, state_dict = find_nearest_ckpt(step, cache_data)

        # ── Compute residual perturbations ──────────────────────────────────
        rank_effs  = {}
        top1_fracs = {}
        top3_fracs = {}
        top_us     = {}          # top activation direction per v_k

        for k in range(n_dir):
            vk = Vt[k]
            dh, eps = compute_residual_perturbation(
                model, state_dict, vk, test_pairs, device)

            rank_e, top1, top3, Vt_dh = effective_rank(dh)
            rank_effs[k]  = rank_e
            top1_fracs[k] = top1
            top3_fracs[k] = top3
            top_us[k]     = Vt_dh[0]   # top right singular vector of Δh matrix

            print(f"    v{k+1}: rank_eff={rank_e:.2f}, top1={top1:.3f}, top3={top3:.3f}")

        # ── Alignment matrix ────────────────────────────────────────────────
        A = np.zeros((n_dir, n_dir))
        for i in range(n_dir):
            for j in range(n_dir):
                u_i = top_us[i]
                u_j = top_us[j]
                A[i, j] = float(np.dot(u_i, u_j) /
                                (np.linalg.norm(u_i) * np.linalg.norm(u_j) + 1e-15))

        # ── Fourier analysis ────────────────────────────────────────────────
        fourier_spectra    = {}
        fourier_peakedness_d = {}
        dominant_freqs     = {}

        if fourier_fn is not None:
            for k in range(min(5, n_dir)):
                vk = Vt[k]
                dh, _ = compute_residual_perturbation(
                    model, state_dict, vk, test_pairs, device)
                power = fourier_spectrum(dh, top_us[k], cos_b, sin_b)
                fourier_spectra[k]      = power
                fourier_peakedness_d[k] = fourier_peakedness(power)
                dominant_freqs[k]       = int(freqs[np.argmax(power)])

                print(f"    v{k+1} Fourier: peak_freq={dominant_freqs[k]}, "
                      f"peakedness={fourier_peakedness_d[k]:.2f}")

        results[phase_name] = {
            "step":               step,
            "kstar":              kstar,
            "rank_eff":           rank_effs,
            "top1_frac":          top1_fracs,
            "top3_frac":          top3_fracs,
            "alignment_matrix":   A.tolist(),
            "fourier_spectrum":   fourier_spectra,
            "fourier_peakedness": fourier_peakedness_d,
            "dominant_freqs":     dominant_freqs,
        }

    return results, freqs


def main():
    device = get_device()
    print(f"Device: {device}")

    all_op_results = {}

    for op_name, op_cfg in OPERATIONS.items():
        out = run_op(op_name, op_cfg, device)
        if out is None:
            continue
        results, freqs = out
        all_op_results[op_name] = results

        op_label = op_cfg["label"]
        plot_figA_effective_rank(results, op_name, op_label)
        plot_figB_alignment_matrix(results, op_name, op_label)
        if freqs is not None:
            plot_figC_fourier_spectrum(results, op_name, op_label, freqs)
            plot_figD_fourier_peakedness(results, op_name, op_label)

        # Print summary
        print(f"\n  ── {op_name} summary ──")
        for ph in PHASE_ORDER:
            if ph not in results: continue
            pd = results[ph]
            v1_rank = pd["rank_eff"].get(0, float("nan"))
            v1_top1 = pd["top1_frac"].get(0, float("nan"))
            v1_freq = pd.get("dominant_freqs", {}).get(0, "N/A")
            v1_peak = pd.get("fourier_peakedness", {}).get(0, float("nan"))
            # top-3 alignment (off-diagonal mean of A[0:3,0:3])
            A  = np.array(pd["alignment_matrix"])
            block = np.abs(A[:3, :3])
            np.fill_diagonal(block, 0)
            top3_align = float(block.mean())
            print(f"  [{PHASE_LABELS[ph]:14s}] step={pd['step']:5d} | "
                  f"v1 rank_eff={v1_rank:.2f} top1={v1_top1:.3f} | "
                  f"top3 inter-align={top3_align:.3f} | "
                  f"v1 Fourier peak={v1_freq} pk={v1_peak:.1f}")

    plot_figE_cross_op(all_op_results)

    torch.save(all_op_results, PLOT_DIR / "alignment_results.pt")
    print(f"\nResults saved to {PLOT_DIR}/alignment_results.pt")
    print(f"Plots saved to   {PLOT_DIR}/")

    # Final interpretation summary
    print("\n" + "="*60)
    print("RESIDUAL STREAM ALIGNMENT — FINDINGS")
    print("="*60)
    for op_name, results in all_op_results.items():
        print(f"\n  {op_name}:")
        for ph in ["grokking_transition", "stable_postgrok"]:
            if ph not in results: continue
            pd = results[ph]
            A  = np.array(pd["alignment_matrix"])

            v1_r  = pd["rank_eff"].get(0, np.nan)
            v1_t1 = pd["top1_frac"].get(0, np.nan)

            # Inter-alignment within top-3 and between top-3 and bulk
            A_abs = np.abs(A)
            n = A_abs.shape[0]
            top3_off = A_abs[:3, :3].copy()
            np.fill_diagonal(top3_off, 0)
            top3_bulk = A_abs[:3, 3:] if n > 3 else np.zeros((3, 1))

            print(f"    [{PHASE_LABELS[ph]}] v1 rank={v1_r:.2f} top1={v1_t1:.3f} | "
                  f"top3 self-align={top3_off.mean():.3f} "
                  f"top3-bulk align={top3_bulk.mean():.3f}")


if __name__ == "__main__":
    main()
