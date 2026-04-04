#!/usr/bin/env python3
"""
sae_fourier_features.py

Two questions:
  1. Is 60% Jaccard significant?
  2. Which SAE features are in the shared pool, and do they correspond to
     Fourier components in the residual stream?

Significance test
-----------------
Three baselines for comparing the v1/v2/v3 Jaccard of 0.60:

  (a) Combinatorial null:
      If top-20 features are drawn uniformly at random from D_SAE=512,
      E[Jaccard] ≈ (20² / 512) / (2·20 - 20²/512) ≈ 0.02.

  (b) Random direction null:
      Draw N_NULL random unit vectors in parameter space, compute their
      Δh through the model, encode through SAE, compute pairwise Jaccard.
      This tests whether ANY two perturbations activate similar features,
      regardless of geometry.

  (c) Angle-matched null (the honest null):
      v1, v2, v3 have ~45° pairwise subspace angles.  Draw synthetic Δh
      pairs with the SAME 45° angle but random content.  Jaccard at 45°
      angle tells us how much of the 60% is purely geometric.

  (d) Bulk baseline:
      Use v4, v5, v6 (below the spectral edge).  If bulk directions give
      similar Jaccard to top-3, the 60% is just an artifact of the SAE
      structure.  If bulk is lower, top-3 are genuinely more co-specialised.

Fourier analysis
----------------
For each SAE feature f in the shared top-20 pool:
  1. Compute feature activations z_f(x) across all test inputs.
  2. Group inputs by output value q = (a+b) mod p → p groups.
     Compute mean z_f per group → a p-point signal s_f[q].
  3. DFT of s_f → Fourier power spectrum P_f[freq].
  4. Dominant frequency: freq* = argmax P_f.
  5. Fourier R²: fraction of variance explained by the dominant mode.

Expected result if shared features are Fourier features:
  - Few dominant frequencies (1-3 out of 48)
  - High R² (>50% variance from a single mode)
  - Different features correspond to different Fourier frequencies,
    but ALL within the frequencies used by the grokking circuit

Compare:
  - Shared features (in all three v1/v2/v3 top-20 pools)
  - v1-exclusive features (top-20 of v1 only)
  - Random non-shared features

Outputs → spectral/sae_fourier_plots/
  figA_significance.png        significance test: observed vs 3 baselines
  figB_shared_features.png     which SAE features are shared
  figC_fourier_spectrum_f*.png Fourier spectrum of each shared feature
  figD_fourier_summary.png     dominant freq + R² for shared vs non-shared
  figE_feature_activation.png  scatter: z_f(x) vs cos(2π·freq*·(a+b)/p)
  results saved to sae_fourier_plots/sae_fourier_results.pt
"""

import math, random, time
from pathlib import Path
from collections import Counter

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
PLOT_DIR    = SCRIPT_DIR / "sae_fourier_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────

P         = 97
D_MODEL   = 128
N_HEADS   = 4
D_HEAD    = D_MODEL // N_HEADS
N_LAYERS  = 2
W_SVD     = 20
N_DIR_TOP = 3     # directions in the "top block"
N_DIR_BULK = 3    # bulk directions to use as empirical null (v4,v5,v6)
TOP_K_FEAT = 20   # features in each direction's "top" set
N_NULL    = 50    # random directions for null distribution
EPS_SCALE = 0.005

D_SAE     = 512
K_SAE     = 32
SAE_EPOCHS = 300
SAE_LR     = 3e-4
SAE_BATCH  = 256

CACHE_PATHS = {
    "add":   RESULTS_DIR / "training_cache.pt",
    "sub":   RESULTS_DIR / "training_cache_sub.pt",
    "mul":   RESULTS_DIR / "training_cache_mul.pt",
    "x2_y2": RESULTS_DIR / "training_cache_x2_y2.pt",
}

PHASE = "stable_postgrok"   # focus on post-grok (cleaner circuit)


# ─── Model ───────────────────────────────────────────────────────────────────

class ModOpTransformer(nn.Module):
    def __init__(self, p=P, d=D_MODEL, nh=N_HEADS, nl=N_LAYERS, dff=256):
        super().__init__()
        self.tok_emb = nn.Embedding(p, d)
        self.pos_emb = nn.Parameter(torch.randn(2, d) / math.sqrt(d))
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nh, dim_feedforward=dff,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nl)
        self.ln   = nn.LayerNorm(d)
        self.head = nn.Linear(d, p)

    def forward(self, a, b):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        return self.head(self.ln(h[:, 0, :]))


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


# ─── SVD helpers ─────────────────────────────────────────────────────────────

def flatten_attn(entry):
    parts = []
    for ld in sorted(entry["layers"], key=lambda x: x["layer"]):
        for k in ["WQ", "WK", "WV", "WO"]:
            parts.append(ld[k].flatten().float())
    return torch.cat(parts)


def compute_updates(attn_logs):
    flat = [flatten_attn(e).numpy() for e in attn_logs]
    return [flat[i] - flat[i-1] for i in range(1, len(flat))]


def window_svd(updates, t_idx, w=W_SVD):
    start = max(0, t_idx - w + 1)
    X = np.stack(updates[start:t_idx+1])
    if X.shape[0] < 3: return None
    X -= X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S, Vt


def find_edge(S):
    k = min(10, len(S)-1)
    if k < 1: return 0
    mass = S[:k] / (S.sum() + 1e-15)
    gap  = S[:k] / (S[1:k+1] + 1e-15)
    return int(np.argmax(mass * gap))


def select_postgrok_checkpoint(attn_logs, metrics):
    steps     = [e["step"] for e in attn_logs]
    n         = len(steps) - 1
    upd_steps = steps[1:]

    def acc(step):
        return min(metrics, key=lambda m: abs(m["step"] - step))

    post = next((i for i in range(n-1, -1, -1)
                 if acc(upd_steps[i])["test_acc"] > 0.95), n-1)
    grok = next((i for i in range(n) if acc(upd_steps[i])["test_acc"] > 0.5), n//2)
    return {"stable_postgrok":     (post, upd_steps[post]),
            "grokking_transition": (grok, upd_steps[grok])}


def nearest_ckpt(step, cache):
    return min(cache["checkpoints"], key=lambda cs: abs(cs[0] - step))


# ─── Forward pass & Δh ───────────────────────────────────────────────────────

@torch.no_grad()
def get_residuals(state_dict, a_np, b_np, device, batch=512):
    """Final-layer residual stream h(x) before LN+head. Returns [N, D_MODEL]."""
    sd  = {k: v.to(device).float() for k, v in state_dict.items()}
    N   = len(a_np)
    out = np.zeros((N, D_MODEL), np.float32)

    for start in range(0, N, batch):
        end = min(start + batch, N)
        a_t = torch.tensor(a_np[start:end], dtype=torch.long, device=device)
        b_t = torch.tensor(b_np[start:end], dtype=torch.long, device=device)

        tok_w = sd["tok_emb.weight"]
        pos_e = sd["pos_emb"]
        h = torch.stack([tok_w[a_t], tok_w[b_t]], dim=1) + pos_e.unsqueeze(0)

        for l in range(N_LAYERS):
            n1w = sd[f"encoder.layers.{l}.norm1.weight"]
            n1b = sd[f"encoder.layers.{l}.norm1.bias"]
            hn  = F.layer_norm(h, (D_MODEL,), n1w, n1b)
            Wqkv = sd[f"encoder.layers.{l}.self_attn.in_proj_weight"]
            bq   = sd.get(f"encoder.layers.{l}.self_attn.in_proj_bias")
            qkv  = hn @ Wqkv.t() + (bq if bq is not None else 0)
            Q, K, V = qkv.split(D_MODEL, dim=-1)
            B2, T = Q.shape[:2]
            Q = Q.view(B2, T, N_HEADS, D_HEAD).transpose(1, 2)
            K = K.view(B2, T, N_HEADS, D_HEAD).transpose(1, 2)
            V = V.view(B2, T, N_HEADS, D_HEAD).transpose(1, 2)
            A = F.softmax(Q @ K.transpose(-2,-1) / math.sqrt(D_HEAD), dim=-1)
            av  = (A @ V).transpose(1, 2).contiguous().view(B2, T, D_MODEL)
            WO  = sd[f"encoder.layers.{l}.self_attn.out_proj.weight"]
            bo  = sd.get(f"encoder.layers.{l}.self_attn.out_proj.bias")
            ao  = av @ WO.t() + (bo if bo is not None else 0)
            h   = h + ao
            n2w = sd[f"encoder.layers.{l}.norm2.weight"]
            n2b = sd[f"encoder.layers.{l}.norm2.bias"]
            hn2 = F.layer_norm(h, (D_MODEL,), n2w, n2b)
            W1  = sd[f"encoder.layers.{l}.linear1.weight"]
            b1  = sd.get(f"encoder.layers.{l}.linear1.bias",
                         torch.zeros(W1.shape[0], device=device))
            W2  = sd[f"encoder.layers.{l}.linear2.weight"]
            b2  = sd.get(f"encoder.layers.{l}.linear2.bias",
                         torch.zeros(W2.shape[0], device=device))
            h   = h + F.gelu(hn2 @ W1.t() + b1) @ W2.t() + b2

        out[start:end] = h[:, 0, :].cpu().numpy()
    return out


def make_perturbed_sd(state_dict, vk_np, eps_scale=EPS_SCALE):
    attn_keys = sorted(k for k in state_dict
                       if "self_attn" in k and "weight" in k and "bias" not in k)
    flat = torch.cat([state_dict[k].float().flatten() for k in attn_keys])
    eps  = float(flat.norm()) * eps_scale
    vk   = torch.from_numpy(vk_np.copy()).float()
    vk  /= vk.norm() + 1e-30
    pert = {k: v.clone().float() for k, v in state_dict.items()}
    off  = 0
    for key in attn_keys:
        n = pert[key].numel()
        pert[key] += (eps * vk[off:off+n]).reshape(pert[key].shape)
        off += n
    return pert


def compute_delta_h(state_dict, vk_np, a_np, b_np, device):
    h_base = get_residuals(state_dict, a_np, b_np, device)
    pert   = make_perturbed_sd(state_dict, vk_np)
    h_pert = get_residuals(pert, a_np, b_np, device)
    return (h_pert - h_base).astype(np.float32)


# ─── SAE ─────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_in=D_MODEL, d_sae=D_SAE, k=K_SAE):
        super().__init__()
        self.k     = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.randn(d_sae, d_in) * 0.02)
        self.W_dec = nn.Parameter(torch.randn(d_in, d_sae) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self._norm_dec()

    def _norm_dec(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def encode(self, x):
        x0     = x - self.b_pre
        z_pre  = x0 @ self.W_enc.t()
        topk   = z_pre.topk(self.k, dim=-1)
        z      = torch.zeros_like(z_pre)
        z.scatter_(-1, topk.indices, topk.values.clamp(min=0))
        return z

    def decode(self, z):
        return z @ self.W_dec.t() + self.b_dec + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_sae(H_np, device):
    H   = torch.from_numpy(H_np).float().to(device)
    N   = H.shape[0]
    sae = TopKSAE().to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    losses = []
    t0 = time.time()
    for ep in range(SAE_EPOCHS):
        idx  = torch.randperm(N, device=device)
        loss_ep = 0.0; nb = 0
        for s in range(0, N, SAE_BATCH):
            x    = H[idx[s:s+SAE_BATCH]]
            xhat, z = sae(x)
            loss = F.mse_loss(xhat, x)
            opt.zero_grad(); loss.backward(); opt.step()
            sae._norm_dec()
            loss_ep += loss.item(); nb += 1
        losses.append(loss_ep / max(nb, 1))
        if (ep+1) % 100 == 0:
            print(f"    SAE epoch {ep+1}/{SAE_EPOCHS}  loss={losses[-1]:.6f}"
                  f"  ({time.time()-t0:.0f}s)")
    return sae, losses


@torch.no_grad()
def get_sae_activations(sae, H_np, device, batch=512):
    """Z[n,f] = activation of feature f on input n.  Returns [N, D_SAE]."""
    H   = torch.from_numpy(H_np).float().to(device)
    out = []
    for s in range(0, len(H), batch):
        out.append(sae.encode(H[s:s+batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def get_delta_z(sae, H_np, dH_np, device, batch=512):
    """Δz[n,f] = z_f(h+Δh) - z_f(h).  Returns [N, D_SAE]."""
    H  = torch.from_numpy(H_np).float().to(device)
    dH = torch.from_numpy(dH_np).float().to(device)
    out = []
    for s in range(0, len(H), batch):
        z_b = sae.encode(H[s:s+batch])
        z_p = sae.encode(H[s:s+batch] + dH[s:s+batch])
        out.append((z_p - z_b).cpu().numpy())
    return np.concatenate(out, axis=0)


def top_features(delta_z, k=TOP_K_FEAT):
    """Top-k feature indices ranked by mean |Δz[f]|."""
    mean_abs = np.abs(delta_z).mean(axis=0)   # [D_SAE]
    return set(np.argsort(mean_abs)[::-1][:k].tolist()), mean_abs


def jaccard(A: set, B: set) -> float:
    if not A and not B: return 1.0
    return len(A & B) / len(A | B)


# ─── Significance baselines ───────────────────────────────────────────────────

def null_jaccard_random(sae, H_np, state_dict, a_np, b_np,
                        device, n_null=N_NULL, k=TOP_K_FEAT, seed=0):
    """
    Draw N_NULL random unit vectors in attention parameter space.
    Compute their top-k SAE feature sets, return all pairwise Jaccards.
    """
    rng     = np.random.RandomState(seed)
    attn_keys = sorted(key for key in state_dict
                       if "self_attn" in key and "weight" in key and "bias" not in key)
    D_attn  = sum(state_dict[k].numel() for k in attn_keys)

    sets = []
    for i in range(n_null):
        v  = rng.randn(D_attn).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-30
        dh = compute_delta_h(state_dict, v, a_np, b_np, device)
        dz = get_delta_z(sae, H_np, dh, device)
        fs, _ = top_features(dz, k=k)
        sets.append(fs)
        if (i+1) % 10 == 0:
            print(f"      null {i+1}/{n_null}")

    jaccards = []
    for i in range(n_null):
        for j in range(i+1, n_null):
            jaccards.append(jaccard(sets[i], sets[j]))
    return np.array(jaccards), sets


def null_jaccard_angle_matched(sae, H_np, state_dict, a_np, b_np,
                               device, target_angle_deg=45.0,
                               n_pairs=30, k=TOP_K_FEAT, seed=1):
    """
    Draw pairs of random directions with a fixed subspace angle (≈target_angle_deg).
    This tests: given two directions that are as similar as v1↔v2, how much
    SAE feature overlap do we expect by geometry alone?
    """
    rng    = np.random.RandomState(seed)
    attn_keys = sorted(key for key in state_dict
                       if "self_attn" in key and "weight" in key and "bias" not in key)
    D_attn = sum(state_dict[k].numel() for k in attn_keys)
    cos_t  = math.cos(math.radians(target_angle_deg))

    jaccards = []
    for i in range(n_pairs):
        u = rng.randn(D_attn).astype(np.float32)
        u /= np.linalg.norm(u) + 1e-30
        # v = cos(θ)·u + sin(θ)·w  where w ⊥ u
        w = rng.randn(D_attn).astype(np.float32)
        w -= (w @ u) * u
        w /= np.linalg.norm(w) + 1e-30
        sin_t = math.sqrt(max(0, 1 - cos_t**2))
        v = cos_t * u + sin_t * w
        v /= np.linalg.norm(v) + 1e-30

        dh_u = compute_delta_h(state_dict, u, a_np, b_np, device)
        dh_v = compute_delta_h(state_dict, v, a_np, b_np, device)
        dz_u = get_delta_z(sae, H_np, dh_u, device)
        dz_v = get_delta_z(sae, H_np, dh_v, device)
        fu, _ = top_features(dz_u, k=k)
        fv, _ = top_features(dz_v, k=k)
        jaccards.append(jaccard(fu, fv))
        if (i+1) % 10 == 0:
            print(f"      angle-matched {i+1}/{n_pairs}")

    return np.array(jaccards)


# ─── Fourier analysis ─────────────────────────────────────────────────────────

def fourier_analysis_feature(z_f, test_pairs, fourier_arg_fn, p=P):
    """
    z_f: [N] activation of SAE feature f across test inputs.
    fourier_arg_fn: (a,b) -> scalar (the "argument" of the operation, e.g. a+b).

    Returns:
      power[freq]   normalised Fourier power  (freqs 1..p//2)
      dominant_freq int
      r_squared     fraction of variance explained by the top Fourier mode
    """
    # Build p-point mean signal: group by output value q = arg(a,b) mod p
    q_vals  = np.array([fourier_arg_fn(int(a), int(b)) % p for a, b in test_pairs])
    means   = np.zeros(p)
    counts  = np.zeros(p)
    for i, q in enumerate(q_vals):
        means[q]  += z_f[i]
        counts[q] += 1
    counts = np.maximum(counts, 1)
    means /= counts            # p-point mean signal s[q]

    # DFT
    freqs    = np.arange(1, p // 2 + 1)
    phases   = np.outer(np.arange(p, dtype=float), freqs) * (2 * np.pi / p)
    cos_basis = np.cos(phases)   # [p, n_freqs]
    sin_basis = np.sin(phases)

    cos_coeffs = means @ cos_basis / p   # [n_freqs]
    sin_coeffs = means @ sin_basis / p
    power      = cos_coeffs**2 + sin_coeffs**2   # [n_freqs]

    total_var    = np.var(means) + 1e-30
    dominant_idx = int(np.argmax(power))
    r_squared    = float(2 * power[dominant_idx] / total_var)   # ×2 for cos+sin

    # Normalise power to sum to 1
    power_norm   = power / (power.sum() + 1e-30)
    return power_norm, int(freqs[dominant_idx]), float(np.clip(r_squared, 0, 1))


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_figA_significance(obs_jaccards, null_random, null_angle, bulk_jaccards,
                           op_name, op_label):
    """Distribution of Jaccards under 3 nulls vs observed top-3 values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histograms for nulls
    bins = np.linspace(0, 1, 26)
    ax.hist(null_random,  bins=bins, alpha=0.55, color="#95a5a6",
            label=f"Random dirs (n={len(null_random)})", density=True)
    ax.hist(null_angle,   bins=bins, alpha=0.55, color="#f39c12",
            label=f"Angle-matched 45° (n={len(null_angle)})", density=True)
    ax.hist(bulk_jaccards, bins=bins, alpha=0.55, color="#3498db",
            label=f"Bulk directions v4-v6 (n={len(bulk_jaccards)})", density=True)

    # Observed top-3 values
    for j_val, label, color in obs_jaccards:
        ax.axvline(j_val, color=color, linewidth=2.5, linestyle="-",
                   label=f"{label}: {j_val:.2f}")

    ax.set_xlabel("Jaccard similarity of top-20 SAE features")
    ax.set_ylabel("Density")
    ax.set_title(f"Significance of v1/v2/v3 SAE Feature Overlap — {op_label}\n"
                 f"Is the 60% Jaccard above chance?", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)

    # Annotate combinatorial null
    comb_null = (TOP_K_FEAT**2 / D_SAE) / (2*TOP_K_FEAT - TOP_K_FEAT**2/D_SAE)
    ax.axvline(comb_null, color="k", linewidth=1.2, linestyle=":",
               label=f"Combinatorial null: {comb_null:.3f}", alpha=0.7)
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figA_significance_{op_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figA_significance_{op_name}.png")


def plot_figB_shared_features(shared_features, mean_deltas, op_name, op_label):
    """Bar chart of mean |Δz| for shared features vs random non-shared."""
    n_shared = len(shared_features)
    if n_shared == 0:
        print("  No shared features to plot")
        return

    fig, axes = plt.subplots(1, N_DIR_TOP, figsize=(5*N_DIR_TOP, 4))
    fig.suptitle(f"SAE Features: Shared Pool vs Exclusive — {op_label}\n"
                 f"Shared = in top-{TOP_K_FEAT} for ALL of v1, v2, v3",
                 fontsize=10, fontweight="bold")

    for k, ax in enumerate(axes):
        md  = mean_deltas[k]               # [D_SAE]
        top = np.argsort(md)[::-1][:TOP_K_FEAT]
        colors = ["#e74c3c" if f in shared_features else "#95a5a6" for f in top]

        ax.bar(range(TOP_K_FEAT), md[top], color=colors, alpha=0.85,
               edgecolor="k", linewidth=0.3)
        ax.set_xticks(range(TOP_K_FEAT))
        ax.set_xticklabels([f"f{f}" for f in top], rotation=60,
                           ha="right", fontsize=5)
        ax.set_ylabel("mean |Δz_k[f]|")
        ax.set_title(f"v{k+1} top-{TOP_K_FEAT}\n"
                     f"(red = shared with all 3)", fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figB_shared_features_{op_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figB_shared_features_{op_name}.png")


def plot_figC_fourier_spectra(shared_feat_fourier, non_shared_fourier,
                              freqs, op_name, op_label):
    """Fourier power spectrum for each shared feature."""
    n_shared = len(shared_feat_fourier)
    if n_shared == 0: return

    n_cols = min(n_shared, 6)
    n_rows = math.ceil(n_shared / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3*n_cols, 2.5*n_rows), squeeze=False)
    fig.suptitle(f"Fourier Spectrum of Shared SAE Features — {op_label}\n"
                 "Power at each frequency in the mean-activation signal s_f[a+b]",
                 fontsize=10, fontweight="bold")

    for i, (feat_id, power, dom_freq, r2) in enumerate(shared_feat_fourier):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        ax.bar(freqs, power, width=0.8, color="#e74c3c", alpha=0.8)
        ax.axvline(dom_freq, color="k", linewidth=1.2, linestyle="--")
        ax.set_title(f"feature {feat_id}\npeak freq={dom_freq}  R²={r2:.2f}",
                     fontsize=7)
        ax.set_xlabel("freq", fontsize=6)
        ax.set_yticks([])
        ax.axhline(1.0/len(freqs), color="gray", linestyle=":", linewidth=0.8)

    # Hide unused axes
    for i in range(len(shared_feat_fourier), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figC_fourier_spectra_{op_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figC_fourier_spectra_{op_name}.png")


def plot_figD_fourier_summary(shared_feat_fourier, non_shared_fourier,
                              op_name, op_label):
    """R² comparison: shared features vs non-shared vs random."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Fourier Alignment of SAE Features — {op_label}\n"
                 "Do shared features encode specific Fourier modes?",
                 fontsize=10, fontweight="bold")

    # Panel 1: dominant frequency distribution
    ax = axes[0]
    shared_freqs = [x[2] for x in shared_feat_fourier]
    nonshr_freqs = [x[2] for x in non_shared_fourier]
    bins = np.arange(0.5, P//2 + 1.5)
    ax.hist(nonshr_freqs, bins=bins, alpha=0.5, color="#95a5a6",
            label=f"Non-shared (n={len(nonshr_freqs)})", density=True)
    ax.hist(shared_freqs, bins=bins, alpha=0.8, color="#e74c3c",
            label=f"Shared in top-3 (n={len(shared_freqs)})", density=True)
    ax.set_xlabel("Dominant Fourier frequency")
    ax.set_ylabel("Density")
    ax.set_title("Dominant frequency distribution")
    ax.legend(fontsize=8)

    # Panel 2: R² comparison
    ax = axes[1]
    shared_r2  = [x[3] for x in shared_feat_fourier]
    nonshr_r2  = [x[3] for x in non_shared_fourier]
    parts = ax.violinplot([nonshr_r2, shared_r2], positions=[0, 1],
                          showmedians=True, showextrema=True)
    parts["bodies"][0].set_facecolor("#95a5a6")
    parts["bodies"][1].set_facecolor("#e74c3c")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-shared", "Shared"], fontsize=9)
    ax.set_ylabel("Fourier R²\n(fraction of variance from dominant mode)")
    ax.set_title("Fourier alignment strength")
    ax.axhline(1/P, color="k", linestyle=":", alpha=0.5, label="random baseline")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # Print summary stats
    if shared_r2:
        print(f"    Shared features:     mean R²={np.mean(shared_r2):.3f}  "
              f"median={np.median(shared_r2):.3f}")
    if nonshr_r2:
        print(f"    Non-shared features: mean R²={np.mean(nonshr_r2):.3f}  "
              f"median={np.median(nonshr_r2):.3f}")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figD_fourier_summary_{op_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figD_fourier_summary_{op_name}.png")


def plot_figE_activation_scatter(shared_feat_fourier, Z, test_pairs, op_name):
    """Scatter: z_f(x) vs cos(2π·freq*·(a+b)/p) for each shared feature."""
    n = min(len(shared_feat_fourier), 6)
    if n == 0: return

    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 3.5), squeeze=False)
    fig.suptitle(f"SAE Feature Activation vs Fourier Mode — {op_name}\n"
                 "Each point = one test input (a,b)",
                 fontsize=10, fontweight="bold")

    for i, (feat_id, power, dom_freq, r2) in enumerate(shared_feat_fourier[:n]):
        ax  = axes[0][i]
        z_f = Z[:, feat_id]
        apb = np.array([(a+b)%P for a,b in test_pairs], dtype=float)
        cos_mode = np.cos(2 * np.pi * dom_freq * apb / P)

        ax.scatter(cos_mode, z_f, s=2, alpha=0.3, color="#e74c3c")
        # Fit line
        m, b = np.polyfit(cos_mode, z_f, 1)
        xs   = np.array([-1.0, 1.0])
        ax.plot(xs, m*xs + b, "k-", linewidth=1.5, zorder=5)
        r = float(np.corrcoef(cos_mode, z_f)[0, 1])
        ax.set_title(f"f{feat_id}  freq={dom_freq}\nR²={r2:.2f}  r={r:.2f}",
                     fontsize=7)
        ax.set_xlabel(f"cos(2π·{dom_freq}·(a+b)/p)", fontsize=6)
        ax.set_ylabel("SAE activation", fontsize=6)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figE_activation_scatter_{op_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figE_activation_scatter_{op_name}.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_op(op_name, op_cfg, device):
    sweep_path = SWEEP_DIR / f"{op_name}_wd1.0_s42.pt"
    cache_path = CACHE_PATHS.get(op_name)
    if not cache_path or not sweep_path.exists() or not cache_path.exists():
        print(f"  Skipping {op_name}"); return None

    print(f"\n{'='*60}\n  {op_name}  —  {op_cfg['label']}\n{'='*60}")

    sweep     = torch.load(sweep_path, map_location="cpu", weights_only=False)
    cache     = torch.load(cache_path, map_location="cpu", weights_only=False)
    attn_logs = sweep["attn_logs"]
    metrics   = cache.get("metrics") or sweep.get("metrics", [])
    test_pairs = [(int(a), int(b)) for a,b in cache["test_pairs"]]
    a_np = np.array([a for a,b in test_pairs])
    b_np = np.array([b for a,b in test_pairs])

    updates = compute_updates(attn_logs)
    ckpts   = select_postgrok_checkpoint(attn_logs, metrics)

    print(f"  Focusing on post-grok checkpoint: step={ckpts[PHASE][1]}")
    t_idx, step = ckpts[PHASE]
    _, state_dict = nearest_ckpt(step, cache)

    svd_res = window_svd(updates, t_idx)
    if svd_res is None: print("  SVD failed"); return None
    S, Vt = svd_res
    n_dir_total = min(N_DIR_TOP + N_DIR_BULK + 2, Vt.shape[0])

    # ── Baseline residuals ────────────────────────────────────────────────────
    print("  Computing baseline residuals...")
    H = get_residuals(state_dict, a_np, b_np, device)
    print(f"  H shape: {H.shape}  ||h|| mean={np.linalg.norm(H, axis=1).mean():.3f}")

    # ── Train SAE ─────────────────────────────────────────────────────────────
    print("  Training SAE...")
    sae, losses = train_sae(H, device)
    Z_base = get_sae_activations(sae, H, device)   # [N, D_SAE]
    recon_err = float(np.mean((H - sae.decode(
        torch.from_numpy(Z_base).to(device)).detach().cpu().numpy())**2))
    print(f"  SAE reconstruction MSE: {recon_err:.6f}")
    print(f"  Mean active features per input: "
          f"{(Z_base > 0).sum(1).mean():.1f}  (target k={K_SAE})")

    # ── Δh and Δz for top-3 and bulk directions ───────────────────────────────
    print(f"  Computing Δh/Δz for {n_dir_total} directions...")
    delta_hs   = {}
    delta_zs   = {}
    mean_deltas = {}
    top_sets   = {}

    for k in range(n_dir_total):
        dh = compute_delta_h(state_dict, Vt[k], a_np, b_np, device)
        dz = get_delta_z(sae, H, dh, device)
        delta_hs[k]    = dh
        delta_zs[k]    = dz
        fs, md         = top_features(dz)
        top_sets[k]    = fs
        mean_deltas[k] = md
        print(f"    v{k+1}: ||Δh||={np.linalg.norm(dh)/math.sqrt(len(test_pairs)):.5f}"
              f"  top-{TOP_K_FEAT} overlap peakedness="
              f"{md.max() / (md.mean()+1e-30):.1f}x")

    # Observed Jaccards among top-3
    obs_jaccards = []
    for i in range(N_DIR_TOP):
        for j in range(i+1, N_DIR_TOP):
            j_val = jaccard(top_sets[i], top_sets[j])
            obs_jaccards.append((j_val, f"v{i+1}↔v{j+1}", ["#e74c3c","#3498db","#2ecc71"][i]))
            print(f"  v{i+1}↔v{j+1} Jaccard: {j_val:.3f}")

    # Bulk baseline Jaccards
    bulk_start  = N_DIR_TOP
    bulk_jaccards = []
    for i in range(bulk_start, min(bulk_start + N_DIR_BULK, n_dir_total)):
        for j in range(i+1, min(bulk_start + N_DIR_BULK, n_dir_total)):
            bulk_jaccards.append(jaccard(top_sets[i], top_sets[j]))
    print(f"  Bulk Jaccard (v{bulk_start+1}-v{bulk_start+N_DIR_BULK}): "
          f"mean={np.mean(bulk_jaccards):.3f}  "
          f"range=[{min(bulk_jaccards):.3f}, {max(bulk_jaccards):.3f}]"
          if bulk_jaccards else "  No bulk pairs")

    # ── Significance baselines ────────────────────────────────────────────────
    print(f"\n  Running random-direction null ({N_NULL} directions)...")
    null_rand_jaccards, null_rand_sets = null_jaccard_random(
        sae, H, state_dict, a_np, b_np, device)
    print(f"  Random null Jaccard: mean={null_rand_jaccards.mean():.3f}  "
          f"95th pct={np.percentile(null_rand_jaccards, 95):.3f}")

    print(f"\n  Running angle-matched null (45°, {30} pairs)...")
    null_angle_jaccards = null_jaccard_angle_matched(
        sae, H, state_dict, a_np, b_np, device)
    print(f"  Angle-matched null: mean={null_angle_jaccards.mean():.3f}  "
          f"95th pct={np.percentile(null_angle_jaccards, 95):.3f}")

    # Significance summary
    obs_min = min(j for j,_,_ in obs_jaccards)
    pval_rand  = float(np.mean(null_rand_jaccards  >= obs_min))
    pval_angle = float(np.mean(null_angle_jaccards >= obs_min))
    print(f"\n  Observed min Jaccard: {obs_min:.3f}")
    print(f"  p-value vs random null:        {pval_rand:.4f}")
    print(f"  p-value vs angle-matched null: {pval_angle:.4f}")
    comb_null = (TOP_K_FEAT**2 / D_SAE) / (2*TOP_K_FEAT - TOP_K_FEAT**2/D_SAE)
    print(f"  Combinatorial null:            {comb_null:.4f}")

    # ── Shared feature pool ───────────────────────────────────────────────────
    shared  = top_sets[0] & top_sets[1] & top_sets[2]
    any_two = (top_sets[0]|top_sets[1]|top_sets[2]) - \
              (top_sets[0] ^ top_sets[1] ^ top_sets[2])   # approximate "at least 2"
    print(f"\n  Features in ALL top-3:    {len(shared)}")
    print(f"  Shared feature IDs: {sorted(shared)}")

    # ── Fourier analysis ──────────────────────────────────────────────────────
    fourier_arg = op_cfg.get("fourier_arg")
    freqs = np.arange(1, P//2 + 1)

    shared_feat_fourier  = []
    non_shared_feat_list = list(
        (top_sets[0] | top_sets[1] | top_sets[2]) - shared)[:20]
    non_shared_fourier   = []

    if fourier_arg is not None:
        print("\n  Fourier analysis of shared features...")
        for feat_id in sorted(shared):
            z_f = Z_base[:, feat_id]
            power, dom_freq, r2 = fourier_analysis_feature(
                z_f, test_pairs, fourier_arg)
            shared_feat_fourier.append((feat_id, power, dom_freq, r2))
            print(f"    f{feat_id:4d}: dom_freq={dom_freq:3d}  R²={r2:.3f}")

        print("\n  Fourier analysis of non-shared features (sample)...")
        for feat_id in non_shared_feat_list:
            z_f = Z_base[:, feat_id]
            power, dom_freq, r2 = fourier_analysis_feature(
                z_f, test_pairs, fourier_arg)
            non_shared_fourier.append((feat_id, power, dom_freq, r2))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_figA_significance(obs_jaccards, null_rand_jaccards,
                           null_angle_jaccards, bulk_jaccards,
                           op_name, op_cfg["label"])
    plot_figB_shared_features(shared, mean_deltas, op_name, op_cfg["label"])
    if shared_feat_fourier:
        plot_figC_fourier_spectra(shared_feat_fourier, non_shared_fourier,
                                  freqs, op_name, op_cfg["label"])
        plot_figD_fourier_summary(shared_feat_fourier, non_shared_fourier,
                                  op_name, op_cfg["label"])
        plot_figE_activation_scatter(shared_feat_fourier, Z_base,
                                     test_pairs, op_name)

    return {
        "op_name":          op_name,
        "obs_jaccards":     [(j,l) for j,l,_ in obs_jaccards],
        "null_rand":        null_rand_jaccards.tolist(),
        "null_angle":       null_angle_jaccards.tolist(),
        "bulk_jaccards":    bulk_jaccards,
        "pval_rand":        pval_rand,
        "pval_angle":       pval_angle,
        "shared_features":  sorted(shared),
        "shared_fourier":   [(f, int(df), float(r2))
                             for f,_,df,r2 in shared_feat_fourier],
        "non_shared_fourier": [(f, int(df), float(r2))
                               for f,_,df,r2 in non_shared_fourier],
    }


OPERATIONS = {
    "add":   {"label": "(a+b) mod p",    "fourier_arg": lambda a,b: (a+b)%P},
    "sub":   {"label": "(a-b) mod p",    "fourier_arg": lambda a,b: (a-b)%P},
    "mul":   {"label": "(a*b) mod p",    "fourier_arg": None},
    "x2_y2": {"label": "(a²+b²) mod p", "fourier_arg": lambda a,b: (a*a+b*b)%P},
}


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Combinatorial null E[Jaccard] = "
          f"{(TOP_K_FEAT**2/D_SAE)/(2*TOP_K_FEAT - TOP_K_FEAT**2/D_SAE):.4f}")

    all_results = {}
    for op_name, op_cfg in OPERATIONS.items():
        res = run_op(op_name, op_cfg, device)
        if res:
            all_results[op_name] = res

    # Cross-op summary
    print("\n" + "="*60)
    print("SIGNIFICANCE SUMMARY")
    print("="*60)
    print(f"{'Op':8s}  {'Obs(min)':9s}  {'p_rand':8s}  {'p_angle':8s}  "
          f"{'null_rand_95':12s}  {'shared_N':8s}  {'shared_R2_mean':14s}")
    for op_name, res in all_results.items():
        obs_min  = min(j for j,_ in res["obs_jaccards"])
        null_95  = float(np.percentile(res["null_rand"], 95))
        n_shared = len(res["shared_features"])
        r2_vals  = [r2 for _,_,r2 in res.get("shared_fourier", [])]
        r2_mean  = np.mean(r2_vals) if r2_vals else float("nan")
        print(f"{op_name:8s}  {obs_min:.3f}      {res['pval_rand']:.4f}    "
              f"{res['pval_angle']:.4f}    {null_95:.3f}         "
              f"{n_shared:8d}  {r2_mean:.3f}")

    torch.save(all_results, PLOT_DIR / "sae_fourier_results.pt")
    print(f"\nResults saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
