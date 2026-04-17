#!/usr/bin/env python3
"""
v123_feature_attribution.py

Measures whether the {v1, v2, v3} block in update-space SVD corresponds to
actual computational features — specifically which attention heads and matrix
types (Q/K/V/O) each spectral direction loads on, and whether directions that
are geometrically concentrated on specific heads are also functionally distinct.

Measurements
------------
1. Head mass decomposition
   For each direction v_k and each (layer, head), compute the fraction of
   ||v_k||^2 that lives in that head's parameter subspace. Reports a
   (N_DIR x N_HEADS_TOTAL) heatmap per training phase.

2. Matrix-type decomposition (QK vs VO)
   Sum head mass over heads, split by matrix type: Q+K vs V+O.
   QK mass  → the direction shapes attention routing (where information flows)
   VO mass  → the direction shapes value extraction (what information is read)

3. Head purity
   purity_k = max_{layer,head} head_mass_{layer,head,k}
   Tracks how head-localized each direction is. purity → 1 means the direction
   lies entirely in one head's parameters; purity → 1/8 means uniform spread.

4. Dominant-head assignment
   For each k, which (layer, head) carries the most mass? Tracks whether v1's
   dominant head stays stable across training.

5. Layer-level attention-output sensitivity
   Perturb θ → θ + ε·v_k. Hook on each TransformerEncoderLayer's self_attn
   output. Report per-layer ||Δ(attn_output)||_F, normalized by perturbation size.
   Tells us: does v_k primarily affect layer-0 or layer-1 attention computations?

6. The Olah bridge (parameter-space version)
   For each direction k:
     head_purity_k      ↔  diagonal dominance of co-usage matrix M̃
     effective_imp_k    =  σ_k × head_purity_k  ↔  corrected importance
     interference_k     =  1 − head_purity_k    ↔  interference fraction
   Plots raw σ_k vs effective_imp_k, and tracks whether top-3 become more
   head-pure at grokking.

Output
------
Figures saved to spectral/feature_attribution_plots/
  figA_head_mass_heatmap.png   — (N_DIR × 8 heads) per phase
  figB_qkvo_decomposition.png  — QK vs VO mass fraction per direction, per phase
  figC_head_purity.png         — purity trajectory for k=0,1,2 and bulk mean
  figD_dominant_head.png       — dominant (layer, head) for each k across phases
  figE_attn_sensitivity.png    — per-layer attention output sensitivity per direction
  figF_olah_bridge.png         — raw vs head-purity-corrected importance
Results also saved to feature_attribution_plots/attribution_results.pt
"""

import math, time, random, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR.parent / "coherence_edge_results"
SWEEP_DIR    = SCRIPT_DIR.parent.parent / "grok_sweep_results"
PLOT_DIR     = SCRIPT_DIR / "feature_attribution_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ─── Model constants (must match training config) ─────────────────────────────

D_MODEL   = 128
N_HEADS   = 4
D_HEAD    = D_MODEL // N_HEADS   # 32
N_LAYERS  = 2
MAT_NAMES = ["WQ", "WK", "WV", "WO"]
MAT_SIZE  = D_MODEL * D_MODEL    # 16384  (each matrix is [128, 128])
LAYER_SIZE = len(MAT_NAMES) * MAT_SIZE   # 65536
TOTAL_DIM  = N_LAYERS * LAYER_SIZE       # 131072

# Layout of the flattened attention vector (matches flatten_attn_from_logs order):
#   Layer 0: WQ[16384] | WK[16384] | WV[16384] | WO[16384]
#   Layer 1: WQ[16384] | WK[16384] | WV[16384] | WO[16384]
# WQ, WK, WV are [D_MODEL, D_MODEL] matrices where rows = n_heads × d_head
#   → head h = rows [h*D_HEAD : (h+1)*D_HEAD] (contiguous in C-order flatten)
# WO is [D_MODEL, D_MODEL] where cols = n_heads × d_head
#   → head h = cols [h*D_HEAD : (h+1)*D_HEAD] (non-contiguous in C-order flatten)

MAT_OFFSETS = {
    "WQ": 0,
    "WK": MAT_SIZE,
    "WV": 2 * MAT_SIZE,
    "WO": 3 * MAT_SIZE,
}

# SVD / analysis hyperparams (must match cousage_experiment)
W      = 20    # sliding window size
N_DIR  = 15    # directions to analyze

# Head labels for plotting
HEAD_LABELS = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
# ['L0H0','L0H1','L0H2','L0H3','L1H0','L1H1','L1H2','L1H3']

PHASE_ORDER  = ["early_coherent", "memorization", "grokking_transition", "stable_postgrok"]
PHASE_LABELS = {
    "early_coherent":    "Early",
    "memorization":      "Memorization",
    "grokking_transition": "Grok Trans.",
    "stable_postgrok":   "Post-Grok",
}
PHASE_COLORS = {
    "early_coherent":    "#2ecc71",
    "memorization":      "#e74c3c",
    "grokking_transition": "#f39c12",
    "stable_postgrok":   "#3498db",
}


# ─── Data config ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    P: int = 97
    TRAIN_FRACTION: float = 0.5
    D_MODEL: int = 128
    N_LAYERS: int = 2
    N_HEADS: int = 4
    D_FF: int = 256
    DROPOUT: float = 0.0
    LR: float = 1e-3
    BATCH_SIZE: int = 512
    GRAD_CLIP: float = 1.0
    ACC_BS: int = 2048
    WEIGHT_DECAY: float = 1.0
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.98
    SEED: int = 42


# ─── Model ───────────────────────────────────────────────────────────────────

class ModOpTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.P, cfg.D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(2, cfg.D_MODEL) / math.sqrt(cfg.D_MODEL))
        enc = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL, nhead=cfg.N_HEADS,
            dim_feedforward=cfg.D_FF, dropout=cfg.DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg.N_LAYERS)
        self.ln   = nn.LayerNorm(cfg.D_MODEL)
        self.head = nn.Linear(cfg.D_MODEL, cfg.P)

    def forward(self, a, b):
        x = torch.stack([a, b], dim=1)
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        return self.head(self.ln(h[:, 0, :]))


# ─── Operations ──────────────────────────────────────────────────────────────

def op_add(a, b, p):    return (a + b) % p
def op_sub(a, b, p):    return (a - b) % p
def op_mul(a, b, p):    return (a * b) % p
def op_x2_y2(a, b, p):  return (a * a + b * b) % p

OPERATIONS = {
    "add":    {"fn": op_add,    "label": "(a+b) mod p",       "restrict_nonzero": False},
    "sub":    {"fn": op_sub,    "label": "(a-b) mod p",       "restrict_nonzero": False},
    "mul":    {"fn": op_mul,    "label": "(a*b) mod p",       "restrict_nonzero": True},
    "x2_y2":  {"fn": op_x2_y2,  "label": "(a²+b²) mod p",    "restrict_nonzero": False},
}

CACHE_PATHS = {
    "add":    RESULTS_DIR / "training_cache.pt",
    "sub":    RESULTS_DIR / "training_cache_sub.pt",
    "mul":    RESULTS_DIR / "training_cache_mul.pt",
    "x2_y2":  RESULTS_DIR / "training_cache_x2_y2.pt",
}


def build_dataset(p, frac, seed, restrict_nonzero=False):
    if restrict_nonzero:
        pairs = [(a, b) for a in range(1, p) for b in range(1, p)]
    else:
        pairs = [(a, b) for a in range(p) for b in range(p)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = int(frac * len(pairs))
    return pairs[:n], pairs[n:]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ─── Head mask construction ───────────────────────────────────────────────────

def build_head_index_arrays():
    """
    Build index arrays into the flattened attention vector for each
    (layer, head, matrix_name) component.

    Returns
    -------
    dict: (layer, head, mat_name) -> np.ndarray of int indices into [0, TOTAL_DIM)
    Also returns per-layer (layer, mat_name) -> indices for matrix-level breakdown.
    """
    head_indices = {}
    for layer in range(N_LAYERS):
        layer_off = layer * LAYER_SIZE
        for mat_name in MAT_NAMES:
            mat_off = layer_off + MAT_OFFSETS[mat_name]
            for head in range(N_HEADS):
                idxs = []
                if mat_name in ("WQ", "WK", "WV"):
                    # Matrix shape [D_MODEL, D_MODEL] = [n_heads*d_head, d_model]
                    # Head h occupies rows [h*D_HEAD : (h+1)*D_HEAD], all columns
                    # In C-order flatten: rows h*D_HEAD..(h+1)*D_HEAD → contiguous block
                    row_start = head * D_HEAD
                    row_end   = (head + 1) * D_HEAD
                    start = mat_off + row_start * D_MODEL
                    end   = mat_off + row_end   * D_MODEL
                    idxs  = list(range(start, end))
                else:  # WO
                    # Matrix shape [D_MODEL, D_MODEL] = [d_model, n_heads*d_head]
                    # Head h occupies cols [h*D_HEAD : (h+1)*D_HEAD], all rows
                    # In C-order flatten: non-contiguous — each row contributes D_HEAD entries
                    col_start = head * D_HEAD
                    col_end   = (head + 1) * D_HEAD
                    for row in range(D_MODEL):
                        row_base = mat_off + row * D_MODEL
                        idxs.extend(range(row_base + col_start, row_base + col_end))
                head_indices[(layer, head, mat_name)] = np.array(idxs, dtype=np.int32)

    # Verify sizes: each (layer, head, mat) should have D_HEAD * D_MODEL = 4096 entries
    for key, idxs in head_indices.items():
        expected = D_HEAD * D_MODEL
        assert len(idxs) == expected, f"mask size mismatch for {key}: {len(idxs)} vs {expected}"
    return head_indices


# ─── Attribution computation ──────────────────────────────────────────────────

def compute_head_attribution(vk, head_indices):
    """
    Decompose direction vk (shape [TOTAL_DIM]) by (layer, head) and matrix type.

    Returns
    -------
    head_mass : dict (layer, head) -> fraction of ||vk||^2  (sums to 1 over all heads)
    mat_mass  : dict (layer, head, mat_name) -> fraction
    type_mass : dict mat_name -> fraction summed over all heads/layers
    lh_mat_mass: dict (layer, mat_name) -> fraction summed over heads
    """
    total_sq = float(np.dot(vk, vk)) + 1e-30

    # Per (layer, head, mat)
    lhm_mass = {}
    for (layer, head, mat), idxs in head_indices.items():
        sq = float(np.sum(vk[idxs] ** 2))
        lhm_mass[(layer, head, mat)] = sq / total_sq

    # Aggregate to (layer, head)
    head_mass = {}
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            head_mass[(layer, head)] = sum(
                lhm_mass[(layer, head, m)] for m in MAT_NAMES
            )

    # Aggregate to matrix type (summed over all layers, heads)
    type_mass = {m: 0.0 for m in MAT_NAMES}
    for (layer, head, mat), v in lhm_mass.items():
        type_mass[mat] += v

    # Aggregate to (layer, mat) summed over heads
    lm_mass = {}
    for layer in range(N_LAYERS):
        for mat in MAT_NAMES:
            lm_mass[(layer, mat)] = sum(lhm_mass[(layer, h, mat)] for h in range(N_HEADS))

    return head_mass, lhm_mass, type_mass, lm_mass


def head_purity(head_mass):
    """Max head mass fraction over all (layer, head) pairs."""
    return max(head_mass.values())


def dominant_head(head_mass):
    """Returns (layer, head) with highest mass."""
    return max(head_mass.keys(), key=lambda k: head_mass[k])


def qkvo_fractions(type_mass):
    """QK fraction = (WQ+WK)/(all), VO fraction = (WV+WO)/(all)."""
    total = sum(type_mass.values()) + 1e-30
    qk = (type_mass["WQ"] + type_mass["WK"]) / total
    vo = (type_mass["WV"] + type_mass["WO"]) / total
    return qk, vo


# ─── SVD helpers (matching cousage_experiment) ───────────────────────────────

def flatten_attn_from_logs(attn_entry):
    parts = []
    for layer_data in sorted(attn_entry["layers"], key=lambda x: x["layer"]):
        for key in ["WQ", "WK", "WV", "WO"]:
            parts.append(layer_data[key].flatten().float())
    return torch.cat(parts)


def compute_updates_from_logs(attn_logs):
    flat = [flatten_attn_from_logs(e).numpy() for e in attn_logs]
    return [flat[i] - flat[i - 1] for i in range(1, len(flat))]


def sliding_window_svd(updates, t_idx):
    start = max(0, t_idx - W + 1)
    end   = t_idx + 1
    if end - start < 3:
        return None
    X = np.stack(updates[start:end])
    X -= X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return {"S": S, "Vt": Vt, "U": U}


def find_edge(S, max_k=10):
    k = min(max_k, len(S) - 1)
    if k < 1:
        return 0, 1.0
    total  = S.sum() + 1e-15
    mass   = S[:k] / total
    gap    = S[:k] / (S[1:k + 1] + 1e-15)
    score  = mass * gap
    kstar  = int(np.argmax(score))
    return kstar, float(score[kstar])


# ─── Checkpoint selection (simplified version) ───────────────────────────────

def select_phase_checkpoints(cache_data, attn_logs, updates):
    """
    Pick 4 representative checkpoints: early / memorization / grokking / post-grok.
    Returns list of (phase_name, t_idx, step).
    """
    metrics = cache_data["metrics"]
    attn_steps = [e["step"] for e in attn_logs]
    update_steps = attn_steps[1:]  # update[i] is the delta arriving at attn_steps[i+1]
    n_updates = len(updates)

    metrics_by_step = {m["step"]: m for m in metrics}

    def acc_at_step(step):
        if step in metrics_by_step:
            return metrics_by_step[step]
        nearest = min(metrics, key=lambda m: abs(m["step"] - step))
        return nearest

    # Grokking step: first step where test_acc > 0.5
    grok_step = next((m["step"] for m in metrics if m["test_acc"] > 0.5), metrics[-1]["step"])
    print(f"    Grokking step: {grok_step}")

    selected = {}

    # 1. Early: first t_idx where some data exists (~10% through)
    t_early = max(W, n_updates // 10)
    selected["early_coherent"] = (t_early, update_steps[t_early])

    # 2. Memorization: midpoint of [start, grok_step] where train_acc > 0.8, test_acc < 0.3
    mem_indices = [
        t for t in range(n_updates)
        if acc_at_step(update_steps[t])["train_acc"] > 0.8
        and acc_at_step(update_steps[t])["test_acc"] < 0.3
    ]
    if mem_indices:
        mid = mem_indices[len(mem_indices) // 2]
        selected["memorization"] = (mid, update_steps[mid])
    else:
        mid_step = grok_step // 2
        best = min(range(n_updates), key=lambda i: abs(update_steps[i] - mid_step))
        selected["memorization"] = (best, update_steps[best])

    # 3. Grokking transition: nearest to grok_step
    best = min(range(n_updates), key=lambda i: abs(update_steps[i] - grok_step))
    selected["grokking_transition"] = (best, update_steps[best])

    # 4. Post-grok: last checkpoint with test_acc > 0.95
    post = next(
        (t for t in range(n_updates - 1, -1, -1)
         if acc_at_step(update_steps[t])["test_acc"] > 0.95),
        n_updates - 1
    )
    selected["stable_postgrok"] = (post, update_steps[post])

    result = [(ph, *selected[ph]) for ph in PHASE_ORDER if ph in selected]
    for phase, t_idx, step in result:
        print(f"      {phase:25s} → t_idx={t_idx}, step={step}")
    return result


def find_nearest_state_dict(step, cache_data):
    checkpoints = cache_data["checkpoints"]
    best = min(checkpoints, key=lambda cs: abs(cs[0] - step))
    return best  # (step, state_dict)


# ─── Attention output sensitivity ────────────────────────────────────────────

def get_attn_keys(model):
    return sorted(
        name for name, _ in model.named_parameters()
        if "self_attn" in name and "weight" in name and "bias" not in name
    )


def perturb_state_dict(state_dict, vk_np, eps_scale, model, attn_keys):
    """
    Returns a new state_dict where attention weights are perturbed along vk
    by eps_scale * ||θ_attn||.
    """
    # Compute norm of current attention weights
    flat_parts = []
    for key in attn_keys:
        flat_parts.append(state_dict[key].float().flatten())
    flat_attn = torch.cat(flat_parts)
    norm = float(flat_attn.norm())

    vk_t = torch.from_numpy(vk_np.copy()).float()
    vk_t = vk_t / (vk_t.norm() + 1e-30)   # ensure unit vector
    delta = eps_scale * norm * vk_t

    new_sd = {k: v.clone() for k, v in state_dict.items()}
    offset = 0
    for key in attn_keys:
        numel = new_sd[key].numel()
        new_sd[key] = (new_sd[key].float() +
                       delta[offset:offset + numel].reshape(new_sd[key].shape))
        offset += numel
    return new_sd


@torch.no_grad()
def compute_attn_layer_sensitivity(model, state_dict, vk_np, test_pairs,
                                   op_fn, cfg, device, eps_scale=0.005,
                                   n_samples=200):
    """
    For each direction vk, measure per-layer attention output sensitivity:
      sens_layer_l = (1/N) Σ_x ||attn_output_l(θ') - attn_output_l(θ)||_F

    Uses forward hooks on each TransformerEncoderLayer.self_attn to capture
    the (B, T, D_MODEL) attention output before the residual add.

    Returns: sens [N_LAYERS] array
    """
    attn_keys = get_attn_keys(model)
    N = min(n_samples, len(test_pairs))
    rng = random.Random(42)
    sample_pairs = rng.sample(test_pairs, N)

    def get_layer_outputs(sd):
        """Run forward pass and collect attention outputs per layer."""
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        model.eval()
        outputs = {}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, inp, out):
                # nn.MultiheadAttention returns (attn_out, attn_weights)
                # attn_out shape: [B, T, D_MODEL]
                if isinstance(out, tuple):
                    outputs[layer_idx] = out[0].detach().cpu()
                else:
                    outputs[layer_idx] = out.detach().cpu()
            return hook

        for i, layer in enumerate(model.encoder.layers):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(i)))

        # Process samples in batch
        ab    = torch.tensor(sample_pairs, device=device)
        a, b  = ab[:, 0], ab[:, 1]
        model(a, b)

        for h in hooks:
            h.remove()
        return outputs   # {layer_idx: [N, T, D_MODEL] tensor}

    # Baseline
    base_outputs = get_layer_outputs(state_dict)

    # Perturbed
    perturbed_sd  = perturb_state_dict(state_dict, vk_np, eps_scale, model, attn_keys)
    pert_outputs  = get_layer_outputs(perturbed_sd)

    # Sensitivity per layer: mean Frobenius norm of delta, normalized by eps
    # Flatten the [N, T, D_MODEL] tensor to [N, T*D_MODEL] and take row norms
    flat_parts = [state_dict[k].float().flatten() for k in attn_keys]
    norm_theta  = float(torch.cat(flat_parts).norm())
    eps         = eps_scale * norm_theta + 1e-30

    sens = np.zeros(N_LAYERS)
    for l in range(N_LAYERS):
        if l not in base_outputs or l not in pert_outputs:
            continue
        delta = (pert_outputs[l] - base_outputs[l]).float()  # [N, T, D_MODEL]
        # Frobenius per sample: mean over (T, D_MODEL), then average over samples
        per_sample = delta.reshape(N, -1).norm(dim=1).mean().item()
        sens[l] = per_sample / eps

    return sens


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_attribution(op_name, op_cfg, cfg, device):
    # attn_logs come from sweep data; state_dicts from training cache
    sweep_path = SWEEP_DIR / f"{op_name}_wd1.0_s42.pt"
    cache_path = CACHE_PATHS[op_name]

    if not sweep_path.exists():
        print(f"  Sweep data not found: {sweep_path}, skipping {op_name}")
        return None
    if not cache_path.exists():
        print(f"  Cache not found: {cache_path}, skipping {op_name}")
        return None

    print(f"\n{'='*60}")
    print(f"  Operation: {op_name}  ({op_cfg['label']})")
    print(f"{'='*60}")

    sweep_data = torch.load(sweep_path, map_location="cpu", weights_only=False)
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

    attn_logs = sweep_data["attn_logs"]
    # Merge metrics: prefer cache (longer), fall back to sweep
    if "metrics" not in cache_data or not cache_data["metrics"]:
        cache_data["metrics"] = sweep_data.get("metrics", sweep_data.get("log", []))

    print(f"  Loaded sweep: {len(attn_logs)} attn_log entries, "
          f"{len(cache_data['checkpoints'])} checkpoints in cache")

    # Build head index arrays (same for all ops)
    print("  Building head index arrays...")
    head_indices = build_head_index_arrays()

    # Compute weight updates
    updates = compute_updates_from_logs(attn_logs)
    print(f"  Computed {len(updates)} updates, dim={len(updates[0])}")

    # Build dataset
    _, test_pairs = build_dataset(cfg.P, cfg.TRAIN_FRACTION, cfg.SEED,
                                  op_cfg["restrict_nonzero"])
    op_fn = op_cfg["fn"]

    # Select phase checkpoints
    print("  Selecting phase checkpoints...")
    phase_checkpoints = select_phase_checkpoints(cache_data, attn_logs, updates)

    # Build model
    model = ModOpTransformer(cfg).to(device)
    model.eval()

    results = {}

    for phase_name, t_idx, step in phase_checkpoints:
        print(f"\n  [{phase_name}] step={step}, t_idx={t_idx}")

        # Get SVD at this checkpoint
        svd = sliding_window_svd(updates, t_idx)
        if svd is None:
            print("    SVD failed (insufficient window), skipping")
            continue

        S  = svd["S"]
        Vt = svd["Vt"]   # [n_svd, TOTAL_DIM]
        kstar, gap_score = find_edge(S)
        print(f"    SVD: {len(S)} singular values, k*={kstar}, gap={gap_score:.3f}")

        # Nearest state_dict
        ckpt_step, state_dict = find_nearest_state_dict(step, cache_data)
        print(f"    Using state_dict from step={ckpt_step}")

        n_dir = min(N_DIR, Vt.shape[0])

        # ── Head attribution for each direction ──────────────────────────────
        per_dir = []
        for k in range(n_dir):
            vk = Vt[k]   # [TOTAL_DIM]

            hm, lhm_mass, tm, lm = compute_head_attribution(vk, head_indices)
            pur  = head_purity(hm)
            dom  = dominant_head(hm)
            qk_f, vo_f = qkvo_fractions(tm)

            per_dir.append({
                "k":          k,
                "sigma":      float(S[k]),
                "head_mass":  hm,           # (layer,head) -> fraction
                "lhm_mass":   lhm_mass,     # (layer,head,mat) -> fraction
                "type_mass":  tm,           # mat_name -> fraction
                "lm_mass":    lm,           # (layer,mat) -> fraction
                "purity":     pur,
                "dominant":   dom,
                "qk_frac":    qk_f,
                "vo_frac":    vo_f,
            })

        # ── Attention sensitivity for top-N directions ────────────────────────
        # Run sensitivity for k=0,1,2 and a couple bulk directions
        sens_dirs = list(range(min(5, n_dir)))  # k = 0..4
        attn_sens = {}
        for k in sens_dirs:
            vk = Vt[k]
            s  = compute_attn_layer_sensitivity(
                    model, state_dict, vk, test_pairs, op_fn, cfg, device,
                    n_samples=200)
            attn_sens[k] = s
            print(f"    k={k}: attn_sens = L0={s[0]:.4f}, L1={s[1]:.4f}")

        results[phase_name] = {
            "step":      step,
            "kstar":     kstar,
            "gap_score": gap_score,
            "S":         S[:n_dir].tolist(),
            "per_dir":   per_dir,
            "attn_sens": attn_sens,
        }

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_figA_head_mass_heatmap(all_results, op_name, op_label):
    """(N_DIR × 8 heads) head mass heatmap per training phase."""
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    n_phases = len(phases)
    if n_phases == 0:
        return

    fig, axes = plt.subplots(1, n_phases, figsize=(5 * n_phases, 5), squeeze=False)
    fig.suptitle(f"Head Mass Decomposition — {op_label}\n"
                 f"||v_k[head]||² / ||v_k||²  (rows = directions, cols = heads)",
                 fontsize=11, fontweight="bold")

    for ax, phase in zip(axes[0], phases):
        phase_data = all_results[phase]
        per_dir    = phase_data["per_dir"]
        n_dir      = len(per_dir)

        # Build matrix [n_dir, N_LAYERS*N_HEADS]
        mat = np.zeros((n_dir, N_LAYERS * N_HEADS))
        for entry in per_dir:
            k = entry["k"]
            for layer in range(N_LAYERS):
                for head in range(N_HEADS):
                    col = layer * N_HEADS + head
                    mat[k, col] = entry["head_mass"][(layer, head)]

        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=0.25)
        ax.set_xticks(range(N_LAYERS * N_HEADS))
        ax.set_xticklabels(HEAD_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_dir))
        ax.set_yticklabels([f"v{k+1}" for k in range(n_dir)], fontsize=7)
        ax.set_title(f"{PHASE_LABELS[phase]}\n(step {phase_data['step']})",
                     fontsize=9)

        # Mark top-3 rows with a box
        for row in range(min(3, n_dir)):
            ax.add_patch(plt.Rectangle((-0.5, row - 0.5), N_LAYERS * N_HEADS, 1,
                                       fill=False, edgecolor="#e74c3c", linewidth=1.5))

        # Annotate uniform line (1/8 = 0.125 per head)
        ax.axhline(2.5, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figA_head_mass_heatmap_{op_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figA_head_mass_heatmap_{op_name}.png")


def plot_figB_qkvo(all_results, op_name, op_label):
    """QK vs VO mass fraction per direction per phase."""
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    n_phases = len(phases)
    if n_phases == 0:
        return

    fig, axes = plt.subplots(1, n_phases, figsize=(4 * n_phases, 4), squeeze=False)
    fig.suptitle(f"QK vs VO Mass Fraction — {op_label}\n"
                 "QK = routing (attention patterns), VO = values (feature extraction)",
                 fontsize=10, fontweight="bold")

    for ax, phase in zip(axes[0], phases):
        phase_data = all_results[phase]
        per_dir    = phase_data["per_dir"]
        n_dir      = len(per_dir)

        ks       = [e["k"] for e in per_dir]
        qk_fracs = [e["qk_frac"] for e in per_dir]
        vo_fracs = [e["vo_frac"] for e in per_dir]

        x = np.arange(n_dir)
        w = 0.35
        bars_qk = ax.bar(x - w/2, qk_fracs, w, label="QK (routing)", color="#3498db", alpha=0.8)
        bars_vo = ax.bar(x + w/2, vo_fracs, w, label="VO (values)",  color="#e74c3c", alpha=0.8)

        ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5, label="equal split")
        ax.set_xticks(x)
        ax.set_xticklabels([f"v{k+1}" for k in ks], fontsize=7)
        ax.set_ylim(0, 0.7)
        ax.set_ylabel("Mass fraction")
        ax.set_title(f"{PHASE_LABELS[phase]}\n(step {phase_data['step']})", fontsize=9)

        # Highlight top-3
        for row in range(min(3, n_dir)):
            ax.axvspan(row - 0.5, row + 0.5, color="#f39c12", alpha=0.08)

        if ax == axes[0, 0]:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figB_qkvo_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figB_qkvo_{op_name}.png")


def plot_figC_head_purity(all_results, op_name, op_label):
    """Head purity trajectory across training phases for k=0,1,2 and bulk mean."""
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    if not phases:
        return

    steps = [all_results[ph]["step"] for ph in phases]
    phase_x = list(range(len(phases)))

    fig, ax = plt.subplots(figsize=(6, 4))
    uniform_baseline = 1.0 / (N_LAYERS * N_HEADS)

    for k, color, ls in [(0, "#e74c3c", "-"), (1, "#3498db", "--"), (2, "#2ecc71", "-.")]:
        purity_vals = []
        for ph in phases:
            per_dir = all_results[ph]["per_dir"]
            if k < len(per_dir):
                purity_vals.append(per_dir[k]["purity"])
            else:
                purity_vals.append(np.nan)
        ax.plot(phase_x, purity_vals, marker="o", color=color, linestyle=ls,
                linewidth=2, label=f"v{k+1} purity")

    # Bulk mean (k=3..N_DIR-1)
    for ph_i, ph in enumerate(phases):
        per_dir = all_results[ph]["per_dir"]
        bulk_p  = [per_dir[k]["purity"] for k in range(3, len(per_dir))]
        if bulk_p:
            ax.scatter([ph_i], [np.mean(bulk_p)], color="#95a5a6",
                       marker="s", s=60, zorder=5)
    # Add bulk label once
    ax.scatter([], [], color="#95a5a6", marker="s", s=60, label="bulk mean")

    ax.axhline(uniform_baseline, color="k", linestyle=":", linewidth=1.2, alpha=0.5,
               label=f"uniform baseline (1/{N_LAYERS*N_HEADS})")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.3)

    ax.set_xticks(phase_x)
    ax.set_xticklabels([PHASE_LABELS[ph] for ph in phases], rotation=15, ha="right")
    ax.set_ylabel("Head purity = max_{layer,head} head_mass")
    ax.set_title(f"Head Purity Across Training — {op_label}\n"
                 "Higher = more concentrated in one head's parameters", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, None)

    # Annotate steps
    for i, step in enumerate(steps):
        ax.annotate(f"step\n{step}", (i, 0.02), ha="center", fontsize=6, color="#7f8c8d",
                    transform=ax.get_xaxis_transform())

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figC_head_purity_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figC_head_purity_{op_name}.png")


def plot_figD_dominant_head(all_results, op_name, op_label):
    """Dominant (layer, head) assignment for each direction k, per phase."""
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    if not phases:
        return

    n_dir = min(N_DIR, len(all_results[phases[0]]["per_dir"]))

    # Encode (layer, head) as integer 0..7
    def encode_head(layer, head):
        return layer * N_HEADS + head

    cmap = plt.cm.get_cmap("Set1", N_LAYERS * N_HEADS)

    fig, axes = plt.subplots(1, len(phases), figsize=(4 * len(phases), 4), squeeze=False)
    fig.suptitle(f"Dominant Head per Direction — {op_label}\n"
                 "Which (layer, head) carries most of v_k's parameter mass",
                 fontsize=10, fontweight="bold")

    for ax, phase in zip(axes[0], phases):
        per_dir = all_results[phase]["per_dir"]
        nd      = len(per_dir)
        S       = all_results[phase]["S"]

        dom_colors = []
        dom_ids    = []
        for entry in per_dir:
            l, h   = entry["dominant"]
            dom_id = encode_head(l, h)
            dom_ids.append(dom_id)
            dom_colors.append(cmap(dom_id))

        # Bar chart: x = direction k, y = purity of dominant head, color = dominant head
        purities = [e["purity"] for e in per_dir]
        bars = ax.bar(range(nd), purities, color=dom_colors, alpha=0.85, edgecolor="k",
                      linewidth=0.5)

        # Overlay: mark top-3 with bold edge
        for k in range(min(3, nd)):
            bars[k].set_linewidth(2.0)
            bars[k].set_edgecolor("#c0392b")

        ax.axhline(1.0 / (N_LAYERS * N_HEADS), color="k", linestyle=":", linewidth=1,
                   alpha=0.5, label="uniform")
        ax.set_xticks(range(nd))
        ax.set_xticklabels([f"v{k+1}" for k in range(nd)], fontsize=7)
        ax.set_ylabel("Purity of dominant head")
        ax.set_ylim(0, 0.55)
        ax.set_title(f"{PHASE_LABELS[phase]}\n(step {all_results[phase]['step']})", fontsize=9)

        # Legend for head colors
        if phase == phases[0]:
            legend_patches = [
                plt.matplotlib.patches.Patch(color=cmap(encode_head(l, h)),
                                             label=HEAD_LABELS[encode_head(l, h)])
                for l in range(N_LAYERS) for h in range(N_HEADS)
            ]
            ax.legend(handles=legend_patches, fontsize=6, loc="upper right",
                      ncol=2, title="Dominant head", title_fontsize=7)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figD_dominant_head_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figD_dominant_head_{op_name}.png")


def plot_figE_attn_sensitivity(all_results, op_name, op_label):
    """Per-layer attention output sensitivity per direction."""
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    if not phases:
        return

    fig, axes = plt.subplots(1, len(phases), figsize=(4 * len(phases), 4), squeeze=False)
    fig.suptitle(f"Attention Output Sensitivity per Direction — {op_label}\n"
                 "||Δ(attn_output_l)||_F per unit perturbation along v_k",
                 fontsize=10, fontweight="bold")

    for ax, phase in zip(axes[0], phases):
        phase_data = all_results[phase]
        attn_sens  = phase_data["attn_sens"]
        per_dir    = phase_data["per_dir"]

        if not attn_sens:
            ax.text(0.5, 0.5, "no sensitivity data", ha="center", transform=ax.transAxes)
            continue

        ks     = sorted(attn_sens.keys())
        sens_l0 = [attn_sens[k][0] for k in ks]
        sens_l1 = [attn_sens[k][1] for k in ks]

        x = np.arange(len(ks))
        w = 0.35
        ax.bar(x - w/2, sens_l0, w, label="Layer 0", color="#3498db", alpha=0.8)
        ax.bar(x + w/2, sens_l1, w, label="Layer 1", color="#e74c3c", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"v{k+1}" for k in ks], fontsize=8)
        ax.set_ylabel("Sensitivity (||ΔA||_F / ε)")
        ax.set_title(f"{PHASE_LABELS[phase]}\n(step {phase_data['step']})", fontsize=9)

        # Highlight top-3
        for k_i, k in enumerate(ks):
            if k < 3:
                ax.axvspan(k_i - 0.5, k_i + 0.5, color="#f39c12", alpha=0.08)

        if phase == phases[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figE_attn_sensitivity_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figE_attn_sensitivity_{op_name}.png")


def plot_figF_olah_bridge(all_results, op_name, op_label):
    """
    Olah bridge in parameter space:
      raw_importance_k = σ_k / Σσ_j
      head_purity_k    ↔ diagonal dominance
      effective_imp_k  = σ_k × purity_k
      interference_k   = 1 - purity_k

    Shows raw vs corrected importance and tracks interference fraction for top-3 vs bulk.
    """
    phases = [ph for ph in PHASE_ORDER if ph in all_results]
    if not phases:
        return

    fig = plt.figure(figsize=(6 * len(phases), 7))
    fig.suptitle(f"Olah Bridge — Parameter-Space Attribution — {op_label}\n"
                 "head_purity ↔ diagonal_dominance  |  effective_imp = σ_k × purity  |  interference = 1 - purity",
                 fontsize=10, fontweight="bold")

    for ph_i, phase in enumerate(phases):
        phase_data = all_results[phase]
        per_dir    = phase_data["per_dir"]
        S          = np.array(phase_data["S"])
        S_total    = S.sum() + 1e-30

        raw_imp  = S / S_total
        purities = np.array([e["purity"] for e in per_dir])
        eff_imp  = raw_imp * purities
        interf   = 1.0 - purities
        nd       = len(per_dir)

        ax_top = fig.add_subplot(2, len(phases), ph_i + 1)
        ax_bot = fig.add_subplot(2, len(phases), len(phases) + ph_i + 1)

        # Top: raw vs effective importance
        x = np.arange(nd)
        ax_top.bar(x - 0.2, raw_imp, 0.4, label="raw (σ_k/Σσ)", color="#3498db", alpha=0.7)
        ax_top.bar(x + 0.2, eff_imp, 0.4, label="effective (×purity)", color="#2ecc71", alpha=0.8)

        for k in range(min(3, nd)):
            ax_top.axvspan(k - 0.5, k + 0.5, color="#f39c12", alpha=0.08)

        ax_top.set_xticks(x)
        ax_top.set_xticklabels([f"v{k+1}" for k in range(nd)], fontsize=6)
        ax_top.set_ylabel("Importance")
        ax_top.set_title(f"{PHASE_LABELS[phase]}\n(step {phase_data['step']})", fontsize=9)
        if ph_i == 0:
            ax_top.legend(fontsize=7)

        # Bottom: interference fraction
        top3_interf = np.mean(interf[:3]) if nd >= 3 else float("nan")
        bulk_interf = np.mean(interf[3:]) if nd > 3 else float("nan")

        ax_bot.bar(x, interf, color=["#e74c3c" if k < 3 else "#95a5a6" for k in range(nd)],
                   alpha=0.8, edgecolor="k", linewidth=0.4)
        ax_bot.axhline(top3_interf if not np.isnan(top3_interf) else 0,
                       color="#e74c3c", linestyle="--", linewidth=1.5,
                       label=f"top-3 mean = {top3_interf:.2f}")
        ax_bot.axhline(bulk_interf if not np.isnan(bulk_interf) else 0,
                       color="#95a5a6", linestyle="--", linewidth=1.5,
                       label=f"bulk mean = {bulk_interf:.2f}")

        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([f"v{k+1}" for k in range(nd)], fontsize=6)
        ax_bot.set_ylabel("Interference fraction\n(1 − head_purity)")
        ax_bot.set_ylim(0, 1.0)
        if ph_i == 0:
            ax_bot.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"figF_olah_bridge_{op_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figF_olah_bridge_{op_name}.png")


def plot_cross_op_summary(all_op_results):
    """
    Summary figure across all operations: head purity of {v1,v2,v3} vs bulk
    at grokking transition, and QK/VO split for top-3 vs bulk post-grok.
    """
    ops     = [op for op in OPERATIONS if op in all_op_results and all_op_results[op]]
    op_lbls = [OPERATIONS[op]["label"] for op in ops]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cross-Operation Attribution Summary — Top-3 vs Bulk",
                 fontsize=12, fontweight="bold")

    # ── Panel 1: Head purity at grokking transition ──────────────────────────
    ax = axes[0]
    x = np.arange(len(ops))
    top3_purities = []
    bulk_purities = []
    for op in ops:
        res = all_op_results[op]
        if "grokking_transition" not in res:
            top3_purities.append(np.nan); bulk_purities.append(np.nan); continue
        per_dir = res["grokking_transition"]["per_dir"]
        top3 = np.mean([per_dir[k]["purity"] for k in range(min(3, len(per_dir)))])
        bulk = np.mean([per_dir[k]["purity"] for k in range(3, len(per_dir))])
        top3_purities.append(top3)
        bulk_purities.append(bulk)

    ax.bar(x - 0.2, top3_purities, 0.4, label="top-3 mean purity", color="#3498db", alpha=0.8)
    ax.bar(x + 0.2, bulk_purities, 0.4, label="bulk mean purity",  color="#95a5a6", alpha=0.8)
    ax.axhline(1.0 / (N_LAYERS * N_HEADS), color="k", linestyle=":", alpha=0.5, label="uniform")
    ax.set_xticks(x); ax.set_xticklabels(op_lbls, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Head purity"); ax.set_title("Purity at Grok Transition")
    ax.legend(fontsize=8)

    # ── Panel 2: QK vs VO fraction for top-3, post-grok ─────────────────────
    ax = axes[1]
    top3_qk, top3_vo = [], []
    bulk_qk, bulk_vo = [], []
    for op in ops:
        res = all_op_results[op]
        if "stable_postgrok" not in res:
            top3_qk.append(np.nan); top3_vo.append(np.nan)
            bulk_qk.append(np.nan); bulk_vo.append(np.nan); continue
        per_dir = res["stable_postgrok"]["per_dir"]
        t3 = per_dir[:3] if len(per_dir) >= 3 else per_dir
        bl = per_dir[3:] if len(per_dir) > 3 else []
        top3_qk.append(np.mean([e["qk_frac"] for e in t3]) if t3 else np.nan)
        top3_vo.append(np.mean([e["vo_frac"] for e in t3]) if t3 else np.nan)
        bulk_qk.append(np.mean([e["qk_frac"] for e in bl]) if bl else np.nan)
        bulk_vo.append(np.mean([e["vo_frac"] for e in bl]) if bl else np.nan)

    w = 0.2
    ax.bar(x - 1.5*w, top3_qk, w, color="#2980b9", alpha=0.8, label="top-3 QK")
    ax.bar(x - 0.5*w, top3_vo, w, color="#c0392b", alpha=0.8, label="top-3 VO")
    ax.bar(x + 0.5*w, bulk_qk, w, color="#7fb3d3", alpha=0.6, label="bulk QK")
    ax.bar(x + 1.5*w, bulk_vo, w, color="#e08080", alpha=0.6, label="bulk VO")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(op_lbls, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Fraction"); ax.set_title("QK vs VO Split (Post-Grok)")
    ax.legend(fontsize=7, ncol=2)

    # ── Panel 3: Dominant head consistency across phases ─────────────────────
    # For v1: how many phases agree on the same dominant head?
    ax = axes[2]
    consistency = []
    for op in ops:
        res = all_op_results[op]
        dom_heads = []
        for ph in PHASE_ORDER:
            if ph not in res:
                continue
            per_dir = res[ph]["per_dir"]
            if per_dir:
                dom_heads.append(per_dir[0]["dominant"])   # v1 dominant head
        if len(dom_heads) > 1:
            from collections import Counter
            most_common_count = Counter(dom_heads).most_common(1)[0][1]
            consistency.append(most_common_count / len(dom_heads))
        else:
            consistency.append(np.nan)

    ax.bar(x, consistency, color="#9b59b6", alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.4, label="perfect")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x); ax.set_xticklabels(op_lbls, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of phases agreeing")
    ax.set_title("v1 Dominant-Head Consistency\nacross training phases")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "figG_cross_op_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figG_cross_op_summary.png")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    cfg    = Config()
    device = get_device()
    print(f"Device: {device}")

    # Quick smoke-test of head index arrays
    head_indices = build_head_index_arrays()
    test_vk = np.random.randn(TOTAL_DIM)
    hm, lhm, tm, lm = compute_head_attribution(test_vk, head_indices)
    total_check = sum(hm.values())
    assert abs(total_check - 1.0) < 1e-5, f"Head mass doesn't sum to 1: {total_check}"
    print(f"Head mask sanity check passed (sum={total_check:.6f})")

    all_op_results = {}

    for op_name, op_cfg in OPERATIONS.items():
        results = run_attribution(op_name, op_cfg, cfg, device)
        if results is None:
            continue
        all_op_results[op_name] = results

        op_label = op_cfg["label"]
        plot_figA_head_mass_heatmap(results, op_name, op_label)
        plot_figB_qkvo(results, op_name, op_label)
        plot_figC_head_purity(results, op_name, op_label)
        plot_figD_dominant_head(results, op_name, op_label)
        plot_figE_attn_sensitivity(results, op_name, op_label)
        plot_figF_olah_bridge(results, op_name, op_label)

        # Print per-op summary
        print(f"\n  ── {op_name} attribution summary ──")
        for ph in PHASE_ORDER:
            if ph not in results:
                continue
            per_dir = results[ph]["per_dir"]
            step    = results[ph]["step"]
            print(f"  [{PHASE_LABELS[ph]:14s}] step={step}")
            for k in range(min(3, len(per_dir))):
                e   = per_dir[k]
                dom = e["dominant"]
                print(f"    v{k+1}: purity={e['purity']:.3f}  dom=L{dom[0]}H{dom[1]}"
                      f"  QK={e['qk_frac']:.2f}  VO={e['vo_frac']:.2f}")

    if len(all_op_results) > 1:
        plot_cross_op_summary(all_op_results)

    # Save results
    save_path = PLOT_DIR / "attribution_results.pt"
    torch.save(all_op_results, save_path)
    print(f"\nResults saved to {save_path}")
    print(f"Plots saved to   {PLOT_DIR}/")

    # Print the Olah bridge interpretation
    print("\n" + "="*60)
    print("OLAH BRIDGE INTERPRETATION")
    print("="*60)
    for op_name, results in all_op_results.items():
        for ph in ["grokking_transition", "stable_postgrok"]:
            if ph not in results:
                continue
            per_dir = results[ph]["per_dir"]
            S       = np.array(results[ph]["S"])
            S_total = S.sum() + 1e-30
            print(f"\n  {op_name} [{PHASE_LABELS[ph]}]:")
            print(f"  {'k':>4}  {'σ_k/Σσ':>8}  {'purity':>8}  {'eff_imp':>8}  "
                  f"{'interf':>8}  {'dom':>6}  QK/VO")
            for k in range(min(6, len(per_dir))):
                e      = per_dir[k]
                raw    = float(S[k]) / S_total
                eff    = raw * e["purity"]
                interf = 1.0 - e["purity"]
                dom    = e["dominant"]
                print(f"  v{k+1:>3}  {raw:>8.4f}  {e['purity']:>8.3f}  {eff:>8.4f}  "
                      f"{interf:>8.3f}  L{dom[0]}H{dom[1]:>1}  "
                      f"{e['qk_frac']:.2f}/{e['vo_frac']:.2f}")


if __name__ == "__main__":
    main()
