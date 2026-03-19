"""
PCA diagnostic: test whether snapshot count (307 vs 30) explains the
PC1% drop from ~84-94% to ~65-82%.

For each weight key in the last layer (layer 2):
  1. Stack all 307 snapshots into a (307, 16384) matrix, run PCA, report PC1%.
  2. Subsample to 30 evenly-spaced snapshots, run PCA, report PC1%.

If subsampling drops PC1% substantially, snapshot count is the main driver.
If PC1% stays high, the drop is due to hyperparameters (wd / architecture).
"""

import torch
import numpy as np

# -- Load data ---------------------------------------------------------------
data = torch.load("/Users/tara-mini/bubble/grok_runs.pt", map_location="cpu")
logs = data["attn_logs"]          # list of 307 dicts
cfg  = data["cfg"]

print(f"Config: lr={cfg['LR']}, wd={cfg['WEIGHT_DECAY']}, "
      f"n_layers={cfg['N_LAYERS']}, steps={cfg['STEPS']}")
print(f"Total snapshots: {len(logs)}")
print(f"Step range: {logs[0]['step']} -> {logs[-1]['step']}")
print()

# -- Settings -----------------------------------------------------------------
LAST_LAYER = cfg["N_LAYERS"] - 1        # layer 2
WEIGHT_KEYS = ["WQ", "WK", "WV", "WO"]
N_SUB = 30                              # match today's sweep count

# -- Helper: compute PC1% from a list of weight tensors ----------------------
def pc1_pct(tensors):
    """Stack flattened weight snapshots, mean-center, SVD -> PC1 variance %."""
    mat = torch.stack([t.flatten() for t in tensors]).numpy()  # (T, D)
    mat = mat - mat.mean(axis=0, keepdims=True)                # center
    # economy SVD (T << D, so this is fast)
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    var = s ** 2
    return 100.0 * var[0] / var.sum()

# -- Extract weight trajectories for the last layer --------------------------
trajectories = {k: [] for k in WEIGHT_KEYS}
for snap in logs:
    layer_dict = snap["layers"][LAST_LAYER]
    for k in WEIGHT_KEYS:
        trajectories[k].append(layer_dict[k])

# -- Subsample indices (evenly spaced) ----------------------------------------
sub_idx = np.round(np.linspace(0, len(logs) - 1, N_SUB)).astype(int)
print(f"Subsample: {N_SUB} snapshots from indices "
      f"[{sub_idx[0]}, {sub_idx[1]}, ..., {sub_idx[-1]}]")
print(f"  -> steps: {logs[sub_idx[0]]['step']}, "
      f"{logs[sub_idx[1]]['step']}, ..., {logs[sub_idx[-1]]['step']}")
print()

# -- Compute & report ---------------------------------------------------------
print(f"{'Key':>4s}  {'Full 307':>10s}  {'Sub 30':>10s}  {'Delta':>8s}")
print("-" * 40)

for k in WEIGHT_KEYS:
    full_list = trajectories[k]
    sub_list  = [full_list[i] for i in sub_idx]

    pct_full = pc1_pct(full_list)
    pct_sub  = pc1_pct(sub_list)
    delta    = pct_sub - pct_full

    print(f"{k:>4s}  {pct_full:9.2f}%  {pct_sub:9.2f}%  {delta:+7.2f}%")

print()
print("Interpretation:")
print("  If Sub-30 ~ Full-307 (Delta small), snapshot count is NOT the cause.")
print("  If Sub-30 << Full-307 (Delta large negative), snapshot count matters.")
