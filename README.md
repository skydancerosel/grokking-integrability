# Integrability of the Grokking Manifold

Empirical evidence that the weight-space trajectory during grokking in modular arithmetic lies on an **integrable (flat) submanifold** of parameter space, and that curvature explosion orthogonal to this manifold serves as a **leading indicator** of the generalization transition.

## Key Findings

1. **Rank-1 execution manifold.** PCA on attention weight trajectories during grokking reveals that 70--94% of variance is captured by a single principal component. Weight evolution during grokking is essentially one-dimensional.

2. **The execution manifold is integrable.** Commutator defect vectors (measuring loss-landscape curvature) are perfectly orthogonal to the PCA submanifold: residual/full = 1.000 +/- 0.000 across 36 conditions (6 operations x 2 weight-decay settings x 3 seeds). The learned subspace is flat.

3. **Curvature explodes orthogonally during grokking.** Operations that grok show 10--1000x higher commutator defect than non-grokking controls, concentrated entirely outside the execution manifold.

4. **Defect spike predicts grokking.** The commutator defect spike precedes the generalization transition by 600--1600 training steps (mean 1117) across all 12 grokking runs (4 operations x 3 seeds), with 100% consistency.

## Experimental Setup

All experiments use the canonical grokking setup from [Power et al. (2022)](https://arxiv.org/abs/2201.02177):
- **Model**: 2-layer Transformer, d_model=128, 4 heads, pre-norm
- **Task**: Binary operations mod 97 (6 operations from Power et al.)
- **Training**: AdamW, lr=1e-3, weight_decay=1.0, beta2=0.98
- **Data split**: 30% train / 70% test

### Operations Tested

| Operation | Groks? | Grok Step (mean) |
|-----------|--------|-----------------|
| (a+b) mod 97 | Yes | ~3000 |
| (a-b) mod 97 | Yes | ~3600 |
| (a*b) mod 97 | Yes | ~2900 |
| (a^2+b^2) mod 97 | Yes | ~2600 |
| (a^2+ab+b^2) mod 97 | No | -- |
| (a^3+ab) mod 97 | No | -- |

## Repository Structure

### Core Scripts (run in order)

| # | Script | What it does | Figures |
|---|--------|-------------|---------|
| 1 | `grok_sweep.py` | Train models across 6 operations x 2 weight-decay x 3 seeds, logging attention weights | -- |
| 2 | `pca_sweep_analysis.py` | PCA eigenanalysis on saved attention weight trajectories | figA--figG |
| 3 | `pca_controls.py` | No-wd baseline, Fourier alignment, random-walk null model | (part of figA, figE) |
| 4 | `pca_compare_regimes.py` | Compare slow (lr=5e-5) vs fast (lr=1e-3) hyperparameter regimes | figH, figI |
| 5 | `grok_commutator_analysis.py` | Forward commutator analysis: project defect onto PCA manifold | figJ--figN |
| 6 | `grok_converse_commutator.py` | Converse: project weight trajectory onto commutator subspace | figO--figR |
| 7 | `grok_multiseed_commutator.py` | Multi-seed replication (3 seeds x 6 ops x 2 wd = 36 runs) | figS--figV |
| 8 | `grok_generalization_dynamics.py` | Temporal: defect spike vs generalization transition timing | figW, figW2, figX |

### Supporting Scripts

| Script | Purpose |
|--------|---------|
| `grok_sweep_slow.py` | Slow-regime training (lr=5e-5, wd=0.1, 3 layers) for regime comparison |
| `pca_diagnostic.py` | Tests whether snapshot count explains PC1% variation |

### Output

- `pca_sweep_plots/` -- All publication figures (figA through figX) and saved result tensors
- `grok_sweep_results/` -- Raw sweep outputs (model checkpoints + attention logs per run)

## Reproducing Results

```bash
pip install -r requirements.txt

# Step 1: Train models and log attention weights (~30 min on MPS/GPU)
python grok_sweep.py

# Step 2: PCA eigenanalysis (~2 min)
python pca_sweep_analysis.py

# Step 3: Control experiments (~5 min)
python pca_controls.py

# Step 4 (optional): Regime comparison (~10 min)
python grok_sweep_slow.py
python pca_compare_regimes.py

# Step 5: Commutator analysis -- single seed (~20 min)
python grok_commutator_analysis.py

# Step 6: Converse commutator analysis (~15 min)
python grok_converse_commutator.py

# Step 7: Multi-seed replication (~90 min)
python grok_multiseed_commutator.py

# Step 8: Generalization dynamics (~15 min)
python grok_generalization_dynamics.py
```

All figures are saved to `pca_sweep_plots/`.

## Figure Index

### PCA Eigenanalysis
- **figA** `figA_grok_vs_nowd_crossop.png` -- PC1% across operations: grok vs no-wd
- **figB** `figB_pc1_heatmap.png` -- PC1% heatmap by operation and weight matrix
- **figC** `figC_eigenspectrum_crossop.png` -- Top-5 eigenspectrum per operation
- **figD** `figD_grok_step_vs_pc1.png` -- Grokking speed vs PC1% concentration
- **figE** `figE_null_zscores_crossop.png` -- Z-scores vs random-walk null model
- **figF** `figF_temporal_crossop.png` -- Temporal PC1% evolution during training
- **figG** `figG_per_weight_crossop.png` -- Per-weight-matrix breakdown

### Regime Comparison
- **figH** `figH_regime_comparison.png` -- Slow vs fast regime PC1%
- **figI** `figI_pc1_drop_decomposition.png` -- Which hyperparameter drives PC1% drop

### Commutator / Integrability (single seed)
- **figJ** `figJ_commutator_defect.png` -- Commutator defect over training
- **figK** `figK_integrability.png` -- Integrability: commutators orthogonal to PCA manifold
- **figL** `figL_grok_vs_nowd_commutator.png` -- Grok vs no-wd defect comparison
- **figM** `figM_defect_integrability.png` -- Defect explosion + integrability combined
- **figN** `figN_attn_weight_fraction.png` -- Attention weight fraction of commutator

### Converse Analysis (single seed)
- **figO** `figO_trajectory_alignment.png` -- Trajectory-curvature alignment
- **figP** `figP_trajectory_in_comm_subspace.png` -- Trajectory projection into commutator subspace
- **figQ** `figQ_alignment_ratio.png` -- Alignment ratio vs random baseline
- **figR** `figR_defect_vs_alignment.png` -- Defect vs alignment scatter

### Multi-Seed Replication
- **figS** `figS_multiseed_integrability.png` -- Integrability bars (mean +/- std, 3 seeds)
- **figT** `figT_multiseed_alignment.png` -- Alignment bars (3 seeds)
- **figU** `figU_multiseed_defect.png` -- Defect bars (3 seeds)
- **figV** `figV_temporal_add_seeds.png` -- Temporal traces with seed overlay

### Generalization Dynamics
- **figW** `figW_defect_predicts_grokking.png` -- Defect vs test accuracy, 4 operations
- **figW2** `figW2_hero_defect_predicts_grok.png` -- Hero figure: single best example
- **figX** `figX_defect_lead_time.png` -- Lead-time quantification

## Hardware

Experiments were run on Apple M-series (MPS backend). GPU (CUDA) and CPU are also supported. Total compute for full reproduction: ~3 hours on a single machine.
