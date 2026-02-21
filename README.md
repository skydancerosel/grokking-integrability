# Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking

We identify an emergent low-dimensional **invariant submanifold**---the *execution manifold*---in the weight space of transformers trained on modular arithmetic. Loss-landscape curvature is confined to the **normal bundle** of this submanifold, curvature growth in the normal bundle consistently **precedes generalization**, and **causal interventions** confirm orthogonal gradient flow is necessary for grokking.

**Paper**: [Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking](https://arxiv.org/abs/2602.16746)

## Key Findings

1. **Rank-1 execution manifold.** PCA on attention weight trajectories during grokking reveals that 70--94% of variance is captured by a single principal component. Weight evolution during grokking is essentially one-dimensional.

2. **Invariant submanifold.** Commutator defect vectors (measuring loss-landscape curvature) are predominantly orthogonal to the execution manifold: residual/full = 1.000 within numerical precision across 36 conditions (6 operations x 2 weight-decay settings x 3 seeds). Curvature is confined to the normal bundle---it does not deflect the trajectory out of its learned subspace. A random-subspace baseline confirms the small parallel component is geometrically structured (exec/random ~ 1.8--2.9x), ruling out dimensionality artifacts.

3. **Curvature explodes orthogonally during grokking.** Operations that grok show 10--1000x higher commutator defect than non-grokking controls, concentrated outside the execution manifold.

4. **Curvature growth precedes generalization.** The onset of commutator defect growth precedes the generalization transition by 600--1600 training steps across all 12 grokking runs (4 operations x 3 seeds), with 100% consistency (sign test p = 2^{-12} < 0.001). Non-grokking operations also show moderate defect growth (30--50x baseline) without generalizing, so onset is a necessary precondition rather than a sufficient predictor. Causal interventions confirm the mechanistic link.

5. **Regime-invariant.** All results replicate across a 100x learning rate sweep ({1e-4, 1e-3, 1e-2}), a qualitatively different slow regime (lr=5e-5, wd=0.1, 3 layers, ~200x timescale difference), and three random seeds.

6. **Causal interventions.** Suppressing orthogonal gradient flow prevents grokking with a monotonic dose-response across four operations (necessary), while artificially boosting curvature defects has no effect (not sufficient). This establishes a directional causal relationship between execution-manifold geometry and generalization.

## Experimental Setup

All experiments use the canonical grokking setup from [Power et al. (2022)](https://arxiv.org/abs/2201.02177):
- **Model**: 2-layer Transformer, d_model=128, 4 heads, d_ff=256, pre-norm, GELU, ~290k params
- **Task**: Binary operations mod 97 (6 operations)
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

### Hyperparameter Regimes

| Parameter | Fast Regime | Slow Regime |
|-----------|-------------|-------------|
| Learning rate | 1e-3 | 5e-5 |
| Weight decay | 1.0 | 0.1 |
| Layers | 2 | 3 |
| Adam beta2 | 0.98 | 0.999 |
| Grok step (add, mean) | ~2,900 | ~570,000 |
| Training budget | 7,500 steps | 650,000 steps |

### Learning Rate Sweep

| LR | Dynamical Regime | Grok Time (add) | Defect Precedes? |
|----|-----------------|-----------------|-----------------|
| 1e-4 | Overdamped | ~30k steps | Yes |
| 1e-3 | Critically damped | ~3k steps | Yes |
| 1e-2 | Underdamped | ~1k steps | Yes |

## Repository Structure

### Core Scripts (run in order)

| # | Script | What it does | Figures |
|---|--------|-------------|---------|
| 1 | `grok_sweep.py` | Train models across 6 ops x 2 wd x 3 seeds, logging attention weights | -- |
| 2 | `pca_sweep_analysis.py` | PCA eigenanalysis on saved attention weight trajectories | figA--figG |
| 3 | `pca_controls.py` | No-wd baseline, Fourier alignment, random-walk null model | (part of figA, figE) |
| 4 | `pca_compare_regimes.py` | Compare slow vs fast hyperparameter regimes | figH, figI |
| 5 | `grok_commutator_analysis.py` | Forward commutator: project defect onto PCA manifold | figJ--figN |
| 6 | `grok_converse_commutator.py` | Converse: project weight trajectory onto commutator subspace | figO--figR |
| 7 | `grok_multiseed_commutator.py` | Multi-seed replication (3 seeds x 6 ops x 2 wd = 36 runs) | figS--figV |
| 8 | `grok_generalization_dynamics.py` | Temporal: defect onset vs generalization transition timing | figW, figW2, figX |
| 9 | `grok_slow_regime_commutator.py` | Slow regime (lr=5e-5, wd=0.1, 3L): invariance + defect timing | figY, figZ |
| 10 | `grok_lr_sweep.py` | LR sweep phase diagram across {1e-4, 1e-3, 1e-2} | figPD, figPD2 |
| 11 | `grok_lr_alignment.py` | Trajectory-curvature alignment across LRs + phase portrait | figPD3, figPD4 |
| 12 | `grok_intervention.py` | Causal interventions: gradient suppression (5 conditions) | figI1--figI5 |
| 13 | `grok_intervention_ablation.py` | PCA vs random projection ablation control | figI6, figI7 |
| 14 | `grok_intervention_sustained_kick.py` | Sustained directional kicks (boosting curvature defect) | figI8, figI9 |
| 15 | `grok_intervention_multiop.py` | Multi-operation dose-response replication | figI10, figI11 |

### Supporting Scripts

| Script | Purpose |
|--------|---------|
| `grok_sweep_slow.py` | Slow-regime training (lr=5e-5, wd=0.1, 3 layers) |
| `grok_integrability_controls.py` | Random subspace baseline control: exec vs random projection |
| `pca_diagnostic.py` | Tests whether snapshot count explains PC1% variation |

### Output

- `pca_sweep_plots/` -- All publication figures (figA through figZ, figPD, figI) and saved result tensors
- `grok_sweep_results/` -- Raw sweep outputs (model checkpoints + attention logs per run)
- `grok_sweep_results_slow/` -- Slow-regime sweep outputs
- `paper/` -- LaTeX source, compiled PDF, bibliography

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

# Step 9: Slow regime verification (~6 hours)
python grok_slow_regime_commutator.py

# Step 10: LR sweep phase diagram (~1-1.5 hours on MPS)
python grok_lr_sweep.py

# Step 11: LR-curvature alignment analysis (~10 min)
python grok_lr_alignment.py

# Step 12: Causal interventions (~2 hours total)
python grok_intervention.py
python grok_intervention_ablation.py
python grok_intervention_sustained_kick.py
python grok_intervention_multiop.py
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

### Commutator / Invariance (single seed)
- **figJ** `figJ_commutator_defect.png` -- Commutator defect over training
- **figK** `figK_integrability.png` -- Invariance: commutators orthogonal to execution manifold
- **figL** `figL_grok_vs_nowd_commutator.png` -- Grok vs no-wd defect comparison
- **figM** `figM_defect_integrability.png` -- Defect explosion + invariance measure combined
- **figN** `figN_attn_weight_fraction.png` -- Attention weight fraction of commutator

### Converse Analysis (single seed)
- **figO** `figO_trajectory_alignment.png` -- Trajectory-curvature alignment
- **figP** `figP_trajectory_in_comm_subspace.png` -- Trajectory projection into commutator subspace
- **figQ** `figQ_alignment_ratio.png` -- Alignment ratio vs random baseline
- **figR** `figR_defect_vs_alignment.png` -- Defect vs alignment scatter

### Multi-Seed Replication
- **figS** `figS_multiseed_integrability.png` -- Invariance bars (mean +/- std, 3 seeds)
- **figT** `figT_multiseed_alignment.png` -- Alignment bars (3 seeds)
- **figU** `figU_multiseed_defect.png` -- Defect bars (3 seeds)
- **figV** `figV_temporal_add_seeds.png` -- Temporal traces with seed overlay

### Generalization Dynamics
- **figW** `figW_defect_predicts_grokking.png` -- Defect vs test accuracy: 4 grokking + 2 non-grokking controls
- **figW2** `figW2_hero_defect_predicts_grok.png` -- Hero figure: single best example
- **figX** `figX_defect_lead_time.png` -- Lead-time quantification (sign test p = 2^{-12})

### Random Subspace Control
- **figC1** `figC1_exec_vs_random.png` -- Exec vs random projection fraction over training
- **figC2** `figC2_exec_over_random_ratio.png` -- Exec/random ratio with defect overlay
- **figC3** `figC3_dimension_sweep.png` -- Projection fraction vs basis dimension K
- **figC4** `figC4_phase_comparison.png` -- Exec/random ratio by training phase
- **figC5** `figC5_hero.png` -- Combined: defect x exec/random ratio x test accuracy

### Slow Regime Verification
- **figY** `figY_regime_comparison_commutator.png` -- Invariance, defect, lead time: slow vs fast
- **figZ** `figZ_slow_regime_hero.png` -- Defect predicts grokking in slow regime

### Learning Rate Sweep
- **figPD** `figPD_lr_phase_diagram.png` -- Phase diagram: grok fraction, grok step, max defect, lead time across 3 LRs x 6 ops
- **figPD2** `figPD2_lr_sweep_hero.png` -- Hero: defect + test accuracy for 3 LRs on addition
- **figPD3** `figPD3_lr_alignment.png` -- Trajectory-curvature alignment across LRs
- **figPD4** `figPD4_alignment_vs_defect.png` -- Phase portrait: alignment ratio vs defect magnitude with dynamical regime labels

### Causal Interventions
- **figI1** `figI1_intervention_defect_trajectories.png` -- Defect trajectories under 5 intervention conditions
- **figI2** `figI2_intervention_accuracy_overlay.png` -- Test accuracy overlay across conditions
- **figI3** `figI3_intervention_grok_timing.png` -- Grok timing comparison
- **figI4** `figI4_intervention_summary_table.png` -- Summary table of all conditions
- **figI5** `figI5_intervention_hparam_sensitivity.png` -- Hyperparameter sensitivity
- **figI6** `figI6_ablation_random_vs_pca.png` -- PCA vs random projection ablation: defect + accuracy
- **figI7** `figI7_ablation_accuracy_overlay.png` -- Ablation accuracy overlay
- **figI8** `figI8_sustained_kick_dose_response.png` -- Sustained directional kick dose-response
- **figI9** `figI9_sustained_kick_overlay.png` -- Kick overlay: accuracy + defect
- **figI10** `figI10_multiop_dose_response.png` -- Multi-operation dose-response (4 ops x 5 strengths)
- **figI11** `figI11_multiop_combined.png` -- Combined multi-operation results

## Hardware

Experiments were run on Apple M-series (MPS backend). GPU (CUDA) and CPU are also supported. Total compute for full reproduction: ~12 hours on a single machine (6 hours slow regime, ~1.5 hours LR sweep, ~2 hours interventions, ~2.5 hours other).

## Citation

```bibtex
@article{xu2026lowdim,
  title={Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking},
  author={Xu, Yongzhong},
  year={2026},
  eprint={2602.16746},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2602.16746}
}
```
