# Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking

We identify an emergent low-dimensional **invariant submanifold**---the *execution manifold*---in the weight space of transformers trained on modular arithmetic. Loss-landscape curvature is confined to the **normal bundle** of this submanifold, curvature growth in the normal bundle consistently **precedes generalization**, and **causal interventions** confirm orthogonal gradient flow is necessary for grokking. **Spectral edge analysis** reveals a characteristic mode-competition-and-collapse cycle in the weight SVD that correlates with grokking across all conditions.

**Papers**:
- [Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking](https://arxiv.org/abs/2602.16746)
- [Spectral Edge Dynamics Reveal Functional Modes of Learning](https://arxiv.org/abs/2604.06256) — see [`spectral/functional_modes/`](spectral/functional_modes/)

## Key Findings

1. **Rank-1 execution manifold.** PCA on attention weight trajectories during grokking reveals that 70--94% of variance is captured by a single principal component. Weight evolution during grokking is essentially one-dimensional.

2. **Invariant submanifold.** Commutator defect vectors (measuring loss-landscape curvature) are predominantly orthogonal to the execution manifold: residual/full = 1.000 within numerical precision across 36 conditions (6 operations x 2 weight-decay settings x 3 seeds). Curvature is confined to the normal bundle---it does not deflect the trajectory out of its learned subspace. A random-subspace baseline confirms the small parallel component is geometrically structured (exec/random ~ 1.8--2.9x), ruling out dimensionality artifacts.

3. **Curvature explodes orthogonally during grokking.** Operations that grok show 10--1000x higher commutator defect than non-grokking controls, concentrated outside the execution manifold.

4. **Curvature growth precedes generalization.** The onset of commutator defect growth precedes the generalization transition by 600--1600 training steps across all 12 grokking runs (4 operations x 3 seeds), with 100% consistency (sign test p = 2^{-12} < 0.001). Non-grokking operations also show moderate defect growth (30--50x baseline) without generalizing, so onset is a necessary precondition rather than a sufficient predictor. Causal interventions confirm the mechanistic link.

5. **Regime-invariant.** All results replicate across a 100x learning rate sweep ({1e-4, 1e-3, 1e-2}), a qualitatively different slow regime (lr=5e-5, wd=0.1, 3 layers, ~200x timescale difference), and three random seeds.

6. **Causal interventions.** Suppressing orthogonal gradient flow prevents grokking with a monotonic dose-response across four operations (necessary), while artificially boosting curvature defects has no effect (not sufficient). This establishes a directional causal relationship between execution-manifold geometry and generalization.

7. **Spectral edge: mode competition and collapse.** Weight matrix SVD reveals a characteristic spectral cycle during grokking: the dominant singular value gap (sigma_1 - sigma_2) narrows as modes compete, reaches a minimum concurrent with the matrix commutator ||[W_Q, W_K]||_F peak, then widens as one mode dominates and the commutator collapses. Non-grokking controls lack this cycle entirely.

8. **Phase portrait geometry.** In (spectral gap, commutator) phase space, grokking traces a characteristic loop through three phases: competition (near-degenerate modes, low non-commutativity), instability (mode separation, peak non-commutativity), and alignment (post-collapse, generalization). Memorizing models remain trapped in the competition region.

9. **Basis-independent integrability.** The commutator's alignment with weight structure is structural, not algorithmic: three independent bases (weight SVD, displacement SVD, gradient SVD) all show a sign flip from exec/random > 1 during memorization to < 1 post-grokking. This holds per-block across all attention weight matrices (W_Q, W_K, W_V, W_O).

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

```
grokking-integrability/
├── training/                  # Model training
│   ├── grok_sweep.py                 # Train 6 ops x 2 wd x 3 seeds
│   └── grok_sweep_slow.py            # Slow regime (lr=5e-5, wd=0.1, 3L)
│
├── pca/                       # PCA eigenanalysis
│   ├── pca_sweep_analysis.py         # Main PCA analysis (figA--figG)
│   ├── pca_controls.py               # Null model baselines (figA, figE)
│   ├── pca_compare_regimes.py        # Slow vs fast comparison (figH, figI)
│   ├── pca_diagnostic.py             # Snapshot count diagnostics
│   └── grok_integrability_controls.py # Random subspace control (figC1--figC5)
│
├── commutator/                # Commutator defect analysis
│   ├── grok_commutator_analysis.py   # Forward commutator (figJ--figN)
│   ├── grok_converse_commutator.py   # Converse projection (figO--figR)
│   ├── grok_multiseed_commutator.py  # Multi-seed replication (figS--figV)
│   ├── grok_generalization_dynamics.py # Defect-grokking timing (figW, figW2, figX)
│   └── grok_slow_regime_commutator.py # Slow regime verification (figY, figZ)
│
├── spectral/                  # Spectral edge & functional mode analysis
│   │
│   │  # ── Spectral edge verification (arXiv:2602.16746, 2603.28964) ──
│   ├── thesis_table7_replication.py  # Gram matrix spectral analysis (g₂₃, R, k*)
│   ├── grok_weight_svd_gaps.py       # Weight SVD gaps (figSVD1--figSVD6)
│   ├── grok_eigenvalue_gaps.py       # Eigenvalue gaps (figEG1--figEG5)
│   ├── grok_phase_portrait.py        # Phase portraits (figPP1--figPP4)
│   ├── grok_geometry_conjecture_test.py # Conjecture tests
│   ├── layerwise_phase_portrait.py   # Per-layer cascades
│   ├── grok_local_integrability.py   # Local integrability (figL1--figL5)
│   ├── grok_multibasis_controls.py   # Multi-basis controls (figM1--figM5)
│   ├── commutator_heatmap.py         # Per-head heatmaps
│   │
│   └── functional_modes/             # arXiv:2604.06256 — Functional Modes of Learning
│       ├── README.md                     # Paper-specific documentation
│       ├── paper_figure_basis.py         # Generates Figure 1 (basis dependence)
│       ├── fig1_basis_dependence.png     # The paper's main figure
│       ├── fourier_functional_view.py    # Fourier profiles, 4 ops, basis test
│       ├── fourier_dlog_mul.py           # Discrete-log basis for mul (5.9× improvement)
│       ├── v123_feature_attribution.py   # Head purity null (purity ≈ 1/8)
│       ├── residual_stream_alignment.py  # Activation rank (≈40) & Fourier peakedness
│       ├── sae_fourier_features.py       # SAE Jaccard null result (p ≥ 0.97)
│       ├── x2y2_composition_test.py      # Composition cross-terms (4× R² boost)
│       └── x2y2_multitask_composition.py # Single-task vs tritask (2.3× reuse)
│
├── intervention/              # Causal interventions
│   ├── grok_intervention.py          # Gradient suppression (figI1--figI5)
│   ├── grok_intervention_ablation.py # PCA vs random ablation (figI6, figI7)
│   ├── grok_intervention_sustained_kick.py # Directional kicks (figI8, figI9)
│   └── grok_intervention_multiop.py  # Multi-op dose-response (figI10, figI11)
│
├── lr_sweep/                  # Learning rate sweep
│   ├── grok_lr_sweep.py              # LR phase diagram (figPD, figPD2)
│   ├── grok_lr_alignment.py          # Trajectory-curvature alignment (figPD3, figPD4)
│   └── grok_pc1_lr_experiment.py     # PC1 vs LR (figPC1)
│
├── plots/                     # All output figures and result tensors
├── commutator_heatmaps/       # Per-head/layer commutator heatmaps
├── layerwise_phase_portraits/ # Per-layer spectral cascade plots
├── requirements.txt
└── README.md
```

## Reproducing Results

```bash
pip install -r requirements.txt

# Step 1: Train models (~30 min on MPS/GPU)
python training/grok_sweep.py

# Step 2: PCA eigenanalysis (~2 min)
python pca/pca_sweep_analysis.py

# Step 3: Control experiments (~5 min)
python pca/pca_controls.py

# Step 4 (optional): Regime comparison (~10 min)
python training/grok_sweep_slow.py
python pca/pca_compare_regimes.py

# Step 5: Commutator analysis -- single seed (~20 min)
python commutator/grok_commutator_analysis.py

# Step 6: Converse commutator analysis (~15 min)
python commutator/grok_converse_commutator.py

# Step 7: Multi-seed replication (~90 min)
python commutator/grok_multiseed_commutator.py

# Step 8: Generalization dynamics (~15 min)
python commutator/grok_generalization_dynamics.py

# Step 9: Slow regime verification (~6 hours)
python commutator/grok_slow_regime_commutator.py

# Step 10: LR sweep phase diagram (~1-1.5 hours on MPS)
python lr_sweep/grok_lr_sweep.py

# Step 11: LR-curvature alignment analysis (~10 min)
python lr_sweep/grok_lr_alignment.py

# Step 12: Causal interventions (~2 hours total)
python intervention/grok_intervention.py
python intervention/grok_intervention_ablation.py
python intervention/grok_intervention_sustained_kick.py
python intervention/grok_intervention_multiop.py

# Step 13: Gram matrix spectral analysis (~5 min for default, ~30 min with large files)
python spectral/thesis_table7_replication.py
# To include 1GB files for x2_xy_y2 and x3_xy:
# MAX_FILE_MB=1200 python spectral/thesis_table7_replication.py

# Step 14: Spectral edge verification (~1 hour)
python spectral/grok_weight_svd_gaps.py
python spectral/grok_eigenvalue_gaps.py
python spectral/grok_phase_portrait.py
python spectral/grok_geometry_conjecture_test.py
python spectral/layerwise_phase_portrait.py
python spectral/grok_local_integrability.py
python spectral/grok_multibasis_controls.py
python spectral/commutator_heatmap.py
```

All figures are saved to `plots/`.

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
- **figPD** `figPD_lr_phase_diagram.png` -- Phase diagram across 3 LRs x 6 ops
- **figPD2** `figPD2_lr_sweep_hero.png` -- Hero: defect + test accuracy for 3 LRs
- **figPD3** `figPD3_lr_alignment.png` -- Trajectory-curvature alignment across LRs
- **figPD4** `figPD4_alignment_vs_defect.png` -- Phase portrait with dynamical regime labels

### Causal Interventions
- **figI1--I5** Gradient suppression: defect trajectories, accuracy, timing, summary, sensitivity
- **figI6--I7** PCA vs random projection ablation
- **figI8--I9** Sustained directional kick dose-response
- **figI10--I11** Multi-operation dose-response (4 ops x 5 strengths)

## Functional Mode Analysis ([arXiv:2604.06256](https://arxiv.org/abs/2604.06256))

The spectral edge directions {v₁, v₂, v₃} above the bulk spectrum are not localized in parameter space (head purity ≈ 1/8) or activation space (effective rank ≈ 40), and SAE feature overlap is not significant against proper null models (p ≥ 0.97). However, when reinterpreted as functions over the input domain via the perturbation response f_k(a,b) = ||Δh_k(a,b)||², they reveal structured functional modes:

| Task | Correct Basis | Structure | Peak F |
|------|--------------|-----------|--------|
| (a+b) mod p | Additive characters | Single mode ω ≈ 25–26 | 0.40 |
| (a·b) mod p | Discrete-log characters | Single mode ω = 29 (5.9× over additive) | 0.32 |
| (a-b) mod p | Additive characters | Multi-mode {6, 16, 32} | 0.19 |
| (a²+b²) mod p | Cross (add × mul) | Compositional (4× R² from cross-terms) | 0.16† |

†Multivariate probe R²; no single-basis F.

Under multitask training (shared trunk for add + mul + x²+y²), the x²+y² spectral edge inherits the addition head's characteristic frequency ω = 26 (2.3× higher concentration than single-task), providing evidence of functional mode reuse across tasks.

All scripts, the paper figure, and paper-specific documentation are in [`spectral/functional_modes/`](spectral/functional_modes/).

### Gram Matrix Spectral Analysis
- Replication of the intra-signal gap framework ([Xu 2026, arXiv:2603.28964](https://arxiv.org/abs/2603.28964))
- Computes three quantities from the rolling-window Gram matrix (W=10) of flattened attention-weight updates:
  - **g₂₃ = σ₂² − σ₃²**: sub-leading eigenvalue gap — declines 15--111× before grokking in 12/12 runs, 1/12 controls
  - **R = σ_{k\*}/σ_{k\*+1}**: gap ratio — separates WD=1.0 (1.40±0.07) from WD=0.0 (2.83±0.35)
  - **k\* (weighted)**: signal rank — stabilizes at k\*=1 in 9/12 grokking runs (75%), matching thesis 10/12 (83%)
- Script: `spectral/thesis_table7_replication.py`
- Output: `spectral/coherence_edge_results/thesis_table7_results.pt`, `spectral/coherence_edge_plots/thesis_table7_singletask.png`

### Spectral Edge Verification
- **figSVD1--SVD6** Weight SVD spectral gaps: timeseries, scatter, phase, per-head, narrative, grok vs control
- **figEG1--EG5** Eigenvalue gaps: timeseries, scatter, multi-seed, phase correlation, layer comparison
- **figPP1--PP4** Phase portraits: hero, grid, grok vs control, 3D
- **figL1--L5** Local integrability: per-block, aggregate, basis rank, multi-op, hero
- **figM1--M5** Multi-basis: ratios, phase bars, per-block heatmap, timeseries, all-ops

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

@article{xu2026functional_modes,
  title={Spectral Edge Dynamics Reveal Functional Modes of Learning},
  author={Xu, Yongzhong},
  year={2026},
  eprint={2604.06256},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2604.06256}
}
```
