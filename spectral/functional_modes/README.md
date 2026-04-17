# Functional Modes of Learning — Code

This folder contains the analysis scripts accompanying:

**Spectral Edge Dynamics Reveal Functional Modes of Learning**
[arXiv:2604.06256](https://arxiv.org/abs/2604.06256)

## Summary

The paper reinterprets spectral edge directions (leading singular vectors of the weight-update trajectory) as **functions over the input domain**, not as localized circuits or features. For each direction $v_k$, the induced perturbation field
$$f_k(a,b) = \|h(a,b; \theta + \varepsilon v_k) - h(a,b; \theta)\|^2$$
captures structured behaviour in appropriately chosen bases.

Main findings (see Figure 1):
- **Addition**: single Fourier mode ($\omega \approx 25$–$26$) under additive basis
- **Multiplication**: single mode ($\omega = 29$) under discrete-log basis, $5.9\times$ concentration improvement
- **Subtraction**: multi-mode subspace, no edge/bulk separation (honest negative)
- **$x^2+y^2$**: no single harmonic basis works; cross-terms of additive and multiplicative features give $4\times$ $R^2$ boost
- **Multitask reuse**: tritask model inherits $\omega = 26$ from the addition head ($2.3\times$ concentration)

## Scripts

### Paper figure
| Script | Purpose |
|---|---|
| `paper_figure_basis.py` | Generates Figure 1 (`fig1_basis_dependence.png`): 5-panel basis-dependence plot |

### Mechanistic-interpretability null results (Section 4)
| Script | Purpose |
|---|---|
| `v123_feature_attribution.py` | Head purity null result (purity $\approx 1/8 =$ uniform baseline) |
| `residual_stream_alignment.py` | Effective rank of $\Delta h$ ($\approx 40$) and Fourier peakedness |
| `sae_fourier_features.py` | SAE Jaccard significance tests (not significant vs angle-matched null; $p \geq 0.97$) |

### Fourier functional analysis (Section 5)
| Script | Purpose |
|---|---|
| `fourier_functional_view.py` | Core Fourier profiles for all 4 ops; peak frequency, $F_k$, basis test |
| `fourier_dlog_mul.py` | Discrete-log basis for modular multiplication (5.9× improvement over raw Fourier) |

### $x^2+y^2$ composition analysis (Section 5.5–5.6)
| Script | Purpose |
|---|---|
| `x2y2_composition_test.py` | Probing with additive + multiplicative + cross-term features (4× $R^2$ boost) |
| `x2y2_multitask_composition.py` | Single-task vs tritask comparison (2.3× concentration, 1.7× synergy) |

## Data requirements

Scripts expect these inputs relative to the spectral folder:

```
spectral/coherence_edge_results/
    training_cache.pt        (add)
    training_cache_sub.pt
    training_cache_mul.pt
    training_cache_x2_y2.pt

grok_sweep_results/
    add_wd1.0_s42.pt
    sub_wd1.0_s42.pt
    mul_wd1.0_s42.pt
    x2_y2_wd1.0_s42.pt

multitask/results/
    tritask_wd1_s42.pt       (only for x2y2_multitask_composition.py)
```

The training cache files are produced by scripts in `training/` and contain checkpoints, metrics, and test pairs. The `grok_sweep_results/` files contain attention weight logs needed for the SVD update analysis. The tritask model comes from the multitask training scripts.

## Reproducing Figure 1

```bash
python spectral/functional_modes/paper_figure_basis.py
```

This runs the perturbation analysis on post-grok checkpoints for all 4 operations under both raw and discrete-log bases and produces `fig1_basis_dependence.png`.

## Citation

```bibtex
@article{xu2026functional_modes,
  title={Spectral Edge Dynamics Reveal Functional Modes of Learning},
  author={Xu, Yongzhong},
  year={2026},
  eprint={2604.06256},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2604.06256}
}
```
