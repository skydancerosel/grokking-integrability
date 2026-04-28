# Gradient SED

Per-task gradient subspaces as a corrected diagnostic for feature
formation in grokking and multitask training. Companion to
[`note/short_note.pdf`](note/short_note.pdf) (Xu, 2026).

## TL;DR

The standard "spectral edge" interpretability diagnostic — top right
singular vectors of a rolling window of AdamW updates `Δθ_t` — does
not reliably point at feature-relevant parameter directions. SVDing
the **gradient** `g(t) = ∇L|_θ_t` instead changes the conclusions
qualitatively. On grokking modular arithmetic:

* In **single-task** training, gradient-SED gives R_k peaks of
  100–650× over random directions; update-SED gives 3–9×. A W
  ablation reveals this signal is dominated by the instantaneous
  gradient direction (W=1 ceiling = 648), so the rolling window is
  acting as denoising rather than identifying a privileged subspace.
* In **multitask** training (4 ops, shared encoder), update-SED
  collapses to R_k ≤ 1 — the diagnostic appears to fail. **Per-task**
  gradient SED — recomputing `g_op(t) = ∇L_op|_θ_t` and SVDing per
  op — recovers R_k of 20–45× across all four ops. Gradient
  aggregation across competing tasks is the obstruction.
* A causal intervention (constrain the AdamW attention update to a
  rank-3 subspace) shows the speedup is **rank-specific, not
  direction-specific**: any rank-3 keep accelerates grokking ~2.3×,
  whether the subspace is SED or random. The "remove SED" speedup
  we initially measured under update-projection (1.21×) disappears
  under gradient-projection (1.02×), revealing it as an AdamW
  pipeline artefact.

The clean takeaway: **to understand feature formation, analyse the
gradient — not what AdamW did to it. In multitask, decompose by task.**

The probe of feature formation throughout is the Linear Centroids
Hypothesis ([Walker, Humayun, Balestriero, Baraniuk
2026](https://arxiv.org/abs/2604.11962)); we measure the
perturbative response of LCH centroids to parameter perturbations
along candidate spectral bases.

## Repository layout

```
gradient_sed/
├── README.md
├── note/
│   ├── short_note.pdf            ← 2-page summary
│   ├── short_note.tex
│   └── figs/                     ← figures used in the note
├── training/                     ← scripts that produce checkpoint caches
│   ├── train_dense.py            ← single-task (one op per call), dense
│   │                                checkpointing every 25 steps
│   ├── sed_intervention.py       ← intervention with update-projection
│   │                                (modes A/B/C/D/E)
│   └── sed_intervention_grad.py  ← intervention with gradient-projection
│                                    (cleaner methodology, cf. main paper §5)
├── diagnostics/                  ← post-hoc analyses on existing caches
│   ├── sed_lch_coupling.py       ← rolling-window SED on Δθ (update SED),
│   │                                with --space {full,attn} and
│   │                                --svd-mode {rolling,expanding}
│   ├── sed_lch_gradient.py       ← rolling-window SED on g (gradient SED)
│   ├── sed_lch_multitask.py      ← multitask aggregated update SED (4 ops)
│   ├── sed_lch_multitask_per_op.py ← multitask per-op gradient SED ★
│   ├── sed_lch_ablation.py       ← W, K, ε, probe-seed ablations
│   ├── per_example_grad_svd.py   ← single-timestep alternative to
│   │                                rolling-window gradient SED (Option 1)
│   ├── centroid_full_vs_logit.py ← single-logit vs. full J^⊤·1 centroid
│   │                                comparison
│   └── fourier_readout.py        ← Fourier-basis decomposition of
│                                    centroid PCs (additive + log basis)
└── plotting/                     ← read .pt outputs, produce .png figures
    ├── cross_op_gradient_plot.py     ← gradient-SED vs update-SED, per op
    ├── cross_op_plot.py              ← per-op summary (single-task)
    ├── sed_lch_compare.py            ← attn-only vs full-θ subspace
    ├── sed_lch_multiseed_plot.py     ← 3-seed overlay
    ├── sed_lch_rolling_vs_expanding.py ← rolling vs expanding window
    ├── intervention_compare.py       ← 5-mode intervention curves
    └── fourier_multiseed_plot.py     ← centroid Fourier across seeds
```

## Setup

Python 3.10+, PyTorch with MPS, CUDA, or CPU backend.

```bash
pip install torch numpy matplotlib
```

The scripts assume the model is a 2-layer pre-norm Transformer with
`d_model=128`, 4 heads, `d_ff=256`, GELU activations, on modular
arithmetic with `p=97`. AdamW with `lr=1e-3`, `wd=1.0`,
`β₂=0.98`, batch 512.

## Reproducing the headline numbers

Each block below assumes you have a training cache (single
state_dict per checkpoint, every 25 steps). The scripts read these
caches and produce result `.pt` files plus `.png` plots.

### 1. Train caches (single-task, ~1 minute per seed/op on M4 Max MPS)

```bash
# single-task: add for three seeds
python training/train_dense.py --seed 42   --op add
python training/train_dense.py --seed 137  --op add
python training/train_dense.py --seed 2024 --op add

# other single-task ops
for op in sub mul x2_y2; do
  for s in 42 137 2024; do
    python training/train_dense.py --seed $s --op $op
  done
done
```

Caches go to `coherence_edge_results/training_cache_s<seed>[_<op>].pt`.
Multitask training (4-head shared encoder) is not in this repo — its
cache `training_cache_quadtask.pt` is needed to reproduce
§4 / `sed_lch_multitask*.py`. Contact the author or train your own.

### 2. The headline diagnostic: per-op gradient SED on multitask

```bash
python diagnostics/sed_lch_multitask_per_op.py \
  --cache <path>/training_cache_quadtask.pt
```

Produces `sed_lch_multitask_per_op.{pt,png}`. Expected
post-grokking values: R_k ∈ [12, 45]× for each of the four ops.

### 3. Single-task gradient SED

```bash
# one op, one seed
python diagnostics/sed_lch_gradient.py \
  --cache <path>/training_cache_s42.pt --op add --tag add_s42

# all four ops × three seeds
for op in add sub mul x2_y2; do
  for s in 42 137 2024; do
    python diagnostics/sed_lch_gradient.py \
      --cache <path>/training_cache_s${s}_${op}.pt \
      --op $op --tag ${op}_s${s}
  done
done
```

Expected R_k peaks: 100–330× across ops and seeds.

### 4. W ablation (gradient-direction limit)

```bash
python diagnostics/sed_lch_ablation.py \
  --cache <path>/training_cache_s42.pt --op add
```

Expected: R_1 monotonically decreasing in W (648 at W=1, 212 at
W=20, 140 at W=40). Also sweeps over K, ε, probe-seed.

### 5. Per-example gradient SVD (Option 1; single-timestep alternative)

```bash
python diagnostics/per_example_grad_svd.py \
  --cache <path>/training_cache.pt
```

Expected: R_k of 27, 92, 431 at init/mid/post-grok; cosine to
mean-gradient direction stays low (0.05–0.35).

### 6. Causal intervention (15 runs each on add and mul, ~3 hours total)

```bash
# update-projected (cf. paper Table 2)
for s in 42 137 2024; do
  for m in A B C D E; do
    python training/sed_intervention.py --seed $s --mode $m --op add
  done
done

# gradient-projected (cf. paper Table 3, cleaner methodology)
for s in 42 137 2024; do
  for m in A B C D E; do
    python training/sed_intervention_grad.py --seed $s --mode $m --op add
  done
done
```

Expected (gradient-projected, mean over 3 seeds):

| mode | description           | grok step | speedup |
|------|-----------------------|-----------|---------|
| A    | control               | 3358      | 1.00×   |
| B    | remove SED rank-3     | 3300      | 1.02×   |
| C    | keep only SED rank-3  | **1525**  | **2.20×** |
| D    | remove random rank-3  | 3358      | 1.00×   |
| E    | keep only random 3D   | **1425**  | **2.36×** |

Pattern: any rank-3 keep accelerates ~2.3×; SED-vs-random within
seed variation; remove-SED has no effect under gradient-projection.

### 7. Plot

```bash
python plotting/cross_op_gradient_plot.py
python plotting/intervention_compare.py
python plotting/sed_lch_multiseed_plot.py
```

PNGs land in `sed_lch_results/` next to the `.pt` files they read.

## Key files at a glance

| What you want                       | File                                            |
|-------------------------------------|-------------------------------------------------|
| Multitask per-op SED (★ headline)   | `diagnostics/sed_lch_multitask_per_op.py`       |
| Gradient SED (single-task)          | `diagnostics/sed_lch_gradient.py`               |
| Update SED (rolling Δθ)             | `diagnostics/sed_lch_coupling.py`               |
| W/K/ε ablation                      | `diagnostics/sed_lch_ablation.py`               |
| Per-example SVD (option 1)          | `diagnostics/per_example_grad_svd.py`           |
| Causal intervention (clean)         | `training/sed_intervention_grad.py`             |
| Causal intervention (original)      | `training/sed_intervention.py`                  |
| Single-logit vs full J^⊤·1 centroid | `diagnostics/centroid_full_vs_logit.py`         |
| Fourier readout                     | `diagnostics/fourier_readout.py`                |

## Methodology details

The diagnostic is a perturbative ratio at each checkpoint t:

```
A(v, t)  = (1/|X|) Σ_x ‖[μ_x(θ_t + ε v) − μ_x(θ_t − ε v)] / (2ε)‖_2²
R_k(t)   = A(v_k, t) / median_j A(r_j, t)        over 20 random Gaussian r_j
```

Centroids `μ_x(θ) = ∇_emb ℓ_y(θ; x)` (single-logit, simplified
form of LCH's J^⊤·1). The perturbation ε v is added only to the
attention-weight slots of θ; embeddings, layer-norms, MLPs, and
the head are left untouched. Probe set X has |X|=1024 random
(a, b) pairs sampled once at the start of analysis. Default
ε=0.005·‖θ_attn_t‖.

The two SED estimators are:

- **Update SED**: `v_k(t) = SVD_K(stack of Δθ_τ for τ ∈ [t-W+1, t])`.
- **Gradient SED**: `v_k(t) = SVD_K(stack of g(τ) for τ ∈ [t-W+1, t])`,
  where `g(τ) = ∇_θ L|_θ_τ` on a fixed batch of size 512.

In multitask, **per-op gradient SED** uses
`g_op(τ) = ∇_θ L_op|_θ_τ` per task instead of the aggregated `g`.

W=20 and K=3 throughout, except in the ablation. ε scales
linearly to keep the linear-response regime; ε=0.005 verified
linear by ε=0.001 yielding R_k=221 (within ~5% of canonical 212),
ε=0.05 yielding R_k=74 (non-linear, breaks the regime).

## Limitations

* Single architecture (2-layer pre-norm Transformer), single
  optimizer (AdamW), single p=97 modular-arithmetic family. The
  intervention rank-3 redundancy claim is established only at
  these hyperparameters.
* Multi-seed verification has substantial spread for sub and mul
  (per-seed gradient-SED peaks: 327, 101, 217 for mul); the
  qualitative claim survives at three seeds, the per-op confidence
  intervals do not.
* The centroid is the simplified single-logit form. The full
  J^⊤·1 form agrees on rank-90 throughout and aligns strongly at
  convergence (top-1 PC cos = 0.79); they diverge mid-training.
* Fixed gradient batch (|B|=512); we have not measured sensitivity
  to batch size or batch reseeding.

## Citation

If you use this code, please cite:

```bibtex
@article{xu2026gradient,
  title={{Optimizer Trajectories Mislead: Gradient Decomposition
         Recovers Feature Directions in Multitask Training}},
  author={Xu, Yongzhong},
  year={2026},
  note={Short note; correspondence: abbyxu@gmail.com}
}
```

Companion / referenced works:

- Linear Centroids Hypothesis: Walker, Humayun, Balestriero,
  Baraniuk. arXiv:2604.11962, 2026.
- Multi-task grokking geometry: Y. Xu. *The Geometry of Multi-Task
  Grokking: Transverse Instability, Superposition, and Weight Decay
  Phase Structure.* arXiv:2602.18523, 2026.
- Spectral Edge Thesis: Y. Xu. *The Spectral Edge Thesis: A
  Mathematical Framework for Intra-Signal Phase Transitions in
  Neural Network Training.* arXiv:2603.28964, 2026.

## Contact

abbyxu@gmail.com
