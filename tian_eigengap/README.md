# Tian Li₂ × Spectral-Edge Eigengap

Empirical tests of two spectral signatures of grokking on
[Tian (2025)](https://arxiv.org/abs/2509.21519)'s exact modular-addition
setup. The technical summary [`paper/note_technical.pdf`](paper/note_technical.pdf)
is the writeup of record; full findings in [`findings.md`](findings.md).

## Headline

- **Stage III lock-in detector (positive result).** The rolling-window
  eigengap σ₂/σ₃ on parameter updates fires at median epoch 174 (IQR=1)
  in 15/15 grok seeds and 0/15 control seeds. Late-stage magnitude
  separation 229× between conditions.
- **Direct Theorem 6 confirmation (mechanistic).** Computing
  B = (F̃ᵀF̃ + ηI)⁻¹ via Woodbury at five deterministic-replay checkpoints,
  the empirical sign-match
  sgn(B<sub>jl</sub>) = −sgn(f̃<sub>j</sub>ᵀ P<sub>η</sub> f̃<sub>l</sub>)
  on the top-200 most-similar feature pairs rises 0.83 → 0.91 →
  0.975 (at the σ₂/σ₃ slope-fire epoch) → 0.995 (deep Stage III).
- **Initiation detector (negative result).** A simpler activation-Gram
  level metric ρ<sub>tian</sub> appears predictive on a single (M, η)
  point and survives an η-sweep, but fires in 60/60 cells of an M×p
  scaling sweep regardless of grokking outcome — a Stage I→II
  initiation marker, not a generalization predictor.

## Quick start

```bash
# 1. Single-seed sanity (η = 2e-4 grok, seed 0). ~70s on M4 Max MPS.
python tian_eigengap.py --M 71 --K 2048 --n-train 2016 \
    --eta 0.0002 --lr 1e-3 --epochs 400 --seed 0 \
    --tag sweep_eta0.0002_seed0

# 2. Headline sweep (15 seeds × 2 conditions). ~35 min.
./run_sweep.sh
python analysis.py
python plot_headline.py

# 3. η sweep — does the lead time generalize? (5 η × 5 seeds, 600 ep). ~30 min.
./eta_sweep.sh
python analyze_eta_sweep.py

# 4. M × p scaling sweep (60 runs). ~2 hours.
./scaling_sweep.sh
python analyze_scaling_sweep.py

# 5. Theorem 6 verification (deterministic replay of seed 0).
python theorem6_verify.py --tag sweep_eta0.0002_seed0 \
    --epochs 50 100 175 250 300 --top-pairs 200

# 6. σ₁/σ₂ reanalysis on existing logs (no retraining).
python sigma12_analysis.py
```

## Files

| File | Purpose |
|------|---------|
| `tian_eigengap.py` | Training. Faithful port of Tian's `ModularAdditionNN` to MPS. Logs train/test acc, Tian Fig. 3 metrics, rolling-window σ₁..σ₅ of ΔW (and ΔV) per epoch. Optional flags: `--static-eig` (CPU SVD on F̃), `--ftf-delta` (rolling F̃ᵀF̃ delta tracker — slow). |
| `analysis.py` | Cross-seed plots and fire-time stats for the headline 15-seed sweep. |
| `analyze_eta_sweep.py` | η-sweep regression; classifies outcome A/B/C. |
| `analyze_scaling_sweep.py` | M×p boundary plot, grok-rate / fft-fire-rate / lead-time heatmaps. |
| `sigma12_analysis.py` | σ₁/σ₂ trajectory reanalysis on existing 30-run sweep. |
| `theorem6_verify.py` | Sign-rule check on top-similarity feature pairs at deterministic-replay checkpoints. Woodbury implementation. |
| `independence_proxy.py` | cos(g<sub>j</sub>, g<sub>j'</sub>) histograms for Tian Stage II decoupling. |
| `plot_headline.py` | 3-panel A/B/C overlay (test acc / Tian level metric / σ₂/σ₃). |
| `plot_two_stage.py` | 4-panel two-stage figure (acc / Tian level / σ₂/σ₃ / fire-time distribution). |
| `plot_pilot_v2.py` | 6-panel pilot overlay including new F̃ᵀF̃ panels and independence proxy. |
| `run_sweep.sh` / `eta_sweep.sh` / `scaling_sweep.sh` | Sequential drivers. |
| `paper/note_technical.tex` / `.pdf` | Technical summary (3 pages, 5 sections + figure). |
| `paper/figures_punch.py` | Generates the σ₂/σ₃ + Theorem 6 sign-match overlay figure. |
| `findings.md` | Detailed writeup of all sweeps, including failed approaches and reframes. |

## Hyperparameter mapping (paper → code)

In Tian's notation **η is the weight decay coefficient, not the learning
rate.** Code uses Adam with `lr=1e-3` (Tian's default) and
`weight_decay=args.eta`.

| Paper (Fig. 3) | This repo | Notes |
|----------------|-----------|-------|
| M = 71 | `--M 71` | modulus |
| K = 2048 | `--K 2048` | hidden width |
| n = 2016 | `--n-train 2016` | 40% of 71² training pairs |
| η = 0.0002 / 0 | `--eta 0.0002 / 0` | weight decay; control is η=0 |
| 400 epochs | `--epochs 400` | full-batch Adam |

## Reproducibility notes

- Runs on Apple M4 Max via PyTorch MPS. SVD and float64 linear algebra
  fall back to CPU (PyTorch 2.5 limitation on MPS).
- Per-run cost: ~60–70s (M=71, 600 epochs); ~15s (M=41, 400 epochs);
  up to ~8 min (M=127, 1000 epochs at p=0.5).
- The rolling-window eigengap is computed by maintaining a deque of
  flattened parameter deltas (size W=20) and calling `torch.linalg.eigvalsh`
  on the W×W Gram. Cost is O(W³), negligible per epoch.
- Theorem 6 verification reduces the K×K matrix inverse to n×n via the
  Woodbury identity and runs in float64 on CPU (~30s per checkpoint).
- Run logs (`runs/`) are not committed — regenerate via the shell scripts.

## Citation

If you use this code or build on these findings:

```bibtex
@misc{xu2026tian_eigengap,
  title={Spectral Marker of Feature Interaction in Tian's Li_2 Grokking Framework},
  author={Xu, Yongzhong},
  year={2026},
  howpublished={\url{https://github.com/skydancerosel/grokking-integrability}}
}
```

Related: [Tian (2025), arXiv:2509.21519](https://arxiv.org/abs/2509.21519).
