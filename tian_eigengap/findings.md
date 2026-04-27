# Tian Li₂ × Spectral-Edge Eigengap

**Date:** 2026-04-24
**Setup:** Tian arXiv:2509.21519 modular addition. M=71, K=2048, n_train=2016 (40% of 71²=5041), 400 epochs, 2-layer MLP with σ(x)=x², MSE loss on zero-meaned target, identity embedding, Adam lr=1e-3.
**Conditions:** η (=weight decay) = 0.0002 (grokking) vs 0 (no grokking), 15 seeds each.
**Eigengap probe:** rolling-window (W=20) Gram of flattened ΔW updates, top-k eigenvalues via `eigvalsh` on the W×W Gram matrix.

## Headline (cross-seed, n = 15 each) — TWO-STAGE SPECTRAL PICTURE

The data on M=71 supports a **two-stage** spectral picture of grokking:

### (1) Stage II onset — leading indicator: Tian off-diagonal magnitude

The `fft_dist_from_ideal` metric (`‖P₁⊥FFᵀ − (a·I + b·11ᵀ)‖_F / ‖FFᵀ‖_F`)
**leads test accuracy by ~80 epochs with perfect specificity**:

| threshold | grok n_fired @ median | control n_fired | lead vs test_acc=0.5 |
|-----------|----------------------|-----------------|----------------------|
| 0.070 | 15/15 @ ep **15** [14, 15] | **0/15** (max 0.065) | **−86 ep (leads)** |
| 0.075 | 15/15 @ ep **17** [17, 18] | **0/15** | **−84 ep (leads)** |
| 0.085 | 15/15 @ ep **22** [21, 22] | **0/15** | **−80 ep (leads)** |
| 0.10  | 15/15 @ ep **28** [27, 28] | **0/15** | **−73 ep (leads)** |

This is the level-metric form of Tian's "F̃ᵀF̃ ≈ multiple of identity" claim:
when the deviation from the (a·I + b·11ᵀ) ideal first crosses ~0.075 of ‖FFᵀ‖,
features are starting to form. Test accuracy follows ~80 epochs later.

### (2) Stage III lock-in — lagging confirmation: rolling-window σ₂/σ₃ on ΔW

| Metric | η = 0.0002 (grok) | η = 0 (control) |
|--------|-------------------|-----------------|
| Late σ₂/σ₃ on ΔW (median epochs 200–400) | **300** [168, 320] | **1.31** [1.25, 1.37] |
| Late-stage ratio (grok / control medians) | **229×** | – |
| Slope-fire (d log(σ₂/σ₃)/dt > 0.04 for ep ≥ 100) | **15 / 15** @ ep **174** [173, 174] | **0 / 15** | **+35 ep (lags)** |
| Test_acc → 0.5 | median ep **102** [99, 102] | never (~0.03) | – |

This signal indicates the rolling-window ΔW spectrum has become effectively
rank-2 (σ₂ stabilizes, σ₃ collapses 30× — see eigenvalue table below). It
**lags grokking completion by ~35 epochs** with perfect specificity.

### What does NOT work

- **Static F̃ᵀF̃ σ₁/σ₂ and σ₂/σ₃** stay at **1.01–1.07** throughout, in BOTH
  conditions. The static activation-Gram top-3 eigenvalues are nearly equal —
  Tian's "multiple of identity" claim holds even more strongly than just
  diagonal-dominance. Static F̃ᵀF̃ eigvals do **not** discriminate.
- **σ₁/σ₂ on rolling-window ΔW does NOT lead** (cf. updated reanalysis below).
  At low thresholds it fires at epoch 2 in both conditions (window-fill
  artifact); at thresholds where control never crosses, it fires at ep ≥ 160
  (lagging). Median grok minus control σ₁/σ₂ is *negative* through ep 125.
- **Rolling-delta F̃ᵀF̃ σ₂/σ₃** discriminates but with the same lagging timing
  as the ΔW-Gram version (rises post-150 in grok, collapses in control).

Plots: `runs/headline_overlay.png` (3-panel test acc / Tian dist / σ₂/σ₃ ΔW),
`runs/pilot_v2_panels.png` (6-panel including new F̃ᵀF̃ and indep proxy).


## Replication checks (single-seed pilot, seed 0)

| Quantity | η=0.0002 (grok) | η=0 (control) | Tian Fig 3 |
|----------|-----------------|---------------|-----------|
| train acc → 1.0 | epoch 25 | epoch 25 | ≲ 100 |
| test acc → 0.5 | epoch 101 | never (~0.03) | ~100 |
| test acc → 0.99 | epoch 139 | never | ~150 |
| ‖G_F‖ peak | 71.2 @ ep 57 | 63.6 @ ep 49 | rises around ep 100 |
| F̃ᵀF̃ off/diag (max) | 0.033 | 0.028 | < 0.08 |
| FF^T dist-from-ideal (peak in Stage II/III) | 0.105 → 0.145 | 0.063 → 0.056 | ↗ in grok, flat in control |
| σ₂/σ₃ on W (median epochs 200–400) | ~290 | ~1.3 | (new — not in Tian) |

## What σ₂/σ₃ actually measures

Inspecting the raw top-3 eigenvalues at the rolling-window Gram of ΔW reveals:

**Grok (η=0.0002):**
| epoch | σ₁ | σ₂ | σ₃ | σ₂/σ₃ | σ₁/σ₂ |
|-------|------|------|------|--------|--------|
| 25 | 3.6 | 0.46 | 0.027 | 17 | 7.8 |
| 100 | 3.5 | 0.54 | 0.020 | 27 | 6.5 |
| 150 | 0.86 | 0.0146 | 6.8e-4 | 22 | 59 |
| 200 | 0.92 | 0.0044 | 2.1e-5 | **212** | 207 |
| 300 | 0.73 | 0.0026 | 8.4e-6 | **310** | 280 |

**Control (η=0):**
| epoch | σ₁ | σ₂ | σ₃ | σ₂/σ₃ |
|-------|------|------|------|--------|
| 100 | 2.7 | 0.15 | 0.008 | 20 |
| 200 | 0.013 | 6.0e-4 | 5.5e-4 | **1.1** |
| 400 | 1.2e-3 | 1.1e-3 | 9.5e-4 | **1.2** |

So the σ₂/σ₃ blow-up in the grok case is driven by **σ₃ collapsing 30x while σ₂ stabilizes** — the rolling-window update spectrum becomes effectively rank-2. In the control, σ₁, σ₂, σ₃ converge to isotropic noise (all ~5e-4 at late times), no preferred direction.

## Interpretation

The σ₂/σ₃ rolling-window eigengap is a **Stage III lock-in detector**, not a Stage II→III transition trigger as initially hypothesized:

- It fires **deterministically ~35 epochs *after* test_acc reaches 0.99** in the grok condition (median lag 35, IQR [34, 36] across 6 of 6 seeds in pilot — extremely tight).
- It **never fires** in the η=0 control (0/6 seeds), because the parameter updates collapse to isotropic small noise.
- Magnitude separation between conditions is ~225x by epochs 200–400.

Mechanism: in Stage III, Tian's Theorem 6 predicts repulsion between similar features (off-diagonal entries of B = (F̃ᵀF̃ + ηI)⁻¹). The rolling-window ΔW shows this as **persistent rank-2 structure** — σ₁ and σ₂ track stable refinement directions while σ₃+ collapses. Without weight decay, no features form, so updates remain isotropic noise.

## Priority 3 — M × p scaling sweep (60 runs, 2026-04-25)

The scaling sweep **falsifies the strong "leading indicator with predictive
specificity" framing** that the η sweep alone supported. **fft_dist_from_ideal
≥ 0.075 fires in 60/60 runs across the entire (M, p) plane**, including 5/5
seeds in cells that fail to grok within the available training time.

### Grokking outcomes (Tian Theorem 4 reproduction)

| M | p | grok rate | med test=0.99 (ep) | med fft fire (ep) | med lead (ep) |
|---|---|-----------|---------------------|--------------------|---------------|
| 41 | 0.10 | 0/5 | – | 13 | – |
| 41 | 0.20 | 0/5 | – | 14 | – |
| 41 | 0.30 | 0/5 | – | 15 | – |
| 41 | 0.50 | **5/5** | 201 | 18 | **+135** |
| 71 | 0.10 | 0/5 | – | 11 | – |
| 71 | 0.20 | **5/5** | 580 | 13 | **+462** |
| 71 | 0.30 | **5/5** | 249 | 15 | **+183** |
| 71 | 0.50 | **5/5** | 88 | 20 | **+55** |
| 127 | 0.10 | 0/5 | – | 11 | – |
| 127 | 0.20 | **5/5** | 205 | 16 | **+153** |
| 127 | 0.30 | **5/5** | 120 | 22 | **+88** |
| 127 | 0.50 | 0/5 (**numerical divergence**) | – | NaN | – |

Tian Theorem 4 predicted p_crit ≈ log(M)/M: M=41 → 0.091, M=71 → 0.060, M=127
→ 0.038. Boundary partially reproduced for M=71 (p=0.1 fails, p≥0.2 succeeds);
M=41 needs more than 400 epochs even at p≥0.1 (Tian's prediction would have
p≥0.09 grokking eventually); M=127 at p=0.5 numerically diverges (train_acc
collapses from 0.027 at ep 100 to 0.008 by ep 250, fft_dist → NaN — likely a
gradient blow-up, would benefit from grad clipping).

### What fft_dist_from_ideal actually is

The signal fires at a **near-constant epoch (~11–25)** across the entire
(M, p) plane regardless of whether the run will grok. **This is outcome A
from the Priority-1 plan**, not outcome C as the η sweep suggested in
isolation. Reconciling:

- The η sweep showed lead times scaling 79–154 epochs across η — but this is
  because η is the *only* hyperparameter that controls whether feature
  dynamics begin at all (η → 0 gives flat updates and fft never fires).
  η sweep was specifically a "feature dynamics on/off" test.
- The M×p sweep keeps η = 2e-4 (always on) and varies what the dynamics will
  *succeed* at producing. **Across M×p, fft fire is essentially constant (~11–25
  ep) and test_acc fire varies wildly (88–580 ep, or never).** The lead
  varies from +55 ep to +462 ep, or doesn't exist when grokking fails.

The honest interpretation: **fft_dist_from_ideal ≥ 0.075 is a Stage I→II
transition detector** — it fires when feature dynamics begin, which happens
in any condition with non-trivial weight decay regardless of whether
grokking will succeed. It is **necessary but not sufficient** for grokking.

### What this means for the spectral picture

The spectral picture as it stands:

1. **fft_dist_from_ideal ≥ 0.075** at ep ~15 (constant): "feature dynamics initiated"
2. **σ₂/σ₃ slope ≥ 0.04 on rolling-window ΔW** at variable ep (only fires if grokking succeeds): "feature dynamics locked in via Theorem 6 repulsion"

Signal (1) is a Stage I→II marker (necessary). Signal (2) is a Stage II→III
marker that gates on success (sufficient — if it fires, the model has groked).

We have NOT empirically demonstrated a signal that **predicts which dynamics will
succeed** before grokking happens. The η-sweep regression slope=0.80 was real but
narrowly applicable to the η axis — it did not generalize to the M×p axis.

Plot: `runs/scaling_boundary.png`.

## Priority 1 — η sweep (5 η × 5 seeds, M=71, K=2048, 600 epochs, 2026-04-25)

Tests whether the 84-epoch lead time of `fft_dist_from_ideal ≥ 0.075` is real
or an artifact of the (M=71, η=2e-4) point. **Outcome: C** — lead is real and
survives the η sweep; the linear regression gives `t_fft = 0.80 × t_test − 75`
with R² = 0.87, close to the canonical y = x − 84 line.

| η | n_grok | fft_fire (ep) | test=0.5 (ep) | lead (ep) | lock-in (ep) | late σ₂/σ₃ |
|---|--------|---------------|---------------|-----------|--------------|-----------|
| 1e-5 | **0/5** | 542 (1 of 5 barely) | (never) | n/a | 397 | 1.8 |
| 5e-5 | 5/5 | 185 | 312 | **+127** | 280 | 1949 |
| 1e-4 | 5/5 | 20 | 174 | **+154** | 199 | 14.3 |
| 2e-4 | 5/5 | 17 | 101 | **+84** | 173 | 6.8 |
| 5e-4 | 5/5 | 23 | 102 | **+79** | 553 | 20.4 |

Robustness:
- **Lead is positive in 20/20 grokking runs** (5 each at η ∈ {5e-5, 1e-4, 2e-4, 5e-4}).
- Lead magnitude ranges 79–154 ep depending on how slow grokking is.
- η=1e-5 doesn't grok in 600 epochs and the fft signal barely touches the 0.075 threshold (1/5 seeds at ep 527, 1/5 at ep 525, others never) — perfect or near-perfect specificity.
- The σ₂/σ₃ lock-in detector also works across the sweep but its lag varies more wildly (η=5e-5 shows a curious −32 ep value because lock-in fires *before* test=0.5 in slow-grokking conditions; η=5e-4 shows +451 because the late-window slope condition is unstable when grokking happens fast).

Interpretation:
- The fft fire time is roughly constant (~17–23 ep) for η ≥ 1e-4 (fast grokking)
  and rises to ~185 ep for η=5e-5 (slow grokking). The signal therefore tracks
  *feature-emergence dynamics*, not wall-clock — when grokking is slow, the
  off-diagonal magnitude rises slower too, and the threshold crossing is later.
- The lead time is **never zero** across the grokking regime: fft fires
  before test_acc transitions in every grok run we collected.

Plot: `runs/eta_sweep_scatter.png`. The grok-run cluster sits between the y=x
diagonal (no lead) and the y=x-84 reference line (canonical lead time).

## Reviewer-response experiments (2026-04-27)

Following an internal hostile review, ran six follow-ups to address load-bearing
concerns. All findings reported honestly even where they weaken the claims.

### B2 — Multi-seed Theorem 6 (#9: single-seed verification was thin)

Re-ran the Woodbury sign-match analysis on seeds 1–4 of `sweep_eta0.0002`,
giving 5-seed totals:

| epoch | sign-match median | IQR | range |
|-------|------|-----|-------|
| 50  | 0.865 | [0.865, 0.875] | [0.830, 0.900] |
| 100 | 0.895 | [0.880, 0.910] | [0.880, 0.920] |
| **175 (lock-in)** | **0.955** | [0.955, 0.965] | [0.945, 0.975] |
| 250 | 0.970 | [0.965, 0.975] | [0.965, 0.975] |
| 300 | **0.985** | [0.980, 0.990] | [0.965, 0.995] |

Sign-match progression 0.83 → 0.99 reproduces tightly across 5 seeds. The
saturation epoch (≥0.95) coincides with the σ₂/σ₃ slope-fire epoch 174
on every seed. Plot: `paper/figures/multi_seed_thm6.png`.

### B3 — Window-size sensitivity (#3: W=20 was ad hoc)

Ran 3 seeds × 2 conditions × W ∈ {5, 10, 30} (W=20 from existing data):

| W  | grok fire epochs | ctrl fire epochs | late σ₂/σ₃ grok | late σ₂/σ₃ ctrl |
|----|------------------|------------------|------------------|-----------------|
| 5  | [144, 149, 153]  | **[179, 193, 263]** | 408 | 8.0 |
| 10 | [—, —, 347]      | **[263, 266, 278]** | 64  | 2.3 |
| 20 | [173, 173, 174]  | [—, —, —]        | 285 | 1.3 |
| 30 | [180, 180, 180]  | [—, —, —]        | 140 | 1.2 |

**The W=20 choice is genuinely load-bearing.** At W ≤ 10 the slope detector
produces false positives in control (3/3 ctrl fire at W=5; 3/3 ctrl + only
1/3 grok at W=10 — even reversed specificity). Specificity holds for
W ∈ {20, 30}; W=30 fires 6 epochs later but with 2× lower late-stage
magnitude separation. The transition is ~25 epochs wide, so windows must
exceed that to average out single-step noise but not so much that they
smear into Stage I/II. Plot: `paper/figures/window_sweep.png`.

### B4 — η = 10⁻⁵ extended to 2000 epochs (#6: small-η was potentially misleading)

Single seed at η = 10⁻⁵ extended to 2000 epochs:

| metric | epoch |
|--------|-------|
| test_acc ≥ 0.5  | 1094 |
| test_acc ≥ 0.99 | 1527 |
| fft_dist ≥ 0.075 | 527 |
| σ₂/σ₃ slope > 0.04 | 640 |

The model **does grok** with extended training — Tian's 1/η scaling
predicts grokking timescale ∝ 1/η, so η=10⁻⁵ should grok ~10× later than
η=10⁻⁴ (which groks at ep ~150–200, so ep ~1500–2000 expected; observed
ep 1527 ✓). **fft leads test_acc=0.5 by 567 epochs** at this η, consistent
with the 79–154-epoch lead at larger η scaling proportionally with
grokking timescale.

However, the σ₂/σ₃ rank-2 lock-in is **much weaker** at η=10⁻⁵: peak
~25 (vs ~300 at η=2×10⁻⁴), with no clean post-grokking plateau. The
late-stage 229× separation is η-dependent.

### B1 — ReLU activation (#2: load-bearing, σ=x² only)

Re-ran the headline 15-seed × 2-condition sweep with σ(x) = ReLU(x).
ReLU groks at η=2×10⁻⁴ but on a longer timescale (test_acc=0.99 at ep
~530 vs ep ~150 for x²). Both spectral signatures behave very differently:

| Quantity | σ=x² grok | σ=ReLU grok |
|----------|-----------|-------------|
| Late σ₂/σ₃ (median, end of training) | 285 | **2.18** |
| Late σ₂/σ₃ ratio (grok/ctrl) | 229× | **1.4×** |
| σ₂/σ₃ slope-fire (15 seeds) | 15/15 at ep 174 | **0/15 ever** |
| ρ_tian ≥ 0.075 fire epoch | ~17 | **0** (init non-lazy) |
| Theorem 6 sign-match at ep 100 | 0.91 | **0.91** |
| Theorem 6 sign-match at "lock-in" | **0.975** at ep 175 | **0.995** at ep 300 |

**Key takeaway:** Theorem 6 *itself* generalizes — the sign rule
sgn(B_{jl}) = −sgn(f̃_jᵀ P_η f̃_l) holds equally well on ReLU,
saturating at 1.0 by ep 500. But **the rolling-ΔW spectral observable
of Theorem 6 is σ=x² specific.** On ReLU the spectrum is rank-1
dominated (σ₁ ≫ σ₂ ≈ σ₃ ≈ σ₄ ≈ σ₅), not rank-2 — meaning the way
Theorem 6 repulsion translates into parameter-update structure depends
on the activation's gradient geometry. ρ_tian fires at ep 0 because
ReLU initialization is already far from the lazy-regime form.

Honest reframe: σ₂/σ₃ on rolling ΔW is a Theorem-6 indicator
**specifically in the σ=x² regime** where rank-2 lock-in occurs. Plot:
`paper/figures/relu_comparison.png`.

### A4 — baseline comparison (#11: no comparison to alternatives)

Compared σ₂/σ₃ to simpler spectral baselines on the same logs (σ=x²):
σ₁ alone, σ₂ alone, σ₃ alone. All separate grok from control late-stage,
but the timing and magnitude differ. σ₃ alone has the cleanest collapse
(control σ₃ falls 4 orders of magnitude; grok σ₃ stabilizes near 10⁻⁵).
σ₁ alone is dominant but doesn't show the same dynamic range as σ₂/σ₃.
Plot: `paper/figures/baseline_sigmas.png`.

(The originally planned cosine-distance baseline turned out to have a
logging bug — `dW_now` was reading `prev_W` from the previous iteration's
end, giving cosine ≈ 1 at every epoch. Fix is in the script; future runs
will produce correct values.)

### #8 — rank-2 evidence partial (σ₄, σ₅ now logged)

Added σ₄, σ₅ logging; the new window-sweep runs include them. Findings:

- At W=5: σ₃, σ₄, σ₅ all collapse to ~10⁻⁵ together (clean rank-2)
- At W=10: σ₃ ~10⁻⁴, σ₄ ~10⁻⁵, σ₅ ~10⁻⁵ (mostly rank-2)
- At W=30: σ₃ ~10⁻⁴, σ₄ ~10⁻⁵, σ₅ ~10⁻⁶ (cascade — closer to rank-3)

So the "rank-2" framing is exact at small W but more like "rank ≤ 3 with
geometric cascade" at larger W. The σ₂/σ₃ ratio is still the cleanest
detector at W=20, but the underlying rank structure is window-dependent.
Plot: `paper/figures/rank2_top5.png`.

## Theorem 6 (repulsion) — empirically confirmed on M=71, η=2e-4, seed 0

Tian's Theorem 6 predicts that for the matrix B = (F̃ᵀF̃ + ηI)⁻¹, the off-diagonal
entry B_{jl} should satisfy `sign(B_{jl}) = -sign(f̃_jᵀ P_{η,−jl} f̃_l)` on
similar feature pairs, encoding the repulsion that drives Stage III. We verified
this directly via deterministic re-construction of seed 0 at five checkpoints,
on the top-200 most-similar feature pairs (using P_η as approximation to
P_{η,−jl}, since K=2048 ≫ 2 the difference is small):

| epoch | median \|sim\| | frac B<0 on +sim | sign-match (Thm 6) |
|-------|---------------|------------------|--------------------|
| 50  (Stage I/II) | 0.20 | 0.83 | **0.83** |
| 100 (mid Stage II) | 0.22 | 0.91 | **0.91** |
| **175 (lock-in)** | 0.40 | 0.97 | **0.975** |
| 250 (deep III) | 0.59 | 0.97 | **0.975** |
| 300 (deep III) | 0.64 | 0.99 | **0.995** |

The agreement starts at 83% (already above chance) and rises monotonically to
99.5% by epoch 300. **Outcome A** in the user's plan: Theorem 6 holds throughout
and intensifies as features differentiate. The σ₂/σ₃ slope-fire at epoch 174
(the Stage III lock-in detector) coincides almost exactly with the moment the
Theorem 6 sign-match jumps from 0.91 to 0.975 — providing direct mechanistic
grounding: **the rolling-window ΔW eigengap is reading the activation-Gram
repulsion mechanism Tian's theorem predicts.**

Plot: `runs/sweep_eta0.0002_seed0_theorem6.png`. Computed via Woodbury identity
on F̃ in float64 on CPU (n=2016 inverse instead of K=2048).

## σ₁/σ₂ on ΔW is NOT a leading indicator on M=71 (reanalysis 2026-04-25)

Tested the hypothesis that σ₁/σ₂ on rolling-window ΔW Gram opens at the Stage I→II
boundary, leading test_acc by ~50 epochs. **Reanalysis on the existing 30 runs
contradicts the hypothesis on this setup.**

| Epoch | grok median σ₁/σ₂ | control median σ₁/σ₂ | Δ |
|-------|------------------|---------------------|---|
| 30 | 7.9 | 19.3 | **−11** (control higher) |
| 50 | 23.3 | 40.4 | **−17** (control higher) |
| 75 | 23.8 | 15.5 | +8 |
| 100 | 6.5 | 17.9 | **−11** (control higher; grok dips) |
| 125 | 15.2 | 25.8 | **−11** (control higher) |
| 150 | 58.8 | 44.6 | +14 (grok crosses ahead) |
| 170 | 140 | 75 | +65 |
| 200 | 208 | 15 | +193 (control collapsing) |

Through epoch 125, control's σ₁/σ₂ is *higher* than grok's at the median.
The two cross at epoch ~150 and only diverge dramatically after epoch 170.
Test_acc=0.5 is at epoch 102 (median, IQR [99, 102]).

Fire-time test, multiple thresholds and slope-based detectors:

| Detector | Grok n_fired @ median epoch | Control n_fired | Lag vs test_acc=0.5 |
|----------|----------------------------|------------------|---------------------|
| σ₁/σ₂ ≥ 25 sustained 5 ep | 15/15 @ ep 2 | **15/15 @ ep 2** | (fires too early in both) |
| σ₁/σ₂ ≥ 100 | 15/15 @ ep 160 | 0/15 | **+58 ep (lags)** |
| σ₁/σ₂ ≥ 200 | 15/15 @ ep 191 | 0/15 | **+89 ep (lags)** |
| d log(σ₁/σ₂)/dt ≥ 0.04, ep≥100 | 15/15 @ ep 139 | 0/15 | **+37 ep (lags)** |

The cleanest specific detector (slope-based, ep≥100) fires at epoch 139 — STILL after test_acc=0.5 (median 102). **No threshold and no slope window we tried makes σ₁/σ₂ on ΔW lead test accuracy on this M=71 setup.**

Why: rolling-window ΔW captures the *consistency* of update direction. In Stage I,
the gradient is dominated by the V (output) layer chasing the ridge solution — this
gives a strong σ₁/σ₂ regardless of whether feature learning will follow. The
discriminating event (control's collapse to isotropic noise) only manifests after
grokking has already happened. So the rolling-window ΔW signal **lags** the
generalization transition rather than leading it, both for σ₂/σ₃ (Stage III lock-in,
~+35 ep) and σ₁/σ₂ (~+37 ep slope, ~+58 ep level).

The previous round's result (σ₂/σ₃ = Stage III lock-in detector with 229× late
separation, 0/15 false positives) stands. The σ₁/σ₂ result is qualitatively the
same signal — both are post-grokking spectral signatures of feature lock-in.
σ₁/σ₂ does not provide a separate Stage I→II window on this setup.

## Anomalies / caveats

- **F̃ᵀF̃ off/diag stays small (<0.04) in BOTH conditions** throughout training. The user's plan hypothesized this metric would cross a threshold at the Stage II→III boundary; empirically it does not. The cleaner Stage II→III signature is `fft_dist_from_ideal`, which rises from 0.058 to 0.105 in Stage II and then 0.087 → 0.145 in Stage III, only in the grok case.
- The σ₂/σ₃ signal **lags grokking** rather than leading it. So the "predict generalization before it happens" framing of the original plan does not hold here. What we have instead is a **specificity-1 Stage III detector**.
- Findings are on M=71. Different M may give different timing (Tian's framework predicts feature-emergence scaling with M log M).
- Independence proxy: median |cos(g_j, g_{j'})| ≈ 0.08 throughout, supporting Tian's Stage II decoupling (each w_j ascends E independently).

## Files

- `tian_eigengap.py` — training + per-epoch logging
- `analysis.py` — cross-seed plots and fire-time stats
- `plot_headline.py` — 3-panel A/B/C overlay (test acc / Tian metric / σ₂/σ₃)
- `independence_proxy.py` — Tian Stage II decoupling check via cos(g_j, g_{j'})
- `runs/sweep_eta{val}_seed{s}/log.jsonl` — per-run logs
- `runs/headline_overlay.png` — main figure
- `runs/eigengap_slope.png` — slope-based detector showing positive peak (grok) vs negative dip (control)
