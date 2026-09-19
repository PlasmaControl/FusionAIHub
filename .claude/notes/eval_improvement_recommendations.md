# E2E Stage-1 foundation model — weaknesses, limitations, and improvement recommendations

**Checkpoint**: `/lustre/orion/proj-shared/fus187/models/e2e_stage1_d1024_48L/e2e_stage1_best.pt`
(1.33 B params, d1024 / 48 L / 16 heads, 12 diagnostics + 9 actuators, 50 ms chunks)
**Evaluation**: 875 held-out validation shots + 875 matched train shots (seed-42 split),
~38.5 k windows per split on a fixed 0.25 s grid. All numbers reproduce the training-time
validation semantics (cross-check gate: worst ratio deviation 0.126, factor skew 1.18×).
**Status**: living document — sections marked *[pending]* fill in as the GPU studies land.
Figures/tables referenced live under `data/outputs/eval_suite/e2e_stage1_best/`.

## Executive summary

1. **The model is underfit, not overfit.** Train and val model/persistence MAE ratios are
   statistically indistinguishable (mean gap 0.011, max 0.024 across 12 modalities;
   `tables/a1_mae_table.csv`), and val windows are no closer to train windows than to
   other val shots in latent space (closer-to-train fraction 0.476 ≈ chance;
   `tables/a6_memorization.json`). The "generative overfitting" hypothesis is ruled out
   by two independent tests. Regularization is not the first-order problem — capacity
   utilization and training signal are.

2. **The latent space is rank-collapsed.** The 1024-dim pooled diagnostic representation
   has participation ratio **7.3** (effective rank ≈ 22; top-10 PCs = 75 % of variance);
   single-modality token slices are worse (ts_core_density ≈ 2 effective dims)
   (`tables/a7_spectrum.csv`, `figures/study_a/a7_eigenspectrum`). The model uses ~1 % of
   its representational width. This one defect coherently predicts the other findings
   below (spectral detail loss, probe plateaus, diluted pooling).

3. **Fine-timescale event structure is lost.** A frozen-probe ELM predictor from the
   global latent reaches AUROC 0.843 — above the operating-point control (0.757) but
   **below trivial persistence (0.925)**, even though the D-alpha trace that defines the
   label is part of the model's *input* (`tables/c_probe_metrics.json`). No token slice
   closes the gap (best 0.88). The ELM-cycle timing present at the input is not linearly
   recoverable after tokenization + 48 backbone layers. *[pending: does LoRA finetuning
   recover it? — flagship comparison, `figures/study_c/c2b_sample_efficiency_lora`]*

4. **Slow "operating-point" physics is encoded well.** Densities/temperatures/D-alpha
   level probe at R² 0.84–0.96 from the frozen latent, clearly above the actuator-scalar
   control (`tables/a8_probe_r2.csv`); concurrent Mirnov/MHR band power probes at
   R² 0.86 / 0.85 (control 0.74 / 0.70) with strong label efficiency (R² ≈ 0.65 from
   5 labeled shots; `figures/study_c/c2_sample_efficiency`). The representation is a
   good *slow-state* summary — its weaknesses are spectral detail, fast events, and
   (probably) actuator response.

5. **The spectral deficit is severe and frequency-resolved (A4).** Predicted
   fluctuation power is ~10⁻³ of ground truth across the whole co2 band (the model
   predicts an almost fluctuation-free interferometer spectrogram), ~10⁻² at low/mid
   frequency rising to only 0.3–0.4 at ≳100 kHz for ece/bes, and 10⁻³–10⁻² for the
   filterscopes PSD (`figures/study_a/a4_spectral_fidelity`). This is
   regression-to-the-mean over-smoothing of a deterministic MAE-trained predictor on
   stochastic fields — and it is the same mechanism that erases ELM timing (finding 3).
   The deficit is worst exactly in the low/mid band where MHD and turbulence live.

6. **Actuator conditioning: right directions, near-zero gain (B1 + B3-swap).**
   Swapping in a *different window's entire actuator trajectory* changes K=1
   predictions by almost nothing — cosine similarity ≥ 0.998 on every modality
   (`study_b/b3_actuator_swap.npz`), confirming the stage-2 audit on this stage-1
   checkpoint. Yet the *directions* of the responses that do exist are largely
   correct: 9/12 textbook sign expectations pass under ±20 % physical scaling,
   including RMP density pump-out on both density channels; the 3 failures
   (beam-voltage→rotation, ECH→Te, RMP→braking) have sign-consistency ≈ 0.5, i.e.
   no coherent response rather than wrong physics (`tables/b1_sign_matrix.csv`).
   The model has learned actuator physics directionally but couples it into
   predictions at negligible amplitude. *[pending: B2 dose-response curves, B4
   sustained-actuation rollout signatures, B3 layerwise propagation, A9 divergence
   horizon.]*

## Detailed findings and evidence

| # | Finding | Evidence | Status |
|---|---------|----------|--------|
| F1 | No memorization: train/val ratio gap ≤ 0.024 everywhere | a1_mae_table, a1_verdict.txt | confirmed |
| F2 | No latent memorization: NN-retrieval at chance | a6_memorization.json (frac 0.476, 20 k queries) | confirmed |
| F3 | Rank collapse: PR 7.3 / 1024 global, ~2–3 per modality slice | a7_spectrum.csv, a7_eigenspectrum | confirmed |
| F4 | ELM probe < persistence (0.843 vs 0.925 AUROC) | c_probe_metrics.json, c2_sample_efficiency | confirmed |
| F5 | Mean-pooling dilutes: modality slices beat global on rotation (0.44 vs 0.16 R²) and Ti (0.63 vs 0.57) | a8_probe_r2.csv | confirmed |
| F6 | Token homogenization: pooled actuator tokens ≈ diag tokens as physics predictors (0.92 vs 0.93 ne R²) | a8_probe_r2.csv | confirmed |
| F7 | Spectrogram modalities weakest relative gain (co2 0.83) | a1_mae_table | confirmed |
| F8 | Spectral deficit: pred/GT fluctuation power ~10⁻³ (co2), 10⁻²–0.4 (ece/bes), 10⁻³–10⁻² (filterscopes) | a4_spectral_fidelity | confirmed |
| F9a | Swap invariance: pred cos-sim ≥ 0.998 all modalities under full actuator-trajectory swap | b3_actuator_swap.npz | confirmed |
| F9b | Sign correctness 9/12 (failures = incoherent response, consistency ≈ 0.5) | b1_sign_matrix.csv | confirmed |
| F9c | Dose-response monotonicity / sustained-actuation signatures | b2_*, b4_* | **[pending]** |
| F10 | Layerwise actuator-information attenuation | b3_layerwise | **[pending]** |
| F11 | Rollout divergence horizon vs persistence | a9_rollout_divergence | **[pending]** |
| F12 | LoRA vs frozen probe at matched label budgets | study_c/lora/*, c2b overlay | **[pending]** |

## Recommendations (standard practices, mapped to findings)

**R1 — Attack rank collapse directly (F3, F5, F6).**
- Add a variance–covariance regularizer on pooled token representations
  (VICReg-style variance + covariance terms, or a whitening/decorrelation penalty).
  This is the standard remedy for dimensional collapse in joint-embedding SSL and
  applies cleanly to a masked-reconstruction world model as an auxiliary term.
- Monitor effective rank / participation ratio of token embeddings as a *training
  metric* (cheap: PCA on a held-out batch each eval step) — collapse is visible long
  before downstream probes degrade.
- Consider per-modality projection heads before pooling for downstream use; the A8
  slice-vs-global gap shows mean-pooling over 1178 tokens is lossy — an attention
  pool or learned query pooling would preserve modality-specific detail.

**R2 — Spectral-fidelity training signal (F7, F8).**
- Add multi-resolution spectral losses on the fast/spectrogram channels (the
  MR-STFT loss standard in neural audio synthesis): L1 on log-magnitude at several
  FFT resolutions. MAE on standardized spectrogram pixels systematically
  under-weights high-frequency, low-power structure — exactly the co2/ece/bes gap.
- Re-examine per-bin standardization: log_standardize with a global std flattens the
  dynamic range the model must reproduce; per-frequency-band statistics (already in
  the stats pipeline) plus a power-law reweighting of the loss would push capacity
  toward the informative bands. *[refine once A4 shows the deficit's frequency profile]*

**R3 — Preserve fast-event information (F4).**
- The filterscopes tokenizer compresses 500 samples/window; ELM spikes survive in the
  input but not the latent. Standard fixes: (a) auxiliary event heads during
  pretraining (predict peak count / max in window — labels are free from the raw
  trace); (b) higher token rate for fast channels; (c) anti-smoothing losses
  (quantile or peak-weighted regression instead of plain MAE).
- Cheap immediate win for applications: concatenate trivial recency features
  (current-window D-alpha stats) with the latent for event tasks — persistence
  information is complementary, not redundant.

**R4 — Strengthen actuator conditioning** *[finalize after B1–B4/A9]*.
- If B-studies confirm weak conditioning: actuator-dropout pretraining (mask actuator
  tokens with probability p so diagnostics-only prediction degrades and the model is
  forced to *use* actuators when present), FiLM-style conditioning of backbone blocks
  on actuator summaries, or a contrastive swap penalty (embeddings under swapped
  actuator futures must diverge).

**R5 — Train longer / rebalance capacity (F1, F3).**
- The model is data/schedule-limited, not regularization-limited. Before adding
  parameters, extend the token budget: 48 L × d1024 at 1.33 B params is far past the
  compute-optimal ratio for the ~10⁹-token corpus implied by 875-shot-scale training
  data. More epochs with the R1/R2 auxiliary losses (which add gradient signal per
  token) is the highest-leverage change; a smaller-but-longer-trained model is a
  worthwhile ablation.

## Limitations of this evaluation

- **Window-mix sensitivity**: absolute MAEs depend on the window grid (dense 0.25 s
  eval grid vs training's sampling); all cross-checkpoint comparisons here use
  model/persistence *ratios*, which are invariant (gate: skew 1.18×).
- **Thomson persistence caveat**: 17.5 % of ts_tangential windows have exactly
  identical input/target (laser rate), inflating the copy baseline's difficulty there.
- **pin validity**: total injected power is valid in only ~55 % of windows; actuator
  studies gate on validity but effective n is smaller.
- **Proxy MHD labels**: mirnov/mhr band power stands in for n=1/n=2 amplitudes until
  the omega fetch lands (`scripts/data_fetching_omega/n1rms_fetch/`, 165 curated
  shots); band power conflates toroidal mode numbers.
- **LoRA eval subset**: LoRA cells validate on a fixed 200-shot val subset (full-875
  forward passes are ~20 min each); the frozen-probe curve is computed on all 875.
  The overlay figure states this.
- **ELM labels**: threshold-based (MAD prominence on D-alpha); the 3-shot validation
  figure is at `data/outputs/eval_suite/label_check/elm_validation.png` and should be
  sign-off reviewed.
