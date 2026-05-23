# Research Plan: End-to-End Foundation Model for Multi-Modal Tokamak Plasma Prediction

**Institution:** Princeton University / Princeton Plasma Physics Laboratory
**Target system:** DIII-D (extensible to other devices)

---

## 1. Scientific Motivation

Real-time prediction of tokamak plasma evolution is essential for advanced control, disruption avoidance, and scenario optimization. Current approaches fall into three categories, each with fundamental limitations:

**Physics-based transport solvers** (TRANSP, TGLF, GENE) solve coupled partial differential equations for particle, momentum, and energy transport. They are accurate but computationally expensive (minutes to hours per simulation), making them unsuitable for real-time control loops that require predictions within milliseconds.

**Single-diagnostic ML models** predict one quantity (e.g., electron temperature profile) from a limited set of inputs. They cannot capture cross-diagnostic physical couplings — for instance, how neutral beam injection simultaneously affects ion temperature (CER), plasma rotation (CER), electron density (Thomson), magnetic field pitch (MSE), and edge emission (filterscopes). These couplings are fundamental to plasma behavior and control.

**Latent-space dynamics models** use pretrained autoencoders to compress diagnostic signals into learned latent representations, then predict dynamics in this compressed space. Our preliminary measurements on DIII-D data (Section 1.1) demonstrate that this approach suffers from a fundamental limitation: reconstruction-trained autoencoders produce latent spaces with no consistent geometric relationship to temporal dynamics. A sufficiently powerful decoder absorbs all geometric complexity, leaving the encoder free to arrange the latent manifold arbitrarily with respect to time. No foundation model architecture or loss function can overcome this upstream representation failure.

**The proposed approach** eliminates this failure mode by training all representations end-to-end under the prediction objective, following the paradigm established by Aurora (Bodnar et al., Nature 2025) for atmospheric prediction. Unlike reconstruction-trained autoencoders, where the latent geometry is unconstrained by the downstream prediction task, end-to-end tokenizers produce representations whose geometry is shaped entirely by the requirement to predict future states. Temporal smoothness and dynamical informativeness emerge as byproducts of the prediction loss — properties that cannot be achieved through post-hoc regularization of separately trained encoders.

### 1.1 Preliminary Evidence: Latent Space Temporal Discontinuity

In prior work on this project, we trained per-modality convolutional autoencoders on eight DIII-D diagnostics (filterscopes, Thomson scattering core/tangential density and temperature, MSE, CER ion temperature and rotation). We measured the Spearman rank correlation between signal-space cosine similarity and latent-space cosine similarity for consecutive 500 ms and 50 ms temporal windows across the training dataset.

**Finding:** All eight modalities exhibit Spearman rank correlation ≤ −0.1 (negative or near-zero). Physically similar consecutive plasma states map to dissimilar latent points, and vice versa. This anti-correlation persists across autoencoder architectures (varying depth, bottleneck dimension), compression ratios (50 ms vs 500 ms windows), regularization strategies (metric-matching temporal loss), and training procedures (unfreezing encoders during prediction training). The result is consistent with the theoretical expectation that a sufficiently expressive decoder can invert any encoding regardless of its geometric structure, rendering the reconstruction loss blind to latent manifold geometry.

This finding motivates the end-to-end approach: the only way to guarantee prediction-friendly geometry is to train the representation under the prediction objective itself.

## 2. Scientific Contributions

This work makes three contributions:

**C1. First multi-modal tokamak foundation model operating on raw heterogeneous signals.** The model simultaneously ingests time series (100 Hz–10 kHz), spectrograms (500 kHz), and video sequences, producing predictions across all modalities conditioned on actuator commands. No prior work handles this heterogeneity in a unified predictive framework.

**C2. Actuator-conditioned prediction for control.** Given a proposed actuator trajectory (beam injection, ECH power, gas fueling, RMP coils), the model predicts the plasma response across all diagnostics. This enables "what-if" scenario evaluation orders of magnitude faster than physics-based simulations, suitable for real-time model-predictive control and between-shot planning.

**C3. Comprehensive verification methodology for autoregressive prediction architectures.** We present an impulse-based test suite (~50 tests) that verifies signal propagation through every architectural component before training begins, and diagnostic metrics (delta-ratio, per-step cosine similarity, per-stage signal pathway analysis) that localize failure modes to specific modules during training. This methodology applies beyond the tokamak domain to any autoregressive prediction system operating on heterogeneous inputs.

## 3. Architecture

### 3.1 Design Principles

1. **Raw signal input, end-to-end representation learning.** No separately trained autoencoders. Per-modality tokenizers are trained jointly with the prediction backbone under the prediction loss. The learned representations are therefore geometrically constrained to be dynamically informative — unlike reconstruction-trained latent spaces where a powerful decoder decouples latent geometry from temporal structure.
2. **50 ms temporal windows.** Each prediction step covers 50 ms, providing 20 Hz temporal resolution. An 80-step rollout covers 4 seconds.
3. **Per-modality tokenization.** Each diagnostic type has a specialized tokenizer matched to its data structure and sampling rate.
4. **Shared backbone.** A single Transformer processes all modality tokens jointly, enabling cross-diagnostic coupling to emerge from data.
5. **Token-space rollout.** During autoregressive prediction, backbone output tokens flow directly to the next step without de-tokenization and re-tokenization. The output heads only fire for loss computation. This eliminates the encode–decode roundtrip information loss that caused rollout collapse in the AE-based architecture.
6. **Actuator conditioning through cross-attention.** The backbone cross-attends to actuator tokens at each rollout step, enabling the prediction to be conditioned on time-varying actuator commands.

### 3.2 Temporal Window: 50 ms

The 50 ms window is chosen to balance three constraints:

- **Diagnostic coverage:** At 100 Hz (Thomson, CER, MSE), 50 ms provides 5 samples per channel — sufficient to capture the local signal shape. At 10 kHz (filterscopes), 50 ms provides 500 samples — rich temporal structure.
- **Temporal smoothness:** Consecutive 50 ms windows differ by ~5% in physical content. The prediction task is geometrically well-posed in raw signal space without any representation engineering.
- **Rollout horizon:** 80 steps cover 4 seconds. The per-step prediction task is easier than with 500 ms windows (smaller changes per step), at the cost of more rollout steps. The pushforward trick and replay buffer (Section 4.3) are specifically designed to handle long rollouts efficiently.

### 3.3 Per-Modality Tokenizers

| Modality Type | Example | Sampling | Window (50 ms) | Tokenization | Tokens |
|---|---|----------|---|---|---|
| Slow time series | Thomson (core + tangential density, temperature), CER (Ti, rotation), MSE | 100 Hz   | 5 samples/ch | Linear per channel | ~90 total (6 modalities × ~15 ch) |
| Fast time series | Filterscopes | 10 kHz   | 500 samples/ch | Conv1d patching (stride 50) | ~80 (8 ch × 10 tokens) |
| Spectrogram | ECE (32 ch), BES (64 ch), CO2 (1 ch) | up to 1 MHz raw  | ~194 frames × 513 freq bins/ch (STFT n_fft=1024, hop_length=256) | Conv2d patches | ~480 (ECE 192 + BES 192 + CO2 96) |
| Video | Fast camera (tangtv, 2 active filters) | 100 fps  | 3 frames × 120 × 360 | Tube-patch Conv3d (T_p, H_p, W_p) = (3, 12, 12) | ~300 (10 × 30 patches per camera) |
| Actuators | NBI, ECH, gas, RMP | varies   | 50 ms | Conv1d patching | ~45 |
| | |          | | **Total (full config):** | **~1180** |

Total token count varies with configuration: time series only (~398), TS + video (~700), or the full BC configuration (~1180). Standard self-attention (O(N²)) remains feasible at all of these on a modern GPU. At N≈1180 with d_model=256 and 8 heads, per-layer attention cost is ~1.1 GFLOPs and the realised per-step cost scales as ~2.1× over TS-only because the FFN (linear in N) dominates the per-layer compute at this width. A Perceiver compression stage after tokenization remains an option if further modalities are added.

Each tokenizer adds a learned modality embedding and positional encoding. All tokenizer weights are trained end-to-end with the backbone.

### 3.4 Shared Backbone

Standard Transformer encoder with pre-norm (LayerNorm before attention, not after). Eight self-attention layers, d_model=256, 8 heads, MLP ratio 4. All diagnostic and actuator tokens attend to each other — cross-diagnostic coupling is learned implicitly through self-attention.

Step conditioning: Fourier features of the rollout step index and absolute time offset, projected through a 2-layer MLP, added to all tokens. This allows the backbone to modulate predictions based on rollout depth.

### 3.5 Per-Modality Output Heads

Each modality has an output head that projects backbone tokens back to the raw signal space. These are approximate inverses of the tokenizers (Linear for slow TS, ConvTranspose1d for fast TS, ConvTranspose2d for spectrograms, single ConvTranspose3d with kernel and stride matching the tube-patch shape for video — each video token reconstructs its own (C, T_p, H_p, W_p) region with no global mixing). Output heads fire only for computing the training loss against ground truth raw signals. During rollout, backbone tokens pass directly to the next step without going through the output heads.

### 3.6 Rollout Architecture

```
Step 0: tokenize(raw_signals) → tokens₀
Step k: backbone(tokensₖ₋₁, actuator_tokensₖ, step=k) → tokensₖ
        output_heads(tokensₖ) → raw_pred  (for loss only, not fed back)
        tokensₖ flows directly to step k+1
```

The 80-step rollout (4 seconds at 50 ms resolution) operates entirely in token space. No re-tokenization between steps. This is the key architectural choice that eliminates the encode–decode roundtrip information loss observed in the AE-based architecture, where the Perceiver encoder–decoder cycle erased temporal variation within 3 rollout steps.

## 4. Training Procedure

The pipeline has three training stages, applied in turn during Phase A (TS only) and again during Phase BC (TS + spectrograms + video). Phase BC additionally warm-starts from the corresponding Phase A checkpoint via an explicit checkpoint loader that allows missing keys for the new modality modules and rejects unexpected keys.

### 4.1 Stage 1: Single-Step Pretraining

**Objective:** Learn tokenizers, backbone, and output heads for one-step (50 ms) prediction.

- Loss: MAE in raw signal space, per-modality, all normalized to unit variance (precomputed statistics). For spectrograms and video the loss is gated by `{name}_channel_mask` (per-channel) and `{name}_valid` (per-batch), so missing modalities and off-channels do not contribute to gradients.
- Data: All available DIII-D shots, chunked into consecutive 50 ms windows with 10 ms step size.
- Duration: until validation MAE plateaus.
- Full weight updates on all parameters by default. In the BC variant, four orthogonal freeze flags (`--freeze_{ts,video,spectro,backbone}_steps`) hold subsets of the model fixed for the first N steps so that freshly-initialised modules can settle without perturbing the warm-started backbone.

### 4.2 Stage 2 (delta): Teacher-Forced Curriculum to K=10

**Objective:** Teach the model to handle its own outputs as input for short horizons.

- Rollout curriculum: K=1 → K=10 steps (50 ms → 500 ms).
- Each rollout step receives ground-truth diagnostic tokens as input (teacher-forced); the prediction at step k is compared against the ground-truth at step k.
- Loss: per-step weighted sum of MAE plus a delta-loss decomposition (cosine similarity of the predicted vs ground-truth displacement, magnitude ratio of the same), computed in token space for TS and in raw signal space for spectrograms and video. The delta decomposition replaces a plain `F.l1_loss(pred-ctx, target-ctx)` formulation, which is algebraically equivalent to the un-decomposed MAE.
- Full backpropagation through all K steps; full weight updates.

### 4.3 Stage 2 Extended: Free Rollout to K=80 with Scheduled-Sampling TF

**Objective:** Stable 80-step (4-second) autoregressive prediction without distribution shift between training and inference.

**Free rollout:** At each rollout step the backbone-output diagnostic tokens are fed directly as input for the next step (no re-tokenization of ground truth). This matches inference behaviour exactly and is what failed catastrophically when attempted naively from a Stage-2b checkpoint — k=1 MAE regressed by 13–69 % because the backbone had been conditioned on `tokenize(GT)`-style prefixes it never sees in pure free-rollout.

**Scheduled-sampling teacher-forcing:** A teacher-forcing probability `p_tf = max(0, 1 − step / tf_anneal_steps)` decays linearly from 1.0 to 0.0. With probability `p_tf` at each k ≥ 1, the next-step diagnostic input is re-tokenized GT instead of the previous step's backbone output. This bridges the Stage-2b regime (always teacher-forced) and pure free-rollout (`p_tf = 0`). Default schedule: `tf_anneal_steps = 40 000` — full TF at step 0, pure free-rollout from step 40 k onward. Validation always runs at `p_tf = 0` so cross-run comparisons stay independent of the in-progress TF schedule.

**Gradient checkpointing:** All K = 80 forward steps run under `torch.utils.checkpoint` so activations memory scales per-step, not K × per-step. TF coin flips for the K rollout steps are pre-drawn outside the checkpointed region so backward replays the same TF decisions on recompute.

- Rollout: K = 80 steps.
- Init from Stage 2 (delta) best; all weights trainable.

## 5. Per-Block Verification Tests

Every architectural block is verified before integration with three categories of tests: impulse tests (does signal propagate?), gradient tests (do parameters receive gradients?), and functional tests (does the block do what it should?). All tests use a small model (d_model=32, 2 backbone blocks, batch_size=2) and run in under 1 minute each.

Hard-won design rules encoded in the tests:
- **Never use constant-valued impulses** — LayerNorm maps constant vectors to the learned bias, erasing the input signal entirely.
- **Never apply LayerNorm after concatenating token groups** — shared normalization dilutes the data-dependent signal relative to learned embeddings.
- **Copy baseline tests must use deterministic targets** — random noise targets make copying the optimal strategy, rendering the test self-defeating.
- **Always test that training resolves random-init bottlenecks** — structural problems and init artifacts look identical at random init; a 50-step training loop distinguishes them.

### 5.1 Slow Time Series Tokenizer (4 tests)

- **Impulse — input reaches tokens:** Zero all channels except one (set to `randn(5) * 5.0`). Active token norm > 2× zero tokens. *Failure: dead projection or embedding dominance.*
- **Impulse — different inputs → different tokens:** Two random inputs. cos_sim < 0.95. *Failure: embedding dominates input.*
- **Gradient — projection weights receive non-zero `.grad`.**
- **Shape — output tokens = n_channels.**

### 5.2 Fast Time Series Tokenizer (5 tests)

- **Impulse — step vs ramp:** Constant vs linearly increasing signal. Total diff norm > 1.0. *Failure: dead Conv1d or signal-killing normalization.*
- **Impulse — temporal localization:** Signal nonzero in one patch only. Corresponding token has highest norm. *Failure: stride/padding misconfigured.*
- **Gradient — Conv1d weights receive `.grad`.**
- **Shape — n_samples // stride = output token count.**
- **Numerical — no NaN with zero input.**

### 5.3 Spectrogram Tokenizer (5 tests)

- **Impulse — frequency band activation:** Energy in one frequency band. Active patch norms > 5× inactive. *Failure: Conv2d not spatially selective, or Perceiver pooling averaging.*
- **Impulse — temporal localization:** Burst at one time frame. Corresponding patch has highest norm.
- **Impulse — cross-frequency mixing (if Perceiver pooling):** Energy in two distant bands. All output tokens have norm > 0.01. *Failure: Perceiver queries only attend locally.*
- **Gradient — full chain receives `.grad`.**
- **Scale — 2× energy scaling → cos_sim < 0.99.** *Failure: energy information discarded.*

### 5.4 Video Tokenizer (tube-patch, 8 tests)

- **Impulse — spatial selectivity:** Bright region against a dark background activates the corresponding tube and not the far-corner tube. cos_sim(bright, black) < 0.9. *Failure: Conv3d patch projection not learning, or modality embedding dominates.*
- **Impulse — temporal_pe perturbation:** Perturbing only the temporal positional encoding (with frames fixed) changes the output. *Failure: temporal information collapsed inside the patch projection.* (Replaces the original input-vs-input motion test, which was insensitive at random init because a near-uniform softmax averaged keys across frames.)
- **Patch locality:** Perturbing one 12×12 patch must change the corresponding token but not the far-corner token. *Failure: patches are not disjoint.*
- **Missing-camera token:** A sample with `valid=False` routes to the learned `missing_token`; output is independent of pixel values.
- **Modality embedding distinctness:** Two different cameras have distinct modality embeddings.
- **Reconstruction pipeline:** Patch + ConvTranspose3d round-trip recovers pixel structure (training-loss decreases >50% in 100 steps with frozen backbone).
- **Gradient — flows from output through ConvTranspose3d, through backbone, through tube-patch Conv3d to pixels.**
- **Memory — full-size forward pass completes without OOM** (gated as GPU-only).

### 5.5 Actuator Tokenizer (4 tests)

- **Impulse — signal reaches tokens:** Active actuator tokens differ from zero tokens (diff norm > 1.0). *Critical: no LayerNorm after concatenation.*
- **Impulse — step/ramp/sinusoid produce different tokens.**
- **Gradient — all parameters receive `.grad`.**
- **Functional — different time offsets → different outputs.**

### 5.6 Shared Backbone (7 tests)

- **Impulse — self-attention spreads information:** One position set to random impulse (not constant!), others small-scale. All positions influenced after one layer (diff norm > 0.01). *Failure: no mixing, or residual dominates.*
- **Impulse — residual preserves impulse advantage:** Impulse position retains largest norm.
- **Impulse — step conditioning changes output:** step_index=0 vs 40. cos_sim < 0.95. *Failure: step embedding too weak.*
- **Impulse — progressive mixing:** CV of per-token norms decreases through layers.
- **Gradient — all layers receive `.grad` (attention, MLP, LayerNorm).**
- **Gradient — step embedding MLP receives `.grad`.**
- **Fixed-point — different inputs → different outputs (cos_sim < 0.99).**

### 5.7 Per-Modality Output Heads (3 tests per type)

- **Reconstruction — loss decreases >50% in 100 training steps** with frozen tokenizer/backbone.
- **Shape — output matches raw signal dimensions.**
- **Gradient — flows back to backbone tokens.**

### 5.8 Full Model End-to-End (4 tests)

- **Cross-modality transfer:** Input one modality only → all outputs non-zero (norm > 0.001).
- **Actuator conditioning:** Same diagnostics, different actuators → outputs differ.
- **Signal pathway:** Two inputs differing by 30%. cos_sim increase < 0.1 per stage, < 0.15 total.
- **Training resolves bottleneck:** After 50 steps, cos_sim of different inputs drops below 0.9.

### 5.9 Rollout Tests (7 tests)

- **Consecutive steps differ:** 10-step rollout, cos_sim < 0.99 at every step.
- **No explosion:** 80-step rollout, max norm < 100× step 1.
- **No collapse:** Min norm > 0.01× step 1.
- **Copy baseline (after training):** Model wins >80% at step 1, >60% at step 10. Deterministic targets only.
- **Fixed-point (after training):** 10-step rollout cos_sim < 0.99 at all steps.
- **model_cos_sim vs gt_cos_sim:** Gap < 0.05 (steps 1–10), < 0.10 (steps 10–80).
- **Actuator sensitivity in rollout:** Two actuator trajectories from same initial condition → predictions diverge by step 10.

### 5.10 Test Execution Summary

| Block | Tests | Runtime | Gate |
|---|---|---|---|
| Slow TS Tokenizer | 4 | <10s | Before integration |
| Fast TS Tokenizer | 5 | <10s | Before integration |
| Spectrogram Tokenizer | 5 | <30s | Before integration |
| Video Tokenizer | 8 | <60s | Before integration |
| Actuator Tokenizer | 4 | <10s | Before integration |
| Shared Backbone | 7 | <30s | Before integration |
| Output Heads | 3/type | <10s each | Before integration |
| Full Model E2E | 4 | <60s | Before Stage 1 |
| Rollout (random init) | 3 | <60s | Before Stage 1 |
| Rollout (after training) | 4 | <10min | Before cluster submission |
| **Total** | **~50** | **<15 min** | — |

**Gating rule:** No cluster job is submitted until all applicable tests pass. No exceptions.

## 6. Experimental Plan

### 6.0 Baseline Archival

Archive the prior AE-based Aurora codebase (autoencoder training scripts, foundation model architecture, training logs, and latent-continuity scatter plots) as a reproducibility snapshot of the approach that motivated the end-to-end design. The archive lives at `archive/ae_baseline/`.

### 6.1 Phase A: Baseline with Time Series Only (Weeks 1–3)

Implement the end-to-end architecture with slow and fast time series only (Thomson, CER, MSE, filterscopes). No spectrograms, no video.

**Milestones (strictly gated):**
- A1: All per-block verification tests pass (Sections 5.1, 5.2, 5.5, 5.6, 5.7, 5.8, 5.9 random-init subset)
- A2: Single-step MAE below the copy baseline for all modalities
- A3: model_cos_sim within 0.05 of gt_cos_sim at steps 1–10 (500 ms)
- A4: model_cos_sim within 0.10 of gt_cos_sim at steps 10–80 (4 seconds)
- A5: Validation rollout plots show tracking of real dynamics, not flatline or constant offset

**Phase A cannot be completed in one week.** Realistic pacing: Week 1 for implementation + A1, Week 2 for Stage 1 training + A2, Week 3 for Stages 2–3 + A3–A5. Attempting to compress this timeline risks repeating the cycle of submitting undertested runs and debugging on the cluster.

### 6.2 Phase BC: Add Spectrograms and Video (Weeks 3–6)

Joint training of spectrogram (ECE / BES / CO2) and video (tangtv) modalities on top of the Phase A backbone. Implemented as a single combined pipeline rather than two sequential phases: a single Stage 1 run (`train_bc_stage1.sh`), a single Stage 2 (delta) run (`train_bc_stage2.sh`), and a single Extended run carry both modality groups together. The TS modules and shared backbone warm-start from the Phase A checkpoint; spectrogram and video tokenizers and output heads init from scratch under the four-flag freeze schedule (TS + backbone held fixed for the first 5 000 steps, spectro + video train freely from step 0).

The combined approach is preferred over two sequential phases because spectrograms and video are independent diagnostic groups and there is no ordering constraint between them. Training jointly halves the warm-start chain length and shares the Phase-A reference checkpoint across both groups.

**Milestones:**
- BC1: Spectrogram and video tokenizers pass §5.3 / §5.4 tests
- BC2: BC-Stage 1 reaches single-step MAE within 5 % of Phase A on TS modalities; per-modality MAE for spectrograms and video below copy baseline
- BC3: BC-Stage 2 (delta) curriculum to K = 10 maintains TS rollout quality without degradation; per-step displacement losses converge for spectrograms and video
- BC4: BC Stage 2 Extended free-rollout to K = 80 stable on all three modality groups
- BC5: Cross-modal coupling verified (NBI → correlated Thomson, spectrogram, and video responses)

### 6.3 Phase D: Actuator Conditioning Evaluation (Weeks 5–6)

- Divergent predictions for different actuator trajectories from the same initial condition
- Comparison against TRANSP for selected scenarios
- Latency measurement for real-time control feasibility (<50 ms for 80-step rollout)

### 6.4 Phase E: Cross-Machine Transfer (Weeks 6–8, exploratory)

Freeze backbone, train new tokenizers on target device diagnostics (EAST, KSTAR). Evaluate zero-shot and few-shot prediction quality.

## 7. Evaluation Metrics

| Metric | What it measures | Target |
|---|---|---|
| MAE (per modality) | Pointwise prediction accuracy | Below copy baseline |
| model_cos_sim vs gt_cos_sim | Per-step dynamics fidelity | Gap < 0.05 (steps 1–10), < 0.10 (steps 10–80) |
| pred_delta / tgt_delta | Displacement magnitude accuracy | Ratio 0.8–1.2 |
| Copy baseline win rate | Model vs trivial copy | >80% at step 1, >60% at step 10 |
| Rollout stability | No explosion or collapse over 80 steps | Norm ratio < 10× |
| Actuator sensitivity | Predictions change with actuator commands | Verified qualitatively and quantitatively |
| Inference latency | Wall-clock time for 80-step rollout | <50 ms on single GPU |

## 8. Risk Assessment and Mitigations

**Risk 1: 80-step rollout error compounding.**
Mitigation: Pushforward + replay buffer train on self-generated states. LoRA preserves single-step quality. Curriculum K=10→80. Fallback: predict 2–5 windows per step (100–250 ms/step), reducing to 16–40 steps.

**Risk 2: 5 samples per channel (slow TS) is too few.**
Mitigation: Linear(5, 256) is an expansion, not compression. Fallback: extend to 100 ms (10 samples) at 2× step count reduction.

**Risk 3: Token count exceeds self-attention budget.**
Mitigation: Full BC configuration produces ~1180 tokens — still feasible for standard attention at d_model=256 (per-step cost scales ~2.1× over TS-only because the FFN dominates per-layer compute). Add Perceiver compression only if further modalities push the count substantially higher.

**Risk 4: High-dimensional output heads (spectrogram, video) cannot reconstruct.**
Mitigation: Output heads are for loss only, not rollout. Approximate reconstruction provides gradient signal. Increase tokens or use U-Net decoder if needed.

**Risk 5: Training instability in Stage 3.**
Mitigation: Well-trained Stage 2 checkpoint. Small LoRA rank (16). Monitor buffer quality. Increase refresh rate if degraded.

**Risk 6: No cross-modal coupling emerges.**
Mitigation: Ablation: mask one modality, check others degrade. Increase backbone depth or add explicit cross-attention if needed.

**Risk 7: Insufficient data for end-to-end tokenizer learning.**
Mitigation: ~500 shots → ~500k chunks (50 ms, 10 ms stride). Fallback: pretrain tokenizers with reconstruction for 10 epochs before switching to end-to-end. This provides initialization without the permanent geometric distortion of fully converged AEs.

## 9. Computational Requirements

Hardware: 1× A100 40 GB per training run unless noted. Step times are realised numbers from production launchers; `wall` columns assume continuous occupancy and include 24 h-wall SLURM chaining via auto-resume.

| Phase | Stage | Steps | Batch | s/step | Wall |
|---|---|---|---|---|---|
| A | Stage 1 (single-step, TS only, 398 tokens) | 336 000 | 256 | 0.97 | ~3.7 days |
| A | Stage 2 (delta, K = 1…10) | 322 000 | 64 | ~2 | ~7.5 days |
| A | Stage 2 Extended (free-rollout K = 80) | ~50 000 | 32 | ~15 | ~9 days |
| BC | Stage 1 (TS + spectro + video, 1180 tokens) | 672 000 | 128 | ~2 (×2.1 over A) | ~16 days |
| BC | Stage 2 (delta, K = 1…10, multimodal) | 322 000 | 64 | ~4 | ~15 days |

Approximate totals:

- Phase A pipeline (Stage 1 → Stage 2 → Extended): **~20 A100-days**.
- Phase BC pipeline (Stage 1 → Stage 2 → Extended once wired): **~35–45 A100-days**, dominated by the 1180-token attention cost relative to Phase A's 398.
- Phase BC step-time scaling is below the 8.8× theoretical attention ceiling at d_model = 256 because the FFN (linear in N) is the per-layer compute bottleneck; the realised slowdown over Phase A is closer to 2× per step.
- Estimated experiments to convergence: 3–5 per phase including failed runs and hyperparameter sweeps.
- Total budget: **~80–120 A100-days** for the full Phase A + Phase BC programme through 80-step rollout, plus Phase D / E which inherit the converged Phase BC checkpoint and require evaluation runs only.

## 10. References

1. C. Bodnar et al., "A foundation model for the Earth system," Nature, 2025.
2. A. Jaegle et al., "Perceiver IO: A general architecture for structured inputs & outputs," ICLR, 2022.
3. E.J. Hu et al., "LoRA: Low-rank adaptation of large language models," ICLR, 2022.
4. A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," ICLR, 2021.
5. Y. Gong et al., "AST: Audio spectrogram transformer," Interspeech, 2021.
6. A. Arnab et al., "ViViT: A video vision transformer," ICCV, 2021.