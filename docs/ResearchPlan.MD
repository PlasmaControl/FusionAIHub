# Research Plan: End-to-End Foundation Model for Multi-Modal Tokamak Plasma Prediction

**PI:** P. Schramowski, E. Kolemen
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

This work makes four contributions:

**C1. First multi-modal tokamak foundation model operating on raw heterogeneous signals.** The model simultaneously ingests time series (100 Hz–10 kHz), spectrograms (500 kHz), and video sequences, producing predictions across all modalities conditioned on actuator commands. No prior work handles this heterogeneity in a unified predictive framework.

**C2. Actuator-conditioned prediction for control.** Given a proposed actuator trajectory (beam injection, ECH power, gas fueling, RMP coils), the model predicts the plasma response across all diagnostics. This enables "what-if" scenario evaluation orders of magnitude faster than physics-based simulations, suitable for real-time model-predictive control and between-shot planning.

**C3. Empirical demonstration that reconstruction-trained latent spaces are geometrically incompatible with temporal prediction, and that end-to-end training resolves this.** We provide a diagnostic framework (signal-to-latent cosine similarity correlation) that quantifies the incompatibility, show that it persists across autoencoder architectures and regularization strategies, and demonstrate that end-to-end tokenizers trained under the prediction objective produce representations where the incompatibility is absent. The comparison uses the AE-based Aurora-style architecture (archived codebase) as controlled baseline.

**C4. Comprehensive verification methodology for autoregressive prediction architectures.** We present an impulse-based test suite (~50 tests) that verifies signal propagation through every architectural component before training begins, and diagnostic metrics (delta-ratio, per-step cosine similarity, per-stage signal pathway analysis) that localize failure modes to specific modules during training. This methodology applies beyond the tokamak domain to any autoregressive prediction system operating on heterogeneous inputs.

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
|---|---|---|---|---|---|
| Slow time series | Thomson (core + tangential density, temperature), CER (Ti, rotation), MSE | 100 Hz | 5 samples/ch | Linear per channel | ~90 total (6 modalities × ~15 ch) |
| Fast time series | Filterscopes | 10 kHz | 500 samples/ch | Conv1d patching (stride 50) | ~80 (8 ch × 10 tokens) |
| Spectrogram | BES, ECE | 500 kHz | ~194 frames × 513 freq bins/ch (STFT n_fft=1024, hop_length=256) | Conv2d (k=64, s=64) patches | ~240 (30 time × 8 freq) |
| Video | Fast camera | 1–10 kHz | 50–500 frames | Spatial CNN + temporal patching + Perceiver pooling | ~16 |
| Actuators | NBI, ECH, gas, RMP | varies | 50 ms | Conv1d patching | ~18 (6 groups × 3) |
| | | | | **Total (full config):** | **~444** |

With ~200–450 total tokens depending on configuration, standard self-attention (O(N²)) is feasible without Perceiver compression. At N=450 with d_model=256 and 8 heads, the per-layer attention cost is ~165M FLOPs — still trivial on a modern GPU. If future modality additions push the count beyond 700, a Perceiver compression stage after tokenization but before the backbone can reduce it.

Each tokenizer adds a learned modality embedding and positional encoding. All tokenizer weights are trained end-to-end with the backbone.

### 3.4 Shared Backbone

Standard Transformer encoder with pre-norm (LayerNorm before attention, not after). Eight self-attention layers, d_model=256, 8 heads, MLP ratio 4. All diagnostic and actuator tokens attend to each other — cross-diagnostic coupling is learned implicitly through self-attention.

Step conditioning: Fourier features of the rollout step index and absolute time offset, projected through a 2-layer MLP, added to all tokens. This allows the backbone to modulate predictions based on rollout depth.

### 3.5 Per-Modality Output Heads

Each modality has an output head that projects backbone tokens back to the raw signal space. These are approximate inverses of the tokenizers (Linear for slow TS, ConvTranspose1d for fast TS, ConvTranspose2d for spectrograms, spatial decoder CNN for video). Output heads fire only for computing the training loss against ground truth raw signals. During rollout, backbone tokens pass directly to the next step without going through the output heads.

### 3.6 Rollout Architecture

```
Step 0: tokenize(raw_signals) → tokens₀
Step k: backbone(tokensₖ₋₁, actuator_tokensₖ, step=k) → tokensₖ
        output_heads(tokensₖ) → raw_pred  (for loss only, not fed back)
        tokensₖ flows directly to step k+1
```

The 80-step rollout (4 seconds at 50 ms resolution) operates entirely in token space. No re-tokenization between steps. This is the key architectural choice that eliminates the encode–decode roundtrip information loss observed in the AE-based architecture, where the Perceiver encoder–decoder cycle erased temporal variation within 3 rollout steps.

## 4. Training Procedure

### 4.1 Stage 1: Single-Step Pretraining

**Objective:** Learn tokenizers, backbone, and output heads for one-step (50 ms) prediction.

- Loss: MAE in raw signal space, per-modality, all normalized to unit variance (precomputed statistics)
- Data: All available DIII-D shots, chunked into consecutive 50 ms windows with 10 ms step size
- Duration: Until validation MAE plateaus (~50–100 epochs)
- Full weight updates on all parameters

### 4.2 Stage 2: Short Rollout Fine-Tuning

**Objective:** Teach the model to handle its own outputs as input for short horizons.

- Rollout curriculum: K=1 → K=10 steps (50 ms → 500 ms) over 30 epochs
- Full backpropagation through all K steps
- Full weight updates

### 4.3 Stage 3: Long Rollout Fine-Tuning (Pushforward + Replay + LoRA)

**Objective:** Stable 80-step (4-second) autoregressive prediction.

**Pushforward trick (Bodnar et al., 2025):** Run K−1 rollout steps with no gradient. Backpropagate only through the final step. Memory cost equals single-step training regardless of K.

**Replay buffer:** In-memory buffer stores ground truth and model-generated states. At each training step: sample from buffer → forward one step → loss → add prediction back to buffer. Periodically refresh with ground truth. This ensures the model trains on the distribution of states it actually produces during inference.

**LoRA (Hu et al., 2022):** Freeze all base weights from Stages 1–2. Attach rank-16 adapters to backbone attention layers. Only LoRA parameters are updated. This preserves the single-step prediction quality while adapting the model for multi-step dynamics.

- Rollout curriculum: K=10 → K=80 steps
- Replay buffer size: 50,000 samples
- Buffer refresh period: every 50 training steps

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

### 5.4 Video Tokenizer (5 tests)

- **Impulse — spatial selectivity:** Bright square in one corner. cos_sim(bright, black) < 0.9. *Failure: spatial CNN not learning.*
- **Impulse — temporal localization:** 5 ms flash. Flash patch has highest norm.
- **Impulse — motion detection:** Static vs moving object. cos_sim < 0.95. *Failure: temporal info lost in spatial compression.*
- **Gradient — flows from output through temporal patching through spatial CNN to pixels.**
- **Memory — full-size forward pass completes without OOM.**

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
| Video Tokenizer | 5 | <60s | Before integration |
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

Archive the current AE-based Aurora codebase (autoencoder training scripts, foundation model architecture, training logs, and the latent continuity scatter plots) as the controlled baseline for C3. The comparison between AE-based and end-to-end architectures requires both codebases to be reproducible.

### 6.1 Phase A: Baseline with Time Series Only (Weeks 1–3)

Implement the end-to-end architecture with slow and fast time series only (Thomson, CER, MSE, filterscopes). No spectrograms, no video.

**Milestones (strictly gated):**
- A1: All per-block verification tests pass (Sections 5.1, 5.2, 5.5, 5.6, 5.7, 5.8, 5.9 random-init subset)
- A2: Single-step MAE below the copy baseline for all modalities
- A3: model_cos_sim within 0.05 of gt_cos_sim at steps 1–10 (500 ms)
- A4: model_cos_sim within 0.10 of gt_cos_sim at steps 10–80 (4 seconds)
- A5: Validation rollout plots show tracking of real dynamics, not flatline or constant offset

**Phase A cannot be completed in one week.** Realistic pacing: Week 1 for implementation + A1, Week 2 for Stage 1 training + A2, Week 3 for Stages 2–3 + A3–A5. Attempting to compress this timeline risks repeating the cycle of submitting undertested runs and debugging on the cluster.

### 6.2 Phase B: Add Spectrograms (Weeks 3–4)

Add spectrogram tokenizer for BES or ECE data. Transfer backbone and time series tokenizers from Phase A checkpoint.

**Milestones:**
- B1: Spectrogram tokenizer passes all Section 5.3 tests
- B2: Cross-modal coupling verified (NBI → correlated Thomson + BES responses)
- B3: Time series rollout quality does not degrade

### 6.3 Phase C: Add Video (Weeks 4–5)

Add video tokenizer for fast camera data. Same transfer strategy.

**Milestones:**
- C1: Video tokenizer passes all Section 5.4 tests (including memory)
- C2: Edge instabilities in video correlate with filterscope signals
- C3: Full multi-modal 80-step rollout stable

### 6.4 Phase D: Actuator Conditioning Evaluation (Weeks 5–6)

- Divergent predictions for different actuator trajectories from the same initial condition
- Comparison against TRANSP for selected scenarios
- Latency measurement for real-time control feasibility (<50 ms for 80-step rollout)

### 6.5 Phase E: Cross-Machine Transfer (Weeks 6–8, exploratory)

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
| Latent continuity (C3) | Spearman(signal_cos, token_cos) | >0.5 for end-to-end tokenizers vs ≤−0.1 for AE |

## 8. Risk Assessment and Mitigations

**Risk 1: 80-step rollout error compounding.**
Mitigation: Pushforward + replay buffer train on self-generated states. LoRA preserves single-step quality. Curriculum K=10→80. Fallback: predict 2–5 windows per step (100–250 ms/step), reducing to 16–40 steps.

**Risk 2: 5 samples per channel (slow TS) is too few.**
Mitigation: Linear(5, 256) is an expansion, not compression. Fallback: extend to 100 ms (10 samples) at 2× step count reduction.

**Risk 3: Token count exceeds self-attention budget.**
Mitigation: Full config produces ~324 tokens — feasible for standard attention. Add Perceiver compression only if >500 tokens from additional modalities.

**Risk 4: High-dimensional output heads (spectrogram, video) cannot reconstruct.**
Mitigation: Output heads are for loss only, not rollout. Approximate reconstruction provides gradient signal. Increase tokens or use U-Net decoder if needed.

**Risk 5: Training instability in Stage 3.**
Mitigation: Well-trained Stage 2 checkpoint. Small LoRA rank (16). Monitor buffer quality. Increase refresh rate if degraded.

**Risk 6: No cross-modal coupling emerges.**
Mitigation: Ablation: mask one modality, check others degrade. Increase backbone depth or add explicit cross-attention if needed.

**Risk 7: Insufficient data for end-to-end tokenizer learning.**
Mitigation: ~500 shots → ~500k chunks (50 ms, 10 ms stride). Fallback: pretrain tokenizers with reconstruction for 10 epochs before switching to end-to-end. This provides initialization without the permanent geometric distortion of fully converged AEs.

## 9. Computational Requirements

- Stage 1 (~100 epochs, ~500k chunks): ~24 hours on 1× A100
- Stage 2 (~30 epochs): ~12 hours on 1× A100
- Stage 3 (~50 epochs with replay): ~48 hours on 1× A100
- Total per experiment: ~3–4 days on 1× A100
- Estimated experiments to convergence: 5–10
- Total budget: 15–40 A100-days

## 10. References

1. C. Bodnar et al., "A foundation model for the Earth system," Nature, 2025.
2. A. Jaegle et al., "Perceiver IO: A general architecture for structured inputs & outputs," ICLR, 2022.
3. E.J. Hu et al., "LoRA: Low-rank adaptation of large language models," ICLR, 2022.
4. A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," ICLR, 2021.
5. Y. Gong et al., "AST: Audio spectrogram transformer," Interspeech, 2021.
6. A. Arnab et al., "ViViT: A video vision transformer," ICCV, 2021.