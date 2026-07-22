# IGNITE — Design Note

**Status:** design-only, no implementation. This is the consolidated architecture and
decision log for a *fresh-start* Genie-style world model for the tokamak. It supersedes
the rollout-native / descriptor / continuous-head / independent-marginal approach
(see `analysis/mode_audit/EXPERIMENTS.md` and the mode-audit conclusion for why).

Date of decisions: 2026-07-14.

---

## 1. Object

A **controllable plasma simulator**:

- **Inputs:** an initial plasma state (`K₀` real seed frames) + a target **actuator
  trajectory** (80 frames × 70 channels), known for the whole horizon.
- **Output:** the predicted plasma state (all diagnostics) for **80 frames**, produced by
  autoregressive rollout.
- Change the actuator trajectory → different predicted evolution. That *is* the
  horizon-controllability claim, delivered by construction.

The model is Genie's architecture adapted to the tokamak. It is **not** a merge of Genie
and FAITH, and it reuses **no** FAITH model code (see §7).

---

## 2. Terminology — the three time units (do not conflate)

| Unit | Size | Role |
|---|---|---|
| **STFT frame** | 1024-sample Hann window, hop 256 → one every **0.512 ms** | Raw spectrogram time resolution. ~**98 per 50 ms window** (0.05 × 500 kHz / 256). Sub-frame *content*, not a stepping axis. |
| **Codec token-time-step** | patch_t = 32 STFT frames ≈ **16.4 ms** | After (64 freq × 32 time) FSQ patching → **3 time-steps × 8 freq-patches = 24 tokens per spectro modality per window**. |
| **Frame = window** | **50 ms** (`chunk_duration_s = 0.05`) | **The world-model frame = one plasma state.** The dynamics advances one frame per AR step. |

**FRAME = one 50 ms window = one plasma state.** `80 frames = 80 × 50 ms = 4 s horizon`.
The ~98 STFT frames within a window are sub-frame content the codec compresses — they are
**not** the world-model's frame. (Data pipeline: `step_size_s = 0.01` = 10 ms overlap
between sampled windows, for diverse training starts — not a prediction stride.)

---

## 3. Architecture overview — two decoupled phases

```
                 ┌── Phase A (frozen after gate) ──┐        ┌──── Phase B ────┐
 raw signal ──► encoder ──► FSQ codes ──────────────────► ST-transformer ──► per-modality
 (on-the-fly    (stats-first,                 ▲            (MaskGIT dynamics)   code logits
  STFT)          shift-invariant)              │ closed         ▲                   │
                                     frozen enc/dec code space  │ additive          ▼
 decoded frame ◄── decoder (adversarial) ◄──── predicted codes  │ actuator     frozen decoder
                                                                └── 70-ch/frame      │
                                                                                     ▼
                                                                            predicted plasma state
```

- **Phase A** = statistics-first FSQ codecs (the tokenizer). Trained first, **frozen**.
- **Phase B** = MaskGIT dynamics over the frozen codes. The **closed code space** means
  the frozen encoder/decoder bracket the dynamics at both ends; the dynamics is a pure
  token-sequence model with **no trainable tokenizer**.
- The two phases are **fully decoupled** (see §6).

---

## 4. Phase A — statistics-first codecs

### 4.1 Principle: encode the *statistic*, not the *realization*

Each spectrogram window splits into:
- **Realization** (nuisance, unpredictable): STFT phase / speckle, sub-window alignment,
  instantaneous jitter. A 0.5 ms shift scrambles ~74 % of the old codec's codes.
- **Statistic** (physics, predictable): which frequencies carry power (mode presence),
  amplitude envelope, bandwidth, drift/growth across the window.

The old codec encoded both → codes inherited the realization's unpredictability, and its
pixel-MAE objective drove the decoder to the mean. Phase A removes **both** failures with
two independent moves:

1. **Codes carry only the statistic** → predictable (passes the oracle gate).
2. **Decoder is generative** → it hallucinates a plausible realization from statistic-codes
   → reconstructions stay sharp and mode-bearing, with no pixel-MAE forcing the mean.

Neither works alone; together → sharp *and* predictable.

### 4.2 Components

- **Bottleneck:** FSQ (discrete; required for MaskGIT downstream). Use
  `vector-quantize-pytorch`'s `FSQ`.
- **Encoder → statistic:** ST-transformer (ST-ViViT lineage), built on `x-transformers`
  primitives. Target = **full-resolution log-power** spectrogram (no smoothing).
- **Invariance = shift-consistency loss:** `‖enc(x) − enc(shift_δ x)‖²` on the **pre-FSQ
  continuous features** (avoids discrete matching). The nuisance pair is generated from the
  **raw signal**, which the data loader already provides on-the-fly (HDF5 → resample → STFT,
  no precomputed cache): take the same shot over `[t₀, t₀+50 ms]` and `[t₀+δ, t₀+50 ms+δ]`,
  STFT both. **δ ~ U[~0.1, 2] ms** (fraction of one STFT hop up to ~1 STFT window; ≪ the
  16 ms codec time-patch, so mode content and within-frame dynamics are untouched).
  Optional mild secondary nuisance: ±few-% multiplicative amplitude jitter + small noise
  floor. **No** frequency transform (freq is the statistic); **no** amplitude-invariance
  beyond the jitter (mode amplitude is signal).
- **Decoder → realization:** generative, **adversarial + small pixel anchor**. The pixel
  anchor buys optimization stability but partially fights invariance, so its weight
  **λ_pix is gated by the stability metric** — raised only while stability ≥ 0.8.
- **Discriminator:** multi-scale, **frequency-aware PatchGAN** (freq-PE channel — the freq
  axis is not translation-invariant, mode@50 kHz ≠ 150 kHz); hinge loss + R1/LeCAM gradient
  penalty; unconditional (real GT frame vs decoded). Precedent: VQ-GAN/MagViT tokenizers +
  vocoder-GANs (HiFi-GAN/MelGAN).

### 4.3 Per-modality

- **Spectrograms (ece, co2, bes, mhr)** — the template; the failing modality. Full design
  above. 24 tokens/window/modality.
- **Fast-TS (filterscopes / ELMs)** — hardest. Statistic = **ELM activity envelope**
  (rate/amplitude), *not* spike timing; the decoder hallucinates plausible spikes.
- **Video (tangtv upper/lower)** — closest to Genie-native (smooth frames); mostly just the
  no-strong-pixel-MSE / generative-decoder move. Two separate up/lower divertor codecs.
- **Slow-TS** — smooth profiles; lightest touch.

### 4.4 Acceptance gate (hard; before Phase B)

All on held-out shots, stratified stable/transition, via the validated detector
(`analysis/mode_audit/`):
- **Stability ≥ 0.80** — fraction of codes unchanged under the nuisance transform.
- **Persistence ≥ 0.5** on steady segments (floor 0.10 = old-codec churn), and measurably
  lower on transitions.
- **Forecastability probe** — a *cheap* predictor (per-code Markov / tiny MLP, not the real
  dynamics) beats persistence at next-frame code **distribution** on transition windows.
  Margin calibrated to beat the old codec's ece transition Δ = +0.075.
- **Mode-bearing decode** — detector F1 ≥ ~0.8, decoded mode distribution matches GT,
  high-freq gradient energy not collapsed.

Stability/persistence are mandate numbers; the forecastability margin and decode-F1 are
build-time calibration targets (floor = beat old codec, ceiling = codec's own recon).
**Codes that fail the gate do not proceed** — with gate-only coupling (§6) there is no
downstream pull toward predictability, so Phase-A design quality matters up front.

---

## 5. Phase B — MaskGIT dynamics

### 5.1 Frame token layout

One frame = all modalities' Phase-A codes concatenated: spectro 24 tokens each × 4 = 96,
plus the video (up/lower) and TS codec tokens. Each token carries **modality-type** +
**within-modality position** (freq-patch, time-patch) + **frame index** (temporal). This
whole multi-modal token set *is* the plasma state at that step. **Per-modality vocab heads**
(each over its own FSQ codebook).

### 5.2 Backbone — factorized ST-transformer over the frozen codes

Fresh code composing `x-transformers` attention primitives into the Genie/ViViT structure,
fed **code embeddings** (closed code space):
- **Spatial attention** — within a frame, over the whole multi-modal token set (cross-modal
  + within-modal mixing of one 50 ms state).
- **Causal temporal attention** — across frames; frame *t* attends only to ≤ *t*.
- FFN, residuals.

### 5.3 Prediction — MaskGIT (the reason for the pivot)

- **Training:** mask a random fraction of a frame's tokens; predict them by categorical CE
  over the per-modality vocab, conditioned on visible tokens + past frames + actuator.
- **Inference:** iterative confidence-based unmasking → **joint** decode of each next
  frame's tokens. This captures the joint distribution (coherent modes across tokens) that
  the old single-pass independent-marginal head structurally could not.

### 5.4 Actuator conditioning (the controllability lever)

Continuous 70-ch `actuator_t` (7 modalities: pin 8, beam_voltage 8, tin 8, ech_power 12,
gas_flow 11, gas_raw 11, rmp 12) → linear embedding → **additive** to the target frame's
tokens (Genie's validated additive-conditioning finding), **causal**: frame *t+1* is
conditioned on actuators ≤ *t+1*, never on future actuators.

### 5.5 Rollout

Seed with `K₀` real frames → MaskGIT-generate frame *t+1* conditioned on all past frames +
`actuator_{t+1}` → **commit the discrete codes** → append → repeat to frame 80. Discrete
commit in a closed space = clean feedback, no decode→re-tokenize round-trip.

### 5.6 Training

- **Base:** single-step MaskGIT, teacher-forced on real past codes.
- **Drift mitigation:** **light scheduled-sampling from the start** — during training,
  occasionally feed the model's own sampled codes for a few steps so it learns to consume
  its own outputs (80 steps / 4 s is a long horizon; Genie itself degrades over long
  rollouts). Ramp schedule = build-time.
- **Seed:** `K₀` = **longer context, ~10–20 frames (0.5–1 s real)** — strong initial
  context for rollout stability.
- **Horizon accounting:** 80 frames = *predicted* (4 s) beyond the seed → total temporal
  context ~90–100 frames.

**Honest trade:** both the longer seed and scheduled sampling favor rollout stability over
minimalism → the "controllability from a *minimal* state" framing weakens to "from ~0.5–1 s
context, predict 4 s of controlled evolution" (still a strong claim).

### 5.7 Rollout gate

Reuse the descriptor / detector infrastructure to measure, on held-out shots: does the
80-step rollout preserve mode structure and track the actuator-driven evolution? Does a
counterfactual actuator trajectory change the prediction (controllability)? If the 4 s
rollout degrades, the scheduled-sampling depth is the first knob.

---

## 6. Training coupling — fully decoupled (Genie staging)

- **Optimization: independent.** Codec **frozen** during Phase B; no dynamics gradient into
  the codec; two separate losses; never jointly backpropagated.
- **Data / order: dependent.** Phase B trains on Phase A's frozen codes; A must **pass the
  oracle gate** before B starts. One-way A → B.
- **Forecastability is gate-only**, not a training gradient (chosen over a forecastability
  aux-objective, and over joint fine-tuning — the latter risks degenerate/collapsed codes).
- **Concurrency:** A before B. The per-modality codecs *within* A are mutually independent →
  parallelizable; B needs the full set done + gated.

---

## 7. Reuse boundary (hard rule)

- **Reuse — allowed:**
  - the **data-loading pipeline** (`data_loader.py` — raw load + on-the-fly STFT);
  - the **new Phase-A codecs** (new code);
  - **external validated packages**: `vector-quantize-pytorch` (FSQ bottleneck),
    `x-transformers` (attention primitives). External libraries are not FAITH code.
- **Written fresh — everything model-side:** the codec encoder/decoder/discriminator, the
  ST-transformer backbone, the MaskGIT dynamics + sampler, the per-modality code heads, the
  trainer.
- **Forbidden:** reusing / porting / subclassing **any** existing FAITH model code —
  `MultiWindowBackbone`, `SharedBackbone`, `BackboneBlock`, `TemporalAttention`,
  `output_heads.py`, `E2EFoundationModel`, `train_e2e_stage1.py` model/loss logic. This is
  the sanctioned exception to the usual "extend, don't create" rule.

**No validated whole-Genie package exists** (DeepMind released neither code nor weights;
`genie2-pytorch` is wip, `open-genie` incomplete). The Genie repos are **design references
only**; the validated, installable pieces are the *components* above.

---

## 8. External-package map

| Package | Phase A | Phase B |
|---|---|---|
| `vector-quantize-pytorch` | FSQ bottleneck (codec quantizer) | — |
| `x-transformers` | ST encoder/decoder blocks | ST-transformer attention (spatial + causal-temporal) |
| *(fresh)* | consistency loss, freq-aware PatchGAN discriminator, log-power target | ST factorization, MaskGIT objective + sampler, per-modality heads, actuator conditioning, scheduled-sampling loop, multimodal frame layout |

"Use where possible" — fall back to custom where a package doesn't fit cleanly (the ST
factorization and the MaskGIT sampler are ours).

---

## 9. What this retires

Continuous mean/flow heads · the independent-marginal code head · the descriptor head · the
K-rollout curriculum + drift penalty + teacher-forcing anneal · FiLM · decode→re-tokenize
feedback · warp/persistence anchors. All replaced by: closed code space + statistics-first
codecs + MaskGIT joint decode + additive actuator conditioning.

---

## 10. Open / build-time items

Not design forks — calibration/spec-out at build time:
- Phase A: exact δ distribution, discriminator depth/scales, oracle-gate numeric thresholds,
  per-modality statistic definitions (esp. fast-TS ELM envelope).
- Phase B: per-frame token budget (from video/TS codec configs), model size
  (d_model / depth), positional-encoding choice (rotary vs learned), training clip length
  (must cover seed + rollout), scheduled-sampling ramp + depth, exact `K₀`.

---

## 11. Decision log (2026-07-14)

1. Fresh start, Genie dynamics; initial plasma state + 80-frame (4 s) actuator trajectory →
   AR rollout. Not a merge.
2. Frame = 50 ms window (user-corrected). 80 frames = 4 s.
3. Codecs: **new statistics-first codecs** (not reuse of existing FSQ codecs).
4. Scope: new modules in the FAITH repo; reuse only data loader + new codecs.
5. Coupling: **gate-only**, fully decoupled (over aux-objective / joint fine-tune).
6. Phase-A invariance: **consistency-loss-only**, via **raw-signal δ-shift + re-STFT**
   (δ ~ U[0.1, 2] ms).
7. Phase-A decoder: **adversarial + small (stability-gated) pixel anchor**; multi-scale
   freq-aware PatchGAN; oracle-gate thresholds accepted.
8. Phase-B drift mitigation: **light scheduled-sampling from the start**.
9. Phase-B seed: **~10–20 frames** of real context.
10. Reuse: **zero FAITH model code**; external `vector-quantize-pytorch` + `x-transformers`
    allowed.
