# Tokamak E2E World Model — Full Architecture (verified from code, 2026-07-13)

One multimodal foundation model: given **one 50 ms window** of all diagnostics +
actuators, predict the **next window** of every diagnostic. Pipeline:
per-modality **tokenizers (encoders)** → **shared Transformer backbone** →
per-modality **output heads (decoders)**. Spectrograms are the modality under
active work; **production predicts them as discrete FSQ codes** (§5).

All shapes below are the production config: `d_model=1024`, 50 ms window,
`SLOW_FS=100 Hz`, `FAST_FS=10 kHz`, STFT `n_fft=1024, hop=256, fs=500 kHz`
(→ 512 freq bins, ~1953 frames/s, 98 frames/50 ms).

---

## 0. Top-level flow

```
 INPUT WINDOW (t…t+50ms)                                          TARGET (t+50…t+100ms)
 ───────────────────────                                          ────────────────────
 slow-TS, fast-TS, spectro, video, actuator                                 ▲
        │  per-modality TOKENIZERS (§2)                                      │ loss (§6)
        ▼                                                                    │
   concat tokens (B, ΣN≈1675, d) + StepConditioning                         │
        ▼                                                                    │
   SHARED TRANSFORMER BACKBONE  (§3)  — 48× pre-norm blocks, full attention  │
        ▼  out_tokens (B, ΣN, d)                                            │
        │  per-modality OUTPUT HEADS (§4) — continuous | FSQ-code | flow     │
        ▼                                                                    │
   predictions[name]  ── spectro FORECAST ANCHOR (§6) ──────────────────────┘
   token order: [slow_ts | fast_ts | spectrogram | video | actuators]  (actuators condition only)
```

Two spectro decode paths coexist (flag-selected):
- **Continuous / generative head** decodes tokens → spectrogram directly (§4.3).
- **FSQ (production):** predict a *frozen codec's discrete codes*, decode through
  the frozen codec (§5).

---

## 1. Modalities & token budget (50 ms window)

| Group | Modalities (channels) | tokens each | Σ |
|---|---|---|---|
| slow-TS | ts_core_density(44), ts_core_temp(44), ts_tangential_density(10), ts_tangential_temp(10), cer_ti(48), cer_rot(48), mse(69) | = n_channels | **273** |
| fast-TS | filterscopes(8) | 8·10 = 80 | **80** |
| spectro | ece(40), co2(4), bes(16), mhr(6) | (F/F_p)·(T/T_p) | **672** (192+96+192+192 @ ece/bes/mhr 32×8, co2 64×8) |
| video | tangtv_lower(2), tangtv_upper(2) | 1·10·30 = 300 | **600** |
| actuator | pin,beam_voltage,tin,ech_×4,gas_×2,rmp | 5 each | **50** |
| | | | **ΣN ≈ 1675** |

---

## 2. ENCODERS (tokenizers) — per-module flow charts

Each → `(B, n_tok, d_model)` + learned **modality embedding** + **positional
embedding**; absent-able modalities carry a learned **`missing_token`**.

```
① SlowTimeSeriesTokenizer     (Thomson/CER/MSE, 100 Hz)
   x (B, C, 5)                     # 5 samples/channel = 50 ms @ 100 Hz  (tiny!)
     │  Linear(5 → d)              # ONE shared weight, ALL channels; no conv/stem/refine
     ▼ (B, C, d)
     + channel_pos(C,d) + modality_embed(d)          [std .02]
   → (B, C, d)                     # 1 token PER CHANNEL

② FastTimeSeriesTokenizer     (filterscopes, 10 kHz)
   x (B, 8, 500)
     │ reshape (B·8,1,500)
     │ STEM: Conv1d(1→64,k3,p1)→GELU→Conv1d(64→64,k3,p1)→GELU     (B·8,64,500)
     │ Conv1d(64→d, k=s=50)                                        (B·8,10,d)
     │ reshape (B,8,10,d) + patch_pos(10,d)+channel_pos(8,d)+modality(d)
     │ reshape (B,80,d);  4× refine  x+[LN→Lin(d→4d)→GELU→Lin(4d→d)]
   → (B, 80, d)

③ SpectrogramTokenizer        (ece/co2/bes/mhr; here ece 40ch, patch 32×8)
   x (B, 40, 512, 98)              # |STFT| magnitude
     │ [freq_stem, OPT-IN, zero-init residual]:
     │    xᵀ(B,40,98,512) → Lin(512→128)→GELU→Lin(128→512) → x += (mix ALL freqs)
     │ truncate T 98→96 (mult of patch_t=8)
     │ Conv2d(40→d, k=s=(32,8))                                    (B,d,16,12)
     │ flatten→transpose (B,192,d) + spatial_pe(192,d)+modality(d)
     │ 12× refine  x+[LN→Lin(d→4d)→GELU→Lin(4d→d)]
   → (B, 192, d)                   # 16 freq-patch × 12 time-patch   (missing_token if absent)

④ VideoTokenizer              (tangtv_lower/upper, 2ch, 3 frames, 120×360)
   x (B, 2, 3, 120, 360)
     │ Conv3d(2→d, k=s=(3,12,12))   tube-patch                     (B,d,1,10,30)
     │ flatten→transpose (B,300,d) + spatial_pe(300,d)+modality_emb(d)
   → (B, 300, d)                   # 1t×10h×30w   (missing_token if camera absent)

⑤ ActuatorTokenizer           (e.g. ech_power 12ch, 10 kHz)
   x (B, C, 500)
     │ Conv1d(C→d, k=s=100)         channel-MIXING (in=C)          (B,d,5)
     │ transpose (B,5,d) + patch_pos(5,d)+modality(d)   [NO LayerNorm]
   → (B, 5, d)                      # CONDITION only — never decoded
```

---

## 3. BACKBONE — `SharedBackbone` (d1024 / 48 layers / 8 heads)

```
concat all tokens  (B, ΣN, d)
   │  StepConditioning(step_index, time_offset_s):
   │     Fourier(16 log-freqs each) → cat(64) → Lin(64→4d)→GELU→Lin(4d→d)   (out std .3)
   │     → (B, d)  broadcast-ADD to EVERY token
   ▼
 48 × BackboneBlock  (pre-norm):
   │   h = LayerNorm(x);   x = x + MultiheadAttention(h,h,h)   # FULL attn, 8 heads, ALL tokens
   │   x = x + MLP(LayerNorm(x));  MLP = Lin(d→4d)→GELU→Dropout→Lin(4d→d)→Dropout
   ▼
 LayerNorm (final)
 [opt] backbone_input_skip:  out = tokens + γ·out         # LayerScale γ init 0.2
   ▼
 out_tokens (B, ΣN, d)
```
- Attention is **bidirectional over all tokens of the window** → cross-modal fusion.
- Per-block **gradient checkpointing** at d1024.
- **Temporal limitation:** operates on ONE window (spatial/modality tokens only);
  no cross-window causal attention → cannot see multi-window **velocity** (the
  lever-1 experiment adds a longer window / temporal attention).

---

## 4. DECODERS (output heads) — continuous & generative

```
① SlowTimeSeriesHead     (B,C,d) → Linear(d→5) → (B,C,5)                         # exact inverse

② FastTimeSeriesHead     (B,80,d) → 4× refine → reshape(B·8,10,d)
                          → ConvTranspose1d(d→64, k=s=50)                (B·8,64,500)
                          → inv_stem: Conv1d(64→64,k3)→GELU→Conv1d(64→1,k3) → (B,8,500)

③ SpectrogramOutputHead  (B,192,d) → 12× refine → reshape(B,d,16,12)
                          → ConvTranspose2d(d→40, k=s=(32,8))            (B,40,512,96)
                          [+ inv_stem residual: ConvT2d(d→64)→GELU→2×Conv2d(3×3)→40]
                          [+ seam_refine: zero-init 3×3 conv, anti-checkerboard]
                          ⚠ deterministic MAE → conditional-mean → BLUR (mode-collapse)

④ SpectrogramFlowHead    μ = SpectrogramOutputHead(tokens)
                          velocity = 3-level 2D U-Net over (C,F,T); conditioning =
                             1×1-conv token map (bilinear↑) + global token vec +
                             sinusoidal flow-time, injected by AdaGN in each ResBlock
                          train: rectified-flow MSE on (target−μ)/σ  (band_weight opt.)
                          eval:  μ + σ·Euler(noise, 6 steps)            # a SAMPLE
                          └ residual_anchor: NO μ; returns σ·Euler; anchor adds input
                          (samples fixed the dampening but → incoherent speckle)

⑤ VideoOutputHead        (B,300,d) → reshape(B,d,1,10,30)
                          → ConvTranspose3d(d→2, k=s=(3,12,12))  OR
                            resize_conv: trilinear↑ → 3×Conv3d(3×3)  → (B,3,2,120,360)
⑥ VideoFlowHead          video analogue of ④ (fold C·T channels, 2D U-Net over H,W)
```

---

## 5. FSQ discrete-code path (PRODUCTION spectro) — the important one

Spectrograms are predicted as **discrete codes of a frozen, adversarially-trained
autoencoder codec**, not as continuous pixels. Two parts:

### 5.1 Frozen codec  `SpectroFSQCodec`  (Phase 1a, trained separately, then FROZEN)
```
  ENCODER  = SpectrogramTokenizer(patch (64,32), freq_stem=TRUE)      # freq_stem ON here
       x (B,C,F,T) → [fold channels] → tokens (B·grp, n_tok_per, d)
  FSQ      = FSQBottleneck(d, levels=[L]*dim)                          # e.g. dim=24, L=8
       tokens → project to `dim` → bounded → ROUND to L levels (straight-through)
       → per-token per-dim INT codes  ∈ {0..L-1}     # NO learned codebook → no collapse
  DECODER  = SpectrogramOutputHead(patch (64,32))                      # ConvTranspose2d unembed
       codes → codes_to_tokens → dec → reconstructed (B,C,F,T)
  (+ adversarial PatchGAN `SpectroDiscriminator` during codec training → sharp recon)
  residual codec: bg_subtract=True → baseline-subtract (gaussian σ freq) BEFORE encode
    (world model then works in R-space where the mode is the signal)
  API:  encode_codes(x)→(B,n_tok,dim) int   ·   decode_codes(codes)→(B,C,F,T)
```

### 5.2 World-model head  `SpectrogramCodeHead`  (Phase 1b, trained with the backbone)
```
  out_tokens[slice]  (B, n_tok, d)
     │ trunk: [Linear(d→pred_hidden)→GELU] × pred_layers
     │ per-dim heads: dim × Linear(pred_hidden → L)
     ▼ code_logits (B, n_tok, dim, L)
   TRAIN: CE( logits , codec.encode_codes(target) )   # categorical → cannot mean-collapse
          (optional class-weight / focal to up-weight rare MODE codes)
   EVAL:  codes = sample(softmax(logits / T))   [T=sample_temperature, default 1.0]
          spectro = codec.decode_codes(codes)          # through the FROZEN decoder
```
`SpectrogramMaskGITHead` = a bidirectional transformer that decodes the whole
code grid jointly (fixes the per-token independent-sampling incoherence).

### 5.3 What we just measured (2026-07-13, residual-FSQ overfit, ece 200729)
- **Mode LOCATION is predicted well:** peak-match **0.90**, capture ~1.0, **beats
  persistence** at T=1. The comb of harmonics shows up at the right frequencies.
- **Amplitude is capped by the CODEC:** `codec-ceiling tvr = 0.36` — encoding the
  *ground-truth* mode and decoding it back yields only 36% of the variance. The
  world model (tvr 0.48) is already at that ceiling. **The frozen codec is the
  amplitude bottleneck, not the world model or the sampler.**
- **Temperature is not the lever:** T=3 raised tvr (1.58) but destroyed the mode
  (capture→0, just noise). T=1 is right.
- ⇒ **Actionable fix = an amplitude-preserving codec** (retrain the codec to raise
  its tvr ceiling: e.g. weight recon toward the mode band / lighter FSQ
  quantization / stronger adversarial), then the existing world model (good at
  location) will render full-amplitude modes.

---

## 6. Forecast anchors & loss

`model.forward`:
```
out_tokens = backbone(concat tokens, step_index, time_offset)
if backbone_input_skip:  out_tokens = tokens + γ·out_tokens
predictions = { m: head_m(out_tokens[slice_m]) }
# spectro-only forecast anchor (continuous heads):
if spec_warp_anchor:          pred[m] = grid_sample(input_m, Δf(tokens)) + pred[m]
elif spec_persistence_anchor: pred[m] = input_m + pred[m]     (+ flow residual_anchor sample)
```
Per-modality loss: **MAE** (continuous), **CE** (FSQ code), **rectified-flow MSE**
(generative); optional mode-band weight, per-bin weight, struct/mask (Dice+BCE).

Rollout: the backbone's token output is fed forward directly (heads bypassed);
output-anchored continuous heads need a decode/re-encode rollout, whereas the
token-space FSQ path is rollout-native.

---

## 7. Verification note
Encoders (§2), backbone (§3), continuous+flow heads (§4), and the FSQ codec +
code head (§5) were read line-by-line from source on 2026-07-13. `freq_stem` is
ON inside the codec encoder, OFF by default in the backbone tokenizer (a known
representation-gap). Token counts are for the tabulated production config.
