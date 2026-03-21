# FAITH Development Log

---

## 2026-03-05 — CO2 MAE Architecture Decisions

### How does MAE handle RGB channels usually?

In the original MAE paper (He et al., 2022) on ImageNet, RGB is handled by
flattening all 3 channels together into a single patch vector of size
`3 × patch_h × patch_w` and projecting with one shared linear. There is no
per-channel specialisation.

The reasoning is that RGB channels are strongly correlated (same scene,
slightly different wavelengths) and the model benefits from seeing all three
together when reconstructing a patch.

Our implementation departs from this by using **per-channel independent
patch embedding kernels and decode heads** (`PerChannelPatchEmbed2d`,
`PerChannelPatchUnembed2d`). The motivation is that fusion diagnostic channels
are less correlated and more semantically distinct than RGB — CO2 chords
measure the same plasma at different impact parameters, and ECE channels
correspond to different radial positions with genuinely different temperature
profiles. They are better treated as separate sensors than as components of a
single measurement.

---

### Is a single linear decode head enough for per-channel specialisation?

A single `Linear(d_model → ph×pw)` per channel has to do two things
simultaneously: decode spatial structure within the patch, and specialise to
the channel's radial/spectral character. One layer conflates these with no
nonlinearity to compose them.

In standard MAE, the lightweight decoder is a deliberate choice — the decoder
is discarded after pretraining and the burden of representation is meant to
fall on the encoder. For reconstruction-quality goals (anomaly detection,
generative modelling of plasma states) a deeper per-channel head is
warranted:

```
d_model → d_model (GELU) → ph×pw
```

However, the practical bottleneck may be upstream. With `d_model=256` and
only 2 decoder transformer layers, the transformer itself may be the limiting
factor before the decode heads are. Recommended approach: profile
reconstruction quality first — if error maps still show a uniform bias rather
than fine-grained residuals, the issue is in the transformer, not the heads.

---

### Is it expected to take a while for positional encoding to become helpful?

Yes — early horizontal stripes in the reconstruction are a normal early
training phase.

The positional embeddings start at small random values (`trunc_normal_,
std=0.02`), so early on the tokens are nearly position-blind. The model's
best strategy at that point is to predict the global mean patch, which for
CO2 is "mostly green with a bright low-frequency band" — hence the stripes.
Horizontal stripes (aligned with frequency rows) indicate the model has
learned the global spectral shape; the positional embeddings now need to
differentiate *where* in the spectrogram each patch sits.

Positional embeddings take longer to converge than transformer weights because
their gradient signal is indirect — the loss only reports total reconstruction
error, not explicit positional error. With only 8 frequency rows at
`patch_h=16` the freq-axis table has few distinct positions and may converge
quickly; the time axis (~123 positions) will take longer.

Rough heuristic: in the original MAE on ImageNet, meaningful spatial structure
appears around epochs 200–400 out of 1600. Running 500 epochs on a smaller
dataset, stripes should fade around epochs 100–200. If they persist past epoch
300, revisit the positional embedding learning rate or initialisation.

---

### Why L1 loss instead of MSE?

The `nn.L1Loss()` in `spectrogram_reconstruction.py` predates this work and
was not introduced here. The choice is reasonable for spectrograms: MSE
penalises large errors quadratically, which can cause the model to
over-focus on a few high-amplitude patches (e.g. the bright low-frequency
band in CO2) at the expense of quieter mid/high-frequency structure. L1
treats all errors more equally across the dynamic range, which is generally
preferred when the signal has a heavy-tailed amplitude distribution — as
log-spectrograms typically do.

---

---

## 2026-03-08 — FSQ-VAE Transformer Autoencoder for Spectrogram Modalities

### Motivation

The current `SpectrogramMAEAutoEncoder` is designed for pre-training an encoder
via masked reconstruction (He et al. 2022). The MAE decoder is a training-time
artifact and is not intended to be used at inference. FAITH's simulation
pipeline, however, requires a high-quality decoder at inference time:

```
encode(spectrogram_t) → latent_tokens → forward_predictor → latent_tokens_t+1
                                                                      │
                                              decode(latent_tokens_t+1) → spectrogram_t+1
```

This motivates a **reconstruction-first** autoencoder design where decoder
quality is the primary objective, not a byproduct.

---

### Why FSQ over VQ or continuous VAE?

**Continuous VAE** suffers from posterior collapse: a powerful transformer
decoder can learn to ignore the latent code and reconstruct from the prior
alone. Requires careful β-scheduling.

**VQ-VAE** avoids posterior collapse but requires maintaining a codebook,
a commitment loss, and EMA updates. Codebook collapse (few codes used) is a
real failure mode.

**FSQ (Finite Scalar Quantization, Mentzer et al. 2023)** is strictly simpler:
project to L dimensions, clamp with bounded tanh, round to integer bins. No
codebook, no commitment loss, no collapse. The effective codebook size is the
product of the per-dimension bin counts (e.g. `[8,5,5,5,5]` → 5000 codes/token).
Increasingly preferred over VQ in recent work.

---

### Why is this faster than diffusion?

A diffusion decoder (e.g. DiTo, arXiv:2501.18593) requires 20–50 iterative
denoising steps per decode. An FSQ-VAE decoder is a single deterministic
forward pass. For simulation where the decoder is called at every rollout
step, this difference is decisive.

Speed comparison for the forward predictor (Stage 2):

| Method | Steps per generated frame |
|---|---|
| Autoregressive (GPT over tokens) | N sequential passes |
| MaskGIT-style masked transformer | 8–12 parallel passes |
| Diffusion in latent space | 20–50 passes |
| FSQ-VAE decode alone | 1 pass |

---

### Recommended architecture

```
Input (B, C, F, T)
      │
      ▼
FlatPatchEmbed              # Linear(C·ph·pw → d_model), 2D factored pos embed
      │
      ▼
Transformer Encoder         # 4–6 layers, d_model=256, norm_first=True
      │
      ▼
Linear(d_model → L)         # Project to FSQ dims (L = 5–8)
      │
      ▼
FSQ                         # levels e.g. [8,5,5,5,5] → 5000 effective codes/token
                            # round() with straight-through gradient, no codebook
      │
      ▼
Linear(L → d_model)         # Project back up
      │
      ▼
Transformer Decoder         # 4–6 layers, separate 2D factored pos embed
      │
      ▼
FlatPatchUnembed            # Linear(d_model → C·ph·pw)
      │
      ▼
Output (B, C, F, T)
```

**Loss:** per-patch variance-weighted L1 (already implemented in
`MAEUnimodalTrainer`). No KL term. No masking — full reconstruction at all
training steps.

**FSQ levels:** the product of levels is the effective codebook size per token.
`[8,5,5,5,5]` (5000 codes) is a good starting point. Keep L small (5–8 dims)
— FSQ works best with a tight bottleneck.

---

### Two-stage training structure

Following MaskGIT / FSQ-VQ-VAE practice, the two stages remain separate:

- **Stage 1:** Train the FSQ-VAE autoencoder to convergence. Freeze encoder
  and decoder.
- **Stage 2:** Train the forward predictor (transformer) over sequences of
  FSQ token indices. The predictor never touches pixels — it operates
  entirely in discrete token space.

This matches FAITH's existing two-stage design. The Stage 1 reconstruction
quality is a hard ceiling on simulation quality, since Stage 2 only ever
operates on token indices.

---

### Caveats

- FSQ with transformer backbone is less battle-tested than VQGAN (which uses
  CNN). Expect to tune the projection dimension L and levels more carefully.
- No commitment loss means no explicit pressure to use all codes — monitor
  codebook utilisation during training (count of unique indices used per batch).
- Recommended: ablate on CO2 at small scale before extending to ECE (48
  channels) or other modalities.

---

### Implementation (2026-03-08)

**Files:**
- `src/tokamak_foundation_model/models/modality/spectrogram_fsq_vae.py` — `FSQ`, `_ViTDecoder`, `SpectrogramFSQVAEAutoEncoder`
- `scripts/training/spectrogram_reconstruction.py` — `FSQUnimodalTrainer`, `--fsq_levels` arg
- `scripts/slurm/train_co2_fsq_vae.sh` — SLURM job script

**Key implementation notes:**

- `FSQ.forward`: `half_levels = levels[i] // 2` (integer floor). For level 8 → half=4, bounded
  to `(-4, 4)` via tanh, rounds to `{-4,...,3}` shifted to `{0,...,7}`. Straight-through gradient
  via `z_q = z_bounded + (z_rounded - z_bounded).detach()`. Scalar index via mixed-radix strides.
- `pre_fsq` weight initialised with `std=0.02` (not the default Xavier) to keep activations in
  the linear regime of tanh at the start and avoid immediate saturation.
- Encoder/decoder reuse `_ViTEncoder` from `spectrogram_mae.py`. Decoder (`_ViTDecoder`) has
  its own separate positional embedding tables — not shared with the encoder.
- Training loss is plain L1 (`loss_fn(reconstructed, data)`) — no patch weighting, no masking.
  Makes train/val directly comparable.
- `FSQUnimodalTrainer` logs `codebook_utilization = unique_indices / n_codes` per batch
  (accumulated as a running mean by the tracker).

**First training run (job 2655612, cancelled after 5 epochs):**

| Epoch | Train L1 | Val L1 |
|-------|----------|--------|
| 1     | 1.289    | 0.319  |
| 2     | 0.307    | 0.316  |
| 3     | 0.298    | 0.319  |
| 4     | 0.295    | 0.317  |
| 5     | 0.294    | 0.315  |

Val loss flat from epoch 2. Train/val gap inverted (train > val), suggesting the model is
predicting close to the dataset mean. Codebook utilisation was not yet logged. Job cancelled
and restarted from scratch (job 2655625) with utilisation logging added.

---

### If training stalls early — diagnostic checklist

**Step 1: Check codebook utilisation first.** This is the canary. Healthy FSQ training should
see utilisation climb from ~5–20% in epoch 1 to >50% by epoch 10–20. If it stays below 10%
after epoch 5, the FSQ is collapsed and the decoder is seeing a near-constant input — it will
learn the dataset mean and val loss will flatline regardless of other settings.

**Step 2: Causes of codebook collapse and fixes.**

| Symptom | Likely cause | Fix |
|---|---|---|
| Util <5%, loss flat | `pre_fsq` saturating tanh | Reduce `pre_fsq` weight init std (try `0.01`); check activations before FSQ |
| Util 5–20%, loss slow | Too many codes for dataset size | Reduce levels: try `[4,4,4]` (64 codes) or `[5,5,5,5]` (625 codes) |
| Util OK but loss flat | Decoder bottleneck | Increase `n_dec_layers` (try 6); increase `d_model` (try 512) |
| Util OK, loss noisy | LR too high | Halve LR; extend warmup to 20 epochs |
| Util OK, loss decreasing slowly | LR too low / warmup too long | Increase LR to `1e-3`; shorten warmup |

**Step 3: If nothing above helps.**

- Switch to a smaller patch size (`patch_h=8, patch_w=8`) to increase the number of tokens
  and give the FSQ more positions to work with.
- Try a CNN encoder (VQGAN-style) instead of ViT — CNNs provide stronger local inductive
  bias that may help the FSQ see meaningful local structure to quantise.
- Revisit whether the data preprocessing (log-standardize for CO2) is producing reasonable
  input magnitudes — very large or very small inputs will push `pre_fsq` outputs into tanh
  saturation before training has a chance to correct it.

## 2026-03-13

● The CNN at 8× proves the encoder/decoder can reconstruct through a compressed bottleneck. But 8× continuous is still far
   from what the fusion transformer needs (few hundred globally-informed tokens). The next steps to bridge that gap:

  1. Find the CNN compression wall — sweep bottleneck_dim down (4, 2, 1) to see where reconstruction breaks. This tells
  you the minimum spatial information needed at stride 4. Takes a few hours and is cheap to run.

  2. Add global context at the bottleneck — this is the real architectural step. Replace the 1×1 projection with a small
  transformer or cross-attention pooling that reads the full CNN feature map and compresses it into N learned tokens
  (e.g., 64–256). Something like:

  CNN encoder → (128, F/4, T/4) feature map
      → flatten to ~15K spatial tokens
      → cross-attention with N learned query tokens
      → N latent tokens (this is what the fusion transformer sees)
      → cross-attention back to 15K spatial positions
  → CNN decoder

  This gives you the best of both worlds: CNN handles local spatial processing (where it excels), and the attention
  bottleneck provides global receptive field + controlled token count. It's essentially a Perceiver-style bottleneck
  sandwiched between CNN stages.

  3. Then add FSQ at the bottleneck — once the continuous version of (2) reconstructs well, quantize the N latent tokens
  with FSQ. Now you have a small number of discrete, globally-informed tokens ready for the fusion transformer.

  I'd start with (1) since it's just a parameter sweep — want me to launch those jobs?

---

## 2026-03-21 — Channel-Attention AST: Breaking the 0.029 L1 Wall

### Problem

Every bottlenecked spectrogram autoencoder (AST-FSQ, ConvNeXt-FSQ, FSQ-VAE) plateaued at
~0.029 train L1 on CO2. The AST-FSQ flattens all channels into each token via
`Linear(C*F*fw, d_model)` — for CO2 (C=4, F=128, fw=2) that's a 1024→256 projection, a
4:1 compression at the very first layer. Hypothesis: the frame projection is the bottleneck.

### Architecture: `SpectrogramChannelASTFSQAutoEncoder`

New model in `spectrogram_channel_ast_fsq.py`. Key insight: **per-channel frame embedding**
avoids the C×F×fw → d_model compression.

```
Input: (B, C, F, T)
  → Pad T to multiple of frame_width
  → Per-channel frame embed: Linear(F*fw, d_model) per channel — nearly 1:1 for CO2
  → Add learned channel_pos_embed + time_pos_embed
  → n_enc_layers × ChannelTimeBlock:
      1. Channel attn: (B*N, C, d_model) → TransformerEncoderLayer
      2. Time conv:    (B*C, d_model, N) → ConvNeXtV2Block1d
  → [Optional: channel merge cross-attention — C tokens/frame → 1 token/frame]
  → [Optional: FSQ bottleneck: pre_fsq → FSQ → post_fsq]
  → Decoder (mirror of encoder with separate pos embeds)
  → Frame unembed: Linear(d_model, F*fw), reshape to (B, C, F, T)
```

Reuses: `FSQ` from `spectrogram_fsq_vae.py`, `_ConvNeXtV2Block1d` from `spectrogram_cnn1d.py`,
`ModalityAutoEncoder` from `base.py`.

### Ablation results (CO2, d_model=256, 4 enc + 4 dec layers, lr=1e-4, flat schedule)

| Variant | Tokens | Compression | Train L1 | Val L1 | Epoch | Job |
|---------|--------|-------------|----------|--------|-------|-----|
| + FSQ [8,8,5,5,5,5,5], fw=2 | 3908 | 1× (quant.) | 0.0290 | 0.0282 | 61 | 2671826 |
| No FSQ, fw=2 | 3908 | 1× | 0.0068 | 0.0055 | 60 | 2672110 |
| **No FSQ, fw=8** | **980** | **4×** | **0.0225** | **0.0219** | **319** | **2672475** |
| Channel merge, no FSQ, fw=8 | 244 | 16× | 0.0282 | 0.0282 | 408 | 2672277 |

**Key findings:**

1. **FSQ was the real bottleneck, not the frame projection.** Removing FSQ (fw=2) dropped L1
   from 0.029 → 0.007, a 4× improvement. The `Linear(256→7)` dimensionality collapse in the
   FSQ path destroys information — 36:1 compression per token.

2. **Temporal compression is cheap.** Going from fw=2 (3908 tokens) to fw=8 (980 tokens) only
   raised L1 from 0.007 → 0.023 despite 4× fewer tokens. Physics is local in time, so packing
   8 time steps per token (4:1 per-frame compression) loses little. The model at 980 tokens
   already beats every prior bottlenecked model.

3. **Channel merge is too aggressive.** Cross-attention merging C=4 channels into 1 per time
   frame hits the 0.028–0.030 wall. A single query attending into 4 channel tokens forces all
   inter-channel information through one d_model vector — the decoder can't recover distinct
   channels from identical replicated inputs + positional embeddings alone.

### Best model for fusion transformer: fw=8, no FSQ, no merge

- **980 tokens × 256 d_model** per CO2 sample
- **12.2M parameters**
- **Train L1 = 0.0225, Val L1 = 0.0219** (epoch 319, still slowly improving)
- No overfitting (train ≈ val throughout)
- Beats ConvNeXt BN32 S4 (0.026) which was the prior best bottlenecked model

### Reproduction

```bash
# Training
sbatch scripts/slurm/train_co2_channel_ast_nofsq_fw8.sh

# Visualization
pixi run python scripts/training/visualize_co2_channel_ast_nofsq_fw8.py
```

SLURM script: `scripts/slurm/train_co2_channel_ast_nofsq_fw8.sh`
Model class: `spectrogram_channel_ast_fsq.py::SpectrogramChannelASTFSQAutoEncoder`
Registry key: `"spectrogram_channel_ast_fsq"` in `model_factory.py`

Key CLI args:
```
--model spectrogram_channel_ast_fsq --fsq_levels --frame_width 8
--time_conv_kernel 7 --d_model 256 --n_fft 256 --hop_length 128
--lr 1e-4 --weight_decay 1e-4 --scheduler none --batch_size 16
```

(`--fsq_levels` with no values → empty list → disables FSQ)

### Files changed

- **Created:** `src/tokamak_foundation_model/models/modality/spectrogram_channel_ast_fsq.py`
  — `_ChannelMerge`, `_ChannelTimeBlock`, `_ChannelASTEncoder`, `_ChannelASTDecoder`,
  `SpectrogramChannelASTFSQAutoEncoder`
- **Modified:** `modality/__init__.py`, `model_factory.py` — added import and registry entry
- **Modified:** `tests/test_model_shapes.py` — 7 test configs (FSQ, no-FSQ, channel merge variants)
- **Modified:** `scripts/training/spectrogram_reconstruction.py` — `--time_conv_kernel`,
  `--channel_merge` args; `fsq_levels` changed to `nargs="*"` for empty-list support
- **Created:** SLURM scripts for all variants in `scripts/slurm/`
- **Created:** Visualization scripts in `scripts/training/`

### Next steps

- Try fw=16 (488 tokens) to see how far temporal compression goes before quality degrades
- Explore whether the fusion transformer can handle ~1000 tokens per modality, or if further
  reduction to ~256 is needed (may require a Perceiver-style bottleneck with multiple merge
  queries rather than the single-query approach that failed)
- Extend to ECE (C=40+) — the per-channel architecture should scale naturally since channel
  attention is O(C²) per time frame and C enters as sequence length, not embedding dimension
