# Perceiver Foundation Model — Architecture and Data Flow

## Overview

The foundation model predicts the future state of a tokamak plasma from a 500 ms context window and actuator commands. It operates entirely in latent space: pre-trained autoencoders (AEs) compress raw diagnostic signals into tokens, the Perceiver processes these tokens, and a dynamics model predicts future latent states autoregressively.

```
Raw signals  ──►  AE encoders (frozen)  ──►  Perceiver  ──►  Dynamics  ──►  Perceiver decoder  ──►  AE decoders (frozen)  ──►  Predicted signals
                  [per modality]              [encode]       [rollout]       [decode]                [per modality]
```

---

## 1. Autoencoder Tokenization (frozen, per-modality)

Each diagnostic modality (e.g. `ts_core_temp`, `filterscopes`, `mse`) has a pre-trained AE that compresses a 500 ms signal window into a fixed number of latent tokens.

**Input:** Raw signal `x_m ∈ R^{C_m × T_m}` for modality `m` (channels × time samples).

**Output:** AE tokens `z_m ∈ R^{N_m × d_lat_m}` where `N_m` is the number of tokens and `d_lat_m` is the per-modality latent dimension.

The AEs are frozen during foundation model training. They define the token vocabulary that the Perceiver reads and writes.

---

## 2. Modality Tokenizer (`ModalityTokenizer`)

Projects all per-modality AE tokens into a common dimension and adds positional/type information.

For each modality `m` present in the input:

```
h_m = W_m · z_m + e_m + PE(t_m)
```

where:
- `W_m ∈ R^{d_model × d_lat_m}` — learned linear projection (no bias)
- `e_m ∈ R^{d_model}` — learned modality embedding (broadcast across tokens)
- `PE(t_m)` — sinusoidal time encoding of each token's center time within the window

All modality token sequences are concatenated:

```
H = [h_1; h_2; ...; h_M]  ∈ R^{B × N_total × d_model}
```

where `N_total = Σ_m N_m`.

---

## 3. Actuator Tokenizer (`ActuatorTokenizer`)

Converts raw actuator time series into transformer tokens via patch embedding.

For each actuator group `a` (e.g. `pin`, `beam_voltage`, `gas_flow`):

```
p_a = Conv1d(u_a) + e_a + PE(t_a)
```

where:
- `Conv1d` has `kernel_size = stride = patch_len` (non-overlapping patches)
- `u_a ∈ R^{B × C_a × T_samples}` — raw actuator signal
- `e_a ∈ R^{d_model}` — learned actuator-type embedding
- `PE(t_a)` — sinusoidal time encoding with absolute offset

All actuator tokens are concatenated and LayerNormed:

```
A = LayerNorm([p_1; p_2; ...; p_A])  ∈ R^{B × N_act × d_model}
```

The actuator tokenizer is used in two places:
1. **Encoder context** — actuator tokens from the 500 ms context window are appended to diagnostic tokens before encoding.
2. **Dynamics input** — actuator tokens from the current and future DT_S windows are used as cross-attention context at each rollout step.

---

## 4. Perceiver Encoder (`PerceiverEncoder` + `LatentProcessor`)

Compresses the variable-length token sequence into a fixed-size latent array.

### 4a. Cross-attention encoding

A set of `N_L` learned latent queries `Q ∈ R^{N_L × d_model}` cross-attends to the input tokens `H` (optionally concatenated with actuator context tokens `A`):

```
Input context:  C = [H; A]  ∈ R^{B × (N_total + N_act) × d_model}

For each cross-attention layer:
    attn = MultiHeadAttn(Q=L, K=C, V=C)
    L = LayerNorm(L + attn)
    L = LayerNorm(L + FFN(L))
```

**Default:** 1 cross-attention layer, 128 latent queries, d_model=256.

### 4b. Self-attention processing

The latent array is refined through self-attention:

```
For each processor layer:
    attn = MultiHeadAttn(Q=L, K=L, V=L)
    L = LayerNorm(L + attn)
    L = LayerNorm(L + FFN(L))
```

**Default:** 1 processor layer.

**Output:** `L ∈ R^{B × N_L × d_model}` — the compressed plasma state.

The encoder and processor use **post-norm** (residual then LayerNorm). This is fine here because they are called once per forward pass, not recurrently.

---

## 5. EMA Target Encoder

A slowly-updated copy of the online encoder (tokenizer + encoder + processor + actuator tokenizer), following the JEPA/BYOL paradigm.

```
θ_ema ← τ · θ_ema + (1 − τ) · θ_online     (τ = 0.996)
```

The EMA encoder produces the **target latents** that the dynamics model predicts. Using a separate encoder prevents representation collapse without contrastive negatives.

No gradients flow through the EMA encoder.

---

## 6. Dynamics Model (`CrossAttentionDynamics`)

Predicts the next latent state from the current state and actuator commands. Called **recurrently** during autoregressive rollout — the output of one step is the input of the next.

### Architecture

```
latent_{k+1} = latent_k + delta_k
```

where `delta_k` is computed in three stages:

### 6a. Actuator extraction (cross-attention, no query residual)

Tokenize the current and future actuator windows, then cross-attend:

```
A_curr = ActuatorTokenizer(u_curr, offset=t_k)
A_fut  = ActuatorTokenizer(u_fut,  offset=t_k + dt)
context = [A_curr; A_fut]

act_info = latent_k                          # initial queries
For each cross-attention layer:
    attn = MultiHeadAttn(Q=act_info, K=context, V=context)
    act_info = LayerNorm(attn)               # NO query residual
    act_info = LayerNorm(act_info + FFN(act_info))
```

**Key design:** No residual from queries. The output `act_info` is built entirely from actuator value vectors. The queries (`latent_k`) only affect attention routing (Q-K alignment), not the output values. This prevents the dynamics from trivially copying the input state.

**Consequence for rollout:** `act_info` is always in the span of actuator values — its magnitude is bounded by the actuator tokenizer's output scale, regardless of `latent_k`'s magnitude.

### 6b. State-actuator fusion (MLP)

Combine the actuator-derived information with the current state:

```
delta = FusionMLP([act_info; latent_k])
```

where `FusionMLP: R^{2·d_model} → R^{4·d_model} → R^{d_model}` with GELU activation.

**Rationale:** Without this, delta would be purely a function of actuators, independent of the plasma state. The fusion MLP enables `delta = f(state, actuators)` — the actuator effect depends on the current plasma regime.

### 6c. Self-attention mixing

```
For each self-attention layer:
    attn = MultiHeadAttn(Q=delta, K=delta, V=delta)
    delta = LayerNorm(delta + attn)
    delta = LayerNorm(delta + FFN(delta))
```

**Default:** 1 self-attention layer. Allows inter-token communication after the per-token fusion.

### 6d. Residual update

```
latent_{k+1} = latent_k + delta_k
```

No output normalization — the latent accumulates freely across rollout steps.

### Known property: LayerNorm in recurrent path

The cross-attention blocks (6a) and self-attention blocks (6c) contain internal LayerNorms that bound the magnitude of `delta_k` at each step. This means:
- `||delta_k|| ≈ sqrt(d_model)` at every step (bounded by post-norm)
- `||latent_k||` grows linearly with steps (accumulation)
- `cos_sim(latent_k, latent_{k+1}) → 1` as k grows — this is a geometric artifact, not a bug

The delta loss (Section 9d) and context augmentation (Section 10) are critical for preventing copy behavior during training. Without them, the model converges to zero delta because the signal loss alone doesn't strongly penalize copy when `target ≈ context`.

### Testing pitfall: `.sum()` through LayerNorm

LayerNorm normalizes to zero mean per token, so `LN(x).sum()` is always zero regardless of `x`. Any test that computes `output.sum().backward()` will get zero gradient through post-normed outputs. Use MSE or another non-trivial loss function for gradient tests.

---

## 7. Perceiver Decoder (`PerceiverDecoder`)

Decodes the latent array back to per-modality token sequences. Each modality has its own set of learned output queries.

```
For each modality m:
    O_m = output_queries_m                   # learned, R^{N_m × d_model}
    For each decoder layer:
        attn = MultiHeadAttn(Q=O_m, K=L, V=L)
        O_m = LayerNorm(O_m + attn)          # WITH query residual
        O_m = LayerNorm(O_m + FFN(O_m))
        attn_self = MultiHeadAttn(Q=O_m, K=O_m, V=O_m)
        O_m = LayerNorm(O_m + attn_self)
        O_m = LayerNorm(O_m + FFN(O_m))
```

**Default:** 2 interleaved (cross-attn + self-attn) layers.

Each modality's output is then projected back to its AE latent dimension:

```
z_hat_m = W_out_m · O_m      where W_out_m ∈ R^{d_lat_m × d_model}
```

---

## 8. Autoregressive Rollout (inference)

The encoder is called once on the initial 500 ms context. All subsequent predictions use the dynamics model only:

```
L_0 = Encode(context)

For k = 0, 1, ..., N_steps-1:
    L_{k+1} = Dynamics(L_k, u_curr_k, u_fut_k)
    z_hat_k = Decode(L_{k+1})
    signal_k = AE_Decode(z_hat_k)          # frozen AE decoder
```

Each step predicts `DT_S` seconds ahead (default 500 ms). The rolled-out signal segments are stitched together to form a continuous prediction.

---

## 9. Training Losses

All losses are computed at each rollout step `k` and averaged. Later steps receive higher weight: `w_k = (k+1) / N_rollout`.

### 9a. Encode loss

Aligns online and EMA encoder representations of the same context:

```
L_enc = MSE(Encode_online(ctx), Encode_ema(ctx))
```

Weight: 0.1. Prevents online/EMA divergence.

### 9b. Reconstruction loss

The Perceiver roundtrip should preserve the AE tokens:

```
L_rec = (1/M) Σ_m MSE(Decode(Encode(ctx))_m, z_ctx_m) / Var(z_ctx_m)
```

Weight: 1.0. Trains the encoder-decoder bottleneck.

### 9c. Signal loss (latent-space prediction)

The dynamics output should match the EMA-encoded target:

```
L_sig = (1/K) Σ_k w_k · MSE(L_k, Encode_ema(target_k)) / Var(target_k)
```

Weight: 1.0. Direct gradient to dynamics without decoder attenuation.

### 9d. Delta loss

The displacement from context should match the target displacement:

```
delta_pred_k = L_k − L_ctx         (total displacement from context)
delta_tgt_k  = Encode_ema(tgt_k) − Encode_ema(ctx)

L_dlt = (1/K) Σ_k w_k · MSE(delta_pred_k, delta_tgt_k) / Var(delta_tgt_k)
```

Weight: 1.0. Explicitly penalizes copy behavior (zero delta).

### 9e. Rollout loss (decode-space prediction)

The decoded AE tokens should match the ground-truth AE tokens:

```
L_rol = (1/KM) Σ_k Σ_m w_k · MSE(Decode(L_k)_m, z_tgt_k_m) / Var(z_tgt_k_m)
```

Weight: 1.0. Ensures the Perceiver decoder can interpret the dynamics output.

### Total loss

```
L = 0.1·L_enc + 1.0·L_rec + 1.0·L_sig + 1.0·L_dlt + 1.0·L_rol
```

---

## 10. Training Curriculum

### Rollout ramp

The number of rollout steps increases linearly from `rollout_start` (1) to `N_ROLLOUT` (16) over `rollout_ramp_epochs` (30) epochs.

### Teacher forcing

At each rollout step, with probability `p_tf`, the dynamics input is replaced with the EMA-encoded ground truth (detached). `p_tf` decays linearly from `teacher_forcing_start` (0.5) to 0 over `teacher_forcing_epochs` (40) epochs.

### Noise injection

When teacher forcing is not applied, Gaussian noise with `rollout_noise_std` (0.1) is added to the dynamics output before the next step.

### Context augmentation

During training, the encoded context is corrupted with Gaussian noise (`context_noise_std=0.1`) and random token dropout (`context_drop_rate=0.1`) to prevent the dynamics from relying on exact encoder outputs.

---

## 11. Tensor Shapes (default config)

| Component | Shape | Description |
|-----------|-------|-------------|
| AE tokens (per modality) | `[B, N_m, d_lat_m]` | N_m ∈ {16, 20}, d_lat ∈ {32, 256} |
| Modality tokens (total) | `[B, N_total, 256]` | N_total = 136 (sum of all N_m) |
| Actuator tokens (context) | `[B, N_act, 256]` | N_act ≈ 6 (one per actuator group) |
| Perceiver latent | `[B, 128, 256]` | N_L=128 queries, d_model=256 |
| Dynamics delta | `[B, 128, 256]` | Same shape as latent |
| Decoder output (per mod) | `[B, N_m, 256]` | Projected to d_lat_m after |

---

## 12. Differentiated Learning Rates

The optimizer uses two parameter groups:

| Group | Default LR | Components |
|-------|-----------|------------|
| Encoder | 1e-5 | tokenizer, encoder, processor, decoder, output projections |
| Dynamics | 1e-3 | dynamics model (cross-attention, fusion MLP, self-attention) |

The 100x higher dynamics LR reflects that the encoder/decoder need to maintain a stable latent space while the dynamics learns to navigate within it.
