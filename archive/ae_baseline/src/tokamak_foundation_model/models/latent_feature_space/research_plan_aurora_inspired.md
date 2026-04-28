# Research Plan: Aurora-Inspired Tokamak Foundation Model

## Problem Statement

The current recurrent dynamics architecture (Perceiver encoder → lightweight dynamics → Perceiver decoder) suffers from a fundamental bottleneck: the dynamics operates in compressed latent space, and the decoder fails to translate latent changes back to signal-space differences. After implementing all 6 fixes from the previous research plan (pre-norm, step embedding, loss rebalance, history buffer, detached online encoder, gated query residual), the diagnostics show non-zero deltas but flat decoded predictions.

The root cause is structural: the encoder-decoder bottleneck compresses away the temporal variation the dynamics is trying to predict. Aurora avoids this entirely by running the full model at every rollout step — there is no compressed latent that accumulates over time.

## Core Design Change

**Current**: Encode once → recurrent dynamics loop in latent space → decode once.

**Proposed**: Full encode → backbone → decode at every rollout step. Predictions are fed back as input in AE token space (observation space), not latent space. No delta accumulation. No distribution drift.

```
Current:
  AE_encode → [Tokenize → Encode → Latent] → Dynamics(L) → Dynamics(L) → ... → [Decode → Deproject] → AE_decode
                                               ↑_________↩  ↑_________↩
                                               recurrent in compressed space

Proposed:
  AE_encode → [Tokenize → Encode → Backbone → Decode → Deproject] → AE_encode_pred → [Tokenize → Encode → ...] → ...
              |________________ full forward pass _________________|  ↑_______________fed back as input__________|
              every step, in observation (AE token) space
```

## Architecture

### Components (5 modules)

**1. ModalityTokenizer** — Existing, no change. Projects per-modality AE tokens into common `d_model` space. Optionally extended to accept T=2 history (concat `[z_{t-1}; z_t]` → `Linear(2*d_lat, d_model)`).

**2. ActuatorTokenizer** — Existing, no change. Conv1d patch embedding with time PE.

**3. PerceiverEncoder** — Existing, switch to pre-norm. Learned latent queries cross-attend to diagnostic + actuator tokens. Output: `(B, N_L, d_model)`.

**4. LatentBackbone** — NEW, replaces the old `CrossAttentionDynamics`. A deep Transformer stack (8-12 blocks) operating on the latent array. Each block has:
- Pre-norm self-attention (latent tokens interact)
- Pre-norm cross-attention to actuator tokens (control conditioning)
- Pre-norm FFN

Conditioned on step index via Fourier + MLP embedding added to all tokens. Optional U-Net skip connections between early and late blocks.

This is the main capacity increase: 8 blocks × (SA + cross-attn + FFN) vs the old 1 SA layer + 2-layer MLP.

**5. PerceiverDecoder** — Existing, switch to pre-norm. Per-modality output queries cross-attend to latent, project back to `d_lat`.

### Forward Pass (single step)

```python
def forward(ae_tokens, actuators, step_index):
    diag_tokens = modality_tokenizer(ae_tokens)     # (B, N_total, d_model)
    act_tokens  = actuator_tokenizer(actuators)     # (B, N_act, d_model)
    latent      = encoder(diag_tokens, act_tokens)  # (B, N_L, d_model)
    latent_next = backbone(latent, act_tokens, step_index)  # (B, N_L, d_model)
    ae_pred     = decoder(latent_next)              # {m: (B, N_m, d_lat_m)}
    return ae_pred
```

### Rollout

```python
current = ae_tokens_context
for k in range(n_steps):
    current = model.forward(current, actuators[k], step_index=k)
    # current is in AE token space — no latent drift
```

## Training (3 phases)

### Phase 1: Single-step pretraining (100 epochs)

- Input: AE tokens at time t. Target: AE tokens at time t+dt.
- Loss: per-modality MAE in AE token space, normalized by modality scale.
- No rollout, no curriculum, no teacher forcing.
- LR: 1e-4 with cosine schedule + warmup.
- This learns the encode → backbone → decode pipeline end-to-end on single-step prediction.

### Phase 2: Multi-step fine-tuning (50 epochs, K=4→8)

- Full backprop through K steps of the complete model.
- Each step runs the full forward pass (tokenize → encode → backbone → decode).
- Loss: weighted MAE at each step, later steps weighted more.
- LR: 3e-5 (lower than pretraining).
- Activation checkpointing on backbone blocks for memory.
- Rollout curriculum: K ramps from 4 to 8 over 30 epochs.

### Phase 3: Long rollout with pushforward (optional)

- Freeze backbone, add LoRA adapters (rank 8) to attention layers.
- Pushforward trick: gradients only through the last step.
- Replay buffer for stability.
- Extends to K=16 without memory issues.

## Loss Function

```
L = (1/K) Σ_k  w_k · (1/M) Σ_m  |pred_m^k - target_m^k| / scale_m
```

- `w_k = (k+1)/K` — later steps weighted more
- `scale_m` — per-modality normalization (estimated from training data)
- MAE (L1), not MSE — more robust to outliers, following Aurora
- **Single loss in AE token space** — no latent-space loss, no EMA, no encode alignment, no delta loss
- The reconstruction loss (decode(encode(x)) ≈ x) can be kept as a regularizer during Phase 1

## Parameter Count

| Config | Backbone | Total | Memory (est.) |
|--------|----------|-------|---------------|
| d=256, 8 blocks | ~16M | ~21M | ~8 GB per rollout step |
| d=384, 12 blocks | ~55M | ~70M | ~20 GB per rollout step |
| d=512, 12 blocks | ~120M | ~150M | ~40 GB per rollout step |

With activation checkpointing on the backbone, an 8-step rollout at d=256 fits in A100 80GB. Larger configs need bfloat16 autocast or pushforward.

Recommended starting config: **d=256, 8 backbone blocks** (~21M params). This is actually smaller than the current model (35M) because the heavy encoder/decoder are thinner without the EMA copy.

## Files to Create/Modify

| File | Action |
|------|--------|
| `perceiver_components.py` | Add `LatentBackbone`, `BackboneBlock` classes. Keep existing encoder/decoder (switch to pre-norm). Remove `CrossAttentionDynamics`. |
| `foundation_model.py` | New `TokamakFoundationModel` class (or refactor `PerceiverFoundationModel`). Forward pass runs full pipeline. Remove EMA encoder, dynamics module. |
| `train_foundation_model.py` | Rewrite training loop. Phase 1: single-step. Phase 2: multi-step with activation checkpointing. Single MAE loss in AE token space. |
| `modality_tokenizer.py` | Optional: `ModalityTokenizerWithHistory` for T=2 input. |
| `test_dynamics_rollout.py` | Rewrite tests for new architecture. Focus on: single-step prediction changes output, multi-step rollout diverges from context, backbone depth matters. |

## Key Differences from Current Architecture

| Aspect | Current | Proposed |
|--------|---------|----------|
| Dynamics | Lightweight MLP + 1 SA layer, recurrent | Deep 8-block Transformer, non-recurrent |
| Rollout space | Compressed latent (128 × 256) | AE token space (~136 × 32-256) |
| Per-step compute | Dynamics only (~2M params) | Full model (~21M params) |
| Target | Detached online encoder (still a learned mapping) | Ground truth AE tokens (frozen, objective) |
| Loss | 5 components (enc, rec, sig, dlt, rol) | 1 component (MAE in AE token space) |
| EMA encoder | Present (unused after P2 fix) | Removed entirely |
| Gradient flow | Through dynamics only (encoder/decoder nearly frozen at 1e-5 LR) | Through entire model |

## Success Metrics

### Phase 1 (single-step)
- Per-modality MAE decreasing
- Reconstruction: decode(encode(target)) ≈ target (the backbone helps, not hurts)

### Phase 2 (multi-step)
- Decoded predictions at step 4+ show temporal structure different from step 1
- `decoded_cos_sim` between consecutive steps drops below 0.9 by epoch 30
- `delta_ratio = pred_delta / tgt_delta` stays in [0.5, 2.0] at all rollout steps

### Phase 3 (long rollout)
- 16-step rollout tracks ground truth evolution qualitatively
- Per-step MAE doesn't blow up exponentially

## Risks

1. **Compute cost**: Full forward pass at every rollout step is ~10x more expensive per training sample than the current recurrent approach. Phase 2 with K=8 requires 8× the compute of Phase 1.

2. **Memory**: 8 full forward passes with gradients. Activation checkpointing is mandatory. May need to reduce batch size.

3. **AE token space may still be too smooth**: If the frozen AEs compress temporal variation (e.g., the AE encoder for `ts_core_temp` produces similar tokens for similar windows), the targets are smooth even in AE token space. This would be a data/AE issue, not a model issue.

4. **Backbone overfitting**: 21M params on ~960 training chunks. Need strong regularization (dropout, weight decay, data augmentation).
