# Aurora vs Tokamak Foundation Model — Architecture Comparison

## Overview

| | Aurora (Earth system) | Ours (Tokamak plasma) |
|---|---|---|
| **Domain** | Global weather, 6h timesteps | Tokamak plasma, 500ms timesteps |
| **Parameters** | 1.3B | ~35M |
| **Backbone** | 3D Swin Transformer U-Net (48 layers) | Perceiver IO (encoder + processor + decoder) |
| **Dynamics** | Non-recurrent (backbone IS the dynamics) | Recurrent (separate dynamics module called per step) |
| **Training** | 32× A100, ~2.5 weeks | 1× GPU, hours |

---

## 1. Autoregressive Rollout

| | Aurora | Ours |
|---|---|---|
| **Approach** | Feed (X^{t-1}, X^t) → backbone → X^{t+1}. The backbone processes the full state at each step. No recurrence — each call is a fresh forward pass. | Encode context once → recurrent dynamics loop: L_{k+1} = L_k + delta(L_k, actuators). The dynamics module is called N times. |
| **Key difference** | The backbone sees the complete observation at every step. The "dynamics" is implicit in the backbone. | The dynamics only sees the latent (compressed) state. The encoder/decoder are called once at the boundaries. |
| **Implication** | No error accumulation through a compressed bottleneck. Each step has full information. | Errors in the latent compress and accumulate. The dynamics must predict from an increasingly stale representation. |

## 2. Temporal Input

| | Aurora | Ours |
|---|---|---|
| **History** | T=2 timesteps as 3D patches: (X^{t-Δt}, X^t). Implicit finite-difference / velocity. | P1 fix: latent_prev fed alongside latent_current in fusion MLP. Similar idea but in compressed latent space. |
| **Time encoding** | Absolute time embedding (seasonal/diurnal cycles) + lead-time Fourier encoding | P0 fix: Fourier-encoded offset_ms through MLP. Similar but simpler — no seasonal/diurnal structure in tokamak data. |
| **Per-step adaptation** | LoRA adapter per rollout step — different weights at different lead times | None. Same dynamics weights at every step. The step embedding is the only differentiation. |

## 3. Prediction Target

| | Aurora | Ours |
|---|---|---|
| **Target space** | Observation space (weather variables at grid points) | Was: EMA-encoded latent space (compressed, co-adapted). P2 fix: detached online encoder (same space as prediction). |
| **Loss function** | Weighted MAE across variables | MSE normalized by target variance, multi-component (signal + delta + rollout + reconstruction) |
| **Residual prediction** | Direct absolute state prediction (no explicit residual) | L_{k+1} = L_k + delta. Explicit residual. |
| **Key difference** | Ground truth is the actual weather observation — no learned target encoder. | Target comes from the same encoder that produces the prediction. Self-referential. |

## 4. Multi-Step Training

| | Aurora | Ours |
|---|---|---|
| **Strategy** | Two-stage: (1) pretrain on single-step, (2) rollout fine-tune with LoRA | Curriculum: ramp rollout from 1→N over epochs + teacher forcing decay |
| **Gradient flow** | Pushforward trick: gradients only through final step. Memory-efficient. | Full backprop through entire rollout chain. Memory scales with N_ROLLOUT. |
| **Stability** | Replay buffer mixes ground truth and model predictions | Teacher forcing (decaying) + rollout noise injection + context augmentation |
| **Memory** | O(1) per step (pushforward) | O(N) per step (full backprop) |

## 5. Backbone Architecture

| | Aurora | Ours |
|---|---|---|
| **Type** | 3D Swin Transformer U-Net: hierarchical, multi-scale, shifted-window attention | Perceiver IO: cross-attention bottleneck with fixed-size latent array |
| **Normalization** | Pre-norm (standard for Swin) | Pre-norm in dynamics (P0 fix), post-norm in encoder/decoder |
| **Scale** | 48 layers, 3 hierarchical stages, skip connections | 1 encoder layer, 1-2 processor layers, 2-3 decoder layers, 1-3 dynamics layers |
| **Attention** | Local shifted-window (linear complexity) | Global (quadratic, but small token count) |

## 6. Modality / Variable Handling

| | Aurora | Ours |
|---|---|---|
| **Input types** | Surface variables (2D) + atmospheric variables (3D, multiple pressure levels) | Diagnostic signals (per-modality AE tokens) + actuator signals (raw patches) |
| **Tokenization** | Variable-specific linear projections + pressure level embeddings, summed | Per-modality AE encoder (frozen) → linear projection + modality embedding + time PE, concatenated |
| **Heterogeneity** | Arbitrary pressure levels per variable, handled by Perceiver cross-attention | Fixed token count per modality, missing modalities skipped |

## 7. Fundamental Design Differences

### Aurora: The backbone IS the dynamics
Aurora's Swin U-Net processes the full atmospheric state (two timesteps) and outputs the next state. There is no separate "dynamics module" — the entire backbone learns the physics. Each rollout step is a fresh forward pass through the full model with full observational context.

### Ours: Separate encoder, dynamics, decoder
We compress observations into a small latent (128 queries × 256 dims), then a lightweight dynamics module predicts the next latent. The decoder must reconstruct the full state from this compressed representation. This creates a bottleneck: the dynamics must predict changes in a space that may not preserve the information needed to reconstruct those changes.

### The key gap
Aurora's backbone sees the raw data at every step. Our dynamics sees only the compressed latent — and the decoder must faithfully translate latent changes back to signal changes. If the encoder/decoder bottleneck smooths out the differences between timesteps (which it does — that's what compression means), the dynamics has no target to learn from.

---

## 8. What We've Adopted from Aurora

| Aurora Feature | Our Implementation | Status |
|---|---|---|
| Pre-norm in recurrent path | Pre-norm in dynamics cross-attn + self-attn blocks | P0 ✓ |
| Lead-time / step encoding | Fourier-encoded offset_ms + MLP | P0 ✓ |
| T=2 history input | latent_prev in fusion MLP | P1 ✓ |
| Observation-space loss | Rollout loss (decoded AE tokens vs ground truth) | P1 ✓ (upweighted to 2.0) |
| No EMA target | Detached online encoder | P2 ✓ |
| Per-step LoRA | Not implemented | — |
| Pushforward trick | Not implemented (full backprop) | — |
| Replay buffer | Not implemented | — |
| Non-recurrent backbone | Not applicable (different architecture) | — |

## 9. What We Can't Adopt

- **Non-recurrent backbone**: Aurora's approach requires the backbone to process the full state at every step. At 1.3B parameters and 32 A100s, this is feasible. At 35M parameters on 1 GPU, processing the full state N times per training sample would be prohibitively expensive.
- **Per-step LoRA**: Requires separate adapter weights per rollout step. Adds parameter count proportional to N_ROLLOUT × rank × n_layers. Could be implemented but adds complexity.
- **Pushforward trick**: Trades gradient quality for memory. Could help if memory is a bottleneck at longer rollouts.

## 10. Remaining Gap Analysis

The fundamental difference is that Aurora predicts in observation space with full state context at every step, while we predict in a compressed latent space where the decoder may not preserve temporal variations.

The diagnostics confirm this: delta norms are non-zero (dynamics is working), but decoded cos_sim stays high (decoder collapses the differences). The encoder-decoder bottleneck is the remaining structural limitation.

Possible directions:
1. **Increase decoder capacity** — more layers, higher-dimensional output queries
2. **Auxiliary decoder loss per rollout step** — force the decoder to differentiate consecutive latents (the rollout loss does this, but at weight 2.0 it may not be enough)
3. **Skip the Perceiver latent for dynamics** — predict directly in AE token space (larger but no bottleneck)
4. **Contrastive loss on consecutive decoded outputs** — explicitly penalize identical decoded outputs at different rollout steps
