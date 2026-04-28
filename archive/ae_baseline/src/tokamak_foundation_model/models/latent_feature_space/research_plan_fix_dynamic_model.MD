# Research Plan: Fixing Autoregressive Copy/Scale/Shift Failure

## Problem Statement

The foundation model for tokamak plasma prediction suffers from a critical failure during autoregressive rollout: after the first prediction step, subsequent steps produce outputs that are merely copies, scalings, or shifts of the initial prediction rather than genuinely evolving dynamics. This failure has persisted despite the model already incorporating residual prediction, delta loss, multi-step rollout with curriculum, teacher forcing, observation-space loss, and context augmentation.

This plan diagnoses the root causes by comparing the current architecture against the Aurora foundation model (Microsoft, Nature 2025), which successfully performs autoregressive rollout over 40+ steps at 1.3B parameters. Specific code-level fixes are proposed, ordered by expected impact.

---

## Diagnosis

### Root Cause 1: LayerNorm in the Recurrent Dynamics Path Bounds Delta Magnitude

**Severity: Critical**

The dynamics model (Section 6 of the architecture README) uses post-norm in both the cross-attention block (6a) and the self-attention mixing block (6c). Post-norm applies `LayerNorm(x + residual)`, which rescales the *output* to approximately unit variance per token.

At step k, the dynamics computes `latent_{k+1} = latent_k + delta_k`. If `latent_k` has grown to magnitude ~10 after accumulating several deltas, but `delta_k` is always bounded to ~1 by the internal LayerNorms, the relative perturbation per step is ~10% and shrinking. The predictions converge to a fixed point — the model literally cannot keep up with its own trajectory.

Aurora's approach is structurally different: its backbone (a 48-layer 3D Swin Transformer U-Net) processes the full state as a single non-recurrent forward pass. There is no accumulation of bounded deltas. All internal LayerNorms operate within a single call, not across recurrent steps.

### Root Cause 2: No Temporal/Step Encoding in the Dynamics Model

**Severity: Critical**

Aurora's backbone receives two temporal signals at every forward pass: a Fourier-encoded lead-time embedding (hours ahead, passed through an MLP, added to every token) and an absolute-time embedding. Additionally, Aurora's LoRA system selects different adaptation weights per rollout step.

The current dynamics model has zero temporal awareness. Every call to `Dynamics(latent_k, u_curr, u_fut)` is structurally identical from the model's perspective — it cannot distinguish step 1 from step 15. If the latent hasn't changed much (because of Root Cause 1) and the actuators are similar across adjacent windows, the model receives near-identical inputs at every step and produces near-identical outputs. Copy behavior is the expected result.

### Root Cause 3: EMA Target Creates a Moving Attractor in Latent Space

**Severity: High**

Aurora does not use EMA targets. It predicts in physical observation space and compares against ground truth directly.

The current architecture trains the dynamics to match `Encode_ema(target_k)`, but the EMA encoder slowly tracks the online encoder. The signal loss (L_sig) pushes the dynamics output toward the EMA representation, while the encode loss (L_enc) pushes the EMA representation toward the online encoder's output. If the online encoder produces smooth, slowly-changing representations (which the reconstruction loss incentivizes), then `Encode_ema(target_1)` and `Encode_ema(target_2)` are also smooth and similar. The dynamics model sees targets that genuinely are close together — learning small deltas correctly minimizes the loss. The model learns the wrong thing because the target space has been compressed.

### Root Cause 4: No History in the Dynamics Model

**Severity: High**

Aurora's patch embeddings have shape `(D, 1, T=2, P, P)` — the model always sees two consecutive timesteps, providing implicit velocity/finite-difference information.

The current dynamics model sees only `latent_k` at each step. At step 1, it receives `L_0` (the encoded 500 ms context). `L_0` encodes a window — it cannot distinguish "stable plasma, now evolving" from "plasma already changing rapidly." Without the previous latent, the model cannot infer a rate of change and defaults to conservative (small delta) predictions.

### Root Cause 5: Actuator Degeneracy Under Slowly Varying Control

**Severity: Moderate**

The "no query residual" design in Section 6a is well-motivated — `act_info` lives entirely in the span of actuator value vectors, preventing identity copying through cross-attention. However, if actuator signals change slowly (typical in tokamak control — the PCS does not change beam power every millisecond), then actuator tokens at step k and step k+1 are nearly identical. The fusion MLP receives nearly identical actuator conditioning at every step and must produce different deltas from `FusionMLP([same_act_info; slowly_changing_latent])`, which is very hard for a 2-layer MLP.

---

## Proposed Fixes

### P0 — Critical (implement together as a single experiment)

#### Fix 1: Pre-Norm in Dynamics Blocks

Switch sections 6a and 6c from post-norm to pre-norm. This unbounds the delta magnitude in the residual stream.

```python
# Post-norm (current — broken for recurrence):
x = LayerNorm(x + attn(x))           # bounds the OUTPUT

# Pre-norm (correct for recurrence):
x = x + attn(LayerNorm(x))           # bounds the INPUT to attention only
```

The residual stream can now carry signals of any magnitude. The LayerNorm controls what goes into the attention/FFN, not what comes out. This is the same principle that makes GPT-style autoregressive Transformers work over thousands of steps.

**Where to change:** `CrossAttentionDynamics` — all cross-attention layers (6a), all self-attention layers (6c), and any FFN blocks in the dynamics path.

#### Fix 2: Add Step/Time Embedding to the Dynamics Model

Fourier-encode the rollout step index (or absolute time) and inject it into the dynamics model.

```python
step_embed = MLP(fourier_encode(k))                          # (B, d_model)
delta = FusionMLP([act_info; latent_k; step_embed.expand(B, N_L, d_model)])
```

This gives the model a critical signal: "the world should be different now than it was at step 0." The FusionMLP input dimension increases from `2 * d_model` to `3 * d_model`.

**Where to change:** `CrossAttentionDynamics.__init__` (add Fourier embedding + MLP), `CrossAttentionDynamics.forward` (accept step index, concatenate embedding), `FusionMLP` (adjust input dimension), and the rollout loop (pass step index).

### P1 — High Priority (add if P0 alone does not resolve the failure)

#### Fix 3: Rebalance Losses — Downweight L_sig, Upweight L_rol

The latent-space signal loss (L_sig, Section 9c) pushes the dynamics toward the EMA-encoded target, which is subject to the compression problem described in Root Cause 3. The rollout loss (L_rol, Section 9e) compares decoded AE tokens against ground truth — this is closer to Aurora's observation-space loss.

```
# Current:  L = 0.1·L_enc + 1.0·L_rec + 1.0·L_sig + 1.0·L_dlt + 1.0·L_rol
# Proposed: L = 0.1·L_enc + 1.0·L_rec + 0.1·L_sig + 1.0·L_dlt + 2.0·L_rol
```

Alternatively, remove L_sig entirely and rely on L_dlt + L_rol to supervise the dynamics.

**Where to change:** Loss weight configuration. No architectural changes.

#### Fix 4: Add 2-Step History Buffer to the Dynamics Model

Feed both `latent_k` and `latent_{k-1}` to the dynamics model, providing implicit velocity information.

```python
L_prev = L_0                           # initialize with encoded context
for k in range(N_steps):
    delta = Dynamics(L_k, L_prev, u_curr, u_fut, step=k)
    L_{k+1} = L_k + delta
    L_prev = L_k
```

The fusion MLP becomes:

```python
delta = FusionMLP([act_info; latent_k; latent_prev; step_embed])
# Input dimension: 4 * d_model
```

**Where to change:** `CrossAttentionDynamics.forward` (accept `latent_prev`), `FusionMLP` (adjust input dimension to `4 * d_model`), and the rollout loop (maintain `L_prev` buffer).

### P2 — Refinement (for accuracy improvement after rollout is unblocked)

#### Fix 5: Replace EMA Target with Frozen/Detached Online Encoder

Replace the EMA encoder with the online encoder run in eval mode with `torch.no_grad()`. This eliminates the co-adaptation between the target representation and the prediction pathway.

Alternatively, take a frozen snapshot of the online encoder at the start of each epoch and use it as the target encoder for that epoch.

**Where to change:** Target computation in the training loop. Remove EMA update step. Replace `Encode_ema(target_k)` with `Encode_online(target_k).detach()`.

#### Fix 6: Gated Query Residual in Cross-Attention (6a)

Add a learned gate that allows a small amount of state information to flow into the dynamics pathway through the cross-attention, breaking the actuator degeneracy when control signals are slowly varying.

```python
gate = sigmoid(W_gate @ latent_k)     # per-token scalar in [0, 1]
act_info = (1 - gate) * cross_attn_output + gate * latent_k
```

Initialized with `W_gate` bias = -3 so the gate starts near zero (minimal state leakage), and the model can learn to increase it where needed.

**Where to change:** `CrossAttentionDynamics` — add gating layer after cross-attention output in Section 6a.

---

## Experimental Protocol

### Experiment 1: P0 Fixes (Pre-Norm + Step Embedding)

1. Implement Fix 1 and Fix 2 together.
2. Train for 50 epochs with rollout ramp from 1 to 8 steps.
3. **Success metric:** At step 8+, the predicted signals should show qualitatively different temporal structure from step 1. Specifically, `||delta_8|| / ||delta_1||` should remain in [0.3, 3.0] rather than decaying to near zero.
4. Monitor per-step delta norms throughout training to verify they do not collapse.

### Experiment 2: P1 Fixes (Loss Rebalance + History)

If Experiment 1 shows improved but insufficient dynamics:

1. Add Fix 3 (loss rebalance) and Fix 4 (history buffer).
2. Train for 50 epochs with rollout ramp from 1 to 16 steps.
3. **Success metric:** Decoded predictions at step 12+ should track ground-truth temporal evolution (not just amplitude) as measured by time-lagged cross-correlation > 0.5.

### Experiment 3: P2 Fixes (Target Encoder + Gated Residual)

If Experiments 1–2 succeed in producing non-trivial rollouts but accuracy plateaus:

1. Add Fix 5 (frozen target encoder) and/or Fix 6 (gated residual).
2. Train for full curriculum (16 rollout steps, 80+ epochs).
3. **Success metric:** Reduction in rollout RMSE at steps 8–16 relative to Experiment 2.

---

## Key Lessons from Aurora's Codebase

| Aurora Design Choice | Current Architecture | Gap |
|---|---|---|
| Non-recurrent backbone (single forward pass for full state) | Recurrent dynamics with LayerNorm accumulating bounded deltas | Post-norm bounds delta magnitude across steps |
| T=2 history input (3D conv patches over 2 timesteps) | Single-timestep latent input | No velocity information available |
| Lead-time + absolute-time Fourier embeddings | No temporal signal to dynamics | Steps are indistinguishable |
| Per-step LoRA adaptation in backbone | Shared dynamics weights across all steps | Cannot learn step-dependent corrections |
| MAE loss in observation space against ground truth | MSE loss in latent space against EMA target | Target space compressed; loss metric squared |
| Modulation heads for residual prediction (`pred + (1 + mod) * prev`) | Additive residual (`latent_k + delta_k`) | Less expressive residual parameterization |
| Pushforward trick + replay buffer for long rollouts | Full backprop through rollout chain + teacher forcing | Memory-limited rollout depth |

---

## References

- Bodnar et al. (2025). "A Foundation Model for the Earth System." *Nature*.
  - Repository: https://github.com/microsoft/aurora
  - Key files: `aurora/model/aurora.py` (forward pass), `aurora/rollout.py` (autoregressive rollout), `aurora/model/lora.py` (per-step LoRA), `aurora/model/swin3d.py` (backbone with pre-norm blocks)
- Brandstetter et al. (2022). "Message Passing Neural PDE Solvers." — Pushforward trick for stabilizing autoregressive rollout training.
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." — Per-step LoRA adaptation used in Aurora's rollout fine-tuning.
