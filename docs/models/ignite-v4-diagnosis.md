---
title: "IGNITE v4 Rollout Diagnosis"
sidebar_position: 9
---

# IGNITE v4 rollouts: what is wrong, and what to change

**Status:** diagnosis, measured on 2026-09-22 (read-only; no retraining). It covers the pinned
v4 bundle (`nc1/IGNITE-v4` @ `d2f12b82`, dynamics `mskfull` step 3,200) and the 16 complete v4
rollouts on Stellar. It builds on the earlier
[IGNITE Rollout Quality Plan](./ignite-rollout-quality-plan.md), which was written against an
older run. The design itself is in the [IGNITE design note](./ignite.md).

**Update, 2026-09-22 (iteration 2).** Stage 1 is in code: the exact KV cache, the placeholder
clamp, fixes for B1, B3, B5, B6, B8 and B11, and `shot_design simulate` rewritten to score
ensembles against persistence with a null arm ([Simulation](../shot-design/simulation.md)).
The decode-settings sweep and the held-out multi-shot evaluation were not run; the owner
asked for the ideas, not a full evaluation or a new architecture.

## The answer

v4 rollouts do not beat persistence in any modality in decoded space, and their paired
real/proposed differences are dominated by sampling noise, not by the actuators. No pipeline bug
explains this: the metrics, frame indices, seeds and actuators line up (checks below). The causes,
in order of how much they cap the result:

1. **The codecs.** Consecutive real spectrogram frames share about 1 % of their tokens, and
   several decoders do not beat a time-mean baseline. The dynamics model is asked to predict
   codec noise, which caps any dynamics model trained on these tokens.
2. **The training recipe.** The pinned checkpoint is an infiller: fully teacher-forced
   (scheduled sampling 0, `gen_mask_p` 0), unregularised, and selected on masked
   (infilling) cross-entropy. Runs trained this way were already known to roll out flat.
3. **The sampler.** Every position is resampled at temperature 1 on every pass, with 4 passes
   and a greedy confidence order. Once one token differs, the two arms decorrelate. A null edit
   of 0.0015 σ ends as far apart as a real edit.
4. **The actuator path.** One additive vector per frame, z-scored per shot with whole-shot
   statistics. The absolute level is removed, the future leaks in, and there is no trained
   unconditional branch.

Stage 1 below (days, no retraining) makes the evaluation honest and the rollouts cheap. Stages 2
and 3 retrain, which needs the decisions listed at the end.

## What v4 is

| part | v4 |
|---|---|
| codecs | 15 FSQ codecs, levels [8,5,5,5] = 1,000 codes each (~10 bits/token) |
| tokens per 50 ms frame | 1,209: five spectrograms × 192 (79 % of the frame), two tangential-TV views × 108, seven profile signals × 4, filterscopes × 5 |
| frame axis | frame 0 at 1.0 s shot time, 219 frames per shot; `frame_embed` has 100 rows, so 20 seed frames allow at most 80 predicted |
| dynamics | MaskGIT over factorised spatio-temporal blocks (attention within a frame over all 1,209 tokens, then causal attention over time per token position); depth 16, d_model 1024, 16 heads, 300.8 M parameters |
| actuators | 88 channels, 50 ms means z-scored per shot, one `Linear(88 → 1024)` added to every token of its frame; no cross-attention, no lag window |
| objective | cosine-prior masked CE over all 100 frames, seed frames included; scheduled sampling 0; `gen_mask_p` 0; dropout 0; weight decay 0; flat lr 1e-3; AdamW β = (0.9, 0.9); equal weight per modality, so 8/15 of the loss sits on 33 of the 1,209 tokens |
| selection | best masked CE, 2.169 at step 3,200 (generation CE 2.728) |
| data | 8,752 shots, 190000–204999 |
| rollout | each new frame starts fully masked; every position sampled with `torch.multinomial` at T = 1 on every pass; the top-k by p(sampled) revealed on a cosine schedule; Gumbel reveal off (now `SamplerConfig.gumbel`, default 0); no KV cache, so every pass re-encodes the whole past (the exact cache is now opt-in, `rollout(kv_cache=True)`); fp32. The 16 v4 rollouts used 4 passes (the default is 10) |

The v2 production run it replaced used scheduled sampling 0.75 and dropout 0.3 and was selected
on generation CE. v4 turned all three off. The trainer's argv, split and loss history are only
on Frontier, and training ran from an uncommitted tree.

## Evidence

All numbers are medians over the 16 complete v4 rollouts (4 named shots in
`ideate/experiments/showcase4_v4`, 12 batch designs in `ideate/experiments/stellar_1k_v4`),
60 predicted frames from 2.00 s, unless stated.

### Decoded error against persistence

RMSE over the predicted frames divided by the measured standard deviation, for the real-actuator
arm and for holding the last seed frame:

| modality | real arm | persistence |
|---|---|---|
| ts_core_density | 3.44 | 1.27 |
| ts_core_temp | 1.96 | 1.29 |
| ts_tangential_density | 2.33 | 1.86 |
| ts_tangential_temp | 1.91 | 1.19 |
| cer_ti | 1.87 | 1.47 |
| cer_rot | 1.23 | 1.12 |
| mse | 1.40 | 1.33 |
| filterscopes | 2.01 | 1.49 |
| mhr | 1.94 | 1.30 |
| mirnov | 2.52 | 1.27 |
| ece | 3.07 | 1.55 |
| bes | 3.21 | 1.32 |
| co2 | 3.69 | 1.26 |

For BES, CO2 and Mirnov this table was computed with the `np.abs` fold (bug B9). With the sign
kept the gap widens: BES 5.16 against 1.23, CO2 3.99 against 1.13, Mirnov 3.25 against 1.27.
The profiles are worse from the first predicted frame (pooled 0.47 against 0.26), and a constant
seed mean (1.50 over 60 frames) beats the real arm (1.71). The mean of the two arms does not
beat persistence either.

### Tokens

Exact-token persistence of the *measured* codes, `mean(gt[t] == gt[t−L])`:

| modality | L = 1 | 5 | 10 | 20 | 60 |
|---|---|---|---|---|---|
| ece, bes, mhr, co2, mirnov | 0.008–0.024 | 0.011–0.021 | 0.005–0.018 | 0.002–0.017 | 0.009–0.016 |
| ts_core_density / temp | 0.13 / 0.14 | 0.12 / 0.10 | 0.10 / 0.08 | 0.07 / 0.05 | 0.05 / 0.03 |
| ts_tangential_density / temp | 0.44 / 0.41 | 0.35 / 0.34 | 0.31 / 0.29 | 0.26 / 0.26 | 0.21 / 0.21 |
| cer_ti / cer_rot | 0.26 / 0.18 | 0.17 / 0.12 | 0.13 / 0.08 | 0.09 / 0.06 | 0.10 / 0.07 |
| mse | 0.23 | 0.22 | 0.18 | 0.17 | 0.18 |
| filterscopes | 0.09 | 0.08 | 0.07 | 0.07 | 0.04 |

Exact-match accuracy therefore cannot score spectrograms at all. For the profiles, the model
loses to holding the last frame one step ahead (ts_core_density 0.25 against 0.50,
ts_tangential_density 0.50 against 0.875, cer_ti 0.25 against 0.50), as an infilling objective
predicts. Codes do not collapse: the predicted window uses 969–994 distinct codes per spectrogram
against 994–1,000 measured. The model over-disperses instead (spectral std 0.715 against 0.268
measured).

### Paired divergence is a sampling split

Rollout 04132144 (shot 195268) edits `gas_flow[0]` by at most 0.0015 σ. Its two arms stay
token-identical until frames 53–65 and then reach divergence 0.65–1.0 by frame 60 (ece 0.65,
mhr 0.66, bes 0.72, mirnov 0.72, ts_core_density 1.0). Real edits have a median of 0.74 σ. So
the proposed-vs-real divergence records *when* two sampling paths split, not how the plasma
responds. The lower divergence of v2 (0.12–0.35) came from collapsed codebooks (v2's ECE used 153
of 32,768 codes in rollout b0d8f066), not from a better model.

### Codecs

Only 6 of the 15 codec files carry a gate record; the other 9 (ece, mirnov, filterscopes, the
four TS codecs, cer_ti, cer_rot) are last-step checkpoints with none. Every spectrogram codec with
a record fails its own stability gate (0.38–0.51, the gate is 0.8) and persistence gate
(0.21–0.30, the gate is 0.5). co2 reconstructs with nRMSE 0.900, against 0.845 for its time
mean; mse with 0.677, against 0.561 for its time mean, using 5.5 effective codes.

### Checks that found nothing wrong

`tokens/gt` equals the reference cache; the first 20 frames of both arms equal the measurement;
both actuator arms equal the cache; every token id is in [0, 999]; persistence uses frame 19 and
scoring starts at frame 20; a frame-offset scan of real-vs-measured accuracy from −20 to +20
peaks at 0; the codec and dynamics vocabularies are both 1,000. The rollout is causal (at frame
t the model sees `actuators[:, :k0+t+1]`).

### Earlier measurements that point the same way

From earlier generations, on the same code paths: runs without scheduled sampling roll out
nearly flat (temporal activity 0.058, against 0.125 at 0.75); the actuator effect on the
additive path was 0.26 (donor shot) and 0.24 (zeroed actuators) of the forecast spread; one-step
token accuracy 0.533 against persistence 0.652; bf16 alone moved token accuracy by −0.040 on
average (Frontier job 5218866), which is the chaotic sensitivity of item 3 above.

## Causes, and the test that settles each

| # | cause | test (no retraining unless stated) |
|---|---|---|
| H1 | codec targets are unstable frame to frame; some decoders lose to a time mean | decode the measured tokens and score them against the raw signals: where codec-only error is at or above raw persistence, no dynamics model can beat persistence in decoded space |
| H2 | infill objective, teacher forcing, selection on masked CE | clean-history one-step CE and accuracy against persistence, and 5-frame rollout CE, on the pinned checkpoint |
| H3 | T = 1 sampling plus greedy reveal turns any difference into a split | per design: real vs real with another seed; real vs real with actuators moved 1e-3 σ; real vs proposed. If the second matches the third, the reported effect is noise |
| H4 | one T = 1 sample scored against deterministic persistence | ensemble mean, CRPS or energy score, FSQ-lattice distances in place of exact match |
| H5 | weak actuator conditioning | `--actuator_mode zero / freeze / donor` on v4, judged against the seed spread in decoded space |
| H6 | no regularisation; 300 M parameters on 8.7 k shots | retraining (Stage 2) |
| H7 | 4 passes waste the first pass on 8 modalities and commit 38 % of the spectrogram tokens in the last one | steps {4, 10, 16, 25} × T {1, 0.7} × top-p 0.9 × Gumbel {0, 1, 4.5} × revision rounds, on 3 designs; every knob exists |
| H8 | absent diagnostics are sampled and fed back (co2 is a placeholder in 10 of 16 rollouts, bes in 7; bf1b260c generates 40 distinct frames into an absent camera) | clamp placeholder modalities to their placeholder code on every pass |
| H9 | Stellar inputs may differ from the Frontier training inputs (CERQUICK vs CERAUTO, the 14.4× gas scale, filterscope padding) | `g_enc.py` on Stellar against cache files copied from Frontier |

## Bugs

| id | where | what | status |
|---|---|---|---|
| B1 | `ignite/eval_dynamics.py:298-305` | `rollout_shot` reads an undefined `shot` whenever `IGNITE_ACT_GLOBAL` is set (NameError) and overwrites the caller's `cache["actuators"]` | fixed: `shot` is a parameter and the cache is left alone |
| B2 | shot_design, `_stellar_common.sh`, `ignite_modalities.yaml` | silent fallbacks to the retired v2 generation and its 0.0 s frame origin | fixed (v4 only) |
| B3 | `ignite/eval_dynamics.py:146-185` | `load_model` cannot restore `lag_embed_k` or tell CTF and self-forcing arms apart | fixed: shapes come from the weights, every saved `cfg_*` field is restored |
| B4 | the pinned `dynamics_best.pt` | selected as the best infiller (`best_metric 'masked'`) | Stage 2 |
| B5 | `shot_design/simulate/decode.py` | `freq_axis_khz` is `None` when `band_pool > 0`, which silently gives the full band | fixed: a pooled band sits at its bins' mean frequency |
| B6 | `shot_design/simulate/core.py:1-11, 125-128` | the docstrings say the arms differ only through the actuators; only the random draw counts are matched | fixed: the arms share random numbers (Gumbel-max), and a null arm measures what sampling alone does |
| B8 | `DynamicsConfig`, `train_dynamics.sh`, the mskfull run | three different scheduled-sampling defaults (0.15 over 40 k steps, 0.75 over 2 k, 0.0) | fixed: one default, 0.75 over 2 k steps, in `DynamicsConfig` |
| B9 | `simulate/decode.py:88` and the paper's decode scripts | `np.abs` of signed standardised log-power folded values below the mean onto values above it (BES 54 %, CO2 86 % of in-band values are negative); it inverted the draft's BES "rise" | fixed |
| B10 | `simulate/report.py` and the paper's figure script | `frac_static` computed on the proposed arm only, so a hallucinated placeholder passed the static filter and a frozen proposed arm was dropped | fixed: the report gives both arms; the figures drop a modality only when its *measured* tokens are static (a placeholder), never on an arm |
| B11 | `ignite/dynamics_config.py:182` | `actuator_dim` defaults to 70; v4 needs 88 (`load_model` infers it, and every v4 log prints `actuator_dim 70 -> 88`) | fixed: 88, the actuator spec's width |

## What to change

### Stage 1: no retraining (days, Stellar)

- **Exact KV cache** (done). It is exact because a past frame never attends to the new one:
  spatial attention stays within a frame, temporal attention is causal, and actuators are added
  per frame. Each committed frame's keys and values are cached per layer and the S passes run
  over the 1,209 new tokens only. Measured on the pinned v4 checkpoint (V100S, fp32): logits
  agree with the uncached path to 4e-5 (total variation 6e-6); 3 frames × 10 passes take 4.6 s
  instead of 42.2 s; an 80-frame, 10-pass member takes 106 s at a 16.6 GiB peak, about 30×
  less than the uncached path. `rollout(..., kv_cache=True)`; the default path is unchanged.
- **Clamp placeholders** (H8, done). `rollout(..., hold=placeholders(seed))` keeps a modality
  whose seed codes never change at that code.
- **An evaluation harness that can say no.** Per design, done: `shot_design simulate` runs
  ensembles of real, proposed and null arms and reports CRPS skill against persistence,
  spread/error and effect against noise. Not built: the multi-shot held-out harness with
  climatology and analog baselines. N-member ensembles per arm; decoded-space RMSE,
  CRPS and energy score against persistence, climatology and a nearest-analog shot; seed-spread
  intervals on every paired effect; the null-edit and seed-only controls (H3) reported beside
  every real edit; the codec-only ceiling (H1) per modality.
- **The decode-settings sweep** (H7), plus the Halton reveal order, which replaces the confidence
  order without retraining. Not run.
- **Fix** B1, B3, B5, B6, B8 and B11 (done); recover the v4 trainer's argv, split and loss
  history from Frontier (not done).

Stage 1 will not make v4 skilful, and it is not meant to. It measures how far the codecs and the
recipe are from skill, and it keeps unsupported effect sizes out of the paper.

### Stage 2: the same codecs, a sound recipe (one Frontier run)

- Keep the context clean: mask only the frame being predicted, as MaskViT does, and train with
  scheduled sampling 0.75 (the value measured best on this code), or corrupt a sampled fraction
  of the history and give the model that rate (the discrete analogue of GameNGen's context noise
  and of Diffusion Forcing).
- Dropout 0.3, weight decay, a decaying learning rate, loss weight proportional to token count,
  and early stopping on a rollout metric (decoded CRPS at 10 and 40 frames), not on masked CE.
- Actuators normalised over the dataset, with absolute-level channels, and 10–20 % actuator
  dropout so a conditional-free branch exists for guidance.

This is the cheapest retraining, but H1 still caps it: it cannot beat persistence on modalities
whose codec reconstruction does not.

### Stage 3: the architecture (recommended)

A hybrid in which each signal is predicted in the space where it is stable:

- **Continuous targets for the low-dimensional signals**: the seven profile signals,
  filterscopes, and per-band powers of each spectrogram, predicted directly with a probabilistic
  head (a small diffusion or flow head per token, as in MAR, or a mixture density). Every
  tokamak model that has worked in practice predicts profiles or scalars (Abbate et al. 2021,
  Char et al. 2023, Seo et al. 2024, Degrave et al. 2022).
- **Temporally stable latents for spectrograms and cameras**: the codec retrain with a
  temporal-stability term, gated against time-mean and dataset-mean baselines
  ([codec retrain spec](./ignite-codec-retrain-spec.md)), or continuous latents with a per-token
  diffusion loss.
- **A causal transformer over per-frame latents**, with the KV cache from Stage 1, a lagged
  actuator window entering by cross-attention, and a multi-step rollout loss (Self-Forcing on
  the model's own rollouts, or a latent-consistency loss).
- **Evaluation by CRPS and skill** against persistence, climatology and analog baselines on
  held-out shots, reported with ensembles.

| option | what | for | against |
|---|---|---|---|
| a | Stage 2 on the v4 codecs | one run; no new data path | capped by the codecs (H1) |
| **b** | **the hybrid above** | **predicts physics where tokens are unstable; directly comparable to the tokamak literature; keeps tokens where they work** | **new heads and a new data path for the continuous targets** |
| c | retrain every codec, then the dynamics | keeps the single discrete pipeline | two sequential training campaigns before the first answer |

## Decisions needed

1. **Where to train, and the budget.** Frontier (the fus187 allocation, where the v4 cache and
   trainer live) or Stellar (about 20 shared A100s; the corpus is 16.9 k shots; scratch is 91 %
   full).
2. **Which option** (a, b or c). b is recommended.
3. **How the paper treats rollouts until then.** The current draft reads paired divergence as an
   effect size and claims a BES rise that B9 produced; neither survives the evidence above.

## Literature

- Gupta et al. MaskViT, ICLR 2023 (arXiv 2206.11894): context frames stay unmasked; the mask
  ratio is drawn from U[0.5, 1) on future frames only.
- Bruce et al. Genie, ICML 2024 (arXiv 2402.15391): MaskGIT world model, 25 decode steps per
  frame.
- Valevski et al. GameNGen (arXiv 2408.14837): noise on context latents, with the noise level as
  an input, against autoregressive drift.
- Hu et al. GAIA-1 (arXiv 2309.17080): argmax loops, full-distribution sampling drifts, top-k 50.
- Besnier et al. Halton scheduler for MaskGIT, ICLR 2025 (arXiv 2503.17076).
- EMERALD, a MaskGIT latent world model, ICML 2025 (arXiv 2507.04075).
- Huang et al. Self-Forcing (arXiv 2506.08009); Self-Forcing++ (arXiv 2510.02283).
- Chen et al. Diffusion Forcing (arXiv 2407.01392).
- Li et al. MAR, autoregression without vector quantisation (arXiv 2406.11838).
- Alonso et al. DIAMOND (arXiv 2405.12399); Micheli et al. Δ-IRIS (arXiv 2406.19320).
- Price et al. GenCast (arXiv 2312.15796): ensembles scored by CRPS.
- Hafner et al. DreamerV3 (arXiv 2301.04104).
- Bengio et al. Scheduled sampling (arXiv 1506.03099).
- Abbate, Conlin & Kolemen, Data-driven profile prediction for DIII-D, Nucl. Fusion 61 (2021);
  Char et al., Offline model-based reinforcement learning for tokamak control, L4DC 2023;
  Seo et al., Avoiding fusion plasma tearing instability with deep reinforcement learning,
  Nature 626 (2024); Degrave et al., Magnetic control of tokamak plasmas through deep
  reinforcement learning, Nature 602 (2022).
