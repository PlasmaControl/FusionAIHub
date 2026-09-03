# E2E Tokamak World Model — Model & Training Summary (paper reference)

_Artifact-grounded. Every number was extracted from checkpoints / model code or obtained by
building the model and counting — not recalled. The d512 pilot rebuild reproduces its
checkpoint `state_dict` key-for-key; the d1024/48L production numbers are from an actual CPU
build with all six real FSQ codecs loaded (zero projections). Raw detail:
`eval_runs/paper_facts/FACT_SHEET.md` (pilot) and `eval_runs/paper_facts/FACT_SHEET_production.md`
(production); reproducible counters `build_and_count.py` / `build_and_count_production.py`._

**Two configurations, reported side by side:**
- **d512 pilot — reduced-modality METHOD-DEVELOPMENT run (trained).** ece-only spectrogram,
  no video. Used to develop the rollout methodology cheaply; it is **not** the production
  model. Checkpoint: `models/e2e_g3fix_anneal/e2e_stage1_beta6.0_step3000.pt`.
- **d1024 / 48L — FULL-modality PRODUCTION (training now, from-scratch rollout-native).**
  4 spectrograms + split video + fast-TS + slow-TS. Trained by
  `scripts/slurm_frontier/train_e2e_stage1_d1024_48L.sh` (`ROLLOUT_NATIVE=1`); live checkpoint
  `models/e2e_d1024_rollout_native/e2e_stage1_latest.pt`.

---

## 1. Overview

A multimodal, actuator-conditioned **world model** for the DIII-D tokamak. It ingests a 50 ms
window of many heterogeneous diagnostics plus seven actuator modalities (70 channels) and autoregressively
forecasts the next window. High-dimensional modalities (spectrograms, video) are modeled as
**discrete FSQ codes** with categorical prediction on top of **frozen FSQ codecs**; the
low-dimensional profile/scalar time-series (Thomson, CER, MSE, filterscopes) are modeled
**continuously**. The world model learns dynamics in a fixed discrete/continuous latent space.

---

## 2. Parameter counts (verified — built & counted)

| | d512 pilot (ece-only, TRAINED) | d1024 / 48L production (full-modality, TRAINING) |
|---|---:|---:|
| **Total** | **120,702,180** | **1,203,520,250** |
| **Trainable** | **105,101,068** | **1,145,387,460** |
| **Frozen** (FSQ codecs) | 15,601,112 (ece only) | 58,132,790 (4 spectro + 2 video) |
| Backbone (Transformer) | 39,011,840 | 609,082,368 |
| Spectrogram tokenizers | 28,224,512 (ece) | 414,800,000 (ece+co2+bes+mhr) |
| Video tokenizers | — | 3,000,000 (upper+lower divertor) |
| Fast-TS tokenizer + head (filterscopes) | ~20,118,000 | 73,800,000 |
| Slow-TS tokenizers + heads (7) | ~0.19M | 370,000 |
| Actuator tokenizers (7) | 14,361,088 | 28,722,176 |
| Spectrogram descriptor heads | 2,283,368 (ece) | 9,150,000 (×4) |
| Spectrogram FSQ code heads | ~0.92M (ece) | 4,730,000 (×4) |
| Video FSQ code heads | — | 1,770,000 (×2) |
| Frozen spectro codecs | 15,601,112 (ece) | 56,240,000 (ece 15.60 / bes 14.03 / mhr 13.37 / co2 13.24) |
| Frozen video codecs | — | 1,890,000 (upper + lower, ~0.94M each) |

FSQ codecs are frozen submodules (internal d_model=256, independent of backbone width) — their
counts are identical whether embedded in a checkpoint or the standalone `.pt` (verified
byte-identical). `n_heads` (8→16) does not change the count (attention is head-count-independent
at fixed d_model). Production per-component counts are **exact** (all codecs exist on disk).

---

## 3. Architecture

**Backbone.** Pre-norm Transformer encoder, full self-attention (`nn.MultiheadAttention`),
GELU, MLP ratio 4.0, dropout 0.1.
- d512 pilot: **12 layers, 8 heads, head_dim 64**.
- d1024 production: **48 layers, 8 heads, head_dim 128** (launcher `--n_heads 8`, confirmed on the live model; parameter count is head-count-independent).

**Token sequence** (single flat backbone sequence, order
`[slow_ts | fast_ts | spectrogram | video | actuators]`):
- Pilot: **772 tokens** (737 diag + 35 actuator; ece=384, filterscopes=80, mse=69, …).
- Production: **2,524 tokens** (2,489 diag + 35 actuator; four spectrograms + two video streams
  dominate).

**Tokenizers (per modality).** Spectrograms → patch-Conv2d tokenizer (patch = 8 freq × 16 time)
+ spatial PE + modality embedding + 12 residual refinement MLP blocks; video → tube-patch
`VideoTokenizer`; fast-TS → `FastTimeSeriesTokenizer` (Conv1d stem + patch + per-token MLP);
slow-TS/actuators → their own learned tokenizers.

**Conditioning.** Actuators enter as **tokens** (5 each × 7 groups = 35). FiLM is implemented
but **off**. Rollout step-index and absolute time are Fourier-encoded and broadcast-added.

**Prediction heads.** FSQ modalities (spectrograms, video) → categorical **code-logits heads**;
spectrograms additionally carry a persistence-anchored **descriptor head**
(`output = anchor·β + Δ(tokens)`). Continuous modalities (slow-TS, fast-TS) → regression heads.

**Codecs.** FSQ (finite scalar quantization), residual/background-subtracted, **frozen**
(`requires_grad_(False)`). Pilot: ece only. Production: 4 spectro (ece, co2, bes, mhr) + 2 video
(tangtv upper/lower divertor).

---

## 4. Modalities

| Modality | Physical channel | Kind | Pilot | Production | Representation |
|---|---|---|:--:|:--:|---|
| ts_core_density | Thomson scattering | slow-TS scalar | ✓ | ✓ | continuous |
| ts_core_temp | Thomson scattering | slow-TS scalar | ✓ | ✓ | continuous |
| ts_tangential_density | Thomson scattering | slow-TS scalar | ✓ | ✓ | continuous |
| ts_tangential_temp | Thomson scattering | slow-TS scalar | ✓ | ✓ | continuous |
| cer_ti | charge-exchange (ion T) | slow-TS scalar | ✓ | ✓ | continuous |
| cer_rot | charge-exchange (rotation) | slow-TS scalar | ✓ | ✓ | continuous |
| mse | Motional Stark Effect | slow-TS scalar | ✓ | ✓ | continuous |
| filterscopes | fast time-series (8 ch) | fast-TS | ✓ | ✓ | continuous **(oracle-confirmed — FSQ fails stability gate 0.315<0.8)** |
| ece | ECE spectrogram | spectrogram | ✓ | ✓ | FSQ-coded |
| co2 | CO₂ interferometer spectrogram | spectrogram | — | ✓ | FSQ-coded |
| bes | beam-emission spectroscopy spectrogram | spectrogram | — | ✓ | FSQ-coded |
| mhr | Mirnov spectrogram | spectrogram | — | ✓ | FSQ-coded |
| tangtv_lower | tangential TV, lower divertor | video | — | ✓ | FSQ-coded |
| tangtv_upper | tangential TV, upper divertor | video | — | ✓ | FSQ-coded |

**Actuators — 7 modalities / 70 channels total** (each modality → 5 tokens regardless of
channel count → 35 actuator tokens):
| Modality | Channels | Physical |
|---|---:|---|
| pin | 8 | per-beamline NBI injected power (`PINJ`, 8 beamlines) |
| beam_voltage | 8 | per-beamline NBI accel voltage |
| tin | 8 | per-beamline NBI injected torque (`TINJ`) |
| ech_power | 12 | per-gyrotron ECH power |
| gas_flow | 11 | gas-valve flow |
| gas_raw | 11 | gas-valve raw |
| rmp | 12 | RMP (resonant magnetic perturbation) coil currents |

_`pin`/`beam_voltage`/`tin` are three quantities of the same 8-beamline NBI system;
`gas_flow`/`gas_raw` two views of the gas system — so the count of independent physical
actuator systems is smaller than 7. Exact physical grouping to be confirmed for the paper._

---

## 5. Data & Training

- **Source:** DIII-D tokamak shots (D3D tree). **Train 7,878 / val 875** (val_fraction 0.1,
  seed 42).
- **Windowing:** 50 ms input chunk (`chunk_duration_s=0.05`) → next-window forecast; 10 ms
  stride; first 1 s per shot skipped (`warmup_s=1.0`); model horizon 200 ms.
- **Spectrograms:** STFT n_fft=1024, hop=256, Hann, 500 kHz, DC dropped → 512 frequency bins.
- **Preprocessing:** per-signal `log_standardize` with per-frequency-bin statistics for STFT
  modalities (`preprocessing_stats.pt`).

**Single from-scratch rollout-native run.** The production model is trained in one run from
random init — **no single-step pre-training / warm-start**; the K-step rollout objective is
active from the start. A **K-curriculum is advanced upward from K=1** at validation-gated
boundaries (target train K ≈ 10–20, evaluated out to K=80 — K-depth is set by the
horizon-controllability claim, not ladder-completeness), with `block_steps=5000` per stage and
scheduled sampling (teacher-forcing → free over `tf_anneal_steps=4000`). Descriptor anchor-β is
**pinned at 6**. Between rollout steps, feedback is **code-space** (argmax FSQ codes re-embedded)
for the FSQ modalities and **continuous** for the time-series. An asymmetric **drift penalty
(weight 0.5)** on the descriptor-centroid displacement is applied; per-k loss weighting is
**uniform**.

**Optimizer / schedule.** AdamW (lr = 5×10⁻⁴, weight_decay = 0.1); linear warmup (4000 steps) →
cosine annealing to `min_lr = 1×10⁻⁶` over the full 118 k-step horizon (one continuous cosine,
preserved across chained resumes). At global batch 128 the 118 k-step horizon is ≈ 1.76 epochs
of the 8.59 M-chunk training set.

**Precision / parallelism.** bf16 autocast (no GradScaler); DDP (`find_unused_parameters=False`);
gradient checkpointing over rollout steps.

**Batch / hardware.** Per-GPU batch 16 across 8 GCDs → **global batch 128, held fixed for the
whole run** (no mid-run batch shift); OLCF Frontier, AMD MI250X (1 GCD/rank, 8 nodes × 1 rank);
seed 42.

**Loss.** Summed per-modality then averaged over K rollout steps: masked-MAE (continuous
slow-TS/fast-TS, with dead-channel masking for MSE/CER), class-weighted FSQ code cross-entropy
(spectrograms, video), and a 6.0×-weighted spectrogram descriptor loss (multi-horizon t+2/t+4
distribution-matching with a persistence anchor and ×5 transition weighting).

**Lever #1 (production memory management).** The dataset future-horizon is decoupled from
`max(K)` and set per curriculum block (`--rollout_dataset_horizon_s`), with `--stop_at_step`
segmenting the run so the one-cosine LR is preserved; the horizon-specific lengths cache is
pre-built offline. This keeps per-GPU memory tractable as the K-curriculum deepens.

---

## 6. Notes / open items for the methods section

- **d512 pilot is a reduced-modality method-development run, NOT the production model.** It runs
  ece-only spectrogram with no video, used to develop the rollout methodology cheaply.
  Production is the full-modality d1024/48L above, now training from-scratch rollout-native.
- **Production FSQ patch/codec family.** The build uses the residual codec family at patch
  (8,16) → 384 tokens (`fsq_resid_p8_all`). If production adopts the 96-token family
  (patch 32,16), the spectrogram tokenizer + head + codec components shrink — rerun
  `build_and_count_production.py` to update.
- **Audit-conditional rows (oracle audits — endgame item 1): BOTH RESOLVED, both confirm the locked spec.**
  (a) **video loss structure** — **RESOLVED: keep exact-code FSQ cross-entropy on tangtv.** The
  tangtv oracle PASSED on the active stratum for both cameras (lower stability 0.846 /
  persistence 0.821; upper 0.873 / 0.870; ≥ 0.8 gate, no in/out-OOD gap) → the codes are stable
  and persistent → an FSQ code-CE world-model target is well-posed.
  (b) **filterscopes FSQ question** — **RESOLVED: keep filterscopes CONTINUOUS.** The fast-TS
  oracle FAILED the gate (active stability 0.315 < 0.8; persistence 0.171;
  corr(persistence, activity) = −0.986 — codes encode burst realization/phase bits, the same
  failure mode as the spectro modes), so FSQ-coding filterscopes would relocate mode-collapse
  into fast-TS code space.
  Both verdicts came from the pre-written oracle rule (stability ≥ 0.8 + persistence ≫ 0.10 gate,
  measured on the ELM/burst-active stratum). The pooled/quiescent numbers were near-1.0 for both
  modalities; the *active*-stratum split is what discriminates (filterscopes active 0.315 vs
  video active 0.85–0.87).
- **The FSQ-production d1024 has not been trained yet.** The most recent trained d1024
  checkpoint (`e2e_stage1_d1024_p64pe`) is a *different* design (generative patch (64,32) heads,
  no FSQ codecs); it is not this configuration.
- **Checkpoint naming.** `beta6.0_step3000` records the β-hold just completed (β=6 over steps
  1500–3000); the running anchor-β at step 3000 was already 5.0.

---

_Generated 2026-07-17 from artifacts; training pipeline + status updated 2026-07-20. Pilot
d512 = trained (reduced modality); production d1024/48L = full-modality, training now
(from-scratch rollout-native, params built + counted + confirmed live at 1,203.52 M)._
