# E2E Tokamak World Model — Paper-Grade Fact Sheet

All numbers are artifact-grounded. Two scales are reported side by side:
- **d512 (pilot / method-development, ACTUALLY trained):** the g3fix β-anneal model,
  ckpt `/lustre/orion/fus187/proj-shared/models/e2e_g3fix_anneal/e2e_stage1_beta6.0_step3000.pt`.
- **d1024 / 48L (ece-only scale-up projection — SUPERSEDED):** an early pure scale-up of the
  ece-only pilot (`d_model=1024, n_layers=48`), TOTAL ≈ 837 M. This is **not** the production
  model. The real full-modality production numbers (1.20 B, `n_heads=8`, 4 spectro + video +
  TS) live in `FACT_SHEET_production.md`; the d1024 counts in §5 below are retained only as the
  ece-only-scale-up reference (built at `n_heads=16`, which — being param-independent — does not
  change the count).

Primary artifacts:
- Checkpoint `args` dict + `model_state_dict` (loaded with `weights_only=False`).
- `src/tokamak_foundation_model/e2e/model.py` (`E2EFoundationModel`), `.../e2e/backbone.py`.
- `scripts/training/train_e2e_stage1.py` (`build_configs`, model ctor, opt/scheduler, losses).
- Launcher `scripts/slurm_frontier/train_e2e_stage1_kanneal.sh` + `_kanneal_g3fix_flags.txt`.
- Codec `.pt` files under `/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all/`.

---

## 1. d512 exact facts (from the checkpoint)

**Checkpoint top-level keys:** `model_state_dict, optimizer_state_dict, scheduler_state_dict,
step, val_loss, best_val_loss, best_step, metrics, diagnostics, actuators, args`.
`step=3000`, `val_loss=1.2364`, `best_val_loss=1.3293`, `best_step=500`.

> **Checkpoint-name nuance (verify against artifact):** milestone files are named
> `beta{β-just-completed}_step{step}` (`train_e2e_stage1.py:4414-4417`). `beta6.0_step3000`
> is the head **saved at step 3000, i.e. after the β=6 anchor hold (steps 1500–3000) completed**;
> the *running* anchor-β at step 3000 was already 5.0 (holds `8,6,5,4,3` × 1500 each,
> from `args.spec_descriptor_anchor_beta_holds='8,6,5,4,3'`, `..._hold_steps=1500`). The
> K-anneal Stage-2 warm-starts from this file and **pins anchor-β = 6** thereafter
> (launcher `--spec_descriptor_anchor_beta_holds 6 --spec_descriptor_anchor_beta_hold_steps 100000`).

### Full `args` dict (verbatim)
```
backbone_grad_checkpoint=False  backbone_input_skip=False  batch_size=16
checkpoint_dir=/…/e2e_g3fix_anneal  chunk_duration_s=0.05  collapse_aware_best=False
collapse_aware_lambda=1.0  d_model=512  data_dir=/…/foundation_model
desc_false_death_abort=0.01  device=None  dropout=0.1
fastts_code_class_weight=4.0  fastts_code_pred_hidden=512  fastts_code_pred_layers=2
fastts_code_temperature=1.0  fastts_code_weight_batches=50  fastts_fsq=False  fastts_fsq_codec_dir=''
freeze_backbone_steps=0  freeze_fast_ts_steps=0  freeze_slow_ts_steps=0  freeze_spectro_steps=0
freeze_ts_steps=0  freeze_video_steps=0  freeze_whole_run=False  grad_clip=5.0
history_windows=1  init_checkpoint=/…/e2e_g3fix/e2e_stage1_best.pt  lazy_optimizer_load=False
lengths_cache_dir=/…/foundation_model_meta  log_every=50  loss_norm_beta=0.99  loss_norm_ema=False
loss_priority_spectro=1.0  lr=0.0002  max_files=None  max_steps=7500  min_lr=1e-06
n_heads=8  n_layers=12  no_amp=False  no_amp_val=False  no_video_presence_filter=False
num_workers=4  prediction_horizon_s=0.2  reinit_act_tokenizers=False  resume_checkpoint=None
seam_refine_hidden_ch=16  seed=42
slow_ts_code_class_weight=4.0  slow_ts_code_pred_hidden=512  slow_ts_code_pred_layers=2
slow_ts_code_temperature=1.0  slow_ts_code_weight_batches=50  slow_ts_fsq=False  slow_ts_fsq_codec_dir=''
spec_autoencode=False  spec_code_class_weight=10.0  spec_code_focal_gamma=0.0
spec_code_pred_hidden=512  spec_code_pred_layers=2  spec_code_temperature=1.0  spec_code_weight_batches=50
spec_descriptor=True  spec_descriptor_anchor=True  spec_descriptor_anchor_beta_hold_steps=1500
spec_descriptor_anchor_beta_holds='8,6,5,4,3'  spec_descriptor_dist_beta=8.0
spec_descriptor_hidden=512  spec_descriptor_horizons='2,4'  spec_descriptor_loss='dist'
spec_descriptor_tcol=6  spec_descriptor_transition_weight=5.0  spec_descriptor_weight=6.0
spec_flow_base_ch=64  spec_flow_freq_pe_ch=0  spec_flow_lambda=1.0  spec_flow_residual_anchor=False
spec_flow_steps=6  spec_flow_time_pe_ch=0  spec_freq_stem=False  spec_freq_stem_from_codec=False
spec_freq_stem_hidden=128  spec_fsq=True  spec_fsq_codec_dir=/…/fsq_resid_p8_all
spec_generative=False  spec_input_cond=False  spec_input_feat=False  spec_inv_stem=False
spec_inv_stem_ch=64  spec_mae_lambda=1.0  spec_mask=False  spec_mask_hidden=64  spec_mask_lambda=0.0
spec_mask_loss='dice'  spec_maskgit=False  spec_maskgit_decode_steps=10  spec_maskgit_decode_temp=0.5
spec_maskgit_dim=512  spec_maskgit_heads=8  spec_maskgit_layers=4  spec_mode_band_hi_khz=40.0
spec_mode_band_lo_khz=5.0  spec_mode_band_weight=1.0  spec_ordinal_eps=0.0  spec_per_bin_loss=False
spec_per_bin_weight_clamp=10.0  spec_per_bin_weight_power=1.0  spec_persistence_anchor=False
spec_struct_lambda=0.0  spec_warp_anchor=False  spec_warp_max_bins=8.0
spectro_patch_f=8  spectro_patch_t=16  spectro_refine_kernel=3  spectro_seam_refine=False
stats_path=/…/foundation_model_meta/preprocessing_stats.pt  step_size_s=0.01
train_shots_yaml=None  use_spectro=['ece']  use_video=[]  val_batch_size=None  val_every=250
val_fraction=0.1  val_max_batches=20  val_shots_yaml=None
video_code_*=…(FSQ video OFF)  video_flow_*=…(OFF)  video_fsq=False  video_generative=False
video_refine_kernel=[1,3,3]  video_resize_conv=False  video_resize_conv_hidden=64
video_seam_refine=False  video_sigma_spatial=False  warmup_s=1.0  warmup_steps=300  weight_decay=0.1
```

### Parameter count — d512 (from the checkpoint `model_state_dict`)
- **TOTAL** = **120,702,212** params (629 tensors; = 120,702,180 trainable/frozen params
  + 32 non-parameter Fourier-frequency buffer elements `backbone.step_cond.{step,time}_freqs`).
- Rebuilding the model on CPU reproduces the state_dict **key-for-key with zero diff**
  and totals **120,702,212** — confirming the counter matches the trained artifact exactly.

| Component (state_dict prefix) | Params | Trainable | Frozen |
|---|---:|---:|---:|
| `backbone` (12× BackboneBlock + step_cond MLP + final_norm) | 39,011,872 | 39,011,840 | 0 (+32 buf) |
| `diag_tokenizers.*` (all 9 diagnostics) | 38,453,568 | 38,453,568 | 0 |
| `diag_heads.ece.codec` (FSQ spectro codec) | **15,601,112** | 0 | **15,601,112** |
| `act_tokenizers.*` (all 7 actuators) | 14,361,088 | 14,361,088 | 0 |
| `diag_heads.*.pred` (all 9 prediction heads) | 10,991,204 | 10,991,204 | 0 |
| `spec_descriptor_heads.ece` | 2,283,368 | 2,283,368 | 0 |
| **TOTAL** | **120,702,180** | **105,101,068** | **15,601,112** |

Per-key detail worth noting:
- `diag_tokenizers.ece` = 28,224,512 (the spectrogram tokenizer dominates the tokenizer bank).
- `diag_tokenizers.filterscopes` = 10,064,192; `diag_heads.filterscopes.pred` = 10,053,953
  (fast-TS conv-stem + transformer-width MLPs); `diag_heads.ece.pred` = 919,296.
- Each `act_tokenizers.{ech_power,rmp}` = 2,461,184; `{gas_flow,gas_raw}` = 2,256,384;
  `{pin,beam_voltage,tin}` = 1,641,984 (scales with `n_channels`).
- Each continuous slow-TS head (`ts_*`, `cer_*`, `mse`) `pred` = 2,565 (a single Linear).

**Frozen FSQ codec — how freezing is applied & where it lives:**
`load_frozen_codec` (`e2e/quantizers/spectro_codec.py:117-135`) calls `codec.eval()` and
`p.requires_grad_(False)` on every codec param. The codec is stored as a submodule
(`SpectrogramCodeHead.codec`, `output_heads.py:1460`), so **its params ARE inside this
checkpoint's `model_state_dict`** (`diag_heads.ece.codec.*` = 15,601,112 = byte-identical to
the standalone `spectro_codec_ece.pt` `ae` state-dict count). They are excluded from the DDP
reducer because `requires_grad=False`; AdamW receives them but never updates them (0 gradient).
So they are loaded **as part of the model checkpoint**, not separately at eval.

---

## 2. Architecture config

Backbone = **pre-norm Transformer encoder** (`SharedBackbone`, `history_windows=1` so the
multi-window path is inactive). Confirmed pre-norm from `BackboneBlock.forward`
(`backbone.py:92-100`): `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`.

| Field | d512 (pilot) | d1024/48L (production) | Source |
|---|---|---|---|
| d_model | 512 | 1024 | args / build |
| n_layers | 12 | 48 | args / build |
| n_heads | 8 | 16 | args / build |
| head_dim | 64 | 64 | derived (d_model/n_heads); attention param count is n_heads-independent |
| MLP hidden | 2048 (mlp_ratio=4.0) | 4096 | `BackboneBlock` `hidden=int(d_model*mlp_ratio)`; `mlp_ratio` default 4.0 |
| Norm | LayerNorm, pre-norm | same | `backbone.py:78,82,94-97` |
| Activation | GELU (attn + MLP + step-cond MLP) | same | `backbone.py:87`, `86` |
| Attention | `nn.MultiheadAttention(batch_first=True)`, full (non-causal) self-attn | same | `backbone.py:79-81,95` |
| Dropout | 0.1 (attn + MLP) | 0.1 | args `dropout=0.1` |
| Positional / conditioning | Fourier features of `(step_index, time_offset_s)` → 2-layer MLP → `d_model`, **broadcast-added to all tokens** before block 0 (`StepConditioning`); per-modality tokenizers carry their own learned patch/spatial PE + modality embed | same | `backbone.py:23-64,258-259` |

**Token layout** (single flat backbone sequence; order = `[slow_ts | fast_ts | spectrogram | video | actuators]`,
`build_configs` comment `train_e2e_stage1.py:207-210`). Identical at both scales
(token count is d_model-independent):

| Modality | tokens | notes |
|---|---:|---|
| ts_core_density | 44 | slow_ts: 1 token / channel |
| ts_core_temp | 44 | |
| ts_tangential_density | 10 | |
| ts_tangential_temp | 10 | |
| cer_ti | 48 | |
| cer_rot | 48 | |
| mse | 69 | |
| filterscopes | 80 | fast_ts: n_channels(8) × (window 500 / patch 50)=10 → 80 |
| **ece** | **384** | spectrogram: (freq_bins 512 / F_p 8) × (trunc_t 96 / T_p 16) = 64×6 = 384 |
| **n_diag_tokens** | **737** | (`model.n_diag_tokens`) |
| pin / beam_voltage / tin / ech_power / gas_flow / gas_raw / rmp | 5 each | actuator: n_tokens=5 |
| **n_total_tokens** | **772** | 737 diag + 35 actuator (`model.n_total_tokens`) |

**Actuator conditioning mechanism (g3fix):** actuators enter as **35 sequence tokens**
(7 groups × 5 tokens each), concatenated after the diagnostics. `use_actuator_film=False`
for g3fix (not passed on the command line; `--use_actuator_film` is an available `store_true`
flag, default off — `train_e2e_stage1.py:2646`, launcher never sets it). The FiLM path
(`E2EFoundationModel.actuator_film`, `model.py:626-641`) is therefore **not instantiated**
and contributes 0 params.

---

## 3. Modalities

g3fix uses **9 diagnostics + 7 actuators**. `use_spectro=['ece']`, `use_video=[]`.
**co2, bes, mhr (spectrograms) and tangtv (video) are registered in the code but NOT used
in g3fix** — confirmed: `use_spectro=['ece']` only, `use_video=[]` empty (args + ckpt
`diagnostics`/`actuators` lists).

### Diagnostics (from `build_configs` + ckpt `diagnostics`)
| Short-code | Physical name | Kind | n_channels | Input shape (per window) | Tokenizer |
|---|---|---|---|---|---|
| ts_core_density | Thomson scattering, core density | slow_ts scalar | 44 | (44, 5) [5 samples = 50 ms @ 100 Hz] | `SlowTimeSeriesTokenizer` = `nn.Linear(window_samples→d_model)`, 1 token/channel |
| ts_core_temp | Thomson scattering, core temperature | slow_ts | 44 | (44, 5) | same |
| ts_tangential_density | Thomson scattering, tangential density | slow_ts | 10 | (10, 5) | same |
| ts_tangential_temp | Thomson scattering, tangential temperature | slow_ts | 10 | (10, 5) | same |
| cer_ti | Charge-Exchange Recombination, ion temperature | slow_ts | 48 | (48, 5) | same |
| cer_rot | Charge-Exchange Recombination, rotation | slow_ts | 48 | (48, 5) | same |
| mse | Motional Stark Effect | slow_ts | 69 | (69, 5) | same |
| filterscopes | fast time-series ("fast-TS") | fast_ts | 8 | (8, 500) [500 samples = 50 ms @ 10 kHz] | `FastTimeSeriesTokenizer`: Conv1d stem (k=3) + Conv1d patch (stride=50) + per-token MLP; 10 patches/channel |
| ece | Electron Cyclotron Emission (STFT spectrogram) | spectrogram | 40 (subset of raw 48) | (40, 512, 98→trunc 96) STFT magnitude | `SpectrogramTokenizer`: Conv2d patch (8×16) + spatial PE + learned `missing_token`; freq_stem OFF |

Notes: filterscopes downselected raw 104 → first 8 channels (`data_loader` `channels_to_use=slice(0,8)`);
ece uses first 40 of 48 raw STFT channels (`data_loader.py:321`).

### Actuators (from `ACTUATOR_MODALITIES`, ckpt `actuators`)
Each = `ActuatorConfig(n_tokens=5)`, tokenized by `ActuatorTokenizer` = Conv1d(n_channels→d_model,
kernel=stride=window/5) + learned patch_pos + modality embed. Window = `prediction_horizon_s(0.2)
× 10 kHz = 2000` samples (actuator tokens span the **prediction horizon**, not the input chunk;
`build_configs:192-197`).

| Short-code | Physical group | n_channels |
|---|---|---:|
| pin | Neutral-beam injected power (`pinj`) | 8 |
| beam_voltage | Neutral-beam voltage | 8 |
| tin | Neutral-beam ion torque / `tinj` | 8 |
| ech_power | Electron-cyclotron heating power | 12 |
| gas_flow | Gas-injection flow | 11 |
| gas_raw | Gas-injection raw command | 11 |
| rmp | Resonant magnetic perturbation coil current | 12 |

> Note (`train_e2e_stage1.py:122-123`): `ech_tor_angle`, `ech_pol_angle`, `ech_polarization`
> DROPPED 2026-07-14 (identically zero corpus-wide → dead inputs). g3fix has **7** actuators.

---

## 4. FSQ codecs

Only the **spectro/ece codec is active in g3fix** (`spec_fsq=True`; `slow_ts_fsq=False`,
`fastts_fsq=False`, `video_fsq=False`). Slow-TS and fast-TS use continuous regression heads
(`SlowTimeSeriesHead` = Linear; `FastTimeSeriesHead` = deconv). The slow-TS/fast-TS FSQ codecs
exist on disk (`.../fsq_slowts_codecs/`, `.../fsq_fastts_codec_tok80/`) but are **not loaded**
in this run.

**Active codec — spectro/ece** (`fsq_resid_p8_all/spectro_codec_ece.pt`, cfg verbatim):
| Field | Value |
|---|---|
| FSQ levels L | 16 (`fsq_L`) |
| fsq_dim | 48 |
| n_tokens (codec) | 384 (= (512/8)×(96/16); matches backbone ece token count) |
| codec internal d_model | 256 (independent of backbone d_model) |
| patch size (F_p, T_p) | (8, 16) |
| Fq × Tq | 512 × 96 |
| residual / bg_subtract | **True** (bg_sigma default 8.0) — the whole ece pathway runs in baseline-subtracted "R-space" |
| per_channel | False (all 40 channels folded into one 384-token budget) |
| frozen | Yes (`requires_grad_(False)`) |
| params | 15,601,112 (enc `SpectrogramTokenizer`+freq_stem ON, FSQ bottleneck, dec `SpectrogramOutputHead`) |

The other codec families are the same architecture (`SpectroFSQCodec`) with identical
`fsq_dim=48, fsq_L=16, patch (8,16), d_model=256, bg_subtract=True`; their standalone param
counts (for reference, NOT in the g3fix model): co2 13,241,780 · bes 14,028,224 · mhr 13,372,854.
Codec internal dim is fixed at 256 regardless of backbone d_model, so **the frozen codec is
identical (15,601,112) in both the d512 and the d1024/48L model**.

---

## 5. d1024/48L build + count (ece-only scale-up — SUPERSEDED by FACT_SHEET_production.md)

> **SUPERSEDED.** This section is the early **ece-only** scale-up projection (TOTAL 837,786,340,
> one frozen codec, no video, built at `n_heads=16`). The actual production model is
> full-modality at **1,203,520,250** params with **`n_heads=8`** (head_dim 128) — see
> `FACT_SHEET_production.md`. The numbers below are correct for the ece-only build but are
> **not** the production model.

Built on CPU with `build_configs(...)` + `E2EFoundationModel(...)` exactly as
`train_e2e_stage1.py` does, changing **only** `d_model=512→1024`, `n_layers=12→48`,
`n_heads=8→16` (head_dim held at 64). Same `use_spectro=['ece']`, same frozen codec, same
patch sizes (8,16), same descriptor head (horizons 2,4). **Model constructed cleanly**
(loaded the frozen codec, no shape errors); token layout identical (772 total / 737 diag).

- **TOTAL = 837,786,340** params
- **TRAINABLE = 822,185,228**
- **FROZEN = 15,601,112** (the frozen ece codec, unchanged)

| Component | d1024/48L Params | Trainable | Frozen |
|---|---:|---:|---:|
| `backbone` (48 blocks) | 609,082,368 | 609,082,368 | 0 (+32 buf) |
| `diag_tokenizers.*` | 144,003,392 | 144,003,392 | 0 |
| `diag_heads.*.pred` | 38,089,828 | 38,089,828 | 0 |
| `act_tokenizers.*` | 28,722,176 | 28,722,176 | 0 |
| `diag_heads.ece.codec` (FSQ) | 15,601,112 | 0 | 15,601,112 |
| `spec_descriptor_heads.ece` | 2,287,464 | 2,287,464 | 0 |
| **TOTAL** | **837,786,340** | **822,185,228** | **15,601,112** |

> **Assumption flagged:** the d1024/48L number is a **pure scale-up of the current g3fix design**
> — every non-arch knob (codecs, patch sizes, descriptor config, spectro=ece-only, no video,
> actuators-as-tokens) held fixed; only `d_model/n_layers/n_heads` changed. Any production run
> that also flips a design flag (e.g. adds co2/bes video, turns on `spec_freq_stem`, or FiLM)
> will differ from this count.

### Side-by-side parameter table
| | d512 (pilot, trained) | d1024/48L (production, built) |
|---|---:|---:|
| **TOTAL** | **120,702,180** | **837,786,340** |
| **TRAINABLE** | **105,101,068** | **822,185,228** |
| **FROZEN (FSQ codec)** | **15,601,112** | **15,601,112** |
| backbone | 39,011,840 | 609,082,368 |
| diag_tokenizers (all) | 38,453,568 | 144,003,392 |
| diag_heads pred (all) | 10,991,204 | 38,089,828 |
| act_tokenizers (all) | 14,361,088 | 28,722,176 |
| spec_descriptor_heads | 2,283,368 | 2,287,464 |
| diag_heads codec [FROZEN] | 15,601,112 | 15,601,112 |

(Reproduce: `eval_runs/paper_facts/build_and_count.py`.)

---

## 6. Training setup

| Field | Value | Source |
|---|---|---|
| Optimizer | `torch.optim.AdamW(model.parameters(), lr, weight_decay)` | `train_e2e_stage1.py:3629-3633` |
| betas / eps | **PyTorch defaults** — betas=(0.9, 0.999), eps=1e-8 (NOT overridden in code) | ctor call (only lr + weight_decay passed) |
| weight_decay | 0.1 | args |
| base lr | 2e-4 | args `lr=0.0002` |
| min lr | 1e-6 | args `min_lr` |
| LR schedule | `SequentialLR`: `LinearLR(start_factor=1e-3→1.0, total_iters=warmup_steps)` then `CosineAnnealingLR(T_max=max_steps−warmup_steps, eta_min=min_lr)` | `_build_scheduler`, `train_e2e_stage1.py:2204-2213` |
| warmup_steps | g3fix Stage-1: 300; K-anneal Stage-2: 300 | args / launcher |
| Cosine T_max retarget on resume | Yes — cosine `T_max` re-set from current `--max_steps` on resume; opt lr synced from `scheduler.get_last_lr()` (PyTorch SequentialLR lr-sync bug fix) | `train_e2e_stage1.py:3851-3884` |
| grad clip | 5.0 (`grad_clip`) | args |
| batch_size (per rank) | 16 | args / launcher |
| global batch | 16 × nodes (1 rank/node) → e.g. **128** at `-N 8` | launcher `--ntasks-per-node=1`, `-N 8` |
| precision | **bf16 autocast**, forward-only; **no GradScaler** (bf16 has fp32 range) | `train_e2e_stage1.py:3643-3654`, `2093` |
| DDP | `DistributedDataParallel`, **`find_unused_parameters=False`** (default) | `distributed.py:60-80` |
| grad checkpointing | backbone GC OFF (`backbone_grad_checkpoint=False`); **rollout GC** every 10 steps in K-anneal (`--rollout_grad_checkpoint_every 10`) | args / launcher:60 |
| hardware | Frontier, AMD **MI250X** (4/node = 8 GCDs/node, each a separate GPU); launcher uses **1 rank/node**, `--gpus-per-task=1 --gpu-bind=closest` → 1 GCD used per node | `_frontier_common.sh` header, launcher SBATCH |
| ranks | RANK=SLURM_PROCID, LOCAL_RANK=SLURM_LOCALID, WORLD_SIZE=SLURM_NTASKS | `_srun_rank_wrapper.sh:10-12` |
| seed | 42 | args / launcher |
| num_workers | 4 | args / launcher |

> **Assumption flagged:** production `-N 8` (per memory + launcher usage comment) gives 8 ranks →
> global batch 128. The launcher's default `#SBATCH -N 1` is overridden at submit time
> (`sbatch -N 8 …`). GCDs-per-node = 8 physically, but this job pins **1 GCD/node**.

---

## 7. Training curriculum / stages

**Stage 1 (pretraining, produced `beta6.0_step3000`):** single-step next-window prediction
(`history_windows=1`, `--k_rollout` OFF). `max_steps=7500`, anchor-β annealed `8→6→5→4→3`
(1500 steps each). Predicts the next window (see §8 windowing). This is the warm-start source.

**Stage 2 = K-anneal rollout fine-tune** (`train_e2e_stage1_kanneal.sh`, `--k_rollout` ON):
- Curriculum **K ∈ {10, 20, 40, 80}** (`--curriculum_Ks`), **block_steps=5000** each (→ max_steps 20000).
- **tf_anneal_steps=4000**: scheduled sampling — GT-fed → free-running by step 4000 within block 0.
- **anchor-β pinned at 6** for the whole run (`--spec_descriptor_anchor_beta_holds 6
  --..._hold_steps 100000`).
- Optional **Lever #1** per-block dataset-horizon ladder `K*0.05+0.2` (K=10→0.7, 20→1.2, 40→2.2,
  80→4.2 s), off by default; `--stop_at_step` block segmentation keeps the one-cosine LR intact.
- Warm-starts from `beta6.0_step3000`; auto-resume/chain via `latest.pt`.
- **Feedback between steps:** ece code-path fed back as codec-decoded state (option to
  `--feedback_normalize`), continuous TS fed back directly (rollout driver, `train_e2e_stage1.py:1949-1978`).

**Loss functions** (`compute_step_loss`, `train_e2e_stage1.py:1190-1684`).
Per-modality loss, summed into `total_loss` (with optional EMA loss-norm; g3fix `loss_norm_ema=False`
→ **plain unweighted sum** of per-modality losses + the descriptor term):

- **Continuous slow-TS (ts_*, cer_*, mse):** masked MAE (`masked_mae`) — masking on dead
  mse/cer channels via the per-modality mask (`SlowTimeSeriesHead`, not FSQ). `loss = mae`.
- **fast-TS (filterscopes):** continuous head; `loss = mae` (masked). (Args carry FSQ fast-TS
  knobs but `fastts_fsq=False`.)
- **ece spectrogram (FSQ code path):** **class-weighted cross-entropy** over the frozen codec's
  per-dim FSQ codes (`SpectrogramCodeHead.code_logits` → `(B, n_tok, dim, levels)`,
  `F.cross_entropy` with per-(dim,level) class weight `spec_code_class_weight=10.0`, cap-normalized
  so background is down-weighted). `spec_code_focal_gamma=0.0` (off). The argmax-decoded
  reconstruction is scored as a logging-only MAE (no gradient — frozen decoder). Ordinal-eps off.
- **ece spectrogram descriptor head (auxiliary, `spec_descriptor=True`):** forecasts the
  shift-stable 5–40 kHz band-power **mode descriptor** at horizons **t+2 and t+4**
  (`spec_descriptor_horizons='2,4'`). Loss = distribution-CE over frequency
  (`spec_descriptor_loss='dist'`, target softmax temperature `spec_descriptor_dist_beta=8.0`),
  active-weighted by target mode prominence and **transition-overweighted ×5**
  (`spec_descriptor_transition_weight=5.0`) on onset/death flips; **persistence anchor**
  `pred_logit = anchor·β + head_residual` (`spec_descriptor_anchor=True`, head zero-init → starts
  at persistence, learns only drift; β = the pinned anchor-β=6 in Stage-2). Multi-horizon terms
  averaged, then added as `spec_descriptor_weight=6.0 × d_loss`.
- **Total:** `total_loss = Σ_modalities loss + 6.0·descriptor_loss`. In K-rollout, the total is
  **averaged over the K rollout steps** (`train_e2e_stage1.py:1978`, `total_loss/K`).
  Per-modality weighting via `loss_priority_spectro=1.0` (i.e. no up/down-weight) since
  `loss_norm_ema=False`.

---

## 8. Data pipeline

| Field | Value | Source |
|---|---|---|
| Machine / source | **DIII-D tokamak shots** (extensible to other devices) | `docs/ResearchPlan.MD:4,102`; `prepare_data.py:56` tree `'D3D'`; `multi_file_dataset.py:562` "DIII-D dataset (~7900 shots)" |
| Shot files on disk | 8753 `*_processed.h5` in `/…/foundation_model` | `ls | wc -l` |
| Train / val split | **train 7878 / val 875** (val_fraction 0.1, seed 42, random glob-split; no shot YAML) | `resolve_shot_files` run directly — exact |
| Input window (chunk) | `chunk_duration_s=0.05` (50 ms) → predicts next window | args |
| Prediction horizon (model) | `prediction_horizon_s=0.2` (200 ms) — sets the actuator-token span + rollout target reach | args, `build_configs:192-197` |
| Step size | `step_size_s=0.01` (10 ms window stride) | args |
| Warm-up skip | `warmup_s=1.0` — skips first 1 s / shot (plasma ramp-up); NOT the LR warmup | args |
| Rollout dataset horizon | default `max(curriculum_Ks)·0.05 + 0.2` (K=80 → 4.2 s); per-block via Lever #1 | launcher:16-23 |
| Slow-TS sample rate | 100 Hz → 5 samples / 50 ms window | `SLOW_FS=100.0`, `train_e2e_stage1.py:129` |
| Fast-TS sample rate | 10 kHz → 500 samples / window | `FAST_FS=10_000.0`, `:130` |
| STFT (spectrograms) | `n_fft=1024`, `hop=256`, target_fs `500e3` (500 kHz), Hann window, `center=True`; **DC bin dropped** → `freq_bins = n_fft//2 = 512`; time_frames = round(0.05·500000/256) = 98 (codec Tq truncated to 96) | `data_loader.py:211-213,310,1144,1173-1174`; `train_e2e_stage1.py:161-173` |
| Preprocessing / standardization | per-signal `log_standardize` (STFT mag) from `preprocessing_stats.pt`; stats hold `raw`, `log`, and (for STFT modalities mhr/ece/co2) **`log_per_bin`** entries | `data_loader.py:313-330`; `preprocessing_stats.pt` keys |
| ece raw channels | 48 → first 40 used (`channels_to_use=slice(0,40)`) | `data_loader.py:316-322` |

---

### Reproducibility
- Counting/build script (kept): `eval_runs/paper_facts/build_and_count.py`
  (sources env via `scripts/slurm_frontier/_frontier_common.sh`, runs on CPU, no GPU/training).
- No repo code was modified; no checkpoints copied; the running chain/cache untouched.
