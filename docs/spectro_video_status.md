# Spectrogram + Video status (BC-Stage 1 / BC-Stage 2)

Snapshot of the joint Phase B (spectrograms — ECE, CO2, BES) and Phase C
(video — tangtv) tracks. Both modalities now share a single combined
training pipeline: **BC-Stage 1** (single-step) and **BC-Stage 2**
(K-step rollout, teacher-forced delta loss). Stage 2 Extended is still
TS-only — spectro/video plumbing through the free-rollout trainer is
deferred.

Last updated: 2026-05-07. Supersedes the older Phase-C-only chronology
(`phase_c_step1_status.md`); historical notes preserved in §10–§12.

---

## 1. Scope and current shipping state

| Stage | Trainer | TS | Spectrograms | Video | Launcher |
|---|---|---|---|---|---|
| BC-Stage 1 | `train_e2e_stage1.py` | ✓ | ✓ ECE / CO2 / BES | ✓ tangtv | `scripts/slurm/train_bc_stage1.sh` |
| BC-Stage 2 (delta / teacher-forced K=1…10) | `train_e2e_stage2_delta.py` | ✓ | ✓ ECE / CO2 / BES | ✓ tangtv | `scripts/slurm/train_bc_stage2.sh` |
| Stage 2 Extended (free-rollout K=80) | `train_e2e_stage2_extended.py` | ✓ | ✗ | ✗ | `scripts/slurm/train_e2e_stage2_extended.sh` |

irtv was dropped from Phase C scope (see §12.1). Only `tangtv` is in
the live diagnostic list.

---

## 2. Modality contracts (what the dataset emits)

### 2.1 Spectrograms (Phase B)

Computed from raw 1-D signals (ECE, CO2 phase, BES) inside
`data_loader.py::_process_signal` via STFT (`n_fft=1024`,
`hop_length=256`, Hann window). Output per signal is a complex
spectrogram tensor of shape `(C, F, T)`:

| Signal | Channels | F (freq bins) | T (time bins per 50 ms chunk) | Tokens |
|---|---|---|---|---|
| ECE | 32 | 513 | 20 (per spectrogram_tokenizer_plan.md) | 192 |
| CO2 | 1 | 513 | 20 | 96 |
| BES | 64 | 513 | 20 | 192 |

Total spectrogram tokens: **480 per chunk**.

Per-channel presence is recorded as `{name}_channel_mask: (C,) bool`
and per-batch presence as `{name}_valid: (B,) {0,1}`. ECE has the
highest coverage (~94% of shots); CO2 is sparsest (~44%); BES sits in
between (~36%). See `docs/spectrogram_step0_findings.md` for the
empirical distributions.

### 2.2 Video (Phase C)

`MOVIE_CONFIGS` in `data_loader.py`:

```python
MOVIE_CONFIGS = [
    MovieConfig("irtv", ["irtv"], 7, 100, 513, 640),  # not used in BC pipeline
    MovieConfig(
        "tangtv", ["tangtv"], 2, 100, 120, 360,
        channels_to_use=[4, 6],
        n_output_frames=3,
    ),
]
```

`tangtv` post-amendment-2026-05-06: the 7 raw "channels" are 7 optical
filters; only filters 4 and 6 carry plasma signal across all shots.
`channels_to_use=[4, 6]` selects them; `MovieConfig.channels_to_use`
accepts `Sequence[int]` in addition to `slice` for this. The previous
`runs/c_stage1` (trained on the 7-channel config) was deleted; all
later runs use the 2-channel layout.

Per-chunk shape: `(B, C=2, T=3, H=120, W=360)` after subsampling
3 evenly-spaced frames from the 5-frame native window.

Sample dict carries:
- `tangtv` — `(C, T, H, W)` data tensor
- `tangtv_channel_mask` — `(C,)` bool mask of active filters
- `tangtv_valid` — `(B,)` int 0/1 = `channel_mask.any()`

Video tokens: **300 per camera per chunk** (see §3.2).

---

## 3. Tokenizer + output-head designs

### 3.1 Spectrogram (Phase B)

`src/tokamak_foundation_model/e2e/tokenizers/spectrogram.py` —
`SpectrogramTokenizer`. Designed and gated by §5 of
`docs/spectrogram_tokenizer_plan.md`. Output head in
`src/tokamak_foundation_model/e2e/output_heads.py`.

Loss: plain MAE over the channel × frequency × time grid, gated by
`{name}_channel_mask` and `{name}_valid`.

### 3.2 Video (Phase C — tube-patch, post-2026-04-27 reset)

`src/tokamak_foundation_model/e2e/tokenizers/video.py` — `VideoTokenizer`:

* Patch shape `(T_p, H_p, W_p) = (3, 12, 12)` — one tube spans all 3
  input frames, so temporal info is encoded directly in each token's
  content (no separate temporal-attention machinery).
* `Conv3d` with kernel and stride both equal to the patch shape: each
  output element is a learned linear projection of one disjoint patch.
* `(120 / 12) × (360 / 12) = 300` tokens per camera per 50 ms window.
  Each token represents a `2 × 3 × 12 × 12 = 864`-pixel region.
* Per-patch spatial PE (std=0.02), single modality embedding (std=0.02),
  learned `missing_token` of shape `(n_tokens, d_model)` for camera-
  level missing rows.
* Param count: ~928 k.

`VideoOutputHead` in `e2e/output_heads.py`:

* Single `ConvTranspose3d` with the same kernel/stride — exact inverse
  of the patch embedding. No bilinear upsample, no multi-stage cascade.
* Each token reconstructs its own `(C, T_p, H_p, W_p)` region; no
  global mixing. Spatial detail preserved by construction.
* Param count: ~774 k.

Total Phase C add-on: **~1.70 M params**.

Loss masking: `_video_loss_gate(cfg, batch, device) -> (B, C, 1, 1, 1)`
combines `{name}_valid` and `{name}_channel_mask`; the existing
`masked_mae(pred, target, mask)` excludes off-channels and missing-
camera samples once `mask` is the gate.

The (now-superseded) Perceiver-pool design and the reasoning that
forced the reset are preserved in §11.

---

## 4. Token budget and memory

### 4.1 Per-chunk token layout

Diagnostic prefix (BC-Stage 1, full configuration):

```
[ slow_ts | fast_ts | spectro (ECE, CO2, BES) | video (tangtv) | actuators ]
   273        80              480                    300              45
                       <------- 1178 total ------->
```

Compared to TS-only (Phase A) at 398 tokens: **2.96× tokens**, so
attention scales as ~8.8× per layer; FFN as ~2.96×.

Stage 2b is configured identically but at smaller batch.

### 4.2 Stage 1 video-only memory benchmark (job 2725293, A100-PCIE 40 GB)

| Config | Batch | Params | Peak | Step time |
|---|---|---|---|---|
| TS-only (Phase A) | 128 | 9.29 M | 7.15 GB | 0.231 s |
| TS + tangtv | 128 | 11.00 M | 14.60 GB | 0.485 s |
| TS-only (Phase A) | 256 | 9.29 M | 14.04 GB | 0.458 s |
| TS + tangtv | 256 | 11.00 M | 28.78 GB | 0.970 s |

Step-time scaling is 2.10×–2.12×, better than the 3.1× theoretical
attention ceiling because FFN is the dominant per-layer cost at
`d_model=256`. No grad checkpointing needed at TS+video / batch 256.

### 4.3 Full BC-Stage 1 (TS + spectro + video) sizing

The 1178-token configuration has not been microbenchmarked yet. The
launcher comment in `train_bc_stage1.sh` flags this and runs at
`--batch_size 128` (rather than 256) for headroom on Stellar A100 40 GB.
Stage 2b uses `--batch_size 64` because of the K=1…10 rollout
multiplier on top.

---

## 5. Freeze API (BC-Stage 1)

Stage 1 has four orthogonal freeze flags in
`scripts/training/train_e2e_stage1.py`. Each freezes a named module
group until step `N`, then releases all of them. The four groups are:

| Flag | Modules frozen |
|---|---|
| `--freeze_ts_steps N` | `diag_tokenizers.{slow_ts,fast_ts}.*`, `diag_heads.{slow_ts,fast_ts}.*` |
| `--freeze_video_steps N` | `diag_tokenizers.{video}.*`, `diag_heads.{video}.*` |
| `--freeze_spectro_steps N` | `diag_tokenizers.{spectro}.*`, `diag_heads.{spectro}.*` |
| `--freeze_backbone_steps N` | shared backbone (Perceiver layers + actuator tokenizers) |

Default 0 = no freeze = byte-identical to the un-augmented trainer.
Implementation lives at `train_e2e_stage1.py:838-850` (argparse) and
`train_e2e_stage1.py:1118-1121` (the `("ts", N), ("video", N),
("spectro", N), ("backbone", N)` tuple list driving the per-group
freeze loop).

The current BC-Stage 1 launcher uses
`--freeze_ts_steps 5000 --freeze_backbone_steps 5000`, so the freshly-
initialised video and spectrogram modules train freely while the
Phase-A-warm-started TS modules and shared backbone are held fixed for
the first 5 000 steps.

Stage 2b (`train_e2e_stage2_delta.py`) does **not** have these freeze
flags — its training schedule assumes everything trains together
(post-warm-start curriculum on K).

---

## 6. BC-Stage 1 — operational summary

### 6.1 Launcher: `scripts/slurm/train_bc_stage1.sh`

Mirror of `train_e2e_stage1.sh` plus:

* `--use_video tangtv`
* `--use_spectro ece co2 bes`
* `--init_checkpoint runs/e2e_stage1/e2e_stage1_best.pt` (warm-start
  TS + actuator weights from Phase A best; video / spectro modules init
  from scratch via `load_state_dict_explicit` `allowed_missing_prefixes`)
* `--freeze_ts_steps 5000 --freeze_backbone_steps 5000`
* Output: `runs/bc_stage1/`. The Phase A `runs/e2e_stage1/` tree is
  not modified; Phase A Stage 2b chain + Stage 2 Extended are unaffected.

The launcher snapshots the Phase A best at job start
(`runs/e2e_stage1/e2e_stage1_best_bc_stage1_init.${SLURM_JOB_ID}.pt`)
so a future Phase A retraining cannot silently change the warm-start
source.

Auto-resume: if `runs/bc_stage1/e2e_stage1_latest.pt` exists, the
trainer resumes from it (and `--resume_checkpoint` overrides
`--init_checkpoint`).

### 6.2 Trainer additions in `train_e2e_stage1.py`

* Module-level `VIDEO_MODALITIES` and a parallel `SPECTRO_MODALITIES`
  list. Argparse uses `choices=` from those lists.
* `--use_video` / `--use_spectro`: `nargs="*"`, default `[]`. Empty
  defaults reproduce TS-only behaviour byte-for-byte.
* `build_configs(...)` appends a `DiagnosticConfig` per requested
  spectrogram (after fast_ts, before video) and per requested video
  camera (after spectro, before actuators), keeping the diagnostic
  prefix contiguous as required by `rollout.py:149` and Guard G1.
* `load_state_dict_explicit(...)` (in `e2e/checkpoint.py`) replaces
  `model.load_state_dict(state, strict=True)` everywhere: it raises on
  unexpected keys and on missing keys not matched by an
  `allowed_missing_prefixes` entry, so warm-starting from a TS-only
  checkpoint into a BC model works while accidental TS renames still
  fail loudly.

### 6.3 Status

Code-complete. First multi-day run not yet recorded in this doc. See
the active-work entries in `MEMORY.md` (e.g. `feedback_*` and any
`project_bc_stage1_*`) for in-flight observations.

---

## 7. BC-Stage 2 — operational summary

### 7.1 Launcher: `scripts/slurm/train_bc_stage2.sh`

Same multimodal additions as BC-Stage 1:
`--use_video tangtv --use_spectro ece co2 bes`. Init source falls back
through:

1. `runs/bc_stage1/e2e_stage1_best.pt` if present (preferred — keeps
   the BC-Stage-1-trained spectro / video weights);
2. `runs/e2e_stage1/e2e_stage1_best.pt` as Phase A fallback (spectro
   and video then init from scratch via `allowed_missing_prefixes`).

Output: `runs/bc_stage2_delta/`. Hyperparameters: `K_max=10`,
`curriculum_steps=322000`, `batch=64`, delta loss with
cos_weight=0.3 / mag_weight=0.1.

### 7.2 Trainer additions in `train_e2e_stage2_delta.py`

`build_configs` extended (parallels Stage 1) — see lines around 123–160
for the spectro / video append logic. `--use_video` and `--use_spectro`
flags around lines 886–892. The video-presence dataset filter at
~968–988 only retains shot files where the requested cameras' HDF5
groups exist.

The rollout machinery (`displacement_losses` per-modality dispatch,
`split_target_by_step`, the per-step gate) was extended at the same
time: video targets are split per-step in 5-D, displacement losses
branch on `cfg.kind == "video"` to drop cos/mag and keep only MAE in
pixel space, and the `_video_loss_gate` is built once per batch.

### 7.3 Status

Code-complete and submission-ready alongside BC-Stage 1.

---

## 8. Stage 2 Extended — what's missing

`train_e2e_stage2_extended.py` is currently TS-only:

* No `--use_video` or `--use_spectro` flags.
* No spectro/video append in its config builder.
* The free-rollout machinery (`TokenSpaceRollout`, scheduled-sampling
  TF schedule from §13) does not propagate spectrogram or video
  diagnostics.

To extend Extended:

1. Mirror Stage 2b's `--use_video` / `--use_spectro` argparse and
   `build_configs` plumbing.
2. Update `_make_chunk_fn` / `rollout_forward_loss_extended` so the
   per-step diagnostic prefix slice carries spectro and video tokens
   alongside TS — `TokenSpaceRollout.forward` already accepts
   per-step GT, so the existing TF logic should work once the prefix
   is multimodal.
3. Per-modality displacement-loss dispatch already exists in Stage 2b;
   port the `cfg.kind == "video"` / spectrogram branches to Extended's
   loss builder.
4. Extend the BC-Stage 1 G2 / G3 byte-identical fixtures with
   Extended-trainer equivalents (or accept the existing fixtures as
   sufficient since they exercise the same model).

Estimated effort: ~1–2 days of focused coding plus a benchmark pass.
Order — A (validate first) vs B (plumbing first) — is still open per
§13.

---

## 9. Tests

Live test files exercising spectro / video paths:

```
tests/data/test_video_loading.py            8 passed
tests/data/test_spectrogram_loading.py      green per spectrogram_tokenizer_plan.md
tests/e2e/test_video_tokenizer.py           7 passed, 1 skipped (GPU OOM)
tests/e2e/test_video_integration.py         5 passed (G1–G5)
tests/e2e/test_spectrogram_*.py             green per plan
```

The five Step-5 guard tests (`test_video_integration.py`):

| Guard | Test | Asserts |
|---|---|---|
| G1 | `test_video_tokens_in_diagnostic_prefix` | every `TokenSlice` named `tangtv` has `slice.stop <= n_diag_tokens` |
| G2 | `test_no_video_state_dict_keys_identical` | TS-only `state_dict()` keys equal a captured fixture |
| G3 | `test_no_video_forward_bitwise_identical` | TS-only forward output is `torch.equal` to a captured fixture |
| G4 | `test_load_old_checkpoint_into_video_model_succeeds` | TS-only state_dict loads cleanly into a TS+video model |
| G5 | `test_load_with_unexpected_key_raises` | the explicit loader raises on renamed keys |

G2 + G3 fixtures live at `tests/e2e/fixtures/no_video_forward.pt`
(captured with `scripts/capture_no_video_fixture.py`). The capture
script's docstring explains when to regenerate; do NOT regenerate
reflexively to silence a failing test.

---

## 10. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-04-27 | tangtv: per-channel availability mask, not per-pixel | "65% NaN" was the fraction of off-channels averaged over shots, not an off-pixel ratio; off-channels are NaN-everywhere slabs, active channels are NaN-free. |
| 2026-04-27 | tangtv: keep near-constant channels (e.g. shot 204510 ch0/ch2 with mean=50 exactly) | Trust the model to learn that low-dynamic-range channels carry little information; no std-based filter. |
| 2026-04-27 | Drop irtv from Phase C scope | Only tangtv is in MOVIE_CONFIGS for the active pipeline. |
| 2026-04-27 | Replace Perceiver-pool video tokenizer with tube-patch | Three Perceiver iterations plateaued at ratio ~0.62 on plasma channels and produced featureless reconstructions. Bounded global tokens cannot encode unbounded local spatial structure. See §11. |
| 2026-04-28 | Tube-patch shape `(3, 12, 12)` → 300 tokens | Option A in the §14 token-budget memo. 24×24 (75 tokens) was cancelled before producing final results. Perceiver-after-tube-patch (option C) was rejected because the skip connection from input tokens doesn't generalise to autoregressive prediction. |
| 2026-04-28 | G3 reference fixture (Q1) | Catches accidental perturbations to the TS forward path; regeneration cost when the TS path changes is acceptable. |
| 2026-04-28 | No runtime `--use_video` flag inside the model (Q2) | Model is list-gated — instantiates video modules only when a `DiagnosticConfig(kind="video")` is present. The trainer owns the on/off decision via its own flag. |
| 2026-05-06 | tangtv 7 → 2 channels (filters 4 and 6 only) | Filters 4 and 6 are the only ones carrying plasma data across all shots; the others are background / calibration / dim. The previous `runs/c_stage1` was deleted. |
| 2026-05-06 | Combined BC launchers (`train_bc_stage1.sh`, `train_bc_stage2.sh`) supersede the separate `train_c_stage1.sh` and Phase-B-only launchers | Joint single-run training of both modalities is cleaner than two parallel pipelines and shares the warm-start from Phase A. |
| 2026-05-06 | Four-flag freeze API (`--freeze_{ts,video,spectro,backbone}_steps`) supersedes the Phase-C-only `--freeze_backbone_steps` | Each modality + the backbone needs to be freezable independently when warm-starting from Phase A; the combined launcher freezes TS+backbone but lets newly-initialised spectro+video train from step 0. |

---

## 11. Historical: Perceiver → tube-patch reset (2026-04-27)

Preserved because the reasoning generalises. The original Phase C
design used 16/32 global Perceiver queries cross-attending over
8×100 stem patches, then a ConvT cascade decoder up to 120×360.

* Three Perceiver iterations (16→32 queries; 3-stage→5-stage decoder;
  width-32 throughout) all hit ratio ~0.62 on plasma channels and
  produced featureless "predict per-(B, C) mean" reconstructions.
* `scripts/diagnose_video_ae.py`'s diagnostic 3 (overfit a fixed
  batch with stem-resolution head) gave ratio 0.32 in 200 steps,
  initially read as "bottleneck has the information". That was a
  *memory* test, not a generalization test: a single batch can be
  encoded by global pooling; diverse plasma shots cannot.
* Generalisation conclusion: bounded global tokens are the wrong
  primitive for plasma video. Patches were always the right answer.

Tube-patch validation results at step 3500 of
`scripts/training/train_video_ae.py`:

```
                old (Perceiver)    new (tube-patch)    improvement
ch4 ratio:      0.62 plateau       0.235               2.6× better
ch6 ratio:      0.71 plateau       0.369               1.9× better
ch0 ratio:      0.97               0.266               3.6× better
ch2 ratio:      0.69               0.233               3.0× better
```

Recon plot at step 3500 showed visible curved plasma filaments in
both input and output columns — structural reconstruction, not mean
prediction.

---

## 12. Historical: dropped designs

### 12.1 irtv

Dropped from Phase C scope on 2026-04-27. Kept in `MOVIE_CONFIGS`
purely so the dataset code path doesn't need to be removed.

### 12.2 Pixel-level NaN mask for video

Replaced with `tangtv_channel_mask: (C,) bool`. The original
per-pixel mask `~np.isnan(data).any(axis=(0, 1))` set the entire
spatial mask to False whenever any one channel was off (because
off-channels are stored as fully-NaN slabs in `ydata`). The new
contract is: "channel is active iff it contains any non-NaN value
in the loaded window."

### 12.3 7-channel tangtv

Used in all runs prior to 2026-05-06 (including the deleted
`runs/c_stage1`). Token count was the same (the tube-patch
tokenizer's token count depends only on H×W÷patch and not on C),
but channel-mask coverage was substantially worse because filters
0/1/2/3/5 were almost always either NaN-everywhere (off) or
near-constant (calibration). Active channel-set was typically
{4, 6} or {0, 2, 4, 6} per shot.

---

## 13. Open work

1. **Stage 2 Extended multimodal extension** (§8). Either order A
   (validate BC-Stage 1 first) or order B (plumbing first) — decision
   pending. Rough effort 1–2 days plus benchmark.
2. **Full BC-Stage 1 memory benchmark** at the 1178-token / batch
   configuration in `train_bc_stage1.sh`. The 698-token (TS+video)
   benchmark in §4.2 is the only one on file; the spectro path adds
   480 more tokens.
3. **First multi-day BC-Stage 1 run** has not been recorded here yet.
   See `MEMORY.md` `project_bc_stage1_*` for the live observations.

---

## 14. Cross-references

* `docs/spectrogram_tokenizer_plan.md` — Phase B implementation plan.
  Steps 0–5 and Stage 2 integration are complete; Step 5's freeze API
  description there still references the old single-flag form (the
  four-flag API in §5 of this doc supersedes it).
* `docs/video_tokenizer_plan.md` — Phase C implementation plan. Early
  sections still describe the abandoned Perceiver-pool design;
  the tube-patch design here in §3.2 is what shipped.
* `docs/spectrogram_step0_findings.md` — empirical presence rates
  for ECE / CO2 / BES.
* `docs/eval_stage1_plan.md`, `docs/eval_stage1_panels_patch.md` —
  Stage 1 evaluation. Multimodal eval support is partial; BC-Stage 1
  diagnostics not yet integrated end-to-end.
* `docs/ResearchPlan.MD` — refers to Phase B and Phase C as separate
  research stages. The "BC" nomenclature in this doc reflects the
  combined training pipeline that landed 2026-05-06; the research-
  level distinction in `ResearchPlan.MD` is unchanged.