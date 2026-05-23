# Stage 2 with video — implementation plan

Goal: train video alongside TS modalities through Stage 2's K=10 rollout, with
real video-loss gradient flowing back through every rollout step. Init from a
Phase C Stage 1 checkpoint (`runs/c_stage1/c_stage1_best.pt`) when available.

## Decisions (locked unless flagged)

- **Video loss = plain MAE only.** No cos / mag-loss terms for video — per
  `project_phase_c_video_design.md`, cos in ~900 k pixels is meaningless. The
  TS displacement-loss formulation (`α·MAE + β·(1−cos) + γ·|log mag|`)
  applies to TS modalities only.
- **Per-batch standardisation on the input window** (`_video_standardize_per_bc`),
  applied identically to all K target windows. Stats computed once from
  step-0 input. Matches Stage 1 convention.
- **Video propagated through the rollout in token space** — same as TS. No
  detokenize-retokenize between steps.
- **Video target geometry:** dataset emits `K · n_output_frames` target frames
  (= 30 frames at K=10, n_output_frames=3), structured so the trainer can
  split into K windows of `n_output_frames` each.
- **`tangtv` only** for now (mirroring C-Stage 1). irtv plumbing comes
  later, same hooks.

## Four edits — files and effort

### Edit 1 — Rollout tokenisation honours `_valid` mask (~10 LOC)

**File:** `src/tokamak_foundation_model/e2e/rollout.py`

`_tokenize_diagnostics` currently calls `tokenizer(x)` for every modality,
ignoring the camera-validity scalar. Branch on `cfg.kind == "video"` and
forward `diag_inputs[f"{cfg.name}_valid"].bool()` as `mask=`. Mirrors the
logic already in `model.py:tokenize`.

Affects: step 0 (`initial_diag_inputs`) and TF re-tokenisation
(`gt_target_per_step`). Without this, the ~45 % of shots without tangtv
get garbage video tokens fed to the backbone.

### Edit 2 — Dataset emits K × n_output_frames target frames (~25 LOC)

**File:** `src/tokamak_foundation_model/data/data_loader.py`

In `_getitem_prediction` (lines 1628–1666), the video target half is
currently subsampled to `n_output_frames` total frames spread across the
entire `prediction_horizon_s`. Change so that when
`prediction_horizon_s > chunk_duration_s` (i.e. K > 1):

1. Compute `K = round(prediction_horizon_s / chunk_duration_s)`.
2. Split `out_chunk` into K equal sub-windows of `n_training_frames` each.
3. Subsample each sub-window to `n_output_frames` evenly-spaced frames.
4. Concat into one `(C, K · n_output_frames, H, W)` tensor.

`channel_mask` and `_valid` scalar are unchanged (per-shot, not per-step).

Backward-compat: K=1 → original single-window behaviour byte-identical.

### Edit 3 — Stage 2 trainer learns video (~80 LOC)

**File:** `scripts/training/train_e2e_stage2_delta.py`

Port from `train_e2e_stage1.py`:

- `VIDEO_MODALITIES` registry (just `tangtv` for now).
- `build_configs` accepts `use_video`, appends video DiagnosticConfigs.
- Helpers: `_video_standardize_per_bc`, `_video_loss_gate`.
- New per-step splitter `split_video_target_by_step(target, K, n_per)` —
  returns K slices of `(B, C, n_per, H, W)`.
- `--use_video tangtv` CLI flag (defaults to none, byte-identical when off).
- `--freeze_backbone_steps` warm-start support (mirrors C-Stage 1).
- `rollout_forward_loss_delta` modifications:
  - Apply per-(B,C) z-score to `diag_initial[video]` and propagate
    `(mu, sd)` to standardise per-step video targets.
  - Pass `f"{name}_valid"` into `diag_initial` so the rollout's tokeniser
    can mask missing-camera rows (Edit 1).
  - Compute per-step video MAE with `_video_loss_gate(channel × valid)`.
  - Permute video predictions `(B, T, C, H, W) → (B, C, T, H, W)` to
    match target shape, per step.
  - Add to `step_loss` with weight `mae_weight` only — no cos/mag for video.
- `validate` extended to include video MAE per step in the val table.
- File-presence filter (`filter_video_present_files`) wired exactly like
  C-Stage 1.

### Edit 4 — Launcher (~5 LOC)

**File:** `scripts/slurm/train_e2e_stage2_delta.sh`

Add:
- `--use_video tangtv` flag
- Optional `--freeze_backbone_steps 5000` (matching C-Stage 1's warm-start
  convention; only relevant if init is from a NON-video checkpoint, which
  shouldn't happen if we init from C-Stage 1 best)
- Snapshot from `runs/c_stage1/c_stage1_best.pt` (replacing the current
  Stage 1 snapshot) — when that file exists; fall back to Stage 1 best
  with explicit `allowed_missing_prefixes` for video keys, like C-Stage 1
  does.

Auto-resume from `runs/e2e_stage2_delta/e2e_stage2_delta_latest.pt` is
already wired and unaffected.

## Order of work

1. Rollout mask fix (Edit 1) — smallest, foundational.
2. Dataset target geometry (Edit 2) — enables K-window video targets.
3. Stage 2 trainer video plumbing (Edit 3) — biggest, depends on 1+2.
4. Launcher update (Edit 4).
5. Smoke test on CPU with `--max_steps 5 --K_max 2 --batch_size 2 --use_video tangtv`.
6. Sanity check: `pixi run pytest tests/e2e/test_rollout.py` still passes
   (Edit 1 must preserve byte-identity for the mask=None TS-only path).

## Open questions

1. **Init source.** When a C-Stage 1 best is available, we want to init
   Stage 2 from it. But C-Stage 1 isn't done yet (still ~32 % through 336 k
   steps as of last check). Two options:
   - Wait for C-Stage 1 to finish, then start Stage 2 with video.
   - Start Stage 2 with video sooner using current C-Stage 1 latest, accepting
     that Stage 2's foundation is a partly-trained Stage 1.

2. **freeze_backbone_steps for Stage 2.** Stage 1 used 5 k frozen steps when
   warm-starting from a TS-only checkpoint, to let video tokenizer/head warm
   up without disturbing TS. If we init Stage 2 from a C-Stage 1 best (where
   video has already been trained for ~10 epochs), the freeze is unnecessary.
   Default to 0 if init has video keys, 5 k otherwise.

3. **Video loss weight in Stage 2's combined sum.** Currently `mae_weight = 1.0`
   for all modalities. Video MAE is in standardised pixel space (~unit-variance
   per channel) and TS MAE is in standardised signal space (also unit-variance).
   Magnitudes should be comparable. Suggest leaving `mae_weight = 1.0` for
   video and watching the per-modality breakdown for one block before deciding
   to weight it down.

## Estimated total LOC and time

~120 LOC across 4 files, ~2–3 h of careful implementation including smoke
testing. Compares with the original Phase C C-Stage 1 effort (~150 LOC for
the same plumbing in train_e2e_stage1.py).

## What I'd like sign-off on

- Locked decisions look right? (video MAE-only, per-batch standardise, K
  target windows of `n_output_frames` each)
- Open question 1: wait for C-Stage 1 to finish, or start sooner with
  current latest?
- Open question 3: any reason to weight video loss differently from TS?
