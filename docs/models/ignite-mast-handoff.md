# IGNITE on MAST: handoff (2026-11-27)

This page hands over the IGNITE-on-MAST work on node13 (branch `nvidia`). It says
what was tried, what worked, what didn't, where everything lives and what to do
next. Detailed results, with the measurements behind them, are in the other
pages of this series:

- [design](ignite-mast-design.md): goals, the bar, the design and its risks;
- [stage 0](ignite-mast-stage0.md): headroom against TokaMind/GenToka on TokaMark;
- [stage 1](ignite-mast-stage1.md): the fixed and learned tokenizers;
- [spectrogram quality](ignite-mast-spectrogram-quality.md): stage 2 (the IGNITE
  port), the Mirnov codecs, TokEye and decode tests. It is current to 2026-11-25
  ~19:00 UTC. Everything measured after that is only on this page (the section
  [New since the spectrogram page](#new-since-the-spectrogram-page)).

Read `docs/clusters/nvidia.md` and `/models/README.md` first for the machine
(aarch64, 4 GPUs at cc 10.7, NVRTC preload, `/models` layout).

## The goal and where it stands

**Goal** (set 2026-11-21, window to about 2026-12-12): pretrain dev-peter's
IGNITE on MAST (FAIR-MAST / TokaMark data) so that, as an autoregressive,
control-ready simulator, it beats TokaMind and GenToka by a wide margin. From
2026-11-24 the owner's priority was: **make the MAST rollouts look as good as
IGNITE's DIII-D rollouts.** Prerequisites took too long, so scope was cut to
reach good rollout plots.

**Where it stands on 2026-11-27:**

- The port works. dev-peter's Phase B (MaskGIT dynamics, trainer, rollout)
  trains on MAST unchanged, with MAST tokenizers. Rollouts follow the plasma
  current ramp and the NBI step. Switching NBI off in a counterfactual moves
  beta_p, SXR, Mirnov activity, T_e and n_e the physical way, checked on one
  shot.
- Slow signals (T_e, n_e, q95, beta_p, SXR, D-alpha, I_p) roll out reasonably.
  The best checkpoints have a median rollout NRMSE of about 0.057–0.061 over 40
  signals (6 validation shots, seed 0).
- **The Mirnov spectrograms do not roll out cleanly.** No checkpoint
  reproduces chirping mode lines. One checkpoint (run 9 at step 10,000) places
  modes slightly better than holding the last measured frame. Everything else
  is below that baseline.
- **IGNITE has not been scored on TokaMark yet**, so there is no
  IGNITE-against-TokaMind/GenToka number. Stage 0 only measured a direct model
  (no pretraining) per task: projected 14-task mean 0.1897 against the bar of
  0.183 (matching the best published score everywhere would give 0.194; lower
  is better). See [stage 0](ignite-mast-stage0.md).

## How a rollout is judged

All MAST rollout scoring since 2026-11-25 uses the same protocol (rulings R26,
R29 and R30 in the ledger):

- CPU sampler (`CUDA_VISIBLE_DEVICES=`), seed 0, `IGNITE_HISTORY_CACHE=1`,
  `IGNITE_MASKGIT_GUMBEL=3`, `--forbid-missing`.
- Windows of 32 frames of 50 ms: 12 seed frames (`k0`), then 20 predicted.
- **40-signal score:** `scripts/mast/port_rollout_plots.py` on the 6 shots
  11914, 16052, 20384, 23403, 24934 and 28101. It reports the per-signal median
  rollout NRMSE and its median over the signals. Reconstructions
  (`port_recon_plots.py`) use 23529 and 20384.
- **Spectrogram score:** TokEye mode fidelity
  (`scripts/mast/tokeye_mode_fidelity.py`), 12 mode-rich validation shots × 3
  windows, which gives 108 OMV and 45 CC probe-windows. It reports the mode-mask
  F1 of the rollout against the measurement. Two references come with it: the
  codec ceiling (decode the true codes: about 0.95 for OMV) and persistence
  (hold the last seed frame: OMV 0.189, CC 0.140). `--one-step` also scores each
  frame predicted from the measured frames before it.
- **"Lit / truth":** how much of the spectrogram the rollout marks as mode,
  relative to the measurement. It tracks the mode F1 closely and is the
  quickest read of "too dim".
- One seed and 6 shots are thin. A median difference under about 0.005 is
  noise.

## Runs

All runs are under `/models/intermediate/mast/stage2r/runs/`. Each has a
`launch.sh` and a `train.log`.

| Run | Directory | What it tested | Outcome |
|---|---|---|---|
| 2 | `v1b_d512L12` | first port run (d512 × 12), linear Mirnov codec | paused at step 5,000 (R8) |
| 3 / 3a | `v1b_logspec_cam_d512L12` / `v1b_logspec_d512L12` | log-power Mirnov codec, with and without the camera | the log codec costs no rollout skill (R16) |
| 4 / 5 | `v1b_logspec_cam_bf_*` / `v1b_logspec_cam_live_f32_d512L12` | block forcing against none, on the live-only cache | no block forcing (R19: run 5 better on 33/40 at equal wall clock) |
| 6 | `v1b_logspec_cam_live_da_f32_d1024L16` | dev-peter's production size (d1024 × 16) from scratch, 256 windows per step | better one frame ahead than run 8, slow on 2 GPUs; paused at 4,500 (R28) |
| 7 | `v1b_peakspec_cam_live_da_f32_d512L12` | peak + baseline Mirnov codec (PeakSpec) + TokEye selection | stopped at ~5,800 for run 8 (R25) |
| **8** | `v1b_logspec_cam_live_da_tokeye_f32_d512L12` | band-DCT Mirnov codec + D-alpha v2 + TokEye selection, 30k steps | **the reference.** Final median 0.057 (G3). OMV mode F1 0.122 final, 0.184 at step 10k |
| **9** | `v1b_r9_r6s4500_r8recipe_d1024L16` | run 6's step-4,500 weights, run 8's recipe | **best spectrograms**: step 10k OMV F1 0.207 (beats persistence), median 0.061. Paused at 11,500 (R31), resumable |
| 10 | `v1b_binspec_cam_live_da_tokeye_f32_d512L12` | per-bin log-power Mirnov codec (BinSpec), else run 8 | best 40-signal median (final 0.060, better than run 8 on 22/40) but the dimmest spectrograms (OMV F1 0.090) |
| 11 | `v1b_binspec_cam_live_da_tokeye_f32_d1024L16` | run 10 at d1024 × 16 from scratch, batch 8 per rank | **diverged**, see below. Ignore its checkpoints |

Checkpoints worth keeping:

- run 8: `dynamics_step{5,10,15,20,25,30}000.pt`, plus `runs/r8_final_eval.pt`;
- run 9: `dynamics_step10000.pt` (best spectrograms) and `dynamics_latest.pt`
  (step 11,500);
- run 6: `dynamics_step4500.pt`.

Frame caches: `frame_codes_v1b_logspec_cam_live_da` (runs 8, 9; band-DCT) and
`frame_codes_v1b_binspec_cam_live_da` (runs 10, 11).

## What worked

- **Porting dev-peter's Phase B unchanged** (R0). The model, trainer and
  rollout run on MAST with MAST codecs and an 8-channel actuator vector.
  Changes live in `ignite/mast/` or are runtime patches. `maskgit.py` and
  dev-peter's other files are not edited.
- **Gumbel 3 decode** (R30, dev-peter's DIII-D setting). With Gumbel 0, T_e and
  n_e froze: the decode copied the last frame (repeat rate 0.46–0.90 against a
  measured 0.19). Gumbel 3 brought the repeat rates to the measured ones and run
  8 final's median from 0.068 to 0.057.
- **A per-modality Gumbel scale** (new on 2026-11-26, not yet adopted
  officially). Gumbel 0 for the two Mirnov arrays and Gumbel 3 for the rest:
  - spectrograms: run 8 final's OMV mode F1 rises 0.122 → 0.143 and lit 0.54 → 0.86;
  - T_e and n_e: error 0.159 → 0.138 and 0.176 → 0.165, repeat rates still at
    the measured level;
  - 40-signal median: 0.057 → 0.061, within one-seed noise.

  It is a runtime wrapper, `gmod_run.py` (see [Scripts](#scripts-not-yet-in-git)).
  Recommended as the standard decode.
- **`--forbid-missing`** (R26). It removes the blank "not recorded" stretches
  from Mirnov rollouts.
- **The band-DCT log-power Mirnov codec.** The owner liked how it looks; codec
  ceiling F1 0.957.
- **D-alpha v2** (peak + baseline). This was the one signal that looked wrong
  with the old codec.
- **The camera** as two modalities through dev-peter's VideoCodec at lr 3e-4
  and `pixel_anchor_weight` 1.0 (R12, R14). The default settings collapsed.
- **TokEye**, used two ways:
  - to select training windows (`presence_tokeye*.json`, `--tokeye_factor 2`);
  - as the spectrogram judge.

  Its mode-mask F1 against the codec ceiling and against persistence is the
  most useful spectrogram metric found. The 40-signal NRMSE does not see
  missing modes.
- **Model size for the Mirnov dynamics.**
  - Run 6 (d1024) beat run 8 (d512) one frame ahead: OMV 0.559 against 0.490,
    CC 0.456 against 0.277 (2 shots).
  - Run 9 (d1024) is the only run whose free rollout beats persistence on OMV
    modes.

## What didn't work

- **BinSpec (runs 10 and 11).** One token per frequency bin removed the blocky
  patchwork of the band-DCT codec and won the 40-signal median. But its Mirnov
  rollouts light only 24–30% of what was measured. Across steps 10k, 20k and
  final, OMV mode F1 stays at 0.08–0.09 and CC at 0.06–0.08, half of run 8's.
  Its level drift: rollouts start at the measured level but lag the power rise
  (level 30 → 40 against a measured 30 → 48 over 20 frames). A dead end for
  spectrograms.
- **Run 11 diverged.** It was d1024 from scratch at lr 1e-3 with run 8's small
  batch (2 ranks × 8). Gradient norms ran from 15 to 943 (run 10's: about 0.2).
  The best validation loss came at step 1,000, and the 40-signal median was
  0.62, ten times run 10's. It used GPUs 2–3 for about 14 h (2026-11-25 23:17
  to 2026-11-26 13:37). Run 6, the same size from scratch, was stable at lr
  1e-3 with 256 windows per step (4 ranks × 32 × accumulation 2). **The
  lesson:** at d1024, either use the large batch, or warm-start (as run 9 did),
  or lower the learning rate.
- **Training longer makes the spectrograms worse.** Run 8's OMV mode F1 by step:

  | Step | 5k | 10k | 15k | 20k | 25k | 30k |
  |---|---|---|---|---|---|---|
  | OMV mode F1 | 0.170 | **0.184** | 0.159 | 0.146 | 0.129 | 0.122 |
  | Lit / truth | 0.88 | 1.05 | 0.73 | 0.58 | 0.54 | 0.54 |

  The validation loss keeps improving (best at step 20k), so selecting on
  validation loss picks a checkpoint with dimmer spectrograms. Run 9 also fell
  from 0.207 at step 10k to 0.184 at 11.5k (one seed).
- **Free rollout loses half of what the model knows one frame ahead.** At step
  10k:
  - run 8: OMV 0.383 one step ahead against 0.184 free; CC 0.221 against 0.123;
  - run 10: 0.242 against 0.084.

  Even one frame ahead, a chirping line (23529, 90 → 70 kHz) comes out as a
  flat, patchy band.
- **The Mirnov dynamics are barely learned.** Run 8's OMV validation `gen_ce`
  went from 3.665 at step 1k to 3.580 at 18k; a uniform guess over 64 codes is
  4.16. The other signals keep improving.
- **Smaller things:**
  - PeakSpec codec (run 7): stopped for the band-DCT codec.
  - Block forcing (run 4): lost (R19).
  - MARS-style coarse-to-fine reveal order: OMV 0.122 → 0.135 on run 8 final,
    less than the per-modality Gumbel.
  - Decode steps, seeds and Gumbel 10/30 on the dark run-10 step-5k frame: the
    darkness was model uncertainty and went away with training.
  - The MARS paper (arXiv 2509.26007) has no public code.

## New since the spectrogram page

Measured between 2026-11-25 19:20 and 2026-11-26 00:56 UTC, plus run 11
(trained until 2026-11-26 13:37, scored by 15:51). The protocol is as above, and
"lit" is lit / truth.

TokEye OMV / CC mode F1 (median):

| Checkpoint | OMV F1 (lit) | CC F1 (lit) |
|---|---|---|
| Hold the last measured frame | 0.189 | 0.140 |
| **Run 9, step 10k** | **0.207** (1.15) | 0.139 (0.75) |
| Run 9, step 11.5k | 0.184 (0.96) | 0.141 (0.75) |
| Run 8, step 10k | 0.184 (1.05) | 0.123 (0.96) |
| Run 6, step 4.5k | 0.172 (0.84) | 0.141 (0.65) |
| Run 8 final, Mirnov Gumbel 0 | 0.143 (0.86) | 0.118 (1.12) |
| Run 8 final | 0.122 (0.54) | 0.108 (0.78) |
| Run 10 final | 0.090 (0.30) | 0.076 (0.38) |
| Run 10, step 10k, Mirnov Gumbel 0 | 0.084 (0.30) | 0.070 (0.34) |
| Run 11 final | 0.056 (0.23) | 0.021 (0.15) |

40-signal median rollout NRMSE (6 shots):

| Checkpoint | Median | Better than run 8 final | Notes |
|---|---|---|---|
| Run 8 final | 0.057 | — | T_e 0.159, n_e 0.176, q95 0.142 |
| Run 8 final, Mirnov Gumbel 0 | 0.061 | 17/40 | T_e 0.138, n_e 0.165 |
| Run 9, step 10k | 0.061 | 9/40 | T_e 0.175, n_e 0.155, q95 0.147 |
| Run 9, step 11.5k | 0.067 | 11/40 | |
| Run 10 final | 0.060 | 22/40 | q95 0.098, D-alpha 0.120 |
| Run 10, step 20k | 0.069 | 13/40 | |
| Run 11 final | 0.616 | 0/40 | diverged |

Plot directories: `/models/intermediate/mast/stage2r/plots/<run>/step<N>forbid_g3[m0]/`.
TokEye results: `/models/intermediate/mast/stage2r/tokeye/cpu_r<run>_s<step>_forbid_g3[m0|_1step]/`.

## Recommended next steps

In the order I would take them:

1. **Adopt the per-modality decode** (Mirnov Gumbel 0, the rest Gumbel 3). Move
   `gmod_run.py` into the port's rollout code as a default-off option, and
   re-score run 9 step 10k with it. On run 8 it gave +0.02 OMV F1, so run 9 with
   it is the best rollout available today. Make its figures first.
2. **Select checkpoints on TokEye mode F1, not on validation loss.** Mode
   quality peaks around step 10k under the 30k cosine schedule. Save snapshots
   every 1k steps and score them on CPU while training continues.
3. **Next training run: d1024 on the band-DCT cache**, either resumed from run 9
   or fresh with run 6's large batch (not run 11's batch 8). Use a shorter or
   flatter schedule, given the fade. Never BinSpec.
4. **More data.** The Mirnov dynamics look data-limited (`gen_ce` barely moves).
   The full FAIR-MAST download was started on 2026-11-25 (a separate subtask);
   its state was not checked after that evening. The actuator pull
   (`logs/act_pull.log`) exited rc=1 after 16,961 s with 22,870 errors (8.7 GB).
   Then rebuild the frame cache with more shots.
5. **Actuators v2.** The model sees 8 channels: gas total, PF coil voltages
   P1/P2/P4/P5, the plasma current and line-density references, and total NBI
   power. DIII-D IGNITE sees 88. On disk or in level 1 there is also:
   - per-beam NBI (anb SS/SW power);
   - inboard and outboard gas totals and the valve voltages;
   - TF current, P6, and the EFPS and error-field coil currents;
   - z_ref.
6. **Diagnostics not encoded.** Candidates from the 2026-11-25 audit: summary
   ohmic and radiated power, neutron rate, bolometers, equilibrium li / W_mhd /
   V_loop / q profile, charge exchange (4,278 shots), TF, passive PF, z_ref,
   the tangential SXR camera, OMAHA, BES and the edge density gradient.
   `diag_value.py` (HistGB, GroupKFold by shot) is ready to rank them once
   level 2 is complete.
7. **Score IGNITE on TokaMark** against TokaMind and GenToka. This is the stated
   goal and it has never been run for IGNITE. Stage 0's harness and tasks are
   in `ignite/mast/` (`harness.py`, `tasks.py`).

## Scripts not yet in git

The analysis scripts of 2026-11-24 to 2026-11-26 live in the session's job
directory, with a copy in `/models/intermediate/mast/stage2r/jobtmp_backup_1ce3f0fb/`.
Move the useful ones to `scripts/mast/` before relying on them:

| Script | What it does |
|---|---|
| `gmod_run.py` | Runs a MAST port script with a per-modality Gumbel scale (`GMOD_SCALE`, `GMOD_SIGNALS`). Runtime patch; `maskgit.py` is untouched |
| `c2f_run.py` | The same idea, for a coarse-to-fine reveal order on the band-DCT Mirnov tokens |
| `cmp_rollouts.py` | Compares scored plot directories: the 40-signal median, wins, key signals, reconstructions and repeat rates |
| `fid_by_h.py` | Pairs TokEye rows across runs: median F1, lit / truth, wins against a reference |
| `level_drift.py` | BinSpec level, contrast and flicker by horizon, truth against rollout |
| `repeat_rate.py` | Frame-to-frame code repeat rate, truth against rollout |
| `r10_watch.sh`, `r11_watch.sh`, `fid_traj.sh`, `fid_1step.sh`, `gmod_eval.sh` | Examples of the CPU scoring watchers |

Example: run 9 step 10k with the per-modality decode
(`B=/models/intermediate/mast/stage2r/jobtmp_backup_1ce3f0fb`,
`S=/models/intermediate/mast/stage2r`):

```bash
export IGNITE_MASKGIT_GUMBEL=3 GMOD_SCALE=0 IGNITE_HISTORY_CACHE=1 CUDA_VISIBLE_DEVICES=
OMP_NUM_THREADS=32 scripts/mast/py.sh $B/gmod_run.py scripts/mast/port_rollout_plots.py \
    --ckpt $S/runs/v1b_r9_r6s4500_r8recipe_d1024L16/dynamics_step10000.pt \
    --cache-dir $S/frame_codes_v1b_logspec_cam_live_da --out-dir <out>/real \
    --shots 11914,16052,20384,23403,24934,28101 --seed 0 --actuator-mode real --device cpu --forbid-missing
OMP_NUM_THREADS=32 scripts/mast/py.sh $B/gmod_run.py scripts/mast/tokeye_mode_fidelity.py \
    --keep $S/tokeye/scores_v2 --out $S/tokeye/<tag> --cache-dir $S/frame_codes_v1b_logspec_cam_live_da \
    --ckpt <ckpt> --device cpu --forbid-missing --figures 4
scripts/mast/py.sh -I $B/cmp_rollouts.py ref=<plotdir> new=<plotdir>
scripts/mast/py.sh -I $B/fid_by_h.py $S/tokeye/cpu_r8_final_forbid_g3 $S/tokeye/<tag>
```

## Records

- **Ledger:** `.superpowers/sdd/2026-11-23-ignite-mast-stage2r-port/progress.md`.
  It is git-ignored, so it exists only in this checkout. It holds every run,
  ruling (R0–R32) and measurement with times. Plans are in `.superpowers/drafts/`.
- **Progress page:** https://claude.ai/artifact/Pngm39wZYuUqxry3nhdVVt. It is
  published up to v94. Later edits are in its source, copied to
  `jobtmp_backup_1ce3f0fb/page/ignite-mast-progress.html`; the publish
  service was failing on 2026-11-25.
- **Logs:** `/models/intermediate/mast/stage2r/logs/`.
- **Last commit** with the docs: `3dbacce8` (pushed to `origin/nvidia`).

## Rules this work kept

- Data and runs go under `/models`, and nothing under `/models/tokamind/`. Pass
  `--frozen` to pixi.
- Don't edit `pyproject.toml`, `pixi.lock` or dev-peter's files outside
  `ignite/mast/`. Runtime patches from scripts are fine.
- Files hashed by stage 0 never change: `ignite/mast/harness.py`, `tasks.py`,
  `baselines.py`, `headroom/{floor,cheap,direct}.py`, and every `*.py` directly
  in `ignite/mast/tokenizers/`.
- Downloaded data is untrusted: run Python on it with `-I`, and never `cd` into
  it or execute anything from it.
- Score on CPU while the GPUs train. A `ProcessPoolExecutor` opened after torch
  is loaded uses the spawn context.
- GPUs 0–1 were reserved for someone else on 2026-11-24/25; that has passed.
  Check `nvidia-smi` before using any GPU. On 2026-11-27 at 17:16 UTC all four
  were idle and had been since run 11 ended (GPU 2 held a 1.3 GB MinerU
  process).
- The owner said a locally running Kimi K3.1 can be used for low-level
  implementation agents, besides Antigravity/Gemini. Its setup has not been
  checked.
