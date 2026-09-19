# Stage-1 Evaluation Pipeline — Design Plan

Working design for `scripts/eval/eval_stage1.py` and supporting modules.
This document is the source of truth for what we're building before any
code lands.

## 1. What stage-1 actually predicts

From `train_e2e_stage1.py`:
- **Input:** diagnostics at time `t` + actuators driving `t → t + 50 ms`
- **Target:** diagnostics at time `t + 50 ms`

So this is **single-step (K=1) next-chunk prediction**, not autoencoder
reconstruction. The evaluator must mirror this: it scores each window
on how well the model predicts the *next* 50 ms diagnostic state given
the current state and the actuator trajectory.

## 2. Goals

For each diagnostic modality, on both train and val splits, answer:

1. **Did the model actually learn dynamics, or is it just copying the
   input?** Per-modality scatter of per-shot model MAE vs **copy-baseline
   MAE** (where "copy" predicts `t+50 ms` identical to `t` — pure
   persistence, no model). Points below the diagonal = model beats
   persistence and has learned something. Points on the diagonal =
   model is just propagating its input. This is the headline
   "did stage-1 work?" plot, per modality.

2. **Which shots reveal failure modes?** Rank shots by **MAE ratio**
   (`model_mae / copy_mae`), not by raw MAE. Raw-MAE ranking is
   confounded by intrinsic shot difficulty — quiet shots will always
   rank "best". The ratio normalises for that. **Worst-by-ratio is
   the more informative pool**: it surfaces shots where the model
   failed despite favorable, predictable input — disruption-adjacent
   windows, rare actuator configurations, missing-data edge cases.
   The bottom-N shots get more attention than the top-N in the
   plotting phase.

3. **Within-shot dynamics evidence is the paper-grade deliverable.**
   The stitched-window view (a single shot, GT vs prediction across
   **4+ seconds**) is the strongest visual evidence that stage-1
   has learned tokamak dynamics. For TS modalities this means
   overlaid traces tracking through transient events; for spectrograms
   it means side-by-side spectrogram evolution showing **mode
   frequency tracking and broadband turbulence changes**. These
   plots are the centrepieces — design and execution must reflect
   that.

Per-shot resolution matters — a single shot has hundreds of 50 ms
windows; aggregate metrics across the whole split can hide failure
clusters in specific shots.

**Plots are the primary deliverable.** The numerical metrics tables
are diagnostic infrastructure, but the plots are what a human will
actually use to judge stage-1 quality. They must be meaningful, with
clear GT-vs-prediction comparison and a layout that's easy to read at
a glance — see the quality bar in §5 before writing any plotting code.

## 3. Outputs

All outputs land in `--output_dir`. Phase 1 auto-names it
`eval_runs/stage1_phase1_<ckpt-stem>_<jobid>/`; Phase 2 and 3 write
**into the same directory** rather than creating new ones, so a full
end-to-end run produces one consolidated artifact set per checkpoint.
The `phase1_` token in the dir name is just a "who created it first"
hint — it stays as-is even after later phases run.

Concrete on-disk layout after all phases complete:

```
eval_runs/stage1_phase1_<ckpt-stem>_<jobid>/
├── config.json                # checkpoint path, split list, args snapshot
├── per_window_metrics.csv.gz  # one row per (shot_id, window_idx, modality, split)
├── per_shot_metrics.csv.gz    # aggregated: one row per (shot_id, modality, split)
├── top_bottom_shots.csv.gz    # top-N + bottom-N per (split, modality), ranked by mae_ratio_mean
└── plots/
    └── val/                                  # (also train/ if --splits train val)
        └── <modality>/
            ├── _aggregate_scatter.png        # Phase 2.0 (one per modality per split)
            ├── <shot_id>_summary.png         # Phase 2.1 (one per selected shot)
            ├── <shot_id>_stitched_<seg>.png  # Phase 3   (one per (shot, segment))
            ├── <shot_id>_stitched_<seg>_ch<n>.png  # spectrogram only (per-channel)
            └── <shot_id>.mp4                 # Phase 3, video modalities only
```

**Idempotency / re-runs.** All phase scripts overwrite existing files
in `<output_dir>`. To compare two runs, point them at different
output dirs; do not re-use a partial dir expecting "merge". The
metric CSVs are atomically rewritten by Phase 1; plot PNGs / mp4s
are atomically rewritten by the phase that produced them.

Per-window metrics columns:
- `shot_id`, `window_idx`, `window_t_s` (window-center time within shot)
- `split` (`train` | `val`)
- `modality` (e.g., `ts_core_density`, `ece`, `filterscopes`, `tangtv`)
- `mae` — masked mean absolute error (model)
- `copy_mae` — masked MAE between input (t) and target (t+50 ms);
  the persistence baseline for this window/modality
- `mae_ratio` — `mae / copy_mae` (< 1 means model beats persistence)
- `dcos` — direction cosine (TS modalities only; NaN otherwise)
- `mag_ratio` — magnitude ratio (TS only)

Per-shot aggregation: for each (shot_id, modality), compute
- `n_windows`
- Model: `mae_mean`, `mae_median`, `mae_p95`, `mae_max`
- Copy: `copy_mae_mean`, `copy_mae_median`
- Ratio: `mae_ratio_mean`, `mae_ratio_median`, `frac_windows_below_diag`
  (fraction of windows where `mae < copy_mae` — a per-shot version
  of the §2-Q1 scatter signal)
- `dcos_mean`, `mag_ratio_mean` (where defined)

**Storage format note.** Plan originally specified parquet, but the
pixi `frontier` env lacks `pyarrow` and `fastparquet`. Phase 1 lands
as **`csv.gz`** instead — pandas writes/reads it natively, no env
changes needed. Estimated worst-case size is ~250 MB compressed
(5000 shots × hundreds of windows × 12 modalities). If size becomes
a problem, adding `pyarrow` to `pyproject.toml` is a one-line change
and the file extension can be swapped without other code rewrites.

## 4. Iteration & DDP strategy

### Shot-level sharding

The training data loader emits windows in shot-major order (per
`DistributedTwoLevelSampler`'s two-level structure). For evaluation we
need every window of every shot:

- Build the dataset with `prediction_mode=True` (matching training).
- Disable random shuffling at the sampler level.
- Each DDP rank gets a contiguous shard of shots (not windows). This
  way per-shot aggregation can happen locally without cross-rank gather
  during the inference loop.
- After inference: rank 0 reads all ranks' per-window parquets,
  concatenates, and computes per-shot aggregates + top/bottom selection.

For single-GPU interactive mode: skip the DDP setup, iterate all
shots on the one rank.

### Copy-baseline computation (free)

The persistence/copy baseline is computed alongside the model forward
pass at zero extra inference cost: it's just `MAE(input_t, target_{t+1})`
per modality per window. Storing it as `copy_mae` lets the entire
downstream analysis (aggregate scatter, top/bottom-N selection,
per-shot metrics) work in **ratio space**, normalising for intrinsic
shot difficulty. No second forward pass and no architectural change
needed.

### Plotting after metrics

The plotting phase runs only on rank 0 after metrics are complete:
1. Read concatenated per-shot metrics.
2. For each (split, modality), pick top-N and bottom-N shot_ids
   **by `mae_ratio_mean`** (not raw MAE — see §2-Q2).
3. Re-run inference on those selected shots (small, ~10–20 shots per
   modality after dedup) and produce plots.

Saving every prediction tensor during phase-1 inference would balloon
disk usage (TB-scale). Re-running inference on selected shots only is
cheaper for both disk and code complexity.

## 5. Plot types

> **Quality bar — non-negotiable.** Plots are the primary artifact a
> human reads to decide whether stage-1 has learned something useful.
> Every plot must be **meaningful, clearly readable, and easy to
> interpret at a glance**. If a plot needs a paragraph of explanation
> to be understood, redesign it. Concretely, every plot must satisfy:
>
> - **Clear comparison:** GT and prediction visually co-located on
>   the same axes (overlaid lines, stacked panels with shared axes,
>   or side-by-side with identical color scales). The reader must be
>   able to see *where* and *how* they differ without flipping back
>   and forth.
> - **Consistent visual language across the whole eval run:** GT and
>   prediction get the same color/linestyle in *every* plot
>   (suggested: GT = solid black, prediction = dashed `tab:blue`,
>   |diff| = `magma` or `viridis` colormap). Channel order, panel
>   layout, and orientation stay consistent so a reader scanning a
>   directory can compare across shots without re-learning the
>   layout.
> - **Honest axes:** physical units in axis labels (samples, ms,
>   Hz, channel index, frame number, etc.); shared y-range for GT
>   and prediction so one isn't visually dominated; colorbars
>   labelled with units and value range; **no rainbow colormaps** —
>   they distort perception of relative magnitude.
> - **Self-documenting titles:** every figure title encodes
>   `shot_id`, `modality`, `split`, the metric value driving its
>   selection (e.g., `mae=0.342`), and `window_idx` for
>   single-window panels. A plot pulled out of context must still be
>   intelligible.
> - **Error visible:** wherever practical include a `|GT − pred|`
>   panel or residual trace so the *magnitude and location* of
>   errors are explicit, not implicit in line spacing.
> - **No clutter, but legend every panel that has lines.** A panel
>   with explicit lines (TS line plots, MAE-over-time, histograms)
>   gets a small 2- or 3-entry legend so the reader can identify
>   GT vs model without guessing. Image-style panels (heatmaps,
>   video frames) use in-figure text labels and an explicit
>   colorbar with a labelled unit instead of a legend. Never plot
>   more series than the eye can untangle (~8 lines is the upper
>   bound per panel — use small multiples beyond that). When
>   plotting many channels in a single panel, label only the first
>   GT line and the first model line so the legend has 2 entries,
>   not 2N.
>
> Concrete review test: if I look at a plot for 5 seconds and can't
> answer "did the model fit this?", the plot has failed and we redo
> it.

### Aggregate-quality scatter (one plot per modality per split)

**This is the headline "did the model learn anything?" plot — must be
the first thing produced by the plotting phase.**

- Scatter, one dot per shot.
- x-axis: per-shot `copy_mae_mean` (intrinsic difficulty of this shot
  for this modality).
- y-axis: per-shot `mae_mean` (model performance).
- y = x diagonal drawn as a reference line.
- Below diagonal = model beats persistence.
- Dot color: per-shot `frac_windows_below_diag` (with a perceptually
  uniform colormap, not rainbow) — so dense dot clouds resolve into
  "shots where the model wins consistently" vs "shots where it wins
  on average but loses on key windows".
- Title encodes: modality, split, total shots, **percent below
  diagonal** (the single most-quotable summary number — e.g.
  "ts_core_density val: 78% of shots beat copy").
- Equal aspect ratio so the diagonal is visually 45°.

### Per-shot summary (one plot per shot per modality)

A 2×2 grid:
- **TL:** time series of `mae` across window_idx for this shot
  (one point per window). Highlights time regions of poor prediction.
- **TR:** GT-vs-prediction for the *best* window of this shot.
- **BL:** GT-vs-prediction for the *worst* window of this shot.
- **BR:** histogram of MAE across all windows of this shot.

Per-window inset rendering depends on modality kind:
- **slow_ts** (e.g., 44 ch × 5 samples): line plot, one panel per
  ~4 representative channels (highest-variance channels of this shot).
- **fast_ts** (8 ch × 500 samples): line plot per channel, all 8 channels.
- **spectrogram** (40 ch × 512 freq × 96 time): one set of 3-panel
  heatmaps (GT, pred, |diff|) **per channel in the representative
  subset** — no channel averaging. Subset defaults: ECE/BES → 4
  channels each; CO2 → all 4. Selection rule is shared with the
  stitched-window plots.
- **video** (2 ch × 3 frames × 120 × 360): single middle frame, GT vs
  pred side-by-side.

### Stitched-window plots — paper-grade centrepiece

**This is the strongest visual evidence stage-1 has learned dynamics
(§2-Q3).** Design and execution must reflect that — these are the
plots that go in talks and papers.

Concatenate consecutive windows of a shot to give a long view of how
the model tracks dynamics over time. Default: **3 segments per shot,
each spanning ~80 windows ≈ 4 s of shot wall-time** (configurable).
Both the count and the length are deliberately longer than a
"diagnostic" view — short stitches don't reveal dynamics.

Modality-specific design:

- **slow_ts / fast_ts** — overlaid line plot, GT solid + prediction
  dashed, GT in the foreground; one panel per channel for fast_ts
  (8 panels); for slow_ts, the ~4 highest-variance channels per shot.
  X-axis in **physical time (seconds since shot start)**, derived
  directly from `window_idx × chunk_duration_s` since chunks are
  strictly monotonic in time within a shot (no gaps from filtering).
  Mark transient events (large GT excursions) where visible so the
  reader's eye is drawn to dynamics rather than baseline.

- **spectrogram** — **per-channel** stacked heatmaps (not averaged):
  for each modality pick a representative subset of channels and
  produce one figure per channel showing **GT (top) / predicted
  (middle) / |diff| (bottom)**, all on a shared frequency axis and
  shared time axis. Time axis in seconds. Shared colorbar between
  GT and pred (same vmin/vmax so the eye reads intensity
  consistently); |diff| uses its own colorbar centred at 0. The
  reader should be able to see:
  - **mode frequency tracking** — coherent horizontal features in
    GT that the model also reproduces;
  - **broadband turbulence changes** — increases/decreases in spectral
    density that the model should anticipate;
  - **what's missing or wrong** — the |diff| panel makes failure
    locations explicit (mode missed, turbulence onset late, etc.).

  Channel subset selection per modality (configurable via CLI):
  - ECE (40 ch): default 4 channels (selection rule TBD — pinning
    physically-meaningful indices is preferable to per-shot variance
    so plots stay comparable across shots; CLI flag to override).
  - CO2 (4 ch): show all 4.
  - BES (16 ch): default 4 channels (same logic as ECE).

  Optional: small text annotations naming features ("L-H transition",
  "sawtooth", "ELM") if a per-shot annotation source is available
  (defer to phase 3; not blocking).

- **video** — two complementary deliverables per selected shot:

  *Static grid plot* — 5×6 (or 6×5, configurable) **grid** of frame
  pairs sampled from the stitched segment, GT frame on top of each
  pair and predicted frame below, identical pixel intensity
  normalisation per pair. Time order reads row-major. Each cell
  titled with the frame's seconds-since-shot-start. Total ~30 frame
  pairs gives a readable single-page view of how the prediction
  tracks across ~4 s of shot wall-time.

  *MP4 video sequence* — one mp4 per selected shot showing GT,
  predicted, and `|GT − pred|` side-by-side, frame-by-frame at the
  video modality's native frame rate. Three panels per frame
  (left: GT, center: pred, right: |diff| with a clearly labelled
  colorbar). Spans the full set of stitched segments for that shot
  (so a viewer can watch ~12 s of dynamics if all 3 segments are
  rendered, with brief separators between segments). Encoded with
  `imageio` / `ffmpeg`; pixi env should already have a usable
  ffmpeg — if not, fall back to a sequence of PNGs in a numbered
  subdir plus a one-liner `ffmpeg` command in the README.

  The mp4 is the most expensive output (encoding cost + disk),
  but for the video modality it's the deliverable that actually
  conveys dynamics; static grids are inherently lossy for video.

## 6. Code organization

The plan document lives at `docs/eval_stage1_plan.md` (this file).

There is **pre-existing code** at `scripts/training/eval_e2e_stage1.py`
(1290 LOC) that implements much of the metric-collection side already:
copy-baseline, per-channel MAE, hexbin scatter, percentile sample
caching, 4-panel TS plot, video modality plot, JSON + `summary.md`
writers. **The plan document it was built against is not trusted**
(see §9 Phase 0) — the script must be audited fresh against this
plan rather than against its original spec.

Target code layout once audit + extensions are complete:

```
scripts/training/
└── eval_e2e_stage1.py            # extended in place if audit shows
                                  # close alignment, otherwise rewritten
scripts/slurm_frontier/
└── eval_e2e_stage1.sh            # Frontier-flavoured SLURM wrapper
                                  # (existing scripts/slurm/eval_e2e_stage1.sh
                                  # is for the legacy Princeton paths)
```

Modules ≤ ~300 lines. Whether helper files (`_shot_iter.py`,
`_metrics.py`, `_plots.py`) are split out depends on the audit
result — extend in place if `eval_e2e_stage1.py` is close enough,
split otherwise.

## 7. CLI surface

```bash
# Path may change after Phase 0 audit; this is the existing script location.
python scripts/training/eval_e2e_stage1.py \
    --checkpoint <path> \
    --data_dir   /lustre/orion/fus187/proj-shared/foundation_model \
    --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
    --splits     train val          # any subset of {train, val}
    --output_dir eval_runs/...      # auto-named if omitted
    --top_n      5                  # plots per modality
    --bottom_n   5
    --stitch_segments 3             # stitched plots per shot
    --stitch_windows  80            # windows per stitched segment (~4 s)
    --max_shots  0                  # 0 = all; small int for test runs
    --use_ddp                       # flip on for SLURM 8-rank
    --no_plots                      # metrics only, skip plotting phase
    --batch_size 8
    --num_workers 4
```

## 8. Reuses from training code (memory: reuse, don't reinvent)

- `build_configs(...)` from `train_e2e_stage1.py` — modality lists.
- The existing dataset class (whichever the trainer instantiates) with
  `prediction_mode=True`. We will *not* reimplement file scanning.
- `load_state_dict_explicit` — same allowed_missing_prefixes pattern.
- `DistributedManager` — for the DDP path.
- `_clean_and_mask` and the masked-MAE helper — exact same metric
  semantics as training.

## 9. Phased delivery

Built and reviewable in four phases. Phase 0 is new — it exists because
prior work in this area produced
`scripts/training/eval_e2e_stage1.py` (1290 LOC) and
`docs/eval_stage1_panels_patch.md` (an unmerged patch).
The prior plan they were built against is **not trusted**, so we
audit the artefacts against *this* plan before any new code lands.

**Phase 0 — Audit existing `eval_e2e_stage1.py` against this plan**
(no new code)
- Read the script end-to-end and the unmerged
  `docs/eval_stage1_panels_patch.md`.
- Build a checklist mapping each requirement in §2 / §3 / §5 of
  this plan to one of:
  - (a) an existing function/class already covers it,
  - (b) covered partially, needs extension,
  - (c) no current implementation.
- Output: an `## Audit findings` section appended to this plan
  document, with explicit references like
  `eval_e2e_stage1.py:168 copy_baseline_for_modality already covers
  §3 copy_mae per-modality, but operates on batch averages — needs
  extension to emit per-window rows`.
- The audit decides whether Phase 1 is "extend in place" or
  "rewrite". Do not skip Phase 0 — skipping is exactly how the
  duplicate plan was created in the first place.

**Phase 1 — Metrics only** ✅ **First cut landed** at
`scripts/training/eval_e2e_stage1_phase1.py` (~500 LOC). Re-uses
the audit-approved helpers (`forward_one_batch`, `copy_baseline_for_modality`,
mask helpers) from `eval_e2e_stage1.py` via direct import; everything
downstream is fresh code.

What the first cut delivers:
- DDP-aware shot-sharded inference loop (env-var-detected; falls
  back cleanly to single-process / single-GPU). World-size 1 runs
  on a login node or a 1-GPU compute node interactively.
- Per-window CSV.gz: `(split, modality, kind, shot_id, window_idx,
  window_t_s, mae, copy_mae, mae_ratio, dcos, mag_ratio)`.
- Per-shot CSV.gz with `n_windows`, `mae_{mean,median,p95,max}`,
  `copy_mae_{mean,median}`, `mae_ratio_{mean,median}`,
  `frac_windows_below_diag`, `dcos_mean`, `mag_ratio_mean`.
- `top_bottom_shots.csv.gz`: top-N + bottom-N per (split, modality),
  ranked by `mae_ratio_mean`.
- Shot identifiers parsed from the `<shot_id>_processed.h5`
  filename convention; window index derived from each rank's local
  dataset `_cumulative_lengths`. No modification to the shared
  dataset class or collate function.
- `config.json` snapshot of args + checkpoint + row counts.

What it deliberately does NOT do (Phase 2/3 work):
- No plotting.
- No mp4.
- No spectrogram per-channel heatmaps.
- No SLURM wrapper yet — submission script lands in Phase 3.

Test plan (smoke before any full-split run):
- `--splits val --max_shots 10` on a 1-GPU interactive node →
  verify `per_window_metrics.csv.gz` has plausible row counts and
  finite-valued mae/copy_mae for at least the TS modalities,
  spot-check against the existing eval script for the same shots
  (regression check).
- Then `--splits train val --max_shots 20` on the same node →
  confirm split column distinguishes correctly.
- Only after both pass: scale to full split (Phase 3 SLURM wrapper).

**Phase 2 — Plots from CSV + per-shot summary** (split into 2.0 / 2.1
during delivery)

**Phase 2.0 — Aggregate-quality scatter** ✅ landed at
`scripts/training/eval_e2e_stage1_phase2_plots.py`.
- CSV-only (reads `per_shot_metrics.csv.gz`); no GPU, no
  checkpoint required. Runs on a login node in ~seconds.
- One scatter per (split, modality), one dot per shot,
  y = model_mae_mean vs x = copy_mae_mean. Diagonal reference,
  color = `frac_windows_below_diag`. Title prints
  percent-below-diagonal — the §2-Q1 headline number.

**Phase 2.1 — Per-shot 2×2 summary plots** ✅ landed at
`scripts/training/eval_e2e_stage1_phase2_per_shot.py`.
- Re-inference required (needs `--checkpoint`).
- Runs in a separate **1-node 1-GPU SLURM job**
  (`scripts/slurm_frontier/eval_e2e_stage1_phase2_per_shot.sh`),
  not as rank-0 of the original Phase 1 DDP eval.
- 2×2 grid per (selected shot, modality):
  - TL: MAE-vs-window time series (from CSV)
  - TR: GT-vs-pred for the best window of this shot (from re-inference)
  - BL: GT-vs-pred for the worst window of this shot (from re-inference)
  - BR: per-window MAE histogram (from CSV)
- **Coverage-aware `--max_shots_to_plot` cap**: greedy set-cover by
  modality **kind** (slow_ts / fast_ts / spectrogram / video), then
  fill remaining capacity by selection count. Guarantees that
  cap ≥ 4 covers one shot of every kind. Implemented in
  `_coverage_aware_shot_order()` in `eval_e2e_stage1_phase2_per_shot.py`.
- Throughput characteristics (1-GPU MI250X, batch=128):
  - First shot: ~6 min (dominated by model load + JIT warmup
    + HDF5 first-open).
  - Subsequent shots: ~1 min each (steady-state).
  - Full run on ~59 unique selected shots: **~1.5–3 h** wall time.

**Phase 3 — Stitched plots + mp4 + SLURM wrapper** ⏳ next
- Stitched-window plots: re-inference produces predictions for
  consecutive windows, then a long-range comparison view of how
  the model tracks dynamics across ~4 s of shot wall time.
- See §5 for per-kind layout and §10 Q9/10/11 for the still-open
  specification decisions (segment selection, mp4 layout,
  per-channel spectrogram filename convention).
- SLURM wrapper mirrors Phase 2.1's (1-node 1-GPU). DDP-shot-sharded
  variant is a future optimisation if the wall time exceeds
  what an overnight run can absorb.
- End-to-end smoke on a small `--max_shots_to_plot` cap before
  full run.

## 10. Open questions

1. ~~Window-time-axis for stitching~~ — **resolved.** Chunks are
   strictly monotonic in time within a shot, so the stitched plot's
   x-axis is `window_idx × chunk_duration_s` (seconds since shot
   start). No timestamp lookup needed.
2. ~~Channel subset for spectro plots~~ — **resolved: no averaging.**
   Use a representative subset of channels per modality (defaults:
   ECE/BES 4 each, CO2 all 4). Selection rule for the subset is still
   TBD — pinning a fixed set of channel indices (physically meaningful
   ones) is preferable to per-shot variance so plots are comparable
   across shots; CLI flag to override.
3. ~~Video stitching layout~~ — **resolved.** Static plot is a 5×6
   grid of GT/pred frame pairs (~30 frame-pairs total per stitched
   segment). Plus an **mp4 sequence per selected shot** showing
   GT / pred / |diff| side-by-side at the video's native frame
   rate — this is the deliverable that actually conveys dynamics
   for video, since static frames are lossy.
4. **Tangential-density magnitude-bias panel** — these had `mrat`
   values far from 1 in earlier training logs. Plan: add the panel
   to the summary-plot infrastructure (so it's a flip-on, not a
   re-architecture), but **defer interpretation** — the panel is
   likely not load-bearing for the paper. Keeps the option without
   committing to it.
5. **`bes`/`co2` `copy_mae ≈ 0` data-pipeline question** —
   surfaced in the Phase 1 smoke. For these two spectrogram
   modalities the per-window `copy_mae` is zero for the vast
   majority of windows, suggesting the dataset emits the same
   spectrogram tensor for both `inputs[name]` (at t) and
   `targets[name]` (at t + 50 ms). `ece` works correctly — it has
   `copy_mae_mean` in the 0.3 range. Either there's a
   data-loader bug specific to BES/CO2, or the dataset's
   spectrogram-rendering path emits identical tensors for both
   sides of the prediction horizon for those modalities only.
   Phase 3 plots will surface the artifact visually; the upstream
   fix is a separate investigation.
6. **Degenerate `mae_ratio_mean` in top/bottom-N selection** —
   `top_bottom_shots.csv.gz` for `filterscopes` / `tangtv` currently
   includes shots with `copy_mae_mean ≈ 0` giving pathological
   ratios (44, 67). These dominate the bottom-N pool without
   reflecting a real failure mode. Need either:
   (a) a minimum-`copy_mae_mean` threshold filter in
   `select_top_bottom()` (drop shots where copy is too close to
   zero to give a meaningful ratio), or
   (b) a minimum-`n_valid_windows` filter, or
   (c) leave the artifact visible and rely on the aggregate scatter
   to flag denominator-degenerate cases. Decision deferred.
7. **Phase 3 stitched segment selection** — plan §5 specifies
   "3 segments × ~80 windows each ≈ 4 s". Still open: where in the
   shot do the 3 segments live?
   Candidates: **(a) evenly spaced at 25%/50%/75% of shot length**
   (deterministic, simple, comparable across shots), (b) centred
   on best/median/worst windows (more informative, less
   comparable), (c) one fixed early segment + the worst run of
   consecutive high-loss windows (highlights failure dynamics).
   **Decision: (a) — evenly spaced 25/50/75% of shot length**,
   span 80 windows each. Future iterations can revisit if shots
   of very different length (≪ 4 s × 3 segments) leave gaps.

   **Stride math (added 2026-05-18 after the first cut shipped):**
   The dataset uses `step_size_s = 0.01s`, not `chunk_duration_s = 0.05s`,
   so raw consecutive windows step every 10 ms and overlap by 80% of
   their content. Naively concatenating "80 consecutive windows × 5
   samples" produces a non-monotonic time series with massive overlap
   — and labels the span as 4 s when the underlying shot wall-time is
   only 0.8 s. Phase 3.0 fixes this with a **stride =
   `chunk_duration_s / step_size_s` = 5**: only every 5th window from
   the segment range is kept, giving 80 stride-5 windows whose
   predictions are exactly non-overlapping and span 4.00 s of real
   shot wall-time. Each segment's raw window range is 400 windows
   wide (`80 × 5`); stride selection happens in
   `collect_stitched_segments_for_shot()`.
8. **Phase 3 mp4 layout** — open until coding starts:
   - **Frame rate**: native (tangtv = 3 frames per 50 ms window
     → 60 frames/s).
   - **Per-shot vs per-segment**: one mp4 per shot, concatenating
     all 3 segments with a brief separator frame between them
     (decision: per-shot — easier downstream playback than
     juggling 3 files per shot).
   - **Resolution**: native 120 × 360 per frame, GT and model
     side-by-side with an additional `|GT − model|` panel ⇒
     final mp4 is 120 × 1080 (3 panels wide).
   - **Codec / container**: `imageio` + `ffmpeg` writer
     (`libx264` default). Fall back to a numbered PNG sequence +
     a `make_mp4.sh` one-liner if ffmpeg isn't on the env's
     PATH.
9. **Phase 3 stitched per-channel spectrogram filename
   convention** — `<shot_id>_stitched_<seg>_ch<channel_idx>.png`
   for spectrograms (since they emit one PNG per channel in the
   representative subset). Non-spectrogram modalities emit one PNG
   per stitched segment: `<shot_id>_stitched_<seg>.png`.
   Documented in §3.
10. **Annotation overlays on stitched plots** (`L-H transition`,
    `sawtooth`, `ELM`, etc.) — flagged as optional in §5; no
    annotation source has been wired up yet. **Decision: defer
    until Phase 3 first cut is reviewed; not load-bearing.**

## 11. Risk / non-goals

- Not building this as a generic eval framework — single-purpose for
  stage-1 reconstruction. Stage-2 delta-rollout evaluation is a
  separate script (different prediction semantics).
- Not measuring inference latency / throughput; this is a *quality*
  evaluator, not a perf evaluator.
- Not handling checkpoints saved by `torch.save` with non-default
  pickle protocols. The trainer uses default protocol so this isn't
  an issue, but worth flagging if checkpoints change format.
- **Not chasing tangtv (video) reconstruction quality**. Phase 1
  smoke established that `tangtv` has `mae_ratio_mean ≈ 13.5` — the
  copy baseline (consecutive video frames at 50 ms separation barely
  change) is hard to beat in this single-step prediction objective.
  The video modality is included in eval for completeness, but
  improving it is a separate research item, not within the eval
  pipeline's scope.

## 12. Audit findings (Phase 0 — `scripts/training/eval_e2e_stage1.py`)

Read of `eval_e2e_stage1.py` (1290 LOC) and
`docs/eval_stage1_panels_patch.md`. The patch is **already integrated**
into the script — the "with this:" block in the patch matches the
current main loop verbatim. Treat `eval_stage1_panels_patch.md` as
a historical artefact only.

### Re-usable as-is (small helpers, semantics match training)

| Plan section | Existing artefact | Status |
|---|---|---|
| §1 (K=1 next-chunk, prediction_mode=True) | `forward_one_batch` (110), dataset build in `main` (1089) | direct reuse |
| §3 `copy_mae` metric | `copy_baseline_for_modality` (168) | direct reuse |
| §3 mask handling | `_clean_and_mask` (61), `_video_loss_gate` (80), `_ts_mask` (96) | direct reuse |
| §1 video standardisation parity with trainer | `_video_standardize_per_bc` (72) | direct reuse |
| Checkpoint load + LoRA detection | `main` lines 1033–1058 | direct reuse |

### Partial — needs extension

| Plan requirement | Existing | Gap |
|---|---|---|
| §2 train + val splits | `resolve_val_files` (194) returns val only | add train-split case; refactor signature to accept a split name |
| §3 per-channel breakdown | `PerChannelAccumulator` (306) covers MAE per channel per modality | Plan doesn't strictly require this, but it's useful — keep |
| `summary.md` with PASS/FAIL on copy-baseline | `write_summary_md` (925) implements milestone A2 gate | keep alongside new parquet outputs |
| JSON metrics dump | `write_metrics_json` (877) | keep alongside parquet — JSON for one-glance global numbers, parquet for the per-window/per-shot tables |
| Quality-bar styling (§5 callout) | Existing plots use varied colours/legends (e.g. Panel D uses `C0`/`C2`/`C3`) | needs a global conventions pass — central palette helper, GT=black/pred=dashed-blue, no legends where convention is global |

### Diverges from plan — needs rewrite or replacement

| Plan requirement | Existing | Why divergence |
|---|---|---|
| §3 per-window metric rows (`per_window_metrics.parquet`) | `GlobalAccumulator` (209) batch-means each metric and aggregates to a single scalar per modality | Cannot produce per-window rows. Replace with a per-window emitter that writes (shot_id, window_idx, modality, mae, copy_mae, mae_ratio, dcos, mag_ratio) directly. |
| §3 per-shot aggregation | not implemented | Needs (shot_id) carried through the data loader / batch; existing dataset emits chunks without exposing shot_id in the batch dict — **first thing to verify in Phase 1**. |
| §5 aggregate-quality **shot-level** scatter (dot per shot) | `HexbinAccumulator` (373) is a value-level density (pred-value vs target-value, every (sample, channel, timestep)) | Different plot entirely. Existing hexbin can stay as a supplementary panel; the new shot-scatter is a separate plotter. |
| §5 top/bottom-N **shot** selection by `mae_ratio` | `PercentileSampleCache` (427) holds *first 8 batches* of *samples*, ranked by raw MAE | Mismatch on three axes: (i) first 8 batches ≠ full split, (ii) samples ≠ shots, (iii) raw MAE ≠ MAE ratio. Rewrite as a full-split shot-aggregator. |
| §5 per-shot 2×2 summary (MAE-vs-time / best window / worst window / MAE histogram) | `plot_ts_4panel` (630) is per-modality, not per-shot; layout is (A demo-shot trajectory / B per-channel bars / C hexbin / D best-median-worst) | Different plot. Existing 4-panel can survive as a "per-modality overview"; the new per-shot 2×2 is a new generator. |
| §5 stitched-window plots at scale (~80 windows × 3 segments × top/bottom-N shots, per modality) | `collect_demo_shot_trajectory` (481) handles one shot, TS only, single channel chosen by best-improvement | Same idea, much smaller scope. Generalise to many shots, configurable segment count/length, all modality kinds (incl. spectrogram per-channel heatmaps and video grid). |
| §5 video 5×6 grid + mp4 | `plot_video_modality` (832) shows sample 0, frame 0, all channels in 4 columns (ctx/tgt/pred/|diff|) | Conceptually similar but different scope (single frame vs. ~30 frame-pairs grid vs. mp4). Rewrite. |
| §4 DDP shot-sharded inference | not present — single-GPU only | New requirement. Either retrofit the existing `main` to wrap the loader with `DistributedSampler` keyed on shots, or restructure into a worker function dispatched by `DistributedManager`. Lean toward the latter for clean rank-0 plotting phase. |

### Missing entirely

- Parquet output schema (`per_window_metrics.parquet`,
  `per_shot_metrics.parquet`, `top_bottom_shots.parquet`).
  Existing outputs are JSON (global) + CSV (per-channel) + PNG.
- `frac_windows_below_diag` per-shot statistic.
- `mae_ratio` per-window column.
- MP4 encoding pipeline.
- Spectrogram per-channel stacked heatmaps (existing video plot does
  per-channel but spectrogram has no analogous plot path).
- Shot-id propagation from the dataset into the batch dict —
  `TokamakMultiFileDataset` emits chunks but the eval script doesn't
  use any shot identifier in the inner loop; verify whether the
  dataset already exposes one and, if not, add it.

### Architectural recommendation (informs Phase 1 scope)

The existing script is ~60% gap and ~40% reusable. The reusable
~40% is concentrated in the helpers and the inference forward pass;
everything downstream (accumulators, output, plots, DDP) needs new
code. Concrete recommendation:

1. **Keep as a module of helpers**, not as the main eval entry. Move
   `_clean_and_mask`, `forward_one_batch`, `copy_baseline_for_modality`,
   video standardisation helpers, checkpoint-load logic into a
   `scripts/training/eval_helpers.py` (or similar).
2. **Rewrite** `main`, all accumulators, all plot functions, and the
   output writers in a new entry script. Whether that entry lives at
   `scripts/training/eval_e2e_stage1.py` (overwriting) or
   `scripts/training/eval_e2e_stage1_v2.py` (parallel during the
   migration) is the user's call.
3. **Preserve** `metrics.json` + `summary.md` as supplementary
   outputs alongside the new parquet files — they're cheap and the
   PASS/FAIL gate is genuinely useful at-a-glance.
4. **Verify shot-id availability** in the dataset's batch dict
   before committing Phase 1 — if it's not there, that's the first
   plumbing change needed.

### Pre-Phase-1 verification (one read, before any code)

Before Phase 1 begins, check `TokamakMultiFileDataset` (in
`src/tokamak_foundation_model/data/multi_file_dataset.py`) to confirm:
- whether each emitted chunk carries a `shot_id` (or equivalent
  file-index) field;
- whether windows from one shot are contiguous in the loader's
  output (assumed by §4 DDP shot-sharding);
- whether `prediction_mode=True` provides the `inputs` / `targets`
  dict shape the existing `forward_one_batch` expects (already
  proven in production, so should be fine).
