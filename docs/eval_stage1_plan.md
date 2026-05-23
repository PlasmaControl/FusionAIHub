# Stage 1 Evaluation Script — Plan

**Goal.** Given a frozen Stage 1 checkpoint (Phase A or Phase C), run single-step
(K=1) prediction over the **full** val set and produce a complete evaluation
report. Answer "did Stage 1 milestone A2 pass?" (single-step MAE below copy
baseline for all modalities, per `ResearchPlan.MD` §6.1).

## Decisions already locked in

- **Supports both Phase A Stage 1 (`runs/e2e_stage1/`) and Phase C Stage 1
  (`runs/c_stage1/`)** checkpoints. Same model class; the only difference is
  `--use_video tangtv` for C-Stage 1.
- **Fresh val loop** (not reusing trainer's `validate()`). ~50 LOC more, but
  decouples eval from trainer changes and lets us cleanly add direction_cos
  and magnitude_ratio.

## Open decision: which tier?

### Tier 1 — Minimum viable (~1 day, ~250 LOC)

Just the numbers, no plots.

- Load checkpoint via the same logic as
  `tests/e2e/test_rollout_trained.py:139–161` (handles LoRA detection, video
  diagnostics, architecture reconstruction from saved configs).
- Build val dataset matching the training split: `val_fraction`, `seed`,
  `chunk_duration_s`, `step_size_s`, `warmup_s` from CLI. Deletes
  `lengths_*.pt` if window params changed (known footgun, see
  `feedback_chunk_cache_bug` memory).
- Full-val K=1 loop. Per modality compute:
  - `MAE_model`
  - `MAE_copy` (predict `t = t + 50ms`, i.e. output = input)
  - `Δ = MAE_copy - MAE_model` (positive = beating copy)
  - **`direction_cos`** = `cos_sim(pred - ctx, tgt - ctx)` averaged over batch
  - **`magnitude_ratio`** = `||pred - ctx|| / ||tgt - ctx||` (target ≈ 1)
- Print a table to stdout in the same format the trainer uses, with the extra
  columns, on the **full** val set (not just 20 batches).
- Write `metrics.json` with per-modality numbers and a top-level `a2_pass: bool`.

### Tier 2 — Adds plots and per-channel detail (+0.5 day)  ← my recommendation

Everything in Tier 1, plus:

- **Per-channel MAE breakdown** as `per_channel.csv`. Catches "ts_core_density
  mean OK but channel 23 is nuked".
- **Per-modality `pred vs target` overlay plots** for N random val samples
  (default 4). One PNG per modality.
- **`summary.md`** — human-readable PASS / FAIL on A2, table of marginal
  modalities, links to plots.

### Tier 3 — Adds C3 latent-continuity (+0.5 day)

Everything in Tier 2, plus:

- Spearman correlation of `cos_sim(window_t, window_{t+1})` between raw signal
  and tokenizer output, per modality. Already implemented in
  `debug_e2e_latent_continuity.py` — would just call its core function.
- This is the metric `ResearchPlan.MD §1.1 / C3` cites as the *headline* Stage 1
  result vs. AE baseline (Spearman ≤ −0.1 for AE, expected > 0.5 for E2E).
- Gated behind `--compute_continuity` flag (slower; needs separate dataset
  iteration with `chunk_duration_s = 0.1`, `step_size_s = 0.1`).

## File layout

```
scripts/training/eval_e2e_stage1.py       # the script
scripts/slurm/eval_e2e_stage1.sh          # SLURM wrapper
                                          # (1× GPU, ~30 min full val at b=128)
```

Output directory layout:

```
runs/e2e_stage1/eval_<jobid_or_step>/
  metrics.json              # all numerical results
  per_channel.csv           # Tier 2+
  plots/<modality>.png      # Tier 2+
  summary.md                # Tier 2+
```

## CLI surface

```bash
pixi run python scripts/training/eval_e2e_stage1.py \
    --checkpoint runs/e2e_stage1/e2e_stage1_best.pt \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path scripts/slurm/preprocessing_stats.pt \
    --output_dir runs/e2e_stage1/eval_best \
    --batch_size 128 \
    --num_workers 8 \
    --val_fraction 0.1 \
    --seed 42 \
    --chunk_duration_s 0.05 \
    --step_size_s 0.01 \
    --warmup_s 1.0 \
    [--use_video tangtv]            # for C-Stage 1 checkpoints
    [--max_batches 50]              # quick smoke-test mode
    [--compute_continuity]          # Tier 3 only
```

## What changes between Phase A and Phase C eval

- `--use_video tangtv` adds the video diagnostic to the model config.
- All other args identical.
- Output `metrics.json` will have an extra `tangtv` entry alongside the TS
  modalities. A2 gate is checked across all modalities present in the
  checkpoint.

## Question for you

**Tier 1, 2, or 3?**

I recommend **Tier 2**: all the numbers needed for the A2 gate, plus plots for
sanity-checking, without coupling to the C3 plumbing. Tier 3 can be added later
as a flag once Tier 2 is working.
