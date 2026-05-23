# Profiling file-open cost for `train_e2e_stage1` on Frontier

**Date:** 2026-05-11
**Author:** nchen
**Status:** Design — approved, plan pending

## Goal

Measure the end-to-end file-open cost of an `e2e_stage1` training job on Frontier
(Lustre filesystem, ~8753 shot HDF5 files at `/lustre/orion/fus187/proj-shared/foundation_model`),
and decide whether it is a real problem that needs mitigation.

## Background

`scripts/training/train_e2e_stage1.py` uses
`tokamak_foundation_model.data.multi_file_dataset.TokamakMultiFileDataset` to read
single-shot HDF5 files. File opens happen in two distinct places:

1. **Startup indexing pass.** `_load_or_compute_lengths()` opens every shot HDF5
   sequentially to read its duration and compute a chunk count. Results are
   cached to a `.pt` sidecar; subsequent runs short-circuit this entirely.
2. **Steady-state, during training.** Each DataLoader worker has its own LRU
   cache of `h5py.File` handles, bounded by `max_open_files=1024`. Cache hits
   are free; cold misses re-open with `h5py.File(path, "r", rdcc_nbytes=0)`.
   Per-worker counters (`_prof_opens`, `_prof_hits`, `_prof_open_s`,
   `_prof_close_s`, `_prof_getitem_s`) are already in place.

Existing infrastructure we'll reuse:
- `scripts/profile_indexing.py` — times Phase 1.
- `scripts/slurm_frontier/profile_indexing.sh` — Frontier launcher for the above.
- `scripts/training/profile_stage1.py` — `torch.profiler` on the full train step.
- `scripts/training/probe_stage1_loading.py` — single-process `__getitem__` timing.

Prior measurements (`logs/4555562_idx_profile.out`):
- 100-file run: 6.00 files/s, predicted ~33 min on full 8753.
- Two full-dataset attempts (jobs 4555563, 4558113) did **not** finish: the first
  timed out at 1 h walltime, the second failed at 7 s (exit 1).
- `runs/lengths_cache_e2e_stage1/` is currently empty.

## Scope

**In:**
- Single Frontier job, one node, production training config (8 DDP ranks ×
  4 workers/rank × batch 16, pulled from `scripts/slurm_frontier/train_e2e_stage1.sh`).
- Both phases: full-dataset indexing + ~200 steady-state training steps.
- A written verdict on whether file-open cost is acceptable or needs work.

**Out:**
- Multi-node coordination measurements.
- Multiple worker-count sweeps (4 vs 8 vs 16). One config only.
- Lustre stripe-config experimentation.
- Changes to the production training script.

## Plan

### Phase A — startup indexing (full dataset)

Run `scripts/profile_indexing.py` with no file cap against the full data
directory, writing the lengths cache to `runs/lengths_cache_e2e_stage1/`. Walltime
budget **3 h** (the prior 1 h attempt timed out).

Measurements:
- Total wall time, files/s, valid/skipped count, total chunks.

Side benefit: populates the lengths cache so all future training jobs skip the
indexing wall entirely.

### Phase B — steady-state opens during training

Run a new thin script `scripts/training/profile_stage1_opens.py` that mirrors
the existing `scripts/training/profile_stage1.py` structure (imports
`build_configs`, `build_datasets`, `resolve_shot_files`, `compute_step_loss`
from `train_e2e_stage1.py` — no changes to the production script).

Configuration to match production (`train_e2e_stage1.sh`):
- 8 DDP ranks per node, 1 GPU per rank, `--gpu-bind=closest`.
- 4 DataLoader workers per rank (32 workers total).
- `batch_size=16`, `chunk_duration_s=0.05`, `step_size_s=0.01`, `warmup_s=1.0`,
  `prediction_horizon_s=0.05`, `d_model=256`, `n_layers=8`, `n_heads=8`.
- Reuse the lengths cache from Phase A.

Run ~200 training steps. At the end, each worker dumps its profiling counters
(`_prof_opens`, `_prof_hits`, `_prof_open_s`, `_prof_close_s`, `_prof_getitem_s`,
`_prof_load_s`, `_prof_process_s`) to a per-worker JSON file in
`runs/profile_e2e_stage1_opens/per_worker/`.

Rank 0 reads all per-worker JSONs after `dist.barrier()`, aggregates, and
writes `summary.json` plus a human-readable `report.md`.

If the existing in-place stdout logging (every 50 calls) is sufficient
to extract these numbers from the SLURM log, the JSON dump can be skipped in
favor of a `parse_log.py` post-processor. We will pick whichever is simpler
during implementation; the spec does not lock in one approach.

### Putting them together

Single launcher `scripts/slurm_frontier/profile_e2e_stage1_opens.sh`:
- `#SBATCH -t 03:00:00`, 1 node, account `fus187`.
- Runs Phase A first (CPU-only mode by calling the python script directly,
  not via `srun`), then Phase B (via `srun -n 8 --gpu-bind=closest …`).
- Each phase writes to its own subdirectory under `runs/profile_e2e_stage1_opens/`.

## Outputs

All in `runs/profile_e2e_stage1_opens/`:

- `indexing.log` — Phase A stdout: wall time, files/s, valid/skipped, total chunks.
- `per_worker/rank{R}_worker{W}.json` — raw per-worker counters from Phase B.
- `summary.json` — aggregated open counts / hit rate / open-wall across the
  32 workers; `__getitem__` time breakdown.
- `report.md` — synthesis and verdicts (see below).

Side effect: `runs/lengths_cache_e2e_stage1/lengths_e2e_stage1_{train,val}.pt`
populated for future runs.

## Verdict criteria (to include in `report.md`)

| Question | Threshold | Source |
|---|---|---|
| Is full-dataset indexing tolerable? | < 30 min OK; 30–60 min worth pre-caching; > 60 min should be a permanent cache or restripe | Phase A wall time |
| Is the training loop open-bound? | Open-wall fraction of `__getitem__` < 5 % = good, 5–20 % = OK, > 20 % = bad | Phase B `_prof_open_s / _prof_getitem_s` |
| Is `max_open_files=1024` right-sized? | Hit rate > 95 % in steps 100–200 = fine; less = LRU churn | Phase B `_prof_hits / (_prof_hits + _prof_opens)` |
| Cold-start to first useful step | Indexing + warm-up; report as a number | Phase A + Phase B step-1 timing |

Each verdict comes with a one-line recommendation: leave alone / pre-cache /
resize LRU / restripe / something else.

## Expected back-of-envelope (sanity check)

- 32 workers, 8753 files → ~274 files/worker. LRU=1024 means every worker fits
  its slice — cold opens should happen at most once per file per worker.
- A pure `h5py.File()` open on Lustre is plausibly 20–100 ms (no duration
  scan). At ~50 ms × 274 files = ~14 s of cold-open wall per worker, amortized
  across the entire epoch.
- If the actual hit rate is much below 95 %, that's a red flag worth digging
  into (DistributedSampler shard, `TwoLevelSampler` interaction, or per-worker
  shard size larger than expected).
- Indexing throughput on Lustre is the dominant unknown. The prior 100-file
  warm-cache extrapolation predicted 33 min but the full run timed out at 1 h,
  so the true rate may be 2–4× slower than the small-N extrapolation suggested.

## Open questions / decisions deferred to plan

- Whether to dump counters via per-worker JSON files or parse the existing
  stdout log (pick simpler at implementation time).
- Whether Phase A and Phase B share one SLURM job or run as two
  `--dependency`-linked jobs (one job is simpler, picked here unless Phase A
  is unstable enough to need re-runs).
- Whether to add an MPI broadcast of `__getitem__` step-1 timing for end-to-end
  cold-start, or just report indexing wall + a single rank's step-1 time.
