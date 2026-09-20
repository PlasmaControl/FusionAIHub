---
title: Frontier SLURM scripts
sidebar_position: 2
---

`scripts/slurm_frontier/` holds two things: a handful of shared setup
scripts sourced by everything else, and one script per job (training run,
evaluation, profiling probe, or `shot_design` pipeline stage). Most of the
directory is one-off experiment scripts named after their own configuration
(`train_e2e_stage1_d1024_48L.sh`, `mode_audit_gate.sh`, and so on) — read
their own header comment for what they do rather than expecting a table
here to cover all of them. This page tables the shared infrastructure and
the `shot_design` family, which other pages link to.

## Shared setup (sourced, never run directly)

| Script | Sets up |
|---|---|
| `_frontier_settings.sh` | modules (`PrgEnv-gnu/8.7.0`, `cpe/26.03`, `rocm/7.1.1`, `craype-accel-amd-gfx90a`), `PATH`/`LD_LIBRARY_PATH` for the `frontier` pixi env, `PYTORCH_ROCM_ARCH`, the MIOpen cache, `FLASH_ATTENTION_TRITON_AMD_ENABLE` |
| `_frontier_common.sh` | ROCm DDP job environment: RCCL/NCCL knobs, MIOpen cache, `MASTER_ADDR`/`MASTER_PORT`, for multi-node training jobs |
| `_shot_design_common.sh` | sources `_frontier_settings.sh`; exports `REPO`, `ROOT` (`$SHOT_DESIGN_DATA_ROOT`), `PY` (the `shot-design-frontier` interpreter path), `SHOT_DESIGN_PATHS`/`LABELER_ROOT`/`SHOT_DESIGN_CORPUS`, `RCCL_PLUGIN=0` (single-GPU/CPU jobs need no plugin) |
| `_gpu_sampler.sh` | background `rocm-smi` sampler, one line every interval to `<jobid>.gpu.csv`; `labeler.jobstats.parse_frontier` reads it back as the only GPU-utilization source OLCF offers |
| `setup_frontier_env.sh` | flash-attention 2 (Triton backend) build for MI250X; run via `pixi run -e frontier setup-flash-attn` on a login node, no GPU needed at build time |

Every wrapper below is submitted from the repo root — `_shot_design_common.sh`
and `_frontier_common.sh` are sourced relative to `$SLURM_SUBMIT_DIR`, which
`sbatch` sets to the directory it was invoked from, not the script's own
location (the spooled copy `sbatch` runs has no relation to the checkout).

## `shot_design` pipeline (Task B3/B4/D4)

| Script | Partition / resources | Purpose |
|---|---|---|
| `shot_design_census.sh` | `batch`, 1 node, CPU, 2h | corpus census → `db/corpus_coverage.parquet` |
| `shot_design_build.sh` | `extended`, 1 node, CPU, 6h | full database rebuild for a shot list (`SHOT_LIST`, `BUILD_ARGS`) |
| `shot_design_encode.sh` | `extended`, array 0-7, 1 GPU each, 6h | IGNITE frame-code encoding, one contiguous chunk of the shot list per task |
| `shot_design_genc.sh` | `batch -q debug`, 1 GPU, 1h | G-ENC gate: fresh encode vs. the production frame-codes cache |
| `shot_design_simulate.sh` | `batch -q debug`, 1 GPU, 1h | one design's paired real/proposed IGNITE rollout (see [Simulation](../shot-design/simulation.md)) |

See [Database build](../shot-design/database-build.md) for the full
census → select → labels → build → blurb → encode sequence and exact
commands, and [Frontier](../clusters/frontier.md#slurm-wrappers) for the
submit-from-repo-root rule.

## Training and evaluation

The foundation-model / IGNITE training scripts (`train_e2e_stage*.sh`,
`train_dynamics*.sh`, `train_fsq_codec.sbatch`, and the codec/eval family
`ignite_codec_prod.sh`, `eval_dynamics.sh`, `eval_e2e_*.sh`) predate this
port and follow their own naming and resource conventions per experiment —
most declare their own `#SBATCH -N`/`--gres`/`-t` rather than sharing a
common wrapper, since a training job's node count and walltime are part of
what is being varied. `train_dynamics.sh` (`extended`, 16 nodes, 8 GPUs
each) is the multi-node dynamics training entry point referenced from
[IGNITE](../models/ignite.md); read a given script's own header before
resubmitting it, since several encode one-off flags for a specific
experiment (`_kanneal_g3fix_flags.txt`, `_gate4_kanneal_k10.sbatch`, and
similar).

## House style

Every script under `scripts/slurm_frontier/` that is not one of the shared
setup files above declares `#SBATCH -A fus187`, a partition of `batch` or
`extended`, and writes its output under `<data_root>/runs/slurm/%j.out` (or
`%A_%a.out` for an array). `tests/shot_design/test_slurm_frontier_scripts.py`
enforces this for the `shot_design_*.sh` family specifically: the account
line, the partition, that `_shot_design_common.sh` is sourced, that no
Stellar path (`/scratch/gpfs`) or Stellar QOS (`gpu-stellar`, `pppl`) leaked
in, and that logs land under `runs/slurm/`.
