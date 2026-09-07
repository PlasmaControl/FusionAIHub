#!/bin/bash
#SBATCH --job-name=ideate-encode
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%A_%a.out

# IGNITE frame-code caches for the 500 recommender_v1 shots: the per-shot seeds a Phase-5
# rollout starts from. Each array task takes one contiguous half of the sorted list
# (`--chunk $SLURM_ARRAY_TASK_ID --n-chunks 2`) so the two tasks walk different regions of the
# corpus directory rather than competing for the same 2-5 GB files.
#
# REPO is the MAIN checkout, not the I8 worktree: this script is written on branch
# recommender-I8 but is run after the merge, from /scratch/gpfs/nc1514/FusionAIHub.
#
# `--ntasks=1`, not the brief's 2. The brief asks for both `--ntasks=2 --gpus-per-task=1` and
# `--array=0-1%2` with `--chunk $SLURM_ARRAY_TASK_ID`, and those two say different things: with
# ntasks=2 under srun BOTH tasks of an array element get the same SLURM_ARRAY_TASK_ID, so they
# would encode the same 250 shots twice on two GPUs. The array index is what the brief actually
# uses to split the list, so the array is what stays: two elements, one task and one GPU each,
# `%2` letting them run concurrently. Set `N_CHUNKS` and the array range together to use more.
#
# ------------------------------------------------- sizing (measured 2026-09-07, login V100S)
# Four pilot runs, 30 shots total, `ideate encode --device cuda --workers 8`:
#     10 shots  23.36 s/shot mean   378 % of one cpu
#      4 shots  21.11 s/shot mean   378 %
#      6 shots  19.85 s/shot mean   403 %
#      5 shots  21.39 s/shot mean   443 %
#   -> 21.6 s/shot, 4.0 cores of demand, GPU memory 1.87 GB, single-process max RSS 2.59 GB,
#      whole-process-tree PSS peak 7.41 GB (median 1.12 GB; the peak is the ece/video window).
#
# THE JOB IS I/O-BOUND, NOT GPU-BOUND, and no amount of sizing changes that. Profiled per shot:
# 2.1 s of GPU forward against 16.5 s of HDF5 read + per-frame STFT in the dataloader workers
# and 2.4 s of presence scan, so the ceiling on GPU utilisation is ~10 % whatever else is done.
# Sampled with `nvidia-smi -l 1`: mean 3.7-4.1 %, p95 25-33 %, max 49-56 %. A prefetch pool
# overlapping the next shot's reads with this shot's GPU work was therefore NOT added -- it can
# recover at most the 2.1 s of GPU time per 21.6 s shot (<= 10 % of wall clock) and cannot reach
# the 70 % GPU-utilisation figure at all; the measurement that settles it is the worker sweep,
# where raising `--workers` from 8 to 16 and 24 made the shot SLOWER (21.1 -> 28.5 -> 27.2
# s/shot) at unchanged CPU%, i.e. the read path is already at its ceiling. The plan's iteration-0
# criterion is jobstats (CPU and memory efficiency), and that is what the sizing below targets.
#
#   cpus-per-task  4.0 cores of demand / 5 = 80 % projected CPU efficiency. 8 would project 50 %
#                  and fail the jobstats gate; 6 projects 67 %, still under it. `--workers 8` is
#                  kept because it was the fastest point of the sweep and the workers are
#                  I/O-blocked, so they oversubscribe 5 cores without adding CPU time.
#   mem-per-cpu    7.41 GB tree PSS * 1.3 = 9.6 GB per task / 5 cpus = 1.93 -> 2G
#                  (10 GB per task, ~74 % projected memory efficiency).
#   time           250 shots * 21.6 s = 90 min per task; 2x for the GPFS contention of two
#                  concurrent tasks -> 3 h. The allocation's A100s are faster than the login
#                  V100S, but on an I/O-bound job that buys nothing, so it is not counted.
#
#   OMP pin       The pilot above was measured WITHOUT the `OMP_NUM_THREADS=1` this script
#                  exports, so it was re-measured on one shot (185786, login V100S, GPU 1 idle,
#                  2026-09-07) in both environments:
#
#                      unpinned            19.38 s/shot   332 % CPU   89.4 CPU-s   26.9 s wall
#                      OMP_NUM_THREADS=1   17.25 s/shot   345 % CPU   77.6 CPU-s   22.5 s wall
#
#                  The pin lowers CPU-time AND wall clock together, so the ratio the jobstats
#                  gate reads is flat to slightly up (332 -> 345 %), not down: the feared
#                  "wall grows while cpu-time shrinks, efficiency falls" does not happen here,
#                  because the work is in the 8 dataloader worker processes and in GPFS reads,
#                  not in a threaded BLAS call. Both figures are single-shot runs whose wall
#                  clock still carries the ~5.6 s codec load, so the CPU % is if anything
#                  understated against a 250-shot task; the 4.0 cores / 21.6 s per shot the
#                  sizing above rests on come from the 30-shot pilot and are kept unchanged
#                  (they are the conservative pair: more cores, more seconds). The script runs
#                  the pinned environment, which is the faster of the two measured.
#
# Re-runnable: `--skip-existing` leaves a cache already written, so a task that hits the wall
# clock can simply be resubmitted.
#
# THE CPU-ONLY ALTERNATIVE. `scripts/ideate/encode_cpu.sbatch` runs the same command with
# `--device cpu` on the `pppl` partition. Measured 40.4 s/shot at OMP_NUM_THREADS=4 against
# 17-22 s/shot here, i.e. ~2x, for no GPU at all -- and plan section 13.3 judges a CPU-only job
# on CPU and CPU-memory alone, which is the gate this workload can actually meet. Read the
# capital-letter note there before choosing: the two devices do NOT always produce identical
# tokens, so a corpus should be encoded entirely on one or the other.

set -euo pipefail
REPO=/scratch/gpfs/nc1514/FusionAIHub
ROOT="${IDEATE_DATA_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/ideate}"
ENV="$REPO/.pixi/envs/ideate"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export IDEATE_DATA_ROOT="$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$ROOT/runs/slurm" "$ROOT/frame_codes"
cd "$REPO"
echo "job ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$ENV/bin/python" -m ideate encode \
    --list "${SHOT_LIST:-recommender_v1}" \
    --out "$ROOT/frame_codes" \
    --device cuda \
    --workers 8 \
    --skip-existing \
    --chunk "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-chunks "${N_CHUNKS:-2}"

echo "finished at $(date -Is)"
