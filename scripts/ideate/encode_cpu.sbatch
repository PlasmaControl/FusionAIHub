#!/bin/bash
#SBATCH --job-name=ideate-encode-cpu
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=01:10:00
#SBATCH --array=0-9%10
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%A_%a.out

# IGNITE frame-code caches for the 500 recommender_v1 shots, encoded on CPU. This is the
# alternative to `scripts/ideate/encode.sbatch` (same command, `--device cuda`, `gpu` partition);
# the controller picks one. NEITHER is submitted by the implementer.
#
# REPO is the MAIN checkout, not the I8 worktree: written on branch recommender-I8, run after
# the merge, from /scratch/gpfs/nc1514/FusionAIHub.
#
# --------------------------------------------------------------- why a CPU job exists at all
# The encode is not GPU work. Profiled per shot on a V100S: 2.1 s of GPU forward against 16.5 s
# of HDF5 read and per-frame STFT in the dataloader workers plus 2.4 s of presence scan. Total
# GPU work for the whole 500-shot corpus is 500 x 2.1 s ~ 18 GPU-MINUTES. A GPU job therefore
# cannot exceed ~10 % GPU utilisation whatever it does, which makes iteration 0's ">= 70 %"
# structurally unattainable on this workload rather than a sizing failure -- and plan section
# 13.3 says CPU-only jobs are judged on CPU and CPU-memory alone. Dropping the GPU request
# removes the criterion the work cannot meet instead of arguing with it, and drops the
# `gpu-stellar` queue as well.
#
# ------------------------------------------------- sizing (measured 2026-09-07, login node)
# `ideate encode --list recommender_v1 --device cpu --limit 2 --workers 8`, /usr/bin/time -v
# around it and a 0.5 s process-tree sampler over /proc/*/smaps_rollup, 40-core login node at
# load ~4, two shots (185786, 185955):
#
#     OMP/MKL threads   s/shot mean   CPU %    max RSS   tree RSS peak   tree PSS peak
#     4                    40.41      380 %    2.47 GB     15.39 GB         7.17 GB
#     8                    31.52      532 %    2.47 GB     16.38 GB         8.04 GB
#
#   for comparison, the same code on the GPU path: 17.25 s/shot (OMP_NUM_THREADS=1, one shot)
#   and 21.6 s/shot (30-shot pilot). So CPU is ~2x the GPU path at 4 threads and ~1.5x at 8 --
#   inside the 3x that made this worth writing.
#
#   cpus-per-task=5  OMP_NUM_THREADS=4 measured 380 % of one core. 380 / 5 = 76 % projected CPU
#                    efficiency, over the 75 % target. 4 cpus would project 95 % but leaves the
#                    8 I/O-blocked dataloader workers nothing to run on; 6 projects 63 %. The
#                    8-thread point is the alternative: 532 % / 7 cpus = 76 % at 31.5 s/shot,
#                    i.e. the same efficiency for 40 % more cores and 22 % less wall clock. The
#                    4-thread point is chosen because it is cheaper per shot in core-seconds
#                    (153 vs 168) and asks the scheduler for less.
#                    Both figures are two-shot runs whose wall clock still carries the ~5.6 s
#                    codec load, so the steady-state CPU % of a 50-shot task is if anything
#                    higher than measured.
#
#   mem=10G          Sized on the process TREE's peak PSS, 7.17 GB x 1.3 = 9.32 -> 10G, which
#                    projects 72 % memory efficiency. NOT on the summed tree RSS (15.39 GB;
#                    x 1.3 = 20G): eight forked dataloader workers share the parent's pages
#                    copy-on-write, summing RSS counts those pages up to eight times, and the
#                    cgroup SLURM charges counts them once. Sizing on 20G would ask for twice
#                    the memory and project 36 % efficiency -- failing the very gate this job
#                    is judged by. Single-process max RSS is 2.47 GB, so 10G also leaves room
#                    for one worker to grow.
#
#   time=01:10:00    500 shots / 10 array tasks = 50 shots per task; 50 x 40.41 s = 2021 s =
#                    33.7 min; x2 for GPFS contention between ten concurrent tasks reading
#                    2-5 GB corpus files -> 67.4 min -> 1 h 10 min.
#
#   array=0-9%10     Ten contiguous chunks of the sorted 500-shot list, `--chunk
#                    $SLURM_ARRAY_TASK_ID --n-chunks 10`, all ten resident at once (the QOS
#                    allows 20 jobs per user). Contiguous, not strided, so the ten tasks walk
#                    ten different regions of the corpus directory. Total ask: 50 cores and
#                    100 GB for ~35 min, against 2 GPUs + 10 cores + 20 GB for ~90 min.
#
# Re-runnable: `--skip-existing` leaves a cache already written, so a task that hits the wall
# clock is simply resubmitted. Ten shots (185786 ... 186036) are already encoded from the pilot.
#
# ------------------------------------------------------------------------- READ THIS FIRST
# CPU AND GPU DO NOT ALWAYS PRODUCE THE SAME TOKENS, so a corpus must be encoded entirely on
# one device or entirely on the other. Measured on the two shots above: 185786 is bit-identical
# across cuda, cpu-with-4-threads and cpu-with-8-threads on all fourteen modalities, but 185955
# is not -- `bes` and `mhr` disagree by 1 to 4 isolated single tokens, one per affected frame,
# and cpu-4-threads versus cpu-8-threads disagree by 3 `bes` tokens on the SAME machine and the
# SAME device with only the BLAS thread count changed. The 88 actuator channels are bit-identical
# in every pairing (they are NumPy arithmetic, not a codec). This is the same one-token-per-frame
# quantiser-bin-boundary signature the G-ENC gate sees against the shipped caches, and it means
# the ten pilot shots already in $ROOT/frame_codes were GPU-encoded: either re-encode them on CPU
# (drop `--skip-existing`) or run the GPU script.
#
# This job is judged on CPU and CPU-memory efficiency (plan section 13.3), projected 76 % and
# 72 %. It requests no GPU, so the GPU rows of jobstats do not apply to it.

set -euo pipefail
REPO=/scratch/gpfs/nc1514/FusionAIHub
ROOT="${IDEATE_DATA_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/ideate}"
ENV="$REPO/.pixi/envs/ideate"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export IDEATE_DATA_ROOT="$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$ROOT/runs/slurm" "$ROOT/frame_codes"
cd "$REPO"
echo "job ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date -Is)"

srun "$ENV/bin/python" -m ideate encode \
    --list "${SHOT_LIST:-recommender_v1}" \
    --out "$ROOT/frame_codes" \
    --device cpu \
    --workers 8 \
    --skip-existing \
    --chunk "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-chunks "${N_CHUNKS:-10}"

echo "finished at $(date -Is)"
