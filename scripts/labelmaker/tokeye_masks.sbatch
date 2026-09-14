#!/bin/bash
# README: L12 measured A100 requests; production is a proposal, NEVER submitted.
# Completed cold pilot 2931065_0: 20 attempts / 15 ok / 5 coverage errors,
# 7478 tiles / 152.72 driver seconds = 48.97 tiles/s; SLURM wall 161 seconds.
# Six prep workers + parent + tail = 8 CPUs. CPU 27.1%, sampled CPU memory
# 4.7GB/24GB = 19.6%, GPU 6.9%, GPU memory 37.9GB/40GB = 94.7%.
# sacct MaxRSS 6750340K = 6.438 GiB; x1.30 = 8.37 GiB -> request 9G.
# Parent VmHWM 2.589 GiB, each prep 0.951 GiB, tail 2.431 GiB (shared pages
# mean these must NOT be summed as a cgroup measurement).
# Prefetch 6 feeds all six workers. Batch 96 was measured; actual blocks are
# 10-55 tiles, so increasing this cap cannot improve this pilot's GPU use.
# For proposed 60-shot chunks: 161/20*60*1.30 = 627.9 s -> time 00:11:00.
# Single-element default is safe for a pilot; production shape is 0-7%2 with
# N_CHUNKS=8 and the separate remaining-480-shot list. One GPU per task AND
# per node describes the same one-GPU allocation (confirmed by sacct).
# At 9G, sampled CPU-memory projects only 52.2%; CPU and GPU also MISS 70%.
# Completed pilot 2931084_0 (1 prep/2 prefetch/3 CPUs/9G) fell to 24.99 tiles/s,
# with prep wait 184.11s vs 32.53s. CPU 39.2%, CPU-mem 35.6%, GPU 3.8%:
# fewer workers worsened throughput and still missed all three gates.
# Workflow: bash scripts/labelmaker/tokeye_text_subset.sh; sbatch this file;
# sbatch --dependency=afterok:$JOBID scripts/labelmaker/tokeye_masks_afterok.sbatch "$JOBID" --pilot
# See docs/LABELMAKER.md and .superpowers/sdd/task-L12-report.md for exact commands.
#SBATCH --job-name=tokeye-masks
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=9G
#SBATCH --time=00:11:00
#SBATCH --array=0-0%1
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out

set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub-L}"
ROOT="${ROOT:-${LABELMAKER_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
PHASE3_PYTHON="${PHASE3_PYTHON:-$RUNTIME_ROOT/envs/phase3/bin/python}"
UNET="${UNET:-$RUNTIME_ROOT/models/tokeye/big_tf_unet_251210.pt}"
ROOT="$(realpath -m -- "$ROOT")"
printf 'Resolved root: %s\n' "$ROOT"
CORPUS="${LABELMAKER_CORPUS:-/scratch/gpfs/EKOLEMEN/foundation_model}"
SHOT_FILE="${SHOT_FILE:?set SHOT_FILE to the pilot or remaining-shot list}"
N_CHUNKS="${N_CHUNKS:-1}"
PREP_WORKERS="${PREP_WORKERS:-6}"
PREFETCH="${PREFETCH:-6}"
TILE_BATCH="${TILE_BATCH:-96}"
if (( PREFETCH < PREP_WORKERS )); then
    echo "PREFETCH must be >= PREP_WORKERS" >&2
    exit 2
fi
if (( SLURM_CPUS_PER_TASK != PREP_WORKERS + 2 )); then
    echo "SLURM_CPUS_PER_TASK must equal PREP_WORKERS + 2" >&2
    exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_CHUNKS )); then
    echo "array task must be within N_CHUNKS" >&2
    exit 2
fi
export LABELMAKER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
cd "$REPO"
echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,utilization.gpu --format=csv
srun --cpu-bind=cores "$PHASE3_PYTHON" -u \
    "$REPO/scripts/labelmaker/tokeye_masks.py" \
    --shot-file "$SHOT_FILE" --chunk "$SLURM_ARRAY_TASK_ID" \
    --n-chunks "$N_CHUNKS" --rank 0 --world 1 \
    --root "$ROOT" --unet "$UNET" --corpus "$CORPUS" --device cuda --plan round1 \
    --tile-batch "$TILE_BATCH" --prep-workers "$PREP_WORKERS" \
    --prefetch "$PREFETCH" --tail-workers 1 --text-subset readonly --no-index \
    --amp --timeout 240 --run-id "tokeye-${SLURM_ARRAY_JOB_ID}"
echo "finished at $(date -Is)"
