#!/bin/bash
# README: L14 pilot 2 capacity basis; production is report-only, NOT validated.
# Cold A100 pilot 2932066_0 at source 56d4521 completed 20/20 shots,
# 244 blocks / 7478 tiles / 26913 events in 76.87 driver seconds = 97.29 tiles/s.
# Its allocation was 12 CPUs (7 prep + 4 tail + 1 parent), prefetch 8,
# 32G, 00:12:00, one A100, batch 96, AMP, readonly text, --no-index.
# Jobstats: CPU 31.3%, CPU memory 9.8GB/32GB = 30.6%, GPU 7.4%,
# GPU memory 38.6GB/40GB = 96.6%. FAIL (pilot-exempt); no production submitted.
# sacct: 82s wall, TotalCPU 06:20.657, MaxRSS 13559492K = 12.931339 GiB.
# Memory capacity: ceil(12.931339 * 1.3) = 17G at the SAME pool sizes.
# Time for 60 shots: startup (82 - 76.87) + one timeout 240 +
# 60 * (76.87 / 20) * 1.3 = 544.923s -> 00:10:00.
# Keep 7 prep / 4 tail / prefetch 8: changing pools invalidates this memory basis.
# Prep wait 44.823s, tail wait 5.803s. GPU feeding still failed; the following
# defaults describe capacity only and do not establish production readiness.
# Parent VmHWM 6.050 GiB, max worker 2.573 GiB; shared pages mean per-PID
# peaks must NOT be added as a cgroup measurement. Live workers: 7 prep, 4 tail.
# Single-element default is for a pilot; proposed production shape remains
# --array=0-7%2, N_CHUNKS=8 (8 x 60 shots), pending resolution of failed gates.
# Workflow: tokeye_text_subset.sh, this array, tokeye_masks_afterok.sbatch.
# See docs/LABELER.md and .claude/superpowers-runtime/sdd/task-L14perf-report.md (Fix loop).
#SBATCH --job-name=tokeye-masks
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=17G
#SBATCH --time=00:10:00
#SBATCH --array=0-0%1
# Slurm cannot expand ROOT in a directive. The shell below opens the full log
# under the resolved ROOT; an explicit --output captures bootstrap diagnostics.
#SBATCH --output=/dev/null

set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ROOT="${ROOT:-$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
PHASE3_PYTHON="${PHASE3_PYTHON:-$RUNTIME_ROOT/envs/phase3/bin/python}"
UNET="${UNET:-$RUNTIME_ROOT/models/tokeye/big_tf_unet_251210.pt}"
ROOT="$(realpath -m -- "$ROOT")"
printf 'Resolved root: %s\n' "$ROOT"
CORPUS="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_CORPUS "/scratch/gpfs/EKOLEMEN/foundation_model")"
SHOT_FILE="${SHOT_FILE:?set SHOT_FILE to the pilot or remaining-shot list}"
N_CHUNKS="${N_CHUNKS:-1}"
PREP_WORKERS="${PREP_WORKERS:-7}"
PREFETCH="${PREFETCH:-8}"
TILE_BATCH="${TILE_BATCH:-96}"
TAIL_WORKERS="${TAIL_WORKERS:-4}"
if (( PREFETCH < PREP_WORKERS )); then
    echo "PREFETCH must be >= PREP_WORKERS" >&2
    exit 2
fi
if (( SLURM_CPUS_PER_TASK != PREP_WORKERS + TAIL_WORKERS + 1 )); then
    echo "SLURM_CPUS_PER_TASK must equal PREP_WORKERS + TAIL_WORKERS + 1" >&2
    exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_CHUNKS )); then
    echo "array task must be within N_CHUNKS" >&2
    exit 2
fi
mkdir -p "$ROOT/runs/slurm"
exec > "$ROOT/runs/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1
printf 'Resolved root: %s\n' "$ROOT"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
cd "$REPO"
echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,utilization.gpu --format=csv
srun --cpu-bind=cores "$PHASE3_PYTHON" -u \
    "$REPO/scripts/labeler/tokeye_masks.py" \
    --shot-file "$SHOT_FILE" --chunk "$SLURM_ARRAY_TASK_ID" \
    --n-chunks "$N_CHUNKS" --rank 0 --world 1 \
    --root "$ROOT" --unet "$UNET" --corpus "$CORPUS" --device cuda --plan round1 \
    --tile-batch "$TILE_BATCH" --prep-workers "$PREP_WORKERS" \
    --prefetch "$PREFETCH" --tail-workers "$TAIL_WORKERS" --text-subset readonly --no-index \
    --amp --timeout 240 --run-id "tokeye-${SLURM_ARRAY_JOB_ID}"
echo "finished at $(date -Is)"
