#!/bin/bash
# Submit only as: sbatch --dependency=afterok:$JOBID this-file "$JOBID" [--pilot]
# Rebuild once after the WHOLE array; a GPU task never owns the shared index.
# First companion 2931072: sacct MaxRSS 400244K (0.382 GiB), 3.87 CPU-s;
# 512M gives 31% headroom. Six minutes bounds the scan plus 300 s accounting
# latency. Numbered srun steps retain payload RSS for the sacct fallback.
# Corrected companion 2931085: 7s wall, 272804K numbered-step MaxRSS, 52% of
# 512M. Too short for sampled CPU/memory; CPU undetermined, memory FAIL via
# sacct (pilot-exempt). Do not pad its runtime to manufacture samples.
# Gate this CPU companion after it leaves squeue;
# GPU metrics are N/A for it. No production job is submitted by this script.
#SBATCH --job-name=tokeye-rebuild-gate
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:06:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

set -euo pipefail
JOBID="${1:?pass the completed GPU array job id}"
gate_mode=()
if [[ "${2:-}" == --pilot ]]; then
    gate_mode+=(--pilot)
elif [[ $# != 1 ]]; then
    echo "usage: $0 JOBID [--pilot]" >&2
    exit 2
fi
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub-L}"
ROOT="${LABELMAKER_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
if [[ ! -x "$ROOT/runs/slurm/jobstats-client/jobstats" ]]; then
    echo "run tokeye_text_subset.sh on the login node first to stage jobstats" >&2
    exit 2
fi
export PATH="$ROOT/runs/slurm/jobstats-client:$PATH"
export LABELMAKER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/tmp/l12-pixi}"
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID:-rebuild}"
cd "$REPO"
srun --cpu-bind=cores pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
    -e labelmaker python "$REPO/scripts/labelmaker/tokeye_masks.py" \
    --root "$ROOT" --rebuild-index --index-out "$ROOT/events/events_index.parquet"
srun --cpu-bind=cores pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
    -e labelmaker python "$REPO/scripts/labelmaker/jobstats_check.py" \
    --job-id "$JOBID" "${gate_mode[@]}" --wait-for-data 300 \
    --preserve-dir "$ROOT/runs/slurm" --out "$ROOT/runs/slurm/jobstats.json"
