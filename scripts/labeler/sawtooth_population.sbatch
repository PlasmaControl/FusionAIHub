#!/bin/bash
#SBATCH --job-name=saw-physics
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/round4/saw/fix5/population_%A_%a.log
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/round4/saw/fix5/population_%A_%a.log

# Four disjoint resumable population shards, then run
# labels --population --records-only for the final JSON/CSV export.
# Geometry fetching stays on the login node. Its completion flag is written
# after the full EFIT bundle and the measured-height attempt have closed.
set -euo pipefail
REPO="${LABELER_REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV="${LABELER_PIXI_ENV:-/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler}"  # another cluster: export LABELER_PIXI_ENV
LABELER_ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler")"
LABELER_LABEL_TABLES="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_LABEL_TABLES "/scratch/gpfs/EKOLEMEN/D3D/events")"
export LABELER_ROOT LABELER_LABEL_TABLES
export TMPDIR="$LABELER_ROOT/scratch/claude-89242e53/r4/tmp/saw"
mkdir -p "$TMPDIR"
export LABELER_NO_FETCH=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/home/nc1514/.pixi/bin:$PATH
cd "$REPO"
export PYTHONPATH=$PWD/src
trap 'bash "$LABELER_ROOT/scratch/bin/tmpsweep.sh"' EXIT
timeout 21600 pixi run --frozen --no-install \
    --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labeler \
    python scripts/labeler/sawtooth_physics.py labels --population --skip-cohort --workers 8 \
    --shard "$SLURM_ARRAY_TASK_ID" --shards 4
