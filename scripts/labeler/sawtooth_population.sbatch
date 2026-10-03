#!/bin/bash
#SBATCH --job-name=saw-physics
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix/population_%A_%a.log
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix/population_%A_%a.log

# Four disjoint resumable population shards; run labels --population afterward
# to aggregate their per-shot records into one population JSON/CSV export.
set -euo pipefail
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/saw
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/home/nc1514/.pixi/bin:$PATH
cd /scratch/gpfs/nc1514/FusionAIHub-r4-saw
export PYTHONPATH=$PWD/src
pixi run --frozen --no-install \
    --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
    python scripts/labeler/sawtooth_physics.py labels --population --skip-cohort --workers 8 \
    --shard "$SLURM_ARRAY_TASK_ID" --shards 4
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/tmpsweep.sh
