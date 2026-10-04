#!/bin/bash
#SBATCH --job-name=saw-edge-context
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix5_edge_context_derivation/run_%j.log
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix5_edge_context_derivation/run_%j.log

# Second worker of the TRAIN-only edge-context derivation run: it walks the TRAIN
# shots from the far end and meets the head-node run; cached shots are skipped.
set -euo pipefail
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/saw
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/home/nc1514/.pixi/bin:$PATH
cd /scratch/gpfs/nc1514/FusionAIHub-r4-saw
export PYTHONPATH=$PWD/src
trap 'bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/tmpsweep.sh' EXIT
timeout 10800 pixi run --frozen --no-install \
    --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
    python scripts/labeler/sawtooth_edge_context.py run --workers 8 --reverse
