#!/bin/bash
#SBATCH -A fus187
#SBATCH -J scan_slowts_qc
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"; source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/data_preparation/scan_slowts_qc.py --workers "${WORKERS:-56}"
echo "=== SLOWTS QC (exit $?) ==="
