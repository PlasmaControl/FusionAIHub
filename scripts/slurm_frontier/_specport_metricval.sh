#!/bin/bash
#SBATCH -A fus187
#SBATCH -J specport_metricval
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# 5-REFERENCE VALIDATION of the mode-track metrics. spec_nrmse ranks the BLUR best, so any
# replacement ranking metric must first reproduce a known-correct ordering on real held-out
# windows. Run under SLURM because the login node SIGKILLs this workload.
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
srun -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 --gpu-bind=closest \
  python analysis/_specport_metric_validation.py "${MV_N:-240}"
echo "=== SPECPORT METRICVAL DONE (exit $?) ==="
