#!/bin/bash
#SBATCH -A fus187
#SBATCH -J specport_ceiling
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 01:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# CAPACITY CEILING at 192 tokens: (1) out-of-sample PCA plateau vs k on BOTH STFT grids,
# (2) the same thing as a picture. Both run in the per-freq log-z space the mhr arms train in.
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
FIG=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/codec_recon_figs/specport_ceiling.png
echo "=== STEP 1: PCA plateau vs k ==="
srun -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 --gpu-bind=closest \
  python analysis/_specport_plateau.py "${PLATEAU_N:-720}"
echo "=== STEP 2: ceiling figure ==="
srun -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 --gpu-bind=closest \
  python analysis/_specport_ceiling_fig.py "${FIG}" "${CEILING_N:-240}"
echo "=== SPECPORT CEILING DONE (exit $?) ==="
