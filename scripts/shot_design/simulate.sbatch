#!/bin/bash
#SBATCH --job-name=sd-simulate
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
# One design's paired real/proposed IGNITE v4 rollout on one A100: what the UI's Simulate
# button submits on Stellar (paths.yaml: simulate_submit_cmd). The Stellar twin of
# scripts/slurm_frontier/shot_design_simulate.sh.
#   sbatch scripts/shot_design/simulate.sbatch <ident> [simulate args]
# Sizing is a pilot guess (v4, 80 frames x 10 passes, estimated 70-130 min on an A100):
# resize --mem and --time from the first run's MaxRSS and elapsed time.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV="$REPO/.pixi/envs/shot-design"
export SHOT_DESIGN_DATA_ROOT="${SHOT_DESIGN_DATA_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/ideate}"
export LABELER_ROOT="${LABELER_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
export SHOT_DESIGN_CORPUS="${SHOT_DESIGN_CORPUS:-/scratch/gpfs/EKOLEMEN/foundation_model}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
IDENT="${1:?design ident}"
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is); design $IDENT; root $SHOT_DESIGN_DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "$ENV/bin/python" -m shot_design simulate "$IDENT" --device cuda "${@:2}"
echo "finished at $(date -Is)"
