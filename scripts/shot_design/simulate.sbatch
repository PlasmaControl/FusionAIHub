#!/bin/bash
#SBATCH --job-name=sd-simulate
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
# One design's ensemble of IGNITE v4 rollouts (real, proposed and null arms) on one A100:
# what the UI's Simulate button submits on Stellar (paths.yaml: simulate_submit_cmd). The
# Stellar twin of scripts/slurm_frontier/shot_design_simulate.sh.
#   sbatch scripts/shot_design/simulate.sbatch <ident> [simulate args]
# Sized from pilot 2939099 at the defaults (8 members x 3 arms, 80 frames x 10 passes, fp32,
# two members per pass): 28:50 elapsed, MaxRSS 3.2 GiB; jobstats CPU 98.9 %, CPU-mem 72.5 %,
# GPU 97.9 %, GPU-mem 98.4 %. Time grows about linearly with --members and --n-predict.
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
