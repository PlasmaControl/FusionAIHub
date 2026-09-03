#!/bin/bash
#SBATCH -A fus187
#SBATCH -J nan_localize
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 0:40:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
export MASTER_PORT=29610
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src:$FMH/scripts/training:$FMH/analysis/mode_audit:${PYTHONPATH:-}"
export EXTRA_DATA_DIR=/lustre/orion/fus187/proj-shared/additional_data
export OUT_DIR="$FMH/eval_runs/nan_localize"
export CACHE_DIR="$FMH/eval_runs/nan_localize_cache"
# span more of the corpus absmax range; keep collection within the 40-min budget
export N_EXTRA_SHOTS="${N_EXTRA_SHOTS:-10}"
export MAX_BATCHES="${MAX_BATCHES:-20}"
export N_HOT="${N_HOT:-14}"
export ISOLATE_ECE="${ISOLATE_ECE:-1}"
srun python analysis/mode_audit/nan_localize.py \
  "${CKPT:-/lustre/orion/fus187/proj-shared/models/e2e_g3fix_anneal/e2e_stage1_beta6.0_step3000.pt}"
