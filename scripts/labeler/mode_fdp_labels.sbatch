#!/bin/bash
#SBATCH --job-name=mode-fdp-labels
#SBATCH --account=nstx-u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%A_%a.out

# omnimode's labeller (`omnimode batch --variant ermap-magrot`, TokEye front end) on one list,
# one array task a slice shots[I::N]. Submitted by mode_all_shots.py with --array=0-<N-1> for the
# whole list: never resubmit a subset of the array (the slices change with the task count).
# Finished shots are cache hits; LABEL_ARGS=--force relabels (mode_all_shots.py redo).
# batch refuses a list holding a sealed shot (seal.guard): the lists here never hold one.
#   LIST=... [LABEL_ARGS=--force] sbatch --array=0-9 scripts/labeler/mode_fdp_labels.sbatch
set -euo pipefail
source "${REPO:-/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub}/scripts/labeler/mode_fdp_env.sh"
cd "$OMNIMODE_REPO"
read -ra EXTRA <<< "${LABEL_ARGS:-}"
.venv/bin/python -I -m omnimode batch --shots-file "$LIST" --variant ermap-magrot \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}" --task-count "${SLURM_ARRAY_TASK_COUNT:-1}" "${EXTRA[@]}"
echo "finished at $(date -Is)"
