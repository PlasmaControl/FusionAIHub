#!/bin/bash
#SBATCH --job-name=bes-abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-19%4
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/conf/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/conf/logs/slurm_%A_%a.out

# The BES confinement ablation's paper-training rows (`full_*`: no padding in the
# convolution, 60,000 steps with no early stopping, best validation-F1 checkpoint, the
# block chosen by geometry), one array task per (row, split): five splits of each of
# four rows, at most four tasks at a time.
#
#   task = 5 * row + split;  rows: full_cum_abcdrgef, full_leak_abcdrge,
#                                  full_cum_abcdrge, full_only_c
#
# The same array runs any four rows (BES_ROWS), for example the single factors added to
# only_ge on 444 shots: BES_ROWS="ge_a ge_b ge_c ge_r".
#
# Environment: the phase-3 venv (torch 2.14.0+cu126). The split's features (up to
# 11 GB at 1 MHz) stay on the GPU (--gpu-gb 16), so a step takes about 10 ms.
# Predictions go to $LABELER_ROOT/round4/conf/ablation/<row>/<split>_predictions.csv.

set -euo pipefail

ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
REPO=/scratch/gpfs/nc1514/FusionAIHub-r4-conf
export PYTHONPATH=$REPO/src
export LABELER_ROOT=$ROOT

# BES_ROWS overrides the rows (sbatch --export=ALL,BES_ROWS="ge_a ge_b ge_c ge_r")
read -r -a ROWS <<< "${BES_ROWS:-full_cum_abcdrgef full_leak_abcdrge full_cum_abcdrge full_only_c}"
TASK=${SLURM_ARRAY_TASK_ID:?array task}
ROW=${ROWS[$((TASK / 5))]}
FOLD=$((TASK % 5))

nvidia-smi || true
echo "job ${SLURM_JOB_ID} task ${TASK} (${ROW} split ${FOLD}) on $(hostname) at $(date -Is)"

srun "$ROOT/envs/phase3/bin/python" -u "$REPO/scripts/labeler/confinement_bes_ablation.py" \
  run "$ROW" --folds "$FOLD" --device cuda:0 --gpu-gb 16

echo "finished at $(date -Is)"
