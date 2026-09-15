#!/bin/bash
#SBATCH --job-name=ae-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out

# Task 7b: the three AE SELDNet runs, as three array tasks.
#
#   0  --target threeway --loss sce   (primary)
#   1  --target threeway --loss bce   (does the symmetric term matter?)
#   2  --target mask     --loss sce   (does the three-way target matter?)
#   3  --target threeway --loss sce --pool-sizes 6,2  (frequency-head fix)
#
# Task 3 was added after watching runs 0-2: with the reference pool sizes
# (6,2,29) the band collapses to ONE frequency row before the GRU, and the
# frequency head converged to the constant-median predictor (16.8 kHz MAE on
# the validation shots) and stopped. Pooling only 6*2 leaves 29 rows - 64*29 =
# 1856 features per frame - so the recurrent stack can actually see where in
# the band the energy is. 20 epochs, to fit alongside task 2.
#
# Environment: the phase-3 venv (torch 2.14.0+cu126). No Lightning, by design.
# The dataset is 180 npz on group storage whose `spec` members are stored
# uncompressed, so the loader reads one shot (20 MB) and draws 8 windows from
# it; --cpus-per-task 8 is the DataLoader worker count. Tasks 6 and 7a were
# both host-I/O bound and the epoch log prints the dataloader wait so the next
# request can be sized from it.

set -euo pipefail

ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
REPO=/scratch/gpfs/nc1514/FusionAIHub
export PYTHONPATH=$REPO/src
export LABELER_GIT_SHA=$(git -C "$REPO" rev-parse HEAD)
export OMP_NUM_THREADS=1   # 8 dataloader workers, each single-threaded numpy

EXTRA=()
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TARGET=threeway; LOSS=sce ;;
  1) TARGET=threeway; LOSS=bce ;;
  2) TARGET=mask;     LOSS=sce ;;
  3) TARGET=threeway; LOSS=sce; EXTRA=(--pool-sizes 6,2 --tag f29 --max-epochs 20) ;;
  *) echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

nvidia-smi || true
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} (${TARGET}/${LOSS}) on $(hostname) at $(date -Is)"

srun "$ROOT/envs/phase3/bin/python" -u "$REPO/scripts/labeler/ae_train.py" \
  --target "$TARGET" --loss "$LOSS" \
  --dataset-dir "$ROOT/ae/dataset" \
  --out-dir "$ROOT/models/d3d_ae_activity_seldnet" \
  --window-frames 710 --windows-per-shot 8 --batch-windows 16 \
  --max-epochs 30 --lr 1e-4 --weight-decay 1e-4 --min-lr 1e-6 --patience 5 \
  --lambda-f 1.0 --seed 0 "${EXTRA[@]+"${EXTRA[@]}"}"

echo "finished at $(date -Is)"
