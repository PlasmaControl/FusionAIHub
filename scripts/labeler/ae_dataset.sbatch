#!/bin/bash
#SBATCH --job-name=ae-dataset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out

# Task 7a: AE frame labels and dataset from big_tf_unet's cleaned coherent mask.
#
# Environment: the tokeye repo's own venv, used read-only (torch 2.9.1+cu128,
# tokeye importable). pyarrow is not installed there and the tokeye tree may not
# be written to, so pyarrow lives in a --target directory on group storage that
# is put on PYTHONPATH, together with this repo's src/ so that
# labeler.ae.labels - the numpy-only rule module the pixi tests cover -
# is the single source of the cleaning rules.

set -euo pipefail

REPO=/scratch/gpfs/nc1514/FusionAIHub
PY=/scratch/gpfs/nc1514/tokeye/.venv/bin/python
ENV="${LABELER_PIXI_ENV:-/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler}"  # another cluster: export LABELER_PIXI_ENV
# The cluster's production root (labeler.env.CLUSTERS), whatever LABELER_ROOT the caller
# exports: the pyarrow --target directory ($ROOT/ae/pylibs) lives there.
ROOT="$(env -u LABELER_ROOT "$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler")"
export PYTHONPATH="$ROOT/ae/pylibs:$REPO/src"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1   # the checkpoint is loaded from a local path

nvidia-smi || true
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun "$PY" "$REPO/scripts/labeler/ae_dataset.py" \
    --stage all \
    --device cuda \
    --batch 4 \
    --root "$ROOT/ae" \
    --out-dir "$REPO/outputs/labeler/ae/dataset"

echo "finished at $(date -Is)"
