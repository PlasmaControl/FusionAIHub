#!/bin/bash
#SBATCH --job-name=ae-supervision-swap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-8%3
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/aeswap/slurm-%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/aeswap/slurm-%A_%a.out

set -euo pipefail
REPO=/scratch/gpfs/nc1514/FusionAIHub-r4-aeswap
# Short on purpose: DataLoader workers add /pymp-*/listener-* to TMPDIR and an
# AF_UNIX socket path must stay under 108 bytes.
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/ae-sw
mkdir -p "$TMPDIR"
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1
export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$REPO"
trap 'bash "$LABELER_ROOT/scratch/bin/tmpsweep.sh"' EXIT

# Three tasks per seed, at most three concurrent jobs. Freeze seeds 0, 1, 2
# in the manifest before submission. AESWAP_RUNS="legacy:2 dense:1" instead maps
# array task i to the i-th listed arm:seed (used for convergence-rule reruns).
TASK=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((TASK / 3))
ARMS=(legacy dense threeway)
ARM=${ARMS[$((TASK % 3))]}
if [[ -n ${AESWAP_RUNS:-} ]]; then
    read -ra RUNS <<< "$AESWAP_RUNS"
    ARM=${RUNS[$TASK]%%:*}
    SEED=${RUNS[$TASK]##*:}
fi
echo "job ${SLURM_ARRAY_JOB_ID:-local}_${TASK} ${ARM} seed ${SEED} on $(hostname) at $(date -Is)"
if [[ -n ${AESWAP_PIXI_ENV:-} ]]; then
    srun pixi run --frozen --no-install \
        --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
        -e "$AESWAP_PIXI_ENV" python -u scripts/labeler/ae_supervision_swap.py \
        train --supervision "$ARM" --seed "$SEED"
else
    PYTHON=${AESWAP_PYTHON:-$LABELER_ROOT/envs/phase3/bin/python}
    srun "$PYTHON" -u scripts/labeler/ae_supervision_swap.py \
        train --supervision "$ARM" --seed "$SEED"
fi
