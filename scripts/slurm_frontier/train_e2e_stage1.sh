#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage1
#SBATCH -o logs/%j_e2e_stage1.out
#SBATCH -e logs/%j_e2e_stage1.err
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# SLURM stages the submit script under /var/spool/slurmd/... so BASH_SOURCE
# is useless for locating the repo. Use SLURM_SUBMIT_DIR — submit from the
# repo root: `cd <repo> && sbatch scripts/slurm_frontier/train_e2e_stage1.sh`.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_settings.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1"
mkdir -p logs "${CHECKPOINT_DIR}"

export MASTER_PORT=29500
source scripts/slurm_frontier/_frontier_settings.sh

# Auto-resume from previous chained submission. Pass --resume_checkpoint
# only when a `_latest.pt` is on disk; the Python script's flag guard
# would otherwise fall through to fresh init anyway, but being explicit
# makes the log line show whether we resumed or started cold.
RESUME_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage1_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[train_e2e_stage1] resuming from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
else
    echo "[train_e2e_stage1] no latest checkpoint at ${LATEST_CKPT}; starting fresh"
fi

# Per-node sampler: one line per node per minute with mean GPU busy%,
# host RAM, and mean VRAM%. Launched as a side srun step with --overlap
# so it shares the allocation without stealing GPUs. Cost ~0.1% of one
# CPU/node. Killed when this script exits (walltime or normal end).
SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage1.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
     --checkpoint_dir "${CHECKPOINT_DIR}" \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --prediction_horizon_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 256 \
     --n_layers 26 \
     --n_heads 8 \
     --dropout 0.1 \
     --lr 5e-4 \
     --min_lr 1e-6 \
     --warmup_steps 4000 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 64 \
     --num_workers 6 \
     --max_steps 672000 \
     --log_every 50 \
     --val_every 1180 \
     --val_max_batches 100 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     --no_amp_val \
     ${RESUME_FLAG}
