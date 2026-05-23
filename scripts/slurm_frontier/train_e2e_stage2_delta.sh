#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage2_delta
#SBATCH -o logs/%j_e2e_stage2_delta.out
#SBATCH -e logs/%j_e2e_stage2_delta.err
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

# Submission pattern (matches Stage 1 chained-job recipe):
#
#   # First job — short to land in `batch` partition (2h cap):
#   sbatch -p batch -t 2:00:00 -N 8 scripts/slurm_frontier/train_e2e_stage2_delta.sh
#
#   # Followup 24h jobs on `extended`, chained via afterany so each
#   # resubmit picks up the previous job's _latest.pt automatically:
#   sbatch -p extended -t 24:00:00 -N 8 --dependency=afterany:<prev> \
#       scripts/slurm_frontier/train_e2e_stage2_delta.sh

# Resolve repo from SLURM_SUBMIT_DIR. SLURM stages the script under
# /var/spool/slurmd/... so BASH_SOURCE is useless. Submit from repo root.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_settings.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage2_delta"
STAGE1_CKPT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1"
STAGE1_BEST="${STAGE1_CKPT_DIR}/e2e_stage1_best.pt"
mkdir -p logs "${CHECKPOINT_DIR}"

# Per-stage MASTER_PORT (different from Stage 1's 29500 so concurrent
# jobs don't collide on the rendezvous port).
export MASTER_PORT=29502
source scripts/slurm_frontier/_frontier_settings.sh

# Auto-resume from previous chained submission. If a `_latest.pt` exists
# we resume (chained-job continuation). Otherwise initialise from
# Stage 1's `e2e_stage1_best.pt` via --init_checkpoint (cold start).
RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage2_delta_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[train_e2e_stage2_delta] resuming from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
elif [ -f "${STAGE1_BEST}" ]; then
    echo "[train_e2e_stage2_delta] cold start — initialising from ${STAGE1_BEST}"
    INIT_FLAG="--init_checkpoint ${STAGE1_BEST}"
else
    echo "ERROR: neither ${LATEST_CKPT} nor ${STAGE1_BEST} found." >&2
    echo "       Stage 2 delta needs Stage 1's best.pt to bootstrap." >&2
    exit 1
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

# Validation cadence: at 8 nodes × batch_size=8 (global batch 512),
# 4,632,251 stage-2 train chunks → 9047 steps/epoch. val_every=9047 ≈ 1
# val per epoch — same "1 val per epoch" pattern Stage 1 settled on.
# val_max_batches=30 because Stage 2 val is K_max=10× more expensive
# per batch than Stage 1's single-step val.
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage2_delta.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
     --checkpoint_dir "${CHECKPOINT_DIR}" \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 256 \
     --n_layers 26 \
     --n_heads 8 \
     --dropout 0.1 \
     --K_max 10 \
     --curriculum_steps 180940 \
     --grad_checkpoint_every 0 \
     --mae_weight 1.0 \
     --cos_weight 0.3 \
     --mag_weight 0.1 \
     --min_disp_norm 0.01 \
     --lr 5e-4 \
     --min_lr 1e-6 \
     --warmup_steps 500 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 8 \
     --num_workers 6 \
     --max_steps 180940 \
     --log_every 50 \
     --val_every 9047 \
     --val_max_batches 30 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
