#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage2_delta_smoke_48L
#SBATCH -o logs/%j_e2e_stage2_delta_smoke_48L.out
#SBATCH -e logs/%j_e2e_stage2_delta_smoke_48L.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# 48-layer Stage 2 delta smoke (2026-05-20). Warm-starts from the 26L
# production Stage 1.5 best via --init_checkpoint; the trainer auto-
# detects the 26→48 layer extension and initialises the 22 new blocks
# as near-identity. grad_checkpoint_every=10 (full-rollout GC) is
# REQUIRED at 48L — the 26L smoke peaked at 58% VRAM without GC; 48L
# without GC projects to ~108% (OOM). Goal: validate that K=10 rollouts
# fit at 2× backbone depth with full-rollout GC.
#
# Submit with: sbatch -q debug scripts/slurm_frontier/train_e2e_stage2_delta_smoke_48L.sh

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_smoke_48L"
STAGE1_PROD_BEST="/lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt"
mkdir -p logs "${CHECKPOINT_DIR}"

# Distinct port from stage1 prod (29500), stage2 prod (29502),
# stage1 smoke (29510), stage2 smoke (29512), stage1 48L smoke (29513).
export MASTER_PORT=29514
source scripts/slurm_frontier/_frontier_common.sh

# Auto-resume from chained submission; otherwise warm-start init from the
# 26L production Stage 1.5 best (trainer auto-applies near-identity init
# to layers 26-47 via warm_start_extend_backbone).
RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage2_delta_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[train_e2e_stage2_delta_smoke_48L] resuming from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
elif [ -f "${STAGE1_PROD_BEST}" ]; then
    echo "[train_e2e_stage2_delta_smoke_48L] warm-starting 26→48L from ${STAGE1_PROD_BEST}"
    INIT_FLAG="--init_checkpoint ${STAGE1_PROD_BEST}"
else
    echo "ERROR: production Stage 1 best not found at ${STAGE1_PROD_BEST}." >&2
    exit 1
fi

SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

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
     --n_layers 48 \
     --n_heads 8 \
     --dropout 0.1 \
     --K_max 10 \
     --curriculum_steps 200 \
     --grad_checkpoint_every 10 \
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
     --max_steps 300 \
     --log_every 25 \
     --val_every 150 \
     --val_max_batches 5 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
