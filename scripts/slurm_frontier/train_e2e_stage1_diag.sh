#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage1_diag
#SBATCH -o logs/%j_e2e_stage1_diag.out
#SBATCH -e logs/%j_e2e_stage1_diag.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Diagnostic Stage-1 run to test whether the AWS-OFI-NCCL plugin is
# the cause of the post-maintenance NCCL hangs (2026-05-27). The plugin
# LD_LIBRARY_PATH export is commented out in _frontier_common.sh for
# this test. Goal: train ~20 steps with 1 val; if collectives complete
# we've isolated the plugin.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"

# Separate checkpoint dir so this test never overwrites the production
# 48L _latest.pt. Resume reads from the production state.
CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1_48L_diag"
PROD_LATEST="/lustre/orion/fus187/proj-shared/models/e2e_stage1_48L/e2e_stage1_latest.pt"
mkdir -p logs "${CHECKPOINT_DIR}"

# Distinct port to avoid collision with held production chain (29500).
export MASTER_PORT=29516
source scripts/slurm_frontier/_frontier_common.sh

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
     --n_layers 48 \
     --n_heads 8 \
     --dropout 0.1 \
     --lr 5e-4 \
     --min_lr 1e-6 \
     --warmup_steps 4000 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 64 \
     --num_workers 6 \
     --max_steps 56680 \
     --log_every 1 \
     --val_every 10 \
     --val_max_batches 5 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     --no_amp_val \
     --resume_checkpoint "${PROD_LATEST}"
