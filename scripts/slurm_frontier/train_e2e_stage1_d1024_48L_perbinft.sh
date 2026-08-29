#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage1_perbinft
#SBATCH -o logs/%j_e2e_stage1_perbinft.out
#SBATCH -e logs/%j_e2e_stage1_perbinft.err
#SBATCH -t 12:00:00
#SBATCH -p extended
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --mail-user=ps9551@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -e

# Per-bin spec-loss fine-tune of Stage 1 d=1024 / 48L. Initialises from
# the converged Stage 1 best.pt (step 118_000) but resets the step
# counter — this is a short fine-tune, not a chain continuation. Saves
# to a SEPARATE checkpoint dir so the original Stage 1 best.pt is
# untouched. Toggle for revert: drop --spec_per_bin_loss from the
# srun args (then this becomes a plain-MAE fine-tune that should
# regress slightly to the original optimum).

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

SOURCE_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L"
SOURCE_BEST="${SOURCE_DIR}/e2e_stage1_best.pt"
CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L_perbinft"
mkdir -p logs "${CHECKPOINT_DIR}"

if [ ! -f "${SOURCE_BEST}" ]; then
    echo "ERROR: source best.pt not found: ${SOURCE_BEST}" >&2
    exit 1
fi

# Distinct port — original Stage 1 d=1024 uses 29515.
export MASTER_PORT=29516
source scripts/slurm_frontier/_frontier_common.sh

# Resume from chain's own latest if this isn't the first job; otherwise
# cold-init from the source best.pt.
RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage1_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[perbinft] resuming chain from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
else
    echo "[perbinft] cold-init from ${SOURCE_BEST}"
    INIT_FLAG="--init_checkpoint ${SOURCE_BEST}"
fi

SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

# Fine-tune knobs (vs the original Stage 1 sbatch):
#   --lr 5e-5      (10× smaller than the original 5e-4; starting from a
#                  converged optimum so we want gentle updates)
#   --max_steps 10000  (~2-3 h at ~3500 steps/hr Stage 1 throughput;
#                       10× val_every gives ~17 val events to track
#                       convergence)
#   --warmup_steps 500 (short warmup since weights are already trained)
#   --val_every 590    (same as original — ~1 epoch at 8N batch=32)
#   --spec_per_bin_loss   NEW: per-(channel, freq-bin) MAE weighting
#                         to counter spec mean-collapse. Reads
#                         'log_per_bin' from preprocessing_stats.pt
#                         (populated by job 4797193).
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
     --d_model 1024 \
     --n_layers 48 \
     --n_heads 8 \
     --dropout 0.1 \
     --lr 5e-5 \
     --min_lr 1e-6 \
     --warmup_steps 500 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 32 \
     --num_workers 6 \
     --max_steps 10000 \
     --log_every 50 \
     --val_every 590 \
     --val_max_batches 100 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     --no_amp_val \
     --backbone_grad_checkpoint \
     --spec_per_bin_loss \
     --spec_per_bin_weight_clamp 10.0 \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
