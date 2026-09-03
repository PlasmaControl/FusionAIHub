#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage2_d1024
#SBATCH -o logs/%j_e2e_stage2_d1024.out
#SBATCH -e logs/%j_e2e_stage2_d1024.err
#SBATCH -t 24:00:00
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

# Stage 2 delta — d_model=1024 / n_layers=48 variant. Warm-starts from
# the d=1024 Stage 1 best.pt and applies K=10-step rollout supervision.
# Forked from train_e2e_stage2_delta.sh (d=256 version) 2026-05-28.
# Memory budget at d=1024:
#   - Stage 1 d=1024 needed --backbone_grad_checkpoint to fit at batch=32.
#   - Stage 2 K=10 rollout uses --grad_checkpoint_every=10 (== K_max) so
#     the entire rollout is one checkpoint group — single forward kept,
#     full recompute in backward. Together with --backbone_grad_checkpoint
#     per layer, batch=2 fits at d=1024 with comfortable VRAM margin.
#   - The stage 2 delta path only supports gc_every=0 (off) or
#     gc_every >= k_steps (single group); per-group chunking is not
#     ported. gc_every=1 worked while curriculum K=1 but raised
#     NotImplementedError as soon as K advanced to 2 (job 4735214,
#     step ~18094). Matches d=256 prod (train_e2e_stage2_delta.sh:109).

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_d1024_48L"
STAGE1_CKPT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L"
STAGE1_BEST="${STAGE1_CKPT_DIR}/e2e_stage1_best.pt"
mkdir -p logs "${CHECKPOINT_DIR}"

# Distinct port — d=256 Stage 2 uses 29502, d=256 Stage 1 uses 29500,
# d=1024 Stage 1 uses 29515.
export MASTER_PORT=29503
source scripts/slurm_frontier/_frontier_common.sh

RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage2_delta_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[train_e2e_stage2_d1024] resuming from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
elif [ -f "${STAGE1_BEST}" ]; then
    echo "[train_e2e_stage2_d1024] cold start — initialising from ${STAGE1_BEST}"
    INIT_FLAG="--init_checkpoint ${STAGE1_BEST}"
else
    echo "ERROR: neither ${LATEST_CKPT} nor ${STAGE1_BEST} found." >&2
    echo "       d=1024 Stage 2 needs d=1024 Stage 1 best.pt to bootstrap." >&2
    exit 1
fi

SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

# Stage 2 dataset has 4,632,251 chunks at the K=10 horizon. With 8 nodes
# × batch_size=2, world batch = 128 → 36,189 steps/epoch. Default "1 val
# per epoch" (36_189) is too sparse here: step time grows with K (1→10
# under the curriculum) so 36k steps takes ~15 h, longer than the 24 h
# walltime can comfortably cover. Without a val we never write a
# latest.pt → chain resumes from Stage 1 best.pt every job, never
# accumulates. val_every=4500 → first val at step 4500 (~1.5 h while
# K=1), ~5-6 vals per 24 h slot.
VAL_EVERY="${VAL_EVERY:-4500}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-30}"
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
     --d_model 1024 \
     --n_layers 48 \
     --n_heads 8 \
     --dropout 0.1 \
     --K_max 10 \
     --curriculum_steps 180940 \
     --grad_checkpoint_every 10 \
     --backbone_grad_checkpoint \
     --mae_weight 1.0 \
     --cos_weight 1.0 \
     --mag_weight 0.5 \
     --min_disp_norm 0.01 \
     --lr 2e-4 \
     --min_lr 1e-6 \
     --warmup_steps 500 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 2 \
     --num_workers 4 \
     --max_steps 180940 \
     --log_every 50 \
     --val_every "${VAL_EVERY}" \
     --val_max_batches "${VAL_MAX_BATCHES}" \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
