#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage2_ext_smoke_d1024
#SBATCH -o logs/%j_e2e_stage2_ext_smoke_d1024.out
#SBATCH -e logs/%j_e2e_stage2_ext_smoke_d1024.err
#SBATCH -t 2:00:00
#SBATCH -p batch
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

# Stage 2 EXTENDED smoke — d_model=1024 / n_layers=48 / K=80 from t=0.
# Goal: validate WORST-CASE RAM + VRAM budget for the extended trainer.
# Configured to exercise the peak-memory rollout step immediately,
# skipping the K∈{10,20,40} curriculum ramp.
#
# Submit with debug QOS for fast scheduling:
#   sbatch -q debug scripts/slurm_frontier/train_e2e_stage2_extended_smoke_d1024.sh
#
# Worst-case memory knobs:
#   --curriculum_Ks 80         # single K, no warm-up via shorter K
#   --grad_checkpoint_every 10 # 8 groups of 10 — gc=80 OOMs (job 4757298)
#   --backbone_grad_checkpoint # per-layer GC inside backbone
#   batch_size=2               # matches production extended_d1024
#   -N 8                       # matches production node count → same
#                              # per-rank world-batch, same per-GPU memory
#
# Init from Stage 2 delta d=1024 best.pt (same as production extended).
# Smoke writes to a separate checkpoint dir so it can be re-run idempotently
# without disturbing production state.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage2_extended_smoke_d1024_48L"
DELTA_CKPT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_d1024_48L"
DELTA_BEST="${DELTA_CKPT_DIR}/e2e_stage2_delta_best.pt"
mkdir -p logs "${CHECKPOINT_DIR}"

# Distinct port — production extended_d1024 uses 29504.
export MASTER_PORT=29516
source scripts/slurm_frontier/_frontier_common.sh

RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage2_extended_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[ext_smoke_d1024] resuming from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
elif [ -f "${DELTA_BEST}" ]; then
    echo "[ext_smoke_d1024] cold start — initialising from ${DELTA_BEST}"
    INIT_FLAG="--init_checkpoint ${DELTA_BEST}"
else
    echo "ERROR: neither ${LATEST_CKPT} nor ${DELTA_BEST} found." >&2
    echo "       Smoke needs Stage 2 delta d=1024 best.pt to bootstrap." >&2
    exit 1
fi

SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

# 50 training steps + one val pass exercises both fwd/bwd peak (training)
# and fwd-only peak (validation). Each K=80 step at d=1024 is expensive,
# so 50 steps is enough to confirm steady-state memory rather than just
# the cold-start spike.
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage2_extended.py \
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
     --backbone_grad_checkpoint \
     --curriculum_Ks 80 \
     --block_steps 50 \
     --grad_checkpoint_every 10 \
     --mae_weight 1.0 \
     --cos_weight 1.0 \
     --mag_weight 0.5 \
     --min_disp_norm 0.01 \
     --lr 1e-4 \
     --min_lr 1e-6 \
     --warmup_steps 500 \
     --weight_decay 0.01 \
     --grad_clip 5.0 \
     --batch_size 2 \
     --val_batch_size 1 \
     --num_workers 4 \
     --max_steps 50 \
     --log_every 5 \
     --val_every 40 \
     --val_max_batches 5 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
