#!/bin/bash
# Frontier launcher — Stage 2 (delta) POC for the generative spectro head +
# resize-conv video. Warm-starts from the Stage-1 generative POC best.pt and
# applies low-K rollout supervision, so the K-step block-mode render can show
# whether (a) the resize-conv removes the autoregressive checkerboard (only
# visible in the rollout, NOT Stage 1) and (b) the flow head keeps coherent
# modes through the rollout.  See docs/stage2_genvid_integration_plan.md.
#
# GATED: submit ONLY after the Stage-1 genvid POC (e2e_poc_genvid) has a
# best.pt with modes recovered. Init checkpoint must exist.
#
# Usage:  sbatch -p extended scripts/slurm_frontier/train_e2e_stage2_poc_genvid.sh
#
#SBATCH -A fus187
#SBATCH -J e2e_s2_poc_genvid
#SBATCH -o logs/%j_e2e_s2_poc_genvid.out
#SBATCH -e logs/%j_e2e_s2_poc_genvid.err
#SBATCH -t 08:00:00
#SBATCH -p extended
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

export MASTER_PORT="${MASTER_PORT:-29531}"
source scripts/slurm_frontier/_frontier_common.sh

NODES="${SLURM_JOB_NUM_NODES:-1}"
TOTAL_RANKS="${SLURM_NTASKS:-$((NODES * 1))}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-7}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre/orion/fus187/proj-shared/models/e2e_stage2_poc_genvid}"
STAGE1_GENVID_BEST="${STAGE1_GENVID_BEST:-/lustre/orion/fus187/proj-shared/models/e2e_poc_genvid/e2e_stage1_best.pt}"
LENGTHS_CACHE_DIR="${LENGTHS_CACHE_DIR:-${CHECKPOINT_DIR}/cache}"
mkdir -p "$CHECKPOINT_DIR" "$LENGTHS_CACHE_DIR"

RESUME_FLAG=""; INIT_FLAG=""
LATEST="${CHECKPOINT_DIR}/e2e_stage2_delta_latest.pt"
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "[s2_poc] resuming from $LATEST"
elif [ -f "$STAGE1_GENVID_BEST" ]; then
    INIT_FLAG="--init_checkpoint $STAGE1_GENVID_BEST"
    echo "[s2_poc] cold start — init from $STAGE1_GENVID_BEST"
else
    echo "ERROR: Stage-1 genvid best.pt not found: $STAGE1_GENVID_BEST" >&2
    echo "       Run the Stage-1 genvid POC first." >&2
    exit 1
fi

# POC scale: match the Stage-1 genvid POC (d512/12L) so the init loads.
# Low K (8) for a cheap rollout that still exposes the checkerboard.
K_MAX="${K_MAX:-8}"
MAX_STEPS="${MAX_STEPS:-3000}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-1500}"   # ramp K 1→8 over the first half
BATCH_SIZE="${BATCH_SIZE:-4}"                  # rollout is memory-heavy (×K)
VAL_EVERY="${VAL_EVERY:-500}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-30}"

echo "[s2_poc/genvid] nodes=$NODES ranks=$TOTAL_RANKS K_max=$K_MAX \
batch=$BATCH_SIZE steps=$MAX_STEPS ckpt=$CHECKPOINT_DIR"

srun --overlap -N "$NODES" -n "$TOTAL_RANKS" -c "$CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage2_delta.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
     --checkpoint_dir "${CHECKPOINT_DIR}" \
     --lengths_cache_dir "${LENGTHS_CACHE_DIR}" \
     --max_files 200 \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 512 \
     --n_layers 12 \
     --n_heads 8 \
     --dropout 0.1 \
     --K_max "${K_MAX}" \
     --curriculum_steps "${CURRICULUM_STEPS}" \
     --grad_checkpoint_every "${K_MAX}" \
     --backbone_grad_checkpoint \
     --mae_weight 1.0 \
     --cos_weight 1.0 \
     --mag_weight 0.5 \
     --min_disp_norm 0.01 \
     --lr 2e-4 \
     --min_lr 1e-6 \
     --warmup_steps 300 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size "${BATCH_SIZE}" \
     --num_workers 4 \
     --max_steps "${MAX_STEPS}" \
     --log_every 50 \
     --val_every "${VAL_EVERY}" \
     --val_max_batches "${VAL_MAX_BATCHES}" \
     --use_video tangtv \
     --use_spectro ece co2 \
     --video_resize_conv \
     --spec_generative \
     --spec_flow_steps 6 \
     --collapse_aware_best \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
