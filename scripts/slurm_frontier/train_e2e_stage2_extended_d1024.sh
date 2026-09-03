#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage2_ext_d1024
#SBATCH -o logs/%j_e2e_stage2_ext_d1024.out
#SBATCH -e logs/%j_e2e_stage2_ext_d1024.err
#SBATCH -t 48:00:00
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

# Stage 2 EXTENDED — d_model=1024 / n_layers=48 variant. Full-backprop
# rollout fine-tune with stepwise K curriculum {10, 20, 40, 80}.
# Forked from train_e2e_stage2_delta_d1024.sh 2026-06-02.
#
# Differences vs Stage 2 delta:
#   - Trainer:    train_e2e_stage2_extended.py (not _delta).
#   - Curriculum: --curriculum_Ks 10,20,40,80 + --block_steps (vs delta's
#                 --K_max + --curriculum_steps).
#   - Loss:       same MAE + cos + log-mag displacement weights.
#   - lr:         1e-5 → 1e-7 cosine (vs delta's 2e-4 → 1e-6). This is a
#                 fine-tune of a converged delta backbone, not a re-train.
#   - Warmup:     500 steps cosine warmup at every job-restart.
#   - Init:       Stage 2 delta d=1024 best.pt (NOT Stage 1 best). Chain
#                 resumes from this script's own _latest.pt thereafter.
#
# Memory budget at d=1024:
#   - --backbone_grad_checkpoint mandatory (per-block GC inside the 1.33B
#     param backbone). Without it, K=80 OOMs trivially.
#   - --grad_checkpoint_every 10 — splits the K=80 rollout into 8
#     groups of 10, keeping the peak activation footprint to one
#     group's worth (~10× lower than gc_every=K_max=80). Smoke 4758855
#     proved: gc_every=80 OOMs the K=80 backward at d=1024 (job 4757298),
#     gc_every=10 fits comfortably at VRAM 82 %. The extended trainer's
#     loop `for group_start in range(0, k_steps, group_size)` supports
#     any group_size <= k_steps (delta path doesn't — only delta needs
#     gc_every >= K_max because of its single-group implementation).
#     Trade-off: ~8× more recompute passes at K=80, but step time is
#     still bounded by GPU compute, not memory traffic.
#   - batch_size=2 matches delta_d1024.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage2_extended_d1024_48L"
DELTA_CKPT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_d1024_48L"
DELTA_BEST="${DELTA_CKPT_DIR}/e2e_stage2_delta_best.pt"
mkdir -p logs "${CHECKPOINT_DIR}"

# Distinct port from Stage 1 d=1024 (29515), Stage 2 delta d=1024 (29503).
export MASTER_PORT=29504
source scripts/slurm_frontier/_frontier_common.sh

RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage2_ext_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[train_e2e_stage2_ext_d1024] resuming from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
elif [ -f "${DELTA_BEST}" ]; then
    echo "[train_e2e_stage2_ext_d1024] cold start — initialising from ${DELTA_BEST}"
    INIT_FLAG="--init_checkpoint ${DELTA_BEST}"
else
    echo "ERROR: neither ${LATEST_CKPT} nor ${DELTA_BEST} found." >&2
    echo "       d=1024 Stage 2 extended needs d=1024 Stage 2 delta best.pt to bootstrap." >&2
    exit 1
fi

SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

# Curriculum: K∈{10,20,40,80}, 5000 steps per block → 20000 total.
# At d=1024 / 8N / batch=2, step time scales ~linearly with K. Rough
# wall-clock per block (extrapolating delta_d1024's ~30k steps/day at K=10):
#   K=10: ~4 h   K=20: ~8 h   K=40: ~16 h   K=80: ~32 h
# → expect 5 jobs of 24 h walltime to chain through.
BLOCK_STEPS="${BLOCK_STEPS:-5000}"
CURRICULUM_KS="${CURRICULUM_KS:-10,20,40,80}"
# max_steps = block_steps × number of curriculum K values.
N_K=$(echo "$CURRICULUM_KS" | tr ',' '\n' | wc -l)
MAX_STEPS=$((BLOCK_STEPS * N_K))

# val_every=500 (was 2500): _latest.pt only saves at vals, so frequent vals
# (a) cap the loss from an NCCL-watchdog crash to ≤500 steps instead of a
# full ~2500-step block, and (b) are REQUIRED so the K=80 block (steps
# 15k-20k) can clear a val interval within one 48h job — see the
# checkpoint-deadlock note. ~2-4% val overhead.
VAL_EVERY="${VAL_EVERY:-500}"
# 2026-06-11: bumped from val_batch_size=1, val_max_batches=30 → 2/60.
# Ext val 1 showed co2/bes=0.000 across all K because the previous
# config + shuffle=False on val_loader meant every rank consumed the
# first 1-2 files of val_ds, and stub-data shots (~62% BES, ~45% CO2)
# clustered there masked the entire aggregate. Combined with the new
# DistributedSampler(shuffle=True) on val_loader in
# train_e2e_stage2_extended.py, each rank now hits a different
# strided shuffle of windows, and the 2× batch + 2× max_batches gives
# 7680 val samples total (4× previous 1920). Smoke 4758855's
# val_batch_size=2 OOM concern was at the K=80 *training* transition
# — observed at K=10 the val with batch=2 leaves ~12 GB VRAM margin
# per GCD (peak ~52 GB / 64 GB).
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-60}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"

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
     --curriculum_Ks "${CURRICULUM_KS}" \
     --block_steps "${BLOCK_STEPS}" \
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
     --val_batch_size "${VAL_BATCH_SIZE}" \
     --num_workers 4 \
     --max_steps "${MAX_STEPS}" \
     --log_every 50 \
     --val_every "${VAL_EVERY}" \
     --val_max_batches "${VAL_MAX_BATCHES}" \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
