#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_s1_specfix_smoke
#SBATCH -o logs/%j_e2e_stage1_specfix_smoke.out
#SBATCH -e logs/%j_e2e_stage1_specfix_smoke.err
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

# 1-1 comparison smoke (2026-06-12): the ORIGINAL Stage 1 d=1024/48L
# production configuration with EXACTLY these deltas and nothing else:
#   (1) new model architecture  — spec inv_stem + 64ch/5x5 seam refine
#   (2) new loss                — per-(channel, freq-bin) weighted MAE
#   (3) frozen backbone         — backbone + slow_ts + fast_ts via
#                                 --freeze_whole_run (pre-DDP-wrap)
#   (4) NO --backbone_grad_checkpoint (dropped per A/B design: with
#       the backbone frozen, static memory falls ~8 GB — params'
#       grads/Adam states — so full activations should fit at
#       batch=32; removing gc also removes the 48-layer recompute
#       from every backward).
# All other trainer args are verbatim from
# train_e2e_stage1_d1024_48L.sh: lr 5e-4, warmup 4000, max_steps
# 118000, batch 32, workers 6, val_every 590, val_max_batches 100,
# dropout 0.1, seed 42, val_fraction 0.1.
#
# Reference step rate to beat/match: production Stage 1 ≈ 2.5-3 s/step.
# Failed specfix attempt 4803320 (with gc + 12h config): ~75 s/step.
#
# Operational deviations (documented, not part of the A/B):
#   - SEPARATE CHECKPOINT_DIR (smoke must never touch production
#     checkpoints — the original's resume logic would otherwise pick
#     up production e2e_stage1_latest.pt).
#   - --init_checkpoint from Stage 1 best (the fine-tune premise).
#   - MIOPEN_FIND_MODE=FAST: without it, MIOpen's exhaustive tuning of
#     the new conv shapes exceeds the 30-min NCCL watchdog and kills
#     the job (4802391). Production never sets it, but production
#     also never runs these conv shapes.
#   - 2 h walltime, -p batch (smoke; g1 is production-only).

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

SOURCE_BEST="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L/e2e_stage1_best.pt"
CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L_specfix_smoke"
mkdir -p logs "${CHECKPOINT_DIR}"

if [ ! -f "${SOURCE_BEST}" ]; then
    echo "ERROR: source best.pt not found: ${SOURCE_BEST}" >&2
    exit 1
fi

# Distinct port — specfix fine-tune uses 29517, stage2 specfix 29518.
export MASTER_PORT=29519
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_FIND_MODE=FAST

RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage1_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[specfix-smoke] resuming from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
else
    echo "[specfix-smoke] cold-init from ${SOURCE_BEST}"
    INIT_FLAG="--init_checkpoint ${SOURCE_BEST}"
fi

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
     --d_model 1024 \
     --n_layers 48 \
     --n_heads 8 \
     --dropout 0.1 \
     --lr 5e-4 \
     --min_lr 1e-6 \
     --warmup_steps 4000 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 32 \
     --num_workers 6 \
     --max_steps 118000 \
     --log_every 50 \
     --val_every 590 \
     --val_max_batches 100 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     --no_amp_val \
     --spec_per_bin_loss \
     --spec_per_bin_weight_clamp 10.0 \
     --spec_inv_stem \
     --spec_inv_stem_ch 64 \
     --spectro_seam_refine \
     --video_seam_refine \
     --seam_refine_hidden_ch 64 \
     --spectro_refine_kernel 5 \
     --video_refine_kernel 3 5 5 \
     --freeze_whole_run \
     --freeze_backbone_steps 1 \
     --freeze_slow_ts_steps 1 \
     --freeze_fast_ts_steps 1 \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
