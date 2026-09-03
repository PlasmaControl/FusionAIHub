#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage1_specfix
#SBATCH -o logs/%j_e2e_stage1_specfix.out
#SBATCH -e logs/%j_e2e_stage1_specfix.err
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

# Spec-fix fine-tune of Stage 1 d=1024/48L — consolidated experiment
# (2026-06-12) attacking spectrogram mean-collapse + patch-grid
# checkerboard in one run. Inits from the converged Stage 1 best.pt;
# saves to a SEPARATE checkpoint dir (original untouched).
#
# Features (all opt-in flags, absent from production sbatches):
#   1. --spec_per_bin_loss        per-(channel, freq-bin) weighted MAE —
#                                 rebalances loss across bins so quiet,
#                                 mode-carrying bins get equal pressure
#   2. --spec_inv_stem            fast-TS-style feature-space decode
#                                 branch on spec heads (zero-init)
#   3. --spectro/video_seam_refine + 64ch/5x5 kernels — strengthened
#                                 anti-checkerboard refine blocks
#                                 (zero-init)
#   4. Frozen backbone + slow_ts + fast_ts (--freeze_whole_run applies
#                                 freezes BEFORE the DDP wrap)
# Trainable: spectro tokenizers+heads (incl. new modules), video
# tokenizer+head (incl. refine), actuator tokenizers.
#
# Revert: this run writes only to e2e_stage1_d1024_48L_specfix/.
# Dropping any flag reverts that feature; the production Stage 1/2
# sbatches never pass these flags and are untouched.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

SOURCE_BEST="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L/e2e_stage1_best.pt"
# _specfix2 (2026-06-13): freq-stem + squared weights run. Fresh dir so
# the resume logic cold-inits from Stage 1 best.pt rather than picking
# up the stale power=1 / no-freq-stem latest.pt from the _specfix run.
CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L_specfix2"
mkdir -p logs "${CHECKPOINT_DIR}"

if [ ! -f "${SOURCE_BEST}" ]; then
    echo "ERROR: source best.pt not found: ${SOURCE_BEST}" >&2
    exit 1
fi

# Distinct port — Stage 1 d=1024 uses 29515, perbinft used 29516.
export MASTER_PORT=29517
source scripts/slurm_frontier/_frontier_common.sh

# MIOpen note (2026-06-12, jobs 4802391 / 4803320 / 4803873): novel
# conv shapes (64ch/5x5 refine, (3,5,5) Conv3d) forced a lose-lose —
# default find mode = >30 min exhaustive tuning > NCCL watchdog
# (4802391 dead); MIOPEN_FIND_MODE=FAST = workspace-starved fallback
# kernels at ~75 s/step with 99% gpu_busy (4803320). Resolution: the
# refine blocks below use the PROVEN Stage 2 shapes (16ch, 3x3,
# (1,3,3)) which resolve instantly from the system find-db, FAST mode
# is NOT set (production MIOpen behavior), and the only near-novel
# shapes left are the inv_stem's (1024->64 deconv — next door to the
# long-tuned 1024->40 patch_unembed — and two 64ch 3x3 convs), whose
# tuning is expected to take minutes, not tens of minutes.

RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage1_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[specfix] resuming chain from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
else
    echo "[specfix] cold-init from ${SOURCE_BEST}"
    INIT_FLAG="--init_checkpoint ${SOURCE_BEST}"
fi

SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

# lr 5e-5 (10x below original Stage 1) — converged init, gentle updates.
# 10k steps ≈ 17 val events at val_every=590.
# freeze_*_steps values are just on-switches under --freeze_whole_run.
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
     --spec_per_bin_weight_clamp 20.0 \
     --spec_per_bin_weight_power 2.0 \
     --spec_inv_stem \
     --spec_inv_stem_ch 64 \
     --spec_freq_stem \
     --spec_freq_stem_hidden 128 \
     --spectro_seam_refine \
     --video_seam_refine \
     --seam_refine_hidden_ch 16 \
     --spectro_refine_kernel 3 \
     --video_refine_kernel 1 3 3 \
     --freeze_whole_run \
     --freeze_backbone_steps 1 \
     --freeze_slow_ts_steps 1 \
     --freeze_fast_ts_steps 1 \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
