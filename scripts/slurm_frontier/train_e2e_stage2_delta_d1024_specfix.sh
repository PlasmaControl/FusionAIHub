#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage2_specfix
#SBATCH -o logs/%j_e2e_stage2_specfix.out
#SBATCH -e logs/%j_e2e_stage2_specfix.err
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

# Stage 2 specfix fine-tune (2026-06-12) — DO NOT SUBMIT before the
# Stage 1 specfix gate (render of e2e_stage1_d1024_48L_specfix best.pt
# shows real spectral modes). Same four features as Stage 1 specfix,
# applied to the K=10 delta-rollout objective:
#   per-bin spec MAE + spec inv_stem + 64ch/5x5 refine + frozen
#   backbone/slow_ts/fast_ts.
# Inits from the FINAL Stage 2 delta best.pt. The delta checkpoint's
# trained 16ch/3x3 refine_block weights are shape-mismatched against
# the 64ch/5x5 blocks and are dropped + re-initialized (zero-init =
# identity); the trainer logs the dropped keys.
# K curriculum: --curriculum_steps 10 ramps K 1→10 within the first
# 10 steps (block=1), i.e. effectively K=10 from the start.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

SOURCE_BEST="/lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_d1024_48L/e2e_stage2_delta_best.pt"
CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_d1024_48L_specfix"
mkdir -p logs "${CHECKPOINT_DIR}"

if [ ! -f "${SOURCE_BEST}" ]; then
    echo "ERROR: source best.pt not found: ${SOURCE_BEST}" >&2
    exit 1
fi

# Distinct port — Stage 1 specfix uses 29517.
export MASTER_PORT=29518
source scripts/slurm_frontier/_frontier_common.sh

# No MIOPEN_FIND_MODE override — refine blocks use the proven Stage 2
# shapes (instant find-db hits) and the inv_stem shapes tune in
# minutes under the default mode. See the Stage 1 specfix sbatch for
# the 2026-06-12 incident note (FAST mode = 75 s/step fallback kernels).

RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage2_delta_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[s2-specfix] resuming chain from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
else
    echo "[s2-specfix] cold-init from ${SOURCE_BEST}"
    INIT_FLAG="--init_checkpoint ${SOURCE_BEST}"
fi

SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

# val_every 1000 → ~10 checkpoint opportunities across 10k steps.
# lr 2e-5 = prod 2e-4 / 10 (fine-tune from converged Stage 2).
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
     --curriculum_steps 10 \
     --grad_checkpoint_every 10 \
     --backbone_grad_checkpoint \
     --mae_weight 1.0 \
     --cos_weight 1.0 \
     --mag_weight 0.5 \
     --min_disp_norm 0.01 \
     --lr 2e-5 \
     --min_lr 1e-6 \
     --warmup_steps 200 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 2 \
     --num_workers 4 \
     --max_steps 10000 \
     --log_every 50 \
     --val_every 1000 \
     --val_max_batches 30 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     --spec_per_bin_loss \
     --spec_per_bin_weight_clamp 20.0 \
     --spec_per_bin_weight_power 2.0 \
     --spec_inv_stem \
     --spec_inv_stem_ch 64 \
     --spec_freq_stem \
     --spec_freq_stem_hidden 128 \
     --seam_refine_hidden_ch 16 \
     --spectro_refine_kernel 3 \
     --video_refine_kernel 1 3 3 \
     --freeze_categories backbone slow_ts fast_ts \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
