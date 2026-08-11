#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_dynamics
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p extended
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
# IGNITE Phase-B (MaskGIT dynamics) trainer — DDP over the pre-encoded frame-code cache.
# 8-node / 1-GCD / global-128 layout (same as the decisive chain + codec runs). Env overrides:
# CACHE_DIR, OUT_DIR, DEPTH (depth study), D_MODEL, STEPS, BATCH_SIZE, LR, NUM_WORKERS.
# Resumes from OUT_DIR/dynamics_latest.pt. Chain with --dependency=afterany:<prev>; multi-partition
# each job (scontrol update Partition=extended,batch,g1), keep -t <=2h for g1 eligibility.
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/slurm_frontier/_frontier_common.sh

CACHE_DIR="${CACHE_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_frame_codes}"
OUT_DIR="${OUT_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_dynamics_d1024}"
DEPTH="${DEPTH:-24}"; D_MODEL="${D_MODEL:-1024}"; STEPS="${STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-2}"; LR="${LR:-3e-4}"; NUM_WORKERS="${NUM_WORKERS:-4}"
CKPT_EVERY="${CKPT_EVERY:-200}"   # < steps-per-job (~700 at 2h/g1) so the chain checkpoints + resumes
PRECOMPUTE="${PRECOMPUTE:-0}"     # 1 = build the frame-code cache (each rank a shot-shard) then exit
MAX_SHOTS="${MAX_SHOTS:-0}"       # cap total shots for a --precompute sanity run (0 = full dataset)
CODEC_TMPL="${CODEC_TMPL:-}"      # precompute codec override, e.g. 'path/codecs/{m}/codec_best.pt'
# probe / architecture overrides (empty = production defaults)
N_HEADS="${N_HEADS:-}"; K0="${K0:-}"; N_PREDICT="${N_PREDICT:-}"
TRAIN_CAP="${TRAIN_CAP:-0}"       # N-shots generalization probe: train on first N shots only
VAL_N="${VAL_N:-0}"               # fixed validation-tail size (shots); 0 = 5% fraction
mkdir -p logs "${CACHE_DIR}"

SPLIT_SEED="${SPLIT_SEED:-0}"     # >0: seeded-random train/val split (0 = sorted tail)
SHOT_SAMPLE="${SHOT_SAMPLE:-0}"   # precompute: random-sample this many shots from ALL data
SHOT_SEED="${SHOT_SEED:-0}"       # seed for SHOT_SAMPLE

# Allocator config must reach the RANKS, not just the submitting shell. Job 5233441 OOM'd at
# step ~650 with 19.26 GiB "reserved but unallocated" — textbook fragmentation, which
# expandable_segments fixes. Re-export explicitly and ECHO it so a log proves whether the
# ranks actually saw it (the stage-1 launcher carries the same line).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
echo "[ignite_dynamics] PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}"

EXTRA=()
[ -n "${CODEC_TMPL}" ] && EXTRA+=(--codec_tmpl "${CODEC_TMPL}")
[ -n "${N_HEADS}" ] && EXTRA+=(--n_heads "${N_HEADS}")
[ -n "${K0}" ] && EXTRA+=(--k0_seed "${K0}")
[ -n "${N_PREDICT}" ] && EXTRA+=(--n_predict "${N_PREDICT}")
[ "${SPLIT_SEED}" != "0" ] && EXTRA+=(--split_seed "${SPLIT_SEED}")
[ -n "${WARMUP_STEPS:-}" ] && EXTRA+=(--warmup_steps "${WARMUP_STEPS}")
[ -n "${MIN_LR_RATIO:-}" ] && EXTRA+=(--min_lr_ratio "${MIN_LR_RATIO}")
[ -n "${BETA2:-}" ] && EXTRA+=(--beta2 "${BETA2}")
[ -n "${WEIGHT_DECAY:-}" ] && EXTRA+=(--weight_decay "${WEIGHT_DECAY}")
[ "${SHOT_SAMPLE}" != "0" ] && EXTRA+=(--shot_sample "${SHOT_SAMPLE}" --shot_seed "${SHOT_SEED}")
[ -n "${T0_START:-}" ] && EXTRA+=(--t0_start "${T0_START}")
[ -n "${SHOT_TIMEOUT_S:-}" ] && EXTRA+=(--shot_timeout_s "${SHOT_TIMEOUT_S}")
[ -n "${TEST_N:-}" ] && EXTRA+=(--test_n "${TEST_N}")
[ -n "${TEST_FRAC:-}" ] && EXTRA+=(--test_frac "${TEST_FRAC}")
[ -n "${PIN_VAL:-}" ] && EXTRA+=(--pin_val "${PIN_VAL}")
[ "${MASK_ABSENT:-0}" = "1" ] && EXTRA+=(--mask_absent)
[ -n "${PRESENCE_PATH:-}" ] && EXTRA+=(--presence_path "${PRESENCE_PATH}")

if [ "${PATCH_ACTUATORS:-0}" = "1" ]; then
  # One-off cache repair: rewrite ONLY the actuators with the 2026-08-11 time-base fix.
  # Codes are untouched (diagnostics were always on the correct absolute-time base), so this
  # is an I/O pass, NOT a re-precompute. DRY_RUN=1 verifies without writing.
  echo "[ignite_dynamics] PATCH_ACTUATORS host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} \
world_size=${SLURM_NTASKS} cache=${CACHE_DIR} dry_run=${DRY_RUN:-0}"
  [ "${DRY_RUN:-0}" = "1" ] && EXTRA+=(--dry_run)
  srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
       --gpus-per-task=1 --gpu-bind=closest \
       scripts/slurm_frontier/_srun_rank_wrapper.sh \
       -m tokamak_foundation_model.ignite.train_dynamics \
       --cache_dir "${CACHE_DIR}" --patch_actuators "${EXTRA[@]}"
elif [ "${PRECOMPUTE}" = "1" ]; then
  echo "[ignite_dynamics] PRECOMPUTE host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} \
world_size=${SLURM_NTASKS} cache=${CACHE_DIR} max_shots=${MAX_SHOTS} codec_tmpl=${CODEC_TMPL:-manifest}"
  srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
       --gpus-per-task=1 --gpu-bind=closest \
       scripts/slurm_frontier/_srun_rank_wrapper.sh \
       -m tokamak_foundation_model.ignite.train_dynamics \
       --cache_dir "${CACHE_DIR}" \
       --precompute --max_shots "${MAX_SHOTS}" "${EXTRA[@]}"
else
  mkdir -p "${OUT_DIR}"
  echo "[ignite_dynamics] TRAIN host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} world_size=${SLURM_NTASKS} \
cache=${CACHE_DIR} out=${OUT_DIR} depth=${DEPTH} d_model=${D_MODEL} steps=${STEPS} bs=${BATCH_SIZE}"
  srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
       --gpus-per-task=1 --gpu-bind=closest \
       scripts/slurm_frontier/_srun_rank_wrapper.sh \
       -m tokamak_foundation_model.ignite.train_dynamics \
       --cache_dir "${CACHE_DIR}" \
       --out_dir "${OUT_DIR}" \
       --depth "${DEPTH}" \
       --d_model "${D_MODEL}" \
       --steps "${STEPS}" \
       --batch_size "${BATCH_SIZE}" \
       --lr "${LR}" \
       --num_workers "${NUM_WORKERS}" \
       --ckpt_every "${CKPT_EVERY}" \
       --ss_final_frac "${SS_FINAL_FRAC:-0}" \
       --train_cap "${TRAIN_CAP}" \
       --val_n "${VAL_N}" "${EXTRA[@]}"
fi
