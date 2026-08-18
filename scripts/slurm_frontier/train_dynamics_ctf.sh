#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_dynamics
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 12:00:00
#SBATCH -p extended
#SBATCH -N 16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
# IGNITE Phase-B (MaskGIT dynamics) trainer — DDP over the pre-encoded frame-code cache.
# PRODUCTION layout: 16 nodes x 8 GCDs, BATCH_SIZE=1 x ACCUM_STEPS=2 -> effective batch 256.
#   sbatch -N 16 --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00 -p extended
# Env overrides: CACHE_DIR, OUT_DIR, DEPTH, D_MODEL, STEPS, BATCH_SIZE, ACCUM_STEPS, LR,
# NUM_WORKERS, MASK_ABSENT, TEST_FRAC, PIN_VAL, SPLIT_SEED, DATA_DIR, PRECOMPUTE.
# Resumes from OUT_DIR/dynamics_latest.pt. Chain with --dependency=afterany:<prev>; multi-partition
# each job (scontrol update Partition=extended,batch,g1), keep -t <=2h for g1 eligibility.
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/slurm_frontier/_frontier_settings.sh

# PRODUCTION cache (8753 shots, actuator time-base fixed, t0_start=0). The old
# ignite_frame_codes (7264 shots, pre-fix) is superseded — a run that silently
# fell back to it would train on the wrong data without erroring.
CACHE_DIR="${CACHE_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes}"
OUT_DIR="${OUT_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_production/runs/ctf_d512L8}"
# ROLLOUT-QUALITY arm (Phase-1). Everything else matches train_dynamics.sh exactly, so this
# launcher IS the baseline plus these four flags — the arms stay comparable.
CTF_FRAC="${CTF_FRAC:-0.5}"
MODALITY_LOSS_WEIGHT="${MODALITY_LOSS_WEIGHT:-sqrt_tokens}"
ACTUATOR_DROPOUT_P="${ACTUATOR_DROPOUT_P:-0.1}"
SF_FRAMES="${SF_FRAMES:-0}"      # stage A is enabled in a SECOND run, after CTF is validated
# Production config: d512xL8 (the capacity arm showed 815 M gave no rollout gain),
# 55399 steps = 10 epochs over 1.42 M windows at effective batch 256.
DEPTH="${DEPTH:-8}"; D_MODEL="${D_MODEL:-512}"; STEPS="${STEPS:-55399}"
# BATCH_SIZE 1 + ACCUM_STEPS 2 = effective 256 on 16 nodes. bs2 is NOT usable at
# d1024xL16 (62.7/64 GiB reserved; killed jobs 5233441 and 5234381) and accumulation
# costs nothing — step time scales linearly with batch at this sequence length.
BATCH_SIZE="${BATCH_SIZE:-1}"; ACCUM_STEPS="${ACCUM_STEPS:-2}"
LR="${LR:-3e-4}"; NUM_WORKERS="${NUM_WORKERS:-4}"
CKPT_EVERY="${CKPT_EVERY:-500}"   # in OPTIMIZER steps (accumulation-independent); must be
                                  # < steps-per-leg so a 12 h leg checkpoints before its wall
PRECOMPUTE="${PRECOMPUTE:-0}"     # 1 = build the frame-code cache (each rank a shot-shard) then exit
MAX_SHOTS="${MAX_SHOTS:-0}"       # cap total shots for a --precompute sanity run (0 = full dataset)
CODEC_TMPL="${CODEC_TMPL:-}"      # precompute codec override, e.g. 'path/codecs/{m}/codec_best.pt'
# probe / architecture overrides (empty = production defaults)
N_HEADS="${N_HEADS:-}"; K0="${K0:-}"; N_PREDICT="${N_PREDICT:-}"
TRAIN_CAP="${TRAIN_CAP:-0}"       # N-shots generalization probe: train on first N shots only
VAL_N="${VAL_N:-0}"               # fixed validation size (shots); 0 = 5% fraction
# Held-out TEST partition, untouched until the end (user convention 0.90/0.05/0.05).
# Without this the split is 0.95/0.05/0 and there is NO test set at all.
TEST_FRAC="${TEST_FRAC:-0.05}"
PIN_VAL="${PIN_VAL:-200729}"      # standing example shot: pinned to val, never trained
VAL_WINDOWS="${VAL_WINDOWS:-32}"  # independent of BATCH_SIZE (see --val_windows)
# MASK_ABSENT: drop ABSENT diagnostics from the CE. ON for production (user 2026-08-12).
# An absent diagnostic feeds its frozen codec a constant, so it encodes to the same null
# codeword in every shot — ~38% of the loss TERMS, because the loss weights every modality
# equally regardless of token count. Their tokens still enter the model as INPUT.
# REQUIRES <cache>/_presence.json to exist ALREADY: train() builds it on rank 0 behind a
# dist.barrier(), so on a 128-rank job the other 127 would wait out a full 8753-shot scan
# and trip the 600 s NCCL watchdog. Build it first with BUILD_PRESENCE=1 — and REBUILD it
# after ANY cache change, since a stale map silently mislabels the shots that changed.
MASK_ABSENT="${MASK_ABSENT:-1}"
mkdir -p logs "${CACHE_DIR}"

# SPLIT_SEED must be NON-ZERO for production: 0 selects the legacy SORTED-TAIL split,
# which puts the highest shot numbers (one whole campaign) in val. Probe v1 showed
# exactly that arrangement collapses cross-campaign — diversity is what flipped the
# sign. 42 is the seed the frozen production split was drawn with.
SPLIT_SEED="${SPLIT_SEED:-42}"
SHOT_SAMPLE="${SHOT_SAMPLE:-0}"   # precompute: random-sample this many shots from ALL data
SHOT_SEED="${SHOT_SEED:-0}"       # seed for SHOT_SAMPLE

# Allocator config must reach the RANKS, not just the submitting shell, so re-export it here
# and ECHO it (the trainer logs the RANK-SEEN value too — the launcher's echo runs in the batch
# step and cannot prove what the srun tasks got).
#
# DO NOT read this as the fix for job 5233441's fragmentation OOM (19.26 GiB "reserved but
# unallocated" at step ~650). MEASURED 2026-08-12, job 5247144 stderr, this ROCm build:
#     UserWarning: expandable_segments not supported on this platform
#     (Triggered internally at c10/hip/HIPAllocatorConfig.h:40)
# The option is accepted and echoed back as if it were active, but the allocator IGNORES it —
# so the log line below is evidence of INTENT, not of effect. What actually keeps the footprint
# survivable is the batch config: BATCH_SIZE 1 + ACCUM_STEPS (measured bs1 30.0 GiB allocated
# vs bs2 49.0 GiB, bs2 reserving 62.7/64) plus the smaller d512xL8 backbone. Left exported so a
# future ROCm that does support it picks the behaviour up for free.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
echo "[ignite_dynamics] PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF} \
(NOTE: expandable_segments is a NO-OP on this ROCm build)"

EXTRA=()
# DATA_DIR: read shots from an OVERLAY instead of the canonical foundation_model
# (which is not group-writable). discover_shots() over the overlay also RESTRICTS
# a re-tokenize to exactly the shots it contains.
[ -n "${DATA_DIR:-}" ] && EXTRA+=(--data_dir "${DATA_DIR}")
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
[ -n "${VAL_WINDOWS:-}" ] && EXTRA+=(--val_windows "${VAL_WINDOWS}")
[ -n "${ACCUM_STEPS:-}" ] && EXTRA+=(--accum_steps "${ACCUM_STEPS}")
[ "${MASK_ABSENT}" = "1" ] && EXTRA+=(--mask_absent)
[ -n "${PRESENCE_PATH:-}" ] && EXTRA+=(--presence_path "${PRESENCE_PATH}")

if [ "${BUILD_PRESENCE:-0}" = "1" ]; then
  # Rebuild <cache>/_presence.json — the absent-diagnostic mask --mask_absent scores against.
  # SINGLE RANK on purpose: build_presence is a serial CPU pass over every cached shot writing
  # ONE file, so extra ranks would each redo the whole scan and race on the output.
  # MUST be re-run whenever the cache changes. train() only builds the map `if not exists`, so a
  # stale one is reused silently: the 2026-08-12 co2 re-tokenize made co2 PRESENT on 190735/190736
  # while the Aug-11 map still recorded it absent — masking away the very data that was added.
  echo "[ignite_dynamics] BUILD_PRESENCE host=$(hostname) cache=${CACHE_DIR} \
out=${PRESENCE_PATH:-${CACHE_DIR}/_presence.json}"
  srun -N 1 -n 1 -c "$SLURM_CPUS_PER_TASK" \
       scripts/slurm_frontier/_srun_rank_wrapper.sh \
       -m tokamak_foundation_model.ignite.train_dynamics \
       --cache_dir "${CACHE_DIR}" --build_presence "${EXTRA[@]}"
elif [ "${PATCH_ACTUATORS:-0}" = "1" ]; then
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
       --val_n "${VAL_N}" \
       --ctf_frac "${CTF_FRAC}" \
       --modality_loss_weight "${MODALITY_LOSS_WEIGHT}" \
       --actuator_dropout_p "${ACTUATOR_DROPOUT_P}" \
       --sf_frames "${SF_FRAMES}" "${EXTRA[@]}"
fi
