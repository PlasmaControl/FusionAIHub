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
# Env overrides: CACHE_DIR, OUT_DIR, DEPTH, D_MODEL, N_HEADS, K0, N_PREDICT, STEPS, BATCH_SIZE,
# ACCUM_STEPS, GRAD_CKPT, LR, WARMUP_STEPS, MIN_LR_RATIO, BETA2, WEIGHT_DECAY, GEN_MASK_P,
# SS_FINAL_FRAC, SS_RAMP_STEPS, NUM_WORKERS, MASK_ABSENT, TEST_FRAC, PIN_VAL, PIN_TRAIN,
# SPLIT_SEED, DATA_DIR, PRECOMPUTE, CODEC_TMPL,
# TEXT_EMBED_PATH, TEXT_EMBED_DIM, TEXT_DROPOUT_P, TEXT_KEY (TEXT_EMBED_DIM 0/empty
# = no text conditioning at all, the default).
# Resumes from OUT_DIR/dynamics_latest.pt. Chain with --dependency=afterany:<prev>; multi-partition
# each job (scontrol update Partition=extended,batch,g1), keep -t <=2h for g1 eligibility.
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/slurm_frontier/_frontier_settings.sh

# ============================================================================================
# LOCKED PRODUCTION CONFIGURATION (user decision, 2026-08-19). Every value below stays env-
# overridable; what changed is which value you get when you override NOTHING. The rationale for
# each is recorded here because a default with no reason attached is the one that silently drifts.
#
#  * OBJECTIVE = GENIE (GEN_MASK_P=0.0). The plain cosine-prior-on-every-frame MaskGIT objective
#    beat the split-point / generation-mode objective at a COMMON condition: 0.1563 vs 0.1693 CE,
#    and it got there in 14k steps instead of 51k. gen_mask_p > 0 is therefore OFF in production.
#  * SS_FINAL_FRAC=0.75 / SS_RAMP_STEPS=2000. 0.75 is the best single scheduled-sampling setting
#    measured (CE 0.1053, token acc 0.9792); SS=1.0 COLLAPSES (no teacher signal left), so this is
#    a peak, not a monotone knob — do not "round it up". The 2000-step ramp reaches the requested
#    fraction early; the config default (40k) only ever reaches HALF of it on a short leg, which
#    silently tests a weaker schedule than the one that was chosen.
#  * LR=1e-3 with MIN_LR_RATIO=1.0 (flat, no cosine decay), WEIGHT_DECAY=0, BETA2=0.9,
#    WARMUP_STEPS=100. 1e-3 is the rate EVERY converged arm used. 2e-3 bought 27% fewer epochs at
#    d512xL16/N=8 but has NOT been validated at d1024 scale, so it is not the production default.
#  * d_model 1024 x depth 16 x 16 heads. DEPTH BEATS WIDTH at matched params: d512xL16 beat
#    d1024xL4 on rollout with HALF the parameters. 16 heads = d_model/64, the standard ratio.
#  * BATCH_SIZE=1 (+ ACCUM_STEPS=2 => effective 256 on 16 nodes x 8 GCDs). bs2 is a MEASURED OOM
#    at d1024xL16 on the 1593-token frame (62.7 of 64 GiB reserved; killed jobs 5233441 and
#    5234381), and accumulation costs nothing here — step time scales linearly with batch at this
#    sequence length. NOTE the rebuilt cache is a 1017-token frame (see CACHE below), so bs1 is
#    if anything MORE headroom than the condition the OOM was measured at; it stays at 1 because
#    nothing has been measured at bs2 on the new shape.
#  * GRAD_CKPT=1 is MANDATORY, not an optimisation: d_model x depth = 1024 x 16 = 16384, far above
#    the 4096 threshold where the un-checkpointed activation stack stops fitting a 64 GiB GCD.
#  * 14 modalities, mirnov EXCLUDED; actuators 88 channels WITH i_coil.
#  * K0=20 seed frames (1.0 s) -> N_PREDICT=80 predicted frames (4.0 s).
#
# CACHE: this config REQUIRES the rebuilt cache. Both the codec set (co2/mhr/ece are the
# instance-norm-OFF retrains, pinned under ignite_codecs_prod_noinorm) and the actuator width
# (70 -> 88 with i_coil) changed, so the pre-2026-08-19 frame_codes cache is NOT compatible and
# NO existing checkpoint can resume against the new layout. The default points at the REBUILT
# cache on purpose: if it does not exist yet the job fails immediately with "no cached shots"
# instead of silently training on the stale one. (The 7264-shot ignite_frame_codes was superseded
# the same way — a run that fell back to it trained on the wrong data without erroring.)
# FRAME WIDTH CHANGES with the rebuild: the pinned noinorm ece codec patches at 16x16 where the
# old one patched at 8x8, so ece contributes 192 tokens instead of 768 and the frame goes
# 1593 -> 1017 tokens (4x192 spectro + 2x108 video + 7x4 slow-TS + 5 fast-TS). Vocabs move too
# (co2 64000 -> 32768, mhr 1000 -> 32768, ece 1000 -> 32768). The trainer derives BOTH from the
# cache, so nothing here needs editing — but any step budget, memory estimate or eval script that
# hardcoded 1593 is now wrong.
CACHE_DIR="${CACHE_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes_noinorm}"
OUT_DIR="${OUT_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_production/runs/prod_d1024L16}"
DEPTH="${DEPTH:-16}"; D_MODEL="${D_MODEL:-1024}"; STEPS="${STEPS:-55399}"
# STEPS is a WINDOW-COUNT budget, not a model property: 55399 = 10 epochs over the 1.42 M windows
# of the old cache at effective batch 256. Re-derive it once the rebuilt cache is on disk.
BATCH_SIZE="${BATCH_SIZE:-1}"; ACCUM_STEPS="${ACCUM_STEPS:-2}"
GRAD_CKPT="${GRAD_CKPT:-1}"       # 0 disables recompute-in-backward (will OOM at d1024xL16)
LR="${LR:-1e-3}"; NUM_WORKERS="${NUM_WORKERS:-4}"
# Optimizer shape. These were previously UNSET, so the argparse defaults applied (min_lr_ratio
# 0.01 = full cosine decay, weight_decay 0.01, beta2 0.999, no warmup) — i.e. the launcher's
# silent defaults did NOT match the arms that converged. Setting a var to "" restores that
# argparse default, so every one of them is still fully env-overridable.
WARMUP_STEPS="${WARMUP_STEPS-100}"
MIN_LR_RATIO="${MIN_LR_RATIO-1.0}"
BETA2="${BETA2-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY-0}"
# Objective + rollout-drift schedule (see the LOCKED block above).
GEN_MASK_P="${GEN_MASK_P:-0.0}"
SS_FINAL_FRAC="${SS_FINAL_FRAC:-0.75}"
SS_RAMP_STEPS="${SS_RAMP_STEPS:-2000}"
CKPT_EVERY="${CKPT_EVERY:-500}"   # in OPTIMIZER steps (accumulation-independent); must be
                                  # < steps-per-leg so a 12 h leg checkpoints before its wall
PRECOMPUTE="${PRECOMPUTE:-0}"     # 1 = build the frame-code cache (each rank a shot-shard) then exit
MAX_SHOTS="${MAX_SHOTS:-0}"       # cap total shots for a --precompute sanity run (0 = full dataset)
CODEC_TMPL="${CODEC_TMPL:-}"      # precompute codec override, e.g. 'path/codecs/{m}/codec_best.pt'
# Architecture / horizon. Formerly empty (= fall through to DynamicsConfig); now pinned to the
# locked values so the launcher, not a dataclass three files away, is the record of the run.
N_HEADS="${N_HEADS:-16}"; K0="${K0:-20}"; N_PREDICT="${N_PREDICT:-80}"
TRAIN_CAP="${TRAIN_CAP:-0}"       # N-shots generalization probe: train on first N shots only
WINDOWS_PER_SHOT="${WINDOWS_PER_SHOT:-0}"   # keep only the first n windows/shot (0 = all); see
                                            # --windows_per_shot. TRAIN set only.
# RUNTIME CAP OVERRIDE. TRAIN_CAP normally arrives via `sbatch --export`, which SLURM captures at
# SUBMIT time -- a queued leg's cap is therefore frozen, and growing the training pool would mean
# resubmitting the whole chain. This launcher, by contrast, is `exec bash`-ed FROM DISK on every
# leg, so a file read here takes effect at the next leg rotation with no resubmission.
# Added 2026-08-21 to grow 1000->1003 / 4000->4003 when the three pinned VALIDATION shots were
# also added to training (user request), without restarting either chain.
if [ -f "${OUT_DIR}/train_cap.override" ]; then
  _TC=$(tr -dc '0-9' < "${OUT_DIR}/train_cap.override")     # sanitise: set -u + stray newline
  if [ -n "${_TC}" ]; then
    echo "[launcher] TRAIN_CAP override ${TRAIN_CAP} -> ${_TC} (${OUT_DIR}/train_cap.override)"
    TRAIN_CAP="${_TC}"
  fi
fi
VAL_N="${VAL_N:-0}"               # fixed validation size (shots); 0 = 5% fraction
# Held-out TEST partition, untouched until the end (user convention 0.90/0.05/0.05).
# Without this the split is 0.95/0.05/0 and there is NO test set at all.
TEST_FRAC="${TEST_FRAC:-0.05}"
PIN_VAL="${PIN_VAL:-200729}"
# PIN_TRAIN: neighbour stratification. COMMA-separated -- set it INSIDE the job script, never
# via sbatch --export, which splits the export list on commas.
PIN_TRAIN="${PIN_TRAIN:-}"      # standing example shot: pinned to val, never trained
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
# Per-shot text conditioning (optional). TEXT_EMBED_DIM 0/empty = no text flags at
# all (byte-identical command line to before this feature existed). Path and dim
# must be set TOGETHER -- see EXTRA assembly below.
TEXT_EMBED_PATH="${TEXT_EMBED_PATH:-}"
TEXT_EMBED_DIM="${TEXT_EMBED_DIM:-0}"
TEXT_DROPOUT_P="${TEXT_DROPOUT_P:-0.1}"
TEXT_KEY="${TEXT_KEY:-input}"
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

# TEXT_EMBED_PATH and TEXT_EMBED_DIM must be set TOGETHER or not at all: dim-without-path
# would pass --text_embed_path "" and die inside the trainer's TOGETHER guard with a
# confusing h5py error, and path-without-dim would silently launch an UNCONDITIONED
# baseline (the gate below keys on TEXT_EMBED_DIM alone).
if { [ -n "${TEXT_EMBED_PATH}" ] && [ "${TEXT_EMBED_DIM}" = "0" ]; } || \
   { [ -z "${TEXT_EMBED_PATH}" ] && [ "${TEXT_EMBED_DIM}" != "0" ]; }; then
  echo "ERROR: TEXT_EMBED_PATH and TEXT_EMBED_DIM must be set together (path='${TEXT_EMBED_PATH}', dim='${TEXT_EMBED_DIM}')" >&2
  exit 1
fi

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
[ -n "${PIN_TRAIN:-}" ] && EXTRA+=(--pin_train "${PIN_TRAIN}")
[ -n "${VAL_WINDOWS:-}" ] && EXTRA+=(--val_windows "${VAL_WINDOWS}")
[ -n "${DROPOUT:-}" ] && EXTRA+=(--dropout "${DROPOUT}")
[ -n "${ACCUM_STEPS:-}" ] && EXTRA+=(--accum_steps "${ACCUM_STEPS}")
[ "${MASK_ABSENT}" = "1" ] && EXTRA+=(--mask_absent)
[ "${BALANCE_PRESENCE:-0}" = "1" ] && EXTRA+=(--balance_presence)
[ -n "${PRESENCE_PATH:-}" ] && EXTRA+=(--presence_path "${PRESENCE_PATH}")
[ "${TEXT_EMBED_DIM}" != "0" ] && [ -n "${TEXT_EMBED_DIM}" ] && EXTRA+=(
  --text_embed_path "${TEXT_EMBED_PATH}" --text_embed_dim "${TEXT_EMBED_DIM}"
  --text_dropout_p "${TEXT_DROPOUT_P}" --text_key "${TEXT_KEY}"
)

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
  # Echo the LOCKED knobs, not just the shape: an arm that silently reverted one of them
  # (a stale export, an unset var) is otherwise indistinguishable from production in the log.
  echo "[ignite_dynamics] TRAIN host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} world_size=${SLURM_NTASKS} \
cache=${CACHE_DIR} out=${OUT_DIR} depth=${DEPTH} d_model=${D_MODEL} n_heads=${N_HEADS} \
k0=${K0} n_predict=${N_PREDICT} steps=${STEPS} bs=${BATCH_SIZE} accum=${ACCUM_STEPS} \
grad_ckpt=${GRAD_CKPT} lr=${LR} warmup=${WARMUP_STEPS} min_lr_ratio=${MIN_LR_RATIO} \
beta2=${BETA2} wd=${WEIGHT_DECAY} gen_mask_p=${GEN_MASK_P} ss_final=${SS_FINAL_FRAC} \
ss_ramp=${SS_RAMP_STEPS} mask_absent=${MASK_ABSENT}"
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
       --ss_final_frac "${SS_FINAL_FRAC}" \
       --ss_ramp_steps "${SS_RAMP_STEPS}" \
       --gen_mask_p "${GEN_MASK_P}" \
       --grad_ckpt "${GRAD_CKPT}" \
       --best_metric "${BEST_METRIC:-masked}" \
       --gen_horizon_alpha "${GEN_HORIZON_ALPHA:-0}" \
       --gen_horizon_max "${GEN_HORIZON_MAX:-0}" \
       --act_cross_attn "${ACT_CROSS_ATTN:-0}" \
       --lag_embed_k "${LAG_EMBED_K:-0}" \
       --window_stride "${WINDOW_STRIDE:-1}" \
       --windows_per_shot "${WINDOWS_PER_SHOT}" \
       --window_sample "${WINDOW_SAMPLE:-random}" \
       --val_window_sample "${VAL_WINDOW_SAMPLE:-}" \
       --mod_weight_pow "${MOD_WEIGHT_POW:-0}" \
       --lr_decay_from "${LR_DECAY_FROM:-0}" \
       --wd_exclude_norms "${WD_EXCLUDE_NORMS:-0}" \
       --label_smoothing "${LABEL_SMOOTHING:-0}" \
       --val_windows_per_shot "${VAL_WINDOWS_PER_SHOT:-0}" \
       --grad_clip "${GRAD_CLIP:-0}" \
       --train_cap "${TRAIN_CAP}" \
       --val_n "${VAL_N}" "${EXTRA[@]}"
fi
