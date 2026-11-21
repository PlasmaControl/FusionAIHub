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
# IGNITE Phase-B (MaskGIT dynamics) trainer: DDP over the pre-encoded frame-code cache.
# Frontier layout: 16 nodes x 8 GCDs, BATCH_SIZE=1 x ACCUM_STEPS=2 -> effective batch 256.
#   sbatch -N 16 --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00 -p extended
# Env overrides: CACHE_DIR, OUT_DIR, DEPTH, D_MODEL, N_HEADS, K0, N_PREDICT, STEPS, BATCH_SIZE,
# ACCUM_STEPS, GRAD_CKPT, LR, WARMUP_STEPS, MIN_LR_RATIO, BETA2, WEIGHT_DECAY, GEN_MASK_P,
# SS_FINAL_FRAC, SS_RAMP_STEPS, NUM_WORKERS, MASK_ABSENT, TEST_FRAC, PIN_VAL, SPLIT_SEED,
# DATA_DIR, PRECOMPUTE, CODEC_TMPL.
# Resumes from OUT_DIR/dynamics_latest.pt. Chain with --dependency=afterany:<prev>. On
# Frontier, set each job to several partitions (scontrol update Partition=extended,batch,g1)
# and keep -t <=2h for g1.
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
# CLUSTER_ENV: environment hook of the cluster (default Frontier/ROCm; StellarAI sets
# scripts/slurm_stellarai/_stellarai_common.sh).
source "${CLUSTER_ENV:-scripts/slurm_frontier/_frontier_common.sh}"

# Defaults. Every value stays env-overridable; the reasons are noted so they do not drift.
#  * GEN_MASK_P=0.0: the plain MaskGIT objective (cosine mask prior on every frame) beat
#    the split-point generation objective, 0.1563 vs 0.1693 CE, in 14k steps instead of 51k.
#  * SS_FINAL_FRAC=0.75 / SS_RAMP_STEPS=2000. 0.75 was the best scheduled-sampling value
#    measured (CE 0.1053, token acc 0.9792); 1.0 collapses (no teacher signal left), so do
#    not round it up. The short ramp reaches the target early; the config default ramp
#    (40k) reaches only half of it on a short leg.
#  * LR=1e-3 flat (MIN_LR_RATIO=1.0, no cosine decay), WEIGHT_DECAY=0, BETA2=0.9,
#    WARMUP_STEPS=100. Every converged run used 1e-3; 2e-3 was faster at d512xL16 but is
#    untested at d1024.
#  * d_model 1024 x depth 16 x 16 heads. Depth beats width at matched parameters: d512xL16
#    beat d1024xL4 on rollout with half the parameters. 16 heads = d_model/64.
#  * BATCH_SIZE=1 with accumulation: bs2 ran out of memory at d1024xL16 on a 64 GiB GCD
#    (62.7 GiB reserved on the 1593-token frame), and accumulation costs nothing since
#    step time scales linearly with batch at this sequence length.
#  * GRAD_CKPT=1 is required: above d_model x depth = 4096 the activations without
#    recompute do not fit a 64 GiB GCD, and 1024 x 16 = 16384.
#  * K0=20 seed frames (1.0 s) -> N_PREDICT=80 predicted frames (4.0 s).
#
# CACHE_DIR: the default points at the rebuilt cache, so that if it is missing the job
# fails with "no cached shots" instead of training on an older, incompatible cache.
# The trainer derives the frame width and vocabularies from the cache, so a cache with
# different codecs needs no edit here; only hardcoded token counts elsewhere would be wrong.
CACHE_DIR="${CACHE_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes_noinorm}"
OUT_DIR="${OUT_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_production/runs/prod_d1024L16}"
DEPTH="${DEPTH:-16}"; D_MODEL="${D_MODEL:-1024}"; STEPS="${STEPS:-55399}"
# STEPS is a window-count budget: 55399 = 10 epochs over the 1.42 M windows of the old
# cache at effective batch 256. Re-derive it for another cache.
BATCH_SIZE="${BATCH_SIZE:-1}"; ACCUM_STEPS="${ACCUM_STEPS:-2}"
GRAD_CKPT="${GRAD_CKPT:-1}"       # 0 disables recompute in backward (runs out of memory at d1024xL16)
LR="${LR:-1e-3}"; NUM_WORKERS="${NUM_WORKERS:-4}"
# Optimizer shape. Setting one of these to "" falls back to the argparse default
# (min_lr_ratio 0.01 = full cosine decay, weight_decay 0.01, beta2 0.999, no warmup).
WARMUP_STEPS="${WARMUP_STEPS-100}"
MIN_LR_RATIO="${MIN_LR_RATIO-1.0}"
BETA2="${BETA2-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY-0}"
# Objective and scheduled sampling (see the defaults above).
GEN_MASK_P="${GEN_MASK_P:-0.0}"
SS_FINAL_FRAC="${SS_FINAL_FRAC:-0.75}"
SS_RAMP_STEPS="${SS_RAMP_STEPS:-2000}"
CKPT_EVERY="${CKPT_EVERY:-500}"   # in optimizer steps (independent of accumulation); keep it
                                  # below the steps per leg so a 12 h leg saves before its wall
PRECOMPUTE="${PRECOMPUTE:-0}"     # 1 = build the frame-code cache (each rank a shard of shots), then exit
MAX_SHOTS="${MAX_SHOTS:-0}"       # cap total shots for a --precompute sanity run (0 = full dataset)
CODEC_TMPL="${CODEC_TMPL:-}"      # precompute codec override, e.g. 'path/codecs/{m}/codec_best.pt'
# Architecture and horizon, set here so the launcher records the run.
N_HEADS="${N_HEADS:-16}"; K0="${K0:-20}"; N_PREDICT="${N_PREDICT:-80}"
TRAIN_CAP="${TRAIN_CAP:-0}"       # train on the first N shots only (0 = all)
WINDOWS_PER_SHOT="${WINDOWS_PER_SHOT:-0}"   # keep only the first n windows per shot (0 = all);
                                            # training set only, see --windows_per_shot
# Runtime TRAIN_CAP override. `sbatch --export` values are fixed at submit time, but this
# script is read from disk on every leg, so a train_cap.override file in OUT_DIR changes
# the cap at the next leg without resubmitting the chain.
if [ -f "${OUT_DIR}/train_cap.override" ]; then
  _TC=$(tr -dc '0-9' < "${OUT_DIR}/train_cap.override")     # digits only (set -u, stray newline)
  if [ -n "${_TC}" ]; then
    echo "[launcher] TRAIN_CAP override ${TRAIN_CAP} -> ${_TC} (${OUT_DIR}/train_cap.override)"
    TRAIN_CAP="${_TC}"
  fi
fi
VAL_N="${VAL_N:-0}"               # fixed validation size (shots); 0 = 5% fraction
# Held-out test fraction (split 0.90/0.05/0.05). With 0 there is no test set.
TEST_FRAC="${TEST_FRAC:-0.05}"
# Paper use-case shots, never trained on: 200729 and the detachment control
# experiments 201081/201085. Applies to new runs only; a resumed run reloads its
# frozen split_shots.json. Comma-separated, so set it here, not via --export.
PIN_VAL="${PIN_VAL:-200729,201081,201085}"
# PIN_TRAIN: shots forced into training. Comma-separated, so set it in the job script,
# not via sbatch --export.
PIN_TRAIN="${PIN_TRAIN:-}"
VAL_WINDOWS="${VAL_WINDOWS:-32}"  # independent of BATCH_SIZE (see --val_windows)
# MASK_ABSENT: leave absent diagnostics out of the CE. An absent diagnostic feeds its codec
# a constant, so it encodes to the same null code in every shot; with every modality
# weighted equally these make up ~38% of the loss terms. Their tokens still enter the
# model as input.
# Needs <cache>/_presence.json to exist already. Otherwise train() builds it on rank 0
# behind a barrier, and on a large job the other ranks wait out the full scan and hit the
# 600 s NCCL timeout. Build it with BUILD_PRESENCE=1, and rebuild it after any cache change.
MASK_ABSENT="${MASK_ABSENT:-1}"
mkdir -p logs "${CACHE_DIR}"

# SPLIT_SEED has to be non-zero: 0 selects the legacy sorted-tail split, which puts the
# highest shot numbers (one whole campaign) in val, and models do not generalise across
# that. 42 is the seed of the production split.
SPLIT_SEED="${SPLIT_SEED:-42}"
SHOT_SAMPLE="${SHOT_SAMPLE:-0}"   # precompute: random-sample this many shots from all data
SHOT_SEED="${SHOT_SEED:-0}"       # seed for SHOT_SAMPLE

# The allocator config has to reach the ranks, so it is exported here (the trainer logs
# the value each rank sees). The ROCm build on Frontier ignores expandable_segments with
# a warning; what keeps memory in bounds there is BATCH_SIZE 1 with accumulation.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
echo "[ignite_dynamics] PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF} \
(NOTE: expandable_segments is a NO-OP on this ROCm build)"

EXTRA=()
# DATA_DIR: read shots from an overlay directory instead of the canonical one (which is
# not group-writable). A re-tokenize then covers exactly the shots in the overlay.
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
[ -n "${PIN_BOTH:-}" ] && EXTRA+=(--pin_both "${PIN_BOTH}")
[ -n "${PIN_VAL:-}" ] && EXTRA+=(--pin_val "${PIN_VAL}")
[ -n "${PIN_TRAIN:-}" ] && EXTRA+=(--pin_train "${PIN_TRAIN}")
[ -n "${VAL_WINDOWS:-}" ] && EXTRA+=(--val_windows "${VAL_WINDOWS}")
[ -n "${DROPOUT:-}" ] && EXTRA+=(--dropout "${DROPOUT}")
[ -n "${ACCUM_STEPS:-}" ] && EXTRA+=(--accum_steps "${ACCUM_STEPS}")
[ "${MASK_ABSENT}" = "1" ] && EXTRA+=(--mask_absent)
[ "${BALANCE_PRESENCE:-0}" = "1" ] && EXTRA+=(--balance_presence)
[ -n "${PRESENCE_PATH:-}" ] && EXTRA+=(--presence_path "${PRESENCE_PATH}")

if [ "${BUILD_PRESENCE:-0}" = "1" ]; then
  # Rebuild <cache>/_presence.json, the absent-diagnostic mask that --mask_absent uses.
  # One rank: it is a serial CPU pass writing one file, so more ranks would redo the scan
  # and race on the output. train() builds the map only if it is missing, so a stale map
  # is reused without warning; rerun this after every cache change.
  echo "[ignite_dynamics] BUILD_PRESENCE host=$(hostname) cache=${CACHE_DIR} \
out=${PRESENCE_PATH:-${CACHE_DIR}/_presence.json}"
  srun -N 1 -n 1 -c "$SLURM_CPUS_PER_TASK" \
       scripts/slurm_frontier/_srun_rank_wrapper.sh \
       -m tokamak_foundation_model.ignite.train_dynamics \
       --cache_dir "${CACHE_DIR}" --build_presence "${EXTRA[@]}"
elif [ "${PATCH_ACTUATORS:-0}" = "1" ]; then
  # Cache repair: rewrite only the actuators with the corrected time base. Codes are left
  # alone, so this is an I/O pass, not a re-precompute. DRY_RUN=1 checks without writing.
  echo "[ignite_dynamics] PATCH_ACTUATORS host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} \
world_size=${SLURM_NTASKS} cache=${CACHE_DIR} dry_run=${DRY_RUN:-0}"
  [ "${DRY_RUN:-0}" = "1" ] && EXTRA+=(--dry_run)
  srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
       --gpus-per-task=1 --gpu-bind="${SRUN_GPU_BIND:-closest}" \
       scripts/slurm_frontier/_srun_rank_wrapper.sh \
       -m tokamak_foundation_model.ignite.train_dynamics \
       --cache_dir "${CACHE_DIR}" --patch_actuators "${EXTRA[@]}"
elif [ "${PRECOMPUTE}" = "1" ]; then
  echo "[ignite_dynamics] PRECOMPUTE host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} \
world_size=${SLURM_NTASKS} cache=${CACHE_DIR} max_shots=${MAX_SHOTS} codec_tmpl=${CODEC_TMPL:-manifest}"
  srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
       --gpus-per-task=1 --gpu-bind="${SRUN_GPU_BIND:-closest}" \
       scripts/slurm_frontier/_srun_rank_wrapper.sh \
       -m tokamak_foundation_model.ignite.train_dynamics \
       --cache_dir "${CACHE_DIR}" \
       --precompute --max_shots "${MAX_SHOTS}" "${EXTRA[@]}"
else
  mkdir -p "${OUT_DIR}"
  # Log every knob, so a run that picked up a wrong value can be told apart in the log.
  echo "[ignite_dynamics] TRAIN host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} world_size=${SLURM_NTASKS} \
cache=${CACHE_DIR} out=${OUT_DIR} depth=${DEPTH} d_model=${D_MODEL} n_heads=${N_HEADS} \
k0=${K0} n_predict=${N_PREDICT} steps=${STEPS} bs=${BATCH_SIZE} accum=${ACCUM_STEPS} \
grad_ckpt=${GRAD_CKPT} lr=${LR} warmup=${WARMUP_STEPS} min_lr_ratio=${MIN_LR_RATIO} \
beta2=${BETA2} wd=${WEIGHT_DECAY} gen_mask_p=${GEN_MASK_P} ss_final=${SS_FINAL_FRAC} \
ss_ramp=${SS_RAMP_STEPS} mask_absent=${MASK_ABSENT} block_forcing=${BLOCK_FORCING:-0} \
forecast_depth=${FORECAST_DEPTH:-1} act_drop_p=${ACT_DROP_P:-0} hist_gap_p=${HIST_GAP_P:-0} hist_gap_max=${HIST_GAP_MAX:-10} \
event_windows=${EVENT_WINDOWS_PER_SHOT:-0} event_weight=${EVENT_WEIGHT:-1} event_frames=${EVENT_FRAMES:-15} \
seed_drop_p=${SEED_DROP_P:-0} seed_column_drop_p=${SEED_COLUMN_DROP_P:-0} \
actuator_noise=${ACTUATOR_NOISE:-0} window_jitter=${WINDOW_JITTER:-0} \
rollout_horizon=${ROLLOUT_HORIZON:-0} rollout_weight=${ROLLOUT_WEIGHT:-1.0} \
rollout_start_jitter=${ROLLOUT_START_JITTER:-0} rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0} \
val_rollout_horizon=${VAL_ROLLOUT_HORIZON:-0} skill_every=${SKILL_EVERY:-0} skill_shots=${SKILL_SHOTS:-32} \
skill_gumbel=${SKILL_GUMBEL:-0} best_metric=${BEST_METRIC:-masked}"
  srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
       --gpus-per-task=1 --gpu-bind="${SRUN_GPU_BIND:-closest}" \
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
       --latent_head "${LATENT_HEAD:-0}" \
       --window_stride "${WINDOW_STRIDE:-1}" \
       --windows_per_shot "${WINDOWS_PER_SHOT}" \
       --window_sample "${WINDOW_SAMPLE:-random}" \
       --val_window_sample "${VAL_WINDOW_SAMPLE:-}" \
       --mod_weight_pow "${MOD_WEIGHT_POW:-0}" \
       --mod_weights_json "${MOD_WEIGHTS_JSON:-}" \
       --live_end_json "${LIVE_END_JSON:-}" \
       --lr_decay_from "${LR_DECAY_FROM:-0}" \
       --wd_exclude_norms "${WD_EXCLUDE_NORMS:-0}" \
       --label_smoothing "${LABEL_SMOOTHING:-0}" \
       --val_windows_per_shot "${VAL_WINDOWS_PER_SHOT:-0}" \
       --grad_clip "${GRAD_CLIP:-0}" \
       --block_forcing "${BLOCK_FORCING:-0}" \
       --forecast_depth "${FORECAST_DEPTH:-1}" \
       --act_drop_p "${ACT_DROP_P:-0}" \
       --hist_gap_p "${HIST_GAP_P:-0}" \
       --hist_gap_max "${HIST_GAP_MAX:-10}" \
       --event_windows_per_shot "${EVENT_WINDOWS_PER_SHOT:-0}" \
       --snapshot_every "${SNAPSHOT_EVERY:-0}" \
       --event_weight "${EVENT_WEIGHT:-1}" \
       --event_frames "${EVENT_FRAMES:-15}" \
       --seed_drop_p "${SEED_DROP_P:-0}" \
       --seed_column_drop_p "${SEED_COLUMN_DROP_P:-0}" \
       --actuator_noise "${ACTUATOR_NOISE:-0}" \
       --window_jitter "${WINDOW_JITTER:-0}" \
       --rollout_horizon "${ROLLOUT_HORIZON:-0}" \
       --rollout_weight "${ROLLOUT_WEIGHT:-1.0}" \
       --rollout_start_jitter "${ROLLOUT_START_JITTER:-0}" \
       --rollout_temperature "${ROLLOUT_TEMPERATURE:-1.0}" \
       --val_rollout_horizon "${VAL_ROLLOUT_HORIZON:-0}" \
       --skill_every "${SKILL_EVERY:-0}" \
       --skill_shots "${SKILL_SHOTS:-32}" \
       --skill_gumbel "${SKILL_GUMBEL:-0}" \
       --skill_codec_tmpl "${SKILL_CODEC_TMPL:-}" \
       --train_cap "${TRAIN_CAP}" \
       --val_n "${VAL_N}" "${EXTRA[@]}"
fi
