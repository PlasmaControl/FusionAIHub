#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage1_d1024_48L
#SBATCH -o logs/%j_e2e_stage1_d1024_48L.out
#SBATCH -e logs/%j_e2e_stage1_d1024_48L.err
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

# SLURM stages the submit script under /var/spool/slurmd/... so BASH_SOURCE
# is useless for locating the repo. Use SLURM_SUBMIT_DIR — submit from the
# repo root: `cd <repo> && sbatch scripts/slurm_frontier/train_e2e_stage1.sh`.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
# d_model=1024 + n_layers=48 Stage 1 — NEW ARCHITECTURE (2026-06-22):
# full-frequency spectro patch (512,4) + generative flow spectro head +
# resize-conv video + mhr spectrogram + 7-channel tangtv + tin actuator.
# ~1.7B params. From-scratch (incompatible with the old 1.34B d1024 run);
# first job COLD STARTs, successors resume from their own _latest.pt. Distinct
# CHECKPOINT_DIR isolates it from the completed old-arch run.
#
# Env overrides (for the VRAM smoke / chain): CHECKPOINT_DIR, BATCH_SIZE,
# MAX_STEPS, VAL_EVERY, MAX_FILES, SMOKE.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_newarch}"
# Batch 16: batch 32 OOM'd in the TRAINING generative flow-loss (4 velocity
# U-Nets + full-freq head decodes spike ~62 GiB; smoke 4888038/4888168). 16
# fits with margin (~44 GiB est). Effective batch 16×64ranks=1024.
BATCH_SIZE="${BATCH_SIZE:-16}"
# Validation batch — smaller than training: fp32 (--no_amp_val) val + the
# 4 generative spectro heads' Euler sampling spikes ~18 GB above the training
# footprint and OOM'd at batch 32 (smoke 4888038). Training stays at 32.
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-118000}"
VAL_EVERY="${VAL_EVERY:-590}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-100}"
# New-arch spectro knobs. Defaults = the ORIGINAL new-arch (512,4 patch, no
# positional embeddings) so this launcher stays byte-identical for the existing
# chain; override via env for the patch-64 + freq/time-PE retrain, e.g.
#   SPECTRO_PATCH_F=64 SPECTRO_PATCH_T=32 SPEC_FLOW_FREQ_PE_CH=16 \
#   SPEC_FLOW_TIME_PE_CH=8 CHECKPOINT_DIR=<fresh> sbatch ...
SPECTRO_PATCH_F="${SPECTRO_PATCH_F:-512}"
SPECTRO_PATCH_T="${SPECTRO_PATCH_T:-4}"
SPEC_FLOW_FREQ_PE_CH="${SPEC_FLOW_FREQ_PE_CH:-0}"
SPEC_FLOW_TIME_PE_CH="${SPEC_FLOW_TIME_PE_CH:-0}"
MAX_FILES_FLAG=""
[ -n "${MAX_FILES:-}" ] && MAX_FILES_FLAG="--max_files ${MAX_FILES}"
# Explicit shot lists (default empty → glob+random-split, existing chain
# unchanged). Used by the overfit-prediction test to pin an exact tiny train
# set + a disjoint filler val set; YAMLs under data/config/shot_list/.
TRAIN_SHOTS_FLAG=""
[ -n "${TRAIN_SHOTS_YAML:-}" ] && TRAIN_SHOTS_FLAG="--train_shots_yaml ${TRAIN_SHOTS_YAML}"
VAL_SHOTS_FLAG=""
[ -n "${VAL_SHOTS_YAML:-}" ] && VAL_SHOTS_FLAG="--val_shots_yaml ${VAL_SHOTS_YAML}"
# Plan B mode-mask branch (default off → existing chain byte-identical). SPEC_MASK=1
# adds the predicted-mode-mask head; SPEC_MASK_LAMBDA weights its soft-Dice+BCE loss.
SPEC_MASK_FLAG=""
[ -n "${SPEC_MASK:-}" ] && SPEC_MASK_FLAG="--spec_mask"
# Input-conditioning / persistence prior on the mask head (default off).
SPEC_INPUT_COND_FLAG=""
[ -n "${SPEC_INPUT_COND:-}" ] && SPEC_INPUT_COND_FLAG="--spec_input_cond"
# LAZY_OPTIMIZER_LOAD=1 → resume holds optimizer state on CPU until the first
# opt.step() (reclaims batch 16 on the near-VRAM-ceiling resume; default off).
LAZY_OPT_FLAG=""
[ -n "${LAZY_OPTIMIZER_LOAD:-}" ] && LAZY_OPT_FLAG="--lazy_optimizer_load"
# DIAGNOSTIC: SPEC_AUTOENCODE=1 targets the current input window's own codes
# instead of the next window's (forecast) — isolates representation capacity
# from forecast-irreducibility. Spectro code head only; default off.
SPEC_AUTOENCODE_FLAG=""
[ -n "${SPEC_AUTOENCODE:-}" ] && SPEC_AUTOENCODE_FLAG="--spec_autoencode"
# Full-frequency encoder stem on the spectro tokenizer (zero-init, warm-start
# safe): mixes all freq bins BEFORE patching so each token knows its global
# frequency position. The frozen codec encoder has this ON; the backbone
# tokenizer defaults OFF — the suspected mode-collapse root cause.
SPEC_FREQ_STEM_FLAG=""
[ -n "${SPEC_FREQ_STEM:-}" ] && SPEC_FREQ_STEM_FLAG="--spec_freq_stem"
# Warm-init the tokenizer freq_stem from the codec's trained freq_stem (fast path).
[ -n "${SPEC_FREQ_STEM_FROM_CODEC:-}" ] && SPEC_FREQ_STEM_FLAG="${SPEC_FREQ_STEM_FLAG} --spec_freq_stem_from_codec"
# JOINT MaskGIT code head (fixes the independent-head collapse). Requires SPEC_FSQ.
SPEC_MASKGIT_FLAG=""
[ -n "${SPEC_MASKGIT:-}" ] && SPEC_MASKGIT_FLAG="--spec_maskgit \
    --spec_maskgit_dim ${SPEC_MASKGIT_DIM:-512} \
    --spec_maskgit_layers ${SPEC_MASKGIT_LAYERS:-4} \
    --spec_maskgit_heads ${SPEC_MASKGIT_HEADS:-8} \
    --spec_maskgit_decode_steps ${SPEC_MASKGIT_DECODE_STEPS:-10} \
    --spec_maskgit_decode_temp ${SPEC_MASKGIT_DECODE_TEMP:-0.5}"
# Video head: default deterministic resize-conv. VIDEO_GENERATIVE=1 swaps in the
# generative VideoFlowHead (resize-conv mean + rectified-flow residual + spatial
# PE + per-pixel σ) — robust to imperfect backbone tokens (no checkerboard, no
# collapse; eval_runs/video_test/SUMMARY.md). Warm-start safe via INIT_CKPT.
# VIDEO_FSQ=1 → discrete VideoCodeHead: predict a FROZEN adversarial-FSQ video
# codec's codes via class-weighted CE (frozen resize-conv decoder → sharp, no
# checkerboard). Requires VIDEO_FSQ_CODEC_DIR (video_codec_<mod>.pt). Takes
# precedence. Else VIDEO_GENERATIVE=1 → generative flow head; else resize-conv.
if [ -n "${VIDEO_FSQ:-}" ]; then
    : "${VIDEO_FSQ_CODEC_DIR:?VIDEO_FSQ=1 requires VIDEO_FSQ_CODEC_DIR}"
    VIDEO_FLAGS="--video_fsq --video_fsq_codec_dir ${VIDEO_FSQ_CODEC_DIR} \
        --video_code_class_weight ${VIDEO_CODE_CLASS_WEIGHT:-4.0} \
        --video_code_weight_batches ${VIDEO_CODE_WEIGHT_BATCHES:-50}"
elif [ -n "${VIDEO_GENERATIVE:-}" ]; then
    VIDEO_FLAGS="--video_generative --video_sigma_spatial \
        --video_flow_base_ch ${VIDEO_FLOW_BASE_CH:-64} \
        --video_flow_steps ${VIDEO_FLOW_STEPS:-16} \
        --video_flow_pe_ch ${VIDEO_FLOW_PE_CH:-16} \
        --video_flow_lambda ${VIDEO_FLOW_LAMBDA:-1.0}"
else
    VIDEO_FLAGS="--video_resize_conv"
fi
# Spectro head selector. Default = the generative SpectrogramFlowHead (--spec_generative
# + flow flags; unchanged). SPEC_FSQ=1 swaps in the discrete SpectrogramCodeHead: predict
# a FROZEN adversarial-FSQ codec's codes via class-weighted CE (Phase 1b). Requires
# SPEC_FSQ_CODEC_DIR (holding spectro_codec_<mod>.pt); pair with SPECTRO_PATCH_F=64
# SPECTRO_PATCH_T=32 so backbone n_tok(24)==codec.
if [ -n "${SPEC_FSQ:-}" ]; then
    : "${SPEC_FSQ_CODEC_DIR:?SPEC_FSQ=1 requires SPEC_FSQ_CODEC_DIR}"
    SPEC_HEAD_FLAGS="--spec_fsq --spec_fsq_codec_dir ${SPEC_FSQ_CODEC_DIR} \
        --spec_code_class_weight ${SPEC_CODE_CLASS_WEIGHT:-4.0} \
        --spec_code_weight_batches ${SPEC_CODE_WEIGHT_BATCHES:-50} \
        --spec_code_focal_gamma ${SPEC_CODE_FOCAL_GAMMA:-0.0} \
        --spec_code_pred_hidden ${SPEC_CODE_PRED_HIDDEN:-512} \
        --spec_code_pred_layers ${SPEC_CODE_PRED_LAYERS:-2}"
else
    SPEC_HEAD_FLAGS="--spec_generative \
        --spec_flow_steps 6 \
        --spec_flow_lambda ${SPEC_FLOW_LAMBDA:-1.0} \
        --spec_flow_freq_pe_ch ${SPEC_FLOW_FREQ_PE_CH} \
        --spec_flow_time_pe_ch ${SPEC_FLOW_TIME_PE_CH} \
        --spec_struct_lambda ${SPEC_STRUCT_LAMBDA:-0.0} \
        --spec_mask_lambda ${SPEC_MASK_LAMBDA:-0.0} \
        ${SPEC_MASK_FLAG} ${SPEC_INPUT_COND_FLAG}"
fi
# Fast-TS (filterscope/ELM) head selector. Default = continuous FastTimeSeriesHead.
# FASTTS_FSQ=1 swaps in the discrete FastTimeSeriesCodeHead (predict a FROZEN fast-TS
# FSQ codec's codes via class-weighted CE → keeps sharp ELM spikes). Requires
# FASTTS_FSQ_CODEC_DIR (holding fastts_codec.pt).
FASTTS_FLAGS=""
if [ -n "${FASTTS_FSQ:-}" ]; then
    : "${FASTTS_FSQ_CODEC_DIR:?FASTTS_FSQ=1 requires FASTTS_FSQ_CODEC_DIR}"
    FASTTS_FLAGS="--fastts_fsq --fastts_fsq_codec_dir ${FASTTS_FSQ_CODEC_DIR} \
        --fastts_code_class_weight ${FASTTS_CODE_CLASS_WEIGHT:-4.0} \
        --fastts_code_weight_batches ${FASTTS_CODE_WEIGHT_BATCHES:-50}"
fi
# Slow-TS (Thomson/CER/MSE) head selector. Default = continuous SlowTimeSeriesHead.
# SLOW_TS_FSQ=1 swaps in the discrete SlowTimeSeriesCodeHead (per-modality frozen FSQ
# codec, unified discrete world-model). Requires SLOW_TS_FSQ_CODEC_DIR (slowts_codec_<mod>.pt).
SLOWTS_FLAGS=""
if [ -n "${SLOW_TS_FSQ:-}" ]; then
    : "${SLOW_TS_FSQ_CODEC_DIR:?SLOW_TS_FSQ=1 requires SLOW_TS_FSQ_CODEC_DIR}"
    SLOWTS_FLAGS="--slow_ts_fsq --slow_ts_fsq_codec_dir ${SLOW_TS_FSQ_CODEC_DIR} \
        --slow_ts_code_class_weight ${SLOW_TS_CODE_CLASS_WEIGHT:-4.0} \
        --slow_ts_code_weight_batches ${SLOW_TS_CODE_WEIGHT_BATCHES:-50}"
fi

# ALL_SHOTS=1 → train on the WHOLE dataset (skip the video-presence filter). Absent
# video is zero-filled + loss-masked per-sample. Default off = video-present shots only.
# NOTE: with all shots, PRE-WARM the lengths cache for the full file list first (a cold
# 7878-file scan at 64-rank startup blows the NCCL watchdog).
ALLSHOTS_FLAG=""
[ -n "${ALL_SHOTS:-}" ] && ALLSHOTS_FLAG="--no_video_presence_filter"

# ─── ROLLOUT-NATIVE d1024 PRODUCTION (opt-in; default OFF → existing chain
#     byte-identical) ─────────────────────────────────────────────────────
# ROLLOUT_NATIVE=1 turns this launcher into the pre-registered rollout-native
# d1024/48L full-modality FROM-SCRATCH production recipe
# (analysis/mode_audit/EXPERIMENTS.md "ROLLOUT-NATIVE d1024 PRODUCTION",
# 2026-07-18). It APPENDS the rollout loss family + descriptor/β=6 anchor to
# the arch this launcher already wires; it does NOT alter any default path.
#   - K-rollout curriculum FROM K=1 (extension under ONE loss family, no
#     objective switch); CURRICULUM_KS overrides the schedule.
#   - drift-penalty IN from step 0 (asymmetric, weight 0.5 — strike-3's).
#   - UNIFORM per-k weighting (k0-protection OUT — do NOT set K_GE1_* here).
#   - descriptor + β=6 anchor (dist loss), FiLM OFF, filterscopes+slow-TS
#     CONTINUOUS, spectro+video FSQ. FROM-SCRATCH (no INIT_CKPT / RESUME).
#   - per-k loss shares logged always (train_e2e_stage1.py) = the contingency
#     trigger (k0-share collapse as K grows).
# Callers set SPEC_FSQ / VIDEO_FSQ codec dirs + patch (8,16) via env (see the
# smoke recipe below); this block only assembles the ROLLOUT + descriptor part.
ROLLOUT_NATIVE_FLAGS=""
if [ -n "${ROLLOUT_NATIVE:-}" ]; then
    CURRICULUM_KS="${CURRICULUM_KS:-1}"
    BLOCK_STEPS="${BLOCK_STEPS:-5000}"
    TF_ANNEAL_STEPS="${TF_ANNEAL_STEPS:-4000}"
    GRAD_CKPT_EVERY="${GRAD_CKPT_EVERY:-10}"
    DRIFT_PENALTY_WEIGHT="${DRIFT_PENALTY_WEIGHT:-0.5}"
    # β=6 anchor: hold β=6 for the whole run (single hold, long hold_steps).
    ANCHOR_BETA_HOLDS="${ANCHOR_BETA_HOLDS:-6}"
    ANCHOR_BETA_HOLD_STEPS="${ANCHOR_BETA_HOLD_STEPS:-100000}"
    # Dataset future span. The FSQ-VIDEO codec is fixed at n_frames=3 = 1 codec
    # window = 300 tok; the rollout splits video_target into n_per = total_frames/K
    # frames per step and feeds each to the codec, so it REQUIRES n_per==3, i.e.
    # total video frames == 3*K, i.e. dataset_horizon_s == K*chunk_duration_s
    # (the loader emits 3 frames per chunk-window). The trainer's DEFAULT
    # dataset_horizon = maxK*chunk + prediction_horizon adds a +prediction_horizon
    # surplus (e.g. +0.2s = +4 windows) → n_per != 3 → video spatial_pe shape
    # crash. So for the FSQ-video path set ROLLOUT_DATASET_HORIZON_S = K*0.05
    # explicitly. Unset → trainer default (spectro-only runs are unaffected).
    ROLLOUT_DS_HORIZON_FLAG=""
    [ -n "${ROLLOUT_DATASET_HORIZON_S:-}" ] && \
        ROLLOUT_DS_HORIZON_FLAG="--rollout_dataset_horizon_s ${ROLLOUT_DATASET_HORIZON_S}"
    ROLLOUT_NATIVE_FLAGS="\
        --k_rollout \
        --curriculum_Ks ${CURRICULUM_KS} \
        --block_steps ${BLOCK_STEPS} \
        --tf_anneal_steps ${TF_ANNEAL_STEPS} \
        --rollout_grad_checkpoint_every ${GRAD_CKPT_EVERY} \
        --drift_penalty_weight ${DRIFT_PENALTY_WEIGHT} \
        ${ROLLOUT_DS_HORIZON_FLAG} \
        --spec_descriptor \
        --spec_descriptor_anchor \
        --spec_descriptor_loss dist \
        --spec_descriptor_dist_beta ${SPEC_DESC_DIST_BETA:-8.0} \
        --spec_descriptor_weight ${SPEC_DESC_WEIGHT:-6.0} \
        --spec_descriptor_hidden ${SPEC_DESC_HIDDEN:-512} \
        --spec_descriptor_horizons ${SPEC_DESC_HORIZONS:-2,4} \
        --spec_descriptor_tcol ${SPEC_DESC_TCOL:-6} \
        --spec_descriptor_transition_weight ${SPEC_DESC_TRANS_WEIGHT:-5.0} \
        --spec_descriptor_anchor_beta_holds ${ANCHOR_BETA_HOLDS} \
        --spec_descriptor_anchor_beta_hold_steps ${ANCHOR_BETA_HOLD_STEPS}"
fi
mkdir -p logs "${CHECKPOINT_DIR}"

# Distinct from existing Stage 1 (29500) / Stage 1 smoke (29510) / etc.
export MASTER_PORT=29515
source scripts/slurm_frontier/_frontier_common.sh

# Optional PyTorch allocator config passthrough (set AFTER sourcing common so it
# is not clobbered). NOTE: expandable_segments is CONFIRMED UNSUPPORTED on this
# Frontier ROCm build (silently ignored — see project-fullfreq-spectro-patch
# memory); the knife's-edge VRAM is fixed by REDUCING batch (BATCH_SIZE). This
# hook remains only for the untried max_split_size_mb long-shot. Empty default →
# existing chains unchanged.
[ -n "${PYTORCH_ALLOC_CONF:-}" ] && export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"

# Resume from chain successor's _latest.pt if present; otherwise cold start.
# RESUME_CKPT env overrides the source checkpoint (default = this dir's latest):
# lets a resume CANARY load another run's checkpoint while writing its own
# throwaway CHECKPOINT_DIR (so it never clobbers the live chain).
RESUME_FLAG=""
LATEST_CKPT="${RESUME_CKPT:-${CHECKPOINT_DIR}/e2e_stage1_latest.pt}"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[train_e2e_stage1_d1024_48L] resuming from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
else
    echo "[train_e2e_stage1_d1024_48L] cold start (no checkpoint at ${LATEST_CKPT})"
fi

# Warm-start INIT from another run's checkpoint (FIRST job only — once this dir
# has its own _latest.pt, RESUME_FLAG takes over and INIT is ignored). Loads the
# trained backbone + encoder + matching heads; a swapped head architecture
# (e.g. VideoFlowHead) inits fresh (allowed_missing + stale-key strip). Optimizer
# / scheduler / step start fresh (warmup re-runs → gentle, backbone-protecting).
INIT_FLAG=""
if [ -z "${RESUME_FLAG}" ] && [ -n "${INIT_CKPT:-}" ]; then
    echo "[train_e2e_stage1_d1024_48L] warm-start INIT from ${INIT_CKPT}"
    INIT_FLAG="--init_checkpoint ${INIT_CKPT}"
fi

# max_steps = 118_000 = 100 epochs × 1180 steps/epoch (val_every=1180 ≈
# 1 epoch at 8N batch=64). The cosine schedule decays from --lr 5e-4
# down to --min_lr 1e-6 across this window. Changing --max_steps here
# retargets the LR schedule even mid-chain — train_e2e_stage1.py:1188
# re-applies T_max from args after scheduler.load_state_dict().

# Per-node sampler: one line per node per minute with mean GPU busy%,
# host RAM, and mean VRAM%. Launched as a side srun step with --overlap
# so it shares the allocation without stealing GPUs. Cost ~0.1% of one
# CPU/node. Killed when this script exits (walltime or normal end).
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
     --prediction_horizon_s "${PREDICTION_HORIZON_S:-0.05}" \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 1024 \
     --n_layers 48 \
     --n_heads 8 \
     --dropout 0.1 \
     --lr "${LR:-5e-4}" \
     --min_lr 1e-6 \
     --warmup_steps "${WARMUP_STEPS:-4000}" \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size "$BATCH_SIZE" \
     --val_batch_size "$VAL_BATCH_SIZE" \
     --num_workers 6 \
     --max_steps "$MAX_STEPS" \
     --log_every "${LOG_EVERY:-50}" \
     --val_every "$VAL_EVERY" \
     --val_max_batches "$VAL_MAX_BATCHES" \
     --lengths_cache_dir "${LENGTHS_CACHE_DIR:-/lustre/orion/fus187/proj-shared/foundation_model_meta}" \
     --use_video ${USE_VIDEO:-tangtv_lower tangtv_upper} \
     --use_spectro ece co2 bes mhr \
     ${VIDEO_FLAGS} \
     ${SPEC_HEAD_FLAGS} \
     ${FASTTS_FLAGS} \
     ${SLOWTS_FLAGS} \
     ${ALLSHOTS_FLAG} \
     --spectro_patch_f "$SPECTRO_PATCH_F" \
     --spectro_patch_t "$SPECTRO_PATCH_T" \
     --collapse_aware_best \
     --no_amp_val \
     --backbone_grad_checkpoint \
     ${MAX_FILES_FLAG} \
     ${TRAIN_SHOTS_FLAG} \
     ${VAL_SHOTS_FLAG} \
     ${LAZY_OPT_FLAG} \
     ${SPEC_AUTOENCODE_FLAG} \
     ${SPEC_FREQ_STEM_FLAG} \
     ${SPEC_MASKGIT_FLAG} \
     ${ROLLOUT_NATIVE_FLAGS} \
     ${EXTRA_FLAGS:-} \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
