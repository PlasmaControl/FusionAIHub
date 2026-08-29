#!/bin/bash
# Frontier DDP launcher: Stage-2 K-ANNEAL (drift intervention) on the g3fix β=6 model.
#
# Extends train_e2e_stage1.py's OPT-IN --k_rollout mode. ONE CHANGE vs the g3fix
# β=6 recipe: the K-rollout extension (curriculum 10→20→40→80). Every other knob
# — architecture, losses, lr recipe, β PINNED at 6 — is the g3fix operating point,
# reconstructed faithfully from the checkpoint args (analysis/mode_audit/EXPERIMENTS.md
# "STAGE-2 K-ANNEAL — PRE-REGISTERED 2026-07-16"). Warm-starts the β=6 landing.
#
# Usage:
#   SMOKE=1 sbatch scripts/slurm_frontier/train_e2e_stage1_kanneal.sh     # real-model warm-start smoke
#   sbatch -N 8 -t 2:00:00 scripts/slurm_frontier/train_e2e_stage1_kanneal.sh   # production (chain + multi-partition after)
#
# Env overrides: SMOKE, MAX_STEPS, BATCH_SIZE, NUM_WORKERS, CURRICULUM_KS, BLOCK_STEPS,
#   TF_ANNEAL_STEPS, GRAD_CKPT_EVERY, CHECKPOINT_DIR, INIT_CKPT, LENGTHS_CACHE_DIR, MASTER_PORT,
#   FEEDBACK_NORMALIZE (=1 → append --feedback_normalize; default off = byte-identical),
#   ROLLOUT_DATASET_HORIZON_S (Lever #1: per-BLOCK dataset future span — set to the CURRENT
#     block's reach = K*chunk + pred_horizon = K*0.05 + 0.2 [K=10→0.7, K=20→1.2, K=40→2.2, K=80→4.2].
#     DEFAULT UNSET → flag omitted → trainer falls back to max(curriculum_Ks) span [byte-identical
#     to non-B runs]. When set, PAIR it with a horizon-specific LENGTHS_CACHE_DIR whose
#     lengths_e2e_stage1_{train,val}.pt were PRE-BUILT OFFLINE at this horizon [the lengths scan
#     is horizon-specific; a cold 7878-shot scan inside a multi-rank job trips NCCL's watchdog →
#     64-rank crash — see scripts/data_preparation/prebuild_lengths_cache.py]).
#
#SBATCH -A fus187
#SBATCH -J e2e_kanneal
#SBATCH -o logs/%j_e2e_kanneal.out
#SBATCH -e logs/%j_e2e_kanneal.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -uo pipefail

# NEVER default to nchen. This run lives entirely in ps9551's tree.
PROJECT_DIR="${PROJECT_DIR:-/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub}"
cd "$PROJECT_DIR"
mkdir -p logs

export MASTER_PORT="${MASTER_PORT:-29540}"
# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

NODES="${SLURM_JOB_NUM_NODES:-1}"
TOTAL_RANKS="${SLURM_NTASKS:-$((NODES * 1))}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-7}"

# ─── Fixed g3fix paths ───────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"
STATS_PATH="${STATS_PATH:-/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt}"
INIT_CKPT="${INIT_CKPT:-/lustre/orion/fus187/proj-shared/models/e2e_g3fix_anneal/e2e_stage1_beta6.0_step3000.pt}"

# ─── K-anneal curriculum ─────────────────────────────────────────────────
CURRICULUM_KS="${CURRICULUM_KS:-10,20,40,80}"
BLOCK_STEPS="${BLOCK_STEPS:-5000}"
TF_ANNEAL_STEPS="${TF_ANNEAL_STEPS:-4000}"     # scheduled sampling: GT-fed → free by step 4000 (within block 0)
GRAD_CKPT_EVERY="${GRAD_CKPT_EVERY:-10}"       # grad-checkpoint the rollout (K≥40 at d512 needs it)

# ─── SMOKE overrides (real-model warm-start + one-rollout-step gate) ──────
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_STEPS="${MAX_STEPS:-4}"
    MAX_FILES="${MAX_FILES:-8}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    LOG_EVERY="${LOG_EVERY:-1}"
    VAL_EVERY="${VAL_EVERY:-1000}"          # skip val in smoke
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-1}"
    WARMUP_STEPS="${WARMUP_STEPS:-1}"
    CURRICULUM_KS="${CURRICULUM_KS_SMOKE:-10}"
    BLOCK_STEPS="${BLOCK_STEPS_SMOKE:-2}"
    # smoke default: pure free rollout (argmax feedback + anchor). Override with
    # TF_ANNEAL_STEPS_SMOKE>0 to exercise the teacher-forcing path (p_tf~1 early).
    TF_ANNEAL_STEPS="${TF_ANNEAL_STEPS_SMOKE:-0}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre/orion/fus187/proj-shared/models/e2e_g3fix_kanneal_smoke}"
    # LOCAL lengths cache — a small-file run must NOT overwrite the shared
    # production cache (foundation_model_meta/lengths_e2e_stage1_train.pt).
    LENGTHS_CACHE_DIR="${LENGTHS_CACHE_DIR:-$CHECKPOINT_DIR}"
    BANNER="[KANNEAL-SMOKE] "
else
    MAX_STEPS="${MAX_STEPS:-20000}"         # 4 blocks × 5000
    BATCH_SIZE="${BATCH_SIZE:-16}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    LOG_EVERY="${LOG_EVERY:-50}"
    VAL_EVERY="${VAL_EVERY:-500}"
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-20}"
    WARMUP_STEPS="${WARMUP_STEPS:-300}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre/orion/fus187/proj-shared/models/e2e_g3fix_kanneal}"
    # Production reuses the shared cache (else ~87-min cold recompute → NCCL crash).
    LENGTHS_CACHE_DIR="${LENGTHS_CACHE_DIR:-/lustre/orion/fus187/proj-shared/foundation_model_meta}"
    BANNER=""
fi
mkdir -p "$CHECKPOINT_DIR" "$LENGTHS_CACHE_DIR"

MAX_FILES_FLAG=""
[ -n "${MAX_FILES:-}" ] && MAX_FILES_FLAG="--max_files $MAX_FILES"

# OPT-IN post-tokenizer feedback-token renorm to the step-0 input band (option-2)
# for the ece proj-conv NaN fix. DEFAULT OFF (env unset) → flag NOT appended →
# byte-identical to the running production chain. Set FEEDBACK_NORMALIZE=1 to
# scale each code-path modality's feedback token slice per-sample DOWN so its
# absmax never exceeds the step-0 input-window band (the fixed near-DC proj
# filter otherwise saturates on the codec-decoded broadband floor → bf16 NaN,
# hottest in the teacher-forcing GT-code-decode path).
FEEDBACK_NORMALIZE_FLAG=""
[ "${FEEDBACK_NORMALIZE:-0}" = "1" ] && FEEDBACK_NORMALIZE_FLAG="--feedback_normalize"

# Lever #1 (per-block dataset horizon). ONLY appended when ROLLOUT_DATASET_HORIZON_S
# is set → non-B runs omit the flag entirely and the trainer's default
# (max(curriculum_Ks)*chunk + pred) applies → byte-identical.
ROLLOUT_DATASET_HORIZON_FLAG=""
[ -n "${ROLLOUT_DATASET_HORIZON_S:-}" ] && \
    ROLLOUT_DATASET_HORIZON_FLAG="--rollout_dataset_horizon_s $ROLLOUT_DATASET_HORIZON_S"

# Lever #1 companion: block-segmented curriculum. STOP_AT_STEP breaks the loop
# at the block boundary while MAX_STEPS stays at the full 20000 → the LR cosine
# T_max is unchanged (one-cosine recipe preserved), and the next block resumes
# with a bumped ROLLOUT_DATASET_HORIZON_S + its own lengths cache. Unset → omitted
# → byte-identical (loop bounded only by MAX_STEPS).
STOP_AT_STEP_FLAG=""
[ -n "${STOP_AT_STEP:-}" ] && STOP_AT_STEP_FLAG="--stop_at_step $STOP_AT_STEP"

# ─── STRIKE-3 (K=10-gate failure fix) — two OPT-IN loss levers ────────────
# Each flag is appended ONLY when its env var is set → non-strike-3 runs omit
# them entirely and the trainer's identity defaults apply → byte-identical.
#   DRIFT_PENALTY_WEIGHT      Lever 1: asymmetric relu(pred_drift-gt_drift) weight
#   K_GE1_WEIGHT              Lever 2: constant k>=1 loss multiplier (k=0 pinned 1.0)
#   K_GE1_WEIGHT_ANNEAL_STEPS Lever 2: linear anneal-up of the k>=1 weight -> 1.0
#   K_GE1_WEIGHT_START        Lever 2: anneal start value for the k>=1 weight
DRIFT_PENALTY_WEIGHT_FLAG=""
[ -n "${DRIFT_PENALTY_WEIGHT:-}" ] && \
    DRIFT_PENALTY_WEIGHT_FLAG="--drift_penalty_weight $DRIFT_PENALTY_WEIGHT"
K_GE1_WEIGHT_FLAG=""
[ -n "${K_GE1_WEIGHT:-}" ] && K_GE1_WEIGHT_FLAG="--k_ge1_weight $K_GE1_WEIGHT"
K_GE1_WEIGHT_ANNEAL_STEPS_FLAG=""
[ -n "${K_GE1_WEIGHT_ANNEAL_STEPS:-}" ] && \
    K_GE1_WEIGHT_ANNEAL_STEPS_FLAG="--k_ge1_weight_anneal_steps $K_GE1_WEIGHT_ANNEAL_STEPS"
K_GE1_WEIGHT_START_FLAG=""
[ -n "${K_GE1_WEIGHT_START:-}" ] && \
    K_GE1_WEIGHT_START_FLAG="--k_ge1_weight_start $K_GE1_WEIGHT_START"

# Auto-resume (chain). --resume_checkpoint overrides --init_checkpoint in the trainer.
LATEST="$CHECKPOINT_DIR/e2e_stage1_latest.pt"
INIT_OR_RESUME="--init_checkpoint $INIT_CKPT"
if [ -f "$LATEST" ]; then
    INIT_OR_RESUME="--resume_checkpoint $LATEST"
    echo "${BANNER}[kanneal] auto-resume from $LATEST"
else
    echo "${BANNER}[kanneal] warm-start from $INIT_CKPT"
fi

echo "${BANNER}[kanneal] nodes=$NODES ranks=$TOTAL_RANKS batch=$BATCH_SIZE steps=$MAX_STEPS K=$CURRICULUM_KS block=$BLOCK_STEPS tf_anneal=$TF_ANNEAL_STEPS gc=$GRAD_CKPT_EVERY ds_horizon=${ROLLOUT_DATASET_HORIZON_S:-<default max-K>} lengths_cache=$LENGTHS_CACHE_DIR"
echo "${BANNER}[kanneal] STRIKE-3 levers: drift_penalty_weight=${DRIFT_PENALTY_WEIGHT:-<off>} k_ge1_weight=${K_GE1_WEIGHT:-<off>} k_ge1_weight_start=${K_GE1_WEIGHT_START:-<off>} k_ge1_weight_anneal_steps=${K_GE1_WEIGHT_ANNEAL_STEPS:-<off>}"

srun --overlap -N "$NODES" -n "$TOTAL_RANKS" -c "$CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage1.py \
     $INIT_OR_RESUME $MAX_FILES_FLAG \
     --data_dir "$DATA_DIR" \
     --stats_path "$STATS_PATH" \
     --checkpoint_dir "$CHECKPOINT_DIR" \
     --lengths_cache_dir "$LENGTHS_CACHE_DIR" \
     --val_fraction 0.1 \
     --seed 42 \
     --num_workers "$NUM_WORKERS" \
     --batch_size "$BATCH_SIZE" \
     --max_steps "$MAX_STEPS" \
     --log_every "$LOG_EVERY" \
     --val_every "$VAL_EVERY" \
     --val_max_batches "$VAL_MAX_BATCHES" \
     --warmup_steps "$WARMUP_STEPS" \
     --k_rollout \
     --curriculum_Ks "$CURRICULUM_KS" \
     --block_steps "$BLOCK_STEPS" \
     --tf_anneal_steps "$TF_ANNEAL_STEPS" \
     --rollout_grad_checkpoint_every "$GRAD_CKPT_EVERY" \
     $FEEDBACK_NORMALIZE_FLAG \
     $ROLLOUT_DATASET_HORIZON_FLAG \
     $STOP_AT_STEP_FLAG \
     $DRIFT_PENALTY_WEIGHT_FLAG \
     $K_GE1_WEIGHT_FLAG \
     $K_GE1_WEIGHT_ANNEAL_STEPS_FLAG \
     $K_GE1_WEIGHT_START_FLAG \
     --spec_descriptor_anchor_beta_holds 6 \
     --spec_descriptor_anchor_beta_hold_steps 100000 \
     `cat scripts/slurm_frontier/_kanneal_g3fix_flags.txt`
