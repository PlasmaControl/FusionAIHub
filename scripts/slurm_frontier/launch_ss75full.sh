#!/bin/bash
#SBATCH -A fus187
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
# =====================================================================================
# ss75full -- THE RECIPE THAT BEATS PERSISTENCE ON HELD-OUT SHOTS (validated 2026-09-24).
#
# d1024xL16 MaskGIT over the prod_v4 frame-code cache (15 modalities, 1209 tok/frame), ALL
# 7877 training shots, scheduled sampling ramped to 0.75 over 2000 steps, dropout 0.1,
# weight decay 0.01, post-shot masking from eval_runs/tend_all.json, the 3 standing example
# shots pinned into BOTH train and val. Rollout skill on the 12-shot protocol
# (analysis/rollout_npz_score.py, static pairs screened): near-1 0.216 -> 0.283 -> 0.310 ->
# 0.339 vs persistence 0.315 at ep 2.3 / 2.9 / 4.2 / 6.8, motion ratio constant 0.72, timing
# corr rising 0.11 -> 0.26. The gain came AFTER masked val CE turned (min ep 4.9): with this
# recipe NEVER select a checkpoint on masked/gen CE -- evaluate dynamics_latest.pt on rollout.
#
# Launch (8 nodes, 12-h legs chained with afterany; TimeLimit/Partition set after sbatch):
#   J=$(sbatch --parsable -J ss75full_l1 -N 8 --ntasks-per-node=8 --cpus-per-task=7 \
#          --gres=gpu:8 -t 02:00:00 scripts/slurm_frontier/launch_ss75full.sh)
#   scontrol update JobId=$J TimeLimit=12:00:00 Partition=extended,batch,g1
#   next leg: add  -d afterany:$J
# Any variable below can be overridden with --export=ALL,VAR=value (NO commas in values --
# sbatch --export splits on commas; comma-bearing pins therefore live in this file).
# Rollout curve point: sbatch -N 1 --ntasks-per-node=8 --gres=gpu:8 -t 02:00:00 -p batch \
#   --export=ALL,RUN=ss75full scripts/slurm_frontier/rollout15_examples.sh
# =====================================================================================
set -uo pipefail
# REPO_DIR: repository root (other clusters set it, see scripts/slurm_stellarai/).
cd "${REPO_DIR:-/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub}"
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-7}

# --- architecture (persisted in the checkpoint; eval infers them from weights) ---
export D_MODEL=${D_MODEL:-1024} DEPTH=${DEPTH:-16} N_HEADS=${N_HEADS:-16}
export K0=${K0:-20} N_PREDICT=${N_PREDICT:-80}
# --- optimisation ---
export BATCH_SIZE=${BATCH_SIZE:-1} ACCUM_STEPS=${ACCUM_STEPS:-2} GRAD_CKPT=${GRAD_CKPT:-1}
export LR=${LR:-1e-3} WARMUP_STEPS=${WARMUP_STEPS:-100} MIN_LR_RATIO=${MIN_LR_RATIO:-1.0}
export LR_DECAY_FROM=${LR_DECAY_FROM:-999999} STEPS=${STEPS:-30000} BETA2=${BETA2:-0.9}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01} WD_EXCLUDE_NORMS=${WD_EXCLUDE_NORMS:-1}
export DROPOUT=${DROPOUT:-0.1} GRAD_CLIP=${GRAD_CLIP:-1.0} LABEL_SMOOTHING=${LABEL_SMOOTHING:-0}
export MOD_WEIGHT_POW=${MOD_WEIGHT_POW:-0} GEN_MASK_P=${GEN_MASK_P:-0.0}
# --- THE recipe axis: scheduled sampling (the infiller recipe had SS_FINAL_FRAC=0 and rolled out flat) ---
export SS_FINAL_FRAC=${SS_FINAL_FRAC:-0.75} SS_RAMP_STEPS=${SS_RAMP_STEPS:-2000}
# BLOCK_FORCING=1: every frame predicted from earlier frames only; SS then uses the model's own forecasts.
export BLOCK_FORCING=${BLOCK_FORCING:-0}
# --- data / windows / split ---
export CACHE_DIR=${CACHE_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes}
export LIVE_END_JSON=${LIVE_END_JSON:-/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/tend_all.json}
# ACTUATOR NORMALISATION. Empty = per-shot z-score with WHOLE-SHOT statistics (the future leak; what
# ss75full trained on). Set to the v4 side cache for dataset-wide stats computed from TRAIN shots:
#   IGNITE_ACT_GLOBAL=/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/act_global
# Persisted in the checkpoint as cfg_act_global; eval_dynamics refuses a mismatch.
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-}
export WINDOWS_PER_SHOT=${WINDOWS_PER_SHOT:-10} WINDOW_SAMPLE=${WINDOW_SAMPLE:-first}
# WINDOW_SAMPLE=live (2026-10-13): train windows drawn each epoch from anywhere the plasma is live
# (>=50% of the window before t_end) instead of starts 0..9 only; validation is forced to 'first'.
export VAL_WINDOW_SAMPLE=${VAL_WINDOW_SAMPLE:-}
export VAL_WINDOWS_PER_SHOT=${VAL_WINDOWS_PER_SHOT:-10} VAL_WINDOWS=${VAL_WINDOWS:-256}
export SPLIT_SEED=${SPLIT_SEED:-42} TEST_FRAC=${TEST_FRAC:-0.05} VAL_N=${VAL_N:-438}
export TRAIN_CAP=${TRAIN_CAP:-0} MASK_ABSENT=${MASK_ABSENT:-0} NUM_WORKERS=${NUM_WORKERS:-2}
# --- checkpointing: BEST_METRIC is kept for the record only; rollout uses dynamics_latest.pt ---
export BEST_METRIC=${BEST_METRIC:-masked} CKPT_EVERY=${CKPT_EVERY:-200}
export OUT_DIR=${OUT_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/ss75full}
# --- the standing example shots: in BOTH train and val (--pin_val would EXCLUDE them from training) ---
export PIN_BOTH="190729,190735,190736"
export PIN_TRAIN="190726,190727,190728,190730,190731,190732,190733,190734,190737,190738,190739"

exec bash scripts/slurm_frontier/train_dynamics.sh
