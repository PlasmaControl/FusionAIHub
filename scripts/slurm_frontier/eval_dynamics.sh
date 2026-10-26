#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_dynamics_eval
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 00:30:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
# IGNITE Phase-B (MaskGIT dynamics) EVALUATION — 1-node / 1-GCD.
# Loads a trained dynamics ckpt, seeds K0 real frames, rolls out, decodes GT + pred codes through
# the FROZEN codecs, and renders per-modality GT-vs-pred panels to OUT_DIR. Env overrides:
# CKPT, SHOT, OUT_DIR, TEMPERATURE.
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
# CLUSTER_ENV: environment hook of the cluster (default Frontier/ROCm).
source "${CLUSTER_ENV:-scripts/slurm_frontier/_frontier_common.sh}"

CKPT="${CKPT:-/lustre/orion/fus187/proj-shared/models/ignite_production/runs/prod_d512L8/dynamics_latest.pt}"
OUT_DIR="${OUT_DIR:-eval_runs/ignite_dynamics_eval}"
SHOT="${SHOT:-200729}"            # comma-separated list evaluates many shots (metrics json)
TEMPERATURE="${TEMPERATURE:-1.0}"
CACHE_DIR="${CACHE_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes}"
CODEC_TMPL="${CODEC_TMPL:-}"      # must match the cache's _codec_manifest.json codecs
mkdir -p logs "${OUT_DIR}"

VAL_TAIL="${VAL_TAIL:-0}"         # >0: evaluate the last N shots of the trainer's val split
VAL_N="${VAL_N:-0}"               # val split size (defaults to VAL_TAIL)
SPLIT_SEED="${SPLIT_SEED:-0}"     # MUST match the training run's split seed

EXTRA=()
[ -n "${CODEC_TMPL}" ] && EXTRA+=(--codec_tmpl "${CODEC_TMPL}")
# ACTUATOR COUNTERFACTUAL. Never exposed before, so no trained IGNITE model has ever been
# tested for whether it USES the actuators. Anything but "real" re-runs the identical rollout
# under real actuators with the same RNG and reports divergence_vs_real -- if that is ~0 the
# forecast is unconditional, which no amount of tokenisation or capacity would fix.
[ -n "${ACTUATOR_MODE:-}" ] && EXTRA+=(--actuator_mode "${ACTUATOR_MODE}")
# GUIDANCE=w: classifier-free guidance on the actuators (checkpoint trained with ACT_DROP_P > 0).
[ -n "${GUIDANCE:-}" ] && EXTRA+=(--guidance "${GUIDANCE}")
# GATE=tau, CONSENSUS_N=n, CONSENSUS_Q=q: confidence gate / consensus rollout (no retraining).
[ -n "${GATE:-}" ] && EXTRA+=(--gate "${GATE}")
[ -n "${CONSENSUS_N:-}" ] && EXTRA+=(--consensus_n "${CONSENSUS_N}")
[ -n "${CONSENSUS_Q:-}" ] && EXTRA+=(--consensus_q "${CONSENSUS_Q}")
[ -n "${CONSENSUS_MODE:-}" ] && EXTRA+=(--consensus_mode "${CONSENSUS_MODE}")
[ "${CONSENSUS_INDEPENDENT:-0}" = "1" ] && EXTRA+=(--consensus_independent)
# RENDER_ALL=1: a figure for every shot, not only the first shot of each rank.
[ "${RENDER_ALL:-0}" = "1" ] && EXTRA+=(--render_all)
if [ "${VAL_TAIL}" != "0" ]; then
  # resolved at RUN time inside eval_dynamics (same split_shots as the trainer), so eval
  # jobs can be chained behind training before the cache/split exists.
  EXTRA+=(--val_tail "${VAL_TAIL}" --val_n "${VAL_N}" --split_seed "${SPLIT_SEED}")
fi

echo "[ignite_dynamics_eval] host=$(hostname) ranks=${SLURM_NTASKS:-1} ckpt=${CKPT} shot=${SHOT} out=${OUT_DIR} T=${TEMPERATURE}"
# multi-rank: submit with --ntasks-per-node=8 --gres=gpu:8 to shard shots across 8 GCDs
# --wait=0: ranks own unequal shot counts; StellarAI kills a step 60 s after its first task exits.
srun -N "${SLURM_JOB_NUM_NODES:-1}" -n "${SLURM_NTASKS:-1}" --gpus-per-task=1 --gpu-bind="${SRUN_GPU_BIND:-closest}" --wait=0 \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     -m tokamak_foundation_model.ignite.eval_dynamics \
     --ckpt "${CKPT}" \
     --shot "${SHOT}" \
     --cache_dir "${CACHE_DIR}" \
     --out_dir "${OUT_DIR}" \
     --temperature "${TEMPERATURE}" "${EXTRA[@]}"
