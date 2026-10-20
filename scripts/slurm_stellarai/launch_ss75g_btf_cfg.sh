#!/bin/bash
#SBATCH --account=nstx-u
#SBATCH --job-name=ss75g_btf_cfg
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =====================================================================================
# ss75g_btf_cfg on Princeton StellarAI (NVIDIA B200): ss75g_btf (block teacher forcing,
# dataset-wide actuator normalisation, effective batch 256 = 2 x 32 GPUs x 4) plus
#   ACT_DROP_P      share of samples trained without any actuator term, so a rollout can
#                   use classifier-free guidance on the actuators (eval GUIDANCE=w > 1)
#   FORECAST_DEPTH  history frames are the model's own FORECAST_DEPTH-deep forecasts
#                   (pushforward multi-step training) instead of one-step ones
#   WINDOW_SAMPLE   live: training windows from anywhere the plasma is live; ss75g_btf
#                   drew only starts 0..9, about one sequence per shot
# warm-started from an ss75g_btf checkpoint.
#
# Before the first leg: WARM_FROM holds dynamics_latest.pt and split_shots.json of the
# ss75g_btf run. The first leg copies both into OUT_DIR and resumes from that step
# (optimizer and scheduler included); next legs resume from OUT_DIR/dynamics_latest.pt:
#   sbatch scripts/slurm_stellarai/launch_ss75g_btf_cfg.sh
#   sbatch -d afterany:<previous job id> scripts/slurm_stellarai/launch_ss75g_btf_cfg.sh
# Evaluate with guidance:
#   sbatch --export=ALL,SWEEP="ss75g_btf_cfg:1.0",GUIDANCE=2 scripts/slurm_stellarai/rollout_val_sweep.sh
# =====================================================================================
set -uo pipefail

# --- EDIT THESE PATHS -------------------------------------------------------------------
export REPO_DIR=${REPO_DIR:-$HOME/FusionAIHub}                 # the cloned repository
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/$USER/ignite}             # where the archives were unpacked
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
export LIVE_END_JSON=${LIVE_END_JSON:-$DATA_ROOT/tend_all.json}
WARM_FROM=${WARM_FROM:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_btf}
export OUT_DIR=${OUT_DIR:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_btf_cfg}
# ----------------------------------------------------------------------------------------

export BLOCK_FORCING=1
export ACT_DROP_P=${ACT_DROP_P:-0.1}
export FORECAST_DEPTH=${FORECAST_DEPTH:-2}
export WINDOW_SAMPLE=${WINDOW_SAMPLE:-live} VAL_WINDOW_SAMPLE=${VAL_WINDOW_SAMPLE:-first}
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUM_STEPS=${ACCUM_STEPS:-4}
# Activation checkpointing stays ON: without it block forcing runs out of 178 GB per GPU.
export GRAD_CKPT=${GRAD_CKPT:-1}
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export NUM_WORKERS=${NUM_WORKERS:-2}

for path in "$REPO_DIR" "$CACHE_DIR" "$IGNITE_ACT_GLOBAL" "$LIVE_END_JSON"; do
  [ -e "$path" ] || { echo "[stellarai] missing: $path -- edit the paths at the top" >&2; exit 1; }
done
# First leg: fork the warm-start run. Later legs find their own checkpoint and skip this.
if [ ! -f "$OUT_DIR/dynamics_latest.pt" ]; then
  for name in dynamics_latest.pt split_shots.json; do
    [ -f "$WARM_FROM/$name" ] || { echo "[stellarai] $WARM_FROM/$name missing" >&2; exit 1; }
  done
  mkdir -p "$OUT_DIR"
  cp "$WARM_FROM/split_shots.json" "$OUT_DIR/"
  cp "$WARM_FROM/dynamics_latest.pt" "$OUT_DIR/dynamics_latest.pt.tmp" \
    && mv "$OUT_DIR/dynamics_latest.pt.tmp" "$OUT_DIR/dynamics_latest.pt"
  echo "[stellarai] warm start: $WARM_FROM -> $OUT_DIR"
fi
[ -f "$OUT_DIR/split_shots.json" ] || {
  echo "[stellarai] $OUT_DIR/split_shots.json missing -- copy it from the ss75g_btf run" >&2; exit 1; }

cd "$REPO_DIR"
exec bash scripts/slurm_frontier/launch_ss75full.sh
