#!/bin/bash
#SBATCH --account=nstx-u
#SBATCH --job-name=ss75g_live_unroll
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =====================================================================================
# ss75g_live_unroll on Princeton StellarAI (NVIDIA B200): the ss75g_live recipe plus the
# ROLLOUT LOSS, warm-started from an ss75g_live checkpoint. After the masked loss, each
# micro-batch unrolls ROLLOUT_HORIZON frames from its real seed: every frame predicted in
# one pass from the seed and the frames already generated, fed back as straight-through
# Gumbel samples so gradients run through the chain, every frame scored against the truth.
# Validation adds rollout_ce (same unroll from k0, fixed noise): the number to compare
# with ss75g_live's dynamics_latest.pt. Effective batch 128 = micro-batch 2 x 64 GPUs.
#
# Before the first leg, from the repository root on a login node:
#   pixi install -e stellarai
#   copy dynamics_latest.pt and split_shots.json of the Frontier run
#     /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/ss75g_live/
#   into WARM_FROM (below), then
#   sbatch scripts/slurm_stellarai/launch_ss75g_live_unroll.sh
# The first leg copies both files into OUT_DIR and resumes from that step. Next legs
# resume from OUT_DIR/dynamics_latest.pt:
#   sbatch -d afterany:<previous job id> scripts/slurm_stellarai/launch_ss75g_live_unroll.sh
# On a different node or GPU count, keep NODES x GPUS x BATCH_SIZE x ACCUM_STEPS = 128.
# =====================================================================================
set -uo pipefail

# --- EDIT THESE PATHS -------------------------------------------------------------------
export REPO_DIR=${REPO_DIR:-$HOME/FusionAIHub}                 # the cloned repository
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/$USER/ignite}             # where the archives were unpacked
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
export LIVE_END_JSON=${LIVE_END_JSON:-$DATA_ROOT/tend_all.json}
export WARM_FROM=${WARM_FROM:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_live}
export OUT_DIR=${OUT_DIR:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_live_unroll}
# ----------------------------------------------------------------------------------------

# The ss75g_live recipe: live training windows, deployment windows for validation.
export WINDOW_SAMPLE=live VAL_WINDOW_SAMPLE=first BLOCK_FORCING=0
export ROLLOUT_HORIZON=${ROLLOUT_HORIZON:-8}
export ROLLOUT_WEIGHT=${ROLLOUT_WEIGHT:-1.0}
export ROLLOUT_START_JITTER=${ROLLOUT_START_JITTER:-0}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUM_STEPS=${ACCUM_STEPS:-1}
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export NUM_WORKERS=${NUM_WORKERS:-2}

for path in "$REPO_DIR" "$CACHE_DIR" "$IGNITE_ACT_GLOBAL" "$LIVE_END_JSON"; do
  [ -e "$path" ] || { echo "[stellarai] missing: $path -- edit the paths at the top" >&2; exit 1; }
done
# launch_ss75full.sh forks WARM_FROM into OUT_DIR on the first leg (dynamics_latest.pt and
# the frozen split_shots.json, so train/val/test match the Frontier arms exactly).
[ -f "$OUT_DIR/dynamics_latest.pt" ] || [ -f "$WARM_FROM/split_shots.json" ] || {
  echo "[stellarai] $WARM_FROM/split_shots.json missing -- copy it from the ss75g_live run" >&2; exit 1; }

cd "$REPO_DIR"
exec bash scripts/slurm_frontier/launch_ss75full.sh
