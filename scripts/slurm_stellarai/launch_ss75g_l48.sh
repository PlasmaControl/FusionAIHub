#!/bin/bash
#SBATCH --account=nstx-u
#SBATCH --job-name=ss75g_l48
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ss75g_l48 on StellarAI (NVIDIA B200): the ss75g_live recipe (live training windows,
# deployment windows for validation, no block forcing) at depth 48 instead of 16, with
# d_model 1024 and 16 heads unchanged. A 16-layer checkpoint cannot seed 48 layers, so
# this trains from scratch on the frozen split of ss75g_live and is compared with
# ss75g_live at matched steps.
#   LR 6e-4 = 1e-3 x sqrt(16/48), with a 1000-step warmup, since the 16-layer rate is
#   untested this deep from scratch.
# On one B200 node: 72 GB of 180 GB per GPU at micro-batch 2, 5.2 s per step at the start
# (before scheduled sampling ramps in), about 3x the 16-layer step.
# Effective batch 128 = micro-batch 2 x 64 GPUs; keep NODES x 8 x BATCH_SIZE x ACCUM_STEPS.
#   sbatch scripts/slurm_stellarai/launch_ss75g_l48.sh
#   sbatch -d afterany:<previous job id> scripts/slurm_stellarai/launch_ss75g_l48.sh
# The rollout loss is off so that only depth differs from ss75g_live. To add it, fork with
# WARM_FROM=<runs>/ss75g_l48 ROLLOUT_HORIZON=8.
set -uo pipefail

export REPO_DIR=${REPO_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub}
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/models}
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
export LIVE_END_JSON=${LIVE_END_JSON:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/tend_all.json}
SPLIT_FROM=${SPLIT_FROM:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_live}
export OUT_DIR=${OUT_DIR:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_l48}

export DEPTH=${DEPTH:-48}
export LR=${LR:-6e-4} WARMUP_STEPS=${WARMUP_STEPS:-1000}
export WINDOW_SAMPLE=live VAL_WINDOW_SAMPLE=first BLOCK_FORCING=0
export STEPS=${STEPS:-30000}
export SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-2000}
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUM_STEPS=${ACCUM_STEPS:-1}
# ACCUM_STEPS=auto: pick it from the allocated node count so that nodes x 8 x BATCH_SIZE x
# ACCUM_STEPS = EFFECTIVE_BATCH. Lets a flexible-size job (sbatch -N 2,4) resume any leg.
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-128}
if [ "$ACCUM_STEPS" = auto ]; then
  per_step=$(( ${SLURM_JOB_NUM_NODES:?} * 8 * BATCH_SIZE ))
  (( EFFECTIVE_BATCH % per_step == 0 )) || {
    echo "[stellarai] $SLURM_JOB_NUM_NODES nodes x 8 x $BATCH_SIZE does not divide $EFFECTIVE_BATCH" >&2; exit 1; }
  export ACCUM_STEPS=$(( EFFECTIVE_BATCH / per_step ))
  echo "[stellarai] $SLURM_JOB_NUM_NODES nodes -> ACCUM_STEPS=$ACCUM_STEPS (effective batch $EFFECTIVE_BATCH)"
fi
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export NUM_WORKERS=${NUM_WORKERS:-2}

for path in "$REPO_DIR" "$CACHE_DIR" "$IGNITE_ACT_GLOBAL" "$LIVE_END_JSON"; do
  [ -e "$path" ] || { echo "[stellarai] missing: $path -- edit the paths at the top" >&2; exit 1; }
done
# First leg: only the split is copied (no checkpoint fits 48 layers); later legs resume
# from OUT_DIR/dynamics_latest.pt.
if [ ! -f "$OUT_DIR/dynamics_latest.pt" ] && [ ! -f "$OUT_DIR/split_shots.json" ]; then
  [ -f "$SPLIT_FROM/split_shots.json" ] || { echo "[stellarai] $SPLIT_FROM/split_shots.json missing" >&2; exit 1; }
  mkdir -p "$OUT_DIR"
  cp "$SPLIT_FROM/split_shots.json" "$OUT_DIR/"
fi

cd "$REPO_DIR"
exec bash scripts/slurm_frontier/launch_ss75full.sh
