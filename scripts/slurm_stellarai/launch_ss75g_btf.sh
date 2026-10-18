#!/bin/bash
#SBATCH --account=nstx-u
#SBATCH --job-name=ss75g_btf
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =====================================================================================
# ss75g_btf on Princeton StellarAI (NVIDIA B200): the ss75full recipe with dataset-wide
# actuator normalisation (the leak fix) and BLOCK TEACHER FORCING. Effective batch 128 =
# micro-batch 2 x 64 GPUs (Frontier: 1 x 64 x 2 accumulation steps), as on the Frontier arms.
#
# Before the first leg, from the repository root on a login node:
#   pixi install -e stellarai
#   mkdir -p logs "${OUT_DIR}" && cp <ss75g run>/split_shots.json "${OUT_DIR}/"
#   sbatch scripts/slurm_stellarai/launch_ss75g_btf.sh
# Next legs resume from OUT_DIR/dynamics_latest.pt:
#   sbatch -d afterany:<previous job id> scripts/slurm_stellarai/launch_ss75g_btf.sh
# On a different node or GPU count, keep NODES x GPUS x ACCUM_STEPS = 128.
# =====================================================================================
set -uo pipefail

# --- EDIT THESE PATHS -------------------------------------------------------------------
export REPO_DIR=${REPO_DIR:-$HOME/FusionAIHub}                 # the cloned repository
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/$USER/ignite}             # where the archives were unpacked
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
export LIVE_END_JSON=${LIVE_END_JSON:-$DATA_ROOT/tend_all.json}
export OUT_DIR=${OUT_DIR:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_btf}
# ----------------------------------------------------------------------------------------

export BLOCK_FORCING=1
# B200s have room for micro-batch 2 (about 78 GB of 179 GB); with no accumulation the
# effective batch stays 64 GPUs x 2 = 128, the recipe of the Frontier arms.
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUM_STEPS=${ACCUM_STEPS:-1}
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export NUM_WORKERS=${NUM_WORKERS:-2}

for path in "$REPO_DIR" "$CACHE_DIR" "$IGNITE_ACT_GLOBAL" "$LIVE_END_JSON"; do
  [ -e "$path" ] || { echo "[stellarai] missing: $path -- edit the paths at the top" >&2; exit 1; }
done
# The run must reuse the frozen split, so train/val/test match the Frontier arms exactly.
[ -f "$OUT_DIR/split_shots.json" ] || {
  echo "[stellarai] $OUT_DIR/split_shots.json missing -- copy it from the ss75g run" >&2; exit 1; }

cd "$REPO_DIR"
exec bash scripts/slurm_frontier/launch_ss75full.sh
