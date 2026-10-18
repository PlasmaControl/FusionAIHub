#!/bin/bash
#SBATCH --account=nstx-u
#SBATCH --job-name=ss75g_live_aug
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =====================================================================================
# ss75g_live_aug on Princeton StellarAI (NVIDIA B200): the ss75g_live recipe (ss75full with
# dataset-wide actuator normalisation and live training windows) plus input augmentation,
# warm-started from an ss75g_live checkpoint. Effective batch 128 = micro-batch 2 x 64 GPUs
# (Frontier: 1 x 64 x 2 accumulation steps).
#
# Augmentation (training only, validation is never augmented):
#   SEED_DROP_P        hide a whole modality from the input in the 20 seed frames
#   SEED_COLUMN_DROP_P otherwise hide a random share of its token positions in every seed frame
#   ACTUATOR_NOISE     Gaussian noise (normalised units) on actuator channels that vary
# Hidden seed tokens are scored, so the model learns to predict a modality missing from the
# seed. Train CE is therefore NOT comparable to ss75g_live; val CE is.
#
# Before the first leg, from the repository root on a login node:
#   pixi install -e stellarai
#   copy dynamics_latest.pt and split_shots.json of the Frontier run
#     /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/ss75g_live/
#   into WARM_FROM (below), then
#   sbatch scripts/slurm_stellarai/launch_ss75g_live_aug.sh
# The first leg copies both files into OUT_DIR and resumes from that step (optimizer and
# scheduler included). Next legs resume from OUT_DIR/dynamics_latest.pt:
#   sbatch -d afterany:<previous job id> scripts/slurm_stellarai/launch_ss75g_live_aug.sh
# On a different node or GPU count, keep NODES x GPUS x BATCH_SIZE x ACCUM_STEPS = 128.
# =====================================================================================
set -uo pipefail

# --- EDIT THESE PATHS -------------------------------------------------------------------
export REPO_DIR=${REPO_DIR:-$HOME/FusionAIHub}                 # the cloned repository
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/$USER/ignite}             # where the archives were unpacked
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
export LIVE_END_JSON=${LIVE_END_JSON:-$DATA_ROOT/tend_all.json}
WARM_FROM=${WARM_FROM:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_live}
export OUT_DIR=${OUT_DIR:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_live_aug}
# ----------------------------------------------------------------------------------------

# The ss75g_live recipe: live training windows, deployment windows for validation.
export WINDOW_SAMPLE=live VAL_WINDOW_SAMPLE=first BLOCK_FORCING=0
export SEED_DROP_P=${SEED_DROP_P:-0.15}
export SEED_COLUMN_DROP_P=${SEED_COLUMN_DROP_P:-0.3}
export ACTUATOR_NOISE=${ACTUATOR_NOISE:-0.1}
export WINDOW_JITTER=0   # live windows already redraw their starts each epoch
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUM_STEPS=${ACCUM_STEPS:-1}
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export NUM_WORKERS=${NUM_WORKERS:-2}

for path in "$REPO_DIR" "$CACHE_DIR" "$IGNITE_ACT_GLOBAL" "$LIVE_END_JSON"; do
  [ -e "$path" ] || { echo "[stellarai] missing: $path -- edit the paths at the top" >&2; exit 1; }
done
# First leg: fork the warm-start run. Later legs find their own checkpoint and skip this.
if [ ! -f "$OUT_DIR/dynamics_latest.pt" ]; then
  for name in dynamics_latest.pt split_shots.json; do
    [ -f "$WARM_FROM/$name" ] || {
      echo "[stellarai] $WARM_FROM/$name missing -- copy it from the Frontier ss75g_live run" >&2; exit 1; }
  done
  mkdir -p "$OUT_DIR"
  cp "$WARM_FROM/split_shots.json" "$OUT_DIR/"
  cp "$WARM_FROM/dynamics_latest.pt" "$OUT_DIR/dynamics_latest.pt.tmp" \
    && mv "$OUT_DIR/dynamics_latest.pt.tmp" "$OUT_DIR/dynamics_latest.pt"
  echo "[stellarai] warm start: $WARM_FROM -> $OUT_DIR"
fi
# The run must reuse the frozen split, so train/val/test match the Frontier arms exactly.
[ -f "$OUT_DIR/split_shots.json" ] || {
  echo "[stellarai] $OUT_DIR/split_shots.json missing -- copy it from the ss75g_live run" >&2; exit 1; }

cd "$REPO_DIR"
exec bash scripts/slurm_frontier/launch_ss75full.sh
