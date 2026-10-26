#!/bin/bash
#SBATCH --account=kolemen
#SBATCH --job-name=ss75g_evt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =====================================================================================
# ss75g_evt on Princeton StellarAI: ss75g_btf_cfg (block teacher forcing, actuator dropout
# for guidance, pushforward depth 2, live windows) plus ACTUATOR-RESPONSE training. The
# scorecard (analysis/checkpoint_scorecard.py) showed no checkpoint lowers T_i reliably
# when the beams go off, so the plasma's response to its actuators is what is missing:
#   EVENT_WINDOWS_PER_SHOT  extra windows per shot and epoch whose predicted frames hold
#                           an NBI (>= 1 MW) or ECH (>= 0.5 MW) power step
#   EVENT_WEIGHT            loss weight of the EVENT_FRAMES frames after each step
# Validation is unchanged, so val CE stays comparable to ss75g_btf_cfg; judge the run on
# the scorecard instead (SNAPSHOT_EVERY steps -> scripts/slurm_stellarai/scorecard_watch.sh).
# Warm-started from the frozen WARM_CKPT of ss75g_btf_cfg (optimizer and scheduler included):
#   sbatch scripts/slurm_stellarai/launch_ss75g_evt.sh
#   sbatch -d afterany:<previous job id> scripts/slurm_stellarai/launch_ss75g_evt.sh
# =====================================================================================
set -uo pipefail

export REPO_DIR=${REPO_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub}
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/models}
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
export LIVE_END_JSON=${LIVE_END_JSON:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/tend_all.json}
WARM_FROM=${WARM_FROM:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_btf_cfg}
WARM_CKPT=${WARM_CKPT:-dynamics_step22400.pt}
export OUT_DIR=${OUT_DIR:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_evt}
export BLOCK_FORCING=1
export ACT_DROP_P=${ACT_DROP_P:-0.1}
export FORECAST_DEPTH=${FORECAST_DEPTH:-2}
export WINDOW_SAMPLE=${WINDOW_SAMPLE:-live} VAL_WINDOW_SAMPLE=${VAL_WINDOW_SAMPLE:-first}
export EVENT_WINDOWS_PER_SHOT=${EVENT_WINDOWS_PER_SHOT:-10}
export EVENT_WEIGHT=${EVENT_WEIGHT:-3}
export EVENT_FRAMES=${EVENT_FRAMES:-15}
export SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-2000}
export STEPS=${STEPS:-34400}
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUM_STEPS=${ACCUM_STEPS:-4}
export GRAD_CKPT=${GRAD_CKPT:-1}
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export NUM_WORKERS=${NUM_WORKERS:-2}

for path in "$REPO_DIR" "$CACHE_DIR" "$IGNITE_ACT_GLOBAL" "$LIVE_END_JSON"; do
  [ -e "$path" ] || { echo "[stellarai] missing: $path -- edit the paths at the top" >&2; exit 1; }
done
if [ ! -f "$OUT_DIR/dynamics_latest.pt" ]; then
  for name in "$WARM_CKPT" split_shots.json; do
    [ -f "$WARM_FROM/$name" ] || { echo "[stellarai] $WARM_FROM/$name missing" >&2; exit 1; }
  done
  mkdir -p "$OUT_DIR"
  cp "$WARM_FROM/split_shots.json" "$OUT_DIR/"
  cp "$WARM_FROM/$WARM_CKPT" "$OUT_DIR/dynamics_latest.pt.tmp" \
    && mv "$OUT_DIR/dynamics_latest.pt.tmp" "$OUT_DIR/dynamics_latest.pt"
  echo "[stellarai] warm start: $WARM_FROM/$WARM_CKPT -> $OUT_DIR"
fi
cd "$REPO_DIR"
exec bash scripts/slurm_frontier/launch_ss75full.sh
