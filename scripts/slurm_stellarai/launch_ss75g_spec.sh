#!/bin/bash
#SBATCH --account=kolemen
#SBATCH --job-name=ss75g_spec
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --exclude=sai-n01g4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ss75g_spec on StellarAI (NVIDIA B200): ss75g_live_unroll with the spectrograms
# up-weighted in the training loss, warm-started from the latest ss75g_live_unroll
# checkpoint. The masked and rollout losses average the token cross-entropy within each
# modality and then weight the 15 modalities equally, so ece/bes/mhr/co2/mirnov (960 of
# ~1017 tokens per frame) get only 5/15 of the gradient. SPEC_WEIGHT=3 gives them
# 15/25 = 60%. Validation always uses equal weights, so val CE stays comparable to
# ss75g_live_unroll.
# Effective batch 128 = micro-batch 2 x 32 GPUs x 2 accumulation steps.
#   sbatch scripts/slurm_stellarai/launch_ss75g_spec.sh                                  # spec3_h8
#   sbatch -J ss75g_spec_h16 --export=ALL,ROLLOUT_HORIZON=16 scripts/slurm_stellarai/launch_ss75g_spec.sh
#   sbatch -d afterany:<previous job id> <same command>                                 # next leg
export REPO_DIR=${REPO_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub}
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/models}
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
export LIVE_END_JSON=${LIVE_END_JSON:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/tend_all.json}
export WARM_FROM=${WARM_FROM:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_live_unroll}
# Loss weight of each spectrogram modality; every other modality keeps weight 1.
SPEC_WEIGHT=${SPEC_WEIGHT:-3}
# Frames of the rollout loss (ss75g_live_unroll: 8).
export ROLLOUT_HORIZON=${ROLLOUT_HORIZON:-8}
export OUT_DIR=${OUT_DIR:-$DATA_ROOT/ignite_prod_v4/runs/ss75g_spec${SPEC_WEIGHT}_h${ROLLOUT_HORIZON}}

# The ss75g_live_unroll recipe: live training windows, deployment windows for validation.
export WINDOW_SAMPLE=live VAL_WINDOW_SAMPLE=first BLOCK_FORCING=0
export ROLLOUT_WEIGHT=${ROLLOUT_WEIGHT:-1.0}
export ROLLOUT_START_JITTER=${ROLLOUT_START_JITTER:-0}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
# Frozen snapshots (and the in-training skill check) every 1000 steps: a 4-node leg makes
# about 4000 steps at horizon 8, half that at horizon 16.
export SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-1000}
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUM_STEPS=${ACCUM_STEPS:-2}
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export NUM_WORKERS=${NUM_WORKERS:-2}

for path in "$REPO_DIR" "$CACHE_DIR" "$IGNITE_ACT_GLOBAL" "$LIVE_END_JSON"; do
  [ -e "$path" ] || { echo "[stellarai] missing: $path -- edit the paths at the top" >&2; exit 1; }
done
# The weights go in a file because sbatch --export would split inline JSON on its commas.
mkdir -p "$OUT_DIR"
export MOD_WEIGHTS_JSON=$OUT_DIR/mod_weights.json
printf '{"ece": %s, "bes": %s, "mhr": %s, "co2": %s, "mirnov": %s}\n' \
  "$SPEC_WEIGHT" "$SPEC_WEIGHT" "$SPEC_WEIGHT" "$SPEC_WEIGHT" "$SPEC_WEIGHT" > "$MOD_WEIGHTS_JSON"

cd "$REPO_DIR"
exec bash scripts/slurm_frontier/launch_ss75full.sh
