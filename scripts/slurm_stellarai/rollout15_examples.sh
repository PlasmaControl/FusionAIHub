#!/bin/bash
#SBATCH --account=nstx-u
#SBATCH --job-name=roll15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =====================================================================================
# One rollout curve point on Princeton StellarAI, on the fixed 12-shot protocol of the
# Frontier arms (3 pinned + 9 held-out shots, all 15 modalities, K0=20, 80 predicted frames,
# T=1.0, dynamics_latest.pt). Shards the 12 shots over 8 GPUs.
#   sbatch --export=ALL,RUN=ss75g_btf scripts/slurm_stellarai/rollout15_examples.sh
# Score (code space; needs no raw data):
#   pixi run -e stellarai python analysis/rollout_npz_score.py --split 190729,190735,190736 \
#       $DATA_ROOT/ignite_prod_v4/runs/ss75g_btf/rollout15_examples_<jobid>
# The H5 signals are not on StellarAI, so figures show the codec reconstruction as ground
# truth (IGNITE_EVAL_RAW_GT=0). For measured-signal figures copy the 12 shots' H5 files and
# set IGNITE_EVAL_RAW_GT=1 IGNITE_DATA_DIR=<their directory>.
# =====================================================================================
set -uo pipefail

# --- paths on StellarAI ------------------------------------------------------------------
export REPO_DIR=${REPO_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub}
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/models}
export RUN=${RUN:-ss75g_btf}
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
# {m} is the modality placeholder; set via a variable because it would close ${...:-...}.
_codec_template="$DATA_ROOT/ignite_codecs_v4/{m}/codec_best.pt"
export CODEC_TMPL=${CODEC_TMPL:-$_codec_template}
export CKPT=${CKPT:-$DATA_ROOT/ignite_prod_v4/runs/$RUN/dynamics_latest.pt}
export OUT_DIR=${OUT_DIR:-$DATA_ROOT/ignite_prod_v4/runs/$RUN/rollout15_examples_${SLURM_JOB_ID:-local}}
# ----------------------------------------------------------------------------------------

export IGNITE_EVAL_RAW_GT=${IGNITE_EVAL_RAW_GT:-0}
# One figure per shot: by default eval renders only the first shot of each rank, and
# the 12 shots over 8 ranks then leave 4 without one.
export RENDER_ALL=${RENDER_ALL:-1}
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}

for path in "$REPO_DIR" "$CACHE_DIR" "$IGNITE_ACT_GLOBAL" "$CKPT" "$DATA_ROOT/ignite_codecs_v4"; do
  [ -e "$path" ] || { echo "[stellarai] missing: $path -- edit the paths at the top" >&2; exit 1; }
done

cd "$REPO_DIR"
exec bash scripts/slurm_frontier/rollout15_examples.sh
