#!/bin/bash
#SBATCH --account=nstx-u
#SBATCH --job-name=roll_val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =====================================================================================
# Held-out rollout sweep on StellarAI: the last VAL_TAIL shots of the frozen validation
# split (not the 12-shot protocol, whose 9 held-out shots are too few to separate +-0.01),
# for each RUN:TEMPERATURE[:GUIDANCE] in SWEEP, one after another on one node (8 GPUs, ~10 min per
# shot and GPU). Each point is scored right away, in code space (rollout_npz_score.py)
# and for hallucinated / missed spectrogram activity (rollout_hallucination.py), so a
# point that finishes is never lost to the time limit.
#   sbatch scripts/slurm_stellarai/rollout_val_sweep.sh
#   sbatch --export=ALL,SWEEP="ss75g_btf:1.0",VAL_TAIL=32 scripts/slurm_stellarai/rollout_val_sweep.sh
#   sbatch --export=ALL,SWEEP="ss75g_btf_cfg:1.0:1 ss75g_btf_cfg:1.0:2" scripts/slurm_stellarai/rollout_val_sweep.sh
# GUIDANCE is classifier-free guidance on the actuators; it needs a checkpoint trained with
# ACT_DROP_P > 0 (launch_ss75g_btf_cfg.sh). Output: runs/<RUN>/rollout_val<N>_T<T>[_g<G>]_<jobid>/
# =====================================================================================
set -uo pipefail

export REPO_DIR=${REPO_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub}
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/models}
SWEEP=${SWEEP:-"ss75g_live_aug:1.0 ss75g_live_aug:0.5 ss75g_live_aug:0.7 ss75g_btf:1.0"}
export VAL_TAIL=${VAL_TAIL:-32} VAL_N=${VAL_N:-438} SPLIT_SEED=${SPLIT_SEED:-42}
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
_codec_template="$DATA_ROOT/ignite_codecs_v4/{m}/codec_best.pt"
export CODEC_TMPL=${CODEC_TMPL:-$_codec_template}
export IGNITE_EVAL_MODALITIES="bes,cer_rot,cer_ti,co2,ece,filterscopes,mhr,mirnov,mse,tangtv_lower,tangtv_upper,ts_core_density,ts_core_temp,ts_tangential_density,ts_tangential_temp"
export IGNITE_EVAL_RAW_GT=${IGNITE_EVAL_RAW_GT:-0}
export RENDER_ALL=${RENDER_ALL:-0}   # one figure per rank is enough for a sweep
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
PINNED="190729,190735,190736"

cd "$REPO_DIR"
# The scorers run in this shell: activate the frozen env here (compute nodes have no
# network, so `pixi run` cannot re-solve the environment there).
source "$CLUSTER_ENV"
for point in $SWEEP; do
  IFS=: read -r RUN T G <<< "$point"
  export CKPT=$DATA_ROOT/ignite_prod_v4/runs/$RUN/dynamics_latest.pt TEMPERATURE=$T GUIDANCE=${G:-1}
  export OUT_DIR=$DATA_ROOT/ignite_prod_v4/runs/$RUN/rollout_val${VAL_TAIL}_T${T}${G:+_g$G}_${SLURM_JOB_ID:-local}
  echo "[sweep] $(date '+%H:%M') $RUN T=$T guidance=${G:-1} -> $OUT_DIR"
  bash scripts/slurm_frontier/eval_dynamics.sh
  python analysis/rollout_npz_score.py --split "$PINNED" "$OUT_DIR"
  srun -N1 -n1 --gpus-per-task=1 --gpu-bind=none \
    python analysis/rollout_hallucination.py --split "$PINNED" \
    --cache_dir "$CACHE_DIR" --codec_tmpl "$CODEC_TMPL" "$OUT_DIR"
done
echo "[sweep] $(date '+%H:%M') done"
