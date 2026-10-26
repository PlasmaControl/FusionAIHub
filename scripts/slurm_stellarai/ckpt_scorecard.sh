#!/bin/bash
#SBATCH --account=kolemen
#SBATCH --job-name=scorecard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =====================================================================================
# Rollout scorecard of ONE frozen checkpoint (what the masked validation loss misses), on
# one node (~10 min per rollout and GPU), in one or two jobs (STAGE):
#   1. held-out rollout: the last 32 shots of the validation split, real actuators, scored
#      in code space and for hallucinated / missed spectrogram activity
#      (skipped when VAL_DIR points at an existing rollout of the same checkpoint)
#   2. NBI response: all beams off at frame 50 (3.5 s) on 32 held-out shots with steady NBI
#   3. ECH response: ECH off at frame 50 on the 22 held-out shots with steady ECH
# STAGE=all runs 1-3 (too long for 4 h); STAGE=val runs 1 and STAGE=resp runs 2-3, as two
# jobs side by side. Whichever finishes second writes the scorecard. Response rollouts run
# only shots without an archive yet, so an older 16-shot scorecard is topped up in place.
# The response rollouts archive their real-actuator baseline (same random stream), so
# analysis/checkpoint_scorecard.py scores dT_i / dT_e from the archives alone.
# Shot lists come from analysis/select_response_shots.py (validation split, pins excluded);
# they live here, not in --export, because sbatch --export splits values on commas.
#   sbatch --export=ALL,RUN=ss75g_btf,CKPT_NAME=dynamics_step16800.pt scripts/slurm_stellarai/ckpt_scorecard.sh
#   sbatch --export=ALL,RUN=ss75g_btf,CKPT_NAME=dynamics_step21000.pt,VAL_DIR=<existing val32 dir> ...
# Output: runs/<RUN>/scorecard_<step>/{val32,nbi_off,ech_off}/
# =====================================================================================
set -uo pipefail

export REPO_DIR=${REPO_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub}
DATA_ROOT=${DATA_ROOT:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/models}
RUN=${RUN:?set RUN}
CKPT_NAME=${CKPT_NAME:?set CKPT_NAME}
STEP=${CKPT_NAME//[!0-9]/}
STAGE=${STAGE:-all}
NBI_SHOTS=${NBI_SHOTS:-200212,201574,200206,203622,204011,203552,203958,201775,201212,201851,204158,201685,201913,200444,192858,204989,190465,190750,201887,204602,190905,201957,204447,201959,190463,204189,192422,193775,192386,202833,204824,191917}
ECH_SHOTS=${ECH_SHOTS:-204989,204602,201893,201775,204722,201887,204673,204007,204011,203958,204676,203500,200212,200206,201240,191022,204824,190633,190523,201685,190632,204627}
export CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/ignite_prod_v4/frame_codes}
export IGNITE_ACT_GLOBAL=${IGNITE_ACT_GLOBAL:-$DATA_ROOT/ignite_prod_v4/act_global}
_codec_template="$DATA_ROOT/ignite_codecs_v4/{m}/codec_best.pt"
export CODEC_TMPL=${CODEC_TMPL:-$_codec_template}
export IGNITE_EVAL_MODALITIES="bes,cer_rot,cer_ti,co2,ece,filterscopes,mhr,mirnov,mse,tangtv_lower,tangtv_upper,ts_core_density,ts_core_temp,ts_tangential_density,ts_tangential_temp"
export IGNITE_EVAL_RAW_GT=0 RENDER_ALL=0 TEMPERATURE=1.0 GUIDANCE=1
export CLUSTER_ENV=${CLUSTER_ENV:-scripts/slurm_stellarai/_stellarai_common.sh}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export CKPT=$DATA_ROOT/ignite_prod_v4/runs/$RUN/$CKPT_NAME
OUT=$DATA_ROOT/ignite_prod_v4/runs/$RUN/scorecard_$STEP
PINNED="190729,190735,190736"

cd "$REPO_DIR"
source "$CLUSTER_ENV"
mkdir -p "$OUT"
VAL=${VAL_DIR:-$OUT/val32}
if [ "$STAGE" != resp ]; then
  if [ -z "${VAL_DIR:-}" ] && [ ! -f "$VAL/hallucination.json" ]; then
    echo "[scorecard] $(date '+%H:%M') $RUN step $STEP: held-out rollout"
    ( export VAL_TAIL=32 VAL_N=438 SPLIT_SEED=42 OUT_DIR=$VAL; bash scripts/slurm_frontier/eval_dynamics.sh )
    python analysis/rollout_npz_score.py --split "$PINNED" "$VAL"
  fi
  if [ ! -f "$VAL/hallucination.json" ]; then
    srun -N1 -n1 --gpus-per-task=1 --gpu-bind=none python analysis/rollout_hallucination.py --split "$PINNED" \
      --cache_dir "$CACHE_DIR" --codec_tmpl "$CODEC_TMPL" "$VAL"
  fi
fi
if [ "$STAGE" != val ]; then
  ok=1
  for pair in "nbi_off|$NBI_SHOTS|scale:pinj=0@50+beam_voltage=0@50+tinj=0@50" "ech_off|$ECH_SHOTS|scale:ech_power=0@50"; do
    IFS='|' read -r name shots mode <<< "$pair"
    todo=$(for s in ${shots//,/ }; do [ -f "$OUT/$name/codes_${s}_step$STEP.npz" ] || echo "$s"; done | paste -sd,)
    if [ -n "$todo" ]; then
      echo "[scorecard] $(date '+%H:%M') $RUN step $STEP: $name on $todo"
      ( export SHOT=$todo ACTUATOR_MODE=$mode OUT_DIR=$OUT/$name; bash scripts/slurm_frontier/eval_dynamics.sh )
    fi
    for s in ${shots//,/ }; do [ -f "$OUT/$name/codes_${s}_step$STEP.npz" ] || { echo "[scorecard] $name $s missing" >&2; ok=0; }; done
  done
  [ "$ok" = 1 ] || echo "[scorecard] scoring without the missing shots" >&2
  touch "$OUT/resp.done"
fi
if [ -f "$VAL/hallucination.json" ]; then
  if [ -f "$OUT/resp.done" ]; then
    python analysis/checkpoint_scorecard.py --codec_tmpl "$CODEC_TMPL" \
      --entry "$RUN@$STEP=$VAL,$OUT/nbi_off,$OUT/ech_off" --json "$OUT/scorecard.json"
  fi
fi
echo "[scorecard] $(date '+%H:%M') $STAGE done"
