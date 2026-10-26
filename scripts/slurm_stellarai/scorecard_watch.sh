#!/bin/bash
# =====================================================================================
# Score every frozen checkpoint of the watched runs on rollouts, as they appear.
# Trainers started with SNAPSHOT_EVERY (default 2000 in launch_ss75full.sh) keep
# runs/<run>/dynamics_step<N>.pt; for each one without runs/<run>/scorecard_<N>/ this
# submits scripts/slurm_stellarai/ckpt_scorecard.sh as two one-node jobs side by side
# (STAGE=val: held-out rollouts, ~1.2 h; STAGE=resp: NBI-off and ECH-off response, ~2.5 h),
# at most MAX_JOBS scorecard jobs at a time, and then refreshes
# runs/scorecard_all.json with analysis/scorecard_table.py. Runs on the login node:
#   nohup bash scripts/slurm_stellarai/scorecard_watch.sh > logs/scorecard_watch.log 2>&1 &
# =====================================================================================
set -uo pipefail

REPO_DIR=${REPO_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub}
RUNS_DIR=${RUNS_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_prod_v4/runs}
RUNS=${RUNS:-"ss75g_evt ss75g_gap ss75g_btf_cfg ss75g_btf ss75g_live_aug"}
MAX_JOBS=${MAX_JOBS:-6}
POLL_S=${POLL_S:-1800}
MIN_AGE_S=${MIN_AGE_S:-300}   # a snapshot younger than this may still be being written

cd "$REPO_DIR"
while true; do
  for run in $RUNS; do
    for ckpt in "$RUNS_DIR/$run"/dynamics_step*.pt; do
      [ -f "$ckpt" ] || continue
      step=$(basename "$ckpt" .pt); step=${step#dynamics_step}
      out="$RUNS_DIR/$run/scorecard_$step"
      [ -e "$out" ] && continue
      [ $(( $(date +%s) - $(stat -c %Y "$ckpt") )) -ge "$MIN_AGE_S" ] || continue
      running=$(squeue -h -u "$USER" -o %j | grep -c '^sc_')
      [ "$running" -lt "$MAX_JOBS" ] || continue 3
      mkdir -p "$out"
      ids=""
      for stage in val resp; do
        ids+=" $(sbatch --qos=gpu-short --job-name="sc_${run#ss75g_}_${step}_$stage" \
          --export=ALL,RUN="$run",CKPT_NAME="dynamics_step$step.pt",STAGE=$stage \
          scripts/slurm_stellarai/ckpt_scorecard.sh | awk '{print $4}')"
      done
      echo "[watch] $(date '+%m-%d %H:%M') $run step $step -> jobs$ids"
    done
  done
  .pixi/envs/stellarai/bin/python analysis/scorecard_table.py --runs_dir "$RUNS_DIR" > /dev/null 2>&1
  sleep "$POLL_S"
done
