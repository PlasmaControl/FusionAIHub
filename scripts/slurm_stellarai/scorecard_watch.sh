#!/bin/bash
# Score every frozen checkpoint of the watched runs on rollouts as it appears.
# Trainers started with SNAPSHOT_EVERY keep runs/<run>/dynamics_step<N>.pt. For each one
# without runs/<run>/scorecard_<N>/ this submits scripts/slurm_stellarai/ckpt_scorecard.sh
# as two one-node jobs side by side (STAGE=val: held-out rollouts; STAGE=resp: NBI-off and
# ECH-off response), at most MAX_JOBS scorecard jobs at a time, then refreshes
# runs/scorecard_all.json with analysis/scorecard_table.py. Runs on the login node:
#   nohup bash scripts/slurm_stellarai/scorecard_watch.sh > logs/scorecard_watch.log 2>&1 &
set -uo pipefail

REPO_DIR=${REPO_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub}
RUNS_DIR=${RUNS_DIR:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_prod_v4/runs}
RUNS=${RUNS:-"ss75g_l48 ss75g_live_unroll ss75g_evt ss75g_gap ss75g_btf_cfg ss75g_btf ss75g_live_aug"}
MAX_JOBS=${MAX_JOBS:-6}
POLL_S=${POLL_S:-1800}
MIN_AGE_S=${MIN_AGE_S:-300}   # a snapshot younger than this may still be being written
# Runs whose rollouts outlast gpu-short's 4 h: 48 layers take ~2.5 h for val, ~6 h for resp.
LONG_RUNS=${LONG_RUNS:-"ss75g_l48"}

cd "$REPO_DIR"
while true; do
  for run in $RUNS; do
    for ckpt in "$RUNS_DIR/$run"/dynamics_step*.pt; do
      [ -f "$ckpt" ] || continue
      step=$(basename "$ckpt" .pt); step=${step#dynamics_step}
      out="$RUNS_DIR/$run/scorecard_$step"
      # A stage whose sbatch was rejected (e.g. by the QOS's 10-job limit) leaves
      # .unsubmitted_<stage> and is retried on the next poll.
      if [ -e "$out" ]; then
        stages=$(ls -A "$out" 2>/dev/null | sed -n 's/^\.unsubmitted_//p')
        [ -n "$stages" ] || continue
      else
        stages="val resp"
      fi
      [ $(( $(date +%s) - $(stat -c %Y "$ckpt") )) -ge "$MIN_AGE_S" ] || continue
      running=$(squeue -h -u "$USER" -o %j | grep -c '^sc_')
      [ "$running" -lt "$MAX_JOBS" ] || continue 3
      mkdir -p "$out"
      ids=""
      limit=(--qos=gpu-short)
      [[ " $LONG_RUNS " == *" $run "* ]] && limit=(--time=12:00:00)
      for stage in $stages; do
        id=$(sbatch --parsable "${limit[@]}" --job-name="sc_${run#ss75g_}_${step}_$stage" \
          --export=ALL,RUN="$run",CKPT_NAME="dynamics_step$step.pt",STAGE=$stage \
          scripts/slurm_stellarai/ckpt_scorecard.sh)
        if [ -n "$id" ]; then
          rm -f "$out/.unsubmitted_$stage"; ids+=" $id"
        else
          touch "$out/.unsubmitted_$stage"; ids+=" ($stage rejected, will retry)"
        fi
      done
      echo "[watch] $(date '+%m-%d %H:%M') $run step $step -> jobs$ids"
    done
  done
  .pixi/envs/stellarai/bin/python analysis/scorecard_table.py --runs_dir "$RUNS_DIR" > /dev/null 2>&1
  sleep "$POLL_S"
done
