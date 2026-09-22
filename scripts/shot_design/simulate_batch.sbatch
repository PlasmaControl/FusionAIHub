#!/bin/bash
#SBATCH --job-name=sd-simulate-batch
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/slurm/%j.out
# Paired real/proposed IGNITE rollouts for MANY saved designs: one design id per line in
# $IDENTS, 4 GPUs on one node, task i taking lines i, i+4, i+8, ... and skipping any design
# whose outputs/<ident>/simulation/status.json already says "complete". The Stellar twin of
# scripts/slurm_frontier/shot_design_simulate.sh (which runs ONE design per job).
#   SHOT_DESIGN_DATA_ROOT=.../experiments/<name> IDENTS=<file> [FOLLOW=1] sbatch scripts/shot_design/simulate_batch.sbatch [simulate args]
# FOLLOW=1 keeps re-reading a growing $IDENTS (see batch_idents.py) until $IDENTS.final exists.
# --wait=0 --kill-on-bad-exit=0: by default srun kills the remaining tasks 60 s after the
# FIRST task exits (job 2937656 lost three quarter-lists that way when task 3 finished early).
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/shot_design/_stellar_common.sh"
IDENTS="${IDENTS:?set IDENTS to a file with one design id per line}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export FOLLOW="${FOLLOW:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks="$SLURM_NTASKS" --wait=0 --kill-on-bad-exit=0 bash -c '
  PY="$0"; IDENTS="$1"; ROOT="$2"; shift 2   # bash -c: the first extra arg is $0
  done_=0; fail=0
  # FOLLOW=1: $IDENTS is append-only and still growing (batch_idents.py adds design ids
  # as the design batch saves them); re-read it every 2 min, skipping every id whose
  # status.json is already complete or failed, until "$IDENTS.final" appears. Line
  # index mod ntasks stays the owner of a line, so an append-only file never hands one
  # design to two tasks.
  while :; do
    n=0
    while IFS= read -r ident; do
      [ -z "$ident" ] && continue
      if [ $((n % SLURM_NTASKS)) -eq "$SLURM_PROCID" ]; then
        st="$ROOT/outputs/$ident/simulation/status.json"
        if [ -f "$st" ] && grep -Eq "\"state\": \"(complete|failed)\"" "$st"; then n=$((n+1)); continue; fi
        t0=$(date +%s)
        if "$PY" -m shot_design simulate "$ident" --device cuda "$@" > "$ROOT/runs/slurm/sim-$ident.log" 2>&1; then
          done_=$((done_+1)); echo "task $SLURM_PROCID $ident ok $(( $(date +%s) - t0 )) s"
        else
          fail=$((fail+1)); echo "task $SLURM_PROCID $ident FAILED $(( $(date +%s) - t0 )) s (see runs/slurm/sim-$ident.log)"
        fi
      fi
      n=$((n+1))
    done < "$IDENTS"
    if [ -z "${FOLLOW:-}" ] || [ -f "$IDENTS.final" ]; then break; fi
    sleep 120
  done
  echo "task $SLURM_PROCID finished: $done_ ok, $fail failed"
' "$PY" "$IDENTS" "$ROOT" "$@"
echo "finished at $(date -Is)"
