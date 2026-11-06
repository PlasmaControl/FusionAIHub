#!/bin/bash
#SBATCH --job-name=elm-tiers
#SBATCH --account=nstx-u
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# GPU steps of scripts/labeler/elm_tiers.py; the arguments are passed through.
#   sbatch scripts/labeler/elm_tiers_gpu.sbatch train --role teacher     (the five folds at once)
#   sbatch scripts/labeler/elm_tiers_gpu.sbatch predict --role teacher --list <root>/lists/bronze.csv
# The model is small, so train runs the five fold processes on the one GPU and predict runs
# PARTS processes, to keep it busy. The log ends with the GPU utilisation over the run
# (nvidia-smi every 0.5 s); gate with python -m labeler.jobstats --job-id <id>.
set -euo pipefail
REPO=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export ELM_TIERS_ROOT="${ELM_TIERS_ROOT:-$LABELER_ROOT/elm_tiers}"
export PYTHONPATH="$REPO/src" LABELER_NO_FETCH=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
JOB="${SLURM_JOB_ID:-none}"
export XDG_CACHE_HOME="$LABELER_ROOT/runs/slurm/cache/$JOB"
mkdir -p "$XDG_CACHE_HOME"
cd "$REPO"
PY="$REPO/.pixi/envs/stellarai/bin/python"
FINE="$LABELER_ROOT/runs/slurm/$JOB.nvsmi_fine.csv"
echo "job $JOB on $(hostname) at $(date -Is): $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 500 > "$FINE" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
T0=$(date +%s)
PIDS=()
if [ "$1" = train ]; then
    for k in 0 1 2 3 4; do
        "$PY" -u scripts/labeler/elm_tiers.py "$@" --fold "$k" --workers 8 > "$LABELER_ROOT/runs/slurm/elm-tiers_${JOB}_f$k.log" 2>&1 &
        PIDS+=($!)
    done
else
    PARTS="${PARTS:-3}"
    for (( p = 0; p < PARTS; p++ )); do
        "$PY" -u scripts/labeler/elm_tiers.py "$@" --parts "$PARTS" --part "$p" --workers 12 &
        PIDS+=($!)
    done
fi
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
kill "$SMI" 2>/dev/null || true
[ "$1" = train ] && tail -n 2 "$LABELER_ROOT"/runs/slurm/elm-tiers_${JOB}_f*.log
awk -F', ' 'NR > 1 { split($2, u, " "); s += u[1]; n++ } END { if (n) printf "GPU utilisation over the run: mean %.1f %% over %d samples\n", s / n, n }' "$FINE"
echo "finished at $(date -Is) after $(( $(date +%s) - T0 )) s"
exit "$RC"
