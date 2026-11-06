#!/bin/bash
#SBATCH --job-name=conf-fdp-infer
#SBATCH --account=nstx-u
#SBATCH --partition=gpu
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# confine-ours (base3: 5 folds x 3 seeds) over every extracted grid not yet inferred, or LIST's
# (scripts/labeler/conf_fdp_label.py infer). The networks are tiny: one batched job covers the run.
#   [LIST=<root>/batches/bNNNN.csv] sbatch scripts/labeler/conf_fdp_infer.sbatch
# GPU rule: the log prints GPU utilisation over the shot loop (nvidia-smi every 0.2 s).
set -euo pipefail
REPO=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export CONF_FDP_ROOT="${CONF_FDP_ROOT:-$LABELER_ROOT/conf_fdp_1ms}"
export PYTHONPATH="$REPO/src" LABELER_NO_FETCH=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
JOB="${SLURM_JOB_ID:-none}"
export XDG_CACHE_HOME="$LABELER_ROOT/runs/slurm/cache/$JOB"
mkdir -p "$XDG_CACHE_HOME"
cd "$REPO"
PY="$REPO/.pixi/envs/stellarai/bin/python"
OUT="$LABELER_ROOT/runs/slurm/conf-fdp-infer_$JOB.out"
FINE="$LABELER_ROOT/runs/slurm/$JOB.nvsmi_fine.csv"
echo "job $JOB on $(hostname) at $(date -Is): list ${LIST:-all}, root $CONF_FDP_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 200 > "$FINE" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
ARGS=(--root "$CONF_FDP_ROOT" infer --workers $(( ${SLURM_CPUS_PER_TASK:-16} - 2 )) --batch "${BATCH:-64}")
[ -n "${LIST:-}" ] && ARGS+=(--list "$LIST")
[ -n "${REDO:-}" ] && ARGS+=(--redo)
RC=0
"$PY" -u scripts/labeler/conf_fdp_label.py "${ARGS[@]}" || RC=$?
kill "$SMI" 2>/dev/null || true
"$PY" - "$OUT" "$FINE" <<'PYEOF'
import sys
from datetime import datetime

fmt = "%Y/%m/%d %H:%M:%S.%f"
out, smi = sys.argv[1:]
marks = {}
for line in open(out):
    mark, _, when = line.partition(" ")
    if mark in ("LOOP_START", "LOOP_WARM", "LOOP_END"):
        marks[mark] = datetime.strptime(when.strip(), fmt)
samples = []
for line in open(smi):
    parts = [p.strip() for p in line.split(",")]
    try:
        samples.append((datetime.strptime(parts[0], fmt), float(parts[1].split()[0]), float(parts[2].split()[0])))
    except (IndexError, ValueError):
        continue
b = marks.get("LOOP_END")
for name, a in (("shot loop", marks.get("LOOP_START")), ("after the first batch", marks.get("LOOP_WARM"))):
    if a is None or b is None:
        continue
    inside = [(u, m) for t, u, m in samples if a <= t <= b]
    if inside:
        print(f"GPU utilisation over {name}: mean {sum(u for u, _ in inside) / len(inside):.1f} % over "
              f"{len(inside)} samples ({(b - a).total_seconds():.1f} s), GPU memory max {max(m for _, m in inside):.0f} MiB")
PYEOF
echo "finished at $(date -Is)"
exit "$RC"
