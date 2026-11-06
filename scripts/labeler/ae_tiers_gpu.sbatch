#!/bin/bash
#SBATCH --job-name=ae-tiers
#SBATCH --account=nstx-u
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%A_%a.out

# GPU steps of scripts/labeler/ae_tiers.py (train, predict); the arguments are passed through.
#   teachers:  sbatch --array=0-4 scripts/labeler/ae_tiers_gpu.sbatch train --name teacher_f{fold}
#   ensemble:  sbatch scripts/labeler/ae_tiers_gpu.sbatch predict --models 'models/teacher_f*/model.pt' --name teachers
#   student:   sbatch scripts/labeler/ae_tiers_gpu.sbatch train --name student_r1 --pseudo <root>/pseudo/r1 --dropout 0.2
# GPU rule: the log ends with GPU utilisation between the script's LOOP_START and LOOP_END
# (nvidia-smi every 0.5 s); gate with python -m labeler.jobstats --job-id <id>.
set -euo pipefail
REPO=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export AE_TIERS_ROOT="${AE_TIERS_ROOT:-$LABELER_ROOT/ae_tiers}"
export PYTHONPATH="$REPO/src" LABELER_NO_FETCH=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${SLURM_ARRAY_TASK_ID:-0}"
export XDG_CACHE_HOME="$LABELER_ROOT/runs/slurm/cache/$JOB"
mkdir -p "$XDG_CACHE_HOME"
cd "$REPO"
PY="$REPO/.pixi/envs/stellarai/bin/python"
FINE="$LABELER_ROOT/runs/slurm/$JOB.nvsmi.csv"
echo "job $JOB on $(hostname) at $(date -Is): $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 500 > "$FINE" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
LOG="$LABELER_ROOT/runs/slurm/ae-tiers_$JOB.log"
"$PY" -u scripts/labeler/ae_tiers.py --root "$AE_TIERS_ROOT" "$@" 2>&1 | tee "$LOG"
kill "$SMI" 2>/dev/null || true
"$PY" - "$LOG" "$FINE" <<'PY'
import sys
from datetime import datetime
fmt = "%Y/%m/%d %H:%M:%S.%f"
log, smi = sys.argv[1:]
marks = {}
for line in open(log):
    mark, _, when = line.partition(" ")
    if mark in ("LOOP_START", "LOOP_WARM", "LOOP_END"):
        try:
            marks[mark] = datetime.strptime(when.strip(), fmt)
        except ValueError:
            pass
rows = []
for line in open(smi):
    p = [x.strip() for x in line.split(",")]
    try:
        rows.append((datetime.strptime(p[0], fmt), float(p[1].split()[0]), float(p[2].split()[0])))
    except (IndexError, ValueError):
        continue
b = marks.get("LOOP_END")
for name, a in (("loop", marks.get("LOOP_START")), ("after warm-up", marks.get("LOOP_WARM"))):
    if a and b:
        x = [(u, m) for t, u, m in rows if a <= t <= b]
        if x:
            print(f"GPU utilisation over {name}: mean {sum(u for u, _ in x) / len(x):.1f} % over {len(x)} samples "
                  f"({(b - a).total_seconds():.0f} s), memory max {max(m for _, m in x):.0f} MiB")
PY
echo "finished at $(date -Is)"
