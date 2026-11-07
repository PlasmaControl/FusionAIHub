#!/bin/bash
#SBATCH --job-name=elm-fdp-infer
#SBATCH --account=nstx-u
#SBATCH --partition=gpu
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# An ELM model run (five folds, labeler.elm.serve layout) over the extracted inputs of one batch
# (scripts/labeler/elm_fdp_label.py infer). DataLoader workers build the next shots' channels
# while the GPU runs the folds; PARTS processes share the GPU on disjoint shares of the shots.
#   MODELS=<run dir> [LIST=<root>/batches/bNNNN.csv] sbatch scripts/labeler/elm_fdp_infer.sbatch
# REPO runs another checkout (a worktree) with the main checkout's pixi env, or LABELER_PIXI_ENV's;
# INFER_ARGS adds infer arguments (e.g. --keep-heat for an instance run).
# GPU rule: the log prints GPU utilisation over the shot loop (nvidia-smi every 0.2 s).
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub}"
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export ELM_FDP_ROOT="${ELM_FDP_ROOT:-$LABELER_ROOT/elm_fdp_1ms}"
export PYTHONPATH="$REPO/src" LABELER_NO_FETCH=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
JOB="${SLURM_JOB_ID:-none}"
export XDG_CACHE_HOME="$LABELER_ROOT/runs/slurm/cache/$JOB"
mkdir -p "$XDG_CACHE_HOME"
cd "$REPO"
PY="${LABELER_PIXI_ENV:-/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.pixi/envs/stellarai}/bin/python"
OUT="$LABELER_ROOT/runs/slurm/elm-fdp-infer_$JOB.out"
FINE="$LABELER_ROOT/runs/slurm/$JOB.nvsmi_fine.csv"
echo "job $JOB on $(hostname) at $(date -Is): models $MODELS, list ${LIST:-all}, root $ELM_FDP_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 200 > "$FINE" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
PARTS="${PARTS:-2}"
ARGS=(--root "$ELM_FDP_ROOT" infer --models "$MODELS" --parts "$PARTS"
      --workers $(( (${SLURM_CPUS_PER_TASK:-16} - PARTS) / PARTS )))
[ -n "${LIST:-}" ] && ARGS+=(--list "$LIST")
[ -n "${INFER_ARGS:-}" ] && read -ra EXTRA <<< "$INFER_ARGS" && ARGS+=("${EXTRA[@]}")
PIDS=()
for (( p = 0; p < PARTS; p++ )); do
    "$PY" -u scripts/labeler/elm_fdp_label.py "${ARGS[@]}" --part "$p" &
    PIDS+=($!)
done
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
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
        t = datetime.strptime(when.strip(), fmt)
        keep = max if mark == "LOOP_END" else min
        marks[mark] = keep(marks.get(mark, t), t)
samples = []
for line in open(smi):
    parts = [p.strip() for p in line.split(",")]
    try:
        samples.append((datetime.strptime(parts[0], fmt), float(parts[1].split()[0]), float(parts[2].split()[0])))
    except (IndexError, ValueError):
        continue
b = marks.get("LOOP_END")
for name, a in (("shot loop", marks.get("LOOP_START")), ("after the first shot", marks.get("LOOP_WARM"))):
    if a is None or b is None:
        continue
    inside = [(u, m) for t, u, m in samples if a <= t <= b]
    if inside:
        print(f"GPU utilisation over {name}: mean {sum(u for u, _ in inside) / len(inside):.1f} % over "
              f"{len(inside)} samples ({(b - a).total_seconds():.1f} s), GPU memory max {max(m for _, m in inside):.0f} MiB")
PYEOF
echo "finished at $(date -Is)"
exit "$RC"
