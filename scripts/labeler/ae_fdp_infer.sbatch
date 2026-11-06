#!/bin/bash
#SBATCH --job-name=ae-fdp-infer
#SBATCH --account=nstx-u
#SBATCH --partition=gpu
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# d3d_ae_activity_seldnet over the extracted CO2 files of one batch, the whole chain on the GPU
# (scripts/labeler/ae_fdp_label.py infer; labeler.ae.gpu as in silver_ae_seldnet.sbatch).
# Seven DataLoader workers read the next shots' compact CO2 files while the GPU resamples,
# transforms and runs the network. Writes <root>/work/<shot>.npz and <root>/results/<job>.*.csv,
# then deletes each shot's CO2 file (KEEP=1 keeps it).
#   LIST=<root>/batches/bNNNN.csv [KEEP=1] [NOTCH=1] sbatch scripts/labeler/ae_fdp_infer.sbatch
# GPU rule: inputs are staged by the CPU extract job; the log prints GPU utilisation over the
# shot loop (nvidia-smi every 0.2 s), gated with python -m labeler.jobstats --job-id <id>.
set -euo pipefail
REPO=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export AE_FDP_ROOT="${AE_FDP_ROOT:-$LABELER_ROOT/ae_fdp_1ms}"
export PYTHONPATH="$REPO/src" LABELER_NO_FETCH=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
JOB="${SLURM_JOB_ID:-none}"
export XDG_CACHE_HOME="$LABELER_ROOT/runs/slurm/cache/$JOB"
mkdir -p "$XDG_CACHE_HOME"
cd "$REPO"
PY="$REPO/.pixi/envs/stellarai/bin/python"
OUT="$LABELER_ROOT/runs/slurm/ae-fdp-infer_$JOB.out"
FINE="$LABELER_ROOT/runs/slurm/$JOB.nvsmi_fine.csv"
echo "job $JOB on $(hostname) at $(date -Is): list ${LIST:-all}, root $AE_FDP_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 200 > "$FINE" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
# PARTS processes share the GPU, each on a disjoint share of the shots: one alone leaves the GPU idle
# while it reads the next shot and writes its outputs (4 % busy on 498 shots, job 9103)
PARTS="${PARTS:-2}"
ARGS=(--root "$AE_FDP_ROOT" infer --workers $(( (${SLURM_CPUS_PER_TASK:-16} - PARTS) / PARTS )) --parts "$PARTS")
[ -n "${LIST:-}" ] && ARGS+=(--list "$LIST")
[ -n "${KEEP:-}" ] && ARGS+=(--keep-inputs)
[ -n "${SPEC_DIR:-}" ] && ARGS+=(--spec-dir "$SPEC_DIR")   # ae_tiers: keep the model input
[ -n "${CKPT:-}" ] && ARGS+=(--checkpoint "$CKPT")         # ae_tiers student instead of the stable model
[ -n "${NOTCH:-}" ] && ARGS+=(--notch)                    # pickup lines notched (labeler.ae.lines)
PIDS=()
for (( p = 0; p < PARTS; p++ )); do
    "$PY" -u scripts/labeler/ae_fdp_label.py "${ARGS[@]}" --part "$p" &
    PIDS+=($!)
done
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
kill "$SMI" 2>/dev/null || true
"$PY" - "$OUT" "$FINE" <<'EOF'
import sys
from datetime import datetime

fmt = "%Y/%m/%d %H:%M:%S.%f"
out, smi = sys.argv[1:]
marks = {}
for line in open(out):
    mark, _, when = line.partition(" ")
    if mark in ("LOOP_START", "LOOP_WARM", "LOOP_END"):
        t = datetime.strptime(when.strip(), fmt)  # PARTS processes: first start, last end
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
EOF
echo "finished at $(date -Is)"
exit "$RC"
