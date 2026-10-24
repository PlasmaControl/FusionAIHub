#!/bin/bash
#SBATCH --job-name=elm-inst2
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out

# elm-inst2 (scripts/labeler/elm_inst2.py): the five fold models trained at once on one GPU,
# one process each, then `predict` (out-of-fold traces, the key shots' five-model mean, and
# elm-inst's mean on his shots). Run `elm_inst2_cpu.sbatch` with STAGE=prepare first.
# The model is small (InstNet), so one process cannot keep the GPU busy; five can.
#
#   run:   export LABELER_ROOT=... LABELER_LABEL_TABLES=... SHOT_DESIGN_DATA_ROOT=...
#          sbatch --output=$LABELER_ROOT/runs/slurm/%j.out --error=$LABELER_ROOT/runs/slurm/%j.out \
#              scripts/labeler/elm_inst2.sbatch
#          pilot: EPOCHS=2 ITERS=50 PREDICT=0 sbatch --time=00:15:00 ... (same flags)
#          REPO=<worktree> runs another checkout's code
#   gate:  this log's "GPU utilisation over the training loop" line (the GPU rule: >= 50 %) and
#          python -m labeler.jobstats --job-id <id>
#
# ------------------------------------------ sizing
#   2977039 (pilot: EPOCHS=2 ITERS=50 PREDICT=0; A100-PCIE-40GB): job 43 s; the five processes
#   load their inputs for about 16 s (GPU 0 %), then train at 53-80 % (2 s sampler), 26 % over
#   the whole loop with the loading; 7 s per 50 steps and 1.7 s of validation an epoch; GPU
#   memory 11.4 GB; MaxRSS 15.1 GB (five processes); 3.3 CPUs busy of 8.
#   Full run: 40 x 100 steps, about 16 s an epoch, 11 min, then predict; mem 20G, time 40 min.
#   2977040 (full run): 10 min 37 s; GPU 65.4 % over the training loop; MaxRSS 14.9 GB;
#   CPU 50 min of 85 (57.7 %). jobstats FAIL on its 70 % bars (CPU 57.7 %, GPU 62.3 %); it
#   passes the GPU rule (>= 50 %). Next run: --cpus-per-task=6.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ROOT="${LABELER_ROOT:?export LABELER_ROOT before sbatch: the job reads and writes under it}"
: "${LABELER_LABEL_TABLES:?export LABELER_LABEL_TABLES before sbatch}"
PY="$ROOT/envs/phase3/bin/python"  # torch with CUDA; the pixi env's torch is CPU-only
EPOCHS="${EPOCHS:-40}"
ITERS="${ITERS:-100}"
export PYTHONPATH="$REPO/src" LABELER_NO_FETCH=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="$ROOT/runs/mplconfig"
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID:-none}"
mkdir -p "$XDG_CACHE_HOME"
JOB="${SLURM_JOB_ID:-none}"
RUN="$ROOT/round4/elm/inst/inst2"
NVSMI="$ROOT/runs/slurm/$JOB.nvsmi.csv"
cd "$REPO"
echo "job $JOB on $(hostname) at $(date -Is), repo $REPO at $(git rev-parse --short HEAD), epochs $EPOCHS x $ITERS"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$NVSMI" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
stamp() { date +'%Y/%m/%d %H:%M:%S.%3N'; }
start=$(stamp)
pids=()
for k in 0 1 2 3 4; do
    "$PY" -u scripts/labeler/elm_inst2.py train --fold "$k" --device cuda:0 \
        --epochs "$EPOCHS" --iters "$ITERS" > "$RUN/train_fold$k.log" 2>&1 &
    pids+=("$!")
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
end=$(stamp)
kill "$SMI" 2>/dev/null || true
"$PY" - "$NVSMI" "$start" "$end" <<'EOF'
import sys
from datetime import datetime

fmt = "%Y/%m/%d %H:%M:%S.%f"
smi, start, end = sys.argv[1], datetime.strptime(sys.argv[2], fmt), datetime.strptime(sys.argv[3], fmt)
u = []
for line in open(smi):
    parts = [p.strip() for p in line.split(",")]
    try:
        t = datetime.strptime(parts[0], fmt)
    except (IndexError, ValueError):
        continue
    if start <= t <= end:
        u.append(float(parts[1].split()[0]))
print(f"GPU utilisation over the training loop: mean {sum(u) / max(len(u), 1):.1f} % over "
      f"{len(u)} samples ({(end - start).total_seconds():.0f} s)")
EOF
[ "$failed" = 0 ] || { echo "a fold failed: see $RUN/train_fold*.log"; tail -5 "$RUN"/train_fold*.log; exit 1; }
if [ "${PREDICT:-1}" = 1 ]; then
    "$PY" -u scripts/labeler/elm_inst2.py predict --device cuda:0 > "$RUN/predict.log" 2>&1
fi
echo "done at $(date -Is)"
