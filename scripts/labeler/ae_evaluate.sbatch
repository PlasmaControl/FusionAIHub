#!/bin/bash
#SBATCH --job-name=ae-eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labeler/runs/slurm/%j.out

# The AE SELDnet checkpoints (d3d_ae_activity_seldnet, every ae_seldnet_*.pt; threeway_sce is the
# shipped one) scored on named shot lists in ONE GPU job (docs/labeler/shotlists.md):
#   scripts/labeler/ae_evaluate.py --shotlist $SHOTLIST [--include-trained]
#   -> outputs/labeler/ae/shotlists/evaluation.json: sets.<list> (shotlists.plan's envelope: n_listed,
#      n_scored, missing, trained_on, leaky) and results.<list>.<checkpoint>.{reviewed,annotated},
#      on the owner's 10 ms frames of 0-2 s against the live review/labels.csv (sha256 in the record);
#      and evaluation.frames.npz beside it (the frame scores: a list rescores without the GPU).
# The deployed chain's numbers (the record's corpus block) need no GPU here: run the listed shots
# without a cache through ae_gold_extend.py --chain gpu (silver_ae_seldnet.sbatch with LIST and
# WORK=$ROOT/runs/ae_eval_lists/work), then `ae_evaluate.py --shotlist ... --chain corpus` as a CPU job.
# SHOTLIST defaults to the AE event's `evaluate` lists (configs/labeler/shotlists.yaml). SHOTLIST=none
# runs task 7b's evaluation instead: every checkpoint on the 60 validation shots, the models dir's
# evaluation.json and the six-shot figures. It was sized for one shot at a time in bf16 (about 2 min a
# checkpoint, 4 checkpoints plus the figures), so give it its old room:
#   SHOTLIST=none sbatch --time=00:30:00 --mem=8G scripts/labeler/ae_evaluate.sbatch
# Seven DataLoader workers (cpus-per-task - 1) read the staged npz ($ROOT/ae/dataset, ae_dataset.py,
# 20 MB a shot) while the GPU runs every checkpoint on 8 shots at a time (BATCH) in float32 and takes
# the 10 ms frame means; the same seven processes then compute the scores and their shot bootstrap.
# PRECISION=bf16 is task 7b's autocast instead: its GRU falls off cuDNN, so the GPU waits on kernel
# launches (job 2965782 below). Trained-on shots are not run (--include-trained runs and scores them, and marks the record
# leaky). The job samples the GPU every 2 s (runs/slurm/<job>.nvsmi.csv, the GPU rule's sampler) and
# every 0.2 s (<job>.nvsmi_fine.csv: the loop lasts seconds) and prints, at the end, each sampler's
# mean utilisation over the shot loop (the script's LOOP_START to LOOP_END) and after its first batch
# (LOOP_WARM to LOOP_END: CUDA start-up and cuDNN's first kernels are before it).
#
#   run:   sbatch scripts/labeler/ae_evaluate.sbatch
#          SHOTLIST=ae_ece,gold INCLUDE_TRAINED=1 OUT=/path/evaluation_leaky.json sbatch <this>
#          (PRECISION=bf16|fp32, BATCH=<shots a forward pass>)
#          INDEPENDENT_TRUTH=204503,... sbatch <this>: the owner's call that those shots, once
#          prefilled from this model, were re-labelled from scratch, so they count again
#          (ae_evaluate.py NOTES.prefilled; the record lists them under prefilled.cleared_by_hand)
#   gate:  cd $REPO && PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labeler \
#              python -m labeler.jobstats --job-id <id>
#          and this log's "GPU utilisation over the shot loop" line (the GPU rule).
#
# ------------------------------------------ sizing
#   2965782 (the five lists, 60 shots x 4 checkpoints, bf16, 4 shots a batch; A100-PCIE-40GB):
#   job 2:48, loop 129 s (2.1 s a checkpoint a batch: the bf16 GRU steps one kernel launch at a
#   time), loader wait 0.000 s after the first batch, GPU 34.6 % over the loop (2 s sampler) and
#   34.2 % (0.2 s); GPU memory 3.6 GB; MaxRSS 1.92 GiB. jobstats: CPU 11 %, GPU 25.5 %: FAIL.
#   2965786 (pilot: ae_valid_fair, 19 shots, fp32, 8 a batch): loop 1.7 s (0.09 s a checkpoint a
#   batch, cuDNN's GRU), GPU after the first batch 90.7 % (0.2 s sampler; 3 samples), 42.8 % over
#   the whole loop (the first batch holds CUDA and cuDNN start-up); GPU memory 11.9 GB; MaxRSS
#   1.55 GiB. Its threeway_sce frame means differ from 2965782's bf16 ones by 2e-4 on average
#   (max 0.029), one hard call of 3,800 flips, AUROC 0.9826 -> 0.9825.
#   2965787 (the five lists, fp32, 8 a batch, scoring in a 7-process pool): job 16 s, loop 3.5 s
#   (GPU 3.1 s, read 0.54 s, loader wait 0.32 s), GPU 74.1 % over the loop (0.2 s sampler, 17
#   samples; the 2 s sampler caught 2: 55.0 %), 98.9 % after the first batch; GPU memory 11.9 GB;
#   scoring 5.1 s; MaxRSS 2.08 GiB. jobstats: n/a on every metric (the job is shorter than its
#   sampling), so the gate is this log's sampler line.
#   mem 2.08 x 1.2 -> 3G; time 10 min for margin (the run takes under a minute).
#   GPU clause: the tiny-model one (440,514 parameters; the whole run is one batched GPU job of
#   about a minute, under 5 min). The loop lasts seconds, so start-up weighs on its mean.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labeler
PY="$ROOT/envs/phase3/bin/python"  # torch 2.14 + CUDA 12.6; the pixi env's torch is CPU-only
SHOTLIST="${SHOTLIST:-ae_ece,ae_co2,ae_co2_ece,ae_valid_fair,gold}"
OUT="${OUT:-$REPO/outputs/labeler/ae/shotlists/evaluation.json}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-$REPO/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="$ROOT/runs/mplconfig"
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID:-none}"
mkdir -p "$XDG_CACHE_HOME"
cd "$REPO"
JOB="${SLURM_JOB_ID:-none}"
LOG="$ROOT/runs/slurm/$JOB.out"
NVSMI="$ROOT/runs/slurm/$JOB.nvsmi.csv"
FINE="$ROOT/runs/slurm/$JOB.nvsmi_fine.csv"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
echo "job $JOB on $(hostname) at $(date -Is), lists $SHOTLIST, $WORKERS workers"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$NVSMI" &
SMI=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 200 > "$FINE" &
SMI_FINE=$!
trap 'kill "$SMI" "$SMI_FINE" 2>/dev/null || true' EXIT
# python directly, not srun: a step started by srun need not get the job's 8 CPUs.
if [ "$SHOTLIST" = none ]; then
    # the old file's four threads: this path reads and scores in the main process
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u "$REPO/scripts/labeler/ae_evaluate.py" \
        --dataset-dir "$ROOT/ae/dataset" \
        --models-dir "$ROOT/models/d3d_ae_activity_seldnet" \
        --fig-dir "$REPO/outputs/labeler/ae/training"
else
    "$PY" -u "$REPO/scripts/labeler/ae_evaluate.py" --shotlist "$SHOTLIST" \
        --dataset-dir "$ROOT/ae/dataset" \
        --models-dir "$ROOT/models/d3d_ae_activity_seldnet" \
        --out "$OUT" --workers "$WORKERS" --device cuda \
        --precision "${PRECISION:-fp32}" --batch-shots "${BATCH:-8}" \
        ${INCLUDE_TRAINED:+--include-trained} \
        ${INDEPENDENT_TRUTH:+--independent-truth "$INDEPENDENT_TRUTH"}
fi
kill "$SMI" "$SMI_FINE" 2>/dev/null || true
"$PY" - "$LOG" "$NVSMI" "$FINE" <<'EOF'
import sys
from datetime import datetime

fmt = "%Y/%m/%d %H:%M:%S.%f"
log, *smis = sys.argv[1:]
marks = {}
for line in open(log):
    mark, _, when = line.partition(" ")
    if mark in ("LOOP_START", "LOOP_WARM", "LOOP_END"):
        marks[mark] = datetime.strptime(when.strip(), fmt)
for smi in smis:
    samples = []
    for line in open(smi):
        parts = [p.strip() for p in line.split(",")]
        try:
            samples.append((datetime.strptime(parts[0], fmt), float(parts[1].split()[0]),
                            float(parts[2].split()[0])))
        except (IndexError, ValueError):
            continue
    print(f"nvidia-smi: {len(samples)} samples in {smi}")
    end = marks.get("LOOP_END")
    for name, start in (("the shot loop", marks.get("LOOP_START")),
                        ("after the first batch", marks.get("LOOP_WARM"))):
        if start is None or end is None:
            print(f"GPU utilisation over {name}: no loop marks in {log}")
            continue
        inside = [(u, m) for t, u, m in samples if start <= t <= end]
        if inside:
            print(f"GPU utilisation over {name}: mean {sum(u for u, _ in inside) / len(inside):.1f} % "
                  f"over {len(inside)} samples ({(end - start).total_seconds():.1f} s), "
                  f"GPU memory max {max(m for _, m in inside):.0f} MiB")
        else:
            print(f"GPU utilisation over {name}: no sample in its {(end - start).total_seconds():.1f} s")
EOF
echo "finished at $(date -Is)"
