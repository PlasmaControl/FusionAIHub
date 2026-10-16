#!/bin/bash
#SBATCH --job-name=overlays
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labeler/runs/slurm/%j.out

# TokEye overlays on every spectrogram row of the review stores (plan section C), in ONE GPU job
# per list, never an array:
#   python -m labeler.events.review.overlays build "$@" --device cuda --workers <cpus - 1>
#   -> $LABELER_ROOT/overlays/tokeye/<model sha8>/<store event>/<shot>.{npz,json}
# Seven DataLoader workers read the stores (each spectrogram row's uint8 pyramid level, or the
# AE store's four CO2 chords from the corpus or the raw cache) while the main process does the
# math on the GPU: clip and z-score each row, tile it, the pinned U-Net over tiles of many rows
# and shots per forward pass (fp16 autocast), the stitch; for AE the CO2 resample, STFT,
# standardise (labeler.ae.gpu), the clean mask, persistence, notch and 2-of-4 union. A shot
# done with the same model on the same store is not rerun (--force reruns it). Nothing is
# fetched (LABELER_NO_FETCH=1); a shot without a store is skipped and counted.
#
# Arguments pass through to `overlays build`, one of:
#   --event E [E ...] --shots-file F   a list's shots (a CSV with a shot column, or one per line)
#   --event E [E ...] --list NAME      a registered shot list (configs/labeler/shotlists.yaml)
#   --requests F                       a queue batch: the review server's drain submits
#                                      `sbatch --job-name=overlays-queue --time=00:05:00
#                                      --cpus-per-task=4 --mem=8G --export=ALL,REPO=<its repo>
#                                      <this> --requests F`, F holding at most 8 validated
#                                      "<store event> <shot>" lines (overlays.QUEUE_*)
# The job samples the GPU every 2 s (runs/slurm/<job>.nvsmi.csv) and every 0.2 s
# (<job>.nvsmi_fine.csv) and prints, for each, the mean utilisation over the shot loop (LOOP_START
# to LOOP_END: the GPU rule's number) and after the first forward pass (LOOP_WARM to LOOP_END:
# CUDA start-up and cuDNN's first kernels behind it), with per-stage seconds and per-event counts
# ("overlays summary": read_wait, prep, network, post, write; by_event).
#
#   pilot:     sbatch --time=00:20:00 <this> --event mhd_n1 edge_localized_mode alfven_eigenmode \
#                  --shots-file <20 shots of the list>
#   full run:  sbatch <this> --event <every store event with spectrogram rows> \
#                  --shots-file data/events/silver/silver_shots.csv   (then gold/gold_shots.csv)
#   AE gate:   python -m labeler.events.review.overlays compare  (IoU against tokeye-full on its
#              180 shots, the bar of ae/full.py: median >= 0.75, <= 10 % below 0.5, both bands)
#   gate:      cd /scratch/gpfs/nc1514/FusionAIHub && PYTHONPATH=$PWD/src pixi run --frozen \
#                  --no-install -e labeler python -m labeler.jobstats --job-id <id> [--pilot], and
#              this log's "GPU utilisation over shot loop" line.
#
# GPU rule, which branch:
#   list runs (--event ... --shots-file/--list): the pilot-then-full-run branch: the full run only
#              if the pilot's loop kept the GPU >= 50 % busy (jobstats' pilot exemption is no pass).
#   queue batches (--requests, from the review page): the tiny-model branch: ONE batched job of at
#              most 5 minutes (--time=00:05:00 from the drain, at most 8 shots a batch, one queue
#              job pending or running at a time). Measured (2966127, 8 AE shots, the heaviest
#              kind, 4 CPUs): 22 s wall, a 11.9 s loop at 73.8 % GPU (2 s sampler); lower
#              overlays.QUEUE_MAX_SHOTS only if a batch ever comes near 5 minutes.
#
# ------------------------------------------ sizing
#   Measured 2026-10-16 on A100-PCIE-40GB nodes (C-gpu report), 7 workers, prefetch 4, batch 32:
#   pilot 2965796 (20 silver shots x 14 events, 40 layers): 26.2 s loop, GPU 72.3 % (2 s) / 81.4 %
#                  (0.2 s), MaxRSS 3.38 GiB.
#   AE-only pilot 2965807 (20 shots): 22.1 s loop, GPU 87.6 % / 80.0 %, MaxRSS 6.89 GiB. AE alone
#                  keeps 28 CO2 records in flight (7 workers x prefetch 4), so it needs the most
#                  memory: the first gate try at 6G (sized from the mixed pilot) died OOM (2965799).
#   gate 2965809 (180 AE shots): 146.7 s loop, GPU 89.9 %, MaxRSS 8.78 GiB of 9G.
#   silver 2966113 (450 x 14): 10 min 58 s, 624.6 s loop, GPU 90.2 %, MaxRSS 4.21 GiB.
#   gold 2966114 (50 x 14): 1 min 15 s, 60.7 s loop, GPU 83.1 %, MaxRSS 3.75 GiB.
#   So: mem=12G (the AE-only MaxRSS 8.78 GiB x 1.2 = 10.5, rounded up; a mixed list peaks near
#   4.2 GiB), time=00:30:00 (silver took 11 min; gold fits in 00:10:00), cpus=8 per the GPU rule.
#   jobstats fails these jobs on CPU (19-23 %) and GPU memory (32-34 %): the GPU is the bottleneck
#   (read_wait is 3-4 % of the loop), the workers idle; 4 CPUs would likely keep up.
#   Queue batches: 4 CPUs, 8G, 5 minutes, passed by the drain on the sbatch line (MaxRSS 4.11 GiB
#   for 8 AE shots).
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"  # the drain exports the server's own worktree
ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labeler
PY="$ROOT/envs/phase3/bin/python"  # torch 2.14 + CUDA 12.6; the pixi env's torch is CPU-only
export LABELER_ROOT="$("$PY" "$REPO/src/labeler/env.py" LABELER_ROOT "$ROOT")" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-$REPO/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # tiles of many shapes: less fragmentation
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID:-none}"
mkdir -p "$XDG_CACHE_HOME"
cd "$REPO"
JOB="${SLURM_JOB_ID:-none}"
OUT="$ROOT/runs/slurm/$JOB.out"
NVSMI="$ROOT/runs/slurm/$JOB.nvsmi.csv"
FINE="$ROOT/runs/slurm/$JOB.nvsmi_fine.csv"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
echo "job $JOB on $(hostname) at $(date -Is), root $LABELER_ROOT, $WORKERS workers, args: $*"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$NVSMI" &
SMI=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 200 > "$FINE" &
SMI_FINE=$!
trap 'kill "$SMI" "$SMI_FINE" 2>/dev/null || true' EXIT
# python directly, not srun: a step started by srun need not get the job's 8 CPUs.
STATUS=0
"$PY" -u -m labeler.events.review.overlays build "$@" --device cuda --workers "$WORKERS" || STATUS=$?
kill "$SMI" "$SMI_FINE" 2>/dev/null || true
"$PY" - "$OUT" "$NVSMI" "$FINE" <<'EOF'
import sys
from datetime import datetime

fmt = "%Y/%m/%d %H:%M:%S.%f"
out, *smis = sys.argv[1:]
loops, loop = [], {}
for line in open(out):
    mark, _, when = line.partition(" ")
    if mark in ("LOOP_START", "LOOP_WARM", "LOOP_END"):
        loop[mark] = datetime.strptime(when.strip(), fmt)
        if mark == "LOOP_END":
            loops.append(loop)
            loop = {}
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
    for k, marks in enumerate(loops):
        b = marks["LOOP_END"]
        for name, a in (("shot loop", marks.get("LOOP_START")),
                        ("after the first pass", marks.get("LOOP_WARM"))):
            if a is None:
                continue
            inside = [(u, m) for t, u, m in samples if a <= t <= b]
            if inside:
                print(f"GPU utilisation over {name} {k}: mean {sum(u for u, _ in inside) / len(inside):.1f} % "
                      f"over {len(inside)} samples ({(b - a).total_seconds():.1f} s), "
                      f"GPU memory max {max(m for _, m in inside):.0f} MiB")
            else:
                print(f"GPU utilisation over {name} {k}: no sample in its {(b - a).total_seconds():.1f} s")
EOF
echo "finished at $(date -Is) with status $STATUS"
exit "$STATUS"
