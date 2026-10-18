#!/bin/bash
#SBATCH --job-name=silver-ae-seldnet
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=9G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labeler/runs/slurm/%j.out

# The best AE model, d3d_ae_activity_seldnet 2026_09_06 (configs/labeler/best_models.yaml), over
# the 450 silver shots in ONE GPU job, the whole chain on the GPU (--chain gpu, labeler.ae.gpu):
#   scripts/labeler/ae_gold_extend.py --shots-file $LIST --work-dir $WORK --chain gpu --device cuda
#       --no-write
#   -> $WORK/<shot>.npz (column probabilities, window end, CO2 tier, chain) or <shot>.skip.json
# Seven DataLoader workers (cpus-per-task - 1) find each shot's CO2 tier and window end and read
# and cut its CO2 record; the main process resamples it to 500 kHz, takes the STFT, clips,
# standardises and runs the network on the GPU (the full windows batched), then writes the npz.
# A shot already in $WORK is not rerun (work/ holds 173 shots from the cancelled cpu-chain array,
# network on the GPU, rest on one CPU). Nothing is fetched (LABELER_NO_FETCH=1): a shot with no CO2 on the corpus or the raw
# cache is skipped and named. The table is then built on the login node from $WORK alone
# (--build-only never loads the model; seconds):
#   ae_gold_extend.py --shots-file $LABELER_LABEL_TABLES/silver/silver_shots.csv --work-dir $WORK --build-only
#   -> $LABELER_LABEL_TABLES/alfven_eigenmode/extend_ae_seldnet/silver_shots.{csv,meta.json}
# CHECK="<gold shots>" also runs those gold shots into $GOLD_WORK (default $WORK/../gold_check_gpu),
# to compare with the gold table's CPU run (runs/ae_gold_extend) on the login node.
# The job samples the GPU every 2 s (runs/slurm/<job>.nvsmi.csv, the GPU rule's sampler) and every
# 0.2 s (<job>.nvsmi_fine.csv: a pilot's loop lasts seconds) and prints, at the end, each sampler's
# mean utilisation over each shot loop (the script's LOOP_START to LOOP_END) and over the loop after
# its first shot (LOOP_WARM to LOOP_END: CUDA start-up and cuDNN's first kernels are behind it).
#
#   pilot:     LIST=$ROOT/runs/silver_infer/ae_seldnet/pilot_gpu2/pilot_shots.csv \
#              WORK=$ROOT/runs/silver_infer/ae_seldnet/pilot_gpu2 GOLD_WORK=$ROOT/runs/silver_infer/ae_seldnet/gold_check_gpu2 \
#              CHECK="176070 199597" sbatch --time=00:20:00 <this>
#              (20 silver shots: 12 already in work/, rerun for the equivalence check, and 8 new)
#   full run:  sbatch <this>          (the silver list into work/, sized below; first move aside work/'s
#              12 npz older than their re-fetched cached Ip, so they are rerun: report gpu_ae.md)
#   gate:      python -m labeler.jobstats --job-id <id> [--pilot], and this log's "GPU utilisation
#              over shot loop" lines. The GPU rule's two branches: the pilot's loop keeps the GPU
#              >= 50 % busy, or a tiny model runs the whole list as one batched GPU job of <= 5 min
#              (jobstats' pilot exemption is never a pass). This job passes on the tiny-model branch:
#              AeSeldNet has 440,514 parameters and the full run is one job of about a minute; its
#              20-shot pilots kept the GPU 28.5-38 % busy over the loop (one-time CUDA start-up) and
#              57-67 % after the first shot. Equivalence: max |delta prob| <= 1e-3 and <= 0.1 % of
#              10 ms frames flipped against the cpu-chain npz (report gpu_ae.md; review gpu_ae.review1.md).
#
# ------------------------------------------ sizing
#   pilots 2965772 (20 shots + 2 gold) and 2965773 (pilot 1's 20 shots again), A100-PCIE-40GB:
#   job 24 s and 12 s; shot loop 5.0 s and 3.8 s (0.25 and 0.19 s a shot), of which the first shot
#   1.4-2.9 s (CUDA libraries loading their kernels, cuFFT plans, cuDNN: one-time); after it 0.11 s a
#   shot (worker read 0.67 s a shot over 7 workers, never waited on; resample 0.03, STFT 0.02,
#   network 0.05 s). GPU utilisation (0.2 s sampler) over the shot loop 28.5 % and 38.0 %, after the
#   first shot 67.3 % and 57.2 % (2 s sampler: 37 % and 16 % over 2 samples each). sacct MaxRSS
#   5.24 GiB and 4.70 GiB -> mem 5.24 x 1.2 = 6.3 -> 7G. GPU memory peak 7.05 GiB allocated.
#   full run: ~290 shots left (450 less work/'s 161 current npz), ~0.11 s a shot -> loop ~35 s,
#   job ~1 min; time=00:10:00 for margin.
#   full run 2965783 (2026-10-16, 289 shots, A100-PCIE-40GB): job 55 s, loop 39.6 s (0.14 s a
#   shot), GPU over the loop 51.6 % (2 s sampler, 20 samples) and 59.4 % (0.2 s, 197); 0 skipped;
#   loader wait above 10 ms on 3 shots. sacct MaxRSS 6.98 GiB, 99.7 % of the 7G it had -> 6.98 x 1.2
#   = 8.4 -> 9G for a rerun. torch's allocator once hit the card's 40 GB with cached blocks and
#   retried after freeing them (a warning, no failure).
#   GPU clause: the tiny-model one (the whole run is one batched GPU job of about a minute, under
#   5 min). A 20-shot pilot's loop is 4-5 s, so the one-time start-up keeps its mean under 50 %;
#   past the first shot the loop holds 57-67 % (report gpu_ae.md).
#   before: the cpu-chain array (pilot 2965454, 1 CPU a task) took 2-8 s a shot at GPU 0 %.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labeler
PY="$ROOT/envs/phase3/bin/python"  # torch 2.14 + CUDA 12.6; the pixi env's torch is CPU-only
LIST="${LIST:-${LABELER_LABEL_TABLES:-/scratch/gpfs/EKOLEMEN/nc1514/events}/silver/silver_shots.csv}"
WORK="${WORK:-$ROOT/runs/silver_infer/ae_seldnet/work}"
GOLD_WORK="${GOLD_WORK:-$(dirname "$WORK")/gold_check_gpu}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/EKOLEMEN/nc1514/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID:-none}"
mkdir -p "$XDG_CACHE_HOME" "$WORK"
cd "$REPO"
JOB="${SLURM_JOB_ID:-none}"
OUT="$ROOT/runs/slurm/$JOB.out"
NVSMI="$ROOT/runs/slurm/$JOB.nvsmi.csv"
FINE="$ROOT/runs/slurm/$JOB.nvsmi_fine.csv"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
echo "job $JOB on $(hostname) at $(date -Is), root $ROOT, list $LIST, work $WORK, $WORKERS workers"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$NVSMI" &
SMI=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 200 > "$FINE" &
SMI_FINE=$!
trap 'kill "$SMI" "$SMI_FINE" 2>/dev/null || true' EXIT
# python directly, not srun: a step started by srun need not get the job's 8 CPUs.
"$PY" -u "$REPO/scripts/labeler/ae_gold_extend.py" --shots-file "$LIST" --work-dir "$WORK" \
    --chain gpu --device cuda --workers "$WORKERS" --no-write
if [ -n "${CHECK:-}" ]; then
    echo "gold check: $CHECK at $(date -Is)"
    mkdir -p "$GOLD_WORK"
    "$PY" -u "$REPO/scripts/labeler/ae_gold_extend.py" --work-dir "$GOLD_WORK" \
        --chain gpu --device cuda --workers "$WORKERS" --shots $CHECK --no-write
fi
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
        for name, a in (("shot loop", marks["LOOP_START"]), ("after the first shot", marks.get("LOOP_WARM"))):
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
echo "finished at $(date -Is)"
