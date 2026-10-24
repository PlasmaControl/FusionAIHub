#!/bin/bash
#SBATCH --job-name=silver-tm-infer
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out

# tm-ours 2026_10_04 on the silver shots that have magnetics features but no
# out-of-fold row (owner's ask 4, 2026-10-15):
#   python scripts/labeler/silver_tm_saw_rwm.py tm-infer --workers $SLURM_CPUS_PER_TASK [--limit N]
# ONE batched GPU pass: every shot's features (round4/tm/magfeatures for the population
# shots, the tm-features stage of silver_tm_saw_rwm.sbatch for the rest; both staged by
# SLURM CPU jobs, so this job only reads them, on --workers threads) are padded into one
# tensor, standardised on the GPU per seed, and run through the five fold checkpoints x
# three seeds with a masked forward (equal to one shot at a time). The same pass
# recomputes the out-of-fold silver shots' stored scores (the check of the whole path),
# and a few shots are rerun through tm_ours.predict on the CPU (the CPU chain); the job
# fails if either differs. Writes $LABELER_ROOT/runs/silver_infer/tm_saw_rwm/tm/pred/
# and tm/infer.json (stage seconds, checks). Python: envs/phase3 (torch 2.14 + CUDA
# 12.6), as tm_ours.py's docstring names it. Nothing is fetched.
#
#   pilot:      LIMIT=20 sbatch <this>            (20 inferred + 20 OOF-check shots)
#   full run:   sbatch --mem=<pilot MaxRSS x 1.2> --time=<...> <this>
#   gate:       python -m labeler.jobstats --job-id <id>   (the pilot: --pilot)
#   GPU:        nvidia-smi samples every 2 s in runs/slurm/<id>.nvsmi.csv (every 0.2 s in
#               <id>.nvsmi_fine.csv); the log prints when the batched pass starts and ends
#
# GPU rule (owner, 2026-10-15 19:04). The full run goes ahead under the TINY-MODEL
# exception, not the 50 % one: tm-ours is three 32-channel dilated 1-D convolutions on
# 33 features per 10 ms bin (about 25k weights a seed), and the whole run is a single
# batched pass of at most 5 minutes; its GPU compute is a fraction of a second, so the
# GPU is idle for most of the job (startup, reading features, the checks).
#
# ------------------------------------------ sizing (measured 2026-10-15)
#   pilot 2965486 (LIMIT=20: 20 inferred + 20 OOF-check shots, 4 CPUs, 8G, A100-PCIE-40GB):
#     11 s elapsed; MaxRSS 956328K (934 MiB, mostly torch + CUDA libraries; the batch
#     was 40 shots x 33 features x 820 bins); TotalCPU 4.5 s of 44 core-s; peak CUDA 31 MiB.
#     stages (s): torch/CUDA init 0.46, plan 0.25, load features 0.02, checkpoints 0.28,
#     pack+H2D 0.01, batched pass 0.94 (first-call cuDNN/kernel loading; the kernels
#     themselves are microseconds), write 0.03, OOF check 0.01, CPU chain 0.18.
#     checks: |batched - stored OOF| <= 1.2e-7 (0 calls differ); |batched - CPU chain|
#     <= 4.2e-7. GPU over the batched pass (19:27:46.486-47.436): mean 0.4 % in the
#     0.2 s samples (1, 1, 0, 0, 0 %), 1 % in the one 2 s sample.
#     jobstats --pilot: FAIL (exempt), CPU and GPU undetermined (too short to sample),
#     CPU-mem 11.4 % (sacct). The exemption is not read as a GPU pass: the full run
#     rests on the tiny-model exception above.
#   full run, sized from the pilot: --cpus-per-task=2 (loading 40 files took 0.02 s; the
#     inputs were staged by CPU jobs), --mem=1400M ((934 MiB + ~230 MiB for the
#     ten-times-larger host batch, 400 shots x 33 features x up to ~1500 bins in float64
#     plus the loaded features) x 1.2), --time=00:05:00 (the exception's 5-minute bound).
#   full run 2965491 (75 inferred + 325 OOF-check shots: 400 x 33 features x 820 bins):
#     6 s elapsed; MaxRSS 804604K (786 MiB); TotalCPU 4.35 s of 12 core-s (36 %); peak
#     CUDA 299 MiB, nvidia-smi 863 MiB used. stages (s): CUDA init 0.30, plan 0.02, load
#     0.44, checkpoints 0.05, pack+H2D 0.05, batched pass 0.26, D2H 0.003, write 0.09,
#     OOF check 0.04, CPU chain 0.11.
#     checks: OOF 325 shots, 186,141 bins, max diff 1.8e-7, 0 calls differ; CPU chain
#     5 shots x 5 folds, max diff 3.0e-7.
#     GPU over the batched pass (19:32:43.324-43.641): 0 % in the one 0.2 s sample inside
#     it and 4 % in the one 2 s sample; the next two 0.2 s samples (nvidia-smi averages
#     over a trailing period) read 4 and 20 %; whole job 1.4 % (0.2 s) and 1.3 % (2 s).
#     jobstats: FAIL. CPU and GPU undetermined (no rows for a 6 s job); CPU-mem 56.1 %:
#     the memory guess assumed the padded batch grows with the shot count, but the
#     longest shot was 820 bins in both runs, and the 2-CPU run kept less resident (786
#     vs 934 MiB). A rerun would take --mem=950M (786 MiB x 1.2); not rerun for the gate.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ROOT=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler
PY="$ROOT/envs/phase3/bin/python"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/EKOLEMEN/D3D/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID:-none}"
mkdir -p "$XDG_CACHE_HOME"
SAMPLES="$ROOT/runs/slurm/${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$SAMPLES.nvsmi.csv" &
SAMPLER=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 200 > "$SAMPLES.nvsmi_fine.csv" &
FINE=$!
trap 'kill $SAMPLER $FINE 2>/dev/null || true; rm -rf "$XDG_CACHE_HOME"' EXIT
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} tm-infer${LIMIT:+ (pilot, $LIMIT shots)} on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun "$PY" -u scripts/labeler/silver_tm_saw_rwm.py tm-infer \
    --workers "${SLURM_CPUS_PER_TASK:-4}" ${LIMIT:+--limit $LIMIT}
echo "finished at $(date -Is)"
