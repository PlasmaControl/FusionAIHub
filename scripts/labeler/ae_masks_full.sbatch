#!/bin/bash
#SBATCH --job-name=ae-masks-full
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# TokEye over the whole shot, for the 180 AE shots (round three, spec §2.2):
#   scripts/labeler/ae_masks_full.py   the raw cache's CO2 over 0 .. min(end, 6000) ms,
#       resampled to 500 kHz; TokEye's STFT, big_tf_unet_251210.pt and v1's cleaning
#       -> $LABELER_ROOT/ae/masks-full/<shot>_<split>_{clean,probs}.npz
#          $LABELER_ROOT/ae/dataset-full/<shot>_<split>.npz
# A shot whose three files exist is skipped (FORCE=1 redoes it), so the full run after the
# pilot does the other 164. ae/masks and ae/dataset are read, never written. Nothing is
# fetched (LABELER_NO_FETCH=1): a shot missing from the raw cache fails and is named.
#
# Environment: the tokeye repo's venv, read-only (torch 2.9.1+cu128, tokeye importable), with
# pyarrow from $LABELER_ROOT/ae/pylibs and this repo's src/ on PYTHONPATH, as in
# ae_dataset.sbatch. One worker (cpus-per-task - 1) prepares the next shot on the CPU while the
# parent drives the GPU. The batch (chords at once) is 2 on a 40 GB A100 and 4 on an 80 GB one,
# read from nvidia-smi, so the GPU-memory peak stays near 82 % on either.
#
#   pilot:      PILOT=1 sbatch --time=00:30:00 <this>   (16 shots, the owner's three first)
#   full run:   sbatch --mem=<the pilot's MaxRSS x 1.3> <this>
#   some shots: SHOTS=170720,176053 sbatch <this>
#   tiled:      TILE_MS=2000 FORCE=1 sbatch <this>      (only if the IoU bar fails; D12)
#   gate:       python -m labeler.jobstats --job-id <id> --gpu
#
# ------------------------------------------ sizing (measured on v1's 0-2 s run, then scaled)
#   v1 (ae_dataset.sbatch over 0-2 s, 7,813 frames): 8.4 s a shot on one A100, GPU peak
#   21.93 GiB at 4 chords a batch, about 5.5 GiB a chord.
#   A whole shot of up to 6 s is up to 23,440 frames: about 16.4 GiB a chord, so batch 2 peaks
#   near 33 GiB of 40 (82 %) and batch 4 near 66 GiB of 80; about 25 s a shot at 6 s.
#   time=02:30:00  180 shots x 25 s = 75 min, x 2 for start-up and cold reads.
#   mem=16G        the parent holds the model and one shot's raw, norm and prob (~0.5 GB), the
#                  worker one 1.667 MHz record (~0.5 GB) and its resampling; torch and CUDA
#                  take a few GB a process. Task 1.11 sets it from the pilot's MaxRSS x 1.3.
#   cpus=2         the parent (near 100 % while the GPU runs) and one worker (~4 s of each
#                  25 s shot). If the pilot's CPU is under 70 %, --cpus-per-task=1 prepares in
#                  the parent (--workers 0).
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
PY=/scratch/gpfs/nc1514/tokeye/.venv/bin/python
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LABELER_ROOT="$ROOT" PYTHONPATH="$ROOT/ae/pylibs:$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
export HF_HUB_OFFLINE=1  # the checkpoint is loaded from a local path
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is), root $ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv
TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
BATCH=2
if [ "$TOTAL_MIB" -gt 60000 ]; then BATCH=4; fi
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))
echo "batch $BATCH, workers $WORKERS, tile ${TILE_MS:-0} ms"
srun "$PY" -u "$REPO/scripts/labeler/ae_masks_full.py" --device cuda --batch "$BATCH" \
    --workers "$WORKERS" --tile-ms "${TILE_MS:-0}" ${PILOT:+--pilot} ${SHOTS:+--shots "$SHOTS"} \
    ${FORCE:+--overwrite}
echo "finished at $(date -Is)"
