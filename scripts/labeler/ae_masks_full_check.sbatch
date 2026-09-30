#!/bin/bash
#SBATCH --job-name=ae-masks-full-check
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=04:40:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# Whole-shot TokEye against v1 over 0-2 s (round three, spec §2.2, D10):
#   python -m labeler.ae.full --check   every shot with a masks-full record: the IoU of the
#       >= 2-chord lit pixels in 0-250 and in 80-250 kHz, each record's notched bins, and
#       SELDNet's 0-2 s frame agreement between ae/dataset and ae/dataset-full
#       -> $LABELER_ROOT/ae/masks-full/check.{json,md}, with each record's lit pixels per
#       compared column beside each IoU
# The bar (median IoU >= 0.75 and at most 10 % of shots below 0.50, in both bands) is
# reported, not raised: the job succeeds either way. Nothing is fetched.
#
#   pilot:      SHOTS=<the pilot's 16, comma-separated> sbatch <this>
#   full run:   sbatch --dependency=afterok:<the ae_masks_full job> <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
# check.{json,md} are no input and are left out of masks-full's manifest.json, so a rerun
# of the check leaves v3's inputs as they are; the manifest's `report` names job 2944173's.
#
# ------------------------------------------ sizing (measured on pilot job 2944155)
#   pilot 2944155 (9f5c1a5, the first pilot's 16 shots, --mem=8G): 4:49, 18.1 s a shot,
#   MaxRSS 4.50 GiB (the first try, 2944136, ran out of memory at 3G).
#   jobstats: CPU 86.9 %  CPU-mem 47.5 %  GPU n/a  GPU-mem n/a  FAIL (exempt)
#   mem=7G         4.50 GiB x 1.4. jobstats reads the peak below sacct's (3.8 GiB of 8G),
#                  so CPU-mem may read under 70 %: the margin is over the per-shot peak.
#   time=04:40:00  18.1 s x 180 shots x 5.
#   before the pilot:
#   ae_xpower_evaluate's pilot ran SELDNet with every other method at ~4 s a test shot, max
#   RSS 1.85 GB, on one thread. The check runs SELDNet twice over 0-2 s (~3 s) and compares
#   two unpacked masks (4 x 512 x 23,445 bytes, ~48 MB each): ~4 s a shot.
#   mem=3G         1.85 GB + two masks, x 1.4; Task 1.11 sets it from the pilot's MaxRSS.
#   time=01:00:00  180 shots x 4 s = 12 min, x 5 for cold reads of masks-full.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.full --check ${SHOTS:+--shots "$SHOTS"}
echo "finished at $(date -Is)"
