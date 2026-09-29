#!/bin/bash
#SBATCH --job-name=ae-below80
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2500M
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# The owner-absent frames where AE may lie only below 80 kHz (round three, spec §2.6, D28):
#   python -m labeler.ae.xpower.below80   every shot of v3's snapshot: v3's AE off an MHD
#       frame, or TokEye (masks-full) lit only below 80 kHz once pseudo-v2's steady lines are
#       removed -> $LABELER_ROOT/suggestions/ae_xpower/v3/below80.{csv,json,md}
# Suggestions, never labels: the owner's labels and the page's band are not touched. Needs v3's
# chosen model and its one test (models/ae_xpower/v3, whose evaluation.json evaluate --test
# writes; the command refuses before it) and pseudo-v2's rules.json. Nothing is fetched.
#
#   pilot:      SHOTS=170720,176053,176041 sbatch <this>
#   full run:   sbatch <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26 on ae_xpower_gallery)
#   ae_xpower_gallery's 180-shot run reads the same stores with the same kind of model: 1 CPU,
#   1500M, 82.1 % CPU, 80 % memory, 3 min 39 s. below80 also reads each shot's masks-full
#   record (~50 MB unpacked) and finds its lines on the level-8 grid: ~2 s a shot.
#   mem=2500M      1.2 GB + one record, x 1.6; Task 1.13 sets it from the pilot's MaxRSS.
#   time=00:45:00  180 shots x 2 s = 6 min, x 7 for cold reads and start-up.
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
srun "$ENV/bin/python" -m labeler.ae.xpower.below80 --workers "${SLURM_CPUS_PER_TASK:-1}" \
    ${SHOTS:+--shots "$SHOTS"}
echo "finished at $(date -Is)"
