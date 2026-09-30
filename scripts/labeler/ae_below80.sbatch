#!/bin/bash
#SBATCH --job-name=ae-below80
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# The owner-absent frames where AE may lie only below 80 kHz (round three, spec §2.6, D28):
#   python -m labeler.ae.xpower.below80   every shot of v3's snapshot: v3's AE off an MHD
#       frame, or TokEye (masks-full) lit only below 80 kHz once pseudo-v2's steady lines are
#       removed, or TokEye lit below 80 kHz past pseudo-v3's per-line MHD markers
#       (tokeye_lines, with the catalog's NTM table; S2 fix)
#       -> $LABELER_ROOT/suggestions/ae_xpower/v3/below80.{csv,json,md}
# Suggestions, never labels: the owner's labels and the page's band are not touched. Needs v3's
# chosen model and its one test (models/ae_xpower/v3, whose evaluation.json evaluate --test
# writes; the command refuses before it) and pseudo-v2's rules.json. Nothing is fetched.
#
#   pilot:      SHOTS=170720,176053,176041 sbatch <this>
#   full run:   sbatch <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured on pilot job 2944607 and full run 2945198)
#   pilot, 3 shots (170720, 176053, 176041), 1 CPU at 2500M: 12 s wall, 4 s a shot with
#   start-up; sacct MaxRSS 488,392 KiB (477 MiB); too short for a CPU sample.
#   full run 2945198, 180 shots at 700M (the pilot's 477 MiB x 1.3): 4 min 44 s, CPU 85.3 %;
#   sacct MaxRSS 714,592 KiB (698 MiB), 99.7 % of 700M: the pilot under-predicted by 46 %.
#   mem=1000M      the full run's 698 MiB x 1.3 is 907 MiB; rounded up to 100M.
#   time=00:30:00  the full run's 4 min 44 s x 6.
#   tokeye_lines (an estimate, not measured): the markers add ~0.2 s and one more (n_y, n)
#   array a shot, with the lines read once for both TokEye reasons; 1000M and 30 min should
#   hold. Check it on the pilot's MaxRSS before the full run.
# before the pilot (measured 2026-09-26 on ae_xpower_gallery):
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
