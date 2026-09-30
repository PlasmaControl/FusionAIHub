#!/bin/bash
#SBATCH --job-name=ae-xpower-final
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4500M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# After all 15 cross-validation tasks (ae_xpower_cv.sbatch): the choice, then the
# final model.
#   python -m labeler.ae.xpower.cv --version v2 --choose
#       pools the 15 out-of-fold records, applies Deviation 11's rule at thresholds
#       0.10-0.90, writes models/ae_xpower/v2/cv/{choice.json,frontier.md}; it
#       refuses while any fold record is missing and names the missing ones.
#   python -m labeler.ae.xpower.train --version v2 --from-cv
#       the chosen candidate on all 120 train and validation shots of the snapshot,
#       for the median of its five folds' best epochs, no early stopping, at the
#       choice's threshold: models/ae_xpower/v2/<candidate>/{model.pt, split.csv,
#       training.json, review/labels.csv} and models/ae_xpower/v2/chosen.json.
# The owner's labels come only from the version's frozen snapshot; the choice also
# checks that the source table ($LABELER_LABEL_TABLES/alfven_eigenmode/format/) is the
# one every fold record scored with, as the test does. Nothing is fetched. CPU only.
# The test shots are scored afterwards, once:
#   VERSION=v2 sbatch --dependency=afterok:<id> ae_xpower_evaluate.sbatch
#
#   pilot:      PILOT=10 sbatch <this>   (after the cv pilot: its choice, then 10 pool
#               shots for 2 epochs, into runs/ae_xpower/pilot/v2; a pilot of 20
#               shots or fewer comes first; it is exempt from the gate but reported)
#   full run:   sbatch --dependency=afterok:<cv array id> <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#   v4:         VERSION=v4 PILOT=10 sbatch <this>, then VERSION=v4 sbatch
#               --dependency=afterok:<v4 cv array id> <this>: models/ae_xpower/v4/
#               {cv/choice.json, <candidate>/, chosen.json}; then
#               VERSION=v4 sbatch --dependency=afterok:<id> ae_xpower_evaluate.sbatch
#
# ------------------------------------------ sizing (measured 2026-09-29, v3's final, job 2944533)
#   The choice: 5 s, MaxRSS 231,232 KiB. The final model, band0-mhd3 on 120 shots for
#   18 epochs: 10 min 1 s, sacct MaxRSS 3,824,656 KiB (3735 MiB), CPU 84.2 %.
#   mem=4500M      3735 MiB x 1.2 is 4482 MiB; rounded up.
#   time=01:30:00  kept: 60 epochs, the most a final trains, take about 33 min; x 2.7.
#   v4 (an estimate, not measured): v2's 120 shots over 0-2 s on 60-250 kHz, about 11 %
#                  more pixels than v1's 80-250 kHz final (~2530 MB, below): 2530 x 1.11 x 1.2
#                  is about 3400M: pass --mem=3400M.
# before v3 (measured 2026-09-26, v1's train header):
#   The choice reads 15 small npz records: seconds.
#   v1's full runs: 88 train shots, ~21 s an epoch (band80, 4 threads; 44 batches of 16
#   crops at 0.47 s); peak RSS 2311 MB with 104 shots loaded, 1182 MB with 20
#   (~13.4 MB a shot). The final model loads 120 shots: 60 batches, ~28 s an epoch;
#   at most 60 epochs (the median of the folds' best epochs is at most that), ~28 min.
#   mem=3200M      2311 + 16 x 13.4 = ~2530 MB x 1.25.
#   time=01:30:00  28 min x 3.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
VERSION="${VERSION:-v2}"
echo "job ${SLURM_JOB_ID:-none} ($VERSION) on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.xpower.cv --version "$VERSION" --choose ${PILOT:+--pilot "$PILOT"}
srun "$ENV/bin/python" -m labeler.ae.xpower.train --version "$VERSION" --from-cv ${PILOT:+--pilot "$PILOT"}
echo "finished at $(date -Is)"
