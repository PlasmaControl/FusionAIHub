#!/bin/bash
#SBATCH --job-name=ae-xpower-cv
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2500M
#SBATCH --time=01:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out

# v2's five-fold cross-validation (the ledger's Deviation 11): one array task per
# (candidate, fold), 3 candidates x 5 folds = 15 tasks. Task id K gives
#   candidate = train.candidates("$VERSION")[K // 5], fold = K % 5:
#   0-4 band80-mhd3    5-9 band80-mhd10    10-14 band80-mhd30   (folds 0-4 in order)
# Task (c, k) trains candidate c on the three folds other than k and (k+1) mod 5,
# early-stops on fold (k+1) mod 5 (validation F1 at 0.5), and writes fold k's
# out-of-fold P(AE) to $LABELER_ROOT/models/ae_xpower/$VERSION/cv/<c>/fold<k>.{npz,json}.
# The owner's labels come only from the version's frozen snapshot
# ($LABELER_ROOT/models/ae_xpower/$VERSION/review/labels.csv, sha256 checked); the
# live review file is never read. The frames scored are the test's, inside the source
# table's windows ($LABELER_LABEL_TABLES/alfven_eigenmode/format/, its sha256 in the
# record). Nothing is fetched. CPU only, 4 threads.
#
# Before the array, write the folds once, on the login node (seconds; a SLURM job
# for it could not meet the gate); a fold task refuses to run without them:
#   python -m labeler.ae.xpower.cv --version v2 --folds
#
#   pilot:      python -m labeler.ae.xpower.cv --version v2 --folds --pilot 10
#               PILOT=10 sbatch <this>   (10 pool shots, 2 a fold, 2 epochs, into
#               runs/ae_xpower/pilot/v2/cv; a pilot of 20 shots or fewer comes
#               first; it is exempt from the gate but reported)
#   full run:   sbatch <this>
#   gate each:  python -m labeler.jobstats --job-id <array id>_<k> --cpu-only
#   then:       sbatch --dependency=afterok:<array id> ae_xpower_final.sbatch
#
# The seed study (post hoc: no decision depends on it). SEED=<S> trains with
# TrainConfig.seed S instead of the default 20260923, and only the chosen candidate
# (chosen.json's): tasks 5-9, band80-mhd10's folds 0-4. It reads cv/folds.csv (its
# sha256 checked against chosen.json's folds_sha256), changes nothing else (folds,
# stop rule, epoch cap, data), and writes only to
# $LABELER_ROOT/runs/ae_xpower/seeds/$VERSION/seed<S>/cv/ (a pilot's under
# seed<S>/pilot/); a seed aimed at the version's cv/ or at runs/ae_xpower/pilot/ is
# refused, and a record already there is never replaced. No --folds step: the folds
# are cv/'s. Pilot first:
#   pilot:      SEED=20260924 PILOT=10 sbatch --array=5-9 <this>   (2 shots a fold,
#               2 epochs, into runs/ae_xpower/seeds/v2/seed20260924/pilot/cv; it is
#               exempt from the gate but reported)
#   study:      SEED=20260924 sbatch --array=5-9 <this>
#               SEED=20260925 sbatch --array=5-9 <this>
#               SEED=20260926 sbatch --array=5-9 <this>
#   gate each:  python -m labeler.jobstats --job-id <array id>_<k> --cpu-only
#   then:       sbatch ae_xpower_posthoc.sbatch   (its summary is the record's (c))
#
# ------------------------------------------ sizing (measured 2026-09-28, A = job 2943054)
#   A, v2's 15 fold tasks: 72 train + 24 early-stopping shots loaded, 4 threads (CPU
#   75.4-89.2 %), 6 min 31 s to 16 min 13 s a task; sacct MaxRSS 1,780,848 to
#   2,156,464 KiB, peak 2106 MiB. A seeded fold loads and trains the same shots.
#   cpus=4         A's.
#   mem=2500M      A's peak 2106 MiB plus 15 % is 2422 MiB; rounded up to 2500M.
#   time=01:00:00  A's longest task, 16 min 13 s, x 3.7. Size from the full runs, not
#                  from the pilot, whose peak under-predicts (v1: 1182 MB at 20 shots).
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
VERSION="${VERSION:-v2}"
K="${SLURM_ARRAY_TASK_ID:-0}"
CANDIDATE="$("$ENV/bin/python" -c "from labeler.ae.xpower.train import candidates; print(list(candidates('$VERSION'))[$K // 5])")"
FOLD=$((K % 5))
echo "job ${SLURM_JOB_ID:-none} task $K ($VERSION $CANDIDATE fold $FOLD${SEED:+ seed $SEED}) on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.xpower.cv --version "$VERSION" \
    --candidate "$CANDIDATE" --fold "$FOLD" ${PILOT:+--pilot "$PILOT"} ${SEED:+--seed "$SEED"}
echo "finished at $(date -Is)"
