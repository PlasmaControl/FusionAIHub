#!/bin/bash
#SBATCH --job-name=ae-xpower-cv
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2800M
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
# Labels come only from the version's frozen snapshot
# ($LABELER_ROOT/models/ae_xpower/$VERSION/review/labels.csv, sha256 checked); the
# live review file is never read. Nothing is fetched. CPU only, 4 threads.
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
# ------------------------------------------ sizing (measured 2026-09-26, v1's train header)
#   v1's full runs, 88 train + 16 val shots loaded, 4 threads: 22-32 min for up to 60
#   epochs at ~21 s an epoch (band80); peak RSS 2311 MB (band80-mhd3), 2470 MB (band0).
#   A fold loads 72 train + 24 early-stopping shots (96, fewer than v1's 104) and
#   predicts its 24 one at a time: ~17 s an epoch, at most ~26 min.
#   mem=2800M      v1's measured size; 96 loaded shots peak below v1's 2311 MB.
#   time=01:00:00  26 min x 2.3. Size from the full runs, not from the pilot, whose
#                  peak under-predicts (v1: 1182 MB at 20 shots).
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub-round-two}"
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
echo "job ${SLURM_JOB_ID:-none} task $K ($VERSION $CANDIDATE fold $FOLD) on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.xpower.cv --version "$VERSION" \
    --candidate "$CANDIDATE" --fold "$FOLD" ${PILOT:+--pilot "$PILOT"}
echo "finished at $(date -Is)"
