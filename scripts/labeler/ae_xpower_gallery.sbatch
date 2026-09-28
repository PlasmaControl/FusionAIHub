#!/bin/bash
#SBATCH --job-name=ae-xpower-gallery
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
# Scheduler: pppl/pppl-short-stellar -> all/stellar-debug; sacct 2942492,2942516-18.

# One JPEG per AE180 shot, with the chosen model:
#   python -m labeler.ae.xpower.gallery --workers 1
# writes $LABELER_ROOT/gallery/alfven_eigenmode/ae_xpower-$VERSION/{reviewed,unreviewed}/<shot>.jpg
# (VERSION defaults to v1; v2's owner labels are its snapshot) and
# index.csv. A shot the owner saves later is redrawn by running it again (its row is replaced).
#
#   pilot:      SHOTS="170659 170813" sbatch <this>
#   full run:   sbatch <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26, login node)
#   pilot, 3 shots, 1 worker: 6.6 s wall with ~4 s of start-up, i.e. ~1 s a shot, 147 % of one
#   core (numpy's resampling threads were not pinned there; they are here).
#   180-shot run at 4 CPUs / 5G: 28.9 % CPU, 50 % memory; workers use about one core.
#   1 CPU / 1500M: 82.1 % CPU, 80 % memory in 3 min 39 s.
#   time=00:30:00   allows for cold reads and start-up beyond the measured run.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub-round-two}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
VERSION="${VERSION:-v1}"  # v1 by default, so v1's commands still mean v1
echo "job ${SLURM_JOB_ID:-none} ($VERSION) on $(hostname) at $(date -Is), root $ROOT"
# SHOTS is left unquoted on purpose: it is a list of shot numbers.
# shellcheck disable=SC2086
srun "$ENV/bin/python" -m labeler.ae.xpower.gallery --workers 1 --version "$VERSION" \
    ${SHOTS:+--shots $SHOTS}
echo "finished at $(date -Is)"
