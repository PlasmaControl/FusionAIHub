#!/bin/bash
#SBATCH --job-name=ae-xpower-extend
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out

# The chosen AE model over the population's CO2 shots, as suggestions, in four shards:
#   python -m labeler.ae.xpower.extend --shard K --of 4 --workers 8
# writes $LABELER_ROOT/suggestions/ae_xpower/v1/shards/K.{csv,summary.csv,npz,failed.txt} and a
# JPEG per shot into the gallery's extension/. The corpus is only read; nothing is fetched.
# Merge afterwards on the login node (seconds; a SLURM job for it could not meet the gate):
#   python -m labeler.ae.xpower.extend --merge --of 4
#
#   pilot:      LIMIT=20 sbatch --array=0 --cpus-per-task=4 --mem=7G <this>
#   full run:   sbatch <this>
#   gate each:  python -m labeler.jobstats --job-id <array id>_<k> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26, login node)
#   probe, 2 shots (2024 and 2025, 5.9 s and 4.7 s windows), 1 worker, pictures on: 10.9 s wall
#   with ~5 s of start-up, i.e. ~3 s a shot; max RSS 1.46 GB.
#   population: 1,924 of the frozen population's 4,885 shots have at least 2 s of
#               corpus CO2, all but 2 of them from 2024-2025.
#   cpus-per-task=8   481 shots a shard x 3 s / 8 workers = 3.0 min of work a shard.
#   mem=14G           8 x 1.46 GB x 1.2; reset from the pilot's MaxRSS.
#   time=01:00:00     3.0 min x 20 for cold GPFS reads of 64 MB of CO2 a shot.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub-round-two}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
K="${SLURM_ARRAY_TASK_ID:-0}"
OF="${OF:-4}"
echo "job ${SLURM_JOB_ID:-none} shard $K of $OF on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.xpower.extend --shard "$K" --of "$OF" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${LIMIT:+--limit "$LIMIT"}
echo "finished at $(date -Is)"
