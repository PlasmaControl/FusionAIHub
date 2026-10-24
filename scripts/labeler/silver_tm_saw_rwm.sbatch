#!/bin/bash
#SBATCH --job-name=silver-tsr
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out

# The CPU stages of the silver tables for tm-ours, saw-ours v2 and rwm-ours (owner's ask
# 4, 2026-10-15), one stage a job:
#   python scripts/labeler/silver_tm_saw_rwm.py $STAGE --workers $WORKERS [--limit N]
#   STAGE=tm-features  magnetics features of the silver corpus shots round four never
#                      featurised, over each shot's Ip window (feeds silver_tm_infer.sbatch)
#   STAGE=saw-frames   saw-ours v2's combined presence per 10 ms frame, from the stored
#                      fix6 fold predictions and the fix5 ECE signals
#   STAGE=rwm-score    the final rwm-ours forest on Hanson's 33 shots, scored on every
#                      silver shot whose inputs load
# into $LABELER_ROOT/runs/silver_infer/tm_saw_rwm/. Nothing is fetched (LABELER_NO_FETCH=1).
#   STAGE=table        the three tables from those outputs and silver_tm_infer.sbatch's
#                      (one process; JOBS_FILE names the stage jobs for the meta):
#                      python scripts/labeler/silver_tm_saw_rwm.py table --jobs-file $JOBS_FILE
#                      with LIMIT or SAMPLE (the pilot) under runs/.../pilot_tables,
#                      never $LABELER_LABEL_TABLES
#
#   pilot:      LIMIT=20 STAGE=saw-frames sbatch <this>   (the first 20 planned shots,
#               the full run's workers and CPUs; single-file outputs get a _pilot suffix;
#               SAMPLE=20 instead spreads the 20 over the plan)
#   full run:   STAGE=saw-frames sbatch --mem=<pilot MaxRSS x 1.2> --time=<...> <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only   (the pilot: --pilot)
#
# ------------------------------------------ sizing (measured 2026-10-15, sacct)
#   tm-features  pilot 2965467 (LIMIT=20, 8 CPUs): 11 s, MaxRSS 4.0 GiB, TotalCPU 58 s
#                full 2965472 (45 shots): --cpus-per-task=8 --mem=5G --time=00:10:00;
#                17 s, MaxRSS 4.0 GiB (CPU-mem 79.9 %), CPU 98 s of 136 core-s (72 %)
#   saw-frames   pilots 2965465 (LIMIT=20, 8 CPUs: 6 s, 397 MiB, CPU 4.6 s of 48, the
#                pool idle) and 2965471 (SAMPLE=20, 2 CPUs: 4 s, 284 MiB, CPU 4.2 s of 8)
#                full 2965473 (322 shots): --cpus-per-task=2 --mem=350M --time=00:10:00;
#                14 s, MaxRSS 294 MiB (CPU-mem 84 %), CPU 22.5 s of 28 core-s (80 %)
#   rwm-score    pilots 2965466 (LIMIT=20: 7 s, 623 MiB) and 2965470 (SAMPLE=20: 6 s,
#                1.05 GiB, the forest fit dominating)
#                full 2965474 (450 shots): --cpus-per-task=8 --mem=1350M --time=00:10:00;
#                42 s, MaxRSS 1.08 GiB (CPU-mem 81.6 %), CPU 283 s of 336 core-s (84 %)
#   table        pilot 2965495 (SAMPLE=20, 1 CPU, 2G): 43 s, MaxRSS 367 MiB, CPU 3.5 s;
#                34 s of it was importing sawtooth_fix6 (it imports torch) on a cold
#                node. full 2965500 (450 shots): --cpus-per-task=1 --mem=450M
#                --time=00:10:00; 12 s, MaxRSS 284 MiB (CPU-mem 63 %), CPU 5.2 s of 12
#                core-s (44 %). The full run kept less resident than the pilot, so
#                MaxRSS x 1.2 of the pilot missed the 70 % line; a rerun: --mem=350M
#   jobstats --cpu-only on every full run: FAIL, CPU undetermined (no CPU rows for jobs
#   of 12-42 s); CPU-mem 79.9-84 % for the stages, 63 % for the table (sacct). The CPU
#   figures above are sacct's TotalCPU / (CPUs x elapsed); none was padded to reach the
#   sampler.
#   The scheduler moved every one of these jobs from pppl/pppl-short-stellar to
#   partition all, QOS stellar-debug (as ae_xpower_gallery.sbatch records).
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV="${LABELER_PIXI_ENV:-/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler}"  # another cluster: export LABELER_PIXI_ENV
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
LABELER_LABEL_TABLES="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_LABEL_TABLES "/scratch/gpfs/EKOLEMEN/D3D/events")"
export LABELER_LABEL_TABLES
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
export XDG_CACHE_HOME="$ROOT/runs/slurm/cache/${SLURM_JOB_ID:-none}"
mkdir -p "$XDG_CACHE_HOME"
trap 'rm -rf "$XDG_CACHE_HOME"' EXIT
cd "$REPO"
: "${STAGE:?set STAGE to tm-features, saw-frames, rwm-score or table}"
echo "job ${SLURM_JOB_ID:-none} $STAGE${LIMIT:+ (pilot, first $LIMIT shots)}${SAMPLE:+ (pilot, $SAMPLE shots spread over the plan)} on $(hostname) at $(date -Is), root $ROOT"
if [ "$STAGE" = table ]; then
    : "${JOBS_FILE:?set JOBS_FILE to the JSON of the stage jobs, for the meta}"
    srun "$ENV/bin/python" -u scripts/labeler/silver_tm_saw_rwm.py table \
        --jobs-file "$JOBS_FILE" ${LIMIT:+--limit $LIMIT} ${SAMPLE:+--sample $SAMPLE}
else
    srun "$ENV/bin/python" -u scripts/labeler/silver_tm_saw_rwm.py "$STAGE" \
        --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" ${LIMIT:+--limit $LIMIT} ${SAMPLE:+--sample $SAMPLE}
fi
echo "finished at $(date -Is)"
