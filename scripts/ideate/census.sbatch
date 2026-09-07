#!/bin/bash
#SBATCH --job-name=ideate-census
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=01:15:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out

# Task I3: the fresh corpus census -- one row per (file, group) for all 16,909 files of
# $IDEATE_CORPUS, written to db/corpus_coverage.parquet with a sibling .json describing the run.
#
# This is the census the plan requires in place of the old shotsearch manifest, which counted a
# group as populated on the CHANNEL axis (`shape[0] > 1`) and so counted every (C, 1) placeholder
# as present. The work is header-only -- a ydata shape and two xdata scalars per group, never a
# ydata read -- so the job is bound by GPFS metadata latency, not by CPU or memory.
#
# ---------------------------------------------------------------------------- sizing (measured)
#
# Pilot (login node, cold cache, 2026-09-07):
#     /usr/bin/time -v pixi run -e ideate-cpu python -m ideate corpus scan \
#         --limit 300 --workers 8 --out $IDEATE_DATA_ROOT/db/census_pilot.parquet
#   -> 300 files, 249 openable, 7,949 rows; scan wall 32.3 s (38.4 s including `pixi run`);
#      max RSS 1,140,716 KB. A second, warm-cache run of the same 300 files took 2.5 s, which is
#      how we know the 32.3 s is cold metadata I/O and nothing else.
#
# A second pilot over 302 files spread across the whole corpus (every 56th shot) rather than its
# first 300: 37.2 s, 298 openable, 9,514 rows. Its per-group fractions are the ones the plan's V2
# reports (mhr 82 %, ece 83 %, co2 43 %, bes 38 %, tangtv 44 %, 1.3 % unopenable), while the
# first 300 shots are one early block and are not representative of anything (co2 0 %, 17 %
# unopenable). Both runs are used below -- the spread one for the rate, because it is the one
# that also opens the video groups.
#
# --time: 37.2 s / 302 files = 0.123 s per file at 8 workers.
#   * scaling linearly to 16 workers: 0.123 * 16,909 / 2 = 1,040 s (17 min);
#   * assuming the I/O server, not the client, is the limit, so that 8 -> 16 workers buys
#     nothing: 0.123 * 16,909 = 2,080 s (35 min).
#   The second is the one to budget against, and 2x it is 4,160 s = 69 min -> --time=01:15:00
#   (the QOS allows far more; the point of the bound is to fail fast if the corpus is colder than
#   the pilots, not to squeeze the queue).
#
# --mem: the pilot's 1,140,716 KB = 1.09 GiB is almost entirely the `pixi run` launcher -- the
#   same command run through the environment's python directly peaks at 116,096 KB = 0.11 GiB --
#   and this script uses `pixi run` too, so both are counted:
#     launcher            1.09 GiB   (independent of workers and of file count)
#     workers             0.11 GiB   (8 -> 16 workers roughly doubles a per-worker footprint that
#                                     is one open file handle and a few header buffers)
#     accumulated rows    0.70 GiB   (7,949 rows -> ~540,000 for the full corpus, at ~0.7 kB per
#                                     row-dict, plus the DataFrame built from them at the end)
#   1.3 * (1.09 + 0.11 + 0.70) = 2.5 GiB -> --mem=4G, rounded up to a whole GB.

set -euo pipefail

REPO=/scratch/gpfs/nc1514/FusionAIHub
IDEATE_DATA_ROOT="${IDEATE_DATA_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/ideate}"

cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is)"

srun pixi run -e ideate-cpu python -m ideate corpus scan \
    --workers 16 \
    --out "$IDEATE_DATA_ROOT/db/corpus_coverage.parquet"

echo "finished at $(date -Is)"
