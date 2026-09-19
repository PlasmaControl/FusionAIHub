#!/bin/bash
#SBATCH -A fus187
#SBATCH -p extended
#SBATCH -J sd-encode
#SBATCH -N 1
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 06:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%A_%a.out
# IGNITE frame-code caches for a shot list: the per-shot seeds a rollout starts from.
# Each array task takes one contiguous slice of the SORTED list (--chunk/--n-chunks), so
# the eight tasks walk different regions of the corpus directory instead of competing for
# the same 2-5 GB files. One GCD and 7 cores each -- one MI250X node's worth in total.
# The work is I/O-bound, not GPU-bound (Stellar measured ~10 % GPU utilisation ceiling),
# so more GCDs per task would buy nothing. `--skip-existing` makes a task that hits the
# wall clock resubmittable. Set N_CHUNKS and the --array range together.
# `extended` for the same reason as shot_design_build.sh: `batch` caps a one-node job at 2 h.
source "$(dirname "$0")/_shot_design_common.sh"

# GPU samples for the utilisation gate: OLCF has no `jobstats`, so this file is the only
# record of how much of the GCD the job used. Named after the id `sacct` reports -- for an
# array element that is <arrayjobid>_<taskid>, not $SLURM_JOB_ID, which is what
# labeler.jobstats.read_gpu_samples looks for.
JOB_TAG="${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
bash "$REPO/scripts/slurm_frontier/_gpu_sampler.sh" \
    "$ROOT/runs/slurm/${JOB_TAG:-${SLURM_JOB_ID:-local}}.gpu.csv" &
trap 'kill %1 2>/dev/null || true' EXIT

srun "$PY" -m shot_design encode --list "${SHOT_LIST:-recommender_frontier_v1}" \
    --out "$ROOT/frame_codes" --device cuda --workers 6 --skip-existing \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks "${N_CHUNKS:-8}"
