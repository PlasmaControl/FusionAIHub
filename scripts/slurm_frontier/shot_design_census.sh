#!/bin/bash
#SBATCH -A fus187
#SBATCH -p batch
#SBATCH -J sd-census
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%j.out
# Census of $SHOT_DESIGN_CORPUS: one header-only read per shot file, in parallel, into
# db/corpus_coverage.parquet -- the table every later selection and coverage question is
# answered from. That exact name is not decoration: it is what `corpus scan --out`,
# `select` and `coverage` default to (src/shot_design/cli.py), so a census written as
# `census.parquet` would be invisible to every consumer.
# CPU-only and I/O-bound; one whole node, 48 of its 56 cores to the workers.
# sbatch runs a spool COPY of this file, so `dirname "$0"` is not the repo; the submit
# directory is (every wrapper is submitted from the repo root). Local runs fall back.
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/slurm_frontier/_shot_design_common.sh"
srun -n1 -c56 "$PY" -m shot_design corpus scan --workers 48 --out "$ROOT/db/corpus_coverage.parquet"
"$PY" -m shot_design corpus summary "$ROOT/db/corpus_coverage.parquet"
