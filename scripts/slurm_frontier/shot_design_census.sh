#!/bin/bash
#SBATCH -A fus187
#SBATCH -p batch
#SBATCH -J sd-census
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%j.out
# Census of $SHOT_DESIGN_CORPUS: one header-only read per shot file, in parallel, into
# db/census.parquet -- the table every later selection and coverage question is answered
# from. CPU-only and I/O-bound; one whole node, 48 of its 56 cores to the workers.
source "$(dirname "$0")/_shot_design_common.sh"
srun -n1 -c56 "$PY" -m shot_design corpus scan --workers 48 --out "$ROOT/db/census.parquet"
"$PY" -m shot_design corpus summary "$ROOT/db/census.parquet"
