#!/bin/bash
#SBATCH --job-name=sd-build-batch
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/slurm/%j.out
# Build a shot list's database into a BATCH data root on Stellar, then join labeler's
# labels/events. Usage (from the repo root):
#   SHOT_DESIGN_DATA_ROOT=.../experiments/<name> LIST=<shot list> sbatch scripts/shot_design/build_batch.sbatch
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/shot_design/_stellar_common.sh"
LIST="${LIST:?set LIST to a configs/shot_design/shot_lists/<name>.yaml name}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
srun "$PY_CPU" -m shot_design build --list "$LIST" --reader corpus --no-encode --workers "${SLURM_CPUS_PER_TASK:-16}" "${@}"
srun "$PY_CPU" -m shot_design labels join --list "$LIST"
"$PY_CPU" -m shot_design coverage || true
echo "finished at $(date -Is)"
