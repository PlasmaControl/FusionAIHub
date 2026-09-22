#!/bin/bash
#SBATCH --job-name=sd-encode-batch
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/slurm/%j.out
# IGNITE frame codes for every shot of a list that has none yet, 4 GPUs on one node, each
# task taking a contiguous quarter of the list (`encode --chunk i --n-chunks 4`).
#   SHOT_DESIGN_DATA_ROOT=.../experiments/<name> LIST=<shot list name, e.g. stellar_1k> sbatch scripts/shot_design/encode_batch.sbatch
# --wait=0 --kill-on-bad-exit=0: by default srun kills the remaining tasks 60 s after the
# FIRST task exits (job 2937656 lost three quarter-lists that way when task 3 finished early).
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/shot_design/_stellar_common.sh"
LIST="${LIST:?set LIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks="$SLURM_NTASKS" --wait=0 --kill-on-bad-exit=0 bash -c '
  echo "task $SLURM_PROCID on $(hostname) gpu $CUDA_VISIBLE_DEVICES"
  exec "$0" -m shot_design encode --list "$1" --out "$2/frame_codes" --device cuda \
      --workers "${SLURM_CPUS_PER_TASK:-8}" --skip-existing --chunk "$SLURM_PROCID" --n-chunks "$SLURM_NTASKS"
' "$PY" "$LIST" "$ROOT"
echo "finished at $(date -Is)"
