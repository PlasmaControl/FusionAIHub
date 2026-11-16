#!/bin/bash
#SBATCH --job-name=mode-fdp-publish
#SBATCH --account=nstx-u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# The MHD mode databases of docs/labeler/fdp_labels.md (mhd_n1, mhd_n2, mhd_n3) from the
# label store (scripts/labeler/mode_fdp_label.py publish --labeller redesign) into
# OUT_ROOT/<event>/RUN/. Submitted by mode_all_shots.py publish; copy OUT_ROOT to
# /projects/EKOLEMEN/tokamaksi/diiid/dev/ on the login node after.
#   STORE=... SHOTS=... OUT_ROOT=... RUN=... sbatch scripts/labeler/mode_fdp_publish.sbatch
set -euo pipefail
source "${REPO:-/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub}/scripts/labeler/mode_fdp_env.sh"
cd "$REPO"
export PYTHONPATH="$REPO/src"
"$REPO/.pixi/envs/fdp/bin/python" -u scripts/labeler/mode_fdp_label.py --labels "$STORE" \
    --shots "$SHOTS" publish --out-root "$OUT_ROOT" --run "$RUN" --labeller redesign
echo "finished at $(date -Is)"
