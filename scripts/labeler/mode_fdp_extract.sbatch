#!/bin/bash
#SBATCH --job-name=mode-fdp-extract
#SBATCH --account=nstx-u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# Staged FDP views -> omnimode's inputs (scripts/labeler/mode_all_shots.py extract), no network:
# <root>/corpus/<shot>_processed.h5 (Mirnov + Ip) and <root>/omnimode/inputs/{efit02er,efit02,
# efitrt2,rot}/<shot>.npz. The stage subcommand submits it per batch with LIST=<root>/batches/bNNNN.txt.
#   LIST=... sbatch scripts/labeler/mode_fdp_extract.sbatch
# --time is above 30 min on purpose: shorter jobs land in QOS test, 5 queued jobs per user.
set -euo pipefail
source "${REPO:-/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub}/scripts/labeler/mode_fdp_env.sh"
unset BEARER_TOKEN
cd "$REPO"
"$REPO/.pixi/envs/fdp/bin/python" -u scripts/labeler/mode_all_shots.py --root "$MODE_FDP_ROOT" \
    extract --list "$LIST" --workers "${WORKERS:-8}"
echo "finished at $(date -Is)"
