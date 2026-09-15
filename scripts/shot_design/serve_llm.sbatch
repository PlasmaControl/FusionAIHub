#!/usr/bin/env bash
#SBATCH -J shot-design-llm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/slurm-%j.out
# Submit from the checkout root. Create <data_root>/llm before sbatch opens its output.
# For an alternate data_root, also override sbatch --output to <data_root>/llm/slurm-%j.out.
# Binds the GPU node hostname so CPU clients on the login node can reach the endpoint.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec bash scripts/shot_design/serve_llm.sh --host "$(hostname -s)" --port 11434
