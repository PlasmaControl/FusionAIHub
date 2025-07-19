#!/bin/bash
#SBATCH --job-name=dataprep   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=500GB               # memory per node
#SBATCH --time=10:00:00          # maximum time needed (HH:MM:SS)
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# Set environment
module purge
source .venv/bin/activate

# Run pipeline
srun python -m faith.preprocess --config-name signals