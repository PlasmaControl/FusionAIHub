#!/bin/bash
#SBATCH --job-name=ae_stats
#SBATCH --output=logs/%j_ae_stats.out
#SBATCH --error=logs/%j_ae_stats.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/compute_ae_token_stats.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model/ \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt \
    --ae_checkpoint_dir /projects/EKOLEMEN/foundation_model/ \
    --output_path /projects/EKOLEMEN/foundation_model/ae_token_stats.pt \
    --batch_size 512 \
    --num_workers 4
