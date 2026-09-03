#!/bin/bash
#SBATCH --job-name=gen_tokens
#SBATCH --output=logs/%j_gen_tokens_%a.out
#SBATCH --error=logs/%j_gen_tokens_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-8

# Each array task processes ~1000 shots
SHOTS_PER_TASK=1000
START=$((SLURM_ARRAY_TASK_ID * SHOTS_PER_TASK))
END=$((START + SHOTS_PER_TASK))

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /scratch/gpfs/nc1514/FusionAIHub

srun pixi run python scripts/training/generate_token_dataset.py \
    --checkpoint_dir runs \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --output_dir data/tokens \
    --stats_path data/preprocessing_stats.pt \
    --chunk_duration_s 0.2 \
    --n_fft 256 \
    --hop_length 256 \
    --batch_size 32 \
    --shot_start $START \
    --shot_end $END
