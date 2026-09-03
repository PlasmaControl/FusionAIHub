#!/bin/bash
#SBATCH --job-name=e2e_stage1_prof
#SBATCH --output=logs/%j_profile_stage1.out
#SBATCH --error=logs/%j_profile_stage1.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Short torch.profiler run (30 training steps) on the real Stage 1 pipeline.
# Output goes to runs/profile_stage1/<JOBID>/ — download trace.json and
# open in chrome://tracing (or Perfetto) to inspect the timeline.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUT_DIR=runs/profile_stage1/$SLURM_JOB_ID

srun pixi run python ../training/profile_stage1.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --output_dir "$OUT_DIR" \
    --batch_size 256 \
    --num_workers 8 \
    --profile_wait 5 \
    --profile_warmup 5 \
    --profile_active 20