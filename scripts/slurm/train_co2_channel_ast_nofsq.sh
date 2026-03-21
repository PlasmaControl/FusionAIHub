#!/bin/bash
#SBATCH --job-name=train_co2_channel_ast_nofsq
#SBATCH --output=logs/%j_train_co2_channel_ast_nofsq.out
#SBATCH --error=logs/%j_train_co2_channel_ast_nofsq.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Channel-AST without FSQ (continuous bottleneck) — ablation to measure
# how much of the ~0.029 L1 floor is due to FSQ quantization
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_channel_ast_fsq \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --fsq_levels \
    --frame_width 2 \
    --time_conv_kernel 7 \
    --d_model 256 \
    --n_tokens 0 \
    --batch_size 16 \
    --num_workers 2 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler none \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir runs/co2_channel_ast_nofsq \
    --resume
