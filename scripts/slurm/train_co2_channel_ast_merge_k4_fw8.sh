#!/bin/bash
#SBATCH --job-name=co2_chast_merge_k4_fw8
#SBATCH --output=logs/%j_co2_chast_merge_k4_fw8.out
#SBATCH --error=logs/%j_co2_chast_merge_k4_fw8.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Channel-AST-Merge: k=4 pool queries, frame_width=8, no FSQ
# Token count: k × ceil(1954/8) = 4 × 244 = 976 tokens × 256 d_model
# For CO2 (C=4, k=4): no channel compression — sanity check vs no-merge baseline
# For ECE (C=40, k=4): 10× channel compression — same 976 token count
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_channel_ast_merge \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --frame_width 8 \
    --n_merge_queries 4 \
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
    --checkpoint_dir runs/co2_channel_ast_merge_k4_fw8 \
    --resume
