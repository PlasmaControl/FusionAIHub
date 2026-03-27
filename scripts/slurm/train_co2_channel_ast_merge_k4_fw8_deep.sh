#!/bin/bash
#SBATCH --job-name=co2_chast_merge_k4_fw8_deep
#SBATCH --output=logs/%j_co2_chast_merge_k4_fw8_deep.out
#SBATCH --error=logs/%j_co2_chast_merge_k4_fw8_deep.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Channel-AST-Merge: k=4, fw=8, deep pool/expand (3 layers each)
# Ablation: does deeper cross-attention close the gap vs no-merge (0.0225)?
# Shallow pool (1 layer): 0.0267; target: ≤0.0225
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_channel_ast_merge \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --frame_width 8 \
    --n_merge_queries 4 \
    --n_pool_layers 3 \
    --n_expand_layers 3 \
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
    --checkpoint_dir runs/co2_channel_ast_merge_k4_fw8_deep \
    --resume
