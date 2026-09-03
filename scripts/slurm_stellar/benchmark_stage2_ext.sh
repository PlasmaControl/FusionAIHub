#!/bin/bash
#SBATCH --job-name=e2e_bench
#SBATCH --output=logs/%j_e2e_bench.out
#SBATCH --error=logs/%j_e2e_bench.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Measure wall time per step for extended-Stage-2 at batch=256, K=80,
# gradient checkpointing every step, upfront async per-modality H2D (pin
# is preserved, transfers overlap with compute). Earlier benchmark 2717509
# logged 28 s/step at batch=128 with blocking transfers and only 27% GPU
# util; this run measures the pin-fix + batch-256 combined effect.
#
# 150 training steps with validation fired once at step 100
# (--val_max_batches 1) to also verify the validation memory fixes
# (collect_history=False + per-step free) hold at K=80.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

INIT="runs/e2e_stage2_delta/e2e_stage2_delta_best.pt"
if [ ! -f "$INIT" ]; then
    echo "Stage 2b best not found; falling back to Stage 2 best"
    INIT="runs/e2e_stage2/e2e_stage2_best.pt"
fi
echo "Init checkpoint: $INIT"

DATA_DIR=/scratch/gpfs/EKOLEMEN/foundation_model
STATS=/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt
BENCH_ROOT=runs/e2e_bench/$SLURM_JOB_ID

COMMON="--data_dir $DATA_DIR --stats_path $STATS \
    --init_checkpoint $INIT \
    --val_fraction 0.05 --seed 42 \
    --chunk_duration_s 0.05 --step_size_s 0.01 --warmup_s 1.0 \
    --d_model 256 --n_layers 8 --n_heads 8 --dropout 0.1 \
    --mae_weight 1.0 --cos_weight 0.3 --mag_weight 0.1 --min_disp_norm 0.01 \
    --lr 1e-5 --min_lr 1e-7 --warmup_steps 10 --weight_decay 0.01 --grad_clip 5.0 \
    --num_workers 8 \
    --max_steps 150 --log_every 25 \
    --val_every 100 --val_max_batches 1 \
    --max_files 200"

# ── Production config: batch 128, K=80, ckpt every step ─────────────
echo ""
echo "================ CONFIG: batch=256 K=80 ckpt=1 pin-fix ================"
srun pixi run python ../training/train_e2e_stage2_extended.py $COMMON \
    --checkpoint_dir $BENCH_ROOT/b256_k80_ckpt1_pin \
    --batch_size 256 \
    --curriculum_Ks 80 --block_steps 1000 \
    --grad_checkpoint_every 1

echo ""
echo "================ BENCHMARK DONE ================"
echo "Parse the .err log — look at step timestamps to compute s/step."