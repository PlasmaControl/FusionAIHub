#!/bin/bash
#SBATCH -A fus187
#SBATCH -J bench_plugin
#SBATCH -o logs/%j_benchmark_plugin_perf.out
#SBATCH -e logs/%j_benchmark_plugin_perf.err
#SBATCH -t 1:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# AWS-OFI-NCCL plugin perf benchmark — 8-node DDP, identical workload
# (100 training steps, fresh-init, no checkpoint resume), once WITH the
# plugin (default after common.sh) and once WITHOUT (LD_LIBRARY_PATH
# stripped + NCCL_NET_PLUGIN=none). Compares step times to measure the
# collective-throughput benefit on real allreduce of gradient tensors.
#
# Per-run cost: ~3 min init + ~10 min for 100 steps ≈ 13 min.
# Two runs sequentially = ~26 min, well under the 1h debug cap.
#
# Submit:
#   sbatch --qos=debug scripts/slurm_frontier/benchmark_plugin_perf.sh
#
# Outputs:
#   logs/<jobid>_benchmark_plugin_perf_with_plugin.{out,err}
#   logs/<jobid>_benchmark_plugin_perf_without_plugin.{out,err}
# Final comparison summary printed to the main .out at end of job.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs

# Distinct port from production (29500) and other eval phases (29520-23).
export MASTER_PORT=29550
source scripts/slurm_frontier/_frontier_common.sh

BENCH_LOG_BASE="logs/${SLURM_JOB_ID}_benchmark_plugin_perf"
BENCH_CKPT_DIR="/tmp/bench_plugin_${SLURM_JOB_ID}"     # NOT production dir!
mkdir -p "${BENCH_CKPT_DIR}"

# Identical hyperparameters for both runs. No --resume_checkpoint → fresh
# init keeps both runs starting at the same model state and avoids any
# interaction with production's _latest.pt at /lustre/.../e2e_stage1/.
COMMON_ARGS=(
    --data_dir /lustre/orion/fus187/proj-shared/foundation_model
    --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt
    --val_fraction 0.1
    --seed 42
    --chunk_duration_s 0.05
    --prediction_horizon_s 0.05
    --step_size_s 0.01
    --warmup_s 1.0
    --d_model 256
    --n_layers 26
    --n_heads 8
    --dropout 0.1
    --lr 5e-4
    --min_lr 1e-6
    --warmup_steps 4000
    --weight_decay 0.1
    --grad_clip 5.0
    --batch_size 64
    --num_workers 6
    --max_steps 100
    --log_every 10
    --val_every 99999
    --val_max_batches 1
    --use_video tangtv
    --use_spectro ece co2 bes
    --no_amp_val
    --checkpoint_dir "${BENCH_CKPT_DIR}"
)

# Per-node sampler (shared between both runs).
SAMPLER_LOG="${BENCH_LOG_BASE}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

PLUGIN_PATH="$HOME/aws-ofi-nccl/install/lib"
echo "=== Pre-benchmark env (should show plugin loaded) ==="
echo "  LD_LIBRARY_PATH first entry: ${LD_LIBRARY_PATH%%:*}"
echo "  Plugin lib present:          $(test -f $PLUGIN_PATH/libnccl-net.so && echo YES || echo NO)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# RUN 1 — WITH plugin (default after common.sh)
# ─────────────────────────────────────────────────────────────────────
echo "=== Run 1: WITH AWS-OFI-NCCL plugin ($(date '+%H:%M:%S')) ==="
T0=$(date +%s)
NCCL_DEBUG=INFO srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" \
        -c "$SLURM_CPUS_PER_TASK" --gpus-per-task=1 --gpu-bind=closest \
        scripts/slurm_frontier/_srun_rank_wrapper.sh \
        scripts/training/train_e2e_stage1.py \
        "${COMMON_ARGS[@]}" \
        > "${BENCH_LOG_BASE}_with_plugin.out" \
        2> "${BENCH_LOG_BASE}_with_plugin.err"
T1=$(date +%s)
WITH_PLUGIN_S=$((T1 - T0))
echo "Run 1 complete at $(date '+%H:%M:%S'), wall=${WITH_PLUGIN_S}s"
echo ""

# Clean scratch dir between runs so the second doesn't accidentally
# resume / load partial state from the first.
rm -rf "${BENCH_CKPT_DIR}"/*

# ─────────────────────────────────────────────────────────────────────
# RUN 2 — WITHOUT plugin (strip from LD_LIBRARY_PATH + force-off env)
# ─────────────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//${PLUGIN_PATH}:/}"
export NCCL_NET_PLUGIN=none

echo "=== Run 2: WITHOUT plugin ($(date '+%H:%M:%S')) ==="
echo "  LD_LIBRARY_PATH first entry: ${LD_LIBRARY_PATH%%:*}"
echo "  NCCL_NET_PLUGIN:             ${NCCL_NET_PLUGIN}"
T0=$(date +%s)
NCCL_DEBUG=INFO srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" \
        -c "$SLURM_CPUS_PER_TASK" --gpus-per-task=1 --gpu-bind=closest \
        scripts/slurm_frontier/_srun_rank_wrapper.sh \
        scripts/training/train_e2e_stage1.py \
        "${COMMON_ARGS[@]}" \
        > "${BENCH_LOG_BASE}_without_plugin.out" \
        2> "${BENCH_LOG_BASE}_without_plugin.err"
T1=$(date +%s)
WITHOUT_PLUGIN_S=$((T1 - T0))
echo "Run 2 complete at $(date '+%H:%M:%S'), wall=${WITHOUT_PLUGIN_S}s"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Comparison summary
# ─────────────────────────────────────────────────────────────────────
echo "=== Benchmark summary ==="
printf "  WITH plugin:    %5d s wall  ←  uses libfabric/cxi via aws-ofi-nccl v10\n" "$WITH_PLUGIN_S"
printf "  WITHOUT plugin: %5d s wall  ←  TCP socket via hsn0\n" "$WITHOUT_PLUGIN_S"
if [ "$WITHOUT_PLUGIN_S" -gt 0 ]; then
    awk -v a="$WITH_PLUGIN_S" -v b="$WITHOUT_PLUGIN_S" \
        'BEGIN{printf "  Speedup ratio: %.3fx (with / without = %d / %d)\n", a/b, a, b}'
fi
echo ""
echo "Per-step timestamps for direct comparison:"
for variant in with_plugin without_plugin; do
    echo "--- $variant (step N at HH:MM:SS) ---"
    grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}.*step [0-9]+/100" \
        "${BENCH_LOG_BASE}_${variant}.err" 2>/dev/null \
        | awk '{print $1, $NF}' | head -12
done
echo ""
echo "Confirm plugin loaded in run 1:"
grep -E "NET/Plugin: Loaded|NET/OFI Selected provider" \
    "${BENCH_LOG_BASE}_with_plugin.err" 2>/dev/null | head -2
echo ""
echo "Confirm plugin NOT loaded in run 2:"
grep -E "NET/Plugin|NET/Socket : Using|NCCL_NET_PLUGIN" \
    "${BENCH_LOG_BASE}_without_plugin.err" 2>/dev/null | head -5
