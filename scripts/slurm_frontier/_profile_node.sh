#!/bin/bash
# Per-node profiling sidecar. Runs alongside the training srun via
# `srun --overlap` and samples GPU + CPU utilization once per second to CSV.
#
# Usage (spawned by the main shape script):
#   srun --overlap --jobid=$SLURM_JOB_ID -N $NODES -n $NODES \
#        --ntasks-per-node=1 --gpus-per-task=0 --cpus-per-task=2 \
#        scripts/slurm_frontier/_profile_node.sh <output_dir> &
#
# Outputs (one of each per node):
#   $1/<host>_gpu.csv     timestamp,card,gpu_pct,vram_pct,power_w,temp_c,sclk,mclk
#   $1/<host>_cpu.csv     mpstat -P ALL output (all cores)
#
# Sampling stops automatically when the job ends (SLURM SIGTERMs the task).

set -uo pipefail

OUT_DIR="${1:-profile}"
HOST="$(hostname -s)"
mkdir -p "$OUT_DIR"

GPU_LOG="$OUT_DIR/${HOST}_gpu.csv"
CPU_LOG="$OUT_DIR/${HOST}_cpu.csv"
META_LOG="$OUT_DIR/${HOST}_meta.txt"

# Capture node-level static metadata once.
{
    echo "host=$HOST"
    echo "date=$(date -Iseconds)"
    echo "kernel=$(uname -r)"
    echo "rocm_version=$(rocm-smi --showdriverversion 2>/dev/null | tail -1 || true)"
    echo "ngcd=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU\[' || true)"
    echo "ncpu=$(nproc)"
    rocm-smi --showproductname 2>/dev/null | head -20 || true
} > "$META_LOG"

# ─── GPU sampler (rocm-smi @ 1 Hz) ─────────────────────────────────────────
# rocm-smi --csv outputs a header + one row per GCD. We strip the per-call
# header and prepend a timestamp column so the file is one big CSV.
{
    echo "timestamp,card,gpu_use_pct,gpu_memory_use_pct,power_w,temp_edge_c,sclk_mhz,mclk_mhz"
    while :; do
        ts="$(date '+%Y-%m-%dT%H:%M:%S')"
        # rocm-smi --csv produces something like:
        # device,GPU use (%),GPU memory use (%),Average Graphics Package Power (W),Temperature (Sensor edge) (C),sclk clock speed: (MHz),mclk clock speed: (MHz)
        # card0,12,8,75.0,42.0,500Mhz,1600Mhz
        rocm-smi --csv \
                 --showuse --showmemuse --showpower --showtemp --showclocks 2>/dev/null \
            | awk -v ts="$ts" -F, '
                NR > 1 && $1 ~ /^card[0-9]+$/ {
                    print ts","$1","$2","$3","$4","$5","$6","$7
                }'
        sleep 1
    done
} > "$GPU_LOG" &
GPU_PID=$!

# ─── CPU sampler (mpstat @ 1 Hz, all cores) ────────────────────────────────
# mpstat keeps printing until SIGTERMed; one row per core per interval.
mpstat -P ALL 1 > "$CPU_LOG" &
CPU_PID=$!

# Forward SIGTERM/SIGINT to children so the samplers exit cleanly when SLURM
# tears down the step.
trap 'kill $GPU_PID $CPU_PID 2>/dev/null; exit 0' SIGTERM SIGINT

wait $GPU_PID $CPU_PID
