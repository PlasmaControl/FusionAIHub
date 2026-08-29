#!/bin/bash
# Two-lane local runner for ROCm training scripts on della-milan.
# Pins one worker lane to HIP_VISIBLE_DEVICES=0 and another to =1, runs them
# concurrently, captures per-script logs and an amd-smi utilization trace.
# Falls back to `sbatch` if an MI210 SLURM partition becomes available.
#
# Options (env vars):
#   CONTINUE_ON_ERR=1   keep going if one script fails (default: 1 in lane mode)
#   DRY_RUN=1           just print what would run
#   ONLY="a.sh b.sh"    space-separated list of script basenames to run (smoke set)
#   GPUS="0 1"          space-separated GPU IDs to use as lanes (default: "0 1")
#   AMD_SMI_INTERVAL=5  amd-smi sampling interval in seconds (default: 5)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
cd "$PROJECT_DIR"
mkdir -p logs

# Decide: sbatch (if available AND we're on a node with MI210 partition) or local lanes.
USE_SBATCH=0
if command -v sbatch >/dev/null 2>&1; then
    if sinfo -h -o "%f %N" 2>/dev/null | grep -qi "mi210\|gfx90a"; then
        USE_SBATCH=1
    fi
fi

# Build script list (full set, or filtered by ONLY).
all_scripts=()
for script in "${SCRIPT_DIR}"/train_*.sh; do
    name=$(basename "$script")
    if [ -n "${ONLY:-}" ]; then
        case " ${ONLY} " in
            *" ${name} "*) all_scripts+=("$script") ;;
        esac
    else
        all_scripts+=("$script")
    fi
done

if [ "${#all_scripts[@]}" -eq 0 ]; then
    echo "[submit_all] no scripts matched (ONLY='${ONLY:-}')"
    exit 1
fi

if [ "$USE_SBATCH" -eq 1 ]; then
    echo "[submit_all] sbatch + MI210 partition detected -> submitting via sbatch"
    failures=()
    for script in "${all_scripts[@]}"; do
        name=$(basename "$script")
        echo "[submit_all] sbatch $name"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY_RUN: would sbatch $script"
            continue
        fi
        sbatch "$script" || failures+=("$name")
    done
    if [ "${#failures[@]}" -gt 0 ]; then
        echo "[submit_all] sbatch submission failures: ${failures[*]}"
        exit 1
    fi
    exit 0
fi

# --- Local two-lane mode ---
GPUS="${GPUS:-0 1}"
read -r -a gpu_arr <<<"$GPUS"
n_lanes=${#gpu_arr[@]}
if [ "$n_lanes" -lt 1 ]; then
    echo "[submit_all] no GPU lanes configured"; exit 1
fi

echo "[submit_all] no MI210 SLURM partition; running locally on $(hostname)"
echo "[submit_all] lanes: ${n_lanes} (GPUs: $GPUS)"
echo "[submit_all] scripts queued: ${#all_scripts[@]}"

ts=$(date +%Y%m%d_%H%M%S)
amd_csv="logs/amd_smi_${ts}.csv"

# Background amd-smi sampler. amd-smi monitor prints a header + repeating rows;
# we redirect to a file and tag rows by timestamp via the -w flag.
interval="${AMD_SMI_INTERVAL:-5}"
amd_pid=""
if [ "${DRY_RUN:-0}" != "1" ] && command -v amd-smi >/dev/null 2>&1; then
    ( amd-smi monitor -w "$interval" --csv >"$amd_csv" 2>&1 ) &
    amd_pid=$!
    echo "[submit_all] amd-smi sampler pid=$amd_pid -> $amd_csv (interval=${interval}s)"
fi

# Distribute scripts to lanes round-robin.
declare -a lane_lists
for i in "${!gpu_arr[@]}"; do
    lane_lists[$i]=""
done
idx=0
for script in "${all_scripts[@]}"; do
    lane=$(( idx % n_lanes ))
    lane_lists[$lane]+="${script}"$'\n'
    idx=$(( idx + 1 ))
done

run_lane() {
    local gpu_id="$1"
    local list="$2"
    local lane_log="logs/lane_gpu${gpu_id}_${ts}.log"
    : >"$lane_log"
    while IFS= read -r script; do
        [ -z "$script" ] && continue
        local name; name=$(basename "$script")
        local script_log="logs/${name%.sh}_gpu${gpu_id}_${ts}.log"
        {
            echo ""
            echo "========================================"
            echo "[lane gpu${gpu_id}] $(date -Is) START $name"
            echo "========================================"
        } | tee -a "$lane_log"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY_RUN: HIP_VISIBLE_DEVICES=$gpu_id bash $script" | tee -a "$lane_log"
            continue
        fi
        # Run script with GPU pin; capture both stdout and stderr per-script.
        if HIP_VISIBLE_DEVICES="$gpu_id" bash "$script" >"$script_log" 2>&1; then
            echo "[lane gpu${gpu_id}] $(date -Is) OK    $name -> $script_log" | tee -a "$lane_log"
        else
            rc=$?
            echo "[lane gpu${gpu_id}] $(date -Is) FAIL($rc) $name -> $script_log" | tee -a "$lane_log"
            if [ "${CONTINUE_ON_ERR:-1}" != "1" ]; then
                echo "[lane gpu${gpu_id}] stopping (CONTINUE_ON_ERR=0)" | tee -a "$lane_log"
                return 1
            fi
        fi
    done <<<"$list"
    echo "[lane gpu${gpu_id}] DONE" | tee -a "$lane_log"
}

# Launch lanes in parallel.
lane_pids=()
for i in "${!gpu_arr[@]}"; do
    gpu_id="${gpu_arr[$i]}"
    list="${lane_lists[$i]}"
    run_lane "$gpu_id" "$list" &
    lane_pids+=($!)
    echo "[submit_all] lane gpu${gpu_id} pid=${lane_pids[-1]}"
done

# Wait for lanes; collect exit statuses.
overall_rc=0
for pid in "${lane_pids[@]}"; do
    if ! wait "$pid"; then
        overall_rc=1
    fi
done

# Stop amd-smi sampler.
if [ -n "$amd_pid" ]; then
    kill "$amd_pid" 2>/dev/null || true
    wait "$amd_pid" 2>/dev/null || true
fi

echo ""
echo "========================================"
# Summarize per-script outcomes by scanning lane logs.
fail_count=$(grep -hE "FAIL\(" logs/lane_gpu*_${ts}.log 2>/dev/null | wc -l | tr -d ' ')
ok_count=$(grep -hE " OK    " logs/lane_gpu*_${ts}.log 2>/dev/null | wc -l | tr -d ' ')
echo "[submit_all] timestamp ${ts}"
echo "[submit_all] OK: ${ok_count}  FAIL: ${fail_count}  total queued: ${#all_scripts[@]}"
if [ "$fail_count" -gt 0 ]; then
    echo "[submit_all] failures:"
    grep -hE "FAIL\(" logs/lane_gpu*_${ts}.log 2>/dev/null | sed 's/^/  /'
fi
[ -f "$amd_csv" ] && echo "[submit_all] amd-smi trace: $amd_csv"

exit "$overall_rc"
