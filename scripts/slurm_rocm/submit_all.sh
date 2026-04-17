#!/bin/bash
# Serial local runner for all ROCm training scripts on della-milan.
# Falls back to `sbatch` if an MI210 SLURM partition becomes available.
#
# Options (env vars):
#   CONTINUE_ON_ERR=1   keep going if one script fails (default: stop)
#   DRY_RUN=1           just print what would run
#   HIP_VISIBLE_DEVICES=N   pin to a specific MI210 (default: 0)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
cd "$PROJECT_DIR"
mkdir -p logs

# Decide: sbatch (if available AND we're on a node with MI210 partition) or bash.
USE_SBATCH=0
if command -v sbatch >/dev/null 2>&1; then
    if sinfo -h -o "%f %N" 2>/dev/null | grep -qi "mi210\|gfx90a"; then
        USE_SBATCH=1
    fi
fi

if [ "$USE_SBATCH" -eq 1 ]; then
    echo "[submit_all] sbatch available and MI210 partition found -> using sbatch"
else
    echo "[submit_all] no MI210 SLURM partition; running serially on $(hostname)"
fi

failures=()
for script in "${SCRIPT_DIR}"/train_*.sh; do
    name=$(basename "$script")
    echo ""
    echo "========================================"
    echo "[submit_all] $name"
    echo "========================================"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN: would run $script"
        continue
    fi
    if [ "$USE_SBATCH" -eq 1 ]; then
        sbatch "$script" || failures+=("$name")
    else
        if bash "$script"; then
            echo "[submit_all] $name OK"
        else
            rc=$?
            echo "[submit_all] $name FAILED (exit $rc)"
            failures+=("$name")
            if [ "${CONTINUE_ON_ERR:-0}" != "1" ]; then
                echo "[submit_all] stopping (set CONTINUE_ON_ERR=1 to continue past failures)"
                break
            fi
        fi
    fi
done

echo ""
echo "========================================"
if [ "${#failures[@]}" -eq 0 ]; then
    echo "[submit_all] all scripts completed"
else
    echo "[submit_all] ${#failures[@]} failure(s):"
    printf '  %s\n' "${failures[@]}"
    exit 1
fi
