#!/bin/bash
# Submit all 20 ROCm training jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "${SCRIPT_DIR}"/train_*.sh; do
    echo "Submitting $(basename "$script")"
    sbatch "$script"
done
