#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_fsq_overfit_test
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --mail-user=ps9551@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Real-shot IGNITE overfit gates on ONE GCD:
#   rung 1: single most-active-window recon-only overfit per modality (bes/co2/ece/mhr)
#   rung 2: whole-shot overfit + FULL stitched-spectrogram reconstruction
#           + GT|recon|diff comparison figure + metrics json per modality
#           (rungs 1+2: tests/ignite/test_fsq_overfit_realshot.py)
#   rung 3 (E2E=1): FULL-IGNITE overfit — ALL-family codecs (every modality incl.
#           filterscopes from the current production template, default v6, with
#           v5/frozen-manifest fallbacks) fine-tuned on the shot + MaskGIT ST-backbone
#           trained over their codes + rollout + physical-units GT-vs-pred figure
#           (tests/ignite/test_e2e_overfit_realshot.py)
# Figures/metrics land in ${OUT_DIR}.
#
# Env overrides:
#   SHOT      shot number                      (default 200729)
#   E2E       "1" runs rung 3 INSTEAD of rungs 1+2          (default 0)
#   STEPS     rung-1 Adam steps / rung-3 backbone steps     (default 1500)
#   EPOCHS    rung-2 epochs over the shot      (default 300)
#   BS        rung-2 batch size                (default 16)
#   OUT_DIR   figure/metrics output dir        (default eval_runs/fsq_overfit_${SHOT},
#                                               E2E: eval_runs/ignite_e2e_overfit_${SHOT})
#   PYTEST_K  pytest -k filter, e.g. "ece" or "whole_shot" (default: all tests)
#   IGNITE_E2E_*  rung-3 knobs pass through the environment (see the test's docstring)

set -u
cd /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
source scripts/slurm_frontier/_frontier_common.sh
# single process, no DDP: let the CPU-side STFT window building use the allocated cores
# (overrides _frontier_common's OMP_NUM_THREADS=1, which is tuned for 8-rank DDP nodes).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SHOT="${SHOT:-200729}"
E2E="${E2E:-0}"
STEPS="${STEPS:-1500}"
EPOCHS="${EPOCHS:-300}"
BS="${BS:-16}"
PYTEST_K="${PYTEST_K:-}"

export IGNITE_OVERFIT_SHOT="${SHOT}"
if [ "${E2E}" = "1" ]; then
    # rung 3: full-IGNITE overfit (all-family codecs + MaskGIT backbone)
    OUT_DIR="${OUT_DIR:-eval_runs/ignite_e2e_overfit_${SHOT}}"
    export IGNITE_E2E=1
    export IGNITE_E2E_STEPS="${STEPS}"
    export IGNITE_E2E_OUT="${OUT_DIR}"
    PYTEST_ARGS=(tests/ignite/test_e2e_overfit_realshot.py -s -q)
else
    OUT_DIR="${OUT_DIR:-eval_runs/fsq_overfit_${SHOT}}"
    export IGNITE_OVERFIT_STEPS="${STEPS}"
    export IGNITE_FULLSHOT=1
    export IGNITE_FULLSHOT_EPOCHS="${EPOCHS}"
    export IGNITE_FULLSHOT_BS="${BS}"
    export IGNITE_FULLSHOT_OUT="${OUT_DIR}"
    PYTEST_ARGS=(tests/ignite/test_fsq_overfit_realshot.py -s -q)
fi

mkdir -p logs "${OUT_DIR}"

# array form so a -k expression with spaces ("freqgrad or fsq6") survives word-splitting
[ -n "${PYTEST_K}" ] && PYTEST_ARGS+=(-k "${PYTEST_K}")

echo "[ignite_fsq_overfit_test] host=$(hostname) shot=${SHOT} e2e=${E2E} steps=${STEPS} \
rung2_epochs=${EPOCHS} bs=${BS} out=${OUT_DIR} k=${PYTEST_K:-all}"

srun -N 1 -n 1 -c "${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 --gpu-bind=closest \
     python -m pytest "${PYTEST_ARGS[@]}"

echo "=== IGNITE FSQ OVERFIT TEST DONE (exit $?) ==="
