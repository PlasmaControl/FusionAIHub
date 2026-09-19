#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_bp_cases
#SBATCH -q debug
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH -o logs/%j_ignite_bp_cases.log

# Actuator-counterfactual case studies on the BAND-POWER dynamics model.
#
# Why this family: at step 13500 the 14-modality model predicts `mhr` AT OR BELOW a
# constant-token baseline on all three case shots, so AE and tearing-mode questions are
# unanswerable on it. bp128_d512L8 (step 11000) trains on `mhr` + `co2` only -- the MHD
# channel -- and does show skill there.
#
# SCOPE LIMIT: this family has NO `filterscopes`, so it cannot address ELMs at all. D-alpha
# stays with the 14-modality model (which genuinely beats the trivial baseline there,
# 0.590 vs 0.198 on 190735). The two families are complementary, not interchangeable.
#
# Each non-real arm is paired against the real-actuator arm at the SAME seed, so any
# difference is the actuator edit rather than sampling. Metrics are restricted to the window
# where the diagnostic actually has data -- mhr is a fixed 2**21+1 sample / 4.194 s digitiser
# record, so a 5 s rollout runs off the end of it and the cached codes there encode padding.
#
# Run on a COMPUTE node: a rollout on the shared login-node GPU hung it (2026-08-14).

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/evaluation/slurm/_eval_env.sh

CKPT=/lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_d512L8/dynamics_latest.pt
BP_ROOT=/lustre/orion/fus187/proj-shared/models/ignite_bp128
CACHE=data/outputs/ignite_bp_cache_ext
OUT=data/outputs/ignite_bp_cases
MAIN_DATA=/lustre/orion/fus187/proj-shared/foundation_model
ADD_DATA=/lustre/orion/fus187/proj-shared/additional_data

mkdir -p "$OUT" logs

# case | shot | data-dir-for-record-end | arms
# TM is already complete (job 5268277). AE lost its arms to a missing donor: encoding
# main-corpus shots runs ~5-7 s/frame vs 10-47 ms for the additional_data shots, so 200728
# would have cost ~48 more min of shared login-node CPU and was dropped rather than waited on.
# AE therefore runs with the one real run-day donor that finished.
CASES=(
  "ae_200729|200729|$MAIN_DATA|real,donor:200730,group_zero:pinj"
)

NGPU=8
for i in "${!CASES[@]}"; do
  IFS='|' read -r TAG SHOT DDIR ARMS <<< "${CASES[$i]}"
  g=$((i % NGPU))
  echo "[gcd $g] $TAG  arms=$ARMS"
  (
    export ROCR_VISIBLE_DEVICES=$g
    export PYTHONPATH=src
    python scripts/evaluation/ignite_bp_cases.py \
      --ckpt "$CKPT" --bp-root "$BP_ROOT" --cache-dir "$CACHE" \
      --shot "$SHOT" --arms "$ARMS" --data-dir "$DDIR" \
      --out "$OUT/$TAG" \
      > "logs/${SLURM_JOB_ID}_${TAG}.log" 2>&1 \
      && echo "[gcd $g] OK   $TAG" || echo "[gcd $g] FAIL $TAG"
  ) &
done

wait

echo
echo "================ SUMMARY ================"
for i in "${!CASES[@]}"; do
  IFS='|' read -r TAG SHOT DDIR ARMS <<< "${CASES[$i]}"
  if [ -f "$OUT/$TAG/bp_cases_traces.npz" ]; then
    echo "OK   $TAG"
    grep -h "divergence_vs_real" "logs/${SLURM_JOB_ID}_${TAG}.log" 2>/dev/null | sed 's/^/       /'
    grep -h "data-valid frames" "logs/${SLURM_JOB_ID}_${TAG}.log" 2>/dev/null | sed 's/^/       /'
  else
    echo "FAIL $TAG"
    tail -n 10 "logs/${SLURM_JOB_ID}_${TAG}.log" 2>/dev/null | sed 's/^/       /'
  fi
done
