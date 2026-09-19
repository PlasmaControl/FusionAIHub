#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_rerender
#SBATCH -q debug
#SBATCH -t 00:25:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH -o logs/%j_ignite_rerender.log

# Re-draw figures from their ARCHIVED CODES -- no rollout, no model.
#
# eval_dynamics writes codes_<shot>_step<N>.npz beside its metrics exactly so a figure tweak
# need not cost a fresh 80-frame rollout, but nothing read them back until
# scripts/evaluation/ignite_rerender.py. That matters here: the ELM figures were drawn at
# 19:36-19:42 with only part of the figure fixes in place, and re-running the rollouts to get
# the rest would waste ~15 min of GPU and a queue slot for output that is already determined.
#
# Fixes these renders pick up that the 19:36 ones lack:
#   * short y-labels ($n_e$ core ...) -- the raw modality names are longer than a row is tall
#     once rotated, so they collided with the rows above and below at any font size
#   * video scale guard now also triggers on an OFFSET (GT in camera counts, mean ~35, vs a
#     standardized prediction, mean ~0); the old ratio-only test at >5x let 190735 through and
#     its prediction still rendered solid black
#   * constant ground truth is reported as "not scored" instead of emitting nRMSE 8.1e11 and
#     max|diff| 3.9e12 sigma, which is what dividing by sigma~0 produces
#
# Runs on a COMPUTE node: the shared login-node GPU hung twice today doing exactly this.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/evaluation/slurm/_eval_env.sh

CODEC_TMPL="/lustre/orion/fus187/proj-shared/models/ignite_codecs_current/{m}/codec_best.pt"
RUNS=(
  data/outputs/ignite_cases/elm_190735_real
  data/outputs/ignite_cases/elm_190735_donor-190736
  data/outputs/ignite_cases/elm_190735_group_zero-rmp
  data/outputs/ignite_cases/elm_190735_group_zero-ech_power
  data/outputs/ignite_cases/elmb_203266_real
  data/outputs/ignite_cases/elmb_203266_donor-203267
  data/outputs/ignite_cases/elmb_203266_group_zero-rmp
)

export IGNITE_FIG_SPECTRO=mhr,ece
NGPU=8
for i in "${!RUNS[@]}"; do
  R="${RUNS[$i]}"
  g=$((i % NGPU))
  TAG=$(basename "$R")
  (
    export ROCR_VISIBLE_DEVICES=$g PYTHONPATH=src
    python scripts/evaluation/ignite_rerender.py --run "$R" --codec-tmpl "$CODEC_TMPL" \
      > "logs/${SLURM_JOB_ID}_rr_${TAG}.log" 2>&1 \
      && echo "[gcd $g] OK   $TAG" || echo "[gcd $g] FAIL $TAG"
  ) &
done
wait

echo
echo "================ SUMMARY ================"
for R in "${RUNS[@]}"; do
  TAG=$(basename "$R")
  F=$(ls "$R"/dynamics_eval_*.png 2>/dev/null | head -1)
  if [ -n "$F" ]; then
    echo "OK   $TAG  ($(stat -c %y "$F" | cut -c1-16), $(stat -c %s "$F") bytes)"
  else
    echo "FAIL $TAG"; tail -n 6 "logs/${SLURM_JOB_ID}_rr_${TAG}.log" 2>/dev/null | sed 's/^/       /'
  fi
done
