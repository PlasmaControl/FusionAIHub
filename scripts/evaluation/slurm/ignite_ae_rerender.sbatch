#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_ae_rerender
#SBATCH -q debug
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH -o logs/%j_ignite_ae_rerender.log

# Two remaining pieces, one allocation, because the two questions need DIFFERENT models.
#
#   ELM  -> the 14-modality model (prod_d512L8). D-alpha lives in `filterscopes`, which the
#           band-power family does not have at all. This is also the one modality where that
#           model clearly beats a constant-token baseline (0.590 vs 0.198 on 190735).
#   TM   -> the band-power model (bp128_d512L8), now at step 20000 vs the 11000 the first
#           TM answer was produced with. `mhr` is degenerate on the 14-modality model.
#
# The ELM arms also re-render `ignite_cases` with the figure fixes:
#   * video GT is raw camera counts while the decode is standardized, so the prediction was
#     painted with GT's colour limits and came out SOLID BLACK; now z-scored when the two
#     spaces disagree by >5x.
#   * spectrograms drew with interpolation="none" on a ~512 x 9600 image squeezed into ~1600
#     px, which aliased it into blocks; now antialiased.
#   * 12 pt rotated y-labels were taller than their own row and collided; now shrunk.
#   * PNG went from 200 to 400 dpi (the PDF keeps its exact \textwidth geometry).
#   * out_dir is pre-created so the codes_*.npz archive stops failing -- that archive is what
#     lets the D-alpha panel be rendered afterwards WITHOUT another rollout.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/evaluation/slurm/_eval_env.sh

E2E_CKPT=/lustre/orion/fus187/proj-shared/models/ignite_production/runs/prod_d512L8/dynamics_latest.pt
E2E_CACHE=data/outputs/ignite_cache_ext
CODEC_TMPL="/lustre/orion/fus187/proj-shared/models/ignite_codecs_current/{m}/codec_best.pt"
BP_CKPT=/lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_d512L8/dynamics_latest.pt
BP_ROOT=/lustre/orion/fus187/proj-shared/models/ignite_bp128
BP_CACHE=data/outputs/ignite_bp_cache_ext
ADD_DATA=/lustre/orion/fus187/proj-shared/additional_data
OUT=data/outputs/ignite_cases
BP_OUT=data/outputs/ignite_bp_cases

export IGNITE_FIG_SPECTRO=mhr,ece
mkdir -p "$OUT" "$BP_OUT" logs

# ---- GCDs 1-7: the ELM case on the 14-modality model ---------------------------------- #
# 190735 is the shot the user asked for; 203266 is the clean-data counterpart from the
# 777-shot core set (all modalities live, ELM rate 103 Hz, sharp RMP turn-on inside the
# predicted region). ech_power is ~0 in both, so group_zero:ech_power edits nothing and its
# divergence is the noise floor -- without it a small response is uninterpretable.
ELM_RUNS=(
  "ae|200729|real"
  "ae|200729|donor:200730"
  "ae|200729|donor:200728"
  "ae|200729|group_zero:pinj"
)
for i in "${!ELM_RUNS[@]}"; do
  g=$(( (i % 7) + 1 ))
  IFS='|' read -r CASE SHOT MODE <<< "${ELM_RUNS[$i]}"
  TAG="${CASE}_${SHOT}_$(echo "$MODE" | tr ':' '-')"
  (
    export ROCR_VISIBLE_DEVICES=$g PYTHONPATH=src
    mkdir -p "$OUT/$TAG"
    python -m tokamak_foundation_model.ignite.eval_dynamics \
      --ckpt "$E2E_CKPT" --cache_dir "$E2E_CACHE" --codec_tmpl "$CODEC_TMPL" \
      --shot "$SHOT" --actuator_mode "$MODE" --split_seed 42 --seed 0 \
      --out_dir "$OUT/$TAG" \
      > "logs/${SLURM_JOB_ID}_${TAG}.log" 2>&1 \
      && echo "[gcd $g] OK   $TAG" || echo "[gcd $g] FAIL $TAG"
  ) &
done

wait

echo
echo "================ SUMMARY ================"
if false; then
  echo "OK   tm_199597_step20000"
  grep -h "ckpt step\|divergence_vs_real\|data-valid" \
    "logs/${SLURM_JOB_ID}_tm_199597_step20000.log" 2>/dev/null | sed 's/^/       /'
else
  echo "FAIL tm_199597_step20000"
  tail -n 8 "logs/${SLURM_JOB_ID}_tm_199597_step20000.log" 2>/dev/null | sed 's/^/       /'
fi
for i in "${!ELM_RUNS[@]}"; do
  IFS='|' read -r CASE SHOT MODE <<< "${ELM_RUNS[$i]}"
  TAG="${CASE}_${SHOT}_$(echo "$MODE" | tr ':' '-')"
  if [ -f "$OUT/$TAG/eval_metrics.json" ]; then
    NA=$(ls "$OUT/$TAG"/codes_*.npz 2>/dev/null | wc -l)
    echo "OK   $TAG   (code archives: $NA)"
  else
    echo "FAIL $TAG"
    tail -n 6 "logs/${SLURM_JOB_ID}_${TAG}.log" 2>/dev/null | sed 's/^/       /'
  fi
done
