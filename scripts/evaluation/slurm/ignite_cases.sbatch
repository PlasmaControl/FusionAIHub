#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_cases
#SBATCH -q debug
#SBATCH -t 01:10:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH -o logs/%j_ignite_cases.log

# Control-reference case studies on the IGNITE MaskGIT dynamics model.
#
# Each reference shot is rolled out (a) with its REAL actuators and (b) with a
# counterfactual control input. eval_dynamics pairs every non-real mode against the
# real-actuator rollout at the same seed automatically, so each run reports
# divergence_vs_real -- the effect size -- alongside the figure.
#
#   TM  199597  tearing modes          THE headline case: the only shot here that sat outside
#                                      every split until it was encoded 2026-08-14. Measured
#                                      TM band power (1-20 kHz) climbs 1.31 -> 1.74 peaking at
#                                      t ~ 2.75 s, i.e. the mode grows AFTER the model loses
#                                      real context at t = 1.0 s -- a true forecast test.
#                                      Caveats: mhr is a fixed 4.194 s digitiser record so it
#                                      covers only 76% of the 5 s window (frames >76 score
#                                      against encoded non-data); co2, ts_core_density and
#                                      both tangtv streams are static in this shot.
#   AE  200729  Alfven eigenmodes      NOT an independent test -- it is the development shot
#                                      (the usage-string example, pinned into val), so its
#                                      cleanliness is circular. Context only.
#   ELM 190735  RMP ELM suppression    bes/co2/tangtv_lower ABSENT (hence not in the core
#                                      set) but mhr + filterscopes live -- enough for ELMs.
#
# Counterfactual choice follows the user's framing: these are CONTROL REFERENCE shots,
# so the most physical "varied actuator" is the actuation of a NEIGHBOURING shot from the
# same run day (donor:<shot>), not a synthetic scaling -- cached actuators are per-shot
# z-scored, which makes gain edits a no-op by construction.
#
# Measured donor contrast (RMS delta / RMS ref over the predicted region):
#   200729 <- 200730  beams 1.51, rmp 0.002   (strong beam contrast; both in core set)
#   200729 <- 200728  beams 1.46, rmp 0.002   (the shot before; also core)
#   200729 <- 200731  donor has NO beams      -> a real-trajectory beams-off variant
#   190735 <- 190736  beams 0.17, rmp 0.46    (moderate RMP contrast)
#
# group_zero arms kill exactly the group the phenomenon should depend on:
#   AE  -> pinj  (fast-ion drive)
#   ELM -> rmp   (the ELM control knob itself)
#   TM  -> pinj  (NBI torque; tearing modes brake and lock when torque drops)
#
# NULL CONTROL: ech_power is ~0 in every one of these shots, so group_zero:ech_power
# edits nothing. Its divergence_vs_real is therefore the model's NOISE FLOOR (MaskGIT
# decoding is stochastic), which is what tells us how large a real response must be to
# count. Without it, a small divergence is uninterpretable.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/evaluation/slurm/_eval_env.sh

CKPT=/lustre/orion/fus187/proj-shared/models/ignite_production/runs/prod_d512L8/dynamics_latest.pt
OUT=data/outputs/ignite_cases

# EXTENDED cache: symlinks to every entry of Peter's production cache (which is NOT
# group-writable) plus real files for shots encoded here. The 199595-199609 block lives in
# proj-shared/additional_data rather than the main corpus, so its shots are absent from the
# production cache; encoding them here is what lets the TM case use a same-run-day donor.
CACHE=data/outputs/ignite_cache_ext
ADDITIONAL_DATA=/lustre/orion/fus187/proj-shared/additional_data

# TM donors to encode if missing. t0_start comes from the cache manifest (0.0) inside the
# script -- NOT guessed, since a wrong window origin silently yields misaligned codes.
TM_DONORS=199598,199596

# Codec resolution: FROZEN_CODEC_CKPTS holds paths relative to the REPO
# (eval_runs/ignite_d5_<m>/codec_best.pt) which only exist in Peter's tree, so the manifest
# fallback resolves nothing here and eval silently logs "loaded 0 frozen codecs" AND CARRIES
# ON -- producing metrics but no decoded panels at all. resolve_codec_path() tries
# --codec_tmpl FIRST, and `repo / "/abs/path"` yields the absolute path, so an absolute
# template works and resolves all 14 (verified).
CODEC_TMPL="/lustre/orion/fus187/proj-shared/models/ignite_codecs_current/{m}/codec_best.pt"

# mhr is the toroidal MHD array: 500 kHz -> 250 kHz Nyquist, 512 bins at 488 Hz, so it
# resolves both the AE band (50-250 kHz) and the TM band (1-20 kHz).
export IGNITE_FIG_SPECTRO=mhr,ece

# case | shot | actuator_mode
RUNS=(
  "ae|200729|real"
  "ae|200729|donor:200730"
  "ae|200729|donor:200728"
  "ae|200729|group_zero:pinj"
  "tm|199597|real"
  "tm|199597|group_zero:pinj"
  "tm|199597|group_zero:rmp"
  "tm|199597|freeze"
  # same-run-day donor, encoded above -- makes TM methodologically identical to AE
  "tm|199597|donor:199598"
  "elm|190735|real"
  "elm|190735|donor:190736"
  "elm|190735|group_zero:rmp"
  "elm|190735|group_zero:ech_power"
  # ELM-B: the clean-data counterpart to 190735, screened out of the 777-shot core set
  # for (a) all 14 modalities live, (b) ELM rate 103 Hz -- plenty of ELMs to suppress,
  # (c) a SHARP RMP turn-on inside the predicted region: the 12-coil RMS amplitude steps
  # 0.30 -> 3.24 between frames ~70 and ~80 (t ~ 3.5-4.0 s), where the rollout starts at
  # frame 20. So the model must PREDICT an RMP-driven ELM collapse it is being told about
  # only through the actuator channel. 203267 is the very next shot on the run day and is
  # also core/all-live, giving a real in-context donor (rmp contrast 1.14, beams 0.55).
  "elmb|203266|real"
  "elmb|203266|donor:203267"
  "elmb|203266|group_zero:rmp"
  "elmb|203266|group_zero:ech_power"
)

NGPU=8
mkdir -p "$OUT" logs

# ---- encode the TM donors (no-op once they exist) -------------------------------------- #
# On a GPU: 14 codecs x 239 frames per shot is far too heavy for a login node.
ROCR_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/evaluation/ignite_encode_shots.py \
  --shots "$TM_DONORS" \
  --data-dir "$ADDITIONAL_DATA" \
  --cache-dir "$CACHE" \
  --codec-tmpl "$CODEC_TMPL" \
  > "logs/${SLURM_JOB_ID}_encode_tm_donors.log" 2>&1 \
  && echo "[encode] TM donors OK" || echo "[encode] TM donors FAILED (TM donor arm will skip)"
tail -n 12 "logs/${SLURM_JOB_ID}_encode_tm_donors.log" 2>/dev/null | sed 's/^/    /'

# Run i -> GCD (i % NGPU); each GCD works through its own list sequentially, so the
# schedule scales past 8 runs without oversubscribing a device.
for g in $(seq 0 $((NGPU - 1))); do
  (
    export ROCR_VISIBLE_DEVICES=$g
    export PYTHONPATH=src
    for i in "${!RUNS[@]}"; do
      [ $((i % NGPU)) -eq "$g" ] || continue
      IFS='|' read -r CASE SHOT MODE <<< "${RUNS[$i]}"
      TAG="${CASE}_${SHOT}_$(echo "$MODE" | tr ':' '-')"
      if [ -f "$OUT/$TAG/eval_metrics.json" ]; then
        echo "[skip] $TAG"
        continue
      fi
      echo "[gcd $g] start $TAG"
      # eval_dynamics archives codes_<shot>_step<N>.npz BEFORE it mkdirs out_dir (the mkdir
      # lives in render_figure, which runs later), so the archive dies with FileNotFoundError
      # and is swallowed by its own try/except -- "archiving must never take an eval down".
      # Pre-create the dir so the archive lands; without it every re-render needs a fresh
      # 80-frame rollout, and D-alpha/rotation panels are impossible (they decode from these).
      mkdir -p "$OUT/$TAG"
      python -m tokamak_foundation_model.ignite.eval_dynamics \
        --ckpt "$CKPT" \
        --cache_dir "$CACHE" \
        --codec_tmpl "$CODEC_TMPL" \
        --shot "$SHOT" \
        --actuator_mode "$MODE" \
        --split_seed 42 \
        --seed 0 \
        --out_dir "$OUT/$TAG" \
        > "logs/${SLURM_JOB_ID}_${TAG}.log" 2>&1 \
        && echo "[gcd $g] OK   $TAG" || echo "[gcd $g] FAIL $TAG"
    done
  ) &
done

wait

echo
echo "================ SUMMARY ================"
for i in "${!RUNS[@]}"; do
  IFS='|' read -r CASE SHOT MODE <<< "${RUNS[$i]}"
  TAG="${CASE}_${SHOT}_$(echo "$MODE" | tr ':' '-')"
  if [ -f "$OUT/$TAG/eval_metrics.json" ]; then
    NC=$(grep -ho "loaded [0-9]* frozen codecs" "logs/${SLURM_JOB_ID}_${TAG}.log" 2>/dev/null | head -1)
    echo "OK   $TAG   ($NC)"
  else
    echo "FAIL $TAG"
    tail -n 10 "logs/${SLURM_JOB_ID}_${TAG}.log" 2>/dev/null | sed 's/^/       /'
  fi
done
