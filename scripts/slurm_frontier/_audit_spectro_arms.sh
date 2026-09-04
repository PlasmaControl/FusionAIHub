#!/bin/bash
# Audit one modality's sweep arms on the LOGIN-NODE GPU: the >= 300-window arm table, then
# the deliverable figure for the winner. Runs analysis/spectro_final_fig.py; trains nothing.
#
# Login-node GPU needs the private ROCm caches (a shared MIOpen DB gives
# miopenStatusInternalError on the decoder's conv refinement head).
#
# Usage: _audit_spectro_arms.sh <modality> <arms_dir> [floor] [n_windows]
#   arms_dir = the job's OUT_DIR; every subdir with a codec_best.pt becomes an arm.
set -e
M="${1:?modality}"
DIR="${2:?arms dir}"
NW="${4:-320}"

# OUT-OF-SAMPLE LINEAR FLOOR at k = n_tok = 192, per modality: the strongest 192-dimensional
# linear code at INFINITE precision on HELD-OUT shots, i.e. the bar an arm must beat for
# criterion 1. Measured 2026-09-03 with analysis/_specport_plateau.py at the reference
# protocol (N=720 windows, 40 fit shots spread across campaigns, 4 held-out shots,
# presence-filtered pool, gate normalisation). They are NOT alike and track values-per-token,
# so a shared bar would be meaningless:
#
#   modality  values/token   floor    ~tmean   note
#   co2           1024      0.6303      -      prior measurement, not re-run
#   mhr           1536      0.8218    0.8768   prior measurement, reproduced by the audit
#   bes           4096      0.8354    0.8662   per-freq log-z OFF (no bes stats file exists)
#   mirnov        7424      0.8709    0.8307   ~tmean BEATS the linear code -> near dead end
#   ece          10240      0.9915      -      prior measurement, not re-run
#
# ~tmean is the "perfect per-frequency envelope, zero temporal structure" predictor. Where it
# beats the floor (mirnov), a 192-dim linear code cannot even reach the time-averaged
# spectrum, and no vocabulary change can help -- the bottleneck is the DIMENSION, not bits.
case "${M}" in
    co2)    DEF_FLOOR=0.6303 ;;
    mhr)    DEF_FLOOR=0.8218 ;;
    bes)    DEF_FLOOR=0.8354 ;;
    mirnov) DEF_FLOOR=0.8709 ;;
    ece)    DEF_FLOOR=0.9915 ;;
    *)      DEF_FLOOR="" ;;
esac
FLOOR="${3:-${DEF_FLOOR}}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SC="${SCRATCH_DIR:-/tmp/ignite_audit_$USER}"
mkdir -p "${SC}/comgr" "${SC}/miopen" eval_runs/codec_recon_figs
export AMD_COMGR_CACHE_DIR="${SC}/comgr" MIOPEN_USER_DB_PATH="${SC}/miopen" MIOPEN_CUSTOM_CACHE_DIR="${SC}/miopen"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PY=.pixi/envs/frontier/bin/python

# BOTH checkpoints per arm. `codec_best.pt` is selected on gate_score, which is NOT nRMSE:
# measured on mhr o_ms20, best was saved at step 22001 (nRMSE 0.7964 on the audit pool) while
# the run went on to 0.7398 at step 28000 -- a 0.056 nRMSE penalty for reading only the "best"
# file. gate_score must not be changed, so the audit reads LAST as well and reports both.
ARMS=""
for d in "${DIR}"/*/; do
    n=$(basename "${d}")
    [ -f "${d}codec_best.pt" ] && ARMS="${ARMS}${ARMS:+,}${n}=${d}codec_best.pt"
    [ -f "${d}codec_last.pt" ] && ARMS="${ARMS}${ARMS:+,}${n}_last=${d}codec_last.pt"
done
[ -n "${ARMS}" ] || { echo "no codec_best.pt under ${DIR}" >&2; exit 1; }
echo "[audit] ${M}: ${ARMS}"

${PY} analysis/spectro_final_fig.py --mode score --modality "${M}" \
    --n_windows "${NW}" --batch_size 8 --device cuda --arms "${ARMS}" \
    ${FLOOR:+--floor ${FLOOR}} \
    --json "eval_runs/codec_recon_figs/${M}_arms.json" 2>&1 | grep --line-buffered -vE "it/s\]|^ *$"

# The figure is built from the FIRST arm listed; pass ARM_FOR_FIG=<name> to pick another.
FIGARM="${ARM_FOR_FIG:-}"
if [ -n "${FIGARM}" ]; then
    ${PY} analysis/spectro_final_fig.py --mode figure --modality "${M}" \
        --arms "${FIGARM}=${DIR}/${FIGARM}/codec_best.pt" \
        --channels "${FIG_CHANNELS:-0}" --device cuda \
        --out "eval_runs/codec_recon_figs/${M}_FINAL_fullshot.png" 2>&1 | grep --line-buffered -vE "it/s\]|^ *$"
fi
