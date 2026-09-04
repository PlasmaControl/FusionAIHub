#!/bin/bash
# Audit one modality's sweep arms on the LOGIN-NODE GPU: the >= 300-window arm table, then
# the deliverable figure for the winner. Runs analysis/spectro_final_fig.py; trains nothing.
#
# Login-node GPU needs the private ROCm caches (a shared MIOpen DB gives
# miopenStatusInternalError on the decoder's conv refinement head).
#
# Usage: _audit_spectro_arms.sh <modality> <arms_dir> [floor] [n_windows]
#   arms_dir = the job's OUT_DIR; every subdir with a codec_best.pt becomes an arm.
#
# THREE stages, in the order a verdict needs them (2026-09-04):
#   1. --mode score      the arm table. nRMSE here is a FLOOR to clear, NOT the ranking key.
#   2. --mode structure  the RANKING stage: per-band hf_ratio for every arm against the two
#                        oracles that calibrate it (tsmooth = the coherent ceiling, the most
#                        any codec should reproduce, measured 0.250 on co2; patchmean = what
#                        the token grid gives away for free, 0.015 on co2). Set STRUCT_ARMS
#                        to a shorter arm list -- it is O(30) full-array metric evaluations
#                        per arm. STRUCT=0 skips it.
#   3. --mode figure     the deliverable, ZOOMED to FIG_BAND_KHZ. At the full 0-250 kHz a
#                        coherent mode line 1-2 STFT bins wide is under one screen pixel, so
#                        the full-band panel cannot distinguish "resolved the tracks" from
#                        "painted a smooth band" and a good codec is judged blurry by
#                        rendering artefact. Both a full-band and a zoomed figure are written.
#
# Env: ARM_PREFIX, ARM_FOR_FIG, FIG_CHANNELS, FIG_BAND_KHZ, STRUCT, STRUCT_ARMS, BANDS_KHZ.
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
#   co2           1024      0.7112    0.6910   ~tmean BEATS it too; prior 0.6303 did NOT reproduce
#   mhr           1536      0.8218    0.8768   prior measurement, reproduced by the audit
#   bes           4096      0.8354    0.8662   per-freq log-z OFF (no bes stats file exists)
#   mirnov        7424      0.8709    0.8307   ~tmean BEATS the linear code -> near dead end
#   ece          10240      0.9915      -      prior measurement, not re-run
#
# ~tmean is the "perfect per-frequency envelope, zero temporal structure" predictor. It is
# CONTEXT ONLY and never a target: it has zero temporal structure by construction, so an arm
# converging on it has deleted exactly the mode evolution the world model exists to predict.
#
# THE RANKING CALIBRATION (2026-09-04, 320 held-out windows per modality, --mode structure).
# hf_ratio 1.0 is NOT the target -- most of a spectrogram's HF gradient energy is
# frame-to-frame realization speckle no codec can carry. Divide an arm's hf by ITS OWN
# modality's coherent ceiling:
#
#   modality  GT ac1      coh_frac    coherent rank90   patchmean hf/nRMSE  tsmooth5 hf/nRMSE
#   co2       0.61-0.68   0.46-0.56   16.9 (PR 11.0)    0.015 / 0.6566      0.250 / 0.3842
#   mirnov    0.42-0.48   0.35-0.41   -                 0.034 / 0.9805      0.290 / 0.6299
#   ece       0.29-0.37   0.29-0.35   22.7 (PR 21.8)    0.001 / 0.9752      0.178 / 0.7646
#
#   ac1        GT lag-1 autocorrelation along the STFT-frame axis (0 = pure speckle).
#   coh_frac   share of band variance surviving a 5-frame time boxcar.
#   rank90     independent time-courses the window's coherent content rides, of T=96.
#              => the 1914-bit budget is 83-114 bits PER TIME-COURSE. Bits are NOT the
#              binding constraint; a flat plate is a statement about the OBJECTIVE.
#   patchmean  the exact per-(channel, patch_f x patch_t) mean at INFINITE precision. On
#              mirnov/ece it barely beats the 1.0 constant anchor, so whatever these codecs
#              deliver must come from INSIDE the patch.
#   tsmooth5   GT low-passed over 5 frames, scored against RAW GT. THE CEILING an arm aims at,
#              and it already beats every trained arm on both hf and nRMSE.
#
# AND hf MUST be read WITH patch_lattice_ratio. The ece production codec reads hf 0.065 (37%
# of its ceiling, better than any co2 arm relatively) at nRMSE 1.2333 and lattice 127-282
# against a GT control of 1.04 -- that HF energy is checkerboard, and its rendered panel is a
# flat plate (std ratio 0.164/0.297, corr 0.007/0.055).
case "${M}" in
    co2)    DEF_FLOOR=0.7112 ;;
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
# ARM_PREFIX filters which subdirs are scored. Needed because one job's OUT_DIR can hold arms
# of TWO modalities (the bes+mirnov pairing), and scoring a mirnov checkpoint as bes would
# silently produce nonsense rather than an error.
# ARMS_OVERRIDE: an explicit "label=ckpt,label=ckpt,..." list, for when a modality's arms are
# spread across SEVERAL job OUT_DIRs (they are -- _mssim, _struct2, _advbest and _advmulti each
# hold arms of the same three modalities). The directory scan below only sees one dir.
ARMS="${ARMS_OVERRIDE:-}"
[ -n "${ARMS}" ] || \
for d in "${DIR}"/*/; do
    n=$(basename "${d}")
    case "${n}" in ${ARM_PREFIX:-*}) ;; *) continue ;; esac
    [ -f "${d}codec_best.pt" ] && ARMS="${ARMS}${ARMS:+,}${n}=${d}codec_best.pt"
    [ -f "${d}codec_last.pt" ] && ARMS="${ARMS}${ARMS:+,}${n}_last=${d}codec_last.pt"
done
[ -n "${ARMS}" ] || { echo "no codec_best.pt under ${DIR}" >&2; exit 1; }
echo "[audit] ${M}: ${ARMS}"

# NO_SEQ=1 skips the forecastability pool. MEASURED 2026-09-04: on mirnov it NEVER finished --
# `seq_pool` needs blocks of 8 CONSECUTIVE windows in which EVERY channel is valid, and
# mirnov's streamed tensor is only 87.3% real, so nearly every candidate block is rejected and
# the scan walks the entire shot list (1.5 h with zero arms scored). The arm table is the
# deliverable; the forecastability margin is not.
SEQFLAG=""
[ "${NO_SEQ:-0}" = "1" ] && SEQFLAG="--no_seq"

${PY} analysis/spectro_final_fig.py --mode score --modality "${M}" \
    --n_windows "${NW}" --batch_size 8 --device cuda --arms "${ARMS}" ${SEQFLAG} \
    ${FLOOR:+--floor ${FLOOR}} \
    --json "eval_runs/codec_recon_figs/${M}_arms.json" 2>&1 | grep --line-buffered -vE "it/s\]|^ *$"

# STAGE 2: the RANKING stage. Defaults to the same arm list; STRUCT_ARMS narrows it.
if [ "${STRUCT:-1}" != "0" ]; then
    ${PY} analysis/spectro_final_fig.py --mode structure --modality "${M}" \
        --n_windows "${NW}" --batch_size 8 --device cuda \
        --arms "${STRUCT_ARMS:-${ARMS}}" \
        --bands_khz "${BANDS_KHZ:-0-10,10-30,30-60,60-120,120-250}" \
        --json "eval_runs/codec_recon_figs/${M}_structure.json" 2>&1 | grep --line-buffered -vE "it/s\]|^ *$"
fi

# STAGE 3: the figure is built from the FIRST arm listed; pass ARM_FOR_FIG=<name> to pick
# another. A ckpt suffixed _last is addressed as <arm>_last -> strip it back to the file.
FIGARM="${ARM_FOR_FIG:-}"
if [ -n "${FIGARM}" ]; then
    case "${FIGARM}" in *_last) FIGCK="${DIR}/${FIGARM%_last}/codec_last.pt" ;;
                        *)      FIGCK="${DIR}/${FIGARM}/codec_best.pt" ;; esac
    for BK in "" "${FIG_BAND_KHZ:-0-60}"; do
        if [ -z "${BK}" ]; then
            OUT="eval_runs/codec_recon_figs/${M}_FINAL_fullshot.png"; BKFLAG=""
        else
            OUT="eval_runs/codec_recon_figs/${M}_FINAL_band.png"
            BKFLAG="--band_khz ${BK/-/,}"
        fi
        ${PY} analysis/spectro_final_fig.py --mode figure --modality "${M}" \
            --arms "${FIGARM}=${FIGCK}" \
            --channels "${FIG_CHANNELS:-0}" --device cuda ${BKFLAG} \
            --out "${OUT}" 2>&1 | grep --line-buffered -vE "it/s\]|^ *$"
    done
fi
