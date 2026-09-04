#!/bin/bash
# Score EVERY co2 / mirnov / ece arm produced by the 2026-09-04 legs, in ONE 1-node allocation.
#
# The arms of one modality are spread across FOUR job OUT_DIRs (_mssim, _struct2, _advbest,
# _advmulti), so the per-dir scan in _audit_spectro_arms.sh cannot see them all; this builds an
# explicit ARMS_OVERRIDE list per modality. `codec_last.pt` is scored beside `codec_best.pt`
# for every arm, because best is selected on gate_score and NOT on nRMSE.
#
# ONE node on purpose. The QOS cap is 16 nodes and prod_nfulldecay permanently holds 8, so a
# multi-node audit would stall an 8-node production leg for its whole walltime.
set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
M_MSSIM=/lustre/orion/fus187/proj-shared/models/ignite_codecs_ece_mirnov_mssim
M_STRUCT=/lustre/orion/fus187/proj-shared/models/ignite_codecs_ece_mirnov_co2_struct2
M_ADVB=/lustre/orion/fus187/proj-shared/models/ignite_codecs_ece_mirnov_co2_advbest
M_ADVM=/lustre/orion/fus187/proj-shared/models/ignite_codecs_ece_mirnov_co2_advmulti
M_CO2OLD=/lustre/orion/fus187/proj-shared/models/ignite_codecs_co2_msk
# 2026-09-04 evening legs: the DOCUMENTED-RECIPE correction (co2fix) and the GAIN-SHAPE x
# MULTISCALE cell + gain_tokens rate ladder (gsmulti). Both are continuation-friendly (the
# launcher resumes from codec_last.pt in the same OUT_DIR), so re-running this audit after a
# continuation leg re-scores the SAME labels at their new steps.
M_CO2FIX=/lustre/orion/fus187/proj-shared/models/ignite_codecs_co2_co2fix
M_GSM=/lustre/orion/fus187/proj-shared/models/ignite_codecs_co2_mirnov_gsmulti
# MODS selects which modality tables to build. ece is PARKED (24 checkpoints across ms_ssim
# 0/5/20/50, two discriminator families, gain-shape and d_model 512 never moved nRMSE below
# 1.0472), so it is NOT in the default -- pass MODS="co2 mirnov ece" to include it.
MODS="${MODS:-co2 mirnov}"

add() {  # add <label> <dir>/<arm>  -> appends "<label>=.../codec_best.pt,<label>_last=.../codec_last.pt"
    # `|| true` is LOAD-BEARING: `set -e` is on, and `[ -f x ] && VAR=...` returns 1 when the
    # file is absent, which would abort the whole audit for one arm that never checkpointed.
    local lab="$1" pth="$2"
    [ -f "${pth}/codec_best.pt" ] && ARMS="${ARMS}${ARMS:+,}${lab}=${pth}/codec_best.pt" || true
    [ -f "${pth}/codec_last.pt" ] && ARMS="${ARMS}${ARMS:+,}${lab}_last=${pth}/codec_last.pt" || true
}

audit() {  # audit <modality> <floor>
    local M="$1" FL="$2"
    echo "=================== ${M} ==================="
    ARM_PREFIX='' ARMS_OVERRIDE="${ARMS}" STRUCT=0 NO_SEQ=1 \
        bash scripts/slurm_frontier/_audit_spectro_arms.sh "${M}" "${M_MSSIM}" "${FL}" "${NW:-320}"
}

# ---- co2 ---------------------------------------------------------------------------------- #
# The INCUMBENTS stay in every table: --mode score OVERWRITES the json with only the arms of
# this run, and the pool is rebuilt from the FIRST arm's cfg, so dropping them would leave a
# table that cannot be compared with the previous one. ms5_incumbent is listed first for that
# reason (its cfg has no gain_shape, i.e. the plainest geometry).
case " ${MODS} " in *" co2 "*)
ARMS=""
add ms5_incumbent  "${M_CO2OLD}/ms5"
for a in co2_m02 co2_m02s2;             do add "${a}" "${M_ADVM}/${a}";  done
for a in co2_gs co2_gsadv;              do add "${a}" "${M_STRUCT}/${a}"; done
for a in co2_r_ctl co2_r_m co2_r_m_s2 co2_r_gsm; do add "${a}" "${M_CO2FIX}/${a}"; done
for a in co2_gsm32_s1 co2_gsm32_s2 co2_gsm16 co2_gsm8; do add "${a}" "${M_GSM}/${a}"; done
audit co2 0.7112
;; esac

# ---- mirnov ------------------------------------------------------------------------------ #
case " ${MODS} " in *" mirnov "*)
ARMS=""
for a in mirnov_a02d_s1 mirnov_a02d_s2; do add "${a}" "${M_ADVB}/${a}"; done
for a in mirnov_gs_s1 mirnov_gs_s2;     do add "${a}" "${M_STRUCT}/${a}"; done
for a in mirnov_ms50_s2;                do add "${a}" "${M_MSSIM}/${a}"; done
for a in mirnov_gsm32_s1 mirnov_gsm32_s2 mirnov_gsm16 mirnov_gsm8; do add "${a}" "${M_GSM}/${a}"; done
audit mirnov 0.8709
;; esac

# ---- ece: the slowest (40 channels); PARKED, off by default ------------------------------- #
case " ${MODS} " in *" ece "*)
ARMS=""
for a in ece_ms0 ece_ms5 ece_ms20 ece_ms50; do add "${a}" "${M_MSSIM}/${a}"; done
for a in ece_gs ece_gsadv;                   do add "${a}" "${M_STRUCT}/${a}"; done
for a in ece_m02_s2 ece_m05 ece_m02ms50;     do add "${a}" "${M_ADVM}/${a}"; done
for a in ece_a02 ece_a02d ece_a02w;          do add "${a}" "${M_ADVB}/${a}"; done
audit ece 0.9915
;; esac
# ---- ORACLE CALIBRATION, LAST so a timeout cannot cost the score tables ----------------- #
# Two questions this stage answers and the score table cannot:
#
#  1. WHERE IS THE CEILING. co2 already has this (co2_structure_peakf1.json): patchmean -- the
#     exact per-(channel, 16x16 patch) mean at INFINITE precision -- scores peak_f1 0.6537 and
#     tsmooth5 scores 0.9936, so co2's best arm at 0.6243 sits just BELOW what the patch grid
#     gives for free. mirnov has NO such calibration, and without it "mirnov peak_f1 0.5022"
#     cannot be called either a near-ceiling result or a failure.
#
#  2. IS THE GRID ITSELF THE BLOCKER. n_tok = (512//patch_f) * (96//patch_t), so 8x32, 16x16,
#     32x8 and 64x4 ALL give 192 tokens: they REALLOCATE the same 192 dimensions between
#     frequency and time and leave FRAME_LAYOUT and the vocab untouched. peak_f1 is computed on
#     the window's per-FREQUENCY profile with time averaged away, and modes are thin in
#     frequency and extended in time, so a freq-finer grid should raise the ceiling. These rows
#     are avg-pools -- no codec decode -- so they cost one metric pass each.
_ORACLE_STRUCT="${ORACLE_STRUCT:-1}"
_SC="${SCRATCH_DIR:-/tmp/ignite_audit_$USER}"
mkdir -p "${_SC}/comgr" "${_SC}/miopen"
export AMD_COMGR_CACHE_DIR="${_SC}/comgr" MIOPEN_USER_DB_PATH="${_SC}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${_SC}/miopen"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
oracle() {  # oracle <modality> <label=ckpt list>
    local M="$1" A="$2"
    echo "=================== ${M} ORACLES + PATCH ASPECT ==================="
    .pixi/envs/frontier/bin/python analysis/spectro_final_fig.py --mode structure \
        --modality "${M}" --n_windows "${NW:-320}" --batch_size 8 --device cuda \
        --arms "${A}" --bands_khz 0-10,10-30,30-60,60-120,120-250 \
        --patch_aspects 8x32,32x8 \
        --json "eval_runs/codec_recon_figs/${M}_structure_peakf1.json" 2>&1 \
        | grep --line-buffered -vE "it/s\]|^ *$" \
        || echo "[oracle] ${M} structure FAILED (score tables already written)"
}
if [ "${_ORACLE_STRUCT}" = "1" ]; then
    case " ${MODS} " in *" co2 "*)
        _A="ms5=${M_CO2OLD}/ms5/codec_best.pt"
        [ -f "${M_CO2FIX}/co2_r_gsm/codec_last.pt" ] && \
            _A="${_A},co2_r_gsm_last=${M_CO2FIX}/co2_r_gsm/codec_last.pt" || true
        oracle co2 "${_A}"
        ;; esac
    case " ${MODS} " in *" mirnov "*)
        _A="a02d_s2_last=${M_ADVB}/mirnov_a02d_s2/codec_last.pt"
        [ -f "${M_GSM}/mirnov_gsm32_s1/codec_last.pt" ] && \
            _A="${_A},gsm32_s1_last=${M_GSM}/mirnov_gsm32_s1/codec_last.pt" || true
        oracle mirnov "${_A}"
        ;; esac
fi
echo "=== ALL SPECTRO ARM TABLES DONE ==="
