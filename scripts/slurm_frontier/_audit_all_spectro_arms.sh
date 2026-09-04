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

add() {  # add <label> <dir>/<arm>  -> appends "<label>=.../codec_best.pt,<label>_last=.../codec_last.pt"
    local lab="$1" pth="$2"
    [ -f "${pth}/codec_best.pt" ] && ARMS="${ARMS}${ARMS:+,}${lab}=${pth}/codec_best.pt"
    [ -f "${pth}/codec_last.pt" ] && ARMS="${ARMS}${ARMS:+,}${lab}_last=${pth}/codec_last.pt"
}

audit() {  # audit <modality> <floor>
    local M="$1" FL="$2"
    echo "=================== ${M} ==================="
    ARM_PREFIX='' ARMS_OVERRIDE="${ARMS}" STRUCT=0 NO_SEQ=1 \
        bash scripts/slurm_frontier/_audit_spectro_arms.sh "${M}" "${M_MSSIM}" "${FL}" "${NW:-320}"
}

# ---- co2: struct2 arms are the ONLY ones on the documented ms_ssim 5 recipe --------------- #
ARMS=""
add ms5_incumbent  "${M_CO2OLD}/ms5"
for a in co2_gs co2_gsadv co2_adv;      do add "${a}" "${M_STRUCT}/${a}"; done
for a in co2_m02 co2_m02s2 co2_m05;     do add "${a}" "${M_ADVM}/${a}";  done
for a in co2_a02 co2_a05;               do add "${a}" "${M_ADVB}/${a}";  done
audit co2 0.7112

# ---- mirnov ------------------------------------------------------------------------------ #
ARMS=""
for a in mirnov_ms50_s1 mirnov_ms50_s2; do add "${a}" "${M_MSSIM}/${a}"; done
for a in mirnov_gs_s1 mirnov_gs_s2 mirnov_gsadv_s1; do add "${a}" "${M_STRUCT}/${a}"; done
for a in mirnov_m02_s3 mirnov_m05_s1;   do add "${a}" "${M_ADVM}/${a}"; done
for a in mirnov_a02_s1 mirnov_a02d_s1 mirnov_a02d_s2; do add "${a}" "${M_ADVB}/${a}"; done
audit mirnov 0.8709

# ---- ece: the slowest (40 channels) ------------------------------------------------------- #
ARMS=""
for a in ece_ms0 ece_ms5 ece_ms20 ece_ms50; do add "${a}" "${M_MSSIM}/${a}"; done
for a in ece_gs ece_gsadv;                   do add "${a}" "${M_STRUCT}/${a}"; done
for a in ece_m02_s2 ece_m05 ece_m02ms50;     do add "${a}" "${M_ADVM}/${a}"; done
for a in ece_a02 ece_a02d ece_a02w;          do add "${a}" "${M_ADVB}/${a}"; done
audit ece 0.9915
echo "=== ALL SPECTRO ARM TABLES DONE ==="
