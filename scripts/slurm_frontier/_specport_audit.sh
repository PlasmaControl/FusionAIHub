#!/bin/bash
#SBATCH -A fus187
#SBATCH -J specport_audit
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# 720-held-out-window audit of the Spectral-Codec port arms against the reference codecs.
# AUDIT_SPECS / AUDIT_FIG / AUDIT_JSON / AUDIT_TAG come from the submitting environment.
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
srun -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 --gpu-bind=closest \
  python analysis/render_codec_recon_figs.py --audit ${AUDIT_SPECS} \
    --modality mhr --n_windows "${AUDIT_WINDOWS:-720}" --audit_shots 4 --eval_n_shots 16 \
    --batch_size "${AUDIT_BS:-8}" --num_workers 7 --device cuda --detrended \
    ${AUDIT_FIG:+--fig ${AUDIT_FIG}} ${AUDIT_JSON:+--json ${AUDIT_JSON}}
echo "=== ARM RANKING: distance to the rank-192 ORACLE profile ==="
# spec_nrmse ranks the blur best, and the 5-reference validation REJECTED ms_ssim and
# mode_track_f1 as ranking keys, so arms are ranked by joint distance to the oracle's
# 4-vector with hard gates. The five references are printed alongside for context.
[ -n "${AUDIT_JSON}" ] && [ -f "${AUDIT_JSON}" ] && \
  python analysis/_specport_rank_arms.py "${AUDIT_JSON}"
echo "=== SPECPORT AUDIT DONE (exit $?) ==="
