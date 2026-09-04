#!/bin/bash
#SBATCH -A fus187
#SBATCH -J codec_audit
#SBATCH -o logs/%j_codec_audit.out
#SBATCH -e logs/%j_codec_audit.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
set -e
# TWO MODES.
#
# ARM_TABLE mode (ARM_TABLE=1) -- the >=300-window arm table, i.e.
# scripts/slurm_frontier/_audit_spectro_arms.sh run INSIDE AN ALLOCATION.
#   Env: MODALITY, ARMS_DIR, FLOOR, NW, ARM_PREFIX, ARM_FOR_FIG, FIG_CHANNELS, FIG_BAND_KHZ,
#        STRUCT, NO_SEQ, AUDIT_WORKERS.
#
# WHY IT MUST NOT RUN ON THE LOGIN NODE (measured 2026-09-04): gate.decode_fidelity is host
# numpy with Python-level loops, so it is single-core bound. A 40-channel ece checkpoint at
# 320 windows ran 2 h 02 m at 99.2% of ONE core and produced nothing -- on login04, a shared
# node. It is now (a) chunk-parallel over AUDIT_WORKERS processes and (b) confined to an
# allocation. 1 node is enough; the QOS cap is 16 nodes and prod_nfulldecay permanently holds
# 8, so keep this SMALL and short so it never competes with an 8-node production leg.
#
# DEFAULT mode: (1) code histogram (imbalance) + (2) faithfulness splice test.
# No world model — frozen codec + data only. Env: CODEC_DIR, MODALITIES, SHOT, OUT_DIR.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs
export MASTER_PORT=29553
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
if [ "${ARM_TABLE:-0}" = "1" ]; then
    export SCRATCH_DIR="${SCRATCH_DIR:-/tmp/ignite_audit_$USER}"
    export AUDIT_WORKERS="${AUDIT_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
    export OMP_NUM_THREADS=1          # the parallelism is the PROCESS POOL, not BLAS threads
    echo "[codec_audit] ARM_TABLE ${MODALITY} dir=${ARMS_DIR} nw=${NW:-320} " \
         "workers=${AUDIT_WORKERS} prefix=${ARM_PREFIX:-*}"
    bash scripts/slurm_frontier/_audit_spectro_arms.sh \
        "${MODALITY:?MODALITY}" "${ARMS_DIR:?ARMS_DIR}" "${FLOOR:-}" "${NW:-320}"
    echo "[codec_audit] ARM_TABLE done"
    exit 0
fi

CODEC_DIR="${CODEC_DIR:-/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all}" \
MODALITIES="${MODALITIES:-ece,co2,bes,mhr}" \
SHOT="${SHOT:-200729}" \
OUT_DIR="${OUT_DIR:-eval_runs/codec_audit}" \
python scripts/training/spectro_codec_audit.py
echo "[codec_audit] done"
