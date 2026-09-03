#!/bin/bash
#SBATCH -A fus187
#SBATCH -J tg_gate_spike
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
# IGNITE Phase-A GATE SPIKE on 1 GPU (single GCD).
# Trains the SpectroCodec + FreqAwarePatchGAN and evaluates the §4.4 oracle gate
# (stability / persistence / forecastability / decode_fidelity) on held-out real ECE.
# See src/tokamak_foundation_model/ignite/spike.py (CLI main).
#
# Configure via env (with broad/firm defaults baked into the CLI):
#   N_SHOTS      number of shots for the data pool          (default 40)
#   N_BATCHES    pool size (train batches)                  (default 400)
#   BATCH_SIZE   pairs per batch                            (default 8)
#   STEPS        training/gate steps                        (default 20000)
#   EVAL_EVERY   gate/checkpoint cadence                    (default 1000)
#   OUT_DIR      output dir for gate_*.json/codec_last.pt/summary.json
#   EXTRA_ARGS   any extra CLI flags (e.g. --resume <ckpt>, --lr 5e-4, --n_frames 6)
#
# NOTE (standing multi-partition rule): after submit, the parent runs
#   scontrol update job=<id> Partition=extended,batch,g1
# Keeping -t <=2h above keeps the g1 partition eligible.
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
# fresh per-job /tmp MIOpen cache (avoids the FIND_MODE poison / shared-cache issues);
# MIOPEN_SHARED=1 to reuse the warm cache, MIOPEN_FAST=1 for FIND_MODE=2 (risky).
if [ -n "${MIOPEN_SHARED:-}" ]; then
    export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
    export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"
fi
[ -n "${MIOPEN_FAST:-}" ] && export MIOPEN_FIND_MODE=2
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"

N_SHOTS="${N_SHOTS:-40}"
N_BATCHES="${N_BATCHES:-400}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
OUT_DIR="${OUT_DIR:-eval_runs/ignite_gate_spike_${SLURM_JOB_ID:-local}}"

echo "[tg_gate_spike] host=$(hostname) n_shots=${N_SHOTS} n_batches=${N_BATCHES} \
batch_size=${BATCH_SIZE} steps=${STEPS} eval_every=${EVAL_EVERY} out=${OUT_DIR} \
extra=${EXTRA_ARGS:-}"

# Bare `python` (no srun) mirrors train_fsq_codec.sbatch and every other single-GPU
# script in this dir; srun is reserved for multi-node DDP jobs here.
python -m tokamak_foundation_model.ignite.spike \
    --n_shots "${N_SHOTS}" \
    --n_batches "${N_BATCHES}" \
    --batch_size "${BATCH_SIZE}" \
    --steps "${STEPS}" \
    --eval_every "${EVAL_EVERY}" \
    --out_dir "${OUT_DIR}" \
    --device cuda \
    ${EXTRA_ARGS:-}

echo "=== TOKAMAK-GENIE GATE SPIKE DONE (exit $?) ==="
