#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_codec_prod
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --mail-user=ps9551@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -e
# IGNITE Phase-A STREAMING, DDP production codec trainer.
# Unlike ignite_gate_spike.sh (single GPU, ~40-shot pre-built pool), this streams
# δ-shift pairs across THOUSANDS of shots on the fly with parallel DataLoader workers
# and trains the SpectroCodec + FreqAwarePatchGAN under DDP across N GPUs.
#
# DDP LAYOUT (mirrors train_e2e_stage1_d1024_48L.sh's srun+rank-wrapper DDP, but
# with ONE rank per node — as requested): -N <nodes> --ntasks-per-node=1
# --gres=gpu:1 => global world size == number of nodes (each rank owns one GCD).
# srun launches one task/node; _srun_rank_wrapper.sh maps SLURM_PROCID/LOCALID/NTASKS
# into RANK/LOCAL_RANK/WORLD_SIZE, which train_codec._DDPState reads to init NCCL.
#
# Env overrides:
#   MODALITY     spectro modality (ece|co2|bes|mhr)        (default ece)
#   N_SHOTS      train shots to stream                     (default 2000)
#   EVAL_N_SHOTS held-out gate shots (disjoint)            (default 16)
#   STEPS        training/gate steps                       (default 20000)
#   EVAL_EVERY   gate/checkpoint cadence                   (default 1000)
#   BATCH_SIZE   pairs per batch PER RANK                  (default 8)
#   NUM_WORKERS  DataLoader workers per rank               (default 6)
#   OUT_DIR      output dir                                (default eval_runs/ignite_codec_<mod>_<jobid>)
#   LENGTHS_CACHE_DIR  dir for the dataset's chunk-length sidecar (default foundation_model_meta;
#                      empty = disable cache / re-scan every job)
#   NODES        node count (informational; set -N to match)
#   LR / EMA / EXTRA_ARGS                                  (optional passthrough)
#   ARMS         MULTI-ARM mode (default: unset = the single DDP run described above).
#                Semicolon-separated "name|extra args" entries; entry i runs as an
#                INDEPENDENT single-process (world_size 1) codec run on node i, all inside
#                ONE SLURM job, writing to ${OUT_DIR}/<name>. Use it to sweep a knob when the
#                per-user RUNNING-job cap (MaxJobs=3) is the binding constraint: 8 arms cost
#                one job slot instead of eight. Every arm gets the common flags above plus
#                its own "extra args" (which win, being appended last), so each arm differs
#                from the shared recipe in exactly the tokens you list. Requires
#                --ntasks-per-node=1 and -N >= (number of arms). Implemented with
#                `srun --multi-prog`, so arm args are WHITESPACE-SPLIT and must not contain
#                spaces inside a single token (comma lists are fine — unlike --export, this
#                path never splits on commas).
#                Example (1 job, 3 nodes, 3 independent arms):
#                  ARMS='base|;je2|--joint_entropy_weight 2.0;je5|--joint_entropy_weight 5.0'
#
# NOTE (standing multi-partition rule): after submit, the parent runs
#   scontrol update job=<id> Partition=extended,batch,g1
# Keeping -t <=2h keeps the g1 partition eligible.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

# Distinct MASTER_PORT from the e2e trainers (29500/29510/29515) and the spike.
export MASTER_PORT=29520
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

MODALITY="${MODALITY:-ece}"
N_SHOTS="${N_SHOTS:-2000}"
EVAL_N_SHOTS="${EVAL_N_SHOTS:-16}"
STEPS="${STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LR="${LR:-1e-3}"
# Shared length-cache dir for the production dataset's per-file chunk-count sidecar. Points
# at foundation_model_meta (same convention as the video-presence cache) so we do NOT cold-
# scan thousands of shot lengths at every job start. Override with LENGTHS_CACHE_DIR="".
# NOTE: use ${VAR-default} (no colon) so an EXPLICIT empty string ("") DISABLES the cache instead of
# falling through to the shared dir. The ":-" form treats "" as unset -> shared dir -> overfit runs
# with --shots would overwrite the real caches with a tiny shot list (cost several crashed full runs
# 2026-07-30). Unset => shared foundation_model_meta (normal); ""=> disabled; else the given dir.
LENGTHS_CACHE_DIR="${LENGTHS_CACHE_DIR-/lustre/orion/fus187/proj-shared/foundation_model_meta}"
OUT_DIR="${OUT_DIR:-eval_runs/ignite_codec_${MODALITY}_${SLURM_JOB_ID:-local}}"
mkdir -p logs "${OUT_DIR}"

# --lengths_cache_dir passthrough (empty = disable caching / re-scan).
LENGTHS_CACHE_FLAG=""
[ -n "${LENGTHS_CACHE_DIR}" ] && LENGTHS_CACHE_FLAG="--lengths_cache_dir ${LENGTHS_CACHE_DIR}"

# EMA passthrough (optional).
EMA_FLAG=""
[ -n "${EMA:-}" ] && EMA_FLAG="--ema"

# Resume from this OUT_DIR's codec_last.pt if present (chain-friendly).
RESUME_FLAG=""
if [ -f "${OUT_DIR}/codec_last.pt" ]; then
    echo "[ignite_codec_prod] resuming from ${OUT_DIR}/codec_last.pt"
    RESUME_FLAG="--resume ${OUT_DIR}/codec_last.pt"
fi

echo "[ignite_codec_prod] host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} \
world_size(=nodes)=${SLURM_NTASKS} modality=${MODALITY} n_shots=${N_SHOTS} \
eval_n_shots=${EVAL_N_SHOTS} steps=${STEPS} eval_every=${EVAL_EVERY} \
batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} out=${OUT_DIR} extra=${EXTRA_ARGS:-}"

# ------------------------------------------------------------------------------------- #
# MULTI-ARM mode (ARMS set): N INDEPENDENT single-process runs, one per node, ONE job.
# Each task leaves WORLD_SIZE unset (we do NOT go through _srun_rank_wrapper.sh), so
# train_codec._DDPState takes its single-process branch and the arms never form a process
# group with each other. Falls through to the normal single DDP run when ARMS is unset.
# ------------------------------------------------------------------------------------- #
if [ -n "${ARMS:-}" ]; then
    MP="${OUT_DIR}/multiprog_${SLURM_JOB_ID:-local}.conf"
    : > "${MP}"
    IFS=';' read -r -a _arm_list <<< "${ARMS}"
    if [ "${#_arm_list[@]}" -gt "${SLURM_NTASKS}" ]; then
        echo "ERROR: ${#_arm_list[@]} arms but only ${SLURM_NTASKS} tasks (-N/--ntasks-per-node)." >&2
        exit 1
    fi
    for i in "${!_arm_list[@]}"; do
        ARM_NAME="${_arm_list[$i]%%|*}"
        ARM_ARGS="${_arm_list[$i]#*|}"
        [ "${ARM_ARGS}" = "${_arm_list[$i]}" ] && ARM_ARGS=""   # no '|' => no extra args
        ARM_OUT="${OUT_DIR}/${ARM_NAME}"
        mkdir -p "${ARM_OUT}"
        ARM_RESUME=""
        if [ -f "${ARM_OUT}/codec_last.pt" ]; then
            echo "[ignite_codec_prod] arm ${ARM_NAME}: resuming from ${ARM_OUT}/codec_last.pt"
            ARM_RESUME="--resume ${ARM_OUT}/codec_last.pt"
        fi
        echo "${i} python -m tokamak_foundation_model.ignite.train_codec \
--modality ${MODALITY} --n_shots ${N_SHOTS} --eval_n_shots ${EVAL_N_SHOTS} \
--steps ${STEPS} --eval_every ${EVAL_EVERY} --batch_size ${BATCH_SIZE} \
--num_workers ${NUM_WORKERS} --lr ${LR} --out_dir ${ARM_OUT} \
${LENGTHS_CACHE_FLAG} ${EMA_FLAG} ${ARM_RESUME} ${EXTRA_ARGS:-} ${ARM_ARGS}" >> "${MP}"
        echo "[ignite_codec_prod] arm ${i} '${ARM_NAME}' -> ${ARM_OUT}  extra: ${ARM_ARGS}"
    done
    echo "[ignite_codec_prod] MULTI-ARM: ${#_arm_list[@]} independent single-GPU runs, one per node"
    cat "${MP}"
    # --label prefixes every line with its task id, so the 8 arms' interleaved stdout in the
    # single job log stays attributable (each arm also writes its own gate_*.json / summary.json).
    srun -N "$SLURM_JOB_NUM_NODES" -n "${#_arm_list[@]}" -c "$SLURM_CPUS_PER_TASK" \
         --gpus-per-task=1 --gpu-bind=closest --label --multi-prog "${MP}"
    echo "=== IGNITE CODEC PROD MULTI-ARM DONE (exit $?) ==="
    exit 0
fi

# One rank per node (--ntasks-per-node=1): global world size == node count. The
# rank wrapper (SLURM_PROCID/LOCALID/NTASKS -> RANK/LOCAL_RANK/WORLD_SIZE) is the
# same srun DDP launch mechanism train_e2e_stage1_d1024_48L.sh uses.
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     -m tokamak_foundation_model.ignite.train_codec \
     --modality "${MODALITY}" \
     --n_shots "${N_SHOTS}" \
     --eval_n_shots "${EVAL_N_SHOTS}" \
     --steps "${STEPS}" \
     --eval_every "${EVAL_EVERY}" \
     --batch_size "${BATCH_SIZE}" \
     --num_workers "${NUM_WORKERS}" \
     --lr "${LR}" \
     --out_dir "${OUT_DIR}" \
     ${LENGTHS_CACHE_FLAG} \
     ${EMA_FLAG} \
     ${RESUME_FLAG} \
     ${EXTRA_ARGS:-}

echo "=== IGNITE CODEC PROD DONE (exit $?) ==="
