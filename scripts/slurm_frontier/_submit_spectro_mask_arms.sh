#!/bin/bash
# Submit ONE 8-arm per-modality spectro codec sweep (2026-09-03 masking retrain).
#
# WHY A WRAPPER. `sbatch --export=VAR=...` SPLITS ON COMMAS, so `--fsq_levels 8,5,5,5` and the
# semicolon/space-separated ARMS string cannot be passed that way — the value is truncated at
# the first comma (verified: job 5416275 arrived with EXTRA_ARGS='--fsq_levels'). This wrapper
# exports the variables into its OWN shell and submits with plain `--export=ALL`, which is the
# form every previously-successful ARMS job used.
#
# Usage: scripts/slurm_frontier/_submit_spectro_mask_arms.sh <modality> [steps] [walltime]
set -e
M="${1:?modality (ece|co2|bes|mhr|mirnov)}"
STEPS_IN="${2:-30000}"
WALL="${3:-02:00:00}"
STATS=/lustre/orion/fus187/proj-shared/models/ignite_codecs_noinorm/stats

# The SHIPPED mhr recipe (arm g512_ctl, nrmse 0.7481 at step 31000 vs its 0.8218 linear floor):
# 1000-code FSQ, paper Adam betas + gamma-0.998 decay, D every 2 steps, dilated refine decoder,
# recon-only (adversarial_weight 0.0), anti-collapse joint entropy with a 2000-step ramp.
BASE="--fsq_levels 8,5,5,5 --consistency_weight 0.0 --eval_batches 32 --seed 1"
BASE="${BASE} --adam_beta1 0.8 --adam_beta2 0.99 --lr_decay_gamma 0.998 --lr_decay_every 1000"
BASE="${BASE} --disc_update_every 2 --adv_warmup_steps 0 --joint_entropy_ramp_steps 2000"
BASE="${BASE} --refine_depth 6 --refine_dilated --pixel_anchor_weight 5.0"
BASE="${BASE} --multiscale_recon_weight 2.0 --adversarial_weight 0.0 --fm_weight 0.0"
BASE="${BASE} --joint_entropy_weight 1.0"
# per-freq log-z input stats, when this modality has a precomputed file (bes has none yet).
[ -f "${STATS}/codec_${M}_perfreq_stats.pt" ] && \
    BASE="${BASE} --logpow_stats_path ${STATS}/codec_${M}_perfreq_stats.pt"

MSK="--mask_missing --spectro_presence any"
# 8 arms: ONE knob moved per arm off a masked control, plus a NO-MASK control so the masking
# fix itself is attributable rather than confounded with the sweep.
ARMS_STR="ctl|${MSK}"
ARMS_STR="${ARMS_STR};ms20|${MSK} --ms_ssim_weight 20"
ARMS_STR="${ARMS_STR};ms5|${MSK} --ms_ssim_weight 5"
ARMS_STR="${ARMS_STR};je05|${MSK} --joint_entropy_weight 0.5"
ARMS_STR="${ARMS_STR};je2|${MSK} --joint_entropy_weight 2.0"
ARMS_STR="${ARMS_STR};adv|${MSK} --adversarial_weight 0.05 --fm_weight 0.5 --adv_warmup_steps 1500"
ARMS_STR="${ARMS_STR};pa1|${MSK} --pixel_anchor_weight 1.0"
ARMS_STR="${ARMS_STR};nomask|"

# MIRNOV overrides the arm set. It has NEVER had a codec, so it must train FROM SCRATCH --
# exactly the regime in which this project measured 27 of 28 arms ending at 1 code and the
# SAME recipe at the SAME seed spanning patch-lattice 61.02 to 22.86. A single-seed from-
# scratch arm is therefore unattributable, so the 8 slots buy 4 configurations x 2 SEEDS
# instead of 8 one-shot knobs, and the sweep axis is the anti-collapse lever
# (joint_entropy_weight) rather than the sharpness levers.
if [ "${M}" = "mirnov" ]; then
    ARMS_STR="ctl_s1|${MSK} --seed 1"
    ARMS_STR="${ARMS_STR};ctl_s2|${MSK} --seed 2"
    ARMS_STR="${ARMS_STR};je2_s1|${MSK} --joint_entropy_weight 2.0 --seed 1"
    ARMS_STR="${ARMS_STR};je2_s2|${MSK} --joint_entropy_weight 2.0 --seed 2"
    ARMS_STR="${ARMS_STR};je4_s1|${MSK} --joint_entropy_weight 4.0 --seed 1"
    ARMS_STR="${ARMS_STR};je4_s2|${MSK} --joint_entropy_weight 4.0 --seed 2"
    ARMS_STR="${ARMS_STR};nomask_s1|--seed 1"
    ARMS_STR="${ARMS_STR};nomask_s2|--seed 2"
fi

export MODALITY="${M}"
export N_SHOTS=9000            # MANDATORY: any other value cold-scans (33 min > NCCL watchdog)
export EVAL_N_SHOTS=16
export STEPS="${STEPS_IN}"
export EVAL_EVERY=1000
export BATCH_SIZE=8
export NUM_WORKERS=6
export LR=2e-4
export OUT_DIR="/lustre/orion/fus187/proj-shared/models/ignite_codecs_${M}_msk"
export EXTRA_ARGS="${BASE}"
export ARMS="${ARMS_STR}"
mkdir -p "${OUT_DIR}"

JID=$(sbatch --parsable -J codec_${M}_msk -N 8 -t "${WALL}" --export=ALL \
      scripts/slurm_frontier/ignite_codec_prod.sh)
echo "[submit] ${M} -> job ${JID}"
scontrol update JobId="${JID}" Partition=extended,batch,g1 && echo "[submit] ${JID} partitions=extended,batch,g1"
