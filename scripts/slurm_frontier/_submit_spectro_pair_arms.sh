#!/bin/bash
# Submit ONE 8-node job carrying 4 arms of modality A and 4 arms of modality B.
#
# WHY PAIRED. The account's QOS caps NODES, not jobs (measured: with prod_nfulldecay_95 on 8
# nodes, a second 8-node job runs and a third sits in QOSMaxNodePerUserLimit -> the cap is 16
# nodes). So exactly ONE 8-node codec job can run at a time while the production chain holds
# its slot. Splitting those 8 nodes across two modalities buys per-modality coverage per
# 2-hour leg instead of depth on one modality.
#
# The 4 arms per modality are: masked CONTROL, the anti-collapse lever (joint_entropy_weight),
# the sharpness lever (ms_ssim_weight, with the control being its own weight-0 arm), and a
# NO-MASK arm so the masking fix itself stays attributable.
#
# `--modality` and `--logpow_stats_path` are set PER ARM (arm args are appended last, so they
# win over EXTRA_ARGS), which is what lets one job carry two modalities.
#
# Usage: _submit_spectro_pair_arms.sh <modA> <modB> [steps] [walltime]
set -e
A="${1:?modality A}"
B="${2:?modality B}"
STEPS_IN="${3:-30000}"
WALL="${4:-02:00:00}"
STATS=/lustre/orion/fus187/proj-shared/models/ignite_codecs_noinorm/stats

BASE="--fsq_levels 8,5,5,5 --consistency_weight 0.0 --eval_batches 32 --seed 1"
BASE="${BASE} --adam_beta1 0.8 --adam_beta2 0.99 --lr_decay_gamma 0.998 --lr_decay_every 1000"
BASE="${BASE} --disc_update_every 2 --adv_warmup_steps 0 --joint_entropy_ramp_steps 2000"
BASE="${BASE} --refine_depth 6 --refine_dilated --pixel_anchor_weight 5.0"
BASE="${BASE} --multiscale_recon_weight 2.0 --adversarial_weight 0.0 --fm_weight 0.0"
BASE="${BASE} --joint_entropy_weight 1.0"

MSK="--mask_missing --spectro_presence any"
ARMS_STR=""
for M in "${A}" "${B}"; do
    SFLAG=""
    [ -f "${STATS}/codec_${M}_perfreq_stats.pt" ] && \
        SFLAG="--logpow_stats_path ${STATS}/codec_${M}_perfreq_stats.pt"
    P="--modality ${M} ${SFLAG}"
    if [ "${M}" = "mirnov" ]; then
        # mirnov has NEVER had a codec, so it trains FROM SCRATCH -- the regime where 27 of 28
        # arms ended at 1 code and the same recipe at the same seed spanned lattice 61.02-22.86.
        # Its 4 slots buy 2 configurations x 2 SEEDS instead of 4 one-shot knobs, and the axis
        # is the anti-collapse lever, not sharpness.
        ARMS_STR="${ARMS_STR};${M}_ctl_s1|${P} ${MSK} --seed 1"
        ARMS_STR="${ARMS_STR};${M}_ctl_s2|${P} ${MSK} --seed 2"
        ARMS_STR="${ARMS_STR};${M}_je2_s1|${P} ${MSK} --joint_entropy_weight 2.0 --seed 1"
        ARMS_STR="${ARMS_STR};${M}_je2_s2|${P} ${MSK} --joint_entropy_weight 2.0 --seed 2"
    else
        ARMS_STR="${ARMS_STR};${M}_ctl|${P} ${MSK}"
        ARMS_STR="${ARMS_STR};${M}_je2|${P} ${MSK} --joint_entropy_weight 2.0"
        ARMS_STR="${ARMS_STR};${M}_ms20|${P} ${MSK} --ms_ssim_weight 20"
        ARMS_STR="${ARMS_STR};${M}_nomask|${P}"
    fi
done
ARMS_STR="${ARMS_STR#;}"

export MODALITY="${A}"        # per-arm --modality overrides this; only names the default
export N_SHOTS=9000           # MANDATORY: a wrong N cold-scans (33 min > NCCL watchdog)
export EVAL_N_SHOTS=16
export STEPS="${STEPS_IN}"
export EVAL_EVERY=1000
export BATCH_SIZE=8
export NUM_WORKERS=6
export LR=2e-4
export OUT_DIR="/lustre/orion/fus187/proj-shared/models/ignite_codecs_${A}_${B}_msk"
export EXTRA_ARGS="${BASE}"
export ARMS="${ARMS_STR}"
mkdir -p "${OUT_DIR}"

JID=$(sbatch --parsable -J codec_${A}_${B}_msk -N 8 -t "${WALL}" --export=ALL \
      scripts/slurm_frontier/ignite_codec_prod.sh)
echo "[submit] ${A}+${B} -> job ${JID}"
scontrol update JobId="${JID}" Partition=extended,batch,g1 && echo "[submit] ${JID} partitions=extended,batch,g1"
