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
        # MEASURED 2026-09-04 (job 5416298): mirnov wrote ZERO files in 70 minutes while the
        # bes arms beside it reached step 3000-20000 -- it never finished step 0's gate. Cause:
        # _activity_overrides gives mirnov min_activity 0.10 / active_bias 0.5, so
        # _stratified_draw does up to 8 re-draws PER ITEM, each a full _build_pair (a 29-channel
        # strided HDF5 read, ~29 Lustre seeks at ~13 ms, plus 2 STFTs), and _stream_eval_data
        # builds --eval_batches 32 x --eval_batch_size 4 = 128 such items SINGLE-PROCESS before
        # training starts. That is ~1 h of every 2 h leg, and it is paid again on every resume
        # because the eval set is rebuilt, never cached.
        #
        # --skip_activity_override removes it. The stratification was introduced to stop mirnov
        # collapsing on degenerate windows -- but the 2026-09-03 audit shows those "degenerate
        # windows" were overwhelmingly the 28-of-29 DEAD CHANNELS that the new mask now excludes
        # properly (only 0.5% of mirnov shots are absent; the loss is temporal). So the hack was
        # compensating for the missing mask and is now redundant as well as ruinously expensive.
        # It only clears min_activity/active_bias here: the adv_warmup_steps / adversarial_weight
        # it also carries are already overridden by BASE (--adversarial_weight 0.0
        # --adv_warmup_steps 0), and CLI wins over _activity_overrides either way.
        # --eval_batches 8 (the script default) cuts the remaining build another 4x.
        MIR="--skip_activity_override --eval_batches 8"
        ARMS_STR="${ARMS_STR};${M}_ctl_s1|${P} ${MSK} ${MIR} --seed 1"
        ARMS_STR="${ARMS_STR};${M}_ctl_s2|${P} ${MSK} ${MIR} --seed 2"
        ARMS_STR="${ARMS_STR};${M}_je2_s1|${P} ${MSK} ${MIR} --joint_entropy_weight 2.0 --seed 1"
        ARMS_STR="${ARMS_STR};${M}_je2_s2|${P} ${MSK} ${MIR} --joint_entropy_weight 2.0 --seed 2"
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
