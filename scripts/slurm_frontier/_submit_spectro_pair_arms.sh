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
# SWEEP MODES (env `SWEEP`, default "mask" = the arm layout described above):
#
#   SWEEP=mask    masked CONTROL / joint-entropy / ms_ssim / no-mask   (the original)
#   SWEEP=mssim   the MODE-STRUCTURE sweep. 2026-09-04 CORRECTION: these codecs feed a world
#                 model whose job is predicting how MODES EVOLVE, so the ranking key is
#                 `hf_ratio` toward GT (read WITH patch_lattice_ratio, GT ~1.1-1.16) and
#                 visible mode tracks in the figure -- NOT spec_nrmse, and specifically NOT
#                 "get close to ~tmean" (tmean is the time-AVERAGED spectrum: it has zero
#                 temporal structure by construction, so optimising toward it optimises
#                 toward the one predictor that carries no mode information at all).
#                 MEASURED cause of the flat plates: every masked arm of 2026-09-03 ran
#                 ms_ssim_weight 0 with adversarial 0 and fm 0, i.e. a pure
#                 L1 + avg-pooled-multiscale objective whose exact minimiser IS the blur
#                 (mirnov ctl_s1 hf 0.008, co2 ctl hf 0.015). ms_ssim is the only term in the
#                 objective that penalises local flatness (its CONTRAST factor
#                 2*sd_x*sd_y/(sd_x^2+sd_y^2) collapses where the recon is flatter than the
#                 target), and it is the lever that produced mhr's resolved mode lines at
#                 weight 50. The sweep is therefore ms_ssim_weight against a weight-0 control.
#                 A modality that already HAS a weight-0 control at the same recipe spends its
#                 slots on 2 weights x 2 SEEDS instead (mirnov, per its measured 0.050-0.120
#                 seed spread).
#   ARMS_EXTRA    appended to EVERY arm of this submit (e.g. "--gain_shape --gain_tokens 8").
#
# Usage: _submit_spectro_pair_arms.sh <modA> <modB> [steps] [walltime]
set -e
# $1 may be a COMMA LIST ("ece,mirnov,co2"); $2 is then optional. The 2-positional form
# <modA> <modB> is unchanged. Arms are emitted per modality and must total <= 8 (one node
# each), which the launcher checks against SLURM_NTASKS.
A="${1:?modality A (or a comma list)}"
B="${2:-}"
MODS="$(printf '%s' "${A}${B:+,${B}}" | tr ',' ' ')"
STEPS_IN="${3:-30000}"
WALL="${4:-02:00:00}"
STATS=/lustre/orion/fus187/proj-shared/models/ignite_codecs_noinorm/stats

BASE="--fsq_levels 8,5,5,5 --consistency_weight 0.0 --eval_batches 32 --seed 1"
BASE="${BASE} --adam_beta1 0.8 --adam_beta2 0.99 --lr_decay_gamma 0.998 --lr_decay_every 1000"
BASE="${BASE} --disc_update_every 2 --adv_warmup_steps 0 --joint_entropy_ramp_steps 2000"
BASE="${BASE} --refine_depth 6 --refine_dilated --pixel_anchor_weight 5.0"
BASE="${BASE} --multiscale_recon_weight 2.0 --adversarial_weight 0.0 --fm_weight 0.0"
BASE="${BASE} --joint_entropy_weight 1.0"

SWEEP="${SWEEP:-mask}"
ARMS_EXTRA="${ARMS_EXTRA:-}"
ARMS_STR=""
for M in ${MODS}; do
    SFLAG=""
    [ -f "${STATS}/codec_${M}_perfreq_stats.pt" ] && \
        SFLAG="--logpow_stats_path ${STATS}/codec_${M}_perfreq_stats.pt"
    P="--modality ${M} ${SFLAG}"
    # PRESENCE FILTER IS PER MODALITY, and the gate is whether its own lengths SIDECAR is warm.
    # `--spectro_presence any` changes the shot list, so it needs codec_<mod>_presence_lengths.pt;
    # with that file cold the arm spends the WHOLE leg scanning (measured: mirnov job 5416298,
    # zero gates in 2 h). Warm today: bes, mirnov. NOT warm: ece, co2, mhr.
    # ece is also the modality the filter matters least for -- 7632/8753 shots keep a live
    # channel (87%) and 87.2% of its streamed tensor is real data (vs bes 0.376 / co2 0.547) --
    # and --mask_missing already excludes the dead (channel, frame) cells inside the loss. So
    # ece runs masked-but-unfiltered rather than burning a leg on a cold scan.
    MSK="--mask_missing"
    # co2 is safe WITHOUT its own sidecar: it has had whole-shot presence filtering via
    # _PRESENCE_FILTER_SIGNALS since July, so `codec_co2_lengths.pt` was itself built from the
    # filtered path list and the presence run HITS it. MEASURED (job 5416277, 2026-09-03): the
    # 7 presence-filtered co2 arms launched 21:06 and wrote gate_0 at 21:08 / gate_1000 at
    # 21:11, and the only "Computing file lengths" in the log is 16 files (the eval split).
    case "${M}" in co2) MSK="${MSK} --spectro_presence any" ;; *)
        if [ -f "/lustre/orion/fus187/proj-shared/foundation_model_meta/codec_${M}_presence_lengths.pt" ]; then
            MSK="${MSK} --spectro_presence any"
        else
            echo "[submit] ${M}: no warm codec_${M}_presence_lengths.pt -> masked but UNFILTERED" >&2
        fi ;;
    esac
    if [ "${SWEEP}" = "struct" ]; then
        # STRUCTURE sweep: the two levers MEASURED to raise hf_ratio, plus the structural one.
        #
        #   ADVERSARIAL. co2 audit 2026-09-04 (320 held-out windows, per-band hf_ratio):
        #   ctl (adv 0, ms_ssim 0) reads hf 0.015 full-band / 0.075 at 0-10 kHz; ms5 reads
        #   0.037 / 0.138; and the `adv` arm -- which is only adversarial_weight 0.05 with
        #   fm 0.5 and NO ms_ssim -- reads 0.073 / 0.217, i.e. roughly DOUBLE ms5 in every
        #   one of the five bands. The winning masked recipe runs adversarial_weight 0.0, so
        #   this lever was switched off in every arm of the flat-plate cohort.
        #   0.2 (not 1.0) because under the adaptive scheme 1.0 is gradient PARITY with the
        #   whole reconstruction reference, and a run-away lam is the documented cause of the
        #   2026-07 co2/video NCCL crash.
        #
        #   ENVELOPE/SHAPE SPLIT (--gain_shape). Makes recon.std(-1) a TRANSMITTED code, so
        #   amplitude collapse is not representable at all -- the direct structural attack on
        #   std_ratio 0.15-0.28. 32 gain tokens = one per freq patch, 160 left for the shape.
        #
        # CALIBRATION for reading the results: GT hf_ratio 1.0 is NOT the target. 75% of the
        # GT's HF gradient energy is frame-to-frame realization speckle -- the tsmooth5 oracle
        # (GT low-passed over 5 STFT frames) reads hf 0.250 on co2 -- so ~0.25 is the ceiling
        # for a codec that reproduces all COHERENT structure and no speckle.
        GS="--gain_shape --gain_tokens 32"
        ADV="--adversarial_weight 0.2 --fm_weight 1.0 --adv_warmup_steps 1500"
        MIR="--eval_batches 8"
        [ "${M}" = "mirnov" ] && MIR="${MIR} --skip_activity_override"
        # per-modality ms_ssim optimum, MEASURED: mhr 20-50, bes 20, co2 5 (co2 at 20 is
        # 0.7015 vs 0.6812 at 5). ece has no measurement yet -> 20, the mhr/bes value.
        case "${M}" in co2) W=5 ;; *) W=20 ;; esac
        case "${M}" in
            ece)
                ARMS_STR="${ARMS_STR};${M}_gs|${P} ${MSK} ${MIR} ${GS} --ms_ssim_weight ${W}"
                ARMS_STR="${ARMS_STR};${M}_gsadv|${P} ${MSK} ${MIR} ${GS} --ms_ssim_weight ${W} ${ADV}"
                ;;
            mirnov)
                # 2 SEEDS on the headline arm: mirnov's measured seed spread (0.050-0.120)
                # exceeds the gap between recipes, so a 1-seed mirnov verdict is not a verdict.
                ARMS_STR="${ARMS_STR};${M}_gs_s1|${P} ${MSK} ${MIR} ${GS} --ms_ssim_weight ${W} --seed 1"
                ARMS_STR="${ARMS_STR};${M}_gs_s2|${P} ${MSK} ${MIR} ${GS} --ms_ssim_weight ${W} --seed 2"
                ARMS_STR="${ARMS_STR};${M}_gsadv_s1|${P} ${MSK} ${MIR} ${GS} --ms_ssim_weight ${W} ${ADV} --seed 1"
                ;;
            *)
                ARMS_STR="${ARMS_STR};${M}_gs|${P} ${MSK} ${MIR} ${GS} --ms_ssim_weight ${W}"
                ARMS_STR="${ARMS_STR};${M}_gsadv|${P} ${MSK} ${MIR} ${GS} --ms_ssim_weight ${W} ${ADV}"
                # adv WITHOUT the split, so the two levers stay attributable to each other.
                ARMS_STR="${ARMS_STR};${M}_adv|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} ${ADV}"
                ;;
        esac
        continue
    fi
    if [ "${SWEEP}" = "mssim" ]; then
        # MODE-STRUCTURE sweep (see header). mirnov ALREADY has its weight-0 control at this
        # exact recipe (ignite_codecs_mirnov_msk/ctl_s{1,2}, 30k steps, ms_ssim_weight 0.0
        # confirmed in the checkpoint cfg), so its slots buy 2 weights x 2 seeds; every other
        # modality gets a 0/5/20/50 ladder with the 0 arm as its own control.
        # --eval_batches 8 for EVERY mssim arm (BASE says 32). The gate set is rebuilt
        # SINGLE-PROCESS at every start AND every resume, and at 29-40 channels one
        # _build_pair is ~29-40 Lustre seeks + 2 STFTs; 32x4=128 of those cost minutes of
        # every leg and buy nothing, because these arms are ranked by the >=300-window audit
        # (analysis/spectro_final_fig.py --mode score), never by the 32-window gate.
        # --skip_activity_override for mirnov only: it is the one modality here with an
        # _activity_overrides entry (min_activity 0.10 / active_bias 0.5), whose per-item
        # re-draws cost ~1 h of a 2 h leg. ece has no entry, so nothing to skip.
        MIR="--eval_batches 8"
        [ "${M}" = "mirnov" ] && MIR="${MIR} --skip_activity_override"
        if [ "${M}" = "mirnov" ]; then
            ARMS_STR="${ARMS_STR};${M}_ms20_s1|${P} ${MSK} ${MIR} --ms_ssim_weight 20 --seed 1"
            ARMS_STR="${ARMS_STR};${M}_ms20_s2|${P} ${MSK} ${MIR} --ms_ssim_weight 20 --seed 2"
            ARMS_STR="${ARMS_STR};${M}_ms50_s1|${P} ${MSK} ${MIR} --ms_ssim_weight 50 --seed 1"
            ARMS_STR="${ARMS_STR};${M}_ms50_s2|${P} ${MSK} ${MIR} --ms_ssim_weight 50 --seed 2"
        else
            for W in 0 5 20 50; do
                ARMS_STR="${ARMS_STR};${M}_ms${W}|${P} ${MSK} ${MIR} --ms_ssim_weight ${W}"
            done
        fi
        continue
    fi
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
# ARMS_EXTRA is appended to EVERY arm (after its own args, so it wins over both BASE and the
# arm), by splicing it before each ';' separator and at the end.
if [ -n "${ARMS_EXTRA}" ]; then
    ARMS_STR="$(printf '%s' "${ARMS_STR}" | sed "s#;# ${ARMS_EXTRA};#g") ${ARMS_EXTRA}"
fi

export MODALITY="${MODS%% *}" # per-arm --modality overrides this; only names the default
export N_SHOTS=9000           # MANDATORY: a wrong N cold-scans (33 min > NCCL watchdog)
export EVAL_N_SHOTS=16
export STEPS="${STEPS_IN}"
export EVAL_EVERY=1000
export BATCH_SIZE=8
export NUM_WORKERS=6
export LR=2e-4
# OUT_DIR is TAGGED by the sweep: the launcher auto-resumes every arm from
# ${OUT_DIR}/<arm>/codec_last.pt, so reusing one dir across sweeps would silently continue a
# DIFFERENT recipe's checkpoint under the new arm's flags. OUT_DIR_TAG overrides it.
_TAG="$(printf '%s' "${MODS}" | tr ' ' '_')"
export OUT_DIR="/lustre/orion/fus187/proj-shared/models/ignite_codecs_${_TAG}_${OUT_DIR_TAG:-${SWEEP}}"
export EXTRA_ARGS="${BASE}"
export ARMS="${ARMS_STR}"
mkdir -p "${OUT_DIR}"

# --dependency=afterany:<id> via DEP_AFTER: MANDATORY when a later leg shares this OUT_DIR,
# so two legs can never write the same arm's codec_last.pt concurrently.
DEP_FLAG=""
[ -n "${DEP_AFTER:-}" ] && DEP_FLAG="--dependency=afterany:${DEP_AFTER}"
JID=$(sbatch --parsable -J codec_${_TAG}_${OUT_DIR_TAG:-${SWEEP}} -N 8 -t "${WALL}" \
      ${DEP_FLAG} --export=ALL \
      scripts/slurm_frontier/ignite_codec_prod.sh)
echo "[submit] ${MODS} (${SWEEP}) -> job ${JID}"
scontrol update JobId="${JID}" Partition=extended,batch,g1 && echo "[submit] ${JID} partitions=extended,batch,g1"
