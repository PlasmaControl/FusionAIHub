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
    # CORRECTION 2026-09-04: co2 is NOT exempt. It looked exempt because job 5416277 wrote
    # gate_0 two minutes after launch with --spectro_presence any and no co2 sidecar -- its
    # 4789-shot filtered list happened to match what was cached. But the list DRIFTS: job
    # 5420031 resolved 4787 shots (the liveness/presence caches moved by 2 shots), the stored
    # path list no longer matched, and both co2 arms sat for 40+ minutes with an EMPTY output
    # directory, cold-scanning 4787 files at ~0.77 s each. A shot-count coincidence is not an
    # exemption. The rule is now uniform: presence needs its own sidecar, and
    # `--build_presence_lengths` derives one in seconds by subsetting the unfiltered cache.
    if [ -f "/lustre/orion/fus187/proj-shared/foundation_model_meta/codec_${M}_presence_lengths.pt" ]; then
        MSK="${MSK} --spectro_presence any"
    else
        echo "[submit] ${M}: no warm codec_${M}_presence_lengths.pt -> masked but UNFILTERED." >&2
        echo "[submit]   build one first (seconds, no HDF5 scan):" >&2
        echo "[submit]   python -m tokamak_foundation_model.ignite.train_codec --modality ${M} \\" >&2
        echo "[submit]     --n_shots 9000 --eval_n_shots 16 --spectro_presence any \\" >&2
        echo "[submit]     --build_presence_lengths --out_dir /tmp/x \\" >&2
        echo "[submit]     --lengths_cache_dir /lustre/orion/fus187/proj-shared/foundation_model_meta" >&2
    fi
    if [ "${SWEEP}" = "gsmulti" ]; then
        # THE UNEXPLORED CELL: ENVELOPE/SHAPE SPLIT **x** MULTISCALE CRITIC, plus the RATE
        # lever. Both corners of the 320-window tables are unshippable in opposite ways:
        #
        #   co2     arm            nRMSE   hf_r  peakF1  lattice  %rate ceiling
        #           co2_m02       0.7028  0.166  0.6243    9.64   92.5   best peakF1
        #           co2_gsadv     0.7300  0.237  0.5959    2.01   73.0   best hf AND lattice
        #   mirnov  a02d_s2_last  1.0004  0.215  0.5022   14.60   85.5   best peakF1
        #           gs_s2         1.1178  0.103  0.3558    2.20   78.5   best lattice
        #
        # Adversarial+multiscale buys peak placement at a 9-24x lattice (GT control 1.12-1.18);
        # gain-shape buys a near-GT lattice (2.01 / 2.20) but loses peakF1 AND spends only
        # 66-78% of the achievable bit rate against 85-94% for the sharp arms. Never run
        # together. This leg runs them together.
        #
        # RATE LEVER: the gain-shape deficit is structural -- `gain_tokens` of the 192 carry
        # the per-(channel, frequency) ENVELOPE, which is smooth across windows and therefore
        # LOW-ENTROPY, so those token positions deliver few bits. Sweeping gain_tokens
        # 32 -> 16 -> 8 hands the shape path 160 -> 176 -> 184 tokens and should walk the rate
        # back toward 90%. If gain-shape reaches ~90% rate while holding lattice near 2, that
        # is the win for both modalities. gain_tokens must DIVIDE freq_bins (512).
        #
        # NOT swept here, with reasons:
        #   pixel_anchor_weight -- already AT its documented optimum 5.0 in BASE (0.05 was the
        #     old default; 1.0 is worse than 5.0 on co2 on both axes; 20 collapses to 1 code).
        #   decoder receptive field -- refine_depth 6 + refine_dilated gives RF 2^(6+1)-1 = 127
        #     against patch_f 16, i.e. already 8x the patch, so the lattice on the adversarial
        #     arms is NOT an RF shortfall.
        MIR="--eval_batches 8"
        [ "${M}" = "mirnov" ] && MIR="${MIR} --skip_activity_override"
        # Per-modality objective = each modality's OWN best-measured sharp arm, + gain-shape.
        # co2: ms_ssim 5 (its documented optimum; 20 triples its lattice, 50 was my error).
        # mirnov: ms_ssim 20 -- the 320-window audit REFUTES my 2-seed ms_ssim-50 screen, the
        #   a02d arms (ms_ssim 20 + multiscale + adversarial) beat every ms50 arm on peakF1,
        #   0.5022 vs 0.4471. Build on a02d.
        case "${M}" in co2) W=5 ;; *) W=20 ;; esac
        B="--ms_ssim_weight ${W} --multiscale_recon_scales 1,2,4 --adversarial_weight 0.2"
        B="${B} --fm_weight 1.0 --adv_warmup_steps 1500 --discriminator multiscale --gain_shape"
        ARMS_STR="${ARMS_STR};${M}_gsm32_s1|${P} ${MSK} ${MIR} ${B} --gain_tokens 32 --seed 1"
        ARMS_STR="${ARMS_STR};${M}_gsm32_s2|${P} ${MSK} ${MIR} ${B} --gain_tokens 32 --seed 2"
        ARMS_STR="${ARMS_STR};${M}_gsm16|${P} ${MSK} ${MIR} ${B} --gain_tokens 16"
        ARMS_STR="${ARMS_STR};${M}_gsm8|${P} ${MSK} ${MIR} ${B} --gain_tokens 8"
        continue
    fi
    if [ "${SWEEP}" = "co2fix" ]; then
        # CORRECTION LEG. Every co2 arm launched earlier today used ms_ssim_weight 50,
        # joint_entropy_weight 1.0 and adversarial 0.2-0.5 + fm 1.0. The MEASURED co2 optima
        # (project-per-modality-codec-tuning-required, from controlled full arms) are
        # ms_ssim 5, joint_entropy 0.5, and adversarial 0.05 + fm 0.5 -- and ms_ssim 20 was
        # measured to cost nRMSE 1.0084->1.0560, collapse sharpness 0.0278->0.0043 and TRIPLE
        # the lattice 8.40->20.51 on co2. I overrode that from a 96-window decoder-only screen,
        # which is exactly the kind of low-window evidence that produced a wrong co2 MS-SSIM
        # verdict once before. The patch-discriminator arms then INVERTED at step 3000
        # (peak_f1 0.000, corr2d -0.349, envelope_corr -0.682).
        #
        # What is kept from today's work, because it is measured and new: the MULTISCALE
        # CRITIC. On the same full-training leg it was the only thing that stopped the
        # adversarial term destabilising (mirnov a02d peak_f1 0.375 stable vs a02 0.365->0.239
        # with nRMSE 1.778; co2 patch-D inverted while co2 multiscale-D reached gate nRMSE
        # 0.852 / peak_f1 0.710 on TWO seeds). And the ENVELOPE/SHAPE SPLIT, which gave the
        # lowest patch lattice ever recorded here (co2_gs gate lattice 2.1 against a GT
        # control of ~1.16-1.26, vs 4.4-6.5 for every non-gain-shape arm).
        MIR="--eval_batches 8"
        DOC="--ms_ssim_weight 5 --joint_entropy_weight 0.5 --adversarial_weight 0.05"
        DOC="${DOC} --fm_weight 0.5 --adv_warmup_steps 1500 --pixel_anchor_weight 5.0"
        case "${M}" in
            co2)
                ARMS_STR="${ARMS_STR};${M}_r_ctl|${P} ${MSK} ${MIR} ${DOC}"
                ARMS_STR="${ARMS_STR};${M}_r_m|${P} ${MSK} ${MIR} ${DOC} --discriminator multiscale"
                ARMS_STR="${ARMS_STR};${M}_r_m_s2|${P} ${MSK} ${MIR} ${DOC} --discriminator multiscale --seed 2"
                ARMS_STR="${ARMS_STR};${M}_r_gsm|${P} ${MSK} ${MIR} ${DOC} --discriminator multiscale --gain_shape --gain_tokens 32"
                ;;
        esac
        continue
    fi
    if [ "${SWEEP}" = "advmulti" ]; then
        # THE MULTISCALE CRITIC IS WHAT MAKES ADVERSARIAL PRESSURE USABLE.
        #
        # MEASURED at FULL training, job 5420031 (advbest), where the ONLY difference between
        # these arms is the discriminator family:
        #
        #   arm              D          step    hf      peak_f1  nRMSE   lattice  corr2d
        #   mirnov_a02d_s1   multiscale 15000   0.1260  0.375    1.061   21.1     0.286
        #   mirnov_a02d_s2   multiscale 15000   0.0809  0.354    1.123   15.1     0.346
        #   mirnov_a02_s1    patch      14000   0.1883  0.239    1.778   16.7     0.197
        #   ece_a02d         multiscale 10000   0.0060  0.285    1.304   27.2     0.105
        #   ece_a02          patch      10000   0.0404  0.256    1.409   34.0     0.063
        #   co2_a02          patch       3000   0.0021  0.000    1.220   19.6    -0.349
        #   co2_a05          patch       3000   0.0018  0.000    1.190   24.6    -0.347
        #
        # The patch-discriminator arms DESTABILISE: mirnov_a02_s1's peak_f1 fell 0.365 -> 0.239
        # between steps 8000 and 14000 while its nRMSE rose to 1.778, and both co2 arms went
        # ANTI-correlated (envelope_corr -0.68, peak_f1 0.000) within 1500 steps of the
        # adversarial term switching on. The multiscale arms are stable and are the best
        # numbers either modality has produced.
        #
        # This is the same mechanism as the lattice: a PATCH critic is satisfiable by one fixed
        # tiled texture, so under pressure the generator races toward that degenerate solution;
        # a critic that scores the WHOLE spectrogram at 1x/2x/4x cannot be answered that way.
        # NOTE the decoder-only screen did NOT predict this -- it fine-tunes an already
        # CONVERGED decoder, where the patch critic is harmless (co2 patch-D combo read hf 82%
        # of ceiling, lattice 6.7). The instability only appears when the ENCODER is also
        # moving. A screen orders levers; only a full arm decides.
        MIR="--eval_batches 8"
        [ "${M}" = "mirnov" ] && MIR="${MIR} --skip_activity_override"
        case "${M}" in co2) W=50 ;; *) W=20 ;; esac
        B="--ms_ssim_weight ${W} --multiscale_recon_scales 1,2,4 --fm_weight 1.0"
        B="${B} --adv_warmup_steps 1500 --discriminator multiscale"
        case "${M}" in
            co2)
                # co2 has never had the multiscale critic at full training, and its patch-D
                # arms are the ones that inverted.
                ARMS_STR="${ARMS_STR};${M}_m02|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2"
                ARMS_STR="${ARMS_STR};${M}_m05|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.5"
                ARMS_STR="${ARMS_STR};${M}_m02s2|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --seed 2"
                ;;
            mirnov)
                # 3rd seed of the leading recipe -- mirnov's seed spread (0.050-0.120) is wider
                # than the gap between recipes, so two seeds is the minimum and three is better.
                ARMS_STR="${ARMS_STR};${M}_m02_s3|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --seed 3"
                ARMS_STR="${ARMS_STR};${M}_m05_s1|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.5 --seed 1"
                ;;
            ece)
                ARMS_STR="${ARMS_STR};${M}_m02_s2|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --seed 2"
                ARMS_STR="${ARMS_STR};${M}_m05|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.5"
                ARMS_STR="${ARMS_STR};${M}_m02ms50|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --ms_ssim_weight 50"
                ;;
        esac
        continue
    fi
    if [ "${SWEEP}" = "advbest" ]; then
        # THE MEASURED RECIPE. Screened on the login GPU with
        # analysis/probe_spectro_objective.py (decoder-only, codes FROZEN so every arm sees the
        # identical codebook, discriminator trained, 1200 steps, 96 held-out windows). co2 from
        # the shipped ms5 @55001, as % of that pool's coherent hf ceiling (0.322):
        #
        #   arm                                              hf   %ceil  lattice  std_r  nRMSE
        #   base (ms_ssim 5, adv 0, fm 0)                0.0472      15      5.1  0.766  0.5447
        #   freq_grad 5 / 20 / 60                    0.0461-0.0442   14   5.4-6.7  0.763  0.5417
        #   target_time_smooth 5                         0.0277       9      5.2  0.743  0.5463
        #   ms_ssim 20 / 50                          0.0551/0.0616  17/19   6.0/5.4 0.763 0.5396
        #   multiscale_recon_scales 1,2,4                0.0665      21      5.2  0.772  0.5428
        #   adversarial 0.2 + fm 1.0                     0.1460      45      7.2  0.790  0.5553
        #   adversarial 0.5 + fm 1.0                     0.2532      79      7.4  0.793  0.5522
        #   adversarial 1.0 + fm 1.0                     1.2025     374     10.1  0.881  0.6612
        #   adv 0.2 + fm 1.0 + ms_ssim 50 + scales 1,2,4 0.2626      82      6.7  0.813  0.5541
        #
        # So: ADVERSARIAL IS THE LEVER (15% -> 45-79% of ceiling), and it was set to 0.0 in
        # every arm of the flat-plate cohort. ms_ssim 50 and scales 1,2,4 each add a few points
        # and cost nothing. freq_grad is a NEGATIVE at all three weights and target smoothing
        # is a NEGATIVE -- both are closed, do not re-run them. adversarial 1.0 OVERSHOOTS
        # (374% of ceiling, nRMSE 0.6612), so the ladder here stops at 0.5.
        #
        # --adv_warmup_steps 1500 is MANDATORY: the recorded failure is that filterscopes ran
        # adversarial_weight 1.0 with ZERO warmup, so D hammered a from-scratch encoder from
        # step 0 and drove an adversarial 1-code collapse.
        #
        # MIRNOV DIFFERS AND GETS THE MULTISCALE CRITIC. On the same screen (from
        # mirnov_ms20_s2), every hf gain came with a LATTICE EXPLOSION -- base 13.1, adv 0.2
        # 31.6, adv 0.5 42.5, against a GT control of 1.16 -- with nRMSE and corr2d getting
        # WORSE. That is checkerboard, not modes: a PATCH discriminator is satisfiable by one
        # fixed tiled texture, which is exactly the documented lattice mechanism, and mirnov
        # has 7424 values per token so the tiled basis is the cheapest way to raise HF energy.
        # discriminator=multiscale scores the WHOLE spectrogram at 1x/2x/4x and cannot be
        # fooled that way; on co2 it matched the patch-D combo exactly (hf 83% vs 82%,
        # lattice 6.8 vs 6.7), so it costs nothing where the patch D already works.
        MIR="--eval_batches 8"
        [ "${M}" = "mirnov" ] && MIR="${MIR} --skip_activity_override"
        case "${M}" in co2) W=50 ;; *) W=20 ;; esac
        B="--ms_ssim_weight ${W} --multiscale_recon_scales 1,2,4 --fm_weight 1.0 --adv_warmup_steps 1500"
        case "${M}" in
            mirnov)
                ARMS_STR="${ARMS_STR};${M}_a02d_s1|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --discriminator multiscale --seed 1"
                ARMS_STR="${ARMS_STR};${M}_a02d_s2|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --discriminator multiscale --seed 2"
                ARMS_STR="${ARMS_STR};${M}_a02_s1|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --seed 1"
                ;;
            ece)
                ARMS_STR="${ARMS_STR};${M}_a02d|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --discriminator multiscale"
                ARMS_STR="${ARMS_STR};${M}_a02|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2"
                # ece's decoder maps ONE 256-column basis to 40*16*16 = 10240 outputs per
                # token (6.7x mhr) and its measured lattice is 127-282 against GT 1.04, so it
                # also gets the only in-scope capacity lever: a wider codec (n_tok, vocab and
                # FRAME_LAYOUT are untouched).
                ARMS_STR="${ARMS_STR};${M}_a02w|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2 --codec_d_model 512"
                ;;
            *)
                ARMS_STR="${ARMS_STR};${M}_a02|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.2"
                ARMS_STR="${ARMS_STR};${M}_a05|${P} ${MSK} ${MIR} ${B} --adversarial_weight 0.5"
                ;;
        esac
        continue
    fi
    if [ "${SWEEP}" = "smooth" ]; then
        # THE PREDICTABLE-TARGET sweep: --target_time_smooth.
        #
        # This is the one lever aimed at the CAUSE rather than a symptom. MEASURED per
        # modality on 320 held-out windows (--mode structure), the GT's lag-1 autocorrelation
        # along the STFT-frame axis is 0.61-0.68 (co2), 0.42-0.48 (mirnov), 0.29-0.37 (ece),
        # with a coherent variance share of 0.42 / ~0.37 / 0.32. Training to reconstruct the
        # RAW window therefore asks the decoder for ~60-70% unpredictable realization speckle,
        # and the exact minimiser of every L-p term in the objective is the conditional mean --
        # so the decoder correctly answers with a low-amplitude blur. THAT is the flat plate;
        # the measured std(recon)/std(GT) of 0.15-0.32 is the arithmetic of it, not a bug.
        #
        # Smoothing the TARGET removes the unpredictable part from the ask, so the optimum
        # becomes the coherent structure AT FULL AMPLITUDE. The encoder still sees the raw
        # window and the audit still scores against RAW GT, so nothing is being graded on a
        # curve. The ceiling is the tsmooth5 oracle in the same table, which already beats
        # every trained arm on BOTH the ranking key and the nRMSE floor:
        #     co2     hf 0.250 / nRMSE 0.3842   vs  ms5    hf 0.037 / 0.6812
        #     mirnov  hf 0.290 / nRMSE 0.6299   vs  ctl_s1 hf 0.022 / 1.1086
        #
        # K=5 is where the oracle was measured; co2 also gets K=3 because it is the most
        # coherent of the three and may not need as much low-passing.
        MIR="--eval_batches 8"
        [ "${M}" = "mirnov" ] && MIR="${MIR} --skip_activity_override"
        case "${M}" in co2) W=5 ;; *) W=20 ;; esac
        FG="--freq_grad_weight 20"
        case "${M}" in
            mirnov)
                ARMS_STR="${ARMS_STR};${M}_sm5_s1|${P} ${MSK} ${MIR} --target_time_smooth 5 --ms_ssim_weight ${W} --seed 1"
                ARMS_STR="${ARMS_STR};${M}_sm5_s2|${P} ${MSK} ${MIR} --target_time_smooth 5 --ms_ssim_weight ${W} --seed 2"
                ARMS_STR="${ARMS_STR};${M}_sm5fg_s1|${P} ${MSK} ${MIR} --target_time_smooth 5 --ms_ssim_weight ${W} ${FG} --seed 1"
                ;;
            ece)
                ARMS_STR="${ARMS_STR};${M}_sm5|${P} ${MSK} ${MIR} --target_time_smooth 5 --ms_ssim_weight ${W}"
                ARMS_STR="${ARMS_STR};${M}_sm5fg|${P} ${MSK} ${MIR} --target_time_smooth 5 --ms_ssim_weight ${W} ${FG}"
                ;;
            *)
                ARMS_STR="${ARMS_STR};${M}_sm5|${P} ${MSK} ${MIR} --target_time_smooth 5 --ms_ssim_weight ${W}"
                ARMS_STR="${ARMS_STR};${M}_sm5fg|${P} ${MSK} ${MIR} --target_time_smooth 5 --ms_ssim_weight ${W} ${FG}"
                ARMS_STR="${ARMS_STR};${M}_sm3|${P} ${MSK} ${MIR} --target_time_smooth 3 --ms_ssim_weight ${W}"
                ;;
        esac
        continue
    fi
    if [ "${SWEEP}" = "sharp" ]; then
        # THE MODE-LINE sweep: freq_grad_weight, the one term in the objective whose gradient
        # actually PAYS for putting a line inside a patch.
        #
        # WHY, from the zoomed co2 figure (ms5, shot 204984 ch 2, 0-60 kHz): the GT has thin
        # coherent tracks 1-2 STFT bins wide (a rising line at 10-20 kHz through 1.3-1.7 s, a
        # multi-line braid at 20-50 kHz through 2.3-2.5 s) and the reconstruction has NONE --
        # it is visibly blocky on the 16-bin x 16-frame patch lattice, i.e. near-constant
        # inside each patch, which is what the patchmean oracle (hf 0.015) looks like.
        #
        # The gradient argument for the lever: a line occupies ~1-2 of a patch's 16 frequency
        # bins, so turning a flat patch into "line in the right bin" barely moves pixel-L1
        # (~10% of the patch area improves, the rest gets marginally worse) -- L1 does not pay
        # for it. losses.freq_gradient_loss is L1 on the FREQUENCY DERIVATIVE, where a line is
        # a large +/- spike and a flat patch is ~0, so the same change moves it a lot. It is
        # also half of the ranking metric by construction (hf_ratio is HF gradient energy over
        # frequency AND time). It has been 0.0 in every production arm to date.
        #
        # multiscale_recon_scales 1,2,4 is the paired control on the OTHER end: at the default
        # (2,4) every multiscale term is L1 on an AVG-POOLED (low-passed) copy, so the recipe
        # carries a 2.0-weighted "match my blurred spectrogram" term and never scores full
        # resolution at all. Adding scale 1 makes it a real multi-resolution pyramid.
        MIR="--eval_batches 8"
        [ "${M}" = "mirnov" ] && MIR="${MIR} --skip_activity_override"
        case "${M}" in co2) W=5 ;; *) W=20 ;; esac
        case "${M}" in
            mirnov)
                ARMS_STR="${ARMS_STR};${M}_fg20_s1|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} --freq_grad_weight 20 --seed 1"
                ARMS_STR="${ARMS_STR};${M}_fg20_s2|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} --freq_grad_weight 20 --seed 2"
                ARMS_STR="${ARMS_STR};${M}_fg20msc_s1|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} --freq_grad_weight 20 --multiscale_recon_scales 1,2,4 --seed 1"
                ;;
            ece)
                ARMS_STR="${ARMS_STR};${M}_fg5|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} --freq_grad_weight 5"
                ARMS_STR="${ARMS_STR};${M}_fg20|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} --freq_grad_weight 20"
                ;;
            *)
                ARMS_STR="${ARMS_STR};${M}_fg5|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} --freq_grad_weight 5"
                ARMS_STR="${ARMS_STR};${M}_fg20|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} --freq_grad_weight 20"
                ARMS_STR="${ARMS_STR};${M}_fg20msc|${P} ${MSK} ${MIR} --ms_ssim_weight ${W} --freq_grad_weight 20 --multiscale_recon_scales 1,2,4"
                ;;
        esac
        continue
    fi
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
# NODES: one node per arm. The QOS cap is 16 and prod_nfulldecay permanently holds 8, so a
# 4-arm leg must ask for 4 nodes, not 8 -- the launcher srun's exactly ${#arms} tasks and the
# spare nodes would sit idle while blocking a production leg.
NODES="${NODES:-8}"
JID=$(sbatch --parsable -J codec_${_TAG}_${OUT_DIR_TAG:-${SWEEP}} -N "${NODES}" -t "${WALL}" \
      ${DEP_FLAG} --export=ALL \
      scripts/slurm_frontier/ignite_codec_prod.sh)
echo "[submit] ${MODS} (${SWEEP}) -> job ${JID}"
scontrol update JobId="${JID}" Partition=extended,batch,g1 && echo "[submit] ${JID} partitions=extended,batch,g1"
