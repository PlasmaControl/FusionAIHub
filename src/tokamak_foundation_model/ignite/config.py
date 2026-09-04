"""Shared config + tensor-shape contracts for IGNITE Phase A (codecs).

This module is the single source of truth for shapes and hyper-parameters that every
Phase-A component builds against. Components depend ONLY on these shapes (and external
libs), not on each other's implementations, so they can be built independently.

Tensor-shape conventions (batch-first):

    raw window          (B, C, W)          C channels, W = window_samples
    spectrogram window  (B, C, F, T)       F = freq_bins, T = time_frames
    pre-FSQ features     (B, n_tok, d_model)
    fsq codes (int)     (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
    quantized (cont)    (B, n_tok, d_model)
    reconstruction      (B, C, F, T)

    n_tok = (freq_bins // patch_f) * (time_frames // patch_t)

Component interface contracts (implemented in sibling modules; TDD):

    quantizer.SpectroQuantizer(cfg)
        .quantize(feats: (B,n_tok,d_model)) -> (quant: (B,n_tok,d_model), codes: (B,n_tok,fsq_dim) long)
        .indices_from_codes / .codebook_size  (thin wrapper over vector_quantize_pytorch.FSQ)

    nets.SpectroEncoder(cfg)(x: (B,C,F,T)) -> feats (B,n_tok,d_model)      # x-transformers based
    nets.SpectroDecoder(cfg)(quant: (B,n_tok,d_model)) -> recon (B,C,F,T)  # x-transformers based

    losses.shift_consistency(feats_a, feats_b) -> scalar    # ||enc(x)-enc(shift_d x)||^2 on PRE-FSQ feats
    losses.recon_objective(recon, target, disc, cfg) -> dict  # adversarial + lambda_pix * pixel-anchor

    discriminator.FreqAwarePatchGAN(cfg)(x: (B,C,F,T)) -> list of patch-score maps  # multi-scale, freq-PE

    gate.stability(codes_x, codes_shifted) -> float in [0,1]
    gate.persistence(codes_t, codes_tp1, steady_mask) -> float
    gate.forecastability(codes_seq, ...) -> dict   # cheap probe vs persistence, transition stratum
    gate.decode_fidelity(recon, target) -> dict    # mode-detector F1 + distributional match

    data.shift_pair_windows(...) -> (spec_a (B,C,F,T), spec_b (B,C,F,T))  # raw d-shift + re-STFT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log2, prod
from typing import Dict, List, Optional, Sequence, Tuple

# --- STFT / windowing (matches the existing spectro pipeline; see data_loader.py) ---
STFT_FS: float = 500_000.0   # Hz, spectro modality target sampling
STFT_N_FFT: int = 1024       # Hann window length -> 0.512 ms/frame at hop 256
STFT_HOP: int = 256          # STFT hop in samples
CHUNK_S: float = 0.05        # 50 ms = ONE FRAME (the world-model stepping unit)

# --- Fast-TS (filterscopes) raw sampling (matches TokamakH5Dataset.SIGNAL_CONFIGS) ---
# The filterscopes SignalConfig is target_fs=10 kHz, channels_to_use=slice(0,8) (8 chans),
# apply_stft=False, preprocess=standardize. So a 50 ms window is round(0.05*10000) = 500
# raw samples per channel. The codec models the ELM ACTIVITY ENVELOPE of those raw samples,
# NOT the raw spike waveform (§4.3: "Statistic = ELM activity envelope ... NOT spike timing").
FASTTS_FS: float = 10_000.0  # Hz, filterscopes target sampling (== SignalConfig.target_fs)

# --- slow-TS windowing (smooth kinetic-profile time-series; see data_loader.py) -----
# All 7 slow-TS signals load at target_fs = 100 Hz (SignalConfig.target_fs=1e2), so ONE
# 50 ms world-model frame = round(CHUNK_S * SLOWTS_FS) = 5 raw samples. A slow-TS window is
# therefore (C_positions, T=5): a short slice of a smooth radial/position profile in time.
SLOWTS_FS: float = 100.0     # Hz, slow-TS (Thomson / CER / MSE) target sampling


@dataclass
class SpectroCodecConfig:
    """Statistics-first spectrogram codec (Phase A). Defaults match the real STFT grid."""

    # data / shape
    channels: int = 1
    freq_bins: int = 512          # cropped from n_fft//2+1 = 513
    time_frames: int = 96         # ~98 STFT frames / 50 ms, cropped to a multiple of patch_t
    # DESIGNED per-frame budget (Phase-B frame layout): 192 tokens / spectro modality.
    # patch_f=16 -> 512/16 = 32 freq-patches (~8 kHz each, freq-FINE to resolve the coherent
    # modes — vs the old 31 kHz at patch_f=64); patch_t=16 -> 96/16 = 6 time-patches.
    #   n_tok = 32 * 6 = 192  (was 8 * 3 = 24 at patch_f=64/patch_t=32).
    patch_f: int = 16             # -> 32 freq patches
    patch_t: int = 16             # -> 6 time patches    => n_tok = 192
    # CHANNEL-FACTORIZED tokens (ece capacity lever, 2026-07-31): with channel_groups=G,
    # each token's patch spans only channels/G channels (a channel-group axis is added to
    # the token grid, group-outer order), so n_tok = G * n_freq_patch * n_time_patch and
    # a code no longer has to describe all C channels at once (ece: 40 ch through one
    # token was the diagnosed entanglement). Default 1 = all-channel tokens, BYTE-
    # IDENTICAL to the prior behavior (no group PE parameter is even created).
    # channels % channel_groups must be 0 (asserted in the encoder).
    channel_groups: int = 1

    # --- PER-CODEC STFT GEOMETRY (NVIDIA-Spectral-Codec port, 2026-09-03) -------------- #
    # The log-power STFT grid used to BUILD this codec's input. Defaults are EXACTLY the
    # module globals ``STFT_N_FFT`` / ``STFT_HOP``, so every pre-existing codec checkpoint and
    # the ``ignite_prod_v2`` token cache (which the live prod_nfulldecay chain depends on) is
    # bit-identical — ``data.log_power_stft`` reads these with ``getattr(cfg, ..., GLOBAL)``,
    # so old pickled configs that lack the fields entirely resolve to the same values.
    #
    # WHY THEY ARE NOW PER-CODEC: at n_fft 1024 / hop 256 the STFT is 75%-overlapped, i.e. 4x
    # OVERSAMPLED in time — 512 freq bins x 96 frames = 49,152 values per channel describe a
    # 25,000-sample window. Halving n_fft to 512 (hop unchanged at 256) is the canonical
    # 50%-overlap Hann COLA setup: freq_bins 512 -> 256, bin width 488 -> 977 Hz, and the
    # covered band is UNCHANGED at 0-250 kHz (Nyquist is set by STFT_FS, not n_fft). Paired
    # with patch_f/patch_t 8/8 that takes values-per-token from 1536 to 384, i.e. 4x the
    # bits/value the FSQ bottleneck has to spend (0.0065 -> 0.0260).
    stft_n_fft: int = STFT_N_FFT      # 1024 today; 512 for the Spectral-Codec port
    stft_hop: int = STFT_HOP          # 256 (unchanged in the port)

    # bottleneck (vector-quantize-pytorch FSQ)
    # Right-sized to the FSQ-paper-recommended ~1024-code config: [8, 5, 5, 5] = prod = 1000.
    # The previous [8, 8, 8, 8, 8] = 32768 was ~30x over-provisioned for a codec that only
    # ever uses O(10-30) codes; the spare 5th dim always died (min_dim_entropy=0). Four dims
    # at 1000 codes removes that dead dim while keeping ample capacity.
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = prod = 1000

    # RAW input standardization (modalities whose raw is NOT O(1); OFF by default -> rich spectros
    # byte-identical). co2's raw is ~1e13 (unnormalized interferometer counts) -> log10(mag^2) ~ 24,
    # which the log_power ceiling (_LOG_CEIL=20) clips to a flat plate (100% clipped), destroying
    # the signal (true std ~1.08). When on, log_power_stft z-scores the RAW per channel BEFORE the
    # STFT (raw_mean / raw_std, shape (C,)), a per-channel linear rescale that shifts log-power into
    # the un-clipped [-10, 20] band while preserving all spectral structure. Set at runtime from the
    # FM's preprocessing_stats[modality]['raw'] by train_codec.apply_spectro_standardization (co2).
    input_standardize: bool = False
    raw_mean: Optional[List[float]] = None   # (C,) per-channel raw mean
    raw_std: Optional[List[float]] = None     # (C,) per-channel raw std

    # PER-FREQ LOG-POWER Z-STANDARDIZATION (the THIN-modality mean-collapse fix; see
    # data.log_power_stft + train_codec --logpow_stats_path). For thin spectros (co2), the
    # log-power STFT window is a large near-constant plate (~20/freq, clipped at _LOG_CEIL=20)
    # with real signal in only a few freq bins, so pure recon-MAE is minimized by predicting that
    # constant -> the FSQ collapses to ONE code. The fix subtracts the per-(channel,freq) mean and
    # divides by the per-freq std (clamped to a floor so near-constant "noise" bins are not blown
    # up), so informative bins become O(1) and a constant can no longer minimize MAE. The stats
    # live in the codec's OWN raw-clipped log_power_stft space (NO raw-std applied), so this
    # REPLACES raw-standardization (input_standardize is turned OFF when this is on). Shapes are
    # (C, F) nested lists (ckpt-serializable). All default to the no-op (byte-identical when off).
    logpow_standardize: bool = False
    logpow_freq_mean: Optional[list] = None   # (C, F) per-(channel,freq) log-power mean
    logpow_freq_std: Optional[list] = None    # (C, F) per-(channel,freq) log-power std
    logpow_std_floor: float = 0.25            # per-freq std clamp floor (guards flat/noise bins)
    # Per-window instance z-score on the log-power input (mean~0/std~1 per window,channel). ROOT-CAUSE
    # fix for the co2 encoder death: strips co2's large DC offset (window mean ~-9.9) that saturates
    # the FSQ tanh bound. No-op when off (byte-identical). Composes with / supersedes the offset the
    # raw-std + per-freq paths leave.
    input_instance_norm: bool = False
    # SHIFT-ROBUST instance norm (2026-08-03): quantize the per-(window,channel) stats —
    # sd onto a log2 grid with step `q`, mu onto a grid of (q * quantized sd) — so a small
    # realization perturbation (δ-shift ≤ 2 ms) almost never changes the APPLIED
    # normalization. Fixes the measured instance-norm stability tax (codes inherited
    # window-stat jitter: bes-no-norm stab 0.70 vs instance-normed trio 0.31-0.51,
    # retrain v3 2026-08-02). 0.0 = OFF (plain instance norm, byte-identical).
    instance_norm_quantize: float = 0.0

    # transformer (x-transformers)
    d_model: int = 256
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8

    # DECODER family (gated drop-in). "linear" = the original transformer + single nn.Linear
    # `to_pixels` unpatchify (byte-identical DEFAULT). "conv" = HiFi-GAN/VQGAN-style 2D transposed-
    # conv upsampling decoder (nets.SpectroConvDecoder): places each FSQ token on a coarse
    # (n_freq_patch, n_time_patch) grid and SYNTHESIZES fine turbulent texture via a stack of
    # ConvTranspose2d upsample + residual-conv blocks. The linear `to_pixels` can only produce
    # SMOOTH patches (a linear map per token), capping recon of broadband bes/mhr/ece at
    # GT↔recon corr ~0.5; the conv decoder is the one architectural lever left to break that.
    decoder: str = "linear"           # {"linear", "conv"}
    conv_dec_base_ch: int = 128       # SpectroConvDecoder proj_in channel width (halved on upsample)
    conv_dec_res_blocks: int = 2      # residual conv blocks per upsample stage
    # Channel FLOOR for the conv decoder's halving schedule. The paper's HiFi-GAN V1 decoder is
    # 55 M params against a 10 M encoder (5.5:1); ours was 6.9 M against 6.7 M (1.02:1). base_ch
    # 128 -> 1024 with this floor at 128 reproduces the paper's decoder-heavy shape.
    conv_dec_min_ch: int = 64

    # --- decoder conv REFINEMENT head (patch-lattice / checkerboard fix) --------------- #
    # MEASURED (mhr v1k ms2_s1, patch 16x16, 32 held-out windows; gate.patch_lattice_metrics):
    # 56 % of the reconstruction's 2-D spectral energy sits EXACTLY on the patch lattice
    # (patch_lattice_ratio 61.6 vs 1.15 on the ground truth) — a visible regular grid. Root
    # cause: `to_pixels` is a per-token linear map from ONE shared 256-column basis, and that
    # basis is heavily loaded on the patch-local Nyquist rows (22 % of its energy in the single
    # freq-Nyquist row vs 6.25 % uniform), so the same texture is tiled into all 192 patches.
    # `refine_depth > 0` appends a RESIDUAL stride-1 kernel-3 2D conv stack over the ASSEMBLED
    # (F, T) spectrogram (final conv ZERO-INIT -> the head starts as an exact identity), giving
    # the decoder full-resolution cross-patch context so its texture no longer has to be a tiled
    # constant. This is the same lever that fixed the VideoDecoder's 20x20 seams (see
    # VideoCodecConfig.refine_depth). Deliberately NOT a ConvTranspose(kernel=stride) upsampler
    # — that family is the documented FAITH checkerboard bug.
    #   refine_dilated: use dilations 1,2,4,8,... so `refine_depth` convs reach a receptive
    #     field of 2^(depth+1)-1 (depth 4 -> 31 >= patch_f), wide enough to see a whole patch
    #     and its neighbours. False = all dilation 1 (receptive field 2*depth+1).
    # 0 = OFF (byte-identical; nothing is constructed and old pickled configs, which lack these
    # fields entirely, unpickle + load unchanged — every access is via getattr).
    refine_depth: int = 0
    refine_hidden: int = 64
    refine_dilated: bool = False
    # --- decoder NOISE INJECTION (StyleGAN-style stochastic detail) -------------------- #
    # MEASURED root cause of the lattice, beyond the tiled linear head: the adversarial +
    # feature-matching terms REWARD high-frequency energy, and a deterministic decoder's only
    # source of it is that shared tiled basis. Head-only fine-tunes of the mhr baseline make
    # this explicit — under a pure reconstruction objective a conv refinement head cuts the
    # lattice ratio 61.2 -> 15.0, but under the PRODUCTION objective (adversarial 1.0 + FM 1.0)
    # the same head holds hf_ratio ~1.0 and only drags the lattice to 39.5 while envelope_corr
    # falls 0.702 -> 0.583. Adding a learned-scale per-pixel Gaussian noise input (StyleGAN2
    # section 3 / "noise inputs", the standard remedy for a generator forced to fake stochastic
    # detail deterministically) gives the decoder an APERIODIC source of exactly the texture the
    # discriminator is asking for. The scale is per-channel and ZERO-INIT, so a fresh
    # noise-enabled decoder is an EXACT identity to the noiseless one and the gradient
    # d(loss)/d(scale) = <grad_x, noise> is still non-zero, i.e. it can grow if it helps.
    # False = OFF (no parameter created; byte-identical). Applied AFTER the refinement head so
    # the head cannot simply filter it away.
    #
    # MEASURED VERDICT SO FAR: NULL, not a win. In a 2000-step decoder-only fine-tune of the mhr
    # baseline under the production objective (Adam 1e-4), the zero-init scale never grew
    # (|scale| 0.0003 at step 2000) and every metric matched the no-noise control within noise
    # (lattice 51.73 vs 52.50, env_corr 0.5830 vs 0.5845). The plausible reading is that the
    # discriminator separates white noise from real spectrogram texture easily, so the gradient
    # never favours the noise path. Keep the lever, but do not assume it helps.
    #
    # TOKEN CONTRACT UNAFFECTED: Phase B tokenizes with the ENCODER only
    # (train_dynamics._load_codec -> codec.quantizer.fsq(codec.encode(x))), so a stochastic
    # decoder changes nothing about the world-model's codes. The decoder is used at RENDER time
    # (eval_dynamics.decode), which is exactly where the checkerboard is visible.
    decoder_noise: bool = False

    # invariance (Phase-A consistency-loss-only; raw d-shift + re-STFT)
    consistency_delta_ms: Tuple[float, float] = (0.1, 2.0)  # sub-window shift range (< 16 ms codec patch)
    consistency_weight: float = 1.0
    amp_jitter: float = 0.03      # +-3% multiplicative secondary nuisance; 0 disables
    noise_floor: float = 0.0      # additive log-power noise std; 0 disables

    # decoder / reconstruction objective
    adversarial_weight: float = 1.0
    pixel_anchor_weight: float = 0.05   # lambda_pix; STABILITY-GATED (raise only while stability >= 0.80)
    # Multi-resolution + freq-gradient reconstruction loss (NeMo/audio-codec-style; the fix for the
    # smooth-envelope reconstruction of TURBULENT modalities that plain pixel-L1 can't sharpen).
    # Both default 0.0 = OFF (byte-identical). Folded into recon_ref so the adaptive adv balances it.
    multiscale_recon_weight: float = 0.0
    # Resolutions the multi-resolution recon L1 is scored at (avg-pool kernel sizes).
    # DEFAULT (2, 4) is the prior hard-coded value -> byte-identical.
    #
    # WHY THIS IS NOW CONFIGURABLE (2026-09-03): every term at the default is L1 on an
    # AVG-POOLED (i.e. low-passed) copy, and full resolution is never scored at all. Weighting
    # that block at 20 -- the paper's reconstruction weight -- therefore weights "match my
    # BLURRED spectrogram" 20x, which is the opposite of the paper's intent: arXiv 2406.05298
    # varies the STFT WINDOW LENGTH ([32 ... 2048]), trading time against frequency resolution
    # while always comparing full-detail log-magnitudes at each resolution. Including scale 1
    # makes this a genuine multi-resolution pyramid rather than a pure low-pass ladder.
    multiscale_recon_scales: Tuple[int, ...] = (2, 4)
    freq_grad_weight: float = 0.0
    # --- MS-SSIM reconstruction term (``losses.ms_ssim_loss``; 1 - MS-SSIM) ------------- #
    # The DIFFERENTIABLE twin of the ``gate.ms_ssim`` ranking metric, so the quantity trained
    # and the quantity judged are identical. SSIM factorises as luminance x CONTRAST x
    # STRUCTURE with a contrast factor 2*sd_x*sd_y/(sd_x^2+sd_y^2) that COLLAPSES where a
    # reconstruction is locally flatter than the target -- i.e. it penalises blur explicitly,
    # which no L-p term does (the conditional mean is L-p's exact minimiser, and that is the
    # measured failure: best spec_nrmse 0.9010 at hf_ratio 0.0143 and no visible structure).
    # 0.0 = OFF (byte-identical: the term is not even evaluated).
    ms_ssim_weight: float = 0.0
    ms_ssim_win: int = 7
    ms_ssim_scales: Tuple[int, ...] = (1, 2, 4)
    # --- DISCRIMINATOR family ---------------------------------------------------------- #
    # "patch" = FreqAwarePatchGAN (default, unchanged). "multiscale" =
    # discriminator.MultiScaleSpectroGAN, which scores the WHOLE spectrogram at 1x/2x/4x with
    # one number per scale. A PATCH discriminator is satisfiable by emitting a single fixed
    # patch texture, and that is precisely the measured artifact (patch_lattice_ratio 61.02
    # against a GT control of 1.14, 92-94% of the reconstruction's HF energy on the lattice);
    # a global multi-resolution critic cannot be fooled that way. The reference pairs a
    # multi-period with a multi-scale complex-STFT discriminator (arXiv 2406.05298 section 4).
    discriminator: str = "patch"        # {"patch", "multiscale"}
    # Discriminator FEATURE-MATCHING weight (HiFi-GAN/MelGAN vocoder-GAN perceptual term;
    # the spectrogram-adapted stand-in for Genie's VGG perceptual loss, which does NOT
    # transfer to spectrograms). Folded INTO the reconstruction reference alongside the
    # pixel anchor, so it enlarges `recon_ref` -> the VQGAN adaptive weight `lam` rises ->
    # the balanced adversarial term regains sharpening strength. Realization-SAFE: it matches
    # the discriminator's INTERNAL features, not raw STFT phase/realization pixels.
    fm_weight: float = 1.0

    # VQGAN/MagViT-style adaptive adversarial weight ("Taming Transformers" §3.3).
    # The hinge GAN's adversarial term can oscillate against the diversity terms
    # (entropy + shift-consistency). When enabled, the adversarial coefficient is
    # auto-scaled every generator step by lam = ||∇ref|| / (||∇adv|| + 1e-4), the ratio
    # of gradient norms of the non-adversarial ("ref") total and the adversarial term at
    # the decoder's last layer, so neither overpowers the other. Only this lever is added
    # (no LeCAM / EMA / spectral-norm; everything else stays hinge-only).
    adaptive_adv_weight: bool = True    # auto-scale the adversarial term (VQGAN adaptive weight)
    # upper clamp on lam (lower clamp is 0). In STABLE training lam sits ~0.01-1; the old 1e4
    # let a collapsing codec's lam run away to ~729 (co2/video crash 2026-07), so the adversarial
    # term dominated + diverged -> one DDP rank desynced -> NCCL watchdog SIGTERM (exit 143). 50
    # is generous headroom over the ~1 stable value while making run-away domination impossible.
    adaptive_adv_clamp: float = 50.0
    adv_warmup_steps: int = 0           # steps with adv_coeff forced to 0 (adv term off during warmup)

    # --- OPTIMIZER / SCHEDULE (NVIDIA Spectral Codec, arXiv 2406.05298 section 4) ------- #
    # Persisted on the cfg (not just argv) so a checkpoint records the recipe it was trained
    # under. Every default is TODAY's behaviour, so nothing existing changes:
    #   adam_beta1/2        torch.optim.Adam defaults (0.9, 0.999). Paper: 0.8 / 0.99.
    #   lr_decay_gamma      multiplicative decay applied every ``lr_decay_every`` steps as
    #                       lr = lr0 * gamma ** (global_step / lr_decay_every). 1.0 = OFF
    #                       (constant LR, byte-identical). Paper: gamma 0.998 per 1000 steps.
    #   disc_update_every   run the DISCRIMINATOR step once every N generator steps. 1 = every
    #                       step (today). Paper: 2.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_decay_gamma: float = 1.0
    lr_decay_every: int = 1000
    disc_update_every: int = 1

    # anti-collapse (codebook-utilization) regularizer — Genie-style entropy term.
    # The shift-consistency loss has a trivial minimum at encoder ≡ constant (all inputs
    # map to ONE code); these terms pressure the encoder toward diverse codebook usage.
    # entropy_loss = per_sample_entropy_mean - diversity_weight * entropy_of_batch_mean.
    # entropy_weight=1.0 is the value that achieved healthy diversity (min_dim_entropy≈0.65)
    # in the diversity spike; the old 0.1 is the known-collapsing value (dead FSQ dim,
    # min_dim_entropy≈0). Paired with FIX 1 (global DDP batch-mean). See FIX 2.
    entropy_weight: float = 1.0     # multiplies entropy_loss into the generator total
    diversity_weight: float = 1.0   # weight on the batch-diversity (spread) reward
    # --- JOINT-code diversity (the FIX for rank-1 FSQ collapse; measured 2026-09-02) ---
    # `diversity_weight` above rewards each FSQ dimension's MARGINAL spread, one dim at a time.
    # A rank-1 encoder satisfies that perfectly and still uses ~nothing of the codebook: the
    # prod mhr codec (fsq_levels [8,8,8,8,8]) scored min_dim_entropy 0.921 (93% of its ln 8
    # ceiling, i.e. the marginal reward is SATURATED — raising entropy_weight can buy at most
    # 0.15 nats) while using 42 of 32768 joint codes. Measured cause: its encoder features are
    # rank-1 (top PC = 95.7% of variance, participation-ratio effective rank 1.09) and the FSQ
    # project_in rows are mutually aligned (|cos| 0.72-0.99), so the 5 per-dim level positions
    # are near-perfectly correlated (|r| >= 0.997, top covariance eigenvalue 99.9%) — one scalar
    # replicated 5 times. Joint entropy 2.656 nats = 14.2 effective codes.
    #
    # These two terms attack that directly and BOTH default to 0.0 = OFF (byte-identical; they
    # are also read with getattr in the quantizer, so the video / slow-TS / fast-TS configs that
    # lack the fields are unaffected):
    #   joint_entropy_weight  reward on the entropy of the batch-mean JOINT soft-code
    #                         distribution over all prod(fsq_levels) codes (MagViT-2 / LFQ
    #                         "codebook entropy"), i.e. the quantity `frac_codes_used` measures.
    #                         Max value ln(codebook_size) (6.908 nats at cb=1000).
    #   decorrelation_weight  penalty on the mean squared OFF-DIAGONAL correlation of the
    #                         per-dim continuous pre-quant level positions — a direct push
    #                         against the rank-1 structure above.
    # NOTE both are applied INSIDE quantizer.entropy_loss, whose return value generator_losses
    # multiplies by `entropy_weight`; the EFFECTIVE weights are therefore
    # entropy_weight * joint_entropy_weight and entropy_weight * decorrelation_weight.
    joint_entropy_weight: float = 0.0
    decorrelation_weight: float = 0.0
    # Linear RAMP on the joint-entropy weight over the first N generator steps (0 = OFF, the
    # weight is constant from step 0 — byte-identical). Needed because the encoder starts
    # SATURATED on real spectro input: every arm of the 2026-09-02 mhr sweep read 1 distinct
    # code at step 0, and the codec has to climb out of that. A large constant joint weight
    # cannot (measured: joint_entropy_weight 5.0 stayed dead at 1 code / pre-quant level std
    # 0.0000 through step 2000, while 1.0 and 2.0 climbed to joint entropies of 5.00 and 5.69
    # nats). Ramping lets recon + the marginal term de-saturate the projection first, then
    # applies the full joint pressure.
    joint_entropy_ramp_steps: int = 0

    # --- MISSING-DATA (dead diagnostic channel) EXCLUSION ------------------------------ #
    # 2026-09-03 audit, generalising the VIDEO finding to the spectro family. The spectro
    # dataset (train_codec.CodecPairDataset._build_pair) DISCARDS the loader's ``nan_mask``
    # and guards a window only on (a) ``valid_len`` (the padded TAIL), (b) global finiteness,
    # and (c) ``raw.std() < min_std`` computed over ALL CHANNELS AT ONCE. So a window in which
    # some channels are a zero slab and at least one channel is live passes every guard, and
    # the dead channels are reconstructed, fed to the discriminator as REAL, feature-matched
    # and entropy-counted. There was no per-channel mask anywhere on the spectro path -- not
    # in the dataset, not in any loss term, not in the discriminator step.
    #
    #   mask_missing   build a REAL per-(channel, STFT-frame) validity mask
    #                  (data.spectro_frame_mask: NaN-projected exactly like the production
    #                  data_loader._raw_to_frame_mask, PLUS all-zero frames, which is how the
    #                  loader represents an absent channel / an out-of-range window) and
    #                  honour it in EVERY loss term, the discriminator step and the gate.
    #                  False = no mask is built and every consumer receives ``None`` =>
    #                  BYTE-IDENTICAL to the pre-2026-09-03 path.
    #   require_live_channels
    #                  dataset-side: re-draw windows in which ANY channel is dead, so the
    #                  codec only ever sees fully-populated windows (the mask then has
    #                  nothing left to exclude). False = the previous "any live channel" rule.
    #   presence_filter
    #                  drop whole shots that carry no live channel for THIS modality, from the
    #                  precomputed foundation_model_meta/spectro_channel_liveness.pt (one
    #                  torch.load, no HDF5 scan). False = no filtering, as before.
    mask_missing: bool = False
    require_live_channels: bool = False
    presence_filter: bool = False

    # --- FSQ-NATIVE anti-collapse knobs (vector_quantize_pytorch.FSQ constructor args) ----
    # These are the library's OWN hyper-parameters, not a re-implementation. Both default to
    # the library defaults, so every existing codec constructs a byte-identical FSQ.
    #
    #   fsq_noise_dropout      With this probability PER ELEMENT, FSQ.maybe_apply_noise adds a
    #                          uniform +/-0.5 offset to the BOUNDED code and re-clamps to
    #                          [-1, 1]. TRAINING ONLY. Note it is applied AFTER
    #                          `codes_to_indices`, so the emitted INDICES stay noise-free and
    #                          only the decoder-facing continuous latent is jittered — it is a
    #                          decoder-robustness regularizer, not index jitter. Also note the
    #                          +/-0.5 is in the NORMALIZED [-1, 1] code space, where the bin
    #                          spacing is 2/(L-1): for [8,5,5] that is +/-1.75 bins on the
    #                          8-level dim and +/-1.0 bins on the 5-level dims, i.e. a LARGE
    #                          perturbation rather than the half-bin it looks like.
    #   fsq_preserve_symmetry  Swaps the bounding function for `symmetry_preserving_bound`
    #                          (arXiv 2411.19842 section 3.2) and the level<->code mapping in
    #                          `_scale_and_shift`. The library ASSERTS
    #                          `not (noise_dropout > 0 and not preserve_symmetry)`, so it is
    #                          mandatory whenever noise dropout is on — and it is NOT a free
    #                          rider: it changes the emitted codes even at noise_dropout 0, so
    #                          any noise-dropout arm needs a preserve_symmetry-only CONTROL to
    #                          be attributable.
    #
    # NOT exposed: FSQ's neighbouring `orthogonal_rotation`. Its source comment advertises
    # "increase codebook utilization" but also "ensure levels are symmetric!", and the
    # production spectro levels ([8,5,5,5] / [8,5,5]) are ASYMMETRIC.
    fsq_noise_dropout: float = 0.0
    fsq_preserve_symmetry: bool = False


    # --- activity-stratified sampling (anti degenerate-window domination) ------------- #
    # Some modalities are mostly quiet / floored (co2 is ~48% present / 52% floored at ~-10,
    # so the built log-power windows have a median std of only ~0.02 vs ~0.85 for ece/bes/mhr).
    # A codec trained on such a batch is swamped by near-constant windows and collapses to the
    # dominant degenerate value, never learning the minority active signal. `active_bias` biases
    # the per-item draw toward ACTIVE windows (activity = the built log-power window's std here):
    # with probability `active_bias`, if a window's activity < `min_activity` the dataset re-draws
    # from nearby chunks looking for an active one (falling back to the drawn window if none is
    # found, so QUIET windows are NOT dropped — the codec still gets a "quiet" code for Phase-B
    # generalization). Both DEFAULT to 0 (OFF): the already-working modalities (ece/bes/mhr) are
    # byte-identical to before; the trainer turns them ON per-modality only for co2 (see
    # train_codec._activity_overrides). min_activity is a log-power std threshold when > 0.
    min_activity: float = 0.0       # per-window activity threshold (log-power std); 0 = OFF
    active_bias: float = 0.0        # P(re-draw a below-threshold window toward active); 0 = OFF

    # --- PEAK-WEIGHTED RECONSTRUCTION (the surrogate for peak_f1) ---------------------- #
    # ``losses.peak_l1_loss``: L1 reweighted by each frequency bin's prominence over its own
    # column mean, so the target's spectral PEAKS carry most of the loss. 0.0 = OFF
    # (byte-identical; the term is not even evaluated).
    #
    # WHY. MEASURED on 320 held-out co2 windows, peak_f1 (top-k spectral peak overlap -- are
    # the mode tracks in the RIGHT PLACE):
    #     tsmooth5 oracle (perfect coherent structure)   0.9936   <- reachable
    #     patchmean oracle (exact patch means, FREE)     0.6537
    #     shipped ms5 arm                                0.6116
    #     best adversarial arm (hf 82% of ceiling)       0.6196
    # Both trained codecs sit BELOW the free patch-mean code, and no sharpness lever moves it:
    # adversarial pressure took hf from 15% to 82% of the coherent ceiling and std_ratio from
    # 0.766 to 0.964 for +0.008 of peak_f1. Its rendered panel gains GT-like granularity and
    # correct burst columns and still has no 10-20 kHz mode track. Nothing in the objective
    # PAYS for peak placement: a line spans 1-2 of a patch's 16 frequency bins, so placing it
    # correctly moves plain L1 by ~10% of the patch area, while the adversarial term rewards
    # the right texture STATISTICS anywhere in the patch.
    peak_weight: float = 0.0

    # --- TIME-SMOOTHED RECONSTRUCTION TARGET (the predictable component) --------------- #
    # ``target_time_smooth = K`` replaces the reconstruction TARGET with a K-frame boxcar
    # moving average of the window along TIME. The ENCODER still sees the raw window (so the
    # codes are computed from real data at inference, unchanged), and every REPORTED metric is
    # still measured against the RAW spectrogram -- only what the decoder is asked to produce
    # changes. 0 or 1 = OFF, bit-identical (``losses.time_smooth`` returns the same object).
    #
    # WHY. MEASURED on 320 held-out windows per modality (analysis/spectro_final_fig.py
    # --mode structure): most of a spectrogram's frame-to-frame variation is REALIZATION
    # SPECKLE that no code can carry. GT lag-1 autocorrelation along the STFT-frame axis is
    # only 0.61-0.68 for co2, 0.42-0.48 for mirnov and 0.29-0.37 for ece, and the coherent
    # share of variance is 0.42 / ~0.37 / 0.32. Training to reconstruct the RAW window
    # therefore asks the decoder for something ~60-70% unpredictable, and the exact minimiser
    # of every L-p term is the conditional mean -- so the decoder correctly answers with a
    # low-amplitude blur. That IS the flat plate (measured std(recon)/std(GT) 0.15-0.32).
    #
    # Smoothing the TARGET removes the unpredictable part from the ask. The optimum is then
    # the coherent structure AT FULL AMPLITUDE rather than the coherent structure diluted by
    # the speckle it cannot predict. The ceiling this aims at is the ``tsmooth5`` oracle in
    # the same table -- GT low-passed over 5 frames, scored against RAW GT:
    #
    #     modality   tsmooth5 hf_ratio   tsmooth5 nRMSE   best trained arm (raw target)
    #     co2            0.250              0.3842        ms5   hf 0.037  nRMSE 0.6812
    #     mirnov         0.290              0.6299        ctl_s1 hf 0.022 nRMSE 1.1086
    #
    # i.e. the smoothed predictor is BETTER than every trained arm on BOTH the ranking key and
    # the nRMSE floor, while carrying no speckle at all. K should match the coherence length;
    # 5 frames is where the oracle above was measured.
    #
    # SCOPE: this changes the training target only. n_tok, the vocab and FRAME_LAYOUT are
    # untouched, and the audit still scores against raw GT, so any gain is honestly measured.
    target_time_smooth: int = 0

    # --- ENVELOPE / SHAPE SPLIT ("gain-shape", spectro port of the fast-TS fix) --------- #
    # THE PROBLEM IT ATTACKS is amplitude collapse, i.e. the flat plate. Every term in the
    # 2026-09-03 recipe except ms_ssim is an L-p distance, and the exact minimiser of an L-p
    # distance is the CONDITIONAL MEAN -- so a decoder that emits each window's smooth
    # time-average scores well while reproducing no temporal structure at all. Measured:
    # mirnov ctl_s1 std(recon)/std(GT) 0.282 and 0.151 on two channels, hf_ratio 0.008;
    # bes_ctl_last 0.278; co2 ctl 0.613. These codecs feed a world model whose job is
    # predicting how MODES EVOLVE, so that failure is total regardless of nRMSE.
    #
    # THE DECOMPOSITION. With gain_shape ON, the leading ``gain_tokens`` tokens carry the
    # per-(channel, frequency) TIME statistics of the window and the remaining
    # ``n_tok - gain_tokens`` carry a NORMALIZED shape:
    #
    #     level (B,C,F) = x.mean(-1)                     the static envelope
    #     sigma (B,C,F) = (x - level).std(-1)            the TEMPORAL amplitude per freq bin
    #     shape (B,C,F,T) = (x - level) / sigma          zero-mean, unit-std along T
    #
    # and the decoder rebuilds ``recon = level_hat + sigma_hat * shape_hat`` with shape_hat
    # mean-removed AND std-normalized along T inside the decoder. Two things then hold BY
    # CONSTRUCTION rather than by hope (both pinned by tests/test_spectro_gain_shape.py):
    #
    #   1. recon.mean(-1) IS level_hat  -> the envelope is transmitted, not re-derived, so the
    #      shape tokens stop spending their budget re-describing the DC level every window.
    #   2. recon.std(-1) IS sigma_hat   -> THE AMPLITUDE IS TRANSMITTED. A flat plate is no
    #      longer representable: the decoder cannot shrink toward the conditional mean, because
    #      the temporal std of its output is set by a code, not by the loss. This is the
    #      structural attack on hf_ratio / std_ratio, and it is the reason to run this arm.
    #      (``gain_scale`` False keeps only guarantee 1 -- level transmitted, amplitude free --
    #      which is the control that isolates how much of the effect is the std normalization.)
    #
    # HOW IT IS *NOT* TO BE JUDGED (2026-09-04). Not by whether it approaches ``~tmean``.
    # ~tmean is the time-AVERAGED spectrum: it has zero temporal structure by construction, so
    # "get closer to ~tmean" is an instruction to delete the modes. Judge these arms on
    # hf_ratio toward GT (with patch_lattice_ratio beside it) and on visible mode tracks in
    # the band where the GT actually has coherent structure (analysis/spectro_final_fig.py
    # --mode structure locates it). If an arm improves nRMSE while the figure stays a smooth
    # plate it has made things WORSE and must be rejected.
    #
    # TOKEN BUDGET IS UNCHANGED: this is a REALLOCATION of the 192 tokens, not a new token.
    # ``FRAME_LAYOUT`` and the Phase-B vocab are untouched.
    #
    # gain_shape False = OFF: no parameter is constructed, no split is computed, and the
    # forward/state_dict are byte-identical to the pre-2026-09-04 codec (every field is read
    # with getattr, so old pickled configs unpickle and load unchanged).
    gain_shape: bool = False
    # Tokens reserved for the (level, sigma) envelope. MUST divide freq_bins: the envelope is
    # patchified along FREQUENCY into exactly this many patches, so each gain token owns one
    # contiguous frequency band's statistics for ALL channels. gain_tokens == n_freq_patch
    # (32 at the production geometry) puts the envelope on the SAME frequency lattice as the
    # main path, which is the natural default; smaller values coarsen it and leave more tokens
    # for the shape. Must be < n_tok (a codec with no shape path is just an envelope coder).
    gain_tokens: int = 32
    # Transmit sigma as well as level (guarantee 2 above). False = level only.
    gain_scale: bool = True
    # Weight on the DIRECT L1 supervision of the decoded (level, log1p sigma) against their
    # analytic targets. The reconstruction term already sees both, but mixed with the shape
    # error at whatever relative scale the window happens to have; the gain head produces the
    # reconstruction's entire envelope AND amplitude, so it gets its own unambiguous target.
    gain_weight: float = 1.0
    # Hidden width of the per-patch gain MLPs (encoder input head / decoder output head).
    gain_hidden: int = 256

    # oracle-gate acceptance thresholds
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    # collapse detection (§4.4 anti-posterior-collapse): a codec fails the gate if it uses
    # too little of the codebook or has too-low per-dim code entropy. NOTE (2026-07): this
    # `collapsed` flag is now INFORMATIONAL ONLY for best-ckpt selection — it no longer forces
    # gate_score = -inf. See `gate_recon_floor` below and `spike.gate_score`.
    gate_min_utilization: float = 0.02   # min fraction of the codebook used to pass
    gate_min_code_entropy: float = 0.3   # min per-dim normalized code entropy to pass
    # HARD best-ckpt floor on ABSOLUTE distinct codes (2026-08-03): below this the gate
    # score is -inf — best.pt must never track a terminally collapsing codec (which can
    # keep reconstructing via decoder pos-emb, so the recon floor alone misses it).
    gate_hard_min_codes: int = 8
    # Reconstruction floor for best-ckpt DISQUALIFICATION (the only hard gate on the score).
    # A codec is disqualified (gate_score = -inf) ONLY when reconstruction genuinely fails:
    # decode envelope_corr is NaN or < gate_recon_floor. Utilization is folded in as a SOFT
    # reward term instead of a hard gate, so a well-reconstructing codec with one dead FSQ dim
    # is no longer wrongly rejected (it just scores slightly lower on the utilization reward).
    gate_recon_floor: float = 0.2
    # CAUTION on what that floor actually measures (2026-09-02): ``envelope_corr`` runs on the
    # TIME-COLLAPSED power envelope (``spec.mean(axis=-1)``), so a codec can clear it while
    # reconstructing no temporal structure at all. Measured on 360 held-out mhr windows, all
    # five saved mhr codecs read envelope_corr 0.67-0.70 but gate.full_spectro_metrics
    # spec_corr2d 0.27-0.32 and spec_nrmse 1.01-1.25 — i.e. FARTHER from the target than the
    # window's own constant mean, and worse than the trivial time-mean-envelope predictor
    # (0.8799 / 0.4443). Those full-spectrogram keys now ride in every gate dict for exactly
    # this reason; the SELECTION score is deliberately unchanged (they are informational).

    @property
    def n_freq_patch(self) -> int:
        return self.freq_bins // self.patch_f

    @property
    def n_time_patch(self) -> int:
        return self.time_frames // self.patch_t

    @property
    def n_tok(self) -> int:
        return self.channel_groups * self.n_freq_patch * self.n_time_patch

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.fsq_levels)

    @property
    def window_samples(self) -> int:
        return round(CHUNK_S * STFT_FS)

    # --- envelope/shape split (see the gain_shape note above) --------------------------- #
    @property
    def n_gain_tok(self) -> int:
        """Tokens carrying the (level, sigma) envelope code; 0 unless gain_shape is on."""
        if not getattr(self, "gain_shape", False):
            return 0
        return int(getattr(self, "gain_tokens", 32))

    @property
    def n_shape_tok(self) -> int:
        """Tokens carrying the normalized shape (== n_tok when gain_shape is off)."""
        return self.n_tok - self.n_gain_tok

    @property
    def uses_gain_scale(self) -> bool:
        """gain_scale, forced OFF when there is no shape path for it to normalize."""
        return bool(getattr(self, "gain_scale", True)) and self.n_shape_tok > 0

    @property
    def gain_patch_f(self) -> int:
        """Frequency bins per gain token (freq_bins / n_gain_tok)."""
        return self.freq_bins // max(1, self.n_gain_tok)

    @property
    def gain_values_per_tok(self) -> int:
        """Numbers ONE gain token transmits: channels x gain_patch_f x (2 if sigma else 1)."""
        if self.n_gain_tok == 0:
            return 0
        return self.channels * self.gain_patch_f * (2 if self.uses_gain_scale else 1)

    @property
    def gain_bits(self) -> float:
        """Bits the gain path owns (n_gain_tok * log2(codebook))."""
        return self.n_gain_tok * log2(self.codebook_size)

    def __post_init__(self) -> None:
        assert self.freq_bins % self.patch_f == 0, "freq_bins must be divisible by patch_f"
        assert self.time_frames % self.patch_t == 0, "time_frames must be divisible by patch_t"
        if getattr(self, "gain_shape", False):
            g = int(self.gain_tokens)
            assert 1 <= g < self.n_tok, (
                f"gain_tokens ({g}) must be in [1, n_tok={self.n_tok}); a codec with no shape "
                f"tokens left is an envelope coder, not a spectrogram codec"
            )
            assert self.freq_bins % g == 0, (
                f"gain_tokens ({g}) must divide freq_bins ({self.freq_bins}): each gain token "
                f"owns one contiguous frequency band's (level, sigma) for all channels"
            )
            assert self.gain_hidden >= 1, "gain_hidden must be >= 1"
            assert int(getattr(self, "channel_groups", 1)) == 1, (
                "gain_shape does not support channel_groups > 1 (the shape token axis is "
                "mixed as one block; a group axis would have to be mixed per group)"
            )


# ---------------------------------------------------------------------------------------- #
# Video codec (tangtv, Phase A) — statistics-first, generative-decoder, NO STFT machinery.
# ---------------------------------------------------------------------------------------- #
# Design (docs/IGNITE_DESIGN.md §4.3): the video codec is "closest to Genie-native (smooth
# frames); mostly just the no-strong-pixel-MSE / generative-decoder move. Two separate
# up/lower divertor codecs." So this config mirrors the spectro codec's INFRASTRUCTURE
# (FSQ bottleneck, adaptive-adv-weight, entropy/utilization regularizer, gate thresholds,
# feature-matching) but drops everything STFT/spectrogram/δ-shift-specific:
#   * NO shift-consistency term. Video has no STFT-phase / speckle realization nuisance —
#     that failure mode was spectrogram-only (a 0.5 ms shift scrambles ~74 % of spectro
#     codes; a divertor movie has no such sub-window-phase confound). The invariance the
#     spectro codec bought with a δ-shift pair is not needed here, so the video codec trains
#     on a SINGLE frame window (no nuisance pair) with the pixel-anchor + FM + adversarial +
#     entropy losses only.
#   * The reconstruction target is the RAW frames (C, T, H, W); the tokenizer patchifies over
#     space AND time -> tokens -> FSQ -> generative deconv decoder.
#
# Divertor mapping (read from data_loader.MovieConfig, NOT hard-coded physics):
#   tangtv_lower = raw camera channels [0, 2]  (LODIV PAR-int + LODIV PERP)
#   tangtv_upper = raw camera channels [4, 6]  (UPDIV0 PERP + UPDIV0 PAR)
# ch1/ch3/ch5 are NaN (off) in all shots. Each divertor codec sees C=2 channels.


# tangtv movie geometry (matches data_loader.MovieConfig for tangtv_lower / tangtv_upper).
VIDEO_TARGET_FPS: int = 100      # target frame rate after resample
VIDEO_HEIGHT: int = 120          # output frame height
VIDEO_WIDTH: int = 360           # output frame width


@dataclass
class VideoCodecConfig:
    """Statistics-first tangtv video codec (Phase A), per-divertor.

    Two instances are used in production — one per divertor (``divertor="lower"`` /
    ``"upper"``) — each its own encoder/decoder/discriminator, matching §4.3's "two separate
    up/lower divertor codecs". ``divertor`` only selects the channel set + logging name; the
    tensor contract is identical.

    Tensor-shape conventions (batch-first):

        frame window    (B, C, T, H, W)   C channels-per-divertor, T frames, H×W pixels
        pre-FSQ feats    (B, n_tok, d_model)
        fsq codes (int) (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
        quantized       (B, n_tok, d_model)
        reconstruction  (B, C, T, H, W)

        n_tok = (T // patch_t) * (H // patch_h) * (W // patch_w)
    """

    # divertor selector (documentation / logging only; the loader maps it to channels).
    divertor: str = "lower"  # "lower" (cams 0,2) or "upper" (cams 4,6)

    # data / shape
    channels: int = 2              # channels-per-divertor (both divertors keep 2 live cameras)
    frames: int = 5                # T; a 50 ms window at 100 fps = round(0.05*100) = 5 frames
    height: int = VIDEO_HEIGHT     # 120
    width: int = VIDEO_WIDTH       # 360
    patch_t: int = 5               # -> 1 time patch (whole window is one temporal patch)
    patch_h: int = 20              # -> 6 height patches
    patch_w: int = 20              # -> 18 width patches   => n_tok = 1*6*18 = 108

    # bottleneck (vector-quantize-pytorch FSQ) — right-sized like the spectro fix:
    # [8, 5, 5, 5] = prod = 1000 codes, 4 dims (ample for a smooth-frame divertor view; the
    # spare-dim death seen at [8,8,8,8,8]=32768 is avoided). Video content is smoother than
    # spectro modes, so 1000 codes is a comfortable start.
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = 1000

    # transformer (x-transformers) — spatial+temporal token set, bidirectional (codec frame).
    d_model: int = 256
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8

    # decoder / reconstruction objective (Genie-native: generative decoder, LOW pixel anchor).
    adversarial_weight: float = 1.0
    # LOW-weight pixel anchor. §4.3's "no-strong-pixel-MSE move": the pixel term only buys
    # optimization stability; the sharp, plausible-frame reconstruction is produced by the
    # adversarial + feature-matching signal, not by minimizing pixel error to the mean.
    pixel_anchor_weight: float = 0.05
    # Discriminator FEATURE-MATCHING weight (VGG-style perceptual term via the discriminator's
    # internal features — the standard image-GAN perceptual loss; unlike spectrograms, VGG-ish
    # perceptual matching IS natural for camera frames, and the FM term is a clean stand-in that
    # needs no external VGG weights). Folded into recon_ref so the VQGAN adaptive weight rises.
    fm_weight: float = 1.0

    # VQGAN/MagViT adaptive adversarial weight ("Taming Transformers" §3.3) — reused verbatim
    # from the spectro codec (auto-scale adv coeff by grad-norm ratio at decoder.last_layer).
    adaptive_adv_weight: bool = True
    # upper clamp on lam (lower clamp is 0) — see SpectroCodecConfig: 50 stops a collapsing
    # codec's lam from running away (old 1e4 -> ~729 -> diverge -> DDP desync -> exit 143).
    adaptive_adv_clamp: float = 50.0
    adv_warmup_steps: int = 0

    # anti-collapse (codebook-utilization) entropy regularizer — Genie/LFQ style, reused
    # from the spectro quantizer's entropy_loss (per-sample entropy - diversity * batch-mean).
    # entropy_weight=1.0 (the diversity-spike value, min_dim_entropy≈0.65); 0.1 was the
    # known-collapsing value. Paired with FIX 1 (global DDP batch-mean). See FIX 2.
    entropy_weight: float = 1.0
    diversity_weight: float = 1.0
    # --- JOINT-code diversity (the fsq-collapse fix; see SpectroCodecConfig for the full
    # measurement note). `diversity_weight` rewards each FSQ dim's MARGINAL spread, which a
    # rank-1 encoder saturates while using ~nothing of the joint codebook; these reward the
    # entropy of the batch-mean JOINT code distribution / penalise the per-dim correlation.
    # Both read via getattr inside quantizer.entropy_loss, so 0.0 = OFF = BYTE-IDENTICAL.
    # NOTE the joint term materialises an (N, codebook_size) tensor and is refused above
    # 4096 codes, so it is usable at fsq_levels [8,5,5,5] (1000) and NOT at [8,8,8,5,5,5]
    # (64000) -- which is exactly why the 1000-code retrain is the one that can use it.
    joint_entropy_weight: float = 0.0
    decorrelation_weight: float = 0.0
    joint_entropy_ramp_steps: int = 0

    # --- MISSING-DATA (dead camera) EXCLUSION ------------------------------------------ #
    # MEASURED 2026-09-03 over all 8753 shots from the full-dataset per-channel liveness scan
    # (foundation_model_meta/video_channel_liveness.pt):
    #     tangtv_lower (ch 0, 2): BOTH cameras live in 2640 shots (30.16%), exactly ONE live in
    #                             1623 (18.54%), NEITHER live in 4490 (51.30%).
    #     tangtv_upper (ch 4, 6): both 1820 (20.79%), one 930 (10.62%), neither 6003 (68.58%).
    # An off camera is stored as a fully-NaN slab and the loader ZERO-FILLS it, so with no
    # filtering 51.3% / 68.6% of the streamed windows are ALL-ZERO clips and, among the
    # surviving shots, a further 19.0% / 16.9% of channel-slots are zero. Net: only 39.4%
    # (lower) / 26.1% (upper) of the training tensor is real camera data. Those zeros reached
    # the DISCRIMINATOR as "real" frames (teaching it that a constant-zero image is realistic,
    # i.e. directly rewarding a mean-collapsed generator), the feature-matching term and the
    # FSQ entropy statistic -- only the pixel anchor was masked, and even that mask was
    # ALL-ONES in practice (VideoCodecPairDataset built `torch.ones(cfg.frames)` unconditionally).
    #
    #   mask_missing          emit a REAL per-(C, T) validity mask from the loader's
    #                         `channel_valid` and honour it in every loss term + the gate.
    #                         False = the previous all-ones mask => BYTE-IDENTICAL.
    #   require_live_channels dataset-side: re-draw clips in which ANY channel is dead, so the
    #                         codec trains only on fully-live windows (the mask then has
    #                         nothing to exclude). False = the previous any-live rule.
    #   presence_filter       drop whole shots with no live camera for THIS divertor, using the
    #                         precomputed liveness cache (no HDF5 scan). False = no filtering.
    mask_missing: bool = False
    require_live_channels: bool = False
    presence_filter: bool = False

    # --- OPTIMIZER / SCHEDULE (NVIDIA Spectral Codec, arXiv 2406.05298 section 4) ------- #
    # Same fields + same defaults (= today's behaviour) as SpectroCodecConfig, so the video
    # trainer can run the paper recipe (betas 0.8/0.99, gamma 0.998 per 1k steps, D every 2)
    # and the checkpoint records which recipe it was trained under.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_decay_gamma: float = 1.0
    lr_decay_every: int = 1000
    disc_update_every: int = 1

    # --- decoder conv refinement head (patch-seam / checkerboard fix) ------------------ #
    # The linear per-token unpatchify renders every 20x20(x5) patch independently, which leaves
    # a visible 6x18 patch-seam checkerboard under the v6 GAN-free recipe (pixel+entropy only;
    # the config note above `pixel_anchor_weight` predicted exactly this: the linear head relied
    # on the adversarial+FM signal for plausible frames, and v6 removed both — confirmed on the
    # 2026-08-05 renders). ``refine_depth > 0`` appends a small RESIDUAL per-frame stride-1 2D
    # conv stack (kernel 3, ``refine_hidden`` channels, final conv ZERO-INIT so the head starts
    # as an exact identity) after the unpatchify, letting the loss blend across patch borders.
    # Deliberately NOT a ConvTranspose(kernel=stride) upsampler — that family is the documented
    # FAITH checkerboard bug; this head is stride-1 smoothing at full resolution. 0 = OFF
    # (byte-identical; old checkpoints unpickle without these fields and load unchanged).
    refine_depth: int = 0
    refine_hidden: int = 64
    # DILATED refinement: dilations 1,2,4,8,... so `refine_depth` convs reach a receptive field
    # of 2^(depth+1)-1 instead of 2*depth+1. The recorded lattice rule is that the head's
    # receptive field must EXCEED the patch size; the video patch is 20x20, so an undilated
    # head needs depth 10 while a dilated depth-4 head already reaches 31 > 20 at a quarter of
    # the cost. False = all dilation 1 => BYTE-IDENTICAL to the pre-2026-09-03 head.
    refine_dilated: bool = False

    # --- activity-stratified sampling (anti degenerate-window domination) ------------- #
    # Same lever as SpectroCodecConfig (activity = the clip's frame std here). tangtv frames
    # are NOT degeneracy-dominated in the diagnostic (all built clips have std >= ~3.8, like the
    # working spectro modalities) — the video collapse is an ADVERSARIAL-instability failure, not
    # a quiet-window one — so the trainer leaves `active_bias = 0` (OFF) for tangtv and instead
    # applies an adversarial warmup + lower adversarial_weight (see train_codec._activity_overrides).
    # The lever is still exposed here for symmetry / future divertors. Both DEFAULT 0 (OFF).
    min_activity: float = 0.0       # per-window activity threshold (clip frame std); 0 = OFF
    active_bias: float = 0.0        # P(re-draw a below-threshold clip toward active); 0 = OFF

    # oracle-gate acceptance thresholds (§4.4) — video analogues.
    gate_stability: float = 0.80        # frame-to-frame code stickiness proxy (see gate note)
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
    gate_hard_min_codes: int = 8         # hard best-ckpt floor on ABSOLUTE distinct codes (see SpectroCodecConfig)
    gate_min_code_entropy: float = 0.3
    # Reconstruction floor for best-ckpt DISQUALIFICATION (the only hard gate on the score):
    # a video codec is disqualified (gate_score = -inf) when frame reconstruction genuinely
    # fails (video envelope_corr NaN or < floor). Mirrors SpectroCodecConfig.gate_recon_floor.
    gate_recon_floor: float = 0.2

    @property
    def n_time_patch(self) -> int:
        return self.frames // self.patch_t

    @property
    def n_height_patch(self) -> int:
        return self.height // self.patch_h

    @property
    def n_width_patch(self) -> int:
        return self.width // self.patch_w

    @property
    def n_tok(self) -> int:
        return self.n_time_patch * self.n_height_patch * self.n_width_patch

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.fsq_levels)

    def __post_init__(self) -> None:
        assert self.frames % self.patch_t == 0, "frames must be divisible by patch_t"
        assert self.height % self.patch_h == 0, "height must be divisible by patch_h"
        assert self.width % self.patch_w == 0, "width must be divisible by patch_w"
        assert self.divertor in ("lower", "upper"), (
            f"divertor must be 'lower' or 'upper', got {self.divertor!r}"
        )


# ---------------------------------------------------------------------------------------- #
# Fast-TS codec (filterscopes / ELMs, Phase A) — SAMPLE-WISE RAW-WAVEFORM target.
# ---------------------------------------------------------------------------------------- #
# 2026-09-03 REDESIGN. The previous fast-TS codec did NOT reconstruct the filterscope signal:
# it encoded an ELM ACTIVITY ENVELOPE (E = 5 coarse RMS statistics per 50 ms window, 5 tokens)
# and its own design note said so explicitly ("NOT the raw spike waveform ... NOT spike
# timing"). That whole statistics layer is REMOVED. The codec now encodes and reconstructs the
# RAW 10 kHz samples themselves, sample by sample:
#
#     raw window   (C, W)   C = 8 filterscope channels, W = round(50 ms * 10 kHz) = 500
#                           => 8 * 500 = 4000 values per world-model frame
#     target       the SAME (C, W) array (per-channel standardized; nothing pooled away)
#
# CONSEQUENCES OF THE CHANGE (all deliberate):
#   * NO envelope / RMS / pooling / log1p compression. `pool`, `baseline_win`, `env_eps`,
#     `env_bins` and `patch_e` are gone.
#   * NO delta-shift consistency term by default. Consistency exists to make the codes
#     INVARIANT to sub-bin spike timing; sample-wise reconstruction needs the codes to CARRY
#     that timing, so the term is exactly counter-productive here (`consistency_weight` is
#     kept as a knob but defaults to 0.0 = OFF).
#   * The RECONSTRUCTION term dominates the objective (`pixel_anchor_weight` 1.0 vs the old
#     0.05) and the adversarial term defaults OFF (`adversarial_weight` 0.0). A GAN
#     hallucinates plausible-but-uncorrelated high-frequency detail, which RAISES sample-wise
#     nRMSE; it is available (`--adversarial_weight`) but is not the default objective.
#
# MEASURED DIFFICULTY (out-of-sample, 8 fit shots / 4 DIFFERENT score shots, 2026-09-03).
# nRMSE below is RMSE / std(target) per (window, channel) over the SAMPLE axis, so predicting
# the window's own constant mean scores EXACTLY 1.0000:
#   * lag-1 autocorrelation of a de-meaned window is only 0.20-0.48 and only 19-33% of the
#     variance sits below 500 Hz => the waveform is BROADBAND, near-incompressible.
#   * the OLD envelope resolution (5 bins of 10 ms) as a signed predictor scores 0.9630.
#   * out-of-sample per-channel PCA (coefficients given for FREE, unquantized): k=25 -> 0.9040,
#     k=64 -> 0.8411, k=128 -> 0.7528, k=400 -> 0.3690.
#   * Gaussian rate-distortion bound (reverse water-filling on the measured PSD), the BEST
#     POSSIBLE at a given token budget with 9.9658 bits/token over 4000 values:
#        5 tok (0.0125 b/value) -> >= 0.8697   |   25 tok (0.0623) -> ~0.766
#       50 tok (0.1246)         -> ~0.686      |  128 tok (0.3189) -> >= 0.5427
#   * a PRACTICAL fixed-rate transform coder (DCT lowpass + scalar quantizer) at ~26 tokens
#     reads 0.9771 and needs ~494 tokens to reach 0.7754.
# So: raw filterscope samples ARE largely unreconstructable at any affordable token count.
# `patch_w` is the rate knob and is measured, not assumed (see the arms in the session log).
#
# The codec tokenizes the (C, W) raw window like a 1-D signal: a PRE-PATCH CONV STEM at the
# full 10 kHz rate (the recorded recommendation for fast-TS spikes -- loss-weighting for
# spikes is a known dead end; a conv stem / overlapping patches is the alternative on record),
# then non-overlapping `patch_w`-sample patches over the TIME axis (channels are a feature
# dim carried into the patch, exactly as freq/space are for the other codecs), FSQ, and a
# decoder that mirrors the path back to (C, W) through a conv head.


# fast-TS geometry (matches the filterscopes SignalConfig + a 50 ms window at FASTTS_FS).
FASTTS_CHANNELS: int = 8         # channels_to_use=slice(0,8) on the filterscopes SignalConfig
FASTTS_WINDOW: int = round(CHUNK_S * FASTTS_FS)   # 500 raw samples / channel in a 50 ms window
# Default rate: 20 raw samples (2 ms) per token => n_tok = 500 // 20 = 25 tokens/frame.
# 25 * log2(1000) = 249.1 bits over 4000 values = 0.0623 bits/value. World-model frame
# 1017 - 5 + 25 = 1037 tokens (+2.0%). W must be divisible by patch_w; the divisors of 500 are
# 1,2,4,5,10,20,25,50,100,125,250,500 -> reachable token counts 500,250,125,100,50,25,20,10,5,4,2,1.
FASTTS_PATCH_W: int = 20
# --- ENVELOPE-mode geometry (the ORIGINAL, still the DEFAULT) -------------------------- #
# The raw-sample codec above is OPT-IN (`target="raw"`). The default `target="envelope"`
# reproduces the pre-2026-09-03 ELM-activity-envelope codec BYTE-IDENTICALLY, so every
# existing pickled FastTSCodecConfig and every existing filterscopes checkpoint
# (models/ignite_codecs_prod_noinorm/filterscopes/codec_best.pt and every
# eval_runs/ignite_codec_* fast-TS run) still loads with a STRICT load_state_dict.
# This matters far beyond fast-TS: train_dynamics._load_codec builds the FULL 14-modality
# codec set, # codec set one family at a time, so a fast-TS load failure kills every consumer of the full
# figure renders for shots that have nothing to do with filterscopes.
FASTTS_POOL: int = round(0.010 * FASTTS_FS)          # envelope pooling bin = 100 samples = 10 ms
FASTTS_ENV_BINS: int = FASTTS_WINDOW // FASTTS_POOL  # 5 envelope bins (10 ms each)


@dataclass
class FastTSCodecConfig:
    """Fast-TS (filterscopes) codec (Phase A), RAW-SAMPLE domain.

    Tensor-shape conventions (batch-first):

        raw window       (B, C, W)          C channels, W = window_samples raw samples
                                            (this IS the codec input AND the recon target)
        pre-FSQ feats    (B, n_tok, d_model)
        fsq codes (int)  (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
        quantized        (B, n_tok, d_model)
        reconstruction   (B, C, W)          reconstructed RAW samples

        n_tok = window // patch_w
    """

    # --- WHICH CODEC THIS IS. "envelope" (DEFAULT) = the ORIGINAL ELM-activity-envelope
    # codec, byte-identical to the pre-2026-09-03 build: same module names, same tensor
    # shapes, same objective defaults, so every existing pickled config and checkpoint loads
    # strictly. "raw" = the 2026-09-03 SAMPLE-WISE codec described in the section note above.
    # The switch is read with getattr(cfg, "target", "envelope") everywhere, because configs
    # pickled before this field existed do not carry it and MUST take the envelope path.
    # Build a raw-mode config with :func:`fastts_raw_config`, which also flips the objective
    # defaults (the envelope defaults below are the OLD ones and are wrong for raw samples).
    target: str = "envelope"

    # data / shape
    channels: int = FASTTS_CHANNELS       # 8 filterscope channels
    # -- envelope mode (default) --
    env_bins: int = FASTTS_ENV_BINS       # E = 5 (10 ms per bin over a 50 ms window)
    patch_e: int = 1                      # -> 5 time patches (10 ms each) => n_tok = 5
    pool: int = FASTTS_POOL               # RMS pooling bin (samples) = 100 = 10 ms
    baseline_win: int = 20                # moving-mean baseline window (samples) = 2 ms; 0 = off
    env_eps: float = 1e-3                 # log1p reference: log1p(env / env_eps)
    # -- raw mode (target="raw"); INERT in envelope mode --
    window: int = FASTTS_WINDOW           # W = 500 raw samples (50 ms at 10 kHz)
    patch_w: int = FASTTS_PATCH_W         # raw samples per token => n_tok = window // patch_w

    # PER-CHANNEL RAW STANDARDIZATION. The raw filterscopes signal is UNSTANDARDIZED with
    # per-channel magnitudes ~1e15 and per-channel std ~1e16, so it must be put on the O(1)
    # scale the network is sized for. The data_loader consumes STANDARDIZED filterscopes
    # (SignalConfig preprocess method="standardize", per-channel raw mean/std from
    # preprocessing_stats.pt), so the codec standardizes the SAME way: x <- (x - mean) / std.
    # GLOBAL / per-channel (NOT per-window) so the relative activity LEVEL is preserved.
    # Length == channels. None (the default) => NO standardization (synthetic tests / callers
    # that already hand in O(1) data are unaffected); the trainer loads + injects the real
    # per-channel stats (see fastts_train.load_fastts_channel_stats).
    #
    # NOTE the gate's nRMSE is invariant to this affine map (it divides by each
    # (window, channel)'s OWN std), so standardization changes optimization, never the score.
    channel_mean: Optional[Sequence[float]] = None
    channel_std: Optional[Sequence[float]] = None

    # bottleneck (vector-quantize-pytorch FSQ): [8, 5, 5, 5] = 1000 codes = 9.9658 bits/token,
    # 4 dims. 4 dims avoids the spare-dim death seen at [8,8,8,8,8]=32768 (min_dim_entropy=0),
    # and cb=1000 keeps joint_entropy_weight (the anti-collapse lever that fixed mhr's
    # 42/32768) affordable -- it needs an (N, codebook_size) soft-assignment tensor.
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = 1000

    # --- PRE-PATCH CONV STEM (the fast-TS spike recommendation on record) -------------- #
    # Runs at the FULL 10 kHz rate BEFORE patching, so a spike that straddles a patch seam is
    # still seen whole: with stem_kernel=15 and stride 1 every output sample mixes +-7 raw
    # samples, i.e. the effective receptive field OVERLAPS neighbouring patches (the
    # "overlapping patches" half of the same recommendation, implemented once). Set
    # stem_layers=0 for the plain linear-patch path (no stem) as a control arm.
    stem_channels: int = 64
    stem_kernel: int = 15
    stem_layers: int = 2

    # transformer (x-transformers) — token set over raw-sample patches, bidirectional.
    d_model: int = 256
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8

    # --- objective ---------------------------------------------------------------------- #
    # RECONSTRUCTION-DOMINANT. The three simultaneous objectives are (1) low sample-wise
    # nRMSE, (2) forecastable codes, (3) >=90% codebook utilization. (1) is served by making
    # the reconstruction term the LEADING term -- the opposite of the shipped envelope codec,
    # which ran pixel_anchor_weight 0.05 against adversarial 1.0 (recon weighted ~20x BELOW
    # the GAN). The NVIDIA Spectral Codec reference (arXiv 2406.05298) weights the other way
    # by 21:1; we go reconstruction-only by default.
    pixel_anchor_weight: float = 0.05     # OLD envelope default; fastts_raw_config sets 1.0
    # "nrmse" | "nmse" | "mse" | "l1" | "huber". DEFAULT "nrmse" == THE REPORTED METRIC.
    # Plain "mse" is NOT a surrogate for it: measured on 4 held-out shots, the per-(window,
    # channel) within-window std is ~0.0073 while the RMS window-mean offset is ~25x larger,
    # so an MSE optimum spends its capacity on the DC level and emits a FLAT window -- and a
    # flat line at the window's own mean scores nRMSE EXACTLY 1.0000, i.e. the trivial
    # baseline. (Demonstrated: a flat-line prediction reads mse 5.0e-05, which looks like a
    # near-perfect reconstruction, and nrmse 1.0000, which is the truth.) The metric divides
    # by each pair's own std, so the loss must too.
    # "nrmse" is preferred over "nmse" for CONDITIONING, not for a different objective: the
    # per-pair ratio spans ~1e6 across the population (the quietest windows have std 0.0009
    # against a p90 of 0.071), so the un-rooted "nmse" is dominated by a handful of quiet,
    # pure-noise windows; the square root compresses that range to ~1e3 and is exactly what
    # the gate reports. "mse"/"l1"/"huber" are kept as un-normalized controls so the
    # flat-line failure can be demonstrated rather than asserted.
    # See FastTSCodec._recon_loss for the full derivation.
    recon_loss: str = "nrmse"
    # --- STRUCTURAL (SSIM) TERM: the amplitude/dynamic-range guard on the objective ----
    # nRMSE's exact minimiser is the conditional mean, so a codec trained on it SHRINKS
    # toward the local mean under uncertainty: at prediction/truth correlation r the
    # nRMSE-optimal output scale is r itself, so r = 0.4 gives an output with 40% of the
    # correct amplitude while nRMSE reads a respectable 0.917. Measured on real held-out
    # windows (gate.fastts_structural_references): the 0.4x-shrunk target scores nRMSE
    # 0.6000 -- which LOOKS good against the 1.0000 anchor -- while its std_ratio is 0.4000
    # and its SSIM contrast factor is 0.6930; a 0.9 ms-smoothed target scores nRMSE 0.8747
    # (also "beats the anchor") with contrast 0.5430 and std_ratio 0.4875. Element-wise
    # error cannot see either failure; the SSIM CONTRAST factor sees both.
    # ssim_weight > 0 adds (1 - contrast*structure) to the reconstruction reference. DEFAULT
    # 0.0 (OFF) so the plain arm is an unchanged-recipe CONTROL and the term's effect is
    # measured, not assumed. NOTE this attacks spike preservation through LOCAL VARIANCE,
    # which is a different mechanism from the per-sample spike loss-WEIGHTING that is a
    # recorded dead end for this modality.
    ssim_weight: float = 0.0
    # Box-window length in SAMPLES for the 1-D SSIM (17 = 1.7 ms at 10 kHz ~ one ELM burst
    # period; filterscope bursts recur every ~1.6 ms). Must match gate._FASTTS_SSIM_WIN for
    # the loss and the reported metric to be the same quantity.
    ssim_win: int = 17

    # Adversarial / feature-matching DEFAULT OFF: a GAN synthesizes uncorrelated high-frequency
    # detail, which is exactly what a sample-wise nRMSE punishes. Kept as knobs so the
    # trade-off can be measured (--adversarial_weight / --fm_weight), not deleted.
    adversarial_weight: float = 1.0       # OLD envelope default; fastts_raw_config sets 0.0
    fm_weight: float = 1.0                # OLD envelope default; fastts_raw_config sets 0.0

    # delta-shift consistency: OFF by default (see the section note -- invariance to spike
    # timing directly contradicts sample-wise reconstruction). Range kept for the gate's
    # stability probe, which still reports how much the codes move under a shift.
    consistency_delta_ms: Tuple[float, float] = (0.1, 5.0)
    consistency_weight: float = 1.0       # OLD envelope default; fastts_raw_config sets 0.0

    # VQGAN/MagViT adaptive adversarial weight ("Taming Transformers" 3.3). Inert while
    # adversarial_weight == 0.
    adaptive_adv_weight: bool = True
    adaptive_adv_clamp: float = 50.0
    adv_warmup_steps: int = 0

    # anti-collapse (codebook-utilization) entropy regularizer — Genie/LFQ style, reused from
    # the shared SpectroQuantizer.entropy_loss (per-sample entropy - diversity * batch-mean).
    entropy_weight: float = 1.0
    diversity_weight: float = 1.0
    # JOINT-code diversity: the lever that took mhr from 42/32768 to 1000/1000 codes. Affordable
    # here because cb=1000 (the (N, cb) soft-assignment tensor is small). 0.0 = OFF.
    joint_entropy_weight: float = 0.0
    joint_entropy_ramp_steps: int = 0

    # --- activity-stratified sampling (bias toward the more-active windows) ------------ #
    # activity = the RAW window's std (per-window, in standardized units). Measured on the
    # 4 held-out score shots: per-(window, channel) std p25/p50/p75 = 0.0025/0.0073/0.0121,
    # and the TOP std quartile is by far the most reconstructable (oracle 0-1 kHz lowpass
    # nRMSE 0.3956 there vs 0.68-0.73 in the lower three quartiles), so biasing toward active
    # windows biases toward the part of the signal that HAS reconstructable structure.
    # Both DEFAULT 0 (OFF); the trainer may turn them on (see train_codec._activity_overrides).
    min_activity: float = 0.0       # per-window activity threshold (raw window std); 0 = OFF
    active_bias: float = 0.0        # P(re-draw a below-threshold window toward active); 0 = OFF

    # --- GAIN-SHAPE DECOMPOSITION (2026-09-03) — ENVELOPE mode only, DEFAULT OFF -------- #
    # WHY. Measured on 4 held-out shots / 483 windows: 96.2% of the ELM-envelope variance is
    # the BETWEEN-WINDOW DC LEVEL (each (window, channel)'s mean over the E=5 bins). A trivial
    # encoder that transmits only that level, quantised to the codec's OWN 49.83-bit budget
    # (5 tokens x log2(1000)), scores pooled nRMSE 0.1935 — 2.7x BETTER than the shipped
    # codec's 0.5252. The shipped codec loses because it spends its bits against the variance
    # structure: level and shape share every token, and the level comes out only correlated
    # (r 0.870), not exact.
    #
    # THE DECOMPOSITION. With gain_shape ON the codec codes
    #     x[c, e]  =  level[c]  +  sigma[c] * shape[c, e]
    # where level = mean_e x, sigma = std_e (x - level), and shape has zero mean and unit std
    # by CONSTRUCTION. The first `gain_tokens` tokens carry (level, log1p sigma) through a
    # dedicated MLP -> FSQ path that never sees the shape; the remaining tokens carry the
    # normalized shape. The DECODER enforces the split structurally: the shape branch's output
    # is mean-removed (and, when gain_scale, std-normalized) before it is combined, so the
    # reconstruction's per-(window, channel) mean is produced by the GAIN head ALONE.
    #
    # WHAT THAT BUYS. The per-window-mean anchor (0.1919) becomes a FLOOR the codec cannot
    # fall below rather than a bar it fails: zero out the shape path and the codec IS the
    # quantised-window-mean encoder. The shape path can then only add. And because sigma is
    # transmitted EXPLICITLY, the reconstruction's within-window amplitude is set by a
    # decoded number rather than by an nRMSE-minimising conditional mean — the direct attack
    # on the measured 36% burst-height retention (channel 0: 8%).
    #
    # DEFAULT False => every field below is INERT and the module is byte-identical to the
    # pre-2026-09-03 build (proved by tests/test_fastts_gain_shape.py::test_gain_shape_off_is_bit_identical).
    gain_shape: bool = False
    # Tokens (of the n_env_patch = 5 total) reserved for the GAIN code. The rest carry shape.
    # gain_tokens == n_env_patch is legal and means LEVEL ONLY (no shape path) — the direct
    # learned-VQ analogue of the rate-matched baseline, and the cleanest test of whether the
    # 49.83-bit budget can code the 8-channel level vector at all.
    gain_tokens: int = 1
    # Transmit the per-(window, channel) AC scale sigma alongside the level, and normalize the
    # decoded shape to unit std so sigma ALONE sets the within-window amplitude (textbook
    # gain-shape VQ: gain = norm, shape = unit vector). False => level only; the shape path's
    # amplitude is then free and can shrink toward the conditional mean. Forced False when
    # there is no shape path (gain_tokens == n_env_patch), where it would be inert anyway.
    gain_scale: bool = True
    # Weight on the AUXILIARY direct supervision of the gain head, |pred - (level, log1p sigma)|.
    # Part of the RECONSTRUCTION reference (it is a fidelity term). 0 = OFF (the gain path is
    # then trained only through the reconstruction term).
    gain_weight: float = 1.0
    # Hidden width of the gain MLP (encoder side and decoder side).
    gain_hidden: int = 256

    # oracle-gate acceptance thresholds (4.4) — fast-TS analogues.
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
    gate_hard_min_codes: int = 8         # hard best-ckpt floor on ABSOLUTE distinct codes
    gate_min_code_entropy: float = 0.3
    # Reconstruction floor for best-ckpt DISQUALIFICATION. spike.gate_score reads
    # decode["envelope_corr"], which for this codec is the SAMPLE-WISE Pearson correlation on
    # the raw waveform (see gate.fastts_decode_fidelity). That number is intrinsically much
    # smaller than an envelope correlation -- the measured out-of-sample linear ceiling at a
    # 25-token budget is corr ~0.37 -- so the 0.2 floor used for the envelope codec would
    # disqualify every honest checkpoint for most of training. Floor lowered to 0.02
    # (still above a dead 1-code codec, which reads ~0.0).
    gate_recon_floor: float = 0.2         # OLD envelope default; fastts_raw_config sets 0.02

    @property
    def is_raw(self) -> bool:
        """True iff this is the 2026-09-03 sample-wise codec. Old pickles have no field."""
        return getattr(self, "target", "envelope") == "raw"

    @property
    def n_env_patch(self) -> int:
        """Envelope-mode token count (kept for the pre-2026-09-03 API)."""
        return self.env_bins // self.patch_e

    @property
    def n_gain_tok(self) -> int:
        """Tokens carrying the GAIN code (0 unless gain_shape is on in envelope mode)."""
        if self.is_raw or not getattr(self, "gain_shape", False):
            return 0
        return int(getattr(self, "gain_tokens", 1))

    @property
    def n_shape_tok(self) -> int:
        """Tokens carrying the level-normalized SHAPE (== n_tok when gain_shape is off)."""
        return self.n_tok - self.n_gain_tok

    @property
    def gain_values(self) -> int:
        """Numbers the gain path transmits: C levels (+ C log1p sigmas when gain_scale)."""
        if self.n_gain_tok == 0:
            return 0
        return self.channels * (2 if self.uses_gain_scale else 1)

    @property
    def uses_gain_scale(self) -> bool:
        """gain_scale, forced OFF when there is no shape path for it to normalize."""
        return bool(getattr(self, "gain_scale", True)) and self.n_shape_tok > 0

    @property
    def gain_bits(self) -> float:
        """Bits the GAIN path owns (n_gain_tok * log2(codebook))."""
        return self.n_gain_tok * log2(self.codebook_size)

    @property
    def n_tok(self) -> int:
        return (self.window // self.patch_w) if self.is_raw else self.n_env_patch

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.fsq_levels)

    @property
    def window_samples(self) -> int:
        """Raw samples in a 50 ms window at FASTTS_FS (== self.window)."""
        return int(self.window)

    @property
    def bits_per_value(self) -> float:
        """Token bits per RAW VALUE: n_tok * log2(codebook) / (channels * window).

        The rate the reconstruction has to live within. 25 tokens of a 1000-code codebook over
        8 x 500 = 4000 values is 0.0623 bits/value.
        """
        n_values = self.channels * (self.window if self.is_raw else self.env_bins)
        return self.n_tok * log2(self.codebook_size) / n_values

    def __post_init__(self) -> None:
        if getattr(self, "target", "envelope") not in ("envelope", "raw"):
            raise ValueError(f"target must be 'envelope' or 'raw', got {self.target!r}")
        if self.is_raw:
            assert self.patch_w >= 1, "patch_w must be >= 1"
            assert self.window % self.patch_w == 0, (
                f"window ({self.window}) must be divisible by patch_w ({self.patch_w}); "
                f"divisors of 500 are 1,2,4,5,10,20,25,50,100,125,250,500"
            )
            assert self.stem_layers >= 0, "stem_layers must be >= 0 (0 disables the conv stem)"
            assert self.stem_kernel % 2 == 1, "stem_kernel must be odd ('same' padding)"
        else:
            assert self.env_bins % self.patch_e == 0, "env_bins must be divisible by patch_e"
            assert self.pool >= 1, "pool must be >= 1"
            assert self.baseline_win >= 0, "baseline_win must be >= 0 (0 disables detrend)"
            if getattr(self, "gain_shape", False):
                assert 1 <= self.gain_tokens <= self.n_env_patch, (
                    f"gain_tokens ({self.gain_tokens}) must be in [1, n_env_patch="
                    f"{self.n_env_patch}]"
                )
                assert self.gain_hidden >= 1, "gain_hidden must be >= 1"
        if getattr(self, "gain_shape", False) and self.is_raw:
            raise ValueError(
                "gain_shape is an ENVELOPE-mode decomposition (level = mean over the E bins); "
                "it has no meaning for target='raw'. Use target='envelope'."
            )


def fastts_raw_config(**overrides) -> FastTSCodecConfig:
    """Build the 2026-09-03 SAMPLE-WISE fast-TS config (``target="raw"``).

    :class:`FastTSCodecConfig`'s bare defaults are the ORIGINAL envelope ones, so that every
    config pickled before the redesign — and every checkpoint written against it — keeps
    loading strictly. The raw codec needs a different objective as well as a different target,
    so this factory flips both together rather than leaving a half-configured object:

        target                 "envelope"  ->  "raw"
        pixel_anchor_weight    0.05        ->  1.0    (reconstruction LEADS the objective)
        adversarial_weight     1.0         ->  0.0    (a GAN raises sample-wise nRMSE)
        fm_weight              1.0         ->  0.0
        consistency_weight     1.0         ->  0.0    (a sample-wise codec must CARRY timing)
        gate_recon_floor       0.2         ->  0.02   (sample-wise corr is much smaller than
                                                       an envelope corr; 0.2 would disqualify
                                                       every honest checkpoint)

    Any keyword overrides are applied last, so a caller still wins.
    """
    base = dict(
        target="raw",
        pixel_anchor_weight=1.0,
        adversarial_weight=0.0,
        fm_weight=0.0,
        consistency_weight=0.0,
        gate_recon_floor=0.02,
    )
    base.update(overrides)
    return FastTSCodecConfig(**base)


# ---------------------------------------------------------------------------------------- #
# Slow-TS codec (Thomson / CER / MSE, Phase A) — the "lightest touch" statistics-first codec.
# ---------------------------------------------------------------------------------------- #
# Design (docs/IGNITE_DESIGN.md §4.3): "Slow-TS — smooth profiles; lightest touch." These are
# smooth kinetic-profile time-series (electron density/temperature profiles from Thomson
# scattering, ion temperature / rotation from charge-exchange, and the motional-Stark-effect
# pitch-angle profile). Each is loaded at target_fs = 100 Hz (SLOWTS_FS), so a 50 ms
# world-model frame is a tiny (C_positions, T=5) window: a short slice of a smooth
# position-vs-time profile.
#
# This config MIRRORS the spectro/video codec INFRASTRUCTURE (FSQ bottleneck, entropy /
# codebook-utilization regularizer, gate thresholds, gate_recon_floor) but drops everything
# spectrogram-specific, and — being the *lightest touch* — everything adversarial too:
#   * NO δ-shift consistency term. Consistency projects out an STFT-phase / speckle
#     *realization* nuisance that exists only for spectrograms (a 0.5 ms sub-window shift
#     scrambles ~74 % of spectro codes). A slow-TS profile has no such sub-window-phase
#     realization — it is a smooth, directly-sampled physical quantity — so there is no
#     nuisance pair to be invariant to.
#   * NO adversarial / discriminator / feature-matching. Two reasons: (1) §4.3 scopes this as
#     the lightest touch — a smooth low-dimensional profile is not a texture-rich signal where
#     a GAN buys sharpness (that was the spectrogram / video mean-collapse problem); a plain
#     masked reconstruction already captures a smooth profile faithfully. (2) CER/MSE/Thomson
#     carry REAL missingness (neutral-beam-off gaps; diagnostic-not-firing zeros). A generative
#     adversarial decoder would *hallucinate* plausible values into those masked regions —
#     exactly the wrong behaviour for a kinetic profile, where "missing" must stay masked, not
#     invented. So the codec is a reconstruction + entropy codec (no `decoder.last_layer`
#     adaptive-adv machinery needed), with the reconstruction MASKED where the signal is
#     genuinely missing.
#
# ONE codec class parameterized by `signal` (like SpectroCodec is parameterized by cfg): each
# of the 7 signals gets its OWN cfg (different `channels` = profile positions + patch sizes)
# and is trained + frozen INDEPENDENTLY. `signal` is documentation/logging + the loader's
# channel-count + missingness-policy selector; the tensor contract is identical across signals.

# The 7 slow-TS signals + their loaded channel counts (positions) — read from
# data_loader.TokamakH5Dataset.SIGNAL_CONFIGS (num_channels; none of the 7 use channels_to_use)
# so this stays a convenience default, NOT a hard-coded physics claim (the trainer reads the
# real count from the loader at runtime, exactly like the spectro/video codecs).
SLOWTS_SIGNALS: Tuple[str, ...] = (
    "ts_core_density", "ts_core_temp", "ts_tangential_density", "ts_tangential_temp",
    "cer_ti", "cer_rot", "mse",
)
# positions (channel count) per signal, from data_loader SIGNAL_CONFIGS.num_channels.
SLOWTS_POSITIONS: Dict[str, int] = {
    "ts_core_density": 44, "ts_core_temp": 44,
    "ts_tangential_density": 10, "ts_tangential_temp": 10,
    "cer_ti": 48, "cer_rot": 48, "mse": 69,
}
# Signals whose missingness is `zero_is_missing` (Thomson spikes-to-zero) vs an explicit NaN
# mask (CER/MSE neutral-beam gaps). Mirrors data_loader.SignalConfig.zero_is_missing so the
# dataset builds the SAME validity mask the loader would (`data != 0` for zero_is_missing;
# `~isnan(data)` otherwise). NOT a new policy — a read of the loader's config.
SLOWTS_ZERO_IS_MISSING: Dict[str, bool] = {
    "ts_core_density": True, "ts_core_temp": True,
    "ts_tangential_density": True, "ts_tangential_temp": True,
    "cer_ti": False, "cer_rot": False, "mse": False,
}
# Per-signal preprocessing METHOD the FM model applies to each slow-TS signal — a READ of
# data_loader.TokamakH5Dataset.SIGNAL_CONFIGS[*].preprocess.method (NOT a new policy). The codec
# must standardize its input the SAME way the FM does (the SCALE FIX; see the module note on
# SlowTSCodecConfig.channel_mean/std): the 4 Thomson signals are "log_standardize"
# (log10(clip(x,-0.99)+1) then per-channel (x-mean)/std, LOG-space stats) and cer_ti/cer_rot/mse
# are "standardize" (per-channel (x-mean)/std, RAW-space stats). Selects which stats sub-dict of
# preprocessing_stats.pt is read ('log' for log_standardize, 'raw' for standardize) exactly like
# data_loader._update_preprocessing_stats.
SLOWTS_PREPROCESS_METHOD: Dict[str, str] = {
    "ts_core_density": "log_standardize", "ts_core_temp": "log_standardize",
    "ts_tangential_density": "log_standardize", "ts_tangential_temp": "log_standardize",
    "cer_ti": "standardize", "cer_rot": "standardize", "mse": "standardize",
}


@dataclass
class SlowTSCodecConfig:
    """Statistics-first slow-TS codec (Phase A), per-signal.

    Seven instances are used in production — one per :data:`SLOWTS_SIGNALS` — each its own
    encoder/decoder (no discriminator; see the module note above), matching §4.3's "lightest
    touch" for smooth profiles. ``signal`` selects the channel (position) count + missingness
    policy + logging name; the tensor contract is identical across signals.

    Tensor-shape conventions (batch-first):

        signal window   (B, C, T)    C = profile positions, T = time samples (5 @ 50 ms/100 Hz)
        validity mask   (B, C, T)    1.0 = valid (real) sample, 0.0 = missing/padded
        pre-FSQ feats    (B, n_tok, d_model)
        fsq codes (int) (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
        quantized       (B, n_tok, d_model)
        reconstruction  (B, C, T)

        n_tok = (C // patch_c) * (T // patch_t)
    """

    # signal selector (documentation / logging + missingness policy; loader maps to channels).
    signal: str = "ts_core_density"

    # data / shape
    channels: int = 44            # C = REAL profile positions (read from the loader at runtime).
    time_steps: int = 5           # T; a 50 ms window at SLOWTS_FS=100 Hz = round(0.05*100) = 5.
    # DESIGNED per-frame budget (Phase-B frame layout): EXACTLY 4 tokens / slow-TS signal =
    # 4 radial-zone patches × 1 time-patch. The C profile positions are split into 4 contiguous
    # radial zones of `patch_c = ceil(C / 4)` positions each; when C is not divisible by 4 the
    # profile is PADDED up to `padded_channels = 4 * patch_c` and the padded tail positions are
    # MASKED as missing (they never contribute to the encoder input or the loss — see
    # SlowTSCodecPairDataset._fit_window). `patch_c` here is the per-ZONE position count; the
    # encoder/decoder patchify over `padded_channels` (NOT `channels`).
    #   n_pos_patch = 4  (the zone count);  n_tok = 4 * 1 = 4.
    # The `slowts_patch_for` helper computes patch_c = ceil(channels / n_zones) for you.
    n_zones: int = 4              # radial-zone patches (position-patches) -> n_pos_patch
    patch_c: int = 11             # positions per radial zone = ceil(44 / 4) = 11 for the default C.
    patch_t: int = 5              # time samples per patch (default: whole window = 1 time-patch).

    # PER-CHANNEL RAW STANDARDIZATION (the SCALE FIX; see SlowTSCodecPairDataset._standardize).
    # The raw slow-TS signal is UNSTANDARDIZED, and the high-magnitude Thomson DENSITY signals
    # carry ~1e19-scale samples (electron density ~1e19 m^-3): fed straight into the codec encoder
    # they collapse (ts_core_density -> 1 code, env_corr=NaN). The FM model consumes STANDARDIZED
    # slow-TS — each signal's SignalConfig.preprocess.method applied: "log_standardize" for the 4
    # Thomson signals, "standardize" for cer_ti/cer_rot/mse — so the codec must standardize the
    # SAME way BEFORE the encoder, putting every signal on the ~O(1) scale the FM sees. This is
    # GLOBAL / per-channel (NOT per-window) so the relative PROFILE LEVEL (which positions carry
    # more, quiet vs active windows) is preserved. Length == channels.
    #
    # `preprocess_method` selects the transform (mirrors data_loader._apply_preprocessing EXACTLY):
    #   "standardize"     -> (x - mean) / std.clamp(min=1e-3)                         [RAW-space stats]
    #   "log_standardize" -> arr = log10(clip(x, min=-0.99) + 1); (arr - mean)/std.clamp(min=1e-3)
    #                        [LOG-space stats; the loader reads the 'log' sub-dict for this method]
    #   "none" / None     -> IDENTITY (byte-identical to the pre-fix path; the default).
    # channel_mean / channel_std being None is ALSO the identity (no-op) regardless of method, so
    # synthetic tests / stat-less callers are unaffected. The trainer loads + injects the real
    # per-channel stats from preprocessing_stats.pt (see train_codec.load_slowts_channel_stats).
    preprocess_method: Optional[str] = None
    channel_mean: Optional[Sequence[float]] = None
    channel_std: Optional[Sequence[float]] = None

    # bottleneck (vector-quantize-pytorch FSQ) — right-sized like the spectro/video fix:
    # [8, 5, 5, 5] = prod = 1000 codes, 4 dims. A smooth low-D profile needs even fewer codes
    # than a spectrogram, so 1000 is a comfortable start (the entropy regularizer prunes the
    # rest). The FSQ math is REUSED verbatim from the spectro quantizer.
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = 1000

    # transformer (x-transformers) — a modest patch/attention encoder (these are smooth,
    # low-dimensional signals, so a small transformer is ample; matches §4.3 "lightest touch").
    d_model: int = 128
    enc_depth: int = 4
    dec_depth: int = 4
    heads: int = 4

    # reconstruction objective — MASKED (see below) reconstruction only. No adversarial,
    # no feature-matching, no consistency (all justified in the module note). The pixel anchor
    # here is the WHOLE reconstruction signal (there is no GAN to balance against), so it is
    # weight 1.0 — not a small "anchor" as in the generative spectro/video codecs.
    recon_weight: float = 1.0

    # MASKED SQUARED-ERROR companion (2026-09-03; default 0.0 = OFF, byte-identical).
    # ``recon_weight`` above weights an L1 (MAE) term, whose optimum is the conditional MEDIAN.
    # The reconstruction objective we are actually judged on — gate.full_slowts_metrics'
    # ``slowts_nrmse`` = RMSE/std(target) — is a SQUARED error, whose optimum is the conditional
    # MEAN. On a bimodal, heavily-masked profile those differ, and an L1-only codec is measured
    # over-smoothed (prod ts_core_density: reconstruction within-window std is 0.45x the target's,
    # MAE 0.318 against 0.361 for predicting the window's own constant — i.e. it captured only 12%
    # of the deviation). This term adds ``recon_mse_weight * masked_MSE`` so the objective can be
    # aligned with the metric; set ``recon_weight 0`` for pure L2, leave both > 0 for a mix.
    recon_mse_weight: float = 0.0

    # anti-collapse (codebook-utilization) entropy regularizer — Genie/LFQ style, REUSED
    # verbatim from the spectro quantizer's entropy_loss (per-sample entropy - diversity *
    # batch-mean). The masked-recon objective alone does not pressure codebook diversity, so
    # this keeps the encoder from posterior-collapsing to a single code.
    # entropy_weight=1.0 (the diversity-spike value, min_dim_entropy≈0.65); 0.1 was the
    # known-collapsing value. Paired with FIX 1 (global DDP batch-mean). See FIX 2.
    entropy_weight: float = 1.0
    diversity_weight: float = 1.0

    # JOINT-code diversity / dim-decorrelation (2026-09-03) — the SAME anti-collapse knobs the
    # spectro config carries, added here because ``SlowTSQuantizer`` IS ``SpectroQuantizer`` and
    # its ``entropy_loss`` already reads these fields via ``getattr(cfg, ..., 0.0)``. Adding the
    # fields therefore changes NOTHING at the defaults (0.0 = OFF, byte-identical) and simply
    # makes ``--joint_entropy_weight`` / ``--decorrelation_weight`` reachable for slow-TS instead
    # of a SystemExit. The per-dim ``diversity_weight`` reward above is MARGINAL and a rank-1
    # encoder saturates it while using almost no joint codes (measured on the prod mhr spectro
    # codec: min_dim_entropy 0.921 with 42 of 32768 joint codes); ``joint_entropy_weight`` is the
    # lever that fixed that. Safe here because the slow-TS codebook is 1000 codes, far below the
    # quantizer's joint-entropy codebook-size limit.
    joint_entropy_weight: float = 0.0
    joint_entropy_ramp_steps: int = 0
    decorrelation_weight: float = 0.0

    # --- activity-stratified sampling (anti degenerate-window domination) ------------- #
    # Same lever as SpectroCodecConfig, but slow-TS is a MASKED modality, so ACTIVITY here is the
    # window's PRESENT-FRACTION (mask.mean()), NOT a std. ts_core_density is strongly bimodal in
    # the diagnostic: median present-fraction ~0.07 (mostly-missing) but ~34% of windows are >=75%
    # present — so ~2/3 of windows are near-empty. Biasing toward present-fraction >= min_activity
    # pulls the batch onto the well-observed windows. Both DEFAULT 0 (OFF); the trainer turns them
    # ON for ts_core_density (see train_codec._activity_overrides). min_activity is a present-
    # fraction in [0, 1] here (interpretation differs from the std-based codecs by design).
    min_activity: float = 0.0       # per-window present-fraction threshold in [0,1]; 0 = OFF
    active_bias: float = 0.0        # P(re-draw a below-threshold window toward active); 0 = OFF

    # oracle-gate acceptance thresholds (§4.4) — slow-TS analogues (same fields the shared
    # spike.gate_score / gate.utilization read).
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
    gate_hard_min_codes: int = 8         # hard best-ckpt floor on ABSOLUTE distinct codes (see SpectroCodecConfig)
    gate_min_code_entropy: float = 0.3
    # Reconstruction floor for best-ckpt DISQUALIFICATION (the only hard gate on the score):
    # a slow-TS codec is disqualified (gate_score = -inf) when profile reconstruction genuinely
    # fails (envelope_corr NaN or < floor). Mirrors SpectroCodecConfig.gate_recon_floor.
    gate_recon_floor: float = 0.2

    @property
    def zero_is_missing(self) -> bool:
        """Missingness policy for this signal (mirrors data_loader.SignalConfig.zero_is_missing).

        True  -> Thomson spikes-to-zero: a sample is missing where the raw value is 0.
        False -> CER/MSE neutral-beam gaps: a sample is missing where the raw value was NaN.
        """
        return SLOWTS_ZERO_IS_MISSING.get(self.signal, False)

    @property
    def padded_channels(self) -> int:
        """Position count the encoder/decoder actually patchify over (``n_zones * patch_c``).

        ``>= channels``; the extra ``padded_channels - channels`` tail positions are the
        radial-zone padding, MASKED as missing by :meth:`SlowTSCodecPairDataset._fit_window` so
        they never contribute to the encoder input or the reconstruction loss. Equals
        ``channels`` exactly when ``channels`` is divisible by ``n_zones`` (no padding needed).
        """
        return self.n_zones * self.patch_c

    @property
    def n_pos_patch(self) -> int:
        # The radial-zone count IS n_zones (padded_channels // patch_c == n_zones by construction).
        return self.n_zones

    @property
    def n_time_patch(self) -> int:
        return self.time_steps // self.patch_t

    @property
    def n_tok(self) -> int:
        return self.n_pos_patch * self.n_time_patch

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.fsq_levels)

    @property
    def window_samples(self) -> int:
        return round(CHUNK_S * SLOWTS_FS)

    def __post_init__(self) -> None:
        # `patch_c` is the per-ZONE position count; the encoder/decoder patchify over
        # `padded_channels = n_zones * patch_c` (divisible by patch_c BY CONSTRUCTION). The real
        # `channels` need NOT be divisible by patch_c — the profile is padded up to
        # `padded_channels` and the pad tail is masked missing. We only require that `patch_c` is
        # big enough to hold the real profile in `n_zones` zones (channels <= padded_channels) and
        # not so big it leaves a WHOLE zone empty (channels > (n_zones-1)*patch_c), i.e. the split
        # is contiguous + near-equal (patch_c == ceil(channels / n_zones)).
        assert self.n_zones >= 1, "n_zones must be >= 1"
        assert self.patch_c >= 1, "patch_c must be >= 1"
        assert self.channels <= self.padded_channels, (
            f"channels {self.channels} > padded_channels {self.padded_channels} "
            f"(patch_c={self.patch_c} too small for {self.n_zones} zones)"
        )
        assert self.channels > (self.n_zones - 1) * self.patch_c, (
            f"channels {self.channels} leaves a whole empty zone at patch_c={self.patch_c}, "
            f"n_zones={self.n_zones}; patch_c must be ceil(channels / n_zones)"
        )
        assert self.time_steps % self.patch_t == 0, "time_steps must be divisible by patch_t"


def slowts_patch_for(
    channels: int, *, time_steps: int = 5, n_zones: int = 4
) -> Tuple[int, int]:
    """Pick ``(patch_c, patch_t)`` so a slow-TS signal yields EXACTLY ``n_zones`` tokens.

    DESIGNED per-frame budget (Phase-B frame layout): every slow-TS signal produces
    ``n_zones`` tokens = ``n_zones`` radial-zone position-patches × 1 time-patch. The C profile
    positions are split into ``n_zones`` contiguous, near-equal radial zones of
    ``patch_c = ceil(channels / n_zones)`` positions each; ``patch_t = time_steps`` keeps the
    whole 50 ms window in a single time-patch (T stays 1 patch).

    ``channels`` is NOT required to be divisible by ``n_zones`` (the real counts 44/10/48/69 are
    not): the profile is PADDED up to ``padded_channels = n_zones * patch_c`` and the padded tail
    positions are MASKED as missing (see :meth:`SlowTSCodecPairDataset._fit_window`), so they
    never enter the encoder input or the loss. With ``n_zones = 4``:
        C=44 -> patch_c=11 (44 = 4*11, no padding)         -> 4 tokens
        C=10 -> patch_c=3  (padded_channels 12, pad 2)     -> 4 tokens
        C=48 -> patch_c=12 (48 = 4*12, no padding)         -> 4 tokens
        C=69 -> patch_c=18 (padded_channels 72, pad 3)     -> 4 tokens
    Kept as a helper so the trainer sizes the codec from the loader's real channel count without
    the caller hand-picking patch sizes.
    """
    patch_c = -(-int(channels) // int(n_zones))   # ceil(channels / n_zones)
    return patch_c, time_steps
