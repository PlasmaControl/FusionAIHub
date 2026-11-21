"""Shared configs and tensor-shape contracts for the IGNITE Phase-A codecs.

This module is the single source of truth for the shapes and hyperparameters every
Phase-A component builds against. Components depend only on these shapes (and external
libraries), not on each other's implementations, so they can be built independently.

Tensor shapes (batch first):

    raw window          (B, C, W)          C channels, W = window_samples
    spectrogram window  (B, C, F, T)       F = freq_bins, T = time_frames
    pre-FSQ features    (B, n_tok, d_model)
    FSQ codes (int)     (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
    quantized           (B, n_tok, d_model)
    reconstruction      (B, C, F, T)

    n_tok = (freq_bins // patch_f) * (time_frames // patch_t)

Component interfaces (implemented in sibling modules):

    quantizer.SpectroQuantizer(cfg)
        .quantize(feats: (B,n_tok,d_model)) -> (quant: (B,n_tok,d_model), codes: (B,n_tok,fsq_dim) long)
        .indices_from_codes / .codebook_size  (thin wrapper over vector_quantize_pytorch.FSQ)

    nets.SpectroEncoder(cfg)(x: (B,C,F,T)) -> feats (B,n_tok,d_model)
    nets.SpectroDecoder(cfg)(quant: (B,n_tok,d_model)) -> recon (B,C,F,T)
        (both x-transformers)

    losses.shift_consistency(feats_a, feats_b) -> scalar
        ||enc(x) - enc(shift_d x)||^2 on pre-FSQ features
    losses.recon_objective(recon, target, disc, cfg) -> dict
        adversarial + lambda_pix * pixel anchor

    discriminator.FreqAwarePatchGAN(cfg)(x: (B,C,F,T)) -> list of patch-score maps
        multi-scale, with a frequency positional encoding

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

# STFT and windowing, matching the spectrogram pipeline in data_loader.py.
STFT_FS: float = 500_000.0   # Hz, spectrogram modality target sampling
STFT_N_FFT: int = 1024       # Hann window length; 0.512 ms per frame at hop 256
STFT_HOP: int = 256          # STFT hop in samples
CHUNK_S: float = 0.05        # 50 ms, one world-model frame

# Fast-TS (filterscopes) raw sampling, matching TokamakH5Dataset.SIGNAL_CONFIGS: 10 kHz,
# 8 channels, no STFT, standardized. A 50 ms window is therefore 500 samples per
# channel.
FASTTS_FS: float = 10_000.0  # Hz, filterscopes target sampling (SignalConfig.target_fs)

# Slow-TS (Thomson, CER, MSE) signals all load at 100 Hz, so one 50 ms frame is 5
# samples and a window is (C_positions, T=5).
SLOWTS_FS: float = 100.0     # Hz, slow-TS target sampling


@dataclass
class SpectroCodecConfig:
    """Spectrogram codec (Phase A). Defaults match the production STFT grid."""

    # Data and shape
    channels: int = 1
    freq_bins: int = 512          # cropped from n_fft//2+1 = 513
    # Band-power input. 0 uses the raw STFT bins. A value > 0 mean-pools the freq_bins
    # bins into that many equal bands, as the last step of data.log_power_stft (after
    # the crop and the per-frequency log z-score). Pooling shrinks the reconstruction
    # target, so losses and calibration numbers at different pools are not comparable.
    # Choose patch_f per pool to keep n_tok at 192.
    band_pool: int = 0
    time_frames: int = 96         # ~98 STFT frames per 50 ms, cropped to k * patch_t
    # Phase-B budget: 192 tokens per spectrogram modality. patch_f=16 gives 32 frequency
    # patches (~8 kHz each, fine enough to resolve coherent modes); patch_t=16 gives 6
    # time patches.
    patch_f: int = 16             # 32 frequency patches
    patch_t: int = 16             # 6 time patches, so n_tok = 192
    # Channel-factorized tokens. With channel_groups=G each token's patch spans
    # channels/G channels (a group axis is added to the token grid, group-outer), so
    # n_tok = G * n_freq_patch * n_time_patch and one code no longer has to describe all
    # channels at once (useful for ece's 40 channels). 1 = all-channel tokens, and no
    # group position embedding is created. channels % channel_groups must be 0.
    channel_groups: int = 1

    # Per-codec STFT geometry used to build the input. The defaults equal the module
    # globals, and data.log_power_stft reads these with getattr(cfg, ..., GLOBAL), so
    # old pickled configs without the fields behave the same. At n_fft 1024 / hop 256
    # the STFT is 4x oversampled in time; n_fft 512 at the same hop is the 50%-overlap
    # Hann setup (freq_bins 256, 977 Hz bins, same 0-250 kHz band).
    stft_n_fft: int = STFT_N_FFT      # 1024; 512 for the Spectral-Codec layout
    stft_hop: int = STFT_HOP          # 256

    # FSQ bottleneck (vector-quantize-pytorch). [8, 5, 5, 5] = 1000 codes, the size the
    # FSQ paper recommends. The earlier [8, 8, 8, 8, 8] = 32768 always left one
    # dimension unused.
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = prod = 1000

    # Raw input standardization, for modalities whose raw signal is not O(1). co2's raw
    # is ~1e13, so its log-power (~24) is clipped flat by _LOG_CEIL=20. When on,
    # log_power_stft z-scores the raw signal per channel before the STFT, which shifts
    # log-power into the unclipped [-10, 20] band. raw_mean/raw_std are set at runtime
    # from preprocessing_stats[modality]['raw'] by
    # train_codec.apply_spectro_standardization.
    input_standardize: bool = False
    raw_mean: Optional[List[float]] = None   # (C,) per-channel raw mean
    raw_std: Optional[List[float]] = None     # (C,) per-channel raw std

    # Per-frequency log-power z-score (see data.log_power_stft and train_codec
    # --logpow_stats_path). For thin spectrograms (co2) the log-power window is mostly a
    # near-constant plate with signal in a few bins, so plain reconstruction MAE is
    # minimized by a constant and the FSQ collapses to one code. Subtracting the
    # per-(channel, freq) mean and dividing by the per-freq std (clamped to a floor so
    # noise bins are not blown up) makes the informative bins O(1). The stats live in
    # the raw, clipped log_power_stft space, so this replaces input_standardize (which
    # is turned off when this is on). Shapes are (C, F) nested lists so they pickle.
    logpow_standardize: bool = False
    logpow_freq_mean: Optional[list] = None   # (C, F) per-(channel,freq) log-power mean
    logpow_freq_std: Optional[list] = None    # (C, F) per-(channel,freq) log-power std
    logpow_std_floor: float = 0.25            # per-freq std clamp floor (guards flat/noise bins)
    # Per-window instance z-score of the log-power input (per window and channel).
    # Strips co2's large DC offset (window mean ~-9.9), which saturates the FSQ tanh
    # bound. Applied on top of the raw-std and per-freq paths.
    input_instance_norm: bool = False
    # Quantize the instance-norm statistics so a small realization shift (<= 2 ms)
    # rarely changes the applied normalization: sd on a log2 grid with step q, mu on a
    # grid of q * quantized sd. Plain instance norm made codes inherit window-statistic
    # jitter. 0.0 = plain instance norm.
    instance_norm_quantize: float = 0.0

    # Transformer (x-transformers)
    d_model: int = 256
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8

    # Decoder family. "linear" is the transformer plus a per-token nn.Linear `to_pixels`
    # unpatchify. "conv" is a HiFi-GAN/VQGAN-style transposed-conv decoder
    # (nets.SpectroConvDecoder) that places each token on the (n_freq_patch,
    # n_time_patch) grid and upsamples with ConvTranspose2d and residual conv blocks. A
    # linear map per token can only produce smooth patches, which capped reconstruction
    # correlation of broadband bes/mhr/ece at ~0.5.
    decoder: str = "linear"           # {"linear", "conv"}
    conv_dec_base_ch: int = 128       # SpectroConvDecoder proj_in channel width (halved on upsample)
    conv_dec_res_blocks: int = 2      # residual conv blocks per upsample stage
    # Channel floor for the conv decoder's halving schedule. base_ch 1024 with a floor
    # of 128 reproduces the decoder-heavy shape of HiFi-GAN V1 (decoder ~5x the
    # encoder).
    conv_dec_min_ch: int = 64

    # Decoder conv refinement head, against the patch-lattice (checkerboard) artifact.
    # `to_pixels` maps every token through one shared linear basis, so the same texture
    # is tiled into all 192 patches and much of the reconstruction's 2-D spectral energy
    # lands on the patch lattice (gate.patch_lattice_metrics). refine_depth > 0 appends
    # a residual stride-1 kernel-3 conv stack over the assembled (F, T) spectrogram,
    # with the last conv zero-initialized so the head starts as the identity. It gives
    # the decoder cross-patch context at full resolution. The same head fixed the video
    # decoder's seams. It is not a ConvTranspose(kernel=stride) upsampler, which is
    # known to cause checkerboards.
    #   refine_dilated: dilations 1, 2, 4, 8, ... so depth convs reach a receptive field
    #     of 2^(depth+1)-1 (depth 4 gives 31 >= patch_f). False = all dilation 1
    #     (receptive field 2*depth+1).
    # 0 = off: nothing is constructed, and old pickled configs without these fields load
    # unchanged because every access goes through getattr.
    refine_depth: int = 0
    refine_hidden: int = 64
    refine_dilated: bool = False
    # Decoder noise injection (StyleGAN2 "noise inputs"): a learned per-channel scale
    # times per-pixel Gaussian noise, applied after the refinement head so the head
    # cannot filter it out. The adversarial and feature-matching terms reward
    # high-frequency energy, and a deterministic decoder can only get it from its tiled
    # basis; the noise gives it an aperiodic source. The scale is zero-initialized, so a
    # fresh decoder matches the noiseless one, but its gradient is non-zero. In a
    # 2000-step decoder-only fine-tune on mhr the scale never grew and every metric
    # matched the control, probably because the discriminator separates white noise from
    # real texture easily. Do not assume it helps. Phase B tokenizes with the encoder
    # only (train_dynamics._load_codec), so a stochastic decoder does not change the
    # world model's codes; it matters only when rendering (eval_dynamics.decode).
    decoder_noise: bool = False

    # Invariance (consistency loss on a raw d-shift + re-STFT pair)
    consistency_delta_ms: Tuple[float, float] = (0.1, 2.0)  # sub-window shift range (< 16 ms codec patch)
    consistency_weight: float = 1.0
    amp_jitter: float = 0.03      # +-3% multiplicative nuisance; 0 disables
    noise_floor: float = 0.0      # additive log-power noise std; 0 disables

    # Reconstruction objective
    adversarial_weight: float = 1.0
    pixel_anchor_weight: float = 0.05   # lambda_pix; raise only while stability >= 0.80
    # Multi-resolution and frequency-gradient reconstruction losses (audio-codec style),
    # for turbulent modalities that plain pixel L1 reconstructs as a smooth envelope.
    # Both are folded into recon_ref, so the adaptive adversarial weight balances them.
    multiscale_recon_weight: float = 0.0
    # Average-pool kernel sizes the multi-resolution L1 is scored at. With the default
    # (2, 4) every term is L1 on a low-passed copy and full resolution is never scored,
    # so a large weight rewards matching a blurred spectrogram. Include 1 to make it a
    # real multi-resolution pyramid (arXiv 2406.05298 compares full-detail magnitudes at
    # every resolution).
    multiscale_recon_scales: Tuple[int, ...] = (2, 4)
    freq_grad_weight: float = 0.0
    # MS-SSIM reconstruction term (losses.ms_ssim_loss, 1 - MS-SSIM), the differentiable
    # twin of the gate.ms_ssim metric. SSIM's contrast factor
    # 2*sd_x*sd_y/(sd_x^2+sd_y^2) drops where the reconstruction is flatter than the
    # target, so it penalizes blur, which no L-p term does (the conditional mean
    # minimizes L-p). 0.0 = off; the term is not evaluated.
    ms_ssim_weight: float = 0.0
    ms_ssim_win: int = 7
    ms_ssim_scales: Tuple[int, ...] = (1, 2, 4)
    # Discriminator family. "patch" = FreqAwarePatchGAN. "multiscale" =
    # discriminator.MultiScaleSpectroGAN, which scores the whole spectrogram at 1x/2x/4x
    # with one number per scale. A patch discriminator can be satisfied by one fixed
    # patch texture, which is the lattice artifact; a global multi-resolution critic
    # cannot (arXiv 2406.05298 section 4 pairs multi-period and multi-scale STFT
    # critics).
    discriminator: str = "patch"        # {"patch", "multiscale"}
    # Discriminator feature-matching weight (HiFi-GAN/MelGAN perceptual term, standing
    # in for Genie's VGG loss, which does not transfer to spectrograms). It is folded
    # into the reconstruction reference with the pixel anchor, so a larger value raises
    # the adaptive weight `lam` and with it the adversarial term. It matches
    # discriminator features, not raw STFT pixels.
    fm_weight: float = 1.0

    # VQGAN/MagViT adaptive adversarial weight ("Taming Transformers" §3.3). The hinge
    # GAN term can oscillate against the entropy and shift-consistency terms. When on,
    # each generator step scales the adversarial term by
    # lam = ||grad ref|| / (||grad adv|| + 1e-4), the ratio of gradient norms of the
    # non-adversarial total and the adversarial term at the decoder's last layer.
    adaptive_adv_weight: bool = True    # auto-scale the adversarial term (VQGAN adaptive weight)
    # Upper clamp on lam (the lower clamp is 0). Stable training keeps lam at ~0.01-1. A
    # clamp of 1e4 let a collapsing codec's lam reach ~729; the adversarial term then
    # diverged, one DDP rank desynced and the NCCL watchdog killed the job (exit 143).
    adaptive_adv_clamp: float = 50.0
    adv_warmup_steps: int = 0           # steps with the adversarial term forced off

    # Optimizer and schedule (NVIDIA Spectral Codec, arXiv 2406.05298 section 4). Stored
    # on the config so a checkpoint records its recipe.
    #   adam_beta1/2        torch.optim.Adam defaults (0.9, 0.999). Paper: 0.8 / 0.99.
    #   lr_decay_gamma      lr = lr0 * gamma ** (global_step / lr_decay_every). 1.0 =
    #                       constant LR. Paper: 0.998 per 1000 steps.
    #   disc_update_every   discriminator step once every N generator steps. Paper: 2.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_decay_gamma: float = 1.0
    lr_decay_every: int = 1000
    disc_update_every: int = 1

    # Anti-collapse (codebook-utilization) entropy regularizer, Genie style. The
    # shift-consistency loss has a trivial minimum at a constant encoder (one code), and
    # this term pushes toward diverse code use:
    # entropy_loss = per_sample_entropy_mean - diversity_weight * entropy_of_batch_mean,
    # with the batch mean taken over all DDP ranks. entropy_weight 1.0 gave healthy
    # diversity (min_dim_entropy ~0.65); 0.1 collapsed (one FSQ dimension unused).
    entropy_weight: float = 1.0     # multiplies entropy_loss into the generator total
    diversity_weight: float = 1.0   # weight on the batch-diversity (spread) reward
    # Joint-code diversity, against rank-1 FSQ collapse. diversity_weight rewards each
    # FSQ dimension's marginal spread separately. A rank-1 encoder satisfies that and
    # still uses almost none of the codebook: the old mhr codec (levels [8,8,8,8,8]) had
    # min_dim_entropy 0.921 (93% of ln 8) with 42 of 32768 joint codes in use, because
    # its features were rank-1 and the five per-dimension levels were almost perfectly
    # correlated.
    #   joint_entropy_weight  reward on the entropy of the batch-mean joint soft-code
    #                         distribution over all prod(fsq_levels) codes (MagViT-2/LFQ
    #                         codebook entropy), the quantity frac_codes_used measures.
    #                         Maximum ln(codebook_size), 6.908 nats at 1000 codes.
    #   decorrelation_weight  penalty on the mean squared off-diagonal correlation of
    #                         the per-dimension continuous pre-quantization levels.
    # Both are applied inside quantizer.entropy_loss (read with getattr, so the other
    # codec configs need not define them), whose result is multiplied by entropy_weight;
    # the effective weights are entropy_weight * joint_entropy_weight and
    # entropy_weight * decorrelation_weight.
    joint_entropy_weight: float = 0.0
    decorrelation_weight: float = 0.0
    # Linear ramp of the joint-entropy weight over the first N generator steps (0 =
    # constant from step 0). The encoder starts saturated on real spectrograms (one
    # distinct code at step 0), and a large constant joint weight keeps it there
    # (5.0 stayed at one code through step 2000, while 1.0 and 2.0 recovered). The ramp
    # lets reconstruction and the marginal term desaturate the projection first.
    joint_entropy_ramp_steps: int = 0

    # Missing-data (dead channel) exclusion. Without it the spectro dataset
    # (train_codec.CodecPairDataset._build_pair) drops the loader's nan_mask and only
    # rejects a window on its padded tail, non-finite values, or raw.std() < min_std
    # over all channels at once. A window with some all-zero channels and one live
    # channel passes, and its dead channels are reconstructed, shown to the
    # discriminator as real, feature-matched and counted in the entropy.
    #   mask_missing   build a per-(channel, STFT frame) validity mask
    #                  (data.spectro_frame_mask: NaN-projected like
    #                  data_loader._raw_to_frame_mask, plus all-zero frames, which is
    #                  how the loader stores an absent channel) and honour it in every
    #                  loss term, the discriminator step and the gate. False = every
    #                  consumer gets None.
    #   require_live_channels
    #                  redraw windows in which any channel is dead, so the codec only
    #                  sees fully populated windows. False = any live channel is enough.
    #   presence_filter
    #                  drop shots with no live channel for this modality, using the
    #                  precomputed foundation_model_meta/spectro_channel_liveness.pt (one
    #                  torch.load, no HDF5 scan).
    mask_missing: bool = False
    require_live_channels: bool = False
    presence_filter: bool = False

    # FSQ's own anti-collapse arguments (vector_quantize_pytorch.FSQ constructor). Both
    # default to the library defaults.
    #   fsq_noise_dropout
    #       With this probability per element, FSQ.maybe_apply_noise adds a uniform
    #       +/-0.5 offset to the bounded code and clamps to [-1, 1], in training only.
    #       It is applied after codes_to_indices, so the emitted indices stay clean and
    #       only the decoder's continuous input is jittered (a decoder regularizer).
    #       The offset is in the normalized [-1, 1] space, where bins are 2/(L-1)
    #       apart: for [8,5,5] that is +/-1.75 bins on the 8-level dimension and +/-1.0
    #       on the 5-level ones, much more than half a bin.
    #   fsq_preserve_symmetry
    #       Uses `symmetry_preserving_bound` (arXiv 2411.19842 section 3.2) and the
    #       matching level<->code mapping in `_scale_and_shift`. The library requires it
    #       whenever noise dropout is on. It changes the codes even at noise_dropout 0,
    #       so a noise-dropout arm needs a preserve_symmetry-only control.
    # FSQ's `orthogonal_rotation` is not exposed: it requires symmetric levels, and the
    # production levels ([8,5,5,5] / [8,5,5]) are not.
    fsq_noise_dropout: float = 0.0
    fsq_preserve_symmetry: bool = False


    # Activity-stratified sampling, so near-constant windows do not dominate. co2 is
    # floored about half the time (median window std ~0.02 vs ~0.85 for ece/bes/mhr),
    # and a codec trained on such batches collapses to the floor value. With probability
    # active_bias, a window whose activity (log-power std) is below min_activity is
    # redrawn from nearby chunks; if no active one is found the original is kept, so
    # quiet windows still get a code. Both 0 = off. The trainer turns them on for co2
    # only (train_codec._activity_overrides).
    min_activity: float = 0.0       # activity threshold (log-power std); 0 = off
    active_bias: float = 0.0        # P(redraw a quiet window); 0 = off

    # Peak-weighted reconstruction (losses.peak_l1_loss), a surrogate for peak_f1: L1
    # reweighted by each frequency bin's prominence over its column mean, so spectral
    # peaks carry most of the loss. 0.0 = off; the term is not evaluated. On held-out
    # co2, trained codecs score peak_f1 ~0.61-0.62, below a free patch-mean code (0.65)
    # and far below a 5-frame time-smoothed oracle (0.99), and sharpness levers barely
    # move it. Nothing else in the objective pays for peak placement: a mode line covers
    # 1-2 of a patch's 16 frequency bins, so placing it moves plain L1 little, and the
    # adversarial term rewards the right texture anywhere in the patch.
    peak_weight: float = 0.0

    # Time-smoothed reconstruction target. K > 1 replaces the target with a K-frame
    # boxcar average along time. The encoder still sees the raw window, and reported
    # metrics are still against the raw spectrogram; only what the decoder is asked to
    # produce changes. 0 or 1 = off (losses.time_smooth returns the same object). Most
    # frame-to-frame variation is realization speckle no code can carry: lag-1
    # autocorrelation along STFT frames is 0.61-0.68 (co2), 0.42-0.48 (mirnov) and
    # 0.29-0.37 (ece). Reconstructing the raw window asks for a mostly unpredictable
    # target, and the L-p optimum is a low-amplitude blur (std(recon)/std(GT)
    # 0.15-0.32). With a smoothed target the optimum is the coherent structure at full
    # amplitude. The 5-frame smoothed ground truth, scored against raw ground truth,
    # beats every raw-target codec on hf_ratio and nRMSE (co2 nRMSE 0.38 vs 0.68, mirnov
    # 0.63 vs
    # 1.11). K should match the coherence length; 5 is where that was measured.
    # n_tok, the vocabulary and FRAME_LAYOUT are unchanged.
    target_time_smooth: int = 0

    # Envelope/shape split ("gain-shape", the spectrogram port of the fast-TS version).
    # It targets amplitude collapse: apart from ms_ssim every reconstruction term is an
    # L-p distance, minimized by the conditional mean, so a decoder that emits each
    # window's time average scores well with no temporal structure (e.g. mirnov
    # std(recon)/std(GT) 0.15-0.28). The world model needs mode evolution, so that
    # failure matters regardless of nRMSE.
    #
    # With gain_shape on, the first gain_tokens tokens carry per-(channel, frequency)
    # time statistics and the other n_tok - gain_tokens carry a normalized shape:
    #
    #     level (B,C,F) = x.mean(-1)                 static envelope
    #     sigma (B,C,F) = (x - level).std(-1)        temporal amplitude per freq bin
    #     shape (B,C,F,T) = (x - level) / sigma      zero mean, unit std along T
    #
    # The decoder rebuilds recon = level_hat + sigma_hat * shape_hat, with shape_hat
    # mean-removed and std-normalized along T inside the decoder. Two properties hold by
    # construction (tested in tests/test_spectro_gain_shape.py):
    #
    #   1. recon.mean(-1) is level_hat, so shape tokens spend no bits on the DC level.
    #   2. recon.std(-1) is sigma_hat, so the amplitude is transmitted by a code and the
    #      decoder cannot shrink toward the conditional mean. gain_scale False keeps
    #      only property 1, the control that isolates the effect of the std
    #      normalization.
    #
    # Do not judge these arms by closeness to the time-averaged spectrum (~tmean): it
    # has no temporal structure, so approaching it means deleting the modes. Judge on
    # hf_ratio against ground truth (with patch_lattice_ratio) and on visible mode
    # tracks where the ground truth has coherent structure
    # (analysis/spectro_final_fig.py --mode structure). An arm that improves nRMSE while
    # the figure stays a smooth plate is worse.
    #
    # The 192-token budget, FRAME_LAYOUT and the Phase-B vocabulary are unchanged; this
    # reallocates tokens. False = off: no parameter or split is created, and old pickled
    # configs load unchanged (fields are read with getattr).
    gain_shape: bool = False
    # Tokens reserved for the (level, sigma) envelope. Must divide eff_freq_bins: the
    # envelope is patchified along frequency, so each gain token holds one contiguous
    # frequency band's statistics for all channels. gain_tokens == n_freq_patch (32 at
    # the production geometry) puts it on the main path's frequency lattice; fewer
    # coarsens it and leaves more shape tokens. Must be < n_tok.
    gain_tokens: int = 32
    # Transmit sigma as well as level (property 2 above). False = level only.
    gain_scale: bool = True
    # Weight on direct L1 supervision of the decoded (level, log1p sigma) against their
    # analytic targets. The reconstruction term sees both too, but mixed with the shape
    # error; the gain head sets the whole envelope and amplitude, so it gets its own
    # target.
    gain_weight: float = 1.0
    # Hidden width of the per-patch gain MLPs (encoder input head, decoder output head).
    gain_hidden: int = 256

    # Amplitude-ratio term: penalizes |log std(recon) - log std(target)| per (window,
    # channel), the quantity analysis.spectro_final_fig.masked_std_ratio reports (ideal
    # 1.0). 0.0 = off; the loss is not computed. ece codecs sat at a std ratio of
    # ~0.12-0.32 across patch sizes, learning rates and EMA, and no other term targets
    # an amplitude ratio (nRMSE even improves under amplitude collapse).
    std_weight: float = 0.0

    # Oracle-gate acceptance thresholds
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    # Collapse detection (§4.4): a codec is flagged `collapsed` if it uses too little of
    # the codebook or has too low a per-dimension code entropy. The flag is
    # informational for best-checkpoint selection; see gate_recon_floor and
    # spike.gate_score.
    gate_min_utilization: float = 0.02   # min fraction of the codebook used to pass
    gate_min_code_entropy: float = 0.3   # min per-dim normalized code entropy to pass
    # Hard floor on distinct codes: below it the gate score is -inf, so best.pt never
    # follows a collapsing codec (which can keep reconstructing through the decoder's
    # position embedding, so the reconstruction floor alone misses it).
    gate_hard_min_codes: int = 8
    # Reconstruction floor for best-checkpoint selection: the gate score is -inf when
    # the decode envelope_corr is NaN or below this. Utilization enters the score as a
    # soft reward instead, so a codec that reconstructs well with one unused FSQ
    # dimension is not rejected. envelope_corr is computed on the time-collapsed power
    # envelope (spec.mean(axis=-1)), so a codec can pass with no temporal structure. On
    # held-out mhr, codecs with envelope_corr 0.67-0.70 had spec_corr2d 0.27-0.32 and
    # spec_nrmse
    # 1.01-1.25, worse than the window's own mean. The full-spectrogram keys
    # (gate.full_spectro_metrics) are reported in every gate dict for that reason, but
    # the selection score does not use them.
    gate_recon_floor: float = 0.2

    @property
    def eff_freq_bins(self) -> int:
        """Frequency rows the model sees: band_pool when pooling, else freq_bins.

        freq_bins keeps its meaning (the STFT crop width and the shape of the per-freq
        log-z stats); only the model-facing geometry follows the pooled width.
        """
        return int(self.band_pool) if int(self.band_pool or 0) > 0 else int(self.freq_bins)

    @property
    def n_freq_patch(self) -> int:
        return self.eff_freq_bins // self.patch_f

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

    # Envelope/shape split (see gain_shape)
    @property
    def n_gain_tok(self) -> int:
        """Tokens carrying the (level, sigma) envelope code; 0 unless gain_shape is on."""
        if not getattr(self, "gain_shape", False):
            return 0
        return int(getattr(self, "gain_tokens", 32))

    @property
    def n_shape_tok(self) -> int:
        """Tokens carrying the normalized shape (n_tok when gain_shape is off)."""
        return self.n_tok - self.n_gain_tok

    @property
    def uses_gain_scale(self) -> bool:
        """gain_scale, forced off when there is no shape path to normalize."""
        return bool(getattr(self, "gain_scale", True)) and self.n_shape_tok > 0

    @property
    def gain_patch_f(self) -> int:
        """Frequency bins per gain token (eff_freq_bins / n_gain_tok)."""
        return self.eff_freq_bins // max(1, self.n_gain_tok)

    @property
    def gain_values_per_tok(self) -> int:
        """Values per gain token: channels x gain_patch_f x (2 if sigma else 1)."""
        if self.n_gain_tok == 0:
            return 0
        return self.channels * self.gain_patch_f * (2 if self.uses_gain_scale else 1)

    @property
    def gain_bits(self) -> float:
        """Bits the gain path owns (n_gain_tok * log2(codebook))."""
        return self.n_gain_tok * log2(self.codebook_size)

    def __post_init__(self) -> None:
        assert self.eff_freq_bins % self.patch_f == 0, (
            f"eff_freq_bins ({self.eff_freq_bins}) must be divisible by patch_f "
            f"({self.patch_f}); band_pool={self.band_pool}")
        if self.band_pool:
            assert self.freq_bins % self.band_pool == 0, (
                f"band_pool ({self.band_pool}) must divide freq_bins ({self.freq_bins})")
        assert self.time_frames % self.patch_t == 0, "time_frames must be divisible by patch_t"
        if getattr(self, "gain_shape", False):
            g = int(self.gain_tokens)
            assert 1 <= g < self.n_tok, (
                f"gain_tokens ({g}) must be in [1, n_tok={self.n_tok}); a codec with no shape "
                f"tokens left is an envelope coder, not a spectrogram codec"
            )
            assert self.eff_freq_bins % g == 0, (
                f"gain_tokens ({g}) must divide eff_freq_bins ({self.eff_freq_bins}): each gain token "
                f"owns one contiguous frequency band's (level, sigma) for all channels"
            )
            assert self.gain_hidden >= 1, "gain_hidden must be >= 1"
            assert int(getattr(self, "channel_groups", 1)) == 1, (
                "gain_shape does not support channel_groups > 1 (the shape token axis is "
                "mixed as one block; a group axis would have to be mixed per group)"
            )


# Video codec (tangtv, Phase A)
#
# docs/IGNITE_DESIGN.md §4.3: the video codec is the closest to Genie (smooth frames),
# with a generative decoder and no strong pixel MSE, and separate codecs for the upper
# and lower divertor. It reuses the spectrogram codec's FSQ bottleneck, adaptive
# adversarial weight, entropy regularizer, gate thresholds and feature matching, and
# drops the STFT machinery:
#   * No shift-consistency term. That nuisance (a 0.5 ms shift scrambles ~74% of
#     spectrogram codes through STFT phase) does not exist for a camera, so the codec
#     trains on single windows with pixel anchor, feature matching, adversarial and
#     entropy losses.
#   * The target is the raw frames (C, T, H, W), patchified over space and time.
#
# Divertor channels (from data_loader.MovieConfig):
#   tangtv_lower = raw camera channels [0, 2]  (LODIV PAR-int + LODIV PERP)
#   tangtv_upper = raw camera channels [4, 6]  (UPDIV0 PERP + UPDIV0 PAR)
# ch1/ch3/ch5 are NaN (off) in all shots, so each divertor codec sees C=2 channels.


# tangtv movie geometry (matches data_loader.MovieConfig for tangtv_lower / tangtv_upper).
VIDEO_TARGET_FPS: int = 100      # target frame rate after resample
VIDEO_HEIGHT: int = 120          # output frame height
VIDEO_WIDTH: int = 360           # output frame width


@dataclass
class VideoCodecConfig:
    """tangtv video codec (Phase A), one per divertor.

    Production uses two instances, ``divertor="lower"`` and ``"upper"``, each with its
    own encoder, decoder and discriminator (§4.3). ``divertor`` only selects the channel
    set and the logging name; the tensor contract is the same.

    Tensor shapes (batch first):

        frame window    (B, C, T, H, W)   C channels per divertor, T frames, H×W pixels
        pre-FSQ feats   (B, n_tok, d_model)
        FSQ codes (int) (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
        quantized       (B, n_tok, d_model)
        reconstruction  (B, C, T, H, W)

        n_tok = (T // patch_t) * (H // patch_h) * (W // patch_w)
    """

    # Divertor selector (logging only; the loader maps it to channels).
    divertor: str = "lower"  # "lower" (cams 0,2) or "upper" (cams 4,6)

    # Data and shape
    channels: int = 2              # channels per divertor (2 live cameras each)
    frames: int = 5                # T; a 50 ms window at 100 fps = 5 frames
    height: int = VIDEO_HEIGHT     # 120
    width: int = VIDEO_WIDTH       # 360
    patch_t: int = 5               # 1 time patch (the whole window)
    patch_h: int = 20              # 6 height patches
    patch_w: int = 20              # 18 width patches, so n_tok = 1*6*18 = 108

    # FSQ bottleneck: [8, 5, 5, 5] = 1000 codes, as for the spectrogram codec. Frames
    # are smoother than spectrograms, so 1000 codes is ample.
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = 1000

    # Transformer (x-transformers), bidirectional over the space-time token set.
    d_model: int = 256
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8

    # Reconstruction objective: generative decoder with a low pixel anchor.
    adversarial_weight: float = 1.0
    # Low pixel anchor (§4.3, "no strong pixel MSE"). The pixel term only stabilizes
    # optimization; sharp frames come from the adversarial and feature-matching terms.
    pixel_anchor_weight: float = 0.05
    # Discriminator feature-matching weight, a perceptual term on the discriminator's
    # features that needs no external VGG weights. Folded into recon_ref, so it raises
    # the adaptive adversarial weight.
    fm_weight: float = 1.0

    # VQGAN/MagViT adaptive adversarial weight, as in SpectroCodecConfig (gradient-norm
    # ratio at decoder.last_layer).
    adaptive_adv_weight: bool = True
    # Upper clamp on lam; see SpectroCodecConfig.adaptive_adv_clamp.
    adaptive_adv_clamp: float = 50.0
    adv_warmup_steps: int = 0

    # Anti-collapse entropy regularizer, shared with the spectrogram quantizer; see
    # SpectroCodecConfig.entropy_weight.
    entropy_weight: float = 1.0
    diversity_weight: float = 1.0
    # Joint-code diversity and decorrelation; see SpectroCodecConfig. Read with getattr
    # inside quantizer.entropy_loss; 0.0 = off. The joint term builds an
    # (N, codebook_size) tensor and is refused above 4096 codes, so it works at
    # [8,5,5,5] (1000) but not at [8,8,8,5,5,5] (64000).
    joint_entropy_weight: float = 0.0
    decorrelation_weight: float = 0.0
    joint_entropy_ramp_steps: int = 0

    # Missing-data (dead camera) exclusion. Over all 8753 shots
    # (foundation_model_meta/video_channel_liveness.pt):
    #     tangtv_lower (ch 0, 2): both cameras live in 30.2% of shots, one in 18.5%,
    #                             neither in 51.3%.
    #     tangtv_upper (ch 4, 6): both 20.8%, one 10.6%, neither 68.6%.
    # An off camera is an all-NaN slab that the loader zero-fills, so without filtering
    # only 39% (lower) / 26% (upper) of the training tensor is real camera data. The
    # zeros reach the discriminator as real frames (rewarding a mean-collapsed
    # generator), the feature-matching term and the FSQ entropy statistic.
    #   mask_missing          emit a per-(C, T) validity mask from the loader's
    #                         `channel_valid` and honour it in every loss term and the
    #                         gate. False = an all-ones mask.
    #   require_live_channels redraw clips in which any channel is dead. False = any
    #                         live channel is enough.
    #   presence_filter       drop shots with no live camera for this divertor, using
    #                         the precomputed liveness cache (no HDF5 scan).
    mask_missing: bool = False
    require_live_channels: bool = False
    presence_filter: bool = False

    # Optimizer and schedule: same fields and defaults as SpectroCodecConfig, so the
    # video trainer can run the paper recipe (betas 0.8/0.99, gamma 0.998 per 1k steps,
    # D every 2).
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_decay_gamma: float = 1.0
    lr_decay_every: int = 1000
    disc_update_every: int = 1

    # Decoder conv refinement head, against patch seams. The linear per-token unpatchify
    # renders each 20x20(x5) patch independently, which leaves a visible 6x18 grid once
    # the adversarial and feature-matching terms are off. refine_depth > 0 appends a
    # small residual per-frame stride-1 conv stack (kernel 3, refine_hidden channels,
    # last conv zero-initialized so it starts as the identity) after the unpatchify, so
    # the loss can blend across patch borders. It is not a ConvTranspose(kernel=stride)
    # upsampler, which is known to cause checkerboards. 0 = off; old checkpoints load
    # unchanged.
    refine_depth: int = 0
    refine_hidden: int = 64
    # Dilations 1, 2, 4, 8, ... so depth convs reach a receptive field of 2^(depth+1)-1
    # instead of 2*depth+1. The receptive field must exceed the patch size (20 here): an
    # undilated head needs depth 10, a dilated depth-4 head reaches 31.
    refine_dilated: bool = False

    # Activity-stratified sampling, as in SpectroCodecConfig (activity = the clip's
    # frame std). tangtv clips are not dominated by quiet windows (all have std >=
    # ~3.8); video collapse came from adversarial instability, so the trainer leaves
    # active_bias at 0 for tangtv and uses an adversarial warmup and a lower
    # adversarial_weight instead (train_codec._activity_overrides). Both 0 = off.
    min_activity: float = 0.0       # activity threshold (clip frame std); 0 = off
    active_bias: float = 0.0        # P(redraw a quiet clip); 0 = off

    # Oracle-gate acceptance thresholds (§4.4).
    gate_stability: float = 0.80        # frame-to-frame code stickiness proxy
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
    gate_hard_min_codes: int = 8         # hard floor on distinct codes (as Spectro)
    gate_min_code_entropy: float = 0.3
    # Reconstruction floor for best-checkpoint selection: the gate score is -inf when
    # the video envelope_corr is NaN or below this. Same as
    # SpectroCodecConfig.gate_recon_floor.
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


# Fast-TS codec (filterscopes, Phase A)
#
# There are two modes, selected by FastTSCodecConfig.target.
#
# "envelope" (the default) encodes an ELM activity envelope: E = 5 RMS statistics (10 ms
# each) per 50 ms window, 5 tokens. It does not reconstruct the waveform or spike
# timing.
#
# "raw" encodes and reconstructs the 10 kHz samples themselves:
#
#     raw window   (C, W)   C = 8 filterscope channels, W = round(50 ms * 10 kHz) = 500,
#                           so 4000 values per world-model frame
#     target       the same (C, W) array (per-channel standardized, nothing pooled)
#
# Raw mode differs from envelope mode in three ways:
#   * No envelope, RMS, pooling or log1p (pool, baseline_win, env_eps, env_bins and
#     patch_e are unused).
#   * No delta-shift consistency term. Consistency makes codes invariant to sub-bin
#     spike timing, and sample-wise reconstruction needs codes that carry it.
#   * Reconstruction dominates (pixel_anchor_weight 1.0) and the adversarial term is
#     off: a GAN invents uncorrelated high-frequency detail, which raises sample-wise
#     nRMSE.
#
# Difficulty, measured out of sample (fit on 8 shots, scored on 4 others). nRMSE is
# RMSE / std(target) per (window, channel) over samples, so the window's own mean scores
# 1.0:
#   * Lag-1 autocorrelation of a de-meaned window is 0.20-0.48, and only 19-33% of the
#     variance lies below 500 Hz: the waveform is broadband and hard to compress.
#   * The envelope resolution (5 bins of 10 ms) as a signed predictor scores 0.963.
#   * Per-channel PCA with free, unquantized coefficients: k=25 0.904, k=64 0.841,
#     k=128 0.753, k=400 0.369.
#   * Gaussian rate-distortion bound (reverse water-filling on the measured PSD) at
#     9.9658 bits/token over 4000 values:
#        5 tok (0.0125 b/value) -> >= 0.870   |   25 tok (0.0623) -> ~0.766
#       50 tok (0.1246)         -> ~0.686     |  128 tok (0.3189) -> >= 0.543
#   * A DCT-lowpass transform coder scores 0.977 at ~26 tokens and needs ~494 tokens for
#     0.775.
# So raw filterscope samples are largely unreconstructable at an affordable token count.
# patch_w sets the rate.
#
# Raw mode treats the (C, W) window as a 1-D signal: a conv stem at the full 10 kHz rate
# (recommended for spikes over loss weighting, which did not work), non-overlapping
# patch_w-sample patches along time with channels as features, FSQ, and a decoder that
# mirrors the path back to (C, W) through a conv head.


# Fast-TS geometry (filterscopes SignalConfig with a 50 ms window at FASTTS_FS).
FASTTS_CHANNELS: int = 8         # channels_to_use=slice(0,8) on the filterscopes SignalConfig
FASTTS_WINDOW: int = round(CHUNK_S * FASTTS_FS)   # 500 raw samples per channel in 50 ms
# Raw-mode default rate: 20 samples (2 ms) per token, so n_tok = 500 // 20 = 25 and
# 25 * log2(1000) = 249.1 bits over 4000 values = 0.0623 bits/value. The world-model
# frame grows from 1017 to 1037 tokens. W must be divisible by patch_w; the divisors of
# 500 give token counts 500, 250, 125, 100, 50, 25, 20, 10, 5, 4, 2, 1.
FASTTS_PATCH_W: int = 20
# Envelope-mode geometry. Envelope mode is the default, so every existing pickled
# FastTSCodecConfig and filterscopes checkpoint loads with a strict load_state_dict.
# That matters beyond fast-TS: train_dynamics._load_codec builds the full codec set one
# family at a time, so a fast-TS load failure breaks every consumer of the full set.
FASTTS_POOL: int = round(0.010 * FASTTS_FS)          # envelope pooling bin = 100 samples = 10 ms
FASTTS_ENV_BINS: int = FASTTS_WINDOW // FASTTS_POOL  # 5 envelope bins (10 ms each)


@dataclass
class FastTSCodecConfig:
    """Fast-TS (filterscopes) codec (Phase A).

    Tensor shapes in raw mode (batch first):

        raw window       (B, C, W)          C channels, W = window_samples raw samples
                                            (the codec input and the reconstruction
                                            target)
        pre-FSQ feats    (B, n_tok, d_model)
        FSQ codes (int)  (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
        quantized        (B, n_tok, d_model)
        reconstruction   (B, C, W)          reconstructed raw samples

        n_tok = window // patch_w
    """

    # Codec mode. "envelope" (default) is the ELM activity-envelope codec; "raw" is the
    # sample-wise codec (see the section note above). Read everywhere with getattr(cfg,
    # "target", "envelope"), because configs pickled before this field existed must take
    # the envelope path. Build a raw config with fastts_raw_config(), which also
    # switches the objective defaults (the ones below suit envelope mode only).
    target: str = "envelope"

    # Data and shape
    channels: int = FASTTS_CHANNELS       # 8 filterscope channels
    # Envelope mode
    env_bins: int = FASTTS_ENV_BINS       # E = 5 (10 ms per bin over a 50 ms window)
    patch_e: int = 1                      # 5 time patches (10 ms each), so n_tok = 5
    pool: int = FASTTS_POOL               # RMS pooling bin (samples) = 100 = 10 ms
    baseline_win: int = 20                # moving-mean baseline window (samples) = 2 ms; 0 = off
    env_eps: float = 1e-3                 # log1p reference: log1p(env / env_eps)
    # Raw mode (target="raw"); unused in envelope mode
    window: int = FASTTS_WINDOW           # W = 500 raw samples (50 ms at 10 kHz)
    patch_w: int = FASTTS_PATCH_W         # samples per token; n_tok = window // patch_w

    # Per-channel raw standardization, x <- (x - mean) / std. Raw filterscope values are
    # ~1e15 with per-channel std ~1e16, and data_loader standardizes them the same way
    # (method "standardize", per-channel stats from preprocessing_stats.pt). The stats
    # are global per channel, not per window, so relative activity levels are kept.
    # Length == channels. None = no standardization (synthetic tests, callers with O(1)
    # data); the trainer injects the real stats
    # (fastts_train.load_fastts_channel_stats). The gate's nRMSE divides by each
    # (window, channel)'s own std, so this affine map changes optimization but not the
    # score.
    channel_mean: Optional[Sequence[float]] = None
    channel_std: Optional[Sequence[float]] = None

    # FSQ bottleneck: [8, 5, 5, 5] = 1000 codes = 9.9658 bits/token. Four dimensions
    # avoid the unused fifth dimension seen at [8,8,8,8,8], and 1000 codes keep
    # joint_entropy_weight affordable (it needs an (N, codebook_size) tensor).
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = 1000

    # Conv stem before patching, at the full 10 kHz rate, so a spike that straddles a
    # patch boundary is still seen whole: with stem_kernel=15 and stride 1 each output
    # sample mixes +-7 raw samples, so the receptive field overlaps neighbouring
    # patches. stem_layers=0 gives the plain linear-patch path (a control).
    stem_channels: int = 64
    stem_kernel: int = 15
    stem_layers: int = 2

    # Transformer (x-transformers), bidirectional over the sample patches.
    d_model: int = 256
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8

    # Objective. Raw mode aims at low sample-wise nRMSE, forecastable codes and >= 90%
    # codebook use, so reconstruction leads; the envelope codec ran pixel anchor 0.05
    # against adversarial 1.0. The NVIDIA Spectral Codec (arXiv 2406.05298) weights
    # reconstruction 21:1 over the GAN; raw mode uses reconstruction only.
    pixel_anchor_weight: float = 0.05     # envelope default; fastts_raw_config sets 1.0
    # "nrmse" | "nmse" | "mse" | "l1" | "huber". "nrmse" is the reported metric. Plain
    # "mse" is not a surrogate for it: the within-window std (~0.0073) is ~25x smaller
    # than the RMS window-mean offset, so an MSE optimum spends its capacity on the DC
    # level and emits a flat window, which scores nRMSE exactly 1.0 (a flat line reads
    # mse 5.0e-05). The metric divides by each pair's own std, so the loss must too.
    # "nrmse" is preferred over "nmse" for conditioning: the per-pair ratio spans ~1e6
    # (quietest windows std 0.0009, p90 0.071), so "nmse" is dominated by a few quiet
    # noise windows; the square root compresses that to ~1e3. "mse"/"l1"/"huber" are
    # unnormalized controls that demonstrate the flat-line failure. See
    # FastTSCodec._recon_loss.
    recon_loss: str = "nrmse"
    # Structural (SSIM) term, an amplitude guard. nRMSE's minimizer is the conditional
    # mean, so under uncertainty the codec shrinks toward the local mean: at correlation
    # r the nRMSE-optimal output scale is r. On held-out windows
    # (gate.fastts_structural_references) a 0.4x-shrunk target scores nRMSE 0.60 with
    # std_ratio 0.40 and SSIM contrast 0.69, and a 0.9 ms-smoothed target scores 0.87
    # with contrast 0.54. Elementwise error misses both; SSIM contrast catches them.
    # ssim_weight > 0 adds (1 - contrast*structure) to the reconstruction reference. 0.0
    # = off, so the plain arm is a control. This preserves spikes through local
    # variance, not through per-sample loss weighting (which did not work for this
    # modality).
    ssim_weight: float = 0.0
    # Box window in samples for the 1-D SSIM (17 = 1.7 ms at 10 kHz, about one ELM burst
    # period of ~1.6 ms). Must match gate._FASTTS_SSIM_WIN so loss and metric agree.
    ssim_win: int = 17

    # Adversarial and feature matching, off in raw mode: a GAN synthesizes uncorrelated
    # high-frequency detail, which sample-wise nRMSE punishes. Kept as knobs
    # (--adversarial_weight / --fm_weight) to measure the trade-off.
    adversarial_weight: float = 1.0       # envelope default; fastts_raw_config sets 0.0
    fm_weight: float = 1.0                # envelope default; fastts_raw_config sets 0.0

    # Delta-shift consistency, off in raw mode (invariance to spike timing contradicts
    # sample-wise reconstruction). The range is still used by the gate's stability
    # probe, which reports how much the codes move under a shift.
    consistency_delta_ms: Tuple[float, float] = (0.1, 5.0)
    consistency_weight: float = 1.0       # envelope default; fastts_raw_config sets 0.0

    # VQGAN/MagViT adaptive adversarial weight ("Taming Transformers" 3.3). Has no
    # effect while adversarial_weight == 0.
    adaptive_adv_weight: bool = True
    adaptive_adv_clamp: float = 50.0
    adv_warmup_steps: int = 0

    # Anti-collapse entropy regularizer, shared with SpectroQuantizer.entropy_loss
    # (per-sample entropy - diversity * batch-mean entropy).
    entropy_weight: float = 1.0
    diversity_weight: float = 1.0
    # Joint-code diversity (see SpectroCodecConfig); took mhr from 42/32768 to 1000/1000
    # codes. Affordable at 1000 codes. 0.0 = off.
    joint_entropy_weight: float = 0.0
    joint_entropy_ramp_steps: int = 0

    # Activity-stratified sampling. Activity is the raw window's std in standardized
    # units. On 4 held-out shots the per-(window, channel) std quartiles are
    # 0.0025/0.0073/0.0121, and the top quartile is the most reconstructable (0-1 kHz
    # lowpass oracle nRMSE 0.40 there vs 0.68-0.73 below), so biasing toward active
    # windows favours the part of the signal with structure. Both 0 = off; the trainer
    # may turn them on (train_codec._activity_overrides).
    min_activity: float = 0.0       # activity threshold (raw window std); 0 = off
    active_bias: float = 0.0        # P(redraw a quiet window); 0 = off

    # Gain-shape decomposition, envelope mode only. On 4 held-out shots (483 windows)
    # 96.2% of the envelope variance is the between-window DC level (each (window,
    # channel)'s mean over the 5 bins). An encoder that sends only that level, quantized
    # to the codec's own 49.83-bit budget (5 tokens x log2(1000)), scores nRMSE 0.19,
    # against 0.53 for the plain codec, which shares every token between level and shape
    # and recovers the level only approximately (r 0.87).
    #
    # With gain_shape on the codec codes
    #     x[c, e]  =  level[c]  +  sigma[c] * shape[c, e]
    # with level = mean_e x, sigma = std_e (x - level), and shape zero-mean and unit-std
    # by construction. The first gain_tokens tokens carry (level, log1p sigma) through
    # their own MLP -> FSQ path, which never sees the shape; the rest carry the shape.
    # The decoder mean-removes the shape branch (and std-normalizes it when gain_scale),
    # so the reconstruction's per-(window, channel) mean comes from the gain head alone.
    #
    # The quantized window mean (nRMSE ~0.19) then becomes a floor: with the shape path
    # zeroed the codec is exactly that encoder, and the shape path can only add. Because
    # sigma is sent explicitly, the within-window amplitude is a decoded number rather
    # than a shrunken conditional mean (the plain codec keeps only ~36% of burst height,
    # 8% on channel 0).
    #
    # False = every field below is unused and the module is unchanged
    # (tests/test_fastts_gain_shape.py::test_gain_shape_off_is_bit_identical).
    gain_shape: bool = False
    # Tokens (of the n_env_patch = 5) reserved for the gain code; the rest carry shape.
    # gain_tokens == n_env_patch means level only (no shape path), the learned-VQ
    # version of the rate-matched baseline.
    gain_tokens: int = 1
    # Send the per-(window, channel) AC scale sigma with the level, and normalize the
    # decoded shape to unit std so sigma alone sets the amplitude (textbook gain-shape
    # VQ). False = level only, and the shape amplitude is free to shrink. Forced off
    # when there is no shape path (gain_tokens == n_env_patch).
    gain_scale: bool = True
    # Weight on direct supervision of the gain head, |pred - (level, log1p sigma)|. Part
    # of the reconstruction reference. 0 = off (the gain path learns only through the
    # reconstruction term).
    gain_weight: float = 1.0
    # Hidden width of the gain MLP (encoder and decoder side).
    gain_hidden: int = 256

    # Oracle-gate acceptance thresholds (4.4).
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
    gate_hard_min_codes: int = 8         # hard best-checkpoint floor on distinct codes
    gate_min_code_entropy: float = 0.3
    # Reconstruction floor for best-checkpoint selection. spike.gate_score reads
    # decode["envelope_corr"], which in raw mode is the sample-wise Pearson correlation
    # of the waveform (gate.fastts_decode_fidelity). That is much smaller than an
    # envelope correlation (the out-of-sample linear ceiling at 25 tokens is ~0.37), so
    # 0.2 would disqualify every reasonable raw checkpoint. Raw mode uses 0.02, still
    # above a dead one-code codec (~0.0).
    gate_recon_floor: float = 0.2         # envelope default; fastts_raw_config: 0.02

    @property
    def is_raw(self) -> bool:
        """True for the sample-wise codec. Old pickles have no ``target`` field."""
        return getattr(self, "target", "envelope") == "raw"

    @property
    def n_env_patch(self) -> int:
        """Envelope-mode token count."""
        return self.env_bins // self.patch_e

    @property
    def n_gain_tok(self) -> int:
        """Tokens carrying the gain code (0 unless gain_shape is on, envelope mode)."""
        if self.is_raw or not getattr(self, "gain_shape", False):
            return 0
        return int(getattr(self, "gain_tokens", 1))

    @property
    def n_shape_tok(self) -> int:
        """Tokens carrying the level-normalized shape (n_tok when gain_shape is off)."""
        return self.n_tok - self.n_gain_tok

    @property
    def gain_values(self) -> int:
        """Numbers the gain path transmits: C levels (+ C log1p sigmas when gain_scale)."""
        if self.n_gain_tok == 0:
            return 0
        return self.channels * (2 if self.uses_gain_scale else 1)

    @property
    def uses_gain_scale(self) -> bool:
        """gain_scale, forced off when there is no shape path to normalize."""
        return bool(getattr(self, "gain_scale", True)) and self.n_shape_tok > 0

    @property
    def gain_bits(self) -> float:
        """Bits the gain path owns (n_gain_tok * log2(codebook))."""
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
        """Raw samples in a 50 ms window at FASTTS_FS (``self.window``)."""
        return int(self.window)

    @property
    def bits_per_value(self) -> float:
        """Token bits per value: n_tok * log2(codebook) / (channels * values/channel).

        The rate the reconstruction has to live within. In raw mode, 25 tokens of a
        1000-code codebook over 8 x 500 = 4000 values is 0.0623 bits/value.
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
    """Build a sample-wise fast-TS config (``target="raw"``).

    The bare :class:`FastTSCodecConfig` defaults are the envelope ones, so configs and
    checkpoints from before raw mode keep loading strictly. Raw mode needs a different
    objective as well as a different target, so this sets both:

        target                 "envelope"  ->  "raw"
        pixel_anchor_weight    0.05        ->  1.0    (reconstruction leads)
        adversarial_weight     1.0         ->  0.0    (a GAN raises sample-wise nRMSE)
        fm_weight              1.0         ->  0.0
        consistency_weight     1.0         ->  0.0    (codes must carry spike timing)
        gate_recon_floor       0.2         ->  0.02   (sample-wise correlation is much
                                                       smaller than envelope
                                                       correlation)

    Keyword overrides are applied last.
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


# Slow-TS codec (Thomson, CER, MSE, Phase A)
#
# docs/IGNITE_DESIGN.md §4.3: "Slow-TS: smooth profiles; lightest touch." These are
# kinetic profiles (Thomson electron density and temperature, CER ion temperature and
# rotation, MSE pitch angle), loaded at 100 Hz (SLOWTS_FS), so a 50 ms frame is a small
# (C_positions, T=5) window.
#
# The config shares the FSQ bottleneck, entropy regularizer and gate thresholds with the
# other codecs, but has no spectrogram-specific or adversarial parts:
#   * No delta-shift consistency term. That removes an STFT-phase realization nuisance
#     which smooth, directly sampled profiles do not have.
#   * No adversarial, discriminator or feature-matching terms. A smooth low-dimensional
#     profile does not need GAN sharpening, and these signals have real missing data
#     (beam-off gaps, zeros where the diagnostic did not fire). An adversarial decoder
#     would invent values in those gaps. The codec is reconstruction plus entropy,
#     with the reconstruction masked where data are missing.
#
# One codec class serves all 7 signals. Each signal has its own config (channel count
# and patch sizes) and is trained and frozen separately. `signal` selects the loader's
# channel count and missingness policy and the logging name; the tensor contract is the
# same for all signals.

# The 7 slow-TS signals and their channel (position) counts, from
# data_loader.TokamakH5Dataset.SIGNAL_CONFIGS (num_channels). These are defaults only;
# the trainer reads the real count from the loader at runtime.
SLOWTS_SIGNALS: Tuple[str, ...] = (
    "ts_core_density", "ts_core_temp", "ts_tangential_density", "ts_tangential_temp",
    "cer_ti", "cer_rot", "mse",
)
# Positions (channel count) per signal, from SIGNAL_CONFIGS num_channels.
SLOWTS_POSITIONS: Dict[str, int] = {
    "ts_core_density": 44, "ts_core_temp": 44,
    "ts_tangential_density": 10, "ts_tangential_temp": 10,
    "cer_ti": 48, "cer_rot": 48, "mse": 69,
}
# Missingness per signal, mirroring data_loader.SignalConfig.zero_is_missing: Thomson
# marks missing samples as zero (`data != 0` is valid); CER and MSE use NaN
# (`~isnan(data)` is valid).
SLOWTS_ZERO_IS_MISSING: Dict[str, bool] = {
    "ts_core_density": True, "ts_core_temp": True,
    "ts_tangential_density": True, "ts_tangential_temp": True,
    "cer_ti": False, "cer_rot": False, "mse": False,
}
# Preprocessing method per signal, as in SIGNAL_CONFIGS[*].preprocess.method. The codec
# standardizes its input the same way the dynamics model's data does (see
# SlowTSCodecConfig.channel_mean): the 4 Thomson signals use "log_standardize"
# (log10(clip(x,-0.99)+1), then per-channel (x-mean)/std with log-space stats), and
# cer_ti/cer_rot/mse use "standardize" (raw-space stats). It selects the 'log' or 'raw'
# sub-dict of preprocessing_stats.pt, as data_loader._update_preprocessing_stats does.
SLOWTS_PREPROCESS_METHOD: Dict[str, str] = {
    "ts_core_density": "log_standardize", "ts_core_temp": "log_standardize",
    "ts_tangential_density": "log_standardize", "ts_tangential_temp": "log_standardize",
    "cer_ti": "standardize", "cer_rot": "standardize", "mse": "standardize",
}


@dataclass
class SlowTSCodecConfig:
    """Slow-TS codec (Phase A), one per signal.

    Production uses seven instances, one per :data:`SLOWTS_SIGNALS`, each with its own
    encoder and decoder and no discriminator (§4.3). ``signal`` selects the position
    count, the missingness policy and the logging name; the tensor contract is the same
    for every signal.

    Tensor shapes (batch first):

        signal window   (B, C, T)    C = profile positions, T = samples (5 at 100 Hz)
        validity mask   (B, C, T)    1.0 = valid sample, 0.0 = missing or padded
        pre-FSQ feats   (B, n_tok, d_model)
        FSQ codes (int) (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
        quantized       (B, n_tok, d_model)
        reconstruction  (B, C, T)

        n_tok = (C // patch_c) * (T // patch_t)
    """

    # Signal selector (logging and missingness policy; the loader maps it to channels).
    signal: str = "ts_core_density"

    # Data and shape
    channels: int = 44            # C = real profile positions (from the loader)
    time_steps: int = 5           # T; a 50 ms window at SLOWTS_FS=100 Hz = 5 samples
    # Phase-B budget: 4 tokens per slow-TS signal, 4 radial zones x 1 time patch. The C
    # positions are split into 4 contiguous zones of patch_c = ceil(C / 4). When C is
    # not divisible by 4 the profile is padded to padded_channels = 4 * patch_c and the
    # padded positions are masked as missing, so they never reach the encoder input or
    # the loss (SlowTSCodecPairDataset._fit_window). The encoder and decoder patchify
    # over padded_channels, not channels. slowts_patch_for() computes patch_c.
    n_zones: int = 4              # radial-zone patches (n_pos_patch)
    patch_c: int = 11             # positions per zone; ceil(44 / 4) for the default C
    patch_t: int = 5              # time samples per patch (whole window = 1 time patch)

    # Per-channel standardization (see SlowTSCodecPairDataset._standardize). Raw slow-TS
    # values are unstandardized, and Thomson density is ~1e19 m^-3; fed to the encoder
    # directly, ts_core_density collapsed to one code. The dynamics data uses each
    # signal's SignalConfig.preprocess.method, so the codec standardizes the same way
    # before the encoder. Stats are global per channel, not per window, so the relative
    # profile level is kept. Length == channels.
    #
    # preprocess_method, matching data_loader._apply_preprocessing:
    #   "standardize"     -> (x - mean) / std.clamp(min=1e-3)   [raw-space stats]
    #   "log_standardize" -> arr = log10(clip(x, min=-0.99) + 1);
    #                        (arr - mean) / std.clamp(min=1e-3)
    #                        [log-space stats; the loader reads the 'log' sub-dict]
    #   "none" / None     -> identity (the default).
    # channel_mean / channel_std of None is also the identity, whatever the method, so
    # synthetic tests are unaffected. The trainer injects the real stats from
    # preprocessing_stats.pt (train_codec.load_slowts_channel_stats).
    preprocess_method: Optional[str] = None
    channel_mean: Optional[Sequence[float]] = None
    channel_std: Optional[Sequence[float]] = None

    # FSQ bottleneck: [8, 5, 5, 5] = 1000 codes, shared with the spectrogram quantizer.
    # A smooth profile needs fewer codes than a spectrogram, and the entropy regularizer
    # prunes the rest.
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = 1000

    # Transformer (x-transformers). These are smooth low-dimensional signals, so a small
    # one is enough.
    d_model: int = 128
    enc_depth: int = 4
    dec_depth: int = 4
    heads: int = 4

    # Masked reconstruction only (no adversarial, feature-matching or consistency terms;
    # see the section note). With no GAN to balance, this L1 term is the whole
    # reconstruction signal, hence weight 1.0.
    recon_weight: float = 1.0

    # Masked squared-error term, added as recon_mse_weight * masked_MSE. recon_weight
    # weights an L1 term, whose optimum is the conditional median, but the gate's
    # slowts_nrmse (gate.full_slowts_metrics, RMSE / std(target)) is a squared error,
    # whose optimum is the conditional mean. On bimodal, heavily masked profiles they
    # differ, and the L1-only ts_core_density codec over-smooths (within-window std
    # 0.45x the target's). Set recon_weight 0 for pure L2, or both > 0 for a mix. 0.0 =
    # off.
    recon_mse_weight: float = 0.0

    # Anti-collapse entropy regularizer, shared with the spectrogram quantizer
    # (per-sample entropy - diversity * batch-mean entropy, batch mean over all DDP
    # ranks). Masked reconstruction alone does not push for code diversity.
    # entropy_weight 1.0 gave min_dim_entropy ~0.65; 0.1 collapsed.
    entropy_weight: float = 1.0
    diversity_weight: float = 1.0

    # Joint-code diversity and dimension decorrelation, as in SpectroCodecConfig.
    # SlowTSQuantizer is SpectroQuantizer, whose entropy_loss reads these with
    # getattr(cfg, ..., 0.0), so 0.0 changes nothing; defining them makes
    # --joint_entropy_weight / --decorrelation_weight usable for slow-TS. The
    # per-dimension diversity_weight reward is marginal, and a rank-1 encoder saturates
    # it while using few joint codes; joint_entropy_weight addresses that. 1000 codes is
    # well below the quantizer's joint-entropy size limit.
    joint_entropy_weight: float = 0.0
    joint_entropy_ramp_steps: int = 0
    decorrelation_weight: float = 0.0

    # Activity-stratified sampling, as in SpectroCodecConfig, but slow-TS is masked, so
    # activity is the window's present fraction (mask.mean()), not a std.
    # ts_core_density is bimodal: median present fraction ~0.07, but ~34% of windows are
    # >= 75% present, so about 2/3 of windows are nearly empty. Biasing toward present
    # fraction >= min_activity fills batches with well-observed windows. Both 0 = off;
    # the trainer turns them on for ts_core_density (train_codec._activity_overrides).
    # Here min_activity is a fraction in [0, 1].
    min_activity: float = 0.0       # present-fraction threshold in [0, 1]; 0 = off
    active_bias: float = 0.0        # P(redraw a sparse window); 0 = off

    # Oracle-gate acceptance thresholds (§4.4), the fields spike.gate_score and
    # gate.utilization read.
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
    gate_hard_min_codes: int = 8         # hard floor on distinct codes (as Spectro)
    gate_min_code_entropy: float = 0.3
    # Reconstruction floor for best-checkpoint selection: the gate score is -inf when
    # the profile envelope_corr is NaN or below this. Same as
    # SpectroCodecConfig.gate_recon_floor.
    gate_recon_floor: float = 0.2

    @property
    def zero_is_missing(self) -> bool:
        """Missingness policy for this signal (as SignalConfig.zero_is_missing).

        True: Thomson; a sample is missing where the raw value is 0.
        False: CER/MSE beam gaps; a sample is missing where the raw value was NaN.
        """
        return SLOWTS_ZERO_IS_MISSING.get(self.signal, False)

    @property
    def padded_channels(self) -> int:
        """Position count the encoder and decoder patchify over (``n_zones * patch_c``).

        At least ``channels``. The extra tail positions are zone padding, masked as
        missing by :meth:`SlowTSCodecPairDataset._fit_window` so they never reach the
        encoder input or the loss. Equals ``channels`` when ``channels`` is divisible by
        ``n_zones``.
        """
        return self.n_zones * self.patch_c

    @property
    def n_pos_patch(self) -> int:
        # padded_channels // patch_c == n_zones by construction.
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
        # The encoder and decoder patchify over padded_channels = n_zones * patch_c, so
        # channels need not be divisible by patch_c. patch_c must hold the profile in
        # n_zones zones (channels <= padded_channels) without leaving a whole zone empty
        # (channels > (n_zones-1)*patch_c), i.e. patch_c == ceil(channels / n_zones).
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
    """Pick ``(patch_c, patch_t)`` so a slow-TS signal gives exactly ``n_zones`` tokens.

    Each slow-TS signal gets ``n_zones`` tokens: ``n_zones`` contiguous, near-equal
    radial zones of ``patch_c = ceil(channels / n_zones)`` positions, times one time
    patch (``patch_t = time_steps``, the whole 50 ms window).

    ``channels`` need not be divisible by ``n_zones`` (the real counts 44/10/48/69 are
    not). The profile is padded to ``padded_channels = n_zones * patch_c`` and the
    padded positions are masked as missing (:meth:`SlowTSCodecPairDataset._fit_window`).
    With ``n_zones = 4``:

        C=44 -> patch_c=11 (no padding)                    -> 4 tokens
        C=10 -> patch_c=3  (padded_channels 12, pad 2)     -> 4 tokens
        C=48 -> patch_c=12 (no padding)                    -> 4 tokens
        C=69 -> patch_c=18 (padded_channels 72, pad 3)     -> 4 tokens
    """
    patch_c = -(-int(channels) // int(n_zones))   # ceil(channels / n_zones)
    return patch_c, time_steps
