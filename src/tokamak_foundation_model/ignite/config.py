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
from math import prod
from typing import Dict, List, Tuple

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
    patch_f: int = 64             # -> 8 freq patches
    patch_t: int = 32             # -> 3 time patches   => n_tok = 24

    # bottleneck (vector-quantize-pytorch FSQ)
    # Right-sized to the FSQ-paper-recommended ~1024-code config: [8, 5, 5, 5] = prod = 1000.
    # The previous [8, 8, 8, 8, 8] = 32768 was ~30x over-provisioned for a codec that only
    # ever uses O(10-30) codes; the spare 5th dim always died (min_dim_entropy=0). Four dims
    # at 1000 codes removes that dead dim while keeping ample capacity.
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = prod = 1000

    # transformer (x-transformers)
    d_model: int = 256
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8

    # invariance (Phase-A consistency-loss-only; raw d-shift + re-STFT)
    consistency_delta_ms: Tuple[float, float] = (0.1, 2.0)  # sub-window shift range (< 16 ms codec patch)
    consistency_weight: float = 1.0
    amp_jitter: float = 0.03      # +-3% multiplicative secondary nuisance; 0 disables
    noise_floor: float = 0.0      # additive log-power noise std; 0 disables

    # decoder / reconstruction objective
    adversarial_weight: float = 1.0
    pixel_anchor_weight: float = 0.05   # lambda_pix; STABILITY-GATED (raise only while stability >= 0.80)
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
    adaptive_adv_clamp: float = 1e4     # upper clamp on lam (lower clamp is 0)
    adv_warmup_steps: int = 0           # steps with adv_coeff forced to 0 (adv term off during warmup)

    # anti-collapse (codebook-utilization) regularizer — Genie-style entropy term.
    # The shift-consistency loss has a trivial minimum at encoder ≡ constant (all inputs
    # map to ONE code); these terms pressure the encoder toward diverse codebook usage.
    # entropy_loss = per_sample_entropy_mean - diversity_weight * entropy_of_batch_mean.
    entropy_weight: float = 0.1     # multiplies entropy_loss into the generator total
    diversity_weight: float = 1.0   # weight on the batch-diversity (spread) reward

    # oracle-gate acceptance thresholds
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    # collapse detection (§4.4 anti-posterior-collapse): a codec fails the gate if it uses
    # too little of the codebook or has too-low per-dim code entropy. NOTE (2026-07): this
    # `collapsed` flag is now INFORMATIONAL ONLY for best-ckpt selection — it no longer forces
    # gate_score = -inf. See `gate_recon_floor` below and `spike.gate_score`.
    gate_min_utilization: float = 0.02   # min fraction of the codebook used to pass
    gate_min_code_entropy: float = 0.3   # min per-dim normalized code entropy to pass
    # Reconstruction floor for best-ckpt DISQUALIFICATION (the only hard gate on the score).
    # A codec is disqualified (gate_score = -inf) ONLY when reconstruction genuinely fails:
    # decode envelope_corr is NaN or < gate_recon_floor. Utilization is folded in as a SOFT
    # reward term instead of a hard gate, so a well-reconstructing codec with one dead FSQ dim
    # is no longer wrongly rejected (it just scores slightly lower on the utilization reward).
    gate_recon_floor: float = 0.2

    @property
    def n_freq_patch(self) -> int:
        return self.freq_bins // self.patch_f

    @property
    def n_time_patch(self) -> int:
        return self.time_frames // self.patch_t

    @property
    def n_tok(self) -> int:
        return self.n_freq_patch * self.n_time_patch

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.fsq_levels)

    @property
    def window_samples(self) -> int:
        return round(CHUNK_S * STFT_FS)

    def __post_init__(self) -> None:
        assert self.freq_bins % self.patch_f == 0, "freq_bins must be divisible by patch_f"
        assert self.time_frames % self.patch_t == 0, "time_frames must be divisible by patch_t"


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
    adaptive_adv_clamp: float = 1e4
    adv_warmup_steps: int = 0

    # anti-collapse (codebook-utilization) entropy regularizer — Genie/LFQ style, reused
    # from the spectro quantizer's entropy_loss (per-sample entropy - diversity * batch-mean).
    entropy_weight: float = 0.1
    diversity_weight: float = 1.0

    # oracle-gate acceptance thresholds (§4.4) — video analogues.
    gate_stability: float = 0.80        # frame-to-frame code stickiness proxy (see gate note)
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
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
# Fast-TS codec (filterscopes / ELMs, Phase A) — statistics-first, ENVELOPE target.
# ---------------------------------------------------------------------------------------- #
# Design (docs/IGNITE_DESIGN.md §4.3): "Fast-TS (filterscopes / ELMs) — hardest. Statistic =
# ELM activity envelope (rate/amplitude), NOT spike timing; the decoder hallucinates
# plausible spikes." This is the fast-TS analogue of the spectro codec: it removes the SAME
# two failure modes (§4.1) with the SAME two moves —
#   (1) codes carry only the STATISTIC (the ELM activity envelope), so a sub-window spike
#       TIMING jitter (a realization nuisance, exactly like STFT phase for spectro) does not
#       scramble the codes → predictable / passes the stability gate;
#   (2) the decoder is generative (adversarial + small envelope-anchor) → it reconstructs a
#       sharp envelope instead of collapsing to the mean.
#
# THE ELM-ENVELOPE STATISTIC (the key design decision — see ignite.data.elm_envelope):
#   raw filterscope window (C, W)  [W = round(50 ms * 10 kHz) = 500 samples/chan]
#     -> subtract a slow moving-mean baseline (remove the DC / slow drift)   [detrend]
#     -> rectify (square)                                                    [|·|²]
#     -> average within non-overlapping pooling bins of `pool` samples (RMS) [mean-pool]
#     -> sqrt                                                                [-> RMS amplitude]
#     -> log1p(env / eps_ref)                                                [compress dynamic range]
#   giving an ENVELOPE (C, env_bins) with env_bins = W // pool. At pool=100 (10 ms bins),
#   env_bins = 5: a coarse "how much ELM activity, and when" curve per channel. A spike's
#   exact position WITHIN a 10 ms bin is discarded (the nuisance); the burst amplitude + timing
#   at bin resolution is kept (the physics). ELM bursts recur every ~1.6 ms so a 10 ms bin
#   averages over several bursts — a detector-free coarse RMS activity level, not a burst rate.
#   This is directly analogous to the spectro codec's per-frequency log-power (which discards
#   STFT phase, the spectro realization).
#
# The codec tokenizes the (C, env_bins) envelope like a 1-D "image": patch over the TIME
# (envelope-bin) axis (channels are a feature dim carried into the patch, as freq/space are
# for the other codecs), FSQ, generative decode back to the envelope.


# fast-TS geometry (matches the filterscopes SignalConfig + a 50 ms window at FASTTS_FS).
FASTTS_CHANNELS: int = 8         # channels_to_use=slice(0,8) on the filterscopes SignalConfig
FASTTS_WINDOW: int = round(CHUNK_S * FASTTS_FS)   # 500 raw samples / channel in a 50 ms window
# COARSE RMS envelope: a 3000-shot survey showed filterscope ELM bursts recur every ~1.6 ms and
# a burst-RATE detector is param-fragile (rate swings 3× with threshold; interval just tracks
# the min-distance floor), so we commit to a DETECTOR-FREE COARSE RMS envelope and coarsen the
# bins from 1 ms to 10 ms — several inter-burst intervals per bin averages out burst-timing
# realization noise while still resolving the shot-scale ELM activity level.
FASTTS_POOL: int = round(0.010 * FASTTS_FS)       # envelope pooling bin = 100 samples = 10 ms at 10 kHz
FASTTS_ENV_BINS: int = FASTTS_WINDOW // FASTTS_POOL  # 5 envelope bins (10 ms each)


@dataclass
class FastTSCodecConfig:
    """Statistics-first fast-TS (filterscopes) codec (Phase A), ENVELOPE-domain.

    Tensor-shape conventions (batch-first):

        raw window       (B, C, W)          C channels, W = window_samples raw samples
        envelope         (B, C, E)          E = env_bins  (the codec INPUT/TARGET)
        pre-FSQ feats    (B, n_tok, d_model)
        fsq codes (int)  (B, n_tok, fsq_dim)   entry i in [0, fsq_levels[i])
        quantized        (B, n_tok, d_model)
        reconstruction   (B, C, E)          reconstructed ENVELOPE (not raw spikes)

        n_tok = env_bins // patch_e
    """

    # data / shape
    channels: int = FASTTS_CHANNELS       # 8 filterscope channels
    env_bins: int = FASTTS_ENV_BINS       # E = 5 (10 ms per bin over a 50 ms window)
    patch_e: int = 1                       # -> 5 time patches (10 ms each)  => n_tok = 5

    # envelope transform params (see ignite.data.elm_envelope). Right-sized to FASTTS_* so the
    # loader's 500-sample / 10 kHz filterscope window maps to a 5-bin (10 ms) envelope.
    pool: int = FASTTS_POOL                # RMS pooling bin (samples) = 100 = 10 ms
    baseline_win: int = 20                 # moving-mean baseline window (samples) = 2 ms; 0 disables detrend
    env_eps: float = 1e-3                  # log1p reference: log1p(env / env_eps); guards flat/zero windows

    # bottleneck (vector-quantize-pytorch FSQ) — right-sized like the spectro/video fix:
    # [8, 5, 5, 5] = prod = 1000 codes, 4 dims. The envelope carries O(10) distinct "activity
    # levels/patterns", so 1000 codes is ample; 4 dims avoids the spare-dim death seen at
    # [8,8,8,8,8]=32768 (min_dim_entropy=0).
    fsq_levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])  # codebook = 1000

    # transformer (x-transformers) — token set over envelope-time patches, bidirectional.
    d_model: int = 256
    enc_depth: int = 6
    dec_depth: int = 6
    heads: int = 8

    # invariance (Phase-A consistency-loss). UNLIKE the video codec (§4.3 "no shift term"),
    # fast-TS DOES take a δ-shift consistency term: the ELM spike TIMING is a
    # realization/nuisance exactly like the spectrogram's STFT phase (§4.3 names it as such),
    # so a sub-bin raw δ-shift produces the SAME envelope statistic but a different spike
    # realization. The consistency loss ‖enc(env) − enc(env_shifted)‖² on the PRE-FSQ features
    # projects that nuisance out (statistics-first).
    #
    # δ RANGE — capped at the pool-bin duration (pool=100 samples = 10 ms at 10 kHz), i.e.
    # δ ~ U[0.1, 5.0] ms. With the COARSE 10 ms envelope bin a δ up to ~5 ms (half a bin) stays
    # WITHIN a single bin, so the per-bin RMS activity statistic is preserved and the envelope
    # curve is invariant (that is the whole reason the cap was pinned at < 1 bin — at the old
    # 1 ms bin that meant < 1 ms; at the new 10 ms bin it can be up to ~5 ms). A shift of a WHOLE
    # bin (≥10 ms) would translate the envelope by whole bins and break the statistic, so the cap
    # stays sub-bin. (Spectro uses [0.1, 2] ms against its 0.512 ms STFT hop because its FSQ
    # time-patch is 16 ms, huge vs δ.)
    consistency_delta_ms: Tuple[float, float] = (0.1, 5.0)
    consistency_weight: float = 1.0

    # decoder / reconstruction objective (generative decoder, LOW envelope anchor).
    adversarial_weight: float = 1.0
    # LOW-weight envelope anchor (the fast-TS analogue of the spectro/video pixel anchor):
    # buys optimization stability; the sharp envelope reconstruction is produced by the
    # adversarial + feature-matching signal, not by minimizing anchor error to the mean.
    pixel_anchor_weight: float = 0.05
    # Discriminator FEATURE-MATCHING weight (HiFi-GAN/MelGAN vocoder-GAN perceptual term —
    # natural here since the envelope is a smooth 1-D signal like a mel-spectrogram row).
    # Folded into recon_ref so the VQGAN adaptive weight rises. Realization-SAFE.
    fm_weight: float = 1.0

    # VQGAN/MagViT adaptive adversarial weight ("Taming Transformers" §3.3) — reused verbatim
    # from the spectro/video codec (auto-scale adv coeff by grad-norm ratio at decoder.last_layer).
    adaptive_adv_weight: bool = True
    adaptive_adv_clamp: float = 1e4
    adv_warmup_steps: int = 0

    # anti-collapse (codebook-utilization) entropy regularizer — Genie/LFQ style, reused from
    # the shared SpectroQuantizer.entropy_loss (per-sample entropy - diversity * batch-mean).
    entropy_weight: float = 0.1
    diversity_weight: float = 1.0

    # oracle-gate acceptance thresholds (§4.4) — fast-TS analogues (same numeric mandates).
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
    gate_min_code_entropy: float = 0.3
    # Reconstruction floor for best-ckpt DISQUALIFICATION (the only hard gate on the score):
    # disqualified (gate_score = -inf) when envelope reconstruction genuinely fails
    # (envelope_corr NaN or < floor). Mirrors SpectroCodecConfig.gate_recon_floor.
    gate_recon_floor: float = 0.2

    @property
    def n_env_patch(self) -> int:
        return self.env_bins // self.patch_e

    @property
    def n_tok(self) -> int:
        return self.n_env_patch

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.fsq_levels)

    @property
    def window_samples(self) -> int:
        """Raw samples in a 50 ms window at FASTTS_FS (== env_bins * pool when consistent)."""
        return round(CHUNK_S * FASTTS_FS)

    def __post_init__(self) -> None:
        assert self.env_bins % self.patch_e == 0, "env_bins must be divisible by patch_e"
        assert self.pool >= 1, "pool must be >= 1"
        assert self.baseline_win >= 0, "baseline_win must be >= 0 (0 disables detrend)"


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
    channels: int = 44            # C = profile positions (read from the loader at runtime).
    time_steps: int = 5           # T; a 50 ms window at SLOWTS_FS=100 Hz = round(0.05*100) = 5.
    patch_c: int = 44             # positions per patch (default: whole profile = 1 position-patch).
    patch_t: int = 5              # time samples per patch (default: whole window = 1 time-patch).

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

    # anti-collapse (codebook-utilization) entropy regularizer — Genie/LFQ style, REUSED
    # verbatim from the spectro quantizer's entropy_loss (per-sample entropy - diversity *
    # batch-mean). The masked-recon objective alone does not pressure codebook diversity, so
    # this keeps the encoder from posterior-collapsing to a single code.
    entropy_weight: float = 0.1
    diversity_weight: float = 1.0

    # oracle-gate acceptance thresholds (§4.4) — slow-TS analogues (same fields the shared
    # spike.gate_score / gate.utilization read).
    gate_stability: float = 0.80
    gate_persistence: float = 0.50
    gate_min_utilization: float = 0.02
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
    def n_pos_patch(self) -> int:
        return self.channels // self.patch_c

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
        assert self.channels % self.patch_c == 0, "channels must be divisible by patch_c"
        assert self.time_steps % self.patch_t == 0, "time_steps must be divisible by patch_t"


def slowts_patch_for(channels: int, *, time_steps: int = 5) -> Tuple[int, int]:
    """Pick a sensible ``(patch_c, patch_t)`` for a slow-TS signal with ``channels`` positions.

    The default codec uses ONE token per window (whole profile × whole time-window), which is
    the "lightest touch" (a single 4-dim code per 50 ms frame per signal → a tiny per-frame
    token budget for Phase B). ``patch_t = time_steps`` (whole window is one time-patch) and
    ``patch_c = channels`` (whole profile is one position-patch). Kept as a helper so the
    trainer can size the codec from the loader's real channel count without the caller
    hand-picking divisible patch sizes. Callers wanting finer position resolution can override
    ``patch_c`` on the returned cfg.
    """
    return channels, time_steps
