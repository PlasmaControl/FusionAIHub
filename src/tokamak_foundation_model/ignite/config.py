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
from typing import List, Tuple

# --- STFT / windowing (matches the existing spectro pipeline; see data_loader.py) ---
STFT_FS: float = 500_000.0   # Hz, spectro modality target sampling
STFT_N_FFT: int = 1024       # Hann window length -> 0.512 ms/frame at hop 256
STFT_HOP: int = 256          # STFT hop in samples
CHUNK_S: float = 0.05        # 50 ms = ONE FRAME (the world-model stepping unit)


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
    fsq_levels: List[int] = field(default_factory=lambda: [8, 8, 8, 8, 8])  # codebook = prod = 32768

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
    # too little of the codebook or has too-low per-dim code entropy.
    gate_min_utilization: float = 0.02   # min fraction of the codebook used to pass
    gate_min_code_entropy: float = 0.3   # min per-dim normalized code entropy to pass

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
