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
