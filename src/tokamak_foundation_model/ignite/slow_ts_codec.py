"""IGNITE Phase-A slow-TS codec — encoder + FSQ quantizer + reconstruction decoder.

Mirrors :class:`~ignite.codec.SpectroCodec` (same composition: encoder + FSQ + decoder,
same ``encode`` / ``quantize`` / ``decode`` / ``forward`` / ``generator_losses`` surface)
for slow-TS windows ``(B, C, T)`` — ``C`` profile positions × ``T`` time samples — but with
the deliberate "lightest touch" differences of docs/IGNITE_DESIGN.md §4.3:

    * NO δ-shift consistency term (that projects out an STFT-phase realization nuisance that
      exists only for spectrograms; a smooth directly-sampled profile has none). So
      ``generator_losses`` takes a single ``x`` (plus its validity mask), never a nuisance
      pair.
    * NO adversarial / discriminator / feature-matching (a smooth low-D profile is not a
      texture-rich signal where a GAN buys sharpness; and a generative adversarial decoder
      would HALLUCINATE values into the real missing regions — beam-off gaps, diagnostic
      not-firing zeros — which is exactly wrong for a kinetic profile). So there is no
      discriminator argument and no ``decoder.last_layer`` adaptive-adversarial machinery.

Loss = **masked reconstruction (MAE over VALID samples only)** + entropy/utilization. The
mask is CRITICAL: without it the reconstruction MAE would drive the codec toward the
zero-/NaN-filled missing samples (Thomson ``zero_is_missing`` zeros; CER/MSE beam-off NaNs the
loader fills with 0), i.e. toward predicting "missing" as a real value. Masking the recon so
missing samples contribute NOTHING is what respects the loader's missingness contract.

Everything else — the FSQ bottleneck + the entropy/utilization anti-collapse regularizer — is
REUSED from the shared IGNITE infra (``slow_ts_quantizer`` == the spectro FSQ machinery). Reuses
**no** FAITH model code (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import SlowTSCodecConfig
from .slow_ts_nets import SlowTSDecoder, SlowTSEncoder
from .slow_ts_quantizer import SlowTSQuantizer


class SlowTSCodec(nn.Module):
    """Encoder + FSQ quantizer + reconstruction decoder for ONE slow-TS signal (Phase A).

    Forward contract
    ----------------
    ``forward(x: (B, C, T)) -> dict`` with keys:
        ``recon``  (B, C, T)            decoded profile-time signal
        ``feats``  (B, n_tok, d_model)  PRE-FSQ continuous encoder features
        ``quant``  (B, n_tok, d_model)  STE-quantized (decoder-facing) features
        ``codes``  (B, n_tok, fsq_dim)  discrete per-dim FSQ codes (long) — Phase-B contract.

    There is NO external discriminator (unlike the spectro / video codecs): the slow-TS codec
    is a reconstruction + entropy codec (§4.3 "lightest touch"). ``generator_losses`` takes the
    validity ``mask`` instead of a discriminator.
    """

    def __init__(self, cfg: SlowTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = SlowTSEncoder(cfg)
        self.quantizer = SlowTSQuantizer(cfg)
        self.decoder = SlowTSDecoder(cfg)

    # ------------------------------------------------------------------ #
    # forward paths
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> PRE-FSQ continuous features (B, n_tok, d_model)."""
        return self.encoder(x)

    def quantize(self, feats: torch.Tensor):
        """Features -> (quant (B,n_tok,d_model) float, codes (B,n_tok,fsq_dim) long)."""
        return self.quantizer.quantize(feats)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Quantized features (B, n_tok, d_model) -> reconstruction (B, C, T)."""
        return self.decoder(quant)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encode(x)
        quant, codes = self.quantize(feats)
        recon = self.decode(quant)
        return {"recon": recon, "feats": feats, "quant": quant, "codes": codes}

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    # ------------------------------------------------------------------ #
    # generator-side loss (the codec's whole loss — there is no GAN alternation)
    # ------------------------------------------------------------------ #
    def generator_losses(
        self,
        x: torch.Tensor,
        cfg: SlowTSCodecConfig,
        step: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full codec loss for one slow-TS training step.

        Runs the codec forward on ``x`` (the reconstruction target is ``x`` itself) and
        combines a MASKED reconstruction MAE with the entropy/utilization regularizer::

            recon   = mean_over_VALID |recon - x|         # masked where the signal is missing
            entropy = quantizer.entropy_loss(enc(x))       # anti-collapse (codebook diversity)
            total   = cfg.recon_weight * recon + cfg.entropy_weight * entropy

        There is **no** adversarial term, **no** feature-matching, **no** δ-shift consistency,
        and hence **no** VQGAN adaptive-adversarial weight (see the module docstring). ``step``
        is accepted for a uniform trainer signature but is unused (there is no adv warmup).

        Parameters
        ----------
        x : (B, C, T)
            The slow-TS window to reconstruct (also the recon target).
        cfg : SlowTSCodecConfig
        step : int
            Unused; accepted for a signature uniform with the spectro/video codecs.
        mask : (B, C, T) or (B, T) or (B, C) bool/float, optional
            Per-sample VALIDITY mask (1.0 = valid/real, 0.0 = missing/padded). When given, the
            reconstruction MAE is averaged over VALID positions ONLY, so a beam-off gap /
            diagnostic-not-firing zero does not drag the codec toward the (zero/NaN-filled)
            missing value. ``None`` = all valid. If the mask selects nothing (a whole batch
            of fully-missing windows) the term falls back to the unmasked MAE so it stays
            finite + differentiable.

        Returns
        -------
        dict with keys:
            ``total``            scalar codec loss (backprop this into the codec).
            ``recon``            (B, C, T) reconstruction (live graph).
            ``pixel``            raw (masked) reconstruction MAE (for logging / gate).
            ``entropy``          raw anti-collapse entropy term.
            ``adaptive_weight``  1.0 constant (no adaptive-adv weight; present for a uniform
                                 trainer/log contract with the spectro/video codecs).
            ``codes``            (B, n_tok, fsq_dim) codes for ``x`` (for gate/logging).
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]

        pixel = self._masked_recon_mae(recon, x, mask)
        entropy = self.quantizer.entropy_loss(feats_x)

        total = cfg.recon_weight * pixel + cfg.entropy_weight * entropy

        return {
            "total": total,
            "recon": recon,
            "pixel": pixel,
            "entropy": entropy,
            "adaptive_weight": 1.0,
            "codes": codes,
        }

    # ------------------------------------------------------------------ #
    # masked reconstruction MAE
    # ------------------------------------------------------------------ #
    @staticmethod
    def _masked_recon_mae(
        recon: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean |recon - x| over VALID positions (all positions if ``mask`` is None).

        ``mask`` may be (B, C, T) (per-sample), (B, T) (per-time, broadcast over C), or
        (B, C) (per-position, broadcast over T). If the mask selects nothing (a whole batch
        of fully-missing windows) the MAE falls back to the unmasked mean so the term is
        always finite and differentiable.
        """
        if mask is None:
            return torch.mean(torch.abs(recon - x))
        m = mask.to(dtype=recon.dtype)
        if m.dim() == 3:            # (B, C, T)
            pass
        elif m.dim() == 2 and m.shape == (recon.shape[0], recon.shape[2]):  # (B, T)
            m = m[:, None, :]
        elif m.dim() == 2 and m.shape == (recon.shape[0], recon.shape[1]):  # (B, C)
            m = m[:, :, None]
        else:
            raise ValueError(
                f"mask must be (B,C,T), (B,T) or (B,C); got {tuple(mask.shape)} "
                f"for recon {tuple(recon.shape)}"
            )
        m = m.expand_as(recon)
        denom = m.sum()
        if float(denom) <= 0.0:
            return torch.mean(torch.abs(recon - x))
        return (torch.abs(recon - x) * m).sum() / denom
