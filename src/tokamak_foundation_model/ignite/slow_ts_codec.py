"""Slow-TS codec: encoder, FSQ quantizer and reconstruction decoder.

Same structure and interface as :class:`~ignite.codec.SpectroCodec` (``encode``,
``quantize``, ``decode``, ``forward``, ``generator_losses``), for slow-TS windows
``(B, C, T)`` of ``C`` profile positions by ``T`` time samples. It differs in two ways
(docs/IGNITE_DESIGN.md §4.3):

* No δ-shift consistency term. That term removes an STFT-phase nuisance that only
  spectrograms have, so ``generator_losses`` takes a single ``x`` and its validity mask.
* No discriminator or feature matching. A smooth low-dimensional profile gains no
  sharpness from a GAN, and an adversarial decoder would invent values in genuinely
  missing regions (beam-off gaps, diagnostics not firing).

The loss is a reconstruction MAE over valid samples only, plus the entropy regularizer.
Without the mask the MAE would pull the codec toward the zeros the loader puts in
missing samples (Thomson ``zero_is_missing`` zeros, CER/MSE beam-off NaNs filled with
0). The FSQ bottleneck and entropy term are the shared spectro machinery
(``slow_ts_quantizer``). No FAITH model code is used (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import SlowTSCodecConfig
from .slow_ts_nets import SlowTSDecoder, SlowTSEncoder
from .slow_ts_quantizer import SlowTSQuantizer


class SlowTSCodec(nn.Module):
    """Encoder, FSQ quantizer and reconstruction decoder for one slow-TS signal.

    ``forward(x: (B, C, T))`` returns a dict with keys:

    - ``recon`` (B, C, T): decoded signal
    - ``feats`` (B, n_tok, d_model): continuous encoder features before FSQ
    - ``quant`` (B, n_tok, d_model): straight-through quantized features
    - ``codes`` (B, n_tok, fsq_dim): FSQ codes (long), the Phase-B input

    There is no discriminator; ``generator_losses`` takes the validity ``mask`` instead.
    """

    def __init__(self, cfg: SlowTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = SlowTSEncoder(cfg)
        self.quantizer = SlowTSQuantizer(cfg)
        self.decoder = SlowTSDecoder(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> continuous features before FSQ, (B, n_tok, d_model)."""
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

    def generator_losses(
        self,
        x: torch.Tensor,
        cfg: SlowTSCodecConfig,
        step: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Codec loss for one slow-TS training step.

        Runs the codec on ``x`` and combines a masked reconstruction loss with the
        entropy regularizer::

            recon   = mean over valid samples of |recon - x|
            entropy = quantizer.entropy_loss(enc(x))
            total   = cfg.recon_weight * recon + cfg.entropy_weight * entropy

        plus ``cfg.recon_mse_weight`` times the masked squared error when that weight is
        nonzero.

        Parameters
        ----------
        x : (B, C, T)
            Window to reconstruct; also the target.
        cfg : SlowTSCodecConfig
        step : int
            Only used for the joint-entropy ramp; kept for a signature shared with the
            spectro and video codecs.
        mask : (B, C, T), (B, T) or (B, C) bool or float, optional
            Validity mask, 1 for real samples and 0 for missing or padded ones. The
            reconstruction terms average over valid samples only. ``None`` means all
            valid. If the mask selects nothing, the unmasked mean is used so the loss
            stays finite.

        Returns
        -------
        dict with keys:
            ``total``            scalar loss to backpropagate.
            ``recon``            (B, C, T) reconstruction, with graph.
            ``pixel``            masked reconstruction MAE.
            ``pixel_mse``        masked squared error (0 when its weight is 0).
            ``entropy``          entropy term.
            ``adaptive_weight``  constant 1.0, kept for the same log keys as the other
                                 codecs.
            ``codes``            (B, n_tok, fsq_dim) codes of ``x``.
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]

        pixel = self._masked_recon_mae(recon, x, mask)
        # Optional squared error, skipped when its weight is 0. The gate's slowts_nrmse
        # is an RMSE, and L1 alone fits the conditional median rather than the mean.
        mse_w = float(getattr(cfg, "recon_mse_weight", 0.0))
        pixel_mse = (self._masked_recon_mse(recon, x, mask) if mse_w != 0.0
                     else recon.new_zeros(()))
        # `step` drives the joint-entropy ramp (cfg.joint_entropy_ramp_steps); with the
        # default ramp and weight of 0 it has no effect.
        entropy = self.quantizer.entropy_loss(feats_x, step=step)

        total = (cfg.recon_weight * pixel + mse_w * pixel_mse
                 + cfg.entropy_weight * entropy)

        return {
            "total": total,
            "recon": recon,
            "pixel": pixel,
            "pixel_mse": pixel_mse,
            "entropy": entropy,
            "adaptive_weight": 1.0,
            "codes": codes,
        }

    @staticmethod
    def _broadcast_mask(recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Expand a (B, C, T), (B, T) or (B, C) mask to the shape of ``recon``."""
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
        return m.expand_as(recon)

    @staticmethod
    def _masked_recon_mae(
        recon: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean |recon - x| over valid positions (all positions if ``mask`` is None).

        ``mask`` may be (B, C, T), (B, T) (broadcast over C) or (B, C) (broadcast over
        T). If it selects nothing, the unmasked mean is returned so the term stays
        finite.
        """
        if mask is None:
            return torch.mean(torch.abs(recon - x))
        m = SlowTSCodec._broadcast_mask(recon, mask)
        denom = m.sum()
        if float(denom) <= 0.0:
            return torch.mean(torch.abs(recon - x))
        return (torch.abs(recon - x) * m).sum() / denom

    @classmethod
    def _masked_recon_mse(
        cls, recon: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean ``(recon - x)**2`` over valid positions.

        Same mask handling and fallback as :meth:`_masked_recon_mae`, so both terms
        average over the same positions.
        """
        if mask is None:
            return torch.mean((recon - x) ** 2)
        m = cls._broadcast_mask(recon, mask)
        denom = m.sum()
        if float(denom) <= 0.0:
            return torch.mean((recon - x) ** 2)
        return (((recon - x) ** 2) * m).sum() / denom
