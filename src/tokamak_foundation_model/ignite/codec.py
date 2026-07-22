"""IGNITE Phase-A codec — encoder + FSQ quantizer + decoder, composed.

This module composes the already-built, independently-tested Phase-A components
(``nets.SpectroEncoder`` / ``nets.SpectroDecoder``, ``quantizer.SpectroQuantizer``)
into a single :class:`SpectroCodec` and provides the generator-side loss helper for a
standard GAN alternation. It reuses **no** FAITH model code (see
``docs/IGNITE_DESIGN.md`` §7) — only ``torch`` + sibling ``ignite`` modules.

Design (§4): the codec is the statistics-first tokenizer. The encoder maps a log-power
spectrogram ``(B, C, F, T)`` to continuous features ``(B, n_tok, d_model)``; the FSQ
bottleneck discretizes them to per-dimension codes (the Phase-B token contract) and yields
STE-differentiable quantized features; the (generative, adversarial) decoder maps those
back to a spectrogram.

The **discriminator lives OUTSIDE the codec** (created alongside it by the trainer): the
GAN alternation trains the codec (encoder+quantizer+decoder) on the generator loss and the
discriminator separately via :func:`losses.discriminator_loss`. Keeping the discriminator
external means (a) the codec's parameters are exactly the tokenizer that Phase B freezes,
and (b) the two optimizers own disjoint parameter sets, which is the standard GAN pattern.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .config import SpectroCodecConfig
from .losses import recon_objective, shift_consistency
from .nets import SpectroDecoder, SpectroEncoder
from .quantizer import SpectroQuantizer


class SpectroCodec(nn.Module):
    """Encoder + FSQ quantizer + decoder for one spectro modality (Phase A).

    Parameters
    ----------
    cfg : SpectroCodecConfig
        Single source of truth for shapes / hyper-parameters.

    Forward contract
    ----------------
    ``forward(x: (B, C, F, T)) -> dict`` with keys:
        ``recon``  (B, C, F, T)         decoded spectrogram (generative output)
        ``feats``  (B, n_tok, d_model)  PRE-FSQ continuous encoder features
        ``quant``  (B, n_tok, d_model)  STE-quantized (decoder-facing) features
        ``codes``  (B, n_tok, fsq_dim)  discrete per-dim FSQ codes (long) — the Phase-B
                                        token contract.

    The discriminator is **not** a submodule (see module docstring); pass it into
    :meth:`generator_losses` / :func:`losses.discriminator_loss` explicitly.
    """

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = SpectroEncoder(cfg)
        self.quantizer = SpectroQuantizer(cfg)
        self.decoder = SpectroDecoder(cfg)

    # ------------------------------------------------------------------ #
    # forward paths
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) -> PRE-FSQ continuous features (B, n_tok, d_model).

        This is the tensor the shift-consistency loss operates on (§4.2: consistency is
        enforced on the PRE-FSQ features, avoiding discrete matching).
        """
        return self.encoder(x)

    def quantize(self, feats: torch.Tensor):
        """Features -> (quant (B,n_tok,d_model) float, codes (B,n_tok,fsq_dim) long)."""
        return self.quantizer.quantize(feats)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Quantized features (B, n_tok, d_model) -> reconstruction (B, C, F, T)."""
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
    # generator-side loss (the codec's half of the GAN alternation)
    # ------------------------------------------------------------------ #
    def generator_losses(
        self,
        x: torch.Tensor,
        x_shift: torch.Tensor,
        disc: nn.Module,
        cfg: SpectroCodecConfig,
    ) -> Dict[str, torch.Tensor]:
        """Full generator loss for one codec (generator) step.

        Runs the codec forward on ``x`` (the reconstruction target is ``x`` itself — the
        decoder reconstructs the window it encoded) and separately encodes ``x_shift`` (the
        δ-shifted nuisance pair) to compute the shift-consistency term on the PRE-FSQ
        features of both.

        total = recon_objective(recon, x, disc, cfg)["total"]
                + cfg.consistency_weight * shift_consistency(enc(x), enc(x_shift))

        The recon objective's ``total`` already folds in ``cfg.adversarial_weight`` and
        ``cfg.pixel_anchor_weight`` (see ``losses.recon_objective``); this helper adds only
        the consistency term on top, weighted by ``cfg.consistency_weight``.

        Parameters
        ----------
        x : (B, C, F, T)
            The window to reconstruct (also the recon target).
        x_shift : (B, C, F, T)
            The δ-shifted nuisance pair of ``x`` (same modes, different realization).
        disc : nn.Module
            The (external) discriminator; scored on ``recon`` for the adversarial term.
        cfg : SpectroCodecConfig

        An anti-collapse entropy regularizer is added on top:
        ``+ cfg.entropy_weight * self.quantizer.entropy_loss(enc(x))`` — this is the
        pressure toward codebook diversity that the consistency loss alone lacks (its
        trivial minimum is encoder ≡ constant, i.e. one code).

        Returns
        -------
        dict with keys:
            ``total``        scalar generator loss (backprop this into the codec).
            ``adversarial``  adversarial (hinge generator) term  [from recon_objective].
            ``pixel``        raw pixel-MAE anchor (un-weighted)   [from recon_objective].
            ``consistency``  raw shift-consistency MSE (un-weighted).
            ``entropy``      raw anti-collapse entropy term (un-weighted).
            ``recon``        (B, C, F, T) reconstruction (detached-free; live graph).
            ``codes``        (B, n_tok, fsq_dim) codes for ``x`` (for gate/logging).
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]

        recon_terms = recon_objective(recon, x, disc, cfg)

        feats_shift = self.encode(x_shift)
        consistency = shift_consistency(feats_x, feats_shift)

        entropy = self.quantizer.entropy_loss(feats_x)

        total = (
            recon_terms["total"]
            + cfg.consistency_weight * consistency
            + cfg.entropy_weight * entropy
        )

        return {
            "total": total,
            "adversarial": recon_terms["adversarial"],
            "pixel": recon_terms["pixel"],
            "consistency": consistency,
            "entropy": entropy,
            "recon": recon,
            "codes": codes,
        }
