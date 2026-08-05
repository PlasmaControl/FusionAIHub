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
from .losses import (
    _as_score_list,
    _mean_over_maps,
    feature_matching_loss,
    freq_gradient_loss,
    multiscale_recon_loss,
    shift_consistency,
)
from .nets import SpectroConvDecoder, SpectroDecoder, SpectroEncoder
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
        # DECODER family (gated drop-in; cfg.decoder defaults to "linear" -> byte-identical).
        # Both expose the SAME forward(quant)->(B,C,F,T) and `last_layer` property, so the
        # generative loss / adaptive-adv machinery below is unchanged for either.
        self.decoder = (
            SpectroConvDecoder(cfg) if getattr(cfg, "decoder", "linear") == "conv"
            else SpectroDecoder(cfg)
        )

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
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Full generator loss for one codec (generator) step.

        Runs the codec forward on ``x`` (the reconstruction target is ``x`` itself — the
        decoder reconstructs the window it encoded) and separately encodes ``x_shift`` (the
        δ-shifted nuisance pair) to compute the shift-consistency term on the PRE-FSQ
        features of both.

        FIXED-WEIGHT (``cfg.adaptive_adv_weight`` is False — the prior behavior EXACTLY):

            ref   = cfg.pixel_anchor_weight * pixel
                    + cfg.consistency_weight * consistency
                    + cfg.entropy_weight * entropy
            total = cfg.adversarial_weight * adversarial + ref

        The recon objective's ``total`` already folds in ``cfg.adversarial_weight`` and
        ``cfg.pixel_anchor_weight`` (see ``losses.recon_objective``); this helper adds the
        consistency + entropy terms on top, weighted by ``cfg.consistency_weight`` /
        ``cfg.entropy_weight``.

        ADAPTIVE (``cfg.adaptive_adv_weight`` is True — VQGAN/MagViT "adaptive weight",
        "Taming Transformers" §3.3): the adversarial coefficient is auto-scaled every step
        so the adversarial gradient never overpowers the diversity terms::

            ref       = cfg.pixel_anchor_weight * pixel
                        + cfg.consistency_weight * consistency
                        + cfg.entropy_weight * entropy      # the NON-adv total
            adv       = adversarial = -mean(D(recon))       # raw generator adv term
            g_ref     = autograd.grad(ref, last_layer, retain_graph=True)[0]
            g_adv     = autograd.grad(adv, last_layer, retain_graph=True)[0]
            lam       = (‖g_ref‖ / (‖g_adv‖ + 1e-4)).clamp(0, cfg.adaptive_adv_clamp).detach()
            adv_coeff = 0.0 if step < cfg.adv_warmup_steps else cfg.adversarial_weight * lam
            total     = ref + adv_coeff * adv

        ``last_layer`` is ``self.decoder.last_layer`` (the ``to_pixels`` weight). ``lam``
        is DETACHED (a constant scale, no grad path). Zero-grad cases are handled
        gracefully (a term whose weight is 0 contributes no grad → its norm is 0, no NaN);
        in particular if ‖g_adv‖ is 0 the ``+ 1e-4`` keeps ``lam`` finite.

        Parameters
        ----------
        x : (B, C, F, T)
            The window to reconstruct (also the recon target).
        x_shift : (B, C, F, T)
            The δ-shifted nuisance pair of ``x`` (same modes, different realization).
        disc : nn.Module
            The (external) discriminator; scored on ``recon`` for the adversarial term.
        cfg : SpectroCodecConfig
        step : int
            Current global training step (for ``cfg.adv_warmup_steps``). Default 0.

        An anti-collapse entropy regularizer is folded into ``ref``:
        ``cfg.entropy_weight * self.quantizer.entropy_loss(enc(x))`` — this is the
        pressure toward codebook diversity that the consistency loss alone lacks (its
        trivial minimum is encoder ≡ constant, i.e. one code).

        Returns
        -------
        dict with keys:
            ``total``            scalar generator loss (backprop this into the codec).
            ``adversarial``      RAW adversarial (hinge generator) term  [-mean(D(recon))].
            ``pixel``            raw pixel-MAE anchor (un-weighted)   [|recon - x|.mean()].
            ``feature_matching`` raw disc feature-matching L1 (un-weighted); folded into
                                 recon_ref (adaptive path only) via ``cfg.fm_weight``.
            ``consistency``      raw shift-consistency MSE (un-weighted).
            ``entropy``          raw anti-collapse entropy term (un-weighted).
            ``adaptive_weight``  float(lam) — the adaptive scale used this step (1.0 in the
                                 fixed-weight path, where it is not applied).
            ``recon``            (B, C, F, T) reconstruction (detached-free; live graph).
            ``codes``            (B, n_tok, fsq_dim) codes for ``x`` (for gate/logging).
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]

        # ONE discriminator pass over the reconstruction, returning BOTH the patch scores
        # (for the adversarial term) and the intermediate features (for feature matching).
        # This reuses the fake scores instead of a second disc(recon) call (as
        # recon_objective would do), so the adversarial term is identical but cheaper.
        fake_scores, fake_feats = disc(recon, return_features=True)
        fake_scores = _as_score_list(fake_scores)
        adversarial = -_mean_over_maps(fake_scores)  # hinge generator term: -mean(D(recon))

        pixel = torch.mean(torch.abs(recon - x))
        # Multi-resolution + freq-gradient reconstruction (NeMo-style; sharpens turbulent detail
        # that plain pixel-L1 blurs). No-op when the weights are 0 (byte-identical).
        multiscale = (multiscale_recon_loss(recon, x)
                      if cfg.multiscale_recon_weight > 0 else recon.new_zeros(()))
        freq_grad = (freq_gradient_loss(recon, x)
                     if cfg.freq_grad_weight > 0 else recon.new_zeros(()))

        # DETACHED real features: generator matches fake -> real (grad only via fake feats).
        with torch.no_grad():
            _, real_feats = disc(x, return_features=True)
        fm = feature_matching_loss(real_feats, fake_feats)

        feats_shift = self.encode(x_shift)
        consistency = shift_consistency(feats_x, feats_shift)

        entropy = self.quantizer.entropy_loss(feats_x)

        # VQGAN-canonical (Taming Transformers §3.3): the adaptive weight balances the
        # adversarial term against the RECONSTRUCTION reference — NOT the consistency/entropy
        # regularizers. Referencing those (the earlier choice) had large gradients that
        # crushed lam to ~1%, over-suppressing the GAN and blurring the decode. The
        # regularizers still enter `total`; they're just outside the adv-vs-recon balance.
        #
        # The reconstruction reference now ALSO includes the disc feature-matching term
        # (HiFi-GAN/MelGAN, the spectrogram-adapted Genie perceptual loss). Because fm is a
        # substantial, realization-safe perceptual reference, recon_ref grows -> lam rises ->
        # the balanced adversarial term regains sharpening strength.
        recon_ref = (cfg.pixel_anchor_weight * pixel + cfg.fm_weight * fm
                     + cfg.multiscale_recon_weight * multiscale
                     + cfg.freq_grad_weight * freq_grad)
        non_adv_total = (
            recon_ref
            + cfg.consistency_weight * consistency
            + cfg.entropy_weight * entropy
        )

        if cfg.adaptive_adv_weight:
            lam = self._adaptive_adv_weight(recon_ref, adversarial, cfg)
            adv_coeff = (
                torch.zeros((), device=non_adv_total.device, dtype=non_adv_total.dtype)
                if step < cfg.adv_warmup_steps
                else cfg.adversarial_weight * lam
            )
            total = non_adv_total + adv_coeff * adversarial
            adaptive_weight = float(lam)
        else:
            # PRIOR fixed-weight behavior, EXACTLY (regression path — fm is NOT folded in
            # here; the feature-matching lever only participates in the adaptive balance):
            #   total = adversarial_weight*adv + pixel_anchor_weight*pixel
            #           + consistency_weight*consistency + entropy_weight*entropy
            total = (
                cfg.adversarial_weight * adversarial
                + cfg.pixel_anchor_weight * pixel
                + cfg.multiscale_recon_weight * multiscale
                + cfg.freq_grad_weight * freq_grad
                + cfg.consistency_weight * consistency
                + cfg.entropy_weight * entropy
            )
            adaptive_weight = 1.0

        return {
            "total": total,
            "adversarial": adversarial,
            "pixel": pixel,
            "multiscale": multiscale,
            "freq_grad": freq_grad,
            "feature_matching": fm,
            "consistency": consistency,
            "entropy": entropy,
            "adaptive_weight": adaptive_weight,
            "recon": recon,
            "codes": codes,
        }

    # ------------------------------------------------------------------ #
    # VQGAN adaptive adversarial weight ("Taming Transformers" §3.3)
    # ------------------------------------------------------------------ #
    def _adaptive_adv_weight(
        self,
        ref: torch.Tensor,
        adv: torch.Tensor,
        cfg: SpectroCodecConfig,
    ) -> torch.Tensor:
        """lam = ‖∇ref‖ / (‖∇adv‖ + 1e-4), clamped to [0, cfg.adaptive_adv_clamp], DETACHED.

        Gradients are taken w.r.t. the decoder's last layer (``self.decoder.last_layer``),
        the tensor at which the reconstruction and adversarial signals are balanced. Both
        grads use ``retain_graph=True`` so the subsequent ``total.backward()`` still has the
        full graph; ``allow_unused=True`` handles the case where a term is 0 by weight (its
        grad is then treated as zero, no NaN). Returned as a detached 0-dim tensor.
        """
        last_layer = self.decoder.last_layer
        zero = torch.zeros((), device=last_layer.device, dtype=last_layer.dtype)

        def _grad_norm(term: torch.Tensor) -> torch.Tensor:
            if not term.requires_grad:
                # term is a constant w.r.t. last_layer (e.g. weighted by 0) -> zero grad.
                return zero
            g = torch.autograd.grad(
                term, last_layer, retain_graph=True, allow_unused=True
            )[0]
            if g is None:
                return zero
            return g.norm()

        g_ref = _grad_norm(ref)
        g_adv = _grad_norm(adv)
        lam = (g_ref / (g_adv + 1e-4)).clamp(0.0, cfg.adaptive_adv_clamp)
        return lam.detach()
