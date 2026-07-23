"""IGNITE Phase-A tangtv **video** codec — encoder + FSQ quantizer + generative decoder.

Mirrors :class:`~ignite.codec.SpectroCodec` — same composition (encoder + FSQ + decoder),
same generator-side GAN loss shape (adversarial + low pixel anchor + feature-matching +
entropy, balanced by the VQGAN adaptive adversarial weight at the decoder's last layer) —
but for video windows ``(B, C, T, H, W)`` and with **ONE deliberate difference**:

    NO shift-consistency term.

The spectrogram codec needs a δ-shift consistency loss because a spectrogram carries an
STFT-phase / speckle *realization* nuisance (a 0.5 ms sub-window shift scrambles ~74 % of
the codes) that must be projected out so the codes carry only the *statistic*
(docs/IGNITE_DESIGN.md §4.1-§4.2). A divertor camera frame has **no such sub-window-phase
realization** — it is Genie-native smooth video — so there is no realization pair to be
invariant to, and §4.3 explicitly scopes the video codec to "mostly just the
no-strong-pixel-MSE / generative-decoder move." The statistics-first property comes from the
low-weight pixel anchor + adversarial/feature-matching generative decoder (which prevents the
mean-collapse a strong pixel-MSE would cause), NOT from a consistency pair. So
``generator_losses`` takes a single ``x`` (plus an optional per-frame validity mask), never a
nuisance pair.

Everything else — FSQ bottleneck, the adaptive adversarial weight, the entropy/utilization
regularizer, the feature-matching perceptual term — is reused from the shared IGNITE infra
(``video_quantizer`` == the spectro FSQ machinery; ``losses.feature_matching_loss`` /
``losses._as_score_list`` / ``losses._mean_over_maps``). Reuses **no** FAITH model code (§7).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import VideoCodecConfig
from .losses import _as_score_list, _mean_over_maps, feature_matching_loss
from .video_nets import VideoDecoder, VideoEncoder
from .video_quantizer import VideoQuantizer


class VideoCodec(nn.Module):
    """Encoder + FSQ quantizer + generative decoder for ONE tangtv divertor (Phase A).

    Forward contract
    ----------------
    ``forward(x: (B, C, T, H, W)) -> dict`` with keys:
        ``recon``  (B, C, T, H, W)      decoded video (generative output)
        ``feats``  (B, n_tok, d_model)  PRE-FSQ continuous encoder features
        ``quant``  (B, n_tok, d_model)  STE-quantized (decoder-facing) features
        ``codes``  (B, n_tok, fsq_dim)  discrete per-dim FSQ codes (long) — Phase-B contract.

    The discriminator (``video_discriminator.FramePatchGAN``) lives OUTSIDE the codec (created
    alongside it by the trainer), exactly as for the spectro codec: the codec's parameters are
    precisely the tokenizer Phase B freezes, and the two optimizers own disjoint parameters.
    """

    def __init__(self, cfg: VideoCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = VideoEncoder(cfg)
        self.quantizer = VideoQuantizer(cfg)
        self.decoder = VideoDecoder(cfg)

    # ------------------------------------------------------------------ #
    # forward paths
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> PRE-FSQ continuous features (B, n_tok, d_model)."""
        return self.encoder(x)

    def quantize(self, feats: torch.Tensor):
        """Features -> (quant (B,n_tok,d_model) float, codes (B,n_tok,fsq_dim) long)."""
        return self.quantizer.quantize(feats)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Quantized features (B, n_tok, d_model) -> reconstruction (B, C, T, H, W)."""
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
        disc: nn.Module,
        cfg: VideoCodecConfig,
        step: int = 0,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full generator loss for one video-codec (generator) step.

        Runs the codec forward on ``x`` (the reconstruction target is ``x`` itself). There is
        **no** δ-shift nuisance pair and **no** consistency term (see module docstring): the
        signature takes a single ``x``.

        ADAPTIVE (``cfg.adaptive_adv_weight`` True — VQGAN/MagViT "adaptive weight", Taming
        Transformers §3.3), the default::

            recon_ref = cfg.pixel_anchor_weight * pixel + cfg.fm_weight * fm
            adv       = -mean(D(recon))
            lam       = (‖∇recon_ref‖ / (‖∇adv‖ + 1e-4)).clamp(0, clamp).detach()
            adv_coeff = 0 if step < adv_warmup_steps else adversarial_weight * lam
            total     = recon_ref + entropy_weight * entropy + adv_coeff * adv

        FIXED-WEIGHT (``cfg.adaptive_adv_weight`` False)::

            total = adversarial_weight * adv
                    + pixel_anchor_weight * pixel
                    + entropy_weight * entropy

        ``last_layer`` for the adaptive balance is ``self.decoder.last_layer`` (the
        ``to_pixels`` weight); ``lam`` is DETACHED. Zero-grad cases are handled gracefully.

        Parameters
        ----------
        x : (B, C, T, H, W)
            The video window to reconstruct (also the recon target).
        disc : nn.Module
            The (external) FramePatchGAN; scored on ``recon`` for the adversarial term.
        cfg : VideoCodecConfig
        step : int
            Current global step (for ``cfg.adv_warmup_steps``). Default 0.
        frame_mask : (B, T) or (B, C, T) bool/float, optional
            Per-frame (optionally per-channel) validity mask. Off-filter tangtv cameras are
            stored as fully-NaN slabs (already zero-filled by the loader) and flagged invalid;
            when a mask is given the **pixel anchor** is computed only over valid frames so a
            dead camera does not drag the reconstruction toward zero. ``None`` = all valid.
            (The adversarial / feature-matching terms still see the whole clip — the
            discriminator is judging realism per frame; the pixel weighting is where masking
            matters most, matching the loader's "use the mask as a per-channel reconstruction
            weighting" guidance.)

        Returns
        -------
        dict with keys:
            ``total``            scalar generator loss (backprop this into the codec).
            ``adversarial``      RAW hinge generator term  [-mean(D(recon))].
            ``pixel``            raw (masked) pixel-MAE anchor.
            ``feature_matching`` raw disc feature-matching L1.
            ``entropy``          raw anti-collapse entropy term.
            ``adaptive_weight``  float(lam) — the adaptive scale used this step (1.0 in the
                                 fixed-weight path).
            ``recon``            (B, C, T, H, W) reconstruction (live graph).
            ``codes``            (B, n_tok, fsq_dim) codes for ``x`` (for gate/logging).
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]

        # ONE discriminator pass: patch scores (adversarial) + intermediate features (FM).
        fake_scores, fake_feats = disc(recon, return_features=True)
        fake_scores = _as_score_list(fake_scores)
        adversarial = -_mean_over_maps(fake_scores)  # hinge generator term: -mean(D(recon))

        pixel = self._masked_pixel_mae(recon, x, frame_mask)

        # DETACHED real features: generator matches fake -> real (grad only via fake feats).
        with torch.no_grad():
            _, real_feats = disc(x, return_features=True)
        fm = feature_matching_loss(real_feats, fake_feats)

        entropy = self.quantizer.entropy_loss(feats_x)

        # VQGAN-canonical: adaptive weight balances adv against the RECONSTRUCTION reference
        # (pixel anchor + feature-matching), NOT the entropy regularizer.
        recon_ref = cfg.pixel_anchor_weight * pixel + cfg.fm_weight * fm

        if cfg.adaptive_adv_weight:
            lam = self._adaptive_adv_weight(recon_ref, adversarial, cfg)
            adv_coeff = (
                torch.zeros((), device=recon_ref.device, dtype=recon_ref.dtype)
                if step < cfg.adv_warmup_steps
                else cfg.adversarial_weight * lam
            )
            total = recon_ref + cfg.entropy_weight * entropy + adv_coeff * adversarial
            adaptive_weight = float(lam)
        else:
            total = (
                cfg.adversarial_weight * adversarial
                + cfg.pixel_anchor_weight * pixel
                + cfg.entropy_weight * entropy
            )
            adaptive_weight = 1.0

        return {
            "total": total,
            "adversarial": adversarial,
            "pixel": pixel,
            "feature_matching": fm,
            "entropy": entropy,
            "adaptive_weight": adaptive_weight,
            "recon": recon,
            "codes": codes,
        }

    # ------------------------------------------------------------------ #
    # masked pixel anchor
    # ------------------------------------------------------------------ #
    @staticmethod
    def _masked_pixel_mae(
        recon: torch.Tensor, x: torch.Tensor, frame_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean |recon - x| over VALID positions (all positions if ``frame_mask`` is None).

        ``frame_mask`` broadcasts a per-(batch, [channel], time) validity flag over the
        (H, W) pixel grid. If the mask selects nothing (a whole batch of dead cameras) the
        anchor falls back to the unmasked MAE so the term is always finite and differentiable.
        """
        if frame_mask is None:
            return torch.mean(torch.abs(recon - x))
        # x: (B, C, T, H, W). mask given as (B, T) or (B, C, T) -> broadcast over C?,H,W.
        m = frame_mask.to(dtype=recon.dtype)
        if m.dim() == 2:            # (B, T) -> (B, 1, T, 1, 1)
            m = m[:, None, :, None, None]
        elif m.dim() == 3:          # (B, C, T) -> (B, C, T, 1, 1)
            m = m[:, :, :, None, None]
        else:
            raise ValueError(f"frame_mask must be (B,T) or (B,C,T); got {tuple(frame_mask.shape)}")
        m = m.expand_as(recon)
        denom = m.sum()
        if float(denom) <= 0.0:
            return torch.mean(torch.abs(recon - x))
        return (torch.abs(recon - x) * m).sum() / denom

    # ------------------------------------------------------------------ #
    # VQGAN adaptive adversarial weight ("Taming Transformers" §3.3)
    # ------------------------------------------------------------------ #
    def _adaptive_adv_weight(
        self,
        ref: torch.Tensor,
        adv: torch.Tensor,
        cfg: VideoCodecConfig,
    ) -> torch.Tensor:
        """lam = ‖∇ref‖ / (‖∇adv‖ + 1e-4), clamped to [0, cfg.adaptive_adv_clamp], DETACHED.

        Gradients are taken w.r.t. the decoder's last layer (``self.decoder.last_layer``).
        Identical in spirit + numerics to ``SpectroCodec._adaptive_adv_weight``.
        """
        last_layer = self.decoder.last_layer
        zero = torch.zeros((), device=last_layer.device, dtype=last_layer.dtype)

        def _grad_norm(term: torch.Tensor) -> torch.Tensor:
            if not term.requires_grad:
                return zero
            g = torch.autograd.grad(term, last_layer, retain_graph=True, allow_unused=True)[0]
            if g is None:
                return zero
            return g.norm()

        g_ref = _grad_norm(ref)
        g_adv = _grad_norm(adv)
        lam = (g_ref / (g_adv + 1e-4)).clamp(0.0, cfg.adaptive_adv_clamp)
        return lam.detach()
