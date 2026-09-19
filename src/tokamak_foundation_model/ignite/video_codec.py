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
    # input standardization (the SCALE FIX — mirrors the FM model)
    # ------------------------------------------------------------------ #
    @staticmethod
    def standardize_input(x: torch.Tensor) -> torch.Tensor:
        """Per-(B, C) z-score of a video window over (T, H, W) — the codec-input SCALE FIX.

        The tangtv frames arrive as RAW camera pixels (measured std ~5.2, range ~[16, 240] —
        NOT O(1)); feeding those straight into the patchify→Linear encoder is a scale leak that
        the FM model never sees. The FM standardizes video PER-(B, C) over (T, H, W) with
        ``sd.clamp(min=1.0)`` (``e2e.multimodal.video_standardize_per_bc`` /
        ``train_e2e_stage1.py``), so the codec mirrors that EXACTLY here. Applied inside
        ``encode`` / ``forward`` / ``generator_losses`` so EVERY path (training, gate stability
        nuisance, gate forecast sequence) sees the SAME O(1) input and the reconstruction target
        is the SAME standardized frames as the reconstruction (a consistent loss).

        ``sd.clamp(min=1.0)`` keeps an off / dead camera (a zero-filled channel) finite and maps
        it to ~0 (standardized-mean / neutral), never a large artifact. Reuses NO FAITH model
        code — this is the same arithmetic, re-implemented from the data-pipeline mechanism.

        Parameters
        ----------
        x : (B, C, T, H, W)

        Returns
        -------
        (B, C, T, H, W) standardized frames (per-(B, C) zero-mean, unit-ish-std).
        """
        if x.dim() != 5:
            raise ValueError(f"video input must be (B, C, T, H, W); got {tuple(x.shape)}")
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        sd = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
        return (x - mu) / sd

    # ------------------------------------------------------------------ #
    # forward paths
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> PRE-FSQ continuous features (B, n_tok, d_model).

        Standardizes the raw frames per-(B, C) (see :meth:`standardize_input`) BEFORE the
        patchify encoder so the encoder input is O(1) (the SCALE FIX). Callers that pre-encode
        a nuisance (the gate) go through here too, so they are standardized identically.
        """
        return self.encoder(self.standardize_input(x))

    def quantize(self, feats: torch.Tensor):
        """Features -> (quant (B,n_tok,d_model) float, codes (B,n_tok,fsq_dim) long)."""
        return self.quantizer.quantize(feats)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Quantized features (B, n_tok, d_model) -> reconstruction (B, C, T, H, W)."""
        return self.decoder(quant)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the codec; ``recon`` is in the STANDARDIZED frame space (see :meth:`encode`).

        Also returns ``x_std`` — the standardized input — so the generator loss compares
        ``recon`` against the SAME standardized frames it was trained to produce (a consistent
        reconstruction target; without this the target would be raw-scale and the recon
        standardized-scale).
        """
        x_std = self.standardize_input(x)
        feats = self.encoder(x_std)
        quant, codes = self.quantize(feats)
        recon = self.decode(quant)
        return {"recon": recon, "feats": feats, "quant": quant, "codes": codes, "x_std": x_std}

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
            stored as fully-NaN slabs (already zero-filled by the loader) and flagged invalid.
            ``None`` = all valid.

            AS OF 2026-09-03 THE MASK IS HONOURED BY EVERY TERM, not just the pixel anchor:

              * ``pixel``            — mean |recon - x| over VALID (b, c, t) positions
                                       (unchanged behaviour).
              * ``adversarial`` + ``feature_matching`` — the discriminator is run on the
                                       SUBSET of frames in which EVERY channel is valid. A
                                       half-dead frame is not a realistic camera image, and a
                                       fully dead one is a constant-zero plate: feeding those
                                       to D as "real" teaches it that a flat frame IS realistic,
                                       which is a direct gradient toward the mean-collapsed
                                       generator this codec is supposed to avoid.
              * ``entropy``          — the FSQ statistic is accumulated over valid CLIPS only.
                                       Tokens mix both channels (patch_dim = C*pt*ph*pw), so a
                                       clip with a dead channel has every one of its 108 tokens
                                       contaminated; there is no per-token repair, only
                                       exclusion.

            WHY IT MATTERS (measured over all 8753 shots, video_channel_liveness.pt):
            51.30% of tangtv_lower shots and 68.58% of tangtv_upper shots have NO live camera
            at all, and among the shots that do, a further 19.0% / 16.9% of channel-slots are
            dead. Only 39.4% (lower) / 26.1% (upper) of the streamed tensor is real data.

            NO-OP GUARANTEE: when the mask is ``None`` OR selects every frame, the code takes
            the ORIGINAL tensors through the identical call sequence, so the default path is
            bit-identical to the pre-2026-09-03 loss (verified by test).

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
        # The reconstruction target is the STANDARDIZED input (see forward / standardize_input):
        # recon lives in standardized frame space, so the pixel anchor + the discriminator's
        # "real" frames MUST be the same standardized frames — comparing to the raw-scale x would
        # make the loss (and the adaptive-adv gradient balance) inconsistent.
        x_std = out["x_std"]

        # MISSING-DATA EXCLUSION. `keep` is the (B, T) "every channel valid" frame selector;
        # None means "everything valid", in which case the ORIGINAL tensors are used and the
        # whole path below is bit-identical to the unmasked loss.
        keep = self._disc_frame_selector(frame_mask, recon.shape)
        d_fake = recon if keep is None else self._select_frames(recon, keep)
        d_real = x_std if keep is None else self._select_frames(x_std, keep)

        # ONE discriminator pass: patch scores (adversarial) + intermediate features (FM).
        fake_scores, fake_feats = disc(d_fake, return_features=True)
        fake_scores = _as_score_list(fake_scores)
        adversarial = -_mean_over_maps(fake_scores)  # hinge generator term: -mean(D(recon))

        pixel = self._masked_pixel_mae(recon, x_std, frame_mask)

        # DETACHED real features: generator matches fake -> real (grad only via fake feats).
        with torch.no_grad():
            _, real_feats = disc(d_real, return_features=True)
        fm = feature_matching_loss(real_feats, fake_feats)

        # ENTROPY over valid CLIPS only (a clip with a dead channel contaminates all its
        # tokens). `_valid_clips` returns None when every clip is usable -> feats_x unchanged.
        clips = self._valid_clips(frame_mask, recon.shape)
        entropy = self.quantizer.entropy_loss(
            feats_x if clips is None else feats_x[clips], step=step
        )

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
    # missing-data selectors (shared with the trainer's discriminator step)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _frame_validity(
        frame_mask: Optional[torch.Tensor], shape
    ) -> Optional[torch.Tensor]:
        """``(B, T)`` bool "every channel of this frame is valid", or None if all valid.

        ``shape`` is the ``(B, C, T, H, W)`` video shape. A ``(B, T)`` mask is already
        per-frame; a ``(B, C, T)`` mask is reduced over C with AND, because the
        discriminator's input channel axis IS C — there is no way to hide one dead camera
        from a conv that consumes both. Returns ``None`` when nothing would be excluded, so
        every caller can short-circuit to the original tensors (the bit-identical path).
        """
        if frame_mask is None:
            return None
        B, C, T = shape[0], shape[1], shape[2]
        m = frame_mask
        if m.dim() == 2:            # (B, T)
            v = m > 0.5
        elif m.dim() == 3:          # (B, C, T) -> AND over channels
            v = (m > 0.5).all(dim=1)
        else:
            raise ValueError(f"frame_mask must be (B,T) or (B,C,T); got {tuple(m.shape)}")
        if v.shape != (B, T):
            raise ValueError(f"frame_mask implies {tuple(v.shape)}, expected {(B, T)}")
        if bool(v.all()):
            return None             # nothing excluded -> caller keeps the original tensors
        return v

    @classmethod
    def _disc_frame_selector(
        cls, frame_mask: Optional[torch.Tensor], shape
    ) -> Optional[torch.Tensor]:
        """:meth:`_frame_validity`, but also None when the mask would leave NO frame.

        A batch in which every frame is invalid (a whole batch of dead cameras) would give the
        discriminator an empty input; the term then has no gradient and the hinge mean is NaN.
        In that degenerate case we fall back to the unmasked tensors, exactly as
        :meth:`_masked_pixel_mae` falls back to the unmasked MAE.
        """
        v = cls._frame_validity(frame_mask, shape)
        if v is None or not bool(v.any()):
            return None
        return v

    @staticmethod
    def _select_frames(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Gather the ``keep`` (B, T) frames of ``(B, C, T, H, W)`` into ``(1, C, N, H, W)``.

        The FramePatchGAN reshapes ``(B, C, T, H, W) -> (B*T, C, H, W)`` and scores every frame
        independently, so packing the surviving frames onto a single clip's time axis gives the
        discriminator EXACTLY the valid frames and nothing else. Order is (batch, time)
        row-major, matching the discriminator's own rearrange.
        """
        # (B, C, T, H, W) -> (B, T, C, H, W) -> select -> (N, C, H, W) -> (1, C, N, H, W)
        sel = x.permute(0, 2, 1, 3, 4)[keep]          # (N, C, H, W)
        return sel.permute(1, 0, 2, 3).unsqueeze(0)   # (1, C, N, H, W)

    @classmethod
    def _valid_clips(
        cls, frame_mask: Optional[torch.Tensor], shape
    ) -> Optional[torch.Tensor]:
        """``(B,)`` bool "every (channel, frame) of this clip is valid", or None if all are.

        Used to restrict the FSQ entropy statistic. Returns None when nothing would be
        excluded AND when the selection would be empty (keeping the term finite and, under
        DDP, keeping every rank's collective call shape well-defined).
        """
        v = cls._frame_validity(frame_mask, shape)
        if v is None:
            return None
        clips = v.all(dim=1)                          # (B,)
        if bool(clips.all()) or not bool(clips.any()):
            return None
        return clips

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
