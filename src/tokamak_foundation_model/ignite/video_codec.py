"""tangtv video codec: encoder, FSQ quantizer and adversarially trained decoder.

Same structure as :class:`~ignite.codec.SpectroCodec` and the same generator loss
(adversarial term, small pixel anchor, feature matching and entropy, balanced by the
VQGAN adaptive weight at the decoder's last layer), for video windows
``(B, C, T, H, W)``. The one difference is that there is no shift-consistency term.

The spectrogram codec needs a δ-shift consistency loss because a spectrogram carries an
STFT-phase / speckle realization (a 0.5 ms sub-window shift changes ~74% of the codes)
that has to be projected out so the codes keep only the statistic (docs/IGNITE_DESIGN.md
§4.1-§4.2). A divertor camera frame has no such realization; it is smooth video, as in
Genie, and §4.3 scopes the video codec to dropping strong pixel MSE in favour of a
generative decoder. The low-weight pixel anchor with the adversarial and
feature-matching terms prevents the mean collapse a strong pixel MSE would cause. So
``generator_losses`` takes a single ``x`` and an optional per-frame validity mask.

The FSQ bottleneck (``video_quantizer``, the spectro machinery), the adaptive weight,
the entropy regularizer and feature matching (``losses``) are shared with the spectro
codec. No FAITH model code is used (§7).
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
    """Encoder, FSQ quantizer and decoder for one tangtv divertor camera.

    ``forward(x: (B, C, T, H, W))`` returns a dict with keys:

    - ``recon`` (B, C, T, H, W): decoded video, in standardized frame space
    - ``feats`` (B, n_tok, d_model): continuous encoder features before FSQ
    - ``quant`` (B, n_tok, d_model): straight-through quantized features
    - ``codes`` (B, n_tok, fsq_dim): FSQ codes (long), the Phase-B input
    - ``x_std``: the standardized input

    As for the spectro codec, the discriminator (``video_discriminator.FramePatchGAN``)
    is created by the trainer outside the codec, so the codec's parameters are exactly
    the tokenizer Phase B freezes and the two optimizers own disjoint parameters.
    """

    def __init__(self, cfg: VideoCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = VideoEncoder(cfg)
        self.quantizer = VideoQuantizer(cfg)
        self.decoder = VideoDecoder(cfg)

    # Input standardization
    @staticmethod
    def standardize_input(x: torch.Tensor) -> torch.Tensor:
        """Per-(B, C) z-score of a video window over (T, H, W).

        tangtv frames arrive as raw camera pixels (std ~5.2, range ~[16, 240]), not
        O(1). The foundation model standardizes video per (B, C) over (T, H, W) with
        ``sd.clamp(min=1.0)`` (``e2e.multimodal.video_standardize_per_bc``), and the
        codec does the same. It is applied in ``encode``, ``forward`` and
        ``generator_losses``, so training and the gate's stability and forecast checks
        all see the same input, and the reconstruction target is the same standardized
        frames.

        ``sd.clamp(min=1.0)`` keeps an off or dead (zero-filled) camera finite and maps
        it to ~0. The arithmetic is re-implemented here, not imported from FAITH.

        Parameters
        ----------
        x : (B, C, T, H, W)

        Returns
        -------
        (B, C, T, H, W) frames with zero mean and roughly unit std per (B, C).
        """
        if x.dim() != 5:
            raise ValueError(f"video input must be (B, C, T, H, W); got {tuple(x.shape)}")
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        sd = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
        return (x - mu) / sd

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> continuous features before FSQ, (B, n_tok, d_model).

        The frames are standardized first (:meth:`standardize_input`). Callers that
        encode a nuisance copy (the gate) go through here too.
        """
        return self.encoder(self.standardize_input(x))

    def quantize(self, feats: torch.Tensor):
        """Features -> (quant (B,n_tok,d_model) float, codes (B,n_tok,fsq_dim) long)."""
        return self.quantizer.quantize(feats)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Quantized features (B, n_tok, d_model) -> reconstruction (B, C, T, H, W)."""
        return self.decoder(quant)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the codec. ``recon`` is in standardized frame space.

        Also returns ``x_std``, the standardized input, so the generator loss compares
        ``recon`` with the frames it is trained to produce rather than raw-scale ones.
        """
        x_std = self.standardize_input(x)
        feats = self.encoder(x_std)
        quant, codes = self.quantize(feats)
        recon = self.decode(quant)
        return {"recon": recon, "feats": feats, "quant": quant, "codes": codes, "x_std": x_std}

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    def generator_losses(
        self,
        x: torch.Tensor,
        disc: nn.Module,
        cfg: VideoCodecConfig,
        step: int = 0,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generator loss for one video-codec step.

        Runs the codec on ``x``; the target is the standardized ``x``. There is no
        δ-shift pair and no consistency term (see the module docstring).

        With the adaptive weight (``cfg.adaptive_adv_weight``, the default;
        VQGAN/MagViT, Taming Transformers §3.3)::

            ref   = cfg.pixel_anchor_weight * pixel + cfg.fm_weight * fm
            adv   = -mean(D(recon))
            lam   = (|grad ref| / (|grad adv| + 1e-4)).clamp(0, clamp).detach()
            coeff = 0 if step < adv_warmup_steps else adversarial_weight * lam
            total = ref + entropy_weight * entropy + coeff * adv

        With fixed weights::

            total = adversarial_weight * adv
                    + pixel_anchor_weight * pixel
                    + entropy_weight * entropy

        Gradients for ``lam`` are taken at ``self.decoder.last_layer``.

        Parameters
        ----------
        x : (B, C, T, H, W)
            Video window to reconstruct.
        disc : nn.Module
            The external FramePatchGAN, scored on ``recon``.
        cfg : VideoCodecConfig
        step : int
            Global step, for ``cfg.adv_warmup_steps``.
        frame_mask : (B, T) or (B, C, T) bool or float, optional
            Per-frame (optionally per-channel) validity mask. Cameras with the filter
            off are stored as all-NaN slabs, zero-filled by the loader, and flagged
            invalid. ``None`` means all valid. The terms use it as follows:

            * ``pixel``: mean |recon - x| over valid (b, c, t) positions.
            * ``adversarial``, ``feature_matching``: the discriminator sees only frames
              in which every channel is valid. A half-dead frame is not a realistic
              camera image, and a dead one is a constant plate; shown to D as real they
              would teach it that flat frames are realistic, pushing toward mean
              collapse.
            * ``entropy``: valid clips only. Tokens mix both channels
              (patch_dim = C*pt*ph*pw), so a dead channel spoils all 108 tokens of the
              clip.

            Over all 8753 shots (video_channel_liveness.pt), 51.30% of tangtv_lower and
            68.58% of tangtv_upper shots have no live camera, and in the rest 19.0% /
            16.9% of channel slots are dead: only 39.4% (lower) / 26.1% (upper) of the
            streamed tensor is real data.

            When the mask is ``None`` or selects every frame, the original tensors go
            through the same calls, so the result is bit-identical to the unmasked loss
            (covered by a test).

        Returns
        -------
        dict with keys:
            ``total``            scalar generator loss.
            ``adversarial``      hinge generator term, -mean(D(recon)).
            ``pixel``            masked pixel MAE.
            ``feature_matching`` feature-matching L1.
            ``entropy``          entropy term.
            ``adaptive_weight``  float(lam) used this step (1.0 with fixed weights).
            ``recon``            (B, C, T, H, W) reconstruction, with graph.
            ``codes``            (B, n_tok, fsq_dim) codes of ``x``.
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]
        # recon is in standardized frame space, so the pixel anchor and the
        # discriminator's real frames use the standardized input too.
        x_std = out["x_std"]

        # `keep` is the (B, T) selector of frames with every channel valid; None means
        # all are valid and the original tensors are used.
        keep = self._disc_frame_selector(frame_mask, recon.shape)
        d_fake = recon if keep is None else self._select_frames(recon, keep)
        d_real = x_std if keep is None else self._select_frames(x_std, keep)

        # One discriminator pass gives the scores and the features for matching.
        fake_scores, fake_feats = disc(d_fake, return_features=True)
        fake_scores = _as_score_list(fake_scores)
        adversarial = -_mean_over_maps(fake_scores)  # hinge generator term

        pixel = self._masked_pixel_mae(recon, x_std, frame_mask)

        # Real features are a detached target; the gradient flows through the fake ones.
        with torch.no_grad():
            _, real_feats = disc(d_real, return_features=True)
        fm = feature_matching_loss(real_feats, fake_feats)

        # Entropy over valid clips only; `_valid_clips` is None when all are usable.
        clips = self._valid_clips(frame_mask, recon.shape)
        entropy = self.quantizer.entropy_loss(
            feats_x if clips is None else feats_x[clips], step=step
        )

        # As in VQGAN, the adaptive weight balances the adversarial term against the
        # reconstruction terms (pixel anchor and feature matching), not the entropy.
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

    # Missing-data selectors, shared with the trainer's discriminator step
    @staticmethod
    def _frame_validity(
        frame_mask: Optional[torch.Tensor], shape
    ) -> Optional[torch.Tensor]:
        """``(B, T)`` bool, True where every channel of the frame is valid.

        ``shape`` is the ``(B, C, T, H, W)`` video shape. A ``(B, C, T)`` mask is
        reduced over C with AND, because the discriminator's conv takes both cameras as
        input channels and cannot ignore a dead one. Returns ``None`` when nothing would
        be excluded, so callers can use the original tensors.
        """
        if frame_mask is None:
            return None
        B, C, T = shape[0], shape[1], shape[2]
        m = frame_mask
        if m.dim() == 2:            # (B, T)
            v = m > 0.5
        elif m.dim() == 3:          # (B, C, T), AND over channels
            v = (m > 0.5).all(dim=1)
        else:
            raise ValueError(f"frame_mask must be (B,T) or (B,C,T); got {tuple(m.shape)}")
        if v.shape != (B, T):
            raise ValueError(f"frame_mask implies {tuple(v.shape)}, expected {(B, T)}")
        if bool(v.all()):
            return None             # nothing excluded
        return v

    @classmethod
    def _disc_frame_selector(
        cls, frame_mask: Optional[torch.Tensor], shape
    ) -> Optional[torch.Tensor]:
        """:meth:`_frame_validity`, but also None when no frame would be left.

        If every frame in the batch is invalid, the discriminator would get an empty
        input and the hinge mean would be NaN, so the unmasked tensors are used instead,
        as :meth:`_masked_pixel_mae` falls back to the unmasked MAE.
        """
        v = cls._frame_validity(frame_mask, shape)
        if v is None or not bool(v.any()):
            return None
        return v

    @staticmethod
    def _select_frames(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Gather the ``keep`` (B, T) frames of ``(B, C, T, H, W)`` into ``(1, C, N, H, W)``.

        FramePatchGAN reshapes to ``(B*T, C, H, W)`` and scores each frame on its own,
        so packing the kept frames onto one clip's time axis gives it exactly the valid
        frames. Order is (batch, time), matching the discriminator's rearrange.
        """
        # (B, C, T, H, W) -> (B, T, C, H, W) -> (N, C, H, W) -> (1, C, N, H, W)
        sel = x.permute(0, 2, 1, 3, 4)[keep]          # (N, C, H, W)
        return sel.permute(1, 0, 2, 3).unsqueeze(0)   # (1, C, N, H, W)

    @classmethod
    def _valid_clips(
        cls, frame_mask: Optional[torch.Tensor], shape
    ) -> Optional[torch.Tensor]:
        """``(B,)`` bool, True for clips whose every (channel, frame) is valid.

        Restricts the FSQ entropy statistic. Returns None when nothing would be excluded
        and when nothing would be left, which keeps the term finite and the collective
        call shapes the same on every DDP rank.
        """
        v = cls._frame_validity(frame_mask, shape)
        if v is None:
            return None
        clips = v.all(dim=1)                          # (B,)
        if bool(clips.all()) or not bool(clips.any()):
            return None
        return clips

    @staticmethod
    def _masked_pixel_mae(
        recon: torch.Tensor, x: torch.Tensor, frame_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean |recon - x| over valid positions (all if ``frame_mask`` is None).

        The per-(batch, [channel], time) mask is broadcast over the (H, W) grid. If it
        selects nothing, the unmasked MAE is returned so the term stays finite.
        """
        if frame_mask is None:
            return torch.mean(torch.abs(recon - x))
        # x is (B, C, T, H, W); the mask is (B, T) or (B, C, T).
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

    # VQGAN adaptive adversarial weight ("Taming Transformers" §3.3)
    def _adaptive_adv_weight(
        self,
        ref: torch.Tensor,
        adv: torch.Tensor,
        cfg: VideoCodecConfig,
    ) -> torch.Tensor:
        """lam = |grad ref| / (|grad adv| + 1e-4), clamped to [0, adaptive_adv_clamp].

        Gradients are taken at ``self.decoder.last_layer``. Same computation as
        ``SpectroCodec._adaptive_adv_weight``; returns a detached 0-dim tensor.
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
