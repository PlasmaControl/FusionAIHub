"""Spectrogram codec: encoder, FSQ quantizer and decoder.

Combines ``nets.SpectroEncoder`` / ``nets.SpectroDecoder`` and
``quantizer.SpectroQuantizer`` into :class:`SpectroCodec`, and provides the generator
loss for the GAN alternation. Only ``torch`` and sibling ``ignite`` modules are used
(``docs/IGNITE_DESIGN.md`` §7).

Design (§4): the encoder maps a log-power spectrogram ``(B, C, F, T)`` to continuous
features ``(B, n_tok, d_model)``; FSQ turns them into per-dimension codes (the Phase-B
tokens) and straight-through quantized features; the adversarially trained decoder maps
those back to a spectrogram.

The discriminator is not part of the codec. The trainer creates it alongside and trains
it with :func:`losses.discriminator_loss`. That way the codec's parameters are exactly
the tokenizer Phase B freezes, and the two optimizers own disjoint parameter sets.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import SpectroCodecConfig
from .losses import (
    _as_score_list,
    _mean_over_maps,
    feature_matching_loss,
    freq_gradient_loss,
    ms_ssim_loss,
    multiscale_recon_loss,
    peak_l1_loss,
    shift_consistency,
    time_smooth,
)
from .nets import (
    SpectroConvDecoder,
    SpectroDecoder,
    SpectroEncoder,
    spectro_gain_shape_split,
)
from .quantizer import SpectroQuantizer


class SpectroCodec(nn.Module):
    """Encoder, FSQ quantizer and decoder for one spectrogram modality.

    Parameters
    ----------
    cfg : SpectroCodecConfig
        Shapes and hyper-parameters.

    Notes
    -----
    ``forward(x: (B, C, F, T))`` returns a dict with keys:

    - ``recon`` (B, C, F, T): decoded spectrogram
    - ``feats`` (B, n_tok, d_model): continuous encoder features before FSQ
    - ``quant`` (B, n_tok, d_model): straight-through quantized features
    - ``codes`` (B, n_tok, fsq_dim): FSQ codes (long), the Phase-B input

    The discriminator is not a submodule; pass it to :meth:`generator_losses` and
    :func:`losses.discriminator_loss`.
    """

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = SpectroEncoder(cfg)
        self.quantizer = SpectroQuantizer(cfg)
        # cfg.decoder picks the decoder ("linear" by default). Both have the same
        # forward(quant) -> (B, C, F, T) and `last_layer`, so the losses below work with
        # either.
        if cfg.n_gain_tok > 0 and getattr(cfg, "decoder", "linear") == "conv":
            raise NotImplementedError(
                "gain_shape requires decoder='linear': SpectroConvDecoder places tokens on a "
                "(n_freq_patch, n_time_patch) grid and has no shape_unmix / gain head, so the "
                "envelope and amplitude guarantees cannot be enforced in it"
            )
        self.decoder = (
            SpectroConvDecoder(cfg) if getattr(cfg, "decoder", "linear") == "conv"
            else SpectroDecoder(cfg)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) -> continuous features before FSQ, (B, n_tok, d_model).

        The shift-consistency loss acts on these, so no discrete matching is needed
        (§4.2).
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
        # With the envelope/shape split the decoder also returns the decoded gain so it
        # can be supervised directly. Otherwise ``gain_pred`` is absent.
        if self.cfg.n_gain_tok > 0:
            recon, gain_pred = self.decoder(quant, return_aux=True)
        else:
            recon, gain_pred = self.decode(quant), None
        out = {"recon": recon, "feats": feats, "quant": quant, "codes": codes}
        if gain_pred is not None:
            out["gain_pred"] = gain_pred
        return out

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    def generator_losses(
        self,
        x: torch.Tensor,
        x_shift: torch.Tensor,
        disc: nn.Module,
        cfg: SpectroCodecConfig,
        step: int = 0,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generator loss for one codec step.

        Runs the codec on ``x`` (the target is ``x`` itself, or its time-smoothed copy,
        see ``cfg.target_time_smooth``) and encodes ``x_shift``, the δ-shifted copy, for
        the shift-consistency term on the pre-FSQ features.

        With fixed weights (``cfg.adaptive_adv_weight`` False)::

            total = cfg.adversarial_weight * adversarial
                    + cfg.pixel_anchor_weight * pixel
                    + (weighted optional reconstruction terms)
                    + cfg.consistency_weight * consistency
                    + cfg.entropy_weight * entropy

        With the adaptive weight (VQGAN/MagViT, "Taming Transformers" §3.3) the
        adversarial coefficient is rescaled each step so the adversarial gradient does
        not overpower the reconstruction::

            ref   = cfg.pixel_anchor_weight * pixel + cfg.fm_weight * fm
                    + (weighted optional reconstruction terms)
            g_ref = autograd.grad(ref, last_layer)
            g_adv = autograd.grad(adversarial, last_layer)
            lam   = (|g_ref| / (|g_adv| + 1e-4)).clamp(0, cfg.adaptive_adv_clamp)
            coeff = 0 if step < cfg.adv_warmup_steps else cfg.adversarial_weight * lam
            total = ref + cfg.consistency_weight * consistency
                    + cfg.entropy_weight * entropy + coeff * adversarial

        ``last_layer`` is ``self.decoder.last_layer``. ``lam`` is detached. A term with
        weight 0 has no gradient, so its norm is 0; the ``1e-4`` keeps ``lam`` finite.

        The entropy term pushes toward codebook diversity, which the consistency loss
        lacks: its trivial minimum is a constant encoder, i.e. one code.

        Parameters
        ----------
        x : (B, C, F, T)
            Window to reconstruct.
        x_shift : (B, C, F, T)
            δ-shifted copy of ``x`` (same modes, different realization).
        disc : nn.Module
            The external discriminator, scored on ``recon``.
        cfg : SpectroCodecConfig
        step : int
            Global training step, for ``cfg.adv_warmup_steps`` and the joint-entropy
            ramp.
        frame_mask : (B, C, T) or (B, T) bool or float, optional
            Per-(channel, STFT frame) validity mask for ``x``
            (``data.spectro_frame_mask``). ``None`` means everything is valid. Without
            it, a window with dead channels, or an all-floor fallback window from a shot
            whose HDF5 group is an empty stub, would be reconstructed, shown to D as
            real, feature-matched and counted in the entropy. The terms use it as
            follows:

            * ``pixel``: mean |recon - x| over valid (b, c, t) positions, broadcast over
              frequency (missingness has no frequency structure).
            * ``adversarial``, ``feature_matching``, ``multiscale``, ``freq_grad``,
              ``ms_ssim``: only windows in which every channel is valid (see
              ``_valid_windows``). D's channel axis is C, so it would see a dead
              channel, and a flat floor scored as real teaches D that flat spectrograms
              are realistic, which pushes toward mean collapse.
            * ``entropy``, ``consistency``: the same windows. With
              ``channel_groups == 1`` every token spans all channels, so one dead
              channel spoils every token of the window. Consistency is also trivially 0
              on a dead channel, which would deflate it.
            * ``peak`` and the gain term use the finer per-column / per-channel mask.

            When the mask is ``None`` or selects everything, the original tensors go
            through the same calls, so the result is bit-identical to the unmasked loss
            (covered by a test).

        Returns
        -------
        dict with keys:
            ``total``            scalar generator loss.
            ``adversarial``      hinge generator term, -mean(D(recon)).
            ``pixel``            unweighted pixel MAE.
            ``multiscale``, ``freq_grad``, ``ms_ssim``, ``peak``, ``gain``
                                 unweighted optional terms (0 when their weight is 0).
            ``feature_matching`` unweighted feature-matching L1; only the adaptive path
                                 adds it to the loss (via ``cfg.fm_weight``).
            ``consistency``      unweighted shift-consistency MSE.
            ``entropy``          unweighted entropy term.
            ``adaptive_weight``  float(lam) used this step (1.0 with fixed weights).
            ``recon``            (B, C, F, T) reconstruction, with graph.
            ``codes``            (B, n_tok, fsq_dim) codes of ``x``.
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]

        # Reconstruction target. The encoder saw the raw window, so the codes are as at
        # inference; only the decoder's target changes. With target_time_smooth = K it
        # is the K-frame time boxcar, the predictable part: lag-1 autocorrelation of the
        # truth is only 0.29-0.68 by modality, so a raw target is ~60-70% unpredictable
        # speckle and L-p terms answer it with a low-amplitude blur. time_smooth returns
        # x itself for K <= 1.
        x_tgt = time_smooth(x, int(getattr(cfg, "target_time_smooth", 0) or 0))

        # Direct supervision of the decoded gain (envelope/shape split only). The
        # reconstruction term sees level and sigma only mixed with the shape error. The
        # gain head alone sets the reconstruction's per-(channel, frequency) time mean
        # and temporal std, so it gets its own L1 target.
        if "gain_pred" in out and float(getattr(cfg, "gain_weight", 0.0)) > 0.0:
            gain_tgt = spectro_gain_shape_split(x_tgt, cfg.uses_gain_scale)[0]
            gain = self._masked_gain_mae(out["gain_pred"], gain_tgt, frame_mask)
        else:
            gain = recon.new_zeros(())

        # `win` is the (B,) selector of fully valid windows; None means all are valid
        # and the original tensors are used.
        win = self._valid_windows(frame_mask, recon.shape)
        r_sel = recon if win is None else recon[win]
        x_sel = x_tgt if win is None else x_tgt[win]

        # One discriminator pass gives both the scores (adversarial term) and the
        # intermediate features (feature matching), saving the second disc(recon) call
        # that recon_objective would make.
        fake_scores, fake_feats = disc(r_sel, return_features=True)
        fake_scores = _as_score_list(fake_scores)
        adversarial = -_mean_over_maps(fake_scores)  # hinge generator term

        pixel = self._masked_pixel_mae(recon, x_tgt, frame_mask)
        # Amplitude ratio, off (0.0) by default.
        _stdw = float(getattr(cfg, "std_weight", 0.0) or 0.0)
        std_ratio_l = (self._masked_std_ratio_loss(recon, x_tgt, frame_mask)
                       if _stdw > 0.0 else recon.new_zeros(()))
        # Peak-weighted L1. It uses the full tensors with the per-(channel, frame) mask
        # instead of the window selector: the weighting is per (b, c, t) column, and a
        # partly dead window still has valid columns.
        _pkw = float(getattr(cfg, "peak_weight", 0.0))
        peak = (peak_l1_loss(recon, x_tgt, frame_mask) if _pkw > 0
                else recon.new_zeros(()))
        # Multi-resolution and frequency-gradient terms (as in NeMo audio codecs),
        # against the blur of plain pixel L1. Skipped when their weights are 0.
        multiscale = (
            multiscale_recon_loss(
                r_sel, x_sel,
                scales=tuple(getattr(cfg, "multiscale_recon_scales", (2, 4))),
            )
            if cfg.multiscale_recon_weight > 0 else recon.new_zeros(())
        )
        freq_grad = (freq_gradient_loss(r_sel, x_sel)
                     if cfg.freq_grad_weight > 0 else recon.new_zeros(()))
        # 1 - MS-SSIM, the differentiable form of the gate metric.
        _msw = float(getattr(cfg, "ms_ssim_weight", 0.0))
        ms_ssim_t = (
            ms_ssim_loss(r_sel, x_sel, win=int(getattr(cfg, "ms_ssim_win", 7)),
                         scales=tuple(getattr(cfg, "ms_ssim_scales", (1, 2, 4))))
            if _msw > 0 else recon.new_zeros(())
        )

        # Real features are a detached target; the gradient flows through the fake ones.
        with torch.no_grad():
            _, real_feats = disc(x_sel, return_features=True)
        fm = feature_matching_loss(real_feats, fake_feats)

        feats_shift = self.encode(x_shift)
        # Consistency and entropy over the same valid windows.
        f_x = feats_x if win is None else feats_x[win]
        f_s = feats_shift if win is None else feats_shift[win]
        consistency = shift_consistency(f_x, f_s)

        # Only the spectro codec passes step, for the joint-entropy ramp
        # (cfg.joint_entropy_ramp_steps).
        entropy = self.quantizer.entropy_loss(f_x, step=step)

        # As in VQGAN (Taming Transformers §3.3), the adaptive weight balances the
        # adversarial term against the reconstruction terms only, not the consistency
        # and entropy regularizers. Including those made lam ~1%, too little GAN
        # pressure and a blurred decode. The reference also includes feature matching
        # (HiFi-GAN/MelGAN, the spectrogram counterpart of Genie's perceptual loss),
        # which raises lam and so the adversarial sharpening.
        recon_ref = (cfg.pixel_anchor_weight * pixel + cfg.fm_weight * fm
                     + cfg.multiscale_recon_weight * multiscale
                     + cfg.freq_grad_weight * freq_grad
                     + _msw * ms_ssim_t
                     + _pkw * peak
                     + _stdw * std_ratio_l
                     + float(getattr(cfg, "gain_weight", 0.0)) * gain)
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
            # Fixed weights. Feature matching is left out here; it only enters through
            # the adaptive balance.
            total = (
                cfg.adversarial_weight * adversarial
                + cfg.pixel_anchor_weight * pixel
                + cfg.multiscale_recon_weight * multiscale
                + cfg.freq_grad_weight * freq_grad
                + _msw * ms_ssim_t
                + _pkw * peak
                + _stdw * std_ratio_l
                + float(getattr(cfg, "gain_weight", 0.0)) * gain
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
            "ms_ssim": ms_ssim_t,
            "peak": peak,
            "gain": gain,
            "feature_matching": fm,
            "consistency": consistency,
            "entropy": entropy,
            "adaptive_weight": adaptive_weight,
            "recon": recon,
            "codes": codes,
        }

    # Missing-data selectors, shared with the trainer's discriminator step. Like
    # VideoCodec's, each returns None when the mask is absent or selects everything, and
    # callers then use the original tensor, so the unmasked path stays bit-identical.
    @staticmethod
    def _frame_validity(
        frame_mask: Optional[torch.Tensor], shape
    ) -> Optional[torch.Tensor]:
        """``(B, T)`` bool, True where every channel of the STFT frame is valid.

        Returns None if everything is valid. ``shape`` is the ``(B, C, F, T)`` shape. A
        ``(B, C, T)`` mask is reduced over C with AND, because the discriminator's input
        channel axis is C and a dead channel cannot be hidden from it.
        """
        if frame_mask is None:
            return None
        B, C, _F, T = shape
        m = frame_mask
        if m.dim() == 3:
            v = (m > 0.5).all(dim=1)                 # (B, C, T) -> (B, T)
        elif m.dim() == 2:
            v = m > 0.5
        else:
            raise ValueError(f"frame_mask must be (B,T) or (B,C,T); got {tuple(m.shape)}")
        if tuple(v.shape) != (B, T):
            raise ValueError(f"frame_mask implies {tuple(v.shape)}, expected {(B, T)}")
        return None if bool(v.all()) else v

    @classmethod
    def _valid_windows(
        cls, frame_mask: Optional[torch.Tensor], shape
    ) -> Optional[torch.Tensor]:
        """``(B,)`` bool, True for fully valid windows; None when all are valid.

        Window granularity rather than per frame (as in the video codec), because:

        * The discriminator, the multi-scale term and MS-SSIM convolve and pool over the
          ``(F, T)`` plane. A subset of frames could leave a ``T = 1`` plane they cannot
          take; dropping whole windows keeps the trained geometry.
        * It loses almost nothing. Spectro missingness is per (shot, channel): a channel
          that is off is off for all 96 frames of every window. The only partial case, a
          window across the record edge, is already rejected by the ``valid_len`` and
          global-std checks.
        * With ``channel_groups == 1`` every token spans all channels, so any dead
          channel spoils all tokens of the window anyway.

        Returns None rather than an empty selection when nothing survives, so the caller
        falls back to the unmasked tensors and the loss stays finite.
        """
        v = cls._frame_validity(frame_mask, shape)
        if v is None:
            return None
        w = v.all(dim=1)                             # (B, T) -> (B,)
        return None if int(w.sum()) == 0 else w

    @staticmethod
    def _masked_pixel_mae(
        recon: torch.Tensor, x: torch.Tensor, frame_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean |recon - x| over valid positions (all if ``frame_mask`` is None).

        The per-(batch, [channel], frame) mask is broadcast over frequency. If it
        selects nothing, the unmasked MAE is returned so the term stays finite.
        """
        if frame_mask is None:
            return torch.mean(torch.abs(recon - x))
        m = frame_mask.to(dtype=recon.dtype)
        if m.dim() == 3:                             # (B, C, T) -> (B, C, 1, T)
            m = m.unsqueeze(2)
        elif m.dim() == 2:                           # (B, T)    -> (B, 1, 1, T)
            m = m[:, None, None, :]
        else:
            raise ValueError(f"frame_mask must be (B,T) or (B,C,T); got {tuple(frame_mask.shape)}")
        denom = m.expand_as(recon).sum()
        if float(denom) <= 0.0:
            return torch.mean(torch.abs(recon - x))
        return (torch.abs(recon - x) * m).sum() / denom

    @classmethod
    def _masked_std_ratio_loss(
        cls,
        recon: torch.Tensor,
        target: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """|log std(recon) - log std(target)| per (window, channel), over valid windows.

        Targets the audit metric ``analysis.spectro_final_fig.masked_std_ratio``, the
        mean over (window, channel) of ``std(recon) / std(target)``, ideally 1.0. It
        exposes amplitude collapse, which nRMSE cannot: the conditional mean minimises
        nRMSE, so shrinking toward the mean improves nRMSE while losing the signal.

        Why it exists: ece stayed at std ratio 0.232-0.267 across six codecs (patch
        geometries 16x16 and 8x32, learning rates 2e-4 and 1e-4, with and without EMA)
        and 0.12-0.32 over 24 earlier checkpoints. ``pixel_anchor_weight`` matches
        pixels and is already at its best value (5.0; 20 collapses to one code); nothing
        else in the objective targets the amplitude ratio.

        The log form penalises under- and overshoot alike; a plain ratio is bounded on
        one side only and so favours overshoot. std is over the whole (F, T) plane per
        (window, channel), as in the metric, and the mask is applied per window for the
        reason in :meth:`_valid_windows`.

        Returns a scalar. Only called when ``std_weight > 0``.
        """
        win = cls._valid_windows(frame_mask, recon.shape)
        r = recon if win is None else recon[win]
        t = target if win is None else target[win]
        if r.numel() == 0:
            return recon.new_zeros(())
        B, C = r.shape[0], r.shape[1]
        rs = r.reshape(B, C, -1).std(dim=-1)
        ts = t.reshape(B, C, -1).std(dim=-1)
        eps = 1e-6
        return (torch.log(rs + eps) - torch.log(ts + eps)).abs().mean()

    @staticmethod
    def _masked_gain_mae(
        pred: torch.Tensor, tgt: torch.Tensor, frame_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean |pred - tgt| over (window, channel) pairs whose whole time axis is real.

        ``pred`` and ``tgt`` are ``(B, G, F)`` with ``G`` = C (level only) or 2C (level
        and log1p sigma stacked on the channel axis, hence the ``repeat``). The gain
        target is a statistic over the whole time axis, so a partly missing axis can
        only be excluded. Selection is per (window, channel), finer than the window rule
        elsewhere, since the gain path does not mix channels.

        Falls back to the unmasked MAE when the mask is absent or selects nothing; with
        every frame valid it is bit-identical to the unmasked MAE.
        """
        if frame_mask is None:
            return torch.mean(torch.abs(pred - tgt))
        m = frame_mask > 0.5
        if m.dim() == 2:                                  # (B, T): all channels share it
            ok = m.all(dim=-1, keepdim=True)              # (B, 1)
        elif m.dim() == 3:                                # (B, C, T)
            ok = m.all(dim=-1)                            # (B, C)
        else:
            raise ValueError(f"frame_mask must be (B,T) or (B,C,T); got {tuple(frame_mask.shape)}")
        if ok.shape[1] != tgt.shape[1]:
            if tgt.shape[1] % ok.shape[1] != 0:
                raise ValueError(
                    f"gain target has {tgt.shape[1]} rows, not a multiple of the mask's "
                    f"{ok.shape[1]} channels"
                )
            ok = ok.repeat(1, tgt.shape[1] // ok.shape[1])
        w = ok.to(dtype=pred.dtype).unsqueeze(-1)         # (B, G, 1), broadcasts over F
        denom = w.expand_as(pred).sum()
        if float(denom) <= 0.0:
            return torch.mean(torch.abs(pred - tgt))
        return (torch.abs(pred - tgt) * w).sum() / denom

    # VQGAN adaptive adversarial weight ("Taming Transformers" §3.3)
    def _adaptive_adv_weight(
        self,
        ref: torch.Tensor,
        adv: torch.Tensor,
        cfg: SpectroCodecConfig,
    ) -> torch.Tensor:
        """lam = |grad ref| / (|grad adv| + 1e-4), clamped to [0, adaptive_adv_clamp].

        Gradients are taken with respect to ``self.decoder.last_layer``, where the
        reconstruction and adversarial signals are balanced. Both use
        ``retain_graph=True`` so ``total.backward()`` still has the graph, and
        ``allow_unused=True`` for a term with weight 0 (its gradient counts as zero).
        Returns a detached 0-dim tensor.
        """
        last_layer = self.decoder.last_layer
        zero = torch.zeros((), device=last_layer.device, dtype=last_layer.dtype)

        def _grad_norm(term: torch.Tensor) -> torch.Tensor:
            if not term.requires_grad:
                # Constant with respect to last_layer (e.g. weighted by 0).
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
