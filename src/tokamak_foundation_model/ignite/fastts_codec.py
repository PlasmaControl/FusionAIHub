"""IGNITE Phase-A fast-TS (filterscopes) codec: encoder, FSQ and decoder.

``cfg.target`` selects the mode (see ``config.FastTSCodecConfig``). Envelope mode (the
default) models a 5-bin ELM activity envelope. In raw mode the input and the
reconstruction target are the 10 kHz waveform ``(B, C, W)`` (C = 8 channels, W = 500
samples in a 50 ms frame).

Objective in raw mode
---------------------
The envelope codec weights the pixel anchor at 0.05 against adversarial 1.0. For a
sample-wise target that is backwards: an adversarial decoder adds plausible
high-frequency detail that is uncorrelated with the true samples, which raises nRMSE.
``config.fastts_raw_config`` therefore sets ``pixel_anchor_weight = 1.0``,
``adversarial_weight = 0.0`` and ``fm_weight = 0.0``: reconstruction plus the FSQ
entropy term. The adversarial path stays available (``--adversarial_weight``).

Delta-shift consistency in raw mode
-----------------------------------
The envelope codec penalizes ``|enc(x) - enc(x_shift)|^2`` to make codes invariant to
sub-bin spike timing. Sample-wise reconstruction needs codes that carry that timing, so
raw mode sets ``consistency_weight = 0.0``, and the second encoder pass is then skipped.
The loader still builds the shifted pair because the gate's ``stability`` probe reports
it; expect it to be low (``spike.gate_score`` does not read it).

The FSQ bottleneck (:class:`~ignite.fastts_quantizer.FastTSQuantizer`, the spectrogram
quantizer), the adaptive adversarial weight, the entropy regularizer and the
feature-matching term (``losses.feature_matching_loss`` / ``losses._as_score_list`` /
``losses._mean_over_maps``) are shared with the other codecs. No FAITH model code (§7).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .config import FastTSCodecConfig
from .fastts_nets import FastTSDecoder, FastTSEncoder, gain_shape_split
from .fastts_quantizer import FastTSQuantizer
# Variance below which a (window, channel) pair counts as dead and is left out of the
# normalized reconstruction loss. The counterpart of gate._FASTTS_STD_EPS (a std
# threshold), so the loss excludes what the reported metric excludes.
_FASTTS_LIVE_VAR_EPS: float = 1e-16

from .losses import (
    _as_score_list,
    _mean_over_maps,
    feature_matching_loss,
    shift_consistency,
)


class FastTSCodec(nn.Module):
    """Encoder, FSQ quantizer and decoder for the fast-TS codec (Phase A).

    Forward contract
    ----------------
    ``forward(x) -> dict``, with ``x`` of shape (B, C, W) in raw mode or (B, C, E) in
    envelope mode, and keys:
        ``recon``      same shape as ``x``     reconstruction
        ``feats``      (B, n_tok, d_model)     pre-FSQ continuous encoder features
        ``quant``      (B, n_tok, d_model)     straight-through quantized features
        ``codes``      (B, n_tok, fsq_dim)     discrete per-dimension FSQ codes (long),
                                               the Phase-B contract
        ``gain_pred``  only with gain-shape    decoded gain

    The discriminator (``fastts_discriminator.Env1DPatchGAN``) lives outside the codec
    and is created by the trainer, as for the spectrogram and video codecs, so the
    codec's parameters are exactly the tokenizer Phase B freezes and the two optimizers
    own disjoint parameter sets.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = FastTSEncoder(cfg)
        self.quantizer = FastTSQuantizer(cfg)
        self.decoder = FastTSDecoder(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Raw window -> pre-FSQ continuous features (B, n_tok, d_model)."""
        return self.encoder(x)

    def quantize(self, feats: torch.Tensor):
        """Features -> (quant (B,n_tok,d_model) float, codes (B,n_tok,fsq_dim) long)."""
        return self.quantizer.quantize(feats)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Quantized features (B, n_tok, d_model) -> reconstruction."""
        return self.decoder(quant)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encode(x)
        quant, codes = self.quantize(feats)
        # With gain-shape the decoder also returns the decoded gain, for direct
        # supervision; otherwise gain_pred is None and is left out of the dict.
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

    # Generator-side loss (the codec's half of the GAN alternation)
    def generator_losses(
        self,
        x: torch.Tensor,
        x_shift: torch.Tensor,
        disc: nn.Module,
        cfg: FastTSCodecConfig,
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Generator loss for one fast-TS codec step.

        ``x`` is the window to reconstruct (also the target) and ``x_shift`` its
        delta-shifted pair, used only when ``cfg.consistency_weight > 0`` (not in raw
        mode; see the module docstring).

        Raw-mode path (``adversarial_weight == 0``, ``fm_weight == 0``,
        ``consistency_weight == 0``)::

            total = pixel_anchor_weight * recon + ssim_weight * (1 - CS_SSIM)
                    + gain_weight * gain + entropy_weight * entropy

        ``recon`` is the sample-wise loss selected by ``cfg.recon_loss`` (default
        "nrmse", the per-(window, channel) std-normalized RMSE that is the reported
        metric; see :meth:`_recon_loss`). The trainer still steps the discriminator, but
        it adds nothing to the generator loss.

        Adaptive adversarial path (``cfg.adaptive_adv_weight`` and adversarial_weight >
        0; the VQGAN/MagViT adaptive weight, Taming Transformers 3.3)::

            recon_ref = pixel_anchor_weight * recon + ssim_weight * ssim
                        + gain_weight * gain + fm_weight * fm
            non_adv   = recon_ref + consistency_weight*consistency + entropy_weight*entropy
            adv       = -mean(D(recon))
            lam       = (grad_norm(recon_ref) / (grad_norm(adv) + 1e-4)).clamp(0, clamp).detach()
            adv_coeff = 0 if step < adv_warmup_steps else adversarial_weight * lam
            total     = non_adv + adv_coeff * adv

        Returns a dict with ``total``, ``adversarial``, ``pixel`` (the reconstruction
        loss), ``ssim`` (``1 - CS-SSIM``, 0 when ``ssim_weight`` is 0), ``gain``,
        ``feature_matching``, ``consistency``, ``entropy``, ``adaptive_weight``,
        ``recon`` and ``codes``, the keys the DDP trainer reads.
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]
        zero = torch.zeros((), device=recon.device, dtype=recon.dtype)

        # Gain-shape: direct supervision of the decoded gain. The reconstruction term
        # sees the level too, but mixed with the shape error. The gain head produces the
        # whole per-(window, channel) mean (96% of the variance), so it gets its own
        # target. Skipped unless gain_shape is on.
        if "gain_pred" in out and float(getattr(cfg, "gain_weight", 0.0)) > 0.0:
            gain_tgt, _, _, _ = gain_shape_split(x, cfg.uses_gain_scale)
            gain = torch.mean(torch.abs(out["gain_pred"] - gain_tgt))
        else:
            gain = zero

        # Reconstruction. Envelope mode uses the original MAE; raw mode uses the
        # metric-aligned loss selected by cfg.recon_loss (see _recon_loss).
        if cfg.is_raw:
            pixel = self._recon_loss(recon, x, cfg)
        else:
            pixel = torch.mean(torch.abs(recon - x))

        # Structural (SSIM) term, an amplitude guard. Off at ssim_weight 0.0.
        if cfg.is_raw and float(getattr(cfg, "ssim_weight", 0.0)) > 0.0:
            ssim = self._ssim_loss(recon, x, cfg)
        else:
            ssim = zero

        # Adversarial and feature matching, only when weighted.
        want_adv = float(cfg.adversarial_weight) > 0.0
        want_fm = float(cfg.fm_weight) > 0.0
        if want_adv or want_fm:
            fake_scores, fake_feats = disc(recon, return_features=True)
            adversarial = -_mean_over_maps(_as_score_list(fake_scores))
            if want_fm:
                with torch.no_grad():
                    _, real_feats = disc(x, return_features=True)
                fm = feature_matching_loss(real_feats, fake_feats)
            else:
                fm = zero
        else:
            adversarial = zero
            fm = zero

        # Delta-shift consistency; at weight 0 the second encoder pass is skipped.
        if float(cfg.consistency_weight) > 0.0:
            consistency = shift_consistency(feats_x, self.encode(x_shift))
        else:
            consistency = zero

        entropy = self.quantizer.entropy_loss(feats_x)

        # SSIM and gain are fidelity terms, so they belong to the reconstruction
        # reference that the adaptive adversarial weight balances against.
        recon_ref = (cfg.pixel_anchor_weight * pixel
                     + float(getattr(cfg, "ssim_weight", 0.0)) * ssim
                     + float(getattr(cfg, "gain_weight", 0.0)) * gain
                     + cfg.fm_weight * fm)
        non_adv_total = (
            recon_ref
            + cfg.consistency_weight * consistency
            + cfg.entropy_weight * entropy
        )

        if not want_adv:
            total = non_adv_total
            adaptive_weight = 0.0
        elif cfg.adaptive_adv_weight:
            lam = self._adaptive_adv_weight(recon_ref, adversarial, cfg)
            adv_coeff = (
                torch.zeros((), device=non_adv_total.device, dtype=non_adv_total.dtype)
                if step < cfg.adv_warmup_steps
                else cfg.adversarial_weight * lam
            )
            total = non_adv_total + adv_coeff * adversarial
            adaptive_weight = float(lam)
        else:
            total = non_adv_total + cfg.adversarial_weight * adversarial
            adaptive_weight = 1.0

        return {
            "total": total,
            "adversarial": adversarial,
            "pixel": pixel,
            "ssim": ssim,
            "gain": gain,
            "feature_matching": fm,
            "consistency": consistency,
            "entropy": entropy,
            "adaptive_weight": adaptive_weight,
            "recon": recon,
            "codes": codes,
        }

    @staticmethod
    def _ssim_loss(recon: torch.Tensor, target: torch.Tensor,
                   cfg: FastTSCodecConfig) -> torch.Tensor:
        """``1 - mean(contrast * structure)`` of the 1-D SSIM along the sample axis.

        The differentiable twin of ``fastts_ssim_cs`` in
        :func:`gate.fastts_ssim_metrics`: same box window (``cfg.ssim_win``), same
        'valid' positions and the same per-(window, channel) target-std dynamic range
        for the constants, so loss and metric are the same quantity.

        Luminance is left out: for a signed waveform ``2 mu_r mu_t`` goes negative when
        the two local means straddle zero (MS-SSIM also uses only CS at all but the
        coarsest scale), and the nRMSE term, which is not mean-removed, already carries
        the DC level.

        The contrast factor ``2 sig_r sig_t / (sig_r^2 + sig_t^2)`` drops when the
        reconstruction's local variance collapses, the failure an elementwise loss
        rewards. It constrains the local second moment instead of reweighting samples
        (spike weighting did not work for fast-TS).

        Dead (window, channel) pairs (constant target) are masked out, as in
        :meth:`_recon_loss` and the gate.
        """
        win = int(getattr(cfg, "ssim_win", 17))
        B, C, W = target.shape
        r = recon.reshape(B * C, 1, W)
        t = target.reshape(B * C, 1, W)
        live = target.reshape(B * C, W).var(dim=-1, unbiased=False) > _FASTTS_LIVE_VAR_EPS
        if not bool(live.any()) or W <= win:
            return torch.zeros((), device=recon.device, dtype=recon.dtype)
        r, t = r[live], t[live]
        L = t.squeeze(1).std(dim=-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)
        c2 = (0.03 * L) ** 2
        box = lambda z: torch.nn.functional.avg_pool1d(z, kernel_size=win, stride=1)
        mu_r, mu_t = box(r), box(t)
        var_r = (box(r * r) - mu_r * mu_r).clamp_min(0.0)
        var_t = (box(t * t) - mu_t * mu_t).clamp_min(0.0)
        cov = box(r * t) - mu_r * mu_t
        sd_r, sd_t = torch.sqrt(var_r + 1e-24), torch.sqrt(var_t + 1e-24)
        contrast = (2 * sd_r * sd_t + c2) / (var_r + var_t + c2)
        structure = (cov + c2 / 2.0) / (sd_r * sd_t + c2 / 2.0)
        return 1.0 - (contrast * structure).mean()

    @staticmethod
    def _recon_loss(recon: torch.Tensor, target: torch.Tensor,
                    cfg: FastTSCodecConfig) -> torch.Tensor:
        """Sample-wise reconstruction loss selected by ``cfg.recon_loss``.

        Why the default is "nrmse" and not "mse"
        ----------------------------------------
        In the per-channel standardized space a filterscope window has a per-(window,
        channel) std of ~0.007, while the window mean swings over ~25 units, so the
        between-window DC level carries about (25 / 0.007)^2 times the energy of the
        waveform inside the window. A plain-MSE optimum spends its capacity on the DC
        level and emits a flat line in each window, and a flat line at the window's own
        mean scores nRMSE exactly 1.0, the trivial baseline. The metric divides by each
        (window, channel)'s own std, so the loss must too.

        "nrmse" (default) is the reported metric: per (window, channel), RMSE divided by
        that pair's target std, averaged over pairs. DC error still enters (the
        numerator is not mean-removed), at the right relative scale. Dead pairs
        (constant target, zero variance) are masked out, as
        :func:`gate.full_fastts_metrics` excludes them.

        "nmse" drops the outer square root. It is worse conditioned: the per-pair
        squared ratio spans ~1e6 (quietest windows std 0.0009, p90 0.071), so its mean
        is dominated by a few quiet noise windows. The square root compresses that to
        ~1e3.

        "mse", "l1" and "huber" are unnormalized controls; "mse" shows the flat-line
        failure (a flat line reads mse 5.0e-05 and nrmse 1.0 on real data).

        There is no spike-weighted option, since that did not work for this modality;
        the conv stem in ``fastts_nets`` is used instead.
        """
        kind = getattr(cfg, "recon_loss", "nrmse")
        if kind in ("nmse", "nrmse"):
            # Per (window, channel) over the sample axis, as the gate groups it.
            diff2 = ((recon - target) ** 2).mean(dim=-1)                    # (B, C)
            var = target.var(dim=-1, unbiased=False)                        # (B, C)
            live = var > _FASTTS_LIVE_VAR_EPS
            if not bool(live.any()):
                return torch.zeros((), device=recon.device, dtype=recon.dtype)
            ratio = diff2[live] / var[live]
            return ratio.mean() if kind == "nmse" else ratio.clamp_min(0.0).sqrt().mean()
        if kind == "mse":
            return torch.mean((recon - target) ** 2)
        if kind == "l1":
            return torch.mean(torch.abs(recon - target))
        if kind == "huber":
            return torch.nn.functional.smooth_l1_loss(recon, target, beta=0.1)
        raise ValueError(
            f"unknown cfg.recon_loss={kind!r} (expected nrmse|nmse|mse|l1|huber)"
        )

    # VQGAN adaptive adversarial weight ("Taming Transformers" §3.3)
    def _adaptive_adv_weight(
        self,
        ref: torch.Tensor,
        adv: torch.Tensor,
        cfg: FastTSCodecConfig,
    ) -> torch.Tensor:
        """lam = ‖∇ref‖ / (‖∇adv‖ + 1e-4), clamped to [0, adaptive_adv_clamp], detached.

        Gradients are taken at ``self.decoder.last_layer``. Same computation as
        ``SpectroCodec._adaptive_adv_weight``.
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
