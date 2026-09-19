"""IGNITE Phase-A fast-TS (filterscopes) codec — encoder + FSQ + RAW-SAMPLE decoder.

2026-09-03 REDESIGN. The codec's input AND reconstruction target is now the RAW 10 kHz
waveform ``(B, C, W)`` (C = 8 filterscope channels, W = 500 samples in a 50 ms frame), not the
5-bin ELM ACTIVITY ENVELOPE it used to model. The envelope / RMS / pooling layer is gone.

OBJECTIVE INVERSION (the substantive change beyond the target)
--------------------------------------------------------------
The shipped envelope codec ran ``pixel_anchor_weight = 0.05`` against ``adversarial_weight =
1.0`` — reconstruction weighted ~20x BELOW the GAN. For a SAMPLE-WISE target that is exactly
backwards: an adversarial decoder synthesizes plausible high-frequency detail that is
UNCORRELATED with the true samples, which lowers no sample-wise error and raises nRMSE. The
defaults are therefore ``pixel_anchor_weight = 1.0``, ``adversarial_weight = 0.0``,
``fm_weight = 0.0`` — a reconstruction-only objective plus the FSQ entropy anti-collapse term.
The adversarial path is retained (``--adversarial_weight``) so the trade-off is measurable.

DELTA-SHIFT CONSISTENCY IS OFF (``consistency_weight = 0.0``)
------------------------------------------------------------
The envelope codec enforced ``|enc(x) - enc(x_shift)|^2`` because sub-bin spike TIMING was a
realization nuisance to be projected out. Sample-wise reconstruction needs the codes to CARRY
that timing, so the term is counter-productive and defaults OFF; when the weight is 0 the
second encoder pass is SKIPPED entirely (it is pure wasted compute). The delta pair is still
built by the loader because the gate's ``stability`` probe reports it — expect it LOW now, and
note that ``spike.gate_score`` does not read stability.

Everything else is reused from the shared IGNITE infra: the FSQ bottleneck
(:class:`~ignite.fastts_quantizer.FastTSQuantizer`, an alias of the spectro FSQ machinery), the
adaptive adversarial weight, the entropy/utilization regularizer, and the feature-matching term
(``losses.feature_matching_loss`` / ``losses._as_score_list`` / ``losses._mean_over_maps``).
Reuses **no** FAITH model code (7).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .config import FastTSCodecConfig
from .fastts_nets import FastTSDecoder, FastTSEncoder, gain_shape_split
from .fastts_quantizer import FastTSQuantizer
# Variance floor below which a (window, channel) pair is treated as dead / absent and
# excluded from the normalized reconstruction loss. Matches gate._FASTTS_STD_EPS**2 in spirit
# (that one thresholds the std, this one the variance) so the loss masks exactly what the
# reported metric excludes.
_FASTTS_LIVE_VAR_EPS: float = 1e-16

from .losses import (
    _as_score_list,
    _mean_over_maps,
    feature_matching_loss,
    shift_consistency,
)


class FastTSCodec(nn.Module):
    """Encoder + FSQ quantizer + decoder for the fast-TS RAW waveform (Phase A).

    Forward contract
    ----------------
    ``forward(x: (B, C, W)) -> dict`` with keys:
        ``recon``  (B, C, W)            reconstructed RAW samples
        ``feats``  (B, n_tok, d_model)  PRE-FSQ continuous encoder features
        ``quant``  (B, n_tok, d_model)  STE-quantized (decoder-facing) features
        ``codes``  (B, n_tok, fsq_dim)  discrete per-dim FSQ codes (long) — Phase-B contract.

    The discriminator (``fastts_discriminator.Env1DPatchGAN``) is OFF by default
    (``adversarial_weight`` 0.0) but still lives OUTSIDE the codec (created
    alongside it by the trainer), exactly as for the spectro / video codecs: the codec's
    parameters are precisely the tokenizer Phase B freezes, and the two optimizers own disjoint
    parameter sets.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = FastTSEncoder(cfg)
        self.quantizer = FastTSQuantizer(cfg)
        self.decoder = FastTSDecoder(cfg)

    # ------------------------------------------------------------------ #
    # forward paths
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, W) raw samples -> PRE-FSQ continuous features (B, n_tok, d_model)."""
        return self.encoder(x)

    def quantize(self, feats: torch.Tensor):
        """Features -> (quant (B,n_tok,d_model) float, codes (B,n_tok,fsq_dim) long)."""
        return self.quantizer.quantize(feats)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Quantized features (B, n_tok, d_model) -> raw-sample reconstruction (B, C, W)."""
        return self.decoder(quant)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encode(x)
        quant, codes = self.quantize(feats)
        # GAIN-SHAPE (opt-in) additionally returns the decoded gain so it can be supervised
        # directly; ``gain_pred`` is None (and the call is the plain single-tensor one) in
        # every other configuration, so the returned dict is a superset of the old one.
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

    # ------------------------------------------------------------------ #
    # generator-side loss (the codec's half of the GAN alternation)
    # ------------------------------------------------------------------ #
    def generator_losses(
        self,
        x: torch.Tensor,
        x_shift: torch.Tensor,
        disc: nn.Module,
        cfg: FastTSCodecConfig,
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Full generator loss for one fast-TS codec (generator) step, RAW-SAMPLE target.

        ``x`` is the (B, C, W) raw window to reconstruct (also the recon target) and
        ``x_shift`` its delta-shifted pair (only touched when ``cfg.consistency_weight > 0``,
        which is NOT the default -- see the module docstring).

        DEFAULT PATH (``adversarial_weight == 0``, ``fm_weight == 0``,
        ``consistency_weight == 0``)::

            total = pixel_anchor_weight * recon + ssim_weight * (1 - CS_SSIM)
                    + entropy_weight * entropy

        with ``recon`` the sample-wise reconstruction loss selected by ``cfg.recon_loss``
        ("nrmse" default -- the per-(window, channel) std-normalized RMSE, which IS the
        reported metric; plain "mse" is NOT a surrogate for it, see :meth:`_recon_loss`). The discriminator
        is still stepped by the trainer but contributes nothing to the generator, so a
        GAN-free run is exactly a reconstruction + anti-collapse run.

        ADAPTIVE ADVERSARIAL PATH (``cfg.adaptive_adv_weight`` and adversarial_weight > 0 --
        VQGAN/MagViT "adaptive weight", Taming Transformers 3.3)::

            recon_ref = pixel_anchor_weight * recon + fm_weight * fm
            non_adv   = recon_ref + consistency_weight*consistency + entropy_weight*entropy
            adv       = -mean(D(recon))
            lam       = (grad_norm(recon_ref) / (grad_norm(adv) + 1e-4)).clamp(0, clamp).detach()
            adv_coeff = 0 if step < adv_warmup_steps else adversarial_weight * lam
            total     = non_adv + adv_coeff * adv

        Returns a dict with keys ``total`` / ``adversarial`` / ``pixel`` (the RAW sample-wise
        reconstruction loss) / ``ssim`` (``1 - CS-SSIM``, 0 when ``ssim_weight`` is 0) /
        ``feature_matching`` / ``consistency`` / ``entropy`` / ``adaptive_weight`` /
        ``recon`` / ``codes`` -- a superset of the previous keys, so the DDP trainer consumes
        it unchanged.
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]
        zero = torch.zeros((), device=recon.device, dtype=recon.dtype)

        # ---- GAIN-SHAPE auxiliary: DIRECT supervision of the decoded gain -------------- #
        # The reconstruction term already sees the level (it is not mean-removed), but it sees
        # it mixed with the shape error at whatever relative scale the window happens to have.
        # The gain head produces the reconstruction's ENTIRE per-(window, channel) mean, which
        # is 96.2% of the variance, so it gets its own unambiguous target as well. OFF (and
        # exactly zero-cost) unless gain_shape is enabled.
        if "gain_pred" in out and float(getattr(cfg, "gain_weight", 0.0)) > 0.0:
            gain_tgt, _, _, _ = gain_shape_split(x, cfg.uses_gain_scale)
            gain = torch.mean(torch.abs(out["gain_pred"] - gain_tgt))
        else:
            gain = zero

        # ---- reconstruction (the LEADING term in raw mode) --------------------------- #
        # ENVELOPE mode (the default) keeps the ORIGINAL envelope-anchor MAE bit-for-bit, so a
        # pre-2026-09-03 config reproduces the old objective exactly. RAW mode uses the
        # metric-aligned loss selected by cfg.recon_loss (see _recon_loss).
        if cfg.is_raw:
            pixel = self._recon_loss(recon, x, cfg)
        else:
            pixel = torch.mean(torch.abs(recon - x))

        # ---- structural (SSIM) term: the amplitude/dynamic-range guard --------------- #
        # OFF by default (ssim_weight 0.0) so the plain arm is an unchanged-recipe control.
        if cfg.is_raw and float(getattr(cfg, "ssim_weight", 0.0)) > 0.0:
            ssim = self._ssim_loss(recon, x, cfg)
        else:
            ssim = zero

        # ---- adversarial / feature-matching: only when actually weighted ------------- #
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

        # ---- delta-shift consistency: OFF by default (skips a whole encoder pass) ----- #
        if float(cfg.consistency_weight) > 0.0:
            consistency = shift_consistency(feats_x, self.encode(x_shift))
        else:
            consistency = zero

        entropy = self.quantizer.entropy_loss(feats_x)

        # The SSIM term is part of the RECONSTRUCTION reference (it is a fidelity term, not a
        # regularizer), so the VQGAN adaptive adversarial weight balances against it too.
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
        """``1 - mean(contrast * structure)`` of the 1-D SSIM along the SAMPLE axis.

        The differentiable twin of :func:`gate.fastts_ssim_metrics`' ``fastts_ssim_cs`` —
        same box window (``cfg.ssim_win``), same 'valid' positions, same per-(window,
        channel) target-std dynamic range for the constants, so the loss and the reported
        metric are the SAME quantity rather than two similar ones.

        Luminance is deliberately excluded: for a SIGNED waveform ``2 mu_r mu_t`` goes
        negative whenever the two local means straddle zero (MS-SSIM itself uses CS at every
        scale but the coarsest), and the DC level is already carried by the nRMSE term, which
        is not mean-removed.

        WHY THIS TERM AND NOT SPIKE WEIGHTING. The contrast factor
        ``2 sig_r sig_t / (sig_r^2 + sig_t^2)`` collapses exactly when the reconstruction's
        LOCAL VARIANCE collapses, which is the variance-collapse failure an element-wise loss
        rewards. It is a local-second-moment constraint, not a per-sample reweighting — the
        latter is the recorded dead end for fast-TS.

        Dead / absent (window, channel) pairs (constant target) are masked out, matching
        both :meth:`_recon_loss` and the gate.
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

        WHY THE DEFAULT IS "nrmse" AND NOT "mse" (measured, 2026-09-03)
        ---------------------------------------------------------------
        Plain MSE is NOT a surrogate for the reported nRMSE on this signal. In the codec's
        per-channel-standardized space the raw filterscope window has a per-(window, channel)
        std of ~0.007 while the WINDOW MEAN swings over ~25 std units -- the between-window
        DC level carries roughly (25 / 0.007)^2 more energy than the within-window waveform.
        A plain-MSE optimum therefore spends essentially all of its capacity on the DC level
        and emits a FLAT line inside each window, and a flat line at the window's own mean
        scores nRMSE EXACTLY 1.0000, i.e. no better than the trivial baseline. The metric
        divides by each (window, channel)'s OWN std, so the loss must too.

        "nrmse" (DEFAULT) is that aligned loss and IS the reported metric: per
        (window, channel), RMSE divided by that pair's own target std, averaged over pairs.
        DC error still enters (the numerator is not mean-removed) but now at the right
        relative scale. Dead / absent pairs -- constant target, zero variance -- are MASKED
        out exactly as :func:`gate.full_fastts_metrics` excludes them, so they can neither
        blow the loss up nor be optimized against.

        "nmse" is the same thing without the outer square root. It is NOT preferred, for
        CONDITIONING reasons rather than objective ones: the per-pair squared ratio spans
        ~1e6 across the population (quietest windows std 0.0009 vs p90 0.071), so an
        un-rooted mean is dominated by a handful of quiet, pure-noise windows that carry no
        reconstructable content. The square root compresses that to ~1e3.

        "mse" / "l1" / "huber" are the un-normalized controls -- "mse" is kept precisely so
        the flat-line failure above can be demonstrated rather than asserted (a flat-line
        prediction reads mse 5.0e-05 and nrmse 1.0000 on the real geometry).

        A spike-WEIGHTED loss is deliberately NOT offered: that is the recorded dead end for
        this modality. The alternative on record -- a pre-patch conv stem / overlapping
        patches -- is implemented in ``fastts_nets`` instead.
        """
        kind = getattr(cfg, "recon_loss", "nrmse")
        if kind in ("nmse", "nrmse"):
            # per (window, channel) over the SAMPLE axis, mirroring the gate's grouping.
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

    # ------------------------------------------------------------------ #
    # VQGAN adaptive adversarial weight ("Taming Transformers" §3.3)
    # ------------------------------------------------------------------ #
    def _adaptive_adv_weight(
        self,
        ref: torch.Tensor,
        adv: torch.Tensor,
        cfg: FastTSCodecConfig,
    ) -> torch.Tensor:
        """lam = ‖∇ref‖ / (‖∇adv‖ + 1e-4), clamped to [0, cfg.adaptive_adv_clamp], DETACHED.

        Gradients w.r.t. the decoder's last layer (``self.decoder.last_layer``). Identical in
        spirit + numerics to ``SpectroCodec._adaptive_adv_weight``.
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
