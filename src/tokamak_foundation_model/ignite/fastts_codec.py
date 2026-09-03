"""IGNITE Phase-A fast-TS (filterscopes) codec — encoder + FSQ + generative decoder.

Mirrors :class:`~ignite.codec.SpectroCodec` — same composition (encoder + FSQ + decoder) and
the SAME generator-side GAN loss shape (adversarial + low envelope-anchor + feature-matching +
**shift-consistency** + entropy, balanced by the VQGAN adaptive adversarial weight at the
decoder's last layer) — but over the 1-D ELM ACTIVITY ENVELOPE ``(B, C, E)`` instead of a
spectrogram ``(B, C, F, T)``.

WHY fast-TS keeps the δ-shift consistency term (unlike the video codec)
----------------------------------------------------------------------
The spectrogram codec needs a δ-shift consistency loss because a spectrogram carries an
STFT-phase realization nuisance (a 0.5 ms sub-window shift scrambles ~74 % of the codes) that
must be projected out so the codes carry only the statistic (docs/IGNITE_DESIGN.md §4.1-§4.2).
Fast-TS has the SAME kind of nuisance: the exact ELM spike *timing* within an envelope bin is a
realization (§4.3 explicitly: statistic = ELM activity envelope, "NOT spike timing"). So the
fast-TS codec, like spectro (and unlike Genie-native video), takes a δ-shifted RAW pair, maps
both to envelopes, and enforces ``‖enc(env_a) − enc(env_b)‖²`` on the PRE-FSQ features. The
δ-pair is built from the raw signal by :func:`ignite.data.fastts_shift_pair_windows`.

Everything else is reused from the shared IGNITE infra: the FSQ bottleneck
(:class:`~ignite.fastts_quantizer.FastTSQuantizer`, an alias of the spectro FSQ machinery), the
adaptive adversarial weight, the entropy/utilization regularizer, the feature-matching term
(``losses.feature_matching_loss`` / ``losses._as_score_list`` / ``losses._mean_over_maps``).
Reuses **no** FAITH model code (§7).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .config import FastTSCodecConfig
from .fastts_nets import FastTSDecoder, FastTSEncoder
from .fastts_quantizer import FastTSQuantizer
from .losses import (
    _as_score_list,
    _mean_over_maps,
    feature_matching_loss,
    shift_consistency,
)


class FastTSCodec(nn.Module):
    """Encoder + FSQ quantizer + generative decoder for the fast-TS envelope (Phase A).

    Forward contract
    ----------------
    ``forward(x: (B, C, E)) -> dict`` with keys:
        ``recon``  (B, C, E)            decoded ELM envelope (generative output)
        ``feats``  (B, n_tok, d_model)  PRE-FSQ continuous encoder features
        ``quant``  (B, n_tok, d_model)  STE-quantized (decoder-facing) features
        ``codes``  (B, n_tok, fsq_dim)  discrete per-dim FSQ codes (long) — Phase-B contract.

    The discriminator (``fastts_discriminator.Env1DPatchGAN``) lives OUTSIDE the codec (created
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
        """(B, C, E) -> PRE-FSQ continuous features (B, n_tok, d_model).

        This is the tensor the shift-consistency loss operates on (§4.2: consistency is
        enforced on the PRE-FSQ features, avoiding discrete matching).
        """
        return self.encoder(x)

    def quantize(self, feats: torch.Tensor):
        """Features -> (quant (B,n_tok,d_model) float, codes (B,n_tok,fsq_dim) long)."""
        return self.quantizer.quantize(feats)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Quantized features (B, n_tok, d_model) -> reconstruction (B, C, E)."""
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
        cfg: FastTSCodecConfig,
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Full generator loss for one fast-TS codec (generator) step.

        Identical in shape + numerics to :meth:`SpectroCodec.generator_losses` (adaptive +
        fixed-weight paths), only over the 1-D envelope and its δ-shift pair. ``x`` is the
        envelope to reconstruct (also the recon target) and ``x_shift`` its δ-shifted nuisance
        pair (same ELM activity statistic, different sub-bin spike realization); the
        consistency term is ``‖enc(x) − enc(x_shift)‖²`` on the PRE-FSQ features.

        ADAPTIVE (``cfg.adaptive_adv_weight`` True — VQGAN/MagViT "adaptive weight", Taming
        Transformers §3.3), the default::

            recon_ref = cfg.pixel_anchor_weight * anchor + cfg.fm_weight * fm
            non_adv   = recon_ref + consistency_weight*consistency + entropy_weight*entropy
            adv       = -mean(D(recon))
            lam       = (‖∇recon_ref‖ / (‖∇adv‖ + 1e-4)).clamp(0, clamp).detach()
            adv_coeff = 0 if step < adv_warmup_steps else adversarial_weight * lam
            total     = non_adv + adv_coeff * adv

        FIXED-WEIGHT (``cfg.adaptive_adv_weight`` False) reproduces the prior fixed behavior
        exactly (fm NOT folded in).

        Returns a dict with keys ``total`` / ``adversarial`` / ``pixel`` (raw envelope-anchor
        MAE) / ``feature_matching`` / ``consistency`` / ``entropy`` / ``adaptive_weight`` /
        ``recon`` / ``codes`` — the SAME keys as :meth:`SpectroCodec.generator_losses`, so
        :func:`spike.codec_train_step` and the DDP trainer consume it unchanged.
        """
        out = self.forward(x)
        recon, feats_x, codes = out["recon"], out["feats"], out["codes"]

        # ONE discriminator pass: patch scores (adversarial) + intermediate features (FM).
        fake_scores, fake_feats = disc(recon, return_features=True)
        fake_scores = _as_score_list(fake_scores)
        adversarial = -_mean_over_maps(fake_scores)  # hinge generator term: -mean(D(recon))

        pixel = torch.mean(torch.abs(recon - x))  # envelope anchor (MAE)

        with torch.no_grad():
            _, real_feats = disc(x, return_features=True)
        fm = feature_matching_loss(real_feats, fake_feats)

        feats_shift = self.encode(x_shift)
        consistency = shift_consistency(feats_x, feats_shift)

        entropy = self.quantizer.entropy_loss(feats_x)

        # VQGAN-canonical: adaptive weight balances adv against the RECONSTRUCTION reference
        # (envelope anchor + feature-matching), NOT the consistency / entropy regularizers.
        recon_ref = cfg.pixel_anchor_weight * pixel + cfg.fm_weight * fm
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
            total = (
                cfg.adversarial_weight * adversarial
                + cfg.pixel_anchor_weight * pixel
                + cfg.consistency_weight * consistency
                + cfg.entropy_weight * entropy
            )
            adaptive_weight = 1.0

        return {
            "total": total,
            "adversarial": adversarial,
            "pixel": pixel,
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
