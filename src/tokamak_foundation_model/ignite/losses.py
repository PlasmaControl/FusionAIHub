"""IGNITE Phase-A codec losses (fresh code, no FAITH model reuse).

Uses only `torch` + `config.py`. See docs/IGNITE_DESIGN.md §4.2.

Three losses:
- ``shift_consistency`` — enforces shift-invariance of the encoder on the PRE-FSQ
  continuous features (statistics-first codec; a δ-shifted raw signal should map to the
  same statistic).
- ``recon_objective`` — generator-side reconstruction loss: adversarial (hinge generator
  term over the discriminator's scores on the reconstruction) + a small,
  stability-gated pixel-MAE anchor.
- ``discriminator_loss`` — hinge GAN discriminator loss over real vs fake, with an
  optional R1 gradient-penalty stub.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .config import SpectroCodecConfig


# --------------------------------------------------------------------------- #
# encoder shift-consistency
# --------------------------------------------------------------------------- #
def shift_consistency(feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
    """MSE between the PRE-FSQ features of ``x`` and its δ-shifted version.

    Both tensors are (B, n_tok, d_model) continuous encoder features. Returns a scalar.
    Zero when the two feature sets are identical, positive otherwise, differentiable.
    """
    if feats_a.shape != feats_b.shape:
        raise ValueError(
            f"shift_consistency shape mismatch: {tuple(feats_a.shape)} vs {tuple(feats_b.shape)}"
        )
    return torch.mean((feats_a - feats_b) ** 2)


# --------------------------------------------------------------------------- #
# discriminator score helpers (hinge)
# --------------------------------------------------------------------------- #
def _as_score_list(scores) -> List[torch.Tensor]:
    """Normalize a discriminator output (Tensor or list of Tensors) to a list."""
    if isinstance(scores, (list, tuple)):
        return list(scores)
    return [scores]


def _mean_over_maps(maps: List[torch.Tensor]) -> torch.Tensor:
    """Mean of per-map means (equal weight per scale)."""
    return torch.stack([m.mean() for m in maps]).mean()


# --------------------------------------------------------------------------- #
# discriminator feature-matching (HiFi-GAN / MelGAN vocoder-GAN perceptual term)
# --------------------------------------------------------------------------- #
def feature_matching_loss(
    feats_real: List[torch.Tensor],
    feats_fake: List[torch.Tensor],
) -> torch.Tensor:
    """Mean L1 distance between matched discriminator feature activations.

    This is the vocoder-GAN feature-matching term (HiFi-GAN §2.2 / MelGAN): it plays the
    Genie perceptual-loss role but adapted to spectrograms, where VGG features do not
    transfer. The generator is pushed to make the discriminator's *internal representation*
    of the reconstruction match that of the real input.

    ``feats_real`` is treated as a DETACHED target (generator matches fake -> real; gradient
    flows only through ``feats_fake``). List lengths are handled defensively via ``zip`` (the
    shorter list bounds the sum), so a mismatch never raises. Returns a scalar:
    - 0 when every matched pair is elementwise equal,
    - > 0 otherwise,
    - differentiable w.r.t. the tensors in ``feats_fake``.

    An empty pairing (either list empty) returns a 0 scalar (no NaN).
    """
    pairs = list(zip(feats_real, feats_fake))
    if not pairs:
        # no features to match -> zero. Anchor dtype/device to a fake tensor if present.
        ref = feats_fake[0] if feats_fake else (feats_real[0] if feats_real else None)
        if ref is None:
            return torch.zeros(())
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    terms = [torch.mean(torch.abs(fk - rl.detach())) for rl, fk in pairs]
    return torch.stack(terms).mean()


# --------------------------------------------------------------------------- #
# generator-side reconstruction objective
# --------------------------------------------------------------------------- #
def recon_objective(
    recon: torch.Tensor,
    target: torch.Tensor,
    disc: torch.nn.Module,
    cfg: SpectroCodecConfig,
) -> Dict[str, torch.Tensor]:
    """Generator-side loss = adversarial term + ``cfg.pixel_anchor_weight`` * pixel-MAE.

    - adversarial (hinge generator): ``-mean(D(recon))`` — the generator wants the
      discriminator's raw scores on the reconstruction to be high.
    - pixel: L1 / MAE between reconstruction and target (the small, stability-gated anchor).

    Returns a dict with keys ``adversarial``, ``pixel`` (raw, un-weighted MAE) and
    ``total`` (adversarial_weight * adversarial + pixel_anchor_weight * pixel).
    """
    fake_scores = _as_score_list(disc(recon))
    adversarial = -_mean_over_maps(fake_scores)

    pixel = torch.mean(torch.abs(recon - target))

    total = cfg.adversarial_weight * adversarial + cfg.pixel_anchor_weight * pixel
    return {"total": total, "adversarial": adversarial, "pixel": pixel}


# --------------------------------------------------------------------------- #
# discriminator hinge loss
# --------------------------------------------------------------------------- #
def _hinge_real(maps: List[torch.Tensor]) -> torch.Tensor:
    # want real scores >= +1  ->  penalise (1 - score)_+
    return torch.stack([torch.relu(1.0 - m).mean() for m in maps]).mean()


def _hinge_fake(maps: List[torch.Tensor]) -> torch.Tensor:
    # want fake scores <= -1  ->  penalise (1 + score)_+
    return torch.stack([torch.relu(1.0 + m).mean() for m in maps]).mean()


def discriminator_loss(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    cfg: SpectroCodecConfig,
    r1_gamma: float = 0.0,
) -> torch.Tensor:
    """Hinge GAN discriminator loss.

    ``L_D = E[relu(1 - D(real))] + E[relu(1 + D(fake))]``  (averaged over scales/maps).

    Lower when real scores are high (>= +1) and fake scores are low (<= -1).

    Optional R1 gradient penalty (``r1_gamma > 0``): adds
    ``0.5 * r1_gamma * E[||∇_real D(real)||²]`` computed on the real inputs. Off by default
    (stub; the real trainer wires the gradient graph explicitly).
    """
    real_maps = _as_score_list(disc(real))
    fake_maps = _as_score_list(disc(fake.detach() if fake.requires_grad else fake))

    loss = _hinge_real(real_maps) + _hinge_fake(fake_maps)

    if r1_gamma > 0.0:
        loss = loss + _r1_penalty(disc, real, r1_gamma)

    return loss


def _r1_penalty(disc: torch.nn.Module, real: torch.Tensor, gamma: float) -> torch.Tensor:
    """R1 gradient penalty on the real samples: 0.5 * gamma * ||grad_real D(real)||^2."""
    real = real.detach().requires_grad_(True)
    maps = _as_score_list(disc(real))
    score = _mean_over_maps(maps)
    grad = torch.autograd.grad(
        outputs=score, inputs=real, create_graph=True, retain_graph=True
    )[0]
    penalty = grad.reshape(grad.shape[0], -1).pow(2).sum(dim=1).mean()
    return 0.5 * gamma * penalty
