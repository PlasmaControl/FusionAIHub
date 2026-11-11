"""Codec losses (docs/IGNITE_DESIGN.md §4.2).

Uses only ``torch`` and ``config.py``. The main terms:

- ``shift_consistency``: makes the encoder's pre-FSQ features invariant to a δ-shift of
  the raw signal, since the codec encodes statistics.
- ``recon_objective``: generator loss, the hinge adversarial term on the reconstruction
  plus a small pixel-MAE anchor.
- ``discriminator_loss``: hinge discriminator loss on real and fake, with an optional R1
  penalty.

The other functions (feature matching, multi-scale L1, peak-weighted L1, frequency
gradient, MS-SSIM) are optional reconstruction terms.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .config import SpectroCodecConfig


def shift_consistency(feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
    """MSE between the pre-FSQ features of ``x`` and of its δ-shifted copy.

    Both are (B, n_tok, d_model) encoder features. Returns a differentiable scalar, zero
    when the two are identical.
    """
    if feats_a.shape != feats_b.shape:
        raise ValueError(
            f"shift_consistency shape mismatch: {tuple(feats_a.shape)} vs {tuple(feats_b.shape)}"
        )
    return torch.mean((feats_a - feats_b) ** 2)


# Discriminator score helpers
def _as_score_list(scores) -> List[torch.Tensor]:
    """Normalize a discriminator output (Tensor or list of Tensors) to a list."""
    if isinstance(scores, (list, tuple)):
        return list(scores)
    return [scores]


def _mean_over_maps(maps: List[torch.Tensor]) -> torch.Tensor:
    """Mean of per-map means (equal weight per scale)."""
    return torch.stack([m.mean() for m in maps]).mean()


def feature_matching_loss(
    feats_real: List[torch.Tensor],
    feats_fake: List[torch.Tensor],
) -> torch.Tensor:
    """Mean L1 distance between matched discriminator activations.

    The vocoder-GAN feature-matching term (HiFi-GAN §2.2, MelGAN). It takes the place of
    Genie's perceptual loss, since VGG features do not transfer to spectrograms: the
    discriminator's internal features of the reconstruction should match those of the
    real input.

    ``feats_real`` is a detached target, so the gradient flows only through
    ``feats_fake``. The lists are paired with ``zip``, so the shorter one sets the
    length and a mismatch does not raise. Returns a scalar, 0 when every pair is equal
    and when either list is empty.
    """
    pairs = list(zip(feats_real, feats_fake))
    if not pairs:
        # Nothing to match. Take dtype and device from a feature tensor if there is one.
        ref = feats_fake[0] if feats_fake else (feats_real[0] if feats_real else None)
        if ref is None:
            return torch.zeros(())
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    terms = [torch.mean(torch.abs(fk - rl.detach())) for rl, fk in pairs]
    return torch.stack(terms).mean()


def recon_objective(
    recon: torch.Tensor,
    target: torch.Tensor,
    disc: torch.nn.Module,
    cfg: SpectroCodecConfig,
) -> Dict[str, torch.Tensor]:
    """Generator loss: adversarial term plus a small pixel-MAE anchor.

    - adversarial: hinge generator term ``-mean(D(recon))``; high discriminator scores
      on the reconstruction are rewarded.
    - pixel: MAE between reconstruction and target.

    Returns a dict with ``adversarial``, ``pixel`` (unweighted) and ``total``
    (``adversarial_weight * adversarial + pixel_anchor_weight * pixel``).
    """
    fake_scores = _as_score_list(disc(recon))
    adversarial = -_mean_over_maps(fake_scores)

    pixel = torch.mean(torch.abs(recon - target))

    total = cfg.adversarial_weight * adversarial + cfg.pixel_anchor_weight * pixel
    return {"total": total, "adversarial": adversarial, "pixel": pixel}


def multiscale_recon_loss(recon: torch.Tensor, target: torch.Tensor,
                          scales=(2, 4)) -> torch.Tensor:
    """Multi-resolution L1 on (B, C, F, T) spectrograms.

    The image counterpart of the multi-resolution STFT/mel loss of audio codecs
    (Spectral Codecs, arXiv 2406.05298). Full-resolution L1 is minimized by the smooth
    conditional mean (a per-frequency envelope), so turbulent modalities come out
    blurred. L1 at coarser average-pooled resolutions also scores band contrast.
    """
    import torch.nn.functional as _F
    loss = recon.new_zeros(())
    for s in scales:
        loss = loss + torch.mean(torch.abs(_F.avg_pool2d(recon, s) - _F.avg_pool2d(target, s)))
    return loss / max(1, len(scales))


def peak_l1_loss(recon: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """L1 weighted toward the target's spectral peaks, a surrogate for ``peak_f1``.

    ``recon`` and ``target`` are ``(B, C, F, T)``. In each ``(b, c, t)`` column the
    weight of a frequency bin is its prominence over the column mean::

        w = relu(target - target.mean(dim=F, keepdim=True))      # 0 on the floor
        w = w / mean(w)                                          # mean weight 1
        loss = mean(w * |recon - target|)

    Why: on 320 held-out co2 windows, ``peak_f1`` (top-k spectral peak overlap, i.e.
    whether the mode tracks are in the right place) was:

        tsmooth5 oracle (perfect coherent structure)  0.9936
        patchmean oracle (exact patch means)          0.6537
        shipped ms5 codec                             0.6116
        best adversarial codec (hf 82% of ceiling)    0.6196

    Both trained codecs score below the plain patch-mean code, so the codes carry almost
    no sub-patch track information, and sharpness did not help: adversarial training
    took hf from 15% to 82% of ceiling and std_ratio from 0.766 to 0.964 for +0.008 of
    peak_f1. No other term rewards peak placement. A mode line covers 1-2 of a patch's
    16 frequency bins, so placing it right changes plain L1 by ~10% of the patch area
    and ms_ssim's local contrast only a little, while the adversarial term rewards the
    right texture statistics anywhere. Weighting by prominence makes the peaks most of
    the loss.

    ``mask`` is an optional ``(B, C, T)`` or ``(B, T)`` validity mask, broadcast over
    frequency. Weights, their normalisation and the mean then use valid positions only,
    so the scale does not change with the amount of missing data. Returns a scalar.
    """
    t_mean = target.mean(dim=-2, keepdim=True)                 # (B, C, 1, T)
    w = torch.relu(target - t_mean)
    if mask is not None:
        m = mask.to(dtype=recon.dtype)
        if m.dim() == 2:
            m = m[:, None, None, :]
        elif m.dim() == 3:
            m = m.unsqueeze(2)
        else:
            raise ValueError(f"mask must be (B,T) or (B,C,T); got {tuple(mask.shape)}")
        w = w * m
        denom = w.sum()
        if float(denom) <= 0.0:
            return torch.mean(torch.abs(recon - target))
        return (w * torch.abs(recon - target)).sum() / denom
    denom = w.sum()
    if float(denom) <= 0.0:                                    # a constant plate has no peaks
        return torch.mean(torch.abs(recon - target))
    return (w * torch.abs(recon - target)).sum() / denom


def time_smooth(x: torch.Tensor, k: int) -> torch.Tensor:
    """``k``-tap boxcar moving average along time, edge-padded, same shape.

    ``x`` is ``(B, C, F, T)``. For ``k <= 1`` the input object itself is returned, so a
    disabled smoothing target is bit-identical.
    """
    if k <= 1:
        return x
    import torch.nn.functional as _F
    pad_l, pad_r = k // 2, k - 1 - k // 2
    # 'replicate' on a 4-D tensor pads the last two dims. To pad time only, fold batch
    # and channels together and use the 3-D form.
    B, C, F, T = x.shape
    xp = _F.pad(x.reshape(B * C, F, T), (pad_l, pad_r), mode="replicate")
    ker = x.new_full((1, 1, k), 1.0 / float(k)).expand(F, 1, k)
    return _F.conv1d(xp, ker, groups=F).reshape(B, C, F, T)


def freq_gradient_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on the frequency derivative of (B, C, F, T) spectrograms.

    A mean reconstruction has a near-zero frequency gradient where the truth has sharp
    bands, so matching the finite difference along F (axis -2) rewards the band detail
    that pixel L1 ignores.
    """
    dr = recon[..., 1:, :] - recon[..., :-1, :]
    dt = target[..., 1:, :] - target[..., :-1, :]
    return torch.mean(torch.abs(dr - dt))


# Discriminator hinge loss
def _hinge_real(maps: List[torch.Tensor]) -> torch.Tensor:
    # Real scores should be >= +1: penalise relu(1 - score).
    return torch.stack([torch.relu(1.0 - m).mean() for m in maps]).mean()


def _hinge_fake(maps: List[torch.Tensor]) -> torch.Tensor:
    # Fake scores should be <= -1: penalise relu(1 + score).
    return torch.stack([torch.relu(1.0 + m).mean() for m in maps]).mean()


def discriminator_loss(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    cfg: SpectroCodecConfig,
    r1_gamma: float = 0.0,
) -> torch.Tensor:
    """Hinge GAN discriminator loss.

    ``L_D = E[relu(1 - D(real))] + E[relu(1 + D(fake))]``, averaged over scales.

    With ``r1_gamma > 0`` it adds the R1 penalty
    ``0.5 * r1_gamma * E[||grad_real D(real)||^2]`` on the real inputs. Off by default;
    the trainer builds its own gradient penalty.
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


def ms_ssim_loss(recon: torch.Tensor, target: torch.Tensor,
                 win: int = 7, scales=(1, 2, 4), c1_frac: float = 0.01,
                 c2_frac: float = 0.03) -> torch.Tensor:
    """``1 - MS-SSIM`` between two (B, C, F, T) log-magnitude spectrograms.

    Why SSIM rather than an L-p term::

        SSIM = luminance x contrast x structure
        contrast factor = 2 sigma_x sigma_y / (sigma_x^2 + sigma_y^2)

    A blurred reconstruction has local ``sigma_y << sigma_x`` where a mode ridge was, so
    the contrast factor drops and SSIM penalises the blur. An L1 or L2 term has no
    variance factor, and the conditional mean minimises it. That is what happened: the
    codec with the best ``spec_nrmse`` (0.9010) had ``hf_ratio`` 0.0143 and no visible
    structure, while a worse one (1.0972 / 0.8258) reproduced the mode track. This is
    the differentiable form of ``gate.ms_ssim``, so training and judging use the same
    quantity. It also matches arXiv 2406.05298, which reports ViSQOL (built on NSIM, a
    structural similarity over spectro-temporal patches) rather than time-domain error.

    Local statistics use a uniform ``win x win`` box filter (``avg_pool2d``), and scales
    come from ``avg_pool2d`` downsampling. The stabilising constants are set from each
    (sample, channel) target's dynamic range, so the loss does not depend on the
    modality's units. Returns a scalar in ``[0, ~2]``, 0 for a perfect reconstruction.
    """
    import torch.nn.functional as _F

    def _box(z: torch.Tensor, k: int) -> torch.Tensor:
        return _F.avg_pool2d(z, kernel_size=k, stride=1)

    rng = (target.amax(dim=(-2, -1), keepdim=True)
           - target.amin(dim=(-2, -1), keepdim=True)).clamp_min(1e-12)
    c1 = (c1_frac * rng) ** 2
    c2 = (c2_frac * rng) ** 2
    terms = []
    a, b = recon, target
    for i, s in enumerate(scales):
        if i > 0:
            f = s // scales[i - 1] if scales[i - 1] > 0 else s
            if f > 1:
                a, b = _F.avg_pool2d(a, f), _F.avg_pool2d(b, f)
        if min(a.shape[-2], a.shape[-1]) <= win:
            break
        mu_a, mu_b = _box(a, win), _box(b, win)
        sa = (_box(a * a, win) - mu_a * mu_a).clamp_min(0.0)
        sb = (_box(b * b, win) - mu_b * mu_b).clamp_min(0.0)
        sab = _box(a * b, win) - mu_a * mu_b
        ssim = (((2 * mu_a * mu_b + c1) * (2 * sab + c2))
                / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2)))
        terms.append(ssim.mean())
    if not terms:
        return recon.new_zeros(())
    return 1.0 - torch.stack(terms).mean()
