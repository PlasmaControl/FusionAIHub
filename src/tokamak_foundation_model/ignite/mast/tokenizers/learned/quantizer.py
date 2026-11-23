"""Rounding of the recoder values during training (plan 1c).

A recoder's values lie in [-1, 1]. The downstream model reads them as numbers rounded to a few levels,
and a prefix of them with the rest zero. The network is trained so that both are harmless: a random
share of the examples is rounded to `LEVELS` or `FINE_LEVELS` levels (the rest is left as it is) and a
random suffix of the values is set to exactly 0, so the early values carry the most. The rounding passes
the gradient straight through. This module imports torch.
"""

from __future__ import annotations

import torch

LEVELS = 8                              # the coarse rounding: eight evenly spaced numbers from -1 to 1
FINE_LEVELS = 64                        # the fine rounding; the eight coarse levels are among them
P_PRECISION = (0.25, 0.25, 0.5)         # probabilities of precision 0 (LEVELS), 1 (FINE_LEVELS), 2 (unrounded)
P_FULL = 0.5                            # probability that all values of an example are kept


def round_levels(v: torch.Tensor, levels: int) -> torch.Tensor:
    """Round `v` (any shape, values in [-1, 1]) to the nearest of `levels` evenly spaced numbers from -1 to 1.

    Returns a tensor of the shape of `v`, values in [-1, 1], equal to the rounded numbers exactly; the
    gradient passes as if nothing was rounded (straight through).
    """
    if levels < 2:
        raise ValueError(f"levels must be at least 2, got {levels}")
    steps = levels - 1
    r = torch.round((v.detach() + 1.0) / 2.0 * steps) / steps * 2.0 - 1.0
    return v - v.detach() + r


def apply(v: torch.Tensor, precision: torch.Tensor, n_keep: torch.Tensor) -> torch.Tensor:
    """Round each example's values and zero the values from `n_keep` on.

    `v` is ``(B, n_values)``, values in [-1, 1]. `precision` is int64 ``(B,)``: 0 rounds example i to
    `LEVELS` levels, 1 to `FINE_LEVELS`, 2 leaves it unrounded. `n_keep` is int64 ``(B,)``: the values
    of example i from index ``n_keep[i]`` on are exactly 0. Returns ``(B, n_values)``, values in
    [-1, 1]; the gradient reaches the kept values.
    """
    p = precision.to(v.device)[:, None]
    rounded = torch.where(p == 0, round_levels(v, LEVELS), torch.where(p == 1, round_levels(v, FINE_LEVELS), v))
    keep = torch.arange(v.shape[1], device=v.device)[None, :] < n_keep.to(v.device)[:, None]
    return torch.where(keep, rounded, torch.zeros_like(rounded))


def draw(batch: int, n_values: int, generator: torch.Generator, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw each example's `precision` and `n_keep` for `apply`, every draw from `generator`.

    Returns two int64 tensors of shape ``(batch,)``: `precision` in {0, 1, 2} with the probabilities
    `P_PRECISION`, and `n_keep` equal to `n_values` with probability `P_FULL`, otherwise uniform over
    1 to `n_values` (so a value count, never 0). The draws are made on the generator's device and then
    moved to `device` (default: the generator's), so the same seed gives the same draws anywhere.
    """
    dev = generator.device
    u = torch.rand(batch, generator=generator, device=dev)
    precision = torch.full((batch,), 2, dtype=torch.int64, device=dev)
    precision[u < P_PRECISION[0] + P_PRECISION[1]] = 1
    precision[u < P_PRECISION[0]] = 0
    full = torch.rand(batch, generator=generator, device=dev) < P_FULL
    some = torch.randint(1, n_values + 1, (batch,), generator=generator, device=dev)
    n_keep = torch.where(full, torch.full_like(some, n_values), some)
    return precision.to(device or dev), n_keep.to(device or dev)
