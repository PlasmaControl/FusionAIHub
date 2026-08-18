"""Decode-policy configuration and helpers for the MaskGIT sampler.

Pure policy: these functions take logits/probabilities and return filtered or
reordered ones. They hold no model state, so they are cheap to unit-test and can be
swapped per eval arm. ``SamplerConfig()`` with no arguments reproduces the original
sampler exactly (see tests/ignite/test_phaseb_compat.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch


@dataclass
class SamplerConfig:
    """How a frame is decoded. Defaults == the original behaviour, bit for bit.

    temperature      : scalar, or {modality_name: float}. Spectro modalities carry a
                       64k vocab and 768 tokens; slow-TS carries 1k and 4. One global
                       temperature over-disperses the former.
    top_p            : nucleus filter applied per token before sampling. None = off.
    global_pool      : budget each decode step's reveals over ALL of a frame's tokens instead
                       of per modality, scoring them by VOCAB-NORMALIZED log-confidence
                       (``norm_log_confidence``), so an uncertain modality can defer while
                       confident ones commit and anchor it (cross-modal coherence).
    revision_rounds  : after the schedule completes, re-mask the least-confident
                       ``revision_frac`` of the frame and re-decode, this many times.
    cfg_scale        : classifier-free guidance on the actuator conditioning.
                       1.0 = off (single forward pass, no cost).
    """

    temperature: Union[float, Dict[str, float]] = 1.0
    top_p: Optional[float] = None
    global_pool: bool = False
    revision_rounds: int = 0
    revision_frac: float = 0.25
    cfg_scale: float = 1.0

    def temp_for(self, name: str) -> float:
        if isinstance(self.temperature, dict):
            return float(self.temperature.get(name, 1.0))
        return float(self.temperature)


def apply_top_p(probs: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    """Nucleus filter over the last dim, renormalized. Identity when ``top_p`` is None.

    The highest-probability token is always kept, so a top_p below the max prob still
    yields a valid distribution rather than an all-zero row.
    """
    if top_p is None:
        return probs
    srt, idx = probs.sort(dim=-1, descending=True)
    cum = srt.cumsum(dim=-1)
    keep = cum - srt < top_p                       # keep while the mass BEFORE this token < p
    keep[..., 0] = True                            # always keep the argmax
    filt = torch.zeros_like(probs).scatter_(-1, idx, srt * keep)
    return filt / filt.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def rank_normalize(conf: torch.Tensor) -> torch.Tensor:
    """Map each row's values to their quantile rank in [0, 1] along the last dim.

    Raw sampled-token probabilities are NOT comparable across modalities: a 64 000-way
    softmax puts far less mass on its argmax than a 1 000-way one, so a global argsort
    over raw confidence would let low-vocab modalities monopolize every reveal step.
    Rank normalization removes the scale while preserving order within each modality.

    The sort is STABLE, so ties resolve by index rather than by whatever the sort backend
    happens to do: among equal values the lower index takes the lower rank (and is therefore
    revealed later by a descending-confidence policy). Equal confidences are reachable in
    practice — a near-uniform head, or a ``top_p`` nucleus of equal-mass tokens — so without
    ``stable=True`` reveal order would not be reproducible across devices.

    NOT used by the global pool: mapping every modality onto the same {0..1} rank grid also
    erases the BETWEEN-modality confidence signal, which makes a pooled allocation provably
    token-count-proportional (i.e. the fixed quota again). The pool scores with
    ``norm_log_confidence`` instead; this stays as a scale-free within-modality ranking.
    """
    n = conf.shape[-1]
    if n == 1:
        return torch.ones_like(conf)
    order = conf.argsort(dim=-1, stable=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(n, device=conf.device).expand_as(order)
    ranks.scatter_(-1, order, ar)
    return ranks.to(conf.dtype) / (n - 1)


def norm_log_confidence(conf: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Vocab-adjusted confidence in (-inf, 1]: ``1 + log(c) / log(V)``.

    0 = uniform (c == 1/V), 1 = certain. Comparable ACROSS modalities because the
    log-vocab denominator removes the structural smallness of a large-vocab softmax's
    probabilities, while — unlike pure within-modality rank normalization — it
    PRESERVES the between-modality confidence signal the global pool exists to use:
    pure ranks map every modality onto the same {0..1} grid, which makes the pooled
    allocation provably confidence-independent (token-count-proportional, i.e. the
    fixed quota again).
    """
    return 1.0 + conf.clamp_min(1e-12).log() / math.log(max(2, vocab_size))
