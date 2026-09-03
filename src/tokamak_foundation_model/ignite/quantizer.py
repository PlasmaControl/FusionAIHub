"""Phase-A FSQ bottleneck for the statistics-first spectrogram codec.

Thin wrapper over ``vector_quantize_pytorch.FSQ``. The encoder produces
``d_model`` features; the FSQ projects them down to ``fsq_dim``, quantizes with
a straight-through estimator, and projects back up to ``d_model`` for the
decoder. The closed code space (Phase B) consumes the per-dimension integer
``codes``; the continuous ``quant`` is decoder-facing and carries the STE
gradient back to the encoder features.

Only external libs (torch, vector_quantize_pytorch) + config.py are used; no
FAITH model code is imported.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from vector_quantize_pytorch import FSQ

from .config import SpectroCodecConfig


class SpectroQuantizer(nn.Module):
    """FSQ bottleneck: (B,n_tok,d_model) <-> discrete per-dim codes.

    ``quantize(feats)`` returns ``(quant, codes)`` where ``quant`` is the
    quantized features re-projected to ``d_model`` (float, STE-differentiable
    w.r.t. ``feats``) and ``codes`` are the per-dimension FSQ level indices,
    shape ``(B, n_tok, fsq_dim)`` with entry ``i`` in ``[0, fsq_levels[i])``.
    """

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # FSQ owns the d_model <-> fsq_dim projections (project_in/project_out)
        # and applies the straight-through estimator internally.
        self.fsq = FSQ(
            levels=list(cfg.fsq_levels),
            dim=cfg.d_model,
            channel_first=False,
            return_indices=True,
        )

    @property
    def codebook_size(self) -> int:
        return self.cfg.codebook_size

    def quantize(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # quant: (B, n_tok, d_model) continuous, STE grad -> feats.
        # flat_indices: (B, n_tok) codebook index in [0, codebook_size).
        quant, flat_indices = self.fsq(feats)
        # Expand the flat index into per-dimension level indices, one per fsq
        # level, each in [0, levels[i]). This is the Phase-B token contract.
        codes = self.fsq.indices_to_level_indices(flat_indices).long()
        return quant, codes

    # ------------------------------------------------------------------ #
    # pre-quant per-dim continuous scalars (the value round() maps to a code)
    # ------------------------------------------------------------------ #
    def pre_quant_levels(self, feats: torch.Tensor) -> torch.Tensor:
        """Differentiable per-dim continuous level-index scalars, shape (B, n_tok, fsq_dim).

        This replicates ``FSQ``'s ``project_in`` + ``bound`` path but WITHOUT the
        straight-through ``round`` — it returns the continuous position of each FSQ
        dimension in level-index space (``[0, levels[i]-1]``). ``round()`` of this tensor
        recovers exactly the integer ``codes`` from :meth:`quantize` (verified against the
        library), so it is the correct handle on the pre-quant per-dim scalar for the
        entropy regularizer.

        The math mirrors ``FSQ.bound``: with ``half_l = (L-1)(1+eps)/2``,
        ``offset = 0.5 if L even else 0`` and ``shift = atanh(offset/half_l)``,
        the bounded value is ``bz = tanh(z + shift) * half_l - offset`` and FSQ then does
        ``round(bz) / half_width``; ``_scale_and_shift`` maps that normalized code back to a
        level index as ``round(bz) + half_width``. Dropping the round gives the continuous
        level position ``bz + half_width``.
        """
        fsq = self.fsq
        # project d_model -> fsq_dim (same layer FSQ.forward uses).
        z = fsq.project_in(feats)  # (B, n_tok, fsq_dim)

        levels = fsq._levels.to(device=z.device, dtype=z.dtype)  # (fsq_dim,)
        eps = 1e-3
        half_l = (levels - 1) * (1 + eps) / 2
        offset = torch.where(levels % 2 == 0, torch.tensor(0.5, dtype=z.dtype, device=z.device),
                             torch.tensor(0.0, dtype=z.dtype, device=z.device))
        shift = torch.atanh(offset / half_l)
        bounded_z = torch.tanh(z + shift) * half_l - offset
        half_width = torch.div(levels, 2, rounding_mode="floor")
        # continuous level-index position; round() recovers the integer code.
        return bounded_z + half_width

    def entropy_loss(self, feats: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        """Differentiable anti-collapse (codebook-utilization) term. Returns a scalar.

        For each FSQ dimension ``i`` we soft-assign the continuous pre-quant level position
        ``x`` (from :meth:`pre_quant_levels`) to that dim's integer grid points
        ``g = 0..levels[i]-1`` via ``softmax(-beta * (x - g)^2)``. This yields per-position
        soft assignment distributions ``p`` (shape ``(N, fsq_dim, max_levels)``, padded
        dims masked out).

        The loss combines two Genie/LFQ-style terms:

          * ``per_sample_entropy_mean`` — the mean entropy of each individual assignment
            distribution. LOW when the encoder is *confident* (commits to a code); this
            term does not by itself prevent collapse.
          * ``entropy_of_batch_mean_prob`` — the entropy of the batch-averaged assignment
            distribution (per dim, then averaged over dims). HIGH when the batch as a whole
            spreads across many grid points (diverse); LOW when the batch collapses to one.
            Under DDP this batch-mean is GLOBAL: the local assignment sum is all-reduced
            across ranks (SUM) and normalized by the global sample count, so the diversity
            reward measures spread across the WHOLE global batch rather than each rank's 8
            local samples — 8 samples cannot represent codebook-wide usage, so a purely-local
            batch-mean gives a far-too-weak spread signal and the codec collapses (dead FSQ
            dim, min_dim_entropy≈0). See FIX 1.

        Returned combined term::

            per_sample_entropy_mean - diversity_weight * entropy_of_batch_mean_prob

        This is LOW when the batch uses many codes diversely (small per-sample entropy,
        large batch-mean entropy) and HIGH when the batch collapses to one code (per-sample
        entropy ~0 but batch-mean entropy also ~0, so the subtracted reward vanishes). The
        caller multiplies the returned value by ``cfg.entropy_weight``.

        Args:
            feats: (B, n_tok, d_model) PRE-FSQ encoder features (``codec.encode(x)``).
            beta:  softmax sharpness on the (level-index) grid distances.

        Returns:
            Scalar tensor, differentiable w.r.t. ``feats``.
        """
        cfg = self.cfg
        levels = list(cfg.fsq_levels)
        max_levels = max(levels)

        x = self.pre_quant_levels(feats)          # (B, n_tok, fsq_dim)
        x = x.reshape(-1, x.shape[-1])            # (N, fsq_dim)
        N, fsq_dim = x.shape

        # grid of level indices 0..max_levels-1, broadcast against x; pad unused entries
        # (for dims with fewer levels) with -inf logits so they get zero probability.
        grid = torch.arange(max_levels, device=x.device, dtype=x.dtype)      # (max_levels,)
        dist2 = (x[..., None] - grid[None, None, :]) ** 2                     # (N, fsq_dim, max_levels)
        logits = -beta * dist2

        lvl_t = torch.tensor(levels, device=x.device)                        # (fsq_dim,)
        valid = grid[None, :] < lvl_t[:, None]                               # (fsq_dim, max_levels) bool
        logits = logits.masked_fill(~valid[None, :, :], float("-inf"))

        p = torch.softmax(logits, dim=-1)                                    # (N, fsq_dim, max_levels)
        eps = 1e-9

        # per-sample entropy: -sum_g p log p, averaged over (N, fsq_dim).
        per_sample_entropy = -(p * torch.log(p + eps)).sum(dim=-1)           # (N, fsq_dim)
        per_sample_entropy_mean = per_sample_entropy.mean()

        # batch-mean distribution per dim, then its entropy, averaged over dims.
        # p_mean = p.sum(dim=0) / N. Under DDP each rank sees only its ~8 local samples, far
        # too few to represent codebook-wide usage, so a purely-local batch-mean gives a weak
        # spread signal and the codec collapses. FIX 1: make p_mean GLOBAL — all-reduce the
        # local assignment sum (SUM, autograd-compatible so gradients flow) and normalize by
        # the GLOBAL sample count. The count is a plain int (NOT a grad tensor); gradients
        # flow only through the summed probabilities. World_size==1 / uninitialized falls back
        # to the exact local p.sum(dim=0) / N — numerically identical to the previous code.
        p_sum = p.sum(dim=0)                                                 # (fsq_dim, max_levels)
        n_total = N
        world_size = 1
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            # SUM-reduce the probability sum so p_mean is GLOBAL. NOTE: dist.all_reduce
            # is NOT autograd-aware — the collective injects the other ranks' VALUES but
            # backward carries only this rank's contribution to p_sum (see FIX 2 below).
            dist.all_reduce(p_sum, op=dist.ReduceOp.SUM)
            # Reduce the (integer) local count separately as a plain scalar — NOT a grad tensor.
            n_tensor = torch.tensor([float(N)], device=p_sum.device)
            dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
            n_total = float(n_tensor.item())
        p_mean = p_sum / n_total                                            # (fsq_dim, max_levels)
        batch_entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=-1)      # (fsq_dim,)
        entropy_of_batch_mean = batch_entropy.mean()

        # FIX 2 (2026-08-02): restore the diversity GRADIENT scale under DDP. Because the
        # all_reduce above is not autograd-aware, each rank's backward holds only
        # ∂H(p̄_global)/∂p_local, and DDP then AVERAGES gradients across ranks — so the
        # effective diversity gradient is 1/world_size of the single-process global-batch
        # equivalent, while the per-sample confidence term keeps full strength (measured
        # consequence: the spread reward was 8x weaker than configured at world_size 8).
        # The identity w*H - (w-1)*H.detach() preserves the VALUE (logs stay comparable)
        # while scaling dH/dθ by w, exactly cancelling DDP's 1/w averaging.
        if world_size > 1:
            w = float(world_size)
            entropy_of_batch_mean = (
                w * entropy_of_batch_mean - (w - 1.0) * entropy_of_batch_mean.detach()
            )

        return per_sample_entropy_mean - cfg.diversity_weight * entropy_of_batch_mean
