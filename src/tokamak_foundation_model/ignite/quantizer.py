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

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from vector_quantize_pytorch import FSQ

from .config import SpectroCodecConfig

# Upper bound on prod(fsq_levels) for which the JOINT-code entropy reward is allowed. The term
# materializes an (N, codebook_size) soft-assignment tensor (N = batch_size * n_tok, e.g. 1536),
# so cb=1000 costs ~6 MB but the 64000-code fsq6 configs would cost ~390 MB per forward plus the
# autograd graph. Refuse loudly rather than silently OOM mid-run.
_JOINT_ENTROPY_MAX_CODEBOOK: int = 4096


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
        # FSQ-native anti-collapse knobs. getattr keeps the video / slow-TS / fast-TS
        # configs (which do not carry these fields) constructing exactly as before, and the
        # defaults (0.0 / False) are the library's own, so the FSQ is byte-identical unless a
        # run explicitly asks for them. The library asserts preserve_symmetry whenever
        # noise_dropout > 0; raise that here with the reason instead of a bare AssertionError.
        noise_dropout = float(getattr(cfg, "fsq_noise_dropout", 0.0))
        preserve_symmetry = bool(getattr(cfg, "fsq_preserve_symmetry", False))
        if noise_dropout > 0.0 and not preserve_symmetry:
            raise ValueError(
                f"fsq_noise_dropout={noise_dropout} requires fsq_preserve_symmetry=True "
                "(vector_quantize_pytorch.FSQ asserts this). preserve_symmetry ALSO changes "
                "the emitted codes on its own, so run a preserve_symmetry-only control "
                "alongside or the effect is unattributable."
            )
        self.fsq = FSQ(
            levels=list(cfg.fsq_levels),
            dim=cfg.d_model,
            channel_first=False,
            return_indices=True,
            noise_dropout=noise_dropout,
            preserve_symmetry=preserve_symmetry,
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

        # `cfg.fsq_preserve_symmetry` swaps FSQ's bounding function AND its level<->code
        # mapping, so this replication has to follow it or the entropy / joint-entropy
        # regularizers would be shaped on a quantity that is NOT the emitted code.
        # FSQ.symmetry_preserving_bound: bracket = floor((L-1)(tanh z + 1)/2 + 0.5), and
        # _scale_and_shift then maps the normalized code back to exactly that bracket. Since
        # floor(u + 0.5) == round(u), the continuous level position is u itself:
        if fsq.preserve_symmetry:
            return (levels - 1) * (torch.tanh(z) + 1) / 2

        eps = 1e-3
        half_l = (levels - 1) * (1 + eps) / 2
        offset = torch.where(levels % 2 == 0, torch.tensor(0.5, dtype=z.dtype, device=z.device),
                             torch.tensor(0.0, dtype=z.dtype, device=z.device))
        shift = torch.atanh(offset / half_l)
        bounded_z = torch.tanh(z + shift) * half_l - offset
        half_width = torch.div(levels, 2, rounding_mode="floor")
        # continuous level-index position; round() recovers the integer code.
        return bounded_z + half_width

    def entropy_loss(self, feats: torch.Tensor, beta: float = 10.0,
                     step: Optional[int] = None) -> torch.Tensor:
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
                                    - joint_entropy_weight * entropy_of_batch_mean_JOINT
                                    + decorrelation_weight * mean_offdiag_corr_squared

        This is LOW when the batch uses many codes diversely (small per-sample entropy,
        large batch-mean entropy) and HIGH when the batch collapses to one code (per-sample
        entropy ~0 but batch-mean entropy also ~0, so the subtracted reward vanishes). The
        caller multiplies the returned value by ``cfg.entropy_weight``.

        JOINT-code diversity (``cfg.joint_entropy_weight``, default 0.0 = OFF, byte-identical).
        The ``entropy_of_batch_mean_prob`` reward above is PER-DIMENSION and MARGINAL: it is
        maximized by spreading each dim over its own grid independently of the others, which a
        RANK-1 encoder does perfectly while using almost none of the joint codebook. Measured on
        the production mhr codec (2026-09-02): min_dim_entropy 0.921 — 93% of its ln 8 ceiling,
        so this term is SATURATED and raising ``entropy_weight`` can buy at most 0.15 nats — yet
        only 42 of 32768 joint codes were used, because the 5 per-dim pre-quant level positions
        were near-perfectly correlated (|r| >= 0.997; top covariance eigenvalue 99.9% of the
        variance; encoder features participation-ratio effective rank 1.09). When
        ``joint_entropy_weight > 0`` we additionally reward the entropy of the batch-mean
        distribution over the FULL joint code space (MagViT-2 / LFQ "codebook entropy"). The
        per-sample joint soft assignment is the outer product of the per-dim soft assignments
        (the per-dim posteriors are independent given the pre-quant position, so this is exact),
        giving an ``(N, prod(levels))`` tensor; its batch mean is GLOBAL under DDP exactly like
        the per-dim term, with the same value-preserving gradient rescale (FIX 2). This is the
        differentiable analogue of ``gate.utilization``'s ``frac_codes_used``; its maximum is
        ``log(codebook_size)`` (6.908 nats at cb=1000). Cost is ``N * codebook_size`` floats, so
        it is refused above ``_JOINT_ENTROPY_MAX_CODEBOOK`` codes.

        DIM DECORRELATION (``cfg.decorrelation_weight``, default 0.0 = OFF, byte-identical).
        A direct penalty on the rank-1 structure itself: the mean squared OFF-DIAGONAL entry of
        the correlation matrix of the continuous pre-quant level positions across the batch. 0
        when the FSQ dims carry independent information, ~1 when they are one scalar replicated.
        Estimated on the LOCAL batch (1536 token positions at batch_size 8 / 192 tokens — ample
        for a ``fsq_dim``-square correlation matrix); it is not all-reduced, so under DDP it is
        a per-rank estimate whose gradients DDP averages.

        Args:
            feats: (B, n_tok, d_model) PRE-FSQ encoder features (``codec.encode(x)``).
            beta:  softmax sharpness on the (level-index) grid distances.
            step:  current generator step; only used to apply
                   ``cfg.joint_entropy_ramp_steps`` (a linear 0 -> ``joint_entropy_weight``
                   ramp). ``None`` (every non-spectro caller) means no ramp.

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

        total = per_sample_entropy_mean - cfg.diversity_weight * entropy_of_batch_mean

        # ---- JOINT-code diversity reward (OFF by default; see the docstring) ------------
        # getattr so the video / slow-TS / fast-TS configs, which do NOT carry these fields,
        # keep the exact previous behavior.
        joint_w = float(getattr(cfg, "joint_entropy_weight", 0.0))
        ramp = int(getattr(cfg, "joint_entropy_ramp_steps", 0) or 0)
        if joint_w != 0.0 and ramp > 0 and step is not None:
            joint_w *= min(1.0, float(step) / float(ramp))
        if joint_w != 0.0:
            codebook_size = 1
            for lv in levels:
                codebook_size *= int(lv)
            if codebook_size > _JOINT_ENTROPY_MAX_CODEBOOK:
                raise ValueError(
                    f"joint_entropy_weight={joint_w} needs an (N, codebook_size) soft-assignment "
                    f"tensor, but codebook_size={codebook_size} (fsq_levels={levels}) exceeds the "
                    f"{_JOINT_ENTROPY_MAX_CODEBOOK}-code limit. Use a smaller fsq_levels product "
                    f"(e.g. [8,5,5,5] = 1000) or set joint_entropy_weight=0."
                )
            # per-sample joint soft assignment = outer product over dims (exact: the per-dim
            # posteriors are independent given the pre-quant position). (N, prod(levels)).
            pj = p[:, 0, : levels[0]]
            for i in range(1, fsq_dim):
                pj = (pj.unsqueeze(-1) * p[:, i, : levels[i]].unsqueeze(1)).reshape(N, -1)
            pj_sum = pj.sum(dim=0)                                           # (codebook_size,)
            n_total_j = N
            if world_size > 1:
                dist.all_reduce(pj_sum, op=dist.ReduceOp.SUM)
                n_total_j = n_total       # same global sample count as the per-dim term
            pj_mean = pj_sum / n_total_j
            joint_entropy = -(pj_mean * torch.log(pj_mean + eps)).sum()
            if world_size > 1:
                w = float(world_size)
                joint_entropy = w * joint_entropy - (w - 1.0) * joint_entropy.detach()
            total = total - joint_w * joint_entropy

        # ---- pre-quant dimension DECORRELATION penalty (OFF by default) -----------------
        decor_w = float(getattr(cfg, "decorrelation_weight", 0.0))
        if decor_w != 0.0 and fsq_dim > 1:
            xc = x - x.mean(dim=0, keepdim=True)                             # (N, fsq_dim)
            sd = xc.pow(2).mean(dim=0).clamp_min(1e-8).sqrt()
            xn = xc / sd
            corr = (xn.t() @ xn) / float(N)                                  # (fsq_dim, fsq_dim)
            off = corr - torch.diag_embed(torch.diagonal(corr))
            decorrelation = off.pow(2).sum() / float(fsq_dim * (fsq_dim - 1))
            total = total + decor_w * decorrelation

        return total
