"""FSQ bottleneck for the spectrogram codec.

Thin wrapper over ``vector_quantize_pytorch.FSQ``. FSQ projects the encoder's
``d_model`` features down to ``fsq_dim``, quantizes them with a straight-through
estimator and projects back up to ``d_model`` for the decoder. Phase B consumes the
integer per-dimension ``codes``; the continuous ``quant`` goes to the decoder and
carries the straight-through gradient back to the encoder. No FAITH model code is used.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from vector_quantize_pytorch import FSQ

from .config import SpectroCodecConfig

# Largest prod(fsq_levels) allowed with the joint-code entropy term. The term builds an
# (N, codebook_size) soft-assignment tensor (N = batch_size * n_tok, e.g. 1536): about
# 6 MB at 1000 codes, but about 390 MB per forward plus the graph at 64000 codes.
_JOINT_ENTROPY_MAX_CODEBOOK: int = 4096


class SpectroQuantizer(nn.Module):
    """FSQ bottleneck between (B, n_tok, d_model) features and per-dimension codes.

    ``quantize(feats)`` returns ``(quant, codes)``. ``quant`` is the quantized features
    projected back to ``d_model`` (float, straight-through differentiable in ``feats``).
    ``codes`` has shape ``(B, n_tok, fsq_dim)``, entry ``i`` in ``[0, fsq_levels[i])``.
    """

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # FSQ owns the d_model <-> fsq_dim projections and the straight-through
        # estimator. The video, slow-TS and fast-TS configs lack the two noise/symmetry
        # fields, hence getattr; the defaults are the library's own. The library asserts
        # preserve_symmetry whenever noise_dropout > 0, so check it here with a message.
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
        # quant: (B, n_tok, d_model), straight-through gradient to feats.
        # flat_indices: (B, n_tok) in [0, codebook_size).
        quant, flat_indices = self.fsq(feats)
        # Split the flat index into one level index per FSQ dimension, each in
        # [0, levels[i]). These are the codes Phase B sees.
        codes = self.fsq.indices_to_level_indices(flat_indices).long()
        return quant, codes

    def pre_quant_levels(self, feats: torch.Tensor) -> torch.Tensor:
        """Continuous level position of each FSQ dimension, shape (B, n_tok, fsq_dim).

        Repeats FSQ's ``project_in`` and ``bound`` without the straight-through
        ``round``, giving each dimension's position in level-index space
        ``[0, levels[i] - 1]``. Rounding it gives exactly the codes of :meth:`quantize`
        (checked against the library). The entropy regularizer works on this.

        The math follows ``FSQ.bound``: with ``half_l = (L-1)(1+eps)/2``,
        ``offset = 0.5`` if ``L`` is even else 0 and ``shift = atanh(offset/half_l)``,
        the bounded value is ``bz = tanh(z + shift) * half_l - offset``. FSQ then takes
        ``round(bz) / half_width``, and ``_scale_and_shift`` maps that back to the level
        index ``round(bz) + half_width``. Without the round this is ``bz + half_width``.
        """
        fsq = self.fsq
        # Same projection FSQ.forward uses.
        z = fsq.project_in(feats)  # (B, n_tok, fsq_dim)

        levels = fsq._levels.to(device=z.device, dtype=z.dtype)  # (fsq_dim,)

        # preserve_symmetry changes both FSQ's bound and its level<->code mapping, so
        # follow it here, or the regularizers would act on something other than the
        # code. FSQ.symmetry_preserving_bound gives floor((L-1)(tanh z + 1)/2 + 0.5),
        # and _scale_and_shift maps back to that bracket. Since floor(u + 0.5) ==
        # round(u), the continuous position is u itself.
        if fsq.preserve_symmetry:
            return (levels - 1) * (torch.tanh(z) + 1) / 2

        eps = 1e-3
        half_l = (levels - 1) * (1 + eps) / 2
        offset = torch.where(levels % 2 == 0, torch.tensor(0.5, dtype=z.dtype, device=z.device),
                             torch.tensor(0.0, dtype=z.dtype, device=z.device))
        shift = torch.atanh(offset / half_l)
        bounded_z = torch.tanh(z + shift) * half_l - offset
        half_width = torch.div(levels, 2, rounding_mode="floor")
        return bounded_z + half_width

    def entropy_loss(self, feats: torch.Tensor, beta: float = 10.0,
                     step: Optional[int] = None) -> torch.Tensor:
        """Differentiable codebook-utilization term. Returns a scalar.

        For each FSQ dimension ``i`` the continuous level position ``x``
        (:meth:`pre_quant_levels`) is soft-assigned to the grid ``g = 0..levels[i]-1``
        with ``softmax(-beta * (x - g)^2)``. This gives distributions ``p`` of shape
        ``(N, fsq_dim, max_levels)``, with the unused grid points of smaller dimensions
        masked out.

        Two Genie/LFQ-style terms:

        * ``per_sample_entropy_mean``: mean entropy of each sample's distribution. Low
          when the encoder commits to a code; on its own it does not prevent collapse.
        * ``entropy_of_batch_mean_prob``: entropy of the batch-mean distribution, per
          dimension and then averaged. High when the batch spreads over many grid
          points, low when it collapses. Under DDP the batch mean is global (the
          assignment sum is all-reduced and divided by the global count), because a
          rank's ~8 local samples cannot show codebook-wide usage and a local mean gave
          too weak a signal: the codec collapsed (a dead FSQ dimension).

        The returned value is::

            per_sample_entropy_mean - diversity_weight * entropy_of_batch_mean_prob
                                    - joint_entropy_weight * entropy_of_batch_mean_JOINT
                                    + decorrelation_weight * mean_offdiag_corr_squared

        It is low when the batch uses many codes and high when it collapses to one (then
        both entropies are ~0). The caller multiplies it by ``cfg.entropy_weight``.

        Joint-code diversity (``cfg.joint_entropy_weight``, default 0, off). The
        per-dimension reward is marginal: a rank-1 encoder maximizes it by spreading
        each dimension over its grid while using almost none of the joint codebook. The
        production mhr codec had min_dim_entropy 0.921 (93% of ln 8, so the term was
        saturated) but used 42 of 32768 joint codes, because its 5 per-dimension
        positions were almost perfectly correlated (|r| >= 0.997, effective rank 1.09).
        This term rewards the entropy of the batch-mean distribution over the full joint
        code space (MagViT-2 / LFQ codebook entropy). A sample's joint assignment is the
        outer product of its per-dimension assignments (exact, since they are
        independent given the position), an ``(N, prod(levels))`` tensor. Its batch mean
        is global under DDP with the same gradient rescale as the per-dimension term. It
        is the differentiable analogue of ``gate.utilization``'s ``frac_codes_used``,
        with maximum ``log(codebook_size)`` (6.908 nats at 1000 codes). It costs
        ``N * codebook_size`` floats, so it is refused above
        ``_JOINT_ENTROPY_MAX_CODEBOOK`` codes.

        Dimension decorrelation (``cfg.decorrelation_weight``, default 0, off).
        Penalizes rank-1 structure directly: the mean squared off-diagonal entry of the
        correlation matrix of the level positions over the batch, 0 for independent
        dimensions and ~1 for one replicated scalar. It is estimated on the local batch
        (e.g. 1536 token positions, plenty for an ``fsq_dim``-square matrix) and not
        all-reduced; DDP averages its gradients.

        Parameters
        ----------
        feats : (B, n_tok, d_model)
            Encoder features before FSQ (``codec.encode(x)``).
        beta : float
            Softmax sharpness on the grid distances.
        step : int, optional
            Current generator step, used only for ``cfg.joint_entropy_ramp_steps`` (a
            linear ramp of ``joint_entropy_weight`` from 0). ``None`` means no ramp.

        Returns
        -------
        torch.Tensor
            Scalar, differentiable in ``feats``.
        """
        cfg = self.cfg
        levels = list(cfg.fsq_levels)
        max_levels = max(levels)

        x = self.pre_quant_levels(feats)          # (B, n_tok, fsq_dim)
        x = x.reshape(-1, x.shape[-1])            # (N, fsq_dim)
        N, fsq_dim = x.shape

        # Grid of level indices; entries beyond a dimension's own level count get -inf
        # logits, so zero probability.
        grid = torch.arange(max_levels, device=x.device, dtype=x.dtype)      # (max_levels,)
        dist2 = (x[..., None] - grid[None, None, :]) ** 2                     # (N, fsq_dim, max_levels)
        logits = -beta * dist2

        lvl_t = torch.tensor(levels, device=x.device)                        # (fsq_dim,)
        valid = grid[None, :] < lvl_t[:, None]                               # (fsq_dim, max_levels) bool
        logits = logits.masked_fill(~valid[None, :, :], float("-inf"))

        p = torch.softmax(logits, dim=-1)                                    # (N, fsq_dim, max_levels)
        eps = 1e-9

        # Per-sample entropy, averaged over (N, fsq_dim).
        per_sample_entropy = -(p * torch.log(p + eps)).sum(dim=-1)           # (N, fsq_dim)
        per_sample_entropy_mean = per_sample_entropy.mean()

        # Entropy of the batch-mean distribution per dimension, averaged over
        # dimensions. Under DDP the sum is all-reduced and divided by the global sample
        # count (a plain number, not a grad tensor). With one process this is
        # p.sum(0) / N.
        p_sum = p.sum(dim=0)                                                 # (fsq_dim, max_levels)
        n_total = N
        world_size = 1
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            # all_reduce is not autograd-aware: it brings in the other ranks' values,
            # but backward only sees this rank's share of p_sum (rescaled below).
            dist.all_reduce(p_sum, op=dist.ReduceOp.SUM)
            n_tensor = torch.tensor([float(N)], device=p_sum.device)
            dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
            n_total = float(n_tensor.item())
        p_mean = p_sum / n_total                                            # (fsq_dim, max_levels)
        batch_entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=-1)      # (fsq_dim,)
        entropy_of_batch_mean = batch_entropy.mean()

        # Restore the gradient scale under DDP. Each rank's backward holds only
        # dH(p_global)/dp_local, and DDP averages gradients over ranks, so the diversity
        # gradient would be 1/world_size of the single-process value while the
        # per-sample term keeps full strength. w*H - (w-1)*H.detach() keeps the value
        # and scales the gradient by w.
        if world_size > 1:
            w = float(world_size)
            entropy_of_batch_mean = (
                w * entropy_of_batch_mean - (w - 1.0) * entropy_of_batch_mean.detach()
            )

        total = per_sample_entropy_mean - cfg.diversity_weight * entropy_of_batch_mean

        # Joint-code diversity (off by default). getattr because the video, slow-TS and
        # fast-TS configs lack these fields.
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
            # Joint assignment as the outer product over dimensions, (N, prod(levels)).
            pj = p[:, 0, : levels[0]]
            for i in range(1, fsq_dim):
                pj = (pj.unsqueeze(-1) * p[:, i, : levels[i]].unsqueeze(1)).reshape(N, -1)
            pj_sum = pj.sum(dim=0)                                           # (codebook_size,)
            n_total_j = N
            if world_size > 1:
                dist.all_reduce(pj_sum, op=dist.ReduceOp.SUM)
                n_total_j = n_total       # same global count as the per-dimension term
            pj_mean = pj_sum / n_total_j
            joint_entropy = -(pj_mean * torch.log(pj_mean + eps)).sum()
            if world_size > 1:
                w = float(world_size)
                joint_entropy = w * joint_entropy - (w - 1.0) * joint_entropy.detach()
            total = total - joint_w * joint_entropy

        # Decorrelation of the FSQ dimensions (off by default).
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
