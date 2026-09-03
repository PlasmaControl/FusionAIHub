"""Finite Scalar Quantization (Mentzer et al., 2023 — "VQ-VAE Made Simple").

FSQ replaces a learned VQ codebook with a fixed integer grid: each latent is
projected to ``len(levels)`` scalar dims, each bounded by tanh and rounded
(straight-through) to one of ``levels[d]`` values. The implicit codebook has
``prod(levels)`` entries but is never stored or trained — so there is **no
codebook collapse and no commitment loss to tune**, the two failure modes that
make classic VQ-VAE finicky.

Why we want it here: continuous regression heads (MAE / rectified-flow) collapse
spectrogram modes to the conditional mean. With FSQ, prediction becomes
*categorical* — the backbone predicts a distribution over code indices and we
**sample** — which cannot collapse to a mean and yields sharp, mode-bearing
output. See ``e2e/quantizers/__init__.py`` for the module export.

The math mirrors the reference implementation:
  bound(z)   = tanh(z + shift) * half_l - offset      (maps R -> the round region)
  quantize   = round_ste(bound(z)) / half_width        (normalized code in ~[-1,1])
  index      = sum_d shifted_code_d * basis_d          (pack per-dim -> one int)
"""
import math
from typing import List

import torch
import torch.nn as nn


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round to nearest integer with a straight-through gradient (grad == 1)."""
    return z + (z.round() - z).detach()


class FSQ(nn.Module):
    """Finite Scalar Quantizer.

    Args:
        levels: per-dimension number of quantization levels, e.g.
            ``[8, 8, 8, 5, 5, 5]`` → 6 dims, codebook size 8·8·8·5·5·5 = 24000.

    forward(z): z is ``(..., dim)`` continuous → returns ``(codes, indices)``
        codes:   ``(..., dim)`` float, straight-through quantized, ~[-1, 1]
                 (feed to the decoder projection).
        indices: ``(...)`` int64 in ``[0, codebook_size)`` (cross-entropy target).
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        levels = [int(x) for x in levels]
        _levels = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("_levels", _levels, persistent=False)
        # basis to pack per-dim integer codes into a single flat index
        basis = torch.cumprod(
            torch.tensor([1] + levels[:-1], dtype=torch.float32), dim=0)
        self.register_buffer("_basis", basis, persistent=False)
        self.dim = len(levels)
        self.levels_list = levels
        # Python int product (float32 torch.prod overflows past ~a few dims).
        self.codebook_size = math.prod(levels)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        lv = self._levels.to(z.device, z.dtype)
        half_l = (lv - 1) * (1 + eps) / 2
        offset = torch.where(lv % 2 == 0,
                             torch.full_like(lv, 0.5), torch.zeros_like(lv))
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """z (...,dim) -> normalized STE-quantized codes ~[-1,1]."""
        q = round_ste(self.bound(z))
        half_width = (self._levels // 2).to(z.device, z.dtype)
        return q / half_width

    def _to_int_codes(self, codes_norm: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels // 2).to(codes_norm.device, codes_norm.dtype)
        return (codes_norm * half_width + half_width)          # -> [0, L-1]

    def _from_int_codes(self, codes_int: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels // 2).to(codes_int.device, codes_int.dtype)
        return (codes_int - half_width) / half_width           # -> ~[-1,1]

    def codes_to_int(self, codes_norm: torch.Tensor) -> torch.Tensor:
        """Per-dim integer codes (..., dim), each in [0, levels[d]). This is the
        overflow-safe Stage-B target (predict each dim independently). Prefer
        this over the flat index for large codebooks."""
        return self._to_int_codes(codes_norm).round().to(torch.long)

    def codes_to_indices(self, codes_norm: torch.Tensor) -> torch.Tensor:
        """Flat index sum_d code_d * basis_d. Only valid when codebook_size fits
        int64 (~prod(levels) < 9.2e18); overflows silently otherwise — use
        codes_to_int for large codebooks."""
        ci = self._to_int_codes(codes_norm)
        basis = self._basis.to(codes_norm.device, codes_norm.dtype)
        return (ci * basis).sum(dim=-1).round().to(torch.long)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (...) int -> normalized codes (..., dim)."""
        idx = indices.unsqueeze(-1).to(torch.float32)
        lv = self._levels.to(indices.device)
        basis = self._basis.to(indices.device)
        codes_int = torch.floor(idx / basis) % lv
        return self._from_int_codes(codes_int)

    def forward(self, z: torch.Tensor):
        """z (..., dim) -> (codes, per_dim_int_codes). codes are the normalized
        STE-quantized values (feed the decoder); per_dim_int_codes (..., dim)
        are the Stage-B cross-entropy targets (predict each dim independently,
        overflow-safe for any codebook size)."""
        codes = self.quantize(z)
        return codes, self.codes_to_int(codes)


class FSQBottleneck(nn.Module):
    """Discrete FSQ bottleneck: project encoder tokens to FSQ dims, quantize
    (straight-through), project back to d_model.

    tokens (B, n_tok, d_model) -> (tokens_q, per_dim_codes)
      tokens_q      (B, n_tok, d_model) — feed the decoder
      per_dim_codes (B, n_tok, dim), int in [0, levels[d]) — Stage-B CE targets

    Helpers for Stage B (predict codes → decode): ``codes_to_tokens`` maps
    predicted per-dim integer codes back through the (frozen) output projection
    to d_model tokens the decoder consumes.
    """

    def __init__(self, d_model: int, levels: List[int]):
        super().__init__()
        self.fsq = FSQ(levels)
        self.proj_in = nn.Linear(d_model, self.fsq.dim)
        self.proj_out = nn.Linear(self.fsq.dim, d_model)
        self.codebook_size = self.fsq.codebook_size
        self.dim = self.fsq.dim

    def forward(self, tokens: torch.Tensor):
        codes, idx = self.fsq(self.proj_in(tokens))
        return self.proj_out(codes), idx

    def codes_to_tokens(self, per_dim_codes: torch.Tensor) -> torch.Tensor:
        """per_dim int codes (B, n_tok, dim) -> decoder tokens (B, n_tok, d_model).
        Used at Stage-B inference: sample codes → normalized codes → proj_out."""
        norm = self.fsq._from_int_codes(per_dim_codes.float())
        return self.proj_out(norm)
