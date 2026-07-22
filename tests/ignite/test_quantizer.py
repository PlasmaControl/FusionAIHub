"""TDD spec for ignite.quantizer.SpectroQuantizer.

Small synthetic CPU-only tensors. Contract (config.py):

    SpectroQuantizer(cfg)
        .quantize(feats: (B,n_tok,d_model)) -> (quant: (B,n_tok,d_model),
                                                codes: (B,n_tok,fsq_dim) long)
        .codebook_size
    - codes[..., i] in [0, fsq_levels[i])
    - straight-through gradient flows from quant back to feats
"""
from __future__ import annotations

import torch

from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.quantizer import SpectroQuantizer


def _small_cfg() -> SpectroCodecConfig:
    # keep tiny + cheap; only the FSQ / d_model path is exercised here.
    return SpectroCodecConfig(d_model=32, fsq_levels=[4, 4, 3])


def test_shapes() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    B, n_tok = 2, cfg.n_tok
    feats = torch.randn(B, n_tok, cfg.d_model)
    quant, codes = q.quantize(feats)
    assert quant.shape == (B, n_tok, cfg.d_model)
    assert codes.shape == (B, n_tok, cfg.fsq_dim)


def test_codes_are_long() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(2, cfg.n_tok, cfg.d_model)
    _, codes = q.quantize(feats)
    assert codes.dtype == torch.long


def test_codes_in_range() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    # large batch of extreme feats to push FSQ to its bounds.
    feats = torch.randn(8, cfg.n_tok, cfg.d_model) * 10.0
    _, codes = q.quantize(feats)
    levels = torch.tensor(cfg.fsq_levels)
    assert (codes >= 0).all()
    assert (codes < levels).all()  # per-dim upper bound


def test_codebook_size() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    assert q.codebook_size == cfg.codebook_size == 4 * 4 * 3


def test_straight_through_gradient() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(2, cfg.n_tok, cfg.d_model, requires_grad=True)
    quant, _ = q.quantize(feats)
    quant.sum().backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
    assert feats.grad.abs().sum() > 0  # gradient actually reached the input


def test_quant_is_differentiable_continuous() -> None:
    # quant must be a float tensor in d_model space (decoder-facing), not indices.
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(2, cfg.n_tok, cfg.d_model)
    quant, _ = q.quantize(feats)
    assert quant.dtype == torch.float32


# --------------------------------------------------------------------------- #
# anti-collapse entropy_loss
# --------------------------------------------------------------------------- #
def _collapsed_feats(q: SpectroQuantizer, B: int = 16) -> torch.Tensor:
    """Every sample identical -> the encoder maps the whole batch to ONE code."""
    n_tok = q.cfg.n_tok
    one = torch.randn(1, 1, q.cfg.d_model)
    return one.expand(B, n_tok, q.cfg.d_model).contiguous()


def _diverse_feats(q: SpectroQuantizer, B: int = 16) -> torch.Tensor:
    """Large-variance random feats that spread across the FSQ grid -> many codes."""
    n_tok = q.cfg.n_tok
    return torch.randn(B, n_tok, q.cfg.d_model) * 5.0


def test_pre_quant_levels_round_recovers_codes() -> None:
    """round(pre_quant_levels) must equal the integer codes from quantize()."""
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(4, cfg.n_tok, cfg.d_model) * 3.0
    _, codes = q.quantize(feats)
    cont = q.pre_quant_levels(feats)
    assert cont.shape == codes.shape
    levels = torch.tensor(cfg.fsq_levels)
    recovered = cont.round().clamp(min=torch.zeros_like(levels), max=levels - 1).long()
    assert torch.equal(recovered, codes)


def test_entropy_loss_returns_scalar() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    loss = q.entropy_loss(_diverse_feats(q))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_entropy_loss_high_for_collapse_low_for_diverse() -> None:
    """Collapse (one code) => HIGH loss; diverse batch (many codes) => LOW loss.

    Correctness of the anti-collapse term: it must penalise the trivial encoder≡const
    minimum (which uses one code) more than a batch that spreads across the grid.
    """
    torch.manual_seed(0)
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    loss_collapsed = q.entropy_loss(_collapsed_feats(q))
    loss_diverse = q.entropy_loss(_diverse_feats(q))
    assert float(loss_collapsed) > float(loss_diverse)


def test_entropy_loss_is_differentiable() -> None:
    """Gradient of entropy_loss must reach the input feats."""
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(8, cfg.n_tok, cfg.d_model, requires_grad=True)
    q.entropy_loss(feats).backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
    assert feats.grad.abs().sum() > 0
