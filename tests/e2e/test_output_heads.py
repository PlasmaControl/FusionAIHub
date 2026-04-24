"""§5.7 verification tests for per-modality output heads.

Three tests per head type: shape, gradient-to-backbone, and reconstruction
loss drops >50% in 100 training steps with tokenizer+backbone frozen.

Run with::

    pixi run pytest tests/e2e/test_output_heads.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokamak_foundation_model.e2e.backbone import SharedBackbone
from tokamak_foundation_model.e2e.output_heads import (
    FastTimeSeriesHead,
    SlowTimeSeriesHead,
)
from tokamak_foundation_model.e2e.tokenizers.fast_time_series import (
    FastTimeSeriesTokenizer,
)
from tokamak_foundation_model.e2e.tokenizers.slow_time_series import (
    SlowTimeSeriesTokenizer,
)

D_MODEL = 32
SLOW_CHANNELS = 15
SLOW_SAMPLES = 5
FAST_CHANNELS = 8
FAST_SAMPLES = 500
FAST_PATCH = 50
BATCH = 4


# ── Slow TS head ──────────────────────────────────────────────────────────


def test_slow_head_output_shape() -> None:
    torch.manual_seed(0)
    head = SlowTimeSeriesHead(D_MODEL, SLOW_CHANNELS, SLOW_SAMPLES)
    tokens = torch.randn(3, SLOW_CHANNELS, D_MODEL)
    out = head(tokens)
    assert out.shape == (3, SLOW_CHANNELS, SLOW_SAMPLES), (
        f"Expected (3, {SLOW_CHANNELS}, {SLOW_SAMPLES}); got {tuple(out.shape)}."
    )


def test_slow_head_gradient_flows_to_backbone_tokens() -> None:
    """Loss backprop must produce non-zero gradients on the upstream tokens."""
    torch.manual_seed(1)
    head = SlowTimeSeriesHead(D_MODEL, SLOW_CHANNELS, SLOW_SAMPLES)
    tokens = torch.randn(2, SLOW_CHANNELS, D_MODEL, requires_grad=True)
    target = torch.randn(2, SLOW_CHANNELS, SLOW_SAMPLES)
    F.mse_loss(head(tokens), target).backward()
    assert tokens.grad is not None and tokens.grad.abs().sum().item() > 0.0


def test_slow_head_reconstruction_loss_decreases(tmp_path) -> None:
    """§5.7 reconstruction — loss drops >50% in 100 head-only training steps.

    Tokenizer + backbone are random-init and frozen; only the head learns.
    """
    torch.manual_seed(2)
    tokenizer = SlowTimeSeriesTokenizer(SLOW_CHANNELS, SLOW_SAMPLES, D_MODEL)
    backbone = SharedBackbone(
        d_model=D_MODEL, n_heads=4, n_layers=2, dropout=0.0
    )
    head = SlowTimeSeriesHead(D_MODEL, SLOW_CHANNELS, SLOW_SAMPLES)
    _freeze(tokenizer)
    _freeze(backbone)

    target = torch.randn(BATCH, SLOW_CHANNELS, SLOW_SAMPLES)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    initial = _slow_loss(tokenizer, backbone, head, target).item()
    for _ in range(100):
        opt.zero_grad()
        loss = _slow_loss(tokenizer, backbone, head, target)
        loss.backward()
        opt.step()
    final = loss.item()
    assert final < 0.5 * initial, (
        f"Slow head reconstruction did not halve: {initial:.4f} → {final:.4f}."
    )


def _slow_loss(
    tokenizer: SlowTimeSeriesTokenizer,
    backbone: SharedBackbone,
    head: SlowTimeSeriesHead,
    target: torch.Tensor,
) -> torch.Tensor:
    tokens = tokenizer(target)
    step = torch.zeros(target.shape[0], dtype=torch.long)
    time = torch.zeros(target.shape[0])
    out = backbone(tokens, step, time)
    pred = head(out)
    return F.mse_loss(pred, target)


# ── Fast TS head ──────────────────────────────────────────────────────────


def test_fast_head_output_shape() -> None:
    torch.manual_seed(3)
    head = FastTimeSeriesHead(D_MODEL, FAST_CHANNELS, FAST_SAMPLES, FAST_PATCH)
    n_patches = FAST_SAMPLES // FAST_PATCH
    tokens = torch.randn(3, FAST_CHANNELS * n_patches, D_MODEL)
    out = head(tokens)
    assert out.shape == (3, FAST_CHANNELS, FAST_SAMPLES), (
        f"Expected (3, {FAST_CHANNELS}, {FAST_SAMPLES}); got {tuple(out.shape)}."
    )


def test_fast_head_gradient_flows_to_backbone_tokens() -> None:
    torch.manual_seed(4)
    head = FastTimeSeriesHead(D_MODEL, FAST_CHANNELS, FAST_SAMPLES, FAST_PATCH)
    n_patches = FAST_SAMPLES // FAST_PATCH
    tokens = torch.randn(
        2, FAST_CHANNELS * n_patches, D_MODEL, requires_grad=True
    )
    target = torch.randn(2, FAST_CHANNELS, FAST_SAMPLES)
    F.mse_loss(head(tokens), target).backward()
    assert tokens.grad is not None and tokens.grad.abs().sum().item() > 0.0


def test_fast_head_reconstruction_loss_decreases() -> None:
    """§5.7 reconstruction — loss drops >50% in 100 head-only training steps."""
    torch.manual_seed(5)
    tokenizer = FastTimeSeriesTokenizer(
        FAST_CHANNELS, FAST_SAMPLES, D_MODEL, FAST_PATCH
    )
    backbone = SharedBackbone(
        d_model=D_MODEL, n_heads=4, n_layers=2, dropout=0.0
    )
    head = FastTimeSeriesHead(D_MODEL, FAST_CHANNELS, FAST_SAMPLES, FAST_PATCH)
    _freeze(tokenizer)
    _freeze(backbone)

    target = torch.randn(BATCH, FAST_CHANNELS, FAST_SAMPLES)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    initial = _fast_loss(tokenizer, backbone, head, target).item()
    for _ in range(100):
        opt.zero_grad()
        loss = _fast_loss(tokenizer, backbone, head, target)
        loss.backward()
        opt.step()
    final = loss.item()
    assert final < 0.5 * initial, (
        f"Fast head reconstruction did not halve: {initial:.4f} → {final:.4f}."
    )


def _fast_loss(
    tokenizer: FastTimeSeriesTokenizer,
    backbone: SharedBackbone,
    head: FastTimeSeriesHead,
    target: torch.Tensor,
) -> torch.Tensor:
    tokens = tokenizer(target)
    step = torch.zeros(target.shape[0], dtype=torch.long)
    time = torch.zeros(target.shape[0])
    out = backbone(tokens, step, time)
    pred = head(out)
    return F.mse_loss(pred, target)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()