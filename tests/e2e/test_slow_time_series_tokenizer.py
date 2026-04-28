"""§5.1 verification tests for :class:`SlowTimeSeriesTokenizer`.

Run with::

    pixi run pytest tests/e2e/test_slow_time_series_tokenizer.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.e2e.tokenizers.slow_time_series import (
    SlowTimeSeriesTokenizer,
)

N_CHANNELS = 15
WINDOW_SAMPLES = 5
D_MODEL = 32


@pytest.fixture
def tokenizer() -> SlowTimeSeriesTokenizer:
    torch.manual_seed(0)
    return SlowTimeSeriesTokenizer(
        n_channels=N_CHANNELS,
        window_samples=WINDOW_SAMPLES,
        d_model=D_MODEL,
    )


def test_impulse_reaches_tokens(tokenizer: SlowTimeSeriesTokenizer) -> None:
    """Impulse — input reaches tokens.

    Zero all channels except one (randn(5) * 5.0). The active-channel token
    must have norm > 2× the mean norm of zero-channel tokens. Failure mode:
    dead projection or learned embeddings dominating the input signal.
    """
    torch.manual_seed(1)
    x = torch.zeros(1, N_CHANNELS, WINDOW_SAMPLES)
    active = 7
    x[0, active] = torch.randn(WINDOW_SAMPLES) * 5.0

    tokens = tokenizer(x)  # (1, C, D)
    norms = tokens[0].norm(dim=-1)
    mask = torch.arange(N_CHANNELS) != active
    ratio = (norms[active] / norms[mask].mean()).item()
    assert ratio > 2.0, (
        f"Active-channel token norm {norms[active].item():.3f} is not > 2× "
        f"zero-channel mean {norms[mask].mean().item():.3f} (ratio={ratio:.3f})."
    )


def test_different_inputs_produce_different_tokens(
    tokenizer: SlowTimeSeriesTokenizer,
) -> None:
    """Impulse — different inputs → different tokens.

    Two independent random inputs must yield token stacks with cosine
    similarity below 0.95. Failure mode: learned embeddings dominate so the
    output is nearly input-independent.
    """
    torch.manual_seed(2)
    x1 = torch.randn(1, N_CHANNELS, WINDOW_SAMPLES)
    x2 = torch.randn(1, N_CHANNELS, WINDOW_SAMPLES)
    t1 = tokenizer(x1).flatten()
    t2 = tokenizer(x2).flatten()
    cos_sim = F.cosine_similarity(t1, t2, dim=0).item()
    assert cos_sim < 0.95, (
        f"Tokens for different inputs too similar (cos_sim={cos_sim:.3f} ≥ 0.95); "
        "learned embeddings likely dominate the signal projection."
    )


def test_projection_weights_receive_gradient(
    tokenizer: SlowTimeSeriesTokenizer,
) -> None:
    """Gradient — projection weights receive non-zero ``.grad``."""
    torch.manual_seed(3)
    x = torch.randn(2, N_CHANNELS, WINDOW_SAMPLES)
    tokens = tokenizer(x)
    tokens.sum().backward()
    grad = tokenizer.proj.weight.grad
    assert grad is not None, "proj.weight.grad is None"
    assert grad.abs().sum().item() > 0.0, "proj.weight.grad is all zeros"


def test_output_token_count_equals_n_channels(
    tokenizer: SlowTimeSeriesTokenizer,
) -> None:
    """Shape — output has one token per channel."""
    x = torch.randn(3, N_CHANNELS, WINDOW_SAMPLES)
    tokens = tokenizer(x)
    assert tokens.shape == (3, N_CHANNELS, D_MODEL), (
        f"Expected (3, {N_CHANNELS}, {D_MODEL}); got {tuple(tokens.shape)}."
    )