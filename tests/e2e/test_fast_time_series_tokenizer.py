"""§5.2 verification tests for :class:`FastTimeSeriesTokenizer`.

Run with::

    pixi run pytest tests/e2e/test_fast_time_series_tokenizer.py -v
"""

import pytest
import torch

from tokamak_foundation_model.e2e.tokenizers.fast_time_series import (
    FastTimeSeriesTokenizer,
)

N_CHANNELS = 8
WINDOW_SAMPLES = 500
PATCH_SIZE = 50
N_PATCHES = WINDOW_SAMPLES // PATCH_SIZE  # 10
D_MODEL = 32
TOTAL_TOKENS = N_CHANNELS * N_PATCHES  # 80


@pytest.fixture
def tokenizer() -> FastTimeSeriesTokenizer:
    torch.manual_seed(0)
    return FastTimeSeriesTokenizer(
        n_channels=N_CHANNELS,
        window_samples=WINDOW_SAMPLES,
        d_model=D_MODEL,
        patch_size=PATCH_SIZE,
    )


def test_step_vs_ramp_produce_different_tokens(
    tokenizer: FastTimeSeriesTokenizer,
) -> None:
    """Impulse — step vs ramp.

    Constant 1.0 vs linearly increasing in ``[0, 1]``. Total token-difference
    norm must exceed 1.0. Failure mode: dead Conv1d or signal-killing
    normalization erasing absolute-value information.
    """
    step = torch.ones(1, N_CHANNELS, WINDOW_SAMPLES)
    ramp_1d = torch.linspace(0.0, 1.0, WINDOW_SAMPLES)
    ramp = ramp_1d.view(1, 1, -1).expand(1, N_CHANNELS, -1).contiguous()

    t_step = tokenizer(step)
    t_ramp = tokenizer(ramp)
    diff_norm = (t_step - t_ramp).norm().item()
    assert diff_norm > 1.0, (
        f"Step-vs-ramp token difference norm {diff_norm:.3f} ≤ 1.0; "
        "Conv1d may be dead or normalization is erasing the signal."
    )


def test_temporal_localization(tokenizer: FastTimeSeriesTokenizer) -> None:
    """Impulse — temporal localization.

    Zero the input, then inject a strong impulse into one patch of one
    channel. The token for ``(channel, patch)`` must have the highest norm
    across all 80 tokens. Failure mode: Conv1d stride/padding misconfigured.
    """
    torch.manual_seed(1)
    x = torch.zeros(1, N_CHANNELS, WINDOW_SAMPLES)
    active_channel = 3
    active_patch = 6
    t0 = active_patch * PATCH_SIZE
    x[0, active_channel, t0 : t0 + PATCH_SIZE] = torch.randn(PATCH_SIZE) * 5.0

    tokens = tokenizer(x)
    # Channel-major layout: flat_index = channel * n_patches + patch
    expected_index = active_channel * N_PATCHES + active_patch
    norms = tokens[0].norm(dim=-1)
    argmax = norms.argmax().item()
    assert argmax == expected_index, (
        f"Expected token {expected_index} (channel={active_channel}, "
        f"patch={active_patch}) to dominate; got token {argmax} with "
        f"norm {norms[argmax].item():.3f} vs expected norm "
        f"{norms[expected_index].item():.3f}."
    )


def test_conv_weights_receive_gradient(
    tokenizer: FastTimeSeriesTokenizer,
) -> None:
    """Gradient — Conv1d weights receive non-zero ``.grad``."""
    torch.manual_seed(2)
    x = torch.randn(2, N_CHANNELS, WINDOW_SAMPLES)
    tokens = tokenizer(x)
    tokens.sum().backward()
    grad = tokenizer.conv.weight.grad
    assert grad is not None, "conv.weight.grad is None"
    assert grad.abs().sum().item() > 0.0, "conv.weight.grad is all zeros"


def test_output_token_count(tokenizer: FastTimeSeriesTokenizer) -> None:
    """Shape — ``n_samples // patch_size`` tokens per channel."""
    x = torch.randn(3, N_CHANNELS, WINDOW_SAMPLES)
    tokens = tokenizer(x)
    assert tokens.shape == (3, TOTAL_TOKENS, D_MODEL), (
        f"Expected (3, {TOTAL_TOKENS}, {D_MODEL}); got {tuple(tokens.shape)}."
    )


def test_zero_input_produces_no_nan(
    tokenizer: FastTimeSeriesTokenizer,
) -> None:
    """Numerical — no NaN with zero input."""
    x = torch.zeros(1, N_CHANNELS, WINDOW_SAMPLES)
    tokens = tokenizer(x)
    assert torch.isfinite(tokens).all(), "Zero input produced NaN or Inf tokens."
