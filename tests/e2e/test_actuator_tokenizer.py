"""§5.5 verification tests for :class:`ActuatorTokenizer`.

Run with::

    pixi run pytest tests/e2e/test_actuator_tokenizer.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.e2e.tokenizers.actuator import ActuatorTokenizer

N_CHANNELS = 4
WINDOW_SAMPLES = 60
N_TOKENS = 3
D_MODEL = 32


@pytest.fixture
def tokenizer() -> ActuatorTokenizer:
    torch.manual_seed(0)
    return ActuatorTokenizer(
        n_channels=N_CHANNELS,
        window_samples=WINDOW_SAMPLES,
        d_model=D_MODEL,
        n_tokens=N_TOKENS,
    )


def test_impulse_reaches_tokens(tokenizer: ActuatorTokenizer) -> None:
    """Impulse — active tokens differ from zero tokens by norm > 1.0.

    Critical check (§5.5): no LayerNorm after the Conv1d patching, otherwise
    the data-dependent signal is washed out relative to the learned
    embeddings and the difference collapses.
    """
    torch.manual_seed(1)
    x_zero = torch.zeros(1, N_CHANNELS, WINDOW_SAMPLES)
    x_active = torch.randn(1, N_CHANNELS, WINDOW_SAMPLES) * 5.0

    t_zero = tokenizer(x_zero)
    t_active = tokenizer(x_active)
    diff_norm = (t_active - t_zero).norm().item()
    assert diff_norm > 1.0, (
        f"Active-vs-zero actuator token diff norm {diff_norm:.3f} ≤ 1.0; "
        "signal is being erased (check for LayerNorm after patching)."
    )


def test_step_ramp_sinusoid_produce_different_tokens(
    tokenizer: ActuatorTokenizer,
) -> None:
    """Impulse — step, ramp, and sinusoid produce pairwise-different tokens."""
    t = torch.linspace(0.0, 1.0, WINDOW_SAMPLES)
    step = torch.ones(1, N_CHANNELS, WINDOW_SAMPLES)
    ramp = t.view(1, 1, -1).expand(1, N_CHANNELS, -1).contiguous()
    sinusoid = torch.sin(2 * math.pi * t).view(1, 1, -1).expand(
        1, N_CHANNELS, -1
    ).contiguous()

    outs = {name: tokenizer(x) for name, x in
            {"step": step, "ramp": ramp, "sinusoid": sinusoid}.items()}

    for a in outs:
        for b in outs:
            if a >= b:
                continue
            cos_sim = F.cosine_similarity(
                outs[a].flatten(), outs[b].flatten(), dim=0
            ).item()
            assert cos_sim < 0.95, (
                f"{a!r} and {b!r} tokens too similar (cos_sim={cos_sim:.3f})."
            )


def test_all_parameters_receive_gradient(
    tokenizer: ActuatorTokenizer,
) -> None:
    """Gradient — all parameters receive non-zero ``.grad``."""
    torch.manual_seed(2)
    x = torch.randn(2, N_CHANNELS, WINDOW_SAMPLES)
    tokens = tokenizer(x)
    tokens.sum().backward()
    for name, param in tokenizer.named_parameters():
        assert param.grad is not None, f"{name}: .grad is None"
        assert param.grad.abs().sum().item() > 0.0, f"{name}: .grad all zeros"


def test_time_offset_changes_output(tokenizer: ActuatorTokenizer) -> None:
    """Functional — different time offsets produce different outputs.

    Two sinusoids with a phase offset must produce distinguishable token
    stacks (cos_sim < 0.95).
    """
    t = torch.linspace(0.0, 2 * math.pi, WINDOW_SAMPLES)
    x_a = torch.sin(t).view(1, 1, -1).expand(1, N_CHANNELS, -1).contiguous()
    x_b = torch.sin(t + 0.7).view(1, 1, -1).expand(1, N_CHANNELS, -1).contiguous()

    t_a = tokenizer(x_a).flatten()
    t_b = tokenizer(x_b).flatten()
    cos_sim = F.cosine_similarity(t_a, t_b, dim=0).item()
    assert cos_sim < 0.95, (
        f"Phase-shifted sinusoids produced near-identical tokens "
        f"(cos_sim={cos_sim:.3f})."
    )
