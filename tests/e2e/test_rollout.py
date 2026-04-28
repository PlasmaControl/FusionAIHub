"""§5.9 random-init rollout tests for :class:`TokenSpaceRollout`.

Three random-init tests (``Before Stage 1`` gate):
  - consecutive steps differ,
  - no norm explosion over 80 steps,
  - no norm collapse over 80 steps.

Trained-model tests (copy baseline, fixed-point after training, model vs gt
cos_sim, actuator sensitivity) gate cluster submission and are not run here.

Run with::

    pixi run pytest tests/e2e/test_rollout.py -v
"""

from typing import Dict, List

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout

# ── Small Phase-A-style config ────────────────────────────────────────────

DIAGS = [
    DiagnosticConfig("ts_core_temp", "slow_ts", n_channels=15, window_samples=5),
    DiagnosticConfig(
        "filterscopes",
        "fast_ts",
        n_channels=4,
        window_samples=100,
        patch_size=20,
    ),
]
ACTS = [
    ActuatorConfig("nbi", n_channels=4, window_samples=60, n_tokens=3),
]
D_MODEL = 32
BATCH = 2


@pytest.fixture
def rollout() -> TokenSpaceRollout:
    torch.manual_seed(0)
    model = E2EFoundationModel(
        diagnostics=DIAGS,
        actuators=ACTS,
        d_model=D_MODEL,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )
    return TokenSpaceRollout(model, dt_s=0.05)


def _initial_diag(batch: int = BATCH) -> Dict[str, torch.Tensor]:
    return {
        cfg.name: torch.randn(batch, cfg.n_channels, cfg.window_samples)
        for cfg in DIAGS
    }


def _act_sequence(
    n_steps: int, batch: int = BATCH
) -> List[Dict[str, torch.Tensor]]:
    return [
        {cfg.name: torch.randn(batch, cfg.n_channels, cfg.window_samples) for cfg in ACTS}
        for _ in range(n_steps)
    ]


def test_consecutive_steps_differ(rollout: TokenSpaceRollout) -> None:
    """10-step rollout: cos_sim between consecutive diag-token tensors < 0.99."""
    torch.manual_seed(1)
    with torch.no_grad():
        result = rollout(_initial_diag(), _act_sequence(10))

    tokens = result.diagnostic_tokens  # length 11: initial + 10 steps
    for k in range(len(tokens) - 1):
        cos_sim = F.cosine_similarity(
            tokens[k].flatten(), tokens[k + 1].flatten(), dim=0
        ).item()
        assert cos_sim < 0.99, (
            f"Step {k}→{k+1}: diag tokens too similar (cos_sim={cos_sim:.4f}). "
            "Rollout appears to be converging to a fixed point."
        )


def test_no_norm_explosion(rollout: TokenSpaceRollout) -> None:
    """80-step rollout: max per-token norm < 100× reference from step 1."""
    torch.manual_seed(2)
    with torch.no_grad():
        result = rollout(_initial_diag(), _act_sequence(80))

    def max_tok_norm(t: torch.Tensor) -> float:
        return t.norm(dim=-1).max().item()

    ref = max_tok_norm(result.diagnostic_tokens[1])  # after step 0 (== "step 1")
    for k, toks in enumerate(result.diagnostic_tokens[1:], start=1):
        m = max_tok_norm(toks)
        assert m < 100.0 * ref, (
            f"Step {k}: max diag-token norm {m:.3f} ≥ 100× reference {ref:.3f} "
            f"(ratio={m / ref:.2f}). Rollout exploding."
        )


def test_no_norm_collapse(rollout: TokenSpaceRollout) -> None:
    """80-step rollout: min per-token norm > 0.01× reference from step 1."""
    torch.manual_seed(3)
    with torch.no_grad():
        result = rollout(_initial_diag(), _act_sequence(80))

    def min_tok_norm(t: torch.Tensor) -> float:
        return t.norm(dim=-1).min().item()

    ref = min_tok_norm(result.diagnostic_tokens[1])
    for k, toks in enumerate(result.diagnostic_tokens[1:], start=1):
        m = min_tok_norm(toks)
        assert m > 0.01 * ref, (
            f"Step {k}: min diag-token norm {m:.3f} ≤ 0.01× reference {ref:.3f} "
            f"(ratio={m / ref:.4f}). Rollout collapsing."
        )
