"""§5.8 end-to-end verification tests for :class:`E2EFoundationModel`.

Run with::

    pixi run pytest tests/e2e/test_full_model.py -v
"""

from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

# ── Small Phase-A-style config (time-series only) ─────────────────────────

DIAGS = [
    DiagnosticConfig("ts_core_temp", "slow_ts", n_channels=15, window_samples=5),
    DiagnosticConfig(
        "ts_tangential_density", "slow_ts", n_channels=8, window_samples=5
    ),
    DiagnosticConfig(
        "filterscopes", "fast_ts", n_channels=8, window_samples=500, patch_size=50
    ),
]
ACTS = [
    ActuatorConfig("nbi", n_channels=4, window_samples=60, n_tokens=3),
    ActuatorConfig("ech", n_channels=2, window_samples=60, n_tokens=3),
]
D_MODEL = 32
BATCH = 2


@pytest.fixture
def model() -> E2EFoundationModel:
    torch.manual_seed(0)
    return E2EFoundationModel(
        diagnostics=DIAGS,
        actuators=ACTS,
        d_model=D_MODEL,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )


def _random_inputs(
    batch: int = BATCH,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    diag = {
        cfg.name: torch.randn(batch, cfg.n_channels, cfg.window_samples)
        for cfg in DIAGS
    }
    acts = {
        cfg.name: torch.randn(batch, cfg.n_channels, cfg.window_samples)
        for cfg in ACTS
    }
    return diag, acts


def _zero_step(batch: int = BATCH) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(batch, dtype=torch.long), torch.zeros(batch)


def test_cross_modality_transfer(model: E2EFoundationModel) -> None:
    """Input one modality only → every diagnostic output has norm > 0.001."""
    torch.manual_seed(1)
    diag = {cfg.name: torch.zeros(1, cfg.n_channels, cfg.window_samples) for cfg in DIAGS}
    diag["ts_core_temp"] = torch.randn(1, 15, 5) * 3.0
    acts = {cfg.name: torch.zeros(1, cfg.n_channels, cfg.window_samples) for cfg in ACTS}
    step, time = _zero_step(batch=1)

    outs = model(diag, acts, step, time)
    for name, out in outs.items():
        norm = out.norm().item()
        assert norm > 0.001, (
            f"{name}: output norm {norm:.5f} ≤ 0.001 when only ts_core_temp is active."
        )


def test_actuator_conditioning_changes_diagnostic_outputs(
    model: E2EFoundationModel,
) -> None:
    """Same diagnostics, different actuators → diagnostic outputs measurably differ.

    At random init, a single self-attention pass spreads each actuator token's
    contribution across all ~100 tokens, so the per-token effect is small and
    cos_sim stays close to 1.0 even though the actuator signal is wired
    through. We therefore require a relative norm difference
    ``||out_a - out_b|| / ||out_a|| > 1e-3`` — enough to rule out the actuator
    branch being silently disconnected while tolerating weak untrained effect.
    """
    torch.manual_seed(2)
    diag, _ = _random_inputs(batch=1)
    acts_a = {
        cfg.name: torch.randn(1, cfg.n_channels, cfg.window_samples) for cfg in ACTS
    }
    acts_b = {
        cfg.name: torch.randn(1, cfg.n_channels, cfg.window_samples) for cfg in ACTS
    }
    step, time = _zero_step(batch=1)

    out_a = model(diag, acts_a, step, time)
    out_b = model(diag, acts_b, step, time)
    for name in out_a:
        rel = (
            (out_a[name] - out_b[name]).norm() / out_a[name].norm()
        ).item()
        assert rel > 1e-3, (
            f"{name}: relative norm change under actuator swap is "
            f"{rel:.2e} ≤ 1e-3 — actuator branch appears disconnected."
        )


def test_signal_pathway_similarity_bounded(model: E2EFoundationModel) -> None:
    """Two distinct inputs: cos_sim increases by < 0.1 per stage, < 0.15 total.

    Stages: post-tokenize concatenation, each backbone block output, final
    post-norm backbone output. This verifies the model does not collapse
    distinct inputs into a near-identical internal representation.
    """
    torch.manual_seed(3)
    diag1, acts1 = _random_inputs(batch=1)
    diag2 = {
        k: v + torch.randn_like(v) * 0.3 for k, v in diag1.items()
    }
    acts2 = {
        k: v + torch.randn_like(v) * 0.3 for k, v in acts1.items()
    }
    step, time = _zero_step(batch=1)

    tokens1 = model.tokenize(diag1, acts1)
    tokens2 = model.tokenize(diag2, acts2)
    intermediates1 = model.backbone(tokens1, step, time, return_intermediates=True)
    intermediates2 = model.backbone(tokens2, step, time, return_intermediates=True)

    # Layout guard — the backbone pins ``len == n_layers + 2`` with index 0
    # post-conditioning and index -1 post-final-norm. Breaking this silently
    # would make the stage-wise cos_sim deltas below meaningless.
    assert isinstance(intermediates1, list) and isinstance(intermediates2, list)
    expected_len = model.backbone.n_layers + 2
    assert len(intermediates1) == expected_len == len(intermediates2), (
        f"Unexpected intermediates length "
        f"({len(intermediates1)} vs {len(intermediates2)} vs expected "
        f"{expected_len}) — backbone layout has drifted."
    )

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    # Stage 0: post-tokenize (input to backbone, after step-conditioning added)
    # — this is intermediates[0].
    stages = intermediates1  # length n_layers + 2
    stage_cos: list[float] = [cos(stages[i], intermediates2[i]) for i in range(len(stages))]

    for i in range(1, len(stage_cos)):
        delta = stage_cos[i] - stage_cos[i - 1]
        assert delta < 0.1, (
            f"Stage {i}: cos_sim jumped by {delta:.3f} ≥ 0.10 "
            f"(from {stage_cos[i-1]:.3f} to {stage_cos[i]:.3f})."
        )
    total = stage_cos[-1] - stage_cos[0]
    assert total < 0.15, (
        f"Total cos_sim increase {total:.3f} ≥ 0.15 "
        f"(start={stage_cos[0]:.3f}, end={stage_cos[-1]:.3f}). "
        "Model is compressing distinct inputs toward a common representation."
    )


def test_training_learns_actuator_conditioning(
    model: E2EFoundationModel,
) -> None:
    """After 100 steps training with actuator-determined targets, swapping
    actuator inputs moves diagnostic outputs by cos_sim < 0.9.

    Companion to ``test_actuator_conditioning_changes_diagnostic_outputs``:
    the relative-norm wiring check verifies the actuator branch reaches the
    heads at all; this test verifies the signal is actually *learnable* — a
    stricter cos_sim threshold becomes meaningful once the model has trained
    enough to amplify the actuator contribution.
    """
    torch.manual_seed(10)
    # Batch of 2 with identical diagnostic input across the batch, so the
    # only signal distinguishing targets is the actuator input.
    diag_single = {
        cfg.name: torch.randn(1, cfg.n_channels, cfg.window_samples)
        for cfg in DIAGS
    }
    diag = {k: v.expand(2, -1, -1).contiguous() for k, v in diag_single.items()}
    acts = {
        cfg.name: torch.randn(2, cfg.n_channels, cfg.window_samples)
        for cfg in ACTS
    }
    target = {
        cfg.name: torch.randn(2, cfg.n_channels, cfg.window_samples)
        for cfg in DIAGS
    }
    step, time = _zero_step(batch=2)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(100):
        opt.zero_grad()
        out = model(diag, acts, step, time)
        loss = sum(F.mse_loss(out[cfg.name], target[cfg.name]) for cfg in DIAGS)
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = model(diag, acts, step, time)
    for cfg in DIAGS:
        y = out[cfg.name]
        cos_sim = F.cosine_similarity(y[0].flatten(), y[1].flatten(), dim=0).item()
        assert cos_sim < 0.9, (
            f"{cfg.name}: after training on actuator-determined targets, "
            f"outputs for different actuator inputs still have cos_sim "
            f"{cos_sim:.4f} ≥ 0.9 — actuator conditioning not learned."
        )


def test_training_resolves_bottleneck(model: E2EFoundationModel) -> None:
    """After 50 training steps, two distinct-target outputs have cos_sim < 0.9."""
    torch.manual_seed(4)
    diag, acts = _random_inputs(batch=2)
    target = {
        cfg.name: torch.randn(2, cfg.n_channels, cfg.window_samples)
        for cfg in DIAGS
    }
    step, time = _zero_step(batch=2)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(50):
        opt.zero_grad()
        out = model(diag, acts, step, time)
        loss = sum(F.mse_loss(out[cfg.name], target[cfg.name]) for cfg in DIAGS)
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = model(diag, acts, step, time)
    for cfg in DIAGS:
        y = out[cfg.name]
        cos_sim = F.cosine_similarity(y[0].flatten(), y[1].flatten(), dim=0).item()
        assert cos_sim < 0.9, (
            f"{cfg.name}: after training, batch[0] vs batch[1] cos_sim "
            f"{cos_sim:.4f} ≥ 0.9 — bottleneck unresolved."
        )