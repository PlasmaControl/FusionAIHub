"""§5.6 verification tests for :class:`SharedBackbone`.

Run with::

    pixi run pytest tests/e2e/test_backbone.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.e2e.backbone import SharedBackbone

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
N_TOKENS = 20
BATCH = 2


@pytest.fixture
def backbone() -> SharedBackbone:
    torch.manual_seed(0)
    return SharedBackbone(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        mlp_ratio=4.0,
        dropout=0.0,
    )


def _zero_step(batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch),
    )


def test_self_attention_spreads_information(backbone: SharedBackbone) -> None:
    """Impulse — after one block, every token is influenced by the impulse.

    Small-scale baseline + one random (non-constant!) impulse at position 10.
    After the first block, every position's output differs from the
    impulse-free baseline by norm > 0.01. Failure: attention not mixing or
    residual stream dominating.
    """
    torch.manual_seed(1)
    x_base = torch.randn(1, N_TOKENS, D_MODEL) * 0.1
    x_imp = x_base.clone()
    x_imp[0, 10] = torch.randn(D_MODEL) * 5.0

    step, time = _zero_step(batch=1)
    # Apply step conditioning exactly as the backbone does, then one block.
    embed = backbone.step_cond(step, time).unsqueeze(1)
    y_base = backbone.blocks[0](x_base + embed)
    y_imp = backbone.blocks[0](x_imp + embed)

    diff = (y_imp - y_base).norm(dim=-1)[0]
    assert (diff > 0.01).all(), (
        f"Positions not all influenced by impulse: min diff {diff.min().item():.4f}"
    )


def test_residual_preserves_impulse_advantage(backbone: SharedBackbone) -> None:
    """Impulse — after the full stack, the impulse position retains the largest norm."""
    torch.manual_seed(2)
    x = torch.randn(1, N_TOKENS, D_MODEL) * 0.1
    impulse_pos = 10
    x[0, impulse_pos] = torch.randn(D_MODEL) * 5.0

    step, time = _zero_step(batch=1)
    y = backbone(x, step, time)
    norms = y[0].norm(dim=-1)
    argmax = int(norms.argmax().item())
    assert argmax == impulse_pos, (
        f"Impulse position {impulse_pos} lost dominance after stack; "
        f"argmax={argmax} (norms: impulse={norms[impulse_pos].item():.3f}, "
        f"max={norms[argmax].item():.3f})."
    )


def test_step_conditioning_changes_output(backbone: SharedBackbone) -> None:
    """Impulse — same tokens, different step index → cos_sim < 0.95."""
    torch.manual_seed(3)
    tokens = torch.randn(1, N_TOKENS, D_MODEL) * 0.5
    time = torch.zeros(1)
    y_0 = backbone(tokens, torch.tensor([0]), time)
    y_40 = backbone(tokens, torch.tensor([40]), time)
    cos_sim = F.cosine_similarity(y_0.flatten(), y_40.flatten(), dim=0).item()
    assert cos_sim < 0.95, (
        f"Step conditioning too weak: cos_sim(step=0, step=40) = {cos_sim:.3f}."
    )


def test_progressive_mixing_cv_decreases(backbone: SharedBackbone) -> None:
    """Impulse — coefficient of variation of per-token norms decreases through layers.

    Starting from a peaked state (one strong impulse), later layers spread
    information so the per-token norm distribution flattens (CV drops).
    """
    torch.manual_seed(4)
    x = torch.randn(1, N_TOKENS, D_MODEL) * 0.1
    x[0, 10] = torch.randn(D_MODEL) * 5.0
    step, time = _zero_step(batch=1)
    intermediates = backbone(x, step, time, return_intermediates=True)

    def cv(t: torch.Tensor) -> float:
        norms = t[0].norm(dim=-1)
        return (norms.std() / (norms.mean() + 1e-8)).item()

    cv_first = cv(intermediates[0])  # post-conditioning, pre-block
    cv_last = cv(intermediates[-2])  # output of final block (before final_norm)
    assert cv_last < cv_first, (
        f"CV did not decrease: start={cv_first:.3f}, end={cv_last:.3f} "
        "(attention is not spreading the impulse)."
    )


def test_all_layers_receive_gradient(backbone: SharedBackbone) -> None:
    """Gradient — every block's attention, MLP, and LayerNorm parameters get ``.grad``."""
    torch.manual_seed(5)
    tokens = torch.randn(BATCH, N_TOKENS, D_MODEL)
    step, time = _zero_step()
    y = backbone(tokens, step, time)
    y.sum().backward()

    for layer_idx, block in enumerate(backbone.blocks):
        for name, param in block.named_parameters():
            assert param.grad is not None, f"block[{layer_idx}].{name}: .grad is None"
            assert param.grad.abs().sum().item() > 0.0, (
                f"block[{layer_idx}].{name}: .grad all zeros"
            )


def test_step_embedding_mlp_receives_gradient(backbone: SharedBackbone) -> None:
    """Gradient — the step-conditioning MLP receives ``.grad``."""
    torch.manual_seed(6)
    tokens = torch.randn(BATCH, N_TOKENS, D_MODEL)
    step = torch.tensor([0, 40])
    time = torch.tensor([0.0, 2.0])
    y = backbone(tokens, step, time)
    y.sum().backward()
    for name, param in backbone.step_cond.mlp.named_parameters():
        assert param.grad is not None, f"step_cond.mlp.{name}: .grad is None"
        assert param.grad.abs().sum().item() > 0.0, (
            f"step_cond.mlp.{name}: .grad all zeros"
        )


def test_return_intermediates_layout(backbone: SharedBackbone) -> None:
    """Pin the ``return_intermediates=True`` layout contract.

    - ``len(intermediates) == n_layers + 2``
    - ``intermediates[0]`` is the post-conditioning input (``tokens + step_embed``),
      before any block.
    - ``intermediates[1:n_layers+1]`` are the per-block outputs.
    - ``intermediates[-1]`` is the post-final-norm output.

    Several tests (``test_progressive_mixing_cv_decreases``,
    ``test_signal_pathway_similarity_bounded`` in ``test_full_model.py``)
    index this list directly; if the layout drifts, they silently become
    meaningless.
    """
    torch.manual_seed(8)
    tokens = torch.randn(1, N_TOKENS, D_MODEL)
    step, time = _zero_step(batch=1)

    intermediates = backbone(tokens, step, time, return_intermediates=True)
    assert isinstance(intermediates, list)
    assert len(intermediates) == N_LAYERS + 2, (
        f"Expected length {N_LAYERS + 2}; got {len(intermediates)}."
    )

    expected_first = tokens + backbone.step_cond(step, time).unsqueeze(1)
    assert torch.allclose(intermediates[0], expected_first, atol=1e-6), (
        "intermediates[0] is not the post-conditioning input."
    )

    expected_last = backbone.final_norm(intermediates[-2])
    assert torch.allclose(intermediates[-1], expected_last, atol=1e-6), (
        "intermediates[-1] is not the post-final-norm output of the last block."
    )


def test_fixed_point_different_inputs_different_outputs(
    backbone: SharedBackbone,
) -> None:
    """Fixed-point — different inputs → different outputs (cos_sim < 0.99)."""
    torch.manual_seed(7)
    x1 = torch.randn(1, N_TOKENS, D_MODEL)
    x2 = torch.randn(1, N_TOKENS, D_MODEL)
    step, time = _zero_step(batch=1)
    y1 = backbone(x1, step, time)
    y2 = backbone(x2, step, time)
    cos_sim = F.cosine_similarity(y1.flatten(), y2.flatten(), dim=0).item()
    assert cos_sim < 0.99, (
        f"Backbone output collapses to a fixed point: cos_sim={cos_sim:.4f}."
    )
