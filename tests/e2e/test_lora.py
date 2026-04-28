"""Unit tests for :class:`LoRAMultiheadAttention` and wrapper helpers."""

import pytest
import torch
import torch.nn as nn

from tokamak_foundation_model.e2e.backbone import SharedBackbone
from tokamak_foundation_model.e2e.lora import (
    LoRAMultiheadAttention,
    apply_lora_to_backbone,
    freeze_non_lora_parameters,
)

D_MODEL = 32
N_HEADS = 4
N_TOKENS = 20
BATCH = 2


@pytest.fixture
def base_mha() -> nn.MultiheadAttention:
    torch.manual_seed(0)
    return nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)


def test_lora_forward_matches_base_at_init(base_mha: nn.MultiheadAttention) -> None:
    """B is zero-initialised so the LoRA delta is zero and the wrapper must
    produce the same output as the base module."""
    torch.manual_seed(1)
    x = torch.randn(BATCH, N_TOKENS, D_MODEL)
    base_mha.eval()
    base_out, _ = base_mha(x, x, x, need_weights=False)
    lora = LoRAMultiheadAttention(base_mha, rank=16).eval()
    lora_out, lora_attn = lora(x, x, x, need_weights=False)
    assert lora_attn is None
    # SDPA path and manual path should agree to within fp32 precision.
    assert torch.allclose(lora_out, base_out, atol=1e-5), (
        f"Max abs diff = {(lora_out - base_out).abs().max().item():.2e}"
    )


def test_base_params_frozen_after_wrap(base_mha: nn.MultiheadAttention) -> None:
    lora = LoRAMultiheadAttention(base_mha, rank=8)
    for name, param in lora.base.named_parameters():
        assert not param.requires_grad, f"base.{name} is not frozen"


def test_lora_params_train(base_mha: nn.MultiheadAttention) -> None:
    torch.manual_seed(2)
    lora = LoRAMultiheadAttention(base_mha, rank=8)
    x = torch.randn(BATCH, N_TOKENS, D_MODEL)
    target = torch.randn(BATCH, N_TOKENS, D_MODEL)
    out, _ = lora(x, x, x)
    (out - target).pow(2).mean().backward()

    for name, param in lora.named_parameters():
        if "lora_" in name:
            assert param.grad is not None, f"{name} .grad is None"
            if "lora_B" in name:
                # B is zero at init — its gradient should still be non-zero
                # because d/dB of (B @ A) · x has A · x as the gradient and A
                # is Kaiming-initialised.
                assert param.grad.abs().sum().item() > 0.0, (
                    f"{name} .grad is all zeros"
                )
            elif "lora_A" in name:
                # A's gradient flows through B which is zero at init — so A's
                # initial gradient should be ZERO (that's the whole point of
                # zero-init B). Verify this invariant.
                assert param.grad.abs().sum().item() == 0.0, (
                    f"{name} .grad unexpectedly non-zero at init (B=0)"
                )
        else:
            # Base params — either .grad is None (never touched) or zero
            # (touched but should not have updated). Frozen params can still
            # receive .grad; what matters is that requires_grad is False so
            # the optimizer won't update them.
            assert not param.requires_grad, f"{name} is not frozen"


def test_lora_delta_is_non_zero_after_one_step(
    base_mha: nn.MultiheadAttention,
) -> None:
    """After one optimizer step on the LoRA params, the delta is non-zero —
    confirming the wrapper really trains and isn't a no-op."""
    torch.manual_seed(3)
    lora = LoRAMultiheadAttention(base_mha, rank=8)
    opt = torch.optim.Adam(
        [p for p in lora.parameters() if p.requires_grad], lr=1e-2
    )
    x = torch.randn(BATCH, N_TOKENS, D_MODEL)
    target = torch.randn(BATCH, N_TOKENS, D_MODEL)
    for _ in range(3):
        opt.zero_grad()
        out, _ = lora(x, x, x)
        (out - target).pow(2).mean().backward()
        opt.step()
    delta_in = lora._delta_in_proj()
    delta_out = lora._delta_out_proj()
    assert delta_in.abs().sum().item() > 0.0
    assert delta_out.abs().sum().item() > 0.0


def test_apply_lora_to_backbone_replaces_attn() -> None:
    torch.manual_seed(4)
    backbone = SharedBackbone(
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=2, dropout=0.0
    )
    apply_lora_to_backbone(backbone, rank=8)
    for block in backbone.blocks:
        assert isinstance(block.attn, LoRAMultiheadAttention)

    # After wrapping + freezing non-LoRA, only lora_ params train.
    freeze_non_lora_parameters(backbone)
    trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
    assert trainable, "expected LoRA params to be trainable"
    for n in trainable:
        assert "lora_" in n, f"unexpected trainable param: {n}"
    # Sanity: MLP weights frozen.
    for block in backbone.blocks:
        for n, p in block.mlp.named_parameters():
            assert not p.requires_grad, f"mlp.{n} is not frozen"


def test_lora_params_placed_on_base_device() -> None:
    """Wrapping a GPU-resident MHA must produce a GPU-resident wrapper.

    Regression test for the Stage 3 launch bug: ``apply_lora_to_backbone``
    was called after ``model.to(device)``, and default tensor creation put
    LoRA params on CPU → device mismatch in the first forward. The
    wrapper's ``__init__`` now reads the base's device and allocates LoRA
    parameters there.
    """
    # Simulate by constructing a "fake CUDA" via ``meta`` device so the test
    # runs on CPU-only CI. ``meta`` is enough to verify the device-propagation
    # invariant without needing a GPU.
    if not hasattr(torch, "device"):  # pragma: no cover — trivially true
        pytest.skip("torch.device unavailable")
    torch.manual_seed(0)
    base = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
    # Move base to ``meta``; this tags every parameter with device=meta.
    base = base.to(torch.device("meta"))
    lora = LoRAMultiheadAttention(base, rank=4)
    for name in ("lora_A_qkv", "lora_B_qkv", "lora_A_out", "lora_B_out"):
        p = getattr(lora, name)
        assert p.device.type == "meta", (
            f"{name} on {p.device}, expected 'meta' (= base's device)."
        )


def test_apply_lora_forward_matches_unlora_at_init() -> None:
    """A full backbone pass with freshly-applied LoRA (zero delta) must match
    the same backbone before LoRA was applied."""
    torch.manual_seed(5)
    backbone = SharedBackbone(
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=2, dropout=0.0
    )
    backbone.eval()

    tokens = torch.randn(BATCH, N_TOKENS, D_MODEL)
    step = torch.zeros(BATCH, dtype=torch.long)
    time = torch.zeros(BATCH)
    y_before = backbone(tokens, step, time)

    apply_lora_to_backbone(backbone, rank=8)
    backbone.eval()
    y_after = backbone(tokens, step, time)

    assert torch.allclose(y_before, y_after, atol=1e-5), (
        f"Max abs diff = {(y_before - y_after).abs().max().item():.2e}"
    )