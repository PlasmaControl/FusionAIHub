"""FlashAttention-4 path of the dynamics attention against an fp32 reference.

Runs only where FA4 can (CUDA, compute capability 9.x/10.x/11.x, flash-attn-4
installed); elsewhere the dispatcher takes SDPA and these tests skip.
"""

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.ignite import attention as attn
from tokamak_foundation_model.ignite.dynamics import (
    DynamicsBackbone,
    _attend_earlier_and_self_explicit,
)
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec


def _fa4_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    probe = torch.zeros(1, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
    return attn.fa4_reason(probe) is None


pytestmark = pytest.mark.skipif(not _fa4_ready(), reason="FlashAttention-4 cannot run here")

ATOL = RTOL = 3e-2  # bf16 inputs against an fp32 reference


def _qkv(B, H, Lq, Lk, D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    make = lambda L: torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16, generator=g)  # noqa: E731
    return make(Lq), make(Lk), make(Lk)


def _grads(fn, tensors):
    leaves = [t.detach().clone().requires_grad_() for t in tensors]
    out = fn(*leaves)
    out.float().square().sum().backward()
    # An input the output does not depend on gets no gradient; the reference gives zeros.
    grads = [torch.zeros_like(t, dtype=torch.float32) if t.grad is None else t.grad.float() for t in leaves]
    return out.detach().float(), grads


def _close(a, b):
    torch.testing.assert_close(a, b, atol=ATOL * max(1.0, b.abs().max().item()), rtol=RTOL)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("L,D", [(77, 64), (300, 64), (40, 128)])
def test_self_attention_matches_fp32(monkeypatch, causal, L, D):
    monkeypatch.setenv(attn.ENV, "fa4")
    q, k, v = _qkv(6, 4, L, L, D)
    out, grads = _grads(lambda a, b, c: attn.attention(a, b, c, causal=causal), (q, k, v))
    ref, ref_grads = _grads(
        lambda a, b, c: F.scaled_dot_product_attention(a.float(), b.float(), c.float(), is_causal=causal),
        (q, k, v),
    )
    _close(out, ref)
    for g, r in zip(grads, ref_grads):
        _close(g, r)


def test_cross_attention_matches_fp32(monkeypatch):
    monkeypatch.setenv(attn.ENV, "fa4")
    q, k, v = _qkv(8, 8, 120, 88, 128)  # frame tokens attending actuator channels
    out, grads = _grads(attn.attention, (q, k, v))
    ref, ref_grads = _grads(
        lambda a, b, c: F.scaled_dot_product_attention(a.float(), b.float(), c.float()), (q, k, v)
    )
    _close(out, ref)
    for g, r in zip(grads, ref_grads):
        _close(g, r)


@pytest.mark.parametrize("L", [1, 2, 33, 80])
def test_earlier_and_self_matches_the_masked_reference(monkeypatch, L):
    monkeypatch.setenv(attn.ENV, "fa4")
    g = torch.Generator(device="cuda").manual_seed(1)
    tensors = [
        torch.randn(5, 4, L, 64, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(5)
    ]
    out, grads = _grads(attn.attend_earlier_and_self_fa4, tensors)
    ref, ref_grads = _grads(
        lambda *t: _attend_earlier_and_self_explicit(*(x.float() for x in t)), tensors
    )
    _close(out, ref)
    for g_, r in zip(grads, ref_grads):
        _close(g_, r)


def _cfg():
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 24, 64), ModalitySpec("b", "slowts", 7, 32)),
        d_model=256,
        depth=2,
        n_heads=4,
        ffn_mult=2,
        k0_seed=2,
        n_predict=6,
        actuator_dim=12,
    )


def test_backbone_fa4_matches_sdpa(monkeypatch):
    cfg = _cfg()
    torch.manual_seed(0)
    model = DynamicsBackbone(cfg).cuda().eval()
    B, Fr = 3, 8
    codes = {m.name: torch.randint(0, m.codebook_size, (B, Fr, m.n_tok), device="cuda") for m in cfg.modalities}
    act = torch.randn(B, Fr, cfg.actuator_dim, device="cuda")
    out = {}
    for mode in ("sdpa", "fa4"):
        monkeypatch.setenv(attn.ENV, mode)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(codes, act)
        loss = sum(v.float().logsumexp(-1).mean() for v in logits.values())
        loss.backward()
        out[mode] = (
            {k: v.detach().float() for k, v in logits.items()},
            torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]),
        )
    for name in out["sdpa"][0]:
        _close(out["fa4"][0][name], out["sdpa"][0][name])
    cos = F.cosine_similarity(out["fa4"][1], out["sdpa"][1], dim=0).item()
    assert cos > 0.999, cos


def test_env_switch(monkeypatch):
    q = torch.zeros(1, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setenv(attn.ENV, "sdpa")
    assert not attn.use_fa4(q)
    monkeypatch.setenv(attn.ENV, "auto")
    assert attn.use_fa4(q)
    monkeypatch.setenv(attn.ENV, "fa4")
    with pytest.raises(RuntimeError):
        attn.use_fa4(q.float())
    monkeypatch.setenv(attn.ENV, "flash")
    with pytest.raises(ValueError):
        attn.use_fa4(q)
