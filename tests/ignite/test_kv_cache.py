"""The KV cache changes what a rollout costs, never what it produces; the clamp holds absent
diagnostics at their placeholder code."""

import pytest
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics, placeholders
from tokamak_foundation_model.ignite.sampling import SamplerConfig


def _model(**kw):
    cfg = DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 3, 5), ModalitySpec("b", "slowts", 2, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=4,
        maskgit_decode_steps=4, actuator_dim=6, **kw)
    torch.manual_seed(0)
    return MaskGITDynamics(cfg).eval(), cfg


def _inputs(cfg, batch=2):
    g = torch.Generator().manual_seed(1)
    codes = {m.name: torch.randint(0, m.codebook_size, (batch, cfg.k0_seed, m.n_tok), generator=g)
             for m in cfg.modalities}
    return codes, torch.randn(batch, cfg.max_frames, cfg.actuator_dim, generator=g)


def _rollout(model, codes, act, **kw):
    return model.rollout(codes, act, generator=torch.Generator().manual_seed(7), **kw)


def test_a_cached_frame_sees_what_the_full_sequence_sees():
    model, cfg = _model()
    codes, act = _inputs(cfg)
    bb = model.backbone
    nxt = {n: torch.randint(0, 3, c[:, 0].shape) for n, c in codes.items()}
    full = bb.encode({n: torch.cat([codes[n], nxt[n].unsqueeze(1)], 1) for n in codes},
                     act[:, : cfg.k0_seed + 1])[:, -1]
    cache = bb.prefill(codes, act[:, : cfg.k0_seed], cfg.max_frames)
    step = bb.step(cache, nxt, act[:, cfg.k0_seed])
    torch.testing.assert_close(step, full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_kw", [{}, {"act_cross_attn": True}, {"text_embed_dim": 4}])
@pytest.mark.parametrize("sampler", [
    SamplerConfig(),
    SamplerConfig(global_pool=True),
    SamplerConfig(revision_rounds=2),
    SamplerConfig(gumbel=1.0, top_p=0.9, temperature=0.7),
])
def test_cached_rollout_is_token_identical(model_kw, sampler):
    model, cfg = _model(**model_kw)
    codes, act = _inputs(cfg)
    kw = {"text": torch.randn(2, 4)} if model_kw.get("text_embed_dim") else {}
    plain = _rollout(model, codes, act, sampler=sampler, **kw)
    cached = _rollout(model, codes, act, sampler=sampler, kv_cache=True, **kw)
    for name in plain:
        assert torch.equal(plain[name], cached[name]), name


def test_the_cache_refuses_what_it_cannot_do_exactly():
    model, cfg = _model()
    codes, act = _inputs(cfg)
    with pytest.raises(ValueError, match="cfg_scale"):
        _rollout(model, codes, act, sampler=SamplerConfig(cfg_scale=2.0), kv_cache=True)
    lagged, cfg = _model(lag_embed_k=1)
    with pytest.raises(ValueError, match="lag_embed_k"):
        _rollout(lagged, codes, act, kv_cache=True)


def test_placeholders_are_the_rows_that_never_change():
    codes = {"a": torch.tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 4]]]),
             "b": torch.tensor([[[0, 1], [1, 0]], [[2, 2], [3, 3]]])}
    held = placeholders(codes)
    assert set(held) == {"a"}
    assert held["a"].tolist() == [True, False]


def test_a_held_row_keeps_its_placeholder_and_the_cache_agrees():
    model, cfg = _model()
    codes, act = _inputs(cfg)
    codes["a"][0] = codes["a"][0, :1]                      # row 0 of "a" is a placeholder
    hold = placeholders(codes)
    for kv in (False, True):
        roll = _rollout(model, codes, act, hold=hold, kv_cache=kv,
                        sampler=SamplerConfig(revision_rounds=1))
        assert (roll["a"][0] == codes["a"][0, 0]).all()
    plain = _rollout(model, codes, act, hold=hold)
    cached = _rollout(model, codes, act, hold=hold, kv_cache=True)
    for name in plain:
        assert torch.equal(plain[name], cached[name]), name
