"""Actuator counterfactual hook: the ``--actuator_mode`` transforms + the PAIRING contract.

Separate file from test_phaseb_dataset.py because this covers the EVAL side (eval_dynamics),
not the cache/dataset side; it follows the existing one-file-per-concern test_phaseb_* layout.

The whole counterfactual programme rests on one property: two rollouts with the same seed must
differ ONLY through the actuator conditioning. If that silently breaks, every effect size measured
afterwards is noise wearing a lab coat — and it would break invisibly, because a rollout that
diverges for RNG reasons looks exactly like one that diverges for actuator reasons. Hence the
``act_embed``-zeroed test below, which pins the causal path rather than merely observing it.
"""

from pathlib import Path
import tempfile

import pytest
import torch

from tokamak_foundation_model.ignite import eval_dynamics as ed
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
from tokamak_foundation_model.ignite.train_dynamics import _ACT_SPEC

K0, F = 4, 12


def _act(seed=0):
    """(F, 70) stand-in for a cache's already-z-scored actuator block."""
    return torch.randn(F, 70, generator=torch.Generator().manual_seed(seed))


def _group_cols(name):
    off = 0
    for key, nch in _ACT_SPEC:
        if key == name:
            return off, off + nch
        off += nch
    raise KeyError(name)


# --------------------------------------------------------------------------- transforms ---
def test_real_is_identity_but_not_an_alias():
    a = _act()
    out = ed.apply_actuator_mode(a, "real", K0, F=F)
    assert torch.equal(out, a)
    assert out.data_ptr() != a.data_ptr(), "must not hand back the caller's tensor to mutate"


@pytest.mark.parametrize("mode", ["zero", "freeze", "shuffle", "group_zero:rmp"])
def test_seed_region_is_never_touched(mode):
    """Every mode edits [K0, F) ONLY — the model must receive identical context in both arms,
    or the comparison confounds conditioning with a different starting state."""
    a = _act()
    out = ed.apply_actuator_mode(a, mode, K0, F=F)
    assert torch.equal(out[:K0], a[:K0])


def test_zero_and_freeze_semantics():
    a = _act()
    z = ed.apply_actuator_mode(a, "zero", K0, F=F)
    assert float(z[K0:F].abs().max()) == 0.0
    fr = ed.apply_actuator_mode(a, "freeze", K0, F=F)
    assert bool((fr[K0:F] == a[K0 - 1]).all()), "freeze holds the LAST SEED frame"


def test_shuffle_is_a_permutation_and_deterministic():
    a = _act()
    s = ed.apply_actuator_mode(a, "shuffle", K0, F=F)
    # a permutation preserves the column sums exactly; it must also actually reorder
    assert torch.allclose(s[K0:F].sum(0), a[K0:F].sum(0), atol=1e-5)
    assert not torch.equal(s[K0:F], a[K0:F])
    assert torch.equal(s, ed.apply_actuator_mode(a, "shuffle", K0, F=F)), "must be reproducible"


def test_group_zero_touches_only_its_own_columns():
    a = _act()
    out = ed.apply_actuator_mode(a, "group_zero:rmp", K0, F=F)
    lo, hi = _group_cols("rmp")
    assert float(out[K0:F, lo:hi].abs().max()) == 0.0
    others = [c for c in range(70) if not (lo <= c < hi)]
    assert torch.equal(out[:, others], a[:, others])


def test_donor_splices_the_other_shot(tmp_path):
    a, donor = _act(0), _act(1)
    torch.save({"codes": {}, "actuators": donor, "n_frames": F}, tmp_path / "999.pt")
    out = ed.apply_actuator_mode(a, "donor:999", K0, cache_dir=tmp_path, F=F)
    assert torch.equal(out[K0:F], donor[K0:F])
    assert torch.equal(out[:K0], a[:K0])


@pytest.mark.parametrize("bad", ["nonsense", "group_zero:not_a_group"])
def test_unknown_modes_raise(bad):
    with pytest.raises(ValueError):
        ed.apply_actuator_mode(_act(), bad, K0, F=F)


# ------------------------------------------------------- pairing + causal-path contract ---
def _tiny_setup():
    cfg = DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 6, 32), ModalitySpec("b", "slowts", 3, 16)),
        d_model=32, depth=2, n_heads=4, k0_seed=4, n_predict=6, actuator_dim=70,
        grad_checkpointing=False, maskgit_decode_steps=3,
    )
    torch.manual_seed(0)
    model = MaskGITDynamics(cfg).eval()
    g = torch.Generator().manual_seed(0)
    n = cfg.max_frames
    cache = {"codes": {m.name: torch.randint(0, m.codebook_size, (n, m.n_tok), generator=g)
                       for m in cfg.modalities},
             "actuators": torch.randn(n, 70, generator=g), "n_frames": n}
    return cfg, model, cache


def _roll(cfg, model, cache, mode, seed=0, cache_dir=None):
    gen = torch.Generator().manual_seed(seed)
    _gt, pred, _k, _f = ed.rollout_shot(model, cfg, cache, cfg.k0_seed, 0.9, gen,
                                        torch.device("cpu"), actuator_mode=mode,
                                        cache_dir=cache_dir)
    return pred


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_rollout_is_bitwise_reproducible_for_a_fixed_seed():
    cfg, model, cache = _tiny_setup()
    assert _same(_roll(cfg, model, cache, "real"), _roll(cfg, model, cache, "real"))


def test_rollout_actually_consumes_the_generator():
    """Guards the opposite failure: a rollout that ignores the seed would make every
    comparison trivially 'reproducible' and the pairing test vacuous."""
    cfg, model, cache = _tiny_setup()
    assert not _same(_roll(cfg, model, cache, "real", seed=0),
                     _roll(cfg, model, cache, "real", seed=1))


def test_actuators_influence_the_rollout_ONLY_through_act_embed():
    """The core contract, pinned from both sides.

    With act_embed zeroed, the actuator tensor cannot reach the network, so 'zero' and 'real'
    must be BITWISE identical — which simultaneously proves (a) there is no second path by which
    actuators leak into the rollout and (b) the two arms consume an identical RNG stream, so a
    later divergence is attributable to conditioning alone. Restoring and amplifying act_embed
    must then make them differ, or the test would pass for a model that simply ignores its input.
    """
    cfg, model, cache = _tiny_setup()
    saved = {k: v.clone() for k, v in model.backbone.act_embed.state_dict().items()}
    with torch.no_grad():
        model.backbone.act_embed.weight.zero_()
        model.backbone.act_embed.bias.zero_()
    assert _same(_roll(cfg, model, cache, "zero"), _roll(cfg, model, cache, "real")), (
        "actuators changed the rollout with act_embed zeroed — either a leakage path exists "
        "or the two arms drew different RNG"
    )
    with torch.no_grad():
        model.backbone.act_embed.load_state_dict(saved)
        model.backbone.act_embed.weight.mul_(50.0)
    assert not _same(_roll(cfg, model, cache, "zero"), _roll(cfg, model, cache, "real"))


@pytest.mark.parametrize("mode", ["zero", "freeze", "shuffle", "group_zero:rmp"])
def test_every_mode_produces_a_full_trajectory(mode):
    cfg, model, cache = _tiny_setup()
    pred = _roll(cfg, model, cache, mode)
    for m in cfg.modalities:
        assert pred[m.name].shape == (cfg.max_frames, m.n_tok)
