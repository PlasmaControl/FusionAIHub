"""Rollout-context fine-tune: train on the model's OWN context distribution."""
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
from tokamak_foundation_model.ignite.selfforce import rollout_context


def _tiny(**kw):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=4,
        maskgit_decode_steps=3, actuator_dim=6, **kw)


def _codes(cfg, B, F):
    return {m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok)) for m in cfg.modalities}


def test_sf_defaults_to_off():
    c = _tiny()
    assert c.sf_frames == 0 and c.sf_prob == 1.0


def test_rollout_context_replaces_only_the_rolled_window():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    codes = _codes(cfg, B=1, F=6)
    act = torch.randn(1, 6, cfg.actuator_dim)
    out = rollout_context(mg, codes, act, boundary=2, n_roll=2,
                          generator=torch.Generator().manual_seed(1))
    for m in cfg.modalities:
        assert out[m.name].shape == codes[m.name].shape
        assert torch.equal(out[m.name][:, :2], codes[m.name][:, :2])   # prefix untouched
        assert torch.equal(out[m.name][:, 4:], codes[m.name][:, 4:])   # suffix untouched
        assert (out[m.name] < m.codebook_size).all()                   # never the MASK id


def test_rollout_context_output_is_detached():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=1, F=6)
    act = torch.randn(1, 6, cfg.actuator_dim)
    out = rollout_context(mg, codes, act, boundary=2, n_roll=2,
                          generator=torch.Generator().manual_seed(1))
    for m in cfg.modalities:
        assert not out[m.name].requires_grad


def test_training_loss_with_sf_frames_backprops():
    cfg = _tiny(sf_frames=2, ctf_frac=1.0)
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=2, F=6)
    act = torch.randn(2, 6, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    g = mg.backbone.blocks[0].spatial.qkv.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_sf_draws_are_device_qualified(monkeypatch):
    """Every generator-bound RNG draw on the sf path must name a device.

    On GPU the trainer's validation pass hands ``training_loss`` a DEVICE-TYPED generator
    (``torch.Generator(device=device)``, train_dynamics.py::_val_loss), and such a generator
    rejects a draw on any other device: an unqualified ``torch.rand((), generator=g)`` draws on
    the default CPU device and raises "Expected a 'cpu' device type for generator but found
    'cuda'". The CTF gate passes ``device=`` for exactly this reason (see its comment), and the
    sf gate sits three lines above it.

    A CPU generator cannot reproduce that mismatch, and the suite is CPU-only, so pin the
    CONVENTION instead: no generator-bound draw may omit ``device``. Fails loudly if a future
    edit drops it, which is what turns the GPU crash into a local test failure.
    """
    seen = []

    def _spy(name):
        real = getattr(torch, name)

        def wrapper(*a, **kw):
            if kw.get("generator") is not None:
                seen.append((name, kw.get("device")))
            return real(*a, **kw)
        return wrapper

    monkeypatch.setattr(torch, "rand", _spy("rand"))
    monkeypatch.setattr(torch, "randint", _spy("randint"))

    # sf_prob 1.0 short-circuits the gate rand and only draws the boundary randint; 0.5 draws
    # the gate rand too. Both together cover every draw the sf block can make.
    for sf_prob in (1.0, 0.5):
        cfg = _tiny(sf_frames=2, sf_prob=sf_prob)
        torch.manual_seed(0)
        mg = MaskGITDynamics(cfg).train()
        codes = _codes(cfg, B=2, F=6)
        act = torch.randn(2, 6, cfg.actuator_dim)
        dev = codes[cfg.modalities[0].name].device
        seen.clear()
        mg.training_loss(codes, act, generator=torch.Generator(device=dev).manual_seed(2))
        assert seen, f"sf_prob={sf_prob}: training_loss made no generator-bound draw"
        assert all(d == dev for _, d in seen), \
            f"sf_prob={sf_prob}: draw(s) not on the codes' device {dev}: {seen}"
        if sf_prob >= 1.0:
            assert any(n == "randint" for n, _ in seen), "the sf boundary draw never happened"
