"""Reading an auton-survival Deep Survival Machines pickle without auton-survival.

The upstream tearing-survival checkpoint is a pickle of the fork's own classes
wrapping plain torch modules. Only three of those classes need to exist to
unpickle it, and none of their code is needed to evaluate it, so a restricted
unpickler stands them in and refuses everything else.
"""
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from labelmaker.models.runners import dsm_pickle

UPSTREAM = Path("/projects/EKOLEMEN/survival_tm_2/models/rt_fixed_rot.pkl")
GOLDEN = Path(__file__).with_name("data") / "tearing_dsm_golden.npz"


def _relu6(a):
    return np.clip(a, 0.0, 6.0)


def _graph():
    rng = np.random.default_rng(3)
    return dsm_pickle.DsmGraph(
        k=2, dist="LogNormal", temp=1.0,
        embedding=(rng.normal(size=(3, 2)), rng.normal(size=(2, 3))),
        gate=rng.normal(size=(2, 2)),
        scaleg=(rng.normal(size=(2, 2)), rng.normal(size=2)),
        shapeg=(rng.normal(size=(2, 2)), rng.normal(size=2)),
        shape=np.array([6.0, 6.5]), scale=np.array([0.2, -0.3]),
    )


def test_survival_is_the_lognormal_mixture_the_fork_defines():
    """Transcribed from auton_survival dsm_torch `_init_dsm_layers` (LogNormal:
    tanh heads added to the shape/scale parameters, gate divided by `temp`)
    and losses `_lognormal_cdf`: S(t) = sum_k softmax(gate)_k *
    (0.5 - 0.5 erf((ln t - mu_k) / (exp(sigma_k) sqrt 2))).
    """
    g = _graph()
    x = np.array([[0.3, -1.2], [1.0, 0.5]])
    h = _relu6(_relu6(x @ g.embedding[0].T) @ g.embedding[1].T)
    mu = np.tanh(h @ g.shapeg[0].T + g.shapeg[1]) + g.shape
    sigma = np.tanh(h @ g.scaleg[0].T + g.scaleg[1]) + g.scale
    logits = (h @ g.gate.T) / g.temp
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    want = np.zeros((2, 2))
    for j, t in enumerate((500.0, 1000.0)):
        z = (np.log(t) - mu) / (np.exp(sigma) * np.sqrt(2.0))
        want[:, j] = (w * (0.5 - 0.5 * torch.erf(torch.tensor(z)).numpy())).sum(axis=1)
    got = dsm_pickle.survival(g, x, horizons_ms=(500.0, 1000.0))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert ((0.0 <= got) & (got <= 1.0)).all()
    # later horizon, lower survival
    assert (got[:, 1] <= got[:, 0]).all()


class _FakeTorch(torch.nn.Module):
    """Shaped like the fork's DeepSurvivalMachinesTorch, without the fork."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.k, self.dist, self.temp, self.risks = 3, "LogNormal", 1.0, 1
        self.act = torch.nn.Tanh()
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(38, 100, bias=False), torch.nn.ReLU6(),
            torch.nn.Linear(100, 1000, bias=False), torch.nn.ReLU6(),
        ).double()
        self.shape = torch.nn.ParameterDict({"1": torch.nn.Parameter(torch.ones(3, dtype=torch.float64))})
        self.scale = torch.nn.ParameterDict({"1": torch.nn.Parameter(torch.ones(3, dtype=torch.float64))})
        self.gate = torch.nn.ModuleDict({"1": torch.nn.Sequential(torch.nn.Linear(1000, 3, bias=False)).double()})
        self.scaleg = torch.nn.ModuleDict({"1": torch.nn.Sequential(torch.nn.Linear(1000, 3)).double()})
        self.shapeg = torch.nn.ModuleDict({"1": torch.nn.Sequential(torch.nn.Linear(1000, 3)).double()})


_FakeTorch.__module__ = "auton_survival.models.dsm.dsm_torch"
_FakeTorch.__qualname__ = _FakeTorch.__name__ = "DeepSurvivalMachinesTorch"


class _FakeDsm:
    pass


_FakeDsm.__module__ = "auton_survival.models.dsm"
_FakeDsm.__qualname__ = _FakeDsm.__name__ = "DeepSurvivalMachines"


class _FakeSurvivalModel:
    pass


_FakeSurvivalModel.__module__ = "auton_survival.estimators"
_FakeSurvivalModel.__qualname__ = _FakeSurvivalModel.__name__ = "SurvivalModel"


def _register_fake_modules(monkeypatch):
    """Pickling checks `sys.modules[cls.__module__].<name> is cls`, so the
    fork's module names must exist while the fake pickle is written."""
    import sys
    import types

    for cls in (_FakeTorch, _FakeDsm, _FakeSurvivalModel):
        parts = cls.__module__.split(".")
        for i in range(1, len(parts) + 1):        # every parent package too
            dotted = ".".join(parts[:i])
            if dotted not in sys.modules:
                pkg = types.ModuleType(dotted)
                pkg.__path__ = []
                monkeypatch.setitem(sys.modules, dotted, pkg)
        setattr(sys.modules[cls.__module__], cls.__name__, cls)


def _fake_pickle(tmp_path, monkeypatch):
    _register_fake_modules(monkeypatch)
    torch_model = _FakeTorch()
    dsm = _FakeDsm()
    dsm.__dict__.update(k=3, dist="LogNormal", temp=1.0, layers=[100, 1000],
                        fitted=True, torch_model=torch_model)
    sm = _FakeSurvivalModel()
    sm.__dict__.update(model="dsm", fitted=True, _model=dsm)
    payload = [[sm, [0.1] * 20, [0.2] * 20, {"k": 3, "layers": [100, 1000]}]]
    path = tmp_path / "fake.pkl"
    path.write_bytes(pickle.dumps(payload, protocol=4))
    return path, torch_model


def test_load_dsm_reads_the_upstream_shape_without_the_fork(tmp_path, monkeypatch):
    path, torch_model = _fake_pickle(tmp_path, monkeypatch)
    g = dsm_pickle.load_dsm(path)
    assert (g.k, g.dist, g.temp) == (3, "LogNormal", 1.0)
    assert [w.shape for w in g.embedding] == [(100, 38), (1000, 100)]
    np.testing.assert_array_equal(g.gate, torch_model.gate["1"][0].weight.detach().numpy())
    np.testing.assert_array_equal(g.shape, torch_model.shape["1"].detach().numpy())
    # and the graph it read evaluates like the torch modules it came from
    x = np.random.default_rng(0).normal(size=(4, 38))
    s = dsm_pickle.survival(g, x, horizons_ms=(1000.0,))
    assert s.shape == (4, 1) and np.isfinite(s).all()


def test_load_dsm_refuses_a_pickle_that_needs_any_other_class(tmp_path):
    path = tmp_path / "other.pkl"
    path.write_bytes(pickle.dumps(Path("/etc/passwd")))
    with pytest.raises(pickle.UnpicklingError, match="pathlib"):
        dsm_pickle.load_dsm(path)


def test_load_dsm_refuses_a_distribution_it_cannot_evaluate(tmp_path, monkeypatch):
    _register_fake_modules(monkeypatch)
    torch_model = _FakeTorch()
    torch_model.dist = "Weibull"
    path = tmp_path / "weibull.pkl"
    path.write_bytes(pickle.dumps([[torch_model]], protocol=4))
    with pytest.raises(dsm_pickle.UnsupportedModel, match="Weibull"):
        dsm_pickle.load_dsm(path)


@pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream survival checkpoint not mounted")
def test_upstream_checkpoint_matches_the_fork_on_the_golden_inputs():
    """The golden file was made once with the auton-survival fork itself
    (`data/make_tearing_dsm_golden.py`); our evaluator must reproduce it.
    Both sides are float64, so the tolerance is arithmetic, not rounding."""
    gold = np.load(GOLDEN)
    g = dsm_pickle.load_dsm(UPSTREAM)
    got = dsm_pickle.survival(g, gold["x"], horizons_ms=tuple(gold["horizons_ms"]))
    np.testing.assert_allclose(got, gold["survival"], rtol=0, atol=1e-9)


def test_mixture_exposes_normalized_gate_and_component_heads():
    g = _graph()
    x = np.array([[0.3, -1.2], [1.0, 0.5]])
    log_w, mu, sigma = dsm_pickle.mixture(g, x)
    assert log_w.shape == mu.shape == sigma.shape == (2, 2)
    np.testing.assert_allclose(np.exp(log_w).sum(axis=1), 1.0, rtol=0, atol=1e-12)
    h = _relu6(_relu6(x @ g.embedding[0].T) @ g.embedding[1].T)
    np.testing.assert_allclose(mu, np.tanh(h @ g.shapeg[0].T + g.shapeg[1]) + g.shape)
    np.testing.assert_allclose(sigma, np.tanh(h @ g.scaleg[0].T + g.scaleg[1]) + g.scale)


def test_single_component_quantiles_are_analytic_lognormal():
    from dataclasses import replace

    from scipy.special import erfinv

    g = replace(_graph(), k=1, gate=np.zeros((1, 2)),
                shapeg=(np.zeros((1, 2)), np.zeros(1)),
                scaleg=(np.zeros((1, 2)), np.zeros(1)),
                shape=np.array([6.0]), scale=np.array([0.2]))
    x = np.array([[0.3, -1.2], [1.0, 0.5]])
    q = np.array([0.1, 0.5, 0.9])
    want = np.exp(6.0 + np.exp(0.2) * np.sqrt(2) * erfinv(2 * q - 1))
    got = dsm_pickle.quantiles(g, x, q)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, np.tile(want, (2, 1)), rtol=1e-6)


def test_mixture_quantiles_are_ordered_and_invert_survival():
    g = _graph()
    x = np.array([[0.3, -1.2], [1.0, 0.5]])
    got = dsm_pickle.quantiles(g, x, (0.1, 0.5, 0.9))
    assert (np.diff(got, axis=1) > 0).all()
    for row, quantiles in zip(x, got, strict=True):
        np.testing.assert_allclose(dsm_pickle.survival(g, row[None], quantiles),
                                   [[0.9, 0.5, 0.1]], atol=1e-6)


def test_quantiles_clamp_rows_outside_the_bracket():
    from dataclasses import replace

    for location, endpoint in ((-100.0, 1e-3), (100.0, 1e7)):
        g = replace(_graph(), shape=np.full(2, location))
        got = dsm_pickle.quantiles(g, np.zeros((2, 2)), (0.1, 0.5, 0.9))
        np.testing.assert_array_equal(got, np.full((2, 3), endpoint))


def test_quantiles_clamp_extreme_row_without_affecting_normal_row():
    from dataclasses import replace

    from scipy.special import erfinv

    g = replace(_graph(), k=1, embedding=(), gate=np.zeros((1, 1)),
                shapeg=(np.zeros((1, 1)), np.zeros(1)),
                scaleg=(np.array([[10.0]]), np.zeros(1)),
                shape=np.array([6.0]), scale=np.array([3.0]))
    q = np.array([0.1, 0.5, 0.9])
    x = np.array([[1.0], [-1.0]])
    normal = dsm_pickle.quantiles(g, x[1:], q)
    got = dsm_pickle.quantiles(g, x, q)
    np.testing.assert_array_equal(got[0, [0, 2]], [1e-3, 1e7])
    np.testing.assert_allclose(got[0, 1], np.exp(6.0), rtol=1e-12)
    np.testing.assert_array_equal(got[1:], normal)
    want = np.exp(6.0 + np.exp(3.0 + np.tanh(-10.0)) * np.sqrt(2) * erfinv(2 * q - 1))
    np.testing.assert_allclose(got[1], want, rtol=1e-12)


def test_gate_entropy_uniform_and_one_hot():
    log_w = np.array([[-np.log(3)] * 3, [0.0, -np.inf, -np.inf]])
    np.testing.assert_allclose(dsm_pickle.gate_entropy(log_w), [np.log(3), 0.0], atol=1e-12)


def test_load_dsm_skips_dropout_layers_the_elm_fork_inserts(tmp_path, monkeypatch):
    """The ELM fork's `create_representation` appends `nn.Dropout(p)` after every
    ReLU6. Dropout is the identity in eval mode - the only mode a checkpoint is
    read in - and carries no weights, so the reader skips it and returns the same
    graph the same weights would give without it."""
    _register_fake_modules(monkeypatch)
    plain = _FakeTorch()
    dropped = _FakeTorch()
    dropped.embedding = torch.nn.Sequential(
        plain.embedding[0], torch.nn.ReLU6(), torch.nn.Dropout(p=0.2),
        plain.embedding[2], torch.nn.ReLU6(), torch.nn.Dropout(p=0.2),
    ).double()
    paths = []
    for i, model in enumerate((plain, dropped)):
        path = tmp_path / f"m{i}.pkl"
        path.write_bytes(pickle.dumps([[model]], protocol=4))
        paths.append(path)
    without, with_dropout = (dsm_pickle.load_dsm(p) for p in paths)
    assert [w.shape for w in with_dropout.embedding] == [(100, 38), (1000, 100)]
    x = np.random.default_rng(0).normal(size=(4, 38))
    np.testing.assert_array_equal(
        dsm_pickle.survival(with_dropout, x, horizons_ms=(1000.0,)),
        dsm_pickle.survival(without, x, horizons_ms=(1000.0,)),
    )


def test_load_dsm_still_refuses_an_embedding_layer_it_cannot_evaluate(tmp_path, monkeypatch):
    _register_fake_modules(monkeypatch)
    model = _FakeTorch()
    model.embedding = torch.nn.Sequential(
        torch.nn.Linear(38, 100, bias=False), torch.nn.Sigmoid(),
    ).double()
    path = tmp_path / "sigmoid.pkl"
    path.write_bytes(pickle.dumps([[model]], protocol=4))
    with pytest.raises(dsm_pickle.UnsupportedModel, match="Sigmoid"):
        dsm_pickle.load_dsm(path)
