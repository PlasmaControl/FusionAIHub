"""d3d_tearing_time_to_event_dsm: inputs in upstream order, preprocessing as
upstream did it, risk as one minus the fork's survival probability."""
import numpy as np
import pytest

from labelmaker.config import Paths
from labelmaker.models import registry
from labelmaker.models.base import BuiltInputs
from labelmaker.models.d3d_tearing_time_to_event_dsm import spec as dsm
from labelmaker.models.runners import dsm_pickle

MODEL_DIR = Paths.from_env().models / dsm.SLUG
needs_weights = pytest.mark.skipif(
    not (MODEL_DIR / dsm.ARTIFACTS[0]).exists(), reason="survival weights not copied"
)

# get_survival_from_shot.py, verbatim
SCALARS = ['bmspinj', 'bmstinj', 'betan_EFITRT2', 'qmin_EFITRT2', 'ech_pwr_total', 'ip',
           'PCBCOIL', 'li_EFITRT2', 'aminor_EFITRT2', 'rmaxis_EFITRT2', 'tribot_EFITRT2',
           'tritop_EFITRT2', 'kappa_EFITRT2', 'volume_EFITRT2']
PROFILES = ['thomson_temp_mtanh_1d', 'cer_temp_csaps_1d', 'thomson_density_mtanh_1d',
            'cer_rot_csaps_1d', 'qpsi_EFITRT2', 'pres_EFITRT2']
NORM_KEYS = ['thomson_temp_mtanh_1d', 'cer_temp_csaps_1d', 'thomson_density_mtanh_1d',
             'rotation_kms', 'qpsi_EFITRT2', 'pres_EFITRT2']


def test_inputs_are_the_upstream_signals_in_upstream_order():
    assert [f.model_name for f in dsm.INPUT_SPEC.scalar_fields] == SCALARS
    assert [f.model_name for f in dsm.INPUT_SPEC.profile_fields] == PROFILES
    assert all(f.lag == "t" for f in dsm.INPUT_SPEC.fields)
    by_name = {f.model_name: f for f in dsm.INPUT_SPEC.fields}
    assert by_name["PCBCOIL"].scale == 1.69861e-5          # the 'fix BT' line
    assert by_name["bmspinj"].scale == 1e-3                 # kW -> MW
    assert [f.name for f in dsm.OUTPUT_SPEC.fields] == OUTPUT_NAMES
    assert dsm.ADAPTER.time_step_ms == 25.0 and dsm.ADAPTER.ensemble_n == 1


def _built(n=7, seed=0):
    rng = np.random.default_rng(seed)
    scalars = np.abs(rng.normal(size=(n, 14))) * np.array(
        [5, 3, 2, 1.5, 1e6, 9e5, 1.7, 1, 0.6, 1.75, 0.3, 0.3, 1.8, 18.0])
    profiles = np.abs(rng.normal(size=(n, 33, 6))) * np.array([2, 2, 4, 50, 3, 5e4])
    profiles[:, :, 4] += 1.0                                # q stays away from zero
    profiles[2, 5, 4] = 0.0                                 # ... except one point: 1/q -> inf -> 1
    return BuiltInputs(t=0.025 * np.arange(n), scalars=scalars, profiles=profiles,
                       valid=np.ones(n, bool), missing=(), resolvers={})


@needs_weights
def test_predict_reproduces_the_upstream_preprocessing_step_by_step():
    """Transcribed from get_survival_from_shot.py: profiles interpolated 33 ->
    100 (65 for 1/q and pressure), 1/q with inf -> 1, PCA projection per
    profile, then z-scores for the four components and for each scalar, in
    the upstream concatenation order; then the fork's survival at the horizon.
    """
    with open(MODEL_DIR / dsm.ARTIFACTS[1], "rb") as fh:
        norm = dsm_pickle.RestrictedUnpickler(fh).load()
    graph = dsm_pickle.load_dsm(MODEL_DIR / dsm.ARTIFACTS[0])
    built = _built()
    x_old, x_long, x_q = np.linspace(0, 1, 33), np.linspace(0, 1, 100), np.linspace(0, 1, 65)
    cols = [(built.scalars - [norm[k]["mean"] for k in SCALARS]) / [norm[k]["std"] for k in SCALARS]]
    for j, key in enumerate(NORM_KEYS):
        prof = built.profiles[:, :, j]
        if key == "qpsi_EFITRT2":
            with np.errstate(divide="ignore"):
                prof = 1.0 / prof
            prof = np.where(prof == np.inf, 1.0, prof)
        grid = x_q if key in ("qpsi_EFITRT2", "pres_EFITRT2") else x_long
        prof = np.array([np.interp(grid, x_old, row) for row in prof])
        pca = (prof - norm[key]["pca_mean"]) @ norm[key]["pca_matrix"].T
        cols.append((pca - norm[key]["mean"]) / norm[key]["std"])
    x = np.concatenate(cols, axis=1)
    assert x.shape == (7, 38)
    want = 1.0 - dsm_pickle.survival(graph, x, horizons_ms=(250.0, 500.0, 1000.0))

    predict = dsm.load(MODEL_DIR)
    members = predict(built)
    assert members.shape == (1, 7, 17)
    np.testing.assert_allclose(members[0, :, :3], want, rtol=0, atol=1e-12)
    decoded = dsm.OUTPUT_SPEC.decode(members)
    np.testing.assert_allclose(decoded["tm_risk_1s"].mean, want[:, 2])
    assert ((0 <= want) & (want <= 1)).all()
    assert (want[:, 0] <= want[:, 1]).all() and (want[:, 1] <= want[:, 2]).all()


OUTPUT_NAMES = [
    "tm_risk_250ms", "tm_risk_500ms", "tm_risk_1s",
    "tm_time_p10", "tm_time_p50", "tm_time_p90", "tm_time_iqr_log",
    "tm_mix_w0", "tm_mix_w1", "tm_mix_w2",
    "tm_mix_mu0", "tm_mix_mu1", "tm_mix_mu2",
    "tm_mix_sigma0", "tm_mix_sigma1", "tm_mix_sigma2", "tm_gate_entropy",
]
OUTPUT_UNITS = [""] * 3 + ["ms"] * 3 + [""] * 4 + ["ln ms"] * 3 + [""] * 3 + ["nat"]


def test_output_contract_and_card_match_including_units():
    assert registry.card_discrepancies(dsm.SLUG) == []
    fields = dsm.OUTPUT_SPEC.fields
    assert [f.name for f in fields] == OUTPUT_NAMES
    assert [f.column for f in fields] == list(range(17))
    assert [f.task for f in fields] == ["binary"] * 3 + ["regression"] * 14
    assert [f.activation for f in fields] == ["none"] * 17
    assert [f.units for f in fields] == OUTPUT_UNITS
    assert [f["units"] for f in registry.read_card(dsm.SLUG)["labelmaker"]["outputs"]] == OUTPUT_UNITS


def test_predict_publishes_all_mixture_columns(tmp_path, monkeypatch):
    from .test_dsm_pickle import _fake_pickle

    path, _ = _fake_pickle(tmp_path, monkeypatch)
    graph = dsm_pickle.load_dsm(path)
    path.rename(tmp_path / dsm.ARTIFACTS[0])
    import pickle

    (tmp_path / dsm.ARTIFACTS[1]).write_bytes(pickle.dumps({}))
    x = np.random.default_rng(17).normal(size=(7, 38))
    monkeypatch.setattr(dsm, "preprocess", lambda built, norm: x)
    got = dsm.load(tmp_path)(_built())
    log_w, mu, sigma = dsm_pickle.mixture(graph, x)
    q = dsm_pickle.quantiles(graph, x, (0.1, 0.5, 0.9))
    want = np.column_stack((1 - dsm_pickle.survival(graph, x, dsm.HORIZONS_MS),
                            q, np.log(q[:, 2]) - np.log(q[:, 0]),
                            np.exp(log_w), mu, sigma, dsm_pickle.gate_entropy(log_w)))
    assert got.shape == (1, 7, 17)
    np.testing.assert_allclose(got[0], want, atol=1e-12)
