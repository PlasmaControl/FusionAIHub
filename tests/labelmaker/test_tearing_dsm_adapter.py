"""d3d_tearing_time_to_event_dsm: inputs in upstream order, preprocessing as
upstream did it, risk as one minus the fork's survival probability."""
import numpy as np

from labelmaker.config import Paths
from labelmaker.models import registry
from labelmaker.models.base import BuiltInputs
from labelmaker.models.d3d_tearing_time_to_event_dsm import spec as dsm
from labelmaker.models.runners import dsm_pickle

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


def test_the_training_shots_are_committed_beside_the_spec():
    """Half the 500-shot pool is in this set, so every pool number needs it.

    The list is generated from the upstream per-row shot pickle by
    `scripts/labelmaker/write_training_shots.py` and committed, because the
    split has to be reproducible where `/projects` is not mounted.
    """
    shots = dsm.TRAINING_SHOTS
    assert isinstance(shots, frozenset) and len(shots) == 8923
    assert min(shots) == 140444 and max(shots) == 193373
    assert all(isinstance(shot, int) for shot in (min(shots), max(shots)))
    assert dsm.ADAPTER.training_shots is shots
    assert 187199 in shots and 186545 not in shots      # the two example shots


def _built(n=7, seed=0):
    rng = np.random.default_rng(seed)
    scalars = np.abs(rng.normal(size=(n, 14))) * np.array(
        [5, 3, 2, 1.5, 1e6, 9e5, 1.7, 1, 0.6, 1.75, 0.3, 0.3, 1.8, 18.0])
    profiles = np.abs(rng.normal(size=(n, 33, 6))) * np.array([2, 2, 4, 50, 3, 5e4])
    profiles[:, :, 4] += 1.0                                # q stays away from zero
    profiles[2, 5, 4] = 0.0                                 # ... except one point: 1/q -> inf -> 1
    return BuiltInputs(t=0.025 * np.arange(n), scalars=scalars, profiles=profiles,
                       valid=np.ones(n, bool), missing=(), resolvers={})


def test_predict_reproduces_the_upstream_preprocessing_step_by_step(tmp_path, monkeypatch):
    """Transcribed from get_survival_from_shot.py: profiles interpolated 33 ->
    100 (65 for 1/q and pressure), 1/q with inf -> 1, PCA projection per
    profile, then z-scores for the four components and for each scalar, in
    the upstream concatenation order; then the fork's survival at the horizon.
    """
    import pickle

    from .test_dsm_pickle import _fake_pickle

    model_dir = tmp_path
    # _fake_pickle seeds every fake torch model with a fixed seed.
    path, _ = _fake_pickle(tmp_path, monkeypatch)
    path.rename(model_dir / dsm.ARTIFACTS[0])
    rng = np.random.default_rng(21)
    norm = {name: {"mean": 1., "std": 2.} for name in SCALARS}
    for key in NORM_KEYS:
        width = 65 if key in ("qpsi_EFITRT2", "pres_EFITRT2") else 100
        norm[key] = {"mean": rng.normal(size=4), "std": rng.uniform(1, 2, 4),
                     "pca_mean": rng.normal(size=width),
                     "pca_matrix": rng.normal(size=(4, width)) * .01}
    (model_dir / dsm.ARTIFACTS[1]).write_bytes(pickle.dumps(norm))
    with open(model_dir / dsm.ARTIFACTS[1], "rb") as fh:
        norm = dsm_pickle.RestrictedUnpickler(fh).load()
    graph = dsm_pickle.load_dsm(model_dir / dsm.ARTIFACTS[0])
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

    predict = dsm.load(model_dir)
    members = predict(built)
    assert members.shape == (1, 7, 20)
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
    "tm_risk_250ms_isotonic", "tm_risk_500ms_isotonic", "tm_risk_1s_isotonic",
]
OUTPUT_UNITS = [""] * 3 + ["ms"] * 3 + [""] * 4 + ["ln ms"] * 3 + [""] * 3 + ["nat"] + [""] * 3


def test_output_contract_and_card_match_including_units():
    assert registry.card_discrepancies(dsm.SLUG) == []
    fields = dsm.OUTPUT_SPEC.fields
    assert [f.name for f in fields] == OUTPUT_NAMES
    assert [f.column for f in fields] == list(range(20))
    assert [f.task for f in fields] == ["binary"] * 3 + ["regression"] * 14 + ["binary"] * 3
    assert [f.activation for f in fields] == ["none"] * 20
    assert [f.units for f in fields] == OUTPUT_UNITS
    assert [f["units"] for f in registry.read_card(dsm.SLUG)["labelmaker"]["outputs"]] == OUTPUT_UNITS


def test_predict_publishes_all_mixture_columns(tmp_path, monkeypatch):
    from .test_dsm_pickle import _fake_pickle

    # _fake_pickle seeds every fake torch model with a fixed seed.
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
    assert got.shape == (1, 7, 20)
    np.testing.assert_allclose(got[0, :, :17], want, atol=1e-12)

    assert np.isnan(got[0, :, 17:]).all()



def test_loaded_calibration_reaches_predictions_and_hdf5(tmp_path, monkeypatch):
    import json
    import pickle
    from types import SimpleNamespace

    from labelmaker import run
    from labelmaker.calibrate import IsotonicMap
    from labelmaker.labels.store import read_label

    from .test_dsm_pickle import _fake_pickle

    paths = Paths(root=tmp_path)
    model_dir = paths.models / dsm.SLUG
    model_dir.mkdir(parents=True)
    # _fake_pickle seeds every fake torch model with a fixed seed.
    path, _ = _fake_pickle(model_dir, monkeypatch)
    path.rename(model_dir / dsm.ARTIFACTS[0])
    (model_dir / dsm.ARTIFACTS[1]).write_bytes(pickle.dumps({}))
    fitted = IsotonicMap(np.array([0., .5, 1.]), np.array([.1, .7, .9]), 8, .5)
    fit_on = {"shots": [1, 3], "n_rows": 8, "prevalence": .5,
              "row_set": "all_pre_onset", "date": "2026-09-05", "git_sha": "abc"}
    calibration = {"labels": {name: {"map": fitted.to_dict(), "fit_on": fit_on}
                               for name in OUTPUT_NAMES[:3]}}
    (model_dir / "calibration.json").write_text(json.dumps(calibration))
    x = np.random.default_rng(17).normal(size=(7, 38))
    monkeypatch.setattr(dsm, "preprocess", lambda built, norm: x)
    monkeypatch.setattr(registry, "verify_artifacts", lambda *args: None)
    monkeypatch.setattr(run, "_PREDICTORS", {})
    monkeypatch.setattr(run, "_build_inputs", lambda *args: _built())
    paths.features_file(123).parent.mkdir(parents=True)
    paths.features_file(123).write_bytes(b"synthetic")
    ctx = SimpleNamespace(paths=paths, force=True, run_id="test")
    assert run.infer_for_shot(123, dsm.SLUG, ctx)["status"] == "ok"
    for name in OUTPUT_NAMES[:3]:
        raw = read_label(paths.labels_file(123), dsm.SLUG, name)
        iso = read_label(paths.labels_file(123), dsm.SLUG, name + "_isotonic")
        np.testing.assert_allclose(iso.y, fitted.apply(raw.y), atol=1e-7)
        assert iso.attrs["calibration"] == (
            "isotonic, fit on held-out all_pre_onset rows")
        assert json.loads(iso.attrs["calibration_fit_on"]) == fit_on
    # A subsequent load without the file must not retain stale module metadata.
    (model_dir / "calibration.json").unlink()
    predict = dsm.load(model_dir)
    assert np.isnan(predict(_built())[0, :, 17:]).all()
    assert all(not f.attrs for f in predict.output_spec.fields)
