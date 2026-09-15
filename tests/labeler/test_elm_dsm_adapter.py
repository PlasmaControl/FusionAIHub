"""d3d_elm_time_to_event_dsm: 60 named columns, upstream's normalisation, and
`1 - S(h + 1)` at four horizons.

Hermetic: the checkpoint is a fake `DeepSurvivalMachinesTorch` of the right
width and the normalisation constants are made up, so nothing here reads
`/projects` or the model root. What is under test is the mapping - which
canonical feature lands in which of the 60 columns, in what units, and what
happens to a column the corpus cannot serve - not the fitted weights.
"""
import json

import numpy as np
import pytest
import torch

from labeler.models import registry
from labeler.models.base import BuiltInputs
from labeler.models.d3d_elm_time_to_event_dsm import spec as elm
from labeler.models.elm_inputs import COLUMN_SETS
from labeler.models.runners import dsm_pickle

from .test_dsm_pickle import (
    _FakeDsm,
    _FakeSurvivalModel,
    _FakeTorch,
    _register_fake_modules,
)

COLUMNS = COLUMN_SETS["no_bes"]
INDEX = {name: i for i, name in enumerate(COLUMNS)}


def test_the_adapter_declares_the_eleven_features_that_fill_sixty_columns():
    assert [f"{f.model_name} <- {f.canonical}" for f in elm.INPUT_SPEC.fields] == [
        "ip_downsampled <- ip",
        "bt_downsampled <- bt",
        "gas_downsampled <- gas",
        "pinj_downsampled <- pinj_total",
        "tinj_downsampled <- tinj_total",
        "ech_downsampled <- ech_power_total",
        "co2_density_slow_r0_downsampled <- co2_r0",
        "co2_density_slow_v1_downsampled <- co2_v1",
        "co2_density_slow_v2_downsampled <- co2_v2",
        "co2_density_slow_v3_downsampled <- co2_v3",
        "ece_slow_downsampled <- ece",
    ]
    by_name = {f.model_name: f for f in elm.INPUT_SPEC.fields}
    assert by_name["ip_downsampled"].scale == 1e-6           # A -> MA
    assert by_name["pinj_downsampled"].scale == 1e3          # kW -> W
    assert all(f.lag == "t" for f in elm.INPUT_SPEC.fields)
    assert len(COLUMNS) == 60
    assert [f.name for f in elm.OUTPUT_SPEC.fields] == [
        "elm_risk_5ms", "elm_risk_10ms", "elm_risk_20ms", "elm_risk_50ms"]
    assert elm.ADAPTER.time_step_ms == 25.0 and elm.ADAPTER.ensemble_n == 1
    assert elm.ADAPTER.artifacts == ("elm_dsm_no_bes.pkl", "normalization.json")


def test_the_card_matches_the_spec_and_the_model_is_implemented():
    assert registry.status("d3d_elm_time_to_event_dsm") == "implemented"
    assert registry.card_discrepancies("d3d_elm_time_to_event_dsm") == []


def _norm(mean=0.0, std=1.0):
    return {"columns": list(COLUMNS),
            "mean": [float(mean)] * len(COLUMNS),
            "std": [float(std)] * len(COLUMNS)}


def _built(n=5, *, missing=()):
    """Distinct values per field so a mis-mapped column is visible."""
    scalars = np.zeros((n, len(elm.INPUT_SPEC.scalar_fields)))
    for j in range(scalars.shape[1]):
        scalars[:, j] = j + 1
    profiles = np.arange(1.0, 49.0)[None, :, None] * np.ones((n, 1, 1)) * 0.1
    return BuiltInputs(t=0.025 * np.arange(n), scalars=scalars, profiles=profiles,
                       valid=np.ones(n, bool), missing=tuple(missing), resolvers={})


def test_every_value_lands_in_the_column_its_name_names():
    x, ok = elm.preprocess(_built(), _norm())
    assert x.shape == (5, 60) and ok.all()
    smoothed = {"pinj_downsampled", "tinj_downsampled"}
    for j, f in enumerate(elm.INPUT_SPEC.scalar_fields):
        col = x[:, INDEX[f.model_name]]
        if f.model_name in smoothed:
            # upstream's 100 ms boxcar: four taps of the 25 ms grid, so only
            # the interior of a five-row constant survives untouched
            np.testing.assert_allclose(col[2:4], j + 1)
        else:
            np.testing.assert_allclose(col, j + 1)
    for k in range(48):
        np.testing.assert_allclose(
            x[:, INDEX[f"ece_slow_channel_{k + 1}_downsampled"]], (k + 1) * 0.1)


def test_the_two_photodiodes_are_always_exactly_the_training_mean():
    x, _ = elm.preprocess(_built(), _norm(mean=7.0, std=2.0))
    for name in elm.ALWAYS_MEAN_FILLED:
        assert name in INDEX
        np.testing.assert_array_equal(x[:, INDEX[name]], 0.0)


def test_normalisation_is_applied_per_column_from_the_json_constants():
    norm = _norm()
    norm["mean"][INDEX["gas_downsampled"]] = 3.0
    norm["std"][INDEX["gas_downsampled"]] = 4.0
    x, _ = elm.preprocess(_built(), norm)
    # `gas` is the third scalar field, so its raw value is 3.0
    np.testing.assert_allclose(x[:, INDEX["gas_downsampled"]], (3.0 - 3.0) / 4.0)


def test_a_feature_absent_for_the_shot_is_mean_filled_not_read_as_a_raw_zero():
    """`build` turns an absent feature into a raw 0.0, which for CO2 line
    density normalises to a large negative z, not to the training mean."""
    norm = _norm(mean=1.0e14, std=3.0e13)
    built = _built(missing=("co2_v1", "ece"))
    x, _ = elm.preprocess(built, norm)
    np.testing.assert_array_equal(x[:, INDEX["co2_density_slow_v1_downsampled"]], 0.0)
    for k in range(48):
        np.testing.assert_array_equal(
            x[:, INDEX[f"ece_slow_channel_{k + 1}_downsampled"]], 0.0)
    # a chord that WAS measured still gets its own z-score, not a fill
    assert not np.allclose(x[:, INDEX["co2_density_slow_r0_downsampled"]], 0.0)


def test_the_z_limit_narrows_validity_but_never_on_a_mean_filled_column():
    norm = _norm(mean=0.0, std=1.0)
    built = _built()
    built.scalars[2, 0] = 1e6                    # ip, way outside |z| <= 10
    _, ok = elm.preprocess(built, norm)
    assert list(ok) == [True, True, False, True, True]


def _fake_checkpoint(tmp_path, monkeypatch, n_inputs=60):
    """A DSM pickle of this model's width, written through the fork's names."""
    import pickle

    _register_fake_modules(monkeypatch)
    torch.manual_seed(0)
    tm = _FakeTorch()
    tm.embedding = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, 128, bias=False), torch.nn.ReLU6(),
        torch.nn.Dropout(0.2),
    ).double()
    tm.gate = torch.nn.ModuleDict(
        {"1": torch.nn.Sequential(torch.nn.Linear(128, 3, bias=False)).double()})
    tm.scaleg = torch.nn.ModuleDict(
        {"1": torch.nn.Sequential(torch.nn.Linear(128, 3)).double()})
    tm.shapeg = torch.nn.ModuleDict(
        {"1": torch.nn.Sequential(torch.nn.Linear(128, 3)).double()})
    dsm = _FakeDsm()
    dsm.__dict__.update(k=3, dist="LogNormal", temp=1.0, layers=[128],
                        fitted=True, torch_model=tm)
    sm = _FakeSurvivalModel()
    sm.__dict__.update(model="dsm", fitted=True, _model=dsm)
    (tmp_path / "elm_dsm_no_bes.pkl").write_bytes(
        pickle.dumps([[sm, [0.1], [0.2], {"k": 3}]], protocol=4))
    (tmp_path / "normalization.json").write_text(json.dumps(_norm()))
    return tmp_path


def test_predict_is_one_minus_survival_at_the_horizon_plus_one_millisecond(
        tmp_path, monkeypatch):
    model_dir = _fake_checkpoint(tmp_path, monkeypatch)
    predict = elm.load(model_dir)
    built = _built(n=4)
    out = predict(built)
    assert out.shape == (1, 4, 4) and np.isfinite(out).all()

    graph = dsm_pickle.load_dsm(model_dir / "elm_dsm_no_bes.pkl")
    x, _ = elm.preprocess(_built(n=4), _norm())
    want = 1.0 - dsm_pickle.survival(graph, x, horizons_ms=(6.0, 11.0, 21.0, 51.0))
    np.testing.assert_allclose(out[0], want)
    assert [f.name for f in predict.output_spec.fields] == [
        f.name for f in elm.OUTPUT_SPEC.fields]
    attrs = dict(predict.output_spec.fields[2].attrs)
    assert attrs["queried_at_ms"] == "21.0"
    assert attrs["mean_filled_columns"] == "pcphd02_downsampled,pcphd03_downsampled"
    assert attrs["trained_grid_ms"] == "1.0"


def test_predict_narrows_the_validity_mask_in_place(tmp_path, monkeypatch):
    model_dir = _fake_checkpoint(tmp_path, monkeypatch)
    predict = elm.load(model_dir)
    built = _built(n=4)
    built.scalars[1, 1] = 1e9                    # bt, far outside |z| <= 10
    out = predict(built)
    assert list(built.valid) == [True, False, True, True]
    assert np.isfinite(out).all(), "an invalid row still gets a probability"


def test_load_refuses_normalisation_constants_for_another_column_set(
        tmp_path, monkeypatch):
    model_dir = _fake_checkpoint(tmp_path, monkeypatch)
    (model_dir / "normalization.json").write_text(json.dumps(
        {"columns": list(COLUMNS)[:59], "mean": [0.0] * 59, "std": [1.0] * 59}))
    with pytest.raises(ValueError, match="normalization.json"):
        elm.load(model_dir)
