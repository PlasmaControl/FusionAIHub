"""LC2 actuator records: measured units, native clocks, honest missing data."""

import h5py
import numpy as np
import pytest

from labelmaker.features import namespace as ns
from labelmaker.features import resolve_fdp as rf
from labelmaker.features.store import missing_names, present, write_features

POINTS = [
    ("lh_power", r"\RF::LH_POWER", "rf", "kW", "kW", 1.0),
    ("helicon_twapwr", r"\RF::TWAPWR", "rf", " ", "", 1.0),
    ("efc_a1_c_ka", r"\OPERATIONS::CN1IAMP", "operations", "Amps", "kA", .001),
    ("efc_a1_iu_ka", r"\OPERATIONS::IUN1IAMP", "operations", "Amps", "kA", .001),
    ("efc_a1_il_ka", r"\OPERATIONS::ILN1IAMP", "operations", "Amps", "kA", .001),
    ("ecoil_a", "ecoil", None, "a", "A", 1.0),
]


def record(values=(1000., 2000., 3000.), units="Amps", times=(0., 1., 2.)):
    return {"data": np.array(values), "times": np.array(times),
            "units": {"data": units, "times": "ms"}}


@pytest.fixture(autouse=True)
def no_imports(monkeypatch):
    monkeypatch.setattr(rf, "_import_diagnosis", lambda: None)


@pytest.mark.parametrize("name,node,tree,raw_units,units,scale", POINTS)
def test_actuator_resolves_its_measured_point_and_units(
    monkeypatch, tmp_path, name, node, tree, raw_units, units, scale,
):
    def fetch(expr, actual_tree, shot, dims=()):
        assert (expr, actual_tree, shot, dims) == (node, tree, 203505, ())
        return record(units=raw_units)

    monkeypatch.setattr(rf, "_fetch_mds", fetch)
    monkeypatch.setattr(rf, "_fetch_ptdata", lambda expr, shot: fetch(expr, None, shot))
    arrays, missing = rf.resolve(203505, [name], retries=0)
    assert missing == {}
    np.testing.assert_array_equal(arrays[name].x, [0., .001, .002])
    np.testing.assert_allclose(arrays[name].y, [[1000*scale, 2000*scale, 3000*scale]])
    assert arrays[name].attrs["units_from_source"] == raw_units.strip()
    path = tmp_path / "features.h5"
    write_features(path, 203505, arrays, missing)
    with h5py.File(path, "r") as f:
        assert f[name].attrs["units"] == units
        assert f[name].attrs["locator"] == node


@pytest.mark.parametrize("tree,node", [
    ("rf", r"\RF::LH_POWER"),
    ("operations", r"\OPERATIONS::CN1IAMP"),
    ("pellet", r"\PELLET::LGIHI_T"),
    ("d3d", r"\D3D::ECOIL"),
])
def test_qualified_tree_controls_fetch_and_never_uses_efit_clock(monkeypatch, tree, node):
    def fetch(expr, actual_tree, shot, dims=()):
        assert (expr, actual_tree) == (node, tree)
        return record(units="ms")

    monkeypatch.setattr(rf, "_fetch_mds", fetch)
    spec = ns.FeatureSpec("probe", "scalar", "", ("fdp",), (node,))
    got = rf._resolve_one(spec, node, 1, lambda key, thunk: thunk())
    np.testing.assert_array_equal(got.x, [0., .001, .002])


def efc_backend(monkeypatch, *, bad=None):
    records = {
        r"\OPERATIONS::CN1IAMP": record((1000., 4000., 2000.)),
        r"\OPERATIONS::IUN1IAMP": record((3000., 2000., 1000.)),
        r"\OPERATIONS::ILN1IAMP": record((2000., 1000., 5000.)),
    }
    if bad:
        bad(records[r"\OPERATIONS::ILN1IAMP"])
    calls = []

    def fetch(expr, tree, shot, dims=()):
        assert tree == "operations"
        calls.append(expr)
        return records[expr]

    monkeypatch.setattr(rf, "_fetch_mds", fetch)
    return calls


def test_efc_max_is_reproducible_and_components_are_fetched_once(monkeypatch, tmp_path):
    calls = efc_backend(monkeypatch)
    names = ["efc_n1_ka", "efc_a1_c_ka", "efc_a1_iu_ka", "efc_a1_il_ka"]
    arrays, missing = rf.resolve(203505, names, retries=0)
    assert missing == {}
    np.testing.assert_allclose(arrays["efc_n1_ka"].y, [[3., 4., 5.]])
    assert len(calls) == len(set(calls)) == 3
    path = tmp_path / "features.h5"
    write_features(path, 203505, arrays, missing)
    with h5py.File(path, "r") as f:
        assert f["efc_n1_ka"].attrs["units"] == "kA"
        np.testing.assert_array_equal(
            f["efc_n1_ka/ydata"][:],
            np.maximum.reduce([f[n + "/ydata"][:] for n in names[1:]]),
        )


@pytest.mark.parametrize("bad", [
    lambda r: r.update(times=np.array([0., 1., 3.])),
    lambda r: r.update(times=np.array([0., 1., 1.])),
    lambda r: r["units"].update(data="V"),
])
def test_bad_efc_component_is_a_miss_not_a_partial_max(monkeypatch, tmp_path, bad):
    efc_backend(monkeypatch, bad=bad)
    arrays, missing = rf.resolve(1, ["efc_n1_ka", "efc_a1_c_ka"], retries=0)
    assert set(arrays) == {"efc_a1_c_ka"}
    assert missing == {"efc_n1_ka": "ValueError"}
    path = tmp_path / "features.h5"
    write_features(path, 1, arrays, missing)
    assert present(path) == {"efc_a1_c_ka"}
    assert missing_names(path) == missing


def test_nonfinite_efc_component_remains_unknown(monkeypatch):
    efc_backend(monkeypatch, bad=lambda r: r.update(data=np.array([np.nan, np.inf, 5000.])))
    arrays, missing = rf.resolve(1, ["efc_n1_ka"], retries=0)
    assert missing == {}
    np.testing.assert_allclose(arrays["efc_n1_ka"].y, [[np.nan, np.nan, 5.]])


def test_component_recovered_by_max_is_returned_even_when_requested_first(monkeypatch):
    efc_backend(monkeypatch)
    fetch = rf._fetch_mds
    failures = 0

    def transient(expr, *args, **kwargs):
        nonlocal failures
        if expr == r"\OPERATIONS::CN1IAMP" and failures < 2:
            failures += 1
            raise ConnectionError("temporary")
        return fetch(expr, *args, **kwargs)

    monkeypatch.setattr(rf, "_fetch_mds", transient)
    arrays, missing = rf.resolve(1, ["efc_a1_c_ka", "efc_n1_ka"], retries=1)
    assert missing == {}
    np.testing.assert_allclose(arrays["efc_a1_c_ka"].y, [[1., 4., 2.]])
    np.testing.assert_allclose(arrays["efc_n1_ka"].y, [[3., 4., 5.]])


@pytest.mark.parametrize("name,node,tree,raw_units,units,scale", POINTS)
def test_bad_actuator_clock_records_failure(
    monkeypatch, name, node, tree, raw_units, units, scale,
):
    bad = record(units=raw_units, times=(0., 1., np.nan))
    monkeypatch.setattr(rf, "_fetch_mds", lambda *a, **k: bad)
    monkeypatch.setattr(rf, "_fetch_ptdata", lambda *a, **k: bad)
    arrays, missing = rf.resolve(1, [name], retries=0)
    assert arrays == {}
    assert missing == {name: "ValueError"}


@pytest.mark.parametrize("name", ["lh_power", "ecoil_a", "efc_a1_c_ka"])
def test_undeclared_or_wrong_physical_units_are_not_relabelled(monkeypatch, name):
    monkeypatch.setattr(rf, "_fetch_mds", lambda *a, **k: record(units="V"))
    monkeypatch.setattr(rf, "_fetch_ptdata", lambda *a, **k: record(units="V"))
    arrays, missing = rf.resolve(1, [name], retries=0)
    assert arrays == {}
    assert missing == {name: "ValueError"}
