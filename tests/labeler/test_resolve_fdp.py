"""The fdp resolver. Network-free by default; live fetches are opt-in.

The fakes below are shaped like the records the Task 12 Step 1 probe
measured, including the two traps: a geqdsk profile whose `times` dim is the
RADIAL axis in normalized psi, and a zipfit profile whose dim lengths are
the transpose of the order they were requested in.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from labeler.features import namespace as ns
from labeler.features import resolve_fdp as rf


def _zipfit_record(n_t=8, n_x=121, units="10^19 m^-3"):
    """As measured: data (n_t, n_x), `x` in rho, `t_ms` in ms."""
    x = np.linspace(0.0, 1.2, n_x)
    return {
        "data": np.tile(x[None, :], (n_t, 1)),
        "x": x,
        "t_ms": np.linspace(0.0, 175.0, n_t),
        "units": {"data": units, "x": "rho", "t_ms": "ms"},
    }


def _geqdsk_profile_record(n_t=6, n_x=65):
    """As measured: data (n_t, 65), `times` is normalized psi, NOT time."""
    psi = np.linspace(0.0, 1.0, n_x)
    return {
        "data": np.tile(psi[None, :], (n_t, 1)),
        "times": psi,
        "units": {"data": " ", "times": "normalized psi"},
    }


def _scalar_record(n=6, units=" ", first=100.0, step=20.0):
    return {
        "data": np.linspace(1.0, 2.0, n),
        "times": first + step * np.arange(n),
        "units": {"data": units, "times": "ms"},
    }


def _time_record(n=6, first=100.0, step=20.0):
    """A gtime/atime node: its own data is the time axis."""
    t = first + step * np.arange(n)
    return {"data": t, "times": t, "units": {"data": "ms", "times": "ms"}}


# --- _to_rho_grid ----------------------------------------------------------


def test_to_rho_grid_resamples_a_profile_onto_the_canonical_grid():
    coord = np.linspace(0.0, 1.2, 121)           # ZIPFIT's own x axis
    values = np.tile(coord[None, :], (4, 1))     # value == coordinate
    out = rf._to_rho_grid(values, coord, ns.RHO_GRID)
    assert out.shape == (33, 4)                  # (n_rho, T)
    np.testing.assert_allclose(out[:, 0], ns.RHO_GRID, atol=1e-12)


def test_to_rho_grid_handles_a_time_dependent_coordinate():
    # rhovn is one rho axis per time slice, so the helper takes (n_t, n_x).
    coord = np.stack([np.linspace(0.0, 1.0, 65), np.linspace(0.0, 0.5, 65)])
    values = np.stack([np.linspace(0.0, 1.0, 65), np.linspace(0.0, 1.0, 65)])
    out = rf._to_rho_grid(values, coord, ns.RHO_GRID)
    assert out.shape == (33, 2)
    np.testing.assert_allclose(out[:, 0], ns.RHO_GRID, atol=1e-12)
    # second slice: the coordinate only reaches 0.5, so beyond it the
    # profile is NaN rather than an extrapolated edge value
    assert np.isnan(out[ns.RHO_GRID > 0.5, 1]).all()


def test_to_rho_grid_rejects_mismatched_axes():
    with pytest.raises(ValueError, match="does not match"):
        rf._to_rho_grid(np.zeros((4, 65)), np.linspace(0, 1, 33), ns.RHO_GRID)


def test_to_rho_grid_sorts_a_descending_coordinate():
    coord = np.linspace(1.0, 0.0, 65)
    out = rf._to_rho_grid(np.tile(coord[None, :], (2, 1)), coord, ns.RHO_GRID)
    np.testing.assert_allclose(out[:, 0], ns.RHO_GRID, atol=1e-12)


def test_to_rho_grid_skips_a_slice_with_fewer_than_two_finite_points():
    coord = np.stack([np.linspace(0.0, 1.0, 65), np.full(65, np.nan)])
    values = np.zeros((2, 65))
    out = rf._to_rho_grid(values, coord, ns.RHO_GRID)
    assert np.isfinite(out[:, 0]).all()
    assert np.isnan(out[:, 1]).all()


# --- axis identification ---------------------------------------------------


def test_the_time_axis_is_found_by_its_units_not_its_position():
    rec = _scalar_record()
    t_s = rf._time_axis_s(rec, 6)
    np.testing.assert_allclose(t_s, [0.1, 0.12, 0.14, 0.16, 0.18, 0.2])


def test_a_radial_dim_is_never_mistaken_for_a_time_axis():
    # The geqdsk trap: `times` is length 65 in normalized psi.
    assert rf._time_axis_s(_geqdsk_profile_record(), 6) is None


def test_profile_axes_reads_the_radial_axis_and_its_units():
    data, t_s, coord, units = rf._profile_axes(
        _zipfit_record(n_t=8), locator="zip"
    )
    assert data.shape == (8, 121)
    assert units == "rho"
    assert coord.size == 121
    np.testing.assert_allclose(t_s[-1], 0.175)


def test_profile_axes_takes_time_from_the_fallback_when_the_record_has_none():
    rec = _geqdsk_profile_record(n_t=6)
    data, t_s, coord, units = rf._profile_axes(
        rec, locator="geq", fallback=lambda n: np.linspace(0.1, 0.2, n)
    )
    assert data.shape == (6, 65)
    assert units == "normalized psi"
    assert coord.size == 65
    np.testing.assert_allclose(t_s[0], 0.1)


def test_profile_axes_transposes_when_the_dim_lengths_say_so():
    rec = _zipfit_record(n_t=8)
    rec["data"] = rec["data"].T                  # (n_x, n_t) instead
    data, _, coord, _ = rf._profile_axes(rec, locator="zip")
    assert data.shape == (8, 121)                # oriented from the lengths
    assert coord.size == 121


def test_profile_axes_refuses_a_record_with_no_time_at_all():
    with pytest.raises(ValueError, match="no time axis"):
        rf._profile_axes(_geqdsk_profile_record(), locator="geq")


# --- resolve ---------------------------------------------------------------


def test_resolve_records_a_miss_when_toksearch_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        rf, "_import_diagnosis",
        lambda: "toksearch_d3d: ImportError: bad juju",
    )
    got, missing = rf.resolve(190000, ["ip", "kappa"])
    assert got == {}
    assert missing == {
        "ip": "ToksearchUnavailable: toksearch_d3d: ImportError: bad juju",
        "kappa": "ToksearchUnavailable: toksearch_d3d: ImportError: bad juju",
    }
    # The literal prefix survives the diagnosis, so the miss is still
    # classified transient (retryable by a plain re-run) - see store.py.
    from labeler.features.store import is_transient

    assert is_transient(missing["ip"])


def test_resolve_bounds_the_diagnosis_length(monkeypatch):
    monkeypatch.setattr(
        rf, "_import_diagnosis", lambda: "toksearch: ImportError: " + "x" * 500
    )
    _, missing = rf.resolve(190000, ["ip"])
    assert len(missing["ip"]) == rf._CAUSE_MAX_LEN
    assert missing["ip"].startswith("ToksearchUnavailable: toksearch: ImportError:")


def test_the_import_diagnosis_names_the_module_that_failed(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_toksearch_d3d(name, *a, **k):
        if name == "toksearch_d3d":
            raise ImportError(
                "/lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found"
            )
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_toksearch_d3d)
    reason = rf._import_diagnosis()
    assert reason is not None
    assert reason.startswith("toksearch_d3d: ImportError:")
    assert "GLIBCXX_3.4.29" in reason


def test_resolve_records_per_signal_failures(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("ptserver said no")

    monkeypatch.setattr(rf, "_fetch_ptdata", boom)
    monkeypatch.setattr(
        rf, "_fetch_mds", lambda expr, tree, shot, dims=(): _scalar_record()
    )
    got, missing = rf.resolve(190000, ["ip", "kappa"], retries=0)
    assert missing == {"ip": "RuntimeError"}
    assert "kappa" in got


def test_resolve_retries_once_and_keeps_the_second_answer(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)
    calls = []

    def flaky(name, shot):
        calls.append(name)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return {
            "data": np.linspace(0.0, 1e6, 100),
            "times": np.linspace(0.0, 5000.0, 100),
            "units": {"data": "a", "times": "ms"},
        }

    monkeypatch.setattr(rf, "_fetch_ptdata", flaky)
    got, missing = rf.resolve(190000, ["ip"], retries=1)
    assert missing == {}
    assert calls == ["ip", "ip"]
    assert got["ip"].attrs["resolver"] == "fdp"


def test_a_feature_with_no_fdp_source_is_refused():
    with pytest.raises(KeyError, match="fdp"):
        rf.resolve(190000, ["ech_rho"])


def test_a_ptdata_point_is_decimated_to_the_specs_step(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)
    monkeypatch.setattr(
        rf, "_fetch_ptdata",
        lambda name, shot: {
            "data": np.full(2000, 1.5e6),
            "times": np.linspace(0.0, 1000.0, 2000),   # 0.5 ms, as measured
            "units": {"data": "a", "times": "ms"},
        },
    )
    got, missing = rf.resolve(190000, ["ip"])
    assert missing == {}
    arr = got["ip"]
    assert arr.y.shape == (1, 1001)                    # 1 ms over 1 s
    np.testing.assert_allclose(np.diff(arr.x), 0.001)
    assert arr.attrs["decimated_to_s"] == "0.001"
    assert arr.attrs["units_from_source"] == "a"


def test_an_efit_scalar_keeps_its_native_20_ms_cadence(monkeypatch):
    # The plan decimated every scalar to `spec.step or 0.001`, which on a
    # 20 ms EFIT node fabricates 19 empty 1 ms bins out of every 20. A spec
    # with step 0.0 keeps the native rate, as the archive resolver does.
    monkeypatch.setattr(rf, "available", lambda: True)
    monkeypatch.setattr(
        rf, "_fetch_mds",
        lambda expr, tree, shot, dims=(): _scalar_record(n=6),
    )
    got, missing = rf.resolve(190000, ["kappa"])
    assert missing == {}
    arr = got["kappa"]
    assert ns.by_name("kappa").step == 0.0
    assert arr.y.shape == (1, 6)
    np.testing.assert_allclose(arr.x, [0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
    assert "decimated_to_s" not in arr.attrs
    assert np.isfinite(arr.y).all()


def test_a_zipfit_profile_lands_on_the_33_point_grid(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)
    monkeypatch.setattr(
        rf, "_fetch_mds",
        lambda expr, tree, shot, dims=(): _zipfit_record(n_t=8),
    )
    got, missing = rf.resolve(190000, ["ne_zipfit"])
    assert missing == {}
    arr = got["ne_zipfit"]
    assert arr.y.shape == (33, 8)
    np.testing.assert_allclose(arr.y[:, 0], ns.RHO_GRID, atol=1e-12)
    assert arr.attrs["radial_coordinate"] == "rho"
    assert arr.attrs["resampled_from"] == "121 points of rho"


def test_a_geqdsk_profile_takes_its_time_axis_from_gtime(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)

    def fetch(expr, tree, shot, dims=()):
        if expr == rf.GEQDSK_TIME:
            return _time_record(n=6)
        return _geqdsk_profile_record(n_t=6)

    monkeypatch.setattr(rf, "_fetch_mds", fetch)
    got, missing = rf.resolve(190000, ["pres"])
    assert missing == {}
    arr = got["pres"]
    assert arr.y.shape == (33, 6)
    np.testing.assert_allclose(arr.x[0], 0.1)
    # The measured coordinate: uniform normalized psi, not rho. See the
    # module docstring for the archive comparison that settled it.
    assert arr.attrs["radial_coordinate"] == "normalized psi"
    assert arr.attrs["resampled_from"] == "65 points of normalized psi"


def test_a_geqdsk_time_axis_of_the_wrong_length_is_a_miss(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)

    def fetch(expr, tree, shot, dims=()):
        if expr == rf.GEQDSK_TIME:
            return _time_record(n=9)             # disagrees with 6 slices
        return _geqdsk_profile_record(n_t=6)

    monkeypatch.setattr(rf, "_fetch_mds", fetch)
    got, missing = rf.resolve(190000, ["pres"], retries=0)
    assert got == {}
    assert missing == {"pres": "ValueError"}


def test_a_scalar_node_that_comes_back_two_dimensional_is_a_miss(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)
    monkeypatch.setattr(
        rf, "_fetch_mds",
        lambda expr, tree, shot, dims=(): _geqdsk_profile_record(),
    )
    got, missing = rf.resolve(190000, ["kappa"], retries=0)
    assert got == {}
    assert missing == {"kappa": "ValueError"}


def test_a_non_monotonic_time_axis_is_a_miss(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)
    rec = {
        "data": np.ones(2000),
        "times": np.concatenate([np.linspace(0.0, 1000.0, 1999), [-5.0]]),
        "units": {"data": "a", "times": "ms"},
    }
    monkeypatch.setattr(rf, "_fetch_ptdata", lambda name, shot: rec)
    got, missing = rf.resolve(190000, ["ip"], retries=0)
    assert got == {}
    assert missing == {"ip": "ValueError"}


def test_the_time_node_is_fetched_once_for_several_geqdsk_profiles(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)
    seen = []

    def fetch(expr, tree, shot, dims=()):
        seen.append(expr)
        if expr == rf.GEQDSK_TIME:
            return _time_record(n=6)
        return _geqdsk_profile_record(n_t=6)

    monkeypatch.setattr(rf, "_fetch_mds", fetch)
    got, _ = rf.resolve(190000, ["qpsi", "pres"])
    assert set(got) == {"qpsi", "pres"}
    assert seen.count(rf.GEQDSK_TIME) == 1


def test_available_is_false_without_toksearch(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_toksearch(name, *a, **k):
        if name.startswith("toksearch"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_toksearch)
    assert rf.available() is False


# --- regression: torch-first loader ordering --------------------------------

#: `find_spec` reads the package's metadata without executing its `__init__`,
#: so this stays True regardless of whether the GLIBCXX loader-ordering bug
#: (Task 16b) is present - it would be wrong to gate this test on anything
#: that the bug itself makes False, which would make it skip exactly when it
#: is needed. `labeler`'s environment always has `toksearch`, so this
#: never skips there; it exists only so the test also collects harmlessly in
#: an environment that genuinely lacks the package.
_HAS_TOKSEARCH = importlib.util.find_spec("toksearch") is not None


@pytest.mark.skipif(not _HAS_TOKSEARCH, reason="toksearch is not installed here")
def test_available_survives_a_fork_after_torch_is_already_loaded():
    """The regression itself: `import torch` then a forked worker's first
    `toksearch` import must still succeed.

    This is `run.py`'s actual configuration - the worker pool is forked
    before toksearch is imported anywhere, and torch is already loaded in
    the parent by the time it forks (`registry.load_adapter` imports it via
    `spec.py` -> `models/runners/keras_h5.py`). `import torch` binds the
    SYSTEM `/lib64/libstdc++.so.6` unless `pyproject.toml`'s
    `tool.pixi.feature.fdp` activation table puts the pixi env's own copy
    first on the loader's path; without that, every worker's first
    `resolve_fdp.available()` call raises `ImportError` and the whole fdp
    scaling path goes dark with no error anywhere (Task 16b).

    Run in a fresh subprocess, not in-process: this pytest process may have
    already imported `toksearch` (e.g. an earlier test's real, unpatched
    check), and a fork after that would hand the child an already-imported
    module - proving nothing about loader ordering. A subprocess guarantees
    the parent-imports-torch-but-not-toksearch precondition asserted below.
    That exact mistake - forking after the parent had already imported
    toksearch - is called out in the Task 16b brief as the one made while
    diagnosing this.
    """
    script = textwrap.dedent("""
        import multiprocessing
        import sys

        import torch  # noqa: F401 - binds libstdc++ first, as the real runner does

        assert "toksearch" not in sys.modules, (
            "toksearch must not be imported in the parent, or the forked "
            "child inherits it and the test proves nothing"
        )

        def _child():
            # First touch of toksearch in this process happens here, fresh,
            # after the fork - exactly like a real worker's first fetch.
            from labeler.features import resolve_fdp
            sys.exit(0 if resolve_fdp.available() else 1)

        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(target=_child)
        p.start()
        p.join(timeout=60)
        sys.exit(0 if p.exitcode == 0 else 1)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "resolve_fdp.available() was False in a forked child after the "
        "parent had already loaded torch - the loader-ordering regression "
        f"is back.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# --- live ------------------------------------------------------------------


@pytest.mark.live
def test_live_fetch_of_the_reference_points_for_one_shot():
    # This is the original measured reference set, not a claim that every
    # future feature exists on this 2021 shot. LC2's LH/Helicon records are
    # measured on later shots by its pilot; both are TreeNNF on 185945.
    names = [
        "ip", "bt", "r0", "kappa", "tritop", "tribot", "gapin", "betan",
        "qpsi", "pres", "ne_zipfit", "te_zipfit", "rot_zipfit", "qmin",
        "li", "aminor", "volume", "pcbcoil", "ti_zipfit",
    ]
    got, missing = rf.resolve(185945, names)
    assert missing == {}, f"misses on a shot the probe reached: {missing}"
    for name, arr in got.items():
        spec = ns.by_name(name)
        want_c = 33 if spec.kind == "profile" else 1
        assert arr.y.shape[0] == want_c, name
        assert arr.x.ndim == 1 and arr.x.size == arr.y.shape[1]
        assert arr.x[-1] < 30.0, f"{name}: time axis looks like ms, not s"
        assert arr.attrs["resolver"] == "fdp"
        assert np.isfinite(arr.y).any(), name


@pytest.mark.live
def test_live_ip_matches_the_archive_in_amps():
    from labeler.features import resolve_archive as ra
    from labeler.timebase import window_mean

    arch, _ = ra.resolve(185945, ["ip"])
    fdp, missing = rf.resolve(185945, ["ip"])
    assert missing == {}
    a = arch["ip"]
    b = window_mean(fdp["ip"].x, fdp["ip"].y, a.x + rf.ARCHIVE_LAG_S, ns.STEP_S)
    ya, yb = a.y.ravel(), np.asarray(b).ravel()
    good = np.isfinite(ya) & np.isfinite(yb) & (np.abs(ya) > 1e4)
    assert good.sum() > 100
    # No unit factor: PTDATA amps against the archive's amps.
    assert 0.99 < np.median(yb[good] / ya[good]) < 1.01


def test_a_single_slice_tree_is_recorded_as_a_short_record(monkeypatch):
    # MEASURED: efit01 comes back length 1 on shot 190199 and ZIPFIT does on
    # 190124 / 191185. It reads as its own miss class, not a bare ValueError.
    monkeypatch.setattr(rf, "available", lambda: True)
    monkeypatch.setattr(
        rf, "_fetch_mds",
        lambda expr, tree, shot, dims=(): {
            "data": np.array([1.5]),
            "times": np.array([2000.0]),
            "units": {"data": " ", "times": "ms"},
        },
    )
    got, missing = rf.resolve(190199, ["kappa"], retries=0)
    assert got == {}
    assert missing == {"kappa": "ShortRecord"}
    assert issubclass(rf.ShortRecord, ValueError)


def _snan(n):
    """A float32 array of signalling NaNs, as shot 199872's `betan` holds."""
    return np.full(n, 0x7FA00000, dtype=np.uint32).view(np.float32)


def test_a_signalling_nan_does_not_turn_a_good_record_into_a_miss(monkeypatch):
    # Widening a signalling NaN raises the FP invalid flag, which numpy
    # reports as a RuntimeWarning; under -W error that would be caught by
    # resolve's per-signal except and recorded as a miss. MEASURED on shot
    # 199872's betan, which is float32 with signalling NaNs in it.
    monkeypatch.setattr(rf, "available", lambda: True)
    data = np.linspace(1.0, 2.0, 8).astype(np.float32)
    data[3:5] = _snan(2)
    monkeypatch.setattr(
        rf, "_fetch_mds",
        lambda expr, tree, shot, dims=(): {
            "data": data,
            "times": 100.0 + 20.0 * np.arange(8, dtype=np.float32),
            "units": {"data": " ", "times": "ms"},
        },
    )
    got, missing = rf.resolve(199872, ["betan"], retries=0)
    assert missing == {}
    arr = got["betan"]
    assert arr.y.shape == (1, 8)
    assert np.isnan(arr.y[0, 3:5]).all()
    assert np.isfinite(arr.y[0, :3]).all()


def test_as_f64_quietens_a_signalling_nan():
    out = rf._as_f64(_snan(4))
    assert out.dtype == np.float64
    assert np.isnan(out).all()
