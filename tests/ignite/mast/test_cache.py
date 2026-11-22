"""Tests of the standardized float32 cache.

The synthetic tests (tiny `ChannelStats`, hand-made `raw` dictionaries) need neither `tokamark`
nor `zarr`; the fidelity test needs the real cache and TokaMark's loader.
"""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import cache, paths, registry, transform
from tokamak_foundation_model.ignite.mast.tasks import SignalSpec
from tokamak_foundation_model.ignite.mast.transform import ChannelStats

FLUX = SignalSpec("magnetics-flux_loop_flux", 0.0002, (15,), -0.25546626077270007, 0.382853499608823)
IP = SignalSpec("summary-ip", 0.00025, (1,), 352977.58724943904, 301642.02514258376)
COIL = SignalSpec("pf_active-coil_voltage", 0.00025, (4,), -6.899173983623492, 3117.527841332377)
SIGNALS = [COIL, FLUX, IP]          # sorted by name, as registry.all_signals() delivers them
T = 400


def _stats(spec, mean=0.0, std=1.0, compress="none", scale=1.0):
    shape = tuple(spec.values_shape)
    return ChannelStats(compress, np.full(shape, scale), np.full(shape, mean), np.full(shape, std))


def _all_stats():
    return {FLUX.name: _stats(FLUX, FLUX.mean, FLUX.std), IP.name: _stats(IP, IP.mean, IP.std),
            COIL.name: _stats(COIL, 0.0, COIL.std)}


def _registry(signals=SIGNALS, roles=None, sat=None):
    """The columns `write_shot` reads: signal, channel, roles, sat_abs."""
    roles = roles or {}
    sat = sat or {}
    rows = []
    for s in signals:
        for c in range(int(np.prod(s.values_shape))):
            rows.append({"signal": s.name, "channel": c, "roles": roles.get(s.name, "inp+out"),
                         "sat_abs": sat.get(s.name, np.nan)})
    return pd.DataFrame(rows)


def _raw(spec, n=T, t0=0.1, seed=0, dt=None):
    rng = np.random.default_rng(seed)
    shape = tuple(spec.values_shape)
    values = spec.mean + spec.std * rng.standard_normal((*shape, n))
    return {"time": t0 + (dt or spec.dt) * np.arange(n), "values": values}


def _absent():
    return {"time": np.array([]), "values": np.array([])}


def _write(tmp_path, raw, signals=SIGNALS, stats=None, reg=None, shot=7, split="train"):
    p = tmp_path / f"{shot}.h5"
    status = cache.write_shot(p, shot, split, raw, signals, stats or _all_stats(), reg if reg is not None else _registry(signals))
    return p, status


# ---------------------------------------------------------------------------- write and read

@pytest.mark.parametrize("compress", ["none", "asinh"])
def test_roundtrip_within_float32(tmp_path, compress):
    raw = {FLUX.name: _raw(FLUX)}
    if compress == "asinh":
        stats = {FLUX.name: _stats(FLUX, mean=0.0, std=1.5, compress="asinh", scale=FLUX.std)}
    else:
        stats = _all_stats()
    p = tmp_path / "7.h5"
    status = cache.write_shot(p, 7, "train", raw, [FLUX], stats, _registry([FLUX]))
    assert status == {FLUX.name: "ok"}
    got = cache.read_shot(p)[FLUX.name]
    assert got.values.dtype == np.float32 and got.values.shape == (15, T)
    back = transform.to_physical(got.values.astype(np.float64), stats[FLUX.name])
    err = np.abs(back - raw[FLUX.name]["values"]).max() / FLUX.std
    assert err < 1e-5
    assert got.t0 == pytest.approx(0.1, abs=1e-12) and got.dt == pytest.approx(FLUX.dt, rel=1e-12)
    assert got.valid.all() and got.saturated is None
    assert got.const_channels.dtype.kind == "i" and got.const_channels.size == 0


def test_file_attributes_and_dataset_layout(tmp_path):
    p, _ = _write(tmp_path, {s.name: _raw(s) for s in SIGNALS}, shot=42, split="val")
    stats_sha = "ab" * 32
    p2 = tmp_path / "43.h5"
    cache.write_shot(p2, 43, "test", {s.name: _raw(s) for s in SIGNALS}, SIGNALS, _all_stats(), _registry(),
                     stats_sha=stats_sha)
    with h5py.File(p, "r") as f:
        assert f.attrs["shot"] == 42 and f.attrs["split"] == "val" and f.attrs["cache_version"] == cache.CACHE_VERSION
        assert set(f.keys()) == {s.name for s in SIGNALS}
        d = f[FLUX.name]
        assert d.dtype == np.float32 and d.chunks is None and d.compression is None
        assert d.attrs["t0"] == pytest.approx(0.1) and d.attrs["dt"] == pytest.approx(FLUX.dt)
        assert d.attrs["dead_nan"] == 0 and d.attrs["const_channels"].size == 0 and "dead_const" not in d.attrs
    with h5py.File(p2, "r") as f:
        assert f.attrs["stats_sha256"] == stats_sha and f.attrs["split"] == "test"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["42.h5", "43.h5"]    # no .tmp left


def test_read_shot_names_and_none_for_unknown(tmp_path):
    p, _ = _write(tmp_path, {FLUX.name: _raw(FLUX), IP.name: _raw(IP)}, [FLUX, IP])
    got = cache.read_shot(p, [IP.name])
    assert list(got) == [IP.name] and got[IP.name].values.shape == (1, T)
    assert cache.read_shot(p, ["nothing-here"]) == {"nothing-here": None}


def test_missing_signal_is_absent_not_zero(tmp_path):
    raw = {IP.name: _absent(), FLUX.name: _raw(FLUX)}
    p, status = _write(tmp_path, raw, [FLUX, IP])
    assert status[IP.name] == "absent" and status[FLUX.name] == "ok"
    got = cache.read_shot(p)
    assert got[IP.name] is None and got[FLUX.name] is not None
    with h5py.File(p, "r") as f:
        assert IP.name not in f


def test_signal_not_in_raw_is_absent(tmp_path):
    p, status = _write(tmp_path, {FLUX.name: _raw(FLUX)}, [FLUX, IP])
    assert status[IP.name] == "absent" and cache.read_shot(p)[IP.name] is None


def test_writer_refuses_dt_mismatch(tmp_path):
    raw = {FLUX.name: _raw(FLUX, dt=FLUX.dt / 10), IP.name: _raw(IP)}
    p, status = _write(tmp_path, raw, [FLUX, IP])
    assert status[FLUX.name] == "dt_mismatch" and status[IP.name] == "ok"
    assert cache.read_shot(p)[FLUX.name] is None
    with h5py.File(p, "r") as f:
        assert FLUX.name not in f


def test_writer_refuses_wrong_shape(tmp_path):
    raw = _raw(FLUX)
    raw["values"] = raw["values"][:10]                      # 10 channels, not 15
    p, status = _write(tmp_path, {FLUX.name: raw}, [FLUX])
    assert status[FLUX.name] == "shape_mismatch" and cache.read_shot(p)[FLUX.name] is None


# ----------------------------------------------------------------------------- validity

def test_dead_channel_is_invalid_and_constant_channel_is_flagged(tmp_path):
    raw = _raw(FLUX)
    raw["values"][3] = 0.5                      # constant: valid, flagged
    raw["values"][7] = np.nan                   # all NaN: dead
    raw["values"][9, :50] = np.nan              # partly NaN: live
    raw["values"][11] = np.nan
    raw["values"][11, 20] = 0.25                # exactly one finite sample: neither dead nor constant
    p, status = _write(tmp_path, {FLUX.name: raw}, [FLUX])
    assert status[FLUX.name] == "ok"
    got = cache.read_shot(p)[FLUX.name]
    valid = got.valid
    assert not valid[7].any()
    assert valid[3].all()
    assert valid[11].sum() == 1 and valid[11, 20]
    live = [c for c in range(15) if c not in (3, 7, 9, 11)]
    assert valid[live].all()
    assert not valid[9, :50].any() and valid[9, 50:].all()
    assert list(got.const_channels) == [3]
    with h5py.File(p, "r") as f:
        assert f[FLUX.name].attrs["dead_nan"] == 1 and list(f[FLUX.name].attrs["const_channels"]) == [3]
    assert cache.summarize_shot(p)["channels"] == {FLUX.name: (1, 1)}


def test_const_channels_are_sorted_flat_indices_of_a_map(tmp_path):
    spec = SignalSpec("equilibrium-psi", 0.005, (2, 3), 0.0, 1.0)
    raw = _raw(spec, n=20)
    raw["values"][1, 0] = 2.0                   # flat channel 3
    raw["values"][0, 2] = -1.0                  # flat channel 2
    p = tmp_path / "1.h5"
    cache.write_shot(p, 1, "train", {spec.name: raw}, [spec], {spec.name: _stats(spec)}, _registry([spec]))
    got = cache.read_shot(p)[spec.name]
    assert list(got.const_channels) == [2, 3] and got.valid.all()


def test_constant_actuator_stays_valid(tmp_path):
    """An actuator held at one value (a beam at 0 W all shot) is information, not a dead sensor."""
    raw = _raw(COIL)
    raw["values"][1] = 0.0                      # constant actuator: valid
    raw["values"][2] = np.nan                   # all NaN: invalid whatever the role
    reg = _registry([COIL], roles={COIL.name: "act"})
    p, status = _write(tmp_path, {COIL.name: raw}, [COIL], reg=reg)
    got = cache.read_shot(p)[COIL.name]
    valid = got.valid
    assert valid[1].all() and not valid[2].any() and valid[0].all() and valid[3].all()
    held = got.values[1].astype(np.float64)
    assert np.allclose(held * COIL.std, 0.0)                # stored as the standardized 0 W
    assert list(got.const_channels) == [1]
    assert cache.summarize_shot(p)["channels"] == {COIL.name: (1, 1)}


def test_a_constant_channel_is_valid_whatever_its_role(tmp_path):
    raw = _raw(COIL)
    raw["values"][0] = 1.0
    for role in ("inp", "act+inp", "out"):
        reg = _registry([COIL], roles={COIL.name: role})
        p, _ = _write(tmp_path, {COIL.name: raw}, [COIL], reg=reg, shot=len(role))
        got = cache.read_shot(p)[COIL.name]
        assert got.valid[0].all() and list(got.const_channels) == [0]


def test_all_invalid_status(tmp_path):
    raw = _raw(IP)
    raw["values"][:] = np.nan
    p, status = _write(tmp_path, {IP.name: raw}, [IP])
    assert status[IP.name] == "all_invalid"
    got = cache.read_shot(p)[IP.name]
    assert got is not None and not got.valid.any()


def test_saturated_samples_are_flagged_and_kept(tmp_path):
    spec = SignalSpec("spectrometer_visible-filter_spectrometer_dalpha_voltage", 0.0002, (1,), 0.0, 1.0)
    raw = _raw(spec)
    raw["values"] = np.clip(raw["values"], -3, 3)
    raw["values"][0, 10] = 5.0
    raw["values"][0, 11] = -5.0
    raw["values"][0, 12] = 4.0                  # below the limit: not saturated
    raw["values"][0, 13] = np.nan
    stats = {spec.name: _stats(spec)}
    reg = _registry([spec], sat={spec.name: 4.99})
    p = tmp_path / "1.h5"
    status = cache.write_shot(p, 1, "train", {spec.name: raw}, [spec], stats, reg)
    assert status[spec.name] == "ok"
    got = cache.read_shot(p)[spec.name]
    assert got.saturated.dtype == bool and got.saturated.shape == got.values.shape
    assert got.saturated[0, 10] and got.saturated[0, 11] and got.saturated.sum() == 2
    assert got.valid[0, 10] and got.valid[0, 11] and not got.valid[0, 13] and not got.saturated[0, 13]
    assert float(got.values[0, 10]) == pytest.approx(5.0, abs=1e-6) and float(got.values[0, 11]) == pytest.approx(-5.0, abs=1e-6)
    with h5py.File(p, "r") as f:
        assert f[spec.name + "__saturated"].shape == f[spec.name].shape


def test_overflow_becomes_invalid(tmp_path):
    raw = _raw(IP)
    raw["values"][0, 5] = IP.mean + IP.std * 1e39        # z = 1e39 > float32 max (3.4e38)
    raw["values"][0, 6] = np.inf
    p, status = _write(tmp_path, {IP.name: raw}, [IP])
    got = cache.read_shot(p)[IP.name]
    assert not got.valid[0, 5] and not got.valid[0, 6]
    assert got.valid[0, 7:].all() and got.valid[0, :5].all() and status[IP.name] == "ok"


def test_asinh_signal_roundtrip(tmp_path):
    st = _stats(IP, mean=0.0, std=2.0, compress="asinh", scale=1000.0)
    raw = _raw(IP)
    raw["values"] = np.sinh(np.linspace(-4, 4, T))[None] * 1000.0
    p = tmp_path / "1.h5"
    cache.write_shot(p, 1, "train", {IP.name: raw}, [IP], {IP.name: st}, _registry([IP]))
    got = cache.read_shot(p)[IP.name]
    back = transform.to_physical(got.values.astype(np.float64), st)
    assert np.isfinite(back).all()
    assert np.abs(back - raw["values"]).max() / 1000.0 < 1e-5


def test_read_shot_refuses_a_float16_file(tmp_path):
    """A stale float16 file must not be read as if it were current."""
    p = tmp_path / "1.h5"
    with h5py.File(p, "w") as f:
        d = f.create_dataset(FLUX.name, data=np.zeros((15, 10), dtype=np.float16))
        d.attrs["t0"], d.attrs["dt"] = 0.0, FLUX.dt
        f.attrs["status"] = json.dumps({FLUX.name: "ok"})
    with pytest.raises(ValueError, match="float32"):
        cache.read_shot(p)
    with pytest.raises(ValueError, match="float32"):
        cache.read_shot(p, [FLUX.name])


# --------------------------------------------------------------------------------- the build

def _fake_loader(missing=(), bad=(), imp=()):
    def load(shot, signals):
        if shot in imp:
            raise ImportError("no tokamark")
        if shot in bad:
            raise RuntimeError(f"cannot read shot {shot}")
        out = {s.name: _raw(s, seed=shot) for s in signals}
        for name in missing:
            out[name] = _absent()
        return out
    return load


@pytest.fixture
def fake_env(monkeypatch):
    """The 3 toy signals instead of the 39; a canned fit; a canned registry; a fake loader."""
    fits = []

    def fit(shots, signals, **kw):
        fits.append((list(shots), kw))
        n = len(shots)
        return _all_stats(), {"n_shots": n, "n_loaded": n, "failed": {}, "missing": {s.name: 0 for s in signals},
                              "degenerate": {s.name: [] for s in signals}}

    reg_calls = []

    def build_reg(zarr_root, sample_shots, *, workers=1):
        reg_calls.append(list(sample_shots))
        df = _registry()
        for col in registry.REGISTRY_COLUMNS:
            if col not in df:
                df[col] = "" if col in registry._STRING_COLUMNS else np.nan
        df["device"] = "MAST"
        return df[list(registry.REGISTRY_COLUMNS)]

    monkeypatch.setattr(registry, "all_signals", lambda: list(SIGNALS))
    monkeypatch.setattr(transform, "fit_stats_with_summary", fit)
    monkeypatch.setattr(registry, "build_registry", build_reg)
    monkeypatch.setattr(transform, "load_shot", _fake_loader())
    monkeypatch.setattr(cache, "_tokamark_revision", lambda: "tm-rev")
    return type("Env", (), {"fits": fits, "reg_calls": reg_calls})


SHOTS = {"train": [1, 2, 3, 4], "val": [5, 6], "test": [7]}


def test_build_writes_every_shot_and_the_manifest(tmp_path, fake_env):
    m = cache.build(tmp_path, SHOTS, workers=1)
    files = sorted(p.name for p in (tmp_path / "shots").iterdir())
    assert files == [f"{s}.h5" for s in range(1, 8)]
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == json.loads(json.dumps(m))
    assert m["stats_sha256"] == transform.stats_sha256(tmp_path / "stats.npz")
    assert m["cache_version"] == cache.CACHE_VERSION and m["tokamark_revision"] == "tm-rev"
    assert m["counts"]["train"]["files"] == 4 and m["counts"]["val"]["files"] == 2 and m["counts"]["test"]["files"] == 1
    assert m["signals"][FLUX.name]["status"] == {"ok": 7}
    assert m["signals"][FLUX.name]["dead_nan"] == 0 and m["signals"][FLUX.name]["const"] == 0
    assert m["errors"] == {} and m["size_bytes"] > 0 and m["wall_seconds"] >= 0
    assert (tmp_path / "registry.csv").exists()
    reg = registry.load_registry(tmp_path / "registry.csv")
    assert reg.loc[reg.signal == FLUX.name, "compress"].eq("none").all()     # statistics attached
    with h5py.File(tmp_path / "shots" / "5.h5", "r") as f:
        assert f.attrs["split"] == "val" and f.attrs["stats_sha256"] == m["stats_sha256"]


def test_build_counts_absent_and_dead_per_signal(tmp_path, fake_env, monkeypatch):
    def load(shot, signals):
        out = {s.name: _raw(s, seed=shot) for s in signals}
        if shot % 2 == 0:
            out[IP.name] = _absent()
        out[FLUX.name]["values"][2] = np.nan
        out[FLUX.name]["values"][3] = 1.0
        return out
    monkeypatch.setattr(transform, "load_shot", load)
    m = cache.build(tmp_path, SHOTS, workers=1)
    assert m["signals"][IP.name]["status"] == {"ok": 4, "absent": 3}
    assert m["signals"][FLUX.name]["dead_nan"] == 7 and m["signals"][FLUX.name]["const"] == 7
    assert m["signals"][COIL.name]["dead_nan"] == 0 and "dead_const" not in m["signals"][COIL.name]


def test_build_statistics_draw_and_registry_sample(tmp_path, fake_env):
    train = list(range(100, 100 + 2500))
    cache.build(tmp_path, {"train": train[:3]}, workers=1, stats_pool=train)
    (stats_shots, kw), = fake_env.fits
    assert len(stats_shots) == 2000 and stats_shots == sorted(stats_shots) and set(stats_shots) <= set(train)
    expected = sorted(int(s) for s in np.random.default_rng(0).choice(train, 2000, replace=False))
    assert stats_shots == expected
    assert fake_env.reg_calls == [expected[::10]]
    assert sorted(p.name for p in (tmp_path / "shots").iterdir() if p.suffix == ".h5") == sorted(f"{s}.h5" for s in train[:3])


def test_build_with_few_training_shots_fits_on_all_of_them(tmp_path, fake_env):
    cache.build(tmp_path, SHOTS, workers=1)
    assert fake_env.fits[0][0] == [1, 2, 3, 4]


def test_build_without_train_and_without_statistics_raises(tmp_path, fake_env):
    with pytest.raises(ValueError, match="train"):
        cache.build(tmp_path, {"val": [5]}, workers=1)
    assert not (tmp_path / "stats.npz").exists()
    cache.build(tmp_path, SHOTS, workers=1)                 # the statistics now exist: val alone is fine
    cache.build(tmp_path, {"val": [5]}, workers=1)


def test_build_reuses_statistics_and_registry(tmp_path, fake_env):
    cache.build(tmp_path, SHOTS, workers=1)
    cache.build(tmp_path, SHOTS, workers=1)
    assert len(fake_env.fits) == 1 and len(fake_env.reg_calls) == 1


def test_failed_fit_does_not_become_a_cache(tmp_path, monkeypatch, fake_env):
    def fit(shots, signals, **kw):
        n = len(shots)
        return _all_stats(), {"n_shots": n, "n_loaded": n - 1, "failed": {1: "RuntimeError: x"},
                              "missing": {}, "degenerate": {}}
    monkeypatch.setattr(transform, "fit_stats_with_summary", fit)
    # 1 of 4 shots failed: below 99%
    with pytest.raises(RuntimeError, match="statistics"):
        cache.build(tmp_path, SHOTS, workers=1)
    assert not (tmp_path / "stats.npz").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_records_the_fit_and_the_asinh_signals(tmp_path, monkeypatch, fake_env):
    stats = _all_stats()
    stats[IP.name] = _stats(IP, mean=0.0, std=1.0, compress="asinh", scale=10.0)
    stats[IP.name].mean[0] = 4.5                           # far from zero: ten scales and more
    summary = {"n_shots": 4, "n_loaded": 4, "failed": {}, "missing": {FLUX.name: 2}, "degenerate": {}}
    monkeypatch.setattr(transform, "fit_stats_with_summary", lambda shots, signals, **kw: (stats, summary))
    m = cache.build(tmp_path, SHOTS, workers=1)
    assert m["asinh_signals"] == [IP.name]
    assert m["asinh_far_from_zero"] == {IP.name: 1}
    assert m["stats_fit"] == summary


def test_build_refuses_other_stats_without_force(tmp_path, fake_env):
    cache.build(tmp_path, SHOTS, workers=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["stats_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different statistics"):
        cache.build(tmp_path, SHOTS, workers=1)
    m = cache.build(tmp_path, SHOTS, workers=1, force=True)
    assert m["stats_sha256"] != "0" * 64


def test_build_skips_existing_and_leaves_no_tmp(tmp_path, fake_env, monkeypatch):
    cache.build(tmp_path, SHOTS, workers=1)
    before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "shots").iterdir()}
    calls = []
    real = _fake_loader()
    monkeypatch.setattr(transform, "load_shot", lambda shot, signals: calls.append(shot) or real(shot, signals))
    m = cache.build(tmp_path, {**SHOTS, "test": [7, 8]}, workers=1)
    assert calls == [8]                                      # only the new shot is loaded
    after = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "shots").iterdir()}
    assert {k: after[k] for k in before} == before
    assert not any(n.endswith(".tmp") for n in after)
    assert m["counts"]["test"] == {"requested": 2, "files": 2, "written": 1, "skipped": 1, "failed": 0}
    assert m["signals"][FLUX.name]["status"] == {"ok": 8}   # the manifest covers skipped files too


def test_build_rewrites_a_file_made_with_other_statistics(tmp_path, fake_env):
    cache.build(tmp_path, {"train": [1]}, workers=1)
    with h5py.File(tmp_path / "shots" / "1.h5", "r+") as f:
        f.attrs["stats_sha256"] = "f" * 64
    m = cache.build(tmp_path, {"train": [1]}, workers=1)
    assert m["counts"]["train"]["written"] == 1
    with h5py.File(tmp_path / "shots" / "1.h5", "r") as f:
        assert f.attrs["stats_sha256"] == m["stats_sha256"]


def test_build_rewrites_a_float16_file_with_the_current_hash(tmp_path, fake_env):
    m = cache.build(tmp_path, {"train": [1]}, workers=1)
    p = tmp_path / "shots" / "1.h5"
    with h5py.File(p, "r+") as f:           # an older float16 file that carries the current hash
        shape = f[FLUX.name].shape
        del f[FLUX.name]
        f.create_dataset(FLUX.name, data=np.zeros(shape, dtype=np.float16))
    m = cache.build(tmp_path, {"train": [1]}, workers=1)
    assert m["counts"]["train"]["written"] == 1
    assert cache.read_shot(p)[FLUX.name].values.dtype == np.float32


def test_a_failing_shot_is_recorded_and_the_others_are_written(tmp_path, fake_env, monkeypatch):
    monkeypatch.setattr(transform, "load_shot", _fake_loader(bad=(2,)))
    m = cache.build(tmp_path, SHOTS, workers=1)
    assert m["errors"] == {"2": "RuntimeError: cannot read shot 2"}
    names = sorted(p.name for p in (tmp_path / "shots").iterdir())
    assert names == [f"{s}.h5" for s in (1, 3, 4, 5, 6, 7)]       # no 2.h5 and no .tmp
    assert m["counts"]["train"]["failed"] == 1 and m["counts"]["train"]["files"] == 3
    assert json.loads((tmp_path / "manifest.json").read_text())["errors"] == m["errors"]


def test_a_write_that_raises_leaves_no_file_and_no_tmp(tmp_path, fake_env, monkeypatch):
    def broken(shot, signals):
        out = _fake_loader()(shot, signals)
        del out[FLUX.name]["values"]
        return out
    monkeypatch.setattr(transform, "load_shot", broken)
    m = cache.build(tmp_path, {"train": [1]}, workers=1)
    assert list(m["errors"]) == ["1"]
    assert list((tmp_path / "shots").iterdir()) == []


def test_import_error_is_not_a_shot_error(tmp_path, fake_env, monkeypatch):
    monkeypatch.setattr(transform, "load_shot", _fake_loader(imp=(2,)))
    with pytest.raises(ImportError):
        cache.build(tmp_path, SHOTS, workers=1)


def test_pick_evenly():
    shots = list(range(100, 200))
    assert cache.pick_evenly(shots, 5) == [100, 125, 150, 174, 199]
    assert cache.pick_evenly(shots, 1) == [100]
    assert cache.pick_evenly(shots, 500) == shots
    assert cache.pick_evenly([], 3) == []
    assert cache.pick_evenly(shots[::-1], 2) == [100, 199]
    with pytest.raises(ValueError):
        cache.pick_evenly(shots, 0)


def test_git_revision_of_a_directory_without_git(tmp_path):
    assert cache._read_git_head(tmp_path) == "unknown"
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert cache._read_git_head(tmp_path) == "unknown"                  # the ref is not there
    (tmp_path / ".git" / "refs" / "heads").mkdir(parents=True)
    (tmp_path / ".git" / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert cache._read_git_head(tmp_path) == "a" * 40
    (tmp_path / ".git" / "refs" / "heads" / "main").unlink()
    (tmp_path / ".git" / "packed-refs").write_text("# pack-refs\n" + "b" * 40 + " refs/heads/main\n")
    assert cache._read_git_head(tmp_path) == "b" * 40
    (tmp_path / ".git" / "HEAD").write_text("c" * 40 + "\n")             # detached
    assert cache._read_git_head(tmp_path) == "c" * 40


# ------------------------------------------------------------------------------ real data

def _fidelity_errors(cache_root, n_shots=5):
    """``{signal: worst relative error over the shots}`` and the shots used; also checks the NaN pattern."""
    manifest = json.loads((cache_root / "manifest.json").read_text())
    assert manifest["stats_sha256"] == transform.stats_sha256(cache_root / "stats.npz"), \
        "manifest and stats.npz disagree"
    stats = transform.load_stats(cache_root / "stats.npz")
    signals = registry.all_signals()
    val = []
    for p in sorted((cache_root / "shots").glob("*.h5")):
        with h5py.File(p, "r") as f:
            if f.attrs["split"] == "val":
                val.append(p)
    assert len(val) >= n_shots, f"only {len(val)} validation shots in the cache"
    chosen = [val[i] for i in np.linspace(0, len(val) - 1, n_shots).round().astype(int)]
    worst: dict[str, float] = {}
    for p in chosen:
        shot = int(p.stem)
        truth = transform.load_shot(shot, signals)
        got = cache.read_shot(p)
        for spec in signals:
            c, t = got[spec.name], truth.get(spec.name)
            has_truth = t is not None and np.asarray(t["values"]).size > 0
            if c is None or not has_truth:
                continue
            x = np.asarray(t["values"], dtype=np.float64)
            assert c.values.shape == x.shape, f"{shot} {spec.name}: shape {c.values.shape} vs {x.shape}"
            back = transform.to_physical(c.values.astype(np.float64), stats[spec.name])
            # the loader's NaN is the cache's NaN
            assert not (np.isfinite(back) & ~np.isfinite(x)).any(), f"{shot} {spec.name}: finite where the loader is NaN"
            both = np.isfinite(back) & np.isfinite(x)
            if not both.any():
                continue
            err = float(np.sqrt(np.mean((back[both] - x[both]) ** 2)) / spec.std)
            worst[spec.name] = max(worst.get(spec.name, 0.0), err)
    return worst, [int(p.stem) for p in chosen]


@pytest.mark.real_data
def test_cache_fidelity_against_the_loader():
    root = paths.cache_dir()
    if not (root / "manifest.json").exists():
        pytest.skip(f"no cache at {root}")
    worst, shots = _fidelity_errors(root)
    assert worst, "no signal was compared"
    bad = {k: v for k, v in worst.items() if v > 1e-3}
    assert not bad, f"relative error above 1e-3 on shots {shots}: {sorted(bad.items(), key=lambda kv: -kv[1])}"
