"""Profiles for the dock (`review.profiles`) on small fixtures: the header, the body
layout and slicing it at a time, each reader, and the EFIT01 fallback's 202."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.review import profiles
from labeler.events.review.fetch_worker import Fdp, FdpWatch, FetchError, Job
from labeler.events.review.signals import Pending, Signals

#: BARE is farther than mhd.equilibrium.TS_NEAREST from SHOT, so it borrows no chords.
SHOT, BARE = 190001, 185001
REPO_CONFIG = Path(__file__).parents[2] / "configs" / "labeler"
R0, A = 1.7, 0.6
TIMES = np.array([100.0, 300.0, 500.0, 700.0, 900.0])
CHORD_Z = np.array([0.1, 0.2, 0.3])


class FakeFetcher:
    def __init__(self):
        self.jobs, self.errors, self.submitted = {}, {}, []

    def submit(self, key, request, *, shot, ident, where, limit_s, on_result):
        job = self.jobs.get(key) or Job(key, request, shot, ident, where, limit_s,
                                        on_result)
        if key not in self.jobs:
            self.jobs[key] = job
            self.submitted.append(job)
        return job

    def job(self, key):
        return self.jobs.get(key)

    def take_error(self, key):
        return self.errors.pop(key, None)

    def position(self, job):
        return 1

    def summary(self):
        return {"queue": len(self.jobs), "running": None}


def _equilibrium(with_freq=True) -> dict:
    r, z = np.linspace(0.9, 2.5, 65), np.linspace(-1.0, 1.0, 65)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    psi = ((rr - R0) ** 2 + zz**2) / A**2
    psin = np.linspace(0, 1, 65)
    k = np.arange(TIMES.size)[:, None]
    out = {
        "times": TIMES, "atime": TIMES,
        "qpsi": -(1 + 3 * psin[None, :] ** 2 + 0.1 * k),  # signed, as EFIT stores it
        "fpol": np.full((TIMES.size, 65), 3.4), "psirz": np.repeat(psi[None], 5, 0),
        "r": r, "z": z, "ssimag": np.zeros(5), "ssibry": np.ones(5),
        "rmaxis": np.full(5, R0), "a_q95": np.full(5, 4.0), "a_qmin": np.full(5, 1.0),
    }
    if with_freq:
        out["ece_freq"] = np.linspace(80.0, 130.0, 48)
        out["ece_zh"] = np.array([0.0])
    return out


@pytest.fixture
def paths(tmp_path):
    p = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus",
              text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
              label_tables=tmp_path / "events", config_dir=REPO_CONFIG,
              raw_cache=tmp_path / "raw")
    (p.root / "mhd" / "efit02").mkdir(parents=True)
    np.savez(p.root / "mhd" / "efit02" / f"{SHOT}.npz", **_equilibrium())
    np.savez(p.root / "mhd" / "efit02" / f"{BARE}.npz", **_equilibrium(False))
    fusion = p.root / "mhd" / "fusion"
    (fusion / "rot").mkdir(parents=True)
    rows = [f"{SHOT},{ch},1.94,{z}" for ch, z in enumerate(CHORD_Z)]
    (fusion / "ts_geom.csv").write_text("shot,ch,R,Z\n" + "\n".join(rows) + "\n")
    rho = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    np.savez(fusion / "rot" / f"{SHOT}.npz", w01=np.tile(rho * 100, (4, 1)),
             rho01=rho, t01=np.array([400.0, 200.0, 600.0, 800.0]))
    p.corpus.mkdir()
    t_ts = np.linspace(0.1, 0.9, 50)
    for shot in (SHOT, BARE):
        with h5py.File(p.corpus_file(shot), "w") as f:
            for name, value in (("ts_core_temp", 2000.0), ("ts_core_density", 5e19)):
                g = f.create_group(name)
                g.create_dataset("xdata", data=t_ts.astype(np.float32))
                g.create_dataset("ydata", data=np.full((3, 50), value, np.float32))
            g = f.create_group("ece")
            x = np.arange(0, 0.05, 1e-4)
            g.create_dataset("xdata", data=x.astype(np.float32))
            g.create_dataset("ydata", data=np.stack([x * 0 + 1, x * 0 + 3]).astype(
                np.float32))
    p.features.mkdir(parents=True)
    with h5py.File(p.features_file(SHOT), "w") as f:
        g = f.create_group("ece")
        g.create_dataset("xdata", data=np.arange(0.1, 0.9, 1e-3))
        g.create_dataset("ydata", data=np.full((48, 800), 2.5, np.float32))
        g.attrs["resolver"] = "corpus"
    return p


@pytest.fixture
def service(paths):
    watch = FdpWatch(reader=lambda: Fdp(True, None, "ok"))
    return Signals(paths, fetcher=FakeFetcher(), watch=watch)


def _decode(profile, entry):
    header = profile.header(entry)
    values = np.frombuffer(profile.body(), "<f4")
    n_t, n_x = header["n_t"], header["n_x"]
    nx = n_t * n_x if header["x_per_time"] else n_x
    assert values.size == n_t + nx + n_t * n_x
    t = values[:n_t]
    x = values[n_t:n_t + nx].reshape((n_t, n_x) if header["x_per_time"] else (n_x,))
    y = values[n_t + nx:].reshape(n_t, n_x)
    return header, t, x, y


def test_park_q_header_body_and_slicing(service):
    profile = profiles.read(service, SHOT, "q_profile")
    header, t, x, y = _decode(profile, service.resolve("q_profile"))
    assert header == {"n_t": 5, "n_x": 65, "x_kind": "psi_n", "x_per_time": False,
                      "label": "q (EFIT02)", "units": "", "x_units": "",
                      "source": "park", "id": "q_profile"}
    assert np.allclose(t, TIMES) and np.allclose(x, np.linspace(0, 1, 65))
    k = int(np.argmin(np.abs(t - 520.0)))  # the page's slice at the cursor
    assert k == 2
    assert np.allclose(y[k], 1 + 3 * np.linspace(0, 1, 65) ** 2 + 0.2, atol=1e-5)


def test_thomson_is_against_the_outboard_midplane_radius(service):
    profile = profiles.read(service, SHOT, "ts_core_te")
    header, _, x, y = _decode(profile, service.resolve("ts_core_te"))
    assert (header["x_kind"], header["x_per_time"], header["x_units"]) == (
        "R_m", True, "m")
    assert header["label"] == "Thomson core Te (R from EFIT02)"
    assert header["units"] == "keV" and np.allclose(y, 2.0)
    expected = R0 + np.sqrt((1.94 - R0) ** 2 + CHORD_Z**2)
    assert np.allclose(x, expected[None, :], atol=0.01)
    ne = profiles.read(service, SHOT, "ts_core_ne")
    assert np.allclose(ne.y, 5.0)


def test_thomson_without_its_geometry_is_by_chord(service):
    header, _, x, _ = _decode(profiles.read(service, BARE, "ts_core_te"),
                              service.resolve("ts_core_te"))
    assert header["x_kind"] == "channel" and list(x) == [1, 2, 3]


def test_rotation_is_against_rho_in_time_order(service):
    header, t, x, y = _decode(profiles.read(service, SHOT, "rotation"),
                              service.resolve("rotation"))
    assert header["x_kind"] == "rho" and header["n_x"] == 5 and x.max() <= 1.0
    assert list(t) == [200.0, 400.0, 600.0, 800.0]
    assert header["units"] == "krad/s" and np.allclose(y[0], x * 100)


def test_ece_is_against_each_channels_radius_per_time(service, paths):
    from labeler.mhd import equilibrium as mhd_eq

    header, t, x, y = _decode(profiles.read(service, SHOT, "ece_te_r"),
                              service.resolve("ece_te_r"))
    assert (header["x_kind"], header["n_x"], header["x_per_time"]) == ("R_m", 40,
                                                                       True)
    assert np.allclose(y, 2.5)
    eq = mhd_eq.equilibrium(paths, SHOT)
    k = int(np.argmin(np.abs(t - 300.0)))
    assert np.allclose(x[k], eq.slice(300.0)["R"][:40], atol=1e-5)


def test_ece_without_frequencies_is_by_channel_and_averaged_to_1_ms(service):
    header, t, _, y = _decode(profiles.read(service, BARE, "ece_te_r"),
                              service.resolve("ece_te_r"))
    assert header["x_kind"] == "channel" and header["n_x"] == 2
    assert np.allclose(np.diff(t), 1.0) and np.allclose(y, [1.0, 3.0])


def test_efit01_q_is_fetched_when_the_park_has_none(service):
    shot = 190003
    got = profiles.read(service, shot, "q_profile")
    assert isinstance(got, Pending) and got.state == "queued"
    [job] = service.fetcher.submitted
    assert job.request == {"via": "efit01_q"} and job.limit_s == 120.0
    del service.fetcher.jobs[job.key]
    job.on_result({"t_ms": TIMES, "x": np.linspace(0, 1, 65),
                   "y": np.ones((5, 65)), "units": ""})
    header, t, _, y = _decode(profiles.read(service, shot, "q_profile"),
                              service.resolve("q_profile"))
    assert header["source"] == "mds" and header["label"] == "q (EFIT01)"
    assert np.allclose(t, TIMES) and np.allclose(y, 1.0)


def test_no_profile_is_missing_and_a_trace_is_invalid(service):
    with pytest.raises(FetchError) as caught:
        profiles.read(service, 190004, "rotation")
    assert caught.value.kind == "missing"
    with pytest.raises(FetchError) as caught:
        profiles.read(service, SHOT, "ip")
    assert caught.value.kind == "invalid"


def test_available(paths):
    assert profiles.available(paths, SHOT, "park_q")
    assert profiles.available(paths, SHOT, "zipfit_rotation")
    assert profiles.available(paths, SHOT, "thomson_te")
    assert profiles.available(paths, BARE, "ece_te")
    assert not profiles.available(paths, 190004, "park_q")
    assert not profiles.available(paths, BARE, "zipfit_rotation")


def test_per_time_picks_the_nearest_slice_and_drops_far_times():
    by_slice = np.array([[1.0], [2.0], [3.0]])
    got = profiles.per_time(by_slice, [100.0, 200.0, 300.0], np.array(
        [90.0, 149.0, 151.0, 310.0, 401.0]))
    assert list(got[:4, 0]) == [1.0, 1.0, 2.0, 3.0] and np.isnan(got[4, 0])
