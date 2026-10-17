"""The fetched profile readers (`review.profile_fetch`) on small fixtures: the EFIT and
ZIPFIT record layouts, the missing-versus-upstream split for a point the shot lacks,
and `profiles.read` through the fetch queue with a source's scale. Nothing fetches:
`fetch_worker._mds` and `_ptdata` are replaced."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.review import catalog as cat
from labeler.events.review import fetch_worker as fw
from labeler.events.review import profile_fetch as pf
from labeler.events.review import profiles
from labeler.events.review.fetch_worker import Fdp, FdpWatch, FetchError, Job
from labeler.events.review.signals import Pending, Signals

REPO_CONFIG = Path(__file__).parents[2] / "configs" / "labeler"
SHOT = 170659
T_MS = np.array([300.0, 100.0, 200.0, 400.0])  # unsorted on purpose
N_X = 65


def _geqdsk_record(values, radial=True):
    """A GEQDSK 2-D node as toksearch returns it: its `times` dim is the radial axis."""
    return {"data": values, "times": np.linspace(0, 1, N_X),
            "units": {"data": "N / m^2", "times": "normalized psi"}}


class Archive:
    """`fetch_worker._mds` and `_ptdata` stand-ins: a table of records by node."""

    def __init__(self):
        self.records, self.calls, self.ptdata = {}, [], {}

    def mds(self, node, tree, shot, dims=None):
        self.calls.append((node, tree, shot, dims))
        got = self.records[node]
        if isinstance(got, Exception):
            raise got
        return got

    def point(self, name, shot, sleep=None):
        got = self.ptdata[name]
        if isinstance(got, Exception):
            raise got
        return got


@pytest.fixture
def archive(monkeypatch):
    a = Archive()
    monkeypatch.setattr(fw, "_mds", a.mds)
    monkeypatch.setattr(fw, "_ptdata", a.point)
    a.ptdata["ip"] = {"data": np.ones(3)}
    return a


def test_every_spec_is_a_catalog_recipe_that_fetches():
    assert len(pf.SPECS) >= 16
    for name, spec in pf.SPECS.items():
        assert cat.RECIPES[name] == "mds" and name in cat.FETCHING_RECIPES
        assert spec.x_kind in cat.X_KINDS and spec.tag == spec.tree
        node = spec.node
        assert cat.node_problem(spec.tree, node) is None, node
        assert pf.TAGS[name] == spec.tree
    assert pf.TAGS["efit01_q"] == "EFIT01" and "efit01_q" not in pf.SPECS


def test_a_geqdsk_profile_takes_its_times_from_gtime_and_sorts(archive):
    values = np.arange(4 * N_X, dtype=float).reshape(4, N_X)
    archive.records[pf.SPECS["efit02_pres"].node] = _geqdsk_record(values)
    archive.records[r"\EFIT02::TOP.RESULTS.GEQDSK:GTIME"] = {"data": T_MS}
    got = pf.run({"via": "efit02_pres"}, SHOT)
    assert list(got["t_ms"]) == [100.0, 200.0, 300.0, 400.0]
    assert got["x"].shape == (N_X,) and got["x"][0] == 0.0 and got["x"][-1] == 1.0
    assert got["y"].shape == (4, N_X) and got["y"].dtype == np.float32
    assert list(got["y"][:, 0]) == [N_X, 2 * N_X, 0, 3 * N_X]  # rows follow the times
    assert got["units"] == "N / m^2"
    assert archive.calls[0][1:] == ("EFIT02", SHOT, None)


def test_the_q_profile_is_unsigned_and_a_transposed_record_is_turned(archive):
    spec = pf.SPECS["efit02_q"]
    assert spec.absolute
    archive.records[spec.node] = _geqdsk_record(-np.ones((N_X, 4)) * 2.0)
    archive.records[r"\EFIT02::TOP.RESULTS.GEQDSK:GTIME"] = {"data": T_MS}
    got = pf.run({"via": "efit02_q"}, SHOT)
    assert got["y"].shape == (4, N_X) and np.all(got["y"] == 2.0)


def test_a_geqdsk_record_that_does_not_fit_its_times_is_upstream(archive):
    archive.records[pf.SPECS["efit01_pprime"].node] = _geqdsk_record(
        np.ones((3, N_X)))
    archive.records[r"\EFIT01::TOP.RESULTS.GEQDSK:GTIME"] = {"data": T_MS}
    with pytest.raises(FetchError) as caught:
        pf.run({"via": "efit01_pprime"}, SHOT)
    assert caught.value.kind == "upstream"


def test_an_empty_or_nan_profile_is_missing(archive):
    node = pf.SPECS["efit01_fpol"].node
    archive.records[r"\EFIT01::TOP.RESULTS.GEQDSK:GTIME"] = {"data": T_MS}
    for record in (np.empty((0, N_X)), np.full((4, N_X), np.nan), np.ones(N_X)):
        archive.records[node] = _geqdsk_record(record)
        with pytest.raises(FetchError) as caught:
            pf.run({"via": "efit01_fpol"}, SHOT)
        assert caught.value.kind == "missing"


def _zipfit_record(values, t_ms, rho):
    return {"data": values, "t_ms": t_ms, "x": rho,
            "units": {"data": "keV", "x": "rho", "t_ms": "ms"}}


def test_a_zipfit_profile_has_its_own_axes(archive):
    rho = np.linspace(0, 1, 11)
    t = np.array([50.0, 150.0, 250.0])
    spec = pf.SPECS["zipfit01_te"]
    archive.records[spec.node] = _zipfit_record(
        np.arange(33, dtype=float).reshape(3, 11), t, rho)
    got = pf.run({"via": "zipfit01_te"}, SHOT)
    assert list(got["t_ms"]) == list(t) and np.allclose(got["x"], rho)
    assert got["y"].shape == (3, 11) and got["units"] == "keV"
    assert archive.calls[0] == (spec.node, "ZIPFIT01", SHOT, ("x", "t_ms"))


def test_a_zipfit_record_in_radius_by_time_order_is_turned(archive):
    rho, t = np.linspace(0, 1, 11), np.array([50.0, 150.0, 250.0])
    archive.records[pf.SPECS["zipfit02_ne"].node] = _zipfit_record(
        np.arange(33, dtype=float).reshape(11, 3), t, rho)
    got = pf.run({"via": "zipfit02_ne"}, SHOT)
    assert got["y"].shape == (3, 11) and got["y"][1, 0] == 1.0


def test_a_zipfit_record_without_axes_is_missing(archive):
    archive.records[pf.SPECS["zipfit01_ti"].node] = {
        "data": np.ones((3, 11)), "units": {"data": "keV"}}
    with pytest.raises(FetchError) as caught:
        pf.run({"via": "zipfit01_ti"}, SHOT)
    assert caught.value.kind == "missing"


SERVER = ("PtDataError: Error opening network connection: getservbyname failed for "
          "task 'PTSERVER' (also tried 'ptserver'): not in /etc/services")


def test_a_point_the_shot_lacks_is_missing_when_the_archive_still_answers(
        archive, monkeypatch):
    archive.ptdata["DENV1"] = RuntimeError(SERVER)
    with pytest.raises(FetchError) as caught:
        pf.run({"via": "ptdata", "name": "DENV1"}, SHOT)
    assert caught.value.kind == "missing" and "PTDATA DENV1" in caught.value.message
    node = r"\EFIT02::TOP.RESULTS.GEQDSK:PRES"
    archive.records[pf.SPECS["efit02_pres"].node] = RuntimeError(
        "TreeFOPENR: file does not exist")
    with pytest.raises(FetchError) as caught:
        pf.run({"via": "efit02_pres"}, SHOT)
    assert caught.value.kind == "missing" and node in caught.value.message


def test_the_same_error_stays_upstream_when_the_control_fails(archive):
    archive.ptdata["DENV1"] = RuntimeError(SERVER)
    archive.ptdata["ip"] = RuntimeError(SERVER)
    with pytest.raises(RuntimeError, match="PTSERVER"):
        pf.run({"via": "ptdata", "name": "DENV1"}, SHOT)


def test_other_errors_and_catalog_nodes_pass_through(archive):
    archive.records[r"\MHD::N1RMS"] = RuntimeError("some other trouble")
    with pytest.raises(RuntimeError, match="other trouble"):
        pf.run({"via": "mds", "node": r"\MHD::N1RMS", "tree": "MHD"}, SHOT)
    archive.records[r"\MHD::N2RMS"] = {
        "data": np.arange(4.0), "times": np.arange(4.0),
        "units": {"data": "G", "times": "ms"}}
    got = pf.run({"via": "mds", "node": r"\MHD::N2RMS", "tree": "MHD"}, SHOT)
    assert got["units"] == "G" and got["y"].shape == (1, 4)
    with pytest.raises(FetchError, match="unknown fetch"):
        pf.run({"via": "nonsense"}, SHOT)


def test_serve_runs_this_modules_work(monkeypatch):
    seen = {}
    monkeypatch.setattr(fw, "serve", lambda conn, token, work: seen.update(
        conn=conn, token=token, work=work))
    pf.serve("pipe", "tok")
    assert seen == {"conn": "pipe", "token": "tok", "work": pf.run}


# ---- through the service ----


class FakeFetcher:
    def __init__(self):
        self.jobs, self.submitted = {}, []

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
        return None

    def position(self, job):
        return 1

    def summary(self):
        return {"queue": len(self.jobs), "running": None}


@pytest.fixture
def service(tmp_path):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus",
                  text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
                  label_tables=tmp_path / "events", config_dir=REPO_CONFIG,
                  raw_cache=tmp_path / "raw")
    watch = FdpWatch(reader=lambda: Fdp(True, None, "ok"))
    return Signals(paths, fetcher=FakeFetcher(), watch=watch)


PROFILE_IDS = ("pressure_efit", "pprime_efit", "ffprime_efit", "fpol_efit",
               "ne_zipfit", "te_zipfit", "ti_zipfit")


@pytest.mark.parametrize("ident", PROFILE_IDS)
def test_each_new_profile_is_a_catalog_profile_that_asks_the_worker(service, ident):
    entry = service.resolve(ident)
    assert entry.kind == "profile" and entry.axis in ("psi_n", "rho")
    assert entry.units and entry.description and entry.group == "Profiles"
    first = entry.sources[0]
    got = profiles.read(service, SHOT, ident)
    assert isinstance(got, Pending) and got.state == "queued"
    [job] = service.fetcher.submitted
    assert job.request == {"via": first.name} and job.limit_s == fw.LONG_S


def test_a_fetched_profile_is_scaled_into_the_entrys_units(service):
    entry = service.resolve("pressure_efit")
    source = entry.sources[0]
    assert source.name.startswith("efit0") and source.scale == 1e-3
    profiles.read(service, SHOT, "pressure_efit")
    [job] = service.fetcher.submitted
    del service.fetcher.jobs[job.key]
    job.on_result({"t_ms": np.array([100.0, 200.0]), "x": np.linspace(0, 1, N_X),
                   "y": np.full((2, N_X), 5000.0), "units": "N / m^2"})
    got = profiles.read(service, SHOT, "pressure_efit")
    header = got.header(entry)
    assert header["x_kind"] == "psi_n" and header["units"] == "kPa"
    assert header["source"] == "mds"
    assert header["label"] == f"{entry.label} ({pf.TAGS[source.name]})"
    assert np.allclose(got.y, 5.0) and got.y.dtype == np.float32
    assert got.body()  # the wire layout is unchanged


def test_a_zipfit_profile_is_against_rho_with_its_tree_tag(service):
    entry = service.resolve("ti_zipfit")
    profiles.read(service, SHOT, "ti_zipfit")
    [job] = service.fetcher.submitted
    del service.fetcher.jobs[job.key]
    job.on_result({"t_ms": np.array([100.0, 200.0]), "x": np.linspace(0, 1, 7),
                   "y": np.full((2, 7), 2.0), "units": "keV"})
    got = profiles.read(service, SHOT, "ti_zipfit")
    assert got.x_kind == "rho" and got.tag == pf.TAGS[entry.sources[0].name]
    assert got.header(entry)["x_per_time"] is False


def test_the_rotation_entry_falls_back_to_a_fetch_when_the_park_has_none(service):
    entry = service.resolve("rotation")
    assert [s.name for s in entry.sources][:1] == ["zipfit_rotation"]
    assert any(s.fetches for s in entry.sources)
    got = profiles.read(service, 190004, "rotation")
    assert isinstance(got, Pending)
