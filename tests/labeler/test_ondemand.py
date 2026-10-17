"""Shots built when they are opened (`events.ondemand`): which are protected, the
temporary cache and its eviction, the fetch of what a build lacks, the 202 and 502
bodies, and the shot search (docs/labeler/on_demand.md)."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from labeler.config import Paths, atomic_path
from labeler.events import ondemand, raw, shotlists
from labeler.events.catalog.cohort import COHORT_COLUMNS
from labeler.events.review import build as review_build
from labeler.events.review import fetch_worker
from labeler.events.ui import shotlist_routes
from labeler.events.ui.app import API_VERSION, COOKIE, create_app
from labeler.events.verify import NoDataError
from labeler.features import resolve_corpus, resolve_fdp
from labeler.features.store import (
    FeatureArray,
    MissingFeature,
    MissingFeatureFile,
    is_complete,
    missing_names,
    permanent_names,
    present,
    read_feature,
    write_features,
)

from .test_events_ui import _built, _one_trace

ROSTERED = 170815  # on the detachment roster, gold and the catalog
SILVER = 170816  # silver (batch 1) and the catalog
CATALOG_ONLY = 180001
OTHER_CATALOG = 180002
GOLD_ONLY = 171234  # gold, not on the catalog
HELD = 195555  # on a list marked protect
BLIND = 175000
TYPED = 123456  # on no list
ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    f"{ROSTERED},unverified,false,,,\n"
)
REGISTRY = """\
lists:
  catalog: {title: "All shots", kind: dataset, on_demand: true}
  held: {title: "Held", kind: dataset, protect: holdout}
  silver_batch_1: {title: "Silver batch 1", parent: silver, kind: derived, hidden: true}
events:
  detachment: {recommended: [gold, silver, catalog], default: catalog}
"""


@pytest.fixture(autouse=True)
def _no_latch(monkeypatch):
    """No refused token is remembered from another test."""
    monkeypatch.setattr(ondemand, "_refused", None)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def paths(tmp_path):
    tables = tmp_path / "events"
    _write(tables / "events.yaml", "taxonomy:\n  mhd_n1:\n    display: MHD n=1\n"
           "    subcategories:\n      m1: {display: '1/1'}\n")
    for event in ("detachment", "locked_mode", "mhd_n1"):
        _write(tables / event / "shots.csv", ROSTER)
    _write(tables / "gold/gold_shots.csv", "shot,run_day,year,classes,sources,notes\n"
           f"{ROSTERED},20260101,2026,,,\n{GOLD_ONLY},20260101,2026,,,\n")
    _write(tables / "silver/silver_shots.csv",
           "shot,batch,order,pool,classes,sources,notes\n"
           f"{SILVER},1,1,corpus,,format,\n")
    catalog = [ROSTERED, SILVER, 170817, CATALOG_ONLY, OTHER_CATALOG, 190000]
    _write(tables / "shotlists/catalog.csv", "shot,note\n" + "".join(
        f"{shot},\n" for shot in catalog))
    _write(tables / "shotlists/held.csv", f"shot,note\n{HELD},\n")
    _write(tables / "shotlists/silver_batch_1.csv", f"shot,batch,order\n{SILVER},1,1\n")
    _write(tables / "shotlists.yaml", REGISTRY)
    blank = dict.fromkeys(COHORT_COLUMNS, "")
    cohort = pd.DataFrame(
        [blank | {"shot": shot, "queue_rank": rank, "blind": shot == BLIND}
         for rank, shot in enumerate([ROSTERED, SILVER, BLIND], 1)],
        columns=list(COHORT_COLUMNS))
    (tables / "catalog").mkdir()
    cohort.to_csv(tables / "catalog/cohort.csv", index=False)
    return Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables, config_dir=tables, raw_cache=tmp_path / "raw",
    )


@pytest.fixture
def app(paths):
    return create_app(paths=paths, token="secret")


@pytest.fixture
def client(app):
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    return transport


def _array(value=1.0, n=10) -> FeatureArray:
    return FeatureArray(np.arange(n) * 0.1, np.full((1, n), value, dtype="float32"))


def _until(check, timeout=10.0):
    end = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def _open(client, shot, event="detachment"):
    return client.get(f"/api/shot?event={event}&shot={shot}")


def _built_and_served(client, app, shot, event="detachment"):
    """Open `shot`, wait for its build, and return the second open's response."""
    assert _open(client, shot, event).status_code == 202
    app.state.builds.running[(event, shot)].result(timeout=30)
    return _open(client, shot, event)


def _cache_shot(paths, shot, size=100, opened=0.0):
    """A shot directory in the temporary cache holding `size` bytes, opened at `opened`."""
    where = ondemand.shot_dir(paths, shot)
    where.mkdir(parents=True)
    (where / "data.bin").write_bytes(b"x" * size)
    stamp = where / ondemand.STAMP
    stamp.touch()
    os.utime(stamp, (opened, opened))
    return where


# ---- the cache's cap ----


def test_the_cap_is_fifty_gigabytes_unless_the_environment_says_otherwise(monkeypatch):
    monkeypatch.delenv(ondemand.CAP_ENV, raising=False)
    assert ondemand.cap_bytes() == int(50e9)
    monkeypatch.setenv(ondemand.CAP_ENV, "2.5")
    assert ondemand.cap_bytes() == int(2.5e9)
    for bad in ("", "lots", "0", "-3"):  # not a positive number: the default
        monkeypatch.setenv(ondemand.CAP_ENV, bad)
        assert ondemand.cap_bytes() == int(50e9), bad


def test_usage_counts_regular_files_and_not_the_links_in_them(paths, tmp_path):
    outside = tmp_path / "big.bin"
    outside.write_bytes(b"y" * 5000)
    where = _cache_shot(paths, 180001, size=100)
    (where / "link.bin").symlink_to(outside)
    assert ondemand.usage(paths) == 100
    assert [(shot, size) for shot, _, size in ondemand.cached(paths)] == [(180001, 100)]
    assert ondemand.usage(Paths(root=tmp_path / "nowhere", corpus=tmp_path)) == 0


def test_eviction_removes_the_least_recently_opened_until_the_cache_fits(paths):
    for shot, opened in ((180001, 300.0), (180002, 100.0), (180003, 200.0)):
        _cache_shot(paths, shot, opened=opened)
    removed = ondemand.evict(paths, cap=150, now=1e6, grace=0)
    assert removed == [180002, 180003]  # oldest first, stopping once under the cap
    assert [shot for shot, _, _ in ondemand.cached(paths)] == [180001]
    assert ondemand.evict(paths, cap=150, now=1e6, grace=0) == []


def test_eviction_spares_the_kept_the_being_built_and_the_just_opened(paths, monkeypatch):
    for shot in (180001, 180002, 180003, 180004):
        _cache_shot(paths, shot, opened=100.0 + shot)
    monkeypatch.setitem(ondemand._active, 180002, 1)
    assert ondemand.evict(paths, keep=(180001,), cap=0, now=1e6, grace=0) == [
        180003, 180004]
    # Opened within the grace period: still there, so the cache stays over its cap.
    _cache_shot(paths, 180005, opened=time.time())
    assert ondemand.evict(paths, keep=(180001,), cap=0) == []
    assert sorted(shot for shot, _, _ in ondemand.cached(paths)) == [
        180001, 180002, 180005]


def test_the_default_grace_is_read_when_evicting(paths, monkeypatch):
    _cache_shot(paths, 180001, opened=time.time())
    assert ondemand.evict(paths, cap=0) == []
    monkeypatch.setattr(ondemand, "GRACE_S", 0.0)
    assert ondemand.evict(paths, cap=0) == [180001]


def test_touching_a_cached_shot_makes_it_the_most_recent(paths):
    _cache_shot(paths, 180001, opened=100.0)
    _cache_shot(paths, 180002, opened=200.0)
    ondemand.touch(paths, 180001)
    ondemand.touch(paths, 180009)  # not cached: nothing made
    assert not ondemand.shot_dir(paths, 180009).exists()
    assert ondemand.evict(paths, cap=100, now=1e12, grace=0) == [180002]


# ---- the overlay ----


def test_an_overlay_links_the_data_roots_files_and_never_writes_through(paths):
    shot = CATALOG_ONLY
    original = paths.features / f"{shot}_features.h5"
    write_features(original, shot, {"ip": _array(2.0)}, {})
    before = original.read_bytes()
    raw_file = paths.raw_cache / f"{shot}_processed.h5"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"raw")

    overlay = ondemand.overlay_paths(paths, shot)
    assert overlay.features_file(shot) == ondemand.shot_dir(paths, shot) / (
        "features") / f"{shot}_features.h5"
    assert overlay.features_file(shot).is_symlink()
    assert (overlay.raw_cache / raw_file.name).is_symlink()
    assert overlay.spectrogram_file("locked_mode", shot) == ondemand.shot_dir(
        paths, shot) / "spectrograms" / "locked_mode" / f"{shot}.h5"
    assert present(overlay.features_file(shot)) == {"ip"}  # read through the link

    write_features(overlay.features_file(shot), shot, {"pinj_total": _array()}, {})
    assert not overlay.features_file(shot).is_symlink()
    assert present(overlay.features_file(shot)) == {"ip", "pinj_total"}
    assert original.read_bytes() == before and present(original) == {"ip"}
    # Paths as the pipelines and the readers of production data build them never see
    # the cache, whether or not the root holds the file (M9).
    assert paths.features_file(shot) == original
    original.unlink()
    assert paths.features_file(shot) == original and not original.exists()
    # Only the server's own Paths (`read_cache`) falls back to the cache's file.
    serving = replace(paths, read_cache=True)
    assert present(serving.features_file(shot)) == {"ip", "pinj_total"}
    assert serving.features_file(shot) == overlay.features_file(shot)


def test_a_shot_with_nothing_on_disk_gets_an_empty_overlay(paths):
    overlay = ondemand.overlay_paths(paths, TYPED)
    assert ondemand.shot_dir(paths, TYPED).is_dir()
    assert not overlay.features_file(TYPED).exists()
    assert paths.features_file(TYPED) == paths.features / f"{TYPED}_features.h5"
    ondemand._discard_if_empty(paths, TYPED)
    assert not ondemand.shot_dir(paths, TYPED).exists()


# ---- which shots are protected ----


def test_protected_shots_build_in_the_data_root_and_the_rest_in_the_cache(paths):
    def protected(shot, event="detachment"):
        return ondemand.protected(paths, event, shot)

    assert protected(ROSTERED)  # the event's roster, and gold
    assert protected(GOLD_ONLY)  # gold, off every roster
    assert protected(SILVER)  # silver
    assert protected(HELD)  # a list marked protect
    assert protected(ROSTERED, "mhd_n1.m1")  # a subcategory reads its parent's roster
    assert not protected(CATALOG_ONLY) and not protected(TYPED)
    assert not protected(CATALOG_ONLY, "mhd_n1.m1")
    _built(paths, event="detachment", shot=OTHER_CATALOG, current=True)
    assert protected(OTHER_CATALOG)  # a store already in the data root stays there
    assert protected(OTHER_CATALOG) and not protected(OTHER_CATALOG, "locked_mode")


def test_a_shot_is_openable_when_it_has_six_digits_and_is_not_blind(paths):
    assert ondemand.openable(paths, TYPED)
    assert not ondemand.openable(paths, BLIND)
    assert not ondemand.openable(paths, 99999)
    assert not ondemand.openable(paths, 1_000_000)


# ---- the missing feature, fetched and the build retried ----


@pytest.fixture
def needs_pinj(monkeypatch):
    """A detachment builder that needs `pinj_total` in the shot's feature file, with a
    stand-in for the fetch that writes it. Returns the calls the two made."""
    calls = SimpleNamespace(builds=[], fetches=[], report=None)

    def build(event, shot, paths):
        calls.builds.append(shot)
        features = paths.features_file(shot)
        if "pinj_total" not in present(features):
            raise MissingFeature("pinj_total", features)
        return _one_trace(event, shot, paths)

    def fetch(shot, names, paths):
        calls.fetches.append((shot, tuple(names)))
        if calls.report is not None:
            return calls.report
        write_features(paths.features_file(shot), shot, {"pinj_total": _array()}, {})
        return ondemand.FetchReport(fetched={"pinj_total": "fdp"})

    monkeypatch.setitem(review_build.BUILDERS, "detachment", build)
    monkeypatch.setattr(ondemand, "fetch_features", fetch)
    return calls


def test_a_protected_shot_missing_a_feature_fetches_it_and_builds_twice(
    client, app, paths, needs_pinj,
):
    served = _built_and_served(client, app, ROSTERED)
    assert served.status_code == 200 and len(served.json()["rows"]) == 1
    assert needs_pinj.builds == [ROSTERED, ROSTERED]
    assert needs_pinj.fetches == [(ROSTERED, ("pinj_total",))]
    # A protected shot's feature file and store are the data root's.
    assert present(paths.features / f"{ROSTERED}_features.h5") == {"pinj_total"}
    assert paths.spectrogram_file("detachment", ROSTERED).is_relative_to(paths.root)
    assert not paths.ondemand.exists()


def test_an_unprotected_shot_builds_and_fetches_into_the_temporary_cache(
    client, app, paths, needs_pinj,
):
    served = _built_and_served(client, app, CATALOG_ONLY)
    assert served.status_code == 200
    assert needs_pinj.builds == [CATALOG_ONLY] * 2 and len(needs_pinj.fetches) == 1
    where = ondemand.shot_dir(paths, CATALOG_ONLY)
    assert present(where / "features" / f"{CATALOG_ONLY}_features.h5") == {"pinj_total"}
    assert (where / "spectrograms/detachment" / f"{CATALOG_ONLY}.h5").is_file()
    # Nothing of it reached the data root.
    assert not (paths.features / f"{CATALOG_ONLY}_features.h5").exists()
    assert not (paths.spectrograms / "detachment" / f"{CATALOG_ONLY}.h5").exists()
    # A later open is served from the cache, no build, no fetch.
    assert _open(client, CATALOG_ONLY).status_code == 200
    assert needs_pinj.builds == [CATALOG_ONLY] * 2 and len(needs_pinj.fetches) == 1


def test_a_shot_off_every_list_opens_when_its_number_is_typed(client, app, paths, monkeypatch):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", _one_trace)
    served = _built_and_served(client, app, TYPED)
    assert served.status_code == 200
    assert served.json()["tier"] == "unverified"
    assert ondemand.shot_dir(paths, TYPED).is_dir()
    assert _open(client, 12345).status_code == 404  # five digits is no shot
    assert _open(client, BLIND).status_code == 404  # a blind shot opens nowhere


def test_a_fetch_that_gets_nothing_says_the_signal_is_missing(
    client, app, needs_pinj,
):
    needs_pinj.report = ondemand.FetchReport(failed={"pinj_total": "fdp: no data"})
    assert _open(client, CATALOG_ONLY).status_code == 202
    with pytest.raises(MissingFeature):
        app.state.builds.running[("detachment", CATALOG_ONLY)].result(timeout=30)
    assert needs_pinj.builds == [CATALOG_ONLY]  # no second build without a fetch
    response = _open(client, CATALOG_ONLY)
    assert response.status_code == 502
    body = response.json()
    assert set(body) == {"error", "detail", "code"}
    assert body["code"] == "missing_signal"
    assert body["error"] == f"No beam power (pinj_total) for {CATALOG_ONLY}"
    assert "fdp: no data" in body["detail"] and "MissingFeature" in body["detail"]


def test_a_refused_fetch_says_fetching_is_off(client, app, paths, needs_pinj):
    needs_pinj.report = ondemand.FetchReport(
        refused="fetching is off on this server (LABELER_NO_FETCH is set)")
    assert _open(client, CATALOG_ONLY).status_code == 202
    with pytest.raises(MissingFeature):
        app.state.builds.running[("detachment", CATALOG_ONLY)].result(timeout=30)
    body = _open(client, CATALOG_ONLY).json()
    assert body["code"] == "fetch_off"
    assert body["error"] == f"No beam power (pinj_total) for {CATALOG_ONLY}: fetching is off"
    assert not ondemand.shot_dir(paths, CATALOG_ONLY).exists()  # an empty cache dir goes


def test_a_failed_open_is_retried_by_the_next_one(client, app, needs_pinj):
    needs_pinj.report = ondemand.FetchReport(failed={"pinj_total": "fdp: no data"})
    for _ in range(2):
        assert _open(client, CATALOG_ONLY).status_code == 202
        with pytest.raises(MissingFeature):
            app.state.builds.running[("detachment", CATALOG_ONLY)].result(timeout=30)
        assert _open(client, CATALOG_ONLY).status_code == 502
    assert needs_pinj.builds == [CATALOG_ONLY] * 2


def test_a_generic_panel_names_every_missing_trace_at_once(paths):
    from labeler.events.panels import _generic

    shot = CATALOG_ONLY
    write_features(paths.features_file(shot), shot, {"ip": _array()}, {})
    with pytest.raises(MissingFeature) as error:
        _generic.panels(shot, paths=paths)
    assert error.value.names == ("betan", "pinj_total")  # one fetch covers both
    assert error.value.name == "betan"
    assert isinstance(error.value, KeyError)


def test_an_absent_feature_file_is_still_a_file_not_found(paths):
    with pytest.raises(FileNotFoundError) as error:
        read_feature(paths.features_file(CATALOG_ONLY), "ip")
    assert isinstance(error.value, MissingFeatureFile)
    assert isinstance(error.value, KeyError) and error.value.names == ("ip",)


# ---- the fetch itself, with the network and the sources replaced ----


@pytest.fixture
def fetching_on(monkeypatch):
    """`fetch_worker.refusal` says fetching is possible."""
    monkeypatch.setattr(fetch_worker, "WATCH", SimpleNamespace(get=lambda: None))
    monkeypatch.setattr(fetch_worker, "refusal", lambda fdp: None)


@pytest.fixture
def fetching_off(monkeypatch):
    monkeypatch.setattr(fetch_worker, "WATCH", SimpleNamespace(get=lambda: None))
    monkeypatch.setattr(fetch_worker, "refusal", lambda fdp: fetch_worker.FetchError(
        "auth", "fetching is off on this server (LABELER_NO_FETCH is set)"))


def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("went to the network")

    monkeypatch.setattr(raw, "_fetch", refuse)
    monkeypatch.setattr(ondemand, "_from_fdp", refuse)


def test_a_feature_in_the_raw_cache_needs_no_fetch_even_with_fetching_off(
    paths, monkeypatch, fetching_off,
):
    _no_network(monkeypatch)
    shot = CATALOG_ONLY
    held = _array(3.0)
    monkeypatch.setattr(raw, "_holds_record", lambda s, group, root: True)
    monkeypatch.setattr(resolve_corpus, "resolve", lambda s, names, *, corpus: (
        ({n: held for n in names}, {}) if Path(corpus) == paths.raw_cache
        else ({}, {n: "FileNotFoundError" for n in names})))
    report = ondemand.fetch_features(shot, ["pinj_total"], paths)
    assert report.fetched == {"pinj_total": "corpus"} and not report.failed
    assert report.refused is None  # nothing was missing on disk, so nothing refused
    assert present(paths.features_file(shot)) == {"pinj_total"}
    assert ondemand.fetching_signals(shot) == []


def test_a_group_missing_from_disk_is_not_fetched_when_fetching_is_off(
    paths, monkeypatch, fetching_off,
):
    _no_network(monkeypatch)
    monkeypatch.setattr(raw, "_holds_record", lambda s, group, root: False)
    monkeypatch.setattr(resolve_corpus, "resolve", lambda s, names, *, corpus: (
        {}, {n: "FileNotFoundError" for n in names}))
    report = ondemand.fetch_features(CATALOG_ONLY, ["pinj_total"], paths)
    assert not report.fetched and "LABELER_NO_FETCH" in report.refused
    assert report.failed == {"pinj_total": "corpus: fetching is off"}
    assert not paths.features_file(CATALOG_ONLY).exists()


def test_a_missing_raw_group_is_fetched_into_the_raw_cache_then_read(
    paths, monkeypatch, fetching_on,
):
    fetched = []
    stored = {}

    def fetch(shot, group, *, channels, t_range, paths):
        fetched.append((shot, group, paths.raw_cache))
        stored["done"] = True

    monkeypatch.setattr(raw, "_fetch", fetch)
    monkeypatch.setattr(ondemand, "_from_fdp", lambda *a: pytest.fail("fdp is not asked"))
    monkeypatch.setattr(raw, "_holds_record", lambda s, group, root: stored.get("done", False))
    where = ondemand.overlay_paths(paths, CATALOG_ONLY)
    monkeypatch.setattr(resolve_corpus, "resolve", lambda s, names, *, corpus: (
        ({n: _array(4.0) for n in names}, {}) if Path(corpus) == where.raw_cache
        else ({}, {n: "FileNotFoundError" for n in names})))
    report = ondemand.fetch_features(CATALOG_ONLY, ["pinj_total"], where)
    assert fetched == [(CATALOG_ONLY, "pinj", where.raw_cache)]  # the overlay's cache
    assert report.fetched == {"pinj_total": "corpus"}


def test_a_feature_without_a_corpus_route_comes_from_fdp(paths, monkeypatch, fetching_on):
    asked = []

    def from_fdp(shot, names):
        asked.append((shot, list(names)))
        return {"ip": _array(5.0)}, {"betan": "no data"}

    monkeypatch.setattr(ondemand, "_from_fdp", from_fdp)
    report = ondemand.fetch_features(CATALOG_ONLY, ["ip", "betan", "ip"], paths)
    assert asked == [(CATALOG_ONLY, ["ip", "betan"])]  # one batched call, names once
    assert report.fetched == {"ip": "fdp"} and report.failed == {"betan": "fdp: no data"}
    assert present(paths.features_file(CATALOG_ONLY)) == {"ip"}
    assert "ip (fdp)" in report.text() and "betan: fdp: no data" in report.text()


def test_fdp_is_not_asked_while_the_server_cannot_fetch(paths, monkeypatch, fetching_off):
    _no_network(monkeypatch)
    report = ondemand.fetch_features(CATALOG_ONLY, ["ip", "no_such_feature"], paths)
    assert not report.fetched and "LABELER_NO_FETCH" in report.refused
    assert report.failed == {"no_such_feature": "no such feature",
                             "ip": "fdp: fetching is off"}
    assert not paths.features_file(CATALOG_ONLY).exists()


def test_an_error_from_fdp_is_recorded_without_losing_the_rest(
    paths, monkeypatch, fetching_on,
):
    def from_fdp(shot, names):
        raise RuntimeError("secret detail at /scratch/x")

    monkeypatch.setattr(ondemand, "_from_fdp", from_fdp)
    report = ondemand.fetch_features(CATALOG_ONLY, ["ip"], paths)
    # The message is kept (the 502 detail has it; the short line never does).
    assert report.failed == {"ip": "fdp: RuntimeError: secret detail at /scratch/x"}


# ---- a fetch failure is never a permanent miss (I1) ----


def test_a_fetch_records_no_miss_so_the_features_pipeline_still_tries_it(
    paths, monkeypatch, fetching_on,
):
    monkeypatch.setattr(ondemand, "_from_fdp", lambda shot, names: (
        {"ip": _array(5.0)}, {"betan": "RuntimeError: HTTP 500 from the archive"}))
    report = ondemand.fetch_features(CATALOG_ONLY, ["ip", "betan"], paths)
    assert report.fetched == {"ip": "fdp"} and set(report.failed) == {"betan"}
    path = paths.features_file(CATALOG_ONLY)
    assert present(path) == {"ip"}
    assert missing_names(path) == {} and permanent_names(path) == set()
    assert not is_complete(path, ["ip", "betan"])  # the pipeline will fetch betan


def test_a_one_sample_fetch_is_a_failure_and_is_not_written(paths, monkeypatch, fetching_on):
    monkeypatch.setattr(ondemand, "_from_fdp", lambda shot, names: (
        {"ip": _array(5.0, n=1)}, {}))
    report = ondemand.fetch_features(CATALOG_ONLY, ["ip"], paths)
    assert not report.fetched and report.failed == {"ip": "fdp: fewer than two samples"}
    assert not paths.features_file(CATALOG_ONLY).exists()


# ---- the feature file's writers (I3) ----


def test_a_feature_write_waits_for_the_paths_writer_lock(paths):
    path = paths.features_file(CATALOG_ONLY)
    with raw._lock_for(path.resolve()):
        writer = threading.Thread(target=ondemand._write_features,
                                  args=(CATALOG_ONLY, {"ip": _array()}, paths))
        writer.start()
        writer.join(0.3)
        assert writer.is_alive() and not path.exists()  # held back, not racing
    writer.join(10)
    assert present(path) == {"ip"}


def test_concurrent_feature_writes_for_one_shot_keep_every_feature(paths):
    names = ("ip", "betan", "bt", "li")
    errors = []

    def write(name):
        try:
            for _ in range(4):
                ondemand._write_features(CATALOG_ONLY, {name: _array()}, paths)
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=write, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    path = paths.features_file(CATALOG_ONLY)
    assert not errors and present(path) == set(names)
    assert not list(path.parent.glob("*.tmp"))


def test_a_unique_working_copy_is_named_after_the_process_and_thread(tmp_path):
    with atomic_path(tmp_path / "f.h5") as tmp:
        assert tmp.name == "f.h5.tmp"
        tmp.write_bytes(b"a")
    with atomic_path(tmp_path / "f.h5", unique=True) as tmp:
        assert tmp.name == f"f.h5.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_bytes(b"b")
    assert (tmp_path / "f.h5").read_bytes() == b"b" and len(list(tmp_path.iterdir())) == 1


# ---- one fdp session, with a timeout (I2) ----


def _lock_is_taken() -> bool:
    """Whether another thread finds the fdp lock held."""
    got = []

    def try_it():
        got.append(resolve_fdp.FETCH_LOCK.acquire(blocking=False))
        if got[0]:
            resolve_fdp.FETCH_LOCK.release()

    thread = threading.Thread(target=try_it)
    thread.start()
    thread.join()
    return not got[0]


def _hold_fdp_lock():
    """Hold the fdp lock in another thread; returns `(release, thread)`."""
    held, release = threading.Event(), threading.Event()

    def hold():
        with resolve_fdp.one_call():
            held.set()
            release.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(10)
    return release, thread


def test_both_toksearch_primitives_hold_the_one_fdp_lock(monkeypatch):
    seen = []

    class Fake:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, shot):
            seen.append(_lock_is_taken())
            return {}

    monkeypatch.setitem(sys.modules, "toksearch_d3d", SimpleNamespace(PtDataSignal=Fake))
    monkeypatch.setitem(sys.modules, "toksearch", SimpleNamespace(MdsSignal=Fake))
    resolve_fdp._fetch_ptdata("ip", CATALOG_ONLY)
    resolve_fdp._fetch_mds("x", "efit01", CATALOG_ONLY)
    assert seen == [True, True] and not _lock_is_taken()  # held for the call only


def test_a_call_waiting_for_a_held_fdp_lock_gives_up_with_fdp_busy(monkeypatch):
    monkeypatch.setattr(resolve_fdp, "LOCK_TIMEOUT_S", 0.05)
    release, thread = _hold_fdp_lock()
    try:
        with pytest.raises(resolve_fdp.FdpBusy, match="fdp is busy"), resolve_fdp.one_call():
            pytest.fail("the lock was not held")
    finally:
        release.set()
        thread.join(10)
    with resolve_fdp.one_call():  # free again
        pass


def test_a_held_fdp_session_stops_the_fetch_once_and_says_so(
    paths, monkeypatch, fetching_on,
):
    monkeypatch.setattr(resolve_fdp, "_import_diagnosis", lambda: None)
    monkeypatch.setattr(resolve_fdp, "LOCK_TIMEOUT_S", 0.05)
    calls = []

    def one(spec, locator, shot, cached):
        calls.append(spec.name)
        with resolve_fdp.one_call():
            return _array()

    monkeypatch.setattr(resolve_fdp, "_resolve_one", one)
    release, thread = _hold_fdp_lock()
    try:
        report = ondemand.fetch_features(CATALOG_ONLY, ["ip", "betan"], paths)
    finally:
        release.set()
        thread.join(10)
    assert calls == ["ip"]  # not retried, and the next signal does not wait again
    assert not report.fetched and set(report.failed) == {"ip", "betan"}
    assert report.refused == "fdp is busy"
    assert not paths.features_file(CATALOG_ONLY).exists()
    error = MissingFeature("ip", "x", names=["ip", "betan"])
    error.ondemand = report
    failure = ondemand.explain(error, "detachment", CATALOG_ONLY)
    assert failure.code == "fetch_off" and failure.short.endswith("fdp is busy, try again")


def test_the_raw_fetch_does_not_wait_twice_for_a_held_fdp_lock(monkeypatch):
    calls = []

    def busy(*args, **kwargs):
        calls.append(1)
        try:
            raise resolve_fdp.FdpBusy("fdp is busy: another fetch held it")
        except resolve_fdp.FdpBusy as error:
            raise NoDataError(f"'x' failed to fetch over fdp: {error}") from error

    monkeypatch.setattr(raw, "fdp_signal", busy)
    monkeypatch.setattr(raw.time, "sleep", lambda s: pytest.fail("a second wait"))
    with pytest.raises(raw.UpstreamError) as error:
        raw._fetch_with_one_retry(CATALOG_ONLY, raw.FETCH_SPECS["pinj"], 3, "pinj")
    assert calls == [1] and "fdp is busy" in str(error.value)
    assert "tried twice" not in str(error.value)


# ---- a refused token stops the fetching until another is read (I4, M4) ----


@pytest.fixture
def token(monkeypatch):
    """The fdp token the server reads; set `.digest` to change it."""
    box = SimpleNamespace(digest="aaaa")
    monkeypatch.setattr(fetch_worker, "WATCH", SimpleNamespace(
        get=lambda: SimpleNamespace(digest=box.digest)))
    monkeypatch.setattr(fetch_worker, "refusal", lambda fdp: None)
    return box


def test_an_fdp_401_stops_every_fetch_until_the_token_changes(
    paths, monkeypatch, token,
):
    asked = []

    def from_fdp(shot, names):
        asked.append(list(names))
        return {}, dict.fromkeys(names, "RuntimeError: HTTP 401 unauthorized")

    monkeypatch.setattr(ondemand, "_from_fdp", from_fdp)
    first = ondemand.fetch_features(CATALOG_ONLY, ["ip"], paths)
    assert asked == [["ip"]] and first.refused == "fdp refused the request"
    assert ondemand.auth_refused() == ondemand.REFUSED_NOTE

    second = ondemand.fetch_features(OTHER_CATALOG, ["betan"], paths)
    assert asked == [["ip"]]  # fdp is not asked again with the refused token
    assert second.refused == ondemand.REFUSED_NOTE
    assert second.failed == {"betan": "fdp: fetching is off"}
    # The raw groups a panel fetches for itself stop too.
    with pytest.raises(raw.FetchDisabledError, match="fdp refused the token"):
        raw._fetch(CATALOG_ONLY, "pinj", channels=[0], t_range=None, paths=paths)
    failure = ondemand.explain(
        raw.FetchDisabledError(ondemand.REFUSED_NOTE), "detachment", CATALOG_ONLY)
    assert failure.code == "fetch_off"
    assert failure.short.endswith("fdp refused the request, log in again")

    token.digest = "bbbb"  # `fdp login` wrote a new token
    assert ondemand.auth_refused() is None
    ondemand.fetch_features(CATALOG_ONLY, ["ip"], paths)
    assert asked == [["ip"], ["ip"]]


def test_a_panels_own_fetch_refused_by_fdp_latches_the_token(paths, monkeypatch, token):
    def refused(event, shot, paths):
        raise raw.UpstreamError("'x' failed to fetch over fdp: HTTP 403 forbidden")

    monkeypatch.setitem(review_build.BUILDERS, "detachment", refused)
    with pytest.raises(raw.UpstreamError) as error:
        ondemand.build("detachment", CATALOG_ONLY, paths)
    assert ondemand.auth_refused() == ondemand.REFUSED_NOTE
    assert ondemand.explain(error.value, "detachment", CATALOG_ONLY).code == "fetch_off"


def test_a_busy_fdp_does_not_latch_the_token(paths, monkeypatch, token):
    ondemand._learn("RuntimeError: fdp is busy: another fetch held it")
    assert ondemand.auth_refused() is None


def test_an_upstream_error_while_the_server_cannot_fetch_says_why(monkeypatch):
    monkeypatch.setattr(fetch_worker, "WATCH", SimpleNamespace(get=lambda: None))
    monkeypatch.setattr(fetch_worker, "refusal", lambda fdp: fetch_worker.FetchError(
        "auth", "the fdp token has expired: run 'fdp login', then try again"))
    failure = ondemand.explain(raw.UpstreamError("fdp answered 503"), "locked_mode", 180001)
    assert (failure.code, failure.short) == (
        "fetch_off", "No data for 180001: fdp token expired")


# ---- the cache stays off the pipelines' paths (C1) ----


def test_pipeline_paths_never_return_a_cache_path_for_an_unprotected_shot(
    paths, monkeypatch,
):
    shot = CATALOG_ONLY
    where = ondemand.overlay_paths(paths, shot)
    write_features(where.features_file(shot), shot, {"ip": _array()}, {})
    store = _write(where.spectrogram_file("detachment", shot), "store")
    assert store.is_file()
    for name, value in (("LABELER_ROOT", paths.root), ("LABELER_CORPUS", paths.corpus),
                        ("LABELER_LABEL_TABLES", paths.label_tables),
                        ("LABELER_CONFIG_DIR", paths.config_dir)):
        monkeypatch.setenv(name, str(value))
    cli = Paths.from_env()  # what the features pipeline and the prebuild CLI use
    assert cli.read_cache is False
    for pipeline in (paths, cli):
        features = pipeline.features_file(shot)
        review = pipeline.spectrogram_file("detachment", shot)
        assert features == paths.features / f"{shot}_features.h5"
        assert review == paths.spectrograms / "detachment" / f"{shot}.h5"
        assert not ondemand.in_cache(paths, features)
        assert not ondemand.in_cache(paths, review)
    # The server's own Paths (create_app) read the cache; a protected build does not.
    assert present(replace(paths, read_cache=True).features_file(shot)) == {"ip"}
    assert ondemand.in_cache(paths, create_app(paths=paths, token="t").state.paths
                             .spectrogram_file("detachment", shot))


def test_a_protected_build_does_not_read_an_older_copy_in_the_cache(paths, monkeypatch):
    seen = []

    def build(event, shot, where):
        seen.append(where.spectrogram_file(event, shot))
        return _one_trace(event, shot, where)

    where = ondemand.overlay_paths(paths, ROSTERED)  # an old on-demand copy
    _write(where.spectrogram_file("detachment", ROSTERED), "old")
    monkeypatch.setitem(review_build.BUILDERS, "detachment", build)
    ondemand.build("detachment", ROSTERED, replace(paths, read_cache=True))
    assert seen == [paths.spectrogram_file("detachment", ROSTERED)]
    assert not ondemand.in_cache(paths, seen[0])


# ---- the failure as the page gets it ----


@pytest.mark.parametrize(("error", "code", "short"), [
    (MissingFeature("pinj_total", "/scratch/x/180001_features.h5"),
     "missing_signal", "No beam power (pinj_total) for 180001"),
    (MissingFeature("ip", "/scratch/x", names=["ip", "betan", "pinj_total", "li"]),
     "missing_signal",
     "No plasma current (ip), normalized beta (betan), beam power (pinj_total) for 180001"),
    (MissingFeature("mystery", "/scratch/x"), "missing_signal", "No mystery for 180001"),
    (NoDataError("group 'co2' has no 'co2'"), "no_data", "No co2 data for shot 180001"),
    (NoDataError("nothing at all"), "no_data", "No data for shot 180001"),
    (FileNotFoundError("/scratch/x/180001.h5"), "no_data",
     "Input data for 180001 is missing"),
    (raw.FetchDisabledError("group 'ip' has no 'ip'"), "fetch_off",
     "No ip for 180001: fetching is off"),
    (raw.UpstreamError("fdp answered 503"), "build_failed",
     "fdp did not answer for 180001; open it again to retry"),
    (raw.UpstreamError("HTTP 401 unauthorized"), "fetch_off",
     "No data for 180001: fdp refused the request, log in again"),
    (raw.UpstreamError("x failed over fdp: fdp is busy: another fetch held it"),
     "fetch_off", "No data for 180001: fdp is busy, try again"),
    (RuntimeError("boom in /scratch/x/y.py"), "build_failed",
     "Could not build locked mode for 180001"),
])
@pytest.mark.usefixtures("fetching_on")
def test_a_failure_is_short_plain_and_coded(error, code, short):
    failure = ondemand.explain(error, "locked_mode", 180001)
    assert failure.code == code and failure.short == short
    assert failure.code in ondemand.CODES
    assert "/" not in failure.short and "Error" not in failure.short
    assert type(error).__name__ in failure.detail  # the full text keeps the class
    assert failure.body() == {"error": short, "detail": str(failure), "code": code}


def test_a_failure_after_a_refused_fetch_names_why_in_a_few_words():
    def failed(refusal):
        error = MissingFeature("pinj_total", "/x")
        error.ondemand = ondemand.FetchReport(refused=refusal)
        return ondemand.explain(error, "locked_mode", 180001)

    expired = failed("the fdp token has expired: run 'fdp login', then try again")
    assert (expired.code, expired.short) == (
        "fetch_off", "No beam power (pinj_total) for 180001: fdp token expired")
    wrapper = failed("this server was not started under the fdp wrapper, so no")
    assert wrapper.short.endswith("server not under fdp")
    assert "On-demand fetch: fetched nothing" in wrapper.detail
    assert "fetching refused" in wrapper.detail


def test_a_note_for_the_activity_bar_is_at_most_forty_characters():
    assert ondemand.note("queued", [], 0) == "Queued, next"
    assert ondemand.note("queued", [], 3) == "Queued, 3 ahead"
    assert ondemand.note("building", []) == "Building the review rows"
    assert ondemand.note("fetching", []) == "Fetching data"
    assert ondemand.note("fetching", ["pinj_total"]) == "Fetching pinj_total"
    many = ondemand.note("fetching", ["pinj_total", "tinj_total", "betan", "ip"])
    assert len(many) == ondemand.NOTE_MAX and many.endswith("…")


# ---- the 202 body ----


@pytest.fixture
def held(monkeypatch):
    """A detachment builder that waits for the test to let it finish."""
    gate, started = threading.Event(), []

    def slow(event, shot, paths):
        started.append(shot)
        assert gate.wait(30), "the test never let the build finish"
        return _one_trace(event, shot, paths)

    monkeypatch.setitem(review_build.BUILDERS, "detachment", slow)
    yield gate, started
    gate.set()


def test_the_202_body_carries_stage_fetching_and_a_short_note(client, held, monkeypatch):
    gate, started = held
    shots = (CATALOG_ONLY, OTHER_CATALOG, 190000)
    for shot in shots:
        assert _open(client, shot).status_code == 202
    _until(lambda: len(started) == 2)  # a pool of two: the third waits
    running, waiting = _open(client, CATALOG_ONLY).json(), _open(client, 190000).json()
    keys = {"building", "progress", "stage", "fetching", "note", "elapsed_s",
            "queue_position", "prebuilt"}
    assert set(running) == keys and set(waiting) == keys
    assert (running["stage"], running["fetching"], running["note"]) == (
        "building", [], "Building the review rows")
    assert (waiting["stage"], waiting["note"]) == ("queued", "Queued, next")
    assert waiting["fetching"] == [] and waiting["queue_position"] == 1

    monkeypatch.setitem(ondemand._fetching, CATALOG_ONLY, ["pinj_total"])
    fetching = _open(client, CATALOG_ONLY).json()
    assert (fetching["stage"], fetching["fetching"], fetching["note"]) == (
        "fetching", ["pinj_total"], "Fetching pinj_total")
    monkeypatch.setitem(ondemand._fetching, CATALOG_ONLY, [
        "pinj_total", "tinj_total", "betan", "ip", "bt"])
    assert len(_open(client, CATALOG_ONLY).json()["note"]) <= ondemand.NOTE_MAX
    monkeypatch.delitem(ondemand._fetching, CATALOG_ONLY)

    # A raw group `raw` is fetching names itself.
    record = {"done": 1, "total": 7, "stage": "fetching co2", "group": "co2"}
    monkeypatch.setattr(raw, "progress_for", lambda s: dict(record))
    body = _open(client, CATALOG_ONLY).json()
    assert (body["stage"], body["fetching"], body["note"]) == (
        "fetching", ["co2"], "Fetching co2")
    gate.set()


def test_the_api_version_tells_a_server_with_search_from_one_without(client):
    version = client.get("/api/version")
    assert version.status_code == 200 and version.json()["api"] == API_VERSION >= 12


# ---- the 502 body, over HTTP ----


def test_a_502_carries_a_short_error_a_detail_and_a_code(client, app, monkeypatch):
    def broken(event, shot, paths):
        raise NoDataError(f"no features for {shot}")

    monkeypatch.setitem(review_build.BUILDERS, "detachment", broken)
    assert _open(client, ROSTERED).status_code == 202
    with pytest.raises(NoDataError):
        app.state.builds.running[("detachment", ROSTERED)].result(timeout=30)
    response = _open(client, ROSTERED)
    assert response.status_code == 502
    assert response.json() == {
        "error": f"No data for shot {ROSTERED}",
        "detail": f"NoDataError: no features for {ROSTERED}",
        "code": "no_data",
    }


# ---- the eviction, over HTTP ----


def test_a_second_unprotected_shot_evicts_the_first_when_the_cache_is_full(
    client, app, paths, monkeypatch,
):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", _one_trace)
    monkeypatch.setenv(ondemand.CAP_ENV, "0.0000001")  # 100 bytes: one store is over it
    monkeypatch.setattr(ondemand, "GRACE_S", 0.0)
    assert _built_and_served(client, app, CATALOG_ONLY).status_code == 200
    first = ondemand.shot_dir(paths, CATALOG_ONLY)
    assert first.is_dir()  # the build keeps its own shot though the cache is over its cap
    assert _built_and_served(client, app, OTHER_CATALOG).status_code == 200
    assert not first.exists() and ondemand.shot_dir(paths, OTHER_CATALOG).is_dir()
    assert [shot for shot, _, _ in ondemand.cached(paths)] == [OTHER_CATALOG]
    # The evicted shot is simply built again; its labels were never in the cache.
    assert _open(client, CATALOG_ONLY).status_code == 202


# ---- the registry as shipped, the hidden lists ----


SHIPPED = Path(__file__).resolve().parents[2] / "configs/labeler/shotlists.yaml"


def test_the_shipped_registry_offers_all_shots_everywhere_and_hides_the_batches():
    registry = shotlists.parse(yaml.safe_load(SHIPPED.read_text()))
    catalog = registry.lists["catalog"]
    assert catalog["title"] == "All shots" and catalog["kind"] == "dataset"
    assert "catalog/population.csv" in catalog["rule"]
    assert registry.events, "events are named"
    for event, lists in registry.events.items():
        assert lists.recommended[-1] == "catalog", event  # after the event's own lists
    batches = [name for name in registry.lists if name.startswith("silver_batch_")]
    assert len(batches) == 5
    assert all(registry.lists[name]["hidden"] is True for name in batches)
    assert not registry.lists.get("silver", {}).get("hidden")


def test_a_hidden_flag_must_be_a_bool():
    with pytest.raises(shotlists.ShotListError, match="hidden must be true or false"):
        shotlists.parse({"lists": {"a": {"title": "A", "hidden": "yes"}}})


def test_hidden_lists_are_left_out_of_the_dropdown_and_the_overview(client, paths):
    offered = client.get("/api/shotlists?event=detachment").json()
    names = [item["name"] for item in offered]
    assert "silver_batch_1" not in names
    assert names[:3] == ["gold", "silver", "catalog"]  # the event's own, catalog last
    assert [item["name"] for item in offered if item["recommended"]] == [
        "gold", "silver", "catalog"]
    assert shotlists.entry(paths, "silver_batch_1").hidden  # its file and name stay
    assert "silver_batch_1" in shotlists.names(paths)
    overview = client.get("/api/overview?list=silver").json()
    assert overview["children"] == []
    asked = client.get("/api/queue?event=detachment&list=silver_batch_1").json()
    assert asked["list"] == "catalog"  # the default
    assert "not offered any more" in asked["note"]


def test_an_event_the_registry_does_not_name_recommends_all_shots_last(client):
    offered = client.get("/api/shotlists?event=locked_mode").json()
    assert [item["name"] for item in offered if item["recommended"]] == [
        "silver", "gold", "catalog"]
    assert client.get("/api/queue?event=detachment&list=catalog").json()["list"] == "catalog"


def test_a_list_of_on_demand_shots_does_not_count_as_prebuilt(paths):
    from labeler.events.ui import overlay_routes

    assert shotlists.on_demand(paths) == frozenset({"catalog"})
    assert overlay_routes.prebuilt(paths, OTHER_CATALOG) is False
    assert overlay_routes.prebuilt(paths, SILVER) is True


def test_any_list_the_registry_flags_opens_by_the_on_demand_build(client, paths):
    """A shot of a flagged list off the catalog (n1_seo's) builds into the temporary
    cache, and the status line does not call it prebuilt; a shot silver also holds
    still is."""
    from labeler.events.ui import overlay_routes

    far = 160001
    _write(paths.label_tables / "shotlists/far.csv", f"shot\n{far}\n{SILVER}\n")
    _write(paths.label_tables / "shotlists.yaml", REGISTRY.replace(
        "  held:", '  far: {title: "Far", kind: dataset, on_demand: true}\n  held:'))
    assert shotlists.on_demand(paths) == frozenset({"catalog", "far"})
    assert shotlists.holding(paths, far) == ["far"] and shotlists.in_any(paths, far)
    assert not ondemand.protected(paths, "detachment", far), "the cache, not the data root"
    assert ondemand.overlay_paths(paths, far).spectrogram_file(
        "detachment", far).is_relative_to(paths.ondemand)
    assert overlay_routes.prebuilt(paths, far) is False
    assert overlay_routes.prebuilt(paths, SILVER) is True, "silver holds it too"
    assert [item["name"] for item in client.get(
        "/api/shotlists?event=detachment").json()].count("far") == 1
    queue = client.get("/api/queue?event=detachment&list=far").json()
    assert queue["list"] == "far" and [row["shot"] for row in queue["shots"]] == [far, SILVER]
    # Dropping the flag makes the same shot count as prebuilt.
    _write(paths.label_tables / "shotlists.yaml", REGISTRY.replace(
        "  held:", '  far: {title: "Far", kind: dataset}\n  held:'))
    assert overlay_routes.prebuilt(paths, far) is True


# ---- the search ----


def _search(client, query, event="detachment", **extra):
    params = {"q": query, **({"event": event} if event else {}), **extra}
    return client.get("/api/shots/search", params=params)


def test_search_finds_shots_by_prefix_with_the_events_own_first(client, paths):
    _built(paths, event="detachment", shot=SILVER, current=True)
    found = _search(client, "17").json()["shots"]
    assert [item["shot"] for item in found] == [ROSTERED, SILVER, GOLD_ONLY, 170817]
    by = {item["shot"]: item for item in found}
    assert by[ROSTERED]["lists"][0] == "roster"
    assert set(by[ROSTERED]["lists"]) == {"roster", "gold", "catalog"}
    assert set(by[SILVER]["lists"]) == {"silver", "catalog"}  # not the hidden batch
    assert by[GOLD_ONLY]["lists"] == ["gold"] and by[170817]["lists"] == ["catalog"]
    assert [item["cached"] for item in found] == [False, True, False, False]
    assert set(found[0]) == {"shot", "lists", "cached"}
    # Without an event nothing is the roster's, and `cached` means any store.
    plain = _search(client, "170815", event="").json()["shots"]
    assert [item["shot"] for item in plain] == [ROSTERED] and plain[0]["cached"] is False
    assert sorted(plain[0]["lists"]) == ["catalog", "gold"]  # no `roster` without an event
    assert _search(client, "170816", event="").json()["shots"][0]["cached"] is True
    assert _search(client, "170816", event="locked_mode").json()["shots"][0][
        "cached"] is False  # another event's store is not this one's


def test_search_orders_by_number_after_the_first_group_and_honours_the_limit(client):
    found = _search(client, "1").json()["shots"]
    numbers = [item["shot"] for item in found]
    assert numbers[:3] == [ROSTERED, SILVER, GOLD_ONLY]
    assert numbers[3:] == sorted(numbers[3:]) and BLIND not in numbers
    assert HELD in numbers  # every list's shots, not the catalog's alone
    assert len(_search(client, "1", limit=2).json()["shots"]) == 2
    assert _search(client, "1", limit=100).status_code == 200
    assert _search(client, "1", limit=0).status_code == 422
    assert _search(client, "1", limit=101).status_code == 422


def test_search_returns_a_typed_number_the_catalog_lacks(client):
    assert _search(client, str(TYPED)).json() == {
        "shots": [{"shot": TYPED, "lists": [], "cached": False}]}
    assert _search(client, "12345").json() == {"shots": []}  # a prefix is not a shot
    assert _search(client, str(BLIND)).json() == {"shots": []}  # never a blind shot
    assert _search(client, "1750").json() == {"shots": []}
    exact = _search(client, str(ROSTERED)).json()["shots"]
    assert [item["shot"] for item in exact] == [ROSTERED]  # in the catalog, once


@pytest.mark.parametrize("query", ["", "abc", "1234567", "12 3", "1\n", "-12", "17.5",
                                   "１７"])
def test_search_takes_one_to_six_digits_and_nothing_else(client, query):
    response = _search(client, query)
    assert response.status_code == 400
    assert response.json() == {"error": "q must be one to six digits"}


def test_search_names_the_event_it_does_not_know(client):
    assert _search(client, "17", event="no_such_event").status_code == 404
    assert _search(client, "17", event="..").status_code in (404, 422)


def test_search_needs_the_token(app):
    anonymous = TestClient(app)
    assert anonymous.get("/api/shots/search?q=17").status_code in (401, 403)


def test_search_is_a_function_of_the_tables_alone(paths):
    body = shotlist_routes.search(paths, "180", roster=(), limit=20)
    assert [item["shot"] for item in body["shots"]] == [CATALOG_ONLY, OTHER_CATALOG]
