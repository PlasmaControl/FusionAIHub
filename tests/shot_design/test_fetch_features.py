"""`scripts/labeler/fetch_features.py`: the login-node fetch that fills shot_design's feature gaps.

The script lives under `scripts/labeler/` but it exists for `shot_design`, which is why its tests
are here. Two invariants carry the weight:

* **it never creates a feature file.** `select.preferred_shots` and the corpus census both read
  the mere EXISTENCE of `<shot>_features.h5` as "this shot has features", so a run that created
  one holding nothing but the features it just fetched would silently promote a shot labeler
  has never featured. The `path.exists()` guard is on every path, the successful one included.
* **it merges.** `store.write_features(..., merge=True)`; every group already in the file stays.

Loaded by path rather than imported: `scripts/` is not a package, and making it one to reach one
module would put it on every environment's import path. Nothing here imports labeler -- the
resolver and the store are reached through `_resolver()`/`_store()`, which exist so the worker
imports toksearch after the fork and which a test can substitute.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "labeler" / "fetch_features.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("fetch_features_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Recorder:
    """A stand-in for `labeler.features.store`, recording every write it is asked for."""

    def __init__(self) -> None:
        self.writes: list[tuple[Path, int, list[str], dict]] = []

    def write_features(self, path, shot, arrays, missing, *, merge=True):
        self.writes.append((Path(path), int(shot), sorted(arrays), dict(missing)))
        Path(path).write_bytes(b"")  # a real writer creates the file; that is the point


def _array(n: int = 8):
    return SimpleNamespace(x=np.arange(n, dtype=float), y=np.ones((1, n), dtype=float), attrs={})


def _install(script, tmp_path, monkeypatch, *, resolve, features=("ip",)):
    store = _Recorder()
    monkeypatch.setattr(script, "_resolver", lambda: SimpleNamespace(resolve=resolve))
    monkeypatch.setattr(script, "_store", lambda: store)
    script._init(str(tmp_path), list(features), 0)
    return store


def _touch(tmp_path, shot: int) -> Path:
    p = tmp_path / f"{shot}_features.h5"
    p.write_bytes(b"")
    return p


# ------------------------------------------------------------------ the feature list


def test_the_default_feature_list_is_the_one_the_old_invocation_asked_for(script):
    """`fetch_ip_features.py --shot-file …` became `fetch_features.py --shot-file …`. The name
    changed; what a bare invocation fetches did not."""
    args = script.build_parser().parse_args(["--shot-file", "x.txt"])
    assert args.features == ["ip"]


def test_several_features_are_taken_in_the_order_they_were_asked_for(script):
    args = script.build_parser().parse_args(["--shot-file", "x.txt", "--features", "bt", "betan"])
    assert args.features == ["bt", "betan"]


def test_a_repeated_feature_is_asked_for_once(script):
    assert script.canonical(["bt", "ip", "bt"]) == ["bt", "ip"]


def test_a_name_that_is_not_a_canonical_feature_is_refused_before_any_fetch(
    script, tmp_path, capsys
):
    """`resolve_fdp.resolve` raises KeyError on an unknown name -- after the pool has been forked
    and per shot. A typo in a flag is a usage error, and it belongs on the first line of output."""
    shots = tmp_path / "shots.txt"
    shots.write_text("190001\n", encoding="utf-8")
    rc = script.main(
        ["--shot-file", str(shots), "--features-dir", str(tmp_path), "--features", "bt", "nope"]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "nope" in err and "bt" not in err.split("nope")[0].split("\n")[-1]


# --------------------------------------------------------------- one shot, one merge


def test_a_successful_fetch_never_creates_a_feature_file(script, tmp_path, monkeypatch):
    """The bug the I5 re-review found in `fetch_ip_features.py`: the miss path guarded on
    `path.exists()` and the SUCCESS path did not."""
    store = _install(script, tmp_path, monkeypatch, resolve=lambda s, n, retries=0: (
        {name: _array() for name in n}, {}
    ))
    row = script._fetch(190001)
    assert store.writes == []
    assert not (tmp_path / "190001_features.h5").exists()
    assert row["status"] == {"ip": "failed"}
    assert "no features file" in row["causes"]["ip"]


def test_a_successful_fetch_merges_into_a_file_that_is_there(script, tmp_path, monkeypatch):
    store = _install(script, tmp_path, monkeypatch, features=("bt", "betan"),
                     resolve=lambda s, n, retries=0: ({name: _array() for name in n}, {}))
    _touch(tmp_path, 190001)
    row = script._fetch(190001)
    assert row["status"] == {"bt": "fetched", "betan": "fetched"}
    assert row["n"] == {"bt": 8, "betan": 8}
    assert [(w[1], w[2], w[3]) for w in store.writes] == [(190001, ["betan", "bt"], {})]


def test_only_the_features_the_file_lacks_are_asked_of_the_resolver(
    script, tmp_path, monkeypatch
):
    """A feature already in the file costs no network call. The point of the per-feature table:
    `present` and `fetched` are different facts and a rerun over the same list is nearly free."""
    asked: list[list[str]] = []

    def resolve(shot, names, retries=0):
        asked.append(list(names))
        return {name: _array() for name in names}, {}

    _install(script, tmp_path, monkeypatch, features=("bt", "betan"), resolve=resolve)
    monkeypatch.setattr(script, "present_features", lambda path, names: {"bt"})
    _touch(tmp_path, 190001)
    row = script._fetch(190001)
    assert asked == [["betan"]]
    assert row["status"] == {"bt": "present", "betan": "fetched"}


def test_a_miss_is_recorded_in_the_file_and_costs_only_that_feature(
    script, tmp_path, monkeypatch
):
    store = _install(script, tmp_path, monkeypatch, features=("bt", "qmin"),
                     resolve=lambda s, n, retries=0: ({"bt": _array()}, {"qmin": "TreeNNF"}))
    _touch(tmp_path, 190001)
    row = script._fetch(190001)
    assert row["status"] == {"bt": "fetched", "qmin": "failed"}
    assert row["causes"] == {"qmin": "TreeNNF"}
    # One write, carrying the array AND the cause: the file records what was tried.
    assert [(w[2], w[3]) for w in store.writes] == [(["bt"], {"qmin": "TreeNNF"})]


def test_a_one_sample_feature_is_reported_as_the_miss_the_store_will_make_of_it(
    script, tmp_path, monkeypatch
):
    """`store.write_features` demotes a group of fewer than two samples into `missing` (it is the
    corpus' "signal absent" sentinel). A table that called it `fetched` would disagree with the
    file it just wrote."""
    _install(script, tmp_path, monkeypatch, features=("bt",),
             resolve=lambda s, n, retries=0: ({"bt": _array(1)}, {}))
    _touch(tmp_path, 190001)
    row = script._fetch(190001)
    assert row["status"] == {"bt": "failed"} and "OneSample" in row["causes"]["bt"]


# ------------------------------------------------------------------- the run and its table


def test_the_table_counts_every_feature_over_every_shot_and_the_run_exits_1(
    script, tmp_path, monkeypatch, capsys
):
    """The exit code is the whole point of running it from a wrapper script: a run that could not
    get everything it was asked for must not look like one that did."""
    shots = tmp_path / "shots.txt"
    shots.write_text("190001\n190002\n", encoding="utf-8")
    for shot in (190001, 190002):
        _touch(tmp_path, shot)
    store = _Recorder()
    monkeypatch.setattr(script, "_store", lambda: store)
    monkeypatch.setattr(
        script,
        "_resolver",
        lambda: SimpleNamespace(
            resolve=lambda s, n, retries=0: (
                {name: _array() for name in n if name != "qmin"}, {"qmin": "TreeNNF"}
            )
        ),
    )
    rc = script.main(
        ["--shot-file", str(shots), "--features-dir", str(tmp_path),
         "--features", "bt", "qmin", "--workers", "1", "--retries", "0"]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "bt" in out and "qmin" in out
    lines = {line.split()[0]: line.split()[1:] for line in out.splitlines() if line[:1].isalpha()}
    assert lines["bt"][:3] == ["2", "0", "0"]      # fetched / present / failed
    assert lines["qmin"][:3] == ["0", "0", "2"]


def test_a_run_that_got_everything_exits_0(script, tmp_path, monkeypatch, capsys):
    shots = tmp_path / "shots.txt"
    shots.write_text("190001\n", encoding="utf-8")
    _touch(tmp_path, 190001)
    monkeypatch.setattr(script, "_store", lambda: _Recorder())
    monkeypatch.setattr(script, "_resolver", lambda: SimpleNamespace(
        resolve=lambda s, n, retries=0: ({name: _array() for name in n}, {})
    ))
    rc = script.main(
        ["--shot-file", str(shots), "--features-dir", str(tmp_path),
         "--features", "bt", "--workers", "1"]
    )
    assert rc == 0
    assert "1 fetched" in capsys.readouterr().out


def test_a_shot_with_no_feature_file_fails_the_run_and_creates_nothing(
    script, tmp_path, monkeypatch, capsys
):
    shots = tmp_path / "shots.txt"
    shots.write_text("190001\n", encoding="utf-8")
    monkeypatch.setattr(script, "_store", lambda: _Recorder())
    monkeypatch.setattr(script, "_resolver", lambda: SimpleNamespace(
        resolve=lambda s, n, retries=0: ({name: _array() for name in n}, {})
    ))
    rc = script.main(
        ["--shot-file", str(shots), "--features-dir", str(tmp_path), "--workers", "1"]
    )
    assert rc == 1
    assert list(tmp_path.glob("*_features.h5")) == []
    assert "no features file" in capsys.readouterr().out
