"""`scripts/labelmaker/fetch_ip_features.py`: the login-node fetch that fills rule (d)'s `ip`.

The script lives under `scripts/labelmaker/` but it exists for `ideate corpus select`, which is
why its tests are here: the invariant under test is ideate's -- `select.preferred_shots` and the
corpus census both read the mere EXISTENCE of `<shot>_features.h5` as "this shot has features",
so a run that creates a file holding nothing but `ip` would silently promote a shot that
labelmaker has never featured.

Loaded by path rather than imported: `scripts/` is not a package, and making it one to reach one
module would put it on every environment's import path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "labelmaker" / "fetch_ip_features.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("fetch_ip_features_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Recorder:
    """A stand-in for `labelmaker.features.store`, recording every write it is asked for."""

    def __init__(self) -> None:
        self.writes: list[tuple[Path, int, list[str], dict]] = []

    def write_features(self, path, shot, arrays, missing, *, merge=True):
        self.writes.append((Path(path), int(shot), sorted(arrays), dict(missing)))
        Path(path).write_bytes(b"")  # a real writer creates the file; that is the point


def _array(n: int = 8):
    return SimpleNamespace(x=np.arange(n, dtype=float), y=np.ones(n, dtype=float), attrs={})


def _install(script, tmp_path, monkeypatch, *, resolve):
    store = _Recorder()
    monkeypatch.setattr(script, "_resolver", lambda: SimpleNamespace(resolve=resolve))
    monkeypatch.setattr(script, "_store", lambda: store)
    monkeypatch.setattr(script, "_FEATURES_DIR", tmp_path)
    monkeypatch.setattr(script, "_RETRIES", 0)
    return store


def test_a_successful_fetch_never_creates_a_feature_file(script, tmp_path, monkeypatch):
    """The bug the I5 re-review found: the miss path guarded on `path.exists()` and the SUCCESS
    path did not, so a shot with no feature file at all came back with one holding only `ip`."""
    store = _install(script, tmp_path, monkeypatch, resolve=lambda s, n, retries=0: (
        {"ip": _array()}, {}
    ))
    row = script._fetch(190001)
    assert store.writes == []
    assert not (tmp_path / "190001_features.h5").exists()
    assert row["status"] == "failed" and "no features file" in row["cause"]


def test_a_successful_fetch_merges_into_a_file_that_is_there(script, tmp_path, monkeypatch):
    store = _install(script, tmp_path, monkeypatch, resolve=lambda s, n, retries=0: (
        {"ip": _array()}, {}
    ))
    (tmp_path / "190001_features.h5").write_bytes(b"")
    row = script._fetch(190001)
    assert row["status"] == "fetched" and row["n"] == 8
    assert [(w[1], w[2]) for w in store.writes] == [(190001, ["ip"])]
