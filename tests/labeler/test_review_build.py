"""Building the review store: provenance, the command line, failures."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import h5py
import numpy as np

from labeler.config import Paths
from labeler.events.review import build as review_build
from labeler.events.review import rows
from labeler.events.review.rows import Grid, TraceRow
from labeler.events.verify import NoDataError


def fake(event, shot, paths):
    if shot == 2:
        raise NoDataError("no co2")
    values = np.zeros((2, 1, 10), dtype=np.float32)
    return Grid(0.0, 1.0, 10), [TraceRow("p0", "Trace", values)], {"params": {"k": 1}}


def test_build_writes_the_rows_and_their_provenance(tmp_path, monkeypatch):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", fake)
    path = review_build.build("detachment", 1, Paths(root=tmp_path))
    assert path == tmp_path / "spectrograms" / "detachment" / "1.h5"
    with h5py.File(path, "r") as f:
        assert f.attrs["event"] == "detachment" and f.attrs["shot"] == 1
        assert f.attrs["builder"] == "test_review_build"
        assert json.loads(f.attrs["params"]) == {"k": 1}
        assert f.attrs["git_sha"] and f.attrs["made_at"]


class _Fixed(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, 29, 12, 0, tzinfo=tz)


def test_out_none_is_todays_build_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", fake)
    monkeypatch.setattr(review_build, "git_sha", lambda: "0123abc")
    monkeypatch.setattr(review_build, "datetime", _Fixed)
    path = review_build.build("detachment", 1, Paths(root=tmp_path / "a"))
    assert path == tmp_path / "a" / "spectrograms" / "detachment" / "1.h5"
    # Today's build, as `build` wrote it before `out` and `force`.
    today = tmp_path / "today.h5"
    grid, built, info = fake("detachment", 1, None)
    rows.write(
        today,
        grid,
        built,
        event="detachment",
        shot=1,
        builder="test_review_build",
        **info,
        made_at="2026-09-29T12:00:00+00:00",
        git_sha="0123abc",
    )
    assert path.read_bytes() == today.read_bytes()
    # The same file wherever `out` puts it.
    elsewhere = review_build.build(
        "detachment", 1, Paths(root=tmp_path / "b"), out=tmp_path / "out"
    )
    assert elsewhere.read_bytes() == today.read_bytes()


def test_an_existing_file_is_rebuilt_only_when_forced(tmp_path, monkeypatch):
    calls = []

    def counted(event, shot, paths):
        calls.append(shot)
        return fake(event, shot, paths)

    monkeypatch.setitem(review_build.BUILDERS, "detachment", counted)
    paths = Paths(root=tmp_path)
    review_build.build("detachment", 1, paths)
    review_build.build("detachment", 1, paths)
    assert calls == [1]
    review_build.build("detachment", 1, paths, force=True)
    assert calls == [1, 1]


def test_the_command_builds_what_is_missing_and_reports_failures(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", fake)
    monkeypatch.setattr(review_build, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    assert review_build.main(["--event", "detachment", "--shots", "1", "2"]) == 1
    out = capsys.readouterr().out
    assert "detachment: 2 of 2 shots to build" in out
    assert "NoDataError: no co2" in out
    assert review_build.main(["--event", "detachment", "--shots", "1"]) == 0
    assert "detachment: 0 of 1 shots to build" in capsys.readouterr().out
