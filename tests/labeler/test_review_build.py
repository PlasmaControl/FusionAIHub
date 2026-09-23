"""Building the review store: provenance, the command line, failures."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from labeler.config import Paths
from labeler.events.review import build as review_build
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
