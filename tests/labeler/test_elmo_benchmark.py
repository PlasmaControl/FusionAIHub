"""The ELM-O detector of scripts/labeler/elmo_benchmark.py, on signals built to the rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"


@pytest.fixture(scope="module")
def elmo():
    spec = importlib.util.spec_from_file_location(
        "elmo_benchmark", SCRIPTS / "elmo_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["elmo_benchmark"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("elmo_benchmark", None)


def trace(spikes: dict[int, int], bes_high: tuple[int, int], size: int = 1000):
    """Rows 0-1 are the chords, 2-4 the filterscopes; each spike is a first difference of 1.

    The signals are exactly flat elsewhere, so a row's 0.997 quantile is 0 and a row
    is a candidate only where it spikes (no up-sampling noise to vote by chance).
    """
    diff = np.zeros((5, size))
    for row, at in spikes.items():
        diff[row, at] = 1.0
    bes = np.zeros(size)
    bes[bes_high[0] : bes_high[1]] = 3.0
    fs03 = np.zeros(size)
    fs03[bes_high[0] + 10] = 2.0
    return SimpleNamespace(diff=diff, bes=bes, fs03=fs03, half=3, gap=3)


def test_runs_bridge_and_dilate(elmo):
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0], dtype=bool)
    starts, stops = elmo.runs(mask)
    assert (starts.tolist(), stops.tolist()) == ([1, 5, 7], [3, 6, 10])
    joined = elmo.bridge(starts, stops, 2)
    assert (joined[0].tolist(), joined[1].tolist()) == ([1], [10])
    apart = elmo.bridge(starts, stops, 1)
    assert (apart[0].tolist(), apart[1].tolist()) == ([1, 5], [3, 10])
    assert elmo.runs(np.zeros(4, dtype=bool))[0].size == 0
    dot = np.zeros(12, dtype=bool)
    dot[5] = True
    assert np.flatnonzero(elmo.dilate(dot, 2)).tolist() == [3, 4, 5, 6, 7]


def test_upsampling_keeps_a_slow_cosine_and_cuts_in_pieces(elmo):
    n, m = 64, 3
    slow = np.cos(np.pi * m * (np.arange(n) + 0.5) / n)
    fine = np.cos(np.pi * m * (np.arange(640) + 0.5) / 640)
    assert np.abs(elmo.lengthen(slow, 640) - fine).max() < 1e-9
    assert np.array_equal(elmo.upsample(slow, 640, chunk=1000), elmo.lengthen(slow, 640))
    long = elmo.upsample(np.arange(300.0), 2500, chunk=1000)
    assert long.shape == (3 * (2500 // 3),)


def test_bes_mean_doubles_the_second_half(elmo):
    y = np.ones((64, 5))
    y[:, 2] = np.nan
    mean = elmo.bes_mean(y)
    assert mean.tolist() == [1.5, 1.5, 0.0, 1.5, 1.5]
    assert np.isnan(y[:, 2]).all()


def test_an_elm_needs_both_signal_kinds_and_the_bes(elmo):
    full = trace({0: 500, 2: 502, 4: 498}, (490, 520))
    starts, stops = elmo.detect(full, eta=0.997, threshold=1.0)
    # the chords vote 497-503, the filterscopes (two of three) 499-501, the BES 490-519
    assert (starts.tolist(), stops.tolist()) == ([499], [502])
    assert elmo.peaks(full, starts, stops).tolist() == [500]
    one_filterscope = trace({0: 500, 2: 502}, (490, 520))
    assert elmo.detect(one_filterscope)[0].size == 0
    no_chord = trace({2: 500, 3: 500, 4: 500}, (490, 520))
    assert elmo.detect(no_chord)[0].size == 0
    dim = trace({1: 500, 3: 500, 4: 500}, (490, 520))
    dim.bes *= 0.2
    assert elmo.detect(dim)[0].size == 0
    assert elmo.detect(dim, use_bes=False)[0].size == 1


def test_a_window_is_scored_by_the_papers_rule(elmo):
    starts, stops = np.array([10, 50]), np.array([20, 60])
    assert elmo.count(starts, stops, 15, 30) == (1, 1, 0)
    assert elmo.count(starts, stops, 70, 80) == (0, 2, 1)
    assert elmo.count(starts[:0], stops[:0], 15, 30) == (0, 0, 1)


def test_the_second_file_keeps_only_marks_not_yet_taken(elmo):
    windows = pd.DataFrame(
        {
            "shot": [1, 1, 1, 2],
            "rank": [0, 1, 1, 1],
            "label_t0_ms": [10.0, 10.5, 30.0, 5.0],
            "label_t1_ms": [11.0, 11.5, 31.0, 6.0],
        }
    )
    kept = elmo.drop_duplicates(windows)
    assert kept[["shot", "label_t0_ms"]].values.tolist() == [
        [1, 10.0],
        [1, 30.0],
        [2, 5.0],
    ]


def test_a_record_that_covers_too_little_is_refused(elmo):
    t = np.arange(0.0, 10.0, 0.25)
    y = np.vstack([t, -t])
    assert elmo.source_slice(t, y, 2.0, 4.0).shape == (2, 8)
    assert elmo.source_slice(t, y, 9.95, 12.0) is None
    assert elmo.source_slice(t, y, 8.0, 12.0) is None


def test_absent_span_elms_are_counted_by_distance_to_a_present_span(elmo):
    spans = pd.DataFrame(
        {
            "kind": ["absent", "individual", "absent"],
            "t_start": [0, 100, 150],
            "t_end": [100, 150, 300],
        }
    )
    found = elmo.detections(np.array([99.5, 120.0, 160.0, 290.0]), spans)
    assert found == {
        "present": 1,
        "absent": 3,
        "other": 0,
        "absent_within_50ms": 2,
        "absent_within_500ms": 3,
    }


def test_a_shot_without_bes_is_told_from_one_with(elmo, tmp_path, monkeypatch):
    import h5py

    for shot, samples in ((1, 1), (2, 5), (4, 0)):
        with h5py.File(tmp_path / f"{shot}_processed.h5", "w") as f:
            if samples:
                f["bes/xdata"] = np.arange(samples, dtype=float)
    monkeypatch.setattr(elmo, "CORPUS", tmp_path)
    assert [elmo.bes_samples(s) for s in (1, 2, 3, 4)] == [1, 5, 0, 0]
