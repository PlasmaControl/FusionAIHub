"""The `events` feature source: one waveform of 46 window features.

Three contracts are asserted here rather than left to be discovered: the
namespace's `SOURCES` grew by an append (indices into it are load-bearing -
`run.features_for_shot` and `validate` both walk it in order), the resolver
has the same shape as `resolve_archive` and `resolve_corpus`, and
`run._resolve_one_source` actually reaches it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from labelmaker import run
from labelmaker.config import Paths
from labelmaker.events import windows
from labelmaker.features import namespace as ns
from labelmaker.features import resolve_events

from .test_events_windows import (
    SHOT,
    _block,
    _lit,
    _t_grid,
    _write_events,
    _write_masks,
)

NAME = "phenomenon_window_features"


def _paths(tmp_path) -> Paths:
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    paths.mkdirs()
    return paths


def _shot_files(tmp_path, *, t1: float = 1.0):
    paths = _paths(tmp_path)
    t = _t_grid(0.0, t1)
    _write_masks(paths, [
        _block("mhr", 0, "wide", t, coh=_lit(t, (0, 8))),
        _block("mhr", 0, "zoom", t, coh=_lit(t, (0, 8))),
    ])
    _write_events(paths, [])
    return paths


# ---------------------------------------------------------- namespace edits

def test_the_events_source_is_appended_and_nothing_is_reordered():
    assert ns.SOURCES == ("archive", "corpus", "fdp", "events")
    assert ns.SOURCES[:3] == ("archive", "corpus", "fdp")
    assert ns.SOURCES[-1] == "events"


def test_the_events_source_declares_its_sampling_convention():
    assert ns.SAMPLING_BY_SOURCE["events"] == "nearest"


def test_the_window_feature_spec_is_registered():
    spec = ns.by_name(NAME)
    assert spec.kind == "waveform"
    assert spec.sources == ("events",)
    assert spec.locator_for("events")
    assert spec.step == windows.STRIDE_S
    # The shape metadata a waveform spec has to carry: 46 channels, named -
    # in order - by `events.windows.FEATURE_NAMES`, which is the frozen
    # list a trained GBDT's feature names are checked against.
    assert "46" in spec.notes
    assert "FEATURE_NAMES" in spec.notes
    assert {f.name for f in ns.by_source("events")} == {NAME}


def test_the_spec_notes_admit_the_indirect_nbi_gate():
    """The spec's circularity note has to match the module's.

    "DIAGNOSTICS ONLY" is the rule and it holds for every direct read, but
    `lh_recent` descends from `heuristics.lh_transitions`, which gates on
    NBI power. Two documents asserting the opposite is how an audit gets
    closed on a false premise.
    """
    notes = ns.by_name(NAME).notes
    assert "lh_recent" in notes
    assert "pinj" in notes


def test_the_new_feature_does_not_disturb_the_existing_sources():
    for source in ("archive", "corpus", "fdp"):
        assert NAME not in {f.name for f in ns.by_source(source)}


# ------------------------------------------------------------- the resolver

def test_available_needs_both_files(tmp_path):
    paths = _paths(tmp_path)
    assert not resolve_events.available(paths, SHOT)
    t = _t_grid(0.0, 0.5)
    _write_masks(paths, [_block("mhr", 0, "wide", t, coh=_lit(t, (0, 8)))])
    assert not resolve_events.available(paths, SHOT)     # events still absent
    _write_events(paths, [])
    assert resolve_events.available(paths, SHOT)


def test_the_resolver_serves_the_46_at_the_window_centres(tmp_path):
    paths = _shot_files(tmp_path)
    arrays, missing = resolve_events.resolve(SHOT, [NAME], paths=paths)
    assert missing == {}
    arr = arrays[NAME]
    centres, x, _ = windows.shot_window_features(SHOT, paths)
    assert arr.y.shape == (46, centres.size)
    np.testing.assert_allclose(arr.x, centres)
    np.testing.assert_allclose(arr.y, x, rtol=1e-6)
    assert arr.attrs["resolver"] == "events"
    assert arr.attrs["n_channels"] == "46"
    assert arr.attrs["window_s"] == str(windows.WINDOW_S)
    assert arr.attrs["stride_s"] == str(windows.STRIDE_S)


def test_the_timebase_is_seconds_like_every_other_feature(tmp_path):
    """Seconds, not milliseconds.

    `FeatureArray` is documented as "seconds on `x`", `labels/store.py`
    writes `xdata` in seconds, and `models.base.InputSpec.build` compares a
    waveform's own axis against `GRID_S`, which is seconds. A window centre
    stored in ms would be a silent factor of 1,000 at every one of those
    seams, so the ms convention the brief asked for is deliberately not
    used here; see the task report.
    """
    paths = _shot_files(tmp_path)
    arrays, _ = resolve_events.resolve(SHOT, [NAME], paths=paths)
    arr = arrays[NAME]
    assert arr.x[0] == pytest.approx(windows.WINDOW_S / 2)
    assert arr.x[-1] < 1.0
    np.testing.assert_allclose(np.diff(arr.x)[:3], windows.STRIDE_S, atol=1e-9)


def test_a_window_no_diagnostic_covers_comes_back_as_nan(tmp_path):
    """The resolver's way of saying "nobody looked here".

    A resolver has no access to `BuiltInputs.valid`; what it can do is what
    `resolve_corpus` does for an all-NaN channel - hand back NaN rather than
    a fabricated zero - and leave the narrowing to the model, the way the AE
    adapter narrows `built.valid` from its own frame counts.
    """
    paths = _paths(tmp_path)
    t_short = _t_grid(0.0, 0.5)
    t_sliver = _t_grid(1.30, 1.34)
    _write_masks(paths, [
        _block("mhr", 0, "wide", t_short, coh=_lit(t_short, (0, 8))),
        _block("ece", 0, "wide", t_sliver, coh=_lit(t_sliver, (0, 8))),
    ])
    _write_events(paths, [])
    _, _, valid = windows.shot_window_features(SHOT, paths)
    assert not valid.all()
    arrays, _ = resolve_events.resolve(SHOT, [NAME], paths=paths)
    y = arrays[NAME].y
    assert np.isfinite(y[:, valid]).all()
    assert np.isnan(y[:, ~valid]).all()
    assert arrays[NAME].attrs["n_valid_windows"] == str(int(valid.sum()))


def test_missing_files_are_a_per_shot_miss_not_a_raise(tmp_path):
    paths = _paths(tmp_path)
    arrays, missing = resolve_events.resolve(SHOT, [NAME], paths=paths)
    assert arrays == {}
    assert missing == {NAME: "FileNotFoundError"}


def test_a_shot_with_no_coverage_is_recorded_as_a_miss(tmp_path):
    """A masks file whose blocks span no time at all yields no windows."""
    paths = _paths(tmp_path)
    t = _t_grid(0.0, 0.0)
    _write_masks(paths, [_block("mhr", 0, "wide", t, coh=_lit(t, (0, 8)))])
    _write_events(paths, [])
    arrays, missing = resolve_events.resolve(SHOT, [NAME], paths=paths)
    assert arrays == {}
    assert list(missing) == [NAME]
    assert "NoWindows" in missing[NAME]


def test_asking_this_resolver_for_someone_elses_feature_is_a_programming_error(
    tmp_path,
):
    paths = _shot_files(tmp_path)
    with pytest.raises(KeyError):
        resolve_events.resolve(SHOT, ["ip"], paths=paths)


def test_the_served_array_round_trips_through_the_feature_store(tmp_path):
    from labelmaker.features.store import read_feature, write_features

    paths = _shot_files(tmp_path)
    arrays, missing = resolve_events.resolve(SHOT, [NAME], paths=paths)
    write_features(paths.features_file(SHOT), SHOT, arrays, missing)
    back = read_feature(paths.features_file(SHOT), NAME)
    assert back.y.shape == arrays[NAME].y.shape
    assert back.y.dtype == np.float32          # a waveform stays float32
    np.testing.assert_allclose(back.x, arrays[NAME].x)


# ----------------------------------------------------------- run's dispatch

def test_run_dispatches_the_events_source_to_this_resolver(tmp_path, monkeypatch):
    seen = {}

    def fake(shot, names, *, paths):
        seen["args"] = (shot, list(names), paths)
        return {"served": "yes"}, {}

    monkeypatch.setattr(resolve_events, "resolve", fake)
    paths = _paths(tmp_path)
    ctx = SimpleNamespace(paths=paths, archive_files=())
    got, missed = run._resolve_one_source("events", SHOT, [NAME], ctx)
    assert got == {"served": "yes"} and missed == {}
    assert seen["args"] == (SHOT, [NAME], paths)


def test_the_other_sources_still_dispatch_where_they_did(tmp_path, monkeypatch):
    """The events branch must not shadow the fdp fall-through."""
    from labelmaker.features import resolve_fdp

    monkeypatch.setattr(
        resolve_fdp, "resolve", lambda shot, names: ({"fdp": shot}, {})
    )
    ctx = SimpleNamespace(paths=_paths(tmp_path), archive_files=())
    got, _ = run._resolve_one_source("fdp", 12345, ["ip"], ctx)
    assert got == {"fdp": 12345}


# --------------------------------------------------------------- the shape

def test_the_resolver_has_the_same_shape_as_the_other_two(tmp_path):
    """`(arrays, missing)`, keyed by canonical name, values `FeatureArray`."""
    from labelmaker.features.store import FeatureArray

    paths = _shot_files(tmp_path)
    arrays, missing = resolve_events.resolve(SHOT, [NAME], paths=paths)
    assert isinstance(arrays, dict) and isinstance(missing, dict)
    assert set(arrays) == {NAME}
    assert isinstance(arrays[NAME], FeatureArray)
    assert arrays[NAME].y.shape[0] == len(windows.FEATURE_NAMES)


def test_two_served_names_do_not_share_one_array(tmp_path, monkeypatch):
    """No aliasing across served names.

    The source serves one feature today, so this monkeypatches the
    namespace lookup to ask for two. A caller that narrowed one name's
    columns in place would otherwise silently narrow the other's.
    """
    paths = _shot_files(tmp_path)
    spec = ns.by_name(NAME)
    monkeypatch.setattr(resolve_events.ns, "by_name", lambda name: spec)
    arrays, missing = resolve_events.resolve(SHOT, [NAME, "another"],
                                             paths=paths)
    assert missing == {}
    first, second = arrays[NAME], arrays["another"]
    assert first.y is not second.y
    assert first.x is not second.x
    first.y[0, 0] = 12345.0
    first.x[0] = 999.0
    assert second.y[0, 0] != 12345.0
    assert second.x[0] != 999.0
