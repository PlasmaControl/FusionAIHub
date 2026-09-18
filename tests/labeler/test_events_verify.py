"""Reading corpus signals for a review, and what a review writes."""

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from labeler.events.interval_tables import INTERVAL_COLUMNS, write_label_grid
from labeler.events.rosters import ROSTER_COLUMNS, read_roster, write_roster
from labeler.events.verify import (
    NoDataError,
    Panel,
    ReviewSession,
    corpus_signal,
    fdp_signal,
    label_panel,
    read_corrections,
    review,
    review_path,
    write_corrections,
)


def _corpus(tmp_path, shot, groups):
    """One corpus-shaped file: `<shot>_processed.h5`, xdata seconds, ydata (C, T)."""
    path = tmp_path / f"{shot}_processed.h5"
    with h5py.File(path, "w") as f:
        for name, (x, y) in groups.items():
            group = f.create_group(name)
            group.create_dataset("xdata", data=np.asarray(x, dtype="float32"))
            group.create_dataset("ydata", data=np.asarray(y, dtype="float32"))
    return path


def test_a_signal_comes_back_in_milliseconds(tmp_path):
    seconds = np.linspace(0.0, 2.0, 2001)
    _corpus(tmp_path, 185601, {"ece": (seconds, np.zeros((48, 2001)))})
    got = corpus_signal(185601, "ece", corpus=tmp_path)
    assert got.y.shape == (48, 2001)
    assert got.x[0] == pytest.approx(0.0)
    assert got.x[-1] == pytest.approx(2000.0)


def test_channels_select_rows(tmp_path):
    seconds = np.linspace(0.0, 1.0, 101)
    values = np.arange(48 * 101, dtype="float32").reshape(48, 101)
    _corpus(tmp_path, 185601, {"ece": (seconds, values)})
    got = corpus_signal(185601, "ece", channels=[20, 24, 28], corpus=tmp_path)
    assert got.y.shape == (3, 101)
    assert got.y[0] == pytest.approx(values[20])
    assert got.attrs["channels"] == "20,24,28"


def test_a_time_range_reads_only_the_windowed_slice_of_ydata(tmp_path, monkeypatch):
    """Bounds and shape alone can't tell a real h5py slice from a full load
    that NumPy then slices - both produce identical output. This spies on
    every read of the `ydata` dataset and asserts none of them cover more
    than the requested window, so a regression to load-then-slice is caught.
    """
    seconds = np.linspace(0.0, 6.0, 6001)
    _corpus(tmp_path, 185601, {"ece": (seconds, np.zeros((48, 6001)))})

    real_file = h5py.File
    ydata_reads = []

    class SpyDataset:
        """Wraps one dataset, recording every key it is read with."""

        def __init__(self, dataset):
            self._dataset = dataset

        def __getitem__(self, key):
            ydata_reads.append(key)
            return self._dataset[key]

        def __getattr__(self, name):
            return getattr(self._dataset, name)

    class SpyGroup:
        """Wraps one group, handing out a `SpyDataset` for `ydata` only."""

        def __init__(self, group):
            self._group = group

        def __getitem__(self, key):
            item = self._group[key]
            return SpyDataset(item) if key == "ydata" else item

        def __contains__(self, key):
            return key in self._group

        def __getattr__(self, name):
            return getattr(self._group, name)

    class SpyFile:
        """Wraps `h5py.File` so `corpus_signal`'s reads can be inspected."""

        def __init__(self, *args, **kwargs):
            self._file = real_file(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return self._file.__exit__(*exc_info)

        def __contains__(self, key):
            return key in self._file

        def __getitem__(self, key):
            return SpyGroup(self._file[key])

    monkeypatch.setattr("h5py.File", SpyFile)

    got = corpus_signal(185601, "ece", t_range=(2000.0, 2100.0), corpus=tmp_path)

    assert got.x[0] >= 2000.0
    assert got.x[-1] <= 2100.0
    assert got.y.shape[1] == got.x.shape[0] < 200

    assert ydata_reads, "corpus_signal never read the ydata dataset"
    window = got.y.shape[1]
    for _row, columns in ydata_reads:
        stop = columns.stop if columns.stop is not None else 6001
        start = columns.start if columns.start is not None else 0
        assert stop - start == window


def test_a_missing_file_names_the_corpus_span(tmp_path):
    with pytest.raises(NoDataError, match="170815"):
        corpus_signal(170815, "co2", corpus=tmp_path)


def test_the_absent_signal_sentinel_is_not_data(tmp_path):
    _corpus(tmp_path, 185601, {"co2": (np.zeros(1), np.zeros((4, 1)))})
    with pytest.raises(NoDataError, match="co2"):
        corpus_signal(185601, "co2", corpus=tmp_path)


def test_a_missing_group_is_named(tmp_path):
    _corpus(tmp_path, 185601, {"ece": (np.zeros(10), np.zeros((48, 10)))})
    with pytest.raises(NoDataError, match="mhr"):
        corpus_signal(185601, "mhr", corpus=tmp_path)


def _fake_fetch_mds(records, calls):
    """A stand-in for `_fetch_mds` returning canned records, call-counted.

    `records` maps expression -> the dict `_fetch_mds` would return; `calls`
    is a list this appends the expression to, so a test can assert exactly
    how many times the network path would have been taken.
    """

    def fake(expr, tree, shot, dims=()):
        calls.append(expr)
        return records[expr]

    return fake


def test_fdp_signal_returns_expressions_in_order(monkeypatch):
    n = 10
    ms = np.linspace(0.0, 20.0, n)
    records = {
        "a": {
            "data": np.arange(n, dtype="float32"),
            "dim0": ms,
            "units": {"data": "keV", "dim0": "ms"},
        },
        "b": {
            "data": np.arange(n, dtype="float32") + 100,
            "dim0": ms,
            "units": {"data": "keV", "dim0": "ms"},
        },
    }
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds", _fake_fetch_mds(records, [])
    )
    got = fdp_signal(178640, ["a", "b"])
    assert got.y.shape == (2, n)
    assert got.y.dtype == np.float32
    assert got.y[0] == pytest.approx(records["a"]["data"])
    assert got.y[1] == pytest.approx(records["b"]["data"])
    assert got.x[0] == pytest.approx(0.0)
    assert got.x[-1] == pytest.approx(20.0)


def test_fdp_signal_t_range_slices_to_the_window(monkeypatch):
    n = 100
    ms = np.linspace(0.0, 990.0, n)
    records = {
        "a": {
            "data": np.arange(n, dtype="float32"),
            "dim0": ms,
            "units": {"data": "keV", "dim0": "ms"},
        },
    }
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds", _fake_fetch_mds(records, [])
    )
    got = fdp_signal(178640, ["a"], t_range=(200.0, 300.0))
    assert got.x[0] >= 200.0
    assert got.x[-1] <= 300.0


def test_fdp_signal_rejects_a_non_millisecond_unit(monkeypatch):
    n = 10
    records = {
        "a": {
            "data": np.arange(n, dtype="float32"),
            "dim0": np.linspace(0.0, 0.02, n),
            "units": {"data": "keV", "dim0": "s"},
        },
    }
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds", _fake_fetch_mds(records, [])
    )
    with pytest.raises(NoDataError, match="'s'"):
        fdp_signal(178640, ["a"])


def test_fdp_signal_rejects_unequal_lengths(monkeypatch):
    records = {
        "a": {
            "data": np.arange(10, dtype="float32"),
            "dim0": np.linspace(0.0, 10.0, 10),
            "units": {"data": "keV", "dim0": "ms"},
        },
        "b": {
            "data": np.arange(20, dtype="float32"),
            "dim0": np.linspace(0.0, 10.0, 20),
            "units": {"data": "keV", "dim0": "ms"},
        },
    }
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds", _fake_fetch_mds(records, [])
    )
    with pytest.raises(NoDataError, match="10"):
        fdp_signal(178640, ["a", "b"])
    with pytest.raises(NoDataError, match="20"):
        fdp_signal(178640, ["a", "b"])


def test_fdp_signal_cache_is_not_refetched(tmp_path, monkeypatch):
    n = 10
    ms = np.linspace(0.0, 20.0, n)
    records = {
        "a": {
            "data": np.arange(n, dtype="float32"),
            "dim0": ms,
            "units": {"data": "keV", "dim0": "ms"},
        },
    }
    calls: list[str] = []
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds", _fake_fetch_mds(records, calls)
    )
    cache = tmp_path / "178640_ece.npz"

    first = fdp_signal(178640, ["a"], cache=cache)
    assert cache.is_file()
    assert len(calls) == 1

    second = fdp_signal(178640, ["a"], cache=cache)
    assert len(calls) == 1, "the second call must not touch the network"
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(first.x, second.x)


def test_fdp_signal_cache_holds_the_full_record_not_the_window(tmp_path, monkeypatch):
    n = 100
    ms = np.linspace(0.0, 990.0, n)
    records = {
        "a": {
            "data": np.arange(n, dtype="float32"),
            "dim0": ms,
            "units": {"data": "keV", "dim0": "ms"},
        },
    }
    calls: list[str] = []
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds", _fake_fetch_mds(records, calls)
    )
    cache = tmp_path / "178640_ece.npz"

    narrow = fdp_signal(178640, ["a"], t_range=(100.0, 200.0), cache=cache)
    assert len(calls) == 1
    assert narrow.x[0] >= 100.0
    assert narrow.x[-1] <= 200.0

    wide = fdp_signal(178640, ["a"], t_range=(0.0, 900.0), cache=cache)
    assert len(calls) == 1, "the wider window must be served from the cache"
    assert wide.x[0] < narrow.x[0]
    assert wide.x[-1] > narrow.x[-1]


def test_fdp_signal_wraps_a_fetch_failure(monkeypatch):
    def boom(expr, tree, shot, dims=()):
        raise RuntimeError("TREE-E-FOPENR")

    monkeypatch.setattr("labeler.features.resolve_fdp._fetch_mds", boom)
    with pytest.raises(NoDataError, match="fdp run"):
        fdp_signal(178640, ["a"])


def _roster(root, event):
    path = root / event / "shots.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_roster(
        pd.DataFrame(
            [[185601, "unverified", "false", "", "", ""]],
            columns=list(ROSTER_COLUMNS),
        ),
        path,
    )
    return path


def _button(session, description):
    """The control button with this description, out of the row of four."""
    return next(
        b
        for b in session.controls.children[0].children
        if b.description == description
    )


def test_corrections_round_trip_through_the_interval_schema(tmp_path):
    path = tmp_path / "review" / "185601.csv"
    frame = pd.DataFrame(
        [[185601, 1, 1200.0, 1450.0, ""], [185601, 0, 1450.0, 1600.0, ""]],
        columns=list(INTERVAL_COLUMNS),
    )
    write_corrections(frame, path)
    got = read_corrections(path)
    assert list(got.columns) == list(INTERVAL_COLUMNS)
    assert got.t_start.tolist() == [1200.0, 1450.0]
    assert got.category.tolist() == [1, 0]


def test_review_path_is_under_the_category(tmp_path):
    got = review_path("fishbone", 185601, root=tmp_path)
    assert got == tmp_path / "fishbone" / "review" / "185601.csv"


def test_marking_a_range_then_saving_writes_both_files(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.mark(1200.0, 1450.0, category=1)
    session.mark(1450.0, 1600.0, category=0)
    session.verify()
    session.save()

    corrections = read_corrections(review_path("fishbone", 185601, root=tmp_path))
    assert len(corrections) == 2
    assert corrections.shot.tolist() == [185601, 185601]

    roster = read_roster(tmp_path / "fishbone" / "shots.csv")
    assert roster.iloc[0].tier == "unverified"
    assert roster.iloc[0].reviewers == "alice"


def test_nothing_is_written_before_save(tmp_path):
    roster_file = _roster(tmp_path, "fishbone")
    before = roster_file.read_text()
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.mark(1200.0, 1450.0, category=1)
    session.verify()
    assert not review_path("fishbone", 185601, root=tmp_path).parent.exists()
    assert roster_file.read_text() == before


def test_a_backwards_range_is_refused(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    with pytest.raises(ValueError, match="t_end"):
        session.mark(1450.0, 1200.0, category=1)


def test_a_figure_stacks_one_row_per_panel(tmp_path):
    _roster(tmp_path, "fishbone")
    panels = [
        Panel(title="mhr B1", x=np.arange(10.0), y=np.zeros((1, 10)), ylabel="T/s"),
        Panel(
            title="spectrogram",
            kind="heatmap",
            x=np.arange(10.0),
            y=np.arange(5.0),
            z=np.zeros((5, 10)),
            ylabel="kHz",
        ),
    ]
    session = ReviewSession(
        event="fishbone", shot=185601, panels=panels, root=tmp_path, reviewer="alice"
    )
    assert len(session.figure.data) >= 2
    assert "185601" in session.figure.layout.title.text


def test_save_without_verify_writes_corrections_but_leaves_tier_unverified(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.mark(1200.0, 1450.0, category=1)
    session.save()

    corrections = read_corrections(review_path("fishbone", 185601, root=tmp_path))
    assert len(corrections) == 1

    roster = read_roster(tmp_path / "fishbone" / "shots.csv")
    assert roster.iloc[0].tier == "unverified"
    assert roster.iloc[0].reviewers == ""


def test_verify_with_no_marks_writes_the_roster_but_no_corrections_file(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.verify()
    session.save()

    assert not review_path("fishbone", 185601, root=tmp_path).exists()
    roster = read_roster(tmp_path / "fishbone" / "shots.csv")
    assert roster.iloc[0].tier == "unverified"
    assert roster.iloc[0].reviewers == "alice"


def test_label_panel_orientation_matches_x_and_y(tmp_path):
    time_ms = np.arange(0.0, 500.0, 50.0)
    labels = np.zeros((len(time_ms), 20))
    labels[:, 0] = 1
    write_label_grid(
        tmp_path / "fishbone" / "format" / "shots" / "185601.npz", time_ms, labels
    )

    panel = label_panel("fishbone", 185601, source="format/shots", root=tmp_path)

    assert panel.kind == "heatmap"
    assert panel.z.shape == (len(panel.y), len(panel.x))


def test_review_with_no_saved_grid_warns_and_keeps_only_given_panels(tmp_path):
    _roster(tmp_path, "fishbone")
    panels = [
        Panel(title="mhr B1", x=np.arange(10.0), y=np.zeros((1, 10)), ylabel="T/s"),
    ]
    with pytest.warns(UserWarning, match="fishbone"):
        session = review("fishbone", 185601, panels, root=tmp_path, reviewer="alice")

    assert len(session.panels) == len(panels)
    assert all(a is b for a, b in zip(session.panels, panels, strict=True))


def test_a_lasso_selection_reports_a_readable_error_from_the_button(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.figure.layout.selections = [
        go.layout.Selection(type="path", path="M0,0L1,1Z")
    ]
    _button(session, "Mark present").click()

    status = session.controls.children[1]
    assert "Box Select" in status.value
    assert session.corrections.empty


def test_marking_present_twice_on_one_selection_records_one_row(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.figure.layout.selections = [
        go.layout.Selection(type="rect", x0=1200.0, x1=1450.0, y0=0.0, y1=1.0)
    ]
    present = _button(session, "Mark present")
    present.click()
    present.click()

    assert len(session.corrections) == 1


def test_every_category_has_a_verification_notebook():
    import json

    from labeler.config import Paths

    root = Paths.from_env().label_tables
    categories = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert len(categories) == 16
    for category in categories:
        path = root / category / "verification.ipynb"
        assert path.is_file(), f"{category} has no verification.ipynb"
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        sources = "".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )
        assert "from labeler.events.verify import" in sources, category
        # The kernel is still called "Python (FAITH labelmaker)" on purpose;
        # what must not survive the rename is the MODULE path.
        assert "labelmaker.events" not in sources, category
        assert "from labelmaker" not in sources, category


@pytest.mark.real_data
def test_the_generic_notebook_names_features_that_exist():
    """The scaffolded panels shipped `ne_line`, which the store has never held.

    A notebook naming a missing feature raises KeyError on the cell a reviewer
    runs first, so it is dead on arrival and nothing in the suite noticed.
    """
    import glob
    import re
    from pathlib import Path

    from labeler.config import Paths
    from labeler.features.store import present

    scaffold = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "labeler"
        / "make_verification_notebook.py"
    )
    named = set(re.findall(r'trace\("([^"]+)"', scaffold.read_text()))
    assert named, "no trace(...) calls found; did the scaffold change shape?"

    files = sorted(glob.glob(str(Paths.from_env().features / "*_features.h5")))
    if not files:
        pytest.skip("no feature files on this host")
    everywhere = set.intersection(*(present(f) for f in files[:40]))
    missing = sorted(named - everywhere)
    assert not missing, f"generic panels name features no shot has: {missing}"


def test_a_cache_refuses_to_serve_different_points(tmp_path, monkeypatch):
    """The cache is keyed by filename, which says nothing about its channels.

    Without this check, asking for two points against a cache built from
    three returns three rows silently mislabelled as the two - wrong data in
    front of a reviewer, with nothing on screen to give it away.
    """
    n = 10
    ms = np.linspace(0.0, 20.0, n)

    def record(offset):
        return {
            "data": np.arange(n, dtype="float32") + offset,
            "dim0": ms,
            "units": {"data": "keV", "dim0": "ms"},
        }

    calls = []
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds",
        _fake_fetch_mds({"a": record(0), "b": record(100), "c": record(200)}, calls),
    )
    cache = tmp_path / "178640_ece.npz"
    fdp_signal(178640, ["a", "b", "c"], cache=cache)
    assert len(calls) == 3

    with pytest.raises(NoDataError, match="Delete it"):
        fdp_signal(178640, ["a", "b"], cache=cache)
    assert len(calls) == 3, "the refusal must not fall through to a fetch"


def test_no_expressions_is_refused():
    with pytest.raises(NoDataError, match="no expressions"):
        fdp_signal(178640, [])
