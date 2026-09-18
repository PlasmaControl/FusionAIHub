"""Reading corpus signals for a review, and what a review writes."""

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from labeler.events.interval_tables import (
    INTERVAL_COLUMNS,
    SAMPLE_MS,
    write_label_grid,
)
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
    with pytest.raises(NoDataError) as raised:
        fdp_signal(178640, ["a", "b"])
    message = str(raised.value)
    assert "10 for 'a'" in message
    assert "20 for 'b'" in message

    # The FIRST expression sets the expected length, so the same pair the
    # other way round names the other one as the reference.
    with pytest.raises(NoDataError, match="20 for 'b' vs 10 for 'a'"):
        fdp_signal(178640, ["b", "a"])


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


def _fake_fetch_ptdata(records, calls):
    """A stand-in for `_fetch_ptdata` returning canned records, call-counted.

    `records` maps point name -> the dict `_fetch_ptdata` would return, with
    the PTDATA time key `times` and `units["times"]` rather than `dim0`.
    `calls` is a list this appends the name to.
    """

    def fake(name, shot):
        calls.append(name)
        return records[name]

    return fake


def test_fdp_signal_via_ptdata_returns_expressions_in_order(monkeypatch):
    n = 10
    ms = np.linspace(0.0, 20.0, n)
    records = {
        "DENR0UF": {
            "data": np.arange(n, dtype="float32"),
            "times": ms,
            "units": {"data": "f", "times": "ms"},
        },
        "DENV1UF": {
            "data": np.arange(n, dtype="float32") + 100,
            "times": ms,
            "units": {"data": "f", "times": "ms"},
        },
    }
    mds_calls: list[str] = []
    ptdata_calls: list[str] = []
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds", _fake_fetch_mds({}, mds_calls)
    )
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_ptdata",
        _fake_fetch_ptdata(records, ptdata_calls),
    )
    got = fdp_signal(178642, ["DENR0UF", "DENV1UF"], via="ptdata")
    assert got.y.shape == (2, n)
    assert got.y.dtype == np.float32
    assert got.y[0] == pytest.approx(records["DENR0UF"]["data"])
    assert got.y[1] == pytest.approx(records["DENV1UF"]["data"])
    assert got.x[0] == pytest.approx(0.0)
    assert got.x[-1] == pytest.approx(20.0)
    assert ptdata_calls == ["DENR0UF", "DENV1UF"]
    assert mds_calls == [], "the ptdata route must never call the mds fetch"


def test_fdp_signal_via_ptdata_rejects_a_non_millisecond_unit(monkeypatch):
    n = 10
    records = {
        "DENR0UF": {
            "data": np.arange(n, dtype="float32"),
            "times": np.linspace(0.0, 0.02, n),
            "units": {"data": "f", "times": "s"},
        },
    }
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_ptdata",
        _fake_fetch_ptdata(records, []),
    )
    with pytest.raises(NoDataError, match="'s'"):
        fdp_signal(178642, ["DENR0UF"], via="ptdata")


def test_fdp_signal_rejects_an_unknown_via(monkeypatch):
    mds_calls: list[str] = []
    ptdata_calls: list[str] = []
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_mds", _fake_fetch_mds({}, mds_calls)
    )
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_ptdata",
        _fake_fetch_ptdata({}, ptdata_calls),
    )
    with pytest.raises(ValueError, match="mdsplus"):
        fdp_signal(178642, ["a"], via="mdsplus")
    assert mds_calls == []
    assert ptdata_calls == []


def test_fdp_signal_via_ptdata_cache_is_not_refetched(tmp_path, monkeypatch):
    n = 10
    ms = np.linspace(0.0, 20.0, n)
    records = {
        "DENR0UF": {
            "data": np.arange(n, dtype="float32"),
            "times": ms,
            "units": {"data": "f", "times": "ms"},
        },
    }
    calls: list[str] = []
    monkeypatch.setattr(
        "labeler.features.resolve_fdp._fetch_ptdata",
        _fake_fetch_ptdata(records, calls),
    )
    cache = tmp_path / "178642_co2.npz"

    first = fdp_signal(178642, ["DENR0UF"], via="ptdata", cache=cache)
    assert cache.is_file()
    assert len(calls) == 1

    second = fdp_signal(178642, ["DENR0UF"], via="ptdata", cache=cache)
    assert len(calls) == 1, "the second call must not touch the network"
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(first.x, second.x)


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
    """One subplot row per panel, and every trace on its own panel's row.

    A trace count alone says nothing about the stacking: two panels whose
    traces all landed on row 1 would satisfy it, so this follows each trace
    to its y axis instead.
    """
    _roster(tmp_path, "fishbone")
    panels = [
        Panel(
            title="mhr B1-B2",
            x=np.arange(10.0),
            y=np.zeros((2, 10)),
            ylabel="T/s",
        ),
        Panel(
            title="spectrogram",
            kind="heatmap",
            x=np.arange(10.0),
            y=np.arange(5.0),
            z=np.zeros((5, 10)),
            ylabel="kHz",
        ),
        Panel(title="ip", x=np.arange(10.0), y=np.zeros((1, 10)), ylabel="A"),
    ]
    session = ReviewSession(
        event="fishbone", shot=185601, panels=panels, root=tmp_path, reviewer="alice"
    )
    figure = session.figure

    layout = figure.layout.to_plotly_json()
    rows = sorted(key for key in layout if key.startswith("yaxis"))
    assert rows == ["yaxis", "yaxis2", "yaxis3"]
    # two traces from panel 1, one from panel 2, one from panel 3
    assert [trace.yaxis for trace in figure.data] == ["y", "y", "y2", "y3"]
    assert [trace.type for trace in figure.data] == [
        "scattergl",
        "scattergl",
        "heatmap",
        "scattergl",
    ]
    assert [layout[row]["title"]["text"] for row in rows] == ["T/s", "kHz", "A"]
    assert "185601" in figure.layout.title.text


def test_an_unknown_panel_kind_names_the_panel_and_the_kind(tmp_path):
    """`__post_init__` deliberately leaves an unknown kind alone so this
    branch, which names both, is the one that reports it.
    """
    _roster(tmp_path, "fishbone")
    panel = Panel(
        title="mystery", kind="contour", x=np.arange(10.0), y=np.zeros((1, 10))
    )
    with pytest.raises(ValueError, match="unknown kind 'contour'") as raised:
        ReviewSession(event="fishbone", shot=185601, panels=[panel], root=tmp_path)
    assert "mystery" in str(raised.value)


def test_a_failing_save_reports_on_the_button_instead_of_raising(tmp_path):
    """The Save handler runs inside a widget callback, where an exception is a
    traceback in the log the reviewer is not reading. It has to land in the
    status line.
    """
    # No roster written, so `record_review` cannot read one.
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.mark(1200.0, 1450.0, category=1)
    session.verify()

    _button(session, "Save").click()

    status = session.controls.children[1]
    assert "#b2182b" in status.value
    assert "shots.csv" in status.value
    assert not (tmp_path / "fishbone" / "shots.csv").exists()


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
    # Plotly reads N+1 coordinates as cell EDGES and N as cell CENTRES, so a
    # label row built from the grid's own left edges has to hand over one more
    # coordinate than it has columns. Nothing on screen distinguishes the two,
    # which is why the edges are asserted here rather than only the shape.
    assert panel.z.shape == (len(panel.y) - 1, len(panel.x) - 1)
    assert panel.x[0] == pytest.approx(time_ms[0])
    assert panel.x[1] - panel.x[0] == pytest.approx(SAMPLE_MS)
    assert panel.x[-1] == pytest.approx(time_ms[-1] + SAMPLE_MS)
    assert panel.y[0] == pytest.approx(0.0)
    assert panel.y[-1] == pytest.approx(1.0)


def test_the_label_row_colour_scale_spans_the_grids_own_categories(tmp_path):
    """Autoscaled, an all-class-1 shot renders in the colours of an all-class-4
    one, and two shots reviewed back to back use different colour->class maps.
    The grid declares its classes, so the scale comes from there.
    """
    time_ms = np.arange(0.0, 500.0, 50.0)
    labels = np.ones((len(time_ms), 20))
    write_label_grid(
        tmp_path / "minimum_safety_factor" / "format" / "shots" / "185601.npz",
        time_ms,
        labels,
        categories={
            "0": "absent",
            "1": "low",
            "2": "hybrid",
            "3": "elevated",
            "4": "high",
        },
    )

    panel = label_panel(
        "minimum_safety_factor", 185601, source="format/shots", root=tmp_path
    )

    assert (panel.zmin, panel.zmax) == (0.0, 4.0)
    session = ReviewSession(
        event="minimum_safety_factor", shot=185601, panels=[panel], root=tmp_path
    )
    heatmap = session.figure.data[0]
    assert (heatmap.zmin, heatmap.zmax) == (0.0, 4.0)


def test_a_grid_without_a_category_mapping_leaves_the_scale_to_plotly(tmp_path):
    time_ms = np.arange(0.0, 500.0, 50.0)
    write_label_grid(
        tmp_path / "fishbone" / "format" / "shots" / "185601.npz",
        time_ms,
        np.zeros((len(time_ms), 20)),
    )
    panel = label_panel("fishbone", 185601, source="format/shots", root=tmp_path)
    assert panel.zmin is None
    assert panel.zmax is None


def test_a_heatmap_without_z_is_refused():
    """Plotly renders `z=None` as an empty row and reports nothing."""
    with pytest.raises(ValueError, match="needs z"):
        Panel(
            title="spectrogram",
            kind="heatmap",
            x=np.arange(10.0),
            y=np.arange(5.0),
        )


def test_a_transposed_heatmap_is_refused_and_names_all_three_lengths():
    with pytest.raises(ValueError, match=r"z is \(10, 5\)") as raised:
        Panel(
            title="spectrogram",
            kind="heatmap",
            x=np.arange(10.0),
            y=np.arange(5.0),
            z=np.zeros((10, 5)),
        )
    message = str(raised.value)
    assert "len(y)=5" in message
    assert "len(x)=10" in message
    assert "spectrogram" in message


def test_both_legitimate_heatmap_shapes_are_accepted():
    """Bin CENTRES, as a spectrogram arrives, and bin EDGES, as the label row
    hands over - plotly reads N coordinates as centres and N+1 as edges, and
    both are correct. Only these two.
    """
    centres = Panel(
        title="centres",
        kind="heatmap",
        x=np.arange(10.0),
        y=np.arange(5.0),
        z=np.zeros((5, 10)),
    )
    edges = Panel(
        title="edges",
        kind="heatmap",
        x=np.arange(11.0),
        y=np.arange(6.0),
        z=np.zeros((5, 10)),
    )
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[centres, edges], root="data/events"
    )
    assert [trace.type for trace in session.figure.data] == ["heatmap", "heatmap"]


def test_a_one_dimensional_line_y_is_refused():
    with pytest.raises(ValueError, match="channels"):
        Panel(title="qmin", x=np.arange(10.0), y=np.zeros(10))


def test_a_line_whose_y_is_shorter_than_x_is_refused():
    """Plotly truncates to the shorter and draws the trace at the wrong times -
    on the one surface whose output is corrected timings.
    """
    with pytest.raises(ValueError, match="truncate"):
        Panel(title="qmin", x=np.arange(10.0), y=np.zeros((2, 7)))


def test_a_band_on_a_heatmap_is_boundary_lines_not_a_wash_over_the_data(tmp_path):
    """plotly's `shape.layer` defaults to "above", so a filled band darkens
    the image under review; `layer="below"` would hide it behind the image.
    """
    _roster(tmp_path, "fishbone")
    panel = Panel(
        title="spectrogram",
        kind="heatmap",
        x=np.arange(10.0),
        y=np.arange(5.0),
        z=np.zeros((5, 10)),
        bands=[(2.0, 30.0)],
    )
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[panel], root=tmp_path
    )
    shapes = list(session.figure.layout.shapes)
    assert [shape.type for shape in shapes] == ["line", "line"]
    assert sorted(shape.y0 for shape in shapes) == [2.0, 30.0]
    assert not any(shape.fillcolor for shape in shapes)


def test_a_band_on_a_line_panel_is_filled_below_the_traces(tmp_path):
    _roster(tmp_path, "minimum_safety_factor")
    panel = Panel(
        title="qmin",
        x=np.arange(10.0),
        y=np.zeros((1, 10)),
        bands=[(0.95, 1.5)],
    )
    session = ReviewSession(
        event="minimum_safety_factor", shot=185601, panels=[panel], root=tmp_path
    )
    (shape,) = session.figure.layout.shapes
    assert shape.type == "rect"
    assert shape.layer == "below"
    assert (shape.y0, shape.y1) == (0.95, 1.5)


def test_hlines_reach_the_figure(tmp_path):
    """A threshold is a line. Shading 0.01 of an axis spanning 0.8-3 at 8%
    opacity, which is what a band had to fake, is invisible.
    """
    _roster(tmp_path, "minimum_safety_factor")
    panel = Panel(
        title="qmin",
        x=np.arange(10.0),
        y=np.zeros((1, 10)),
        ylabel="q",
        hlines=[0.95, 1.5, 2.0],
    )
    session = ReviewSession(
        event="minimum_safety_factor", shot=185601, panels=[panel], root=tmp_path
    )
    shapes = list(session.figure.layout.shapes)
    assert [shape.type for shape in shapes] == ["line"] * 3
    assert [shape.y0 for shape in shapes] == [0.95, 1.5, 2.0]
    assert all(shape.line.dash == "dash" for shape in shapes)


def test_a_missing_label_grid_is_named_in_the_figure_title(tmp_path):
    """The `warnings.warn` goes to stderr above the widget, which is not where
    the reviewer is looking.
    """
    _roster(tmp_path, "fishbone")
    panels = [Panel(title="mhr B1", x=np.arange(10.0), y=np.zeros((1, 10)))]
    with pytest.warns(UserWarning, match="fishbone"):
        session = review("fishbone", 185601, panels, root=tmp_path, reviewer="alice")

    assert "NO LABEL ROW" in session.figure.layout.title.text


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
    # No count: a pinned 16 fails the next time a category is added or
    # removed, which happened twice while this surface was being built, and
    # fails as `assert 17 == 16`. The per-category assertions below are the
    # content of this test.
    assert categories, f"no category directories under {root}"
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


def test_notebooks_module_is_only_the_generic_helpers():
    from labeler.events import notebooks

    assert hasattr(notebooks, "load_shot")
    assert hasattr(notebooks, "plot_shot")
    assert not hasattr(notebooks, "plot_original"), (
        "per-category original-label plotting belongs in each example.ipynb"
    )


def test_no_example_notebook_imports_plot_original():
    import json

    from labeler.config import Paths

    root = Paths.from_env().label_tables
    for path in sorted(root.glob("*/example.ipynb")):
        try:
            cells = json.loads(path.read_text())["cells"]
        except json.JSONDecodeError:
            # data/events/detachment/example.ipynb is a pre-existing empty
            # file (not valid JSON); nothing about this task touches it.
            continue
        sources = "".join("".join(cell["source"]) for cell in cells)
        assert "plot_original" not in sources, path
        assert "labelmaker.events" not in sources, path
        assert "from labelmaker" not in sources, path
