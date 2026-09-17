"""Reading corpus signals for a review, and what a review writes."""

import h5py
import numpy as np
import pytest

from labeler.events.verify import NoDataError, corpus_signal


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
