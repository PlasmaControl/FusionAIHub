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


def test_a_time_range_slices_rather_than_loading(tmp_path):
    seconds = np.linspace(0.0, 6.0, 6001)
    _corpus(tmp_path, 185601, {"ece": (seconds, np.zeros((48, 6001)))})
    got = corpus_signal(185601, "ece", t_range=(2000.0, 2100.0), corpus=tmp_path)
    assert got.x[0] >= 2000.0
    assert got.x[-1] <= 2100.0
    assert got.y.shape[1] == got.x.shape[0] < 200


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
