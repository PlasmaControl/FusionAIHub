"""The three-tier raw read, and the cache file it writes."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events import raw
from labeler.events.verify import NoDataError


def write_corpus_file(path, group, times_ms, values):
    """A corpus file by hand: seconds on xdata, no attributes anywhere."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g.create_dataset("xdata", data=np.asarray(times_ms, "float32") / 1000.0)
        g.create_dataset("ydata", data=np.asarray(values, "float32"))


@pytest.fixture
def roots(tmp_path):
    return Paths(corpus=tmp_path / "corpus", raw_cache=tmp_path / "cache")


def test_tier_one_is_the_corpus(roots):
    times, values = np.arange(10.0), np.arange(20.0).reshape(2, 10)
    write_corpus_file(roots.corpus / "1_processed.h5", "co2", times, values)
    got = raw.raw_signal(1, "co2", paths=roots)
    assert np.allclose(got.x, times)
    assert np.allclose(got.y, values)
    assert got.attrs["tier"] == "corpus"


def test_tier_two_is_the_cache(roots):
    times, values = np.arange(10.0), np.arange(20.0).reshape(2, 10)
    write_corpus_file(roots.raw_cache / "2_processed.h5", "co2", times, values)
    got = raw.raw_signal(2, "co2", paths=roots)
    assert np.allclose(got.y, values)
    assert got.attrs["tier"] == "cache"


def test_the_corpus_wins_over_the_cache(roots):
    """Both tiers hold shot 3; the corpus copy is the one served."""
    write_corpus_file(roots.corpus / "3_processed.h5", "co2",
                      np.arange(4.0), np.zeros((1, 4)))
    write_corpus_file(roots.raw_cache / "3_processed.h5", "co2",
                      np.arange(4.0), np.ones((1, 4)))
    assert np.allclose(raw.raw_signal(3, "co2", paths=roots).y, 0.0)


def test_channels_and_t_range_slice_the_way_the_corpus_does(roots):
    times = np.arange(100.0)
    values = np.arange(400.0).reshape(4, 100)
    write_corpus_file(roots.corpus / "4_processed.h5", "co2", times, values)
    got = raw.raw_signal(4, "co2", channels=[1, 3], t_range=(10.0, 20.0),
                         paths=roots)
    # The window is asserted in SAMPLE SPACINGS, not in absolute ms, because
    # the edge is exact only up to one sample. `xdata` is float32 seconds, so
    # 10 ms round-trips to 9.9999998 ms - just below the bound - and
    # `corpus_signal`'s searchsorted(side="left") therefore starts one sample
    # late. Here a sample is 1 ms; on real CO2 at 1.667 MHz it is 0.6 us.
    #
    # This cannot bias a recorded correction. A mark's times come from the
    # reviewer's drag on the figure, never from this array's extent.
    spacing = 1.0
    assert 10.0 <= got.x[0] <= 10.0 + spacing
    assert 20.0 - spacing <= got.x[-1] <= 20.0
    assert got.y.shape[0] == 2
    assert got.y.shape[-1] == got.x.shape[-1]
    # Whatever the window's edges, the rows are the ones asked for and the
    # values line up with the times - which is what would break if `channels`
    # were applied before the time slice, or to the wrong axis.
    start = round(float(got.x[0]))
    assert np.allclose(got.y[0], values[1, start : start + got.y.shape[-1]])
    assert np.allclose(got.y[1], values[3, start : start + got.y.shape[-1]])


def test_a_group_the_file_lacks_and_cannot_be_fetched_raises(roots):
    write_corpus_file(roots.corpus / "5_processed.h5", "co2",
                      np.arange(4.0), np.zeros((1, 4)))
    with pytest.raises(NoDataError, match="no fetch route"):
        raw.raw_signal(5, "not_a_group", paths=roots)


def test_write_group_is_additive(roots, tmp_path):
    path = tmp_path / "6_processed.h5"
    raw.write_group(path, "co2", np.arange(4.0), np.zeros((1, 4)))
    raw.write_group(path, "ece", np.arange(4.0), np.ones((1, 4)))
    assert raw.groups_in(path) == {"co2", "ece"}


def test_write_group_leaves_no_partial_file_when_it_fails(roots, tmp_path):
    path = tmp_path / "7_processed.h5"
    with pytest.raises(ValueError):
        # y must be (C, T); a 1-D y is rejected BEFORE anything is created,
        # so a reader never meets a half-written shot.
        raw.write_group(path, "co2", np.arange(4.0), np.zeros(4))
    assert not path.exists()
    assert list(tmp_path.glob("*")) == []


def test_write_group_stores_seconds_so_the_corpus_can_read_it(tmp_path):
    path = tmp_path / "8_processed.h5"
    raw.write_group(path, "co2", np.array([0.0, 1000.0]), np.zeros((1, 2)))
    with h5py.File(path, "r") as f:
        assert np.allclose(f["co2"]["xdata"][:], [0.0, 1.0])
