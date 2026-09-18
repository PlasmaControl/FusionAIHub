"""The three-tier raw read, and the cache file it writes."""
from __future__ import annotations

import shutil
import threading

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events import raw
from labeler.events.verify import NoDataError
from labeler.features.store import FeatureArray


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


def test_write_group_leaves_no_partial_file_when_the_h5py_write_fails(
    tmp_path, monkeypatch
):
    """The pre-flight ValueError test above never touches h5py.File at all.

    This exercises the OTHER half of the atomicity claim: a failure once the
    scratch copy already exists, which is what the `finally: scratch.unlink`
    block in `write_group` is for. `h5py.File` is patched to succeed on the
    first call (the real `co2` write, establishing a baseline) and raise on
    the second (the `ece` write, after `shutil.copy2` has already produced a
    scratch file to write into).
    """
    path = tmp_path / "9_processed.h5"
    real_file = h5py.File
    calls = {"n": 0}

    def flaky_file(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated failure mid-write")
        return real_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", flaky_file)

    raw.write_group(path, "co2", np.arange(4.0), np.zeros((1, 4)))
    with pytest.raises(RuntimeError, match="simulated failure"):
        raw.write_group(path, "ece", np.arange(4.0), np.ones((1, 4)))

    # The original file is exactly what it was before the failed write -
    # still holding co2, not holding a half-written ece.
    assert raw.groups_in(path) == {"co2"}
    with h5py.File(path, "r") as f:
        assert np.allclose(f["co2"]["ydata"][:], 0.0)
    # And the scratch file the failed write created is gone, not left
    # behind as a stray `.tmp` sibling.
    assert list(tmp_path.glob(".*")) == []


def test_write_group_survives_concurrent_writers_to_the_same_path(
    tmp_path, monkeypatch
):
    """Two threads adding different groups to one shot must not race.

    The server this feeds runs synchronous routes in Starlette's threadpool,
    so `co2` and `ece` landing on the same shot at once is a real scenario:
    two THREADS in one process, not two processes. A barrier placed right
    after `shutil.copy2` forces both threads to have copied the SAME
    pre-write file before either writes its own group and renames over it -
    exactly the window `write_group`'s per-path lock exists to close.

    Against the unfixed function this reliably reproduces the lost update
    (confirmed by hand before this test was committed - see the fix report).
    Against the fixed one, the second thread is still waiting on the lock
    when the first reaches the barrier, so the barrier just times out
    harmlessly and the assertion is what actually matters: both groups
    survive.
    """
    path = tmp_path / "10_processed.h5"
    write_corpus_file(path, "base", np.arange(4.0), np.zeros((1, 4)))

    barrier = threading.Barrier(2)
    real_copy2 = shutil.copy2

    def synced_copy2(*args, **kwargs):
        result = real_copy2(*args, **kwargs)
        try:
            barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError:
            pass
        return result

    monkeypatch.setattr(shutil, "copy2", synced_copy2)

    errors = []

    def run(group, values):
        try:
            raw.write_group(path, group, np.arange(4.0), values)
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors`, not swallowed
            errors.append(exc)

    t1 = threading.Thread(target=run, args=("co2", np.zeros((1, 4))))
    t2 = threading.Thread(target=run, args=("ece", np.ones((1, 4))))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors
    assert raw.groups_in(path) == {"base", "co2", "ece"}


def test_a_miss_fetches_and_fills_the_cache(roots, monkeypatch):
    """The fetch runs once; the second read is served off disk."""
    calls = []

    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        calls.append((shot, tuple(exprs), via))
        times = np.arange(50.0)
        return FeatureArray(
            x=times,
            y=np.arange(4 * 50, dtype="float32").reshape(4, 50),
            attrs={"units": "ms"},
        )

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    first = raw.raw_signal(9, "co2", paths=roots)
    assert first.attrs["tier"] == "fetch"
    assert len(calls) == 1
    assert calls[0][2] == "ptdata"

    assert raw.cache_path(9, paths=roots).is_file()
    second = raw.raw_signal(9, "co2", paths=roots)
    assert second.attrs["tier"] == "cache"
    assert len(calls) == 1, "a cached shot must not refetch"
    assert np.allclose(second.y, first.y)


def test_a_fetch_honours_channels_and_t_range_after_caching_everything(
    roots, monkeypatch
):
    """The cache holds the WHOLE record; the slice is applied to the return."""
    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        return FeatureArray(
            x=np.arange(100.0),
            y=np.arange(400.0, dtype="float32").reshape(4, 100),
            attrs={"units": "ms"},
        )

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    # Half-integer bounds, not (5.0, 9.0): xdata round-trips through seconds
    # as float32 (see `write_group`), so an integer-ms boundary can land
    # 1e-7 below its true value and lose the edge sample - the same
    # single-sample slop `test_channels_and_t_range_slice_the_way_the_corpus_does`
    # documents above. Sitting the bounds mid-sample keeps this assertion
    # exact without depending on which way that rounding falls.
    got = raw.raw_signal(10, "co2", channels=[0, 2], t_range=(4.5, 9.5),
                         paths=roots)
    assert got.y.shape == (2, 5)
    with h5py.File(raw.cache_path(10, paths=roots), "r") as f:
        assert f["co2"]["ydata"].shape == (4, 100), "the cache is not sliced"


def test_ece_fetches_over_mds_not_ptdata(roots, monkeypatch):
    seen = {}

    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        seen.update(via=via, tree=tree, n=len(exprs), exprs=list(exprs))
        return FeatureArray(
            x=np.arange(10.0),
            y=np.zeros((len(exprs), 10), dtype="float32"),
            attrs={"units": "ms"},
        )

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    raw.raw_signal(11, "ece", paths=roots)
    assert seen["via"] == "mds"
    assert seen["tree"] == "D3D"
    assert seen["n"] == 48


def test_ece_fetches_the_channels_a_corpus_row_would_index(roots, monkeypatch):
    """Fetched row i must be the same channel corpus row i is.

    The corpus group's row i holds TECEF i+1, so the fetch has to start at
    TECEF01 and run in ascending order. The two paths used to disagree here,
    which put a reviewer's ECE panel one channel off depending only on
    whether the shot happened to be in the corpus - a difference nothing on
    screen would reveal. Asserted rather than left to inspection, because a
    later `range(0, 48)` would reintroduce it silently.
    """
    seen = {}

    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        seen["exprs"] = list(exprs)
        return FeatureArray(
            x=np.arange(10.0),
            y=np.zeros((len(exprs), 10), dtype="float32"),
            attrs={"units": "ms"},
        )

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    raw.raw_signal(12345, "ece", paths=roots)
    assert seen["exprs"][0].endswith("TECEF01")
    assert seen["exprs"][-1].endswith("TECEF48")
    assert seen["exprs"] == sorted(seen["exprs"]), "ascending, like a corpus row"
    assert len(set(seen["exprs"])) == 48, "no channel fetched twice"
