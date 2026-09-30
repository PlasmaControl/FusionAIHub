"""No live fetch while LABELER_NO_FETCH is set, which tier holds a record, and the
review rows' resampling as one function."""

from __future__ import annotations

from fractions import Fraction

import h5py
import numpy as np
import pytest
from scipy import signal

from labeler.config import Paths
from labeler.events import raw
from labeler.events.review import alfven
from labeler.events.verify import NoDataError
from labeler.features.store import FeatureArray


def _write(path, group, times_ms, values):
    """One corpus-layout group: seconds on xdata, float32 ydata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        g = f.create_group(group)
        g.create_dataset("xdata", data=np.asarray(times_ms, "float32") / 1000.0)
        g.create_dataset("ydata", data=np.asarray(values, "float32"))


@pytest.fixture
def roots(tmp_path):
    return Paths(corpus=tmp_path / "corpus", raw_cache=tmp_path / "cache")


@pytest.fixture
def fetches(monkeypatch):
    calls = []

    def fake(shot, exprs, *, tree, via, on_progress=None, **kwargs):
        calls.append(shot)
        return FeatureArray(
            x=np.arange(10.0), y=np.ones((len(exprs), 10), "float32"), attrs={}
        )

    monkeypatch.setattr(raw, "fdp_signal", fake)
    return calls


@pytest.mark.parametrize("value", ["1", "yes"])
def test_no_fetch_raises_before_any_fdp_call(roots, fetches, monkeypatch, value):
    monkeypatch.setenv(raw.NO_FETCH_ENV, value)
    with pytest.raises(raw.FetchDisabledError, match="LABELER_NO_FETCH") as err:
        raw.raw_signal(7, "co2", paths=roots)
    assert isinstance(err.value, NoDataError)
    assert fetches == []
    assert not (roots.raw_cache / "7_processed.h5").exists()


def test_no_fetch_raises_for_a_group_without_a_route_too(roots, fetches, monkeypatch):
    monkeypatch.setenv(raw.NO_FETCH_ENV, "1")
    with pytest.raises(raw.FetchDisabledError):
        raw.raw_signal(7, "no_such_group", paths=roots)


@pytest.mark.parametrize("value", [None, "", "0"])
def test_fetching_is_on_when_the_switch_is_unset_empty_or_0(
    roots, fetches, monkeypatch, value
):
    if value is None:
        monkeypatch.delenv(raw.NO_FETCH_ENV, raising=False)
    else:
        monkeypatch.setenv(raw.NO_FETCH_ENV, value)
    assert raw.fetching_disabled() is False
    got = raw.raw_signal(8, "co2", paths=roots)
    assert fetches == [8] and got.attrs["tier"] == "fetch"


def test_no_fetch_still_reads_the_corpus_and_the_cache(roots, fetches, monkeypatch):
    monkeypatch.setenv(raw.NO_FETCH_ENV, "1")
    _write(roots.corpus / "9_processed.h5", "co2", np.arange(5.0), np.zeros((4, 5)))
    _write(roots.raw_cache / "10_processed.h5", "co2", np.arange(5.0), np.ones((4, 5)))
    assert raw.raw_signal(9, "co2", paths=roots).attrs["tier"] == "corpus"
    assert raw.raw_signal(10, "co2", paths=roots).attrs["tier"] == "cache"
    assert fetches == []


def test_record_tier_names_the_first_tier_holding_a_record(roots, fetches):
    _write(roots.corpus / "11_processed.h5", "co2", np.arange(5.0), np.zeros((4, 5)))
    _write(roots.raw_cache / "11_processed.h5", "ece", np.arange(5.0), np.zeros((2, 5)))
    _write(roots.raw_cache / "11_processed.h5", "co2", np.arange(5.0), np.ones((4, 5)))
    _write(roots.corpus / "12_processed.h5", "co2", np.arange(1.0), np.zeros((4, 1)))
    assert raw.record_tier(11, "co2", paths=roots) == "corpus"
    assert raw.record_tier(11, "ece", paths=roots) == "cache"
    assert raw.record_tier(11, "sxr", paths=roots) is None
    assert raw.record_tier(12, "co2", paths=roots) is None  # the sentinel is no record
    assert raw.record_tier(13, "co2", paths=roots) is None  # no file at all
    assert fetches == []


def _old_resample(time_ms, chords):
    """The lines `spectrogram_rows` held before this task, verbatim."""
    time_ms = np.asarray(time_ms, dtype=np.float64)
    rate = (len(time_ms) - 1) / ((time_ms[-1] - time_ms[0]) / 1000)
    ratio = Fraction(alfven.RATE_HZ / rate).limit_denominator(100)
    x = signal.resample_poly(
        np.asarray(chords, dtype=np.float32),
        ratio.numerator,
        ratio.denominator,
        axis=-1,
    )
    return x, rate * ratio.numerator / ratio.denominator


@pytest.mark.parametrize("rate_hz", [1_666_666.7, 1_000_000.0, 500_000.0])
def test_resample_is_the_review_rows_own_lines(rate_hz):
    rng = np.random.default_rng(20260923)
    n = 6000
    time_ms = (np.arange(n) / rate_hz * 1000 - 3.0).astype(np.float32)
    chords = rng.normal(size=(4, n)).astype(np.float32)
    x, fs = alfven.resample(time_ms, chords)
    want_x, want_fs = _old_resample(time_ms, chords)
    assert fs == want_fs
    assert x.dtype == want_x.dtype and np.array_equal(x, want_x)


def test_spectrogram_rows_resample_through_resample(monkeypatch):
    seen = []
    real = alfven.resample

    def spy(time_ms, chords):
        seen.append(np.asarray(chords).shape)
        return real(time_ms, chords)

    monkeypatch.setattr(alfven, "resample", spy)
    n = 4000
    time_ms = np.arange(n) / 1000.0  # 1 MHz
    chords = np.random.default_rng(0).normal(size=(4, n))
    _, rows = alfven.spectrogram_rows(time_ms, chords)
    assert seen == [(4, n)]
    assert [r.name for r in rows] == ["R0xV1", "R0xV2", "R0xV3"]
