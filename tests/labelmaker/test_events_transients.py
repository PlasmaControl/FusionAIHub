"""Transient column activity -> bursts -> ELM times, a clock, quiet intervals.

Everything here is measured against a train that was DRAWN: bursts every 60
columns of a WIDE pass, whose column lasts 0.256 ms, so the period is 15.36
ms and the rate 65.104 Hz - the number the spec pins - and every expected
value below is arithmetic on the columns the bursts were put at.

The two ports are pinned the way `test_events_tracks.py` pins its merge:
`extract_bursts` is compared against `_tokeye_events`, a literal
transcription of `tokeye.elmspec.events`' inclusive-extent version, on 100
random activity vectors, and `column_activity` is compared against the
`_col_act` array `masks.block_arrays` already writes.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from labelmaker.events import masks, schema, transients


def test_slow_baseline_with_gap_contains_no_elm_spikes():
    t = np.arange(20001) / 10000
    y = 1 + 0.2 * np.sin(2 * np.pi * t)
    y[(t >= 0.8) & (t <= 1.1)] = np.nan
    rows = transients.elm_clock_events(y, t, shot=2)
    peaks = [e for e in rows if e.phenomenon == "elm"]
    assert not peaks, [(e.t0_s, e.attrs["width_ms"]) for e in peaks]
    assert [e.phenomenon for e in rows] == ["elm_free", "elm_free"]
    np.testing.assert_allclose(
        [(e.t0_s, e.t1_s) for e in rows], [(0.0, 0.7999), (1.1001, 2.0)]
    )


@pytest.mark.parametrize("width_ms", [1.0, 2.0])
def test_narrow_dalpha_pulses_survive_slow_baseline_and_gap(width_ms):
    t = np.arange(20001) / 10000
    peaks = np.array([0.2, 0.6, 1.3, 1.7])
    sigma = width_ms / 1000 / 2.355  # Gaussian full width at half maximum.
    y = 1 + 0.2 * np.sin(2 * np.pi * t)
    y += sum(np.exp(-0.5 * ((t - p) / sigma) ** 2) for p in peaks)
    y[(t >= 0.8) & (t <= 1.1)] = np.nan
    rows = transients.elm_clock_events(y, t, shot=2)
    points = [e for e in rows if e.phenomenon == "elm"]
    np.testing.assert_allclose([e.t0_s for e in points], peaks, atol=0.0001)
    assert all(e.t1_s < 0.8 or e.t0_s > 1.1 for e in rows)


def test_dalpha_units_do_not_turn_small_noise_into_elms():
    t = np.arange(10001) / 10000
    peaks = np.array([0.2, 0.4, 0.6, 0.8])
    y = 1 + 0.001 * np.sin(2 * np.pi * 170 * t)
    y += sum(np.exp(-0.5 * ((t - p) / 0.001) ** 2) for p in peaks)
    for scale in (1.0, 1e15):
        rows = transients.elm_clock_events(y * scale, t, shot=1)
        points = [e for e in rows if e.phenomenon == "elm"]
        assert len(points) == 4
        np.testing.assert_allclose([e.t0_s for e in points], peaks, atol=0.0001)


def test_dalpha_padding_and_gaps_create_neither_peaks_nor_quiet_intervals():
    t = np.arange(10001) / 10000
    y = np.zeros_like(t)
    y[:1000] = y[9001:] = y[4000:6001] = np.nan
    rows = transients.elm_clock_events(y, t, shot=1)
    assert [e.phenomenon for e in rows] == ["elm_free", "elm_free"]
    np.testing.assert_allclose([(e.t0_s, e.t1_s) for e in rows],
                               [(0.1, 0.3999), (0.6001, 0.9)])
    assert all((e.t_cov0_s, e.t_cov1_s) == (0.1, 0.9) for e in rows)

#: The synthetic train is a WIDE pass of a 500 kHz record: hop 128 samples,
#: so 0.256 ms per column, 60 columns per ELM, 15.36 ms, 65.104 Hz.
TRAIN_FS_HZ = 5.0e5
TRAIN_DECIM = 1
TRAIN_HOP_S = 2.56e-4
TRAIN_PERIOD_COLS = 60
TRAIN_PERIOD_S = TRAIN_PERIOD_COLS * TRAIN_HOP_S
TRAIN_RATE_HZ = 1.0 / TRAIN_PERIOD_S
TRAIN_COL0 = 200

#: Rows of 512 a burst lights, and the column activity that makes.
BURST_ROWS = 320
BURST_ACTIVITY = BURST_ROWS / masks.N_BINS
#: A row count well under `ACTIVITY_MIN * 512` = 51.2: noise, not a burst.
NOISE_ROWS = 32


def _times(n_cols: int) -> np.ndarray:
    """The wide pass' own column grid, starting at t=0 of the record."""
    return masks.col_times_s(int(n_cols), TRAIN_FS_HZ, TRAIN_DECIM, 0.0)


def _train_cols(n: int, col0: int = TRAIN_COL0, period: int = TRAIN_PERIOD_COLS):
    return [col0 + i * period for i in range(n)]


def _activity(cols, n_cols: int, *, value: float = BURST_ACTIVITY, width: int = 1):
    """A column-activity trace with a `width`-column burst at each of `cols`."""
    out = np.zeros(int(n_cols), dtype=np.float64)
    for col in cols:
        out[col:col + width] = value
    return out


def _tra(cols, n_cols: int, *, rows: int = BURST_ROWS, prob: float = 0.9):
    """The `(512, T)` transient probability map those bursts came from."""
    out = np.zeros((masks.N_BINS, int(n_cols)), dtype=np.float32)
    for col in cols:
        out[:rows, col] = prob
    return out


def _write_block(path, tra, *, shot=198658, diag="mhr", channel=0, thr=None):
    """One stored `(diag, channel, "wide")` block holding `tra`."""
    zeros = np.zeros_like(tra)
    block = masks.MaskBlock(
        diag=diag, channel=channel, pass_name="wide",
        coh=zeros, tra=tra, raw_logpow=zeros, t_s=_times(tra.shape[1]),
        meta={"fs_hz": TRAIN_FS_HZ, "decim": TRAIN_DECIM},
    )
    extra = {} if thr is None else {"thr": float(thr)}
    return masks.write_masks(
        path, shot, [masks.block_arrays(block, unet_sha256="c" * 64, **extra)]
    )


# --------------------------------------------------------------- constants

def test_the_thresholds_are_ours_and_the_row_threshold_is_the_masks_own():
    # The one threshold that is NOT ours: a row counts as transient at the
    # probability the masks file already thresholded it at, so
    # `column_activity` recomputes the array the file stores.
    assert transients.ACTIVITY_THR == masks.PROB_THRESHOLD == 0.2
    assert transients.ACTIVITY_MIN == 0.1
    assert transients.MIN_GAP_COLS == 3
    assert transients.SMOOTH_MS == 0.64
    assert transients.PROMINENCE == 0.03
    assert transients.MIN_DISTANCE_MS == 3.0
    assert transients.RATE_WINDOW_S == 0.1
    assert transients.ELM_FREE_MAX_RATE_HZ == 5.0
    assert transients.ELM_FREE_MIN_S == 0.05


def test_the_sources_and_phenomena_are_ones_the_events_table_knows():
    assert transients.SOURCE in schema.KNOWN_SOURCES
    assert transients.FREE_SOURCE in schema.KNOWN_SOURCES
    assert transients.PHENOMENON == "transient"
    assert transients.FREE_PHENOMENON == "elm_free"


# -------------------------------------------------------- column activity

def test_column_activity_is_the_fraction_of_rows_at_or_over_the_threshold():
    tra = np.zeros((masks.N_BINS, 4), dtype=np.float32)
    tra[:128, 1] = 0.9
    tra[:256, 2] = transients.ACTIVITY_THR          # at the threshold: lit
    tra[:256, 3] = transients.ACTIVITY_THR - 1e-6   # under it: not
    got = transients.column_activity(tra)
    assert got.dtype == np.float32
    assert got.tolist() == [0.0, 0.25, 0.5, 0.0]


def test_column_activity_is_the_array_the_masks_file_already_stores(tmp_path):
    # `masks.block_arrays` writes `_col_act` so nothing has to unpack a
    # mask to get it; the two must be the same array, bit for bit.
    rng = np.random.default_rng(11)
    tra = rng.random((masks.N_BINS, 400), dtype=np.float32)
    zeros = np.zeros_like(tra)
    block = masks.MaskBlock(
        diag="mhr", channel=0, pass_name="wide",
        coh=zeros, tra=tra, raw_logpow=zeros, t_s=_times(400),
        meta={"fs_hz": TRAIN_FS_HZ, "decim": TRAIN_DECIM},
    )
    stored = masks.block_arrays(block, unet_sha256="a" * 64)["mhr_00_wide_col_act"]
    assert np.array_equal(transients.column_activity(tra), stored)


def test_column_activity_wants_a_full_height_transient_map():
    with pytest.raises(ValueError):
        transients.column_activity(np.zeros((16, 100), dtype=np.float32))


# ------------------------------------------------------------------ bursts

def test_a_run_of_lit_columns_is_one_half_open_burst():
    act = _activity([50], 200, width=7)
    act[52] = 0.9                                   # the peak of the burst
    got = transients.extract_bursts(act)
    assert len(got) == 1
    assert (got[0].col0, got[0].col1) == (50, 57)
    assert got[0].n_cols == 7
    assert got[0].peak_activity == pytest.approx(0.9)


def test_a_gap_of_two_columns_is_bridged_and_one_of_five_is_not():
    bridged = _activity([50, 55], 200, width=3)     # 53, 54 quiet: a gap of 2
    assert [(b.col0, b.col1) for b in transients.extract_bursts(bridged)] == [
        (50, 58)
    ]
    split = _activity([50, 58], 200, width=3)       # 53..57 quiet: a gap of 5
    assert [(b.col0, b.col1) for b in transients.extract_bursts(split)] == [
        (50, 53), (58, 61)
    ]


def test_columns_below_the_activity_minimum_are_not_a_burst():
    act = _activity([50], 200, value=NOISE_ROWS / masks.N_BINS, width=9)
    assert act.max() < transients.ACTIVITY_MIN
    assert transients.extract_bursts(act) == []


def test_a_burst_shorter_than_the_minimum_duration_is_dropped():
    act = _activity([50, 100], 200, width=1)
    act[100:104] = BURST_ACTIVITY
    got = transients.extract_bursts(act, min_duration_cols=4)
    assert [(b.col0, b.col1) for b in got] == [(100, 104)]


def test_extract_bursts_needs_one_trace():
    with pytest.raises(ValueError):
        transients.extract_bursts(np.zeros((2, 10)))


# ------------------------------------------------------ the TokEye port

def _tokeye_contiguous_runs(active):
    """`tokeye.elmspec.events._contiguous_runs`, transcribed. INCLUSIVE."""
    padded = np.concatenate(([False], active, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, ends = edges[::2], edges[1::2] - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _tokeye_fill_gaps(active, max_gap):
    """`tokeye.elmspec.events._fill_gaps`, transcribed.

    Its `zip(runs, runs[1:], strict=False)` is spelt `itertools.pairwise`
    here, which is the same pairs; nothing else is changed.
    """
    if max_gap <= 0:
        return active
    filled = active.copy()
    runs = _tokeye_contiguous_runs(active)
    for (_, prev_end), (next_start, _) in itertools.pairwise(runs):
        if next_start - prev_end - 1 <= max_gap:
            filled[prev_end:next_start + 1] = True
    return filled


def _tokeye_events(activity, activity_min, min_gap_cols, min_duration_cols):
    """`tokeye.elmspec.events.extract_elm_events`' tail, transcribed."""
    active = _tokeye_fill_gaps(activity >= activity_min, min_gap_cols)
    return [
        (start, end, float(activity[start:end + 1].max()))
        for start, end in _tokeye_contiguous_runs(active)
        if end - start + 1 >= min_duration_cols
    ]


def test_extract_bursts_is_the_tokeye_port_on_half_open_extents():
    rng = np.random.default_rng(3)
    for _ in range(100):
        act = rng.random(int(rng.integers(20, 200)))
        gap = int(rng.integers(0, 6))
        duration = int(rng.integers(1, 4))
        got = transients.extract_bursts(
            act, activity_min=0.5, min_gap_cols=gap, min_duration_cols=duration
        )
        want = _tokeye_events(act, 0.5, gap, duration)
        assert [(b.col0, b.col1 - 1, b.peak_activity) for b in got] == want


# -------------------------------------------------------------- ELM times

def test_elm_events_recovers_every_burst_of_the_train():
    cols = _train_cols(30)
    t_s = _times(2048)
    got = transients.elm_events(_activity(cols, 2048), t_s)
    assert len(got) == len(cols)
    # Within one column: the boxcar is two columns wide on this grid, so a
    # one-column burst smooths to a two-column plateau.
    assert np.abs(got - t_s[cols]).max() <= TRAIN_HOP_S
    assert np.diff(got) == pytest.approx(TRAIN_PERIOD_S)
    assert 1.0 / float(np.median(np.diff(got))) == pytest.approx(TRAIN_RATE_HZ)


def test_a_quiet_channel_has_no_elms():
    t_s = _times(2048)
    got = transients.elm_events(np.zeros(2048), t_s)
    assert got.shape == (0,)
    assert got.dtype == np.float64


def test_two_bursts_closer_than_the_minimum_distance_are_one_elm():
    # 3.0 ms is 11.7 columns of the wide pass, so `find_peaks` may not put
    # two ELMs 8 columns apart; 12 columns apart it may.
    t_s = _times(400)
    near = _activity([200, 208], 400)
    assert len(transients.elm_events(near, t_s)) == 1
    far = _activity([200, 212], 400)
    assert len(transients.elm_events(far, t_s)) == 2


def test_a_bump_below_the_prominence_is_not_an_elm():
    t_s = _times(400)
    act = _activity([200], 400)
    act[100] = transients.PROMINENCE / 2.0
    got = transients.elm_events(act, t_s)
    assert len(got) == 1
    assert got[0] == pytest.approx(t_s[200], abs=TRAIN_HOP_S)


# ------------------------------------------------------------- the clock

def test_the_clock_counts_the_trains_own_rate_in_a_hundred_millisecond_window():
    cols = _train_cols(45)
    t_s = _times(3000)
    elms = transients.elm_events(_activity(cols, 3000), t_s)
    clock = transients.elm_clock(elms, t_s)
    assert set(clock) == {"rate_hz", "time_since_last_s", "time_to_next_s", "phase"}
    assert all(v.shape == t_s.shape for v in clock.values())
    half = transients.RATE_WINDOW_S / 2.0
    inside = (t_s >= elms[0] + half) & (t_s <= elms[-1] - half)
    rate = clock["rate_hz"][inside]
    # 100 ms holds 6.51 periods, so a COUNT is 6 or 7 and the rate it makes
    # is 60 or 70 Hz; what averages to the train's 65.104 is the trace.
    assert set(np.unique(rate).tolist()) == {60.0, 70.0}
    assert float(rate.mean()) == pytest.approx(TRAIN_RATE_HZ, abs=0.5)


def test_the_phase_runs_from_zero_to_one_between_two_elms():
    cols = _train_cols(10)
    t_s = _times(2048)
    elms = transients.elm_events(_activity(cols, 2048), t_s)
    clock = transients.elm_clock(elms, t_s)
    seg = (t_s >= elms[0]) & (t_s < elms[1])
    assert int(seg.sum()) == TRAIN_PERIOD_COLS
    phase = clock["phase"][seg]
    assert phase[0] == pytest.approx(0.0)
    assert bool(np.all(np.diff(phase) > 0.0))
    assert phase[-1] == pytest.approx(1.0 - 1.0 / TRAIN_PERIOD_COLS)
    assert clock["time_since_last_s"][seg][0] == pytest.approx(0.0)
    assert clock["time_to_next_s"][seg][0] == pytest.approx(TRAIN_PERIOD_S)
    assert clock["time_since_last_s"][seg][-1] == pytest.approx(
        TRAIN_PERIOD_S - TRAIN_HOP_S
    )
    # Outside the train there is no interval to be a fraction of. The last
    # ELM opens one that never closes, so the phase is half-open too.
    assert bool(np.isnan(clock["phase"][t_s < elms[0]]).all())
    assert bool(np.isnan(clock["phase"][t_s >= elms[-1]]).all())
    assert bool(np.isnan(clock["time_since_last_s"][t_s < elms[0]]).all())
    assert bool(np.isnan(clock["time_to_next_s"][t_s >= elms[-1]]).all())


def test_the_clock_of_a_channel_with_no_elms_is_a_rate_of_zero():
    t_s = _times(200)
    clock = transients.elm_clock(np.zeros(0), t_s)
    assert bool((clock["rate_hz"] == 0.0).all())
    for name in ("time_since_last_s", "time_to_next_s", "phase"):
        assert bool(np.isnan(clock[name]).all())


# ------------------------------------------------------ ELM-free intervals

def test_a_train_that_stops_leaves_one_elm_free_interval():
    gap_cols = 1600                                  # 409.6 ms of quiet
    cols = _train_cols(15)
    cols += _train_cols(15, col0=cols[-1] + gap_cols)
    t_s = _times(3600)
    elms = transients.elm_events(_activity(cols, 3600), t_s)
    assert len(elms) == 30
    t_cov = (float(t_s[0]), float(t_s[-1]))
    got = transients.elm_free_intervals(elms, t_cov)
    half = transients.RATE_WINDOW_S / 2.0
    # An ELM at T puts the rate over 5 Hz for the whole 100 ms window it can
    # be counted in, so the quiet interval is the 409.6 ms gap less 100 ms.
    assert got.shape == (1, 2)
    assert got[0, 0] == pytest.approx(elms[14] + half)
    assert got[0, 1] == pytest.approx(elms[15] - half)
    assert float(got[0, 1] - got[0, 0]) == pytest.approx(
        gap_cols * TRAIN_HOP_S - transients.RATE_WINDOW_S
    )
    assert float(got[0, 1] - got[0, 0]) >= 0.3


def test_a_record_with_no_elms_in_it_is_elm_free_end_to_end():
    got = transients.elm_free_intervals(np.zeros(0), (1.0, 3.0))
    assert got.tolist() == [[1.0, 3.0]]


def test_the_coverage_window_bounds_the_elm_free_intervals():
    got = transients.elm_free_intervals(np.array([2.0]), (1.0, 3.0))
    assert got.shape == (2, 2)
    assert got[0] == pytest.approx([1.0, 1.95])
    assert got[1] == pytest.approx([2.05, 3.0])
    # An ELM on the window's own edge still costs it the first 50 ms.
    inside = transients.elm_free_intervals(np.array([2.0]), (2.0, 3.0))
    assert inside.shape == (1, 2)
    assert inside.ravel() == pytest.approx([2.05, 3.0])


def test_an_elm_free_gap_shorter_than_the_minimum_is_not_an_elm_free_phase():
    elms = np.array([1.5, 1.62])                     # 20 ms of quiet between
    got = transients.elm_free_intervals(elms, (1.0, 3.0))
    assert got.shape == (2, 2)
    assert got.ravel() == pytest.approx([1.0, 1.45, 1.67, 3.0])
    relaxed = transients.elm_free_intervals(elms, (1.0, 3.0), min_duration_s=0.01)
    assert relaxed.shape == (3, 2)
    assert relaxed.ravel() == pytest.approx([1.0, 1.45, 1.55, 1.57, 1.67, 3.0])


def test_an_empty_coverage_window_is_no_intervals():
    got = transients.elm_free_intervals(np.zeros(0), (2.0, 2.0))
    assert got.shape == (0, 2)


# ------------------------------------------------------------- the events

def _built(n_elms=15, gap_cols=1600, n_cols=3600):
    """The stopping train, as `(elms, bursts, activity, t_s)`."""
    cols = _train_cols(n_elms)
    cols += _train_cols(n_elms, col0=cols[-1] + gap_cols)
    act = _activity(cols, n_cols, width=3)
    t_s = _times(n_cols)
    return transients.elm_events(act, t_s), transients.extract_bursts(act), act, t_s


def _events(elms, bursts, act, t_s):
    return transients.transients_to_events(
        elms, bursts, shot=198658, diag="mhr", channel=0, pass_name="wide",
        t_s=t_s, t_cov=(float(t_s[0]), float(t_s[-1])),
        unet_sha256="b" * 64, activity=act,
    )


def test_every_mask_peak_is_a_transient_point_without_an_elm_claim():
    elms, bursts, act, t_s = _built()
    got = _events(elms, bursts, act, t_s)
    points = [e for e in got if e.phenomenon == "transient"]
    free = [e for e in got if e.phenomenon == "elm_free"]
    assert len(points) == len(elms)
    assert free == []
    one = points[0]
    assert one.shot == 198658
    assert one.source == "tokeye_transient"
    assert one.evidence_kind == "detector"
    assert one.t0_s == one.t1_s == pytest.approx(elms[0])
    assert (one.diag, one.channel, one.pass_name) == ("mhr", 0, "wide")
    assert (one.t_cov0_s, one.t_cov1_s) == (float(t_s[0]), float(t_s[-1]))
    assert math.isnan(one.f0_khz) and math.isnan(one.f1_khz)
    # An ELM inside a burst is as confident as that burst is active.
    assert one.confidence == pytest.approx(BURST_ACTIVITY)


def test_an_elm_outside_every_burst_is_confident_of_the_smoothed_activity():
    # A peak whose columns never reach `ACTIVITY_MIN` is still a peak - the
    # boxcar halves a one-column spike, and 0.045 clears `PROMINENCE`. What
    # it is not is a burst, and its confidence is the trace under it.
    t_s = _times(400)
    act = np.zeros(400)
    act[200] = 0.09
    elms = transients.elm_events(act, t_s)
    assert len(elms) == 1
    assert transients.extract_bursts(act) == []
    got = _events(elms, [], act, t_s)
    smoothed = transients.smooth_activity(act, t_s)
    point = next(e for e in got if e.phenomenon == "transient")
    assert point.confidence == pytest.approx(
        float(smoothed[int(np.abs(t_s - elms[0]).argmin())])
    )
    # Without the trace there is nothing to be confident of, and NaN says so.
    bare = transients.transients_to_events(
        elms, [], shot=1, diag="mhr", channel=0, pass_name="wide",
        t_s=t_s, t_cov=(float(t_s[0]), float(t_s[-1])), unet_sha256="b" * 64,
        activity=None,
    )
    assert math.isnan(next(e for e in bare if e.phenomenon == "transient").confidence)


def test_the_attrs_say_what_the_evidence_was_and_survive_json():
    elms, bursts, act, t_s = _built()
    got = _events(elms, bursts, act, t_s)
    point = next(e for e in got if e.phenomenon == "transient")
    assert point.attrs["unet_sha256"] == "b" * 64
    assert point.attrs["burst_col0"] is not None
    assert point.attrs["burst_cols"] == 3
    assert json.loads(json.dumps(point.attrs, allow_nan=False)) == dict(
        point.attrs
    )


def test_the_events_go_into_the_shots_table(tmp_path):
    elms, bursts, act, t_s = _built()
    path = tmp_path / "198658_events.parquet"
    schema.write_events(path, 198658, _events(elms, bursts, act, t_s), run_id="t")
    back = schema.read_events(path)
    assert set(back["source"]) == {"tokeye_transient"}
    assert len(schema.read_events(path, phenomenon="transient")) == len(elms)
    free = schema.intervals(back, "elm_free")
    assert free.shape == (0, 2)


def test_the_activity_the_confidence_comes_from_is_not_optional():
    # The confidence rule is "the burst you are in, else the trace under
    # you"; with `activity` defaulted a caller got the NaN fallback by
    # forgetting an argument rather than by deciding anything. Passing
    # `activity=None` is still how a caller says there is no trace.
    t_s = _times(400)
    common = {
        "shot": 1, "diag": "mhr", "channel": 0, "pass_name": "wide",
        "t_s": t_s, "t_cov": (float(t_s[0]), float(t_s[-1])),
        "unet_sha256": "b" * 64,
    }
    with pytest.raises(TypeError, match="activity"):
        transients.transients_to_events(np.zeros(0), [], **common)
    assert transients.transients_to_events(
        np.zeros(0), [], activity=None, **common
    ) == []


def test_events_of_a_channel_with_nothing_on_it():
    t_s = _times(400)
    got = transients.transients_to_events(
        np.zeros(0), [], shot=1, diag="mhr", channel=0, pass_name="wide",
        t_s=t_s, t_cov=(float(t_s[0]), float(t_s[-1])), unet_sha256="b" * 64,
        activity=None,
    )
    # A quiet mask makes no ELM claim; completion belongs in sources.parquet.
    assert got == []


# ---------------------------------------------------- transients_for_block

def test_transients_for_block_reads_a_stored_block(tmp_path):
    cols = _train_cols(15)
    cols += _train_cols(15, col0=cols[-1] + 1600)
    path = _write_block(tmp_path / "198658_masks.npz", _tra(cols, 3600))
    got = transients.transients_for_block("mhr_00_wide", path)
    t_s = _times(3600)
    assert np.array_equal(got.t_s, t_s)
    half = 0.5 * float(t_s[1] - t_s[0])
    assert got.t_cov == pytest.approx(
        (float(t_s[0]) - half, float(t_s[-1]) + half)
    )
    # The file's own `_col_act`, which is `column_activity` of what was packed.
    assert got.activity[cols[0]] == pytest.approx(BURST_ACTIVITY)
    assert len(got.bursts) == 30
    assert len(got.elm_times_s) == 30
    assert np.diff(got.elm_times_s[:15]) == pytest.approx(TRAIN_PERIOD_S)
    assert got.elm_free_s.shape == (1, 2)
    assert got.clock["rate_hz"].shape == t_s.shape
    assert float(np.nanmax(got.clock["rate_hz"])) == 70.0


def test_a_block_thresholded_somewhere_else_is_refused(tmp_path):
    # `_col_act` is `column_activity` at the file's OWN threshold, which is
    # the whole reason this path never unpacks a mask. Reading a file cut at
    # some other threshold would silently return a different trace, so it is
    # refused - naming both numbers, because which one is wrong depends on
    # who wrote the file.
    path = _write_block(tmp_path / "198658_masks.npz", _tra([200], 400), thr=0.5)
    with pytest.raises(ValueError, match=r"0\.5.*0\.2|0\.2.*0\.5"):
        transients.transients_for_block("mhr_00_wide", path)


def test_the_coverage_of_a_block_is_its_columns_true_span(tmp_path):
    # `t_s` holds column CENTRES, so the record the block covers runs half a
    # column either side of the first and last of them: reporting the centres
    # would give away half a column of coverage at each end and make a
    # detector claim it had not looked where it had.
    path = _write_block(tmp_path / "198658_masks.npz", _tra([200], 400))
    got = transients.transients_for_block("mhr_00_wide", path)
    t_s = _times(400)
    half = 0.5 * float(t_s[1] - t_s[0])
    assert got.t_cov == pytest.approx(
        (float(t_s[0]) - half, float(t_s[-1]) + half)
    )


def test_transients_for_block_needs_a_block_that_is_there(tmp_path):
    path = _write_block(tmp_path / "198658_masks.npz", _tra([], 400))
    got = transients.transients_for_block("mhr_00_wide", path)
    assert got.bursts == [] and got.elm_times_s.shape == (0,)
    with pytest.raises(KeyError):
        transients.transients_for_block("ece_08_wide", path)
