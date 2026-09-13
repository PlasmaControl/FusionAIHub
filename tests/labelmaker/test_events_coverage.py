"""Coverage per source and per quantity, and extent clipped into it.

Two of the iteration-0 critic's findings, on shot 198658:

* `pipeline._span` gave every actuator event the union of every actuator
  axis. Gas runs approximately -10 to 94.8576 s on that shot, NBI 0 to
  13.1001 s and the RMP coils -1.06286 to 10.20114 s; all three rows
  declared -10 to 94.8576 s of coverage. Those three spans are the ones
  the synthetic features here are drawn to, so a failure names the
  behaviour the critic measured.
* 9, 11 and 15 track rows on the three pilot shots ended up to 0.000772,
  0.000772 and 0.002052 s after their own `t_cov1_s`. That overrun is the
  transform's edge support, and it is reproduced here through the real
  `masks.prep` and `masks.col_times_s` rather than by writing an
  out-of-range time by hand: a 40 ms record at 500 kHz comes back as 164
  columns whose last centre is 0.962 ms past the last sample.

Everything is drawn; nothing opens `/scratch/gpfs/EKOLEMEN`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from labelmaker.events import coverage, heuristics, masks, schema, tracks

SHOT = 198658
SHA = "0" * 64

#: The critic's independently inspected axes for shot 198658, in seconds.
GAS_SPAN = (-10.0, 94.8576)
NBI_SPAN = (0.0, 13.1001)
RMP_SPAN = (-1.06286, 10.20114)


def _axis(span, step=0.002):
    lo, hi = span
    return np.arange(lo, hi + 0.5 * step, step, dtype=np.float64)


def _pulse(t, lo, hi, level, off=0.0):
    return np.where((t >= lo) & (t <= hi), float(level), float(off))


# ------------------------------------------------------------ finite_span

def test_finite_span_is_the_first_and_last_sample_that_exists():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
    assert coverage.finite_span(t, y) == (1.0, 3.0)
    # Without a trace, the axis itself is the question asked.
    assert coverage.finite_span(t) == (0.0, 4.0)


def test_trailing_nan_padding_is_not_coverage():
    # The corpus' fast groups end in NaN and a filterscope's head and tail
    # are NaN; a span read off the axis would claim them as observed.
    t = np.linspace(0.0, 1.0, 101)
    y = np.ones(101)
    y[-10:] = np.nan
    assert coverage.finite_span(t, y)[1] == pytest.approx(0.90)


def test_one_dead_channel_does_not_end_a_multi_channel_observation():
    # "The RMP coils are on" is a claim about the SET of them, and one
    # dead coil is not the end of the record.
    t = np.linspace(0.0, 1.0, 11)
    y = np.full((3, 11), np.nan)
    y[1] = 1.0
    assert coverage.finite_span(t, y) == (0.0, 1.0)


def test_nothing_finite_is_unknown_not_zero():
    t = np.linspace(0.0, 1.0, 5)
    got = coverage.finite_span(t, np.full(5, np.nan))
    assert all(math.isnan(v) for v in got)
    assert all(math.isnan(v) for v in coverage.finite_span([], None))


def test_a_length_mismatch_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="samples against"):
        coverage.finite_span(np.zeros(4), np.zeros(5))


# -------------------------------------------------------------- intersect

def test_the_intersection_is_where_every_input_was_measured():
    assert coverage.intersect([(0.0, 5.0), (1.0, 9.0), (-2.0, 4.0)]) == (1.0, 4.0)


def test_one_unmeasured_input_makes_the_intersection_unknown():
    # A heuristic whose density trace was never measured did not observe
    # the D-alpha's stretch; it observed nothing.
    got = coverage.intersect([(0.0, 5.0), coverage.UNKNOWN])
    assert all(math.isnan(v) for v in got)
    assert all(math.isnan(v) for v in coverage.intersect([]))


def test_inputs_that_never_overlapped_cover_no_time_together():
    got = coverage.intersect([(0.0, 1.0), (2.0, 3.0)])
    assert all(math.isnan(v) for v in got)


# --------------------------------------------------------- clip_to_coverage

def test_clipping_trims_both_ends_and_says_it_happened():
    assert coverage.clip_to_coverage(-0.1, 1.1, (0.0, 1.0)) == (0.0, 1.0, True)
    assert coverage.clip_to_coverage(0.2, 0.8, (0.0, 1.0)) == (0.2, 0.8, False)


def test_an_unknown_bound_clips_nothing_on_its_side():
    nan = float("nan")
    assert coverage.clip_to_coverage(-1.0, 2.0, (nan, 1.0)) == (-1.0, 1.0, True)
    assert coverage.clip_to_coverage(-1.0, 2.0, (0.0, nan)) == (0.0, 2.0, True)
    assert coverage.clip_to_coverage(-1.0, 2.0, coverage.UNKNOWN) == (
        -1.0, 2.0, False
    )


def test_an_extent_wholly_outside_its_coverage_is_not_repaired():
    # Not a transform edge: a detector claiming an event in a stretch it
    # did not measure. Moving it onto the boundary would make a bug look
    # like a row; left alone, `Event` refuses it.
    assert coverage.clip_to_coverage(5.0, 6.0, (0.0, 1.0)) == (5.0, 6.0, False)
    with pytest.raises(ValueError, match="t1_s must not exceed t_cov1_s"):
        schema.Event(shot=SHOT, source="s", phenomenon="p", t0_s=5.0,
                     t1_s=6.0, t_cov0_s=0.0, t_cov1_s=1.0)


def test_the_coverage_invariant_only_fires_on_two_finite_bounds():
    # Most rows of most files have a known coverage; a row that has none
    # is not thereby invalid, it is unchecked, and it says so.
    schema.Event(shot=SHOT, source="s", phenomenon="p", t0_s=5.0, t1_s=6.0)
    schema.Event(shot=SHOT, source="s", phenomenon="p", t0_s=0.5, t1_s=1.0,
                 t_cov0_s=0.0, t_cov1_s=1.0)


# ----------------------------------------- per-actuator coverage (defect 2a)

@pytest.fixture
def three_axes():
    """Gas, NBI and RMP on the three axes the critic inspected on 198658."""
    gas_t = _axis(GAS_SPAN, step=0.02)
    nbi_t = _axis(NBI_SPAN)
    rmp_t = _axis(RMP_SPAN)
    return {
        "gas": (gas_t, _pulse(gas_t, 1.0, 2.0, 4.0)),
        "pinj_total": (nbi_t, _pulse(nbi_t, 1.0, 3.0, 5.0e3)),
        "rmp": (rmp_t, _pulse(rmp_t, 1.0, 2.0, 2.0)[None, :]),
    }


def test_each_actuator_event_carries_its_own_axis(three_axes):
    spans = coverage.feature_spans(three_axes)
    found = heuristics.actuator_intervals(three_axes, shot=SHOT, t_cov=spans)
    got = {
        e.phenomenon: (e.diag, round(e.t_cov0_s, 5), round(e.t_cov1_s, 5))
        for e in found
    }
    assert got == {
        "gas_on": ("gas", pytest.approx(GAS_SPAN[0]),
                   pytest.approx(GAS_SPAN[1], abs=0.02)),
        "nbi_on": ("pinj_total", pytest.approx(NBI_SPAN[0]),
                   pytest.approx(NBI_SPAN[1], abs=0.002)),
        "rmp_on": ("rmp", pytest.approx(RMP_SPAN[0], abs=0.002),
                   pytest.approx(RMP_SPAN[1], abs=0.002)),
    }
    # And the defect, stated: the union of the three is the gas axis, and
    # giving it to the NBI row is claiming 81 s nobody measured.
    union = (min(s[0] for s in spans.values()),
             max(s[1] for s in spans.values()))
    assert union[1] - got["nbi_on"][2] > 80.0


def test_a_single_span_is_still_accepted_and_is_still_the_old_answer(
    three_axes,
):
    # The old signature has to go on working - `heuristics` is used
    # outside the pipeline - and this is what it does: one span, every row.
    found = heuristics.actuator_intervals(
        three_axes, shot=SHOT, t_cov=(-10.0, 94.8576)
    )
    assert {(e.t_cov0_s, e.t_cov1_s) for e in found} == {(-10.0, 94.8576)}


def test_an_actuator_axis_of_nothing_but_padding_is_unknown_coverage():
    t = np.linspace(0.0, 1.0, 51)
    spans = coverage.feature_spans({"gas": (t, np.full(51, np.nan))})
    assert all(math.isnan(v) for v in spans["gas"])


def test_nbi_counter_takes_the_intersection_of_the_three_it_needs():
    # Torque, power and current, all on their own digitisers, and the
    # claim is a comparison of all three at one instant.
    t_tinj = np.arange(0.0, 4.0, 0.002)
    t_pinj = np.arange(0.5, 5.0, 0.002)
    t_ip = np.arange(-1.0, 3.0, 0.002)
    features = {
        "tinj_total": (t_tinj, _pulse(t_tinj, 1.0, 2.5, 3.0)),
        "pinj_total": (t_pinj, _pulse(t_pinj, 0.9, 2.6, 5.0e3)),
        "ip": (t_ip, np.full(t_ip.size, -1.0e6)),
    }
    spans = coverage.feature_spans(features)
    found = heuristics.actuator_intervals(features, shot=SHOT, t_cov=spans)
    counter = [e for e in found if e.phenomenon == "nbi_counter"]
    assert counter, "the fixture is drawn to produce one"
    want = coverage.intersect(
        [spans["tinj_total"], spans["pinj_total"], spans["ip"]]
    )
    assert want[0] == pytest.approx(0.5) and want[1] == pytest.approx(2.998)
    for e in counter:
        assert (e.t_cov0_s, e.t_cov1_s) == pytest.approx(want)
    # `nbi_on` is a claim about the power alone and keeps the power's span.
    nbi = next(e for e in found if e.phenomenon == "nbi_on")
    assert (nbi.t_cov0_s, nbi.t_cov1_s) == pytest.approx(spans["pinj_total"])


# -------------------------------------------- track extent vs coverage (9)

def _stitched_block(n_samples: int = 20_000, fs_hz: float = 5.0e5):
    """A real transform of a real record: its columns run past its samples.

    `masks.prep` pads the record at both ends (`masks.COL_ORIGIN` is -3),
    so the last column CENTRE lands after the last sample. This is the
    mechanism behind the pilot shots' 9/11/15 overrunning track rows, and
    it is reproduced rather than asserted.
    """
    y = np.zeros(n_samples, dtype=np.float32)
    _, meta = masks.prep(y, fs_hz=fs_hz, decim=1)
    t_s = masks.col_times_s(meta["n_cols"], fs_hz, 1, 0.0)
    t_cov = (0.0, (n_samples - 1) / fs_hz)
    return t_s, t_cov


def test_the_stitched_transform_really_does_run_past_the_record():
    t_s, t_cov = _stitched_block()
    assert t_s[-1] > t_cov[1]
    assert 0.0 < (t_s[-1] - t_cov[1]) * 1e3 < 3.0     # sub-3 ms, as measured
    assert t_s[0] < t_cov[0]


def _track(t0, t1, **kw):
    fields = {
        "t0_s": float(t0), "t1_s": float(t1), "f0_khz": 5.0, "f1_khz": 9.0,
        "f_centroid_khz": 7.0, "chirp_khz_per_ms": 0.0, "chirp_r2": 0.0,
        "bandwidth_khz": 4.0, "duration_ms": float((t1 - t0) * 1e3),
        "duty": 1.0, "mean_prob": 0.9, "conf": 0.9, "n_pix": 400,
        "n_components": 1, "row_lit_fraction": 0.08, "pickup": False,
    }
    fields.update(kw)
    return tracks.Track(**fields)


def test_a_track_that_runs_past_the_record_is_clipped_and_says_so():
    t_s, t_cov = _stitched_block()
    over = _track(t_s[0], t_s[-1])
    inside = _track(0.01, 0.02)
    rows = tracks.tracks_to_events(
        [over, inside], shot=SHOT, diag="mhr", channel=0, pass_name="wide",
        t_cov=t_cov, unet_sha256=SHA,
    )
    clipped, kept = rows
    assert (clipped.t0_s, clipped.t1_s) == (t_cov[0], t_cov[1])
    assert clipped.attrs["clipped"] is True
    # The measurement is not rewritten: `attrs` still holds the duration
    # the descriptors computed off the mask.
    assert clipped.attrs["duration_ms"] == pytest.approx(
        (t_s[-1] - t_s[0]) * 1e3
    )
    # An ordinary row carries no key at all rather than `false`.
    assert "clipped" not in kept.attrs
    assert (kept.t0_s, kept.t1_s) == (0.01, 0.02)


def test_every_track_row_ends_inside_its_own_coverage():
    # The invariant, stated as the critic asked for it: the schema refuses
    # a row that does not, so this passing IS the guarantee.
    t_s, t_cov = _stitched_block()
    rows = tracks.tracks_to_events(
        [_track(t_s[0], t_s[-1]), _track(t_s[5], t_s[-2])],
        shot=SHOT, diag="mhr", channel=0, pass_name="wide", t_cov=t_cov,
        unet_sha256=SHA,
    )
    assert rows
    for e in rows:
        assert t_cov[0] <= e.t0_s <= e.t1_s <= t_cov[1]


def test_a_block_with_no_known_coverage_clips_nothing():
    rows = tracks.tracks_to_events(
        [_track(0.0, 1.0)], shot=SHOT, diag="mhr", channel=0,
        pass_name="wide", t_cov=coverage.UNKNOWN, unet_sha256=SHA,
    )
    assert (rows[0].t0_s, rows[0].t1_s) == (0.0, 1.0)
    assert "clipped" not in rows[0].attrs
