"""Raw-file reads behind an actuator seed (`shot_design.retrieval.actuation`'s trace helpers).

These are the tests of `Trace`, `decimate`, `_clip`, `_json_values`, `_trace`, `_empty_trace`
and `read_traces` -- the block that lived in `shotrec.ui.waveforms` upstream and moved into
`retrieval.actuation` here, because `seed_reference`/`seed_median` are the only callers left
once the browser page is out of scope. What they pin is the contract a seed depends on: a
decimated trace never loses a spike, a window that selects nothing is not "present with no
data", a name that cannot be served still comes back carrying a status, and a sample no member
recorded stays None rather than becoming a 0.0 that reads as "every beam was off".

Fixture facts: shot 900001 has Ip 1.2 MA, beams 15l 2.0 MW / 30l 1.5 MW / 33l 1.0 MW on
1000-4500 ms sampled at 1 kHz, Bt 2.0 T; shot 900002 has Ip only plus a (1,1) NaN placeholder
`ech` group.

The upstream module's `groups`, `spectrogram` and `profile` tests are not here: those three
functions render the browser page's channel picker, MHD spectrogram and profile slice, and none
of them was ported.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import numpy as np
import pytest

from shot_design.retrieval import actuation

from .conftest import BEAMS, write_frame

# ------------------------------------------------------------------------------------ decimate


def test_decimate_keeps_every_buckets_min_and_max():
    t = np.arange(100_000, dtype=float)
    y = np.sin(t / 500.0)
    y[54_321] = 40.0  # a one-sample spike
    y[12_345] = -40.0
    t2, y2 = decimated = actuation.decimate(t, y, 1500)
    assert len(decimated[0]) <= 3000
    assert y2.max() == 40.0 and y2.min() == -40.0
    assert 54_321.0 in t2 and 12_345.0 in t2
    assert np.all(np.diff(t2) > 0), "output stays in time order"


def test_decimate_leaves_short_series_alone():
    t = np.arange(10, dtype=float)
    y = t * 2
    t2, y2 = actuation.decimate(t, y, 1500)
    assert t2 is t and y2 is y


def test_decimate_survives_all_nan_buckets():
    t = np.arange(10_000, dtype=float)
    y = np.full(10_000, np.nan)
    y[:5000] = 1.0
    t2, y2 = actuation.decimate(t, y, 100)
    assert t2.size > 0 and np.nanmax(y2) == 1.0


# ------------------------------------------------------------------------- _clip, _json_values


def test_clip_keeps_the_closed_window_and_returns_the_inputs_when_unbounded():
    t = np.arange(0.0, 10.0, 1.0)
    y = t * 2
    unbounded = actuation._clip(t, y, None, None)
    assert unbounded[0] is t and unbounded[1] is y  # no copy when there is no window
    t2, y2 = actuation._clip(t, y, 2.0, 5.0)
    assert t2.tolist() == [2.0, 3.0, 4.0, 5.0]  # both ends inclusive
    assert y2.tolist() == [4.0, 6.0, 8.0, 10.0]
    assert actuation._clip(t, y, 4.0, None)[0].tolist() == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert actuation._clip(t, y, None, 1.0)[0].tolist() == [0.0, 1.0]
    assert actuation._clip(t, y, 100.0, 200.0)[0].size == 0


def test_json_values_turns_every_non_finite_sample_into_none():
    """JSON has no NaN, and 0.0 would read as a measurement. None is "not recorded"."""
    y = np.array([1.5, np.nan, -0.0, np.inf, -np.inf, 2.0])
    assert actuation._json_values(y) == [1.5, None, -0.0, None, None, 2.0]
    assert all(v is None or isinstance(v, float) for v in actuation._json_values(y))


# ----------------------------------------------------------------------------------- read_traces


def test_read_traces_registry_total_and_unknown(paths, staged_shot_a):
    traces = actuation.read_traces(staged_shot_a, ["ip", "nbi.total", "bt", "nope"], paths)
    by = {t.name: t for t in traces}
    assert by["ip"].status == "present" and by["ip"].units == "A"
    assert abs(max(by["ip"].y) - 1.2e6) < 1.0e4  # the trapezoid's flat top, |Ip| in A
    assert by["nbi.total"].status == "present"
    assert abs(max(by["nbi.total"].y) - 4.5e6) < 1.0e3  # 2.0 + 1.5 + 1.0 MW summed
    assert set(by["nbi.total"].members) >= {"15L", "30L", "33L"}
    assert by["bt"].units == "T" and abs(by["bt"].y[0] - 2.0) < 1e-6
    assert by["nope"].status == "unknown_signal" and by["nope"].t_ms == []
    for tr in traces:
        assert len(tr.t_ms) == len(tr.y)
        assert len(tr.t_ms) <= 3000


def test_read_traces_raw_channel_and_window(paths, staged_shot_a):
    (tr,) = actuation.read_traces(
        staged_shot_a, ["raw:p_inj/pinjf_15l"], paths, t0=2000.0, t1=3000.0
    )
    assert tr.status == "present" and tr.label == "p_inj/pinjf_15l"
    assert min(tr.t_ms) >= 2000.0 and max(tr.t_ms) <= 3000.0
    assert abs(max(tr.y) - 2.0e6) < 1.0  # beam 15L at 2.0 MW throughout the window


def test_read_traces_missing_group_reports_status(paths, staged_shot_b):
    (tr,) = actuation.read_traces(staged_shot_b, ["raw:p_inj/pinjf_15l"], paths)
    assert tr.status == "unavailable" and tr.y == []


def test_read_traces_actuator_member_keys(paths, staged_shot_a):
    traces = actuation.read_traces(
        staged_shot_a,
        ["nbi.15L", "nbi.21L", "ech.LUKE", "nbi.nope", "gas.GASA"],
        paths,
    )
    by = {t.name: t for t in traces}
    assert by["nbi.15L"].status == "present"
    assert by["nbi.15L"].units == "W"
    assert by["nbi.15L"].name == "nbi.15L"
    assert max(by["nbi.15L"].y) == pytest.approx(2.0e6)
    assert by["nbi.21L"].status == "present"
    assert all(v == 0.0 for v in by["nbi.21L"].y)  # fixture writes zeros for unused beams
    assert by["ech.LUKE"].status != "present"  # no ech group in 900001
    assert by["ech.LUKE"].units == "W"
    assert by["nbi.nope"].status == "unknown_signal"
    assert by["gas.GASA"].status == "present"
    assert max(by["gas.GASA"].y) == pytest.approx(20.0)


def test_an_absent_trace_names_no_source_shot(paths, staged_shot_a):
    """`_empty_trace` leaves `source` at None. A trace that carries no samples must not also
    carry a provenance, or `seed_reference` reports a shot as the origin of a waveform that
    shot never recorded."""
    absent = actuation.read_traces(
        staged_shot_a, ["ech.LUKE", "nbi.nope", "nope", "raw:p_inj/nope"], paths
    )
    for tr in absent:
        assert tr.status != "present"
        assert tr.t_ms == [] and tr.y == [] and tr.members == []
        assert tr.source is None
    # by contrast, a trace that did come from a file says where it came from
    (present,) = actuation.read_traces(staged_shot_a, ["nbi.15L"], paths)
    assert present.status == "present" and present.source == "staged"


def test_a_single_member_actuator_is_not_a_total(paths, staged_shot_a):
    """`gas.GASA` is one member of a one-per-species system: it reads through the member branch
    (no `members` list), while `gas.total` goes through `_total` and names what it summed."""
    (member,) = actuation.read_traces(staged_shot_a, ["gas.GASA"], paths)
    assert member.status == "present" and member.members == []
    assert max(member.y) == pytest.approx(20.0)
    (total,) = actuation.read_traces(staged_shot_a, ["gas.total"], paths)
    assert total.status == "present" and total.name == "gas.total"
    assert "GASA" in total.members
    assert max(total.y) == pytest.approx(20.0)  # only GASA puffs; the rest are zeros


def test_total_is_none_where_no_member_recorded(paths, staged_shot_a):
    """`features.system_totals`' rule, carried into the trace: a sample no member recorded is
    NaN (None in JSON), never the 0.0 that would read as "every beam was off"."""
    t = np.arange(0.0, 100.0, 1.0)
    gap = (t >= 40) & (t < 60)
    pin = {f"pinjf_{b}": np.zeros_like(t) for b in BEAMS}
    pin["pinjf_15l"][:] = 1.0e6
    for b in BEAMS:
        pin[f"pinjf_{b}"][gap] = np.nan
    write_frame(paths.staged_raw_dir / f"{staged_shot_a}.h5", "p_inj", t, pin)
    (tr,) = actuation.read_traces(staged_shot_a, ["nbi.total"], paths)
    assert tr.status == "present" and len(tr.t_ms) == t.size  # 100 samples: no decimation
    assert tr.y[10] == 1.0e6
    assert tr.y[50] is None


def test_read_traces_window_outside_the_signal(paths, staged_shot_a):
    """A present signal clipped to nothing is not "present with no data"."""
    (tr,) = actuation.read_traces(staged_shot_a, ["ip"], paths, t0=9e6, t1=9.1e6)
    assert tr.status == "no_samples_in_window" and tr.t_ms == [] and tr.y == []
