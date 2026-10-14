"""Sawtooth crash times from ECE steps (`labeler.events.crash`), on synthetic Te.

A 700 ms record at 100 kHz: the core channel ramps up and drops every 25 ms from
150 to 550 ms while an off-axis channel does the opposite. Both carry 0.01 keV of
noise on 5 and 2 keV. The record's ends are left out of every check: a step's
noise is its running median over 200 ms, which an edge does not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events import crash

T = np.arange(0.0, 700.0, 0.01)
PERIOD = 25.0
RAMP = (150.0, 550.0)
#: A crash is the drop at the end of each ramp.
CRASHES = np.arange(RAMP[0] + PERIOD, RAMP[1] + 1.0, PERIOD)
INTERIOR = (100.0, 600.0)


def _noise(rng, *shape):
    return 0.01 * rng.standard_normal(shape)


def _ramp(amp=0.5):
    y = np.zeros_like(T)
    on = (T >= RAMP[0]) & (T < RAMP[1])
    y[on] = ((T[on] - RAMP[0]) % PERIOD) / PERIOD * amp
    return y


def _inside(times):
    return times[(times > INTERIOR[0]) & (times < INTERIOR[1])]


def _sawtooth(rng):
    core = 5.0 + _ramp() + _noise(rng, len(T))
    outer = np.stack(
        [2.0 - _ramp() + _noise(rng, len(T)), 2.0 + _noise(rng, len(T))]
    )
    return core, outer


def _elms(rng, times):
    """Core and off-axis Te with a 0.3 ms spike at each of `times` (the core dips,
    an off-axis channel jumps: a crash's pattern, but gone in a moment), and the
    D-alpha rise of an ELM at the same instants."""
    core = 5.0 + _noise(rng, len(T))
    outer = 2.0 + _noise(rng, 2, len(T))
    dalpha = 1.0 + _noise(rng, len(T))
    for t0 in times:
        spike = (T >= t0) & (T < t0 + 0.3)
        core[spike] -= 1.0
        outer[0, spike] += 1.0
        decay = (T >= t0) & (T < t0 + 3.0)
        dalpha[decay] += 5.0 * np.exp(-(T[decay] - t0))
    return core, outer, (T, dalpha)


def test_a_sawtooth_gives_a_crash_at_every_drop_to_within_a_millisecond():
    core, outer = _sawtooth(np.random.default_rng(3))
    found = _inside(crash.find_crashes(T, core, outer))
    assert len(found) == len(CRASHES)
    assert np.abs(found - CRASHES).max() < 1.0


def test_elm_spikes_are_crashes_to_the_ece_but_not_once_dalpha_says_so():
    rng = np.random.default_rng(3)
    elms = np.array([320.0, 380.5, 460.0])
    core, outer, dalpha = _elms(rng, elms)
    # Alone, the 0.3 ms spike has a crash's pattern: the core down, an off-axis
    # channel up, at the same instant.
    without = _inside(crash.find_crashes(T, core, outer))
    assert len(without) == len(elms) and np.abs(without - elms).max() < 1.0
    # With D-alpha, an ELM's spike that the core is back from 1-3 ms later is the
    # ELM's, not a crash.
    assert len(_inside(crash.find_crashes(T, core, outer, dalpha))) == 0
    elm_times, _ = crash.find_elms(*dalpha)
    assert len(_inside(elm_times)) == len(elms)


def test_a_crash_inside_an_elm_stays_a_crash_when_the_core_stays_down():
    rng = np.random.default_rng(3)
    core, outer, dalpha = _elms(rng, [400.0])
    core[T >= 400.0] -= 1.0  # still down 1-3 ms later: not the spike's, a crash
    found = _inside(crash.find_crashes(T, core, outer, dalpha))
    assert len(found) == 1 and abs(found[0] - 400.0) < 1.0
    events = crash.find_events(
        crash.crash_score(T, core, outer), T, core, dalpha
    )
    assert {e["kind"] for e in events if 300 < e["t"] < 500} == {"sawtooth", "elm"}


def test_a_flat_signal_has_no_crash():
    rng = np.random.default_rng(3)
    flat = 5.0 + _noise(rng, len(T))
    found = crash.find_crashes(T, flat, np.stack([flat, flat + 1.0]))
    assert len(found) == 0


def test_a_step_is_measured_in_units_of_its_channels_own_noise():
    rng = np.random.default_rng(3)
    quiet = 5.0 + _noise(rng, len(T))
    loud = 5.0 + 10 * _noise(rng, len(T))
    for y in (quiet, loud):
        y[T >= 350.0] += 3.0  # the same 3 keV step; the loud channel is 10x noisier
    grid, z_quiet = crash.step_z(T, quiet)
    _, z_loud = crash.step_z(T, loud)
    at = np.argmin(np.abs(grid - 350.0))
    assert z_loud[at] > crash.THRESHOLD
    assert z_quiet[at] / z_loud[at] == pytest.approx(10, rel=0.3)
    assert len(grid) == len(z_quiet) and np.isfinite(z_quiet).all()


def test_a_score_needs_the_core_to_fall_and_an_off_axis_channel_to_rise():
    rng = np.random.default_rng(3)
    core = 5.0 + _noise(rng, len(T))
    outer = 2.0 + _noise(rng, len(T))
    core[T >= 350.0] -= 1.0
    grid, down_only = crash.crash_score(T, core, outer)
    at = np.argmin(np.abs(grid - 350.0))
    assert down_only[at] < crash.THRESHOLD, "the core alone is not a crash"
    outer[T >= 350.0] += 1.0
    _, both = crash.crash_score(T, core, outer)
    assert both[at] > 10 * crash.THRESHOLD
    with pytest.raises(ValueError, match="off-axis"):
        crash.crash_score(T, core, np.empty((0, len(T))))


def test_a_record_shorter_than_the_step_window_has_no_steps():
    grid, z = crash.step_z([0.0, 0.01], [1.0, 1.0])
    assert len(grid) == len(z) == 0
    assert len(crash.find_crashes([0.0], [1.0], [[1.0]])) == 0
