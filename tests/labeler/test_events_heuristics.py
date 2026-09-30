"""The non-TokEye heuristics: sawteeth, L->H, actuators, the QH proxy.

Everything is measured against the `synth_shot` fixture, which DRAWS ten
sawtooth crashes 76 ms apart with the inversion between channels 20 and 21,
an ELM impostor that drops the whole array together, an L->H at 0.30 s and
an H->L at 0.60 s, and beams whose torque flips counter-current for 200 ms -
so every expected value below is the constant the fixture put there rather
than the number the detector happened to return.

The one port is pinned the way `test_events_tracks.py` pins its merge:
`_omnimode_envelope` is a literal transcription of
`omnimode.mrms.ece.envelope`, and this module's `envelope` must equal it to
the last bit on random data. It must also be faster, and the structural
reason it is - one envelope per SHOT rather than one per candidate crash -
is asserted by counting the calls, which no loaded machine can perturb.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from labeler.events import coverage, heuristics, schema
from labeler.events.tracks import Track

from .conftest import (
    SYNTH_CORE_CH,
    SYNTH_COUNTER_S,
    SYNTH_DALPHA_CHANNELS,
    SYNTH_DALPHA_H,
    SYNTH_DALPHA_L,
    SYNTH_ECE_CHANNELS,
    SYNTH_EDGE_CH,
    SYNTH_ELM_WIDTH_MS,
    SYNTH_HL_S,
    SYNTH_LH_S,
    SYNTH_PINJ_KW,
    SYNTH_PINJ_ON_S,
    SYNTH_SAWTOOTH_N,
    SYNTH_SAWTOOTH_PERIOD_MS,
    SYNTH_TINJ_NM,
)

# ------------------------------------------------------------ the reference

def _omnimode_envelope(X, t_ms, env_ms=1.0):
    """`omnimode.mrms.ece.envelope`, transcribed.

    Transcribed rather than imported: omnimode is not a dependency of this
    package and a test that skips when it is absent would pin nothing. Its
    defect is not here - it is that `find_sawteeth` calls this once per
    CANDIDATE - but the per-channel `bincount` loop is also the slow way to
    bin a sorted axis, and both are what `heuristics.envelope` replaces.
    """
    X = np.atleast_2d(np.asarray(X, float))
    t = np.asarray(t_ms, float)
    nbin = max(1, round((t[-1] - t[0]) / env_ms))
    edges = t[0] + np.arange(nbin + 1) * env_ms
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbin - 1)
    Xe = np.zeros((X.shape[0], nbin))
    cnt = np.bincount(idx, minlength=nbin)
    for c in range(X.shape[0]):
        Xe[c] = np.bincount(idx, weights=X[c], minlength=nbin) \
            / np.maximum(cnt, 1)
    te = edges[:-1] + env_ms / 2
    return Xe, te


def _random_record(n=200_000, channels=48, dtype=np.float32, seed=6):
    """A sorted random time axis in seconds and a `(channels, n)` record.

    The times are RANDOM rather than a grid on purpose: a uniform grid puts
    samples exactly on bin edges, where the reference's arithmetic (searched
    per sample) and this module's (searched per bin) can round to different
    sides and disagree about one sample. Random draws never land within
    1e-12 of an edge, so the equality below tests the binning rather than
    the last bit of the clock.
    """
    rng = np.random.default_rng(seed)
    t_ms = np.sort(rng.uniform(0.0, n / 500.0, n))
    return t_ms * 1e-3, rng.normal(size=(channels, n)).astype(dtype)


# ------------------------------------------------------------------ envelope

def test_the_constants_are_the_references_and_the_sources_are_known():
    assert heuristics.ENV_MS == 1.0
    assert heuristics.DROP_FRAC == 0.02
    assert heuristics.MIN_CHANNELS == 2
    assert heuristics.MIN_INTERVAL_MS == 10.0
    assert heuristics.STEP_GAP_MS == 2.0
    assert heuristics.STEP_SPAN_MS == 8.0
    assert heuristics.REL_NEG == 0.25
    assert heuristics.REL_POS == 0.3
    assert heuristics.MIN_BLOCK == 3
    assert heuristics.MAX_GAP == 2
    for source in (
        heuristics.SAWTOOTH_SOURCE, heuristics.LH_SOURCE,
        heuristics.ACTUATOR_SOURCE, heuristics.QH_SOURCE,
    ):
        assert source in schema.KNOWN_SOURCES


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_envelope_is_the_references_to_the_last_bit(dtype):
    t_s, y = _random_record(dtype=dtype)
    got, t_env_s = heuristics.envelope(y, t_s)
    want, te_ms = _omnimode_envelope(y, t_s * 1e3)
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) < 1e-12
    assert np.max(np.abs(t_env_s * 1e3 - te_ms)) < 1e-12


def test_the_envelope_beats_the_references_way_of_computing_it():
    # The 10x the plan asks for lives at the SHOT level (see the call-count
    # test below): `find_sawteeth` recomputes the envelope once per candidate
    # crash, which on shot 198658 is 429 times. `envelope` ITSELF cannot be
    # ten times faster - both versions have to read the record once and this
    # one is already near memory bandwidth - but it does read it once instead
    # of once per channel, and it never materialises the float64 copy the
    # reference's `asarray(X, float)` makes of a float32 record. Measured
    # 7.5x on a (48, 200_000) float32 array; asserted at 3x so a loaded
    # machine cannot fail it.
    #
    # The controller's decision on plan number 3 (in the ledger): the
    # binding target is the PER-SHOT one - under a second, measured 0.39 s
    # against the plan's 1.55 s - and the isolated envelope is memory-bound,
    # so this 3x assertion and the one-call structural pin below are both
    # kept and neither is raised to 10x.
    t_s, y = _random_record()

    def best_of(fn, n=3):
        elapsed = []
        for _ in range(n):
            start = time.perf_counter()
            fn()
            elapsed.append(time.perf_counter() - start)
        return min(elapsed)

    ours = best_of(lambda: heuristics.envelope(y, t_s))
    theirs = best_of(lambda: _omnimode_envelope(y, t_s * 1e3))
    assert theirs > 3.0 * ours, f"ours {ours:.4f}s, reference {theirs:.4f}s"


def test_a_bin_no_sample_falls_in_is_zero_like_the_reference():
    # `np.add.reduceat` returns the element AT an empty segment's start
    # rather than nothing, which would make a gap in the digitiser record
    # look like one sample repeated. The reference's `bincount` gives 0.
    t_ms = np.concatenate([np.arange(0.0, 2.0, 0.1), np.arange(6.0, 8.0, 0.1)])
    y = np.arange(2 * t_ms.size, dtype=np.float64).reshape(2, t_ms.size)
    got, _ = heuristics.envelope(y, t_ms * 1e-3)
    want, _ = _omnimode_envelope(y, t_ms)
    assert np.array_equal(got, want)
    assert (got[:, 2:6] == 0.0).all()


def test_the_envelope_grid_is_bin_centres_in_seconds():
    t_s = np.arange(0.0, 0.010, 1e-5)
    y = np.ones((1, t_s.size))
    env, t_env_s = heuristics.envelope(y, t_s, env_ms=2.0)
    assert env.shape == (1, 5)
    assert t_env_s == pytest.approx([0.001, 0.003, 0.005, 0.007, 0.009])


# ----------------------------------------------------------------- sawteeth

def _sawteeth(shot_data, **kw):
    return heuristics.sawtooth_events(
        shot_data["ece_y"], shot_data["ece_t_s"],
        shot=198658, t_cov=shot_data["t_cov"], **kw,
    )


def test_every_drawn_crash_is_found_within_a_millisecond(synth_shot):
    got = _sawteeth(synth_shot)
    assert len(got) == SYNTH_SAWTOOTH_N
    found = np.array([e.t0_s for e in got])
    assert np.max(np.abs(found - synth_shot["crash_times_s"])) < 1e-3


def test_the_period_of_the_train_is_the_one_that_was_drawn(synth_shot):
    summary = heuristics.sawtooth_summary(_sawteeth(synth_shot))
    assert summary["n"] == SYNTH_SAWTOOTH_N
    assert summary["median_period_ms"] == pytest.approx(
        SYNTH_SAWTOOTH_PERIOD_MS, abs=1.0
    )
    assert summary["mean_period_ms"] == pytest.approx(
        SYNTH_SAWTOOTH_PERIOD_MS, abs=1.0
    )


def test_a_summary_of_nothing_is_a_count_and_no_period():
    summary = heuristics.sawtooth_summary([])
    assert summary["n"] == 0
    assert math.isnan(summary["median_period_ms"])
    assert math.isnan(summary["mean_period_ms"])


def test_a_crash_says_which_channels_it_inverted_between(synth_shot):
    one = _sawteeth(synth_shot)[0]
    assert one.attrs["inversion_channel_lo"] == SYNTH_CORE_CH[0]
    assert one.attrs["inversion_channel_stop"] == SYNTH_CORE_CH[1]
    assert one.attrs["n_channels_dropping"] == SYNTH_CORE_CH[1] - SYNTH_CORE_CH[0]
    assert one.attrs["n_channels_rising"] == SYNTH_EDGE_CH[1] - SYNTH_EDGE_CH[0]
    assert one.attrs["drop_frac_max"] == pytest.approx(0.05, abs=0.01)
    # Confidence is the share of the array that took part in the inversion:
    # 17 dropping plus 20 rising of 48 channels.
    assert one.confidence == pytest.approx(37 / SYNTH_ECE_CHANNELS)
    assert 0.0 < one.confidence <= 1.0
    assert json.loads(json.dumps(one.attrs, allow_nan=False)) == dict(one.attrs)


def test_the_elm_impostor_is_a_crash_candidate_and_not_a_sawtooth(synth_shot):
    # Every channel drops together, so the dominant negative block runs the
    # whole array, touches both ends and has no heat pulse next door.
    impostor = synth_shot["elm_impostor_s"]
    found = np.array([e.t0_s for e in _sawteeth(synth_shot)])
    assert np.min(np.abs(found - impostor)) > SYNTH_ELM_WIDTH_MS * 1e-3
    steps = np.full(SYNTH_ECE_CHANNELS, -0.05)
    assert heuristics.inversion_block(steps) is None
    assert not heuristics.has_inversion(steps)


def test_nan_channels_are_ignored_rather_than_poisoning_the_array(synth_shot):
    y = synth_shot["ece_y"].copy()
    y[44:] = np.nan
    got = _sawteeth({**synth_shot, "ece_y": y})
    assert len(got) == SYNTH_SAWTOOTH_N
    assert np.max(
        np.abs(np.array([e.t0_s for e in got]) - synth_shot["crash_times_s"])
    ) < 1e-3
    # The dead channels leave the block where it was and leave the confidence
    # a fraction of the channels that were actually LOOKED at.
    one = got[0]
    assert one.attrs["inversion_channel_lo"] == SYNTH_CORE_CH[0]
    assert one.confidence == pytest.approx(37 / 44)


def test_the_envelope_is_computed_once_a_shot_not_once_a_candidate(
    synth_shot, monkeypatch
):
    # The reference's defect, and the only reason this module exists as a
    # port rather than an import: `find_sawteeth` -> `crash_steps` ->
    # `envelope` per candidate is O(candidates x record), which is 3.5
    # minutes on shot 198658 against 0.9 seconds here.
    calls = []
    real = heuristics.envelope

    def counted(*args, **kw):
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(heuristics, "envelope", counted)
    got = heuristics.sawtooth_events(
        synth_shot["ece_y"], synth_shot["ece_t_s"],
        shot=1, t_cov=synth_shot["t_cov"],
    )
    assert len(got) == SYNTH_SAWTOOTH_N
    assert sum(calls) == 1


def test_candidates_exactly_the_minimum_interval_apart_both_survive():
    # The thinning is `|i - k| * env_ms >= min_interval_ms` in BIN space.
    # Between two bin TIMES a few seconds into the shot the same quantity
    # carries a rounding of ~1e-13 ms, and on a 1 ms grid with a 10 ms
    # threshold candidates land EXACTLY on that boundary all shot - so which
    # side of it a rounding falls is which crashes the shot is said to have.
    def kept(*bins, interval=10.0):
        drop = np.zeros((4, 40))
        drop[:3, list(bins)] = 0.5
        return heuristics._crash_bins(
            drop, env_ms=1.0, drop_frac=0.02, min_channels=2,
            min_interval_ms=interval,
        )

    assert kept(5, 15) == [5, 15]        # exactly 10 bins: both
    assert kept(5, 14) == [14]           # 9 bins: the later one wins the tie


def test_the_crash_window_is_the_same_seven_bins_everywhere_in_the_record():
    # The step windows are `gap_ms <= |i - k| * env_ms <= span_ms` in BIN
    # space, which is the reference's arithmetic in the reference's unit.
    # Selecting them by comparing bin CENTRES IN SECONDS against `tc_s -
    # span_s` rounds independently on the two sides, and on a 7000-bin grid
    # that dropped one of the seven bins from 86 before-windows and 78
    # after-windows - a 1.2% chance, per crash, of measuring the step over a
    # window one bin short.
    #
    # Every bin here holds its own index, so a seven-bin window either side
    # of bin `k` has means `k - 5` and `k + 5` and the step is EXACTLY 10.0
    # wherever it is measured. One bin missing moves it by 1/7 or more.
    t_s = np.arange(700_000, dtype=np.float64) / 1.0e5
    _, t_env_s = heuristics.envelope(np.zeros((1, t_s.size)), t_s)
    assert t_env_s.size == 7000
    offsets = heuristics._step_offsets(
        heuristics.ENV_MS, heuristics.STEP_GAP_MS, heuristics.STEP_SPAN_MS
    )
    assert offsets.tolist() == [2, 3, 4, 5, 6, 7, 8]
    env = np.tile(np.arange(t_env_s.size, dtype=np.float64), (2, 1))
    steps = np.array([
        heuristics._crash_step(
            env, k, env_ms=heuristics.ENV_MS, gap_ms=heuristics.STEP_GAP_MS,
            span_ms=heuristics.STEP_SPAN_MS,
        )[0]
        for k in range(int(offsets[-1]), t_env_s.size - int(offsets[-1]))
    ])
    assert np.array_equal(steps, np.full(steps.size, 10.0))


def test_the_step_windows_are_whole_bins_on_a_grid_that_is_not_whole_ms():
    # The docstring's claim is that the windows are the bins `gap_ms` to
    # `span_ms` away EXACTLY. Comparing `d * env_ms` against the two bounds
    # in floating point does not keep it on a grid whose spacing does not
    # divide them: 3 * 0.7 is 2.0999999999999996, a hair under a `gap_ms`
    # of 2.1, so the nearest bin is dropped, and 8 * 0.7 is
    # 5.6000000000000005, a hair over a `span_ms` of 5.6, so the farthest
    # one is too. Asking the question in bins - how many bins is 2.1 ms -
    # has no such edge.
    assert heuristics._step_offsets(0.7, 2.1, 5.6).tolist() == [3, 4, 5, 6, 7, 8]
    # The pipeline's own grid is unchanged by saying it that way.
    assert heuristics._step_offsets(1.0, 2.0, 8.0).tolist() == [2, 3, 4, 5, 6, 7, 8]


def test_a_window_that_falls_off_the_end_of_the_record_is_no_step():
    # The reference's answer: NaN for every channel when either side has
    # nothing in it, and the bins that ARE there when it is merely short.
    env = np.tile(np.arange(20.0), (3, 1))
    assert np.isnan(heuristics._crash_step(env, 0)).all()
    assert np.isnan(heuristics._crash_step(env, 19)).all()
    # Bin 3: the before window is bins 1 and 0 only, mean 0.5; the after
    # window is the full seven, mean 8.
    assert heuristics._crash_step(env, 3) == pytest.approx(np.full(3, 7.5))


def test_a_sawtooth_is_a_point_event_the_table_accepts(synth_shot, tmp_path):
    got = _sawteeth(synth_shot)
    one = got[0]
    assert one.source == "ece_sawtooth"
    assert one.phenomenon == "sawtooth"
    assert one.evidence_kind == "heuristic"
    assert one.t0_s == one.t1_s
    assert (one.diag, one.channel) == ("ece", -1)
    assert (one.t_cov0_s, one.t_cov1_s) == synth_shot["t_cov"]
    assert math.isnan(one.f0_khz) and math.isnan(one.f1_khz)
    path = tmp_path / "198658_events.parquet"
    back = schema.write_events(path, 198658, got, run_id="t")
    assert len(back) == SYNTH_SAWTOOTH_N
    assert set(back["source"]) == {"ece_sawtooth"}


def test_a_record_with_no_crash_in_it_is_no_events(synth_shot):
    flat = np.ones_like(synth_shot["ece_y"])
    assert _sawteeth({**synth_shot, "ece_y": flat}) == []


def test_the_ece_array_has_to_be_the_ece_array(synth_shot):
    with pytest.raises(ValueError, match="48"):
        _sawteeth({**synth_shot, "ece_y": synth_shot["ece_y"][:8]})


def test_the_committed_reference_comparison_pins_the_sawtooth_acceptance():
    # `scripts/labeler/sawtooth_reference_check.py` runs omnimode's own
    # `find_sawteeth` and this port over the WHOLE of shot 198658 and writes
    # what it found here. The port is only a port if somebody has compared
    # the two crash for crash on real data, and this is that comparison's
    # committed record - which is also where the module docstring's
    # acceptance band comes from. Re-run the script when the crash search
    # changes; do not edit the file.
    record = json.loads(
        (Path(__file__).parent / "data"
         / "sawtooth_198658_reference.json").read_text()
    )
    assert record["shot"] == 198658
    assert record["max_seconds"] is None          # the whole record, no prefix
    assert record["same_count"] and record["agree"]
    assert record["max_abs_delta_ms"] <= record["agree_tolerance_ms"]
    assert len(record["crashes"]) == record["reference"]["n"]
    for side in ("reference", "port"):
        assert abs(record[side]["n"] - 47) <= 3
        assert abs(record[side]["median_period_ms"] - 69.0) <= 5.0
    # The per-shot target of plan number 3, which is the binding one.
    assert record["port"]["elapsed_s"] < 1.0
    assert record["reference"]["elapsed_s"] > 10.0 * record["port"]["elapsed_s"]
    # The omnimode side of the record is expensive (about eight minutes) and
    # is not re-run when the port changes; `--port-only` re-runs the PORT on
    # the same shot and amends the record, so the crash count the acceptance
    # rests on is a measurement at the current code and not at whatever the
    # code was when omnimode last ran.
    rerun = record["port_rerun"]
    assert rerun["n"] == record["reference"]["n"]
    assert abs(rerun["median_ms"] - record["reference"]["median_period_ms"]) < 1e-6
    assert rerun["max_abs_dt_ms"] <= record["agree_tolerance_ms"]
    assert rerun["elapsed_s"] < 1.0
    assert len(rerun["labelmaker_sha"]) == 40
    # `labelmaker_sha` is HEAD at run time, which is the PARENT of the
    # commit carrying the record - the script has to run before the commit
    # its output goes into. That names the tree that was MEASURED only if
    # nothing under `src/labeler` was uncommitted when it ran, so the
    # stanza says which, and this asserts it was clean.
    assert rerun["dirty"] is False


# ------------------------------------------------------------- sawteeth, v3
#
# v3 judges each step in its channel's own noise, so its synthetic array has
# noise: a peaked Te profile over 48 channels, the core (17-23, the hottest 20
# in the middle) ramping and crashing 20 % every 50 ms, the channels either side
# of it (13-16, 24-27) taking a 5 % heat pulse, 1 % noise a sample at 10 kHz.

V3_FS_HZ = 1.0e4
V3_T_MS = (0.0, 2000.0)
V3_PERIOD_MS = 50.0
#: Thirty crashes, each 0.3 ms into its 1 ms bin, so the bin is unambiguous.
V3_CRASHES_MS = 300.3 + V3_PERIOD_MS * np.arange(30)
V3_CORE = (17, 24)
V3_PULSE = ((13, 17), (24, 28))


def v3_rows(
    *,
    n_channels=48,
    centre=20.0,
    core=V3_CORE,
    pulse=V3_PULSE,
    crashes_ms=V3_CRASHES_MS,
    drop=0.2,
    rise=0.05,
    noise=0.01,
    t_ms=V3_T_MS,
    seed=0,
):
    """`(t_s, y)`: a sawtooth train on a peaked profile, with noise.

    `core` falls by `drop` at each crash and ramps back over the period, the
    `pulse` channels jump by `rise` and relax back; before the first period
    the core is low and flat, and after the last crash it ramps up and holds.
    """
    t = np.arange(t_ms[0], t_ms[1], 1e3 / V3_FS_HZ)
    crashes = np.asarray(crashes_ms, dtype=np.float64)
    phase = np.zeros_like(t)
    if crashes.size:
        last = np.searchsorted(crashes, t, side="right") - 1
        since = np.where(
            last >= 0,
            t - crashes[np.maximum(last, 0)],
            t - (crashes[0] - V3_PERIOD_MS),
        )
        phase = np.clip(since / V3_PERIOD_MS, 0.0, 1.0)
    level = 0.3 + 2.7 * np.exp(-(((np.arange(n_channels) - centre) / 6.0) ** 2))
    shape = np.ones((n_channels, t.size))
    shape[slice(*core)] = 1.0 - drop + drop * phase
    for lo, hi in pulse:
        shape[lo:hi] = 1.0 + rise * (1.0 - phase)
    y = level[:, None] * shape
    y += noise * level[:, None] * np.random.default_rng(seed).normal(size=y.shape)
    return t * 1e-3, y.astype(np.float32)


def _v3(t_s, y, diag="ece", **kw):
    got = heuristics.sawtooth_crashes(
        y, t_s, shot=198658, diag=diag, t_cov=(float(t_s[0]), float(t_s[-1])),
        **kw,
    )
    return got


def _times_ms(crashes):
    return np.array([e.t0_s * 1e3 for e in crashes.events])


def _verdicts(t_s, y, pulse=True):
    env, _ = heuristics.envelope(y, t_s)
    steps, _, z = heuristics.step_z(env)
    return [
        heuristics.crash_test(env, steps, z, k, pulse=pulse)[0]
        for k in heuristics.crash_candidates(z)
    ]


def test_v3s_constants_are_the_briefs():
    h = heuristics
    assert (h.ENV_MS, h.CRASH_GAP_BINS, h.CRASH_SPAN_BINS) == (1.0, 1, 3)
    assert (h.NOISE_BINS, h.MAD_SIGMA, h.Z_DROP, h.Z_PULSE) == (401, 1.4826, 5, 3)
    assert (h.CRASH_BLOCK, h.CRASH_SEP_BINS, h.HOT_BINS, h.HOT_SLACK) == (3, 5, 10, 1)
    assert (h.MIN_FALL, h.COLLAPSE_FALL, h.FAST_FRAC) == (0.02, 0.5, 0.5)
    assert (h.PULSE_BLOCK, h.PULSE_REACH, h.PULSE_DIAG) == (3, 6, "ece")
    assert (h.MERGE_MS, h.COLLAPSE_GUARD_MS) == (5.0, 300.0)
    assert h.SXR_FANS == (
        ("SX90RM1F", 192), ("SX90RP1F", 256), ("SX90RM1S", 224),
        ("SX90RP1S", 288),
    )
    assert (h.SXR_CHORDS, h.SXR_MIN_CHORDS, h.SXR_LIT_FRAC) == (32, 8, 0.5)


def test_the_step_skips_the_crash_bin_and_is_nan_off_the_record():
    # A crash halfway through bin 30: the bin holds half of each level.
    env = np.tile(np.r_[np.zeros(30), 0.5, np.ones(29)], (2, 1))
    env[1, 35] = np.nan
    steps, sigma, z = heuristics.step_z(env)
    assert np.isnan(steps[:, :3]).all() and np.isnan(steps[:, -3:]).all()
    assert steps[0, 30] == 1.0, "bins 31-33 less 27-29: bin 30 is skipped"
    assert steps[0, 29] == pytest.approx(2.5 / 3) and steps[0, 31] == pytest.approx(
        2.5 / 3
    )
    nan = np.isnan(steps[1, 31:37])
    assert nan.tolist() == [False, True, True, True, False, True], (
        "a window with the NaN bin is no step; the step at bin 35 skips it"
    )
    # 7 of the 54 steps move: their median absolute deviation is 0.
    assert (sigma == 0).all() and (z == 0).all(), "no noise: sigma is 0 and z is 0"


def test_the_noise_is_not_underestimated_at_the_records_ends():
    # The prototype padded the steps with the edge value and counted a NaN
    # step as 0: the windows at each end were half zeros, sigma too small.
    t_s, y = v3_rows(crashes_ms=[])
    env, _ = heuristics.envelope(y, t_s)
    steps, sigma, z = heuristics.step_z(env)
    finite = np.isfinite(steps)
    assert (sigma[finite] > 0).all() and (sigma[~finite] == 0).all()
    ends = sigma[:, 3:200].mean() / sigma[:, 900:1100].mean()
    assert ends == pytest.approx(1.0, abs=0.1)
    assert heuristics.crash_candidates(z) == []


def test_every_crash_of_a_v3_train_is_found_within_a_bin():
    t_s, y = v3_rows()
    got = _v3(t_s, y)
    times = _times_ms(got)
    assert len(times) == len(V3_CRASHES_MS)
    assert np.abs(times - V3_CRASHES_MS).max() <= 1.0
    assert got.collapses_s == ()
    one = got.events[0]
    assert (one.source, one.phenomenon, one.diag) == ("ece_sawtooth", "sawtooth", "ece")
    assert one.t0_s == one.t1_s and one.channel == -1
    assert (one.t_cov0_s, one.t_cov1_s) == (t_s[0], t_s[-1])
    for e in got.events:
        a = e.attrs
        assert (a["inversion_channel_lo"], a["inversion_channel_stop"]) == V3_CORE
        lo, stop = a["pulse_channel_lo"], a["pulse_channel_stop"]
        # The median across channels lets a noisy neighbour lengthen a run.
        assert any(abs(lo - b) <= 1 and abs(stop - c) <= 1 for b, c in V3_PULSE)
        assert a["fall"] == pytest.approx(0.2, abs=0.02)
        assert a["z_min"] < -heuristics.Z_DROP
        # The 7 dropping and the pulse's channels of the 48 that could vote.
        assert e.confidence == pytest.approx((7 + stop - lo) / 48)
    pulses = [
        (e.attrs["pulse_channel_lo"], e.attrs["pulse_channel_stop"]) for e in got.events
    ]
    assert sum(run in V3_PULSE for run in pulses) >= 25


def test_the_same_train_time_reversed_is_no_crash():
    t_s, y = v3_rows()
    assert _v3(t_s, y[:, ::-1].copy()).events == ()
    # The reversed ramps and the reversed pulses are candidates; each fails.
    verdicts = set(_verdicts(t_s, y[:, ::-1].copy()))
    assert verdicts and heuristics.CRASH not in verdicts


def test_pure_noise_is_no_crash():
    t_s, y = v3_rows(crashes_ms=[], seed=3)
    got = _v3(t_s, y)
    assert got.events == () and got.collapses_s == ()


def test_a_smooth_fall_at_the_core_is_no_crash():
    t_s, y = v3_rows(crashes_ms=[], noise=0.002)
    t_ms = t_s * 1e3
    fall = 1.0 - 0.3 * np.clip((t_ms - 1000.0) / 100.0, 0.0, 1.0)  # 30 % in 100 ms
    y[slice(*V3_CORE)] *= fall.astype(np.float32)
    assert _v3(t_s, y).events == ()
    verdicts = _verdicts(t_s, y)
    assert verdicts, "the fall is a candidate: the test is not vacuous"
    assert set(verdicts) <= {"fast", "fall"}


def test_a_collapse_drops_the_train_in_the_300_ms_after_it():
    t_s, y = v3_rows()
    t_ms = t_s * 1e3
    # Everything falls by 90 % at 220.3 ms and recovers over 30 ms.
    quench = np.where(
        t_ms < 220.3, 1.0, 0.1 + 0.9 * np.clip((t_ms - 220.3) / 30.0, 0.0, 1.0)
    )
    y = y * quench.astype(np.float32)
    got = _v3(t_s, y)
    [collapse] = got.collapses_s
    assert collapse * 1e3 == pytest.approx(220.5, abs=1.0)
    assert len(got.events) == len(V3_CRASHES_MS), "each diagnostic keeps its own"
    kept = np.array([e.t0_s * 1e3 for e in heuristics.sawtooth_events_v3([got])])
    guarded = V3_CRASHES_MS <= 220.3 + heuristics.COLLAPSE_GUARD_MS
    assert guarded.sum() == 5
    assert kept == pytest.approx(V3_CRASHES_MS[~guarded] + 0.2, abs=1.0)


def test_a_dropping_block_away_from_the_hottest_channel_is_no_crash():
    t_s, y = v3_rows(core=(31, 38), pulse=((27, 31), (38, 42)))
    assert _v3(t_s, y).events == ()
    assert set(_verdicts(t_s, y)) == {"core"}


def test_an_ece_crash_needs_a_heat_pulse_and_an_sxr_crash_does_not():
    t_s, y = v3_rows(rise=0.0)
    assert _v3(t_s, y).events == ()
    assert set(_verdicts(t_s, y)) == {"pulse"}
    got = _v3(t_s, y, diag="sxr", channels=np.arange(48) + 100, attrs={"fan": "F"})
    times = _times_ms(got)
    assert len(times) == len(V3_CRASHES_MS)
    assert np.abs(times - V3_CRASHES_MS).max() <= 1.0
    one = got.events[0]
    assert one.diag == "sxr" and "pulse_channel_lo" not in one.attrs
    assert one.attrs["fan"] == "F"
    assert (one.attrs["inversion_channel_lo"], one.attrs["inversion_channel_stop"]) == (
        117,
        124,
    ), "in the channels' own indices, end-exclusive"


def test_v3_reads_the_ece_array_or_an_sxr_fan():
    t_s, y = v3_rows()
    with pytest.raises(ValueError, match="48"):
        _v3(t_s, y[:24])
    with pytest.raises(ValueError, match="v3 reads"):
        _v3(t_s, y, diag="co2")
    with pytest.raises(ValueError, match="channel indices"):
        _v3(t_s, y, channels=[1, 2])
    # SXR has no pulse test, so any number of chords: 0-23 hold the core.
    assert len(_v3(t_s, y[:24], diag="sxr", channels=range(24)).events) == 30


def _crash(diag, t_ms):
    return schema.Event(
        shot=1, source="ece_sawtooth", evidence_kind="heuristic",
        phenomenon="sawtooth", t0_s=t_ms * 1e-3, t1_s=t_ms * 1e-3,
        confidence=0.5, diag=diag, channel=-1,
    )


def _found(diag, times_ms, collapses_ms=()):
    return heuristics.Crashes(
        diag, tuple(_crash(diag, t) for t in times_ms),
        tuple(t * 1e-3 for t in collapses_ms),
    )


def test_the_union_merges_crashes_within_5_ms_and_keeps_the_earlier():
    ece = _found("ece", [100.5, 200.5, 300.5, 400.5, 407.5])
    sxr = _found("sxr", [104.5, 206.5, 305.5, 403.5])
    got = heuristics.sawtooth_events_v3([sxr, ece])
    assert [(e.diag, round(e.t0_s * 1e3, 1)) for e in got] == [
        ("ece", 100.5),  # 104.5 is 4 ms after it
        ("ece", 200.5),
        ("sxr", 206.5),  # 6 ms: another crash
        ("ece", 300.5),  # 305.5 is 5.0 ms after it: within
        ("ece", 400.5),  # 403.5 merges into it; 407.5 is 7 ms after
        ("ece", 407.5),
    ]


def test_the_guard_drops_crashes_from_a_collapse_to_300_ms_after_it():
    ece = _found("ece", [590.0, 600.0, 750.0, 900.0, 910.0])
    sxr = _found("sxr", [], collapses_ms=[600.0])
    got = heuristics.sawtooth_events_v3([ece, sxr])
    assert [round(e.t0_s * 1e3, 1) for e in got] == [590.0, 910.0], "900 is in it"
    assert heuristics.sawtooth_events_v3([]) == []


# ------------------------------------------------------------- L->H and H->L

def _lh(shot_data, **kw):
    fields = ("ne_t_s", "ne_y", "betan_t_s", "betan_y", "pinj_t_s", "pinj_y")
    kwargs = {name: shot_data[name] for name in fields}
    kwargs.update(kw)
    return heuristics.lh_transitions(
        shot_data["dalpha_t_s"], shot_data["dalpha_y"],
        shot=198658, t_cov=shot_data["t_cov"], **kwargs,
    )


def test_the_drawn_transition_is_found_where_it_was_drawn(synth_shot):
    got = [e for e in _lh(synth_shot) if e.phenomenon == "lh_transition"]
    assert len(got) == 1
    one = got[0]
    assert one.t0_s == one.t1_s == pytest.approx(SYNTH_LH_S, abs=1e-3)
    assert one.source == "dalpha_lh"
    assert one.evidence_kind == "heuristic"
    assert one.diag == "filterscopes"
    assert (one.t_cov0_s, one.t_cov1_s) == synth_shot["t_cov"]
    # The drop is 45% of a 60% full-confidence drop.
    measured = 1.0 - SYNTH_DALPHA_H / SYNTH_DALPHA_L
    assert one.attrs["drop_frac"] == pytest.approx(measured, abs=0.01)
    assert one.confidence == pytest.approx(measured / 0.6, abs=0.02)
    assert one.attrs["pinj_kw"] == pytest.approx(SYNTH_PINJ_KW)
    assert one.attrs["betan"] == pytest.approx(1.0, abs=0.05)
    assert json.loads(json.dumps(one.attrs, allow_nan=False)) == dict(one.attrs)


def test_a_transition_just_after_the_beam_record_ends_is_kept_and_clipped(
    synth_shot,
):
    # The beam gate reads `[when - 5 ms, when]`, so a transition up to 5 ms
    # after the NBI digitiser's last finite sample clears it with every
    # input measured - and `when` then lies past the INTERSECTION coverage
    # the pipeline hands this detector (task Lfix-C1, deliverable 2). The
    # schema would refuse that row, and `finish_shot` isolates per step, so
    # one 5 ms coincidence would cost the shot every L-H claim it has. The
    # point is moved onto the bound instead, says so, and keeps the
    # measured instant.
    t = synth_shot["pinj_t_s"]
    keep = t <= SYNTH_LH_S - 0.002
    pinj_t, pinj_y = t[keep], synth_shot["pinj_y"][keep]
    cov = coverage.intersect([
        coverage.finite_span(synth_shot["dalpha_t_s"], synth_shot["dalpha_y"]),
        coverage.finite_span(synth_shot["ne_t_s"], synth_shot["ne_y"]),
        coverage.finite_span(pinj_t, pinj_y),
    ])
    assert cov[1] == pytest.approx(float(pinj_t[-1])) and cov[1] < SYNTH_LH_S
    got = heuristics.lh_transitions(
        synth_shot["dalpha_t_s"], synth_shot["dalpha_y"],
        ne_t_s=synth_shot["ne_t_s"], ne_y=synth_shot["ne_y"],
        betan_t_s=synth_shot["betan_t_s"], betan_y=synth_shot["betan_y"],
        pinj_t_s=pinj_t, pinj_y=pinj_y, shot=198658, t_cov=cov,
    )
    lh = [e for e in got if e.phenomenon == "lh_transition"]
    assert len(lh) == 1
    one = lh[0]
    assert one.t0_s == one.t1_s == cov[1] == one.t_cov1_s
    assert one.attrs["clipped"] is True
    assert one.attrs["t_measured_s"] == pytest.approx(SYNTH_LH_S, abs=1e-3)
    # The back transition at 0.60 s has no beam record under its gate at
    # all, and is (rightly) not claimed rather than clipped.
    assert not [e for e in got if e.phenomenon == "hl_transition"]


def test_a_transition_inside_its_coverage_is_not_marked_clipped(synth_shot):
    one = next(e for e in _lh(synth_shot) if e.phenomenon == "lh_transition")
    assert "clipped" not in one.attrs and "t_measured_s" not in one.attrs


def test_the_reverse_step_is_a_back_transition(synth_shot):
    got = [e for e in _lh(synth_shot) if e.phenomenon == "hl_transition"]
    assert len(got) == 1
    assert got[0].t0_s == pytest.approx(SYNTH_HL_S, abs=1e-3)
    assert got[0].source == "dalpha_lh"


def test_without_beams_there_is_no_transition(synth_shot):
    off = np.zeros_like(synth_shot["pinj_y"])
    assert _lh(synth_shot, pinj_y=off) == []
    assert _lh(synth_shot, pinj_t_s=None, pinj_y=None) == []


def test_a_drop_with_no_density_rise_is_not_a_transition(synth_shot):
    flat = np.full_like(synth_shot["ne_y"], 2.0)
    got = _lh(synth_shot, ne_y=flat)
    assert [e.phenomenon for e in got] == []


def test_betan_is_recorded_and_decides_nothing(synth_shot):
    got = _lh(synth_shot, betan_t_s=None, betan_y=None)
    lh = [e for e in got if e.phenomenon == "lh_transition"]
    assert len(lh) == 1
    assert lh[0].attrs["betan"] is None


def test_an_elmy_dalpha_trace_is_not_a_train_of_transitions(synth_shot):
    # The finding that shaped the detector: on shot 185946, comparing a
    # sample with the one 5 ms later called 17,989 of 80,001 samples a 30%
    # drop, because that is exactly what the fall from every ELM spike back
    # to the inter-ELM baseline is. The rolling median either side sees the
    # LEVEL, and the level does not move.
    t = synth_shot["dalpha_t_s"]
    elmy = np.tile(synth_shot["dalpha_y"][:, :1], (1, t.size)).astype(np.float64)
    elmy[:, ::100] *= 6.0                      # an ELM every 10 ms
    got = _lh({**synth_shot, "dalpha_y": elmy})
    assert got == []
    # The same spikes on top of the real step leave the step where it was.
    on_top = synth_shot["dalpha_y"].astype(np.float64).copy()
    on_top[:, ::100] *= 6.0
    lh = [e for e in _lh({**synth_shot, "dalpha_y": on_top})
          if e.phenomenon == "lh_transition"]
    assert len(lh) == 1
    assert lh[0].t0_s == pytest.approx(SYNTH_LH_S, abs=2e-3)


def test_realistic_elm_bursts_are_not_transitions(synth_shot):
    # A type-I ELM's D-alpha burst is MILLISECONDS wide, not the tenth of
    # one the spike test above uses, and a 4 ms burst is most of a 5 ms
    # median window: the level itself steps up and back down at every ELM,
    # so the drop gate fires once per ELM. What separates a transition from
    # an ELM is that the transition STAYS: the level 20-50 ms after it is
    # still down, and after an ELM it is exactly where it was before.
    t = synth_shot["dalpha_t_s"]
    fs = 1.0 / float(t[1] - t[0])
    burst = np.zeros(t.size, dtype=bool)
    width = round(0.004 * fs)
    for start in range(0, t.size, round(0.015 * fs)):
        burst[start:start + width] = True
    elmy = np.tile(synth_shot["dalpha_y"][:, :1], (1, t.size)).astype(np.float64)
    elmy[:, burst] *= 6.0
    assert _lh({**synth_shot, "dalpha_y": elmy}) == []


def test_a_transition_followed_by_an_elm_train_is_one_transition(synth_shot):
    # The H-mode the transition enters is ELMy 60 ms later, which is the
    # ordinary case: one transition, and then a burst train that is not a
    # train of transitions.
    t = synth_shot["dalpha_t_s"]
    fs = 1.0 / float(t[1] - t[0])
    y = synth_shot["dalpha_y"].astype(np.float64).copy()
    width = round(0.004 * fs)
    first = int(np.searchsorted(t, SYNTH_LH_S + 0.060))
    last = int(np.searchsorted(t, SYNTH_HL_S))
    for start in range(first, last, round(0.015 * fs)):
        y[:, start:start + width] *= 6.0
    got = [e for e in _lh({**synth_shot, "dalpha_y": y})
           if e.phenomenon == "lh_transition"]
    assert len(got) == 1
    assert got[0].t0_s == pytest.approx(SYNTH_LH_S, abs=2e-3)


# ----------------------------------------- the hold gate's acceptance matrix

# The hold gate compares the D-alpha level 20-50 ms AFTER a step with the
# level 20-50 ms before it, and what these five cells settle is whether that
# comparison may be made against the pre window's own TREND rather than its
# median. A trend cancels a drifting baseline out of the ratio - cells (a)
# and (b) - but the pre window of a transition whose fall begins before the
# detected edge is not a baseline at all, and carrying it forward predicts
# the rest of the fall, so the transition explains itself away - cells (c)
# and (e). Measured here, `*` marking a wrong answer:
#
#   cell                          stationary    linear trend   geometric
#   (a) drift + ELMs, no step     4* 4* 0 0     0 0 0 0        0 0 0 0
#   (b) drift + step              1 1 1 1       0* 1 1 1       1 1 1 1
#   (c) lead-in + step            1 1 1 1 1     1 1 0* 0* 0*   1 1 0* 0* 1
#   (d) ELMs, no step             0             0              0
#   (e) H->L lead-in + step       1 1           1 0*           1 0*
#
# The stationary gate's wrong answers are a TRAIN OF FALSE transitions,
# which is noisy and which a consumer can see; both trend gates' are a
# genuine transition reported as nothing at all. So the trend gate was
# withdrawn (task L7-fix) and the stationary one stands, with cell (a) as a
# documented limitation rather than a fixed bug.

#: The lead-in cells below draw a transition whose fall STARTS before the
#: edge the detector finds: the level slides linearly from `SYNTH_DALPHA_L`
#: to `LEAD_MID_LH` over the lead-in and then steps the rest of the way to
#: `SYNTH_DALPHA_H`. A third of the fall before the edge and two thirds at
#: it, so the step is still 35% - comfortably over `LH_DROP_FRAC`, which is
#: what makes it a candidate at all - while the lead-in is steep enough to
#: reach into the `[t - 50 ms, t - 20 ms]` window the hold is measured over.
#: A dithering or slow L->H does exactly this, and it is not exotic.
LEAD_MID_LH = 0.85
#: The H->L mirror: the level climbs a third of the way back and then steps
#: the rest, a 43% rise at the edge.
LEAD_MID_HL = 0.70

#: Half-lives of the decaying baseline, in ms, and lead-ins in ms.
MATRIX_HALF_LIVES = (70.0, 100.0, 150.0, 300.0)
MATRIX_LEADS = (0.0, 30.0, 40.0, 50.0, 70.0)


def _dalpha(synth_shot, level):
    """The fixture's eight channels, carrying the 1-D trace `level`."""
    first = synth_shot["dalpha_y"][:, :1].astype(np.float64)
    return first * (np.asarray(level, dtype=np.float64) / SYNTH_DALPHA_L)


def _flat_level(t):
    return np.full(t.shape, SYNTH_DALPHA_L)


def _step_level(t):
    """L, then H over `[SYNTH_LH_S, SYNTH_HL_S)`, then L again."""
    return np.where(
        (t >= SYNTH_LH_S) & (t < SYNTH_HL_S), SYNTH_DALPHA_H, SYNTH_DALPHA_L
    )


def _decayed(level, t, half_ms):
    """`level` under a baseline decaying with the given half-life."""
    return level * np.exp(-t * math.log(2.0) / (half_ms * 1e-3))


def _elm_train(level, t, *, width_ms=4.0, period_ms=15.0):
    """`level` with a 4 ms D-alpha burst every 15 ms on top of it."""
    fs = 1.0 / float(t[1] - t[0])
    out = level.copy()
    width = round(width_ms * 1e-3 * fs)
    for start in range(0, out.size, round(period_ms * 1e-3 * fs)):
        out[start:start + width] *= 6.0
    return out


def _with_lead_in(level, t, *, edge, lead_ms, mid):
    """`level` with a linear lead-in of `lead_ms` ending at `edge`."""
    if lead_ms <= 0.0:
        return level
    out = level.copy()
    lead_s = lead_ms * 1e-3
    m = (t >= edge - lead_s) & (t < edge)
    out[m] = out[m] + (mid - out[m]) * (t[m] - (edge - lead_s)) / lead_s
    return out


def _n_lh(synth_shot, level, phenomenon="lh_transition"):
    got = _lh({**synth_shot, "dalpha_y": _dalpha(synth_shot, level)})
    return len([e for e in got if e.phenomenon == phenomenon])


#: Cell (a)'s measured answer: the number of false `lh_transition`s a
#: falling baseline under an ELM train produces, per half-life. Four is one
#: per ELM whose fall also clears the density gate, in 0.8 s.
MATRIX_A_FALSE = (4, 4, 0, 0)


@pytest.mark.parametrize("half_ms, n_false",
                         list(zip(MATRIX_HALF_LIVES, MATRIX_A_FALSE)))
def test_matrix_a_a_falling_baseline_under_an_elm_train_is_false_transitions(
    synth_shot, half_ms, n_false,
):
    # THE KNOWN LIMITATION, pinned rather than fixed. The hold assumes the
    # inter-ELM baseline is stationary over its +/-50 ms; a baseline that is
    # itself falling is genuinely lower after every ELM than before it, so
    # every ELM's fall passes the hold and the detector reports one
    # transition per ELM although no step was drawn. Below a ~30% fall per
    # 70 ms the drift no longer clears `LH_HOLD_FRAC` and the cell is clean.
    # Both fixes tried made cells (c) and (e) silently miss a real
    # transition, which is worse; see the table above and `LH_HOLD_FRAC`.
    t = synth_shot["dalpha_t_s"]
    drifting = _decayed(_flat_level(t), t, half_ms)
    # The drift ALONE is no transition at any half-life: it is smooth, so
    # the 5 ms drop gate never fires. It takes the ELMs to make candidates.
    assert _n_lh(synth_shot, drifting) == 0
    assert _n_lh(synth_shot, _elm_train(drifting, t)) == n_false


@pytest.mark.parametrize("half_ms", MATRIX_HALF_LIVES)
def test_matrix_b_a_transition_under_a_falling_baseline_is_still_found(
    synth_shot, half_ms,
):
    # The other half of (a): whatever the gate does about a drift, the
    # transition drawn UNDER that drift is still one transition.
    t = synth_shot["dalpha_t_s"]
    assert _n_lh(synth_shot, _decayed(_step_level(t), t, half_ms)) == 1


@pytest.mark.parametrize("lead_ms", MATRIX_LEADS)
def test_matrix_c_a_transition_whose_fall_reaches_the_pre_window_is_found(
    synth_shot, lead_ms,
):
    # The transition takes `lead_ms` to fall, so its own fall is INSIDE the
    # window the hold is measured against. A gate that reads that window as
    # a baseline and carries it forward predicts the rest of the fall, and
    # the transition explains itself away - a SILENT miss, which is the one
    # thing this detector may not do.
    t = synth_shot["dalpha_t_s"]
    level = _with_lead_in(_step_level(t), t, edge=SYNTH_LH_S,
                          lead_ms=lead_ms, mid=LEAD_MID_LH)
    assert _n_lh(synth_shot, level) == 1


def test_matrix_d_a_stationary_baseline_under_an_elm_train_is_no_transition(
    synth_shot,
):
    # The cell the whole hold gate exists for, and the one that is not in
    # doubt: 4 ms bursts every 15 ms on a level that does not move.
    t = synth_shot["dalpha_t_s"]
    assert _n_lh(synth_shot, _elm_train(_flat_level(t), t)) == 0
    assert _n_lh(synth_shot, _elm_train(_flat_level(t), t),
                 "hl_transition") == 0


@pytest.mark.parametrize("lead_ms", [0.0, 40.0])
def test_matrix_e_a_back_transition_with_a_lead_in_is_found(
    synth_shot, lead_ms,
):
    # (c)'s mirror: the level climbs before the step up as often as it
    # falls before the step down, and the same gate has to survive it.
    t = synth_shot["dalpha_t_s"]
    level = _with_lead_in(_step_level(t), t, edge=SYNTH_HL_S,
                          lead_ms=lead_ms, mid=LEAD_MID_HL)
    assert _n_lh(synth_shot, level, "hl_transition") == 1


def test_a_transition_taken_in_two_stages_is_one_transition(synth_shot):
    # Measured on real shots: a transition routinely qualifies in two or
    # three bursts a couple of milliseconds apart, and reporting each is
    # saying there were three transitions.
    t = synth_shot["dalpha_t_s"]
    staged = np.where(
        t < SYNTH_LH_S, SYNTH_DALPHA_L,
        np.where(t < SYNTH_LH_S + 0.002, 0.75, SYNTH_DALPHA_H),
    )
    staged = np.where(t >= SYNTH_HL_S, SYNTH_DALPHA_L, staged)
    y = np.tile(staged, (SYNTH_DALPHA_CHANNELS, 1))
    got = [e for e in _lh({**synth_shot, "dalpha_y": y})
           if e.phenomenon == "lh_transition"]
    assert len(got) == 1
    assert got[0].t0_s == pytest.approx(SYNTH_LH_S, abs=3e-3)


def test_dalpha_channels_that_are_dark_are_stepped_over(synth_shot):
    # Filterscope records begin and end in NaN, and a whole channel is
    # routinely dead. Neither may reach the median as a warning or as a
    # number - under `-W error` the first would fail here.
    y = synth_shot["dalpha_y"].copy()
    y[3] = np.nan
    y[:, :10] = np.nan
    y[:, -10:] = np.nan
    got = [e for e in _lh({**synth_shot, "dalpha_y": y})
           if e.phenomenon == "lh_transition"]
    assert len(got) == 1
    assert got[0].t0_s == pytest.approx(SYNTH_LH_S, abs=1e-3)


def test_only_the_eight_real_filterscope_channels_are_accepted(synth_shot):
    # Channels 8 and up are NaN on every shot (plan V4); they are the
    # caller's to drop, and passing them is a mistake this refuses to make
    # quietly.
    wide = np.full((104, synth_shot["dalpha_y"].shape[1]), np.nan, np.float32)
    wide[:SYNTH_DALPHA_CHANNELS] = synth_shot["dalpha_y"]
    with pytest.raises(ValueError, match="8"):
        _lh({**synth_shot, "dalpha_y": wide})


# ---------------------------------------------------------------- actuators

def _steps(levels, *, fs_hz=1.0e3):
    """A `(t_s, y)` trace holding `levels`, one sample per millisecond."""
    y = np.asarray(levels, dtype=np.float64)
    return np.arange(y.size, dtype=np.float64) / fs_hz, y


def test_the_beams_and_their_counter_torque_become_intervals(synth_shot):
    got = heuristics.actuator_intervals(
        {
            "pinj_total": (synth_shot["pinj_t_s"], synth_shot["pinj_y"]),
            "tinj_total": (synth_shot["tinj_t_s"], synth_shot["tinj_y"]),
            "ip": (synth_shot["ip_t_s"], synth_shot["ip_y"]),
        },
        shot=198658, t_cov=synth_shot["t_cov"],
    )
    nbi = [e for e in got if e.phenomenon == "nbi_on"]
    counter = [e for e in got if e.phenomenon == "nbi_counter"]
    assert len(nbi) == len(counter) == 1
    assert (nbi[0].t0_s, nbi[0].t1_s) == pytest.approx(SYNTH_PINJ_ON_S, abs=2e-3)
    assert (counter[0].t0_s, counter[0].t1_s) == pytest.approx(
        SYNTH_COUNTER_S, abs=2e-3
    )
    assert nbi[0].source == "actuator"
    assert nbi[0].evidence_kind == "heuristic"
    assert nbi[0].attrs == {
        "mean_level": pytest.approx(SYNTH_PINJ_KW),
        "max_level": pytest.approx(SYNTH_PINJ_KW),
        "units": "kW",
    }
    assert counter[0].attrs["units"] == "N m"
    assert counter[0].attrs["max_level"] == pytest.approx(SYNTH_TINJ_NM)


def test_without_ip_there_is_no_counter_injection_claim(synth_shot):
    got = heuristics.actuator_intervals(
        {
            "pinj_total": (synth_shot["pinj_t_s"], synth_shot["pinj_y"]),
            "tinj_total": (synth_shot["tinj_t_s"], synth_shot["tinj_y"]),
        },
        shot=1, t_cov=synth_shot["t_cov"],
    )
    assert [e.phenomenon for e in got] == ["nbi_on"]


def test_a_level_dipping_between_half_and_full_does_not_toggle():
    # 2:1 hysteresis: on at 500 kW, off at 250. A beam notching to 400 for 50
    # ms is one interval, not three.
    level = np.concatenate([
        np.zeros(100), np.full(100, 800.0), np.full(50, 400.0),
        np.full(100, 800.0), np.zeros(100),
    ])
    got = heuristics.actuator_intervals(
        {"pinj_total": _steps(level)}, shot=1, t_cov=(0.0, 0.449)
    )
    assert len(got) == 1
    assert (got[0].t0_s, got[0].t1_s) == pytest.approx((0.100, 0.349))
    assert got[0].attrs["max_level"] == pytest.approx(800.0)
    assert got[0].attrs["mean_level"] == pytest.approx(
        (800 * 200 + 400 * 50) / 250
    )


def test_a_gap_shorter_than_the_minimum_is_bridged():
    # Off for 10 ms is a notch; off for 30 ms is two pulses.
    short = np.concatenate([
        np.zeros(50), np.full(100, 800.0), np.zeros(10),
        np.full(100, 800.0), np.zeros(50),
    ])
    long = np.concatenate([
        np.zeros(50), np.full(100, 800.0), np.zeros(30),
        np.full(100, 800.0), np.zeros(50),
    ])
    bridged = heuristics.actuator_intervals(
        {"pinj_total": _steps(short)}, shot=1, t_cov=(0.0, 0.309)
    )
    assert len(bridged) == 1
    assert (bridged[0].t0_s, bridged[0].t1_s) == pytest.approx((0.050, 0.259))
    split = heuristics.actuator_intervals(
        {"pinj_total": _steps(long)}, shot=1, t_cov=(0.0, 0.329)
    )
    assert len(split) == 2


def test_an_interval_shorter_than_the_minimum_is_dropped():
    blip = np.concatenate([np.zeros(50), np.full(15, 800.0), np.zeros(50)])
    held = np.concatenate([np.zeros(50), np.full(25, 800.0), np.zeros(50)])
    assert heuristics.actuator_intervals(
        {"pinj_total": _steps(blip)}, shot=1, t_cov=(0.0, 0.114)
    ) == []
    assert len(heuristics.actuator_intervals(
        {"pinj_total": _steps(held)}, shot=1, t_cov=(0.0, 0.124)
    )) == 1


def test_the_gyrotrons_are_on_at_a_hundred_kilowatts_of_watts():
    # `ech_power_total` is WATTS in `features/namespace.py` while
    # `pinj_total` is kilowatts, and 1e5 of one is a hundred times 1e5 of
    # the other: a threshold read in the wrong unit is the whole event list.
    below = np.full(200, 0.9e5)
    above = np.concatenate([np.zeros(50), np.full(100, 1.0e5), np.zeros(50)])
    assert heuristics.actuator_intervals(
        {"ech_power_total": _steps(below)}, shot=1, t_cov=(0.0, 0.199)
    ) == []
    got = heuristics.actuator_intervals(
        {"ech_power_total": _steps(above)}, shot=1, t_cov=(0.0, 0.199)
    )
    assert [e.phenomenon for e in got] == ["ech_on"]
    assert got[0].attrs["units"] == "W"
    assert got[0].attrs["max_level"] == pytest.approx(1.0e5)
    # Hysteresis is 2:1 here too: half of 1e5 is still on.
    notched = above.copy()
    notched[80:100] = 0.6e5
    assert len(heuristics.actuator_intervals(
        {"ech_power_total": _steps(notched)}, shot=1, t_cov=(0.0, 0.199)
    )) == 1


def test_counter_injection_is_not_claimed_where_ip_and_pinj_are_not_measured():
    # `ip` and `pinj_total` come off other digitisers than the torque and
    # routinely stop earlier. Interpolation returns NaN there, `_schmitt`
    # HOLDS the last state across a NaN, and holding it is how a beam that
    # stopped being measured goes on injecting counter-current torque to the
    # end of the record.
    t_tinj = np.arange(600, dtype=np.float64) / 1.0e3
    tinj = np.full(t_tinj.size, -5.0)
    t_short = t_tinj[:200]
    got = heuristics.actuator_intervals(
        {
            "tinj_total": (t_tinj, tinj),
            "pinj_total": (t_short, np.full(t_short.size, 3000.0)),
            "ip": (t_short, np.full(t_short.size, 1.0e6)),
        },
        shot=1, t_cov=(0.0, 0.599),
    )
    counter = [e for e in got if e.phenomenon == "nbi_counter"]
    assert len(counter) == 1
    assert counter[0].t1_s == pytest.approx(float(t_short[-1]), abs=2e-3)


def test_an_interval_and_a_gap_exactly_at_the_boundary():
    # Measured edge to edge - every sample owns half a step either side - so
    # 20 samples at 1 kHz is exactly 20 ms and survives, 19 is 19 ms and does
    # not; and a gap is bridged only when it is STRICTLY under 20 ms, so 19
    # off-samples are a notch and 20 are the end of the interval.
    def run(*lengths, level=800.0):
        parts, on = [], True
        for n in lengths:
            parts.append(np.full(n, level if on else 0.0))
            on = not on
        return heuristics.actuator_intervals(
            {"pinj_total": _steps(np.concatenate([np.zeros(50), *parts,
                                                  np.zeros(50)]))},
            shot=1, t_cov=(0.0, 1.0),
        )

    assert len(run(20)) == 1
    assert run(19) == []
    assert len(run(30, 19, 30)) == 1
    assert len(run(30, 20, 30)) == 2


def test_a_multi_coil_actuator_is_on_when_any_one_of_them_is():
    t = np.arange(200, dtype=np.float64) / 1.0e3
    coils = np.zeros((12, t.size))
    coils[7, 50:150] = -0.8            # kA, and the sign is not the question
    gas = np.zeros((11, t.size))
    gas[3, 60:160] = 0.9               # V
    got = heuristics.actuator_intervals(
        {"rmp": (t, coils), "gas": (t, gas)}, shot=1, t_cov=(0.0, 0.199)
    )
    by = {e.phenomenon: e for e in got}
    assert set(by) == {"rmp_on", "gas_on"}
    assert (by["rmp_on"].t0_s, by["rmp_on"].t1_s) == pytest.approx((0.050, 0.149))
    assert by["rmp_on"].attrs["max_level"] == pytest.approx(0.8)
    assert by["rmp_on"].attrs["units"] == "kA"
    assert (by["gas_on"].t0_s, by["gas_on"].t1_s) == pytest.approx((0.060, 0.159))


def test_a_dead_coil_decides_nothing_and_warns_about_nothing():
    # Real coil and valve arrays carry dead channels and NaN ends. Under
    # `-W error` a bare `nanmax` over an all-NaN column would fail this test
    # rather than merely muddy a log, which is the point of asserting it.
    t = np.arange(200, dtype=np.float64) / 1.0e3
    coils = np.full((12, t.size), np.nan)
    coils[7] = 0.0
    coils[7, 50:150] = -0.8
    got = heuristics.actuator_intervals(
        {"rmp": (t, coils)}, shot=1, t_cov=(0.0, 0.199)
    )
    assert len(got) == 1
    assert (got[0].t0_s, got[0].t1_s) == pytest.approx((0.050, 0.149))
    assert got[0].attrs["max_level"] == pytest.approx(0.8)
    # A column with nothing finite in it holds the state rather than ending it.
    dead = np.full((12, t.size), np.nan)
    assert heuristics.actuator_intervals(
        {"rmp": (t, dead)}, shot=1, t_cov=(0.0, 0.199)
    ) == []


def test_a_scalar_feature_given_as_an_array_is_refused():
    t = np.arange(200, dtype=np.float64) / 1.0e3
    beams = np.full((8, t.size), 400.0)      # eight beams, not their total
    with pytest.raises(ValueError, match="pinj_total"):
        heuristics.actuator_intervals(
            {"pinj_total": (t, beams)}, shot=1, t_cov=(0.0, 0.199)
        )


def test_an_actuator_nobody_passed_is_no_events():
    assert heuristics.actuator_intervals({}, shot=1, t_cov=(0.0, 1.0)) == []


# --------------------------------------------------------------- the QH proxy

def _track(t0, t1, *, f0=6.0, f1=10.0, chirp=0.0, conf=0.9):
    """One EHO-shaped `Track`; the defaults pass every gate."""
    return Track(
        t0_s=t0, t1_s=t1, f0_khz=f0, f1_khz=f1,
        f_centroid_khz=0.5 * (f0 + f1), chirp_khz_per_ms=chirp, chirp_r2=0.9,
        bandwidth_khz=f1 - f0, duration_ms=(t1 - t0) * 1e3, duty=1.0,
        mean_prob=0.8, conf=conf, n_pix=5000, n_components=1,
        row_lit_fraction=0.1, pickup=False,
    )


def test_the_proxy_is_where_all_four_conditions_hold_at_once(synth_shot):
    got = heuristics.qh_candidates(
        [_track(1.0, 2.0)],
        elm_free=np.array([[1.2, 3.0]]),
        nbi=np.array([[0.0, 1.8]]),
        ip_flattop=np.array([[1.3, 5.0]]),
        shot=1, t_cov=(0.0, 5.0),
    )
    assert len(got) == 1
    one = got[0]
    assert (one.t0_s, one.t1_s) == pytest.approx((1.3, 1.8))
    assert one.source == "qh_proxy"
    assert one.phenomenon == "qh"
    assert one.evidence_kind == "heuristic"
    assert (one.f0_khz, one.f1_khz) == pytest.approx((6.0, 10.0))
    assert "definitional" in one.attrs["note"]
    assert "Appendix C item 8" in one.attrs["note"]
    assert json.loads(json.dumps(one.attrs, allow_nan=False)) == dict(one.attrs)


def test_the_confidence_is_the_weaker_of_the_track_and_the_quiet():
    # The track runs 1.0-2.0 s and is ELM-free over 1.2-2.0, so 80% of it is
    # quiet - less than the track's own 0.9, and the claim is worth the
    # weaker of the two.
    got = heuristics.qh_candidates(
        [_track(1.0, 2.0, conf=0.9)],
        elm_free=np.array([[1.2, 2.5]]),
        nbi=np.array([[0.0, 5.0]]),
        ip_flattop=np.array([[0.0, 5.0]]),
        shot=1, t_cov=(0.0, 5.0),
    )
    assert got[0].confidence == pytest.approx(0.8)
    assert got[0].attrs["elm_free_fraction"] == pytest.approx(0.8)
    quiet = heuristics.qh_candidates(
        [_track(1.0, 2.0, conf=0.5)],
        elm_free=np.array([[0.0, 5.0]]),
        nbi=np.array([[0.0, 5.0]]),
        ip_flattop=np.array([[0.0, 5.0]]),
        shot=1, t_cov=(0.0, 5.0),
    )
    assert quiet[0].confidence == pytest.approx(0.5)


def test_a_track_that_is_not_an_eho_is_not_a_qh_candidate():
    whole = np.array([[0.0, 5.0]])
    every = {"elm_free": whole, "nbi": whole, "ip_flattop": whole,
             "shot": 1, "t_cov": (0.0, 5.0)}
    assert heuristics.qh_candidates([], **every) == []
    # Out of band, too short, and chirping too hard.
    assert heuristics.qh_candidates(
        [_track(1.0, 2.0, f0=40.0, f1=60.0)], **every) == []
    assert heuristics.qh_candidates([_track(1.0, 1.05)], **every) == []
    assert heuristics.qh_candidates([_track(1.0, 2.0, chirp=-0.5)], **every) == []


def test_an_eho_in_an_elmy_phase_is_not_a_qh():
    whole = np.array([[0.0, 5.0]])
    assert heuristics.qh_candidates(
        [_track(1.0, 2.0)], elm_free=np.zeros((0, 2)), nbi=whole,
        ip_flattop=whole, shot=1, t_cov=(0.0, 5.0),
    ) == []
    assert heuristics.qh_candidates(
        [_track(1.0, 2.0)], elm_free=whole, nbi=np.zeros((0, 2)),
        ip_flattop=whole, shot=1, t_cov=(0.0, 5.0),
    ) == []


# ------------------------------------------------- the Ip flat-top and q-min

def _ip_trace(step_s: float = 0.02):
    """A current record with two flat stretches, the second one longer.

    Ramp up to 1 MA over 0.5 s, hold to 1.5 s, notch to half current for
    0.1 s, hold again to 3.0 s, then a disruption-fast drop to zero. The
    longest run above `FLATTOP_FRAC` of the peak is therefore 1.6-3.0 s and
    not 0.5-1.5 s, which is what `ip_flattop` has to pick.
    """
    t = (np.arange(round(3.5 / step_s) + 1) * step_s).round(10)
    y = np.zeros_like(t)
    y = np.where(t < 0.5, 2.0e6 * t, y)
    y = np.where((t >= 0.5) & (t <= 1.5), 1.0e6, y)
    y = np.where((t > 1.5) & (t < 1.6), 5.0e5, y)
    y = np.where((t >= 1.6) & (t <= 3.0), 1.0e6, y)
    return t, y


def test_the_flattop_is_the_longest_run_above_nine_tenths_of_the_peak():
    t, y = _ip_trace()
    lo, hi = heuristics.ip_flattop(t, y)
    assert (lo, hi) == pytest.approx((1.6, 3.0))


def test_the_flattop_is_the_currents_magnitude_and_not_its_sign():
    t, y = _ip_trace()
    assert heuristics.ip_flattop(t, -y) == pytest.approx(
        heuristics.ip_flattop(t, y)
    )


def test_a_current_record_with_nothing_finite_has_no_flattop():
    t = np.arange(0.0, 1.0, 0.02)
    assert heuristics.ip_flattop(t, np.full_like(t, np.nan)) == coverage.UNKNOWN
    assert heuristics.ip_flattop(
        np.zeros(0), np.zeros(0)
    ) == coverage.UNKNOWN


def test_the_flattop_takes_one_channel_and_refuses_an_array():
    t, y = _ip_trace()
    assert heuristics.ip_flattop(t, y[None, :]) == pytest.approx(
        heuristics.ip_flattop(t, y)
    )
    with pytest.raises(ValueError, match="one trace"):
        heuristics.ip_flattop(t, np.tile(y, (2, 1)))


def _qmin_trace(step_s: float = 0.02, *, t1: float = 6.0):
    """A 20 ms q-min axis, EFIT01's own cadence, over `[0.1, t1]`.

    Rounded, so that a painted window's own bounds ARE sample times: a
    boundary test measured against a float-drifted axis would be testing
    `np.arange`.
    """
    return (0.1 + np.arange(round((t1 - 0.1) / step_s) + 1) * step_s).round(10)


def _painted(t, pieces, fill=0.8):
    """`[(t0, t1, value)]` painted onto `t`, `fill` everywhere else."""
    y = np.full(t.size, float(fill))
    for lo, hi, value in pieces:
        y[(t >= lo) & (t <= hi)] = float(value)
    return y


def test_a_shot_stepping_through_the_three_regimes_gives_three_intervals():
    t = _qmin_trace()
    # 1.2 hybrid, 1.7 elevated, 3.0 high - each a second long, inside the
    # flat-top - and a fourth second at 3.0 OUTSIDE it, which is the
    # ramp-down the gate exists to refuse.
    y = _painted(t, [(1.0, 2.0, 1.2), (2.5, 3.5, 1.7), (4.0, 5.0, 3.0),
                   (5.5, 6.0, 3.0)])
    got = heuristics.qmin_regimes(t, y, (0.5, 5.2), shot=7)
    assert [e.phenomenon for e in got] == [
        "qmin_hybrid", "qmin_elevated", "qmin_high",
    ]
    assert [(round(e.t0_s, 3), round(e.t1_s, 3)) for e in got] == [
        (1.0, 2.0), (2.5, 3.5), (4.0, 5.0),
    ]
    assert all(e.source == "qmin_rule" for e in got)
    assert all(e.evidence_kind == "heuristic" for e in got)
    assert all(math.isnan(e.confidence) for e in got)


def test_nothing_outside_the_flattop_is_claimed():
    t = _qmin_trace()
    y = _painted(t, [(1.0, 3.0, 3.0)])
    assert heuristics.qmin_regimes(t, y, (4.0, 5.0), shot=7) == []


def test_a_band_shorter_than_the_minimum_is_no_event():
    t = _qmin_trace()
    # 400 ms of hybrid q-min, inside the flat-top and a hundred milliseconds
    # short of the rule's own minimum.
    y = _painted(t, [(1.0, 1.4, 1.2)])
    assert heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7) == []


def test_a_band_exactly_at_the_minimum_is_kept():
    """The documented side of the duration test: `>=`, not `>`.

    Measured on the 500 `recommender_v1` shots, five of them own a band
    that is exactly 25 sampling intervals long, so this boundary is worth
    three shots of the hybrid count and two of the high count.
    """
    t = _qmin_trace()
    y = _painted(t, [(1.0, 1.5, 1.2)])
    got = heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7)
    assert len(got) == 1
    assert (got[0].t1_s - got[0].t0_s) * 1e3 == pytest.approx(
        heuristics.QMIN_MIN_MS
    )


def test_a_q_min_exactly_on_a_threshold_falls_in_the_lower_band():
    """Bands are `lo < q <= hi`, so a threshold belongs to the band below."""
    t = _qmin_trace()
    for value, phenomenon in ((1.5, "qmin_hybrid"), (2.0, "qmin_elevated")):
        y = _painted(t, [(1.0, 2.0, value)])
        got = heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7)
        assert [e.phenomenon for e in got] == [phenomenon]
    # And the bottom of the lowest band is open: 0.95 itself is no regime.
    y = _painted(t, [(1.0, 2.0, heuristics.QMIN_HYBRID)])
    assert heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7) == []


def test_the_bands_are_exclusive_and_never_overlap():
    t = _qmin_trace()
    y = _painted(t, [(1.0, 2.0, 1.2), (2.0, 3.0, 1.7), (3.0, 4.0, 3.0)])
    got = heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7)
    assert len(got) == 3
    for a, b in itertools.pairwise(got):
        assert a.t1_s <= b.t0_s


def test_nan_samples_split_a_band_and_leave_the_coverage():
    t = _qmin_trace()
    y = _painted(t, [(1.0, 3.0, 1.2)])
    # A 60 ms dropout in the middle of what would be a 2 s hybrid band, and
    # a NaN head and tail: two 900-odd ms bands, and a coverage that starts
    # at the first finite sample rather than at the first time.
    y[(t > 1.9) & (t < 2.0)] = np.nan
    y[t < 0.5] = np.nan
    y[t > 5.5] = np.nan
    got = heuristics.qmin_regimes(t, y, (0.0, 6.0), shot=7)
    assert [e.phenomenon for e in got] == ["qmin_hybrid", "qmin_hybrid"]
    assert (got[0].t1_s, got[1].t0_s) == pytest.approx((1.9, 2.0))
    assert got[0].t_cov0_s == pytest.approx(0.5)
    assert got[0].t_cov1_s == pytest.approx(5.5)
    # A band made of nothing but NaN is no band, however long it is.
    assert heuristics.qmin_regimes(t, np.full_like(t, np.nan), (0.0, 6.0),
                                  shot=7) == []


def test_the_coverage_is_the_flattop_met_with_the_finite_q_min_record():
    t = _qmin_trace()
    y = _painted(t, [(1.0, 2.0, 1.2)])
    got = heuristics.qmin_regimes(t, y, (0.5, 3.0), shot=7)
    assert (got[0].t_cov0_s, got[0].t_cov1_s) == pytest.approx((0.5, 3.0))


def test_the_rule_says_which_equilibrium_and_which_thresholds_it_used():
    t = _qmin_trace()
    y = _painted(t, [(1.0, 2.0, 1.2)])
    one = heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7)[0]
    assert one.attrs["efit"] == "efit01"
    assert one.attrs["qmin_min"] == pytest.approx(1.2)
    assert one.attrs["qmin_max"] == pytest.approx(1.2)
    assert one.attrs["thresholds"] == {
        "lo": heuristics.QMIN_HYBRID, "hi": heuristics.QMIN_ELEVATED,
        "min_ms": heuristics.QMIN_MIN_MS,
    }
    assert one.diag == "qmin"
    assert json.loads(json.dumps(one.attrs, allow_nan=False)) == dict(one.attrs)
    # The open-topped band says so with a null rather than an infinity JSON
    # has no word for.
    y = _painted(t, [(1.0, 2.0, 3.0)])
    top = heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7)[0]
    assert top.attrs["thresholds"]["hi"] is None
    assert json.dumps(top.attrs, allow_nan=False)


def test_the_extremes_are_the_bands_own_and_not_the_records():
    t = _qmin_trace()
    y = _painted(t, [(1.0, 1.4, 1.1), (1.4, 2.0, 1.45)], fill=5.0)
    one = heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7)[0]
    assert one.attrs["qmin_min"] == pytest.approx(1.1)
    assert one.attrs["qmin_max"] == pytest.approx(1.45)


def test_without_a_flattop_there_is_no_regime_claim():
    t = _qmin_trace()
    y = _painted(t, [(1.0, 3.0, 1.2)])
    assert heuristics.qmin_regimes(t, y, coverage.UNKNOWN, shot=7) == []
    assert heuristics.qmin_regimes(t, y, None, shot=7) == []


def test_a_q_min_record_and_its_axis_have_to_agree():
    t = _qmin_trace()
    with pytest.raises(ValueError, match="samples"):
        heuristics.qmin_regimes(t, np.zeros(t.size + 1), (0.5, 5.0), shot=7)


def test_a_regime_row_is_one_the_events_table_accepts(tmp_path):
    t = _qmin_trace()
    y = _painted(t, [(1.0, 2.0, 1.2)])
    got = heuristics.qmin_regimes(t, y, (0.5, 5.0), shot=7)
    path = tmp_path / "7_events.parquet"
    schema.write_events(path, 7, got, run_id="test")
    df = schema.read_events(path)
    assert list(df["phenomenon"]) == ["qmin_hybrid"]
    assert list(df["source"]) == ["qmin_rule"]
    assert df["confidence"].isna().all()


# ------------------------------- what the rule does on the real distribution

#: The census `scripts/labeler/qmin_regime_census.py` wrote, from the
#: real features store over the 500 shots of `recommender_v1`. Committed
#: rather than recomputed, because the store is not in this repository and
#: the suite is hermetic - and read on every run rather than quoted in a
#: comment, because the gate it justifies is the whole of the rule.
QMIN_CENSUS = Path(__file__).with_name("data") / "qmin_regimes_recommender_v1.json"


@pytest.fixture(scope="module")
def qmin_census():
    return json.loads(QMIN_CENSUS.read_text())


def test_the_census_was_measured_with_the_thresholds_this_module_still_has(
    qmin_census,
):
    """The record and the code have to agree, or the record is a fossil.

    Every number below is a measurement of a particular set of thresholds.
    Moving one without re-running the census would leave the file claiming
    a rule that no longer exists, so the thresholds travel WITH the
    numbers and are checked against the module's here.
    """
    thresholds = qmin_census["thresholds"]
    assert thresholds["bands"] == [
        [name, lo, hi] for name, lo, hi in heuristics.QMIN_BANDS
    ]
    assert thresholds["min_ms"] == heuristics.QMIN_MIN_MS
    assert thresholds["flattop_frac"] == heuristics.FLATTOP_FRAC
    assert thresholds["efit"] == heuristics.QMIN_EFIT


def test_without_the_flattop_gate_the_rule_fires_on_over_99_percent(
    qmin_census,
):
    """MEASURED: the gate is not optional, and this is the number.

    `qmin > 0.95` for 500 ms ANYWHERE in the record - the shipped rule with
    the flat-top gate taken out - fires on 497 of the 500 `recommender_v1`
    shots. Every discharge ramps its current up through every q band and
    back down again, so an ungated q-min threshold is a label that says
    "this is a tokamak discharge". The gated rule claims 271.
    """
    n = qmin_census["n_shots"]
    assert n == 500
    assert qmin_census["ungated"]["any_band"] == 497
    assert qmin_census["ungated"]["any_band"] / n > 0.99
    assert qmin_census["gated"]["qmin_hybrid"] / n < 0.60


def test_the_regime_counts_on_the_five_hundred_are_pinned(qmin_census):
    """hybrid 271 / elevated 60 / high 50, exactly.

    The rule is deterministic, so this is an exact pin and not a
    tolerance: a change to the bands, to the duration test's `>=`, or to
    how the flat-top is found moves these, and they are the numbers the
    task's own exit criterion names.
    """
    assert qmin_census["gated"] == {
        "qmin_hybrid": 271, "qmin_elevated": 60, "qmin_high": 50,
    }
    assert qmin_census["n_skipped"] == {
        "features": 0, "ip": 0, "qmin": 0, "flattop": 0,
    }


def test_the_census_totals_are_its_own_per_shot_rows(qmin_census):
    """The file cannot be edited to say something its rows do not.

    Without this the totals above would be three numbers anybody could
    retype; with it they are a reduction of 500 per-shot answers.
    """
    per_shot = qmin_census["per_shot"]
    assert len(per_shot) == qmin_census["n_shots"]
    for band, n in qmin_census["gated"].items():
        assert sum(band in found for found in per_shot.values()) == n
    # And the bands are exclusive per SHOT only in the sense that a shot may
    # hold several: every name in the file is one the rule can write.
    names = {band for found in per_shot.values() for band in found}
    assert names <= {name for name, _lo, _hi in heuristics.QMIN_BANDS}
