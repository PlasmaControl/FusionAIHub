"""The 88-channel IGNITE actuator contract: layout, per-frame means, z-scoring, and edits.

The fixture below is a corpus file whose actuator groups are PIECEWISE CONSTANT on the frame
grid, so every expected number in this module is exact arithmetic rather than a tolerance: a
50 ms frame either sits wholly in the low level or wholly in the high one. That is what makes
`frame_means` testable at all -- a real actuator trace would only ever support "close to".
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from shot_design.design import actuators as act
from shot_design.schema import ActuationSet, ActuatorWaveform, Vertex
from shot_design.shotdb.corpus import CorpusReader

SHOT = 990088
#: 10 kHz over [-0.5, 2.0] s -- 25001 samples, so `fs = (n - 1) / span` is exactly 10000 Hz and
#: shot time 0.0 s is sample 5000. The level step is at sample 15000, i.e. shot time 1.0 s, which
#: is the start of frame 20 on the 50 ms grid.
N_SAMPLES = 25001
STEP_SAMPLE = 15000
N_FRAMES = 30
LOW, HIGH = 100.0, 200.0


def _write(path: Path, group: str, x_s: np.ndarray, y: np.ndarray) -> None:
    with h5py.File(path, "a") as f:
        g = f.create_group(group)
        g.create_dataset("xdata", data=np.asarray(x_s, dtype=np.float32))
        g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))


def _step(n_channels: int, low: float = LOW, high: float = HIGH) -> np.ndarray:
    """Channel i steps from (i + 1) * low to (i + 1) * high at `STEP_SAMPLE`."""
    scale = (np.arange(n_channels, dtype=np.float64) + 1.0)[:, None]
    y = np.full((n_channels, N_SAMPLES), low, dtype=np.float64)
    y[:, STEP_SAMPLE:] = high
    return y * scale


@pytest.fixture
def act_corpus(tmp_path: Path) -> CorpusReader:
    """A corpus shot carrying three of the eight actuator groups.

    * `pinj` steps -- the group every arithmetic assertion is written against.
    * `ech_power` is constant, so its per-shot std is 0 and its z must be exactly 0.
    * `gas_raw` steps as well but has a NaN burst over the first half of frame 1, which
      production averages as ZERO rather than skipping it (`np.nan_to_num` before the mean).
    * the remaining five groups are absent, which `frame_means` must report as NaN rows.
    """
    d = tmp_path / "corpus"
    d.mkdir()
    p = d / f"{SHOT}_processed.h5"
    x = np.linspace(-0.5, 2.0, N_SAMPLES)
    _write(p, "pinj", x, _step(8))
    _write(p, "ech_power", x, np.full((12, N_SAMPLES), 3.0))
    gas = _step(11)
    gas[:, 5500:5750] = np.nan  # frame 1 is samples 5500..6000: half of it recorded nothing
    _write(p, "gas_raw", x, gas)
    return CorpusReader(d)


PAD_SHOT = 990089
#: 10 kHz over [0.0, 0.22] s -- 2201 samples, so `fs = (n - 1) / span` is exactly 10 kHz and a
#: 50 ms frame is exactly 500 samples. The record therefore ends 200 samples into frame 4, and
#: the LAST of those 2201 samples is the corpus's all-NaN pad, which `CorpusReader.read` strips.
PAD_N = 2201
PAD_LEVEL = 100.0
PAD_FRAMES = 5


@pytest.fixture
def pad_corpus(tmp_path: Path) -> CorpusReader:
    """A shot whose record ends MID-FRAME behind a trailing all-NaN pad sample.

    This is the shape of every real actuator group in the corpus (rmp, gas_flow, i_coil and
    beam_voltage each carry a one-sample pad on 190090/202537/204346) and it is the one case the
    step fixture above cannot reach: its NaN burst is interior, so `CorpusReader.read` returns
    the full array and `_record_length` is never asked for anything.
    """
    d = tmp_path / "pad"
    d.mkdir()
    p = d / f"{PAD_SHOT}_processed.h5"
    y = np.full((8, PAD_N), PAD_LEVEL, dtype=np.float64)
    y[:, -1] = np.nan  # the pad: production averaged it in as a zero, the reader drops it
    _write(p, "pinj", np.arange(PAD_N) / 10000.0, y)
    return CorpusReader(d)


SPIKE_SHOT = 990090


@pytest.fixture
def spike_corpus(tmp_path: Path) -> CorpusReader:
    """A reference shot that is ALREADY outside the training envelope before any edit.

    Real reference shots are: the shipped caches carry 22/88 channels at |z| > 3 on 190090
    (max 9.88), 2/88 on 202537 and 26/88 on 204346. `rmp[0]` here is at full current for exactly
    one of the thirty frames and at zero for the other twenty-nine, which puts its peak at
    sqrt(29) = 5.39 z -- the same shape, in a fixture whose arithmetic is exact. `pinj` and
    `ech_power` step as in `act_corpus`, so an edit can be aimed at either of them and `rmp` is
    addressed by no waveform key at all (actuators.yaml gives it no `system:`).
    """
    d = tmp_path / "spike"
    d.mkdir()
    p = d / f"{SPIKE_SHOT}_processed.h5"
    x = np.linspace(-0.5, 2.0, N_SAMPLES)
    _write(p, "pinj", x, _step(8))
    _write(p, "ech_power", x, _step(12))
    rmp = np.zeros((12, N_SAMPLES), dtype=np.float64)
    rmp[0, 12500:13000] = 1000.0  # frame 15 only (sample 5000 is shot time 0.0)
    _write(p, "rmp", x, rmp)
    return CorpusReader(d)


# ------------------------------------------------------------------------------- the layout


def test_act_spec_offsets_are_the_prefix_sums_and_the_total_is_88():
    assert [g.group for g in act.ACT_SPEC] == [
        "ech_power",
        "pinj",
        "beam_voltage",
        "tinj",
        "gas_flow",
        "gas_raw",
        "rmp",
        "i_coil",
    ]
    assert [g.n_channels for g in act.ACT_SPEC] == [12, 8, 8, 8, 11, 11, 12, 18]
    assert [g.offset for g in act.ACT_SPEC] == [0, 12, 20, 28, 36, 47, 58, 70]
    assert act.N_CHANNELS == 88
    assert sum(g.n_channels for g in act.ACT_SPEC) == act.N_CHANNELS


def test_channel_index_addresses_a_group_member_and_rejects_a_bad_one():
    assert act.channel_index("ech_power", 0) == 0
    assert act.channel_index("pinj", 0) == 12
    assert act.channel_index("i_coil", 17) == 87
    with pytest.raises(IndexError):
        act.channel_index("pinj", 8)
    with pytest.raises(KeyError):
        act.channel_index("no_such_group", 0)


def test_channel_names_are_88_long_and_name_the_group_and_member():
    names = act.channel_names()
    assert len(names) == 88
    assert names[0] == "ech_power[0]"
    assert names[12] == "pinj[0]"
    assert names[87] == "i_coil[17]"


# ------------------------------------------------------------------------------- frame_means


def test_frame_means_are_the_exact_level_of_each_frame(act_corpus):
    raw = act.frame_means(SHOT, act_corpus, n_frames=N_FRAMES)
    assert raw.shape == (88, N_FRAMES)
    assert raw.dtype == np.float32
    for ch in range(8):
        row = raw[act.channel_index("pinj", ch)]
        expected_low = (ch + 1) * LOW
        expected_high = (ch + 1) * HIGH
        np.testing.assert_allclose(row[:20], expected_low, rtol=0, atol=0)
        np.testing.assert_allclose(row[20:], expected_high, rtol=0, atol=0)


def test_frame_means_leave_an_absent_group_all_nan(act_corpus):
    raw = act.frame_means(SHOT, act_corpus, n_frames=N_FRAMES)
    for group in ("beam_voltage", "tinj", "gas_flow", "rmp", "i_coil"):
        spec = act.group_spec(group)
        block = raw[spec.offset : spec.offset + spec.n_channels]
        assert np.isnan(block).all(), f"{group} is absent and must be NaN, not 0"
    # ... while a present group is finite everywhere.
    assert np.isfinite(raw[act.channel_index("pinj", 0)]).all()


def test_frame_means_average_a_nan_sample_as_zero(act_corpus):
    """Production's `np.nan_to_num(seg).mean()`: the NaN samples are counted, valued 0."""
    raw = act.frame_means(SHOT, act_corpus, n_frames=N_FRAMES)
    row = raw[act.channel_index("gas_raw", 0)]
    # frame 1 covers samples 5500..6000; 250 of its 500 samples are NaN and count as 0.
    assert row[1] == pytest.approx(LOW * 250 / 500)
    assert row[0] == pytest.approx(LOW)


def test_frame_means_past_the_record_are_zero_not_an_error(act_corpus):
    """A frame whose window starts past the last sample has nothing to average: production
    emits zeros there rather than raising or wrapping to the array tail."""
    raw = act.frame_means(SHOT, act_corpus, n_frames=60)
    row = raw[act.channel_index("pinj", 0)]
    assert row[59] == 0.0


def test_frame_means_divide_the_straddling_frame_by_the_untrimmed_window(pad_corpus):
    """The trailing-pad quirk, pinned to exact arithmetic.

    Frame 4 of `pad_corpus` covers samples 2000..2201 of a 2201-sample record whose last sample
    is the corpus's all-NaN pad. Production divided that frame by the full 201-sample window
    (the pad counted as a zero); `CorpusReader.read` hands us only the 2200 finite samples, so
    dividing by what survives would give 100.0 instead. 200 samples of 100.0 over a 201-sample
    window is 100 * 200 / 201 = 99.502487... exactly, and getting this wrong is worth ~0.04 z on
    all twelve `rmp` channels of a real shot -- the single measurement that turns 77/88 actuator
    channels bit-identical to the shipped cache into 88/88.
    """
    raw = act.frame_means(PAD_SHOT, pad_corpus, n_frames=PAD_FRAMES, dtype=np.float64)
    row = raw[act.channel_index("pinj", 0)]
    np.testing.assert_allclose(row[:4], PAD_LEVEL, rtol=0, atol=0)
    assert row[4] == pytest.approx(PAD_LEVEL * 200 / 201, rel=1e-12)
    assert row[4] != pytest.approx(PAD_LEVEL, rel=1e-6), "the pad must not be dropped"


def test_record_length_recovers_the_untrimmed_sample_count(pad_corpus, act_corpus):
    """`coverage()` spans the pad, `read()` does not: the two together give the real length."""
    t_ms, _ = pad_corpus.read(PAD_SHOT, "pinj")
    assert t_ms.size == PAD_N - 1, "the reader is expected to strip the trailing all-NaN sample"
    t0_ms, t1_ms = pad_corpus.coverage(PAD_SHOT, "pinj")
    assert act._record_length(t_ms, t1_ms - t0_ms) == PAD_N

    # ... and a group with no pad is returned whole, so the untrimmed length is what was read.
    t_ms, _ = act_corpus.read(SHOT, "pinj")
    lo, hi = act_corpus.coverage(SHOT, "pinj")
    assert t_ms.size == N_SAMPLES
    assert act._record_length(t_ms, hi - lo) == N_SAMPLES


# ------------------------------------------------------------------------------- z_score


def test_z_score_uses_population_std_ddof_zero():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    z, (mean, std) = act.z_score(x)
    assert mean[0] == pytest.approx(2.5)
    assert std[0] == pytest.approx(np.sqrt(1.25))  # ddof 0; ddof 1 would be sqrt(5/3)
    np.testing.assert_allclose(z.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(z.std(axis=1), 1.0, atol=1e-5)


def test_z_score_of_a_constant_channel_is_exactly_zero():
    x = np.full((1, 5), 7.0, dtype=np.float32)
    z, (_, std) = act.z_score(x)
    assert std[0] == 0.0
    assert (z == 0.0).all()


def test_z_score_with_given_stats_does_not_recompute_them():
    x = np.array([[0.0, 10.0]], dtype=np.float32)
    stats = (np.array([0.0]), np.array([1.0]))
    z, got = act.z_score(x, stats=stats)
    np.testing.assert_allclose(z, [[0.0, 10.0]], atol=1e-5)
    assert got is stats


# ------------------------------------------------------------------------------- build + apply


def _nbi_total(reference: act.Actuators, factor: float) -> ActuationSet:
    """An `nbi.total` waveform equal to `factor` x the reference shot's own pinj total."""
    t_s = reference.frame_times
    total = reference.raw[act.channel_index("pinj", 0) : act.channel_index("pinj", 0) + 8].sum(0)
    return ActuationSet(
        source_shot=SHOT,
        waveforms={
            "nbi.total": ActuatorWaveform(
                key="nbi.total",
                vertices=[
                    Vertex(t_s=float(t), y=float(y * factor))
                    for t, y in zip(t_s, total, strict=True)
                ],
            )
        },
    )


def test_build_actuators_returns_88_z_scored_channels_and_its_own_stats(act_corpus):
    ref = act.build_actuators(SHOT, act_corpus, N_FRAMES)
    assert ref.raw.shape == ref.z.shape == (88, N_FRAMES)
    assert np.isfinite(ref.z).all(), "an absent group must reach z as 0, never NaN"
    # an absent group is all zeros in raw AND in z, exactly as production zero-fills it
    spec = act.group_spec("i_coil")
    assert (ref.raw[spec.offset : spec.offset + spec.n_channels] == 0.0).all()
    assert (ref.z[spec.offset : spec.offset + spec.n_channels] == 0.0).all()
    assert "i_coil" in ref.missing and "pinj" not in ref.missing


def test_reference_stats_policy_moves_the_edited_channels(act_corpus):
    """The V9 fact this whole contract exists for: a +20 % NBI edit is visible in z ONLY when
    the edited traces are re-z-scored with the REFERENCE shot's statistics."""
    ref = act.build_actuators(SHOT, act_corpus, N_FRAMES)
    aset = _nbi_total(ref, 1.2)
    z, report = act.apply(aset, ref.raw, ref.stats, policy="reference_stats")
    shift = np.abs(z - ref.z).max(axis=1)
    assert int((shift >= 0.1).sum()) >= 8
    assert len(report.edited_channels) == 8
    assert report.edited_channels[0] == "pinj[0]"
    assert min(report.z_shift.values()) >= 0.1


def test_self_stats_policy_is_the_no_op_trap(act_corpus):
    """Re-z-scoring the EDITED traces with their own statistics cancels the edit: a pure scale
    of a channel leaves its z trace where it was. Kept only to demonstrate the trap."""
    ref = act.build_actuators(SHOT, act_corpus, N_FRAMES)
    aset = _nbi_total(ref, 1.2)
    z, _ = act.apply(aset, ref.raw, ref.stats, policy="self_stats")
    assert np.abs(z - ref.z).max() <= 0.01


def test_apply_warns_when_the_edit_leaves_the_training_envelope(act_corpus):
    ref = act.build_actuators(SHOT, act_corpus, N_FRAMES)
    small = act.apply(aset := _nbi_total(ref, 1.2), ref.raw, ref.stats)[1]
    assert small.edited_extrapolated == ()
    big = act.apply(_nbi_total(ref, 4.0), ref.raw, ref.stats)[1]
    assert big.edited_extrapolated, "|z| > 3 on an edited channel must be reported"
    assert all("pinj" in w for w in big.edited_extrapolated)
    assert aset.waveforms["nbi.total"].key == "nbi.total"


def test_apply_does_not_blame_the_edit_for_the_reference_shot_s_own_excursions(spike_corpus):
    """V10 is about the EDIT. A reference channel that was already past |z| = 3 before anyone
    touched it is reported separately and informationally, not as a warning about the edit --
    otherwise the one warning that matters arrives twenty-second in a list of twenty-two."""
    ref = act.build_actuators(SPIKE_SHOT, spike_corpus, N_FRAMES)
    assert np.abs(ref.z[act.channel_index("rmp", 0)]).max() == pytest.approx(np.sqrt(29.0))

    _, report = act.apply(_nbi_total(ref, 1.2), ref.raw, ref.stats)
    assert report.edited_channels == tuple(f"pinj[{i}]" for i in range(8))
    assert report.edited_extrapolated == (), "the edit moved nothing past |z| = 3"
    assert len(report.reference_extrapolated) == 1
    assert "rmp[0]" in report.reference_extrapolated[0]


def test_apply_warns_about_the_edited_channel_and_only_that_one(spike_corpus):
    """The other half: a single-member edit that does leave the envelope produces exactly one
    edit warning, with the pre-existing `rmp[0]` excursion still in its own list."""
    ref = act.build_actuators(SPIKE_SHOT, spike_corpus, N_FRAMES)
    aset = ActuationSet(
        source_shot=SPIKE_SHOT,
        waveforms={
            "ech.LUKE": ActuatorWaveform(
                key="ech.LUKE",
                vertices=[Vertex(t_s=0.0, y=4000.0), Vertex(t_s=2.0, y=4000.0)],
            )
        },
    )
    _, report = act.apply(aset, ref.raw, ref.stats)
    assert report.edited_channels == ("ech_power[7]",)
    assert len(report.edited_extrapolated) == 1
    assert "ech_power[7]" in report.edited_extrapolated[0]
    assert not any("rmp" in w for w in report.edited_extrapolated)
    assert len(report.reference_extrapolated) == 1
    assert "rmp[0]" in report.reference_extrapolated[0]


def test_apply_rejects_an_unknown_policy(act_corpus):
    ref = act.build_actuators(SHOT, act_corpus, N_FRAMES)
    with pytest.raises(ValueError, match="policy"):
        act.apply(_nbi_total(ref, 1.2), ref.raw, ref.stats, policy="nonsense")
