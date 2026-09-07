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

from ideate.design import actuators as act
from ideate.schema import ActuationSet, ActuatorWaveform, Vertex
from ideate.shotdb.corpus import CorpusReader

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
    assert small.extrapolation_warnings == ()
    big = act.apply(_nbi_total(ref, 4.0), ref.raw, ref.stats)[1]
    assert big.extrapolation_warnings, "|z| > 3 on an edited channel must be reported"
    assert any("pinj" in w for w in big.extrapolation_warnings)
    assert aset.waveforms["nbi.total"].key == "nbi.total"


def test_apply_rejects_an_unknown_policy(act_corpus):
    ref = act.build_actuators(SHOT, act_corpus, N_FRAMES)
    with pytest.raises(ValueError, match="policy"):
        act.apply(_nbi_total(ref, 1.2), ref.raw, ref.stats, policy="nonsense")
