"""CorpusReader over the FAITH corpus layout, on synthetic files written under tmp_path.

The corpus layout is not ours and it has three traps this file pins, because every one of them
was found by measurement on the real corpus and every one of them is silent if unhandled:

* the timebase is SECONDS while the rest of shot_design is milliseconds;
* an absent diagnostic is written as a `(C, 1)` placeholder, not as a missing group -- and the
  time axis is the LAST one, so `shape[0] > 1` (what the old shotsearch manifest tested) counts
  a placeholder as present;
* the 500 kHz groups end in a trailing all-channel NaN sample that no consumer expects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shot_design import config
from shot_design.shotdb.corpus import CorpusReader
from shot_design.shotdb.reader import Reader, ShotFailed, Unavailable

from .conftest import CORPUS_ABSENT as MISSING
from .conftest import CORPUS_FULL as SHOT
from .conftest import CORPUS_TRUNCATED as TRUNCATED


@pytest.fixture
def reader(corpus_dir: Path) -> CorpusReader:
    return CorpusReader(corpus_dir)


# ------------------------------------------------------------------------------- the file layer


def test_path_names_the_corpus_file_whether_or_not_it_exists(reader, corpus_dir):
    assert reader.path(SHOT) == corpus_dir / f"{SHOT}_processed.h5"
    assert reader.path(MISSING) == corpus_dir / f"{MISSING}_processed.h5"


def test_available_is_false_for_a_missing_and_for_an_unreadable_file(reader):
    assert reader.available(SHOT) is True
    assert reader.available(MISSING) is False
    assert reader.available(TRUNCATED) is False


def test_groups_lists_the_present_groups_only(reader):
    assert reader.groups(SHOT) == ["gas_flow", "mhr", "pinj", "tangtv"]  # co2 is a placeholder


def test_a_truncated_file_fails_the_shot_and_chains_the_oserror(reader):
    for call in (
        lambda: reader.groups(TRUNCATED),
        lambda: reader.read(TRUNCATED, "pinj"),
        lambda: reader.coverage(TRUNCATED, "pinj"),
    ):
        with pytest.raises(ShotFailed) as e:
            call()
        assert str(e.value) and isinstance(e.value.__cause__, OSError)


def test_a_missing_file_fails_the_shot_too(reader):
    with pytest.raises(ShotFailed):
        reader.read(MISSING, "pinj")


# ------------------------------------------------------------------------------------ read


def test_read_converts_seconds_to_milliseconds(reader):
    t_ms, y = reader.read(SHOT, "pinj")
    assert t_ms.dtype == np.float64 and y.dtype == np.float32
    assert y.shape == (8, 5)
    np.testing.assert_allclose(t_ms, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(y[0], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_read_returns_the_channels_asked_for_in_the_order_asked(reader):
    _, all_y = reader.read(SHOT, "pinj")
    _, y = reader.read(SHOT, "pinj", channels=[3, 1])
    np.testing.assert_array_equal(y, all_y[[3, 1]])


def test_read_strips_the_trailing_pad_sample_and_keeps_the_time_axis_aligned(reader):
    t_ms, y = reader.read(SHOT, "mhr")
    assert y.shape == (2, 8) and t_ms.shape == (8,)
    assert np.isfinite(y).all()
    np.testing.assert_allclose(t_ms, np.arange(8) * 2.0e-6 * 1000.0, rtol=1e-6)


def test_the_strip_is_over_all_returned_channels_not_any_one_of_them(reader):
    # gas_flow's channel 1 recorded nothing; the last sample is NaN on every channel. Read
    # together, only the pad goes; read alone, a channel that recorded nothing keeps its timebase
    # rather than being truncated to nothing -- "recorded nothing" is a coverage fact, not padding.
    _, y = reader.read(SHOT, "gas_flow")
    assert y.shape == (3, 5)
    t_ms, dead = reader.read(SHOT, "gas_flow", channels=[1])
    assert dead.shape == (1, 6) and t_ms.shape == (6,) and not np.isfinite(dead).any()


def test_a_placeholder_group_is_unavailable_not_an_empty_read(reader):
    with pytest.raises(Unavailable):
        reader.read(SHOT, "co2")
    with pytest.raises(Unavailable):
        reader.coverage(SHOT, "co2")


def test_a_group_this_shot_does_not_carry_is_unavailable(reader):
    with pytest.raises(Unavailable):
        reader.read(SHOT, "ece")


def test_read_refuses_a_video_group(reader):
    with pytest.raises(ValueError, match="video"):
        reader.read(SHOT, "tangtv")


# --------------------------------------------------------------------------------- coverage


def test_coverage_is_milliseconds_from_xdata_alone(reader):
    t0, t1 = reader.coverage(SHOT, "mhr")
    # The pad sample's timestamp is finite and is what xdata's last entry says: coverage is the
    # recorded span of the diagnostic, not of the samples read() hands back.
    assert (t0, t1) == pytest.approx((0.0, 8 * 2.0e-6 * 1000.0))
    assert reader.coverage(SHOT, "tangtv") == pytest.approx((0.0, 40.0))


# ------------------------------------------------------------------------------- the protocol


def test_corpus_reader_satisfies_the_reader_protocol(reader):
    assert isinstance(reader, Reader)


# ------------------------------------------------------------- the actuator group mapping


def test_actuators_yaml_maps_the_eight_corpus_actuator_groups(reader):
    got = config.corpus_actuators()
    assert {k: v.group for k, v in got.items()} == {
        "nbi": "pinj",
        "nbi_voltage": "beam_voltage",
        "nbi_torque": "tinj",
        "ech": "ech_power",
        "gas": "gas_flow",
        "gas_raw": "gas_raw",
        "rmp": "rmp",
        "icoil": "i_coil",
    }
    assert all(v.channels is None for v in got.values())  # "all"
