"""Which corpus channels a shot's mask run may use, and over what time.

Two questions, both answered from the file's headers alone. `plan_for` says
which of the eleven round-1 channels this shot can actually run - a corpus
group is ABSENT when its `ydata` last axis is shorter than 2 (a `(C, 1)`
placeholder), and `mirnov` stands in for `mhr` only when `mhr` is one of
those. `coverage` says over what interval the group has data at all, which
is what lets a later "no EHO on this shot" be told apart from "no magnetics
on this shot".

The fixtures here are synthetic files in the corpus layout - group,
`xdata (n,)` seconds, `ydata (C, n)` - and tiny: nothing here needs a real
2-million-sample record, because nothing here reads a value.
"""
from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import h5py
import numpy as np
import pytest

from labeler.events import channels

#: Channel counts of the real corpus groups, so a fixture that says
#: "everything is here" is the shape of a shot that really has everything.
REAL_WIDTHS = {"mhr": 8, "ece": 48, "co2": 4, "bes": 64, "mirnov": 29}


def _write(path, groups, *, n=16):
    """A corpus-layout file: `{diag: n_channels or None}`; None means absent.

    "Absent" is written the way the corpus writes it - the group is there,
    with a `(C, 1)` placeholder - because that is the case a header-only
    reader has to get right.
    """
    with h5py.File(path, "w") as f:
        for diag, width in groups.items():
            g = f.create_group(diag)
            cols = 1 if width is None else n
            channels_ = REAL_WIDTHS[diag] if width is None else width
            g.create_dataset(
                "ydata", data=np.zeros((channels_, cols), dtype=np.float32)
            )
            g.create_dataset(
                "xdata",
                data=np.linspace(0.0, 1.0, cols, dtype=np.float32),
            )
    return path


@pytest.fixture
def complete(tmp_path):
    """A shot with every round-1 group, at its real channel count."""
    return _write(tmp_path / "199999_processed.h5", dict(REAL_WIDTHS))


def test_the_round1_plan_is_the_eleven_channels_the_pilot_chose():
    got = [(s.diag, s.channel, s.role, s.fallback_for) for s in channels.ROUND1_PLAN]
    assert got == [
        ("mhr", 0, "magnetics", ""),
        ("mhr", 4, "magnetics", ""),
        ("ece", 8, "ece_core", ""),
        ("ece", 20, "ece_mid", ""),
        ("ece", 40, "ece_edge", ""),
        ("co2", 0, "density", ""),
        ("co2", 2, "density", ""),
        ("bes", 26, "bes", ""),
        ("bes", 28, "bes", ""),
        ("mirnov", 0, "magnetics", "mhr"),
        ("mirnov", 8, "magnetics", "mhr"),
    ]


def test_every_role_in_the_plan_is_a_documented_one():
    assert set(channels.ROLES) == {s.role for s in channels.ROUND1_PLAN}


def test_a_channel_spec_is_frozen_and_keys_itself_by_diag_and_channel():
    spec = channels.ChannelSpec("ece", 8, "ece_core")
    assert spec.key == "ece:8"
    assert spec.fallback_for == ""
    with pytest.raises(FrozenInstanceError):
        spec.channel = 9


def test_a_complete_shot_runs_the_nine_primaries_and_no_fallback(complete):
    specs, reasons = channels.plan_for(complete)
    assert [s.key for s in specs] == [
        "mhr:0", "mhr:4", "ece:8", "ece:20", "ece:40",
        "co2:0", "co2:2", "bes:26", "bes:28",
    ]
    assert reasons == {
        "mirnov:0": "fallback not needed",
        "mirnov:8": "fallback not needed",
    }


def test_a_missing_mhr_promotes_the_mirnov_fallback(tmp_path):
    groups = dict(REAL_WIDTHS, mhr=None)
    specs, reasons = channels.plan_for(_write(tmp_path / "a.h5", groups))
    assert [s.key for s in specs][-2:] == ["mirnov:0", "mirnov:8"]
    assert "mhr:0" not in [s.key for s in specs]
    assert reasons["mhr:0"] == "group absent"
    assert reasons["mhr:4"] == "group absent"
    assert "mirnov:0" not in reasons


def test_an_absent_fallback_group_is_skipped_for_being_absent(tmp_path):
    # mhr gone AND mirnov gone: the fallback is wanted but not there, and
    # the reason has to say which of the two it is.
    groups = dict(REAL_WIDTHS, mhr=None, mirnov=None)
    specs, reasons = channels.plan_for(_write(tmp_path / "a.h5", groups))
    assert [s.key for s in specs] == [
        "ece:8", "ece:20", "ece:40", "co2:0", "co2:2", "bes:26", "bes:28",
    ]
    assert reasons["mirnov:0"] == "group absent"


def test_a_group_missing_altogether_is_absent_too(tmp_path):
    groups = {k: v for k, v in REAL_WIDTHS.items() if k != "co2"}
    _, reasons = channels.plan_for(_write(tmp_path / "a.h5", groups))
    assert reasons["co2:0"] == "group absent"
    assert reasons["co2:2"] == "group absent"


def test_the_absent_sentinel_is_one_column_not_zero_columns(tmp_path):
    # `shape[-1] < 2`: a two-sample group is data, a one-sample group is the
    # corpus' placeholder. The boundary is the whole rule, so it is pinned.
    two = _write(tmp_path / "two.h5", dict(REAL_WIDTHS), n=2)
    _, reasons = channels.plan_for(two)
    assert "mhr:0" not in reasons


def test_a_channel_beyond_the_group_is_reported_out_of_range(tmp_path):
    groups = dict(REAL_WIDTHS, ece=21)         # channels 0..20; 40 is gone
    specs, reasons = channels.plan_for(_write(tmp_path / "a.h5", groups))
    assert reasons["ece:40"] == "channel out of range"
    assert [s.key for s in specs if s.diag == "ece"] == ["ece:8", "ece:20"]


def test_the_plan_never_looks_at_a_value(tmp_path):
    # Header-only: an all-NaN group still plans, because `plan_for` decides
    # from `shape` alone. Stripping NaN is `read_waveform`'s job, and reading
    # 3.85 GB per shot to make a plan is not affordable.
    path = tmp_path / "a.h5"
    _write(path, dict(REAL_WIDTHS))
    with h5py.File(path, "r+") as f:
        f["mhr"]["ydata"][...] = np.nan
    specs, _ = channels.plan_for(path)
    assert "mhr:0" in [s.key for s in specs]


def test_specs_and_reasons_partition_the_plan(complete):
    specs, reasons = channels.plan_for(complete)
    assert not {s.key for s in specs} & set(reasons)
    assert {s.key for s in specs} | set(reasons) == {
        s.key for s in channels.ROUND1_PLAN
    }
    assert set(reasons.values()) <= set(channels.REASONS)


def test_a_caller_may_pass_its_own_plan(complete):
    plan = (channels.ChannelSpec("ece", 3, "ece_core"),)
    specs, reasons = channels.plan_for(complete, plan)
    assert [s.key for s in specs] == ["ece:3"] and reasons == {}


def test_coverage_is_the_finite_span_of_the_group(tmp_path):
    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("mhr")
        g.create_dataset("ydata", data=np.zeros((8, 5), dtype=np.float32))
        g.create_dataset(
            "xdata", data=np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        )
    assert channels.coverage(path, "mhr") == pytest.approx((0.5, 0.9))


def test_coverage_ignores_leading_and_trailing_non_finite_samples(tmp_path):
    # Measured on shot 198658: `mirnov` is NaN for the first 1,669,828
    # samples and the last 1,443,111, and its coverage is the finite middle.
    path = tmp_path / "a.h5"
    y = np.zeros((29, 6), dtype=np.float32)
    y[:, :2] = np.nan
    y[:, -1] = np.nan
    with h5py.File(path, "w") as f:
        g = f.create_group("mirnov")
        g.create_dataset("ydata", data=y)
        g.create_dataset("xdata", data=np.arange(6, dtype=np.float32) / 10.0)
    assert channels.coverage(path, "mirnov") == pytest.approx((0.2, 0.4))


def test_coverage_of_an_absent_group_is_two_nans(tmp_path):
    path = _write(tmp_path / "a.h5", dict(REAL_WIDTHS, co2=None))
    t0, t1 = channels.coverage(path, "co2")
    assert math.isnan(t0) and math.isnan(t1)


def test_coverage_of_a_group_that_is_not_in_the_file_is_two_nans(tmp_path):
    path = _write(tmp_path / "a.h5", {"mhr": 8})
    t0, t1 = channels.coverage(path, "bes")
    assert math.isnan(t0) and math.isnan(t1)


def test_coverage_of_an_all_nan_group_is_two_nans(tmp_path):
    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("mhr")
        g.create_dataset("ydata", data=np.full((8, 5), np.nan, dtype=np.float32))
        g.create_dataset("xdata", data=np.arange(5, dtype=np.float32))
    t0, t1 = channels.coverage(path, "mhr")
    assert math.isnan(t0) and math.isnan(t1)


def test_a_fallback_is_excluded_by_a_target_group_the_plan_never_names(tmp_path):
    # The presence check reads the FILE for `mhr`, not the plan: a plan of
    # nothing but the mirnov fallback - which is what a re-run of the one
    # channel that failed looks like - would otherwise see no `mhr` among
    # its own specs and promote a duplicate of the channel it exists to
    # replace.
    path = _write(tmp_path / "a.h5", {"mhr": 8, "mirnov": 29})
    plan = (channels.ChannelSpec("mirnov", 0, "magnetics", fallback_for="mhr"),)
    specs, reasons = channels.plan_for(path, plan)
    assert specs == []
    assert reasons == {"mirnov:0": "fallback not needed"}


def test_the_same_lone_fallback_plan_runs_when_the_target_is_absent(tmp_path):
    path = _write(tmp_path / "a.h5", {"mhr": None, "mirnov": 29})
    plan = (channels.ChannelSpec("mirnov", 0, "magnetics", fallback_for="mhr"),)
    specs, reasons = channels.plan_for(path, plan)
    assert [s.key for s in specs] == ["mirnov:0"] and reasons == {}
