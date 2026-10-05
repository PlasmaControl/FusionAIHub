"""Physical regressions for four-state sawtooth labels."""

from dataclasses import replace

import numpy as np
import pytest

from labeler.events.panels.ece_geometry import Geometry
from labeler.sawtooth.physics import Rule, detect
from labeler.sawtooth.preprocessing import state_spans


def waveform(crashes=(0.08, 0.16, 0.24, 0.32), boundaries=None, amplitude=0.7):
    t = np.arange(0, 0.64, 0.0001)
    y = np.full((12, len(t)), 2.0)
    y[2:5] = 3.0
    if boundaries is None:
        boundaries = [5] * len(crashes)
    for crash, boundary in zip(crashes, boundaries, strict=True):
        transient = np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
        y[2:boundary] -= amplitude * transient
        y[boundary : boundary + 2] += 0.4 * transient
    y += np.random.default_rng(87).normal(0, 0.003, y.shape)
    return t, y


def test_magnetics_q_conflict_abstains_without_rejecting_crashes():
    t, y = waveform()
    result = detect(t, y, shot=1, qmin=([0.0, 0.64], [1.8, 1.8]))
    assert len(result.crashes) == 4
    assert not result.intervals
    assert len(result.uncertain_intervals) == 1
    assert result.uncertain_intervals[0].attrs["state"] == "uncertain"
    assert all(
        "qmin_conflict" in e.attrs["uncertainty_reasons"] for e in result.crashes
    )
    assert "qmin_above_one" not in result.rejected


def test_uncertain_train_last_crash_cannot_be_an_assessed_negative():
    t, y = waveform()
    result = detect(t, y, shot=1, qmin=([0.0, 0.64], [1.8, 1.8]))
    _, assessed = state_spans(
        t, result.observable, [], [(r.t0_s, r.t1_s) for r in result.uncertain_intervals]
    )
    for crash in result.crashes:
        assert not assessed[np.searchsorted(t, crash.t0_s)]


def test_calibrated_core_supersedes_missing_fallback_proxy():
    t, small = waveform()
    y = np.full((48, len(t)), np.nan)
    y[:12] = small
    positions = np.full(48, 0.8)
    positions[2:7] = [0.04, 0.08, 0.16, 0.36, 0.49]
    psi = np.repeat(positions[:, None], 2, axis=1)
    geometry = Geometry(
        np.array([0.0, 640.0]),
        np.where(psi < 0.25, 0.8, 1.2),
        psi,
        np.array([0.25, 0.25]),
    )
    result = detect(t, y, shot=1, geometry=geometry, core_channels=range(20, 36))
    assert len(result.crashes) == 4
    assert all(r.attrs["central_channel"] in (2, 3, 4) for r in result.crashes)


def test_insufficient_profile_support_is_unassessed():
    t, _ = waveform()
    y = np.full((48, len(t)), np.nan)
    y[20:22] = 3
    result = detect(t, y, shot=1, core_channels=range(20, 36))
    assert not result.observable.any()


def test_minimum_central_amplitude_is_measured_from_pre_post_temperature():
    t, y = waveform(amplitude=0.7)
    result = detect(t, y, shot=1, core_channels=[2, 3, 4])
    assert len(result.crashes) == 4
    assert all(0.18 < e.attrs["central_relative_drop"] < 0.26 for e in result.crashes)
    assert not detect(
        t, y, shot=1, rule=replace(Rule(), central_relative_drop=0.3)
    ).crashes


def test_missing_core_and_low_temperature_are_unobservable():
    t, y = waveform()
    gap = (t >= 0.18) & (t <= 0.2)
    y[2:5, gap] = np.nan
    low = (t >= 0.38) & (t <= 0.4)
    y[2:5, low] = 0.1
    result = detect(t, y, shot=1, core_channels=[2, 3, 4])
    assert result.observable.dtype == np.bool_
    assert not result.observable[gap | low].any()
    assert result.observable[(t > 0.02) & (t < 0.06)].all()
    assert not result.intervals  # Two crashes on each side cannot form a train.
    assert all(e.attrs["state"] == "uncertain" for e in result.uncertain_intervals)


def test_caller_density_cutoff_mask_splits_periodic_trains():
    t, y = waveform(crashes=(0.08, 0.16, 0.24, 0.4, 0.48, 0.56))
    observable = ~((t > 0.29) & (t < 0.33))
    result = detect(t, y, shot=1, observability=observable)
    assert len(result.intervals) == 2
    assert not result.observable[~observable].any()
    assert all(not (e.t0_s < 0.3 < e.t1_s) for e in result.intervals)


def test_large_inversion_jump_cannot_make_a_present_train():
    t, y = waveform(boundaries=[5, 9, 5, 9])
    result = detect(t, y, shot=1)
    assert not result.intervals
    assert result.uncertain_intervals
    assert all(e.attrs["state"] == "uncertain" for e in result.uncertain_intervals)


def test_isolated_candidate_is_uncertain_and_not_point_training_truth():
    t, y = waveform(crashes=(0.16,))
    result = detect(t, y, shot=1)
    assert not result.crashes
    assert not result.intervals
    assert len(result.uncertain_intervals) == 1
    assert (
        "isolated_candidate"
        in result.uncertain_intervals[0].attrs["uncertainty_reasons"]
    )


def test_dalpha_spike_with_edge_loss_is_rejected_before_proxy_gate():
    t, y = waveform()
    dalpha = np.ones(len(t))
    for crash in (0.08, 0.16, 0.24, 0.32):
        dalpha[np.abs(t - crash) < 0.0004] = 5.0
    result = detect(t, y, shot=1, core_channels=[0, 1], dalpha=(t, dalpha))
    assert not result.crashes
    assert result.rejected["elm_edge_only"] == 4


def test_calibration_rejects_outer_loss_and_inner_gain():
    t, y = waveform()
    positions = np.array(
        [0.8, 0.7, 0.64, 0.49, 0.36, 0.16, 0.08, 0.04, 0.64, 0.75, 0.85, 0.95]
    )
    psi = np.repeat(positions[:, None], 2, axis=1)
    geometry = Geometry(
        np.array([0.0, 640.0]),
        np.where(psi < 0.25, 0.8, 1.2),
        psi,
        np.array([0.25, 0.25]),
    )
    result = detect(t, y, shot=1, geometry=geometry)
    assert not result.crashes
    assert result.rejected["calibrated_direction"] == 4


def test_auxiliary_neutron_drop_and_mirnov_burst_have_explicit_evidence():
    t, y = waveform()
    neutrons = np.full(len(t), 100.0)
    mirnov = np.zeros(len(t))
    for crash in (0.08, 0.16, 0.24, 0.32):
        neutrons -= 20 * np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
        mirnov[np.abs(t - crash) < 0.0004] = 40.0
    result = detect(
        t,
        y,
        shot=1,
        neutron=(t, neutrons),
        mirnov=(t, mirnov),
        sxr=y,
        nbi=(t, np.full(len(t), 1e6)),
    )
    assert len(result.crashes) == 4
    assert all(e.attrs["neutron_drop_corroboration"] is True for e in result.crashes)
    assert all(e.attrs["mirnov_burst_corroboration"] is True for e in result.crashes)
    assert all(e.attrs["sxr_corroboration"] is None for e in result.crashes)


def test_nonuniform_observability_shape_is_an_error():
    t, y = waveform()
    with pytest.raises(ValueError, match="observability"):
        detect(t, y, shot=1, observability=np.ones(len(t) - 1, dtype=bool))


def test_auxiliary_without_positive_crash_evidence_does_not_veto_train():
    t, y = waveform()
    result = detect(t, y, shot=1, neutron=(t, np.full(len(t), 100.0)))
    assert len(result.crashes) == 4
    assert len(result.intervals) == 1
    assert not result.uncertain_intervals
    assert all(e.attrs["state"] == "present" for e in result.crashes)
    assert all(e.attrs["mirnov_burst_corroboration"] is None for e in result.crashes)


@pytest.mark.parametrize("nbi_power", [None, 0.0])
def test_neutron_drop_without_known_nbi_on_is_not_corroboration(nbi_power):
    t, y = waveform()
    neutrons = np.full(len(t), 100.0)
    for crash in (0.08, 0.16, 0.24, 0.32):
        neutrons -= 20 * np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
    nbi = None if nbi_power is None else (t, np.full(len(t), nbi_power))
    result = detect(t, y, shot=1, neutron=(t, neutrons), nbi=nbi)
    assert len(result.intervals) == 1
    assert all(e.attrs["neutron_drop_corroboration"] is None for e in result.crashes)


def test_noise_sized_neutron_drop_is_unknown_and_records_window_noise():
    t, y = waveform()
    neutrons = 100 + np.tile([-10.0, 10.0], len(t) // 2)
    for crash in (0.08, 0.16, 0.24, 0.32):
        neutrons -= 3 * np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
    result = detect(t, y, shot=1, neutron=(t, neutrons), nbi=(t, np.full(len(t), 1e6)))
    assert len(result.intervals) == 1
    assert all(e.attrs["neutron_drop_corroboration"] is None for e in result.crashes)
    assert all(e.attrs["neutron_window_noise"] > 15 for e in result.crashes)
    assert all(e.attrs["neutron_noise_k"] == 3 for e in result.crashes)


def test_auxiliary_positive_evidence_is_attached_only_to_coincident_crash():
    t, y = waveform()
    mirnov = np.zeros(len(t))
    mirnov[np.abs(t - 0.16) < 0.0004] = 40
    result = detect(t, y, shot=1, mirnov=(t, mirnov))
    assert len(result.intervals) == 1
    assert [e.attrs["mirnov_burst_corroboration"] for e in result.crashes] == [
        None,
        True,
        None,
        None,
    ]


def test_all_missing_auxiliary_samples_leave_evidence_unknown():
    t, y = waveform()
    missing = (t, np.full(len(t), np.nan))
    result = detect(t, y, shot=1, neutron=missing, mirnov=missing)
    assert len(result.intervals) == 1
    assert not result.uncertain_intervals
    assert all(e.attrs["neutron_drop_corroboration"] is None for e in result.crashes)
    assert all(e.attrs["mirnov_burst_corroboration"] is None for e in result.crashes)
