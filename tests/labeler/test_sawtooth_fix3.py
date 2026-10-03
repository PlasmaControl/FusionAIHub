"""Regression checks for bias-aware equilibrium and physical relaxation support."""

import json
from pathlib import Path

import numpy as np

from labeler.sawtooth import physics
from labeler.sawtooth.geometry import RadiusGeometry


def waveform(qvalue, *, crashes=True):
    t = np.arange(0, 1.2, 0.0001)
    y = np.full((12, len(t)), 2.0)
    y[2:5] = 3.0
    if crashes:
        for crash in np.arange(0.16, 1.05, 0.08):
            transient = np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
            y[2:5] -= 0.7 * transient
            y[5:7] += 0.4 * transient
    y += np.random.default_rng(87).normal(0, 0.003, y.shape)
    return t, y, (t, np.full(len(t), qvalue))


def test_efit01_bias_does_not_demote_qmin_122_train():
    t, y, qmin = waveform(1.22)
    result = physics.detect(t, y, shot=1, qmin=qmin)
    assert result.intervals
    assert all(e.attrs["state"] == "present" for e in result.crashes)


def test_sustained_high_q_is_absence_evidence_even_with_unresolved_ece_noise():
    t, y, qmin = waveform(1.5, crashes=False)
    y += np.random.default_rng(99).normal(0, 0.15, y.shape)
    result = physics.detect(t, y, shot=1, qmin=qmin)
    middle = (t > 0.1) & (t < 1.1)
    assert result.absent_mask[middle].mean() > 0.95
    assert result.absence_diagnostics["reason_samples"]["sustained_high_q_absence"] > 0


def test_high_q_cannot_override_observed_conflicting_physical_train():
    t, y, qmin = waveform(2.0)
    result = physics.detect(t, y, shot=1, qmin=qmin)
    assert not result.intervals
    for event in result.uncertain_intervals:
        selected = (t >= event.t0_s) & (t < event.t1_s)
        assert not result.absent_mask[selected].any()


def test_unsustained_q_excursion_does_not_supply_absence():
    t, y, qmin = waveform(0.8, crashes=False)
    y += np.random.default_rng(99).normal(0, 0.15, y.shape)
    qmin[1][(t > 0.5) & (t < 0.52)] = 2.0
    result = physics.detect(t, y, shot=1, qmin=qmin)
    assert result.absence_diagnostics["reason_samples"]["sustained_high_q_absence"] == 0


def test_high_q_preserves_isolated_nonprofile_core_edge_support():
    t, y, qmin = waveform(2.0, crashes=False)
    y[2:5, t >= 0.6] -= 0.4
    result = physics.detect(t, y, shot=1, qmin=qmin, core_channels=[2, 3, 4])
    assert not result.absent_mask[(t > 0.596) & (t < 0.604)].any()
    assert result.absent_mask[(t > 0.7) & (t < 1.0)].all()


def test_sub_10ms_phase_is_never_physical_support():
    assert not physics.core_relaxation_phases(np.arange(0.1, 0.3, 0.00515), [(0, 1)])


def test_periodicity_null_rejects_refractory_shuffled_noise():
    edges = np.cumsum(0.00515 + np.random.default_rng(3).exponential(0.025, 50))
    assert physics.periodicity_null(edges)["p_value"] > 0.05
    periodic = np.arange(0.1, 1.0, 0.08)
    assert physics.periodicity_null(periodic)["p_value"] <= 0.05
    phases = physics.core_relaxation_phases(periodic, [(0, 1.2)])
    assert phases and phases[0]["null_p_value"] <= 0.05


def test_default_detect_rule_is_authoritative_frozen_rule():
    path = Path(physics.__file__).with_name("freeze.json")
    frozen = physics.Rule(**json.loads(path.read_text())["rule"])
    assert physics.DEFAULT_RULE == frozen
    assert frozen.central_relative_drop == 0.05


def test_standalone_profile_cannot_use_unverified_terminal_gain():
    level = np.full(48, 3.0)
    step = np.zeros(48)
    step[33:40] = -0.5
    step[40:48] = 0.5
    verdict, attrs = physics.inversion_profile(step, level)
    assert verdict != "accept"
    assert set(range(40, 48)) <= set(attrs["masked_channels"])


def test_central_drop_uses_efit_axis_instead_of_hottest_nearby_channel():
    t, y, qmin = waveform(0.8)
    y[4] *= 1.4
    radius = np.repeat((3.5 - 0.2 * np.arange(12))[:, None], len(t), axis=1)
    geometry = RadiusGeometry(
        t,
        radius,
        np.full(len(t), 2.9),
        np.full(len(t), 2.6),
        np.full(len(t), 3.2),
    )
    result = physics.detect(
        t,
        y,
        shot=1,
        qmin=qmin,
        core_channels=[2, 3, 4],
        radius_geometry=geometry,
        spatially_verified=True,
    )
    assert result.crashes
    assert all(event.attrs["central_channel"] == 3 for event in result.crashes)


def test_direct_detector_excludes_harmonic_overlap_profile_evidence():
    t, y, qmin = waveform(0.8)
    radius = np.repeat((3.5 - 0.2 * np.arange(12))[:, None], len(t), axis=1)
    geometry = RadiusGeometry(
        t,
        radius,
        np.full(len(t), 2.9),
        np.full(len(t), 2.6),
        np.full(len(t), 3.2),
        np.full(len(t), 4.5),
    )
    result = physics.detect(
        t,
        y,
        shot=1,
        qmin=qmin,
        core_channels=[2, 3, 4],
        radius_geometry=geometry,
        spatially_verified=True,
    )
    assert not result.crashes
