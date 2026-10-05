"""Regression checks for bias-aware equilibrium and physical relaxation support."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

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


def test_sustained_high_q_alone_is_a_prior_not_absence():
    t, y, qmin = waveform(1.5, crashes=False)
    y += np.random.default_rng(99).normal(0, 0.15, y.shape)
    result = physics.detect(t, y, shot=1, qmin=qmin)
    middle = (t > 0.1) & (t < 1.1)
    # ECE noise too large to resolve an edge: the quiet-core test is unresolved.
    assert not result.absent_mask[middle].any()
    assert result.q_prior_mask[middle].mean() > 0.9
    reasons = result.absence_diagnostics["reason_samples"]
    assert reasons["q_prior_only"] == int(result.q_prior_mask.sum()) > 0
    assert reasons["tested_absence"] == 0


def test_tested_absence_needs_no_high_q_and_high_q_is_not_double_counted():
    t, y, qmin = waveform(0.8, crashes=False)
    result = physics.detect(t, y, shot=1, qmin=qmin)
    assert result.absent_mask[(t > 0.45) & (t < 0.75)].all()
    assert not result.q_prior_mask.any()
    t, y, qmin = waveform(1.8, crashes=False)
    both = physics.detect(t, y, shot=1, qmin=qmin)
    assert both.absent_mask[(t > 0.45) & (t < 0.75)].all()
    assert not (both.absent_mask & both.q_prior_mask).any()
    reasons = both.absence_diagnostics["reason_samples"]
    assert reasons["tested_absence_with_high_q"] > 0
    assert reasons["tested_absence_without_high_q"] == 0


def test_high_q_cannot_override_observed_conflicting_physical_train():
    t, y, qmin = waveform(2.0)
    result = physics.detect(t, y, shot=1, qmin=qmin)
    assert not result.intervals
    for event in result.uncertain_intervals:
        selected = (t >= event.t0_s) & (t < event.t1_s)
        assert not result.absent_mask[selected].any()
        assert not result.q_prior_mask[selected].any()


def test_unsustained_q_excursion_does_not_supply_a_prior():
    t, y, qmin = waveform(0.8, crashes=False)
    y += np.random.default_rng(99).normal(0, 0.15, y.shape)
    qmin[1][(t > 0.5) & (t < 0.52)] = 2.0
    result = physics.detect(t, y, shot=1, qmin=qmin)
    assert result.absence_diagnostics["reason_samples"]["sustained_high_q"] == 0
    assert not result.q_prior_mask.any()


def test_isolated_core_edge_is_not_absent_and_high_q_prior_stays_apart():
    t, y, qmin = waveform(2.0, crashes=False)
    y[2:5, t >= 0.6] -= 0.4
    result = physics.detect(t, y, shot=1, qmin=qmin, core_channels=[2, 3, 4])
    veto = physics.DEFAULT_RULE.isolated_edge_context_ms / 1000
    assert not result.absent_mask[np.abs(t - 0.6) < veto - 0.001].any()
    assert result.absent_mask[(t > 0.45) & (t < 0.6 - veto - 0.002)].all()
    assert result.q_prior_mask[(t > 0.95) & (t < 1.1)].all()
    assert not (result.absent_mask & result.q_prior_mask).any()


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
    # Every rule field, including the cutoff and quiet-core settings, is frozen.
    assert set(json.loads(path.read_text())["rule"]) == set(asdict(frozen))


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


def _cutoff_profile(step_start, step_stop, *, factor=0.3, channels=12):
    """Flat 2 keV profile with a core-side step, cutoff-like, over a time window."""
    t = np.arange(0, 1.0, 0.0001)
    y = np.full((channels, len(t)), 2.0)
    window = (t >= step_start) & (t < step_stop)
    y[3:7, window] *= factor
    # A live channel always fluctuates; a flat one is a dead channel.
    y += np.random.default_rng(4).normal(0, 0.003, y.shape)
    radius = np.repeat((3.4 - 0.1 * np.arange(channels))[:, None], len(t), axis=1)
    rho = np.abs(radius - 2.9) / 0.6
    geometry = RadiusGeometry(
        t,
        radius,
        np.full(len(t), 2.9),
        None,
        None,
        np.full(len(t), 3.5),
        rho,
    )
    return t, y, geometry


def test_ece_validity_flags_sustained_cutoff_step_but_not_a_brief_dip():
    t, y, geometry = _cutoff_profile(0.4, 0.6)
    flag, info = physics.ece_validity(y, t, radius=geometry)
    assert flag[(t > 0.42) & (t < 0.58)].all()
    assert not flag[t < 0.38].any() and not flag[t > 0.62].any()
    assert info["status"] == "tested_nominal_geometry"
    t, y, geometry = _cutoff_profile(0.4, 0.41)
    assert not physics.ece_validity(y, t, radius=geometry)[0].any()


def test_ece_validity_flags_cold_axis_channel_and_passes_a_peaked_profile():
    t = np.arange(0, 1.0, 0.0001)
    radius = np.repeat((3.4 - 0.1 * np.arange(12))[:, None], len(t), axis=1)
    rho = np.abs(radius - 2.9) / 0.6
    geometry = RadiusGeometry(
        t, radius, np.full(len(t), 2.9), None, None, np.full(len(t), 3.5), rho
    )
    peaked = np.repeat((4.0 - 0.3 * np.abs(np.arange(12) - 5.0))[:, None], len(t), 1)
    peaked = peaked + np.random.default_rng(6).normal(0, 0.003, peaked.shape)
    assert not physics.ece_validity(peaked, t, radius=geometry)[0].any()
    hollow = peaked.copy()
    hollow[5, (t >= 0.3) & (t < 0.7)] = 0.5 * peaked[5, 0]
    hollow[4:7, (t >= 0.3) & (t < 0.7)] = 1.2
    flag, _ = physics.ece_validity(hollow, t, radius=geometry)
    assert flag[(t > 0.35) & (t < 0.65)].all()


def test_ece_validity_without_geometry_is_not_tested():
    t, y, _ = _cutoff_profile(0.4, 0.6)
    flag, info = physics.ece_validity(y, t, radius=None)
    assert not flag.any()
    assert info["status"] == "geometry_unavailable_not_tested"


def test_detect_drops_cutoff_time_from_observable_and_accounts_for_it():
    t, y, geometry = _cutoff_profile(0.4, 0.6)
    result = physics.detect(
        t, y, shot=1, core_channels=[2, 3, 4], radius_geometry=geometry
    )
    assert not result.observable[(t > 0.45) & (t < 0.55)].any()
    assert not result.absent_mask[(t > 0.4) & (t < 0.6)].any()
    accounting = result.absence_diagnostics["guard_accounting"]
    assert accounting["removed_by_ece_validity"] > 1000
    assert result.absence_diagnostics["ece_validity"]["removed_samples"] > 1000


def test_axis_field_uses_the_local_resonance_field_not_bt_at_r0():
    from labeler.sawtooth.geometry import ELECTRON_CYCLOTRON_HZ_PER_T, axis_field_T

    t = np.arange(4) * 0.001
    bt_r0, r0 = 2.0, 1.67
    radius_ch = np.array([2.2, 1.9, 1.7, 1.5])  # channel index rises with F
    frequency = 2 * ELECTRON_CYCLOTRON_HZ_PER_T * bt_r0 * r0 / radius_ch
    radius = np.repeat(radius_ch[:, None], len(t), axis=1)
    axis_R = np.full(len(t), 1.75)
    geometry = RadiusGeometry(t, radius, axis_R)
    field = axis_field_T(geometry, frequency.tolist())
    assert field == pytest.approx(bt_r0 * r0 / 1.75, rel=1e-6)
    assert field[0] < bt_r0
    assert axis_field_T(None, frequency.tolist()) is None


def test_x2_cutoff_density_scales_with_the_square_of_the_field():
    from labeler.sawtooth.geometry import x2_cutoff_density

    low, high = x2_cutoff_density(1.9), x2_cutoff_density(2.0)
    assert high / low == pytest.approx((2.0 / 1.9) ** 2)
    # 2 T: 0.9 * 2 * (27.992e9 * 2 / 8.98)^2 = about 7.0e19 m^-3.
    assert x2_cutoff_density(2.0) == pytest.approx(7.0e19, rel=0.01)
    assert x2_cutoff_density(-2.0) == x2_cutoff_density(2.0)


def test_dead_channel_is_not_read_as_a_cutoff_step_but_a_real_step_still_is():
    t, y, geometry = _cutoff_profile(0.4, 0.4)
    y[5] = 0.02 + np.random.default_rng(5).normal(0, 0.0005, len(t))
    flag, info = physics.ece_validity(y, t, radius=geometry)
    assert not flag.any()
    assert info["dead_channels"] == [5]
    # A channel stuck at a constant value carries no fluctuation: also dead.
    t, y, geometry = _cutoff_profile(0.4, 0.4)
    y[8] = 2.0
    assert physics.ece_validity(y, t, radius=geometry)[1]["dead_channels"] == [8]
    # The same dead channel does not hide a one-sided cutoff step beside it.
    t, y, geometry = _cutoff_profile(0.4, 0.6)
    y[9] = 0.02 + np.random.default_rng(5).normal(0, 0.0005, len(t))
    flag, info = physics.ece_validity(y, t, radius=geometry)
    assert info["dead_channels"] == [9]
    assert flag[(t > 0.42) & (t < 0.58)].all()


def test_dead_channels_are_below_the_floor_or_flat_for_the_record():
    rng = np.random.default_rng(1)
    values = 2.0 + rng.normal(0, 0.05, (4, 500))
    values[1] = 0.02 + rng.normal(0, 0.001, 500)
    values[2] = 3.0
    values[3, :400] = np.nan
    values[3, 400:] = 0.3  # below the 0.5 keV floor wherever it is finite
    assert physics.dead_channels(values).tolist() == [False, True, True, True]
    assert not physics.dead_channels(np.full((1, 5), np.nan)).any()


def _edge_shot(qvalue, *, outer_rise):
    """Periodic core edges with or without a profile (outer rise) around them."""
    t = np.arange(0, 2, 0.0001)
    y = np.full((8, len(t)), 2.0)
    for crash in np.arange(0.4, 1.7, 0.08):
        transient = np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
        y[2:5] -= 0.5 * transient
        if outer_rise:
            y[5:7] += 0.3 * transient
    y += np.random.default_rng(12).normal(0, 0.002, y.shape)
    return t, y, (t, np.full(len(t), qvalue))


def test_q_prior_splits_into_ece_contradicted_and_untested():
    # Periodic core edges without an outer rise: the ECE contradicts the q prior.
    t, y, qmin = _edge_shot(2.0, outer_rise=False)
    found = physics.detect(t, y, shot=1, qmin=qmin, core_channels=[2, 3, 4])
    contradicted = found.q_prior_contradicted_mask
    # The first edge is at 0.4 s: its context (375 ms) starts at 0.025 s.
    assert contradicted.any() and not contradicted[t < 0.02].any()
    assert (found.q_prior_mask | ~contradicted).all()  # a subset of the q prior
    assert not (contradicted & found.absent_mask).any()
    reasons = found.absence_diagnostics["reason_samples"]
    assert reasons["q_prior_ece_contradicted"] == int(contradicted.sum())
    assert (
        reasons["q_prior_ece_contradicted"] + reasons["q_prior_untested"]
        == reasons["q_prior_only"]
        == int(found.q_prior_mask.sum())
    )
    # Noise too large for any edge test: the prior is untested, not contradicted.
    t, y, qmin = waveform(1.5, crashes=False)
    y += np.random.default_rng(99).normal(0, 0.15, y.shape)
    quiet = physics.detect(t, y, shot=1, qmin=qmin)
    assert quiet.q_prior_mask.any() and not quiet.q_prior_contradicted_mask.any()


def test_longer_edge_context_only_removes_absence_next_to_an_edge():
    t, y, qmin = waveform(0.8, crashes=False)
    y[2:5, t >= 0.6] -= 0.4
    frame = replace(physics.DEFAULT_RULE, isolated_edge_context_ms=5.15)
    found = physics.detect(
        t, y, shot=1, qmin=qmin, core_channels=[2, 3, 4], rule=frame
    )
    edges = found.absence_diagnostics["core_relaxation_test"]["ambiguous_edge_times_s"]
    assert len(edges) == 1
    same = physics.apply_edge_context(t, found.absent_holdoff_mask, edges, 5.15)
    assert (same == found.absent_mask).all()
    wide = physics.apply_edge_context(t, found.absent_holdoff_mask, edges, 50.0)
    assert not wide[np.abs(t - 0.6) < 0.0499].any()
    assert wide.sum() < found.absent_mask.sum()
    assert not (wide & ~found.absent_mask).any()
    # Applying the veto of the rule itself reproduces the rule's own mask.
    own = replace(physics.DEFAULT_RULE, isolated_edge_context_ms=50.0)
    again = physics.detect(t, y, shot=1, qmin=qmin, core_channels=[2, 3, 4], rule=own)
    assert (again.absent_mask == wide).all()
    assert (again.absent_holdoff_mask == found.absent_holdoff_mask).all()


def _density_file(density_m3):
    import h5py

    file = h5py.File("memory.h5", "w", driver="core", backing_store=False)
    tx = np.arange(-0.02, 1.05, 0.02)  # covers the whole analysis window
    group = file.create_group("ts_core_density")
    group["xdata"] = tx
    group["ydata"] = np.full((4, len(tx)), density_m3)
    return file


def _axis_geometry(t, bt_r0=2.0, r0=1.67, r_axis=1.75):
    from labeler.sawtooth.geometry import ELECTRON_CYCLOTRON_HZ_PER_T

    radius_ch = np.array([2.2, 1.9, 1.7, 1.5])
    frequency = 2 * ELECTRON_CYCLOTRON_HZ_PER_T * bt_r0 * r0 / radius_ch
    radius = np.repeat(radius_ch[:, None], len(t), axis=1)
    return RadiusGeometry(t, radius, np.full(len(t), r_axis)), frequency.tolist()


@pytest.fixture
def driver(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/labeler"
    monkeypatch.syspath_prepend(str(scripts))
    import sawtooth_physics as driver

    from labeler.features import resolve_archive

    monkeypatch.setattr(resolve_archive, "resolve", lambda shot, names: ({}, {}))
    return driver


def test_local_axis_field_cutoff_is_used_when_bt_is_missing(driver, monkeypatch):
    t = np.arange(0.0, 1.0, 0.001)
    geometry, frequency = _axis_geometry(t)
    monkeypatch.setattr(driver, "local_scalar", lambda shot, name, paths: None)
    # Local axis field 2.0 T * 1.67 / 1.75 = 1.909 T: cutoff 6.4e19, below the
    # density, whereas the fixed 8e19 guard and Bt at R0 (7.0e19) would keep it.
    with _density_file(6.8e19) as file:
        support, reference, info = driver.density_support(
            file, t, 1, None, None, radius=geometry, frequency_hz=frequency
        )
    assert info["status"] == "Thomson_90percentile_and_local_axis_field"
    assert info["cutoff_field"].startswith("local field")
    assert info["fallback_guard"] == "none" and info["axis_field_samples"] == len(t)
    assert not support.any()
    assert reference.all()  # the fixed guard it replaces would have kept all of it
    assert info["high_density_samples"] == len(t)


def test_fixed_guard_is_the_branch_only_when_axis_field_and_bt_are_missing(
    driver, monkeypatch
):
    t = np.arange(0.0, 1.0, 0.001)
    monkeypatch.setattr(driver, "local_scalar", lambda shot, name, paths: None)
    with _density_file(6.8e19) as file:
        support, _, info = driver.density_support(file, t, 1, None, None)
    assert info["status"].endswith("fixed_density_guard_axis_field_and_bt_unavailable")
    assert info["cutoff_field"].startswith("none")
    assert info["fallback_guard"] == "fixed_density_guard"
    assert support.all() and info["high_density_samples"] == 0
    with _density_file(8.5e19) as file:
        support, _, _ = driver.density_support(file, t, 1, None, None)
    assert not support.any()


def test_bt_at_r0_is_only_the_fallback_where_the_axis_field_is_unmapped(
    driver, monkeypatch
):
    t = np.arange(0.0, 1.0, 0.001)
    bt = (t, np.full(len(t), 2.0))
    monkeypatch.setattr(driver, "local_scalar", lambda shot, name, paths: bt)
    with _density_file(6.8e19) as file:
        support, reference, info = driver.density_support(file, t, 1, None, None)
        assert info["status"].endswith("reference_bt_axis_unmapped")
        assert support.all() and reference.all()  # 6.8e19 < 7.0e19
        geometry, frequency = _axis_geometry(t)
        geometry.R_m[:, 500:] = np.nan  # axis field unmapped in the second half
        support, reference, info = driver.density_support(
            file, t, 1, None, None, radius=geometry, frequency_hz=frequency
        )
    assert not support[:500].any() and support[500:].all()
    assert reference.all()
    assert info["fallback_guard"] == "reference_field_Bt_at_R0"
    assert info["axis_field_samples"] == 500 and info["fallback_samples"] == 500
