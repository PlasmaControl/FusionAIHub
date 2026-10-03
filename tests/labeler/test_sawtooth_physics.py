import numpy as np
import pytest

from labeler.events.panels.ece_geometry import Geometry
from labeler.sawtooth.metrics import (
    aggregate,
    bin_times,
    event_cells,
    interval_cells,
    score_histogram,
)
from labeler.sawtooth.models import HL3, PhasePicker, soft_crash_target
from labeler.sawtooth.physics import detect, inversion_profile, trains


def synthetic():
    t = np.arange(0, 0.4, 0.0001)
    y = np.ones((8, len(t))) * 2
    # Core drop and recovery, outer rise at the same times.
    for crash in (0.08, 0.16, 0.24, 0.32):
        transient = np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
        y[2:5] -= 0.5 * transient
        y[5:7] += 0.3 * transient
    y += np.random.default_rng(10).normal(0, 0.005, y.shape)
    return t, y


def test_crashes_and_train_with_point_and_span_attrs():
    t, y = synthetic()
    found = detect(t, y, shot=1)
    assert len(found.crashes) == 4
    assert (
        np.max(
            np.abs(np.array([r.t0_s for r in found.crashes]) - [0.08, 0.16, 0.24, 0.32])
        )
        < 0.0005
    )
    assert len(found.intervals) == 1
    assert found.intervals[0].attrs["crowd"] is True
    assert all(r.attrs["crowd"] is False for r in found.crashes)
    assert found.intervals[0].attrs["period_ms"] == pytest.approx(80, abs=1)
    assert all(r.attrs["inversion_rho"] is None for r in found.crashes)


def test_qmin_conflict_and_missing_geometry_do_not_fabricate_radius():
    t, y = synthetic()
    qmin = (np.array([0.0, 0.4]), np.array([1.2, 1.2]))
    found = detect(t, y, shot=1, qmin=qmin)
    assert len(found.crashes) == 4
    assert not found.intervals
    assert len(found.uncertain_intervals) == 1
    assert all(r.attrs["inversion_rho"] is None for r in found.crashes)
    assert all(r.attrs["state"] == "uncertain" for r in found.crashes)


def test_global_drop_and_channel_noise_rejected():
    step = np.full(8, -0.5)
    assert inversion_profile(step, np.full(8, 2.0))[0] == "redistribution"
    t, y = synthetic()
    y[5:7] = 2
    assert not detect(t, y, shot=1).crashes


def test_inversion_uses_adjacent_gain_instead_of_stronger_remote_gain():
    step = np.array([0, 0, -0.5, -0.5, 0.15, 0.15, 0, 0.6, 0.6, 0])
    verdict, attrs = inversion_profile(step, np.full(10, 3.0))
    assert verdict == "accept"
    assert attrs["inversion_channel"] == 3.5


def test_implausibly_hot_gain_channels_cannot_define_inversion():
    step = np.array([0, 0, -0.5, -0.5, 0, 0, 0.7, 0.7])
    level = np.array([3, 3, 3, 3, 2, 2, 12, 12])
    verdict, _ = inversion_profile(step, level, core_level=3.0)
    assert verdict == "redistribution"


def test_absence_needs_complete_quiet_core_context():
    t = np.arange(0, 2, 0.0001)
    y = np.full((8, len(t)), 2.0)
    found = detect(t, y, shot=1, core_channels=[2, 3, 4])
    assert found.absent_mask[(t > 0.5) & (t < 1.5)].all()
    assert not found.absent_mask[t < 0.375].any()
    assert found.absence_diagnostics["core_relaxation_test"]["periodic_edges"] == 0


def test_periodic_core_relaxation_without_outer_rise_prevents_absence():
    t = np.arange(0, 2, 0.0001)
    y = np.full((8, len(t)), 2.0)
    for crash in np.arange(0.4, 1.7, 0.08):
        transient = np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
        y[2:5] -= 0.5 * transient
    y += np.random.default_rng(12).normal(0, 0.002, y.shape)
    found = detect(t, y, shot=1, core_channels=[2, 3, 4])
    assert not found.intervals
    assert not found.absent_mask[(t > 0.5) & (t < 1.5)].any()
    assert found.absence_diagnostics["core_relaxation_test"]["periodic_edges"] >= 10


@pytest.mark.parametrize("opposing_rises", [False, True], ids=["diluted", "cancelled"])
def test_subset_core_drops_cannot_be_hidden_by_proxy_averaging(opposing_rises):
    t = np.arange(0, 2, 0.0001)
    y = np.full((8, len(t)), 2.0)
    for crash in np.arange(0.4, 1.7, 0.08):
        transient = np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
        y[2:4] -= 0.12 * transient
        if opposing_rises:
            # Separate single-channel gains fail the contiguous profile test
            # and exactly cancel the two negative channels in a signed mean.
            y[[0, 6]] += 0.12 * transient
    y += np.random.default_rng(43).normal(0, 0.0005, y.shape)
    found = detect(t, y, shot=1, core_channels=range(7))
    assert not found.intervals
    assert found.absence_diagnostics["profile_passing_candidates"] == 0
    assert not found.absent_mask[(t > 0.5) & (t < 1.5)].any()
    assert found.absence_diagnostics["core_relaxation_test"]["periodic_edges"] >= 10


def test_slow_core_relaxation_phase_cannot_have_absent_islands():
    t = np.arange(0, 4.5, 0.0001)
    y = np.full((8, len(t)), 2.0)
    for crash in (0.6, 1.5, 2.4, 3.3):
        transient = np.where(t >= crash, np.exp(-(t - crash) / 0.08), 0)
        y[2:4] -= 0.2 * transient
    y += np.random.default_rng(59).normal(0, 0.0005, y.shape)
    found = detect(t, y, shot=1, core_channels=range(7))
    assert not found.intervals
    assert found.absence_diagnostics["profile_passing_candidates"] == 0
    assert not found.absent_mask[(t > 0.6) & (t < 3.3)].any()
    phases = found.absence_diagnostics["core_relaxation_test"]["phase_spans"]
    assert len(phases) == 1
    assert phases[0]["period_ms"] == pytest.approx(900, abs=1)


def test_core_relaxation_phases_do_not_cross_unobserved_support():
    from labeler.sawtooth.physics import core_relaxation_phases

    edges = [0.6, 1.5, 2.4, 3.3]
    assert core_relaxation_phases(edges, [(0, 2), (2.1, 4.5)]) == []
    phases = core_relaxation_phases(edges, [(0, 4.5)])
    assert phases == [
        {
            "start_s": pytest.approx(0.225),
            "end_s": pytest.approx(3.675),
            "first_edge_s": 0.6,
            "last_edge_s": 3.3,
            "edges": 4,
            "period_ms": pytest.approx(900),
            "minimum_gap_ms": pytest.approx(900),
            "maximum_gap_ms": pytest.approx(900),
        }
    ]


def test_profile_candidate_below_central_amplitude_prevents_absence():
    t = np.arange(0, 2, 0.0001)
    y = np.full((8, len(t)), 2.0)
    transient = np.where(t >= 1.0, np.exp(-(t - 1.0) / 0.018), 0)
    y[2:5] -= 0.12 * transient
    y[5:7] += 0.1 * transient
    y += np.random.default_rng(14).normal(0, 0.001, y.shape)
    found = detect(t, y, shot=1, core_channels=[2, 3, 4])
    assert not found.crashes
    assert not found.absent_mask[(t > 0.7) & (t < 1.3)].any()
    assert found.absence_diagnostics["profile_passing_candidates"] >= 1


def test_noisy_core_without_resolved_edges_is_not_tested_absence():
    t = np.arange(0, 2, 0.0001)
    y = np.full((8, len(t)), 2.0)
    y[2:5] += np.random.default_rng(19).normal(0, 0.3, (3, len(t)))
    found = detect(t, y, shot=1, core_channels=[2, 3, 4])
    assert found.observable[(t > 0.5) & (t < 1.5)].all()
    assert not found.absent_mask[(t > 0.5) & (t < 1.5)].any()
    assert found.absence_diagnostics["reason_samples"]["unresolved_core_noise"] > 0


@pytest.mark.parametrize(
    "crashes",
    [np.arange(0.4, 1.6, 0.012), [0.45, 0.92, 1.47]],
    ids=["rapid_periodic", "irregular"],
)
def test_significant_nonprofile_core_edges_are_not_absence(crashes):
    t = np.arange(0, 2, 0.0001)
    y = np.full((8, len(t)), 2.0)
    for crash in crashes:
        transient = np.where(t >= crash, np.exp(-(t - crash) / 0.003), 0)
        y[2:5] -= 0.2 * transient
    y += np.random.default_rng(25).normal(0, 0.001, y.shape)
    found = detect(t, y, shot=1, core_channels=[2, 3, 4])
    assert not found.intervals
    for crash in crashes:
        assert not found.absent_mask[np.abs(t - crash) < 0.2].any()
    assert found.absence_diagnostics["reason_samples"]["core_edge_ambiguous"] > 0


def test_equilibrium_uncertainty_is_decided_per_crash():
    t, y = synthetic()
    q = np.full(len(t), 0.8)
    q[np.abs(t - 0.16) < 0.001] = 1.2
    found = detect(t, y, shot=1, qmin=(t, q))
    assert [event.attrs["state"] for event in found.crashes] == [
        "present",
        "uncertain",
        "present",
        "present",
    ]


def test_steep_physical_core_peak_survives_channel_temperature_screen():
    t, _ = synthetic()
    level = np.array([1.5, 2, 4, 4.5, 4, 2, 1.5, 1.0])
    y = np.repeat(level[:, None], len(t), axis=1)
    for crash in (0.08, 0.16, 0.24, 0.32):
        transient = np.where(t >= crash, np.exp(-(t - crash) / 0.018), 0)
        y[2:5] -= 0.7 * transient
        y[5:7] += 0.4 * transient
    y += np.random.default_rng(31).normal(0, 0.003, y.shape)
    found = detect(t, y, shot=1, core_channels=range(7))
    assert len(found.crashes) == 4
    assert all(3 not in event.attrs["masked_channels"] for event in found.crashes)


def test_period_trains_do_not_bridge_gap_or_isolated_crash():
    assert trains([0.1, 0.2, 0.3, 0.8, 0.9, 1.0]) == [(0, 3), (3, 6)]
    assert trains([0.1, 0.2]) == []
    assert trains([0.1, 0.2, 0.21, 0.22]) == []


def test_nonuniform_sampling_and_missing_filter_support():
    t, y = synthetic()
    bad = t.copy()
    bad[100:] += 0.001
    with pytest.raises(ValueError, match="uniform"):
        detect(bad, y, shot=1)
    y[:, (t > 0.078) & (t < 0.082)] = np.nan
    found = detect(t, y, shot=1)
    assert len(found.crashes) == 3


def test_one_to_one_matching_and_shot_bootstrap():
    cells = event_cells([0.1, 0.2], [0.101, 0.102, 0.2], 2.0)
    assert cells.tolist() == [2, 1, 0]
    rows = [
        {"shot": 1, "cells": cells, "histogram": score_histogram([0, 1], [0.1, 0.9])}
    ]
    result = aggregate(rows, replicates=10)
    assert result["crash"]["f1"] == 0.8
    assert result["presence"]["auroc"] == 1
    assert result["ci95"]["crash_f1"] == [0.8, 0.8]


def test_models_shapes_gradients_and_gaussian_target():
    import torch

    torch.set_num_threads(2)
    for model, x, expected in (
        (HL3(), torch.randn(2, 4, 200), (2, 3)),
        (PhasePicker(), torch.randn(2, 48, 1000), (2, 3, 1000)),
    ):
        prediction = model(x)
        assert tuple(prediction.shape) == expected
        prediction.square().mean().backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
    t = np.arange(100) * 0.0001
    target = soft_crash_target(t, [0.005])
    assert target[50] == 1
    assert target[20] == 0


def test_interval_matches_do_not_count_multiple_fragments_as_multiple_recalled_spans():
    cells = interval_cells([(0.0, 1.0)], [(0.0, 0.3), (0.5, 0.8)], 0.1)
    assert cells.tolist() == [1, 1, 0]


def test_core_proxy_rejects_edges_without_calibrated_core_geometry():
    t, y = synthetic()
    assert not detect(t, y, shot=1, core_channels=[0, 1]).crashes


def test_presence_grid_does_not_depend_on_native_clock_roundoff():
    assert np.allclose(bin_times((0.0060000001, 0.012)), [0.007, 0.009, 0.011])


@pytest.mark.parametrize(
    "surface,reason",
    [(0.25, None), (np.nan, "no_q1_surface"), (0.8, "calibrated_direction")],
)
def test_calibrated_radius_acceptance_and_q1_conflicts(surface, reason):
    t, y = synthetic()
    positions = np.array([0.8, 0.7, 0.04, 0.08, 0.16, 0.36, 0.49, 0.64])
    psi = np.repeat(positions[:, None], 2, axis=1)
    geometry = Geometry(
        np.array([0.0, 400.0]),
        np.where(psi < 0.25, 0.8, 1.2),
        psi,
        np.array([surface, surface]),
    )
    found = detect(t, y, shot=1, geometry=geometry)
    if reason is None:
        assert len(found.crashes) == 4
        assert found.crashes[0].attrs["inversion_rho"] == pytest.approx(np.sqrt(0.26))
        assert found.crashes[0].attrs["q1_rho"] == 0.5
    elif reason == "no_q1_surface":
        assert len(found.crashes) == 4
        assert not found.intervals
        assert len(found.uncertain_intervals) == 1
        assert reason in found.uncertain_intervals[0].attrs["uncertainty_reasons"]
    else:
        assert not found.crashes
        assert found.rejected[reason] == 4
