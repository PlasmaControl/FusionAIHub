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
