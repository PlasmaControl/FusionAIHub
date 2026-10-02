import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.panels import sawtooth_oscillation as saw
from labeler.events.panels.ece_geometry import classify_q1
from labeler.features.store import FeatureArray, write_features


def test_q_only_metadata_classifies_bands_without_spatial_core_assumptions():
    q = np.array([[0.8, 1.2], [0.9, 0.8], [1.1, 0.9], [1.2, 1.1]])
    core, outside = classify_q1(q)
    assert core[:, 0].tolist() == [True, True, False, False]
    assert outside[:, 0].tolist() == [False, False, True, True]
    assert core[:, 1].tolist() == [False, True, True, False]
    assert outside[:, 1].tolist() == [True, False, False, True]


def test_reverse_shear_and_incomplete_profiles_cannot_establish_core_q1():
    profiles = np.array([
        [0.8, 1.2, 1.5], [1.2, 0.8, 1.5],
        [0.8, np.nan, 1.5], [0.8, 0.9, 0.95],
    ])
    surface = saw.ece_geometry.q1_surface([0, 0.5, 1], profiles)
    assert surface[0] == pytest.approx(0.25)
    assert np.isnan(surface[1:]).all()


def test_invalid_and_q1_channels_are_gaps():
    q = np.array([[0.8, np.nan], [1.0, 1.0], [1.2, 1.0]])
    core, outside = classify_q1(q)
    assert core[:, 0].tolist() == [True, False, False]
    assert outside[:, 0].tolist() == [False, False, True]
    assert not core[:, 1].any() and not outside[:, 1].any()


def test_align_q_does_not_interpolate_across_original_missing_sample():
    x = np.arange(5.0)
    q_x = np.arange(3.0)
    q = np.array([[0.8, np.nan, 1.2]])
    aligned = saw.ece_geometry.align_q(x, q_x, q)
    assert np.isnan(aligned[0, 1])


def test_psi_geometry_maps_against_qpsi_on_normalized_psi_grid():
    psi = np.array([[0.1, 0.1], [0.4, 0.4], [0.8, 0.8]])
    grid = np.array([0.0, 0.5, 1.0])
    qpsi = np.array([[0.8, 1.2, 1.4], [1.2, 0.8, 1.4]])
    q = saw.ece_geometry.q_from_psi(psi, grid, qpsi)
    np.testing.assert_allclose(q[:, 0], [0.88, 1.12, 1.32])
    np.testing.assert_allclose(q[:, 1], [1.12, 0.88, 1.16])


def test_mapping_leaves_a_missing_radial_profile_sample_unknown():
    q = saw.ece_geometry.q_from_psi(
        [[0.25], [0.5], [0.75]], [0, 0.5, 1], [[0.8, np.nan, 1.4]],
    )
    assert np.isnan(q).all()


def test_membership_does_not_extend_past_geometry_coverage_or_across_gaps():
    aligned = saw.ece_geometry.align_mask(
        [-1, 0, 0.5, 1, 2, 3, 4, 4.5, 5, 6],
        [0, 1, 4, 5], [[True, True, True, True]],
    )
    assert aligned[0].tolist() == [
        False, True, True, True, False, False, True, True, True, False,
    ]


def test_missing_geometry_slice_leaves_both_neighboring_intervals_unknown():
    aligned = saw.ece_geometry.align_mask(
        [0, 0.5, 1, 1.5, 2], [0, 1, 2], [[True, False, True]],
    )
    assert aligned[0].tolist() == [True, False, False, False, True]


def test_calibrated_psi_builds_core_and_outside_rows_without_reverse_shear_claims(
    tmp_path,
):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    geometry_t = np.array([0, 0.1, 0.2])
    grid = np.linspace(0, 1, 33)
    profiles = np.stack([0.8 + 0.8 * grid, 0.5 + 0.8 * grid, 1.2 - 0.4 * grid])
    psi = np.repeat(np.array([0.0, 0.2, 0.6, 0.9])[:, None], 3, axis=1)
    write_features(paths.features_file(11), 11, {
        "ece_psi": FeatureArray(geometry_t, psi),
        "qpsi": FeatureArray(geometry_t, profiles.T),
    }, {})
    t = np.arange(201) / 1000
    temperatures = np.repeat(np.array([10, 20, 30, 40])[:, None], len(t), axis=1)
    write_features(paths.corpus_file(11), 11, {
        "ece": FeatureArray(t, temperatures),
    }, {})
    core, outside = saw._mapped_ece_panels(11, paths=paths)
    assert "core" in core.title and "outside q=1" in outside.title
    assert core.legend == ["ch 0", "ch 1", "ch 2"]
    assert outside.legend == ["ch 2", "ch 3"]
    assert np.isfinite(core.y[0, 0])
    assert np.isnan(core.y[:, -1]).all(), "q0>1 is not an axis-connected core"
    assert np.isnan(outside.y[:, -1]).all()


def test_inversion_difference_keeps_all_nan_channels_as_gaps():
    x = np.arange(3.0)
    y = np.full((36, 3), np.nan)
    _, difference = saw.ece_geometry.inversion_difference(x, y)
    assert np.isnan(difference).all()


def test_unmapped_inversion_baseline_is_independent_of_the_requested_view(tmp_path):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    t = np.arange(101) / 1000
    values = np.zeros((36, len(t)))
    values[20:28] = np.linspace(0, 10, len(t))
    values[28:36] = -np.linspace(0, 10, len(t))
    write_features(paths.corpus_file(11), 11, {"ece": FeatureArray(t, values)}, {})
    whole = saw.panels(11, paths=paths)[-1]
    cropped = saw.panels(11, paths=paths, t_range=(20, 40))[-1]
    keep = (whole.x >= 20) & (whole.x <= 40)
    np.testing.assert_allclose(cropped.y, whole.y[:, keep])


def test_mapped_panels_keep_time_varying_core_and_outside_membership(monkeypatch):
    x = np.arange(4.0)
    values = np.tile(np.arange(36.0)[:, None], (1, 4))
    q = np.full((36, 4), 1.2)
    q[20:24, 0] = 0.8
    q[24:28, 1] = 0.8
    q[20:24, 2] = 0.8
    q[28:32, 3] = 0.8
    monkeypatch.setattr(
        saw, "raw_signal", lambda *args, **kwargs: FeatureArray(x=x, y=values, attrs={})
    )
    monkeypatch.setattr(saw.ece_geometry, "load_q", lambda shot, paths: (x, q))
    panels = saw._mapped_ece_panels(1)
    assert [panel.title for panel in panels] == [
        "ECE Te, q<1 inversion group (measured q geometry)",
        "ECE Te, q>1 inversion group (measured q geometry)",
    ]
    assert np.isfinite(panels[0].y[0, 0]) and np.isnan(panels[0].y[0, 1])
    assert np.isfinite(panels[0].y[4, 1]) and np.isnan(panels[0].y[4, 0])
