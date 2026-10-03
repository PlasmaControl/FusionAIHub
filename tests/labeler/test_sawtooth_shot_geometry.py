"""Per-shot ECE proxies do not manufacture unavailable calibration."""

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.features.store import FeatureArray, write_features
from labeler.sawtooth.geometry import (
    RadiusGeometry,
    load_radius_geometry,
    nominal_frequencies,
    radius_evidence,
    second_harmonic_R,
    select_core,
)


def test_frequency_metadata_requires_rf_units_and_one_value_per_channel(tmp_path):
    with h5py.File(tmp_path / "metadata.h5", "w") as file:
        group = file.create_group("ece")
        group.attrs["frequencies"] = [100, 110, 120]
        assert nominal_frequencies(file, 3) == (None, None)
        group.attrs["frequency_units"] = "GHz"
        frequency, source = nominal_frequencies(file, 3)
        np.testing.assert_allclose(frequency, [100e9, 110e9, 120e9])
        assert source.endswith("/ece@frequencies")
        assert nominal_frequencies(file, 4) == (None, None)


def test_nominal_radius_uses_field_reference_not_magnetic_axis_and_keeps_gaps():
    target = np.array([-1, 0, 0.5, 1, 2, 3, 4, 4.5, 5, 6.0])
    field = (np.array([0, 1, 4, 5.0]), np.full(4, -2.0))
    radius = second_harmonic_R(target, [100e9, 120e9], field, 1.7)
    expected = 2 * 27.992e9 * 2 * 1.7 / 100e9
    assert radius[0, 2] == pytest.approx(expected)
    assert np.isnan(radius[:, [0, 4, 5, 9]]).all()
    assert radius[0, 6] == pytest.approx(expected)
    assert radius[1, 2] == pytest.approx(expected * 100 / 120)


def test_missing_frequency_and_field_reference_are_explicit_not_guessed(tmp_path):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    clock = np.array([0, 0.1, 0.2])
    write_features(
        paths.corpus_file(1),
        1,
        {
            "ece": FeatureArray(clock, np.ones((3, 3))),
        },
        {},
    )
    write_features(
        paths.features_file(1),
        1,
        {
            "bt": FeatureArray(clock, np.full((1, 3), -2.0)),
            "r0": FeatureArray(clock, np.full((1, 3), 1.8)),
        },
        {},
    )
    radius, info = load_radius_geometry(1, clock, 3, paths)
    assert radius is None
    assert info["status"] == "frequency_metadata_unavailable"
    with h5py.File(paths.corpus_file(1), "a") as file:
        file["ece"].attrs["frequency_ghz"] = [100, 110, 120]
    radius, info = load_radius_geometry(1, clock, 3, paths)
    assert radius is None
    assert info["status"] == "bt_reference_radius_unavailable"


def test_local_metadata_mapping_records_exact_sources(tmp_path):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    clock = np.array([0, 0.1, 0.2])
    write_features(
        paths.corpus_file(1),
        1,
        {
            "ece": FeatureArray(
                clock,
                np.ones((3, 3)),
                {
                    "frequency_ghz": "[100,110,120]",
                },
            ),
        },
        {},
    )
    write_features(
        paths.features_file(1),
        1,
        {
            "bt": FeatureArray(
                clock,
                np.full((1, 3), -2.0),
                {
                    "reference_radius_m": "1.7",
                    "locator": "bt",
                },
            ),
            "r0": FeatureArray(
                clock,
                np.full((1, 3), 1.8),
                {
                    "locator": "EFIT01:rmaxis",
                },
            ),
        },
        {},
    )
    radius, info = load_radius_geometry(1, clock, 3, paths)
    assert info["status"] == "nominal_second_harmonic_R"
    assert info["frequency_source"].endswith("/ece@frequency_ghz")
    assert "@reference_radius_m" in info["field_reference_source"]
    np.testing.assert_allclose(radius.axis_R_m, 1.8)
    assert radius.R_m[0, 1] == pytest.approx(2 * 27.992e9 * 2 * 1.7 / 100e9)


def test_hottest_physical_proxy_moves_with_shot_and_rejects_harmonic_hot_tail():
    channels = np.arange(48)
    level = 0.2 + 6 * np.exp(-(((channels - 12) / 7) ** 2))
    level[40:47] = 12
    values = np.repeat(level[:, None], 300, axis=1)
    core = select_core(values)
    assert 12 in core.core_channels
    assert max(core.core_channels) < 20
    assert not core.physical_channels[40:47].any()
    assert set(core.core_channels).isdisjoint(core.outer_channels)
    assert core.info["radius_status"] == "unavailable"
    assert core.info["central_channel"] == 12


def test_an_isolated_nonthermal_spike_does_not_define_core():
    channels = np.arange(48)
    level = 0.2 + 5 * np.exp(-(((channels - 27) / 7) ** 2))
    level[3] = 50
    core = select_core(np.repeat(level[:, None], 100, axis=1))
    assert 27 in core.core_channels
    assert not core.physical_channels[3]
    assert 3 not in core.core_channels


def test_same_shot_archive_maps_field_product_and_q1_with_transposed_grid(tmp_path):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    archive = tmp_path / "archive"
    (archive / "raw").mkdir(parents=True)
    clock = np.array([0, 0.1, 0.2])
    r = np.array([0.65, 1, 1.35, 1.7, 2.05, 2.4, 2.75])
    z = np.array([-0.7, -0.35, 0, 0.35, 0.7])
    psi = ((r[None] - 1.7) / 0.7) ** 2 + (z[:, None] / 0.35) ** 2
    with h5py.File(archive / "raw/1.h5", "w") as file:
        file.attrs["shot"] = 1
        ece = file.create_group("ece")
        for index in range(3):
            ece.create_dataset(f"ECEVS{index + 1:02d}", data=[0])
        setup = file.create_group("ecegeom")
        setup.attrs["source"] = "ELECTRONS"
        setup.create_dataset(
            "FREQ", data=2 * 27.992 * 3.4 / np.array([1.35, 1.7, 2.05])
        )
        setup.create_dataset("ECEZH", data=0.0)
        eq = file.create_group("eq")
        eq.attrs["source"] = "efit01"
        for name, values in {
            "gtime": clock * 1000,
            "r": r,
            "z": z,
            "rmaxis": np.full(3, 1.7),
            "zmaxis": np.zeros(3),
            "ssimag": np.zeros(3),
            "ssibry": np.ones(3),
            "fpol": np.full((3, 5), -3.4),
            "qpsi": np.repeat((0.8 + 0.8 * np.linspace(0, 1, 5))[None], 3, axis=0),
            "psirz": np.repeat(psi.T[None], 3, axis=0),
        }.items():
            eq.create_dataset(name, data=values)
    radius, info = load_radius_geometry(1, clock, 3, paths, archive_root=archive)
    assert info["frequency_scope"] == "same_shot"
    assert info["field_product_source"].endswith("/eq/fpol[:, -1] (F=R*Bphi)")
    assert not info["calibrated_flux"]
    np.testing.assert_allclose(radius.R_m[:, 1], [1.35, 1.7, 2.05])
    evidence = radius_evidence(radius, 0.1, 1.5)
    assert evidence["inversion_R_m"] == pytest.approx(1.875)
    assert evidence["q1_R_m"] == pytest.approx(2.05)
    assert evidence["q1_radius_difference_m"] == pytest.approx(-0.175)
    missing, missing_info = load_radius_geometry(
        2, clock, 3, paths, archive_root=archive
    )
    assert (
        missing is None and missing_info["status"] == "frequency_metadata_unavailable"
    )


def test_radius_evidence_does_not_cross_missing_calibration_or_coverage():
    radius = RadiusGeometry(
        np.array([0, 1.0]),
        np.array([[1.4, 1.4], [np.nan, np.nan], [2, 2]]),
        np.array([1.7, 1.7]),
    )
    assert radius_evidence(radius, 0.5, 0.5)["inversion_R_m"] is None
    assert radius_evidence(radius, -1, 0)["inversion_R_m"] is None
