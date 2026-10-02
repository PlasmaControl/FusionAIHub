"""Diamagnetic-loop calibration and D-alpha-corroborated ELM losses."""

import json

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events import elm_energy
from labeler.events.panels import edge_localized_mode as elm
from labeler.events.review import panel_rows
from labeler.events.verify import NoDataError
from labeler.features.store import FeatureArray, write_features


def _shot(*, polarity=1):
    t = np.arange(-40.0, 340.01, 0.1)
    energy = np.where((t >= 0) & (t < 300), 1.0e6, 0.0)
    for peak, loss in ((100.0, 50_000.0), (200.0, 30_000.0)):
        energy[(t >= peak) & (t < peak + 10)] -= loss
    gain = polarity * 4.0e7
    loop = energy / gain + 2.0 + 0.001 * t
    ef_t = np.arange(10.0, 300, 20.0)
    ef_w = np.interp(ef_t, t, energy)
    return t, energy, loop, ef_t, ef_w, gain


def test_drift_uses_quiet_windows_and_preserves_plasma_energy_evolution():
    t, energy, loop, _ef_t, _ef_w, gain = _shot()
    corrected, fit = elm_energy.correct_drift(
        t, loop, ((-40, -10), (310, 340)),
    )
    np.testing.assert_allclose(corrected * gain, energy, atol=1e-6)
    assert fit.slope_per_ms == pytest.approx(0.001)
    assert fit.intercept == pytest.approx(2)


@pytest.mark.parametrize("polarity", [1, -1])
def test_scale_recovers_energy_for_either_loop_polarity(polarity):
    t, energy, loop, ef_t, ef_w, gain = _shot(polarity=polarity)
    corrected, _ = elm_energy.correct_drift(t, loop, ((-40, -10), (310, 340)))
    scaled, fit = elm_energy.scale_to_efit(
        t, corrected, ef_t, ef_w, calibration_window_ms=(10, 290),
    )
    assert fit.gain_j_per_unit == pytest.approx(gain)
    assert fit.matched_samples == len(ef_t)
    np.testing.assert_allclose(scaled, energy, atol=1e-6)


def test_pipeline_detects_loop_drops_matches_dalpha_and_measures_loss():
    t, energy, loop, ef_t, ef_w, _ = _shot()
    analysis = elm_energy.analyze(
        t, loop, ef_t, ef_w, [100, 200, 250],
        baseline_windows_ms=((-40, -10), (310, 340)),
        calibration_window_ms=(10, 290),
    )
    np.testing.assert_allclose(analysis.energy_j, energy, atol=1e-6)
    assert [loss.dalpha_time_ms for loss in analysis.losses] == [100, 200]
    assert [loss.loss_j for loss in analysis.losses] == pytest.approx([50_000, 30_000])
    assert [loss.fraction for loss in analysis.losses] == pytest.approx([0.05, 0.03])
    assert analysis.unmatched_dalpha_ms == (250,)
    assert analysis.metadata()["drift"]["baseline_windows_ms"] == [
        [-40.0, -10.0], [310.0, 340.0],
    ]


def test_dalpha_alone_or_an_unmatched_energy_drop_cannot_produce_a_size():
    t, energy, _loop, _ef_t, _ef_w, _ = _shot()
    assert elm_energy.detect_losses(t, energy, [150, 250])[0] == ()
    assert elm_energy.detect_losses(t, np.ones(len(t)) * 1e6, [100])[0] == ()


def test_long_quiet_record_cannot_lower_the_plasma_loss_noise_threshold():
    t = np.arange(-1000.0, 6000.01, 0.1)
    plasma = (t >= 0) & (t < 1000)
    # Smooth fluctuations, with no crashes. Quiet samples dominate the full
    # record but do not describe the noise where energy losses are measured.
    energy = np.where(plasma, 1e6 + 4000 * np.sin(2 * np.pi * t / 8), 0.0)
    loop = energy / 4e7 + 0.3 + 0.0001 * t
    ef_t = np.arange(100.0, 901, 20)
    analysis = elm_energy.analyze(
        t, loop, ef_t, np.interp(ef_t, t, energy), [108, 308, 508, 708],
        baseline_windows_ms=((-1000, -900), (5900, 6000)),
        calibration_window_ms=(100, 900),
    )
    assert analysis.losses == ()
    assert analysis.unmatched_dalpha_ms == (108, 308, 508, 708)


@pytest.mark.parametrize("window,matched", [
    ((10, 150), [100]),
    ((99, 150), []),  # The first drop lacks a full pre-window in this domain.
    ((150, 290), [200]),
])
def test_sizes_require_complete_support_inside_the_calibrated_regime(window, matched):
    t, energy, loop, ef_t, ef_w, _ = _shot()
    analysis = elm_energy.analyze(
        t, loop, ef_t, ef_w, [100, 200, 250],
        baseline_windows_ms=((-40, -10), (310, 340)),
        calibration_window_ms=window,
    )
    assert [loss.dalpha_time_ms for loss in analysis.losses] == matched
    assert analysis.unmatched_dalpha_ms == tuple(
        peak for peak in (100, 200, 250) if peak not in matched
    )
    np.testing.assert_array_equal(analysis.time_ms, t)
    np.testing.assert_allclose(analysis.energy_j, energy, atol=1e-6)
    assert analysis.metadata()["measurement_window_ms"] == list(window)


def test_energy_gap_through_the_crash_leaves_the_event_unsized():
    t, energy, _loop, _ef_t, _ef_w, _ = _shot()
    energy[(t > 98) & (t < 102)] = np.nan
    losses, _ = elm_energy.detect_losses(t, energy, [100, 200])
    assert [loss.dalpha_time_ms for loss in losses] == [200]


def test_unresolved_neighboring_dalpha_peaks_do_not_get_a_shared_loss_size():
    t, energy, _loop, _ef_t, _ef_w, _ = _shot()
    losses, _ = elm_energy.detect_losses(t, energy, [100, 101, 200])
    assert [loss.dalpha_time_ms for loss in losses] == [200]


def test_one_dalpha_peak_with_two_possible_loop_drops_stays_unresolved():
    t = np.arange(-10.0, 10.001, 0.1)
    energy = np.where(t < 1.0, 100.0, np.where(t < 4.1, 90.0, 80.0))
    losses, unmatched = elm_energy.detect_losses(t, energy, [-2.0, 2.4])
    # Matching the first loop drop to -2 would size another D-alpha event
    # inside its support; choosing the nearest shared peak also cannot
    # establish which of the two observed drops belongs to it.
    assert losses == ()
    assert unmatched == (-2.0, 2.4)


def test_two_dalpha_peaks_near_one_loop_drop_stay_unresolved():
    t, energy, _loop, _ef_t, _ef_w, _ = _shot()
    losses, unmatched = elm_energy.detect_losses(t, energy, [97.5, 102.5, 200])
    assert [loss.dalpha_time_ms for loss in losses] == [200]
    assert unmatched == (97.5, 102.5)


def test_calibration_does_not_bridge_a_loop_gap_or_extrapolate():
    t = np.arange(11.0)
    loop = np.ones(len(t))
    loop[4:7] = np.nan
    _scaled, fit = elm_energy.scale_to_efit(
        t, loop, [-1, 1, 3, 5, 7, 9, 11], np.ones(7) * 10,
        calibration_window_ms=(-1, 11),
    )
    assert fit.matched_samples == 4


@pytest.mark.parametrize("windows", [((-40, -10),), ((-40, -10), (-20, -5))])
def test_drift_requires_distinct_quiet_windows(windows):
    t, _energy, loop, _ef_t, _ef_w, _ = _shot()
    with pytest.raises(ValueError, match="baseline"):
        elm_energy.correct_drift(t, loop, windows)


def test_calibration_refuses_missing_or_inconsistent_scale():
    t = np.arange(8.0)
    with pytest.raises(ValueError, match="calibration"):
        elm_energy.scale_to_efit(
            t, np.zeros(len(t)), t, np.ones(len(t)),
            calibration_window_ms=(0, 7),
        )
    with pytest.raises(ValueError, match="polarity"):
        elm_energy.scale_to_efit(
            t, np.array([1, -1] * 4), t, np.ones(len(t)),
            calibration_window_ms=(0, 7),
        )


def test_measurement_records_parameters_and_native_clock():
    t, _energy, loop, ef_t, ef_w, _ = _shot()
    analysis = elm_energy.analyze(
        t, loop, ef_t, ef_w, [100],
        baseline_windows_ms=((-40, -10), (310, 340)),
        calibration_window_ms=(10, 290),
    )
    np.testing.assert_array_equal(analysis.time_ms, t)
    metadata = analysis.metadata()
    assert metadata["method"] == "diamagnetic_loop_v2"
    assert metadata["calibration"]["matched_samples"] == len(ef_t)
    assert metadata["settings"]["match_tolerance_ms"] > 0


def _write_shot(tmp_path):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    t, _energy, loop, ef_t, ef_w, _ = _shot()
    dalpha = np.ones((8, len(t)))
    for peak in (100, 200, 250):
        dalpha[0] += np.exp(-0.5 * ((t - peak) / 0.3) ** 2)
    write_features(paths.features_file(11), 11, {
        "diamagnetic_loop": FeatureArray(t / 1000, loop[None], attrs={
            "units": "Wb", "source": "test-loop",
            "baseline_windows_ms": json.dumps([[-40, -10], [310, 340]]),
            "calibration_window_ms": json.dumps([10, 290]),
        }),
        "wmhd": FeatureArray(ef_t / 1000, ef_w[None]),
    }, {})
    write_features(paths.corpus_file(11), 11, {
        "filterscopes": FeatureArray(t / 1000, dalpha),
    }, {})
    return paths


def test_labeler_reads_local_loop_and_retains_calibration_provenance(tmp_path):
    paths = _write_shot(tmp_path)
    analysis = elm_energy.load(11, paths)
    # Feature stores retain float32 loop amplitudes, including their offset.
    assert [loss.loss_j for loss in analysis.losses] == pytest.approx(
        [50_000, 30_000], rel=1e-4,
    )
    assert analysis.metadata()["provenance"]["loop_units"] == "Wb"
    assert analysis.metadata()["provenance"]["dalpha_channel"] == "FS01"


def test_elm_panel_replaces_direct_efit_with_loop_energy_and_size_rows(
    tmp_path, monkeypatch,
):
    paths = _write_shot(tmp_path)
    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    built = elm.panels(11, paths=paths)
    energy, loss, fraction = built[-3:]
    assert "diamagnetic" in energy.title.lower() and energy.ylabel == "J"
    assert "Stored energy, EFIT01 WMHD" not in [p.title for p in built]
    assert loss.ylabel == "kJ" and np.nanmax(loss.y) == pytest.approx(50, rel=1e-4)
    assert fraction.ylabel == "%" and np.nanmax(fraction.y) == pytest.approx(5, rel=1e-4)
    assert energy.metadata["method"] == "diamagnetic_loop_v2"
    cropped = elm.energy_panels(11, paths=paths, t_range=(90, 110))
    keep = (energy.x >= 90) & (energy.x <= 110)
    np.testing.assert_allclose(cropped[0].y, energy.y[:, keep])
    assert np.max(cropped[1].y) == pytest.approx(50, rel=1e-4)
    assert cropped[0].metadata == energy.metadata
    _grid, _rows, info = panel_rows.build("edge_localized_mode", 11, paths)
    assert any(m["method"] == "diamagnetic_loop_v2"
               for m in info["params"]["panel_metadata"].values())


def test_wmhd_alone_does_not_fabricate_a_fast_loop_energy_row(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    write_features(paths.features_file(11), 11, {
        "wmhd": FeatureArray(np.arange(4.0), np.ones((1, 4)) * 1e6),
    }, {})
    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    assert elm.energy_panels(11, paths=paths) == []


def test_missing_baseline_metadata_refuses_drifting_loop_energy(tmp_path):
    paths = _write_shot(tmp_path)
    t, _energy, loop, _ef_t, _ef_w, _ = _shot()
    write_features(paths.features_file(11), 11, {
        "diamagnetic_loop": FeatureArray(t / 1000, loop[None]),
    }, {})
    with pytest.raises(NoDataError, match="baseline_windows_ms"):
        elm_energy.load(11, paths)


def test_energy_rows_leave_uncalibrated_time_unavailable(tmp_path):
    paths = _write_shot(tmp_path)
    built = elm.energy_panels(11, paths=paths)
    assert len(built) == 3
    for panel in built:
        outside = (panel.x < 10) | (panel.x > 290)
        assert outside.any()
        assert np.isnan(panel.y[:, outside]).all()
        assert np.isfinite(panel.y[:, ~outside]).all()


@pytest.mark.parametrize("shape", [(10,), (1, 9)])
def test_malformed_loop_store_keeps_other_diagnostics_usable(tmp_path, shape):
    paths = _write_shot(tmp_path)
    with h5py.File(paths.features_file(11), "a") as store:
        group = store["diamagnetic_loop"]
        del group["ydata"]
        group.create_dataset("ydata", data=np.zeros(shape))
    with pytest.raises(NoDataError, match="malformed.*diamagnetic_loop"):
        elm_energy.load(11, paths)
    assert elm.energy_panels(11, paths=paths) == []
