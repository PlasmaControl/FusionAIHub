"""Independent review provenance, target evidence and frozen-store safety."""

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.panels import detachment
from labeler.events.review import geometry, labels, producer, versions
from labeler.events.ui.app import Builds, LabelIn


@pytest.mark.parametrize(
    "reader", [producer.load, geometry.load, detachment.indicator_panels]
)
def test_missing_default_producer_context_fails_loudly(tmp_path, reader):
    with pytest.raises(FileNotFoundError, match="LABELER_DETACHMENT_"):
        reader(200977, Paths(root=tmp_path))


@pytest.mark.parametrize("version", [5, 6])
def test_stale_store_is_never_rebuilt_by_opening_it(tmp_path, version):
    paths = Paths(root=tmp_path)
    path = paths.spectrogram_file("detachment", 200977)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as store:
        store.attrs["panel_version"] = version
    original = path.read_bytes()
    builds = Builds(paths)
    try:
        result, error = builds.status("detachment", 200977)
        assert result is None and "resume" in error.lower()
        assert not builds.running
        assert path.read_bytes() == original
    finally:
        builds.pool.shutdown(wait=True)


@pytest.mark.parametrize(
    "shown,prefilled", [(False, False), (True, False), (True, True), (None, None)]
)
def test_every_saved_version_keeps_exposure_flags(tmp_path, shown, prefilled):
    event = tmp_path / "detachment"
    event.mkdir()
    body = LabelIn(
        event="detachment",
        shot=200977,
        window=(0, 100),
        intervals=[(0, 50, 4)],
        suggestions_shown=shown,
        prefilled=prefilled,
    )
    labels.save(
        event,
        body.shot,
        labels.normalise(body.window, body.intervals),
        source=None,
        suggestions_shown=body.suggestions_shown,
        prefilled=body.prefilled,
    )
    [entry] = versions.shot_versions(event, body.shot)
    assert entry["suggestions_shown"] is shown
    assert entry["prefilled"] is prefilled


def test_jsat_peak_keeps_probe_identity_and_missing_bins(tmp_path, detachment_inputs):
    root = tmp_path / "round4/detach/bins"
    np.savez(
        root / "200977.npz",
        start_ms=[0, 50, 100, 150],
        aux_jsat_peak=[3, np.nan, -1, 2],
        aux_jsat_probe=[12, 9, 7, 14],
    )
    [panel] = detachment.indicator_panels(200977, Paths(root=tmp_path))
    assert panel.metadata["quantity"] == "aux_jsat_peak"
    assert panel.metadata["probe_id"] == [12, None, None, 14]
    assert "probe" in panel.metadata["caveat"].lower()
    assert "rollover" in panel.metadata["caveat"].lower()
    assert np.isnan(panel.y[0, 1:3]).all()


def test_bolo_raw_chords_become_a_heatmap_not_power(tmp_path, detachment_inputs):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    path = paths.corpus_file(200977)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as store:
        group = store.create_group("bolo")
        group["xdata"] = np.arange(10) * 0.001
        group["ydata"] = np.arange(480).reshape(48, 10)
    [panel] = detachment.panels(200977, paths=paths)
    assert panel.kind == "heatmap" and panel.z.shape == (48, 10)
    assert "voltage" in panel.title.lower()
    assert "power" in panel.metadata["caveat"]
    with h5py.File(path, "a") as store:
        del store["bolo/ydata"]
        voltage = np.arange(480, dtype=float).reshape(48, 10)
        voltage[[0, 12]] = np.nan
        store["bolo/ydata"] = voltage
    [panel] = detachment.panels(200977, paths=paths)
    assert panel.z.shape == (48, 10)
    assert np.isnan(panel.z[[0, 12]]).all()
    np.testing.assert_array_equal(panel.z[13], voltage[13])
