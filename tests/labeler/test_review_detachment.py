"""Detachment camera extraction, local panels, frame API and state roundtrips."""

from __future__ import annotations

import io
import json

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labeler.config import Paths
from labeler.events.panels import detachment as panels
from labeler.events.review import build, labels, rows, video
from labeler.events.review.rows import Grid
from labeler.events.ui.app import COOKIE, create_app


def corpus(paths, shot=170815):
    path = paths.corpus_file(shot)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as store:
        t = np.arange(13) * 0.02
        tangtv = store.create_group("tangtv")
        tangtv["xdata"] = t
        y = np.broadcast_to(np.arange(13)[None, :, None, None], (2, 13, 4, 6))
        y = y.astype(float).copy()
        y[1] = np.nan
        tangtv["ydata"] = y
        bolo = store.create_group("bolo")
        bolo["xdata"] = t
        bolo["ydata"] = np.ones((2, len(t)))  # traces are not images
        irtv = store.create_group("irtv")
        irtv["xdata"] = [0.0]
        irtv["ydata"] = np.full((7, 1), np.nan)
    return path


def test_frame_decimation_preserves_corpus_times_and_rejects_bad_clocks():
    assert video.frame_indices(np.arange(13) * 0.02).tolist() == [0, 3, 6, 9, 12]
    assert video.frame_indices([0, np.nan, 0.05, 0.1]).tolist() == [0, 2, 3]
    assert not len(video.frame_indices([0]))
    for clock in ([0, 0], [0.1, 0]):
        with pytest.raises(ValueError, match="increasing"):
            video.frame_indices(clock)
    with pytest.raises(ValueError, match="positive"):
        video.frame_indices([0, 1], max_fps=0)


def test_builder_extracts_frames_fixed_scale_stubs_and_movie_only_shots(tmp_path):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    corpus(paths)
    path = build.build("detachment", 170815, paths)
    manifest = video.meta(path)
    bolo, tangtv, irtv = manifest["cameras"]
    assert not bolo["channels"] and "traces" in bolo["reason"]
    assert not irtv["channels"] and "stub" in irtv["reason"]
    assert len(tangtv["channels"]) == 1  # all-NaN channel omitted
    channel = tangtv["channels"][0]
    np.testing.assert_allclose(channel["times_ms"], [0, 60, 120, 180, 240])
    assert channel["shape"] == [4, 6]
    with h5py.File(path) as store:
        assert store["videos/tangtv/0/frames"].chunks == (1, 4, 6)
        assert store["videos/tangtv/0/frames"].dtype == np.uint8
        assert store["videos/tangtv/0/frames"][0].max() == 0
        assert store["videos/tangtv/0/frames"][-1].min() == 255
        assert build.current(path, "detachment")
    # No scalar rows is also a valid store (the movie supplies its clock).
    with h5py.File(paths.corpus_file(170815), "a") as source:
        del source["bolo"]
    path = build.build("detachment", 170815, paths, force=True)
    assert rows.meta(path)["rows"] == []
    assert rows.meta(path)["t_range"][0] <= 0
    assert rows.meta(path)["t_range"][1] >= 240


def test_external_producer_shot_is_trimmed_to_cached_current_window(
    tmp_path, monkeypatch
):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    source = corpus(paths)
    cache = tmp_path / "producer_cache"
    cache.mkdir()
    monkeypatch.setenv("LABELER_DETACHMENT_GEOMETRY_ROOT", str(cache))
    np.savez(
        cache / "170815.npz",
        ipmeas__t=[0, 60, 120, 180, 240],
        ipmeas__y=[0, 600000, 600000, 600000, 0],
    )
    with h5py.File(source, "a") as store:
        group = store.create_group("co2")
        group["xdata"] = [0, 0.06, 0.12, 0.18, 0.24]
        group["ydata"] = np.ones((4, 5))
    path = build.build("detachment", 170815, paths)
    assert rows.meta(path)["t_range"] == [60, 180]
    np.testing.assert_allclose(
        video.meta(path)["cameras"][1]["channels"][0]["times_ms"], [60, 120, 180]
    )
    with h5py.File(path) as store:
        assert json.loads(store.attrs["params"])["plasma_window_source"] == (
            "cached |Ip| >= 300 kA"
        )
    assert rows.meta(path)["rows"]  # Scalar context is retained inside that window.


@pytest.mark.parametrize("time,index", [(-10, 0), (30, 0), (31, 1), (1000, 2)])
def test_nearest_frame_clamps_and_ties_choose_earlier(time, index):
    assert video.nearest_index([0, 60, 120], time) == index


def test_missing_cameras_malformed_shapes_and_nan_frames_are_explicit(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    source = corpus(paths)
    with h5py.File(source, "a") as store:
        store["tangtv/ydata"][:] = np.nan
        del store["irtv/ydata"]
        del store["bolo/ydata"]
        store["bolo/ydata"] = np.zeros((2, 4, 6))  # wrong clock length
    path = build.build("detachment", 170815, paths)
    reasons = [c["reason"] for c in video.meta(path)["cameras"]]
    assert reasons == [
        "camera shape and clock disagree",
        "no finite image frames",
        "camera has no xdata/ydata",
    ]
    missing = build.build("detachment", 7, paths)
    assert all(not c["channels"] for c in video.meta(missing)["cameras"])
    assert rows.meta(missing)["t_range"] == [0, 10000]


def test_three_dimensional_bolo_images_are_downsampled(tmp_path):
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as store:
        group = store.create_group("bolo")
        group["xdata"] = [0, 0.1]
        group["ydata"] = np.ones((2, 160, 240))
    path = tmp_path / "review.h5"
    rows.write(path, Grid(0, 100, 2), [], video_corpus=source)
    assert video.meta(path)["cameras"][0]["channels"][0]["shape"] == [80, 120]


def test_flattened_irtv_has_no_known_image_geometry(tmp_path):
    source = tmp_path / "flat.h5"
    with h5py.File(source, "w") as f:
        group = f.create_group("irtv")
        group["xdata"] = [0, 0.1]
        group["ydata"] = np.ones((7, 2, 24, 1))
    path = tmp_path / "review.h5"
    rows.write(path, Grid(0, 100, 2), [], video_corpus=source)
    camera = video.meta(path)["cameras"][2]
    assert not camera["channels"]
    assert "flattened" in camera["reason"] and "geometry" in camera["reason"]


def test_panel_sampling_and_indicator_validity(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    source = corpus(paths)
    with h5py.File(source, "a") as store:
        gas = store.create_group("gas_flow")
        gas["xdata"] = [-1, 0, 0.1, 0.2]
        gas["ydata"] = np.ones((11, 4))
        probes = store.create_group("langmuir")
        probes["xdata"] = np.arange(101) * 0.001
        probes["ydata"] = np.tile([[1], [3]], (1, 101))
    indicator_root = tmp_path / "indicators"
    indicator_root.mkdir()
    (indicator_root / "170815.csv").write_text(
        "t_ms,afrac,afrac_valid,prad_div,tangtv_front_height,"
        "tangtv_front_height_valid\n"
        "0,1,1,2,0.3,1\n50,99,0,3,9,0\n100,2,1,4,0.4,1\n"
    )
    monkeypatch.setenv("LABELER_DETACHMENT_INDICATORS", str(indicator_root))
    built = panels.panels(170815, paths=paths)
    by_title = {p.title: p for p in built}
    assert np.isnan(by_title["Afrac"].y[0, 1])
    assert np.isnan(by_title["TangTV front height (source not recorded)"].y[0, 1])
    assert "Divertor radiated power" not in by_title  # no validity mask
    assert not any(
        "Langmuir" in title or "Bolometer raw" in title for title in by_title
    )
    assert np.min(by_title["Gas flow"].x) >= 0


def test_context_block_means_do_not_alias_fast_signal(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    source = corpus(paths)
    with h5py.File(source, "a") as store:
        group = store.create_group("filterscopes")
        group["xdata"] = np.arange(40) * 0.000256
        group["ydata"] = np.tile([0, 2], (8, 20))
    built = panels.panels(170815, paths=paths, t_range=(0, 10))
    assert len(built) == 8  # Unknown chord locations must not share one scale.
    for i, panel in enumerate(built, 1):
        np.testing.assert_allclose(panel.y[:, :9], 1)
        np.testing.assert_allclose(np.diff(panel.x[:9]), 1.024)
        assert panel.legend == [f"FS{i:02d} (ph/(sr cm2 s))"]
        assert panel.ylabel == "ph/(sr cm2 s)"
        assert panel.title == f"D-alpha FS{i:02d} (location not recorded)"
        assert panel.metadata["location"] == "not recorded in corpus"
        assert panel.x[-1] <= 10


def test_density_falls_back_to_local_thomson_when_co2_is_a_stub(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    source = corpus(paths)
    with h5py.File(source, "a") as store:
        co2 = store.create_group("co2")
        co2["xdata"] = [0.0]
        co2["ydata"] = np.full((4, 1), np.nan)
        ts = store.create_group("ts_core_density")
        ts["xdata"] = [0.0, 0.1, 0.2]
        ts["ydata"] = np.array([[1e19, 2e19, 3e19], [np.nan, 4e19, 5e19]])
    density = [p for p in panels.panels(170815, paths=paths) if "density" in p.title]
    assert len(density) == 1
    panel = density[0]
    assert "Thomson" in panel.title and "local" in panel.title
    assert "not line-averaged" in panel.title
    assert panel.ylabel == "m^-3"
    np.testing.assert_allclose(panel.y, [[1e19, 2e19, 3e19], [np.nan, 4e19, 5e19]])
    assert panel.legend == ["core channel 0 (m^-3)", "core channel 1 (m^-3)"]


def test_usable_co2_density_keeps_the_line_average_instead_of_thomson(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    source = corpus(paths)
    with h5py.File(source, "a") as store:
        for name, channels in (("co2", 4), ("ts_core_density", 2)):
            group = store.create_group(name)
            group["xdata"] = [0.0, 0.1, 0.2]
            group["ydata"] = np.ones((channels, 3))
    density = [p for p in panels.panels(170815, paths=paths) if "density" in p.title]
    assert [p.title for p in density] == ["CO2 R0 density proxy (corpus DENUF)"]
    assert density[0].ylabel == "native units (unverified)"


@pytest.mark.parametrize("width", [20.0, 50.0, 100.0])
def test_indicator_bin_width_is_inferred_and_clipped_by_bin_overlap(
    tmp_path, monkeypatch, width
):
    starts = 100 + np.arange(3) * width
    np.savez(
        tmp_path / "170815.npz",
        start_ms=starts,
        afrac_value=[0.1, 99, 0.2],
        afrac_valid=[1, 0, 1],
    )
    monkeypatch.setenv("LABELER_DETACHMENT_INDICATORS", str(tmp_path))
    result = panels.indicator_panels(
        170815, Paths(root=tmp_path), t_range=(100, 100 + width / 4)
    )
    assert len(result) == 1  # First bin overlaps even though its centre is outside.
    panel = result[0]
    assert panel.metadata["bin_width_ms"] == width
    assert panel.metadata["trace_style"] == "step"
    assert panel.metadata["bin_start_ms"] == [100.0]
    assert panel.metadata["bin_end_ms"] == [100 + width / 4]
    np.testing.assert_allclose(panel.y, [[0.1]])


def test_video_store_drops_frames_outside_the_plasma_window(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    source = corpus(paths)
    path = paths.spectrogram_file("detachment", 170815)
    rows.write(path, Grid(40, 20, 6), [], video_corpus=source)
    channel = video.meta(path)["cameras"][1]["channels"][0]
    np.testing.assert_allclose(channel["times_ms"], [60, 120])
    with h5py.File(path) as store:
        assert store["videos/tangtv/0/source_indices"][:].tolist() == [3, 6]


def test_tangtv_manifest_describes_linear_resampling_instead_of_native_exposures(
    tmp_path,
):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    corpus(paths)
    manifest = video.meta(build.build("detachment", 170815, paths))
    note = manifest["cameras"][1]["sampling_note"]
    assert "50 Hz" in note and "linear" in note
    assert "adjacent exposures" in note


def test_actual_detach_bin_schema_keeps_gates_and_dimensionless_ratios(
    tmp_path, monkeypatch
):
    np.savez(
        tmp_path / "170815.npz",
        start_ms=[100, 150, 200],
        afrac_value=[0.1, 99, 0.2],
        afrac_valid=[1, 0, 1],
        prad_value=[0.3, 99, 0.4],
        prad_valid=[1, 0, 1],
        tangtv_value=[0.5, 99, 0.6],
        tangtv_valid=[1, 0, 1],
        tangtv_reason=["", "strike_on_floor", ""],
        tangtv_vote=[1, 0, 2],
    )
    monkeypatch.setenv("LABELER_DETACHMENT_INDICATORS", str(tmp_path))
    result = panels.indicator_panels(170815, Paths(root=tmp_path))
    assert len(result) == 3
    for panel in result:
        np.testing.assert_allclose(panel.x, [125, 175, 225])
        assert np.isnan(panel.y[0, 1])
        assert panel.ylabel == "dimensionless"
        assert panel.legend and "schema" in panel.metadata
    assert "fraction" in result[1].title
    assert "DZ" in result[2].title


def test_irtv_area_average_preserves_aspect(tmp_path):
    with h5py.File(tmp_path / "camera.h5", "w") as f:
        data = f.create_dataset("frames", data=np.tile([0, 2], (2, 512, 160)))
        frame = video._frame(data, 0, 0, (256, 320))
        assert frame.shape == (256, 160)
        np.testing.assert_allclose(frame, 1)


def test_manifest_names_views_including_unavailable_channels(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    corpus(paths)
    manifest = video.meta(build.build("detachment", 170815, paths))
    tangtv = manifest["cameras"][1]
    assert len(tangtv["views"]) == 7
    assert "LODIV_240RM1" in tangtv["channels"][0]["view_name"]
    assert tangtv["channels"][0]["region"] == "lower divertor"
    assert tangtv["default_channel"] == 0
    assert "not recorded" in tangtv["spectral_note"]


def test_manifest_prefers_lower_view_with_frames_in_plasma_window(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    source = corpus(paths)
    with h5py.File(source, "a") as f:
        del f["tangtv/ydata"]
        data = np.full((7, 13, 4, 6), np.nan)
        data[0, 0] = np.arange(24).reshape(4, 6)  # pre-plasma only
        data[2, 6] = np.arange(24).reshape(4, 6)  # inside plasma
        data[4, 6] = np.arange(24).reshape(4, 6)  # upper divertor
        f["tangtv/ydata"] = data
    path = paths.spectrogram_file("detachment", 170815)
    rows.write(path, Grid(100, 20, 3), [], video_corpus=source)
    assert video.meta(path)["cameras"][1]["default_channel"] == 2


def test_detachment_grid_and_context_clip_to_plasma_window(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    corpus(paths)
    monkeypatch.setattr(panels, "plasma_window", lambda *_: (40, 160))
    grid, _, _ = build.detachment.build("detachment", 170815, paths)
    assert grid.t0_ms == 40
    assert grid.t0_ms + grid.dt_ms * grid.n == pytest.approx(160)


def test_frame_endpoint_auth_roster_missing_cameras_and_detachment_save(tmp_path):
    paths = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        label_tables=tmp_path / "tables",
    )
    corpus(paths)
    event = paths.label_tables / "detachment"
    event.mkdir(parents=True)
    (event / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n170815,unverified,false,,,\n"
    )
    build.build("detachment", 170815, paths)
    with TestClient(create_app(paths, token="test")) as client:
        base = "/api/frame?event=detachment&shot=170815&camera=tangtv"
        assert client.get(base).status_code == 401
        client.cookies.set(COOKIE, "test")
        response = client.get(base + "&t_ms=95")
        assert (
            response.status_code == 200
            and response.headers["content-type"] == "image/png"
        )
        assert float(response.headers["x-frame-time-ms"]) == pytest.approx(120)
        assert Image.open(io.BytesIO(response.content)).size == (6, 4)
        assert client.get(base + "&t_ms=nan").status_code == 400
        assert client.get(base + "&channel=1").status_code == 404
        assert client.get(base.replace("170815", "999")).status_code == 404
        assert client.get(base.replace("tangtv", "../bolo")).status_code == 404
        assert client.get(base.replace("tangtv", "irtv")).status_code == 404
        described = client.get("/api/shot?event=detachment&shot=170815").json()
        assert described["video"]["cameras"][1]["channels"]
        categories = client.get("/api/events").json()["events"][0]["categories"]
        assert categories == {
            "1": "attached",
            "2": "detached",
            "3": "marfe",
            "4": "uncertain",
        }
        response = client.post(
            "/api/label",
            json={
                "event": "detachment",
                "shot": 170815,
                "window": [0, 240],
                "intervals": [[0, 60, 1], [60, 120, 2], [120, 180, 3], [180, 240, 4]],
            },
        )
        assert response.status_code == 200
        assert [s[2] for s in labels.read_saved(event)[170815].intervals] == [
            1,
            2,
            3,
            4,
        ]
