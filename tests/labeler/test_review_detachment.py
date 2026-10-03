"""Detachment camera extraction, local panels, frame API and state roundtrips."""

from __future__ import annotations

import io

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


def test_frame_decimation_preserves_native_times_and_rejects_bad_clocks():
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
    assert np.isnan(by_title["TangTV front height"].y[0, 1])
    assert "Divertor radiated power" not in by_title  # no validity mask
    probes = by_title["Langmuir raw channel median (10 ms samples)"]
    assert np.min(np.diff(probes.x)) >= 10
    assert np.all(probes.y == 2)
    assert np.min(by_title["Gas flow (corpus channels)"].x) >= 0


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
