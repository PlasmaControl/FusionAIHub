"""GET /api/slices and /api/slice: E's contract (plan D.B), 200, 404 and 422."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labeler.config import Paths
from labeler.events.ui.app import COOKIE, create_app

from .test_review_slices import SHOT, write_equilibrium, write_inversion


@pytest.fixture
def client(tmp_path):
    paths = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        label_tables=tmp_path / "tables",
    )
    for event in ("detachment", "edge_localized_mode"):
        directory = paths.label_tables / event
        directory.mkdir(parents=True)
        (directory / "shots.csv").write_text(
            "shot,tier,holdout,reviewers,verified_on,notes\n"
        )
    write_equilibrium(paths)
    write_inversion(paths)
    with TestClient(create_app(paths, token="test")) as client:
        client.cookies.set(COOKIE, "test")
        yield client


def test_slices_lists_the_kinds_and_any_event_may_ask(client):
    kinds = client.get(f"/api/slices?event=edge_localized_mode&shot={SHOT}").json()
    assert [k["kind"] for k in kinds["kinds"]] == ["equilibrium", "inversion"]
    assert kinds["kinds"][0] == {
        "kind": "equilibrium", "label": "EFIT equilibrium", "n_t": 3,
        "t_range_ms": [100.0, 300.0],
    }
    assert client.get("/api/slices?event=detachment&shot=1").json() == {"kinds": []}


def test_equilibrium_answers_json_in_metres(client):
    response = client.get(
        f"/api/slice?event=detachment&shot={SHOT}&kind=equilibrium&t_ms=180"
    )
    assert response.status_code == 200
    view = response.json()
    assert view["t_ms"] == 200.0 and view["source"] == "EFIT02"
    assert view["xpoints"] == [[1.4, -1.1]] and view["strikes"] == [[1.5, -1.25]]
    assert [level["psi_n"] for level in view["levels"]] == [
        0.2, 0.4, 0.6, 0.8, 1.0, 1.02, 1.05,
    ]


def test_inversion_answers_png_with_x_slice(client):
    response = client.get(
        f"/api/slice?event=detachment&shot={SHOT}&kind=inversion&t_ms=1001"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    record = json.loads(response.headers["x-slice"])
    assert set(record) == {
        "t_ms", "r_range", "z_range", "vmin", "vmax", "units", "source",
    }
    assert record["t_ms"] == 1000.0
    pixels = np.asarray(Image.open(io.BytesIO(response.content)))
    assert pixels.shape == (3, 4) and pixels[0, 0] > pixels[2, 0]


@pytest.mark.parametrize(
    "query",
    [
        "event=detachment&shot=1&kind=equilibrium&t_ms=100",  # nothing on disk
        f"event=detachment&shot={SHOT}&kind=equilibrium&t_ms=500",  # 200 ms gap
        f"event=detachment&shot={SHOT}&kind=inversion&t_ms=900",  # 100 ms gap
    ],
)
def test_missing_slices_answer_404(client, query):
    response = client.get(f"/api/slice?{query}")
    assert response.status_code == 404 and response.json()["error"]


@pytest.mark.parametrize(
    "query",
    [
        f"event=detachment&shot={SHOT}&kind=equilibrium&t_ms=100.5",
        f"event=detachment&shot={SHOT}&kind=equilibrium&t_ms=soon",
        "event=detachment&shot=abc&kind=equilibrium&t_ms=100",
        f"event=detachment&shot={SHOT}&kind=profile&t_ms=100",
        f"event=nonsense&shot={SHOT}&kind=equilibrium&t_ms=100",
        f"event=..&shot={SHOT}&kind=equilibrium&t_ms=100",
        f"event=detachment&shot={SHOT}&kind=equilibrium",
    ],
)
def test_bad_parameters_answer_422(client, query):
    response = client.get(f"/api/slice?{query}")
    assert response.status_code == 422 and response.json()["error"]


def test_slices_rejects_a_bad_event_or_shot(client):
    assert client.get(f"/api/slices?event=nonsense&shot={SHOT}").status_code == 422
    assert client.get("/api/slices?event=detachment&shot=x").status_code == 422
    client.cookies.clear()
    assert client.get(f"/api/slices?event=detachment&shot={SHOT}").status_code == 401


def test_a_broken_inversion_is_unlisted_and_answers_404(client, tmp_path):
    broken = tmp_path / "root/round4/detach/inversions" / f"{SHOT}.npz"
    broken.write_bytes(b"not an npz")
    kinds = client.get(f"/api/slices?event=detachment&shot={SHOT}").json()["kinds"]
    assert [k["kind"] for k in kinds] == ["equilibrium"]
    response = client.get(
        f"/api/slice?event=detachment&shot={SHOT}&kind=inversion&t_ms=1000"
    )
    assert response.status_code == 404 and "unreadable" in response.json()["error"]
