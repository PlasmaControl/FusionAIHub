"""Authenticated HTTP transport for actual immutable design revisions."""

import io

import pytest
import torch
from fastapi.testclient import TestClient

from shot_design.ui.app import COOKIE, create_app

from .test_program import SHOT, averaging_source, program_source, vertices  # noqa: F401


@pytest.fixture
def client(program_source):  # noqa: F811 - imported shared pytest fixture
    paths, _ = program_source
    with TestClient(create_app(paths=paths, token="secret")) as client:
        client.cookies.set(COOKIE, "secret")
        yield client


def test_save_reopen_download(client):
    body = {"reference_shot": SHOT, "edits": {"pinj[0]": vertices()}}
    preview = client.post("/api/design/preview", json=body)
    assert preview.status_code == 200
    assert preview.json()["validation"]["can_export"]
    saved = client.post("/api/design", json=preview.json()["program"])
    assert saved.status_code == 200
    ident = saved.json()["program"]["id"]
    assert client.get(f"/api/design/{ident}").json() == saved.json()
    assert client.get("/api/design").json()[0]["id"] == ident
    editable = client.get(f"/api/design/{ident}/json")
    assert editable.json() == saved.json()["program"]
    exported = client.get(f"/api/design/{ident}/ignite")
    assert exported.status_code == 200
    assert f"design-{ident}.pt" in exported.headers["content-disposition"]
    payload = torch.load(io.BytesIO(exported.content), weights_only=True)
    assert payload["actuators"][20, 12].item() == pytest.approx(-0.6, abs=0.001)


def test_invalid_draft_saves_but_download_refuses(client):
    saved = client.post(
        "/api/design",
        json={"reference_shot": SHOT, "edits": {"no-such-channel": vertices()}},
    )
    assert saved.status_code == 200
    ident = saved.json()["program"]["id"]
    refused = client.get(f"/api/design/{ident}/ignite")
    assert refused.status_code == 422
    assert "Unknown" in str(refused.json())


def test_all_design_routes_require_authentication(client):
    client.cookies.clear()
    for method, path in [
        ("get", "/api/design"),
        ("post", "/api/design"),
        ("post", "/api/design/preview"),
        ("post", "/api/design/prepare"),
        ("post", "/api/design/merge"),
        ("get", "/api/design/x"),
        ("get", "/api/design/x/json"),
        ("get", "/api/design/x/ignite"),
    ]:
        assert getattr(client, method)(path).status_code == 401


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"reference_shot": "../123"},
        {"reference_shot": SHOT, "start_s": "NaN"},
        {"reference_shot": SHOT, "comparison_shots": [1, 1]},
        {"reference_shot": SHOT, "schema_version": "unknown"},
        {"reference_shot": SHOT, "edits": {"pinj[0]": [{"t_s": 1, "y": "Infinity"}]}},
    ],
)
def test_body_errors_are_422(client, body):
    assert client.post("/api/design/preview", json=body).status_code == 422


def test_unknown_and_unsafe_ids_are_not_filesystem_paths(client):
    for ident in ["unknown", "0" * 32, "..%2Fsecret"]:
        assert client.get(f"/api/design/{ident}").status_code == 404


def test_uncached_shot_preview_and_save_work_without_encoding(client, program_source):  # noqa: F811
    paths, _ = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    preview = client.post("/api/design/preview", json={"reference_shot": SHOT})
    assert preview.status_code == 200
    assert preview.json()["channels"]
    assert preview.json()["validation"]["needs_seed"]
    saved = client.post("/api/design", json=preview.json()["program"])
    assert saved.status_code == 200
    ident = saved.json()["program"]["id"]
    assert client.get(f"/api/design/{ident}/json").status_code == 200
    assert client.get(f"/api/design/{ident}/ignite").status_code == 422


def test_preparation_errors_are_actionable(client, program_source, monkeypatch):  # noqa: F811
    from shot_design.design import seed

    paths, _ = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()

    def unavailable(*args, **kwargs):
        raise RuntimeError("codec unavailable")

    monkeypatch.setattr(seed, "encode_frame_codes", unavailable)
    result = client.post("/api/design/prepare", json={"reference_shot": SHOT})
    assert result.status_code == 422
    assert "codec unavailable" in str(result.json())
    assert not list((paths.ignite_inputs_dir / "reference_cache").glob("*.pt"))


def test_source_drift_refuses_download_and_new_revision(client, program_source):  # noqa: F811
    paths, cache = program_source
    saved = client.post("/api/design", json={"reference_shot": SHOT}).json()
    ident = saved["program"]["id"]
    cache["codes"]["ece"][0, 0] = 1
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    reopened = client.get(f"/api/design/{ident}").json()
    assert not reopened["validation"]["can_export"]
    assert client.get(f"/api/design/{ident}/ignite").status_code == 422
    assert client.post("/api/design", json=saved["program"]).status_code == 422
    assert len(client.get("/api/design").json()) == 1


def test_invalid_literal_nonfinite_numbers_are_422(client):
    response = client.post(
        "/api/design/preview",
        content='{"reference_shot": 990091, "start_s": NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_average_route_returns_editable_snapshot(client, averaging_source):  # noqa: F811
    got = client.post("/api/design/merge", json={
        "reference_shot": SHOT, "comparison_shots": [SHOT + 1]
    })
    assert got.status_code == 200
    assert got.json()["program"]["proposal"]["curves"]["pinj[0]"][0]["y"] == 200
    assert got.json()["program"]["edits"] == {}
    invalid = client.post("/api/design/merge", json={
        "reference_shot": SHOT, "comparison_shots": [SHOT + 9]
    })
    assert invalid.status_code == 422
    assert str(SHOT + 9) in str(invalid.json())


def test_malformed_reference_time_returns_validation_not_500(client, program_source):  # noqa: F811
    import h5py
    import numpy as np

    paths, _ = program_source
    saved = client.post("/api/design", json={"reference_shot": SHOT}).json()
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as f:
        f["pinj/ydata"][:, -1] = np.nan
        f["pinj/xdata"][-1] = np.inf
    preview = client.post("/api/design/preview", json={"reference_shot": SHOT})
    assert preview.status_code == 200
    assert "time" in " ".join(preview.json()["validation"]["errors"])
    assert client.post("/api/design", json={"reference_shot": SHOT}).status_code == 422
    ident = saved["program"]["id"]
    assert client.get(f"/api/design/{ident}/ignite").status_code == 422
