"""The authenticated harness exposes real progress, completion, and errors."""

import io
import time

import h5py
import pytest
from fastapi.testclient import TestClient

from shot_design.ui.app import COOKIE, create_app

from .test_assistant import candidate_search, model_client  # noqa: F401
from .test_program import program_source  # noqa: F401


@pytest.fixture
def assistant_client(program_source, candidate_search, monkeypatch):  # noqa: F811
    from shot_design.design import assistant
    from shot_design.ui import assistant_routes

    paths, _ = program_source
    monkeypatch.setattr(assistant, "LLMClient", lambda paths: model_client(paths))
    monkeypatch.setattr(assistant_routes.tools, "_db", lambda: (object(), None))
    app = create_app(paths=paths, token="secret")
    with TestClient(app) as client:
        client.cookies.set(COOKIE, "secret")
        yield client


def finished(client, ident):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/design-assistant/{ident}")
        body = response.json()
        if body["status"] in {"complete", "failed"}:
            return body
        time.sleep(0.01)
    raise AssertionError("Design job did not finish")


def test_poll_complete_download_and_open_editor(assistant_client):
    client = assistant_client
    response = client.post(
        "/api/design-assistant", json={"prompt": "Control tearing modes"}
    )
    assert response.status_code == 202
    job = finished(client, response.json()["id"])
    assert job["status"] == "complete", job
    assert all(stage["status"] == "complete" for stage in job["stages"])
    result = job["result"]
    assert client.get(f"/api/design/{result['design_id']}").status_code == 200
    download = client.get(result["hdf5_url"])
    assert download.status_code == 200
    with h5py.File(io.BytesIO(download.content), "r") as file:
        assert file["actuators"].shape == (80, 88)


def test_failure_is_pollable_and_has_no_download(assistant_client, monkeypatch):
    from shot_design.design import assistant

    monkeypatch.setattr(
        assistant, "LLMClient", lambda paths: model_client(paths, fail=True)
    )
    response = assistant_client.post(
        "/api/design-assistant", json={"prompt": "Control ELMs"}
    )
    job = finished(assistant_client, response.json()["id"])
    assert job["status"] == "failed"
    assert job["error"]
    assert job["stages"][0]["status"] == "failed"
    assert job["stages"][-1]["status"] == "pending"
    assert job["result"] is None
    assert (
        assistant_client.get(f"/api/design-assistant/{job['id']}/hdf5").status_code
        == 409
    )


def test_assistant_routes_require_existing_authentication(assistant_client):
    assistant_client.cookies.clear()
    assert (
        assistant_client.post(
            "/api/design-assistant", json={"prompt": "ELMs"}
        ).status_code
        == 401
    )
    assert assistant_client.get("/api/design-assistant/" + "a" * 32).status_code == 401
    assert (
        assistant_client.get("/api/design-assistant/" + "a" * 32 + "/hdf5").status_code
        == 401
    )


def test_invalid_prompts_and_unknown_ids(assistant_client):
    for prompt in ["", " ", "x" * 4001]:
        assert (
            assistant_client.post(
                "/api/design-assistant", json={"prompt": prompt}
            ).status_code
            == 422
        )
    assert assistant_client.get("/api/design-assistant/" + "a" * 32).status_code == 404
