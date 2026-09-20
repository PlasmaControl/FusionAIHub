"""HTTP transport for D4: submit a simulation, poll status, read back report/panels.

The real sbatch submission is never exercised here (no Slurm, no GPU); every test
injects ``app.state.submit`` so the wrapper's default (``subprocess.run`` against
the repo root) is covered by its own unit test instead.
"""

import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from shot_design import config as sd_config
from shot_design.ui.app import COOKIE, create_app
from shot_design.ui.simulate_routes import REPO_ROOT, default_submit

IDENT = "0" * 32


@pytest.fixture
def app(paths):  # noqa: F811 - shared pytest fixture from conftest.py
    application = create_app(paths=paths, token="secret")
    application.state.submit = lambda cmd: "Submitted batch job 4242"
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        client.cookies.set(COOKIE, "secret")
        yield client


def _sim_dir(paths, ident=IDENT):
    d = paths.data_root / "outputs" / ident / "simulation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_submit_returns_job_id_and_status_url(client):
    resp = client.post(f"/api/design/{IDENT}/simulate")
    assert resp.status_code == 202
    assert resp.json() == {
        "ident": IDENT,
        "job_id": "4242",
        "status_url": f"/api/design/{IDENT}/simulate",
    }


def test_submit_runs_the_configured_command_with_ident_filled_in(paths):
    received = {}

    def fake_submit(cmd):
        received["cmd"] = cmd
        return "Submitted batch job 55"

    application = create_app(paths=paths, token="secret")
    application.state.submit = fake_submit
    with TestClient(application) as client:
        client.cookies.set(COOKIE, "secret")
        resp = client.post(f"/api/design/{IDENT}/simulate")
    assert resp.status_code == 202
    expected = f"sbatch scripts/slurm_frontier/shot_design_simulate.sh {IDENT}"
    assert received["cmd"] == expected


def test_unparseable_submit_output_is_502(app):
    app.state.submit = lambda cmd: "sbatch: error: something bad happened"
    with TestClient(app) as client:
        client.cookies.set(COOKIE, "secret")
        resp = client.post(f"/api/design/{IDENT}/simulate")
    assert resp.status_code == 502
    assert "something bad happened" in resp.json()["detail"]


def test_failing_sbatch_maps_to_502_with_its_stderr(app):
    def raise_called_process_error(cmd):
        raise subprocess.CalledProcessError(
            1, ["sbatch"], output="", stderr="sbatch: error: invalid qos"
        )

    app.state.submit = raise_called_process_error
    with TestClient(app) as client:
        client.cookies.set(COOKIE, "secret")
        resp = client.post(f"/api/design/{IDENT}/simulate")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "sbatch: error: invalid qos"


def test_sbatch_not_on_path_maps_to_502(app):
    def raise_file_not_found(cmd):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'sbatch'")

    app.state.submit = raise_file_not_found
    with TestClient(app) as client:
        client.cookies.set(COOKIE, "secret")
        resp = client.post(f"/api/design/{IDENT}/simulate")
    assert resp.status_code == 502
    assert "sbatch" in resp.json()["detail"]


def test_missing_simulate_block_in_ui_yaml_is_500(client, monkeypatch):
    orig_load_yaml = sd_config.load_yaml

    def fake_load_yaml(name):
        doc = orig_load_yaml(name)
        if name == "ui.yaml":
            doc.pop("simulate", None)
        return doc

    monkeypatch.setattr(sd_config, "load_yaml", fake_load_yaml)
    resp = client.post(f"/api/design/{IDENT}/simulate")
    assert resp.status_code == 500


def test_status_is_not_started_when_no_status_file_exists(client):
    resp = client.get(f"/api/design/{IDENT}/simulate")
    assert resp.status_code == 200
    assert resp.json() == {"state": "not_started"}


def test_status_returns_the_status_json_contents(client, paths):
    out = _sim_dir(paths)
    status = {
        "state": "running", "started": "2026-09-20T00:00:00+00:00",
        "finished": None, "error": None, "report": None,
    }
    (out / "status.json").write_text(json.dumps(status))
    resp = client.get(f"/api/design/{IDENT}/simulate")
    assert resp.status_code == 200
    assert resp.json() == status


def test_report_is_served_when_present(client, paths):
    out = _sim_dir(paths)
    (out / "report.md").write_text("# Report\n\nAll good.\n")
    resp = client.get(f"/api/design/{IDENT}/simulate/report")
    assert resp.status_code == 200
    assert resp.text == "# Report\n\nAll good.\n"


def test_report_missing_is_404(client):
    resp = client.get(f"/api/design/{IDENT}/simulate/report")
    assert resp.status_code == 404


def test_panel_is_served_as_png(client, paths):
    out = _sim_dir(paths)
    (out / "panels").mkdir()
    (out / "panels" / "mhr.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    resp = client.get(f"/api/design/{IDENT}/simulate/panels/mhr.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == b"\x89PNG\r\n\x1a\n"


def test_panel_missing_is_404(client, paths):
    _sim_dir(paths)
    resp = client.get(f"/api/design/{IDENT}/simulate/panels/mhr.png")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "name", ["..%2Fsecret.png", "mhr.jpg", "MHR.png", "mhr", "a%2Fb.png", "mhr..png"]
)
def test_panel_name_validation_rejects_anything_else(client, paths, name):
    out = _sim_dir(paths)
    (out / "panels").mkdir()
    (out / "panels" / "secret.png").write_bytes(b"nope")
    resp = client.get(f"/api/design/{IDENT}/simulate/panels/{name}")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "ident", ["..%2Fetc", "not-hex", "0" * 31, "0" * 33, "UPPERCASE" + "0" * 23]
)
def test_malformed_idents_are_404_not_filesystem_paths(client, ident):
    routes = [
        ("get", ""), ("post", ""), ("get", "/report"), ("get", "/panels/mhr.png"),
    ]
    for method, suffix in routes:
        resp = getattr(client, method)(f"/api/design/{ident}/simulate{suffix}")
        assert resp.status_code == 404


def test_all_simulate_routes_require_authentication(client):
    client.cookies.clear()
    for method, path in [
        ("post", f"/api/design/{IDENT}/simulate"),
        ("get", f"/api/design/{IDENT}/simulate"),
        ("get", f"/api/design/{IDENT}/simulate/report"),
        ("get", f"/api/design/{IDENT}/simulate/panels/mhr.png"),
    ]:
        assert getattr(client, method)(path).status_code == 401


def test_repo_root_resolves_to_the_actual_repo_root():
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_default_submit_runs_subprocess_from_the_repo_root(monkeypatch):
    calls = {}

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args, 0, stdout="Submitted batch job 99\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    cmd = f"sbatch scripts/slurm_frontier/shot_design_simulate.sh {IDENT}"
    out = default_submit(cmd)
    assert out == "Submitted batch job 99\n"
    assert calls["args"] == [
        "sbatch", "scripts/slurm_frontier/shot_design_simulate.sh", IDENT,
    ]
    assert calls["kwargs"]["cwd"] == REPO_ROOT
    assert calls["kwargs"]["check"] is True
