"""Serve lifecycle with a fake Ollama binary; all state stays under tmp_path."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from shot_design import config

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "shot_design"
MANIFEST = "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml"


def test_llm_shell_scripts_parse():
    for name in ("serve_llm.sh", "serve_llm.sbatch", "blurb_all.sh"):
        subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


@pytest.fixture
def shim_env(paths, tmp_path, monkeypatch):
    if shutil.which("curl") is None:
        pytest.skip("serve lifecycle needs curl")
    shims = tmp_path / "shims"
    shims.mkdir()
    calls = tmp_path / "calls.jsonl"
    pixi = shims / "pixi"
    pixi.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
expected = ["run", "--frozen", "--no-install", "--manifest-path",
            "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml", "-e", "ideate-cpu", "python"]
assert sys.argv[1:9] == expected, sys.argv
with Path(os.environ["SHIM_CALLS"]).open("a") as f:
    f.write(json.dumps({"pixi": sys.argv[1:]}) + "\\n")
os.execv(sys.executable, [sys.executable, *sys.argv[9:]])
''')
    pixi.chmod(0o755)
    bindir = tmp_path / "ollama install"
    (bindir / "bin").mkdir(parents=True)
    ollama = bindir / "bin" / "ollama"
    ollama.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
with Path(os.environ["SHIM_CALLS"]).open("a") as f:
    f.write(json.dumps({"ollama": sys.argv[1:], "env": {
        k: os.environ.get(k) for k in ("HOME", "OLLAMA_MODELS", "OLLAMA_HOST",
        "OLLAMA_CONTEXT_LENGTH", "OLLAMA_KEEP_ALIVE", "OLLAMA_MAX_LOADED_MODELS")
    }}) + "\\n")
if sys.argv[1] == "serve":
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"version":"shim"}')
        def log_message(self, *args):
            pass
    host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
    HTTPServer((host, int(port)), Handler).serve_forever()
elif sys.argv[1] == "--version":
    print("ollama version is shim")
elif sys.argv[1] not in ("show", "pull"):
    sys.exit(2)
''')
    ollama.chmod(0o755)
    cfg = config.load_yaml("llm.yaml")
    cfg.update(
        ollama_bin_dir=str(bindir),
        ollama_models_dir=str(tmp_path / "models"),
        ollama_home_dir=str(tmp_path / "ollama home"),
    )
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "llm.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    monkeypatch.setenv("SHOT_DESIGN_CONFIG_DIR", str(cfgdir))
    monkeypatch.setenv("PYTHONPATH", str(REPO / "src"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PATH", str(shims) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SHIM_CALLS", str(calls))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    for key in ("OLLAMA_KEEP_ALIVE", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_CONTEXT_LENGTH"):
        monkeypatch.delenv(key, raising=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return cfg, calls, port


def _wait_for_endpoint(process, endpoint):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if endpoint.exists():
            return json.loads(endpoint.read_text())
        if process.poll() is not None:
            pytest.fail(f"serve shim exited: {process.communicate()}")
        time.sleep(0.05)
    pytest.fail("serve shim did not publish its endpoint within 15 seconds")


@pytest.mark.parametrize("replace_endpoint", [None, "url", "started"])
def test_serve_endpoint_double_start_and_cleanup(paths, shim_env, replace_endpoint):
    cfg, calls, port = shim_env
    endpoint = paths.data_root / "llm" / "endpoint.json"
    command = ["bash", str(SCRIPTS / "serve_llm.sh"), "--port", str(port)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        try:
            doc = _wait_for_endpoint(proc, endpoint)
            assert set(doc) == {"url", "models", "host", "job_id", "started", "version"}
            assert doc["url"] == f"http://127.0.0.1:{port}"
            assert doc["models"] == list(cfg["models"].values())
            assert doc["host"] and doc["started"] and doc["job_id"] == "12345"
            assert doc["version"] == "ollama version is shim"
            assert not list(endpoint.parent.glob("*.part"))
            before = endpoint.read_bytes()
            duplicate = subprocess.run(
                command, capture_output=True, text=True, timeout=10, check=False,
            )
            assert duplicate.returncode == 2 and "already listening" in duplicate.stderr
            assert endpoint.read_bytes() == before and proc.poll() is None
            rows = [json.loads(line) for line in calls.read_text().splitlines()]
            assert sum("pixi" in row for row in rows) == 1
            served = [r for r in rows if r.get("ollama") == ["serve"]]
            assert len(served) == 1
            assert served[0]["env"] == {
                "HOME": cfg["ollama_home_dir"], "OLLAMA_MODELS": cfg["ollama_models_dir"],
                "OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_CONTEXT_LENGTH": "16384",
                "OLLAMA_KEEP_ALIVE": "24h", "OLLAMA_MAX_LOADED_MODELS": "2",
            }
            assert not any(r.get("ollama", [None])[0] == "pull" for r in rows)
            if replace_endpoint:
                doc[replace_endpoint] = (
                    "http://replacement:11434" if replace_endpoint == "url" else "later run"
                )
                endpoint.write_text(json.dumps(doc))
        finally:
            proc.terminate()
            proc.communicate(timeout=10)
    assert endpoint.exists() is bool(replace_endpoint)
    if replace_endpoint:
        assert json.loads(endpoint.read_text()) == doc


def test_serve_missing_binary_exits_two_without_installing(paths, shim_env):
    cfg, calls, port = shim_env
    binary = Path(cfg["ollama_bin_dir"]) / "bin" / "ollama"
    binary.unlink()
    result = subprocess.run(
        ["bash", str(SCRIPTS / "serve_llm.sh"), "--port", str(port)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 2 and str(binary) in result.stderr
    assert not (paths.data_root / "llm").exists()
    assert not binary.exists()
    assert all("pixi" in json.loads(line) for line in calls.read_text().splitlines())


def test_blurb_wrapper_uses_frozen_environment_without_local_install(tmp_path, monkeypatch):
    wrapper = tmp_path / "pixi"
    wrapper.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        ["bash", str(SCRIPTS / "blurb_all.sh"), "--dry-run", "--limit", "5"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == [
        "run", "--frozen", "--no-install", "--manifest-path", MANIFEST,
        "-e", "ideate-cpu", "python", "-m", "shot_design", "blurb", "--all",
        "--dry-run", "--limit", "5",
    ]
