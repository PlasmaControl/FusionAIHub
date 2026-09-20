"""Submit, poll, and read back one design's Frontier simulation.

D3's ``python -m shot_design simulate <ident>`` writes ``status.json``, ``report.md``
and ``panels/<m>.png`` under ``paths.data_root / "outputs" / <ident> / "simulation"``
(see ``shot_design/simulate/cli.py``). These routes never run that command directly --
Frontier compute nodes are not this process -- they submit it through ``sbatch`` (via
``app.state.submit``, mockable in tests) and read back whatever that job wrote.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .. import config
from ..design import program as service

router = APIRouter(prefix="/api/design")

# src/shot_design/ui/simulate_routes.py -> parents[3] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

_JOB_ID = re.compile(r"Submitted batch job (\d+)")
_PANEL_NAME = re.compile(r"^[a-z_]+\.png$")


def default_submit(cmd: str) -> str:
    """Production ``app.state.submit``: run ``cmd`` from the repo root via sbatch.

    The wrapper locates ``_shot_design_common.sh`` through ``$SLURM_SUBMIT_DIR``, so
    ``sbatch`` must be invoked with the repo root as its working directory.
    """
    result = subprocess.run(
        shlex.split(cmd), cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout


def _check_ident(ident: str) -> None:
    if not re.fullmatch(service.ID_PATTERN, ident):
        raise HTTPException(404, "Unknown design revision")


def _simulation_dir(ident: str, request: Request) -> Path:
    return request.app.state.paths.data_root / "outputs" / ident / "simulation"


@router.post("/{ident}/simulate", status_code=202)
def submit(ident: str, request: Request):
    _check_ident(ident)
    try:
        simulate_cfg = config.load_yaml("ui.yaml")["simulate"]
    except KeyError as exc:
        raise HTTPException(
            500, "configs/shot_design/ui.yaml is missing its simulate: block"
        ) from exc
    cmd = simulate_cfg["submit_cmd"].format(ident=ident)
    submit_fn = getattr(request.app.state, "submit", default_submit)
    output = submit_fn(cmd)
    match = _JOB_ID.search(output)
    if not match:
        raise HTTPException(502, output)
    return {
        "ident": ident,
        "job_id": match.group(1),
        "status_url": f"/api/design/{ident}/simulate",
    }


@router.get("/{ident}/simulate")
def status(ident: str, request: Request):
    _check_ident(ident)
    status_path = _simulation_dir(ident, request) / "status.json"
    if not status_path.exists():
        return {"state": "not_started"}
    return json.loads(status_path.read_text())


@router.get("/{ident}/simulate/report")
def report(ident: str, request: Request):
    _check_ident(ident)
    path = _simulation_dir(ident, request) / "report.md"
    if not path.exists():
        raise HTTPException(404, "No simulation report for this design yet")
    return FileResponse(path, media_type="text/markdown")


@router.get("/{ident}/simulate/panels/{name}")
def panel(ident: str, name: str, request: Request):
    _check_ident(ident)
    if not _PANEL_NAME.fullmatch(name):
        raise HTTPException(404, "Unknown panel")
    path = _simulation_dir(ident, request) / "panels" / name
    if not path.exists():
        raise HTTPException(404, "Unknown panel")
    return FileResponse(path, media_type="image/png")
