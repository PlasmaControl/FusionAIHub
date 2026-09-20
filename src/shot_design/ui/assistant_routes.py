"""Pollable, authenticated jobs for the small model design harness."""

from __future__ import annotations

import copy
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .. import config
from ..design import assistant
from ..llm.client import LLMUnavailable
from ..mcp import tools

router = APIRouter(prefix="/api/design-assistant")
_INIT_LOCK = threading.Lock()


class DesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000)
    model: Literal["quality", "fast"] = "quality"

    @field_validator("prompt")
    @classmethod
    def nonempty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Describe the shot you want to design")
        return value


class Jobs:
    """One in-flight job per app, bounded history, immutable saved artifacts."""

    def __init__(self, paths):
        self.paths = paths
        self.lock = threading.Lock()
        self.jobs = {}

    def snapshot(self, ident):
        with self.lock:
            if not re.fullmatch(r"[0-9a-f]{32}", ident) or ident not in self.jobs:
                raise HTTPException(404, "Unknown design assistant job")
            return copy.deepcopy(self.jobs[ident])

    def submit(self, body):
        with self.lock:
            if any(
                job["status"] in {"queued", "running"} for job in self.jobs.values()
            ):
                raise HTTPException(
                    409, "A design is already running; wait for it to finish"
                )
            while len(self.jobs) >= 50:
                self.jobs.pop(next(iter(self.jobs)))
            ident = uuid.uuid4().hex
            self.jobs[ident] = {
                "id": ident,
                "status": "queued",
                "prompt": body.prompt,
                "created": datetime.now(UTC).isoformat(),
                "stages": [
                    {"key": key, "label": label, "status": "pending", "detail": ""}
                    for key, label in assistant.STAGES
                ],
                "result": None,
                "error": None,
            }
        threading.Thread(
            target=self._run,
            args=(ident, body),
            name=f"shot-design-{ident[:8]}",
            daemon=True,
        ).start()
        return self.snapshot(ident)

    def _run(self, ident, body):
        with self.lock:
            self.jobs[ident]["status"] = "running"

        def progress(key, status, detail):
            with self.lock:
                stage = next(
                    stage for stage in self.jobs[ident]["stages"] if stage["key"] == key
                )
                stage.update(status=status, detail=detail)

        try:
            # ContextVars do not automatically follow a new threading.Thread.
            with config.using_paths(self.paths):
                db, error = tools._db()
                if error:
                    raise ValueError(error["error"])
                result = assistant.run_design(
                    body.prompt,
                    self.paths,
                    db,
                    model=body.model,
                    progress=progress,
                )
            result["hdf5_url"] = f"/api/design-assistant/{ident}/hdf5"
            with self.lock:
                self.jobs[ident].update(status="complete", result=result)
        except Exception as exc:  # noqa: BLE001 - jobs require a terminal state
            if isinstance(exc, (ValueError, LLMUnavailable)):
                detail = str(exc)[:1200]
            elif isinstance(exc, OSError):
                detail = "The design could not read or write its files. Check the configured data and outputs directories."
            else:
                detail = f"The design stopped with {type(exc).__name__}. Retry after checking the server configuration."
            with self.lock:
                job = self.jobs[ident]
                job.update(status="failed", error=detail)
                active = next(
                    (stage for stage in job["stages"] if stage["status"] == "running"),
                    None,
                )
                if active is None:
                    active = next(
                        (
                            stage
                            for stage in job["stages"]
                            if stage["status"] == "pending"
                        ),
                        None,
                    )
                if active:
                    active.update(status="failed", detail=detail)


def _jobs(request):
    with _INIT_LOCK:
        if not hasattr(request.app.state, "design_assistant_jobs"):
            request.app.state.design_assistant_jobs = Jobs(request.app.state.paths)
        return request.app.state.design_assistant_jobs


@router.post("", status_code=202)
def submit(body: DesignRequest, request: Request):
    return _jobs(request).submit(body)


@router.get("/{ident}")
def status(ident: str, request: Request):
    return _jobs(request).snapshot(ident)


@router.get("/{ident}/hdf5")
def download(ident: str, request: Request):
    job = _jobs(request).snapshot(ident)
    if job["status"] != "complete":
        raise HTTPException(409, "This design has no completed HDF5 artifact")
    path = Path(job["result"]["artifact_path"])
    if path.is_symlink() or not path.is_file():
        raise HTTPException(404, "The saved HDF5 artifact is unavailable")
    return FileResponse(path, filename=path.name, media_type="application/x-hdf5")
