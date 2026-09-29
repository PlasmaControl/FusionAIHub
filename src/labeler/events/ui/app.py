"""The label review server: one label per shot, over rows prebuilt per shot.

Loopback only, behind a token: `?token=` sets two cookies and redirects to
the same URL without it. The server reads `shots.csv` and never writes it;
its writes are `POST /api/label` and `POST /api/masks`, both into the event's
`review/` directory. `GET /api/history` lists a shot's saved versions (see
`review.versions`); `GET /api/masks` gives an AE shot's pseudo-mask regions and
the reviewer's last word on them (see `labeler.ae.seg.regions`).
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...ae import seg
from ...ae.seg import regions
from ...ae.seg.pseudo import PseudoMask
from ...config import Paths, sha256_of
from .. import raw, rosters
from ..review import build as review_build
from ..review import labels, rows, versions

STATIC = Path(__file__).parent / "static"
COOKIE = "labeler_verify_token"
#: The same token again for a FRAMED page: VS Code's Simple Browser frames
#: every page, and drops a Lax cookie there. SameSite=None needs Secure, which
#: browsers allow on 127.0.0.1 over plain http.
FRAMED_COOKIE = "labeler_verify_token_framed"
NO_TOKEN = "no token: reopen the link printed by the verify server"
BAD_TOKEN = "bad token"
#: Bumped when the server gains a route or a field the page depends on. The
#: page asks `/api/version` first and, from an older server, saves without a
#: name and hides the history instead of failing every save. 3 added the masks.
API_VERSION = 3

log = logging.getLogger(__name__)


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    event: str
    shot: int
    window: tuple[float, float]
    intervals: list[tuple[float, float, int]] = Field(max_length=1000)
    name: str | None = Field(default=None, max_length=versions.NAME_MAX)


class MaskIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str
    shot: int
    #: The version of the pseudo-mask the page drew; a save on another is refused.
    pseudo_sha256: str = Field(min_length=64, max_length=64)
    revision: int = Field(ge=0)
    rejected: list[int] = Field(max_length=regions.MAX_REGIONS)
    name: str | None = Field(default=None, max_length=versions.NAME_MAX)


class Builds:
    """Store files built on first open, two at a time, off the request thread."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.running: dict = {}
        self.lock = threading.Lock()

    def status(self, event: str, shot: int) -> tuple[Path | None, str | None]:
        """`(path, None)` once built, `(None, error)` once failed, else building."""
        key = (event, shot)
        path = self.paths.spectrogram_file(event, shot)
        with self.lock:
            future = self.running.get(key)
            if future is not None and not future.done():
                return None, None
            self.running.pop(key, None)
            if future is not None and future.exception() is not None:
                error = future.exception()
                return None, f"{type(error).__name__}: {error}"
            if path.is_file():
                return path, None
            self.running[key] = self.pool.submit(
                review_build.build, event, shot, self.paths
            )
            return None, None


def require_event(event: str, paths: Paths) -> Path:
    """The event's directory, else 404; decided on the name before any join."""
    if event != Path(event).name or event in {"", ".", ".."} or "\0" in event:
        raise HTTPException(404, f"unknown event {event!r}")
    directory = paths.label_tables / event
    try:
        found = (directory / rosters.ROSTER_NAME).is_file()
    except OSError:  # ENAMETOOLONG is raised, not reported as False
        found = False
    if not found:
        raise HTTPException(404, f"unknown event {event!r}")
    return directory


def _roster(directory: Path):
    return rosters.read_roster(directory / rosters.ROSTER_NAME)


def roster_tier(roster, shot: int) -> str:
    match = roster[roster.shot == int(shot)]
    if match.empty:
        raise HTTPException(404, f"shot {int(shot)} is not on this event's roster")
    return str(match.tier.iloc[0])


def _admit(request: Request, token: str) -> Response | None:
    """None for a request carrying the token; else its 401, or the 303 that sets it."""
    expected = token.encode()
    given = request.query_params.get("token")
    if given is not None:
        if not secrets.compare_digest(given.encode(), expected):
            return JSONResponse({"error": BAD_TOKEN}, status_code=401)
        url = request.url.remove_query_params("token")
        path = url.path if not url.path.startswith("//") else "/"
        response = RedirectResponse(path + (f"?{url.query}" if url.query else ""), 303)
        response.set_cookie(COOKIE, token, httponly=True, samesite="lax")
        response.set_cookie(
            FRAMED_COOKIE, token, httponly=True, samesite="none", secure=True
        )
        return response
    cookie = request.cookies.get(COOKIE) or request.cookies.get(FRAMED_COOKIE, "")
    if not cookie or not secrets.compare_digest(cookie.encode(), expected):
        return JSONResponse({"error": NO_TOKEN}, status_code=401)
    return None


def create_app(paths: Paths | None = None, token: str | None = None) -> FastAPI:
    paths = Paths.from_env() if paths is None else paths
    app = FastAPI(
        title="labeler review", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.paths = paths
    app.state.token = token or secrets.token_hex(16)
    app.state.builds = Builds(paths)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        # One log line per request, by PATH: the token rides in the query string.
        started = time.monotonic()
        response = _admit(request, app.state.token)
        if response is None:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        log.info(
            "%s %s -> %d in %.1fs",
            request.method,
            request.url.path,
            response.status_code,
            time.monotonic() - started,
        )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def refused(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def malformed(request: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"][1:]) or "query"
        return JSONResponse({"error": f"{where}: {first['msg']}"}, status_code=422)

    @app.exception_handler(ValueError)
    @app.exception_handler(OSError)
    async def failed(request: Request, exc: Exception):
        log.error("%s failed", request.url.path, exc_info=exc)
        return JSONResponse({"error": str(exc) or type(exc).__name__}, status_code=500)

    @app.get("/api/version")
    def version():
        return {"api": API_VERSION}

    @app.get("/api/events")
    def events():
        try:
            directories = sorted(
                p
                for p in paths.label_tables.iterdir()
                if (p / rosters.ROSTER_NAME).is_file()
            )
        except OSError:
            return {"events": []}
        found = []
        for directory in directories:
            event = directory.name
            try:
                roster = _roster(directory)
                saved = labels.read_saved(directory)
                categories = labels.categories(event)
                found.append(
                    {
                        "event": event,
                        "n_shots": len(roster),
                        "n_reviewed": int(roster.shot.isin(list(saved)).sum()),
                        "categories": {str(k): v for k, v in categories.items()},
                    }
                )
            # One bad roster must not hide the rest.
            except Exception as error:  # noqa: BLE001
                message = str(error) or type(error).__name__
                found.append({"event": event, "error": message})
        return {"events": found}

    @app.get("/api/queue")
    def queue(event: str):
        directory = require_event(event, paths)
        return labels.queue(directory, _roster(directory))

    @app.get("/api/shot")
    def shot_view(event: str, shot: int):
        directory = require_event(event, paths)
        tier = roster_tier(_roster(directory), shot)
        path, error = app.state.builds.status(event, shot)
        if error is not None:
            return JSONResponse({"error": error}, status_code=502)
        if path is None:
            body = {"building": True, "progress": raw.progress_for(shot)}
            return JSONResponse(body, status_code=202)
        return {
            "event": event,
            "shot": shot,
            "tier": tier,
            **rows.meta(path, review_build.HIDDEN.get(event, frozenset())),
            **labels.shot_labels(directory, shot),
        }

    @app.get("/api/rows")
    def rows_view(
        event: str,
        shot: int,
        t0: float,
        t1: float,
        cols: Annotated[int, Query(ge=16, le=8192)] = 1500,
    ):
        require_event(event, paths)
        path = paths.spectrogram_file(event, shot)
        if not path.is_file():
            raise HTTPException(404, f"shot {shot} has no rows yet: open it first")
        try:
            hide = review_build.HIDDEN.get(event, frozenset())
            data, grid = rows.read_window(path, t0, t1, cols, hide)
        except (ValueError, OverflowError) as error:
            raise HTTPException(400, str(error)) from None
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"X-Grid": json.dumps(grid)},
        )

    @app.post("/api/label")
    def save_label(body: LabelIn):
        directory = require_event(body.event, paths)
        tier = roster_tier(_roster(directory), body.shot)
        known = set(labels.categories(body.event))
        try:
            label = labels.normalise(body.window, body.intervals, known=known)
            name = versions.clean_name(body.name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        table = labels.source_path(directory)
        try:
            entry = labels.save(
                directory,
                body.shot,
                label,
                source=table.name if table else None,
                name=name,
                source_sha256=sha256_of(table) if table else None,
            )
        except labels.SaveRefused as error:
            raise HTTPException(409, str(error)) from None
        source = labels.read_source(directory).get(body.shot)
        row = {
            "shot": body.shot,
            "tier": tier,
            "state": labels.state(label, source),
            "saved_at": entry["saved_at"],
        }
        return {"row": row, "saved": label.as_json(), "last_save": entry}

    @app.get("/api/history")
    def history_view(event: str, shot: int):
        directory = require_event(event, paths)
        roster_tier(_roster(directory), shot)
        return {"shot": shot, "versions": versions.shot_versions(directory, shot)}

    def mask_file(event: str, shot: int) -> tuple[Path, Path]:
        directory = require_event(event, paths)
        roster_tier(_roster(directory), shot)
        path = regions.pseudo_file(paths, shot)
        if event != seg.EVENT or not path.is_file():
            raise HTTPException(404, f"shot {int(shot)} has no {event} pseudo-mask")
        return directory, path

    @app.get("/api/masks")
    def masks_view(event: str, shot: int):
        directory, _ = mask_file(event, shot)
        with mask_lock:
            return regions.shot_view(paths, directory, shot)

    @app.post("/api/masks")
    def save_masks(body: MaskIn):
        directory, path = mask_file(body.event, body.shot)
        with mask_lock:
            revision = len(regions.shot_history(directory, body.shot))
            if revision != body.revision:
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": (
                            "the mask decisions changed since the page drew them: "
                            "reopen it"
                        ),
                        "revision": revision,
                    },
                )
            if regions.file_sha256(path) != body.pseudo_sha256:
                raise HTTPException(
                    409, "the pseudo-mask changed since the page drew it: reopen it"
                )
            _, count = regions.label_regions(PseudoMask.load(path).mask)
            unknown = sorted({k for k in body.rejected if not 1 <= k <= count})
            if unknown:
                raise HTTPException(400, f"no region {unknown} (1..{count})")
            try:
                name = versions.clean_name(body.name)
            except ValueError as error:
                raise HTTPException(400, str(error)) from None
            entry = regions.save_decision(
                directory,
                body.shot,
                body.rejected,
                pseudo_sha256=body.pseudo_sha256,
                name=name,
            )
        return {
            "rejected": entry["rejected"],
            "last_save": entry,
            "revision": revision + 1,
        }

    mask_lock = threading.Lock()
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
