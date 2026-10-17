"""Review-page routes for TokEye overlays on every spectrogram row and
for build status (plan section C).

`install` adds them to the app that `app.create_app` builds, before the static mount:

    GET  /api/overlays?event=E&shot=S    every row's overlay state, by row name
    POST /api/overlays/request {event, shot}
                                          queue the shot for TokEye: 202 {state, queue_position}
    GET  /  and  /index.html              the page, each local script and stylesheet URL
                                          carrying ?v=<its content's sha8>

`app.py` calls the helpers below in its own regions: `annotate` (each `/api/shot` row's
`overlay`), `tokeye` (the `/api/tokeye` body) and `cache_control` (the gate's header:
`no-store` for everything but a static file asked for at its current `?v=`, which is
cached for a year). A daemon thread (`Drains`) submits waiting requests to SLURM through
`overlays.drain`. See docs/labeler/overlays.md.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ...ae import seg
from ...config import Paths, store_event
from .. import raw
from ..review import overlays
from . import shotlist_routes

log = logging.getLogger(__name__)

#: The page's files (`app.STATIC`).
STATIC = Path(__file__).parent / "static"
#: A static file asked for at its current version.
IMMUTABLE = "private, max-age=31536000, immutable"
NO_STORE = "no-store"
#: A script's src or a link's href in the page.
ASSET = re.compile(
    r"""(<(?:script|link)\b[^>]*?\s(?:src|href)\s*=\s*)(["'])([^"'?#]+)(\?[^"'#]*)?\2""",
    re.IGNORECASE,
)
#: How often the drain thread retries while requests wait.
DRAIN_EVERY_S = 15

_hashes = shotlist_routes.Stamped(size=64)
_pages = shotlist_routes.Stamped(size=2)


def asset(path: str) -> Path | None:
    """The static file a local URL path names, or None (outside STATIC, or missing)."""
    if not path or "://" in path or path.startswith(("//", "data:")):
        return None
    root = STATIC.resolve()
    try:
        found = (root / path.lstrip("/")).resolve()
    except (OSError, ValueError):
        return None
    if root not in found.parents or not found.is_file():
        return None
    return found


def sha8(path: Path) -> str:
    """The first eight hex digits of the file's sha256, until the file changes."""
    return _hashes.get(str(path), [path],
                       lambda: hashlib.sha256(path.read_bytes()).hexdigest()[:8])


def versioned_index() -> str:
    """index.html with `?v=<sha8>` on every local script src and link href."""
    index = STATIC / "index.html"

    def compute() -> str:
        def swap(match: re.Match) -> str:
            head, quote, url, query = match.groups()
            found = asset(url)
            if found is None or found.suffix == ".html":
                return match.group(0)
            query = query or ""
            joiner = "&" if query else "?"
            return f"{head}{quote}{url}{query}{joiner}v={sha8(found)}{quote}"

        return ASSET.sub(swap, index.read_text(encoding="utf-8"))

    files = [index, *sorted(p for p in STATIC.rglob("*") if p.is_file())]
    return _pages.get("index", files, compute)


def cache_control(request, response) -> str:
    """The gate's Cache-Control: a year for a static file asked for at its current
    `?v=`, `no-store` for the page, the API and everything else."""
    path = request.url.path
    version = request.query_params.get("v")
    if (version and response.status_code == 200 and not path.startswith("/api/")
            and path not in ("/", "/index.html")):
        found = asset(path)
        if found is not None and version == sha8(found):
            return IMMUTABLE
    return NO_STORE


def reading(shot: int) -> bool:
    """True while this process holds the shot's corpus or raw-cache file open."""
    name = f"/{int(shot)}_processed.h5"
    try:
        fds = os.listdir("/proc/self/fd")
    except OSError:
        return False
    for fd in fds:
        with contextlib.suppress(OSError):
            if os.readlink(f"/proc/self/fd/{fd}").endswith(name):
                return True
    return False


def build_stage(paths: Paths, store: str, shot: int) -> str:
    """A running store build's stage: fetching (raw is fetching a signal), writing (the
    store's temporary file exists), reading (the shot's corpus file is open), else
    building."""
    if raw.progress_for(shot) is not None:
        return "fetching"
    path = paths.spectrogram_file(store, shot)
    if path.with_name(path.name + ".tmp").exists():
        return "writing"
    return "reading" if reading(shot) else "building"


def prebuilt(paths: Paths, shot: int) -> bool:
    """Whether a registered shot list holds the shot (its stores are pre-built); the
    all-shots list does not count, its stores are built when a shot is opened."""
    try:
        from .. import shotlists

        return any(name not in shotlists.ON_DEMAND
                   for name in shotlists.holding(paths, shot))
    except Exception:  # noqa: BLE001 - a status line must not fail on a bad list file
        return False


def annotate(paths: Paths, event: str, shot: int, rows: list[dict]) -> None:
    """Give every `/api/shot` row its `overlay` state (`overlays.row_states`)."""
    try:
        states = overlays.row_states(paths, event, shot, rows)
    except Exception as error:  # noqa: BLE001 - the overlay is advisory; the shot opens
        log.warning("overlay states of %s %s: %r", event, shot, error)
        states = {row["name"]: {"state": "failed",
                                "reason": f"overlay state unreadable: {error}"}
                  for row in rows if overlays.applicable(row)}
    for row in rows:
        row["overlay"] = states.get(row["name"], dict(overlays.NOT_APPLICABLE))


def tokeye(paths: Paths, event: str, owner: str, shot: int) -> dict | None:
    """The `/api/tokeye` body: AE and its children keep their one layer (this model's
    whole-record union, else tokeye-full); any other event `{rows: {name: layer}}` for
    its rows with a done layer; None when there is nothing to serve."""
    if owner == seg.EVENT:
        return overlays.ae_view(paths, shot)
    try:
        rows = overlays.store_rows_meta(paths, event, shot)
    except FileNotFoundError:
        return None
    found = overlays.layers(paths, event, shot, rows)
    return {"rows": found} if found else None


class Drains:
    """Submits waiting requests from one daemon thread while any wait."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def kick(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            if not overlays.pending(self.paths):
                return
            self.thread = threading.Thread(target=self._loop, name="overlay-drain",
                                           daemon=True)
            self.thread.start()

    def _loop(self) -> None:
        while overlays.pending(self.paths):
            try:
                found = overlays.drain(self.paths)
                if found.get("submitted"):
                    log.info("overlays: submitted job %s for %s shot(s)",
                             found["submitted"], found["shots"])
            except Exception:  # the thread must outlive one bad drain
                log.exception("overlay drain failed")
            time.sleep(DRAIN_EVERY_S)


class OverlayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str = Field(max_length=64, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?$")
    shot: Annotated[int, Field(strict=True, ge=1, lt=10_000_000)]


def install(app: FastAPI, paths: Paths) -> None:
    """Add this module's routes to `app`."""
    from .app import require_event, shot_tier  # app imports this module

    app.state.overlay_drains = Drains(paths)

    def store_rows(event: str, shot: int) -> list[dict]:
        require_event(event, paths)
        shot_tier(event, shot, paths)
        try:
            return overlays.store_rows_meta(paths, event, shot)
        except FileNotFoundError:
            raise HTTPException(
                404, f"shot {int(shot)} has no {store_event(event)} review store yet: "
                     "open it first"
            ) from None

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    def page():
        return HTMLResponse(versioned_index())

    @app.get("/api/overlays")
    def overlays_view(event: str, shot: int):
        rows = store_rows(event, shot)
        app.state.overlay_drains.kick()
        return {"event": event, "shot": int(shot),
                "rows": overlays.row_states(paths, event, shot, rows)}

    @app.post("/api/overlays/request")
    def overlays_request(body: OverlayRequest):
        rows = store_rows(body.event, body.shot)
        found = overlays.request(paths, body.event, body.shot, rows)
        app.state.overlay_drains.kick()
        return JSONResponse(found, status_code=202)
