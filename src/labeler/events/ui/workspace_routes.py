"""Review-page routes for the signal workspace (plan section B): the
catalog, on-demand signals, profiles at a time slice and saved views.

`install` adds them to the app that `app.create_app` builds, before the static mount.

- `GET /api/catalog?event&shot`: the entries the picker offers, each with whether the
  shot has it on disk, would fetch it or lacks it, its one-line `description`, the
  PTDATA names and MDS nodes it fetches (`points`, for a typed name to match) and,
  when it was probed, `avail` (both, early, late or none: the reference shots 170659 and
  200958); the MDS trees and the patterns a free id must match.
- `GET /api/signal?shot&id&t0&t1&cols`: a trace window in the `/api/rows` format, with
  `X-Grid` and `X-Signal`; 202 `{state, stage, elapsed_s, queue_position}` while it is
  fetched.
- `GET /api/profile?shot&id`: a profile (`review.profiles`), with `X-Profile`; the
  same 202.
- `GET /api/fetch/status`: fdp's state and the fetch queue.
- `GET /api/views?event&reviewer`, `POST /api/views`: saved views (`review.views`).

Errors are `{error, kind}`: 422 invalid, 404 missing, 503 auth, 502 upstream, 504
timeout. A catalog blind shot is refused like a missing one. The app's token gate
covers every route.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from ...config import Paths
from ..review import catalog as cat
from ..review import profiles, silver
from ..review import views as saved_views
from ..review.fetch_worker import WATCH, FetchError
from ..review.signals import COLS_DEFAULT, Pending, Signals, http_status

log = logging.getLogger(__name__)

SHOT = re.compile(r"^\d{1,7}$")
#: The most a `POST /api/views` body may hold (the layout's own limit is 64 KB).
BODY_MAX = 256 * 1024


def version_extra(paths: Paths) -> dict:
    """What `/api/version` adds: fdp's state from the token read (reused for a
    minute; no fdp call) and that the workspace routes are here."""
    del paths  # the same for every root: the token is the user's
    return {"fdp": WATCH.get().public(), "workspace": True}


def _error(error: FetchError) -> JSONResponse:
    return JSONResponse({"error": error.message, "kind": error.kind},
                        status_code=http_status(error))


def _shot(text) -> int:
    if text is None or not SHOT.fullmatch(text):
        raise FetchError("invalid", "shot is a shot number")
    return int(text)


def _number(text, name: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        raise FetchError("invalid", f"{name} is a number of milliseconds") from None


def _cols(text) -> int:
    if text is None:
        return COLS_DEFAULT
    if not re.fullmatch(r"\d{1,5}", text):
        raise FetchError("invalid", "cols is a whole number")
    return int(text)


class _Blind:
    """The catalog cohort's blind shots, re-read when cohort.csv changes."""

    def __init__(self, paths: Paths):
        self.cohort = Path(paths.label_tables) / "catalog" / "cohort.csv"
        self.label_tables = paths.label_tables
        self.stamp, self.shots = None, frozenset()

    def __contains__(self, shot: int) -> bool:
        try:
            st = self.cohort.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        if stamp != self.stamp:
            self.shots = frozenset(silver.blind_shots(self.label_tables,
                                                      missing_ok=True))
            self.stamp = stamp
        return shot in self.shots


def install(app: FastAPI, paths: Paths) -> None:
    """Add this module's routes to `app`."""
    from .app import require_event  # app imports this module

    signals = Signals(paths)
    app.state.workspace = signals
    blind = _Blind(paths)

    def reviewable(shot: int) -> None:
        if shot in blind:
            raise FetchError("missing", f"shot {shot} is not reviewable here")

    @app.get("/api/catalog")
    def catalog_view(request: Request):
        query = request.query_params
        event = query.get("event") or None
        if event is not None:
            require_event(event, paths)
        try:
            shot = None if query.get("shot") in (None, "") else _shot(query["shot"])
            if shot is not None:
                reviewable(shot)
            table = signals.catalog()
            found = signals.availability(shot, table.entries) if shot is not None \
                else {}
        except FetchError as error:
            return _error(error)
        entries = []
        for entry in table.entries:
            local, source = found.get(entry.id) or (
                "disk" if entry.sources[0].local else "fetch", entry.sources[0])
            entries.append({
                "id": entry.id, "label": entry.label, "group": entry.group,
                "units": entry.units, "kind": entry.kind, "channels": entry.channels,
                "axis": entry.axis, "suggested": entry.suggested(event),
                "source": source.wire, "where": source.where(), "local": local,
                "description": entry.description, "points": list(entry.points),
                **({"draw": entry.draw} if entry.draw else {}),
                **({"avail": entry.avail} if entry.avail else {}),
            })
        return {"entries": entries, "trees": list(cat.TREES),
                "ptdata_pattern": cat.PTDATA.pattern, "node_pattern": cat.NODE_PATTERN,
                "node_max": cat.NODE_MAX}

    @app.get("/api/signal")
    def signal_view(request: Request):
        query = request.query_params
        try:
            shot = _shot(query.get("shot"))
            t0 = _number(query.get("t0"), "t0")
            t1 = _number(query.get("t1"), "t1")
            cols = _cols(query.get("cols"))
            reviewable(shot)
            got = signals.trace(shot, query.get("id"), t0, t1, cols)
        except FetchError as error:
            return _error(error)
        except OSError as error:
            log.exception("/api/signal %s", query.get("id"))
            return _error(FetchError("upstream", f"could not read it: {error}"))
        if isinstance(got, Pending):
            return JSONResponse(got.body(), status_code=202)
        return Response(got.data, media_type="application/octet-stream", headers={
            "X-Grid": json.dumps(got.grid), "X-Signal": json.dumps(got.signal),
        })

    @app.get("/api/profile")
    def profile_view(request: Request):
        query = request.query_params
        try:
            shot = _shot(query.get("shot"))
            reviewable(shot)
            got = profiles.read(signals, shot, query.get("id"))
        except FetchError as error:
            return _error(error)
        except OSError as error:
            log.exception("/api/profile %s", query.get("id"))
            return _error(FetchError("upstream", f"could not read it: {error}"))
        if isinstance(got, Pending):
            return JSONResponse(got.body(), status_code=202)
        entry = signals.resolve(query.get("id"))
        return Response(got.body(), media_type="application/octet-stream",
                        headers={"X-Profile": json.dumps(got.header(entry))})

    @app.get("/api/fetch/status")
    def fetch_status():
        return signals.status()

    @app.get("/api/views")
    def views_list(event: str, reviewer: str | None = None):
        require_event(event, paths)
        try:
            return {"views": saved_views.list_views(paths, event, reviewer)}
        except saved_views.ViewError as error:
            return JSONResponse({"error": error.message}, status_code=error.status)

    @app.post("/api/views")
    async def views_change(request: Request):
        raw = await request.body()
        if len(raw) > BODY_MAX:
            return JSONResponse({"error": "a request is at most 256 KB"},
                                status_code=400)
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            return JSONResponse({"error": "send a JSON object"}, status_code=400)
        if isinstance(body, dict) and isinstance(body.get("event"), str):
            require_event(body["event"], paths)
        try:
            found = await run_in_threadpool(saved_views.change, paths, body)
        except saved_views.ViewError as error:
            return JSONResponse({"error": error.message}, status_code=error.status)
        return {"views": found}
