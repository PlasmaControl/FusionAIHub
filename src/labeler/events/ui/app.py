"""Browser transport for label verification.

The token gate and the static-app structure come from `shot_design/ui/app.py`,
which took them from shot-recommender-system. Panels come from the panel
registry and corrections go through `verify.write_corrections`, which is the
same append-only path the notebooks used.

This app READS `shots.csv` and never writes it. That file is how a person
tracks what has been processed, and it holds the hand-set `tier` and
`holdout` calls; a surface that edited it on a button press is a surface that
can destroy somebody's curation by accident. Note in particular that
`ReviewSession.save()` is NOT used here - it calls `record_review`, which
writes that file.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...config import Paths
from .. import panels as registry
from .. import rosters
from ..verify import corrections_for

STATIC = Path(__file__).parent / "static"
COOKIE = "labeler_verify_token"
NO_TOKEN = "no token: reopen the link printed by the verify server"
BAD_TOKEN = "bad token"


def _json(value, **kwargs) -> Response:
    return Response(
        json.dumps(value, default=str), media_type="application/json", **kwargs
    )


def _unauthorized(why: str) -> Response:
    return JSONResponse({"error": why}, status_code=401)


def _events(paths: Paths) -> list[dict]:
    """Every event directory that has a roster, with its builder and guidance.

    The rosters are hand-edited, so one of them being malformed is a normal
    Tuesday. A single bad file must not hide the other fifteen events, and it
    must not vanish either: it comes back carrying its own `error` so the
    reviewer can see which roster to go and fix.
    """
    rows = []
    try:
        directories = sorted(p for p in paths.label_tables.iterdir() if p.is_dir())
    except OSError:
        # A label_tables root that is missing or unreadable is a server
        # misconfiguration, not a reason to answer every request with a 500.
        return rows
    for directory in directories:
        roster = directory / rosters.ROSTER_NAME
        if not roster.is_file():
            continue
        event = directory.name
        row = {
            "event": event,
            # "generic" here is the honest answer and the page says so:
            # a reviewer should know the panels in front of them were not
            # chosen for this phenomenon.
            "builder": event if event in registry.BUILDERS else "generic",
            "guidance": registry.guidance(event),
            "n_shots": None,
        }
        try:
            row["n_shots"] = len(rosters.read_roster(roster))
        except Exception as error:  # noqa: BLE001 - any unreadable roster
            row["error"] = str(error) or error.__class__.__name__
        rows.append(row)
    return rows


def require_event(event: str, paths: Paths) -> str:
    """Return `event` if it names a real event directory, else raise 404.

    Without this, `event` is a client-supplied path segment: `../elsewhere`
    escapes `label_tables`, and an ABSOLUTE value reaches anywhere the
    server's uid can read, because pathlib's `/` discards the left operand.
    Every endpoint taking an `event` must come through here first - the read
    endpoints today, and any endpoint that later WRITES under `event`.
    """
    if event not in {row["event"] for row in _events(paths)}:
        raise HTTPException(status_code=404, detail=f"unknown event {event!r}")
    return event


def create_app(paths: Paths | None = None, token: str | None = None) -> FastAPI:
    paths = Paths.from_env() if paths is None else paths
    app = FastAPI(
        title="labeler verify", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.paths = paths
    app.state.token = token or secrets.token_hex(16)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        expected = app.state.token.encode("utf-8")
        given = request.query_params.get("token")
        if given is not None:
            if not secrets.compare_digest(given.encode("utf-8"), expected):
                return _unauthorized(BAD_TOKEN)
            url = request.url.remove_query_params("token")
            path = url.path if not url.path.startswith("//") else "/"
            response = RedirectResponse(
                path + (f"?{url.query}" if url.query else ""), 303
            )
            response.set_cookie(COOKIE, app.state.token, httponly=True, samesite="lax")
            return response
        cookie = request.cookies.get(COOKIE, "")
        if not cookie or not secrets.compare_digest(cookie.encode("utf-8"), expected):
            return _unauthorized(NO_TOKEN)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(StarletteHTTPException)
    async def as_error(request: Request, exc: StarletteHTTPException) -> Response:
        """One error shape for the page: every refusal reads `{"error": ...}`."""
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.get("/api/events")
    def events():
        return _json({"events": _events(app.state.paths)})

    @app.get("/api/shots")
    def shots(event: str):
        paths = app.state.paths
        event = require_event(event, paths)
        frame = rosters.read_roster(rosters.roster_path(event, root=paths.label_tables))
        rows = []
        for record in frame.to_dict("records"):
            shot = int(record["shot"])
            rows.append(
                {
                    "shot": shot,
                    "tier": record.get("tier", ""),
                    # The roster stores holdout as the STRING "true"/"false",
                    # so `bool(...)` on the cell would call every shot a
                    # holdout and quietly hide the whole roster from training.
                    "holdout": record.get("holdout", "") == "true",
                    "reviewers": rosters.split_reviewers(record.get("reviewers", "")),
                    "verified_on": record.get("verified_on", ""),
                    "notes": record.get("notes", ""),
                    # Files on disk, counted fresh: a review saved a moment
                    # ago shows here against an unrecorded reviewer, which is
                    # the prompt to go edit shots.csv by hand.
                    "n_corrections": len(
                        corrections_for(event, shot, root=paths.label_tables)
                    ),
                }
            )
        return _json({"event": event, "shots": rows})

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
