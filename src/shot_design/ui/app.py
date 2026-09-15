"""Browser transport, ported from shot-recommender-system's shotrec/ui/app.py.

The token gate and static app structure come from shotrec. Retrieval belongs to the
existing MCP tools and phenomenon CLI path. Successful tool replies are serialized
verbatim with default=str; the MCP's own guard also handles incomplete databases.
"""

from __future__ import annotations

import json
import secrets
import threading
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import config
from ..mcp import tools
from ..retrieval import phenomena

STATIC = Path(__file__).parent / "static"
COOKIE = "ideate_token"
_LOCATE_LOCK = threading.Lock()


def _json(value, **kwargs) -> Response:
    return Response(json.dumps(value, default=str), media_type="application/json", **kwargs)


def _registry() -> list[dict]:
    return [
        {"id": pid, "title": ph.title, "aliases": ph.aliases,
         "sources": ph.sources, "covering_sources": ph.covering_sources}
        for pid, ph in sorted(phenomena.registry().items())
    ]


def create_app(
    db_dir: Path | None = None, token: str | None = None, paths: config.Paths | None = None
) -> FastAPI:
    paths = paths or config.load_paths()
    if db_dir is not None:
        paths = paths.model_copy(update={"db_dir": Path(db_dir)})
    app = FastAPI(title="shot_design", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.paths = paths
    app.state.token = token or secrets.token_hex(16)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        expected = app.state.token.encode("utf-8")
        given = request.query_params.get("token")
        if given is not None:
            if not secrets.compare_digest(given.encode("utf-8"), expected):
                return _unauthorized(request, "bad token")
            url = request.url.remove_query_params("token")
            path = url.path if not url.path.startswith("//") else "/"
            response = RedirectResponse(path + (f"?{url.query}" if url.query else ""), 303)
            response.set_cookie(COOKIE, app.state.token, httponly=True, samesite="lax")
            return response
        cookie = request.cookies.get(COOKIE, "")
        if not cookie or not secrets.compare_digest(cookie.encode("utf-8"), expected):
            return _unauthorized(request, "no token: reopen the link printed by shot_design serve")
        # ContextVar follows this request into FastAPI's synchronous worker thread.
        with config.using_paths(app.state.paths):
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/meta")
    def meta():
        def summary():
            info = {
                "segments": list(tools.SEGMENTS),
                "phenomena": [
                    {"id": ph["id"], "title": ph["title"],
                     "has_detector": bool(ph["covering_sources"])}
                    for ph in _registry()
                ],
            }
            db, error = tools._db()
            if error:
                return {**info, **error}
            shots = sorted(int(s) for s in db.shots.index)
            return {
                "db": {
                    "n_shots": len(shots),
                    "shot_range": [shots[0], shots[-1]] if shots else None,
                    "n_segments": db.manifest.get("n_segments", len(db.segments)),
                    "built_at": db.manifest.get("built_at"),
                    "git_sha": db.manifest.get("git_sha"),
                },
                **info,
            }

        return _json(tools.never_raises(summary)())

    @app.post("/api/search")
    def search(body: dict):
        return _json(tools.never_raises(tools.search_shots)(**body))

    @app.get("/api/shot/{shot}")
    def shot(shot: int, segment: str = "flat_top"):
        return _json(tools.never_raises(tools.describe_shot)(shot, segment))

    @app.get("/api/shot/{shot}/events")
    def events(
        shot: int, phenomenon: str | None = None,
        t0_s: float | None = None, t1_s: float | None = None,
    ):
        return _json(tools.never_raises(tools.get_events)(
            shot, phenomenon=phenomenon, t0_s=t0_s, t1_s=t1_s,
        ))

    @app.get("/api/phenomena")
    def registry():
        return _json(tools.never_raises(_registry)())

    @app.get("/api/locate")
    def locate(
        phenomenon: str, n: int = 20, segment: str = "flat_top",
        min_confidence: float = 0.0,
        avoid: Annotated[list[str] | None, Query()] = None,
    ):
        notes: list[str] = []

        def run():
            # cmd_phenomenon --json: resolve, locate, model_dump. Do not rank here.
            resolved = phenomena.resolve(phenomenon)
            if not resolved:
                titles = ", ".join(
                    f"{pid} ({ph.title})" for pid, ph in phenomena.registry().items()
                )
                return {"error": f"no phenomenon resolved; try one of: {titles}",
                        "caveats": []}
            db, error = tools._db()
            if error:
                return error
            hits = phenomena.locate(
                resolved[0][0], db, n, segment=segment, min_confidence=min_confidence,
                avoid=avoid or (), notes=notes,
            )
            return [h.model_dump(mode="json") for h in hits]

        # The upstream curated-list cache keys on the config name, not its resolved
        # path. Serialize UI locate calls and reset it so two app factories cannot
        # reuse each other's CSV membership. Retrieval itself stays in locate().
        with _LOCATE_LOCK:
            phenomena._database_shots.cache_clear()
            payload = tools.never_raises(run)()
        # CLI stderr becomes a header: the JSON body remains the CLI's bare list.
        return _json(payload, headers={"X-Ideate-Caveats": json.dumps(notes)})

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unknown_api(path: str):
        return JSONResponse({"error": f"unknown API route: /api/{path}"}, status_code=404)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _unauthorized(request: Request, why: str):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": why}, status_code=401)
    return HTMLResponse(f"<!doctype html><title>401</title><h1>401</h1><p>{why}</p>", 401)
