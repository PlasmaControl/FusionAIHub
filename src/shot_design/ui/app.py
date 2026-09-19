"""Browser transport, ported from shot-recommender-system's shotrec/ui/app.py.

The token gate and static app structure come from shotrec. Retrieval belongs to the
existing MCP tools and phenomenon CLI path. Tool fields are serialized verbatim
with default=str, alongside browser-only descriptions, units and timeline domains;
the MCP's own guard also handles incomplete databases.
"""

from __future__ import annotations

import json
import math
import secrets
import threading
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import config, schema
from ..mcp import tools
from ..retrieval import phenomena
from ..retrieval.describe import describe_parts, phenomenon_rows, scalar_units
from .scoring import scoring_info

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


def _db_summary(db) -> dict:
    """Shared metadata for the banner and the scoring page."""
    shots = sorted(int(s) for s in db.shots.index)
    return {
        "n_shots": len(shots),
        "shot_range": [shots[0], shots[-1]] if shots else None,
        "n_segments": db.manifest.get("n_segments", len(db.segments)),
        "built_at": db.manifest.get("built_at"),
        "git_sha": db.manifest.get("git_sha"),
    }


def _filter_confidence(payload: dict, limit: float) -> dict:
    """HTTP-only filter on timed evidence; preserve MCP status and named claims."""
    if limit == 0:
        return payload
    result = {**payload, "caveats": list(payload.get("caveats", []))}
    for key, count in (("events", "n"), ("forecasts", "n_forecasts")):
        rows = payload.get(key, [])
        kept = [r for r in rows if phenomena._keeps_confidence(r, limit) is True]
        result[key], result[count] = kept, len(kept)
        if len(rows) != len(kept):
            result["caveats"].append(
                f"{len(rows) - len(kept)} {key} excluded: confidence below {limit} or unrecorded"
            )
        if key == "forecasts" and rows:
            old = tools._FORECAST_CAVEAT.format(n=len(rows))
            result["caveats"] = [c for c in result["caveats"] if c != old]
            if kept:
                result["caveats"].append(tools._FORECAST_CAVEAT.format(n=len(kept)))
    return result


def _timeline_domain(rec: schema.ShotRecord) -> dict:
    """Shot scale in seconds, independent of coverage and event-window filters.

    Prefer the full segment, then the union of valid segments, then -2..8 s.
    Pad with min(-2, floor(t0_s)) .. max(8, ceil(t1_s + 0.5)). Stored segment
    times are milliseconds. Malformed legacy spans cannot define an axis.
    """
    segments = [s for s in rec.segments if math.isfinite(s.t0_ms) and
                math.isfinite(s.t1_ms) and s.t1_ms >= s.t0_ms]
    full = next((s for s in segments if s.name == "full"), None)
    if full is not None:
        segments = [full]
    if not segments:
        return {"t0_s": -2, "t1_s": 8, "source": "default"}
    return {
        "t0_s": min(-2, math.floor(min(s.t0_ms for s in segments) / 1000)),
        "t1_s": max(8, math.ceil(max(s.t1_ms for s in segments) / 1000 + 0.5)),
        "source": "full segment" if full is not None else "segments",
    }


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
            return {
                "db": _db_summary(db),
                **info,
            }

        return _json(tools.never_raises(summary)())

    @app.get("/api/scoring")
    def scoring():
        def summary():
            db, error = tools._db()
            return error if error else scoring_info(_db_summary(db), db._phenomenon_config)

        return _json(tools.never_raises(summary)())

    @app.post("/api/search")
    def search(body: dict):
        return _json(tools.never_raises(tools.search_shots)(**body))

    @app.get("/api/shot/{shot}")
    def shot(shot: int, segment: str = "flat_top"):
        def structured():
            payload = tools.describe_shot(shot, segment)
            if "error" in payload:
                return payload
            db, error = tools._db()
            if error:
                return error
            rec = db.get(shot)
            return {
                **payload,
                "describe_parts": describe_parts(rec, payload["segment"], db=db),
                "units": scalar_units(
                    key for seg in rec.segments for key in {**seg.raw, **seg.derived}
                ),
            }

        return _json(tools.never_raises(structured)())

    @app.get("/api/shot/{shot}/events")
    def events(
        shot: int, phenomenon: str | None = None,
        t0_s: float | None = None, t1_s: float | None = None,
        min_confidence: float = 0.0,
    ):
        def with_domain():
            if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
                return {"error": "Minimum confidence must be between 0 and 1", "caveats": []}
            payload = tools.get_events(
                shot, phenomenon=phenomenon, t0_s=t0_s, t1_s=t1_s,
            )
            if "error" in payload:
                return payload
            db, error = tools._db()
            if error:
                return error
            payload = _filter_confidence(payload, min_confidence)
            rec = db.get(shot)
            window = None if t0_s is None and t1_s is None else (t0_s, t1_s)
            # Untimed legacy claims remain in the tool payload. They cannot make
            # a drawable Interval or break an otherwise valid event response.
            timed = [r for r in payload.get("events", []) + payload.get("forecasts", [])
                     if all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                            for k in ("t0_s", "t1_s")) and r["t1_s"] >= r["t0_s"]]
            rows = phenomenon_rows(
                rec, None, db=db, phenomenon=phenomenon, window=window,
                event_rows=timed,
                min_confidence=min_confidence,
            )
            return {**payload, "domain": _timeline_domain(rec), "phenomena": rows}

        return _json(tools.never_raises(with_domain)())

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
            return [
                {**h.model_dump(mode="json"), "domain": _timeline_domain(db.get(h.shot))}
                for h in hits
            ]

        # The upstream curated-list cache keys on the config name, not its resolved
        # path. Serialize UI locate calls and reset it so two app factories cannot
        # reuse each other's CSV membership. Retrieval itself stays in locate().
        with _LOCATE_LOCK:
            phenomena._database_shots.cache_clear()
            payload = tools.never_raises(run)()
        # CLI stderr becomes a header; the bare hit list adds only UI domains.
        return _json(payload, headers={"X-Ideate-Caveats": json.dumps(notes)})

    from .design_routes import router as design_router
    from .assistant_routes import router as assistant_router

    app.include_router(design_router)
    app.include_router(assistant_router)

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unknown_api(path: str):
        return JSONResponse({"error": f"unknown API route: /api/{path}"}, status_code=404)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _unauthorized(request: Request, why: str):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": why}, status_code=401)
    return HTMLResponse(f"<!doctype html><title>401</title><h1>401</h1><p>{why}</p>", 401)
