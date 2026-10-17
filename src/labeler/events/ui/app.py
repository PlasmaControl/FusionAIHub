"""The label review server: one label per shot, over rows prebuilt per shot.

Loopback only, behind a token: `?token=` sets two cookies and redirects to
the same URL without it. The server reads `shots.csv` and never writes it;
its writes are `POST /api/label` and `POST /api/masks`, both into the event's
`review/` directory, and `POST /api/names`, which adds a reviewer's name to
`reviewers.txt` beside the events (`GET /api/names` lists them; see
`review.reviewers`). `GET /api/history` lists a shot's saved versions (see
`review.versions`), and `GET /api/shot` and both saves name everyone who has
saved the shot (see `review.reviewers.shot_reviewers`); `GET /api/masks` gives
an AE shot's pseudo-mask regions (pseudo-v1-full: pseudo-v1's rules over the
whole window, 60-250 kHz) and the reviewer's last
word on them (see `labeler.ae.seg.regions`); `GET /api/tokeye` gives TokEye's
lines over the whole shot, drawn under them (see `labeler.ae.seg.whole`).

A subcategory (`mhd_n1.m1`, `alfven_eigenmode.rsae`; see `taxonomy`) is a subset of its
parent: it is read and saved only inside the time the parent's SAVED label marks (see
`review.labels.Region`), over the parent's roster, window and rows; the parent unsaved
on a shot is a 409. `GET /api/events` lists top-level events, each with its display
name, group and subcategories.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...ae import seg
from ...ae.seg import regions, whole
from ...ae.seg.pseudo import PseudoMask
from ...config import Paths, sha256_of, store_event
from .. import grid, ondemand, raw, rosters, rwm, shotlists, taxonomy
from ..review import build as review_build
from ..review import gold, labels, reviewers, rows, silver, versions, video
from . import overlay_routes, shotlist_routes, slice_routes, workspace_routes

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
#: name and hides the history instead of failing every save. 3 added the masks,
#: 4 the whole-shot TokEye layer, 5 the list of names the page asks from,
#: 6 the exact RWM onset annotations, 7 individual/group annotation resolution,
#: 8 independent, overlapping individual and span (iscrowd) annotation lanes,
#: 9 detachment camera manifests and lazy frame requests.
#: 10 saved suggestion exposure and producer-prefill provenance.
#: 11 named shot lists (the queue's `list=`, `/api/shotlists`, `/api/overview`),
#: build status, TokEye row states and the signal workspace.
#: 12 on-demand shots: `/api/shots/search`, the 202 body's `stage` (queued, fetching
#: or building only), `fetching` and `note`, the 502 body's `detail` and `code`, and
#: any six-digit shot opening on any event.
API_VERSION = 12

log = logging.getLogger(__name__)


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    event: str
    shot: int
    window: tuple[float, float]
    intervals: list[tuple[float, float, int]] = Field(max_length=1000)
    iscrowd: list[object] | None = Field(default=None, max_length=1000)
    overlap_edit: bool = False
    name: str | None = Field(default=None, max_length=versions.NAME_MAX)
    suggestions_shown: bool | None = None
    prefilled: bool | None = None


class MaskIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str
    shot: int
    #: The version of the pseudo-mask the page drew; a save on another is refused.
    pseudo_sha256: str = Field(min_length=64, max_length=64)
    revision: int = Field(ge=0)
    rejected: list[int] = Field(max_length=regions.MAX_REGIONS)
    name: str | None = Field(default=None, max_length=versions.NAME_MAX)


class NameIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=versions.NAME_MAX)


class Builds:
    """Store files built on first open, two at a time, off the request thread.
    `progress` says where one stands, for the 202 body."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.running: dict = {}
        self.lock = threading.Lock()
        #: Each build's submit order and time, and when a worker took it.
        self.asked: dict = {}
        self.started: dict = {}
        self.order = 0

    def _build(self, key: tuple, event: str, shot: int) -> Path:
        with self.lock:
            self.started[key] = time.monotonic()
        return ondemand.build(event, shot, self.paths)

    def progress(self, event: str, shot: int) -> dict:
        """`stage` (queued, fetching or building), `fetching` (the signals being
        fetched), `note` (a line of at most `ondemand.NOTE_MAX` characters for the
        activity bar), `elapsed_s` since it was asked for, `queue_position` (0 while it runs,
        else its place among the waiting) and `prebuilt` (a registered shot list
        holds the shot)."""
        key = (store_event(event), shot)
        now = time.monotonic()
        with self.lock:
            order, asked = self.asked.get(key, (self.order + 1, now))
            started = key in self.started
            ahead = sum(
                1 for other, (o, _) in self.asked.items()
                if o < order and other not in self.started
                and other in self.running and not self.running[other].done()
            )
        fetching = ondemand.fetching_signals(shot) if started else []
        if not started:
            stage = "queued"
        elif fetching or overlay_routes.build_stage(self.paths, key[0], shot) == "fetching":
            stage = "fetching"
        else:
            stage = "building"
        return {
            "stage": stage,
            "fetching": fetching,
            "note": ondemand.note(stage, fetching, ahead),
            "elapsed_s": round(now - asked, 1),
            "queue_position": 0 if started else ahead + 1,
            "prebuilt": overlay_routes.prebuilt(self.paths, shot),
        }

    def status(self, event: str, shot: int) -> tuple[Path | None, str | None]:
        """`(path, None)` once built, `(None, error)` once failed, else building."""
        event = store_event(event)
        key = (event, shot)
        path = self.paths.spectrogram_file(event, shot)
        with self.lock:
            future = self.running.get(key)
            if future is not None and not future.done():
                return None, None
            self.running.pop(key, None)
            self.asked.pop(key, None)
            self.started.pop(key, None)
            if future is not None and future.exception() is not None:
                return None, ondemand.explain(future.exception(), event, shot)
            if review_build.current(path, event):
                if ondemand.in_cache(self.paths, path):
                    ondemand.touch(self.paths, shot)
                return path, None
            if event == "detachment" and path.is_file() and not ondemand.in_cache(
                    self.paths, path):
                return None, ondemand.BuildFailure(
                    f"The detachment store for {shot} is out of date",
                    f"Stale frozen detachment store: {path}. Opening a shot never "
                    "overwrites it. The controller must run the documented "
                    "resume-build after the producer is final.",
                    "build_failed",
                )
            self.order += 1
            self.asked[key] = (self.order, time.monotonic())
            self.running[key] = self.pool.submit(self._build, key, event, shot)
            return None, None


def require_event(event: str, paths: Paths) -> Path:
    """The event's directory, else 404; decided on the name before any join.

    A subcategory (`<parent>.<key>`) is looked up in the taxonomy, never joined from
    the request: its directory is `taxonomy.Taxonomy.child_dir`, which need not exist
    until the first save. Its parent must be a known event.
    """
    if event != Path(event).name or event in {"", ".", ".."} or "\0" in event:
        raise HTTPException(404, f"unknown event {event!r}")
    if event in labels.FOLDED:
        raise HTTPException(404, f"{event} is reviewed as {labels.FOLDED[event]}")
    if "." in event:
        tree = _taxonomy(paths)
        if not tree.is_child(event):
            raise HTTPException(404, f"unknown event {event!r}")
        require_event(tree.parent(event), paths)
        return tree.child_dir(paths.label_tables, event)
    directory = paths.label_tables / event
    try:
        found = (directory / rosters.ROSTER_NAME).is_file()
    except OSError:  # ENAMETOOLONG is raised, not reported as False
        found = False
    if not found:
        raise HTTPException(404, f"unknown event {event!r}")
    return directory


def _taxonomy(paths: Paths) -> taxonomy.Taxonomy:
    try:
        return taxonomy.load(paths.events_manifest)
    except taxonomy.TaxonomyError as error:
        raise HTTPException(500, str(error)) from None


def roster_directory(event: str, paths: Paths) -> Path:
    """Where the roster of `event` lives: a subcategory shares its parent's."""
    parent = _taxonomy(paths).parent(event)
    return require_event(event if parent is None else parent, paths)


def _roster(directory: Path):
    return rosters.read_roster(directory / rosters.ROSTER_NAME)


def _region(
    event: str, shot: int, paths: Paths
) -> tuple[str, Path, labels.Label, labels.Region]:
    """`(parent, parent directory, parent label, region)` a subcategory shot may use.

    The region is where the parent's SAVED label marks something; a parent that is
    not saved on the shot opens nothing (409), so a subcategory is never labelled
    before its parent is.
    """
    tree = _taxonomy(paths)
    parent = tree.parent(event)
    directory = require_event(parent, paths)
    saved = labels.page_label(
        parent, labels.read_labels(labels.labels_path(directory)).get(int(shot))
    )
    region = labels.region_of(parent, saved)
    if region is None:
        raise HTTPException(409, f"review {tree.display(parent)} first")
    return parent, directory, saved, region


def roster_tier(roster, shot: int) -> str:
    match = roster[roster.shot == int(shot)]
    if match.empty:
        raise HTTPException(404, f"shot {int(shot)} is not on this event's roster")
    return str(match.tier.iloc[0])


def shot_tier(event: str, shot: int, paths: Paths) -> str:
    """The shot's roster tier; a shot on any shot list may be reviewed on any event,
    on its roster or not (`shotlists.in_any`), and so may any six-digit shot that is not
    blind (`ondemand.openable`: it builds if fdp has data); it is unverified off the
    roster."""
    roster = _roster(roster_directory(event, paths))
    if not (roster.shot == shot).any() and (
            shotlists.in_any(paths, shot) or ondemand.openable(paths, shot)):
        return "unverified"
    return roster_tier(roster, shot)


def borrowed(event: str, shot: int, paths: Paths) -> bool:
    """Whether `event` reviews `shot` though it is not its own: a silver or gold shot,
    or one off the event's roster that some list holds or that is any open six-digit
    shot. Only such a shot skips a broken source pointer and may open on a silver
    draft; a roster shot keeps its source table (`labels.source_path` raises on a
    pointer to a missing table)."""
    tables = paths.label_tables
    if silver.is_silver(tables, shot) or gold.is_gold(tables, shot):
        return True
    roster = _roster(roster_directory(event, paths))
    return not (roster.shot == shot).any() and (
        shotlists.in_any(paths, shot) or ondemand.openable(paths, shot))


def shot_labels(event: str, shot: int, paths: Paths, *, suppress_source=False,
                region=None, onsets=None,
                t_range=None) -> tuple[dict, Path | None, str | None]:
    """Snapped page labels, with a silver fallback and its exact source table.

    `onsets` (ms) are the rows file's candidate onsets, or RWM's curated ones: a shot
    with no save opens with them in its draft as Individual annotations
    (`labels.with_onsets`, each over its present span for `labels.ONSET_EXTENTS`),
    over `t_range` when it has no source label. Only `/api/shot` passes them; a save
    compares with the source table as it is."""
    directory = require_event(event, paths)
    saved = labels.page_label(
        event, labels.read_labels(labels.labels_path(directory)).get(shot)
    )
    source, table, origin = None, None, None
    if not suppress_source:
        other = borrowed(event, shot, paths)
        try:
            table = labels.source_path(directory)
            source = labels.page_label(
                event, labels.read_labels(table).get(shot) if table else None
            )
        except (ValueError, OSError) as error:
            if not other:
                raise
            log.warning("silver source skipped for %s/%s: %s", event, shot, error)
        if source is None and other:
            found = shotlist_routes.draft(paths, event, shot)
            if found is not None:
                source, layer = found
                source = labels.span_default(event, source)
                table, origin = silver.layer_file(paths, layer), layer.name
    if region is not None:
        saved, source = region.clip(saved), region.clip(source)
    candidates = 0
    if onsets and saved is None and not suppress_source:
        drafted = labels.with_onsets(source, onsets, t_range,
                                     extend=event in labels.ONSET_EXTENTS)
        if drafted is not source:
            candidates = sum(flag == 0 for flag in drafted.iscrowd)
        source = drafted
    last = next((entry for entry in reversed(labels.read_history(directory))
                 if entry["shot"] == shot), None)
    return ({
        "source": None if source is None else source.as_json(),
        "saved": None if saved is None else saved.as_json(),
        "state": labels.state(saved, source), "last_save": last,
        **({"source_origin": origin} if origin is not None else {}),
        **({"onset_candidates": candidates} if candidates else {}),
    }, table, origin)


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
    # The server serves the shots it built on demand from the temporary cache; no other
    # code's `Paths` (pipelines, the prebuild CLI, the paper) finds the cache.
    paths = replace(paths, read_cache=True)
    app = FastAPI(
        title="labeler review", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.paths = paths
    app.state.token = token or secrets.token_hex(16)
    app.state.builds = Builds(paths)
    if (paths.label_tables / silver.SILVER_FILE).is_file():
        threading.Thread(target=silver.warm, args=(paths,), daemon=True).start()
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        # One log line per request, by PATH: the token rides in the query string.
        started = time.monotonic()
        response = _admit(request, app.state.token)
        if response is None:
            response = await call_next(request)
        response.headers["Cache-Control"] = overlay_routes.cache_control(request, response)
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
        return {"api": API_VERSION, **workspace_routes.version_extra(paths)}

    @app.get("/api/names")
    def names_view():
        return {"names": reviewers.read(paths.label_tables)}

    @app.post("/api/names")
    def add_name(body: NameIn):
        with names_lock:
            try:
                names, name = reviewers.add(paths.label_tables, body.name)
            except ValueError as error:
                raise HTTPException(400, str(error)) from None
        return {"names": names, "name": name}

    #: The events list, until the taxonomy, a roster or a saves table changes.
    events_cache = shotlist_routes.Stamped(size=1)

    @app.get("/api/events")
    def events():
        tree = _taxonomy(paths)
        try:
            directories = sorted(
                p
                for p in paths.label_tables.iterdir()
                if p.name not in labels.FOLDED and (p / rosters.ROSTER_NAME).is_file()
                and tree.reviewed(p.name)
            )
        except OSError:
            return {"events": []}
        inputs = [Path(paths.events_manifest), Path(paths.label_tables)]
        for directory in directories:
            inputs += [directory / rosters.ROSTER_NAME, labels.labels_path(directory)]
            inputs += [labels.labels_path(tree.child_dir(paths.label_tables, child))
                       for child in tree.children(directory.name)]
        return events_cache.get(
            "events", inputs, lambda: {"events": _events(tree, directories)}
        )

    def _events(tree: taxonomy.Taxonomy, directories: list[Path]) -> list[dict]:
        found = []
        for directory in directories:
            event = directory.name
            named = {"display_name": tree.display(event), "group": tree.group(event)}
            try:
                roster = _roster(directory)
                saved = labels.read_saved(directory)
                categories = labels.categories(event)
                found.append(
                    {
                        "event": event,
                        **named,
                        "n_shots": len(roster),
                        "n_reviewed": int(roster.shot.isin(list(saved)).sum()),
                        "categories": {str(k): v for k, v in categories.items()},
                        "subcategories": [
                            _subcategory(tree, child, roster) for child in
                            tree.children(event)
                        ],
                    }
                )
            # One bad roster must not hide the rest.
            except Exception as error:  # noqa: BLE001
                message = str(error) or type(error).__name__
                found.append({"event": event, **named, "error": message})
        return found

    def _subcategory(tree: taxonomy.Taxonomy, child: str, roster) -> dict:
        """One chip of the subcategory bar: its name and how many shots it has saved."""
        entry = {"event": child, "display_name": tree.display(child)}
        try:
            saved = labels.read_saved(tree.child_dir(paths.label_tables, child))
            return {**entry, "n_reviewed": int(roster.shot.isin(list(saved)).sum())}
        except Exception as error:  # noqa: BLE001 - one bad chip must not hide the event
            return {**entry, "n_reviewed": 0, "error": str(error) or type(error).__name__}

    def _gold_extras(directory: Path, roster) -> list[dict]:
        """Gold shots this event's roster lacks, flagged `extra`: the page offers them
        only with "Gold set" ticked, since a gold shot is reviewable on every event."""
        try:
            listed = sorted(int(s) for s in gold.read_gold(paths.label_tables).shot)
        except (FileNotFoundError, ValueError, OSError):
            return []
        missing = [s for s in listed if s not in set(roster.shot)]
        if not missing:
            return []
        rows = pd.DataFrame({"shot": missing, "tier": "unverified", "holdout": "false",
                             "reviewers": "", "verified_on": "", "notes": ""})
        return [{**row, "extra": True} for row in labels.queue(directory, rows)["shots"]]

    #: Each event's queue per list asked for, until a file it was read from changes.
    queues = shotlist_routes.Stamped()

    @app.get("/api/queue")
    def queue(event: str, shotlist: Annotated[str | None, Query(alias="list")] = None):
        """The event's queue. With no `list` it is the roster as before, with the gold
        shots it lacks flagged `extra`; with one, that list's shots in its order, the
        ones off the roster flagged `extra` (a subcategory's: those whose parent's
        label marks something). Either way `list` names what is shown, `lists` is
        `/api/shotlists` and `note` says why a list asked for is not shown."""
        directory = require_event(event, paths)
        parent = _taxonomy(paths).parent(event)
        parent_dir = None if parent is None else require_event(parent, paths)
        requested = shotlist or None
        return queues.get(
            (event, requested),
            shotlist_routes.queue_inputs(paths, directory, parent_dir),
            lambda: _queue(event, requested, directory, parent, parent_dir),
        )

    def _queue(event: str, requested: str | None, directory: Path,
               parent: str | None, parent_dir: Path | None) -> dict:
        roster = _roster(directory if parent_dir is None else parent_dir)
        offered = shotlist_routes.offer(
            paths, event, labels.read_saved(directory), roster
        )
        name, note = shotlist_routes.choose(paths, event, requested, offered)
        listed = (requested is not None and name != shotlists.ROSTER
                  and name == shotlists.resolve(paths, requested, event))
        rows = (shotlist_routes.as_roster(roster, shotlists.shots(paths, name))
                if listed else roster)
        if parent_dir is not None:
            found = labels.child_queue(event, directory, parent, parent_dir, rows)
        else:
            found = labels.queue(directory, rows)
            if not listed:
                found["shots"] += _gold_extras(directory, roster)
        if listed:
            on_roster = {int(shot) for shot in roster.shot}
            found["shots"] = [row if row["shot"] in on_roster else {**row, "extra": True}
                              for row in found["shots"]]
        return {**found, "list": name, "lists": offered,
                **({"note": note} if note else {})}

    def silver_events():
        return shotlist_routes.review_events(paths)

    @app.get("/api/silver")
    def silver_view():
        try:
            frame = silver.read_silver(paths.label_tables)
        except FileNotFoundError:
            return {"grid_ms": grid.GRID_MS, "events": silver_events(), "batches": [],
                    "error": "Silver list missing: silver/silver_shots.csv"}
        batches = []
        for batch in range(1, 6):
            shots = []
            for row in frame[frame.batch == batch].itertuples():
                progress = silver.progress(paths, int(row.shot))
                saved, total = len(progress["saved"]), progress["total"]
                shots.append({
                    "shot": int(row.shot), "order": int(row.order), "pool": row.pool,
                    "classes": [part for part in row.classes.split(";") if part],
                    "saved": saved, "total": total, "done": saved == total,
                })
            batches.append({"batch": batch, "shots": shots})
        return {"grid_ms": grid.GRID_MS, "events": silver_events(), "batches": batches}

    @app.get("/api/gold/shots")
    def gold_shots():
        """Just the gold shot numbers, for the event page's gold-only queue."""
        try:
            return {"shots": sorted(int(shot) for shot in gold.read_gold(paths.label_tables).shot)}
        except FileNotFoundError:
            return {"shots": [], "error": "Gold list missing: gold/gold_shots.csv"}
        except ValueError as error:
            return {"shots": [], "error": str(error)}

    @app.get("/api/silver/shot")
    def silver_shot_view(shot: int):
        if not silver.is_silver(paths.label_tables, shot):
            raise HTTPException(404, f"shot {shot} is not in the silver set")
        row = silver.read_silver(paths.label_tables).query("shot == @shot").iloc[0]
        events, t_range = shotlist_routes.shot_events(paths, shot)
        return {
            "shot": shot, "batch": int(row.batch), "order": int(row.order),
            "pool": row.pool, "classes": [p for p in row.classes.split(";") if p],
            "grid_ms": grid.GRID_MS, "t_range": t_range, "events": events,
        }

    @app.get("/api/shot")
    def shot_view(event: str, shot: int):
        directory = require_event(event, paths)
        tier = shot_tier(event, shot, paths)
        # A subcategory opens only once its parent is saved on the shot.
        child = _region(event, shot, paths) if _taxonomy(paths).is_child(event) else None
        store = store_event(event)
        path, error = app.state.builds.status(event, shot)
        if error is not None:
            return JSONResponse(error.body(), status_code=502)
        if path is None:
            body = {"building": True, "progress": raw.progress_for(shot),
                    **app.state.builds.progress(event, shot)}
            return JSONResponse(body, status_code=202)
        described = rows.meta(path, review_build.HIDDEN.get(store, frozenset()))
        band = review_build.BANDS.get(store)
        for row in described["rows"]:
            if band is not None and "band" in row:
                row["band"] = list(band)
        overlay_routes.annotate(paths, event, shot, described["rows"])
        producer = described.get("params", {}).get("detachment_producer", {})
        suppress_source = event == "detachment" and producer.get("source_suppressed", False)
        under_parent = {}
        if child is not None:
            parent, _, parent_label, region = child
            under_parent = {
                "parent": {
                    "event": parent,
                    "display_name": _taxonomy(paths).display(parent),
                    "label": parent_label.as_json(),
                },
                "allowed": region.as_json(),
                "categories": {
                    str(k): v for k, v in labels.categories(event).items()
                },
            }
        curated = rwm.onsets(shot, paths) if event == "resistive_wall_mode" else None
        reviewed, _, _ = shot_labels(
            event, shot, paths, suppress_source=suppress_source,
            region=None if child is None else child[3],
            onsets=(labels.onset_times(event, described.get("markers"))
                    if curated is None else [point["t_ms"] for point in curated]),
            t_range=described.get("t_range"),
        )
        return {
            "event": event,
            "shot": shot,
            "tier": tier,
            "silver": silver.is_silver(paths.label_tables, shot),
            **described,
            **({"video": video.meta(path)} if event == "detachment" else {}),
            **reviewed,
            **under_parent,
            "reviewers": reviewers.shot_reviewers(directory, shot),
            **({"onsets": curated} if curated is not None else {}),
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
        shot_tier(event, shot, paths)
        path = paths.spectrogram_file(event, shot)
        if not path.is_file():
            raise HTTPException(404, f"shot {shot} has no rows yet: open it first")
        try:
            hide = review_build.HIDDEN.get(store_event(event), frozenset())
            data, grid = rows.read_window(path, t0, t1, cols, hide)
        except (ValueError, OverflowError) as error:
            raise HTTPException(400, str(error)) from None
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"X-Grid": json.dumps(grid)},
        )

    @app.get("/api/frame")
    def frame_view(
        event: str, shot: int, camera: str,
        channel: Annotated[int, Query(ge=0, le=255)] = 0,
        t_ms: float = 0.0,
    ):
        require_event(event, paths)
        shot_tier(event, shot, paths)
        if event != "detachment":
            raise HTTPException(404, "this event has no camera previews")
        path = paths.spectrogram_file(event, shot)
        if not path.is_file():
            raise HTTPException(404, f"shot {shot} has no previews yet: open it first")
        try:
            data, frame = video.read_frame(path, camera, channel, t_ms)
        except KeyError:
            raise HTTPException(404, "no frames for this camera/channel") from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        return Response(data, media_type="image/png", headers={
            "X-Frame-Time-Ms": str(frame["time_ms"]),
            "X-Frame-Index": str(frame["index"]),
        })

    @app.post("/api/label")
    def save_label(body: LabelIn):
        directory = require_event(body.event, paths)
        tier = shot_tier(body.event, body.shot, paths)
        known = set(labels.categories(body.event))
        try:
            label = labels.normalise(
                grid.snap_window(*body.window),
                [(*grid.snap_span(a, b), c) for a, b, c in body.intervals],
                known=known, iscrowd=body.iscrowd,
            )
            name = versions.clean_name(body.name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        extra, region = None, None
        if _taxonomy(paths).is_child(body.event):
            # A subcategory is a subset of its parent: whatever lies outside the
            # parent's saved spans (either lane) is dropped, and the window is the
            # parent's. Checked after `normalise`, so a bad span is still a 400.
            parent, parent_dir, parent_label, region = _region(
                body.event, body.shot, paths
            )
            label = region.clip(label)
            extra = labels.parent_stamp(parent, parent_dir, body.shot, parent_label)
        reviewed, table, origin = shot_labels(
            body.event, body.shot, paths, region=region
        )
        source = (labels.Label(
            tuple(reviewed["source"]["window"]),
            tuple(tuple(span) for span in reviewed["source"]["intervals"]),
            tuple(reviewed["source"].get("iscrowd", ())),
        ) if reviewed["source"] else None)
        if source is not None and reviewed["source"].get("attrs"):
            raise HTTPException(409, "The review page cannot edit evidence attributes")
        if not body.overlap_edit and any(labels.has_overlaps(value) for value in (
            label,
            labels.read_saved(directory).get(body.shot),
            source,
        )):
            raise HTTPException(
                409, "Reload the review page before editing overlapping annotations"
            )
        try:
            entry = labels.save(
                directory,
                body.shot,
                label,
                source=origin or (table.name if table else None),
                name=name,
                source_sha256=sha256_of(table) if table else None,
                crowd_edit=body.iscrowd is not None,
                suggestions_shown=body.suggestions_shown,
                prefilled=body.prefilled,
                extra=extra,
            )
        except labels.SaveRefused as error:
            raise HTTPException(409, str(error)) from None
        row = {
            "shot": body.shot,
            "tier": tier,
            # As the queue will show it on reload: a silver fallback is not the
            # event's own source.
            "state": labels.state(label, None if origin else source),
            "saved_at": entry["saved_at"],
        }
        return {
            "row": row,
            "saved": label.as_json(),
            "last_save": entry,
            "reviewers": reviewers.shot_reviewers(directory, body.shot),
        }

    @app.get("/api/history")
    def history_view(event: str, shot: int):
        directory = require_event(event, paths)
        shot_tier(event, shot, paths)
        found = versions.shot_versions(directory, shot)
        if event.split(".")[0] in labels.SPAN_EVENTS:
            # An older save without resolution restores as spans, as the page opens it.
            found = [v if "iscrowd" in v or not v["intervals"]
                     else {**v, "iscrowd": [1] * len(v["intervals"])} for v in found]
        return {"shot": shot, "versions": found}

    def mask_file(event: str, shot: int, *, write: bool = False) -> tuple[Path, Path]:
        """The AE pseudo-mask of `shot`. A subcategory of AE reads its parent's mask and
        decisions, read-only: rejecting a region is a decision on the parent."""
        require_event(event, paths)
        owner = _taxonomy(paths).parent(event) or event
        directory = require_event(owner, paths)
        shot_tier(owner, shot, paths)
        path = whole.review_file(paths, shot)
        if owner != seg.EVENT or not path.is_file():
            raise HTTPException(404, f"shot {int(shot)} has no {event} pseudo-mask")
        if write and owner != event:
            raise HTTPException(404, f"reject regions on {owner}, not on {event}")
        return directory, path

    @app.get("/api/masks")
    def masks_view(event: str, shot: int):
        directory, path = mask_file(event, shot)
        with mask_lock:
            return regions.shot_view(
                paths, directory, shot, path=path, pseudo=whole.REVIEW
            )

    @app.get("/api/tokeye")
    def tokeye_view(event: str, shot: int):
        require_event(event, paths)
        owner = _taxonomy(paths).parent(event) or event
        shot_tier(owner, shot, paths)
        # AE and its children: one layer; any other event: {rows: {name: layer}}.
        found = overlay_routes.tokeye(paths, event, owner, shot)
        if found is None:
            raise HTTPException(404, f"shot {int(shot)} has no {event} TokEye layer")
        return found

    @app.post("/api/masks")
    def save_masks(body: MaskIn):
        directory, path = mask_file(body.event, body.shot, write=True)
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
                pseudo=whole.REVIEW,
            )
        return {
            "rejected": entry["rejected"],
            "last_save": entry,
            "revision": revision + 1,
            "reviewers": reviewers.shot_reviewers(directory, body.shot),
        }

    mask_lock = threading.Lock()
    names_lock = threading.Lock()
    # Each feature module owns its routes; they must precede the static mount.
    shotlist_routes.install(app, paths)
    workspace_routes.install(app, paths)
    overlay_routes.install(app, paths)
    slice_routes.install(app, paths)
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
