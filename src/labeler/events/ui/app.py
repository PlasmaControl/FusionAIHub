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
import logging
import math
import os
import secrets
import warnings
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...config import Paths
from .. import panels as registry
from .. import rosters
from ..raw import UpstreamError, WindowEmptyError
from ..verify import (
    NoDataError,
    correction_path,
    corrections_for,
    label_panel,
    write_corrections,
)

STATIC = Path(__file__).parent / "static"
COOKIE = "labeler_verify_token"
NO_TOKEN = "no token: reopen the link printed by the verify server"
BAD_TOKEN = "bad token"

#: Figures carry log10 magnitudes and physical coordinates; neither is known
#: to seventeen significant digits, and full-precision float64 text is most
#: of the bytes in a heatmap response. `clean(row, DECIMALS)` applies this to
#: every heatmap's `z` regardless of builder or units - safe today for all
#: four (AE log-magnitude, `minimum_safety_factor`'s q ~ 0.8-8, `fishbone`'s
#: +-pi cross-phase, and the label grid's integer codes), but any future
#: heatmap whose `z` has a dynamic range below ~0.01 must not use this
#: default.
DECIMALS = 3

#: The largest `|t|` a mark may carry, in milliseconds. A DIII-D shot runs
#: for a few seconds, so 10,000 s is already absurd; the point is not to
#: police the edges of a plausible window but to stop a value that cannot be
#: a dragged range - a 1e30 out of a broken axis transform, say - from being
#: written into a label file that later feeds training.
MAX_ABS_TIME_MS = 1e7

#: Marks per save. A review is a handful of intervals; four figures of them
#: is a loop in the page, and this endpoint writes what it is given.
MAX_MARKS = 1000

#: Longest a posted `reviewer` id may be. `correction_path` sanitises the id
#: but not its length, and `<shot>__<reviewer>__<stamp>.csv` already has
#: ~30 characters of its own overhead against a filesystem's ~255-character
#: name limit; staying well under that margin buys a 400 here instead of an
#: `ENAMETOOLONG` off `path.exists()`.
MAX_REVIEWER_LEN = 200

log = logging.getLogger(__name__)


def _json(value, **kwargs) -> Response:
    # `allow_nan=False` on purpose: the default emits a bare `NaN` token,
    # which is not JSON and makes `JSON.parse` throw on the WHOLE response,
    # so one unknown cell loses the entire figure. Raising here instead
    # turns that into a server error a log records, and keeps every new
    # float-carrying field honest about going through `_finite` first.
    return Response(
        json.dumps(value, default=str, allow_nan=False),
        media_type="application/json",
        **kwargs,
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
    `/api/save` is such an endpoint now, not later: the same 404 stands
    between a client-supplied `event` and an arbitrary-directory WRITE,
    not only a foreign read.

    The check is a name test plus one `is_file`, not a scan of `_events`:
    this runs on every pan and every zoom, and reading and parsing all
    sixteen rosters to test one string for membership put that whole cost on
    each interactive frame. It is also STRICTER than membership was. The
    accepted set is unchanged - a direct child ENTRY of `label_tables`
    holding a roster - but it is decided on the name before any path is
    joined, so `..`, a separator and an absolute value are refused by
    inspection rather than by failing to match a listing. Nothing is cached,
    so an event added while the server runs shows up on the next request.

    "Direct child entry" is deliberate: this does not resolve symlinks. A
    directory entry that is itself a symlink to somewhere outside
    `label_tables`, or a real child whose `shots.csv` is a symlink to a
    foreign roster, both pass and serve their target's content. That is
    unchanged from the membership test this replaced, and `label_tables` is
    a hand-maintained tree, so a symlinked entry is trusted rather than
    resolved and rechecked.
    """
    # `Path(event).name` is `event` itself only for a bare filename: it is
    # "b" for "a/b", "secret_area" for "/tmp/secret_area", and "" for "..",
    # ".", "" and any trailing-slash form. A NUL cannot be in a path at all.
    if event != Path(event).name or event in {"", ".", ".."} or "\0" in event:
        raise HTTPException(status_code=404, detail=f"unknown event {event!r}")
    try:
        found = (paths.label_tables / event / rosters.ROSTER_NAME).is_file()
    except OSError:
        # `is_file` re-raises rather than reporting False for some failures
        # pathlib treats as ignorable ones - ENAMETOOLONG among them - so an
        # over-long `event` must be caught here, not left to become an
        # unhandled 500 that breaks the module's one-error-shape contract.
        found = False
    if not found:
        raise HTTPException(status_code=404, detail=f"unknown event {event!r}")
    return event


class _Malformed(ValueError):
    """A posted field of the wrong shape: a 422, before any value is judged."""


def _number(value, where: str) -> float:
    """One posted number, refusing the things `float()` would accept.

    `float("early")` raises, but `float("2100")` and `float(True)` do not,
    and neither is a time a figure produced. The wire is JSON: a time is a
    JSON number or it is malformed.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Malformed(f"{where} must be a number, not {type(value).__name__}")
    return float(value)


def _whole(value, where: str) -> int:
    """One posted integer. `int(1.5)` truncates, which is a silent wrong label."""
    number = _number(value, where)
    if not number.is_integer():
        raise _Malformed(f"{where} must be a whole number, not {value!r}")
    return int(number)


def _finite(value):
    """One scalar as JSON: `null` where it is None or not finite."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _panel_json(panel) -> dict:
    """One `Panel` as the plain arrays plotly.js wants.

    NaN is not JSON, and a label grid is full of it where a cell is unknown.
    `json.dumps` would emit a bare `NaN` token that `JSON.parse` rejects, so
    every non-finite value becomes `null` - which plotly draws as a gap,
    which is what an unknown cell is.
    """

    def clean(array, decimals=None):
        values = np.asarray(array, dtype="float64")
        if decimals is not None:
            values = np.round(values, decimals)
        if values.size and not np.isfinite(values).all():
            # Only this branch builds an object array, i.e. a boxed Python
            # float per element. A heatmap is ~1.8 M elements and normally
            # all finite, so the fast path below is worth the extra pass.
            return np.where(np.isfinite(values), values, None).tolist()
        return values.tolist()

    payload = {
        "title": panel.title,
        "kind": panel.kind,
        "ylabel": panel.ylabel,
        "x": clean(panel.x),
        # Through `_finite` like everything else: these are small, but a NaN
        # in any one of them is the same invalid-JSON response as a NaN in
        # `z`, and being small is not a reason to find that out in the field.
        "bands": [[_finite(low), _finite(high)] for low, high in panel.bands],
        "hlines": [_finite(level) for level in panel.hlines],
        "zmin": _finite(panel.zmin),
        "zmax": _finite(panel.zmax),
    }
    if panel.kind == "heatmap":
        payload["y"] = clean(panel.y)
        payload["z"] = [clean(row, DECIMALS) for row in panel.z]
    else:
        payload["y"] = [clean(row) for row in panel.y]
        payload["legend"] = list(
            panel.legend or [f"ch {i}" for i in range(len(panel.y))]
        )
    return payload


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

    @app.exception_handler(RequestValidationError)
    async def as_bad_query(request: Request, exc: RequestValidationError) -> Response:
        """A `t0=early` is a refusal too, and reads like every other one.

        FastAPI's own handler answers with `{"detail": [...]}`, so the page
        would have to carry a second error shape for this one case.
        """
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"][1:]) or "query"
        return JSONResponse({"error": f"{where}: {first['msg']}"}, status_code=422)

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

    @app.get("/api/panels")
    def panels_for(
        event: str, shot: int, t0: float | None = None, t1: float | None = None
    ):
        paths = app.state.paths
        # First, before anything is built from `event`: it is a path segment
        # for `label_panel` below, so an unchecked one reaches any directory
        # the server's uid can read.
        event = require_event(event, paths)
        # Milliseconds on the wire, milliseconds into the builders. The
        # corpus stores seconds and `raw_signal` converts; nothing here does.
        t_range = None if t0 is None or t1 is None else (float(t0), float(t1))
        if t_range is not None and t_range[1] <= t_range[0]:
            # Refused before a byte is read. A window that cannot contain a
            # sample has no answer worth going to disk - let alone to
            # PTDATA - for, and the builders only find that out after they
            # have already paid for it.
            return _json(
                {"error": f"window {t_range[0]}-{t_range[1]} ms ends before it starts"},
                status_code=400,
            )
        try:
            built = registry.build(event, shot, t_range=t_range, paths=paths)
        except WindowEmptyError as error:
            # The record IS on disk and this window is off the end of it -
            # a scroll wheel, not a fault. `raw_signal` raises this INSTEAD
            # of falling through to the live fetch tier, which is what keeps
            # a pan past the end of a shot from costing ~240 MB and minutes.
            return _json({"error": str(error)}, status_code=400)
        except UpstreamError as error:
            # fdp itself could not answer. The only bad-gateway case here.
            return _json({"error": str(error)}, status_code=502)
        except NoDataError as error:
            # This shot is simply not here - no corpus file, an absent-signal
            # sentinel, no fetch route. An ordinary outcome, and a 404 says
            # which of the two it is without the reviewer reading the prose.
            return _json({"error": str(error)}, status_code=404)
        except OSError:
            # Unreadable bytes: a truncated HDF5 file, a permission, a dead
            # mount. The raw message carries absolute server paths and errno
            # noise, which belongs in the log, not at the top of a reviewer's
            # page.
            log.exception("reading %s panels for shot %s", event, shot)
            return _json(
                {
                    "error": (
                        f"could not read the data for shot {int(shot)}; "
                        f"the server log has the details"
                    )
                },
                status_code=502,
            )
        except ValueError as error:
            if t_range is None:
                # With no window there is no window to blame, so this is a
                # builder bug. Reporting it as a 400 told the reviewer their
                # input was bad and hid the regression - surfacing it as a
                # 500 is right and stays. Only the shape was wrong: a bare
                # `raise` gave Starlette's plain-text response instead of
                # this module's one {"error": ...} shape, which is what
                # every other refusal reads and what the page's JavaScript
                # depends on. The exception text itself is logged, not put
                # in the response, the same as the OSError branch above.
                log.exception("building %s panels for shot %s", event, shot)
                return _json(
                    {
                        "error": (
                            f"internal error building {event} panels for "
                            f"shot {int(shot)}; see the server log"
                        )
                    },
                    status_code=500,
                )
            return _json(
                {"error": f"window {t_range[0]}-{t_range[1]} ms: {error}"},
                status_code=400,
            )

        # The label row is appended the way `verify.review` appends it, and
        # its absence is said on the page rather than only in a warning on
        # stderr, which is not where the reviewer is looking.
        note = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                built.append(
                    label_panel(
                        event, shot, source="format/shots", root=paths.label_tables
                    )
                )
        except FileNotFoundError:
            note = "NO LABEL ROW"
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            # `data/events` is a hand-maintained tree, so a half-written or
            # truncated `.npz` (BadZipFile, or ValueError out of np.load) and
            # a grid saved without `rho_edges` (KeyError) are both realistic.
            # Neither is a reason to lose the diagnostic panels the reviewer
            # came for, and neither reads as "there is no label row" - the
            # file is there and it is broken, which is a different errand.
            log.exception("reading the %s label grid for shot %s", event, shot)
            note = "LABEL ROW UNREADABLE"

        return _json(
            {
                "event": event,
                "shot": int(shot),
                "t_range": None if t_range is None else list(t_range),
                "note": note,
                "panels": [_panel_json(p) for p in built],
            }
        )

    @app.post("/api/save")
    async def save(request: Request):
        """Persist one reviewer's corrected intervals, and nothing else.

        The ONLY endpoint in this app that writes. Two things follow from
        that. First, `event` is checked by `require_event` before a path is
        built from it: on the read endpoints an unchecked `event` served a
        foreign file, but here it would be an arbitrary-directory write.
        Second, this goes straight to `write_corrections` and deliberately
        NOT through `ReviewSession.save()`, which also calls `record_review`
        and so edits `shots.csv` - the one file this surface promises never
        to touch.
        """
        import pandas as pd

        from ..interval_tables import INTERVAL_COLUMNS

        try:
            body = await request.json()
        except (ValueError, RecursionError):
            # A deeply nested body (tens of thousands of `[`) makes the C
            # parser blow the recursion limit instead of raising ValueError,
            # and an unhandled RecursionError here would fall through to
            # Starlette's bare-text 500 - a different shape than every other
            # refusal on this endpoint reads.
            return _json({"error": "body is not valid JSON"}, status_code=422)
        if not isinstance(body, dict):
            return _json({"error": "body must be a JSON object"}, status_code=422)
        if not isinstance(body.get("event"), str):
            return _json({"error": "event must be a string"}, status_code=422)
        # First, before a path exists to write to.
        event = require_event(body["event"], app.state.paths)
        try:
            shot = _whole(body.get("shot"), "shot")
            marks = body.get("marks")
            if not isinstance(marks, list):
                raise _Malformed("marks must be a list of intervals")
        except _Malformed as error:
            return _json({"error": str(error)}, status_code=422)

        # Ahead of the parse loop, not after it: the cap exists so a
        # ridiculous post cannot buy itself the cost of being parsed, and
        # checking it only once `rows` was already built defeats that.
        if len(marks) > MAX_MARKS:
            return _json(
                {
                    "error": (
                        f"{len(marks)} marks in one save is more than the "
                        f"{MAX_MARKS} a review can hold"
                    )
                },
                status_code=400,
            )

        try:
            rows = []
            for index, mark in enumerate(marks):
                if not isinstance(mark, dict):
                    raise _Malformed(f"marks[{index}] must be an object")
                for field in ("t_start", "t_end", "category"):
                    if field not in mark:
                        raise _Malformed(f"marks[{index}] has no {field}")
                # Milliseconds, like `t_range` and every other time on this
                # wire. The corpus stores seconds; nothing here converts.
                t_start = _number(mark["t_start"], f"marks[{index}].t_start")
                t_end = _number(mark["t_end"], f"marks[{index}].t_end")
                category = _whole(mark["category"], f"marks[{index}].category")
                rows.append([shot, category, t_start, t_end, ""])
        except _Malformed as error:
            return _json({"error": str(error)}, status_code=422)

        if not rows:
            return _json(
                {"error": "no marks to save; drag a range and mark it first"},
                status_code=400,
            )
        for row in rows:
            t_start, t_end = row[2], row[3]
            if not (math.isfinite(t_start) and math.isfinite(t_end)):
                # json.loads accepts the bare `NaN` and `Infinity` tokens.
                return _json(
                    {"error": "mark times must be finite milliseconds"},
                    status_code=400,
                )
            if max(abs(t_start), abs(t_end)) > MAX_ABS_TIME_MS:
                return _json(
                    {
                        "error": (
                            f"{t_start}-{t_end} is not a window on a shot; "
                            f"times are milliseconds"
                        )
                    },
                    status_code=400,
                )
            if t_end <= t_start:
                # `validate_intervals` would take t_end == t_start, but a
                # drag that never moved marks nothing, and `/api/panels`
                # refuses the same window rather than rendering it.
                return _json(
                    {"error": f"t_end {t_end} does not follow t_start {t_start}"},
                    status_code=400,
                )

        reviewer = body.get("reviewer")
        if reviewer is not None and not isinstance(reviewer, str):
            # Every other field on this endpoint is type-checked before it is
            # used; `reviewer` was the one exception, and an int or a list
            # sails through `str()` into a filename like `178642__178642__...`.
            return _json(
                {"error": f"reviewer must be a string, not {type(reviewer).__name__}"},
                status_code=422,
            )
        reviewer = reviewer or os.environ.get("USER", "unknown")
        if len(reviewer) > MAX_REVIEWER_LEN:
            # Past this length, sanitisation still leaves it long enough for
            # `path.exists()` to raise `ENAMETOOLONG` - a client-caused
            # refusal that must not be dressed up as the `OSError` -> 500
            # branch below, the same reasoning commit 70b561f applied to
            # `event`.
            return _json(
                {
                    "error": (
                        f"reviewer id is {len(reviewer)} characters; the "
                        f"limit is {MAX_REVIEWER_LEN}"
                    )
                },
                status_code=400,
            )
        paths = app.state.paths
        try:
            # `correction_path` mints a name that is free and
            # `write_corrections` refuses one that is not, so two reviewers
            # cannot collide and a second save cannot erase a first.
            target = correction_path(
                event, shot, reviewer=str(reviewer), root=paths.label_tables
            )
            frame = pd.DataFrame(rows, columns=list(INTERVAL_COLUMNS))
            write_corrections(frame, target)
        except ValueError as error:
            # `DatabaseError` out of `validate_intervals` and the unusable
            # reviewer id out of `correction_path` are both about what was
            # posted, and both name the field, so the text is safe to show.
            return _json({"error": str(error)}, status_code=400)
        except OSError:
            # A full disk, a read-only mount, an `event` whose name survived
            # the guard but not the filesystem. Nothing was written, and the
            # errno and absolute path belong in the log, not on the page.
            log.exception("writing %s corrections for shot %s", event, shot)
            return _json(
                {
                    "error": (
                        f"could not write the corrections for shot {shot}; "
                        f"the server log has the details"
                    )
                },
                status_code=500,
            )
        return _json({"written": target.name, "n": len(rows)})

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
