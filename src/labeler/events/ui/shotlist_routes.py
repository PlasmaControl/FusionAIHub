"""Review-page routes for named shot lists (plan section A): the list
dropdown, per-list queues and the by-shot overview.

`install` adds them to the app that `app.create_app` builds, before the static mount:

    GET /api/shotlists?event=E       the lists the page offers on E, in its order
    GET /api/shotlist?name=N         one list's rows and sidecar
    GET /api/overview?list=N         a list shot by shot: how many events each has saved
    GET /api/overview?list=N&shot=S  one shot on every event, as the by-shot strips
    GET /api/shots/search?q=&event=  shots whose number starts with q (digits), the
                                     gold, silver and roster ones first

`/api/queue?event&list=` stays in `app.py`; `offer`, `choose`, `as_roster` and
`queue_inputs` below are its list half. Answers that read many files are kept in a
`Stamped` cache until one of their inputs changes. See docs/labeler/shotlists.md.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from ...config import Paths
from .. import grid, hf_export, ondemand, rosters, shotlists, taxonomy
from ..review import gold, labels, silver

log = logging.getLogger(__name__)

#: What the page calls the virtual list of an event's own roster.
ROSTER_TITLE = "Event roster"
#: A search query: one to six ASCII digits, nothing else reaches a lookup.
SEARCH_QUERY = re.compile(r"[0-9]{1,6}\Z")


def versions(files) -> tuple:
    """Each file's path and (mtime, inode, size), None while it is missing: it changes
    whenever one is written, replaced, created or removed (a directory's whenever an
    entry of it is)."""
    found = []
    for path in files:
        try:
            stat = os.stat(path)
        except OSError:
            found.append((str(path), None))
        else:
            found.append((str(path), stat.st_mtime_ns, stat.st_ino, stat.st_size))
    return tuple(found)


class Stamped:
    """Answers kept until one of their input files changes (`versions`), the least
    recently used dropped past `size`. The inputs are stamped before an answer is
    computed, so a write during it costs one more computation, never a stale answer.
    Callers must not mutate what `get` returns."""

    def __init__(self, size: int = 64):
        self.size = size
        self.lock = threading.Lock()
        self.found: OrderedDict = OrderedDict()

    def get(self, key, files, compute):
        stamp = versions(files)
        with self.lock:
            hit = self.found.get(key)
            if hit is not None and hit[0] == stamp:
                self.found.move_to_end(key)
                return hit[1]
        value = compute()
        with self.lock:
            self.found[key] = (stamp, value)
            self.found.move_to_end(key)
            while len(self.found) > self.size:
                self.found.popitem(last=False)
        return value


def offer(paths: Paths, event: str, saved, roster: pd.DataFrame) -> list[dict]:
    """The `/api/shotlists?event` body: every list the page offers on `event`, in
    `shotlists.offered` order, with its size and how many of its shots `event` has
    saved, then the virtual roster entry. `saved` holds the shots saved on `event`;
    `roster` is its roster (a subcategory's parent's). `default` marks the list the
    event opens on: its registered default, or the roster when that list cannot be
    read. A registry that cannot be read offers the roster alone (logged)."""
    saved = {int(shot) for shot in saved}
    try:
        default = shotlists.default(paths, event)
        listed = shotlists.offered(paths, event)
    except shotlists.ShotListError as error:
        log.warning("shot lists not offered: %s", error)
        default, listed = shotlists.ROSTER, []
    sizes = shotlists.members(paths)
    found = [{
        "name": item.name, "title": item.title, "n": len(sizes[item.name]),
        "kind": item.kind if item.registered else "other",
        "recommended": recommended, "default": item.name == default,
        "parent": item.parent, "protect": item.protect,
        "n_saved": len(sizes[item.name] & saved),
    } for item, recommended in listed]
    on_roster = {int(shot) for shot in roster.shot}
    found.append({
        "name": shotlists.ROSTER, "title": ROSTER_TITLE, "n": len(on_roster),
        "kind": "roster", "recommended": False,
        "default": not any(item["default"] for item in found),
        "parent": None, "protect": None, "n_saved": len(on_roster & saved),
    })
    return found


def choose(paths: Paths, event: str, requested: str | None,
           offered: list[dict]) -> tuple[str, str | None]:
    """`(name, note)`: the list `/api/queue` shows, out of `offer`'s entries, and why
    it is not the one asked for. No request shows the default; an old name that
    `renamed` has moved shows its list, with a note; one the page does not offer falls
    back to the default, and so does a default that cannot be read."""
    fallback = next(item["name"] for item in offered if item["default"])
    if requested is None:
        try:
            registered = shotlists.default(paths, event)
        except shotlists.ShotListError:
            registered = fallback
        if registered != fallback:
            return fallback, f"List {registered} cannot be read; showing {fallback}."
        return fallback, None
    if any(item["name"] == requested for item in offered):
        return requested, None
    current = shotlists.resolve(paths, requested, event)
    if current != requested and any(item["name"] == current for item in offered):
        return current, f"List {requested[:64]} is now {current}."
    try:
        known = current in shotlists.names(paths)
        hidden = known and shotlists.entry(paths, current).hidden
    except (shotlists.ShotListError, KeyError):
        known = hidden = False
    reason = ("is not offered any more" if hidden
              else "cannot be read" if known else "no longer exists")
    return fallback, f"List {requested[:64]} {reason}; showing {fallback}."


def as_roster(roster: pd.DataFrame, shots) -> pd.DataFrame:
    """`shots` in a roster's shape for `labels.queue`: each shot with its roster tier,
    or unverified off the roster (`shot_tier` opens those on any listed shot)."""
    tiers = {int(shot): tier
             for shot, tier in zip(roster.shot, roster.tier, strict=True)}
    shots = [int(shot) for shot in shots]
    return pd.DataFrame({"shot": shots,
                         "tier": [tiers.get(shot, "unverified") for shot in shots]})


def queue_inputs(paths: Paths, directory: Path, parent_dir: Path | None) -> list[Path]:
    """The files `/api/queue` answers from: the event's roster, saves, history and
    source (and its parent's, for a subcategory), the gold list, the taxonomy and
    every shot list."""
    found = [Path(paths.events_manifest), Path(paths.label_tables) / gold.GOLD_FILE]
    for where in (directory, parent_dir):
        if where is not None:
            found += [where / rosters.ROSTER_NAME, labels.labels_path(where),
                      labels.history_path(where), labels.pointer_path(where),
                      where / "format"]
    try:
        source = labels.source_path(directory)
    except (OSError, ValueError):  # the queue itself reports it
        source = None
    return found + ([source] if source is not None else []) + shotlists.inputs(paths)


def overview_inputs(paths: Paths) -> list[Path]:
    """The files a list's overview answers from: the taxonomy, the event directories,
    every shot list and every review event's saves (a child's lock is its parent's)."""
    found = [Path(paths.events_manifest), Path(paths.label_tables),
             *shotlists.inputs(paths)]
    try:  # a roster added to a directory makes it a review event
        found += sorted(path / rosters.ROSTER_NAME
                        for path in Path(paths.label_tables).iterdir() if path.is_dir())
    except OSError:
        pass
    for event in silver.review_events(paths):
        found.append(labels.labels_path(silver.event_dir(paths, event)))
    return found


def _cell(value):
    """A table cell as JSON takes it: a numpy scalar as Python's, a missing value as
    null."""
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    return None if missing is True else value


def records(frame: pd.DataFrame) -> list[dict]:
    """A list's rows, `shot` first, every other column as the file has it."""
    columns = list(frame.columns)
    return [{column: _cell(value) for column, value in zip(columns, row, strict=True)}
            for row in frame.itertuples(index=False, name=None)]


def columns(row: dict) -> dict:
    """A list row as the overview serves it: a `classes` cell ("a:present;b:absent",
    as silver's file has it) as a list, as `/api/silver` gives it."""
    if "classes" not in row:
        return row
    text = row["classes"]
    return {**row, "classes": [] if text is None else [
        part for part in str(text).split(";") if part]}


def withheld(paths: Paths, layer: silver.Layer) -> bool:
    """Whether `layer` is read from a table the release withholds
    (`hf_export.WITHHELD`: the ELM challenge's answer key). Such a layer never feeds a
    draft and is never drawn in the by-shot view."""
    try:
        name = silver.layer_file(paths, layer).relative_to(paths.label_tables)
    except ValueError:
        return False
    return name.with_suffix("").as_posix() in hf_export.WITHHELD


def draft(paths: Paths, event: str, shot: int):
    """`silver.draft` for a shot `event` reviews though it is not its own, or None:
    never on a shot of a `protect: holdout` list, and never when a withheld table
    holds the shot for this event (its layer would be the draft, or fill it)."""
    if shotlists.held_out(paths, shot):
        return None
    if any(withheld(paths, layer) for layer in silver.layers(paths, event, shot)):
        return None
    return silver.draft(paths, event, shot)


def review_events(paths: Paths) -> list[dict]:
    """The page's review events, each parent followed by its subcategories, each with
    its `default` class: what time a saved label leaves unmarked reads as (0 absent)."""
    tree = taxonomy.load(paths.events_manifest)
    return [{"event": event, "display_name": tree.display(event),
             "parent": tree.parent(event), "default": tree.default(event)}
            for event in silver.review_events(paths)]


def shot_events(paths: Paths, shot: int) -> tuple[list[dict], list[int]]:
    """`(events, t_range)`: every review event of `shot` as the by-shot view draws
    it (its layers in precedence order, saved or not, locked behind its parent, the
    parent's region) and the time all the layers span. A withheld table's layer is
    left out. Each event carries its `default` class and each layer its `fill`, what
    the time its label's window leaves unmarked reads as (0 absent)."""
    events, groups = [], []
    for entry in review_events(paths):
        event = entry["event"]
        layers = [layer for layer in silver.layers(paths, event, shot)
                  if not withheld(paths, layer)]
        groups.append(layers)
        region = silver.parent_region(paths, event, shot)
        events.append({
            **entry, "saved": any(layer.kind == "saved" for layer in layers),
            "locked": silver.locked(paths, event, shot),
            "categories": {"0": "absent", **{
                str(k): v for k, v in labels.categories(event).items()
            }},
            **({"allowed": region.as_json()} if region is not None else {}),
            "layers": [{"kind": layer.kind, "name": layer.name,
                        "label": layer.label.as_json(), "fill": layer.fill,
                        "unknown": [list(span) for span in layer.unknown],
                        "held": [list(span) for span in layer.held]}
                       for layer in layers],
        })
    return events, list(silver.time_range(groups))


def overview(paths: Paths, item: shotlists.Entry) -> dict:
    """`/api/overview?list`: the list's shots in its order, each with its columns and
    how many of the events it is open on it has saved (a child locked behind an
    unsaved parent is not counted), and the lists drawn from it."""
    rows = records(shotlists.read(paths, item.name))
    progress = silver.progress_many(paths, [row["shot"] for row in rows])
    shots = []
    for row in rows:
        found = progress[row["shot"]]
        saved, total = len(found["saved"]), found["total"]
        shots.append({**columns(row), "saved": saved, "total": total,
                      "done": saved == total})
    sizes = shotlists.members(paths)
    children = [{"name": child.name, "title": child.title, "n": len(sizes[child.name])}
                for child in shotlists.entries(paths)
                if child.parent == item.name and child.name in sizes
                and not child.hidden]
    return {"list": item.name, "title": item.title, "grid_ms": grid.GRID_MS,
            "events": review_events(paths), "shots": shots, "children": children}


def search(paths: Paths, query: str, *, event: str | None = None, roster=(),
           limit: int = 20) -> dict:
    """The `/api/shots/search` body: up to `limit` shots whose number starts with
    `query` (digits), as `{"shot", "lists", "cached"}`, the gold, silver and `event`'s
    roster shots first, then by number. A shot is any on a list or the roster, the
    blind test shots excepted; `lists` names the registered lists holding it (and
    `roster` for the event's own), `cached` says whether `event`'s review store for
    it is built and current (any store, with no event). A full six-digit number no
    list holds still comes back, `lists` empty and `cached` false: the page may try to
    open it, and it opens if fdp has data (`ondemand.openable`)."""
    sets = shotlists.members(paths)
    on_roster = {int(shot) for shot in roster}
    blind = shotlists.blind_set(paths)
    pool = set().union(*sets.values(), on_roster) - blind
    first = on_roster | sets.get("gold", frozenset()) | sets.get("silver", frozenset())
    found = sorted((shot for shot in pool if str(shot).startswith(query)),
                   key=lambda shot: (shot not in first, shot))
    typed = int(query) if len(query) == 6 else None
    try:
        hidden = {item.name for item in shotlists.entries(paths) if item.hidden}
    except (shotlists.ShotListError, OSError):
        hidden = set()
    found = found[:limit]
    if typed is not None and typed in found:  # the exact number first
        found.remove(typed)
        found.insert(0, typed)
    shots = [{
        "shot": shot,
        "lists": [*(["roster"] if shot in on_roster else []),
                  *(name for name in shotlists.holding(paths, shot) if name not in hidden)],
        "cached": ondemand.has_store(paths, shot, event),
    } for shot in found]
    if typed is not None and typed not in pool and ondemand.openable(paths, typed):
        shots.insert(0, {"shot": typed, "lists": [], "cached": False})
        del shots[limit:]
    return {"shots": shots}


def install(app: FastAPI, paths: Paths) -> None:
    """Add this module's routes to `app`."""
    from .app import require_event, roster_directory  # app imports this module

    overviews = Stamped(16)

    def listed(name: str) -> shotlists.Entry:
        """List `name`, else 404; the roster is no list of its own here."""
        if name == shotlists.ROSTER:
            raise HTTPException(
                404, "the roster is each event's own: /api/queue?event= serves it"
            )
        try:
            item = shotlists.entry(paths, name)
        except KeyError:
            raise HTTPException(404, f"no shot list {name[:64]!r}") from None
        if not item.path.is_file():
            raise HTTPException(404, f"shot list {name} has no file yet")
        return item

    @app.get("/api/shotlists")
    def shotlists_view(event: str):
        directory = require_event(event, paths)
        roster = rosters.read_roster(
            roster_directory(event, paths) / rosters.ROSTER_NAME
        )
        return offer(paths, event, labels.read_saved(directory), roster)

    @app.get("/api/shotlist")
    def shotlist_view(name: str):
        item = listed(name)
        return {
            "name": item.name, "title": item.title,
            "kind": item.kind if item.registered else "other",
            "rows": records(shotlists.read(paths, name)),
            "meta": shotlists.sidecar(item.path),
        }

    @app.get("/api/overview")
    def overview_view(
        name: Annotated[str, Query(alias="list")], shot: int | None = None
    ):
        item = listed(name)
        if shot is None:
            return overviews.get(name, overview_inputs(paths),
                                 lambda: overview(paths, item))
        if not shotlists.in_any(paths, shot):
            raise HTTPException(404, f"shot {shot} is on no shot list")
        frame = shotlists.read(paths, name)
        row = columns(next(iter(records(frame[frame.shot == shot])), {}))
        events, t_range = shot_events(paths, shot)
        return {**{k: v for k, v in row.items() if k != "shot"},
                "list": item.name, "shot": shot, "grid_ms": grid.GRID_MS,
                "t_range": t_range, "events": events}

    @app.get("/api/shots/search")
    def shots_search(q: str, event: str = "",
                     limit: Annotated[int, Query(ge=1, le=100)] = 20):
        if not SEARCH_QUERY.match(q):
            raise HTTPException(400, "q must be one to six digits")
        roster = ()
        if event:
            require_event(event, paths)
            roster = rosters.read_roster(
                roster_directory(event, paths) / rosters.ROSTER_NAME
            ).shot
        return search(paths, q, event=event or None, roster=roster, limit=limit)
