"""Saved views of the review page: a reviewer's named layouts per event (plan sections
B and D.3), and the read-only presets of configs/labeler/views.yaml.

The store is `$LABELER_ROOT/review_views/views.json`,
`{reviewer: {event: {name: {layout, etag, saved_at}}}}`, rewritten whole (a new file
renamed into place) under a thread lock and an `flock` on `views.json.lock`, so two
tabs or two servers never interleave a write. The reviewer's name follows the page's
rule (`versions.clean_name`); so does a view's, which may not be "Default" (the
page's own layout).

Every change names the etag the page last saw: saving over a view, renaming it or
deleting it with another etag (or none) is a conflict (409), so a second tab never
overwrites the first one's view unseen. A layout is at most `LAYOUT_MAX` bytes of
JSON.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import yaml

from ...config import Paths
from . import versions

log = logging.getLogger(__name__)

VIEWS_DIR = "review_views"
VIEWS_FILE = "views.json"
PRESETS_FILE = "views.yaml"
LAYOUT_MAX = 64 * 1024
#: The page's own layout of every event; no saved view takes its name.
DEFAULT_NAME = "Default"
OPS = ("save", "rename", "delete")
CONFLICT = ("This view changed in another tab. Reload it, or save yours under a new "
            "name.")

_LOCK = threading.Lock()


class ViewError(Exception):
    """A refused change: `status` is its HTTP code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def store_path(paths: Paths) -> Path:
    return paths.root / VIEWS_DIR / VIEWS_FILE


def presets_path(paths: Paths) -> Path:
    return paths.config_dir / PRESETS_FILE


def reviewer_name(name) -> str | None:
    if name is not None and not isinstance(name, str):
        raise ViewError(400, "reviewer is text")
    try:
        return versions.clean_name(name)
    except ValueError as error:
        raise ViewError(400, str(error)) from None


def view_name(name, field: str = "name") -> str:
    if name is not None and not isinstance(name, str):
        raise ViewError(400, f"{field} is text")
    try:
        cleaned = versions.clean_name(name)
    except ValueError as error:
        raise ViewError(400, f"a view's {error}") from None
    if cleaned is None:
        raise ViewError(400, "name the view")
    if cleaned.casefold() == DEFAULT_NAME.casefold():
        raise ViewError(400, f"\"{DEFAULT_NAME}\" is the page's own view: pick "
                             "another name")
    return cleaned


def etag_of(layout: dict, saved_at: str) -> str:
    text = json.dumps(layout, sort_keys=True, ensure_ascii=False) + saved_at
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _layout_bytes(layout) -> int:
    return len(json.dumps(layout, separators=(",", ":"), ensure_ascii=False).encode())


# ---- presets ----


def presets(paths: Paths, event: str) -> list[dict]:
    """The event's presets, `{name, layout, etag, preset: true}` each."""
    path = presets_path(paths)
    try:
        st = path.stat()
    except OSError:
        return []
    return [dict(view) for view in _presets(str(path), st.st_mtime_ns, st.st_size)
            .get(event, ())]


@lru_cache(maxsize=4)
def _presets(path: str, _mtime_ns: int, _size: int) -> dict:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        log.warning("%s is unreadable: %s", path, error)
        return {}
    found = raw.get("presets") if isinstance(raw, dict) else None
    out: dict = {}
    for event, items in (found or {}).items():
        for item in items if isinstance(items, list) else ():
            if not (isinstance(item, dict) and isinstance(item.get("name"), str)
                    and isinstance(item.get("layout"), dict)):
                log.warning("%s: a preset of %s is malformed; skipped", path, event)
                continue
            layout = item["layout"]
            out.setdefault(str(event), []).append({
                "name": item["name"], "layout": layout,
                "etag": etag_of(layout, "preset"), "preset": True,
            })
    return {event: tuple(views) for event, views in out.items()}


def _is_preset(paths: Paths, event: str, name: str) -> bool:
    return any(v["name"].casefold() == name.casefold() for v in presets(paths, event))


# ---- the store ----


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ViewError(500, f"{path.name} is unreadable ({error}); it was left as "
                             "it is") from None
    if not isinstance(data, dict):
        raise ViewError(500, f"{path.name} is not a mapping; it was left as it is")
    return data


def _write(path: Path, data: dict) -> None:
    scratch = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        scratch.write_text(json.dumps(data, ensure_ascii=False, indent=1,
                                      sort_keys=True), encoding="utf-8")
        os.replace(scratch, path)
    finally:
        if scratch.exists():
            scratch.unlink()


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(path.with_name(path.name + ".lock"), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def list_views(paths: Paths, event: str, reviewer=None) -> list[dict]:
    """The reviewer's views of the event by name, then the event's presets."""
    reviewer = reviewer_name(reviewer)
    own = {}
    if reviewer is not None:
        found = _read(store_path(paths)).get(reviewer, {})
        own = found.get(event, {}) if isinstance(found, dict) else {}
        own = own if isinstance(own, dict) else {}
    out = [{"name": name, "layout": view["layout"], "etag": view["etag"]}
           for name, view in sorted(own.items(), key=lambda item: item[0].casefold())
           if isinstance(view, dict) and "layout" in view]
    return out + presets(paths, event)


def change(paths: Paths, body) -> list[dict]:
    """Apply one `POST /api/views` body; the reviewer's views after it."""
    if not isinstance(body, dict):
        raise ViewError(400, "send a JSON object")
    op = body.get("op")
    if op not in OPS:
        raise ViewError(400, f"op is one of {', '.join(OPS)}")
    event = body.get("event")
    if not isinstance(event, str) or not event:
        raise ViewError(400, "name the event")
    reviewer = reviewer_name(body.get("reviewer"))
    if reviewer is None:
        raise ViewError(400, "Pick your name to save views")
    name = view_name(body.get("name"))
    etag = body.get("etag")
    path = store_path(paths)
    with _locked(path):
        data = _read(path)
        mine = data.setdefault(reviewer, {})
        views = mine.setdefault(event, {}) if isinstance(mine, dict) else None
        if not isinstance(views, dict):
            raise ViewError(500, f"{path.name} is malformed for {reviewer}; it was "
                                 "left as it is")
        existing = views.get(name)
        if op == "save":
            layout = body.get("layout")
            if not isinstance(layout, dict):
                raise ViewError(400, "layout is a JSON object")
            if _layout_bytes(layout) > LAYOUT_MAX:
                raise ViewError(400, f"a view's layout is at most {LAYOUT_MAX // 1024} "
                                     "KB")
            if existing is None and _is_preset(paths, event, name):
                raise ViewError(400, f"\"{name}\" is a preset view: save yours under "
                                     "another name")
            if (existing is None and etag) or (
                existing is not None and etag != existing.get("etag")
            ):
                raise ViewError(409, CONFLICT)
            saved_at = datetime.now(UTC).isoformat(timespec="milliseconds")
            views[name] = {"layout": layout, "etag": etag_of(layout, saved_at),
                           "saved_at": saved_at}
        else:
            if existing is None:
                raise ViewError(404, f"no view named \"{name}\"")
            if etag != existing.get("etag"):
                raise ViewError(409, CONFLICT)
            if op == "rename":
                new = view_name(body.get("new_name"), "new_name")
                if new != name:
                    if new in views or _is_preset(paths, event, new):
                        raise ViewError(409, f"a view named \"{new}\" exists: pick "
                                             "another name")
                    views[new] = views.pop(name)
            else:
                del views[name]
                if not views:
                    del mine[event]
                if not mine:
                    del data[reviewer]
        _write(path, data)
    return list_views(paths, event, reviewer)
