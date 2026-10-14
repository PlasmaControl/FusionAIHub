"""Which events nest under which, and what the review page calls them.

`configs/labeler/events.yaml` (`Paths.events_manifest`) holds one `taxonomy:`
block, the single source for the page's display names, its dropdown groups and its
subcategories:

    taxonomy:
      mhd_n1: {display: "MHD n=1", group: "MHD mode",
               subcategories: {m1: {display: "1/1 (...)", m: 1, n: 1}}}

An event with `review: false` keeps its directory and tables but is not offered
on the review page or in the silver set (disruption and VDE, owner 2026-10-11).

A subcategory is a SUBSET of its parent, not an exclusive category: it is labelled
only inside the time its parent's saved label marks, and sibling subcategories
overlap freely. Its id is `<parent>.<key>` (`mhd_n1.m1`) and its review files live
in `<label tables>/<parent>/subcategories/<key>/review/`. A new kind of subcategory
(an ELM type) is one more entry in the block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

MANIFEST = "events.yaml"
SUBDIR = "subcategories"
#: Event ids and subcategory keys: a plain name, so none can walk a path.
NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


class TaxonomyError(ValueError):
    """`events.yaml`'s taxonomy block is malformed."""


@dataclass(frozen=True)
class Child:
    """One subcategory of an event."""

    id: str
    parent: str
    key: str
    display: str
    m: int | None = None
    n: int | None = None


@dataclass(frozen=True)
class Entry:
    """One top-level event the taxonomy names."""

    display: str
    group: str | None
    children: tuple[Child, ...] = ()
    review: bool = True


@dataclass(frozen=True)
class Taxonomy:
    """The parsed block; every lookup is by dictionary, never by path."""

    top: dict[str, Entry] = field(default_factory=dict)
    kids: dict[str, Child] = field(default_factory=dict)

    def display(self, event: str) -> str:
        """The page's name for `event`; an event the block does not name keeps its id."""
        if event in self.top:
            return self.top[event].display
        if event in self.kids:
            return self.kids[event].display
        return event

    def group(self, event: str) -> str | None:
        """The dropdown group of `event` (a subcategory's is its parent's), or None."""
        parent = self.parent(event)
        entry = self.top.get(event if parent is None else parent)
        return None if entry is None else entry.group

    def reviewed(self, event: str) -> bool:
        """Whether the page offers `event` (a subcategory goes with its parent)."""
        entry = self.top.get(self.parent(event) or event)
        return entry is None or entry.review

    def children(self, parent: str) -> list[str]:
        """The ids of `parent`'s subcategories, in the order the block lists them."""
        entry = self.top.get(parent)
        return [] if entry is None else [child.id for child in entry.children]

    def parent(self, event: str) -> str | None:
        """The event a subcategory nests under; None for anything else."""
        child = self.kids.get(event)
        return None if child is None else child.parent

    def is_child(self, event: str) -> bool:
        return event in self.kids

    def child(self, event: str) -> Child | None:
        return self.kids.get(event)

    def child_dir(self, label_tables, child_id: str) -> Path:
        """`<label tables>/<parent>/subcategories/<key>`; KeyError for a non-child."""
        child = self.kids[child_id]
        return Path(label_tables) / child.parent / SUBDIR / child.key


def load(manifest) -> Taxonomy:
    """The taxonomy of the `events.yaml` at `manifest` (`Paths.events_manifest`),
    re-read when the file changes.

    No file, or no `taxonomy:` block, is an empty taxonomy: every event keeps its
    id as its name and has no subcategories.
    """
    path = Path(manifest)
    try:
        stat = path.stat()
    except OSError:
        return Taxonomy()
    return _parse(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _parse(path: str, _mtime_ns: int, _size: int) -> Taxonomy:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise TaxonomyError(f"{path}: not readable: {error}") from error
    block = raw.get("taxonomy") if isinstance(raw, dict) else None
    if block is None:
        return Taxonomy()
    return parse(block, where=path)


def parse(block, *, where: str = "taxonomy") -> Taxonomy:
    """The taxonomy of one `taxonomy:` mapping; every mistake names its entry."""
    if not isinstance(block, dict):
        raise TaxonomyError(f"{where}: `taxonomy` must be a mapping")
    top, kids = {}, {}
    for event, spec in block.items():
        if not isinstance(event, str) or not NAME.match(event):
            raise TaxonomyError(f"{where}: {event!r} is not a plain event name")
        spec = {} if spec is None else spec
        if not isinstance(spec, dict):
            raise TaxonomyError(f"{where}: {event} must be a mapping")
        children = []
        subs = spec.get("subcategories") or {}
        if not isinstance(subs, dict):
            raise TaxonomyError(f"{where}: {event}.subcategories must be a mapping")
        for key, sub in subs.items():
            if not isinstance(key, str) or not NAME.match(key):
                raise TaxonomyError(f"{where}: {event}.{key!r} is not a plain name")
            sub = {} if sub is None else sub
            if not isinstance(sub, dict):
                raise TaxonomyError(f"{where}: {event}.{key} must be a mapping")
            child = Child(
                id=f"{event}.{key}", parent=event, key=key,
                display=_text(sub.get("display", key), f"{event}.{key}.display", where),
                m=_integer(sub.get("m"), f"{event}.{key}.m", where),
                n=_integer(sub.get("n"), f"{event}.{key}.n", where),
            )
            children.append(child)
            kids[child.id] = child
        group = spec.get("group")
        review = spec.get("review", True)
        if type(review) is not bool:
            raise TaxonomyError(f"{where}: {event}.review must be true or false")
        top[event] = Entry(
            display=_text(spec.get("display", event), f"{event}.display", where),
            group=None if group is None else _text(group, f"{event}.group", where),
            children=tuple(children),
            review=review,
        )
    return Taxonomy(top, kids)


def _text(value, name: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaxonomyError(f"{where}: {name} must be a non-empty string")
    return value


def _integer(value, name: str, where: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise TaxonomyError(f"{where}: {name} must be a positive integer")
    return value
