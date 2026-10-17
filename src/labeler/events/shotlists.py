"""Named shot lists: files that each list shots, one registry that names them and the
lists each event recommends, and the one rule for which shots the review page opens
off an event's roster.

A list is `<label tables>/shotlists/<name>.csv` (a `shot` column; other columns, such
as batch, order, pool, classes, run_day or note, are kept) or `<name>.txt` (one shot
per line, `#` comments), with a `<name>.meta.json` sidecar saying how it was made
(`make` writes it: title, kind, rule, derived_from, provenance, made_at, made_by,
n_shots, sha256). The name is the file's stem, a plain name (`taxonomy.NAME`). `blind`
is reserved, and so is `roster`, each event's own roster.

The registry, `configs/labeler/shotlists.yaml` (`Paths.shotlists_manifest`), is
source, so it is tracked:

    lists:
      gold: {title: "Gold set", file: gold/gold_shots.csv, protect: test}
      silver_batch_1: {title: "Silver batch 1", parent: silver, hidden: true,
                       rule: "where(silver, batch==1)"}
    events:
      alfven_eigenmode: {recommended: [ae_co2_ece, silver, gold],
                         default: ae_co2_ece, evaluate: [ae_co2, gold]}

`file:` registers a list kept elsewhere under the label tables: gold and silver stay
where they are and keep their stricter readers (`review.gold`, `review.silver`), and
without a registry they are still lists. A list file the registry does not name is
still a list ("Other lists" on the page, kind `other`). A subcategory with no
`events:` entry takes its parent's; an event with none recommends silver, gold and
`catalog` ("All shots") and defaults to its roster. `protect: test | holdout` lists are
never overwritten by `make`. `hidden: true` keeps a list's file and name but leaves it out of the page's dropdown and
a parent's children (`offered`); a request for it falls back to the default.

No list holds a blind test shot (`review.silver.blind_shots`): when the catalog is
present, `read` refuses a list that does, and `make` will not write one (it fails
closed without the catalog). Any shot on any list may be reviewed on any event
(`in_any`), so that refusal is what keeps the blind shots off the page; a tree without
the catalog (a release) trusts its lists. `offered` orders the lists for the page's
dropdown, `holding` names the registered lists a shot is on, `held_out` says whether
a `protect: holdout` list holds it, and `plan` picks the shots an evaluator scores on
a list.

    python -m labeler.events.shotlists list
    python -m labeler.events.shotlists show NAME
    python -m labeler.events.shotlists make NAME (--from-txt F | --from-table F
        | --shots S [S ...] | --union A,B | --from LIST) [--where EXPR]
        [--sort COLUMN] [--title T] [--kind K] [--provenance TEXT] [--drop-blind]
    python -m labeler.events.shotlists check
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from ..config import Paths, atomic_path, git_sha, sha256_of
from .review import gold, silver
from .taxonomy import NAME

log = logging.getLogger(__name__)

DIRECTORY = "shotlists"
#: Each event's own roster, offered as a list beside the files.
ROSTER = "roster"
RESERVED = frozenset({"blind", ROSTER})
#: Longer names are no list (a path that long fails with ENAMETOOLONG, not "not found").
MAX_NAME = 128
KINDS = ("curated", "derived", "dataset")
PROTECT = ("test", "holdout")
SUFFIXES = (".csv", ".txt")
SIDECAR = ".meta.json"
#: The two lists kept outside `shotlists/`, as they are when the registry is silent.
BUILTIN = {
    "gold": {"title": "Gold set", "file": gold.GOLD_FILE, "protect": "test",
             "kind": "curated"},
    "silver": {"title": "Silver set", "file": silver.SILVER_FILE, "kind": "curated"},
}
LIST_KEYS = frozenset({"title", "file", "parent", "rule", "protect", "kind", "sha256",
                       "description", "hidden"})
#: Lists whose shots have no pre-built review store: the review server builds one
#: when a shot is opened (`ondemand`), so holding a shot says nothing about its store.
ON_DEMAND = frozenset({"catalog"})
EVENT_KEYS = frozenset({"recommended", "default", "evaluate"})
SCORING = ("heldout", "oof")
#: An event id: a plain name, or a subcategory's `<event>.<key>`.
EVENT = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ShotListError(ValueError):
    """A list, its sidecar or the registry is malformed, or `make` refuses."""


@dataclass(frozen=True)
class Entry:
    """One named list: where its file is and what the registry or sidecar says."""

    name: str
    path: Path
    title: str
    registered: bool
    kind: str | None = None
    parent: str | None = None
    rule: str | None = None
    protect: str | None = None
    #: A digest the registry pins the file to (`check` compares it).
    sha256: str | None = None
    #: Left out of the page's dropdown (`offered`); the file and name stay.
    hidden: bool = False


@dataclass(frozen=True)
class EventLists:
    """The lists one event recommends, opens on and is evaluated on."""

    recommended: tuple[str, ...] = ()
    default: str = ROSTER
    evaluate: tuple[str, ...] = ()


@dataclass(frozen=True)
class Registry:
    lists: dict[str, dict]
    events: dict[str, EventLists]


# ---- the registry ----


def registry(paths: Paths) -> Registry:
    """The registry at `paths.shotlists_manifest`, re-read when the file changes;
    with gold and silver added where it does not name them. No file is no registry."""
    version = _version(paths.shotlists_manifest)
    return _registry(str(paths.shotlists_manifest), version)


@lru_cache(maxsize=8)
def _registry(path: str, version) -> Registry:
    raw = {}
    if version is not None:
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise ShotListError(f"{path}: not readable: {error}") from error
    return parse(raw, where=path)


def parse(raw, *, where: str = "shotlists.yaml") -> Registry:
    """The registry of one parsed `shotlists.yaml`; every mistake names its entry."""
    if not isinstance(raw, dict):
        raise ShotListError(f"{where}: must be a mapping")
    block, events_block = raw.get("lists") or {}, raw.get("events") or {}
    if not isinstance(block, dict) or not isinstance(events_block, dict):
        raise ShotListError(f"{where}: `lists` and `events` must be mappings")
    lists: dict[str, dict] = {}
    for name, spec in block.items():
        check_name(name, where=where)
        spec = {} if spec is None else spec
        if not isinstance(spec, dict):
            raise ShotListError(f"{where}: {name} must be a mapping")
        unknown = set(spec) - LIST_KEYS
        if unknown:
            raise ShotListError(f"{where}: {name}: unknown keys {sorted(unknown)}")
        for key in ("title", "rule", "description"):
            value = spec.get(key)
            if key in spec and (not isinstance(value, str) or not value.strip()):
                raise ShotListError(
                    f"{where}: {name}.{key} must be a non-empty string"
                )
        if "file" in spec:
            file = spec["file"]
            if (not isinstance(file, str) or Path(file).is_absolute()
                    or ".." in Path(file).parts or Path(file).suffix not in SUFFIXES):
                raise ShotListError(
                    f"{where}: {name}.file must be a .csv or .txt path under the label "
                    "tables"
                )
        if spec.get("protect") not in (None, *PROTECT):
            raise ShotListError(f"{where}: {name}.protect must be one of {PROTECT}")
        if spec.get("kind") not in (None, *KINDS):
            raise ShotListError(f"{where}: {name}.kind must be one of {KINDS}")
        if "sha256" in spec and not SHA256.match(str(spec["sha256"])):
            raise ShotListError(f"{where}: {name}.sha256 must be 64 hex digits")
        if not isinstance(spec.get("hidden", False), bool):
            raise ShotListError(f"{where}: {name}.hidden must be true or false")
        lists[name] = dict(spec)
    # Gold and silver keep their files and gold its protection unless the entry says
    # otherwise; an entry that only retitles one does not move it.
    for name, spec in BUILTIN.items():
        lists[name] = {**spec, **lists.get(name, {})}
    for name, spec in lists.items():
        if spec.get("parent") is not None and spec["parent"] not in lists:
            raise ShotListError(
                f"{where}: {name}.parent {spec['parent']!r} is not a list"
            )
    events = {}
    for event, spec in events_block.items():
        if not isinstance(event, str) or not EVENT.match(event):
            raise ShotListError(f"{where}: {event!r} is not an event id")
        spec = {} if spec is None else spec
        if not isinstance(spec, dict) or set(spec) - EVENT_KEYS:
            raise ShotListError(
                f"{where}: events.{event} must be a mapping of {sorted(EVENT_KEYS)}"
            )
        found = {}
        for key in ("recommended", "evaluate"):
            value = spec.get(key) or []
            if not isinstance(value, list) or any(n not in lists for n in value):
                raise ShotListError(
                    f"{where}: events.{event}.{key} must list registered lists"
                )
            found[key] = tuple(value)
        default = spec.get("default", ROSTER)
        if default != ROSTER and default not in lists:
            raise ShotListError(
                f"{where}: events.{event}.default must be {ROSTER} or a registered list"
            )
        events[event] = EventLists(found["recommended"], default, found["evaluate"])
    return Registry(lists, events)


def check_name(name, *, where: str = "shot list") -> str:
    if not isinstance(name, str) or len(name) > MAX_NAME or not NAME.match(name):
        raise ShotListError(f"{where}: {name!r:.{MAX_NAME}} is not a plain list name")
    if name in RESERVED:
        raise ShotListError(f"{where}: {name!r} is reserved")
    return name


def event_lists(paths: Paths, event: str) -> EventLists:
    """What the registry says of `event`, else of its parent, else silver, gold and
    the all-shots list recommended, the roster as default and nothing to evaluate."""
    found = registry(paths)
    entry_ = found.events.get(event) or found.events.get(event.partition(".")[0])
    if entry_ is not None:
        return entry_
    return EventLists(tuple(name for name in ("silver", "gold", "catalog")
                            if name in found.lists))


def recommended(paths: Paths, event: str) -> list[str]:
    return list(event_lists(paths, event).recommended)


def default(paths: Paths, event: str) -> str:
    """The list `event` opens on: a list name, or `ROSTER` for its own roster."""
    return event_lists(paths, event).default


def evaluate(paths: Paths, event: str) -> list[str]:
    return list(event_lists(paths, event).evaluate)


# ---- the lists ----


def _version(path) -> tuple[int, int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_ino, stat.st_size


def _files(paths: Paths, *, sidecars: bool = False) -> list[Path]:
    """The list files under `shotlists/`, by name (temporary siblings are not lists),
    and with `sidecars` theirs too."""
    try:
        with os.scandir(paths.shotlists) as found:
            files = [Path(item.path) for item in found if item.is_file() and (
                Path(item.name).suffix in SUFFIXES
                or (sidecars and item.name.endswith(SIDECAR))
            )]
    except OSError:
        return []
    return sorted(files)


def names(paths: Paths) -> list[str]:
    """Every list: the registry's in its order, then each other file's by name. A
    file whose stem is not a plain name, or is reserved, is not one (`check` says
    so)."""
    found = list(registry(paths).lists)
    for path in _files(paths):
        stem = path.stem
        if stem not in found and NAME.match(stem) and stem not in RESERVED:
            found.append(stem)
    return found


def entry(paths: Paths, name: str) -> Entry:
    """List `name`; KeyError when there is no such list."""
    spec = registry(paths).lists.get(name)
    if spec is None and (not isinstance(name, str) or len(name) > MAX_NAME
                         or not NAME.match(name) or name in RESERVED):
        raise KeyError(name)
    if spec is not None and spec.get("file"):
        path = Path(paths.label_tables) / spec["file"]
    else:
        present = [Path(paths.shotlists) / f"{name}{suffix}" for suffix in SUFFIXES
                   if (Path(paths.shotlists) / f"{name}{suffix}").is_file()]
        if len(present) > 1:
            raise ShotListError(
                f"list {name} has two files: {present[0]}, {present[1]}"
            )
        if not present and spec is None:
            raise KeyError(name)
        path = present[0] if present else Path(paths.shotlists) / f"{name}.csv"
    spec = spec or {}
    meta = sidecar(path)
    return Entry(
        name=name, path=path, title=spec.get("title") or meta.get("title") or name,
        registered=name in registry(paths).lists,
        kind=spec.get("kind") or meta.get("kind"), parent=spec.get("parent"),
        rule=spec.get("rule"), protect=spec.get("protect"), sha256=spec.get("sha256"),
        hidden=bool(spec.get("hidden")),
    )


#: Each tree's resolved lists, kept until the registry or the lists' directory
#: changes (`make` renames every list and sidecar into it) or a file comes or goes
#: (the listing, so two changes within one clock tick are not missed).
_ENTRIES: dict = {}


def entries(paths: Paths) -> list[Entry]:
    """Every list as `names` orders them; one that cannot be resolved is logged."""
    stamp = (_version(paths.shotlists_manifest), _version(paths.shotlists),
             tuple(path.name for path in _files(paths, sidecars=True)))
    key = (str(paths.label_tables), str(paths.shotlists_manifest), str(paths.shotlists))
    cached = _ENTRIES.get(key)
    if cached is not None and cached[0] == stamp:
        return list(cached[1])
    found = []
    for name in names(paths):
        try:
            found.append(entry(paths, name))
        except (KeyError, ValueError) as error:
            _warn_once(name, str(error))
    _ENTRIES[key] = (stamp, tuple(found))
    return found


def sidecar_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + SIDECAR)


def sidecar(path) -> dict:
    """The `<stem>.meta.json` beside a list file, or {} when there is none."""
    meta = sidecar_path(path)
    version = _version(meta)
    return {} if version is None else dict(_sidecar(str(meta), version))


@lru_cache(maxsize=128)
def _sidecar(path: str, _version) -> dict:
    try:
        found = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        _warn_once(path, str(error))
        return {}
    return found if isinstance(found, dict) else {}


def read(paths: Paths, name: str) -> pd.DataFrame:
    """List `name`'s rows: `shot` (int64) first, then the file's other columns.

    Cached on the file's version and the catalog's, so never mutate the frame.
    Refuses a list that is malformed or holds a blind test shot.
    """
    return _read_entry(paths, entry(paths, name))


def shots(paths: Paths, name: str) -> list[int]:
    """List `name`'s shots in its order."""
    return [int(shot) for shot in read(paths, name).shot]


def _read_entry(paths: Paths, found: Entry) -> pd.DataFrame:
    version = _version(found.path)
    if version is None:
        raise FileNotFoundError(f"list {found.name}: no file {found.path}")
    cohort = _version(_cohort(paths))
    tables = Path(paths.label_tables)
    if found.path == tables / gold.GOLD_FILE:
        kind = "gold"
    elif found.path == tables / silver.SILVER_FILE:
        kind = "silver"
    else:
        kind = found.path.suffix
    return _read_file(kind, str(found.path), str(tables), version, cohort)


@lru_cache(maxsize=64)
def _read_file(kind: str, path: str, tables: str, _version, _cohort) -> pd.DataFrame:
    if kind == "gold":
        return gold.read_gold(tables)
    if kind == "silver":
        return silver.read_silver(tables)
    blind = blind_set_at(tables)
    if kind == ".txt":
        frame = pd.DataFrame({"shot": _txt_shots(path)})
    else:
        # The shot column is read as text to be checked strictly; the others keep
        # the types pandas finds (batch and order are integers, as silver's reader
        # has them).
        frame = pd.read_csv(path, dtype={"shot": str}, keep_default_na=False)
        if "shot" not in frame.columns:
            raise ShotListError(f"{path}: a list needs a `shot` column")
    values = pd.to_numeric(frame["shot"], errors="coerce")
    if not (values.notna() & (values % 1 == 0) & (values >= 0)).all():
        raise ShotListError(f"{path}: shot must be a nonnegative integer")
    frame["shot"] = values.astype("int64")
    repeated = frame.shot[frame.shot.duplicated()]
    if len(repeated):
        raise ShotListError(f"{path}: duplicate shot {int(repeated.iloc[0])}")
    refused = sorted(set(frame.shot) & blind)
    if refused:
        raise ShotListError(f"{path}: blind test shots may not be listed: {refused}")
    return frame[["shot", *(column for column in frame.columns if column != "shot")]]


def _txt_shots(path) -> list[int]:
    found = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if not re.fullmatch(r"[0-9]+", text):
            raise ShotListError(f"{path}:{number}: {text!r} is not a shot number")
        found.append(int(text))
    return found


@lru_cache(maxsize=1024)
def _warn_once(name: str, error: str) -> None:
    log.warning("shot list %s skipped: %s", name, error)


# ---- membership ----


def _cohort(paths: Paths) -> Path:
    return Path(paths.label_tables) / "catalog" / "cohort.csv"


def blind_set(paths: Paths) -> frozenset[int]:
    """The catalog's blind test shots, cached on the cohort file's version; empty
    without a catalog (a tree restored from the release, whose lists were scrubbed
    of them when they were made)."""
    return blind_set_at(paths.label_tables)


def blind_set_at(tables) -> frozenset[int]:
    cohort = Path(tables) / "catalog" / "cohort.csv"
    version = _version(cohort)
    return frozenset() if version is None else _blind(str(tables), version)


@lru_cache(maxsize=8)
def _blind(tables: str, _version) -> frozenset[int]:
    return frozenset(silver.blind_shots(tables))


#: Each tree's per-list shot sets, kept until any list file, the registry or the
#: catalog changes.
_MEMBERS: dict = {}


def members(paths: Paths) -> dict[str, frozenset[int]]:
    """Every readable list's shots, by name; a list that cannot be read is logged
    and left out, so one bad file never hides the others."""
    try:
        found = entries(paths)
    except ValueError as error:  # the registry itself: fall back to gold and silver
        _warn_once(str(paths.shotlists_manifest), str(error))
        tables = Path(paths.label_tables)
        found = [Entry(name, tables / spec["file"], spec["title"], False)
                 for name, spec in BUILTIN.items()]
    stamp = (tuple((item.name, str(item.path), _version(item.path)) for item in found),
             _version(_cohort(paths)))
    key = (str(paths.label_tables), str(paths.shotlists_manifest))
    cached = _MEMBERS.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    sets = {}
    for item in found:
        if _version(item.path) is None:
            continue
        try:
            sets[item.name] = frozenset(int(s) for s in _read_entry(paths, item).shot)
        except (OSError, ValueError) as error:
            _warn_once(f"{item.name} {_version(item.path)}", str(error))
    _MEMBERS[key] = (stamp, sets)
    return sets


def in_any(paths: Paths, shot: int) -> bool:
    """Whether `shot` is on some list (gold, silver, every registered or found list
    file), and so reviewable on every event; a blind shot never is."""
    shot = int(shot)
    return (any(shot in found for found in members(paths).values())
            and shot not in blind_set(paths))


def held_out(paths: Paths, shot: int) -> bool:
    """Whether a list marked `protect: holdout` (in the registry or its sidecar; the
    ELM challenge key's shots) holds `shot`. The review page opens no draft on such
    a shot: a draft could carry the withheld answer key into a save. Fails closed:
    an unreadable registry, or a holdout list whose file is there but cannot be
    read, holds every shot."""
    shot = int(shot)
    try:
        found = entries(paths)
    except ValueError:
        return True
    sets = members(paths)
    for item in found:
        if (item.protect or sidecar(item.path).get("protect")) != "holdout":
            continue
        if item.name in sets:
            if shot in sets[item.name]:
                return True
        elif _version(item.path) is not None:  # there, but unreadable
            return True
    return False


def holding(paths: Paths, shot: int) -> list[str]:
    """The registered lists `shot` is on, in `names` order (their stores are the
    ones prebuilt in SLURM, except those of the `ON_DEMAND` lists); cheap, from the
    cached `members`."""
    shot = int(shot)
    if shot in blind_set(paths):
        return []
    try:
        registered = registry(paths).lists
    except ValueError:  # an unreadable registry still has gold and silver
        registered = BUILTIN
    return [name for name, found in members(paths).items()
            if name in registered and shot in found]


def offered(paths: Paths, event: str) -> list[tuple[Entry, bool]]:
    """The readable lists the page offers on `event`, in its order, each with whether
    the event recommends it: the recommended ones in the order the registry gives for
    the event, then the rest in `names` order with each child right after its parent.
    A child of a recommended list opens the rest, in its parent's order: as near its
    parent as the recommended lists coming first allows. Raises ShotListError when
    the registry cannot be read."""
    found = {item.name: item for item in entries(paths) if not item.hidden}
    readable = members(paths)
    first = [name for name in event_lists(paths, event).recommended
             if name in readable and name in found]
    rest = [name for name in found if name in readable and name not in first]
    ordered: list[str] = []

    def place(name: str) -> None:
        ordered.append(name)
        for child in rest:
            if found[child].parent == name and child not in ordered:
                place(child)

    for parent in first:
        for name in rest:
            if found[name].parent == parent and name not in ordered:
                place(name)
    for name in rest:
        if name not in ordered and found[name].parent not in rest:
            place(name)
    ordered += [name for name in rest if name not in ordered]  # a parent cycle
    return ([(found[name], True) for name in first]
            + [(found[name], False) for name in ordered])


def inputs(paths: Paths) -> list[Path]:
    """The files every list answer depends on, for a cache keyed on their versions:
    the registry, the lists' directory, each list file and sidecar, and the catalog."""
    found = [Path(paths.shotlists_manifest), Path(paths.shotlists), _cohort(paths)]
    try:
        items = entries(paths)
    except ValueError:  # the registry itself, whose version is above
        return found
    return found + [path for item in items
                    for path in (item.path, sidecar_path(item.path))]


# ---- evaluation ----


def plan(paths: Paths, names, population, trained_on, reference, *,
         include_trained: bool = False, scoring: str = "heldout"):
    """Which shots of each named list an evaluator scores, and each list's envelope
    for the record's `sets.<name>`.

    `population` holds the shots the evaluator has inputs for (for an out-of-fold
    evaluator, its cross-validation population), `trained_on` the shots the model
    was fitted on and `reference` the shots the reference label table covers. A
    listed shot outside `population` is `missing.no_input` (`missing.not_in_cv` out
    of fold); one the model trained on is reported under `trained_on` and not scored
    unless `include_trained`, which marks the envelope `leaky`; one the reference
    lacks is `missing.no_reference`. Nothing is dropped without being named, so
    `n_listed` = `n_scored` + every reported shot not scored.

    Returns `(scored, sets)`: each list's shots to score, in list order, and its
    envelope.
    """
    if scoring not in SCORING:
        raise ValueError(f"scoring must be one of {SCORING}")
    population, trained_on, reference = (
        frozenset(int(shot) for shot in group)
        for group in (population, trained_on, reference)
    )
    scored, sets = {}, {}
    for name in names:
        found = entry(paths, name)
        listed = [int(shot) for shot in _read_entry(paths, found).shot]
        missing = {"no_input": [], "not_in_cv": [], "no_reference": []}
        trained, keep = [], []
        for shot in listed:
            if shot not in population:
                why = "no_input" if scoring == "heldout" else "not_in_cv"
                missing[why].append(shot)
                continue
            if shot in trained_on:
                trained.append(shot)
                if not include_trained:
                    continue
            if shot not in reference:
                missing["no_reference"].append(shot)
                continue
            keep.append(shot)
        scored[name] = keep
        sets[name] = {
            "list": name, "title": found.title, "list_sha256": sha256_of(found.path),
            "n_listed": len(listed), "n_scored": len(keep), "missing": missing,
            "trained_on": trained, "scoring": scoring,
            "leaky": bool(include_trained and trained),
        }
    return scored, sets


# ---- making and checking lists ----


def display(paths: Paths, path) -> str:
    """`path` as a sidecar records it: relative to the label tables, else under
    `$LABELER_ROOT/`, else absolute."""
    path = Path(path).resolve()
    for base, prefix in ((paths.label_tables, ""), (paths.root, "$LABELER_ROOT/")):
        try:
            return prefix + path.relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            continue
    return str(path)


def undisplay(paths: Paths, text: str) -> Path:
    if text.startswith("$LABELER_ROOT/"):
        return Path(paths.root) / text.removeprefix("$LABELER_ROOT/")
    return Path(text) if Path(text).is_absolute() else Path(paths.label_tables) / text


_CLAUSE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<=|>=|<|>)\s*(.+?)\s*")


def _literal(text: str):
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


def where(frame: pd.DataFrame, expr: str) -> pd.DataFrame:
    """The rows of `frame` meeting `expr`: `column OP value` clauses joined by
    `and`, OP one of == != < <= > >=; a value is a number, true/false or text."""
    keep = pd.Series(True, index=frame.index)
    for clause in re.split(r"\s+and\s+", expr.strip()):
        match = _CLAUSE.fullmatch(clause)
        if match is None:
            raise ShotListError(f"--where: {clause!r} is not `column OP value`")
        column, op, text = match.groups()
        if column not in frame.columns:
            raise ShotListError(
                f"--where: no column {column!r} in {list(frame.columns)}"
            )
        value, cells = _literal(text), frame[column]
        if isinstance(value, bool):
            if op not in {"==", "!="}:
                raise ShotListError(f"--where: {clause!r}: true/false takes == or !=")
            cells, value = cells.astype(str).str.lower(), str(value).lower()
        elif isinstance(value, int | float):
            cells = pd.to_numeric(cells, errors="coerce")
        else:
            cells = cells.astype(str)
        keep &= {
            "==": cells == value, "!=": cells != value, "<": cells < value,
            "<=": cells <= value, ">": cells > value, ">=": cells >= value,
        }[op].fillna(False).astype(bool)
    return frame[keep]


def _table_shots(path: Path, column: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if column is None:
        matches = [name for name in frame.columns if name.lower() == "shot"]
        if len(matches) != 1:
            raise ShotListError(f"{path}: name the shot column with --column")
        column = matches[0]
    if column not in frame.columns:
        raise ShotListError(f"{path}: no column {column!r}")
    return frame.rename(columns={column: "shot"})


def source(paths: Paths, args) -> tuple[pd.DataFrame, str, dict, bool]:
    """`(rows, rule, derived_from, keep_columns)` for `make`'s source options."""
    if args.from_txt:
        path = Path(args.from_txt)
        frame = pd.DataFrame({"shot": _txt_shots(path)})
        rule, inputs = f"txt({display(paths, path)})", {display(paths, path): path}
    elif args.from_table:
        path = Path(args.from_table)
        frame = _table_shots(path, args.column)
        rule, inputs = f"table({display(paths, path)})", {display(paths, path): path}
    elif args.shots:
        frame, rule, inputs = pd.DataFrame({"shot": args.shots}), "shots", {}
    elif args.union:
        parts = [part.strip() for part in args.union.split(",") if part.strip()]
        if len(parts) < 2:
            raise ShotListError("--union needs two or more lists, comma-separated")
        frame = pd.concat([read(paths, part)[["shot"]] for part in parts],
                          ignore_index=True)
        rule = f"union({', '.join(parts)})"
        inputs = {part: entry(paths, part).path for part in parts}
    else:
        frame = read(paths, args.from_list).copy()
        rule = args.from_list
        inputs = {args.from_list: entry(paths, args.from_list).path}
    keep_columns = bool(args.from_list)
    if args.where:
        frame = where(frame, args.where)
        rule = f"where({rule}, {args.where.strip()})"
    if args.sort:
        if args.sort not in frame.columns:
            raise ShotListError(f"--sort: no column {args.sort!r}")
        numbers = pd.to_numeric(frame[args.sort], errors="coerce")
        key = numbers if numbers.notna().all() else frame[args.sort].astype(str)
        frame = frame.iloc[key.argsort(kind="mergesort")]
        rule = f"sort({rule}, {args.sort})"
    derived = {name: {"sha256": sha256_of(path)} for name, path in inputs.items()}
    return frame, rule, derived, keep_columns


def make(paths: Paths, name: str, frame: pd.DataFrame, *, rule: str,
         derived_from: dict | None = None, keep_columns: bool = False,
         title: str | None = None, kind: str | None = None,
         provenance: str | None = None, made_by: str | None = None,
         drop_blind: bool = False) -> dict:
    """Write list `name` under `shotlists/` with its sidecar, both atomically, and
    return the sidecar. Refuses a reserved or `file:`-registered name, a protected
    list that exists, blind shots (unless `drop_blind`: they are left out and
    counted) and an empty list."""
    check_name(name, where="make")
    spec = registry(paths).lists.get(name, {})
    if spec.get("file"):
        raise ShotListError(
            f"{name} is kept at {spec['file']}; make writes only under {DIRECTORY}/"
        )
    if kind is not None and kind not in KINDS:
        raise ShotListError(f"kind must be one of {KINDS}")
    target = Path(paths.shotlists) / f"{name}.csv"
    if (Path(paths.shotlists) / f"{name}.txt").exists():
        raise ShotListError(f"{name}.txt exists: remove it to make {name}.csv")
    protect = spec.get("protect") or sidecar(target).get("protect")
    if protect and target.exists():
        raise ShotListError(f"{name} is protected ({protect}) and is never overwritten")
    values = pd.to_numeric(frame["shot"], errors="coerce")
    if not (values.notna() & (values % 1 == 0) & (values >= 0)).all():
        raise ShotListError(f"{name}: shot must be a nonnegative integer")
    frame = frame.assign(shot=values.astype("int64")).drop_duplicates("shot")
    # Fails closed: without the catalog nothing could be kept out of the blind shots.
    blind = silver.blind_shots(paths.label_tables)
    hit = sorted(set(frame.shot) & blind)
    if hit and not drop_blind:
        raise ShotListError(
            f"{name}: {len(hit)} blind test shot(s) {hit}; pass --drop-blind to leave "
            "them out"
        )
    frame = frame[~frame.shot.isin(blind)]
    if frame.empty:
        raise ShotListError(f"{name}: no shots")
    columns = (["shot", *(c for c in frame.columns if c != "shot")] if keep_columns
               else ["shot"])
    with atomic_path(target) as tmp:
        frame[columns].to_csv(tmp, index=False)
    meta = {
        "title": title or spec.get("title") or name,
        "kind": kind or spec.get("kind") or "derived",
        "rule": rule,
        "derived_from": derived_from or {},
        **({"provenance": provenance} if provenance else {}),
        **({"protect": spec["protect"]} if spec.get("protect") else {}),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "made_by": made_by or f"{getpass.getuser()} (labeler.events.shotlists make)",
        "git": git_sha(),
        "n_shots": len(frame),
        **({"blind_dropped": len(hit)} if hit else {}),
        "sha256": sha256_of(target),
    }
    with atomic_path(sidecar_path(target)) as tmp:
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def check(paths: Paths) -> tuple[list[str], list[str]]:
    """`(problems, notes)` over the registry and every list: each reads (integer,
    unique, no blind shot), each sidecar matches its file (n_shots, sha256, the
    registry's rule), no input a derived list was made from has changed since, a
    child is inside its parent, and a digest the registry pins still holds."""
    problems, notes = [], []
    try:
        found = registry(paths)
    except ValueError as error:
        return [str(error)], notes
    for path in _files(paths):
        if not NAME.match(path.stem) or path.stem in RESERVED:
            problems.append(f"{path.name}: {path.stem!r} is not a usable list name")
    read_ok: dict[str, set[int]] = {}
    for name in names(paths):
        try:
            item = entry(paths, name)
            frame = _read_entry(paths, item)
        except (KeyError, OSError, ValueError) as error:
            problems.append(f"{name}: {error}")
            continue
        read_ok[name] = set(frame.shot)
        if not item.registered:
            notes.append(f"{name}: not in the registry (shown under Other lists)")
        if item.sha256 and sha256_of(item.path) != item.sha256:
            problems.append(f"{name}: {item.path} differs from the registry's sha256")
        meta = sidecar(item.path)
        if not meta:
            if not (item.registered and found.lists[name].get("file")):
                notes.append(f"{name}: no sidecar")
            continue
        if meta.get("sha256") not in (None, sha256_of(item.path)):
            problems.append(f"{name}: the file changed since its sidecar was written")
        if meta.get("n_shots") not in (None, len(frame)):
            problems.append(
                f"{name}: sidecar n_shots {meta['n_shots']}, file {len(frame)}"
            )
        if item.rule and meta.get("rule") and item.rule != meta["rule"]:
            problems.append(f"{name}: rule {meta['rule']!r}, registry {item.rule!r}")
        for source_, recorded in (meta.get("derived_from") or {}).items():
            try:
                now = sha256_of(entry(paths, source_).path if NAME.match(source_)
                                else undisplay(paths, source_))
            except (KeyError, OSError, ValueError):
                problems.append(f"{name}: its input {source_} is gone")
                continue
            if isinstance(recorded, dict) and recorded.get("sha256") != now:
                problems.append(
                    f"{name}: stale, {source_} changed since {meta.get('made_at')}"
                )
    for name, spec in found.lists.items():
        parent = spec.get("parent")
        if parent and name in read_ok and parent in read_ok:
            outside = sorted(read_ok[name] - read_ok[parent])
            if outside:
                problems.append(
                    f"{name}: {len(outside)} shots not in its parent {parent}"
                )
    return problems, notes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every list, its size and what it is")
    show = commands.add_parser("show", help="one list's registry entry and rows")
    show.add_argument("name")
    made = commands.add_parser("make", help="write a list and its sidecar")
    made.add_argument("name")
    sources = made.add_mutually_exclusive_group(required=True)
    sources.add_argument("--from-txt", help="a text file, one shot per line")
    sources.add_argument("--from-table",
                         help="a CSV table's shot column (one row each)")
    sources.add_argument("--shots", nargs="+", type=int, help="the shots themselves")
    sources.add_argument("--union", help="two or more lists, comma-separated")
    sources.add_argument("--from", dest="from_list", help="a list, keeping its columns")
    made.add_argument("--column", help="--from-table's shot column (default: shot)")
    made.add_argument("--where", help="keep rows meeting `column OP value [and ...]`")
    made.add_argument("--sort", help="order by this column (numeric when it is)")
    made.add_argument("--title")
    made.add_argument("--kind", choices=KINDS)
    made.add_argument("--provenance", help="where the shots come from, in words")
    made.add_argument("--made-by")
    made.add_argument("--drop-blind", action="store_true",
                      help="leave the catalog's blind test shots out (counted)")
    commands.add_parser("check", help="validate the registry and every list")
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    try:
        if args.command == "list":
            for item in entries(paths):
                try:
                    count = str(len(_read_entry(paths, item)))
                except (OSError, ValueError) as error:
                    count = f"unreadable ({error})"
                flags = [flag for flag in (
                    item.kind, item.protect and f"protect:{item.protect}",
                    item.parent and f"parent:{item.parent}",
                    None if item.registered else "other",
                ) if flag]
                print(f"{item.name:20} {count:>6}  {item.title}  [{', '.join(flags)}]")
        elif args.command == "show":
            item = entry(paths, args.name)
            print(json.dumps({"name": item.name, "file": display(paths, item.path),
                              "title": item.title, "kind": item.kind,
                              "parent": item.parent, "rule": item.rule,
                              "protect": item.protect, "registered": item.registered,
                              "sidecar": sidecar(item.path)}, indent=1))
            print(_read_entry(paths, item).to_csv(index=False), end="")
        elif args.command == "make":
            frame, rule, derived, keep = source(paths, args)
            meta = make(paths, args.name, frame, rule=rule, derived_from=derived,
                        keep_columns=keep, title=args.title, kind=args.kind,
                        provenance=args.provenance, made_by=args.made_by,
                        drop_blind=args.drop_blind)
            dropped = meta.get("blind_dropped")
            print(f"wrote {args.name}: {meta['n_shots']} shots, rule {meta['rule']}"
                  + (f", {dropped} blind shot(s) left out" if dropped else ""))
        else:
            problems, notes = check(paths)
            for line in notes:
                print(f"note: {line}")
            for line in problems:
                print(f"PROBLEM: {line}")
            print(f"{len(names(paths))} lists, {len(problems)} problems")
            return 1 if problems else 0
    except KeyError as error:
        print(f"no shot list {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
