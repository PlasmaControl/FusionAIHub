"""A labels-only release of the DIII-D event tables, for a private Hugging Face dataset.

Owner decision H1 (2026-10-14): the label tables, the 10 ms grids and the best model
per event go to the private dataset repo `nc1/tokamak-si`; no signals, stores,
spectrograms, checkpoints or third-party raw files. A release directory holds

    README.md                         the dataset card: YAML `configs`, then the text
    signals.yaml                      event -> best model -> input signals -> score
    taxonomy.yaml                     events, subcategories, categories, grid rule, provenance
    release.json                      version, counts and every file's sha256
    datasets.parquet                  one row per source table: tier, method, version, counts
    labels/<tier>/{train,holdout,test}.parquet   one row per span as written
    grid/{silver,holdout,gold}.parquet   the 10 ms grid of `review.silver`, run-length
                                      encoded (silver train shots, silver holdout shots, gold)
    shots/{silver,gold,holdout,cohort_split}.parquet
    tables/...                        a scrubbed mirror of the label tables, for `pull`
                                      (and configs/labeler's events.yaml and
                                      best_models.yaml, for reference: `pull` skips them)

The tiers follow the silver precedence: `reviewed` (review/labels.csv saves),
`original` (format/: third-party tables in the interval schema), `derived`
(extend_*: our producers; the sawtooth physics chunks, whose boolean `crowd` column
stands for attrs `iscrowd`, included) and `source` (machine drafts: the table each
review page opens on, and the latest suggestion of every method). Splits are per
shot: `test` the gold shots, `holdout` the shots any event roster holds out
(`holdout` true: kept from training and tuning), `train` the rest. `WITHHELD` tables
(the ELM challenge answer key: shared with permission, which is not a licence) stay
out unless a build includes them by name; the card says where to get them.
Scrub, everywhere: the catalog cohort's blind test shots are withheld from every
file; reviewers become R1..Rn in first-save order; the rosters' placeholder rows
(`EXAMPLE - replace with a real shot`) are dropped; attrs keep only `ATTRS` keys with
scalar values; absolute and home paths become `<group storage>/<name>`;
history.jsonl, raw/, npz grids and .meta.json bodies never ship (a mirrored table
gets a minimal sidecar). A table the grid takes frames from, or a review pointer
names, that the release does not hold fails the build unless allowed, and is then
flagged (`datasets.parquet` `shipped`).
The build writes into a hidden sibling and replaces `out` only when `out` is empty
or a previous release, never the label tables, the labeler root or what holds them.
`verify` re-reads the built directory and fails on any blind shot, absolute path,
login or reviewer name, or a shot in the wrong split, so a bad release is never left
looking finished. `restore` checks every mirrored file against the sha256 values in
`release.json` first, and never replaces a file or a pointer.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from ..config import Paths, atomic_path
from . import grid, rosters, taxonomy
from .catalog.states import STATE_NAMES
from .interval_tables import (
    INTERVAL_COLUMNS,
    WITH_ATTRS,
    attrs_text,
    category_labels,
    parse_attrs,
    validate_intervals,
)
from .review import labels, silver

log = logging.getLogger(__name__)

REPO_ID = "nc1/tokamak-si"
#: The default release name; a rebuild that is uploaded again takes the next one
#: (`build(release=...)`), and `upload` refuses a name the repo already has as a tag.
RELEASE = "v1"
RELEASE_NAME = re.compile(r"^v\d+(?:\.\d+)*$")
BEST_MODELS = "best_models.yaml"
#: The configuration a release carries under `tables/` for reference (`events.yaml`,
#: `best_models.yaml`, from `configs/labeler/`). The code reads them from the
#: checkout, never from the label tables, so `pull` does not copy them there.
CONFIG_FILES = frozenset({taxonomy.MANIFEST, BEST_MODELS})
#: No license yet (owner, 2026-10-14: "license dont put one yet"): the card has no
#: `license` field and the release no license file. Earlier releases shipped this
#: terms file, so `build` still accepts it in a release directory it replaces.
LICENSE_FILE = "LICENSE.md"
#: Tables a release leaves out unless a build includes them by name: dataset id ->
#: why, and where a user gets them instead.
WITHHELD = {
    "edge_localized_mode/format/edge_localized_mode_challenge_format_2026_v1": {
        "why": "the DIII-D ELM detection challenge answer key (David Eldon, General "
        "Atomics), shared with the maintainers' permission, which is not a licence to "
        "redistribute it",
        "get": "ask the maintainers of the DIII-D ELM detection challenge "
        "(github.com/DIII-D/elm_detection_challenge, "
        "`src/elm_detection_challenge/answer_key/ELM_database.json`, commit b71f991); "
        "its 24 shots stay `holdout` rows here, so nothing in this release trains on "
        "them",
    },
}
#: The sawtooth physics chunks' schema: a boolean `crowd` column instead of attrs
#: `iscrowd` (read like `review.silver` reads them: crowd marks the nonzero rows).
CROWD_COLUMNS = ("shot", "category", "t_start", "t_end", "crowd", "confidence", "attrs")
#: A roster row that only shows the format (`notes` says so); never a real shot.
PLACEHOLDER = re.compile(r"^\s*EXAMPLE\b")
PLACEHOLDER_TEXT = re.compile(r"EXAMPLE - replace with a real shot")
#: Directories of the label tables that hold no event.
NON_EVENTS = frozenset({"catalog", "silver", "gold", "shotlists"})
#: Silver layer kind -> release tier.
TIERS = {
    "saved": "reviewed",
    "folded": "reviewed",
    "format": "original",
    "extend": "derived",
    "source": "source",
    "suggest": "source",
}
TIER_ORDER = ("reviewed", "original", "derived", "source")
TIER_TEXT = {
    "reviewed": "saved by a reviewer on the review page (one label per shot and event)",
    "original": "third-party label tables, reformatted to the interval schema without "
    "changing what they say",
    "derived": "tables our producers made (`extend_*`: a rule or a model each)",
    "source": "machine drafts: the table each review page opens on (`layer` source) and "
    "the latest suggestion of every method (`layer` suggest); nobody reviewed "
    "them, so they are not labels",
}
#: Release splits, per shot: `test` the gold shots, `holdout` the shots any event
#: roster holds out (for every event: the shot is kept from training and tuning),
#: `train` the rest.
SPLITS = ("train", "holdout", "test")
#: The grid file each split's shots go to.
GRID_FILES = {"train": "silver", "holdout": "holdout", "test": "gold"}
#: What a release directory holds at its top level: `build` replaces only such a one.
RELEASE_ENTRIES = frozenset(
    {
        "README.md",
        LICENSE_FILE,
        "signals.yaml",
        "taxonomy.yaml",
        "release.json",
        "datasets.parquet",
        "labels",
        "grid",
        "shots",
        "tables",
    }
)
#: The attrs keys a release keeps, and what each says: scalar values only (a number,
#: flag, short text or a flat list of numbers; `scrub_attrs` drops any other value),
#: never a path, login or name. Anything else is dropped (the build report counts
#: what was), so a producer's new key ships only once listed here.
ATTRS = {
    "iscrowd": "1 a span of activity (crowd), 0 one countable event",
    "tier": "the producer's own confidence tier: confine-ours (`extend_confine_ours` "
    "gold_shots, roster_review) and detach_vote v1 (`extend_detach_vote/detach_shots`: "
    "certain, tangtv_only, conflict and others); the detach_vote v3 source table "
    "has no attrs",
    "predicted": "confine-ours: the class the network predicted before the confidence floor",
    "source": "confine-ours: held_out (cross-validated fold) or ensemble (all folds)",
    "extrapolated": "confine-ours: the shot lies after the training range",
    "n": "tm_interval: toroidal mode number",
    "reason": "why the row is uncertain: tm_interval (`extend_tm_interval`) and "
    "saw_physics (the `extend_saw_physics` chunks, e.g. q_prior_ece_contradicted)",
    "ended": "tm_interval: how the mode ended (decay, locked, plasma_end, unknown)",
    "locked": "tm_interval: a lock was confirmed",
    "locked_known": "tm_interval: whether lock status could be read",
    "locked_candidate": "tm_interval: a lock is possible but unconfirmed",
    "lock_time_ms": "tm_interval: the confirmed lock time",
    "lock_candidates_ms": "tm_interval: candidate lock times",
    "onset_window_ms": "tm_interval: [start of the leading weak track, interval start]",
    "onset_window_degenerate": "tm_interval: the onset window has zero length",
    "evidence_tier": "rwm_windows: what supports the row",
    "coverage_verified": "rwm_windows: whether coverage of the window was verified",
    "state": "saw_physics: the rule's state (present, absent, uncertain, unassessed)",
    "candidate_state": "saw_physics: the state of the candidate the crash belongs to",
    "train_id": "saw_physics: the crash train within the shot (<shot>:<k>)",
    "period_ms": "saw_physics: the train's crash period",
    "central_relative_drop": "saw_physics: the core ECE channel's relative drop at the "
    "crash",
    "inversion_nominal_rho": "saw_physics: the inversion radius in nominal rho",
    "geometry_status": "saw_physics: how the radius was mapped (nominal_second_harmonic_R "
    "or channel_order_only)",
    "qmin": "saw_physics: EFIT q_min at the crash",
    "q_source": "saw_physics: the equilibrium tree q_min came from",
    "tree": "the equilibrium tree that decided the row (EFIT02, EFIT02ER, EFIT01)",
}
#: Sidecar keys a mirrored table keeps.
META_KEYS = (
    "category",
    "event",
    "categories",
    "method",
    "version",
    "producer",
    "made_at",
    "made_by",
    "time_units",
    "schema_version",
    "table_kind",
)
#: Absolute paths: a release names files by dataset id, never by where they sit.
#: The storage roots of the clusters and laptops the tables were made on, and home
#: paths (`~/...`, `~login/...`), are cut by `scrub_text`.
ROOTS = (
    "scratch|projects|home|fusion|cscratch|pscratch|tmp|u|p|global|gpfs|lustre|"
    "mnt|net|nfs|work|private|Users|Volumes|opt|var|usr"
)
_PATH_TAIL = r"[^\s\"'`,;:)\]}]*"
ABSOLUTE = re.compile(
    rf"(?<![\w.<>])/(?:{ROOTS})/{_PATH_TAIL}"
    rf"|(?<![\w.<>~-])~(?:[A-Za-z][\w.-]*)?/[\w.-]{_PATH_TAIL}"
)
#: What `verify` refuses: a storage root or home path, or any other absolute path of
#: two parts or more (a URL's `//host/` is not one).
FORBIDDEN_PATH = re.compile(
    rf"(?<![\w.<>])/(?:{ROOTS})/"
    r"|(?<![\w.<>~-])~(?:[A-Za-z][\w.-]*)?/[\w.-]"
    r"|(?<![\w.<>/:~-])/[A-Za-z][\w.-]*/[\w.@-]+"
)
SHOT_TOKEN = re.compile(r"(?<!\d)(\d{6})(?!\d)")
HASH = re.compile(r"\b[0-9a-f]{40,64}\b")
#: File kinds a release never holds.
NEVER_SHIP = (".npz", ".npy", ".pkl", ".h5", ".sav", ".tar", ".pt", ".ckpt", ".jsonl")
MAX_BYTES = 200 * 2**20


class ReleaseError(ValueError):
    """The tables cannot make a clean release, or a built one is not clean."""


class NotClean(ReleaseError):
    """A built release that `verify` rejects (kept where it was built, to inspect)."""


# ---- what a release scrubs against ----


@dataclass(frozen=True)
class Context:
    """The shot sets and identities one build scrubs against."""

    paths: Paths
    blind: frozenset[int]
    gold: frozenset[int]
    silver: frozenset[int]
    #: shot -> the events whose roster holds it out
    holdout: dict[int, tuple[str, ...]] = field(repr=False)
    cohort: pd.DataFrame = field(repr=False)
    #: identity (a reviewer's recorded name, or a roster entry) -> R<k>
    reviewers: dict[str, str] = field(repr=False)
    #: logins and reviewer names that must not appear anywhere in a release
    secrets: tuple[str, ...] = field(repr=False)

    def split(self, shot) -> str:
        """`test` for a gold shot, `holdout` for one a roster holds out, else `train`."""
        shot = int(shot)
        if shot in self.gold:
            return "test"
        return "holdout" if shot in self.holdout else "train"


def placeholders(frame: pd.DataFrame) -> pd.Series:
    """The roster rows that only show the format (`notes` starts `EXAMPLE`)."""
    if "notes" not in frame:
        return pd.Series(False, index=frame.index)
    return frame["notes"].astype(str).str.match(PLACEHOLDER)


def read_roster(path) -> pd.DataFrame:
    """An event roster as text cells, without its placeholder rows."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
    return frame[~placeholders(frame)]


def _event_rosters(label_tables) -> list[Path]:
    return [
        path
        for path in sorted(Path(label_tables).glob(f"*/{rosters.ROSTER_NAME}"))
        if path.parent.name not in NON_EVENTS
    ]


def holdouts(label_tables) -> dict[int, tuple[str, ...]]:
    """Every shot an event roster holds out (`holdout` true: reserved from training
    and tuning for final evaluation), with the events whose rosters do."""
    found: dict[int, list[str]] = {}
    for path in _event_rosters(label_tables):
        frame = read_roster(path)
        if "holdout" not in frame:
            continue
        for shot in frame.shot[frame.holdout.str.strip().str.lower() == "true"]:
            found.setdefault(int(shot), []).append(path.parent.name)
    return {shot: tuple(events) for shot, events in sorted(found.items())}


def cohort(label_tables) -> pd.DataFrame:
    """The catalog cohort's shot, split and blind columns (empty when absent)."""
    path = Path(label_tables) / "catalog" / "cohort.csv"
    if not path.is_file():
        return pd.DataFrame({"shot": [], "split": [], "blind": []})
    frame = silver.read_cohort(path)
    return frame[["shot", "split", "blind"]].astype({"shot": "int64", "blind": bool})


def context(paths: Paths) -> Context:
    """Read the blind, gold and silver sets and the reviewer identities.

    A build with no blind list is refused: it could not show that nothing leaked.
    """
    tables = Path(paths.label_tables)
    frame = cohort(tables)
    blind = frozenset(int(s) for s in frame.shot[frame.blind])
    if not blind:
        raise ReleaseError(
            f"no blind cohort list under {tables}/catalog; refusing to build"
        )
    gold_file = tables / "gold" / "gold_shots.csv"
    gold = (
        frozenset(int(s) for s in pd.read_csv(gold_file).shot)
        if gold_file.is_file()
        else frozenset()
    )
    try:
        silver_shots = frozenset(int(s) for s in silver.read_silver(tables).shot)
    except FileNotFoundError:
        silver_shots = frozenset()
    for name, shots in (("gold", gold), ("silver", silver_shots)):
        if shots & blind:
            raise ReleaseError(
                f"{name} holds blind cohort shots {sorted(shots & blind)}"
            )
    if gold & silver_shots:
        raise ReleaseError(f"gold and silver share shots {sorted(gold & silver_shots)}")
    reviewers, secrets = _identities(tables)
    return Context(
        paths=paths,
        blind=blind,
        gold=gold,
        silver=silver_shots,
        holdout={s: e for s, e in holdouts(tables).items() if s not in blind},
        cohort=frame,
        reviewers=reviewers,
        secrets=secrets,
    )


def _histories(tables: Path) -> list[Path]:
    return sorted(tables.glob("**/review/history.jsonl"))


def _tokens(cells) -> set[str]:
    return {
        token.strip() for cell in cells for token in cell.split(";") if token.strip()
    }


def _identities(tables: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    """R1..Rn by each reviewer's first save (then real roster entries, sorted), and
    the logins and names a release must not carry: every recorded reviewer login and
    name, every roster entry (a `$USER` login) and the placeholder rows' names."""
    saves, logins, names = [], {getpass.getuser()}, set()
    for path in _histories(tables):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            logins.add(str(entry.get("reviewer") or ""))
            if entry.get("name"):
                names.add(str(entry["name"]))
                saves.append((str(entry.get("saved_at") or ""), str(entry["name"])))
    for pointer in tables.glob("**/review/source.json"):
        logins.add(str(json.loads(pointer.read_text()).get("set_by") or ""))
    for legacy in tables.glob("**/review/legacy/*__*__*"):
        logins.add(legacy.name.split("__")[1])
    order = list(dict.fromkeys(name for _, name in sorted(saves)))
    entries, examples = set(), set()
    for path in _event_rosters(tables):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
        if "reviewers" not in frame:
            continue
        example = placeholders(frame)
        entries |= _tokens(frame.reviewers[~example])
        examples |= _tokens(frame.reviewers[example])
    roster = sorted(entries - set(order))
    reviewers = {name: f"R{k}" for k, name in enumerate([*order, *roster], 1)}
    secrets = tuple(
        sorted(s for s in (logins | names | entries | examples) if len(s) >= 3)
    )
    return reviewers, secrets


def scrub_text(text: str) -> str:
    """`text` with every absolute path cut to `<group storage>/<its last part>`."""

    def short(match):
        last = match.group(0).rstrip("/").rsplit("/", 1)[-1]
        return f"<group storage>/{last}" if last else "<group storage>"

    return ABSOLUTE.sub(short, text)


def _scalar(value) -> bool:
    """A value an attrs key may ship with: a number, flag, short text or null, or a
    flat list of numbers (an interval, candidate times)."""
    if value is None or isinstance(value, bool | int | float):
        return True
    if isinstance(value, str):
        return len(value) <= 64
    if isinstance(value, list):
        return all(
            v is None or (isinstance(v, int | float) and not isinstance(v, bool))
            for v in value
        )
    return False


def scrub_attrs(cell) -> tuple[str, tuple[str, ...]]:
    """One attrs cell with only `ATTRS` keys whose value is scalar (canonical JSON,
    blank for none), and the keys it dropped."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return "", ()
    return _scrub_attrs(str(cell)) if isinstance(cell, str) else _scrub_value(cell)


@lru_cache(maxsize=1 << 16)
def _scrub_attrs(cell: str) -> tuple[str, tuple[str, ...]]:
    return _scrub_value(cell)


def _scrub_value(cell) -> tuple[str, tuple[str, ...]]:
    found = parse_attrs(cell)
    kept = {k: v for k, v in found.items() if k in ATTRS and _scalar(v)}
    return scrub_text(attrs_text(kept)), tuple(sorted(set(found) - set(kept)))


# ---- the tables a release reads ----


@dataclass(frozen=True)
class Table:
    """One interval table, in the silver precedence, named by its dataset id."""

    event: str
    layer: str
    path: Path = field(repr=False)
    dataset: str
    method: str
    version: str
    meta: dict = field(default_factory=dict, repr=False)

    @property
    def tier(self) -> str:
        return TIERS[self.layer]


def events(paths: Paths) -> list[str]:
    """Every event directory and every subcategory the taxonomy names."""
    tables = Path(paths.label_tables)
    tree = taxonomy.load(paths.events_manifest)
    tops = sorted(
        path.name
        for path in tables.iterdir()
        if path.is_dir()
        and path.name not in NON_EVENTS
        and taxonomy.NAME.match(path.name)
    )
    return [event for top in tops for event in [top, *tree.children(top)]]


def dataset_id(paths: Paths, path) -> str:
    """A table's name in a release: its path under the label tables, or under
    `suggestions/`, without the suffix."""
    path = Path(path).resolve()
    for base, prefix in (
        (Path(paths.label_tables).resolve(), ""),
        ((paths.root / "suggestions").resolve(), "suggestions/"),
    ):
        if path.is_relative_to(base):
            return prefix + path.relative_to(base).with_suffix("").as_posix()
    raise ReleaseError(f"{path.name}: neither a label table nor a suggestion table")


def _method_version(
    layer: str, dataset: str, path: Path, meta: dict
) -> tuple[str, str]:
    if dataset.startswith("suggestions/"):
        parts = dataset.split("/")
        return parts[1], parts[2]
    if layer in ("saved", "folded"):
        return "review", ""
    if layer == "format":
        found = re.search(r"(\d{4}_v\d+)$", path.stem)
        return "format", found.group(1) if found else ""
    producer = next(
        (
            part[len("extend_") :]
            for part in reversed(path.parts[:-1])
            if part.startswith("extend_")
        ),
        "",
    )
    found = re.search(r"(v\d+)$", path.stem)
    return str(meta.get("producer") or producer), str(
        meta.get("version") or (found.group(1) if found else "")
    )


def withheld(include=()) -> dict[str, dict]:
    """The `WITHHELD` tables a build leaves out: all but those it includes by name."""
    unknown = set(include) - set(WITHHELD)
    if unknown:
        raise ReleaseError(f"--include names no withheld table: {sorted(unknown)}")
    return {name: item for name, item in WITHHELD.items() if name not in include}


def tables(paths: Paths, *, exclude=(), include=()) -> tuple[list[Table], list[dict]]:
    """Every table the silver layers read, once, and the ones left out with why.

    A table other than a save needs its `.meta.json` sidecar (shards and shot lists
    have none) and the interval schema (or the sawtooth chunks' `CROWD_COLUMNS`);
    `exclude` names dataset ids left out on purpose (a table not cleared for
    release; an id that is no table of the labeler raises, as an unknown `include`
    does), and the `WITHHELD` ones are left out unless `include` names them.
    """
    found, skipped, seen, known = [], [], set(), set()
    exclude = set(exclude)
    held = withheld(include)
    for event in events(paths):
        for layer, path in silver._tables(paths, event):
            if not path.is_file() or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            name = dataset_id(paths, path)
            known.add(name)
            sidecar = path.with_suffix(".meta.json")
            if name in held:
                skipped.append(
                    {"dataset": name, "why": "withheld: " + held[name]["why"]}
                )
                continue
            if name in exclude:
                skipped.append({"dataset": name, "why": "excluded at build"})
                continue
            if layer != "saved" and not sidecar.is_file():
                skipped.append({"dataset": name, "why": "no .meta.json sidecar"})
                continue
            if tuple(pd.read_csv(path, nrows=0).columns) not in (
                INTERVAL_COLUMNS,
                WITH_ATTRS,
                CROWD_COLUMNS,
            ):
                skipped.append({"dataset": name, "why": "not the interval schema"})
                continue
            meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            found.append(
                Table(
                    event,
                    layer,
                    path,
                    name,
                    *_method_version(layer, name, path, meta),
                    meta,
                )
            )
    if exclude - known:
        raise ReleaseError(
            f"--exclude names no table of the labeler: {sorted(exclude - known)}"
        )
    return found, skipped


def _categories(table: Table, codes=()) -> dict[str, str]:
    """The table's category names: the event's map with its sidecar's names over it,
    so a code the sidecar leaves out keeps the event's name. A code neither names (a
    frame model's 3, not observable, on a present/absent event) takes the catalog
    state name; one with no name at all fails the build."""
    names = table.meta.get("categories")
    found = dict(category_labels(table.event))
    if isinstance(names, dict):
        found.update({str(k): str(v) for k, v in names.items()})
    for code in sorted({int(c) for c in codes}):
        if str(code) in found:
            continue
        if code not in STATE_NAMES:
            raise ReleaseError(
                f"{table.dataset}: category {code} has no name in its sidecar, the "
                "event's categories or the catalog states"
            )
        found[str(code)] = STATE_NAMES[code]
    return found


def read_table(path) -> pd.DataFrame:
    """An interval table in the interval schema (with attrs). A sawtooth chunk's
    `crowd` flag becomes attrs `iscrowd` on its nonzero rows, as `review.silver`
    reads it (category 0 is never a crowd)."""
    frame = pd.read_csv(path, index_col=False)
    if tuple(frame.columns) == CROWD_COLUMNS:
        crowd = frame.pop("crowd").astype(str).str.strip().str.lower() == "true"
        frame["attrs"] = [
            attrs_text({**parse_attrs(cell), "iscrowd": int(flag)} if c else cell)
            for cell, flag, c in zip(
                frame["attrs"], crowd, frame.category != 0, strict=True
            )
        ]
    return validate_intervals(frame)


def _last_reviewer(history: Path, reviewers: dict[str, str]) -> dict[int, str | None]:
    """Each shot's reviewer pseudonym from its newest save (None when the save
    recorded no name)."""
    found = {}
    if history.is_file():
        for line in history.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                found[int(entry["shot"])] = reviewers.get(entry.get("name") or "")
    return found


def rows(table: Table, ctx: Context) -> tuple[pd.DataFrame, dict]:
    """The table's rows as a release holds them, and what the scrub took out."""
    frame = read_table(table.path)
    blind = frame.shot.isin(ctx.blind)
    frame = frame[~blind].reset_index(drop=True)
    cells = frame["attrs"] if "attrs" in frame else pd.Series([""] * len(frame))
    scrubbed = [scrub_attrs(cell) for cell in cells]
    dropped = sorted({key for _, keys in scrubbed for key in keys})
    attrs = [text for text, _ in scrubbed]
    names = _categories(table, frame.category.unique())
    reviewer = (
        _last_reviewer(table.path.parent / "history.jsonl", ctx.reviewers)
        if table.tier == "reviewed"
        else {}
    )
    crowd = [parse_attrs(text).get("iscrowd") for text in attrs]
    out = pd.DataFrame(
        {
            "shot": frame.shot.astype("int32"),
            "event": table.event,
            "category": frame.category.astype("int16"),
            "category_name": frame.category.map(lambda c: names.get(str(int(c)), "")),
            "t_start_ms": frame.t_start.astype("float64"),
            "t_end_ms": frame.t_end.astype("float64"),
            "confidence": frame.confidence.astype("float64"),
            "iscrowd": pd.array(
                [None if c is None else bool(c) for c in crowd], dtype="boolean"
            ),
            "attrs": attrs,
            "tier": table.tier,
            "layer": table.layer,
            "dataset": table.dataset,
            "method": table.method,
            "version": table.version,
            "reviewer": pd.array(
                [reviewer.get(int(s)) for s in frame.shot], dtype="string"
            ),
            "split": [ctx.split(s) for s in frame.shot],
        }
    )
    return out, {"withheld_rows": int(blind.sum()), "dropped_attrs": dropped}


# ---- the 10 ms grid ----


def export_grid(paths: Paths, shots, out_dir, ctx: Context) -> Path:
    """`review.silver.export` of `shots` into `out_dir`, refusing blind shots."""
    shots = [int(s) for s in shots]
    if set(shots) & ctx.blind:
        raise ReleaseError(
            f"blind cohort shots are never exported: {sorted(set(shots) & ctx.blind)}"
        )
    silver.export(paths, shots, out_dir)
    return Path(out_dir)


def _origin(paths: Paths, origin: str) -> tuple[str, str]:
    """(layer, dataset id) of one grid origin name (`kind:name`)."""
    kind = origin.split(":", 1)[0]
    path = silver.layer_file(paths, silver.Layer(kind, origin, None))
    return kind, dataset_id(paths, path)


def grid_rle(npz_dir, paths: Paths) -> pd.DataFrame:
    """Every shot of an exported grid as runs of one category from one origin.

    Runs cover each event's whole frame range, unassessed (-1) runs included, so a
    shot's grid is rebuilt exactly from its runs (`decode_grid`).
    """
    npz_dir = Path(npz_dir)
    index = json.loads((npz_dir / "index.json").read_text())
    origins: dict[str, tuple[str, str]] = {}
    records = []
    for shot in index["shots"]:
        with np.load(npz_dir / f"{int(shot)}.npz", allow_pickle=False) as data:
            values, owners = data["labels"], data["origin"]
            names, sources = (
                [str(e) for e in data["events"]],
                [str(o) for o in data["origins"]],
            )
            step = int(data["grid_ms"])
            first = int(data["t0_ms"]) // step
        for name in sources:
            if name not in origins:
                origins[name] = _origin(paths, name)
        for i, event in enumerate(names):
            value, owner = values[i], owners[i]
            if not len(value):
                continue
            edges = np.flatnonzero((np.diff(value) != 0) | (np.diff(owner) != 0)) + 1
            starts = np.concatenate([[0], edges])
            stops = np.concatenate([edges, [len(value)]])
            for a, b in zip(starts.tolist(), stops.tolist(), strict=True):
                k = int(owner[a])
                layer, dataset = origins[sources[k]] if k >= 0 else (None, None)
                records.append(
                    (
                        int(shot),
                        event,
                        first + a,
                        b - a,
                        int(value[a]),
                        layer,
                        None if layer is None else TIERS[layer],
                        dataset,
                    )
                )
    frame = pd.DataFrame(
        records,
        columns=[
            "shot",
            "event",
            "frame",
            "n_frames",
            "category",
            "layer",
            "tier",
            "dataset",
        ],
    )
    frame["t_start_ms"] = frame.frame * grid.GRID_MS
    frame["t_end_ms"] = (frame.frame + frame.n_frames) * grid.GRID_MS
    return frame.astype(
        {
            "shot": "int32",
            "frame": "int32",
            "n_frames": "int32",
            "category": "int8",
            "t_start_ms": "int32",
            "t_end_ms": "int32",
            "layer": "string",
            "tier": "string",
            "dataset": "string",
        }
    )[
        [
            "shot",
            "event",
            "frame",
            "n_frames",
            "t_start_ms",
            "t_end_ms",
            "category",
            "tier",
            "layer",
            "dataset",
        ]
    ]


def decode_grid(runs: pd.DataFrame, shot: int) -> tuple[list[str], int, np.ndarray]:
    """One shot's (events, t0 in ms, event x frame categories) back from its runs."""
    runs = runs[runs.shot == int(shot)]
    first = int(runs.frame.min())
    n = int((runs.frame + runs.n_frames).max()) - first
    names = list(dict.fromkeys(runs.event))
    out = np.full((len(names), n), grid.UNASSESSED, dtype=np.int8)
    for row in runs.itertuples():
        start = int(row.frame) - first
        out[names.index(row.event), start : start + int(row.n_frames)] = row.category
    return names, first * grid.GRID_MS, out


# ---- the mirror a fresh clone restores ----


def _mirror_csv(source: Path, dest: Path, ctx: Context, *, roster=False) -> int:
    """`source` without blind shots and attrs whitelisted; a `roster` also without
    its placeholder rows and with its reviewers pseudonymised; cells otherwise as
    written. Returns the rows withheld."""
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, index_col=False)
    shots = pd.to_numeric(frame.shot, errors="coerce")
    blind = shots.isin(ctx.blind)
    example = placeholders(frame) if roster else pd.Series(False, index=frame.index)
    frame = frame[~blind & ~example]
    if "attrs" in frame:
        frame["attrs"] = [scrub_attrs(cell)[0] for cell in frame["attrs"]]
    if roster and "reviewers" in frame:
        frame["reviewers"] = [
            ";".join(
                ctx.reviewers.get(t.strip(), "R?") for t in cell.split(";") if t.strip()
            )
            for cell in frame["reviewers"]
        ]
    for column in frame.columns:  # every cell is text (read with dtype=str)
        frame[column] = frame[column].map(scrub_text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return int(blind.sum())


def _mirror_meta(
    meta: dict, dest: Path, *, dataset: str, n_rows: int, n_shots: int
) -> None:
    kept = {key: meta[key] for key in META_KEYS if key in meta}
    kept.update(
        {
            "n_rows": n_rows,
            "n_shots": n_shots,
            "release": f"scrubbed mirror of {dataset}: blind cohort rows withheld, attrs "
            "keys whitelisted; the full provenance stays with the source table",
        }
    )
    dest.write_text(scrub_text(json.dumps(kept, indent=1, default=str)) + "\n")


def _mirror_path(dataset: str) -> str:
    return f"tables/{dataset}.csv"


def mirror(out: Path, found: list[Table], ctx: Context) -> dict:
    """The label tables a fresh clone restores with `pull`, under `out/tables`."""
    tables_dir = Path(ctx.paths.label_tables)
    report = {"tables": 0, "withheld_rows": 0, "pointers": 0}
    for table in found:
        dest = out / _mirror_path(table.dataset)
        report["withheld_rows"] += _mirror_csv(table.path, dest, ctx)
        report["tables"] += 1
        # A suggestion's per-onset drafts (elm_jump) sit beside it; the review page
        # drafts onsets from them, so a restored tree needs them too.
        onsets = table.path.with_name(f"{table.path.stem}_onsets.csv")
        if onsets.is_file():
            report["withheld_rows"] += _mirror_csv(
                onsets, dest.with_name(f"{dest.stem}_onsets.csv"), ctx
            )
            report["onsets"] = report.get("onsets", 0) + 1
        if table.meta:
            kept = pd.read_csv(dest, usecols=["shot"])
            _mirror_meta(
                table.meta,
                dest.with_suffix(".meta.json"),
                dataset=table.dataset,
                n_rows=len(kept),
                n_shots=int(kept.shot.nunique()),
            )
    for pointer in sorted(tables_dir.glob("**/review/source.json")):
        entry = json.loads(pointer.read_text())
        try:
            target = dataset_id(ctx.paths, entry["table"])
        except ReleaseError:
            continue
        if not (out / _mirror_path(target)).is_file():
            report.setdefault("pointers_skipped", []).append(target)
            continue
        dest = out / "tables" / pointer.relative_to(tables_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(
                {
                    "table": _mirror_path(target),
                    "method": entry.get("method"),
                    "version": entry.get("version"),
                    "set_at": entry.get("set_at"),
                },
                indent=1,
            )
            + "\n"
        )
        report["pointers"] += 1
    for roster in _event_rosters(tables_dir):
        _mirror_csv(
            roster, out / "tables" / roster.relative_to(tables_dir), ctx, roster=True
        )
    for listing in ("silver/silver_shots.csv", "gold/gold_shots.csv"):
        if (tables_dir / listing).is_file():
            _mirror_csv(tables_dir / listing, out / "tables" / listing, ctx)
    # The checkout's configuration, for reference: `pull` never writes these into the
    # label tables (`CONFIG_FILES`); the code reads them from `configs/labeler/`.
    for source, name in (
        (ctx.paths.events_manifest, taxonomy.MANIFEST),
        (ctx.paths.best_models, BEST_MODELS),
    ):
        if Path(source).is_file():
            (out / "tables" / name).write_text(scrub_text(Path(source).read_text()))
    return report


def pull_plan(folder, paths: Paths) -> list[tuple[Path, Path]]:
    """(release file, destination) for every mirrored table of a downloaded release:
    `tables/suggestions/...` under `<root>/suggestions`, the rest under the label
    tables. Pointers are written by `restore`, not copied."""
    folder = Path(folder) / "tables"
    plan = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder)
        if relative.parts[-2:] == ("review", "source.json"):
            continue
        if relative.as_posix() in CONFIG_FILES:
            continue
        if relative.parts[0] == "suggestions":
            plan.append((path, paths.root / relative))
        else:
            plan.append((path, Path(paths.label_tables) / relative))
    return plan


def check_files(folder) -> list[str]:
    """What is wrong with a downloaded release's `tables/`: a file missing, changed
    (its sha256 is not the one `release.json` lists) or not in the release at all."""
    folder = Path(folder)
    try:
        listed = json.loads((folder / "release.json").read_text())["files"]
    except (OSError, ValueError, KeyError) as error:
        return [f"release.json: {error}"]
    wanted = {name: item for name, item in listed.items() if name.startswith("tables/")}
    problems = []
    for path in sorted((folder / "tables").rglob("*")):
        name = path.relative_to(folder).as_posix()
        if not path.is_file():
            continue
        if name not in wanted:
            problems.append(f"{name}: not in release.json")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != wanted[name]["sha256"]:
            problems.append(f"{name}: sha256 differs from release.json")
    problems += [
        f"{name}: missing" for name in sorted(wanted) if not (folder / name).is_file()
    ]
    return problems


def restore(folder, paths: Paths) -> dict:
    """Copy a downloaded release's tables into place and set each review pointer
    through `labels.write_pointer`. Every mirrored file must match the sha256 values
    in `release.json`, with nothing extra. Nothing is ever replaced: a file that
    exists and differs, or a pointer at another table, refuses the whole restore
    before anything is written. Label tables whose catalog lists the blind cohort are
    the full tables a release was cut from, and are refused outright (the scrubbed
    mirror differs from them by design); restore into a fresh directory instead
    (`pull --into DIR`)."""
    folder = Path(folder)
    tables_dir = Path(paths.label_tables)
    if cohort(tables_dir).blind.any():
        raise ReleaseError(
            f"{tables_dir} holds the full label tables (its catalog lists the blind "
            "cohort); restore into a fresh directory instead"
        )
    problems = check_files(folder)
    if problems:
        raise ReleaseError(
            f"{folder} is not the release its release.json lists: {problems[:3]}"
        )
    plan = pull_plan(folder, paths)
    conflicts = [
        str(dest)
        for src, dest in plan
        if dest.is_file() and dest.read_bytes() != src.read_bytes()
    ]
    where, pointers = dict(plan), []
    for pointer in sorted((folder / "tables").glob("**/review/source.json")):
        entry = json.loads(pointer.read_text())
        target = where.get(folder / entry["table"])
        if target is None:
            raise ReleaseError(
                f"{pointer.name} names {entry['table']}, not in the release"
            )
        event_dir = tables_dir / pointer.parent.parent.relative_to(folder / "tables")
        local = labels.pointer_path(event_dir)
        if local.is_file():
            current = Path(json.loads(local.read_text())["table"])
            if current.resolve() != target.resolve():
                conflicts.append(str(local))
            continue
        pointers.append((event_dir, target, entry))
    if conflicts:
        raise ReleaseError(
            f"{len(conflicts)} local files differ from the release (never replaced), "
            f"e.g. {conflicts[:3]}"
        )
    copied = 0
    for src, dest in plan:
        if dest.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(dest) as tmp:
            shutil.copyfile(src, tmp)
        copied += 1
    for event_dir, target, entry in pointers:
        labels.write_pointer(
            event_dir, target, method=entry["method"], version=entry["version"]
        )
    return {"files": len(plan), "copied": copied, "pointers": len(pointers)}


# ---- the release ----


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _git() -> dict:
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": sha, "git_dirty": dirty}


def _taxonomy(paths: Paths, event_ids: list[str], release: str = RELEASE) -> dict:
    """taxonomy.yaml: every event's display, parent, categories and folding, the grid
    rule and the third-party provenance, with no raw paths."""
    tree = taxonomy.load(paths.events_manifest)
    manifest = yaml.safe_load(Path(paths.events_manifest).read_text()) or {}
    grid_events = set(silver.review_events(paths))
    entries = {}
    for event in event_ids:
        child = tree.child(event)
        entry = {
            "display": tree.display(event),
            "group": tree.group(event),
            "parent": tree.parent(event),
            "on_review_page": tree.reviewed(event) and event in grid_events,
            "categories": {int(k): v for k, v in category_labels(event).items()},
        }
        if child is not None and child.m is not None:
            entry.update({"m": child.m, "n": child.n})
        if tree.default(event):  # what unmarked time in the owner's save reads as
            entry["default"] = tree.default(event)
        if event in silver.FOLD:
            parent, mapping = silver.FOLD[event]
            entry["folded_into"] = {"event": parent, "categories": dict(mapping)}
        entries[event] = {k: v for k, v in entry.items() if v is not None}
    return {
        "release": release,
        "grid": {
            "grid_ms": grid.GRID_MS,
            "unassessed": grid.UNASSESSED,
            "rule": silver.GRID_RULE,
            "precedence": list(silver.PRECEDENCE),
            "zero": {e: silver.zero_text(tree, e) for e in silver.review_events(paths)},
            "default": {e: tree.default(e) for e in silver.review_events(paths)},
        },
        "tiers": dict(TIER_TEXT),
        "attrs": dict(ATTRS),
        "events": entries,
        "original_datasets": [
            {
                "table": item.get("raw_path"),
                "event": item.get("name"),
                "date": item.get("date"),
                "sources": item.get("sources") or [],
            }
            for item in manifest.get("format_datasets") or []
        ],
        "third_party_sources": [
            {
                "stem": item.get("stem"),
                "provenance": scrub_text(str(item.get("provenance") or "")),
                "formatter": item.get("formatter"),
                "citation": item.get("citation"),
            }
            for item in manifest.get("raw_datasets") or []
        ],
    }


def _signals(paths: Paths) -> dict:
    path = Path(paths.best_models)
    if not path.is_file():
        raise ReleaseError(f"no {path}")
    return yaml.safe_load(path.read_text())


def pending(signals: dict) -> dict[str, str]:
    """Events whose best-model entry still carries a `todo` marker."""
    return {
        event: str(entry["todo"])
        for event, entry in (signals.get("events") or {}).items()
        if isinstance(entry, dict) and entry.get("todo")
    }


def pointers(paths: Paths) -> dict[str, str]:
    """Each review pointer's event directory and the dataset id it names (a table
    outside the label tables and suggestions by its scrubbed path)."""
    tables_dir = Path(paths.label_tables)
    found = {}
    for pointer in sorted(tables_dir.glob("**/review/source.json")):
        table = str(json.loads(pointer.read_text()).get("table") or "")
        where = pointer.parent.parent.relative_to(tables_dir).as_posix()
        try:
            found[where] = dataset_id(paths, table)
        except ReleaseError:
            found[where] = scrub_text(table)
    return found


def review_sources(pointed: dict[str, str], provenance: list[dict]) -> dict[str, dict]:
    """Each event directory -> the method and version of the table its review
    pointer names, for the pointers whose table the release ships. The card prints
    it from here, so it cannot go stale the way a hand-kept version can."""
    shipped = {item["dataset"]: item for item in provenance if item.get("shipped")}
    return {
        where: {
            "dataset": dataset,
            "method": shipped[dataset].get("method"),
            "version": shipped[dataset].get("version"),
        }
        for where, dataset in sorted(pointed.items())
        if dataset in shipped
    }


def _reads_item(text) -> tuple[str, str]:
    """`crash v3` -> ("crash", "v3"); a `reads_tables` item of best_models.yaml."""
    method, _, version = str(text).strip().rpartition(" ")
    if not method or not version:
        raise ReleaseError(
            f"best_models.yaml reads_tables item {text!r} is not '<method> <version>'"
        )
    return method, version


def check_reads_tables(signals: dict, provenance: list[dict]) -> None:
    """Every table a best-model entry says it also reads (`reads_tables`: a list of
    '<method> <version>') must be one the release ships."""
    shipped = {
        (str(item.get("method")), str(item.get("version")))
        for item in provenance
        if item.get("shipped")
    }
    missing = {
        f"{event}: {item}"
        for event, entry in (signals.get("events") or {}).items()
        if isinstance(entry, dict)
        for item in entry.get("reads_tables") or []
        if _reads_item(item) not in shipped
    }
    if missing:
        raise ReleaseError(
            f"best_models.yaml names tables the release does not ship: {sorted(missing)}"
        )


def _source_text(entry: dict, event: str, sources: dict[str, dict]) -> str:
    """The tables a card line's label set rests on: the one the event's review page
    opens on (from the shipped pointer; a subcategory takes its event's), unless
    it is the model's own name and version, and the entry's `reads_tables`."""
    found = sources.get(event.split(".")[0]) or {}
    bits = []
    if (
        found.get("method")
        and found.get("version")
        and (found["method"], found["version"])
        != (entry.get("model"), entry.get("version"))
    ):
        bits.append(
            f"Review source (the table the review page opens on): "
            f"`{found['method']}` {found['version']}."
        )
    reads = [_reads_item(item) for item in entry.get("reads_tables") or []]
    if reads:
        bits.append("Also reads: " + ", ".join(f"`{m}` {v}" for m, v in reads) + ".")
    return " ".join(bits)


def _newest(tables: list[Table]) -> Table | None:
    """The table with the highest version (`v2` before `v10`)."""
    return max(
        tables,
        key=lambda t: [int(n) for n in re.findall(r"\d+", t.version)],
        default=None,
    )


def mixed_sources(found: list[Table]) -> list[dict]:
    """Tables that are not one method's labels: their sidecar's `bins_from`
    ({method: bins}) says bins of several methods were merged into one (`detach_vote`
    v3: the vote, with divertor Thomson Te thresholded where the vote assessed
    nothing). Per table: the counts as the sidecar gives them, its merge rule, the
    other methods' own state rules (from their shipped tables' sidecars) and whether
    the rows can tell the methods apart (they cannot unless the table has an `attrs`
    column), so the card quotes what the tables say and nothing counted by hand."""
    out = []
    for table in found:
        parts = table.meta.get("bins_from")
        if not isinstance(parts, dict) or len(parts) < 2:
            continue
        counts = {str(k): int(v) for k, v in parts.items()}
        rule = table.meta.get("rule")
        components = {}
        for method in counts:
            part = _newest(
                [t for t in found if t.event == table.event and t.method == method]
            )
            own = part.meta.get("rule") if part else None
            own = own if isinstance(own, dict) else {}
            components[method] = {
                "version": part.version if part else None,
                "states": scrub_text(str(own["states"])) if own.get("states") else None,
                "rule": scrub_text("; ".join(str(v) for v in own.values())),
            }
        coverage = table.meta.get("coverage")
        out.append(
            {
                "dataset": table.dataset,
                "event": table.event,
                "method": table.method,
                "version": table.version,
                "tier": table.tier,
                "bins": sum(counts.values()),
                "bin_ms": table.meta.get("bin_ms"),
                "shots": coverage.get("shots") if isinstance(coverage, dict) else None,
                "bins_from": counts,
                "rule": scrub_text("; ".join(str(v) for v in rule.values()))
                if isinstance(rule, dict)
                else "",
                "components": components,
                "per_row_marker": "attrs" in pd.read_csv(table.path, nrows=0).columns,
            }
        )
    return out


def zero_exceptions(
    provenance: list[dict], tree: taxonomy.Taxonomy | None = None
) -> dict[str, dict]:
    """The events whose category 0 does not say assessed-absent: those `silver.ZERO`
    names (a regime or state table's 0 only says nothing was marked there, and the
    grid maps it to -1), those `tree` gives a default class (the owner's unmarked time
    reads as that class), and any other event a shipped table gives a 0 name other
    than `absent`. Per event: the owner's-save meaning (`silver.zero_text`; None where
    there is none) and the names its shipped tables give 0."""
    tree = taxonomy.Taxonomy() if tree is None else tree
    found = {
        event: {"owner": silver.zero_text(tree, event), "names": set()}
        for event in sorted({*silver.ZERO, *(e for e in tree.top if tree.default(e))})
    }
    for item in provenance:
        zero = json.loads(item["categories"]).get("0")
        if zero is None:
            continue
        if item["event"] in found:
            found[item["event"]]["names"].add(zero)
        elif zero != "absent":
            found[item["event"]] = {"owner": None, "names": {zero}}
    return {
        event: {"owner": item["owner"], "names": sorted(item["names"])}
        for event, item in sorted(found.items())
    }


def unshipped(
    shipped: set[str], skipped: list[dict], pointed: dict[str, str], runs=()
) -> tuple[dict[str, dict], dict[str, dict]]:
    """The tables a review pointer names or the grid takes frames from that the
    release does not hold: per dataset, and per directory (with the grid frames and
    shots they set in each split)."""
    why = {item["dataset"]: item["why"] for item in skipped}
    found: dict[str, dict] = {}

    def entry(dataset):
        return found.setdefault(
            dataset, {"why": why.get(dataset, "not a table the release reads")}
        )

    for where, dataset in pointed.items():
        if dataset not in shipped:
            entry(dataset).setdefault("pointers", []).append(where)
    groups: dict[str, dict] = {}
    for frame in runs:
        part = frame[frame.dataset.notna() & ~frame.dataset.isin(shipped)]
        for (dataset, split), rows_ in part.groupby(["dataset", "split"]):
            item = entry(dataset)
            item.update(tier=str(rows_.tier.iloc[0]), layer=str(rows_.layer.iloc[0]))
            item.setdefault("grid", {})[split] = {
                "frames": int(rows_.n_frames.sum()),
                "shots": int(rows_.shot.nunique()),
            }
        part = part.assign(group=part.dataset.str.rsplit("/", n=1).str[0])
        for (group, split), rows_ in part.groupby(["group", "split"]):
            groups.setdefault(group, {})[split] = {
                "tables": int(rows_.dataset.nunique()),
                "frames": int(rows_.n_frames.sum()),
                "shots": int(rows_.shot.nunique()),
            }
    return found, groups


def _unshipped_text(found: dict[str, dict], groups: dict[str, dict]) -> str:
    parts = [
        f"{group}/* ("
        + ", ".join(
            f"{v['tables']} tables set {v['frames']:,} {split} frames on {v['shots']} shots"
            for split, v in sorted(splits.items())
        )
        + ")"
        for group, splits in sorted(groups.items())
    ]
    parts += [
        f"{dataset} (the review pointer of {', '.join(item['pointers'])})"
        for dataset, item in sorted(found.items())
        if item.get("pointers")
    ]
    return "; ".join(parts)


def _check_target(path, paths: Paths, what: str) -> Path:
    """`path`, resolved, unless it is or holds the label tables, the labeler root, its
    suggestions, the repo or home, or lies inside the label tables or suggestions."""
    path = Path(path).resolve()
    tables_dir = Path(paths.label_tables).resolve()
    suggestions = (Path(paths.root) / "suggestions").resolve()
    for keep in (
        tables_dir,
        Path(paths.root).resolve(),
        suggestions,
        Path(__file__).resolve().parents[3],
        Path.home().resolve(),
    ):
        if keep.is_relative_to(path):
            raise ReleaseError(f"{what} {path} is or holds {keep}; refusing")
    for inside in (tables_dir, suggestions):
        if path.is_relative_to(inside):
            raise ReleaseError(f"{what} {path} lies inside {inside}; refusing")
    return path


def _replaceable(path: Path, *, partial: bool = False) -> None:
    """Refuse an existing `path` that is not a previous release (or empty); a
    `partial` one (a build that stopped) need not have its release.json yet."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        raise ReleaseError(f"{path} is not a release directory; refusing to replace it")
    names = {entry.name for entry in path.iterdir()}
    stray = sorted(names - RELEASE_ENTRIES)
    if stray or (names and not partial and "release.json" not in names):
        held = stray[:5] or ["no release.json"]
        raise ReleaseError(
            f"{path} is not a previous release (it holds {held}); refusing to replace it"
        )


def build(
    paths: Paths,
    out,
    *,
    silver_grid=None,
    gold_grid=None,
    work=None,
    exclude=(),
    include=(),
    allow_unshipped: bool = False,
    release: str = RELEASE,
) -> dict:
    """Write release `release` into `out` and verify it; return the report.

    `out` must be new, empty or a previous release, and neither the label tables,
    the labeler root, the repo, home nor anything holding them. The release is built
    in a hidden sibling and swapped in once it verifies; a release that does not
    verify stays in that sibling for inspection and `out` is left as it was. A
    split's grid is read from `silver_grid`/`gold_grid` when given (its shots must be
    that split's), else exported from the current tables into `work/<split>`. A table
    the grid takes frames from, or a review pointer names, that the release does not
    hold fails the build unless `allow_unshipped`, which flags it instead. `WITHHELD`
    tables stay out unless `include` names them; `exclude` leaves others out (an id
    that is no table of the labeler raises).
    """
    if not RELEASE_NAME.match(str(release)):
        raise ReleaseError(f"release {release!r} is not a name like v1 or v2.1")
    ctx = context(paths)
    out = _check_target(out, paths, "out")
    work = _check_target(
        work if work is not None else out.parent / f"{out.name}.work", paths, "work"
    )
    if work.is_relative_to(out):
        raise ReleaseError(f"work {work} lies inside out {out}; refusing")
    _replaceable(out)
    found, skipped = tables(paths, exclude=exclude, include=include)
    shipped = {table.dataset for table in found}
    pointed = pointers(paths)
    missing, _ = unshipped(shipped, skipped, pointed)
    if missing and not allow_unshipped:
        raise ReleaseError(
            "review pointers name tables the release does not hold: "
            + "; ".join(f"{d} ({v['why']})" for d, v in sorted(missing.items()))
        )
    stage = out.parent / f".{out.name}.building"
    _replaceable(stage, partial=True)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        report = _write(
            stage,
            ctx,
            found,
            skipped,
            pointed,
            silver_grid,
            gold_grid,
            work,
            release=release,
            held=withheld(include),
        )
    except NotClean:
        raise
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if report["unshipped"] and not allow_unshipped:
        shutil.rmtree(stage, ignore_errors=True)
        raise ReleaseError(
            "the grid takes frames from tables the release does not hold: "
            + _unshipped_text(report["unshipped"], report["unshipped_groups"])
            + "; give them a .meta.json in the interval schema, or build with "
            "allow_unshipped to flag them"
        )
    if out.exists():
        old = out.parent / f".{out.name}.old"
        _replaceable(old)
        if old.exists():
            shutil.rmtree(old)
        out.rename(old)
        stage.rename(out)
        shutil.rmtree(old)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(out)
    return report


def _write(
    out,
    ctx,
    found,
    skipped,
    pointed,
    silver_grid,
    gold_grid,
    work,
    *,
    release: str = RELEASE,
    held: dict | None = None,
) -> dict:
    """The release, written into the (empty) `out`."""
    paths = ctx.paths
    report = {
        "release": release,
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **_git(),
        "skipped": skipped,
        "withheld": {
            name: dict(item)
            for name, item in sorted((held or {}).items())
            if any(entry["dataset"] == name for entry in skipped)
        },
        "tables": {},
        "dropped_attrs": {},
    }

    frames, provenance = [], []
    for table in found:
        frame, scrub = rows(table, ctx)
        frames.append(frame)
        report["tables"][table.dataset] = {"rows": len(frame), **scrub}
        for key in scrub["dropped_attrs"]:
            report["dropped_attrs"][key] = report["dropped_attrs"].get(key, 0) + 1
        provenance.append(
            {
                "dataset": table.dataset,
                "event": table.event,
                "tier": table.tier,
                "layer": table.layer,
                "method": table.method,
                "version": table.version,
                "made_at": str(table.meta.get("made_at") or ""),
                "made_by": scrub_text(str(table.meta.get("made_by") or "")),
                "n_rows": len(frame),
                "n_shots": int(frame.shot.nunique()),
                "categories": json.dumps(
                    _categories(table, frame.category.unique()), sort_keys=True
                ),
                "shipped": True,
                "why": "",
            }
        )
    report["mixed_sources"] = mixed_sources(found)
    report["zero_exceptions"] = zero_exceptions(
        provenance, taxonomy.load(paths.events_manifest)
    )
    report["review_sources"] = review_sources(pointed, provenance)
    spans = pd.concat(frames, ignore_index=True).sort_values(
        ["event", "dataset", "shot", "t_start_ms", "t_end_ms", "category"],
        kind="stable",
    )
    counts = {}
    for tier in TIER_ORDER:
        for split in SPLITS:
            part = spans[(spans.tier == tier) & (spans.split == split)]
            if len(part):
                _write_parquet(part, out / "labels" / tier / f"{split}.parquet")
                counts[f"{tier}/{split}"] = {
                    "rows": len(part),
                    "shots": int(part.shot.nunique()),
                    "events": int(part.event.nunique()),
                }

    grids, every_run = {}, []
    for name, shots, given in (
        ("silver", ctx.silver, silver_grid),
        ("gold", ctx.gold, gold_grid),
    ):
        if not shots:
            continue
        where = (
            Path(given)
            if given is not None
            else export_grid(paths, sorted(shots), work / name, ctx)
        )
        index = json.loads((where / "index.json").read_text())
        if {int(s) for s in index["shots"]} != set(shots):
            raise ReleaseError(
                f"the {name} grid at {where} is not the {name} shot list"
            )
        runs = grid_rle(where, paths)
        runs["split"] = runs.shot.map(ctx.split).astype("string")
        every_run.append(runs)
        for split, part in runs.groupby("split", sort=False):
            stem = GRID_FILES[split]
            _write_parquet(part, out / "grid" / f"{stem}.parquet")
            grids[stem] = {
                "split": split,
                "shots": int(part.shot.nunique()),
                "runs": len(part),
                "events": index["events"],
                "made_at": index.get("made_at"),
                "assessed_frames": int(part.loc[part.category >= 0, "n_frames"].sum()),
                "frames": int(part.n_frames.sum()),
            }
    missing, groups = unshipped(
        {table.dataset for table in found}, skipped, pointed, every_run
    )
    report["unshipped"], report["unshipped_groups"] = missing, groups
    painted = sorted(d for d in held or {} if missing.get(d, {}).get("grid"))
    if painted:  # its content would ship in the grid; allow_unshipped cannot cover it
        raise ReleaseError(
            f"the grid takes frames from withheld tables {painted}; include them by "
            "name or keep their shots out of the silver and gold lists"
        )
    taken = (
        pd.concat(every_run).groupby("dataset").n_frames.sum().to_dict()
        if every_run
        else {}
    )
    for item in provenance:
        item["grid_frames"] = int(taken.get(item["dataset"], 0))
    for dataset, item in sorted(missing.items()):
        provenance.append(
            {
                "dataset": dataset,
                "event": None
                if dataset.startswith("suggestions/")
                else dataset.split("/")[0],
                "tier": item.get("tier"),
                "layer": item.get("layer"),
                "shipped": False,
                "why": item["why"],
                "grid_frames": int(taken.get(dataset, 0)),
            }
        )
    _write_parquet(
        pd.DataFrame(provenance).astype({"n_rows": "Int64", "n_shots": "Int64"}),
        out / "datasets.parquet",
    )

    report["shots"] = _write_shots(out / "shots", ctx)

    signals = _signals(paths)
    check_reads_tables(signals, provenance)
    (out / "signals.yaml").write_text(
        "# Event -> best model -> input signals -> score (from best_models.yaml).\n#\n"
        + scrub_text(
            _header(Path(paths.best_models))
            + yaml.safe_dump(signals, sort_keys=False, allow_unicode=True, width=100)
        )
    )
    event_ids = sorted(
        set(spans.event) | {e for g in grids.values() for e in g["events"]}
    )
    tax = _taxonomy(paths, event_ids, release)
    (out / "taxonomy.yaml").write_text(
        yaml.safe_dump(tax, sort_keys=False, allow_unicode=True, width=100)
    )
    report["mirror"] = mirror(out, found, ctx)
    by_event = {}
    for events_ in ctx.holdout.values():
        for event in events_:
            by_event[event] = by_event.get(event, 0) + 1
    report.update(
        {
            "labels": counts,
            "grids": grids,
            "pending": pending(signals),
            "withheld_blind_rows": sum(
                t["withheld_rows"] for t in report["tables"].values()
            ),
            "blind_shots_withheld": len(ctx.blind),
            "gold_shots": len(ctx.gold),
            "holdout": {"shots": len(ctx.holdout), "by_event": by_event},
        }
    )
    (out / "README.md").write_text(card(report, signals, tax))
    files = _files(out)
    (out / "release.json").write_text(
        json.dumps(
            {
                "release": release,
                "repo": REPO_ID,
                "made_at": report["made_at"],
                "git_sha": report["git_sha"],
                "git_dirty": report["git_dirty"],
                "labels": counts,
                "grids": {
                    k: {kk: vv for kk, vv in v.items() if kk != "events"}
                    for k, v in grids.items()
                },
                "unshipped": sorted(missing),
                "withheld": sorted(report["withheld"]),
                "files": files,
            },
            indent=1,
        )
        + "\n"
    )
    every = [path for path in out.rglob("*") if path.is_file()]
    report["bytes"] = sum(path.stat().st_size for path in every)
    report["n_files"] = len(every)
    problems = verify(out, ctx)
    report["problems"] = problems
    if problems:
        raise NotClean(f"{len(problems)} problems in {out} (kept): {problems[:5]}")
    return report


def _write_shots(shots_dir: Path, ctx: Context) -> dict:
    """The shot lists, one file (and one card config) per schema; returns what each
    holds. A missing gold list writes no gold file."""
    written = {}

    def put(stem, frame):
        _write_parquet(frame.astype({"shot": "int32"}), shots_dir / f"{stem}.parquet")
        written[stem] = {"shots": len(frame), "columns": list(frame.columns)}

    put(
        "silver",
        pd.DataFrame(
            {
                "shot": sorted(ctx.silver),
                "set": "silver",
                "split": [ctx.split(s) for s in sorted(ctx.silver)],
            }
        ),
    )
    gold_file = Path(ctx.paths.label_tables) / "gold" / "gold_shots.csv"
    if gold_file.is_file():
        gold_list = pd.read_csv(gold_file)
        frame = pd.DataFrame({"shot": gold_list.shot, "set": "gold", "split": "test"})
        if "run_day" in gold_list:
            frame["run_day"] = gold_list.run_day.astype("int32")
        put("gold", frame)
    if ctx.holdout:
        put(
            "holdout",
            pd.DataFrame(
                {
                    "shot": list(ctx.holdout),
                    "set": "holdout",
                    "split": [ctx.split(s) for s in ctx.holdout],
                    "events": [";".join(e) for e in ctx.holdout.values()],
                    "silver": [s in ctx.silver for s in ctx.holdout],
                }
            ),
        )
    # The catalog's own split, except where the release decides: a gold shot is test
    # and a roster holdout is holdout here too, so no file offers either for training.
    split = ctx.cohort[~ctx.cohort.blind][["shot", "split"]].copy()
    ours = split.shot.map(ctx.split)
    split["split"] = ours.where(ours != "train", split.split)
    split["gold"] = split.shot.isin(ctx.gold)
    split["holdout"] = split.shot.isin(set(ctx.holdout))
    put("cohort_split", split.sort_values("shot"))
    written["cohort_split"].update(
        splits=split.split.value_counts().to_dict(), gold=int(split.gold.sum())
    )
    return written


def _header(path: Path) -> str:
    """The comment block a YAML file opens with (its field documentation)."""
    lines = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        if not line.startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines) + "\n\n" if lines else ""


def _files(folder: Path) -> dict[str, dict]:
    return {
        path.relative_to(folder).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != "release.json"
    }


# ---- checks ----


def front_matter(text: str) -> dict:
    """The YAML block a dataset card opens with."""
    if not text.startswith("---\n"):
        raise ReleaseError("the card has no YAML front matter")
    block, _, _ = text[4:].partition("\n---\n")
    return yaml.safe_load(block) or {}


def wrong_splits(shots, splits, ctx: Context) -> list[int]:
    """Shots whose split breaks the rule: a gold shot is always `test`, a roster
    holdout always `holdout` (unless gold), and no other shot is either. Other values
    (the catalog cohort's `val`) are left alone."""
    wrong = set()
    for shot, split in zip(shots, splits, strict=True):
        expected = ctx.split(shot)
        if (split in ("test", "holdout") or expected != "train") and split != expected:
            wrong.add(int(shot))
    return sorted(wrong)


def verify(folder, ctx: Context, *, max_bytes: int = MAX_BYTES) -> list[str]:
    """Everything wrong with a built release (an empty list when it is clean)."""
    folder = Path(folder)
    problems, total = [], 0
    names = [s for s in ctx.secrets if not s.isdigit()]
    words = (
        re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE)
        if names
        else None
    )
    blind = {str(s) for s in ctx.blind}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder).as_posix()
        total += path.stat().st_size
        if (
            path.suffix in NEVER_SHIP
            or "/raw/" in f"/{relative}"
            or path.name == "history.jsonl"
        ):
            problems.append(f"{relative}: never shipped")
            continue
        if path.suffix == ".parquet":
            table = pq.read_table(path)
            texts = []
            for column in table.column_names:
                values = table.column(column).to_pylist()
                if column == "shot":
                    leaked = {int(v) for v in values if v is not None} & ctx.blind
                    if leaked:
                        problems.append(f"{relative}: blind shots {sorted(leaked)[:5]}")
                elif table.schema.field(column).type in (
                    "string",
                    "large_string",
                ) or isinstance(next((v for v in values if v is not None), None), str):
                    texts.extend(v for v in values if isinstance(v, str))
            text = "\n".join(texts)
            if {"shot", "split"} <= set(table.column_names):
                wrong = wrong_splits(
                    table.column("shot").to_pylist(),
                    table.column("split").to_pylist(),
                    ctx,
                )
                if wrong:
                    problems.append(f"{relative}: shots in the wrong split {wrong[:5]}")
        else:
            text = path.read_text(errors="replace")
            header = text.split("\n", 1)[0].split(",")
            if path.suffix == ".csv" and "shot" in header:
                shots = pd.read_csv(path, usecols=["shot"], dtype=str).shot
                found = pd.to_numeric(shots, errors="coerce").dropna().astype(int)
                leaked = set(found) & ctx.blind
                if leaked:
                    problems.append(f"{relative}: blind shots {sorted(leaked)[:5]}")
        if FORBIDDEN_PATH.search(text):
            problems.append(
                f"{relative}: absolute path {FORBIDDEN_PATH.search(text).group(0)}"
            )
        if words is not None and words.search(text):
            problems.append(f"{relative}: names a login or reviewer")
        if PLACEHOLDER_TEXT.search(text):
            problems.append(f"{relative}: a roster placeholder row")
        leaked = blind.intersection(SHOT_TOKEN.findall(HASH.sub("", text)))
        if leaked:
            problems.append(f"{relative}: blind shot numbers {sorted(leaked)[:5]}")
    if total > max_bytes:
        problems.append(f"{total} bytes, above {max_bytes}")
    named = (
        set(pd.read_parquet(folder / "datasets.parquet", columns=["dataset"]).dataset)
        if (folder / "datasets.parquet").is_file()
        else set()
    )
    try:
        held = json.loads((folder / "release.json").read_text()).get("withheld") or []
    except (OSError, ValueError):
        held = []
    for name in held:
        if name in named or (folder / _mirror_path(name)).is_file():
            problems.append(f"{name}: withheld, yet in the release")
    for path in sorted((folder / "grid").glob("*.parquet")):
        unknown = (
            set(pd.read_parquet(path, columns=["dataset"]).dataset.dropna()) - named
        )
        if unknown:
            problems.append(
                f"grid/{path.name}: datasets missing from datasets.parquet "
                f"{sorted(unknown)[:3]}"
            )
    card_path = folder / "README.md"
    if not card_path.is_file():
        problems.append("no README.md card")
    else:
        try:
            meta = front_matter(card_path.read_text())
        except (ReleaseError, yaml.YAMLError) as error:
            problems.append(f"README.md: {error}")
            meta = {}
        for config in meta.get("configs") or []:
            for item in config.get("data_files") or []:
                for path in (
                    [item["path"]] if isinstance(item["path"], str) else item["path"]
                ):
                    if not (folder / path).is_file():
                        problems.append(
                            f"README.md: config {config['config_name']} names "
                            f"missing {path}"
                        )
    return problems


# ---- the card ----


#: How the card names a score key of best_models.yaml.
SCORE_NAMES = {
    "phase_controlled_auroc": "phase-controlled AUROC",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1",
}


def _score(value) -> str:
    if not isinstance(value, dict) or value.get("value") is None:
        return "-"
    if value.get("lo") is None:
        return f"{value['value']:.3f}"
    return f"{value['value']:.3f} [{value['lo']:.3f}, {value['hi']:.3f}]"


def _scores(scores) -> str:
    """Every scalar score of an entry, named, with its interval when it has one."""
    if not isinstance(scores, dict):
        return ""
    return ", ".join(
        f"{SCORE_NAMES.get(key, key)} {_score(value)}"
        for key, value in scores.items()
        if isinstance(value, dict) and value.get("value") is not None
    )


def _size_category(n: int) -> str:
    for bound, name in (
        (1_000, "n<1K"),
        (10_000, "1K<n<10K"),
        (100_000, "10K<n<100K"),
        (1_000_000, "100K<n<1M"),
    ):
        if n < bound:
            return name
    return "1M<n<10M"


def _configs(report: dict) -> list[dict]:
    configs = []
    for tier in TIER_ORDER:
        files = [
            {"split": split, "path": f"labels/{tier}/{split}.parquet"}
            for split in SPLITS
            if f"{tier}/{split}" in report["labels"]
        ]
        if files:
            configs.append({"config_name": tier, "data_files": files})
    if report["grids"]:
        configs.append(
            {
                "config_name": "grid",
                "data_files": [
                    {"split": split, "path": f"grid/{GRID_FILES[split]}.parquet"}
                    for split in SPLITS
                    if GRID_FILES[split] in report["grids"]
                ],
            }
        )
    # One config per shot list: their columns differ, and a config's files must
    # share one schema.
    for stem in report["shots"]:
        name = stem if stem == "cohort_split" else f"shots_{stem}"
        configs.append(
            {
                "config_name": name,
                "data_files": [{"split": stem, "path": f"shots/{stem}.parquet"}],
            }
        )
    configs.append(
        {
            "config_name": "datasets",
            "data_files": [{"split": "train", "path": "datasets.parquet"}],
        }
    )
    configs[0]["default"] = True
    return configs


def _mixed_text(item: dict, *, rule: bool = False) -> str:
    """What the card says of a merged source table (`mixed_sources`): its counts per
    method, each method's state rule where its table is shipped, and that its rows
    do not say which method set a bin. `rule` adds the merge rule."""
    split = []
    for method, n in sorted(item["bins_from"].items(), key=lambda kv: (-kv[1], kv[0])):
        part = item["components"].get(method) or {}
        rule_text = f" ({part['states']})" if part.get("states") else ""
        split.append(f"{n:,} from `{method}`{rule_text}")
    wide = isinstance(item.get("bin_ms"), int | float) and item["bin_ms"]
    width = f"{item['bin_ms']:g} ms " if wide else ""
    bins = f"{item['bins']:,} {width}bins"
    shots = f" on {item['shots']:,} shots" if item.get("shots") else ""
    text = f"of its {bins}{shots}, " + "; ".join(split)
    if not item["per_row_marker"]:
        text += (
            "; its rows carry no per-row marker of which method set a bin, so the "
            "methods cannot be told apart afterwards"
        )
    if rule and item.get("rule"):
        text += f". Merge rule, from its sidecar: {item['rule']}"
    if rule:
        for method, part in sorted(item["components"].items()):
            if method != item["method"] and part.get("rule"):
                text += (
                    f". `{method}` {part['version']}, from its sidecar: {part['rule']}"
                )
    return text


def _zero_text(exceptions: dict[str, dict]) -> str:
    """The card's account of category 0: absent only where the table assessed the
    event, with the events whose 0 only says nothing was marked (`zero_exceptions`)."""
    each = []
    for event, item in exceptions.items():
        bits = []
        if item["names"]:
            bits.append(
                "its tables name 0 " + " or ".join(f"`{n}`" for n in item["names"])
            )
        if item["owner"]:
            bits.append(item["owner"])
        each.append(f"`{event}` ({'; '.join(bits)})")
    text = (
        "Category 0 rows are assessed-absent time (the table looked for the event and "
        "saw none) only where `category_name` says `absent`"
    )
    if each:
        text += (
            ", and not even then for the events whose 0 only says nothing was marked: "
            + "; ".join(each)
            + ". The 10 ms grid already maps those 0s to -1 (unassessed) in every "
            "table but the owner's save, where the unmarked time inside its window is "
            "0 (no regime or state saved) or, for an event with a default class, that "
            "class"
        )
    return (
        text + ". Time no row covers was not assessed by that table, whatever its "
        "category 0 means."
    )


def card(report: dict, signals: dict, tax: dict) -> str:
    """The dataset card: front matter for the Hub, then what the release holds."""
    head = {
        "pretty_name": "Tokamak-SI: DIII-D event labels",
        "tags": ["fusion", "tokamak", "diii-d", "time-series", "event-detection"],
        "size_categories": [
            _size_category(sum(v["rows"] for v in report["labels"].values()))
        ],
        "configs": _configs(report),
    }
    lines = ["---", yaml.safe_dump(head, sort_keys=False).rstrip(), "---", ""]
    add = lines.append
    add("# Tokamak-SI: DIII-D event labels")
    add("")
    add(
        "Time-interval labels of plasma events on DIII-D discharges: confinement regime, "
        "ELMs, Alfvén eigenmodes, MHD modes by toroidal number n (and poloidal m inside "
        "them), q_min band, poloidal beta, detachment, resistive wall modes and others. "
        "Labels only: no signals, spectrograms, model weights or third-party raw files. "
        f"Release {report['release']}, made {report['made_at']} from code "
        f"{report['git_sha']}"
        f"{' (uncommitted changes)' if report['git_dirty'] else ''}."
    )
    add("")
    add("## Files")
    add("")
    add("| Path | What |")
    add("|---|---|")
    add(
        "| `labels/<tier>/{train,holdout,test}.parquet` | one row per span, as each table "
        "wrote it |"
    )
    add(
        "| `grid/{silver,holdout,gold}.parquet` | the 10 ms event x frame grid, "
        "run-length encoded |"
    )
    add(
        "| `shots/*.parquet` | the silver, gold and holdout shot lists and the catalog "
        "cohort split (one config each: `shots_silver`, `shots_gold`, `shots_holdout`, "
        "`cohort_split`) |"
    )
    add(
        "| `datasets.parquet` | one row per source table: tier, method, version, rows, "
        "shots, grid frames, `shipped` |"
    )
    add(
        "| `signals.yaml` | the best model per event: inputs (MDSplus/PTDATA pointnames), score |"
    )
    add(
        "| `taxonomy.yaml` | events, subcategories, category names, the grid rule, provenance |"
    )
    add(
        "| `tables/` | the label tables themselves, scrubbed, for `hf_publish.py pull` |"
    )
    add("| `release.json` | counts and the sha256 of every file |")
    add("")
    add("## Tiers")
    add("")
    for tier in TIER_ORDER:
        add(f"- **{tier}**: {TIER_TEXT[tier]}.")
    for item in report["mixed_sources"]:
        add(
            f"- **Not a label set**: `{item['dataset']}` ({item['event']}, tier "
            f"{item['tier']}) merges the bins of more than one method: "
            + _mixed_text(item, rule=True)
            + "."
        )
    add("")
    add("| Tier | Split | Rows | Shots | Events |")
    add("|---|---|---:|---:|---:|")
    for key, value in report["labels"].items():
        tier, split = key.split("/")
        add(
            f"| {tier} | {split} | {value['rows']:,} | {value['shots']:,} | {value['events']} |"
        )
    add("")
    add(
        "Columns: `shot`, `event` (a subcategory is `<event>.<key>`, e.g. `mhd_n1.m2`), "
        "`category` and `category_name` (the table's own map; see `taxonomy.yaml`), "
        "`t_start_ms`, `t_end_ms` (ms from the DIII-D t=0), `confidence` (blank when the "
        "table gives none), `iscrowd` (true a span of activity, false one countable event, "
        "null unspecified), `attrs` (JSON, whitelisted keys), `tier`, `layer`, `dataset`, "
        "`method`, `version`, `reviewer` (reviewed rows: R1..Rn, null where the save "
        "recorded no name) and `split`. " + _zero_text(report["zero_exceptions"])
    )
    add("")
    add("## Splits")
    add("")
    add(
        "Splits are per shot and the same in every file: the labels, the grid and the "
        "shot lists."
    )
    add("")
    add(
        f"- **test**: the {report['gold_shots']} "
        "gold shots, held out for evaluation. Do not train or tune on them."
    )
    held = report["holdout"]
    if held["shots"]:
        add(
            f"- **holdout**: the {held['shots']} shots an event roster reserves for final "
            "evaluation only (`holdout` true in `tables/<event>/shots.csv`: "
            + ", ".join(f"`{e}` {n}" for e, n in sorted(held["by_event"].items()))
            + "). A shot any roster holds out is held out for every event. Do not train "
            "or tune on them; `shots/holdout.parquet` lists them with the events that "
            "hold them out."
        )
    add("- **train**: every other shot (the silver review set and the wider tables).")
    cohort = report["shots"]["cohort_split"]
    add(
        f"- The DIII-D event catalog's {report['blind_shots_withheld']} blind test shots are "
        "withheld from every file of this release, labels and shot lists alike. "
        "`shots/cohort_split.parquet` lists the catalog cohort's other shots with the "
        "catalog's own split, except that a gold shot is `test` and a roster holdout "
        f"`holdout` there too ({cohort['gold']} gold shots are in the cohort; "
        f"columns `gold`, `holdout`): "
        f"{', '.join(f'{k} {v}' for k, v in sorted(cohort['splits'].items()))}. "
        "The catalog's `val` shots are in the release's train split elsewhere."
    )
    add("")
    add("## The 10 ms grid")
    add("")
    for name, value in report["grids"].items():
        add(
            f"- `grid/{name}.parquet` (split {value['split']}): {value['shots']} shots, "
            f"{len(value['events'])} events, {value['frames']:,} event-frames of which "
            f"{value['assessed_frames']:,} assessed; grid made {value['made_at']}."
        )
    add("")
    add(
        "Each row is a run of frames of one event with one category from one table: "
        "`frame` (frame k covers [10k, 10k+10) ms), `n_frames`, `t_start_ms`, `t_end_ms`, "
        "`category` (-1 unassessed), and the `tier`, `layer` and `dataset` that set it. "
        f"Rule: {tax['grid']['rule']}."
    )
    add("")
    add("```python")
    add("import numpy as np, pandas as pd")
    add("runs = pd.read_parquet('grid/gold.parquet')")
    add("one = runs[runs.shot == runs.shot.iloc[0]]")
    add("events = list(dict.fromkeys(one.event)); f0 = one.frame.min()")
    add(
        "grid = np.full((len(events), (one.frame + one.n_frames).max() - f0), -1, np.int8)"
    )
    add("for r in one.itertuples():")
    add(
        "    grid[events.index(r.event), r.frame - f0:r.frame - f0 + r.n_frames] = r.category"
    )
    add("```")
    add("")
    if report["unshipped"]:
        add("### Not in this release")
        add("")
        add(
            "The grid takes frames from these tables (or a review page opens on them), "
            "but the release does not hold them, so `pull` cannot rebuild those frames. "
            "`datasets.parquet` lists each with `shipped` false and why: "
            + _unshipped_text(report["unshipped"], report["unshipped_groups"])
            + "."
        )
        add("")
    if report.get("withheld"):
        add("## Withheld tables")
        add("")
        add(
            "These tables are read by the labeler but are not in this release (a build "
            "includes one only by name, once its owners allow redistribution):"
        )
        add("")
        for name, item in report["withheld"].items():
            add(f"- `{name}`: {item['why']}. To get it, {item['get']}.")
        add("")
    add("## Events and categories")
    add("")
    add("| Event | Name | Categories |")
    add("|---|---|---|")
    for event, entry in tax["events"].items():
        cats = ", ".join(f"{k} {v}" for k, v in entry["categories"].items())
        fold = entry.get("folded_into", {}).get("event")
        name = entry["display"] + (f" (folded into {fold})" if fold else "")
        add(f"| `{event}` | {name} | {cats} |")
    add("")
    add("## Best model per event")
    add("")
    add(
        "From `signals.yaml`, which lists each model's input signals with their MDSplus or "
        "PTDATA pointnames and rates, the reference each score is against, its interval "
        "and its caveats. Quote the headline: several scores are agreement with a rule, "
        "or show no skill, and are not a detector's accuracy."
    )
    add("")
    for event, entry in (signals.get("events") or {}).items():
        if not isinstance(entry, dict) or not entry.get("model"):
            continue
        scores = entry.get("scores") if isinstance(entry.get("scores"), dict) else {}
        flag = " (pending)" if entry.get("todo") else ""
        bins = f", {entry['bin_ms']} ms bins" if entry.get("bin_ms") else ""
        shots = f", {entry['n_shots']} shots" if entry.get("n_shots") else ""
        version = entry.get("version") or ""
        text = [
            f"- `{event}`: {entry['model']} {version}{flag} "
            + f"({entry.get('task')}{bins}{shots})."
        ]
        if entry.get("headline"):
            text.append(f"**{entry['headline']}.**")
        if scores.get("reference"):
            text.append(f"Against: {scores['reference']}.")
        if _scores(scores):
            interval = scores.get("interval")
            text.append(
                f"Scores: {_scores(scores)}"
                + (f" (brackets: {interval})" if interval else "")
                + "."
            )
        caveats = entry.get("caveats") or []
        if caveats:
            text.append(f"Caveat: {str(caveats[0]).rstrip('.')}.")
        for item in report["mixed_sources"]:
            if item["event"] == event:
                text.append(
                    f"The table that ships for this event, `{item['method']}` "
                    f"{item['version']}, is not this model's label set: "
                    + _mixed_text(item)
                    + "."
                )
        source = _source_text(entry, event, report.get("review_sources") or {})
        if source:
            text.append(source)
        add(" ".join(text))
    if report["pending"]:
        add("")
        add(
            "Pending (still being changed when this release was built): "
            + "; ".join(f"`{e}`" for e in report["pending"])
            + "."
        )
    add("")
    add("## Provenance")
    add("")
    add(
        "The `original` tier reformats these third-party tables; their owners' terms apply."
    )
    add("")
    for item in tax["third_party_sources"]:
        add(f"- `{item['stem']}`: {item['provenance']}")
    add("")
    add("## Scrubbing")
    add("")
    add(
        f"Withheld: every row on a blind cohort shot ({report['withheld_blind_rows']:,} rows "
        "across the tables). Reviewer identities are R1..Rn in the order of their first save. "
        "`attrs` keep only the keys listed in `taxonomy.yaml`. No file names a path on the "
        "storage the tables were made on; tables are named by `dataset` id. Review "
        "histories, third-party raw files and per-shot grids are not in the release."
    )
    add("")
    add("## License")
    add("")
    add(
        "None yet: no license has been chosen. The repository is private and shared "
        "with named collaborators only; do not redistribute it. The original tier "
        "carries third-party labels under their owners' terms (see Provenance); "
        "DIII-D data-use rules apply to everything here."
    )
    add("")
    add("## Use")
    add("")
    add("```python")
    add("from datasets import load_dataset")
    add(
        f"original = load_dataset('{REPO_ID}', 'original')"
        "   # splits: train, holdout, test (gold)"
    )
    add(f"grid = load_dataset('{REPO_ID}', 'grid', split='test')")
    add("```")
    add("")
    add(
        "To put the label tables back where the labeler reads them, on a clone without "
        f"them: `python scripts/labeler/hf_publish.py pull --repo {REPO_ID}`, or "
        "`pull --into DIR` for a fresh directory. `pull` never replaces a file or a "
        "review pointer that differs."
    )
    add("")
    return "\n".join(lines)


# ---- the Hub ----


def upload(
    folder,
    ctx: Context,
    *,
    repo_id: str = REPO_ID,
    api=None,
    allow_pending: bool = False,
    allow_unshipped: bool = False,
    message: str | None = None,
) -> dict:
    """Send a verified release to a PRIVATE dataset repo and tag it with its release.

    Refuses a release that does not verify, one whose signals still carry `todo`
    markers (unless `allow_pending`), one whose grid or review pointers use tables it
    does not hold (unless `allow_unshipped`), one whose release name the repo already
    has as a tag (rebuild with the next `release`), and any repo that is or turns out
    public.
    """
    folder = Path(folder)
    problems = verify(folder, ctx)
    if problems:
        raise ReleaseError(f"{len(problems)} problems in {folder}: {problems[:5]}")
    marked = pending(yaml.safe_load((folder / "signals.yaml").read_text()) or {})
    if marked and not allow_pending:
        raise ReleaseError(f"signals.yaml still has todo markers: {sorted(marked)}")
    datasets = pd.read_parquet(folder / "datasets.parquet")
    missing = sorted(datasets.dataset[~datasets.shipped.astype(bool)])
    if missing and not allow_unshipped:
        raise ReleaseError(
            f"{len(missing)} tables the grid or a review pointer uses are not in the "
            f"release, e.g. {missing[:3]}"
        )
    release = json.loads((folder / "release.json").read_text())["release"]
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    exists = api.repo_exists(repo_id, repo_type="dataset")
    if exists and not api.repo_info(repo_id, repo_type="dataset").private:
        raise ReleaseError(f"{repo_id} is public; refusing to upload")
    if exists and release in _tags(api, repo_id):
        raise ReleaseError(
            f"{repo_id} already has release {release}; build the next one "
            "(`build --release`) instead of moving main under that tag"
        )
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise ReleaseError(f"{repo_id} is not private; refusing to upload")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        commit_message=message or f"release {release}",
        delete_patterns="*",
    )
    api.create_tag(
        repo_id,
        repo_type="dataset",
        tag=release,
        revision=getattr(commit, "oid", None),
    )
    return {"repo": repo_id, "commit": getattr(commit, "oid", None), "tag": release}


def _tags(api, repo_id: str) -> set[str]:
    return {ref.name for ref in api.list_repo_refs(repo_id, repo_type="dataset").tags}


def download(repo_id: str, dest_root, *, revision: str | None = None, api=None) -> Path:
    """The `tables/` of a release (and its card and manifest) into a directory of its
    own, `dest_root/<commit>`. A directory left by an earlier pull of the same commit
    is reused only when `check_files` finds it exactly the release; otherwise it is
    refused, so a file deleted upstream is never restored from a stale download."""
    from huggingface_hub import HfApi, snapshot_download

    api = api or HfApi()
    commit = api.dataset_info(repo_id, revision=revision).sha
    dest = Path(dest_root) / commit
    if dest.is_dir() and any(dest.iterdir()):
        problems = check_files(dest)
        if problems:
            raise ReleaseError(
                f"{dest} holds an earlier download that is not this release "
                f"({problems[:3]}); remove it first"
            )
        return dest
    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=commit,
            local_dir=str(dest),
            allow_patterns=["tables/**", "README.md", "release.json"],
        )
    )
