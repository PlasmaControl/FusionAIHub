"""The review page's signal catalog: `configs/labeler/signals.yaml`, validated, and
the free ids a client may send besides (plan sections B and D.A).

An id the page sends is one of three things, and anything else is `Invalid` before
any file is read or anything is fetched:

- a catalog id (`ID`), one entry of signals.yaml;
- `custom:<NAME>`, a PTDATA point name (`PTDATA`);
- `mds:<TREE>:<node>`, a free MDSplus node. TREE is on `TREES`, and the node names
  it (`\\TREE::...`). The node matches plan B's pattern, tightened: no parentheses,
  quotes, commas, spaces or `$`, no segment starting with `_` or a digit, at most
  `NODE_MAX` characters, and no `DIM_OF` or `PTDATA2` segment.

No client text reaches MDSplus but a node that passed `node_problem`. The time axis
comes from toksearch's own dimension mechanism, so `dim_of(...)` and PTDATA2 are
never built from what a client typed.

An entry's `sources` are tried in order: local tiers first (the corpus, the raw
cache, the features store, the MHD park), then DIII-D through the fetch worker
(`ptdata`, `mds`). A profile's sources are `recipe`s, the fixed readers in
`review.profiles`. Each source may scale its values into the entry's units, and may
carry a `tag` (the EFIT tree, say) that the page shows after the label.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ...config import store_event

#: The catalog file in the config directory (`Paths.config_dir`).
CATALOG_FILE = "signals.yaml"

ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
PTDATA = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,11}$")
#: Plan B's node pattern; `node_problem` tightens it (D.A).
NODE = re.compile(r"^\\?[A-Za-z0-9_]+(::|\.)?[A-Za-z0-9_:.]*$")
NODE_MAX = 96
#: The tightened rule as one expression the page can test (`/api/catalog`'s
#: `node_pattern`); the tree match, `NODE_MAX` and `REFUSED_SEGMENTS` are the
#: server's alone.
NODE_PATTERN = (r"^\\?[A-Za-z][A-Za-z0-9_]*::[A-Za-z][A-Za-z0-9_]*"
                r"((::|[.:])[A-Za-z][A-Za-z0-9_]*)*$")
NODE_STRICT = re.compile(NODE_PATTERN)
#: One segment of a node path: a letter first, never `_` or a digit.
SEGMENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
#: Segments refused outright, case-insensitively, though they could never call
#: anything without parentheses: belt and braces for "never sent".
REFUSED_SEGMENTS = ("DIM_OF", "PTDATA2")
#: The MDSplus trees a node may come from (plan B).
TREES = (
    "EFIT01", "EFIT02", "EFIT02ER", "MHD", "SPECTROSCOPY", "ELECTRONS", "IONS", "D3D",
    "ZIPFIT01", "ZIPFIT02", "ECE", "BOLOM",
)
CUSTOM_PREFIX = "custom:"
MDS_PREFIX = "mds:"

KINDS = ("trace", "profile")
#: A profile's x axis (the `/api/profile` header's `x_kind`).
X_KINDS = ("psi_n", "rho", "R_m", "channel")
DRAWS = ("lines", "map")
#: Source kinds read on this machine; the others go through the fetch worker.
LOCAL = ("corpus", "raw", "feature", "park", "recipe")
FETCHED = ("ptdata", "mds")
#: Park traces: the A-EQDSK scalars `mhd/efit02` (or `efit_fallback`) holds.
PARK_TRACES = ("a_q95", "a_qmin")
#: The profile readers in `review.profiles`, each with the `source` the page shows.
RECIPES = {
    "park_q": "park",
    "efit01_q": "mds",
    "thomson_te": "corpus",
    "thomson_ne": "corpus",
    "zipfit_rotation": "park",
    "ece_te": "corpus",
}
#: A recipe that fetches (the EFIT01 q fallback) runs under the long limit.
FETCHING_RECIPES = ("efit01_q",)
#: What the contract calls each source kind (`source` in the catalog and headers):
#: the features store is a local cache like the raw cache.
WIRE = {"corpus": "corpus", "raw": "raw", "feature": "raw", "park": "park",
        "ptdata": "ptdata", "mds": "mds"}
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class CatalogError(ValueError):
    """signals.yaml is malformed; the message names the entry."""


class Invalid(ValueError):
    """An id the server will not resolve: answered 422 `invalid`, never fetched."""


@dataclass(frozen=True)
class Source:
    """One place an entry's values come from."""

    kind: str
    name: str
    tree: str | None = None
    rows: tuple[int, ...] | None = None
    total: bool = False
    scale: float = 1.0
    tag: str = ""
    free: bool = False

    @property
    def local(self) -> bool:
        return self.kind in LOCAL

    @property
    def wire(self) -> str:
        """The contract's name for this source: corpus, raw, park, mds or ptdata."""
        return RECIPES[self.name] if self.kind == "recipe" else WIRE[self.kind]

    @property
    def fetches(self) -> bool:
        return self.kind in FETCHED or (
            self.kind == "recipe" and self.name in FETCHING_RECIPES
        )

    @property
    def long_limit(self) -> bool:
        """Whether the fetch gets the 120 s limit: a free node or a recipe of several
        nodes. PTDATA and a catalog node get 30 s."""
        return self.free or self.kind == "recipe"

    def where(self) -> str:
        """For display only: the node, point, group or reader."""
        if self.kind == "mds":
            return self.name
        if self.kind == "ptdata":
            return f"PTDATA {self.name}"
        if self.kind in ("corpus", "raw"):
            rows = "" if self.rows is None else f" rows {_span(self.rows)}"
            summed = ", summed" if self.total else ""
            tier = "corpus" if self.kind == "corpus" else "raw cache"
            return f"{tier} {self.name}{rows}{summed}"
        if self.kind == "feature":
            return f"features store {self.name}"
        if self.kind == "park":
            return f"MHD park {self.name}"
        return f"profile reader {self.name}"

    def spec(self) -> dict:
        """What the values depend on; its hash keys the signals cache."""
        out = {"kind": self.kind, "name": self.name}
        if self.tree is not None:
            out["tree"] = self.tree
        if self.rows is not None:
            out["rows"] = list(self.rows)
        if self.total:
            out["total"] = True
        return out

    def key(self) -> str:
        """The signals-cache group of this source's record (scale and tag excluded:
        they are applied on read)."""
        text = json.dumps(self.spec(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Entry:
    """One signal the page can add: a trace (C channels against time) or a profile."""

    id: str
    label: str
    group: str
    units: str
    kind: str
    sources: tuple[Source, ...]
    events: tuple[str, ...] = ()
    legend: tuple[str, ...] | None = None
    channels: int | None = None
    axis: str = "time"
    draw: str | None = None

    def suggested(self, event: str | None) -> bool:
        """Whether this entry is suggested for `event`: named for it, for the store it
        shows (mhd_n2 shows mhd_n1's), or for its parent (a subcategory)."""
        if not event or not self.events:
            return False
        names = {event, store_event(event), event.partition(".")[0]}
        return bool(names & set(self.events))

    def legend_for(self, n: int) -> list[str]:
        if self.legend is not None and len(self.legend) == n:
            return list(self.legend)
        if n == 1:
            return [self.label]
        return [f"ch {i + 1}" for i in range(n)]


@dataclass(frozen=True)
class Catalog:
    entries: tuple[Entry, ...] = ()
    by_id: dict = field(default_factory=dict)

    def resolve(self, ident) -> Entry:
        """The entry an id names: a catalog entry, or one made for a `custom:` or
        `mds:` id. Raises `Invalid` for anything else."""
        if not isinstance(ident, str) or not ident:
            raise Invalid("name a signal: a catalog id, custom:<PTDATA> or "
                          "mds:<TREE>:<node>")
        if ident.startswith(CUSTOM_PREFIX):
            return custom_entry(ident[len(CUSTOM_PREFIX):])
        if ident.startswith(MDS_PREFIX):
            tree, sep, node = ident[len(MDS_PREFIX):].partition(":")
            if not sep:
                raise Invalid(f"{_shown(ident)}: write mds:<TREE>:<node>")
            return mds_entry(tree, node)
        entry = self.by_id.get(ident)
        if entry is None:
            raise Invalid(f"{_shown(ident)} is not in the catalog")
        return entry


def _shown(text: str, limit: int = 120) -> str:
    """A client's text for an error message: repr, cut short."""
    shown = repr(text)
    return shown if len(shown) <= limit else shown[:limit] + "..."


def _span(rows: tuple[int, ...]) -> str:
    if len(rows) > 1 and rows == tuple(range(rows[0], rows[-1] + 1)):
        return f"{rows[0]}-{rows[-1]}"
    return ",".join(str(r) for r in rows)


def ptdata_problem(name: str) -> str | None:
    """Why `name` is not a PTDATA point name, or None."""
    if isinstance(name, str) and PTDATA.fullmatch(name):
        return None
    return (f"{_shown(name)} is not a PTDATA point name: use a letter, then letters, "
            "digits or _, up to 12 characters")


def node_problem(tree: str, node: str) -> str | None:
    """Why `node` in `tree` may not be fetched, or None (plan B, tightened by D.A)."""
    if not isinstance(tree, str) or tree.upper() not in TREES:
        return f"{_shown(tree)} is not an allowed MDS tree: {', '.join(TREES)}"
    if not isinstance(node, str) or not node:
        return "name an MDS node"
    if len(node) > NODE_MAX:
        return f"an MDS node is at most {NODE_MAX} characters"
    if not NODE.fullmatch(node):
        return (f"{_shown(node)} is not an MDS node path: letters, digits and _ "
                "joined by ::, . or :, with no parentheses, quotes, commas, spaces or $")
    path = node.removeprefix("\\")
    head, sep, _ = path.partition("::")
    if not sep:
        return f"{_shown(node)}: name its tree, as \\{tree.upper()}::..."
    if head.upper() != tree.upper():
        return f"{_shown(node)} is in {head}, not {tree.upper()}"
    for segment in re.split(r"::|[.:]", path):
        if not segment:
            return f"{_shown(node)} has an empty segment"
        if not SEGMENT.fullmatch(segment):
            return f"{_shown(node)}: segment {segment!r} must start with a letter"
        if segment.upper().startswith(REFUSED_SEGMENTS):
            return f"{_shown(node)}: segment {segment!r} is refused"
    if not NODE_STRICT.fullmatch(node):
        return f"{_shown(node)} is not an MDS node path"
    return None


def custom_entry(name: str) -> Entry:
    """The entry for `custom:<NAME>`, a PTDATA point. PTDATA and MDSplus names are
    case-insensitive, so a free source is upper case: one cache key per point."""
    problem = ptdata_problem(name)
    if problem:
        raise Invalid(problem)
    return Entry(
        id=CUSTOM_PREFIX + name, label=name, group="PTDATA", units="", kind="trace",
        sources=(Source("ptdata", name.upper(), free=False),), channels=1,
    )


def mds_entry(tree: str, node: str) -> Entry:
    """The entry for `mds:<TREE>:<node>`, a free MDSplus node (the 120 s limit)."""
    problem = node_problem(tree, node)
    if problem:
        raise Invalid(problem)
    tree = tree.upper()
    node = node if node.startswith("\\") else "\\" + node
    return Entry(
        id=f"{MDS_PREFIX}{tree}:{node}", label=node, group="MDS", units="",
        kind="trace", sources=(Source("mds", node.upper(), tree=tree, free=True),),
    )


# ---- signals.yaml ----


def load(path) -> Catalog:
    """The catalog at `path`, re-read when the file changes; an empty catalog when
    the file is absent."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return Catalog()
    return _load(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load(path: str, _mtime_ns: int, _size: int) -> Catalog:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CatalogError(f"{path}: not readable: {error}") from error
    return parse(raw, where=path)


def parse(raw, *, where: str = CATALOG_FILE) -> Catalog:
    """A validated catalog from signals.yaml's parsed content."""
    if not isinstance(raw, dict) or not isinstance(raw.get("signals"), list):
        raise CatalogError(f"{where}: needs a `signals:` list")
    entries, by_id = [], {}
    for index, item in enumerate(raw["signals"]):
        label = item.get("id", f"#{index}") if isinstance(item, dict) else f"#{index}"
        entry = _entry(item, f"{where}: signal {label}")
        if entry.id in by_id:
            raise CatalogError(f"{where}: signal {entry.id} is listed twice")
        by_id[entry.id] = entry
        entries.append(entry)
    return Catalog(tuple(entries), by_id)


ENTRY_KEYS = {"id", "label", "group", "units", "kind", "events", "sources", "legend",
              "channels", "axis", "draw"}


def _entry(item, where: str) -> Entry:
    if not isinstance(item, dict):
        raise CatalogError(f"{where}: must be a mapping")
    unknown = set(item) - ENTRY_KEYS
    if unknown:
        raise CatalogError(f"{where}: unknown keys {sorted(unknown)}")
    ident = item.get("id")
    if not isinstance(ident, str) or not ID.fullmatch(ident):
        raise CatalogError(f"{where}: id must match {ID.pattern}")
    for key in ("label", "group"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise CatalogError(f"{where}: needs a {key}")
    units = item.get("units", "")
    if not isinstance(units, str):
        raise CatalogError(f"{where}: units must be text")
    kind = item.get("kind")
    if kind not in KINDS:
        raise CatalogError(f"{where}: kind must be one of {KINDS}")
    events = item.get("events", [])
    if not isinstance(events, list) or not all(
        isinstance(e, str) and NAME.fullmatch(e) for e in events
    ):
        raise CatalogError(f"{where}: events must be a list of event names")
    sources = item.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CatalogError(f"{where}: needs a non-empty sources list")
    parsed = tuple(_source(s, f"{where}: source {i + 1}") for i, s in enumerate(sources))
    recipes = [s.kind == "recipe" for s in parsed]
    if kind == "profile" and not all(recipes):
        raise CatalogError(f"{where}: a profile's sources are recipes")
    if kind == "trace" and any(recipes):
        raise CatalogError(f"{where}: a trace cannot use a profile recipe")
    axis = item.get("axis", "time" if kind == "trace" else None)
    if kind == "profile" and axis not in X_KINDS:
        raise CatalogError(f"{where}: a profile's axis is one of {X_KINDS}")
    if kind == "trace" and axis != "time":
        raise CatalogError(f"{where}: a trace's axis is time")
    channels = item.get("channels")
    if channels is not None and (
        not isinstance(channels, int) or isinstance(channels, bool) or channels < 1
    ):
        raise CatalogError(f"{where}: channels must be a positive integer")
    if kind == "trace":
        widths = {1 if s.total or s.rows is None else len(s.rows) for s in parsed}
        if len(widths) > 1 or (channels is not None and channels not in widths):
            raise CatalogError(f"{where}: its sources give different channel counts")
        channels = widths.pop()
    legend = item.get("legend")
    if legend is not None:
        if not isinstance(legend, list) or not all(isinstance(x, str) for x in legend):
            raise CatalogError(f"{where}: legend must be a list of names")
        if channels is not None and len(legend) != channels:
            raise CatalogError(f"{where}: legend names {len(legend)} channels, "
                               f"not {channels}")
        legend = tuple(legend)
    draw = item.get("draw")
    if draw is not None and (draw not in DRAWS or kind != "trace"):
        raise CatalogError(f"{where}: draw is one of {DRAWS}, for a trace")
    return Entry(
        id=ident, label=item["label"].strip(), group=item["group"].strip(),
        units=units, kind=kind, sources=parsed, events=tuple(events), legend=legend,
        channels=channels, axis=axis, draw=draw,
    )


SOURCE_KEYS = {"rows", "total", "scale", "tag", "tree"}


def _source(item, where: str) -> Source:
    if not isinstance(item, dict):
        raise CatalogError(f"{where}: must be a mapping")
    kinds = [k for k in (*LOCAL, *FETCHED) if k in item]
    if len(kinds) != 1:
        raise CatalogError(f"{where}: names exactly one of {(*LOCAL, *FETCHED)}")
    kind = kinds[0]
    unknown = set(item) - SOURCE_KEYS - {kind}
    if unknown:
        raise CatalogError(f"{where}: unknown keys {sorted(unknown)}")
    name = item[kind]
    scale = item.get("scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not (
        math.isfinite(scale) and scale > 0
    ):
        raise CatalogError(f"{where}: scale must be a positive number")
    tag = item.get("tag", "")
    if not isinstance(tag, str):
        raise CatalogError(f"{where}: tag must be text")
    rows, total, tree = None, bool(item.get("total", False)), item.get("tree")
    if not isinstance(item.get("total", False), bool):
        raise CatalogError(f"{where}: total is true or false")
    if kind in ("corpus", "raw"):
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise CatalogError(f"{where}: {kind} names a group")
        rows = _rows(item.get("rows"), where)
        if rows is None:
            raise CatalogError(f"{where}: a {kind} source lists its rows")
    elif "rows" in item or total:
        raise CatalogError(f"{where}: rows and total are for corpus and raw groups")
    if kind == "feature" and not (isinstance(name, str) and NAME.fullmatch(name)):
        raise CatalogError(f"{where}: feature names a features-store quantity")
    if kind == "park" and name not in PARK_TRACES:
        raise CatalogError(f"{where}: park is one of {PARK_TRACES}")
    if kind == "recipe" and name not in RECIPES:
        raise CatalogError(f"{where}: recipe is one of {sorted(RECIPES)}")
    if kind == "ptdata":
        problem = ptdata_problem(name)
        if problem:
            raise CatalogError(f"{where}: {problem}")
    if kind == "mds":
        problem = node_problem(tree, name)
        if problem:
            raise CatalogError(f"{where}: {problem}")
        tree = tree.upper()
    elif tree is not None:
        raise CatalogError(f"{where}: tree is for an mds source")
    return Source(kind, name, tree=tree, rows=rows, total=total, scale=float(scale),
                  tag=tag.strip())


def _rows(value, where: str) -> tuple[int, ...] | None:
    """`rows`: an index, a list of indices or "first-last"."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        found = [value]
    elif isinstance(value, str) and re.fullmatch(r"\d+-\d+", value):
        first, last = (int(v) for v in value.split("-"))
        found = list(range(first, last + 1))
    elif isinstance(value, list) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        found = list(value)
    else:
        raise CatalogError(f"{where}: rows is an index, a list or \"first-last\"")
    if not found or min(found) < 0 or len(set(found)) != len(found):
        raise CatalogError(f"{where}: rows must be distinct non-negative indices")
    return tuple(found)
