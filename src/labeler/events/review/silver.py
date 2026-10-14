"""The silver review set and the single giant grid for DIII-D.

Every shot exports one event × 10 ms frame array, with -1 for unassessed
frames and the winning source named separately from its category.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Paths, atomic_path
from .. import grid, rosters, taxonomy
from ..catalog.cohort import read_cohort
from . import labels

log = logging.getLogger(__name__)
SILVER_FILE = "silver/silver_shots.csv"
COLUMNS = ("shot", "batch", "order", "pool", "classes", "sources", "notes")
#: Where a silver shot came from, and its tag on the page ("" for none).
POOLS = {
    "corpus": "",
    "ae_audited": "AE audited",
    "ae_ece_box": "AE ECE box",
    "rwm_onsets": "RWM onset",
}
FOLD = {
    "high_confinement_mode": ("confinement", {1: 1, 2: 5}),
    "low_confinement_mode": ("confinement", {1: 2, 2: 5}),
    "quiescent_high_confinement_mode": ("confinement", {1: 3, 2: 5}),
    "wide_pedestal_quiescent_high_confinement_mode": ("confinement", {1: 4, 2: 5}),
    "improved_energy_confinement_mode": ("confinement", {1: 6, 2: 5}),
    "sawtooth_oscillation": ("mhd_n1", {1: 1, 2: 2}),
    "fishbone": ("mhd_n1", {1: 1, 2: 2}),
    "neoclassical_tearing_mode": ("mhd_n1", {1: 1, 2: 2}),
}
GRID_RULE = (
    "10 ms frames [10k, 10k+10) from t=0; nearest edges, halves up; "
    "vanishing spans claim their midpoint frame; windows snap like spans; "
    "-1 unassessed: outside every window, where a table marks not observable or "
    "unassessed or a category the event lacks (when the owner's save of the "
    "event marks it, the frame names that save and no lower layer fills it), for a child outside its "
    "parent's saved spans (all of it when the parent is saved absent) or its "
    "own table's window, and where a folded table, or a confinement or "
    "detachment table other than the owner's save, marks nothing; "
    "0 is what `zero` names per event; first assessing layer wins, in the order "
    "saved, folded saves, format, extend, source, suggest"
)
#: The order layers win in: the owner's saves, then curated tables, then machines.
#: Every review pointer's source today is a machine suggestion.
PRECEDENCE = ("saved", "folded", "format", "extend", "source", "suggest")
#: What 0 means where it differs from "assessed and not seen". Only the owner's
#: save says it there: another table's 0 names no regime, so it is -1 and a lower
#: layer may speak.
ZERO = {
    "confinement": "the owner saved no regime here",
    "detachment": "the owner saved no state here",
}
#: Category names that say nobody assessed the time (RWM's 4): -1 in the grid.
UNASSESSED_NAMES = {"unassessed", "not_observable"}


@dataclass(frozen=True)
class Layer:
    """One candidate label, in precedence order, with its table's origin."""

    kind: str
    name: str
    label: labels.Label
    #: Snapped stretches the table marks unassessed or not observable: -1, not 0.
    unknown: tuple[tuple[int, int], ...] = ()
    #: The owner's own "not observable" or "unassessed" in a save of this event:
    #: -1 that no layer under it fills.
    held: tuple[tuple[int, int], ...] = ()


def read_silver(label_tables) -> pd.DataFrame:
    """The validated silver list, cached on its file version."""
    path = Path(label_tables) / SILVER_FILE
    stat = path.stat()
    return _read_silver(str(path), stat.st_mtime_ns, stat.st_ino, stat.st_size)


def blind_shots(label_tables, *, missing_ok: bool = False) -> set[int]:
    """The catalog cohort's blind test shots, which silver never holds.

    Fails closed: without `catalog/cohort.csv` it raises FileNotFoundError, because
    every writer that drops these shots (the sawtooth, crash and TM sources, the
    ELM jump and detachment sources, the silver export) would otherwise drop
    nothing. `missing_ok=True` is only for re-validating a list already built: a
    tree restored from the Hugging Face release has no catalog (the release
    withholds it) and its lists were scrubbed of these shots when it was built.
    """
    cohort = Path(label_tables) / "catalog" / "cohort.csv"
    if not cohort.is_file():
        if missing_ok:
            return set()
        raise FileNotFoundError(
            f"{cohort} is missing: the catalog's blind test shots are unknown, so "
            "nothing can be kept out of them"
        )
    return set(read_cohort(cohort).query("blind").shot.astype(int))


@lru_cache(maxsize=8)
def _read_silver(path, _mtime, _ino, _size):
    frame = pd.read_csv(path, keep_default_na=False)
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError(f"{path}: silver list needs {', '.join(COLUMNS)}")
    for column in ("shot", "batch", "order"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not (np.isfinite(values) & (values == np.floor(values))).all():
            raise ValueError(f"{path}: {column} must be integer")
        frame[column] = values.astype("int64")
    if frame.shot.duplicated().any():
        raise ValueError(f"{path}: duplicate silver shots")
    if (frame.shot < 0).any():
        raise ValueError(f"{path}: shot must be nonnegative")
    if not frame.batch.between(1, 5).all():
        raise ValueError(f"{path}: batch must be 1 to 5")
    if not frame.order.between(1, 100).all():
        raise ValueError(f"{path}: order must be 1 to 100")
    if frame.duplicated(["batch", "order"]).any():
        raise ValueError(f"{path}: duplicate batch/order")
    blind = blind_shots(Path(path).parent.parent, missing_ok=True) & set(frame.shot)
    if blind:
        raise ValueError(f"{path}: blind cohort shots {sorted(blind)}")
    if not frame.pool.isin(list(POOLS)).all():
        raise ValueError(f"{path}: pool must be one of {sorted(POOLS)}")
    return frame.sort_values(["batch", "order"]).reset_index(drop=True)


def is_silver(label_tables, shot: int) -> bool:
    try:
        return bool((read_silver(label_tables).shot == int(shot)).any())
    except FileNotFoundError:
        return False
    except (ValueError, OSError) as error:
        log.warning("silver list unreadable: %s", error)
        return False


def review_events(paths: Paths) -> list[str]:
    """The page's top-level events, each followed by its subcategories."""
    tree = taxonomy.load(paths.events_manifest)
    try:
        parents = sorted(
            path.name for path in Path(paths.label_tables).iterdir()
            if path.name not in labels.FOLDED
            and (path / rosters.ROSTER_NAME).is_file() and tree.reviewed(path.name)
        )
    except OSError:
        return []
    return [event for parent in parents for event in [parent, *tree.children(parent)]]


def event_dir(paths: Paths, event: str) -> Path:
    tree = taxonomy.load(paths.events_manifest)
    return (tree.child_dir(paths.label_tables, event) if tree.is_child(event)
            else paths.label_tables / event)


def _read(path: Path, shot: int) -> labels.Label | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    version = (str(path), stat.st_mtime_ns, stat.st_size)
    try:
        # Most tables (a sawtooth chunk, the 28k-shot TM table) lack the shot.
        if int(shot) not in _shots(*version):
            return None
        if _crowd_schema(*version):
            return _crowd_table(*version).get(int(shot))
        return labels.read_labels(path).get(int(shot))
    except Exception as error:  # noqa: BLE001 - one bad table cannot hide a shot
        _warn_once(*version, str(error))
        return None


@lru_cache(maxsize=1024)
def _shots(path, _mtime, _size) -> frozenset[int]:
    return frozenset(pd.read_csv(path, usecols=["shot"]).shot.astype(int))


@lru_cache(maxsize=1024)
def _crowd_schema(path, _mtime, _size) -> bool:
    """The sawtooth physics tables' schema: a boolean `crowd` column, not attrs."""
    return "crowd" in pd.read_csv(path, nrows=0).columns


@lru_cache(maxsize=256)
def _crowd_table(path, _mtime, _size) -> dict[int, labels.Label]:
    frame = pd.read_csv(path)
    frame = frame[frame.t_end - frame.t_start >= 1]
    found = {}
    for shot, rows in frame.groupby("shot", sort=False):
        spans = rows[rows.category != 0]
        found[int(shot)] = labels.normalise(
            (rows.t_start.min(), rows.t_end.max()),
            zip(spans.t_start, spans.t_end, spans.category.astype(int)),
            iscrowd=[int(bool(flag)) for flag in spans.crowd],
            grid_ms=1,
        )
    return found


@lru_cache(maxsize=1024)
def _warn_once(path, _mtime, _size, error) -> None:
    log.warning("skipping %s: %s", path, error)


def parent_region(paths: Paths, event: str, shot: int) -> labels.Region | None:
    parent = taxonomy.load(paths.events_manifest).parent(event)
    if parent is None:
        return None
    saved = _read(labels.labels_path(event_dir(paths, parent)), shot)
    return labels.region_of(parent, labels.page_label(parent, saved))


def locked(paths: Paths, event: str, shot: int) -> bool:
    if not taxonomy.load(paths.events_manifest).is_child(event):
        return False
    region = parent_region(paths, event, shot)
    return region is None or not region.spans


def _version(path: Path):
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", path.name))


def _tables(paths: Paths, event: str):
    directory = event_dir(paths, event)
    yield "saved", labels.labels_path(directory)
    curated = [("format", path) for path in sorted(directory.glob("format/*.csv"))]
    curated += [("extend", path) for path in sorted(directory.glob("extend_*/**/*.csv"))]
    yield from curated
    source = None
    if labels.pointer_path(directory).is_file():
        try:
            source = labels.source_path(directory)
        except Exception as error:  # noqa: BLE001
            log.warning("skipping %s: %s", labels.pointer_path(directory), error)
    # A pointer at a curated table keeps that table's rank.
    if source is not None and source.resolve() not in {
        path.resolve() for _, path in curated
    }:
        yield "source", source
    for method in sorted((paths.root / "suggestions").glob("*")):
        versions = [path for path in method.glob("*") if path.is_dir()]
        latest = max(versions, key=_version, default=None)
        if latest is None:
            continue
        path = latest / f"{event}_suggest_{method.name}_{latest.name}.csv"
        if source is None or path.resolve() != source.resolve():
            yield "suggest", path


def _split(event: str, label: labels.Label, mapping=None, *, zero=True) -> Layer:
    """A whole-ms label as the silver view shows it: the event's categories on the
    grid, and apart from them the stretches nobody assessed. A table whose 0 does
    not say "assessed and not seen" (`zero` false) asserts only its categories:
    the rest of its window is unknown too."""
    shown = labels.categories(event)
    names = {int(k): v for k, v in labels.category_labels(event).items()}
    keep, unknown, held = [], [], []
    for i, (a, b, c) in enumerate(label.intervals):
        mapped = c if mapping is None else mapping.get(c)
        if c == 0 and (mapping is None or label.attrs):
            keep.append((i, (a, b, 0)))  # an evidence table's explicit absent row
        elif c == 0:
            continue
        elif mapped in shown and shown[mapped] not in UNASSESSED_NAMES:
            keep.append((i, (a, b, mapped)))
        else:
            unknown.append(grid.snap_span(a, b))
            if mapping is None and names.get(c) in UNASSESSED_NAMES:
                held.append(grid.snap_span(a, b))
    kept = labels.Label(
        label.window, tuple(span for _, span in keep),
        tuple(label.iscrowd[i] for i, _ in keep) if label.iscrowd else (),
        tuple(label.attrs[i] for i, _ in keep) if label.attrs else (),
    )
    kept = labels.on_grid(kept)
    if not zero:
        unknown += _outside(kept.window, [(a, b) for a, b, c in kept.intervals if c])
    return Layer("", "", kept, tuple(sorted(set(unknown))), tuple(sorted(set(held))))


def _outside(window, spans) -> list[tuple[int, int]]:
    """The parts of `window` no span covers."""
    gaps, at = [], window[0]
    for a, b in sorted(spans):
        if a > at:
            gaps.append((at, min(a, window[1])))
        at = max(at, b)
    if at < window[1]:
        gaps.append((at, window[1]))
    return [(a, b) for a, b in gaps if b > a]


def _name(paths: Paths, kind: str, path: Path) -> str:
    if kind == "suggest":
        name = path.relative_to(paths.root / "suggestions").with_suffix("")
    elif kind == "format":
        relative = path.relative_to(paths.label_tables)
        name = relative.parent.parent / path.stem
    else:
        try:
            name = path.relative_to(paths.label_tables).with_suffix("")
        except ValueError:
            name = path.resolve().with_suffix("")
    return f"{kind}:{name.as_posix()}"


def layer_file(paths: Paths, layer: Layer) -> Path:
    """The exact table named by an origin, for save provenance."""
    name = Path(layer.name.split(":", 1)[1] + ".csv")
    if layer.kind == "suggest":
        return paths.root / "suggestions" / name
    if layer.kind == "format":
        return paths.label_tables / name.parent / "format" / name.name
    return paths.label_tables / name


def _inside(region: labels.Region, label: labels.Label) -> labels.Label | None:
    """A child table's label cut to its parent's region, on the part of the
    parent's window the table itself covers (None when they do not meet)."""
    clipped = region.clip(label)
    lo = max(region.window[0], label.window[0])
    hi = min(region.window[1], label.window[1])
    return dataclasses.replace(clipped, window=(lo, hi)) if hi > lo else None


def layers(paths: Paths, event: str, shot: int) -> list[Layer]:
    """Every readable label for this event on the shared grid, in `PRECEDENCE`."""
    tree = taxonomy.load(paths.events_manifest)
    region = parent_region(paths, event, shot) if tree.is_child(event) else None
    if tree.is_child(event) and (region is None or not region.spans):
        return []
    inputs = [(event, None)]
    for folded, (parent, mapping) in FOLD.items():
        if event == parent or (event == "mhd_n1.m1" and folded in {
            "sawtooth_oscillation", "fishbone",
        }):
            inputs.append((folded, mapping))
    found = []
    for origin, mapping in inputs:
        for kind, path in _tables(paths, origin):
            # A folded event's saves are the owner's evidence, not this event's save.
            kind = "folded" if mapping is not None and kind == "saved" else kind
            value = _read(path, shot)
            if value is None:
                continue
            try:
                # A folded table's 0 is "not this subtype"; a regime table's 0 names
                # no regime: neither says the event was assessed absent.
                zero = mapping is None and (event not in ZERO or kind == "saved")
                split = _split(event, value, mapping, zero=zero)
                label = split.label if region is None else _inside(region, split.label)
                if label is None:
                    continue
                label = labels.span_default(event, label)
                found.append(Layer(kind, _name(paths, kind, path), label, split.unknown,
                                   split.held if kind == "saved" else ()))
            except Exception as error:  # noqa: BLE001
                log.warning("skipping %s: %s", path, error)
    return sorted(found, key=lambda layer: PRECEDENCE.index(layer.kind))


def warm(paths: Paths) -> None:
    """Read every table the first silver shot touches (about 30 s on DIII-D's), so
    the page's first shot opens at once; meant for a background thread."""
    try:
        frame = read_silver(paths.label_tables)
        # A corpus shot is in the big population tables; an old AE shot is not.
        corpus = frame[frame.pool == "corpus"]
        shot = int((corpus if len(corpus) else frame).shot.iloc[0])
        for event in review_events(paths):
            layers(paths, event, shot)
    except Exception as error:  # noqa: BLE001 - a cold cache is only slower
        log.warning("silver cache not warmed: %s", error)


def progress(paths: Paths, shot: int) -> dict:
    events = [event for event in review_events(paths)
              if not locked(paths, event, shot)]
    saved = [event for event in events
             if _read(labels.labels_path(event_dir(paths, event)), shot) is not None]
    return {"saved": saved, "total": len(events)}


def time_range(event_layers) -> tuple[int, int]:
    windows = [layer.label.window for layers_ in event_layers for layer in layers_]
    return ((min(lo for lo, _ in windows), max(hi for _, hi in windows))
            if windows else (0, grid.GRID_MS))


def paint(group, t0: int, n: int, allowed=None) -> tuple[np.ndarray, np.ndarray]:
    """`n` frames from `t0` where the first layer that assessed a frame wins, and
    the index of that layer in `group` (-1 where none did). A save's held
    stretches stay -1 but are taken: they name the save."""
    values = np.full(n, grid.UNASSESSED, dtype=np.int8)
    taken = np.full(n, -1, dtype=np.int16)
    for k, layer in enumerate(group):
        cells = grid.frames(layer.label.window, layer.label.intervals, t0, n,
                            unknown=layer.unknown)
        free = taken == -1
        if allowed is not None:
            free &= allowed
        take = free & (cells != grid.UNASSESSED)
        values[take], taken[take] = cells[take], k
        if layer.held:
            held = grid.frames((t0, t0 + n * grid.GRID_MS),
                               [(a, b, 1) for a, b in layer.held], t0, n) == 1
            taken[free & held & ~take] = k
    return values, taken


def draft(paths: Paths, event: str, shot: int) -> tuple[labels.Label, Layer] | None:
    """The page's draft when the event has no source row for a silver shot: its
    first non-saved layer, filled where that layer said nothing by the layers
    under it as the export fills them (then one label without lanes), and that
    first layer for provenance."""
    found = [layer for layer in layers(paths, event, shot) if layer.kind != "saved"]
    if not found:
        return None
    t0, hi = time_range([found])
    n = (hi - t0) // grid.GRID_MS
    values, _ = paint(found, t0, n)
    if np.array_equal(values, paint(found[:1], t0, n)[0]):
        return found[0].label, found[0]
    spans, start = [], None
    for k, value in enumerate([*values.tolist(), grid.UNASSESSED]):
        if start is not None and value != values[start]:
            spans.append((t0 + start * grid.GRID_MS, t0 + k * grid.GRID_MS,
                          int(values[start])))
            start = None
        if start is None and value > 0:
            start = k
    return labels.Label((t0, hi), tuple(spans)), found[0]


def export(paths: Paths, shots, out_dir):
    """Write one compressed grid per shot and its readable dataset index."""
    events = review_events(paths)
    shots = [int(shot) for shot in shots]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    origins = []
    for shot in shots:
        found = [layers(paths, event, shot) for event in events]
        t0, hi = time_range(found)
        n = (hi - t0) // grid.GRID_MS
        values = np.full((len(events), n), grid.UNASSESSED, dtype=np.int8)
        origin = np.full_like(values, -1)
        shot_origins = list(dict.fromkeys(layer.name for group in found for layer in group))
        if len(shot_origins) > 128:
            raise ValueError(f"shot {shot} has more than 128 origins for int8 indices")
        origins.extend(name for name in shot_origins if name not in origins)
        for i, (event, group) in enumerate(zip(events, found, strict=True)):
            region = parent_region(paths, event, shot)
            allowed = (grid.frames((t0, hi), [(a, b, 1) for a, b in region.spans],
                                   t0, n) == 1 if region is not None else None)
            values[i], taken = paint(group, t0, n, allowed)
            for k, layer in enumerate(group):
                origin[i, taken == k] = shot_origins.index(layer.name)
        with atomic_path(out_dir / f"{shot}.npz") as tmp, tmp.open("wb") as stream:
            np.savez_compressed(
                stream, labels=values, origin=origin,
                events=np.asarray(events, dtype=str),
                origins=np.asarray(shot_origins, dtype=str),
                t0_ms=np.int64(t0), grid_ms=np.int64(grid.GRID_MS),
            )
    index = {
        "events": events,
        "categories": {event: {"0": "absent", **{
            str(k): v for k, v in labels.categories(event).items()
        }} for event in events},
        "zero": {event: ZERO.get(event, "absent: assessed and not seen")
                 for event in events},
        "grid_ms": grid.GRID_MS, "grid_rule": GRID_RULE,
        "shots": shots, "origins": origins,
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(out_dir / "index.json") as tmp:
        tmp.write_text(json.dumps(index, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("export", help="the single giant grid for DIII-D")
    command.add_argument("--out", required=True, type=Path)
    command.add_argument("--shots-file", type=Path)
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    shots = (read_silver(paths.label_tables).shot if args.shots_file is None
             else pd.read_csv(args.shots_file)["shot"])
    blind = blind_shots(paths.label_tables) & set(shots)
    if blind:
        parser.error(f"blind cohort shots are never exported: {sorted(blind)}")
    export(paths, shots, args.out)


if __name__ == "__main__":
    main()
