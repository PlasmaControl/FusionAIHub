"""The cohort: 500 population shots drawn with known weights.

The population passes rules 1-4 from `catalog.population`, then rule 5 (D2e)
drops runaway-electron plateaus using the committed Thomson scan, runaway.csv.
It requires median channel-p90 Te below 60 eV and at least one usable profile
in the assessed window: ceil(n_channels / 2) finite, positive core channels.
Te's median still uses all samples with any valid channel; `n_profile` counts
profile samples. Shots with no usable profile are retained (`no_thomson`).

Groups partition the population, each shot taking the first that applies:
- L: at least one verified OSTI link at freeze (a `papers.csv` row, source osti);
- G: named by a legacy label table (every table `events.yaml` registers);
- R: everything else.

Each group crossed with campaign year (2021-2025) is a cell, 15 in all. L takes all of
its shots up to 200, G up to 100, and R the rest of the 500. Within a group the shots
are shared across years in proportion to the population, by largest remainder, at
least one per non-empty cell. A cell's shots are a simple random sample, its shots
with the smallest draw keys, and each weighs N_h / n_h.

Every cohort shot then gets an order key u. The blind subset (the test split) is 50
shots shared across L, G and R in proportion to their cohort counts, each group's
smallest u; val is the next 50 by the same rule; train is the rest. The review queue
is the blind shots, then the rest, each by u, so any prefix of it is a random
subsample of every group.

Keys are hashes of (seed, purpose, shot), not draws from one generator, so a shot's
key does not depend on which other shots are in the population.

    pixi run -e labelmaker python -m labeler.events.catalog.cohort

reads `$LABELER_ROOT/catalog/{pool.csv, ip.jsonl}`, `data/events/catalog/papers.csv`,
`data/events/catalog/runaway.csv`
and the legacy tables, and writes `cohort.csv`, `cohort_manifest.yaml` and
`population.csv` to `$LABELER_ROOT/catalog/`. The freeze copies all three files,
`cohort.csv`, `population.csv` and `cohort_manifest.yaml`, into
`data/events/catalog/`, so anyone holding the release can rederive the draw (D20).
Freezing the cohort is the owner's call.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction
from numbers import Real
from pathlib import Path

import pandas as pd
import yaml

from ...config import NamedBytes, Paths, atomic_path, git_dirty, git_sha
from ...literature.papers import read_papers
from .. import databases
from . import population as pop
from . import runaway, window
from .check import CatalogError, Finding, require
from .points import validate_csv_fields

SEED = 20260923
COHORT_SIZE = 500
GROUPS = ("L", "G", "R")
#: The most shots L and G take; R takes the rest of the cohort.
CAPS = {"L": 200, "G": 100}
YEARS = (2021, 2022, 2023, 2024, 2025)
BLIND_SIZE = VAL_SIZE = 50
SPLITS = ("test", "val", "train")
SPANS = tuple(f"span_{group}_s" for group in pop.SPAN_GROUPS)
COHORT_COLUMNS = (
    "shot",
    "year",
    "group",
    "cell",
    "weight",
    "split",
    "blind",
    "queue_rank",
    "u",
    "window_start_ms",
    "window_end_ms",
    "ip_ma",
    "pulse_length_s",
    "flattop_s",
    "ip_peak_ma",
    "pbeam_max_mw",
    "pech_max_mw",
    "run_id",
    "legacy_sets",
    "n_links_verified",
    *SPANS,
)
POPULATION_COLUMNS = (
    "shot",
    "year",
    "group",
    "cell",
    "in_cohort",
    "window_start_ms",
    "window_end_ms",
    "flattop_s",
    "legacy_sets",
    "n_links_verified",
    *SPANS,
)
DRAW_COLUMNS = (
    "shot",
    "year",
    "group",
    "cell",
    "weight",
    "split",
    "blind",
    "queue_rank",
    "u",
)


def key(seed: int, purpose: str, shot: int) -> float:
    """A uniform number in [0, 1) fixed by the seed, the purpose and the shot alone."""
    digest = hashlib.sha256(f"{seed}:{purpose}:{int(shot)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def allocate(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """`n` shared in proportion to `sizes`, by largest remainder.

    Every non-empty cell gets at least 1, taken one at a time from the cell furthest
    above its quota. A cell never gets more than its size, and an `n` at or above the
    total takes everything.
    """
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    live = sorted(c for c in sizes if sizes[c] > 0)
    if n < len(live):
        raise ValueError(
            f"{n} shots cannot give each of {len(live)} non-empty cells one"
        )
    quota = {c: Fraction(n * sizes[c], total) for c in sizes}
    share = {c: math.floor(quota[c]) for c in sizes}
    left = n - sum(share.values())
    for c in sorted(sizes, key=lambda c: (share[c] - quota[c], c))[:left]:
        share[c] += 1
    for c in live:
        if share[c] == 0:
            donor = max(
                (d for d in live if share[d] > 1),
                key=lambda d: (share[d] - quota[d], d),
            )
            share[donor] -= 1
            share[c] = 1
    return share


def legacy_sets(
    root, *, inputs=None
) -> tuple[dict[int, tuple[str, ...]], list[databases.TableSpec]]:
    """`{shot: the directories of the legacy tables naming it}`, and the tables read."""
    specs = list(databases.load_manifest(root))
    named: dict[int, set[str]] = defaultdict(set)
    for spec in specs:
        path = spec.path(root)
        data = path.read_bytes()
        source = NamedBytes(data, path)
        validate_csv_fields(source)
        table = databases._parse(source)
        for column, expected in (
            ("source", spec.source),
            ("phenomenon", spec.phenomenon),
        ):
            if column in table and not table[column].eq(expected).all():
                raise CatalogError(f"{path}: {column} must match manifest {expected!r}")
        if inputs is not None:
            inputs.append(_input(path, Path(root), data=data))
        for shot in table.shot:
            named[int(shot)].add(spec.dir)
    return {shot: tuple(sorted(dirs)) for shot, dirs in named.items()}, specs


def osti_links(papers: pd.DataFrame) -> dict[int, int]:
    """Verified OSTI links per shot; arXiv links never move a shot between groups."""
    return papers.loc[papers["source"].eq("osti"), "shot"].value_counts().to_dict()


def assign_groups(
    frame: pd.DataFrame, links: Mapping[int, int], legacy: Mapping[int, tuple[str, ...]]
) -> pd.DataFrame:
    """The population with its group, cell, legacy sets and verified-link count."""
    bad = sorted({int(y) for y in frame["year"].dropna()} - set(YEARS)) + (
        ["none"] if frame["year"].isna().any() else []
    )
    if bad:
        raise CatalogError(
            f"population shots with campaign years outside {YEARS}: {bad}"
        )
    out = frame.copy()
    out["n_links_verified"] = out["shot"].map(lambda s: int(links.get(s, 0)))
    out["legacy_sets"] = out["shot"].map(lambda s: ";".join(legacy.get(s, ())))
    out["group"] = "R"
    out.loc[out["legacy_sets"].ne(""), "group"] = "G"
    out.loc[out["n_links_verified"].gt(0), "group"] = "L"
    out["cell"] = out["group"] + out["year"].astype(int).astype(str)
    return out


def _cells(frame: pd.DataFrame) -> dict[str, int]:
    """N per cell, all 15 of them, the empty ones at 0."""
    counts = frame["cell"].value_counts()
    return {f"{g}{y}": int(counts.get(f"{g}{y}", 0)) for g in GROUPS for y in YEARS}


def group_sizes(sizes: Mapping[str, int], n: int = COHORT_SIZE) -> dict[str, int]:
    """n per group: L and G up to their caps, R the rest, none over its population."""
    out = {g: min(sizes.get(g, 0), CAPS[g]) for g in CAPS}
    out["R"] = min(sizes.get("R", 0), n - sum(out.values()))
    return out


def draw(frame: pd.DataFrame, seed: int = SEED, n: int = COHORT_SIZE):
    """The drawn shots with weights, splits and queue, and `{cell: {"N", "n"}}`.

    `frame` is the population with groups (`assign_groups`).
    """
    cohort, table = _draw(frame, seed, n)
    return cohort[list(COHORT_COLUMNS)], table


def _draw(frame: pd.DataFrame, seed: int, n: int):
    """Derive the draw columns, retaining any other columns supplied by the caller."""
    cells = _cells(frame)
    by_group = frame["group"].value_counts().to_dict()
    per_group = group_sizes(by_group, n)
    if sum(per_group.values()) < n:
        sizes = {g: int(by_group.get(g, 0)) for g in GROUPS}
        raise CatalogError(
            f"cannot draw n={n}: N per group {sizes}, caps {CAPS}; "
            f"only {sum(per_group.values())} shots available under the caps"
        )
    alloc = {}
    for g in GROUPS:
        alloc |= allocate({c: cells[c] for c in cells if c[0] == g}, per_group[g])
    drawn = frame.assign(_d=frame["shot"].map(lambda s: key(seed, "draw", s)))
    drawn = drawn.sort_values(["cell", "_d"], kind="stable")
    picked = pd.concat(
        [part.head(alloc[cell]) for cell, part in drawn.groupby("cell", sort=True)]
    ).drop(columns="_d")
    picked["weight"] = picked["cell"].map(lambda c: cells[c] / alloc[c])
    picked["u"] = picked["shot"].map(lambda s: key(seed, "order", s))
    cohort = _split(picked)
    table = {c: {"N": cells[c], "n": alloc[c]} for c in cells}
    return cohort, table


def _split(picked: pd.DataFrame) -> pd.DataFrame:
    """Blind, val and train by u within each group, and the review queue."""
    counts = picked["group"].value_counts()
    sizes = {g: int(counts.get(g, 0)) for g in GROUPS}
    blind = allocate(sizes, BLIND_SIZE)
    val = allocate({g: sizes[g] - blind[g] for g in GROUPS}, VAL_SIZE)
    out = picked.sort_values("u", kind="stable").copy()
    rank = out.groupby("group").cumcount()
    edge_blind = out["group"].map(blind)
    edge_val = edge_blind + out["group"].map(val)
    out["split"] = "train"
    out.loc[rank < edge_val, "split"] = "val"
    out.loc[rank < edge_blind, "split"] = "test"
    out["blind"] = out["split"].eq("test")
    out = out.sort_values(["blind", "u"], ascending=[False, True], kind="stable")
    out["queue_rank"] = range(1, len(out) + 1)
    return out.sort_values("shot", ignore_index=True)


def _shot_id(value) -> int | None:
    if (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 1
        and value == int(value)
    ):
        return int(value)
    return None


def check_cohort(cohort: pd.DataFrame, cells: Mapping[str, int]) -> list[Finding]:
    """Check unique integral shots, group/year cells, weights and their cell sums.

    Also check u in [0, 1), queue ranks as a permutation with blind shots first,
    split/blind consistency, per-group split counts and u ordering across splits.
    `cells` maps each cell to its N (the manifest's `cells.<cell>.N`).
    Population membership, exact redraw, group provenance and shared scientific
    metadata (including window/flat-top validity) are left to `verify_cohort`.
    """
    where = "cohort"
    found = [
        Finding("unique", where, _shot_id(s), f"shot {s!r} appears twice")
        for s in cohort.loc[cohort["shot"].duplicated(), "shot"]
    ]
    counts = cohort["cell"].value_counts()
    duplicate_ranks = cohort["queue_rank"].duplicated(keep=False)
    n_blind = int(cohort["blind"].sum())
    for row, duplicate_rank in zip(cohort.itertuples(index=False), duplicate_ranks):
        shot = _shot_id(row.shot)
        if shot is None:
            found.append(
                Finding(
                    "shot",
                    where,
                    None,
                    f"shot {row.shot!r}: expected a finite whole number >= 1",
                )
            )
        if row.group not in GROUPS or row.cell != f"{row.group}{row.year}":
            found.append(
                Finding(
                    "cell",
                    where,
                    shot,
                    f"group {row.group!r}, cell {row.cell!r}, year {row.year}",
                )
            )
        expected = cells.get(row.cell)
        if not (row.weight >= 1.0) or (
            expected is not None
            and not math.isclose(row.weight, expected / counts[row.cell], rel_tol=1e-9)
        ):
            found.append(
                Finding(
                    "weight",
                    where,
                    shot,
                    f"weight {row.weight!r}; cell {row.cell} requires N / n",
                )
            )
        if not (0 <= row.u < 1):
            found.append(Finding("u", where, shot, f"u {row.u!r} is outside [0, 1)"))
        rank = _shot_id(row.queue_rank)
        if (
            rank is None
            or rank > len(cohort)
            or duplicate_rank
            or bool(row.blind) != (rank <= n_blind)
        ):
            found.append(
                Finding(
                    "queue_rank",
                    where,
                    shot,
                    f"rank {row.queue_rank!r}: expected a permutation of "
                    f"1..{len(cohort)}, blind ranks 1..{n_blind}",
                )
            )
        if row.split not in SPLITS or row.blind != (row.split == "test"):
            found.append(
                Finding(
                    "split",
                    where,
                    shot,
                    f"split {row.split!r}, blind {row.blind!r}",
                )
            )
    sums = cohort.groupby("cell")["weight"].sum()
    for cell, n_pop in sorted(cells.items()):
        got = float(sums.get(cell, 0.0))
        if not math.isclose(got, n_pop, rel_tol=1e-9, abs_tol=1e-9):
            found.append(
                Finding(
                    "weights",
                    where,
                    None,
                    f"cell {cell}: the weights sum to {got:g}, not N = {n_pop}",
                )
            )
    for cell in sorted(set(sums.index) - set(cells)):
        found.append(Finding("weights", where, None, f"cell {cell} has no N"))
    sizes = {g: int(cohort["group"].eq(g).sum()) for g in GROUPS}
    blind = allocate(sizes, BLIND_SIZE)
    val = allocate({g: sizes[g] - blind[g] for g in GROUPS}, VAL_SIZE)
    for group, part in cohort.groupby("group"):
        if group not in GROUPS:
            continue
        expected = {
            "test": blind[group],
            "val": val[group],
            "train": sizes[group] - blind[group] - val[group],
        }
        for split, n_split in expected.items():
            got = int(part["split"].eq(split).sum())
            if got != n_split:
                found.append(
                    Finding(
                        "split",
                        where,
                        _shot_id(part.iloc[0].shot),
                        f"group {group}: {split} has {got} shots, expected {n_split}",
                    )
                )
        preceding = float("-inf")
        for split in SPLITS:
            rows = part[part["split"].eq(split)]
            for row in rows[rows["u"] <= preceding].itertuples(index=False):
                found.append(
                    Finding(
                        "split",
                        where,
                        _shot_id(row.shot),
                        f"{split} u {row.u!r} is not above earlier splits "
                        f"in group {group}",
                    )
                )
            if not rows.empty:
                preceding = max(preceding, rows["u"].max())
    return found


def verify_cohort(
    cohort: pd.DataFrame,
    population: pd.DataFrame,
    cells: Mapping[str, int] | None = None,
    *,
    seed: int = SEED,
    n: int = COHORT_SIZE,
) -> list[Finding]:
    """Rederive the draw from the frozen population and seed, comparing exactly."""
    found = []
    for name, frame in (("population", population), ("cohort", cohort)):
        for shot in frame["shot"]:
            if _shot_id(shot) is None:
                found.append(Finding("shot", name, None, f"invalid shot {shot!r}"))
        for shot in frame.loc[frame["shot"].duplicated(), "shot"]:
            found.append(Finding("unique", name, _shot_id(shot), "appears twice"))
    if found:
        return found
    for name, frame in (("population", population), ("cohort", cohort)):
        for row in frame.itertuples(index=False):
            lo, hi = row.window_start_ms, row.window_end_ms
            for column, value in (
                ("window_start_ms", lo),
                ("window_end_ms", hi),
                ("flattop_s", row.flattop_s),
            ):
                valid = isinstance(value, Real) and math.isfinite(value)
                if column == "flattop_s":
                    valid = valid and value >= window.MIN_FLATTOP_S
                if not valid:
                    found.append(
                        Finding(
                            "measurement",
                            name,
                            int(row.shot),
                            f"{column}: invalid value {value!r}",
                        )
                    )
            if pd.notna(lo) and pd.notna(hi) and lo >= hi:
                found.append(
                    Finding(
                        "measurement",
                        name,
                        int(row.shot),
                        "window_end_ms must exceed window_start_ms",
                    )
                )
    for row in population.itertuples(index=False):
        group = "L" if row.n_links_verified > 0 else "G" if row.legacy_sets else "R"
        if row.group != group:
            found.append(
                Finding(
                    "group",
                    "population",
                    int(row.shot),
                    f"group {row.group!r} must be {group!r}",
                )
            )
        if row.cell != f"{group}{row.year}":
            found.append(
                Finding(
                    "cell",
                    "population",
                    int(row.shot),
                    f"cell {row.cell!r} must be {group}{row.year}",
                )
            )
    actual = set(cohort["shot"])
    eligible = set(population["shot"])
    marked = set(population.loc[population["in_cohort"], "shot"])
    for shots, detail in (
        (actual - eligible, "shot is outside the population"),
        (actual - marked, "shot is not marked in_cohort"),
        (marked - actual, "in_cohort shot is missing from the cohort"),
    ):
        found.extend(
            Finding("membership", "cohort", int(s), detail) for s in sorted(shots)
        )
    if cells is not None:
        counts = _cells(population)
        for cell in sorted(cells.keys() | counts.keys()):
            if cells.get(cell) != counts.get(cell):
                found.append(
                    Finding(
                        "population",
                        "population",
                        None,
                        f"cell {cell}: N {cells.get(cell)!r}, "
                        f"population has {counts.get(cell)!r}",
                    )
                )
    try:
        expected, _ = _draw(population, seed, n)
    except (ValueError, KeyError) as error:
        found.append(Finding("draw", "population", None, str(error)))
        return found
    expected = expected.set_index("shot")
    got = cohort.set_index("shot")
    for shot in sorted(set(expected.index) ^ actual):
        found.append(
            Finding("draw", "cohort", int(shot), "shot membership differs from redraw")
        )
    common = expected.index.intersection(got.index)
    columns = sorted((set(population.columns) & set(cohort.columns)) - {"shot"})
    columns += [c for c in DRAW_COLUMNS[1:] if c not in columns]
    for column in columns:
        a, b = got.loc[common, column], expected.loc[common, column]
        same = (a.eq(b) | (a.isna() & b.isna())).fillna(False)
        different = common[~same]
        for shot in different[:20]:
            found.append(
                Finding(
                    "draw",
                    "cohort",
                    int(shot),
                    f"{column}: {got.loc[shot, column]!r}, "
                    f"redraw {expected.loc[shot, column]!r}",
                )
            )
        if len(different) > 20:
            found.append(
                Finding(
                    "draw",
                    "cohort",
                    None,
                    f"{column}: {len(different)} differences, first 20 shown",
                )
            )
    return found


def read_cohort(path) -> pd.DataFrame:
    """Read the cohort with exact floats and strictly spelled booleans."""
    return _read_table(path, COHORT_COLUMNS, "blind")


def read_population(path) -> pd.DataFrame:
    """Read the frozen sampling frame with the same precision as the cohort."""
    return _read_table(path, POPULATION_COLUMNS, "in_cohort")


def _read_table(path, columns, boolean) -> pd.DataFrame:
    validate_csv_fields(path)
    strings = ("run_id", "legacy_sets", "split", "group", "cell", boolean)
    frame = pd.read_csv(
        path,
        dtype={c: str for c in strings},
        float_precision="round_trip",
        keep_default_na=False,
        na_values={c: [""] for c in columns if c not in strings},
    )
    if tuple(frame.columns) != columns:
        raise CatalogError(f"{path}: expected the columns {columns}")
    for row, value in enumerate(frame[boolean], 2):
        if value not in ("True", "False"):
            raise CatalogError(
                f"{path}: row {row}: {boolean} {value!r}, expected True or False"
            )
    frame[boolean] = frame[boolean].eq("True")
    return frame.astype({"window_start_ms": "Int64", "window_end_ms": "Int64"})


def _input(path: Path, root: Path | None = None, *, data: bytes | None = None) -> dict:
    """The path, relative to `root` when it lies under it, and its checksum."""
    path = path.resolve()
    root = root.resolve() if root is not None else None
    inside = root is not None and path.is_relative_to(root)
    return {
        "path": str(path.relative_to(root) if inside else path),
        "sha256": hashlib.sha256(
            path.read_bytes() if data is None else data
        ).hexdigest(),
    }


def manifest(
    frame, cohort, cells, *, seed, inputs, pool_meta, ip_log_meta, papers_meta, outputs
) -> dict:
    """What fixed the cohort: rules, seeds, inputs, counts, N and n per cell."""
    in_pop = frame[frame["reasons"].eq("")]
    groups = {}
    for g in GROUPS:
        mine = cohort[cohort["group"].eq(g)]
        groups[g] = {
            "N": int(in_pop["group"].eq(g).sum()),
            "n": len(mine),
            **{s: int(mine["split"].eq(s).sum()) for s in SPLITS},
        }
    return {
        "catalog": "DIII-D event catalog v1",
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(full=True),
        "git_dirty": git_dirty(),
        "seed": seed,
        "keys": "sha256 of '<seed>:<purpose>:<shot>', its first 8 bytes as a "
        "big-endian unsigned integer, / 2**64; "
        "purposes draw (the cell sample) and order (u)",
        "rules": {
            **pop.rules_record(),
            "rule_5": runaway.definition(),
            "groups": "L: >= 1 verified OSTI link; G: in a legacy table; R: the rest",
            "caps": dict(CAPS),
            "cohort_size": COHORT_SIZE,
            "allocation": "largest remainder across years within a group; "
            "remainder ties go to the first cell by name; each non-empty cell "
            "left at zero, in name order, takes one shot from the cell furthest "
            "above its quota among those with more than one shot "
            "(donor ties go to the last cell by name)",
            "blind_size": BLIND_SIZE,
            "val_size": VAL_SIZE,
        },
        "inputs": inputs,
        "pool": pool_meta,
        "ip_log_meta": ip_log_meta,
        "papers_meta": papers_meta,
        "outputs": outputs,
        "counts": {
            **pop.funnel(frame),
            "population": len(in_pop),
            "rejections": pop.rejections(frame),
            "groups": groups,
        },
        "cells": {
            c: {
                "N": v["N"],
                "n": v["n"],
                "weight": (v["N"] / v["n"]) if v["n"] else None,
            }
            for c, v in cells.items()
        },
        "replacements": [],
        "replacements_note": "rule 4 was measured on the whole pool before the draw, "
        "so no drawn shot can fail it",
    }


def _supersedes(path: Path, reason: str, current: dict) -> dict:
    try:
        data = path.read_bytes()
        old = yaml.safe_load(data)
    except (OSError, yaml.YAMLError) as error:
        raise CatalogError(f"{path}: invalid superseded manifest: {error}") from error
    if not isinstance(old, dict) or old.get("seed") != current["seed"]:
        raise CatalogError(f"{path}: superseded seed differs from the requested seed")
    for field in ("git_sha", "written_at", "inputs"):
        if field not in old:
            raise CatalogError(f"{path}: superseded manifest has no {field}")
    changed = [
        name
        for name in sorted(old["inputs"].keys() | current["inputs"].keys())
        if old["inputs"].get(name) != current["inputs"].get(name)
    ]
    for name, key in (
        ("pool_meta", "pool"),
        ("ip_log_meta", "ip_log_meta"),
        ("papers_meta", "papers_meta"),
    ):
        if old.get(key) != current.get(key):
            changed.append(name)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        **{k: old[k] for k in ("git_sha", "written_at", "seed")},
        "reason": reason,
        "changed_inputs": changed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m labeler.events.catalog.cohort",
        description="Draw the cohort from the population and write its manifest.",
    )
    parser.add_argument(
        "--pool", type=Path, help="default: $LABELER_ROOT/catalog/pool.csv"
    )
    parser.add_argument(
        "--ip-log", type=Path, help="default: $LABELER_ROOT/catalog/ip.jsonl"
    )
    parser.add_argument(
        "--papers", type=Path, help="default: <label tables>/catalog/papers.csv"
    )
    parser.add_argument(
        "--runaway", type=Path, help="default: runaway.csv beside papers.csv"
    )
    parser.add_argument("--out", type=Path, help="default: $LABELER_ROOT/catalog")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--supersedes", type=Path, help="manifest this freeze replaces")
    parser.add_argument("--reason", help="why this freeze replaces the previous one")
    args = parser.parse_args(argv)
    if args.supersedes and (not args.reason or not args.reason.strip()):
        parser.error("--supersedes requires a nonblank --reason")
    if args.reason and not args.supersedes:
        parser.error("--reason requires --supersedes")
    try:
        return _run(args, Paths.from_env())
    except (CatalogError, pd.errors.ParserError) as error:
        parser.error(str(error))


def _run(args, paths) -> int:
    pool_path = args.pool or paths.catalog / "pool.csv"
    log_path = args.ip_log or paths.catalog / "ip.jsonl"
    papers_path = args.papers or paths.label_tables / "catalog" / "papers.csv"
    runaway_path = args.runaway or papers_path.with_name("runaway.csv")
    for path in (pool_path, log_path, papers_path, runaway_path):
        if not path.is_file():
            raise CatalogError(f"{path} does not exist")
    meta_path = pool_path.with_suffix(".meta.json")
    ip_meta_path = log_path.with_suffix(".meta.json")
    papers_meta_path = papers_path.with_suffix(".meta.json")
    for path in (meta_path, ip_meta_path, papers_meta_path):
        if not path.is_file():
            raise CatalogError(f"{path}: missing provenance record")
    pool_meta = json.loads(meta_path.read_bytes())
    ip_log_meta = json.loads(ip_meta_path.read_bytes())
    papers_meta_data = papers_meta_path.read_bytes()
    papers_record = json.loads(papers_meta_data)
    pool_data, log_data, papers_data = (
        path.read_bytes() for path in (pool_path, log_path, papers_path)
    )
    pool_input = _input(pool_path, data=pool_data)
    log_input = _input(log_path, data=log_data)
    if pool_meta.get("pool_sha256") != pool_input["sha256"]:
        raise CatalogError(f"{meta_path}: pool_sha256 differs from {pool_path}")
    if pool_meta.get("rules") != pop.rules_record():
        raise CatalogError(f"{meta_path}: rules differ from population.rules_record()")
    pool = pop.read_pool(NamedBytes(pool_data, pool_path))
    for field, expected in {
        "definition": window.definition(),
        "version": window.LOG_VERSION,
        "shot_file_sha256": pool_meta.get("pool_shots_sha256"),
        "shots": int(pop.passes_screen(pool).sum()),
    }.items():
        if expected is None or ip_log_meta.get(field) != expected:
            raise CatalogError(f"{ip_meta_path}: {field} differs from the pool/window")
    if (
        papers_record.get("outputs", {}).get("papers.csv")
        != hashlib.sha256(papers_data).hexdigest()
    ):
        raise CatalogError(f"{papers_meta_path}: papers.csv sha256 differs")
    papers_meta = {
        "sha256": hashlib.sha256(papers_meta_data).hexdigest(),
        **{k: papers_record.get(k) for k in ("git_sha", "written_at", "summary")},
    }
    runaway_data = runaway_path.read_bytes()
    runaway_meta_path = runaway_path.with_suffix(".meta.json")
    try:
        runaway_meta_data = runaway_meta_path.read_bytes()
        runaway_record = json.loads(runaway_meta_data)
        if (
            runaway_record.get("outputs", {}).get("runaway.csv")
            != hashlib.sha256(runaway_data).hexdigest()
        ):
            raise ValueError("runaway.csv sha256 differs")
        scan_inputs = runaway_record.get("inputs", {})
        for name, field, expected in (
            ("pool", "sha256", pool_input["sha256"]),
            ("ip_log", "sha256", log_input["sha256"]),
            ("ip_log", "version", window.LOG_VERSION),
        ):
            if scan_inputs.get(name, {}).get(field) != expected:
                raise ValueError(f"inputs.{name}.{field} differs from the pool/Ip log")
        if {
            field: runaway_record.get(field) for field in runaway.definition()
        } != runaway.definition():
            raise ValueError("definition differs from runaway.definition()")
    except (OSError, ValueError, AttributeError) as error:
        raise CatalogError(f"{runaway_meta_path}: {error}") from error
    frame = pop.population(pool, window.read_log(NamedBytes(log_data, log_path)))
    frame = runaway.apply_rule(
        frame,
        runaway.read_runaway(NamedBytes(runaway_data, runaway_path)),
        runaway_path,
    )
    legacy_inputs = []
    legacy, _ = legacy_sets(paths.label_tables, inputs=legacy_inputs)
    grouped = assign_groups(
        frame[frame["reasons"].eq("")],
        osti_links(read_papers(NamedBytes(papers_data, papers_path))),
        legacy,
    )
    cohort, cells = draw(grouped, args.seed)
    grouped["in_cohort"] = grouped["shot"].isin(cohort["shot"])
    tables = {
        "cohort.csv": cohort.to_csv(index=False),
        "population.csv": grouped[list(POPULATION_COLUMNS)].to_csv(index=False),
    }
    saved_cohort = read_cohort(io.StringIO(tables["cohort.csv"]))
    saved_population = read_population(io.StringIO(tables["population.csv"]))
    n_pop = {c: v["N"] for c, v in cells.items()}
    require(
        check_cohort(saved_cohort, n_pop)
        + verify_cohort(saved_cohort, saved_population, n_pop, seed=args.seed)
    )
    inputs = {
        "pool": pool_input,
        "ip_log": log_input,
        "papers": _input(papers_path, paths.label_tables, data=papers_data),
        "runaway": {
            **_input(runaway_path, paths.label_tables, data=runaway_data),
            "meta_sha256": hashlib.sha256(runaway_meta_data).hexdigest(),
            **{k: runaway_record.get(k) for k in ("git_sha", "counts")},
        },
        "legacy_tables": legacy_inputs,
    }
    frame = frame.merge(grouped[["shot", "group", "cell"]], on="shot", how="left")
    encoded = {name: text.encode("utf-8") for name, text in tables.items()}
    doc = manifest(
        frame,
        cohort,
        cells,
        seed=args.seed,
        inputs=inputs,
        pool_meta=pool_meta,
        ip_log_meta=ip_log_meta,
        papers_meta=papers_meta,
        outputs={
            name: hashlib.sha256(data).hexdigest() for name, data in encoded.items()
        },
    )
    if args.supersedes:
        doc["supersedes"] = _supersedes(args.supersedes, args.reason.strip(), doc)
    out = args.out or paths.catalog
    out.mkdir(parents=True, exist_ok=True)
    for name, data in encoded.items():
        with atomic_path(out / name) as tmp:
            Path(tmp).write_bytes(data)
    with atomic_path(out / "cohort_manifest.yaml") as tmp:
        Path(tmp).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                **doc["counts"]["groups"],
                "population": doc["counts"]["population"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
