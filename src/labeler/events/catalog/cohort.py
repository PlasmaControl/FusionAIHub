"""The cohort: 500 population shots drawn with known weights.

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

reads `$LABELER_ROOT/catalog/{pool.csv, ip.jsonl}`, `data/events/catalog/papers.csv`
and the legacy tables, and writes `cohort.csv`, `cohort_manifest.yaml` and
`population.csv` to `$LABELER_ROOT/catalog/`. Copying the first two into
`data/events/catalog/` freezes the cohort, and that is the owner's call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pandas as pd
import yaml

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...literature.papers import read_papers
from .. import databases
from . import population as pop
from . import window
from .check import CatalogError, Finding, require

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


def legacy_sets(root) -> tuple[dict[int, tuple[str, ...]], list[databases.TableSpec]]:
    """`{shot: the directories of the legacy tables naming it}`, and the tables read."""
    specs = list(databases.load_manifest(root))
    named: dict[int, set[str]] = defaultdict(set)
    for spec in specs:
        for shot in databases.shots(spec, root):
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
    cells = _cells(frame)
    by_group = frame["group"].value_counts().to_dict()
    per_group = group_sizes(by_group, n)
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
    return out.sort_values("shot", ignore_index=True)[list(COHORT_COLUMNS)]


def check_cohort(cohort: pd.DataFrame, cells: Mapping[str, int]) -> list[Finding]:
    """Every shot has a group, cell, weight and split, and the weights reproduce N.

    `cells` maps each cell to its N (the manifest's `cells.<cell>.N`).
    """
    where = "cohort"
    found = [
        Finding("unique", where, int(s), "appears twice")
        for s in cohort.loc[cohort["shot"].duplicated(), "shot"]
    ]
    for row in cohort.itertuples(index=False):
        if row.group not in GROUPS or row.cell != f"{row.group}{row.year}":
            found.append(
                Finding(
                    "cell",
                    where,
                    int(row.shot),
                    f"group {row.group!r}, cell {row.cell!r}, year {row.year}",
                )
            )
        if not (row.weight >= 1.0):
            found.append(
                Finding("weight", where, int(row.shot), f"weight {row.weight!r}")
            )
        if row.split not in SPLITS or row.blind != (row.split == "test"):
            found.append(
                Finding(
                    "split",
                    where,
                    int(row.shot),
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
    return found


def read_cohort(path) -> pd.DataFrame:
    """Read the cohort with exact floats and strictly spelled booleans."""
    return _read_table(path, COHORT_COLUMNS, "blind")


def read_population(path) -> pd.DataFrame:
    """Read the frozen sampling frame with the same precision as the cohort."""
    return _read_table(path, POPULATION_COLUMNS, "in_cohort")


def _read_table(path, columns, boolean) -> pd.DataFrame:
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


def _input(path: Path, root: Path | None = None) -> dict:
    """The path, relative to `root` when it lies under it, and its checksum."""
    inside = root is not None and path.is_relative_to(root)
    return {
        "path": str(path.relative_to(root) if inside else path),
        "sha256": sha256_of(path),
    }


def manifest(frame, cohort, cells, *, seed, inputs, pool_meta) -> dict:
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
        "git_sha": git_sha(),
        "seed": seed,
        "keys": "sha256 of '<seed>:<purpose>:<shot>', first 8 bytes / 2**64; "
        "purposes draw (the cell sample) and order (u)",
        "rules": {
            **pop.rules_record(),
            "groups": "L: >= 1 verified OSTI link; G: in a legacy table; R: the rest",
            "caps": dict(CAPS),
            "cohort_size": COHORT_SIZE,
            "allocation": "largest remainder across years within a group, "
            ">= 1 per non-empty cell",
            "blind_size": BLIND_SIZE,
            "val_size": VAL_SIZE,
        },
        "inputs": inputs,
        "pool": pool_meta,
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
    parser.add_argument("--out", type=Path, help="default: $LABELER_ROOT/catalog")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    pool_path = args.pool or paths.catalog / "pool.csv"
    log_path = args.ip_log or paths.catalog / "ip.jsonl"
    papers_path = args.papers or paths.label_tables / "catalog" / "papers.csv"
    for path in (pool_path, log_path, papers_path):
        if not path.is_file():
            parser.error(f"{path} does not exist")
    frame = pop.population(pop.read_pool(pool_path), window.read_log(log_path))
    legacy, specs = legacy_sets(paths.label_tables)
    grouped = assign_groups(
        frame[frame["reasons"].eq("")], osti_links(read_papers(papers_path)), legacy
    )
    cohort, cells = draw(grouped, args.seed)
    require(check_cohort(cohort, {c: v["N"] for c, v in cells.items()}))
    meta_path = pool_path.with_suffix(".meta.json")
    inputs = {
        "pool": _input(pool_path),
        "ip_log": _input(log_path),
        "papers": _input(papers_path, paths.label_tables),
        "legacy_tables": [
            _input(s.path(paths.label_tables), paths.label_tables) for s in specs
        ],
    }
    pool_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    frame = frame.merge(grouped[["shot", "group", "cell"]], on="shot", how="left")
    doc = manifest(
        frame, cohort, cells, seed=args.seed, inputs=inputs, pool_meta=pool_meta
    )
    out = args.out or paths.catalog
    out.mkdir(parents=True, exist_ok=True)
    grouped["in_cohort"] = grouped["shot"].isin(cohort["shot"])
    for name, table in (
        ("cohort.csv", cohort),
        ("population.csv", grouped[list(POPULATION_COLUMNS)]),
    ):
        with atomic_path(out / name) as tmp:
            table.to_csv(tmp, index=False)
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
