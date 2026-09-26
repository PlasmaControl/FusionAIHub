"""The population: the corpus shots 185,601-204,999 that pass four rules.

Rules 1-3 are `shot_design.shotdb.select.eligible()` with its text-length clause
dropped, and the rejections keep its names:

1. the shot-table row: SHOT_TYPE plasma, |IP| >= 0.5 MA, PULSE-LENGTH >= 2 s, and
   PBEAM-MAX >= 1 MW or PECH-MAX > 0 (`shot_type`, `ip`, `pulse_length`, `heating`);
2. the shot's own shot-table block, not the session fallback, under a run title that
   names no machine activity (`session_fallback`, `title`);
3. at least 2 s of census coverage for mhr, ece and filterscopes (`census_<group>`).

A shot with no text bundle fails them all, as `no_bundle`.

Rule 4, an Ip flat-top of at least 1 s, is measured from high-rate Ip
(`catalog.window`) for every shot that passes rules 1-3, not only for the drawn ones,
so N per cell is exact and no drawn shot is ever replaced. Its rejections are
`flattop` (under 1 s), `no_plasma` (|Ip| never reached 50 kA) and
`flattop_unmeasured` (the fetch kept failing).

    pixi run -e labelmaker python -m labeler.events.catalog.population

writes `$LABELER_ROOT/catalog/pool.csv`, every corpus shot in range with the rules it
fails; `pool_shots.txt`, the shots that pass rules 1-3, for the Ip fetch; and
`pool.meta.json`, the inputs it read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shot_design.shotdb import select

from ...catalog import corpus_shots, write_shot_file
from ...config import Paths, atomic_path, git_sha, sha256_of
from .check import CatalogError

FIRST_SHOT, LAST_SHOT = 185_601, 204_999
#: Census groups whose spans a pool row carries; the first three are rule 3's.
SPAN_GROUPS = ("mhr", "ece", "filterscopes", "co2", "sxr", "mirnov")
POOL_COLUMNS = (
    "shot",
    "year",
    "run_id",
    "ip_ma",
    "pulse_length_s",
    "pbeam_max_mw",
    "pech_max_mw",
    *(f"span_{group}_s" for group in SPAN_GROUPS),
    "reasons",
)
RULE4_COLUMNS = (
    "ip_status",
    "window_start_ms",
    "window_end_ms",
    "flattop_s",
    "ip_peak_ma",
)
#: The rules in the order the funnel applies them, and the rejections each one makes.
RULES = {
    "bundle": ("no_bundle",),
    "rule_1": ("shot_type", "ip", "pulse_length", "heating"),
    "rule_2": ("session_fallback", "title"),
    "rule_3": tuple(f"census_{group}" for group in select.CENSUS_GROUPS),
    "rule_4": ("flattop", "no_plasma", "flattop_unmeasured"),
}


def screen(
    shots: Iterable[int],
    bundles: Mapping[int, str],
    spans: Mapping[int, Mapping[str, float]],
) -> pd.DataFrame:
    """One row per shot: its shot-table values, census spans and the rules 1-3 it fails.

    `reasons` joins the rejections with ";" and is empty for a shot that passes.
    """
    rows = []
    for shot in sorted(set(shots)):
        if shot not in bundles:
            rows.append({"shot": shot, "reasons": "no_bundle"})
            continue
        facts = select.parse_facts(shot, bundles[shot])
        found = spans.get(shot, {})
        _, why = select.eligible(facts, found, min_shot_chars=0)
        rows.append(
            {
                "shot": shot,
                "year": facts.year,
                "run_id": facts.run_id,
                "ip_ma": facts.ip_ma,
                "pulse_length_s": facts.pulse_length_s,
                "pbeam_max_mw": facts.pbeam_max_mw,
                "pech_max_mw": facts.pech_max_mw,
                **{f"span_{g}_s": round(found.get(g, 0.0), 3) for g in SPAN_GROUPS},
                # rule 4 is measured from Ip, not from the pulse-length proxy
                "reasons": ";".join(r for r in why if r != "flattop"),
            }
        )
    frame = pd.DataFrame(rows, columns=list(POOL_COLUMNS))
    return frame.astype({"shot": "int64", "year": "Int64", "run_id": object})


def read_pool(path) -> pd.DataFrame:
    """`pool.csv` as `screen` wrote it.

    An empty `reasons` means the shot passes; any other empty value is missing.
    """
    frame = pd.read_csv(
        path,
        dtype={"run_id": str, "reasons": str},
        keep_default_na=False,
        na_values={c: [""] for c in POOL_COLUMNS if c != "reasons"},
    )
    if tuple(frame.columns) != POOL_COLUMNS:
        raise CatalogError(f"{path}: expected the columns {POOL_COLUMNS}")
    return frame.astype({"year": "Int64"})


def passes_screen(pool: pd.DataFrame) -> pd.Series:
    return pool["reasons"].eq("")


def population(pool: pd.DataFrame, ip_log: pd.DataFrame) -> pd.DataFrame:
    """`pool` with rule 4 applied from the Ip log (`window.read_log`).

    Every shot that passes rules 1-3 gains its Ip status, window and flat-top, and its
    `reasons` become rule 4's; the population is the rows whose `reasons` is empty.
    `CatalogError` if any of those shots has no line in the log.
    """
    judged = passes_screen(pool)
    log = ip_log.set_index("shot")
    missing = sorted(set(pool.loc[judged, "shot"]) - set(log.index))
    if missing:
        raise CatalogError(
            f"{len(missing)} shots pass rules 1-3 but have no Ip log line (first: "
            f"{missing[:5]}); run catalog.window on pool_shots.txt first"
        )
    frame = pool.copy()
    for column in RULE4_COLUMNS:
        source = log["status" if column == "ip_status" else column]
        frame[column] = frame["shot"].map(source).where(judged)
    status, flattop = frame["ip_status"], frame["flattop_s"].astype(float)
    rule4 = np.select(
        [
            status.eq("error"),
            status.eq("no_plasma"),
            ~(flattop >= select.MIN_FLATTOP_S),
        ],
        ["flattop_unmeasured", "no_plasma", "flattop"],
        "",
    )
    frame.loc[judged, "reasons"] = rule4[judged.to_numpy()]
    return frame.astype({"window_start_ms": "Int64", "window_end_ms": "Int64"})


def rejections(frame: pd.DataFrame) -> dict[str, int]:
    """How often each rule rejected a shot; a shot counts under every rule it fails."""
    counts = Counter(r for text in frame["reasons"] for r in text.split(";") if r)
    order = [r for rule in RULES.values() for r in rule]
    return {r: counts[r] for r in order if counts[r]}


def funnel(frame: pd.DataFrame) -> dict[str, int]:
    """The shots left after each rule in turn: the selection table's rows."""
    failed = [set(text.split(";")) for text in frame["reasons"]]
    left, applied = {"corpus_in_range": len(frame)}, set()
    for name, rule in RULES.items():
        applied |= set(rule)
        left[f"after_{name}"] = sum(applied.isdisjoint(f) for f in failed)
    return left


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m labeler.events.catalog.population",
        description="Screen the corpus shots in range against rules 1-3.",
    )
    parser.add_argument(
        "--census",
        type=Path,
        help="default: shot_design's <db_dir>/corpus_coverage.parquet",
    )
    parser.add_argument("--out", type=Path, help="default: $LABELER_ROOT/catalog")
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    if args.census is None:
        from shot_design.config import load_paths

        args.census = load_paths().db_dir / "corpus_coverage.parquet"
    out = args.out or paths.catalog
    shots = [s for s in corpus_shots(paths) if FIRST_SHOT <= s <= LAST_SHOT]
    census = pd.read_parquet(
        args.census, columns=["shot", "group", "present", "t0_s", "t1_s"]
    )
    pool = screen(
        shots, select.read_bundles(paths.text_root, shots), select.spans(census)
    )
    out.mkdir(parents=True, exist_ok=True)
    with atomic_path(out / "pool.csv") as tmp:
        pool.to_csv(tmp, index=False)
    write_shot_file(out / "pool_shots.txt", pool.loc[passes_screen(pool), "shot"])
    meta = {
        "census": str(args.census),
        "census_sha256": sha256_of(args.census),
        "text_root": str(paths.text_root),
        "corpus": str(paths.corpus),
        "git_sha": git_sha(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(out / "pool.meta.json") as tmp:
        Path(tmp).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    left = funnel(pool)
    del left["after_rule_4"]  # measured later, from the Ip log
    summary = {"out": str(out), **left, "rejections": rejections(pool)}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
