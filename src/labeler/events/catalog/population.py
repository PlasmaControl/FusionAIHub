"""The population: the corpus shots 185,601-204,999 that pass four rules.

Rules 1-3 use `shot_design.shotdb.select.eligible()` with its text-length clause
dropped and its title filter replaced by machine time; the rejections keep its names:

1. the shot-table row: SHOT_TYPE plasma, |IP| >= 0.5 MA, PULSE-LENGTH >= 2 s, and
   PBEAM-MAX >= 1 MW or PECH-MAX > 0 (`shot_type`, `ip`, `pulse_length`, `heating`);
2. the shot's own shot-table block, not the session fallback, under a run title that
   names no machine activity (`session_fallback`, `title`): startup and power-system
   test titles, or the lexicon's machine theme after physics themes have been tried
   first against the title and mini-proposal subject;
3. at least 2 s of census coverage for mhr, ece and filterscopes (`census_<group>`),
   in effect presence in this corpus: mhr spans 4.194 s and ece 6.193 s when present.

A shot with no text bundle fails them all, as `no_bundle`; a bundle with another
shot's table row fails as `bundle_mismatch`.

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
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shot_design.shotdb import select
from shot_design.shotdb.text import shot_table_row

from ...catalog import corpus_shots, write_shot_file
from ...config import Paths, atomic_path, git_sha, sha256_of
from . import window
from .check import CatalogError

FIRST_SHOT, LAST_SHOT = 185_601, 204_999
MAX_IP_DT_MS = 0.5
MACHINE_TITLE = re.compile(
    r"(?i)^\s*(?:plasma\s+)?(?:start-?up|starup)\b|^\s*power\s+systems?\s+test"
)
#: Census groups whose spans a pool row carries; the first three are rule 3's.
SPAN_GROUPS = ("mhr", "ece", "filterscopes", "co2", "sxr", "mirnov")
POOL_COLUMNS = (
    "shot",
    "year",
    "run_id",
    "shot_type",
    "has_shot_table",
    "title",
    "mp_subject",
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
    "ip_dt_ms",
)
#: The rules in the order the funnel applies them, and the rejections each one makes.
RULES = {
    "bundle": ("no_bundle", "bundle_mismatch"),
    "rule_1": ("shot_type", "ip", "pulse_length", "heating"),
    "rule_2": ("session_fallback", "title"),
    "rule_3": tuple(f"census_{group}" for group in select.CENSUS_GROUPS),
    "rule_4": ("flattop", "no_plasma", "flattop_unmeasured"),
}


def machine_time(title, subject, themes=None) -> bool:
    """An explicit machine title or the lexicon's physics-first machine theme."""
    if themes is None:
        themes = select.lexicon_themes()
    return bool(MACHINE_TITLE.search(title or "")) or (
        select.assign_theme(title, themes, subject=subject) == select.FALLBACK_THEME
    )


def screen(
    shots: Iterable[int],
    bundles: Mapping[int, str],
    spans: Mapping[int, Mapping[str, float]],
    themes=None,
) -> pd.DataFrame:
    """One row per shot: its shot-table values, census spans and the rules 1-3 it fails.

    `reasons` joins the rejections with ";" and is empty for a shot that passes.
    Rule 2 rejects explicit startup/power-system test titles and the lexicon's
    machine theme, assigned physics first from title and mini-proposal subject.
    """
    if themes is None:
        themes = select.lexicon_themes()
    rows = []
    for shot in sorted(set(shots)):
        if shot not in bundles:
            rows.append({"shot": shot, "reasons": "no_bundle"})
            continue
        table_row = shot_table_row(bundles[shot])
        if table_row and table_row.get("SHOT", "").strip() != str(shot):
            rows.append({"shot": shot, "reasons": "bundle_mismatch"})
            continue
        facts = select.parse_facts(shot, bundles[shot])
        missing = {
            field: None
            for field in ("ip_ma", "pulse_length_s", "pbeam_max_mw", "pech_max_mw")
            if (value := getattr(facts, field)) is not None and not math.isfinite(value)
        }
        facts = replace(facts, **missing)
        found = spans.get(shot, {})
        _, why = select.eligible(facts, found, min_shot_chars=0)
        # Rule 4 uses measured Ip, and rule 2 uses machine time instead of words
        # such as "test", which occur in physics titles as well.
        reasons = set(why) - {"flattop", "title"}
        if machine_time(facts.title, facts.mp_subject, themes):
            reasons.add("title")
        rows.append(
            {
                "shot": shot,
                "year": facts.year,
                "run_id": facts.run_id,
                "shot_type": facts.shot_type,
                "has_shot_table": facts.has_shot_table,
                "title": facts.title,
                "mp_subject": facts.mp_subject,
                "ip_ma": facts.ip_ma,
                "pulse_length_s": facts.pulse_length_s,
                "pbeam_max_mw": facts.pbeam_max_mw,
                "pech_max_mw": facts.pech_max_mw,
                **{f"span_{g}_s": round(found.get(g, 0.0), 3) for g in SPAN_GROUPS},
                "reasons": ";".join(
                    r for rule in RULES.values() for r in rule if r in reasons
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=list(POOL_COLUMNS))
    return frame.astype(
        {
            "shot": "int64",
            "year": "Int64",
            "run_id": object,
            "has_shot_table": "boolean",
        }
    )


def read_pool(path) -> pd.DataFrame:
    """`pool.csv` as `screen` wrote it.

    An empty `reasons` means the shot passes; any other empty value is missing.
    """
    frame = pd.read_csv(
        path,
        dtype={
            **{
                c: str
                for c in ("run_id", "shot_type", "title", "mp_subject", "reasons")
            },
            "has_shot_table": "boolean",
        },
        keep_default_na=False,
        na_values={c: [""] for c in POOL_COLUMNS if c != "reasons"},
    )
    if tuple(frame.columns) != POOL_COLUMNS:
        raise CatalogError(f"{path}: expected the columns {POOL_COLUMNS}")
    duplicates = frame.loc[frame["shot"].duplicated(), "shot"]
    if not duplicates.empty:
        raise CatalogError(f"{path}: duplicate shot {duplicates.iloc[0]}")
    _reason_sets(frame)
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
    measured = log.loc[pool.loc[judged, "shot"]]
    invalid = measured.index[~measured["status"].isin((*window.SETTLED, "error"))]
    if len(invalid):
        raise CatalogError(f"invalid Ip status for shots {invalid.tolist()[:5]}")
    versions = measured.get("version", pd.Series(1, index=measured.index))
    stale = measured.index[~versions.eq(window.LOG_VERSION).fillna(False)]
    if len(stale):
        raise CatalogError(
            f"{len(stale)} shots have an Ip log version other than "
            f"{window.LOG_VERSION} (first: {stale.tolist()[:5]}); "
            "run catalog.window on pool_shots.txt first"
        )
    frame = pool.copy()
    for column in RULE4_COLUMNS:
        name = {"ip_status": "status", "ip_dt_ms": "dt_ms"}.get(column, column)
        source = log.get(name, pd.Series(dtype=float))
        frame[column] = frame["shot"].map(source).where(judged)
    status, flattop = frame["ip_status"], frame["flattop_s"].astype(float)
    dt = frame["ip_dt_ms"].astype(float)
    rule4 = np.select(
        [
            status.eq("error"),
            status.eq("no_plasma"),
            flattop.isna() | dt.isna() | dt.gt(MAX_IP_DT_MS),
            flattop < select.MIN_FLATTOP_S,
        ],
        ["flattop_unmeasured", "no_plasma", "flattop_unmeasured", "flattop"],
        "",
    )
    frame.loc[judged, "reasons"] = rule4[judged.to_numpy()]
    return frame.astype({"window_start_ms": "Int64", "window_end_ms": "Int64"})


def _reason_sets(frame: pd.DataFrame) -> list[set[str]]:
    """Validate rejection codes before any count can hide an unknown rule."""
    known = {reason for rule in RULES.values() for reason in rule}
    failed = []
    for shot, text in zip(frame["shot"], frame["reasons"], strict=True):
        reasons = [reason for reason in text.split(";") if reason]
        for reason in reasons:
            if reason not in known:
                raise CatalogError(f"shot {shot}: unknown rejection {reason!r}")
        failed.append(set(reasons))
    return failed


def rejections(frame: pd.DataFrame) -> dict[str, int]:
    """How often each rule rejected a shot; a shot counts under every rule it fails."""
    counts = Counter(r for reasons in _reason_sets(frame) for r in reasons)
    order = [r for rule in RULES.values() for r in rule]
    return {r: counts[r] for r in order}


def funnel(frame: pd.DataFrame) -> dict[str, int]:
    """The shots left after each rule in turn: the selection table's rows."""
    failed = _reason_sets(frame)
    left, applied = {"corpus_in_range": len(frame)}, set()
    for name, rule in RULES.items():
        applied |= set(rule)
        left[f"after_{name}"] = sum(applied.isdisjoint(f) for f in failed)
    return left


def rules_record() -> dict:
    """The population's rules and deliberately dropped eligibility clauses."""
    return {
        "shot_range": [FIRST_SHOT, LAST_SHOT],
        "rule_1": {
            "shot_type": "plasma",
            "min_abs_ip_ma": select.MIN_IP_MA,
            "min_pulse_length_s": select.MIN_PULSE_LENGTH_S,
            "min_pbeam_mw": select.MIN_PBEAM_MW,
            "pech_max_mw_gt": 0.0,
            "heating": "PBEAM-MAX >= min_pbeam_mw or PECH-MAX > pech_max_mw_gt",
        },
        "rule_2": {
            "own_shot_table_block": True,
            "machine_time": {
                "pattern": MACHINE_TITLE.pattern,
                "lexicon_theme": select.FALLBACK_THEME,
                "assignment": "physics themes first, title and mini-proposal subject",
            },
        },
        "rule_3": {
            "groups": list(select.CENSUS_GROUPS),
            "min_group_span_s": select.MIN_GROUP_SPAN_S,
            "corpus_effect": (
                "In effect presence: mhr spans 4.194 s and ece 6.193 s when present."
            ),
        },
        "rule_4": window.definition()
        | {
            "max_ip_dt_ms": MAX_IP_DT_MS,
            "measured_on": "every shot passing rules 1-3",
        },
        "dropped": {
            "shot_text": {"min_shot_chars": 0},
            "flattop": "select.eligible pulse-length flattop proxy",
            "title": select.TITLE_EXCLUDE.pattern,
        },
    }


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
    bundles = select.read_bundles(paths.text_root, shots)
    pool = screen(shots, bundles, select.spans(census))
    left = funnel(pool)
    if left["after_rule_3"] != int(passes_screen(pool).sum()):
        raise CatalogError("funnel after_rule_3 disagrees with passes_screen(pool)")
    del left["after_rule_4"]  # measured later, from the Ip log
    rejected = {
        code: count
        for code, count in rejections(pool).items()
        if code not in RULES["rule_4"]
    }
    out.mkdir(parents=True, exist_ok=True)
    with atomic_path(out / "pool.csv") as tmp:
        pool.to_csv(tmp, index=False)
    write_shot_file(out / "pool_shots.txt", pool.loc[passes_screen(pool), "shot"])
    inputs = pd.DataFrame(
        [
            {
                "shot": shot,
                "bundle_sha256": (
                    sha256_of(paths.text_root / f"shot_{shot}.txt")
                    if shot in bundles
                    else ""
                ),
            }
            for shot in shots
        ],
        columns=["shot", "bundle_sha256"],
    )
    with atomic_path(out / "pool_inputs.csv") as tmp:
        inputs.to_csv(tmp, index=False)
    from shot_design.config import CONFIG_DIR

    lexicon = CONFIG_DIR / "labels.yaml"
    meta = {
        "census": str(args.census),
        "census_sha256": sha256_of(args.census),
        "text_root": str(paths.text_root),
        "corpus": str(paths.corpus),
        "git_sha": git_sha(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pool_sha256": sha256_of(out / "pool.csv"),
        "pool_shots_sha256": sha256_of(out / "pool_shots.txt"),
        "pool_inputs_sha256": sha256_of(out / "pool_inputs.csv"),
        "lexicon": str(lexicon),
        "lexicon_sha256": sha256_of(lexicon),
        "corpus_in_range": len(shots),
        "bundles": len(bundles),
        "funnel": left,
        "rejections": rejected,
        "rules": rules_record(),
    }
    with atomic_path(out / "pool.meta.json") as tmp:
        Path(tmp).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    summary = {"out": str(out), **left, "rejections": rejected}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
