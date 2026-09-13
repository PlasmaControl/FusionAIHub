"""Warm-cache timings for the five operations the plan sets a budget on.

The plan's Appendix B budget table (§B8) is the contract:

| operation | budget |
| --- | --- |
| `load` a built database | < 3 s |
| `search` with `--text` | < 400 ms |
| `search` without text (a reference shot) | < 200 ms |
| `phenomenon locate` | < 300 ms |
| `describe` one shot | (no budget stated) |

Three things about how this measures, because each of them changes what the number means:

* **Warm.** Every operation is run `WARMUP` times before the clock starts, and those runs are
  thrown away. The first `--text` search loads MiniLM (~5 s offline, minutes on a node that can
  reach out to huggingface and shouldn't), the first `ShotDB.load` faults the parquet files into
  the page cache, and neither is what a budget on an interactive query is about. A cold number is
  a different measurement and this module does not claim to make it.
* **Median and p95, not mean.** A mean over 20 samples is one GC pause away from being a
  different number; the median is what a user feels and the p95 is what they complain about.
* **The verdict is on the median, and a row with no budget gets no verdict.** `describe` has no
  Appendix B number, so it reports `n/a` rather than a free PASS -- a table full of passes, one of
  which was never tested against anything, is worse than an honest gap.

Nothing here is tuned to pass. If a row says FAIL it is a finding, and the report says by how much.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..schema import LatencyReport, LatencyRow, QueryState

__all__ = ["BUDGETS_S", "DEFAULT_REPEATS", "WARMUP", "markdown", "measure", "time_it"]

# The plan's Appendix B numbers, in seconds. None = the plan sets none.
BUDGETS_S: dict[str, float | None] = {
    "load": 3.0,
    "search_text": 0.4,
    "search_no_text": 0.2,
    "phenomenon_locate": 0.3,
    "describe": None,
}

WHAT = {
    "load": "`ShotDB.load` of the built database",
    "search_text": "`rank.search` with a free-text query",
    "search_no_text": "`rank.search` from a reference shot, no text",
    "phenomenon_locate": "`phenomena.locate` over the whole database",
    "describe": "`describe.describe` of one shot's flat top",
}

DEFAULT_REPEATS = 20
WARMUP = 1

# What is typed and what is looked for, when the caller says nothing. The text is one of the
# frozen evalset's own prompts, so the timed query is a query somebody would really make; `elm` is
# the phenomenon with by far the most rows, which is the slow case rather than the flattering one.
DEFAULT_TEXT = "ELM suppression with n=3 RMP"
DEFAULT_PHENOMENON = "elm"


def time_it(fn: Callable[[], object], repeats: int, warmup: int = WARMUP) -> list[float]:
    """`repeats` wall-clock samples of `fn`, after `warmup` throwaway calls.

    `perf_counter`, not `process_time`: a database load is mostly the kernel reading files, and a
    CPU-time number would report that as free.
    """
    for _ in range(max(0, warmup)):
        fn()
    out = []
    for _ in range(max(0, repeats)):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _row(name: str, samples: list[float], repeats: int) -> LatencyRow:
    arr = np.asarray(samples, dtype=float)
    return LatencyRow(
        name=name,
        what=WHAT.get(name, name),
        repeats=repeats,
        median_s=float(np.median(arr)) if arr.size else float("nan"),
        p95_s=float(np.percentile(arr, 95)) if arr.size else float("nan"),
        budget_s=BUDGETS_S.get(name),
    )


def measure(
    db_dir: Path | str,
    *,
    repeats: int = DEFAULT_REPEATS,
    text: str = DEFAULT_TEXT,
    phenomenon: str = DEFAULT_PHENOMENON,
    ref_shot: int | None = None,
    segment: str = "flat_top",
    n: int = 10,
) -> LatencyReport:
    """Time the five operations against the database at `db_dir`."""
    from ..retrieval import describe as describe_mod
    from ..retrieval import phenomena as ph_mod
    from ..retrieval import rank as rank_mod
    from ..shotdb import store

    db_dir = Path(db_dir)
    db = store.ShotDB.load(db_dir)
    shots = [int(s) for s in db.shots.index]
    if not shots:
        raise ValueError(f"no shots in the database at {db_dir}")
    # A reference shot has to HAVE the segment, or `search` returns nothing and the timing is of
    # the empty path rather than of a search.
    ref = ref_shot if ref_shot is not None else _first_with_segment(db, shots, segment)
    notes: list[str] = []
    if ref is None:
        ref = shots[0]
        notes.append(
            f"no shot has a {segment} segment; the no-text search was timed from {ref} anyway "
            "and is not a measurement of a search that returns results"
        )

    rows = [
        _row("load", time_it(lambda: store.ShotDB.load(db_dir), repeats), repeats),
        _row(
            "search_text",
            time_it(
                lambda: rank_mod.search(QueryState(text=text, segment=segment, n=n), db), repeats
            ),
            repeats,
        ),
        _row(
            "search_no_text",
            time_it(
                lambda: rank_mod.search(QueryState(ref_shot=ref, segment=segment, n=n), db),
                repeats,
            ),
            repeats,
        ),
        _row(
            "phenomenon_locate",
            time_it(lambda: ph_mod.locate(phenomenon, db, 20, segment=segment), repeats),
            repeats,
        ),
        _row(
            "describe",
            time_it(lambda: describe_mod.describe(db.get(ref), segment), repeats),
            repeats,
        ),
    ]
    notes.append(
        f"warm: every operation was run {WARMUP}x and discarded before timing, so the MiniLM "
        "load and the first parquet read are not in these numbers"
    )
    notes.append(f"text query {text!r}; phenomenon {phenomenon!r}; reference shot {ref}")
    return LatencyReport(
        db_dir=str(db_dir),
        n_shots=len(shots),
        n_segment_rows=len(db.segments),
        repeats=repeats,
        warm=True,
        rows=rows,
        notes=notes,
        generated_at=dt.datetime.now(dt.UTC),
    )


def _first_with_segment(db, shots: list[int], segment: str) -> int | None:
    index = db.segments.index
    return next((s for s in shots if f"{s}:{segment}" in index), None)


def _ms(seconds: float) -> str:
    return "n/a" if math.isnan(seconds) else f"{1000.0 * seconds:.1f} ms"


def markdown(report: LatencyReport) -> str:
    lines = [
        (
            f"### latency, warm, N={report.repeats} "
            f"({report.n_shots} shots, {report.n_segment_rows} segment rows)"
        ),
        "",
        f"database `{report.db_dir}`",
        "",
        "| operation | what | median | p95 | budget | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.rows:
        budget = "—" if r.budget_s is None else _ms(r.budget_s)
        lines.append(
            f"| `{r.name}` | {r.what} | {_ms(r.median_s)} | {_ms(r.p95_s)} | {budget} | "
            f"{r.verdict} |"
        )
    if report.notes:
        lines += [""] + [f"- {n}" for n in report.notes]
    return "\n".join(lines)
