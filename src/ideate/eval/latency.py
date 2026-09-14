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
* **Separate runs, and a verdict only where they agree.** `runs` blocks of `repeats` samples each,
  with each block's own median kept. One block on a shared login node measures the NODE: the I11
  review re-ran this harness three times and got `search_no_text` at 126 / 156 / 235 ms against a
  200 ms budget and `search_text` anywhere from 174 ms to 2.3 s, while `phenomenon_locate` was
  over budget in 6 runs out of 6. So PASS and FAIL are reported only where every block agrees;
  a row that straddles its budget says **`load-dependent`**, which is the honest answer and is
  not the same as a failure. The headline `median_s` is the median OF THE BLOCK MEDIANS, so one
  bad block cannot carry it, and the spread is printed.
* **The machine is recorded.** Load average, core count and `torch.get_num_threads()` go into the
  report, because a budget table the next reader cannot compare with their own run is a table of
  one afternoon.
* **A row with no budget gets no verdict.** `describe` has no Appendix B number, so it reports
  `n/a` rather than a free PASS -- a table full of passes, one of which was never tested against
  anything, is worse than an honest gap.

Nothing here is tuned to pass. If a row says FAIL it is a finding, and the report says by how much.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..schema import LatencyReport, LatencyRow, QueryState

__all__ = [
    "BUDGETS_S",
    "DEFAULT_REPEATS",
    "DEFAULT_RUNS",
    "WARMUP",
    "markdown",
    "measure",
    "time_it",
]

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
# Three separate blocks, which is the smallest number that can show a straddle at all.
DEFAULT_RUNS = 3
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


def _row(name: str, blocks: list[list[float]], repeats: int) -> LatencyRow:
    """One row from `runs` blocks of samples.

    `median_s` is the median of the per-block medians rather than of the pooled samples: pooling
    lets one slow block of 20 drag the headline, which is exactly the contamination this is meant
    to expose. `p95_s` IS pooled -- a tail is a tail whichever block it came from.
    """
    medians = [float(np.median(b)) for b in blocks if b]
    pooled = np.asarray([s for b in blocks for s in b], dtype=float)
    return LatencyRow(
        name=name,
        what=WHAT.get(name, name),
        repeats=repeats,
        median_s=float(np.median(medians)) if medians else float("nan"),
        p95_s=float(np.percentile(pooled, 95)) if pooled.size else float("nan"),
        budget_s=BUDGETS_S.get(name),
        run_medians=medians,
    )


def measure(
    db_dir: Path | str,
    *,
    repeats: int = DEFAULT_REPEATS,
    runs: int = DEFAULT_RUNS,
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

    ops: dict[str, Callable[[], object]] = {
        "load": lambda: store.ShotDB.load(db_dir),
        "search_text": lambda: rank_mod.search(QueryState(text=text, segment=segment, n=n), db),
        "search_no_text": lambda: rank_mod.search(
            QueryState(ref_shot=ref, segment=segment, n=n), db
        ),
        "phenomenon_locate": lambda: ph_mod.locate(phenomenon, db, 20, segment=segment),
        "describe": lambda: describe_mod.describe(db.get(ref), segment),
    }
    # Interleaved, not one operation at a time: the node's load drifts over the minutes a full
    # sweep takes, and running all 3 blocks of `search_text` back to back would put each
    # operation's blocks in a different part of that drift.
    blocks: dict[str, list[list[float]]] = {name: [] for name in ops}
    runs = max(1, int(runs))
    for _ in range(runs):
        for name, fn in ops.items():
            blocks[name].append(time_it(fn, repeats))
    rows = [_row(name, blocks[name], repeats) for name in ops]

    load_avg = _load_avg()
    cpu_count = os.cpu_count()
    threads = _torch_threads()
    notes.append(
        f"warm: every operation was run {WARMUP}x and discarded before timing, so the MiniLM "
        "load and the first parquet read are not in these numbers"
    )
    notes.append(
        f"{runs} separate blocks of {repeats}, interleaved; the row median is the median of the "
        "block medians and PASS/FAIL is reported only where every block agrees"
    )
    notes.append(
        f"machine at the end of the sweep: load average "
        f"{'/'.join(f'{x:.2f}' for x in load_avg) if load_avg else 'unknown'}, "
        f"{cpu_count} cpus, torch threads {threads if threads is not None else 'unknown'}"
    )
    notes.append(f"text query {text!r}; phenomenon {phenomenon!r}; reference shot {ref}")
    return LatencyReport(
        db_dir=str(db_dir),
        n_shots=len(shots),
        n_segment_rows=len(db.segments),
        repeats=repeats,
        runs=runs,
        load_avg=load_avg,
        cpu_count=cpu_count,
        torch_threads=threads,
        warm=True,
        rows=rows,
        notes=notes,
        generated_at=dt.datetime.now(dt.UTC),
    )


def _load_avg() -> tuple[float, float, float] | None:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:  # pragma: no cover - not every platform has one
        return None
    return (one, five, fifteen)


def _torch_threads() -> int | None:
    """Read only if torch is ALREADY imported. Importing it here to measure it would change the
    thing being measured (and cost several seconds on a cold process)."""
    torch = sys.modules.get("torch")
    getter = getattr(torch, "get_num_threads", None)
    if getter is None:
        return None
    try:
        return int(getter())
    except (RuntimeError, ValueError, TypeError):  # pragma: no cover - an unusual torch build
        return None


def _first_with_segment(db, shots: list[int], segment: str) -> int | None:
    index = db.segments.index
    return next((s for s in shots if f"{s}:{segment}" in index), None)


def _ms(seconds: float) -> str:
    return "n/a" if math.isnan(seconds) else f"{1000.0 * seconds:.1f} ms"


def markdown(report: LatencyReport) -> str:
    lines = [
        (
            f"### latency, warm, N={report.repeats} x {report.runs} runs "
            f"({report.n_shots} shots, {report.n_segment_rows} segment rows)"
        ),
        "",
        f"database `{report.db_dir}`",
        "",
        "| operation | what | median | p95 | spread over runs | budget | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.rows:
        budget = "—" if r.budget_s is None else _ms(r.budget_s)
        spread = (
            f"{_ms(min(r.run_medians))} – {_ms(max(r.run_medians))}"
            if r.run_medians
            else "—"
        )
        lines.append(
            f"| `{r.name}` | {r.what} | {_ms(r.median_s)} | {_ms(r.p95_s)} | {spread} | "
            f"{budget} | {r.verdict} |"
        )
    if report.notes:
        lines += [""] + [f"- {n}" for n in report.notes]
    return "\n".join(lines)
