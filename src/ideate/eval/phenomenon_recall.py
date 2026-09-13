"""Recall of `phenomena.locate` against the human annotation sheets -- and a refusal when there
is not enough of a sheet to say anything.

The sheets live at `$LABELMAKER_ROOT/annotate/<phenomenon>/`: `sheet.csv`, whose header is
exactly `file,shot,t0_s,t1_s,phenomenon,label,notes` and whose `label` is `y`, `n`, `?` or empty,
beside `manifest.parquet`, which carries the same rows keyed by `file` plus the columns the
renderer kept back from the annotator -- `is_negative`, `prior_score`, `why`, `seed`, `diags` --
and the `split`. The sheet is what a human filled in; the manifest is what the split and the
hard-negative flags come from, and the two are joined on `file`. A sheet whose keys do not match
its manifest is refused, not silently inner-joined: the numbering is a seeded shuffle, so a
mismatch means the sheet and the manifest are from different renderings and any join between them
is nonsense.

**Only `split=test` rows, and only `y`/`n` labels.** A `?` or an empty cell is an annotator who
did not answer. Counting those as negatives would manufacture a recall number out of windows
nobody read, which is the same error as reading "no data" as "no phenomenon" everywhere else in
this package.

**It refuses below 20 labelled test rows** (`MIN_TEST_ROWS`). A recall computed on twelve windows
has a 95 % interval roughly half the unit interval wide; printing it with three decimals would be
a claim the data cannot support. The refusal is a `RecallRefused` with the counts in the message,
and the CLI exits 2 on it -- an empty table would read as "recall 0", and the two are opposite
answers.

**What counts as a detection.** An OBSERVED interval overlapping the annotated window, and
nothing else. A forecast is a model's estimate of what was about to happen and a text claim is a
sentence somebody typed; neither is a diagnostic showing the phenomenon inside that window, and
scoring either as a hit would inflate the recall of the detectors with the confidence of the
label models. `locate`'s ranking keeps those classes apart for the same reason (plan §2, §7).

Today no sheet exists anywhere: `$LABELMAKER_ROOT/annotate/` is empty, so every real call to this
module refuses, and its tests build a synthetic sheet under `tmp_path`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..retrieval import phenomena as ph_mod
from ..schema import RecallReport

__all__ = [
    "MIN_TEST_ROWS",
    "SHEET_COLUMNS",
    "RecallRefused",
    "load_sheet",
    "markdown",
    "recall",
    "sheet_path",
]

# The sheet's header, exactly (plan §B, the annotate loop). A sheet with any other header is a
# different file and is refused rather than column-guessed.
SHEET_COLUMNS = ("file", "shot", "t0_s", "t1_s", "phenomenon", "label", "notes")

LABELS_YES, LABELS_NO = "y", "n"
TEST_SPLIT = "test"
MIN_TEST_ROWS = 20

# The segment the evidence is read over. `full` is the whole discharge, which is what an
# annotation window is indexed against: clipping to the flat top would score a ramp-down ELM as a
# miss when the annotator was looking at the ramp-down.
DEFAULT_SEGMENT = "full"


class RecallRefused(Exception):
    """Not enough labelled test rows, or no sheet at all. The CLI exits 2 on this."""


def sheet_path(root: Path | str, phenomenon: str) -> Path:
    return Path(root) / "annotate" / phenomenon / "sheet.csv"


def load_sheet(root: Path | str, phenomenon: str) -> pd.DataFrame:
    """`sheet.csv` joined to `manifest.parquet` on `file`, with the split attached.

    Raises `RecallRefused` when either file is missing, when the header is not the header, or
    when the two disagree about which windows exist.
    """
    sheet = sheet_path(root, phenomenon)
    manifest = sheet.with_name("manifest.parquet")
    if not sheet.exists():
        raise RecallRefused(
            f"no annotation sheet at {sheet}. Render one with labelmaker's annotate stage and "
            "have it filled in; recall against nothing is not a number."
        )
    if not manifest.exists():
        raise RecallRefused(
            f"{sheet} has no manifest.parquet beside it, so there is no split to select "
            "`test` rows by."
        )
    rows = pd.read_csv(sheet, dtype={"label": str, "notes": str})
    if tuple(rows.columns) != SHEET_COLUMNS:
        raise RecallRefused(
            f"{sheet} header is {tuple(rows.columns)}; it must be exactly {SHEET_COLUMNS}"
        )
    man = pd.read_parquet(manifest)
    if "split" not in man.columns:
        raise RecallRefused(
            f"{manifest} carries no `split` column; only `split={TEST_SPLIT}` rows may be scored."
        )
    if set(rows["file"]) != set(man["file"]):
        raise RecallRefused(
            f"{sheet} and {manifest} disagree about which windows exist "
            f"({len(set(rows['file']) ^ set(man['file']))} keys differ). The numbering is a "
            "seeded shuffle, so they are from different renderings and must not be joined."
        )
    keep = [c for c in man.columns if c == "file" or c not in rows.columns]
    return rows.merge(man[keep], on="file", how="left", validate="one_to_one")


def _detected(db, shot: int, ph, t0: float, t1: float, segment: str, min_confidence: float) -> bool:
    """Did a DETECTOR see this phenomenon inside the annotated window on this shot?

    Observed intervals only -- see the module docstring for why a forecast and a text claim do
    not count here.
    """
    ev = ph_mod.evidence(shot, ph, db, segment, min_confidence=min_confidence)
    return any(iv.t0_s <= t1 and iv.t1_s >= t0 for iv in ev.intervals)


def recall(
    phenomenon: str,
    db,
    root: Path | str,
    *,
    segment: str = DEFAULT_SEGMENT,
    min_confidence: float = 0.0,
    min_rows: int = MIN_TEST_ROWS,
) -> RecallReport:
    """Recall (and specificity) of the detectors for `phenomenon` over the sheet's test rows."""
    ph = ph_mod.registry()[phenomenon]
    frame = load_sheet(root, phenomenon)
    n_rows = len(frame)
    test = frame.loc[frame["split"].astype(str) == TEST_SPLIT]
    labelled = test.loc[test["label"].astype(str).str.strip().isin({LABELS_YES, LABELS_NO})]
    if len(labelled) < min_rows:
        raise RecallRefused(
            f"{phenomenon}: {len(labelled)} labelled `{TEST_SPLIT}` rows in "
            f"{sheet_path(root, phenomenon)} ({len(test)} test rows of {n_rows}); "
            f"{min_rows} are needed. A recall over fewer is a number the data cannot support, "
            "so none is reported."
        )

    known = {int(s) for s in getattr(db, "shots", pd.DataFrame()).index}
    counts = dict(tp=0, fn=0, fp=0, tn=0)
    absent = 0
    shots: set[int] = set()
    for row in labelled.to_dict("records"):
        shot = int(row["shot"])
        if shot not in known:
            absent += 1
            continue
        shots.add(shot)
        hit = _detected(db, shot, ph, float(row["t0_s"]), float(row["t1_s"]), segment,
                        min_confidence)
        positive = str(row["label"]).strip() == LABELS_YES
        counts["tp" if hit else "fn"] += 1 if positive else 0
        counts["fp" if hit else "tn"] += 0 if positive else 1

    caveats: list[str] = []
    if not ph.events:
        caveats.append(
            f"no detector writes `{phenomenon}` (phenomena.yaml lists no event rule for it), so "
            "this measures the recall of nothing and is 0 by construction, not by failure"
        )
    if absent:
        caveats.append(
            f"{absent} labelled test windows are on shots this database does not hold; they are "
            "excluded, not counted as misses"
        )
    if counts["tp"] + counts["fn"] == 0:
        caveats.append("no positive test window survived, so recall is undefined rather than 0")
    if counts["tn"] + counts["fp"] == 0:
        caveats.append("no negative test window survived, so specificity is undefined")
    caveats.append(
        "observed intervals only: a forecast and an operator's sentence are not a detector "
        "seeing the phenomenon inside the window, and neither is scored as a hit"
    )
    return RecallReport(
        phenomenon=phenomenon,
        sheet=str(sheet_path(root, phenomenon)),
        n_rows=n_rows,
        n_test_rows=len(test),
        n_labelled=len(labelled),
        n_positive=int((labelled["label"].astype(str).str.strip() == LABELS_YES).sum()),
        n_negative=int((labelled["label"].astype(str).str.strip() == LABELS_NO).sum()),
        n_rows_not_in_db=absent,
        true_positive=counts["tp"],
        false_negative=counts["fn"],
        false_positive=counts["fp"],
        true_negative=counts["tn"],
        recall=_ratio(counts["tp"], counts["tp"] + counts["fn"]),
        specificity=_ratio(counts["tn"], counts["tn"] + counts["fp"]),
        n_shots=len(shots),
        caveats=caveats,
        generated_at=dt.datetime.now(dt.UTC),
    )


def _ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _pct(x: float | None) -> str:
    return "undefined" if x is None else f"{100.0 * x:.1f} %"


def markdown(report: RecallReport) -> str:
    lines = [
        f"### `{report.phenomenon}` recall vs `split=test` annotation rows",
        "",
        f"sheet `{report.sheet}` — {report.n_labelled} labelled test rows "
        f"({report.n_positive} y, {report.n_negative} n) over {report.n_shots} shots, "
        f"of {report.n_test_rows} test rows in {report.n_rows}",
        "",
        "| | detected | not detected |",
        "| --- | --- | --- |",
        f"| annotated `y` | {report.true_positive} | {report.false_negative} |",
        f"| annotated `n` | {report.false_positive} | {report.true_negative} |",
        "",
        f"recall {_pct(report.recall)} · specificity {_pct(report.specificity)}",
    ]
    if report.caveats:
        lines += [""] + [f"- {c}" for c in report.caveats]
    return "\n".join(lines)
