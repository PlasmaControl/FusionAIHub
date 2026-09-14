"""Phenomenon recall against an annotation sheet -- and the refusals, which are most of it.

No sheet exists anywhere today (`$LABELMAKER_ROOT/annotate/` is empty), so every test here builds
one under `tmp_path`. That is not a workaround: the refusal path is what this module does in
production right now, and it is the part that has to be right first.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ideate.eval import phenomenon_recall as rec
from ideate.shotdb import store

from .conftest import shot_record, write_db

SHOT_A, SHOT_B = 300, 301


@pytest.fixture
def db(tmp_path: Path) -> store.ShotDB:
    d = tmp_path / "db"
    d.mkdir()
    write_db(
        d,
        [
            shot_record(SHOT_A, "r1", 1.0e6, 5.0e6, "sawtooth crashes seen on the core ECE"),
            shot_record(SHOT_B, "r1", 1.1e6, 5.5e6, "quiet shot"),
        ],
    )
    _events(
        d,
        [
            # A detector saw a sawtooth at 2.0-2.1 s on shot 300, and nothing on 301.
            _event(SHOT_A, "300-ece_sawtooth-0", t0_s=2.0, t1_s=2.1),
            _event(SHOT_A, "300-ece_sawtooth-1", t0_s=3.0, t1_s=3.1),
        ],
    )
    return store.ShotDB.load(d)


def _event(shot: int, event_id: str, **over) -> dict:
    from labelmaker.events import schema as events_schema

    row = {
        "shot": shot,
        "event_id": event_id,
        "source": "ece_sawtooth",
        "evidence_kind": "heuristic",
        "phenomenon": "sawtooth",
        "t0_s": 2.0,
        "t1_s": 2.1,
        "f0_khz": np.nan,
        "f1_khz": np.nan,
        "confidence": 0.9,
        "horizon_s": np.nan,
        "diag": "ece",
        "channel": 0,
        "pass_name": "",
        "attrs": {},
        "t_cov0_s": 0.0,
        "t_cov1_s": 6.0,
        "run_id": "r1",
        "git_sha": "abc",
        "written_at": "2026-09-13T00:00:00+00:00",
    }
    row.update(over)
    row["attrs"] = json.dumps(row["attrs"], sort_keys=True)
    assert set(row) == set(events_schema.COLUMNS)
    return row


def _events(db_dir: Path, rows: list[dict]) -> None:
    from labelmaker.events import schema as events_schema

    pd.DataFrame(rows, columns=list(events_schema.COLUMNS)).astype(
        events_schema.DTYPES
    ).to_parquet(db_dir / "events.parquet", index=False)


def _sheet(root: Path, phenomenon: str, rows: list[dict], *, splits=None, header=None) -> Path:
    """A sheet.csv + manifest.parquet pair, the way the annotate stage writes them."""
    d = root / "annotate" / phenomenon
    d.mkdir(parents=True, exist_ok=True)
    cols = list(header or rec.SHEET_COLUMNS)
    with (d / "sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    man = pd.DataFrame(
        [
            {
                "file": r["file"],
                "shot": r["shot"],
                "t0_s": r["t0_s"],
                "t1_s": r["t1_s"],
                "phenomenon": phenomenon,
                "is_negative": not r.get("truth", True),
                "prior_score": 0.5,
                "why": "fixture",
                "seed": 1,
                "diags": "ece",
                "split": (splits or {}).get(r["file"], "test"),
            }
            for r in rows
        ]
    )
    man.to_parquet(d / "manifest.parquet", index=False)
    return d / "sheet.csv"


def _rows(n: int, *, shot: int = SHOT_A, hit_window=(2.0, 2.1), label="y") -> list[dict]:
    return [
        {
            "file": f"{i:04d}.png",
            "shot": shot,
            "t0_s": hit_window[0],
            "t1_s": hit_window[1],
            "phenomenon": "sawtooth",
            "label": label,
            "notes": "",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------- the refusals


def test_no_sheet_at_all_is_a_refusal_naming_the_path(tmp_path, db):
    with pytest.raises(rec.RecallRefused, match="no annotation sheet at"):
        rec.recall("sawtooth", db, tmp_path)


def test_a_sheet_with_no_manifest_beside_it_is_refused(tmp_path, db):
    _sheet(tmp_path, "sawtooth", _rows(25))
    (tmp_path / "annotate" / "sawtooth" / "manifest.parquet").unlink()
    with pytest.raises(rec.RecallRefused, match="no manifest.parquet"):
        rec.recall("sawtooth", db, tmp_path)


def test_a_sheet_whose_header_is_not_the_header_is_refused_not_column_guessed(tmp_path, db):
    _sheet(tmp_path, "sawtooth", _rows(25), header=[*rec.SHEET_COLUMNS, "extra"])
    with pytest.raises(rec.RecallRefused, match="header is"):
        rec.recall("sawtooth", db, tmp_path)


def test_a_sheet_and_manifest_from_different_renderings_are_refused(tmp_path, db):
    path = _sheet(tmp_path, "sawtooth", _rows(25))
    man_path = path.with_name("manifest.parquet")
    man = pd.read_parquet(man_path)
    man.loc[0, "file"] = "9999.png"  # a seeded shuffle renumbered: the join would be nonsense
    man.to_parquet(man_path, index=False)
    with pytest.raises(rec.RecallRefused, match="different renderings"):
        rec.recall("sawtooth", db, tmp_path)


def test_fewer_than_twenty_labelled_test_rows_is_a_refusal_with_the_counts(tmp_path, db):
    _sheet(tmp_path, "sawtooth", _rows(19))
    with pytest.raises(rec.RecallRefused, match=r"19 scorable labelled .test. rows"):
        rec.recall("sawtooth", db, tmp_path)


def test_unlabelled_rows_do_not_count_towards_the_twenty(tmp_path, db):
    """A `?` or an empty cell is an annotator who did not answer. Counting those would
    manufacture a recall number out of windows nobody read."""
    rows = _rows(15) + _rows(10, label="?") + _rows(10, label="")
    for i, r in enumerate(rows):
        r["file"] = f"{i:04d}.png"
    _sheet(tmp_path, "sawtooth", rows)
    with pytest.raises(rec.RecallRefused, match="15 scorable labelled"):
        rec.recall("sawtooth", db, tmp_path)


def test_rows_outside_the_test_split_do_not_count_towards_the_twenty(tmp_path, db):
    rows = _rows(30)
    splits = {r["file"]: ("test" if i < 12 else "train") for i, r in enumerate(rows)}
    _sheet(tmp_path, "sawtooth", rows, splits=splits)
    with pytest.raises(rec.RecallRefused, match=r"12 scorable labelled .test. rows"):
        rec.recall("sawtooth", db, tmp_path)


def test_the_floor_is_applied_to_the_rows_actually_SCORED_not_to_the_sheet(tmp_path, db):
    """22 labelled test rows, 20 of them on shots this database does not hold, is TWO windows.

    The floor counted rows before the absent-shot filter, so this scored two and printed
    `recall = 0.0` with a caveat beside it and exit 0 -- precisely the "a recall over twelve
    windows is a claim the data cannot support" the module refuses everywhere else.
    """
    rows = _rows(2) + _rows(20, shot=999999)
    for i, r in enumerate(rows):
        r["file"] = f"{i:04d}.png"
    _sheet(tmp_path, "sawtooth", rows)
    with pytest.raises(rec.RecallRefused, match=r"2 scorable labelled .test. rows"):
        rec.recall("sawtooth", db, tmp_path)


def test_the_floor_counts_windows_on_shots_the_database_holds(tmp_path, db):
    """The mirror of the above: 20 scorable windows plus any number of absent ones is enough."""
    rows = _rows(20) + _rows(5, shot=999999)
    for i, r in enumerate(rows):
        r["file"] = f"{i:04d}.png"
    _sheet(tmp_path, "sawtooth", rows)
    report = rec.recall("sawtooth", db, tmp_path)
    assert report.true_positive + report.false_negative == 20
    assert report.n_rows_not_in_db == 5


def test_a_manifest_with_no_split_column_is_refused(tmp_path, db):
    path = _sheet(tmp_path, "sawtooth", _rows(25))
    man_path = path.with_name("manifest.parquet")
    pd.read_parquet(man_path).drop(columns=["split"]).to_parquet(man_path, index=False)
    with pytest.raises(rec.RecallRefused, match="no `split` column"):
        rec.recall("sawtooth", db, tmp_path)


# ------------------------------------------------------------------------------ the counting


def test_a_window_the_detector_covers_is_a_true_positive(tmp_path, db):
    _sheet(tmp_path, "sawtooth", _rows(20))
    report = rec.recall("sawtooth", db, tmp_path)
    assert report.true_positive == 20 and report.false_negative == 0
    assert report.recall == 1.0
    assert report.specificity is None  # no negatives in this sheet
    assert report.n_shots == 1


def test_a_window_the_detector_missed_is_a_false_negative(tmp_path, db):
    _sheet(tmp_path, "sawtooth", _rows(20, hit_window=(4.5, 4.6)))
    report = rec.recall("sawtooth", db, tmp_path)
    assert report.true_positive == 0 and report.false_negative == 20
    assert report.recall == 0.0


def test_negatives_are_scored_as_specificity_not_folded_into_recall(tmp_path, db):
    hits = _rows(12)
    misses = _rows(12, hit_window=(4.5, 4.6), label="n")
    for i, r in enumerate(hits + misses):
        r["file"] = f"{i:04d}.png"
    _sheet(tmp_path, "sawtooth", hits + misses)
    report = rec.recall("sawtooth", db, tmp_path)
    assert (report.true_positive, report.false_negative) == (12, 0)
    assert (report.true_negative, report.false_positive) == (12, 0)
    assert report.recall == 1.0 and report.specificity == 1.0


def test_a_detected_window_annotated_no_is_a_false_positive(tmp_path, db):
    hits = _rows(12)
    wrong = _rows(12, label="n")  # the detector DOES fire here; the annotator said no
    for i, r in enumerate(hits + wrong):
        r["file"] = f"{i:04d}.png"
    _sheet(tmp_path, "sawtooth", hits + wrong)
    report = rec.recall("sawtooth", db, tmp_path)
    assert report.false_positive == 12 and report.true_negative == 0
    assert report.specificity == 0.0


def test_a_window_on_a_shot_this_database_lacks_is_excluded_not_counted_as_a_miss(tmp_path, db):
    rows = _rows(20) + _rows(5, shot=999999)
    for i, r in enumerate(rows):
        r["file"] = f"{i:04d}.png"
    _sheet(tmp_path, "sawtooth", rows)
    report = rec.recall("sawtooth", db, tmp_path)
    assert report.n_rows_not_in_db == 5
    assert report.true_positive + report.false_negative == 20
    assert any("not counted as misses" in c for c in report.caveats)


def test_only_observed_intervals_count_as_a_detection(tmp_path, db):
    """A forecast is a model's estimate of what was about to happen. Scoring it as a hit would
    inflate the detectors' recall with the label models' confidence."""
    _events(
        db.db_dir,
        [
            _event(
                SHOT_A, "300-label_forecast-0", source="label_forecast",
                evidence_kind="forecast", phenomenon="tearing", horizon_s=1.0, diag="",
                attrs={"label": "tm_risk_1s"},
            )
        ],
    )
    reloaded = store.ShotDB.load(db.db_dir)
    _sheet(tmp_path, "tearing", _rows(20))
    report = rec.recall("tearing", reloaded, tmp_path)
    assert report.true_positive == 0 and report.recall == 0.0
    assert any("observed intervals only" in c for c in report.caveats)


def test_a_phenomenon_no_detector_writes_is_refused_rather_than_scored_as_recall_zero(
    tmp_path, db
):
    """`fast_ion`, `detachment` and `rwm` have no event rule and no label head, so `_detected`
    can only ever be False and the confusion table is all-misses BY CONSTRUCTION.

    Publishing `recall = 0.0` from that would be a number about the registry's shape, not about
    any detector -- the same shape as the twelve-window recall this module already refuses, and
    a caveat beside the zero is an apology, not a refusal. So the refusal comes first.
    """
    for phenomenon in ("fast_ion", "detachment", "rwm"):
        _sheet(tmp_path, phenomenon, _rows(20))
        with pytest.raises(rec.RecallRefused, match="no detector") as excinfo:
            rec.recall(phenomenon, db, tmp_path)
        assert phenomenon in str(excinfo.value)


def test_the_no_detector_refusal_fires_before_the_sheet_is_read(tmp_path, db):
    """Today `fast_ion` refuses only because `$LABELMAKER_ROOT/annotate/` is empty. That is an
    accident of the corpus, so the check is made against the REGISTRY and does not wait for a
    sheet to arrive."""
    with pytest.raises(rec.RecallRefused, match="no detector") as excinfo:
        rec.recall("fast_ion", db, tmp_path)
    assert "no annotation sheet" not in str(excinfo.value)


def test_a_phenomenon_with_only_a_label_head_is_still_scorable(tmp_path, db):
    """The bar is "no detector source at all". A label model is a detector source -- its recall
    is a real question -- so the refusal must not swallow one."""
    _sheet(tmp_path, "sawtooth", _rows(20))
    assert rec.recall("sawtooth", db, tmp_path).recall == 1.0


# -------------------------------------------------------------------------------- reporting


def test_the_report_carries_the_sheet_it_read_and_the_row_counts(tmp_path, db):
    _sheet(tmp_path, "sawtooth", _rows(20))
    report = rec.recall("sawtooth", db, tmp_path)
    assert report.sheet.endswith("annotate/sawtooth/sheet.csv")
    assert (report.n_rows, report.n_test_rows, report.n_labelled) == (20, 20, 20)
    assert (report.n_positive, report.n_negative) == (20, 0)


def test_the_markdown_is_a_confusion_table_with_the_caveats(tmp_path, db):
    _sheet(tmp_path, "sawtooth", _rows(20))
    md = rec.markdown(rec.recall("sawtooth", db, tmp_path))
    assert "| annotated `y` | 20 | 0 |" in md
    assert "recall 100.0 % · specificity undefined" in md
    assert "observed intervals only" in md
