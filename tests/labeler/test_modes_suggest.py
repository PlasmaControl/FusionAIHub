"""`labeler.modes.suggest`: the v6 `mhd_modes` tables in v5's format, on synthetic port-v6 files.

Written into a temporary root only. The real suggestion folders, the events tree and the
tokamaksi tree are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from labeler.events.review import labels as review_labels
from labeler.modes import cells as C
from labeler.modes import suggest
from labeler.modes import tracks as T

from .modes_synth import SHOT, make_shot, write_shot

REPO = Path(__file__).resolve().parents[2]
COLUMNS = ["shot", "category", "t_start", "t_end", "confidence"]


@pytest.fixture
def labels(tmp_path):
    folder = tmp_path / "labels"
    write_shot(folder, make_shot())
    # a shot the caller did not list: never read, never in a table
    write_shot(folder, make_shot(shot=100001))
    write_shot(folder, make_shot(shot=100002, model_error="boom"))
    write_shot(folder, make_shot(shot=100003, window=None, status="no ip"))
    return folder


def test_load_sorts_listed_shots_into_ready_and_skipped_without_reading_others(labels):
    ready, skipped, dropped = suggest.load(labels, [SHOT, 100002, 100003, 100004])
    assert [s for s, _ in ready] == [SHOT]
    assert dropped[SHOT] == {"harmonics": 1, "out_of_scope": 1, "kept": 7}
    assert skipped[100002].startswith(suggest.FAILED)
    assert skipped[100003].startswith(suggest.NO_WINDOW)
    assert skipped[100004] == suggest.NO_LABELS
    assert 100001 not in skipped and 100001 not in dropped  # listed by nobody


def test_subcategories_are_the_classes_labelled_plus_the_five_of_v5(labels):
    ready, _, _ = suggest.load(labels, [SHOT])
    kids = suggest.subcategories(ready)
    assert set(kids) >= {C_ for C_ in map(T.event_of, suggest.V5_CLASSES)}
    # 4/2 (the independent one) and the labelled 4/3 are new to v5's five
    assert kids["mhd_n2.m4"] == "4/2" and kids["mhd_n3.m4"] == "4/3"
    assert list(kids) == sorted(kids, key=lambda e: (int(e[5]), int(e.split(".m")[1])))


def test_write_makes_one_table_and_sidecar_an_event_in_v5_format(labels, tmp_path):
    root = tmp_path / "root"
    summary = suggest.write(labels, [SHOT, 100002, 100004], root, repo=REPO)
    folder = root / "suggestions" / "mhd_modes" / "v6port"
    assert summary["shots"] == 1 and set(summary["skipped"]) == {100002, 100004}
    for event in (*C.PARENTS, *summary["classes"]):
        table = folder / f"{event}_suggest_mhd_modes_v6port.csv"
        assert table.is_file() and table.with_suffix(".meta.json").is_file(), event
        frame = pd.read_csv(table)
        assert list(frame.columns) == COLUMNS
        assert set(frame.shot) == {SHOT}  # the unlisted shots are not in a table
        # the rows tile the assessed window, with no gap or overlap
        assert frame.t_start.iloc[0] == 0 and frame.t_end.iloc[-1] == 1000
        assert (
            frame.t_start.iloc[1:].to_numpy() == frame.t_end.iloc[:-1].to_numpy()
        ).all()
    n2 = pd.read_csv(folder / "mhd_n2_suggest_mhd_modes_v6port.csv")
    present = n2[n2.category == 1]
    assert list(zip(present.t_start, present.t_end)) == [(398, 502), (598, 702)]
    meta = json.loads(
        (folder / "mhd_n2.m4_suggest_mhd_modes_v6port.meta.json").read_text()
    )
    assert meta["method"] == "mhd_modes" and meta["version"] == "v6port"
    assert "proposed, not checked" in meta["source"]["status"]
    assert meta["skipped"]["100004"] == suggest.NO_LABELS
    assert "harm" in meta["rule"]["harmonics"] and "n>3" in meta["rule"]["scope"]
    assert meta["per_shot"][str(SHOT)]["harmonics_dropped"] == 1


def test_a_harmonic_adds_nothing_to_the_tables_but_an_independent_4_over_2_does(
    labels, tmp_path
):
    suggest.write(labels, [SHOT], tmp_path / "root", repo=REPO)
    folder = tmp_path / "root" / "suggestions" / "mhd_modes" / "v6port"
    m4 = pd.read_csv(folder / "mhd_n2.m4_suggest_mhd_modes_v6port.csv")
    present = m4[m4.category == 1]
    # 4/2 is labelled on the harmonic track (100-300) and the independent one (400-500)
    assert list(zip(present.t_start, present.t_end)) == [(398, 502)]
    assert float(present.confidence.iloc[0]) == pytest.approx(0.93)


def test_a_version_is_never_written_into_twice(labels, tmp_path):
    root = tmp_path / "root"
    suggest.write(labels, [SHOT], root, repo=REPO)
    with pytest.raises(FileExistsError):
        suggest.write(labels, [SHOT], root, repo=REPO)
    suggest.write(
        labels, [SHOT], root, version="v7test", repo=REPO
    )  # another version is fine
    assert (root / "suggestions" / "mhd_modes" / "v7test").is_dir()


def test_point_writes_a_review_source_for_each_event_and_makes_subcategory_folders(
    labels, tmp_path
):
    root, events = tmp_path / "root", tmp_path / "events"
    summary = suggest.write(labels, [SHOT], root, repo=REPO)
    (events / "mhd_n1" / "review").mkdir(parents=True)
    entries = suggest.point(events, root, summary["classes"])
    assert set(entries) == {*C.PARENTS, *summary["classes"]}
    for event in entries:
        parent, _, key = event.partition(".")
        base = events / parent / "subcategories" / key if key else events / parent
        pointer = json.loads((base / "review" / "source.json").read_text())
        assert pointer["method"] == "mhd_modes" and pointer["version"] == "v6port"
        assert pointer["table"].endswith(f"{event}_suggest_mhd_modes_v6port.csv")
        # the review page reads the table the pointer names, 1 ms cells
        assert review_labels.read_source(base)
