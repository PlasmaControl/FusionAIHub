"""`labeler.modes.fdp`: the 1 ms databases of `docs/labeler/fdp_labels.md`, on synthetic tracks.

Written into a temporary folder only; the writer never chooses a destination.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labeler.modes import fdp, suggest

from .modes_synth import SHOT, make_shot, write_shot

REPO = Path(__file__).resolve().parents[2]
AUDITED = {
    SHOT: {
        "tier": "silver",
        "reviewers": "Nathaniel Chen",
        "verified_on": "2026-11-07",
        "notes": "audited",
    }
}


@pytest.fixture
def database(tmp_path):
    labels = tmp_path / "labels"
    write_shot(labels, make_shot())
    write_shot(labels, make_shot(shot=100002, model_error="boom"))
    write_shot(labels, make_shot(shot=100003, window=None, status="no ip"))
    shots = [SHOT, 100002, 100003, 100004]
    ready, skipped, dropped = suggest.load(labels, shots)
    kids = suggest.subcategories(ready)
    out = tmp_path / "db" / "mhd_n2" / "run1"
    meta = fdp.write(
        out,
        "mhd_n2",
        ready,
        kids,
        shots,
        skipped,
        dropped,
        AUDITED,
        run="run1",
        source={"labels": "synthetic"},
        rules=["r"],
        repo=REPO,
        note=["proposed"],
    )
    return out, meta, kids


def test_the_grid_bits_name_the_parent_the_subcategories_and_not_observable(database):
    _, meta, _ = database
    assert meta["grid_bits"] == {"parent": 0, "m3": 1, "m4": 2, "not_observable": 7}
    assert meta["subcategories"] == {"m3": "3/2", "m4": "4/2"}


def test_spans_are_the_labelled_modes_on_the_1_ms_grid_with_the_subcategories_inside(
    database,
):
    out, _, _ = database
    spans = pd.read_parquet(out / "spans.parquet")
    assert set(spans.shot) == {SHOT}
    parent = spans[spans.subcategory == ""]
    assert list(zip(parent.t_start_ms, parent.t_end_ms)) == [(398, 502), (598, 702)]
    assert set(parent.n) == {2} and parent.m.isna().all()
    # the parent's confidence is the largest of the subcategories under it
    assert list(parent.confidence) == pytest.approx([0.93, 0.95])
    m4 = spans[spans.subcategory == "m4"]
    assert list(zip(m4.t_start_ms, m4.t_end_ms, m4.m, m4.mn)) == [(398, 502, 4, "4/2")]
    m3 = spans[spans.subcategory == "m3"]
    assert list(zip(m3.t_start_ms, m3.t_end_ms, m3.mn)) == [(598, 702, "3/2")]
    assert (spans.tier == "silver").all()
    assert str(spans.t_start_ms.dtype) == "int32" and str(spans.shot.dtype) == "int32"


def test_the_harmonic_is_not_in_the_spans(database):
    out, _, _ = database
    spans = pd.read_parquet(out / "spans.parquet")
    assert not (
        (spans.t_start_ms < 300) & (spans.t_end_ms > 100)
    ).any()  # 100-300 is the harmonic


def test_the_grid_file_has_one_byte_a_ms_with_the_bits_of_meta(database):
    out, meta, _ = database
    index = pd.read_csv(out / "grid_1ms" / "index.csv").set_index("shot")
    a, b = int(index.loc[SHOT, "start_ms"]), int(index.loc[SHOT, "end_ms"])
    data = np.fromfile(out / "grid_1ms" / str(SHOT // 1000) / f"{SHOT}.bin", np.uint8)
    assert len(data) == b - a == 1000
    bits = meta["grid_bits"]
    for t, parent, m3, m4 in [
        (200, 0, 0, 0),
        (450, 1, 0, 1),
        (650, 1, 1, 0),
        (750, 0, 0, 0),
    ]:
        byte = int(data[t - a])
        assert byte & 1 == parent
        assert (byte >> bits["m3"]) & 1 == m3 and (byte >> bits["m4"]) & 1 == m4
        assert (byte >> bits["not_observable"]) & 1 == 0
    assert index.loc[SHOT, "job"] == "run1"


def test_shots_and_valid_list_every_listed_shot_with_why_one_has_no_labels(database):
    out, meta, _ = database
    shots = pd.read_parquet(out / "shots.parquet").set_index("shot")
    assert dict(shots.status) == {
        SHOT: "labeled",
        100002: "failed",
        100003: "no_window",
        100004: "no_labels",
    }
    assert (
        shots.loc[SHOT, "harmonics_dropped"] == 1
        and shots.loc[SHOT, "tier"] == "silver"
    )
    assert shots.loc[SHOT, "reviewers"] == "Nathaniel Chen"
    assert (shots.drop(SHOT).tier == "unverified").all()
    valid = pd.read_csv(out / "valid.csv").set_index("shot")
    assert dict(valid.valid) == {SHOT: 1, 100002: 0, 100003: 0, 100004: 0}
    assert meta["counts"]["listed"] == 4 and meta["counts"]["labeled"] == 1
    assert meta["counts"]["harmonic_tracks_dropped"] == 1
    assert meta["counts"]["by_subcategory"] == {"3/2": 1, "4/2": 1}
    assert json.loads((out / "meta.json").read_text())["category"] == "mhd_n2"
    assert (out / "README.md").read_text().startswith("# MHD n2 modes")


def test_a_database_is_never_written_into_twice(database, tmp_path):
    out, _, kids = database
    with pytest.raises(FileExistsError):
        fdp.write(
            out,
            "mhd_n2",
            [],
            kids,
            [],
            {},
            {},
            {},
            run="x",
            source={},
            rules=[],
            repo=REPO,
            note=["n"],
        )


def test_too_many_subcategories_for_the_grid_byte_are_refused():
    kids = {f"mhd_n1.m{m}": f"{m}/1" for m in range(1, 9)}
    with pytest.raises(ValueError):
        fdp.kids_of("mhd_n1", kids)
