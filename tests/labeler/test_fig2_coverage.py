"""Figure 2's coverage record: how labelled time is counted, and the committed
record's shape and internal consistency. The generator reads the label tables, so
the counting is tested on small frames and the record is checked as committed."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/labeler/paper/fig2_coverage.py"
spec = importlib.util.spec_from_file_location("fig2_coverage", SCRIPT)
cov = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cov
spec.loader.exec_module(cov)

STATUSES = {"measured", "experimental", "point_events", "none", "pending"}
COUNTED = {"measured", "experimental"}


def frame(rows):
    return pd.DataFrame(rows, columns=["shot", "category", "t_start", "t_end"])


def record():
    if not cov.OUT.is_file():
        pytest.skip("docs/labeler/figure2_coverage.json is not generated")
    return json.loads(cov.OUT.read_text())


def test_overlapping_and_touching_intervals_count_once_per_shot():
    seconds = cov.union_seconds(
        frame(
            [
                (1, 1, 0, 1000),
                (1, 1, 500, 1500),  # overlaps the first
                (1, 1, 3000, 4000),  # a separate span
                (2, 1, 0, 1000),
                (2, 1, 1000, 2000),  # touches the first
            ]
        )
    )
    assert seconds[1] == pytest.approx(1.5 + 1.0)
    assert seconds[2] == pytest.approx(2.0)


def test_labelled_counts_only_the_stated_states_on_the_asked_shots():
    table = frame(
        [
            (1, 1, 0, 1000),  # present
            (1, 0, 2000, 4000),  # absent
            (1, 2, 5000, 9000),  # uncertain: not labelled time
            (2, 1, 0, 500),
            (3, 2, 0, 1000),  # only uncertain: not a labelled shot
        ]
    )
    both = cov.labelled(table, (0, 1))
    assert both == {"shots": 2, "seconds": pytest.approx(3.5)}
    assert cov.labelled(table, (0, 1), shots=[2]) == {
        "shots": 1,
        "seconds": pytest.approx(0.5),
    }
    assert cov.labelled(table, (3,)) == {"shots": 0, "seconds": 0.0}


def test_a_point_event_adds_no_time_and_no_shot():
    table = frame([(1, 1, 700, 700), (2, 1, 0, 1000)])
    assert cov.labelled(table, (1,)) == {"shots": 1, "seconds": pytest.approx(1.0)}


def test_the_record_has_the_seven_sets_in_the_figures_order_and_shape():
    rec = record()
    assert rec["schema"] == "figure2_coverage"
    assert rec["order"] == list(cov.SETS) and set(rec["sets"]) == set(cov.SETS)
    assert rec["definition"] == cov.DEFINITION
    for key in cov.SETS:
        entry = rec["sets"][key]
        assert entry["name"] == cov.NAMES[key]
        for side in ("legacy", "tokamak_si"):
            block = entry[side]
            assert {"status", "label", "shots", "seconds", "definition"} <= set(block)
            assert "paths" in block
            assert block["status"] in STATUSES
            counted = block["status"] in COUNTED
            if counted or block["status"] == "point_events":
                assert block["paths"], f"{key}.{side} names no source"
                assert block["shots"] > 0
            else:
                assert block["shots"] is None
            if counted:
                assert block["seconds"] > 0
            else:
                assert block["seconds"] is None


def test_a_set_that_departs_from_the_shared_definition_says_how():
    rec = record()
    shared = 0
    for key in cov.SETS:
        for side in ("legacy", "tokamak_si"):
            block = rec["sets"][key][side]
            assert isinstance(block["shared_definition"], bool)
            assert block["shared_definition"] == block["definition"].startswith(
                cov.SHARED
            )
            shared += block["shared_definition"]
            if not block["shared_definition"]:
                assert len(block["definition"]) > 20, f"{key}.{side} is unexplained"
    assert 0 < shared < 14  # some follow the definition, some depart from it


def test_the_reviewed_subset_sits_inside_the_tokamak_si_labels():
    rec = record()
    seen = 0
    for key in cov.SETS:
        si = rec["sets"][key]["tokamak_si"]
        rev = si.get("reviewed")
        assert rev is not None, f"{key} states no reviewed subset"
        assert rev["status"] in STATUSES
        if rev["status"] == "measured":
            seen += 1
            assert rev["shots"] <= si["shots"]
            assert rev["seconds"] <= si["seconds"] + 1e-9
        else:
            assert rev["shots"] is None and rev["seconds"] is None
    assert seen >= 2  # AE and ELM have reviewed spans


def test_a_value_read_from_a_stream_record_equals_that_records_key():
    rec = record()
    stream = json.loads(cov.TM_RECORD.read_text())
    for side in ("legacy", "tokamak_si"):
        block = rec["sets"]["tm"][side]
        node = stream
        for part in block["record_key"].split("."):
            node = node[part]
        assert block["shots"] == node["labelled_shots"]
        assert block["seconds"] == pytest.approx(node["labelled_seconds"], abs=5e-4)


def test_sawtooth_counts_the_released_present_and_tested_absent_population():
    block = record()["sets"]["sawtooth"]["tokamak_si"]
    manifest = json.loads(cov.SAW_MANIFEST.read_text())
    assert block["status"] == "measured" and block["shared_definition"]
    assert block["shards"] == sum(
        f["file"].startswith("population-") for f in manifest["files"]
    )
    assert block["shards_sha256sums"] == manifest["manifest_sha256"]
    # the stream's own record: present plus tested-absent seconds of the population
    stream = json.loads(
        (cov.SAW_MANIFEST.parent / "population_labels.json").read_text()
    )["state_seconds"]
    labelled_s = stream["present"] + stream["absent"]
    assert block["seconds"] == pytest.approx(labelled_s, rel=1e-3)
    assert (
        block["shots"]
        > json.loads((cov.SAW_MANIFEST.parent / "population_labels.json").read_text())[
            "positive_shots"
        ]
    )  # the tested-absent shots add to the shots with a train
    assert record()["sets"]["sawtooth"]["legacy"]["status"] == "none"


def test_the_sawtooth_reviewed_subset_is_the_three_shots_a_person_drew_spans_on():
    reviewed = record()["sets"]["sawtooth"]["tokamak_si"]["reviewed"]
    history = cov.table("sawtooth_oscillation/review/history.jsonl")
    lines = history.read_text().splitlines()
    seen = {json.loads(ln)["shot"] for ln in lines if ln.strip()}
    assert reviewed["status"] == "measured" and reviewed["shots"] == len(seen) == 3
    assert 0 < reviewed["seconds"] < 60


def test_a_shard_that_does_not_match_the_manifests_digest_is_refused(
    tmp_path, monkeypatch
):
    import hashlib

    shard = tmp_path / "population-000.csv"
    shard.write_text("shot,category,t_start,t_end\n1,1,0,1000\n")
    manifest = tmp_path / "manifest.json"

    def write(digest):
        manifest.write_text(
            json.dumps(
                {
                    "labels_path": str(tmp_path),
                    "manifest_sha256": "0" * 64,
                    "files": [{"file": shard.name, "sha256": digest}],
                }
            )
        )

    monkeypatch.setattr(cov, "SAW_MANIFEST", manifest)
    monkeypatch.setattr(cov, "SAW_TABLES", tmp_path / "none")
    write(hashlib.sha256(shard.read_bytes()).hexdigest())
    assert cov.saw_shards()[0] == [shard]
    write("1" * 64)
    with pytest.raises(ValueError, match="digest"):
        cov.saw_shards()


def test_detachment_reads_this_checkouts_record_and_names_no_worktree():
    block = record()["sets"]["detachment"]["tokamak_si"]
    stream = json.loads(cov.DETACH_RECORD.read_text())["coverage"]
    cover = stream["labelled_certain_or_tangtv_only"]
    assert block["shots"] == cover["shots"]
    assert block["seconds"] == pytest.approx(cover["seconds"], abs=5e-4)
    assert block["paths"] == ["docs/labeler/figure2_detach.json"]
    assert cov.DETACH_RECORD.relative_to(cov.REPO) == Path(
        "docs/labeler/figure2_detach.json"
    )


def test_no_recorded_source_and_no_default_path_names_a_stream_worktree():
    assert not [s for s in record()["sources"] if "FusionAIHub-r4-" in s]
    assert "FusionAIHub-r4-" not in SCRIPT.read_text()
    assert "worktree" not in SCRIPT.read_text().lower()


def test_the_record_names_the_files_it_was_made_from():
    rec = record()
    assert rec["sources"], "no source digests"
    assert all(len(d) == 64 for d in rec["sources"].values())
    for key in cov.SETS:
        for side in ("legacy", "tokamak_si"):
            for path in rec["sets"][key][side]["paths"]:
                assert path in rec["sources"], path


def test_the_generator_types_no_count():
    tree = ast.parse(SCRIPT.read_text())
    numbers = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, int | float)
        and not isinstance(n.value, bool)
    }
    # state codes, digits of rounding, ms per s: nothing a table could change
    assert numbers <= {0, 1, 2, 3, 4, 1000.0}, numbers
