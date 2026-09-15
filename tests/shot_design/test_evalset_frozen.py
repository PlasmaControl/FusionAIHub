"""The evalset is FROZEN. This file is the lock.

An evaluation set that can be edited after the numbers come in measures nothing: the temptation
is not to cheat but to "fix a badly worded prompt" that happens to be the one that missed, and
there is no way to tell the two apart afterwards. So the 200 prompts and the dev/eval split are
committed once, their sha256 is written into the README, and this module asserts that hash
against a literal recorded HERE as well. Changing the CSV therefore takes three deliberate edits
in three files, and every one of them shows up in a diff.

The structural assertions below are the evalset's own contract (task I11's brief): 200 rows, at
least 12 categories with at least 8 prompts each, exactly 20 hand-graded rows each carrying a
rubric, every `expect_phenomena` id in the phenomenon registry, every `expect_constraints`
column a real segment column, every `expect_segment` a real segment name.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import pytest
import yaml

from shot_design import config
from shot_design.eval import prompts as ev
from shot_design.retrieval import phenomena as ph_mod
from shot_design.schema import SegName

REPO = Path(__file__).resolve().parents[2]
EVALSET = REPO / "configs" / "shot_design" / "evalsets" / "reference_shot_prompts.csv"
README = REPO / "configs" / "shot_design" / "evalsets" / "README.md"
SPLIT = REPO / "configs" / "shot_design" / "evalsets" / "split.yaml"

# The frozen hash. Authored 2026-09-13 (task I11) and never edited to make a number pass.
#
# v1.1 -- the ONE re-freeze, and it happened before any retrieval tuning. An independent review
# found eleven prompts with physics defects (an expectation that named the AE *mode* for a
# fast-ion *topic*, a row whose hard filter selected against its own expected phenomenon, a
# duplicate pair, a garbled sentence) plus nine that are unanswerable on this corpus by
# construction. Those are corrections to the QUESTIONS, not to the answers: nothing in retrieval,
# the ranking or the scoring changed with them. The changelog is in README.md, prompt by prompt,
# and the v1.0 hash is kept below so a v1.0 number can still be told apart from a v1.1 one.
EVALSET_SHA256_V1_0 = "a6059fbccd3aec79547d5a7ced3896c0ea6496c4e60927aa108675caef63a6a9"
EVALSET_SHA256 = "aed8547330414e5b427579b58b207d661609bbfca1b048ca6df666e71172ef0e"
EVALSET_VERSION = "1.1"

HEADER = [
    "prompt_id",
    "category",
    "prompt",
    "expect_phenomena",
    "expect_constraints",
    "expect_segment",
    "hand_graded",
    "notes",
]


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    with EVALSET.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------------- the lock


def test_the_evalset_hash_is_the_one_this_test_and_the_readme_both_record():
    """Two independent records of the same hash. One alone could be updated with the CSV in a
    single edit and nobody would notice; two cannot be."""
    digest = hashlib.sha256(EVALSET.read_bytes()).hexdigest()
    assert digest == EVALSET_SHA256, (
        "the frozen evalset changed. If that was deliberate, say so in the report and update "
        "BOTH this constant and the hash in configs/shot_design/evalsets/README.md."
    )
    text = README.read_text(encoding="utf-8")
    assert EVALSET_SHA256 in text
    assert f"version: {EVALSET_VERSION}" in text
    # The superseded hash stays recorded, so a number produced against v1.0 is still identifiable.
    assert EVALSET_SHA256_V1_0 in text
    assert EVALSET_SHA256 != EVALSET_SHA256_V1_0


def test_the_loader_reports_the_same_hash_it_read():
    loaded = ev.load_evalset()
    assert loaded.sha256 == EVALSET_SHA256
    assert len(loaded.prompts) == 200


# --------------------------------------------------------------------------- the CSV contract


def test_the_header_is_exactly_the_eight_columns(rows):
    with EVALSET.open(newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == HEADER


def test_two_hundred_prompts_with_unique_ids(rows):
    assert len(rows) == 200
    ids = [r["prompt_id"] for r in rows]
    assert len(set(ids)) == 200
    assert ids == sorted(ids), "ids are p001..p200 in order, so a diff shows an insertion"


def test_at_least_twelve_categories_each_with_at_least_eight_prompts(rows):
    counts = Counter(r["category"] for r in rows)
    assert len(counts) >= 12
    assert min(counts.values()) >= 8
    # The three the exit bar is stated on (plan §B8) must be there by name.
    assert {"qh_mode", "elm_rmp", "fast_ions"} <= set(counts)


def test_exactly_twenty_hand_graded_prompts_and_every_one_has_a_rubric(rows):
    graded = [r for r in rows if r["hand_graded"] == "1"]
    assert len(graded) == 20
    for r in graded:
        assert len(r["notes"]) > 40, f"{r['prompt_id']} has no rubric"


def test_the_annotated_rows_are_exactly_the_ones_the_changelog_names(rows):
    """`notes` on a non-hand-graded row is a v1.1 annotation, and the set of them is pinned.

    In v1.0 the column was empty on every ungraded row, and asserting that was how an annotation
    could not be slipped in unnoticed. v1.1 needs annotations -- eleven corrections and ten rows
    that are unanswerable on this corpus -- so the guarantee moves from "none" to "exactly
    these", which is the same guarantee with a list attached. Every id below is in README.md's
    changelog.
    """
    annotated = {r["prompt_id"] for r in rows if r["hand_graded"] == "0" and r["notes"]}
    assert annotated == {
        # corrected in v1.1
        "p036", "p038", "p039", "p041", "p046", "p048", "p066", "p072", "p186", "p197",
        # annotated, not changed: a lexicon plural gap
        "p053",
        # annotated, not changed: unanswerable on the whole corpus
        "p079", "p100", "p120", "p125", "p134", "p163", "p170",
        # annotated, not changed: satisfiable corpus-wide, empty on the eval split
        "p028", "p123", "p166",
    }
    # p111 is hand-graded AND unanswerable, so its rubric and its annotation are both there.
    p111 = next(r for r in rows if r["prompt_id"] == "p111")
    assert "unanswerable on this corpus" in p111["notes"] and p111["notes"].count(" || ") == 1
    for r in rows:
        if r["notes"] and r["hand_graded"] == "0":
            assert r["notes"].startswith("v1.1"), r["prompt_id"]


def test_no_two_prompts_ask_the_same_question(rows):
    """v1.0 shipped `p079` and `p186` with byte-identical expectations and near-identical text,
    so one unanswerable question was charged twice against coverage. v1.1 replaced `p186`."""
    seen: dict[tuple, str] = {}
    for r in rows:
        key = (r["expect_phenomena"], r["expect_constraints"], r["expect_segment"], r["prompt"])
        assert key not in seen, f"{r['prompt_id']} duplicates {seen.get(key)}"
        seen[key] = r["prompt_id"]
    assert len({r["prompt"] for r in rows}) == 200


def test_the_hand_graded_prompts_are_spread_over_the_categories(rows):
    graded = Counter(r["category"] for r in rows if r["hand_graded"] == "1")
    assert len(graded) >= 12
    assert max(graded.values()) <= 2


def test_every_expected_phenomenon_is_a_registry_id(rows):
    registry = set(ph_mod.registry())
    for r in rows:
        for pid in filter(None, r["expect_phenomena"].split("|")):
            assert pid in registry, f"{r['prompt_id']}: {pid!r} is not a phenomenon"


def test_every_expected_segment_is_a_real_segment_name(rows):
    from typing import get_args

    names = set(get_args(SegName))
    assert {r["expect_segment"] for r in rows} <= names


def test_every_constraint_column_is_one_the_builder_writes(rows):
    """A constraint on a column `ShotDB.mask` does not know raises KeyError at query time, which
    would make the harness crash instead of measuring. The names are rebuilt here from the signal
    registry the way `shotdb.build` builds them, so a renamed signal fails this rather than the
    real-database run.

    Being a column is NOT the same as being recorded: `q95_mean` is in this set and is NaN for
    every shot of `recommender_v1`. That is the harness's business to report, not this test's.
    """
    shot = 196336  # any shot of the development universe; the registry is banded by shot number
    known = {
        f"{spec.name}_{stat}"
        for spec in config.expand_registry(shot, include_not_installed=True)
        for stat in spec.stats
    }
    known |= {
        f"{sysdef.prefix}_total_{stat}"
        for sysdef in config.actuator_systems(shot).values()
        for stat in sysdef.stats
    }
    for r in rows:
        for col in ev.parse_constraints(r["expect_constraints"]):
            assert col in known, f"{r['prompt_id']}: no segment column {col!r}"


def test_the_prompts_carry_no_leading_or_trailing_whitespace(rows):
    for r in rows:
        assert r["prompt"] == r["prompt"].strip() and r["prompt"]


# ------------------------------------------------------------------------------- the split


def test_the_split_is_a_function_of_the_run_id_only():
    """Shot-separated means run-separated here: two shots of one run day share a session, a
    mini-proposal and often a logbook sentence, so a split that cut between them would put a
    near-copy of an eval shot in dev."""
    doc = yaml.safe_load(SPLIT.read_text(encoding="utf-8"))
    assert set(doc["dev"]["run_ids"]) & set(doc["eval"]["run_ids"]) == set()
    assert set(doc["dev"]["shots"]) & set(doc["eval"]["shots"]) == set()


def test_the_committed_split_is_what_the_rule_produces_from_recommender_v1():
    """The file is frozen, but it is not a magic list: recomputing the rule over the committed
    shot list has to reproduce it exactly."""
    doc = yaml.safe_load(SPLIT.read_text(encoding="utf-8"))
    rebuilt = ev.build_split(config.CONFIG_DIR / "shot_lists" / "recommender_v1.yaml")
    # The WHOLE document, not just the two shot lists: `n_shots`, `n_run_ids`, `buckets`,
    # `eval_bucket_max` and the run-id lists are part of what a reader trusts, and comparing only
    # the shots left them free to drift.
    assert rebuilt == doc


def test_the_split_covers_all_five_hundred_shots_about_four_to_one():
    doc = yaml.safe_load(SPLIT.read_text(encoding="utf-8"))
    n_dev, n_eval = len(doc["dev"]["shots"]), len(doc["eval"]["shots"])
    assert n_dev + n_eval == 500
    assert 80 <= n_eval <= 130, "the brief asks for ~400/100; this is the measured outcome"


def test_split_of_is_stable_across_processes():
    """sha256 of the run id, not `hash()`: Python's string hash is salted per process, and a
    split that moved between runs would not be a split."""
    assert ev.split_of("20210412") == ev.split_of("20210412")
    assert ev.split_of("20230616") in {"dev", "eval"}
    assert ev.run_bucket("20210412") == 944  # a fixed point of the rule, recomputed by hand


def test_loading_one_side_of_the_split_returns_only_that_sides_shots():
    doc = yaml.safe_load(SPLIT.read_text(encoding="utf-8"))
    assert ev.split_shots("eval") == set(doc["eval"]["shots"])
    assert ev.split_shots("dev") == set(doc["dev"]["shots"])
    assert ev.split_shots("all") == set(doc["eval"]["shots"]) | set(doc["dev"]["shots"])


def test_an_unknown_split_name_is_refused():
    with pytest.raises(ValueError, match="split"):
        ev.split_shots("holdout")
