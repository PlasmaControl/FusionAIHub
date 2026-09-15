"""The prompt harness: what it measures, and the three things it must never do.

It must never let a dev shot into an eval-split run (that is the whole point of the split), it
must never crash on a prompt it cannot answer (a miss is a datum, a traceback is a lost run), and
it must never report the machine proxy as if it were one of the 20 human grades.

Every database here is synthetic and MiniLM is stubbed: nothing in this file is a statement about
the encoder, and loading one would cost several seconds per test.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from shot_design.eval import prompts as ev
from shot_design.retrieval import channels as ch_mod
from shot_design.schema import EvalReport, Range
from shot_design.shotdb import store

from .conftest import shot_record, write_db

# Two run days that land on opposite sides of the frozen rule, checked below so the fixture
# cannot quietly stop testing the split.
DEV_RUN, EVAL_RUN = "20210412", "20230616"


@pytest.fixture(autouse=True)
def _no_minilm(monkeypatch):
    """A fixed query vector. The text channel is then a deterministic cosine against the
    hand-written database vectors, which is what every test here actually wants."""
    from shot_design.shotdb import text as text_mod

    monkeypatch.setattr(
        text_mod, "embed_texts", lambda texts: np.tile([1.0, 0.0, 0.0], (len(texts), 1))
    )


@pytest.fixture
def db(tmp_path: Path) -> store.ShotDB:
    """Six shots over two run days: 300/301/302 in dev, 400/401/402 in eval."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    write_db(
        db_dir,
        [
            shot_record(300, DEV_RUN, 1.0e6, 5.0e6, "QH-mode with a clear EHO at low torque"),
            shot_record(301, DEV_RUN, 1.1e6, 5.5e6, "repeat of 300, still QH-mode"),
            shot_record(302, DEV_RUN, 1.2e6, 6.0e6, "tearing mode locked and disrupted"),
            shot_record(400, EVAL_RUN, 1.3e6, 2.0e6, "ELM suppression with n=3 RMP", betan=2.6),
            shot_record(401, EVAL_RUN, 1.4e6, 2.5e6, "sawtooth crashes on the core ECE"),
            shot_record(402, EVAL_RUN, 1.5e6, 3.0e6, "L-H transition power threshold scan"),
        ],
    )
    return store.ShotDB.load(db_dir)


@pytest.fixture
def split_file(tmp_path: Path) -> Path:
    path = tmp_path / "split.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "rule": ev.SPLIT_RULE,
                "dev": {"run_ids": [DEV_RUN], "shots": [300, 301, 302]},
                "eval": {"run_ids": [EVAL_RUN], "shots": [400, 401, 402]},
            }
        ),
        encoding="utf-8",
    )
    return path


ROWS = [
    # id, category, prompt, phenomena, constraints, segment, graded, notes
    ("p001", "qh_mode", "QH-mode with a clear EHO", "qh|eho", "", "flat_top", "1", "x" * 50),
    ("p002", "elm_rmp", "ELM suppression with n=3 RMP", "elm", "", "flat_top", "0", ""),
    ("p003", "tearing", "island growth after the ECH went off", "tearing", "", "flat_top", "0", ""),
    ("p004", "actuator_only", "Ip above 1.35 MA", "", '{"ip_mean":[1350000.0,null]}',
     "flat_top", "1", "y" * 50),
    ("p005", "actuator_only", "nothing can be this big", "", '{"ip_mean":[9e9,null]}',
     "flat_top", "0", ""),
    ("p006", "nbi_program", "beam power stepped down at 3 s", "", "", "flat_top", "0", ""),
]


def _evalset(tmp_path: Path, rows=ROWS) -> ev.Evalset:
    path = tmp_path / "prompts.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            ["prompt_id", "category", "prompt", "expect_phenomena", "expect_constraints",
             "expect_segment", "hand_graded", "notes"]
        )
        w.writerows(rows)
    return ev.load_evalset(path)


# --------------------------------------------------------------------------------- the split


def test_the_two_fixture_run_days_really_are_on_opposite_sides():
    assert ev.split_of(DEV_RUN) == "dev"
    assert ev.split_of(EVAL_RUN) == "eval"


def test_an_eval_run_never_returns_a_dev_shot(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="eval", split_path=split_file)
    returned = {s for o in report.outcomes for s in o.shots}
    assert returned and returned <= {400, 401, 402}
    assert report.split_shots == 3


def test_a_dev_run_never_returns_an_eval_shot(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="dev", split_path=split_file)
    returned = {s for o in report.outcomes for s in o.shots}
    assert returned and returned <= {300, 301, 302}


def test_the_split_is_applied_before_the_search_not_after(db, split_file, tmp_path):
    """Filtering the results afterwards would leave the dev shots in the BM25 corpus statistics
    and in every k-NN neighbourhood. Excluding them means the candidate count itself shrinks."""
    evalset = _evalset(tmp_path)
    whole = ev.run(db, evalset, split="all", split_path=split_file)
    half = ev.run(db, evalset, split="eval", split_path=split_file)
    assert whole.outcomes[0].candidates == 6
    assert half.outcomes[0].candidates == 3


def test_split_all_is_labelled_as_such_in_the_report(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    assert report.split == "all"
    assert report.split_shots == 6
    assert "split `all`" in ev.markdown(report)


# ------------------------------------------------------------------------------ the metrics


def test_coverage_is_the_fraction_of_prompts_that_got_anything_at_all(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    empty = [o for o in report.outcomes if o.n_results == 0]
    assert [o.prompt_id for o in empty] == ["p005"]  # the impossible Ip range
    assert report.coverage == pytest.approx(5 / 6)


def test_hard_filter_survival_is_candidates_over_segment_rows(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    by_id = {o.prompt_id: o for o in report.outcomes}
    assert by_id["p001"].candidates == 6 and by_id["p001"].segment_rows == 6
    assert by_id["p004"].candidates == 2  # ip_mean >= 1.35 MA: shots 401 and 402
    assert by_id["p005"].candidates == 0
    assert report.hard_filter_survival == pytest.approx(
        sum(o.candidates / o.segment_rows for o in report.outcomes) / 6
    )


def test_channel_participation_counts_the_channels_that_had_something_to_say(
    db, split_file, tmp_path
):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    # Text is stubbed and every fixture shot carries a log entry, so the dense channel fires on
    # every prompt; ignite_knn has no embeddings in this database and fires on none.
    assert report.channel_participation["text_knn"] == pytest.approx(5 / 6)
    assert report.channel_participation["ignite_knn"] == 0.0
    # The phenomenon channel reads the four evidence classes -- event rows, label products,
    # `text_claims` rows and the curated tables -- and `write_db` writes none of them: the
    # fixture's log sentences are dense/BM25 text, not claims. So it has nothing to say on any
    # prompt, p001's resolvable "QH-mode with a clear EHO" included.
    assert report.channel_participation["phenomenon"] == 0.0
    # Every registered channel is reported, so a new channel cannot go unmeasured.
    assert set(report.channel_participation) == set(ch_mod.CHANNELS) == {
        "scalar_knn", "text_knn", "bm25", "ignite_knn", "phenomenon"
    }


def test_run_diversity_counts_distinct_run_days_not_shots(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    assert {o.prompt_id: len(o.run_ids) for o in report.outcomes}["p001"] == 2
    assert report.mean_run_diversity > 0


def test_run_diversity_averages_over_the_prompts_that_ANSWERED(db, split_file, tmp_path):
    """A prompt that returned nothing contributes no top-10 to be diverse. Dividing by it made
    "mean distinct run days per top-10" a smaller number than any top-10 ever showed -- 7.91 on
    the real run against 8.46 over the prompts that actually answered."""
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    answered = [o for o in report.outcomes if o.n_results > 0]
    assert len(answered) == 5 and len(report.outcomes) == 6  # p005 retrieves nothing
    assert report.mean_run_diversity == pytest.approx(
        sum(len(o.run_ids) for o in answered) / len(answered)
    )
    by_cat = {c.category: c for c in report.categories}
    # `actuator_only` holds p004 (answers) and p005 (does not); only p004 is averaged.
    p004 = next(o for o in report.outcomes if o.prompt_id == "p004")
    assert by_cat["actuator_only"].mean_run_diversity == pytest.approx(len(p004.run_ids))


def test_a_category_where_nothing_answered_reports_no_diversity_rather_than_zero(
    db, split_file, tmp_path
):
    rows = [("p001", "impossible", "nothing is this big", "", '{"ip_mean":[9e9,null]}',
             "flat_top", "0", "")]
    report = ev.run(db, _evalset(tmp_path, rows), split="all", split_path=split_file)
    assert report.categories[0].mean_run_diversity is None
    assert report.mean_run_diversity is None


def test_the_duplicate_rate_is_zero_when_every_result_is_a_different_shot(
    db, split_file, tmp_path
):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    assert all(o.duplicate_shots == 0 for o in report.outcomes)
    assert report.duplicate_rate == 0.0


# ----------------------------------------------------------------------------- resolution


def test_resolution_is_true_only_when_every_expected_phenomenon_resolves(
    db, split_file, tmp_path
):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    by_id = {o.prompt_id: o for o in report.outcomes}
    assert by_id["p001"].expected_resolved is True
    assert set(by_id["p001"].resolved) >= {"qh", "eho"}
    assert by_id["p002"].expected_resolved is True


def test_a_prompt_whose_words_are_not_aliases_is_recorded_as_a_miss_not_dropped(
    db, split_file, tmp_path
):
    """"island growth" is how an operator writes a tearing mode and is not in the lexicon. The
    harness's job is to say so, not to widen the lexicon until it passes."""
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    p003 = next(o for o in report.outcomes if o.prompt_id == "p003")
    assert p003.expected_resolved is False
    assert p003.resolved == []
    assert p003.n_results > 0  # it still retrieves; only the resolution missed


def test_a_prompt_expecting_no_phenomenon_has_no_resolution_verdict(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    assert next(o for o in report.outcomes if o.prompt_id == "p004").expected_resolved is None


def test_resolution_is_reported_per_category_and_skipped_where_nothing_expects_one(
    db, split_file, tmp_path
):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    by_cat = {c.category: c for c in report.categories}
    assert by_cat["qh_mode"].resolution == 1.0
    assert by_cat["tearing"].resolution == 0.0
    assert by_cat["actuator_only"].resolution is None
    assert by_cat["actuator_only"].n_with_expectation == 0
    assert report.resolution_overall == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- the proxy grade


def _write_claim(db_dir: Path, shot: int, phenomenon: str) -> None:
    from shot_design.labels.claims import CLAIMS_DTYPES

    pd.DataFrame(
        [{
            "shot": shot,
            "phenomenon": phenomenon,
            "polarity": "pos",
            "temporality": "observed",
            "snippet": "an operator wrote it down",
            "scope": "shot",
        }],
        columns=list(CLAIMS_DTYPES),
    ).astype(CLAIMS_DTYPES).to_parquet(db_dir / "text_claims.parquet", index=False)


def test_the_proxy_is_only_computed_for_hand_graded_prompts(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    graded = {o.prompt_id for o in report.outcomes if o.proxy_hit is not None}
    assert graded <= {"p001", "p004"}
    assert report.proxy.n_prompts == 2


def test_a_constraint_prompt_passes_the_proxy_only_if_every_top_five_shot_satisfies_it(
    db, split_file, tmp_path
):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    p004 = next(o for o in report.outcomes if o.prompt_id == "p004")
    assert p004.proxy_hit is True  # the hard filter already guarantees it


def test_a_phenomenon_prompt_misses_the_proxy_when_no_shot_carries_the_evidence(
    db, split_file, tmp_path
):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    p001 = next(o for o in report.outcomes if o.prompt_id == "p001")
    assert p001.proxy_hit is False  # nothing in this database claims qh or eho


def test_a_phenomenon_prompt_hits_the_proxy_once_a_shot_carries_the_evidence(
    tmp_path, split_file, db, monkeypatch
):
    _write_claim(db.db_dir, 300, "qh")
    rows = [r for r in ROWS if r[0] != "p001"]
    rows.insert(0, ("p001", "qh_mode", "QH-mode", "qh", "", "flat_top", "1", "z" * 50))
    reloaded = store.ShotDB.load(db.db_dir)
    report = ev.run(reloaded, _evalset(tmp_path, rows), split="all", split_path=split_file)
    p001 = next(o for o in report.outcomes if o.prompt_id == "p001")
    assert p001.proxy_hit is True
    assert 300 in p001.shots[: ev.PROXY_K]


def test_the_proxy_says_in_the_report_that_it_is_not_the_human_grade(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    assert "PROXY, NOT A HUMAN GRADE" in report.proxy.caveat
    assert "PROXY, NOT A HUMAN GRADE" in ev.markdown(report)


def test_a_hand_graded_prompt_with_no_machine_checkable_expectation_is_not_a_free_pass(
    db, split_file, tmp_path
):
    rows = [("p001", "mixed", "beam power stepped down at 3 s", "", "", "flat_top", "1", "q" * 50)]
    report = ev.run(db, _evalset(tmp_path, rows), split="all", split_path=split_file)
    assert report.outcomes[0].proxy_hit is None
    assert report.proxy.n_prompts == 1 and report.proxy.n_checkable == 0
    assert report.proxy.fraction is None


# ------------------------------------------------------------------------- refusing to crash


def test_a_constraint_on_a_column_this_database_lacks_is_reported_not_raised(
    db, split_file, tmp_path
):
    rows = [("p001", "actuator_only", "pellet rate above 2 Hz", "",
             '{"pellet_rate_mean":[2.0,null]}', "flat_top", "0", "")]
    report = ev.run(db, _evalset(tmp_path, rows), split="all", split_path=split_file)
    assert report.n_errors == 1
    assert "unknown column" in report.outcomes[0].error
    assert report.outcomes[0].n_results == 0
    assert "pellet_rate_mean" in ev.markdown(report)


# ------------------------------------------------------------------------------- the loader


def test_the_loader_parses_every_column_into_its_own_type(tmp_path):
    evalset = _evalset(tmp_path)
    p = {x.prompt_id: x for x in evalset.prompts}
    assert p["p001"].expect_phenomena == ["qh", "eho"]
    assert p["p001"].hand_graded is True and p["p002"].hand_graded is False
    assert p["p004"].expect_constraints == {"ip_mean": Range(lo=1.35e6, hi=None)}
    assert p["p002"].expect_constraints == {}
    assert p["p002"].expect_phenomena == ["elm"]
    assert p["p006"].expect_phenomena == []
    assert p["p001"].expect_segment == "flat_top"


def test_parse_constraints_keeps_an_open_bound_open():
    assert ev.parse_constraints('{"betan_mean":[2.0,null]}') == {
        "betan_mean": Range(lo=2.0, hi=None)
    }
    assert ev.parse_constraints('{"li_mean":[null,0.8]}') == {"li_mean": Range(lo=None, hi=0.8)}
    assert ev.parse_constraints("") == {} and ev.parse_constraints(None) == {}


# ------------------------------------------------------------------------------- the report


def test_the_report_carries_the_evalset_hash_so_a_number_cannot_be_misattributed(
    db, split_file, tmp_path
):
    evalset = _evalset(tmp_path)
    report = ev.run(db, evalset, split="eval", split_path=split_file)
    assert report.evalset_sha256 == evalset.sha256
    assert evalset.sha256 in ev.markdown(report)


def test_the_report_round_trips_through_json(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="eval", split_path=split_file)
    back = EvalReport.model_validate(json.loads(report.model_dump_json()))
    assert back.coverage == report.coverage
    assert [o.prompt_id for o in back.outcomes] == [o.prompt_id for o in report.outcomes]


def test_the_markdown_marks_the_two_bars_the_plan_states(db, split_file, tmp_path):
    report = ev.run(db, _evalset(tmp_path), split="eval", split_path=split_file)
    md = ev.markdown(report)
    assert "coverage (prompts with >= 1 result)" in md
    assert ">= 95.0 %" in md
    assert "| `qh_mode` |" in md
    assert "| channel | prompts it contributed to |" in md


def test_the_markdown_says_resolution_is_not_a_split_number(db, split_file, tmp_path):
    """`_resolution` reads the prompt text and the lexicon -- no database, no split -- so the
    column is identical on dev, eval and all. It shares a table with metrics that ARE split
    numbers, and a reader who assumes it is one will read `fast_ions 56.2 %` as a statement about
    the eval split's zero fast-ion shots."""
    md = ev.markdown(ev.run(db, _evalset(tmp_path), split="eval", split_path=split_file))
    assert "identical on every split" in md


def test_resolution_really_is_identical_on_every_split(db, split_file, tmp_path):
    evalset = _evalset(tmp_path)
    per_split = {}
    for split in ("dev", "eval", "all"):
        report = ev.run(db, evalset, split=split, split_path=split_file)
        per_split[split] = {c.category: c.resolution for c in report.categories}
    assert per_split["dev"] == per_split["eval"] == per_split["all"]


def test_the_proxy_grade_reads_evidence_at_the_floor_locate_uses(
    db, split_file, tmp_path, monkeypatch
):
    """The proxy grade is about the shots a user would have been shown, so `_proxy_hit` reads
    `evidence()` at the label floor `locate` passes -- explicitly, not by defaulting to it. A
    configured floor that `locate` honoured and this call left implicit would let the two drift
    apart silently the day the default and the configured value differ."""
    from shot_design.retrieval import phenomena as ph

    seen: list = []
    real = ph.evidence

    def spy(*args, **kwargs):
        seen.append(kwargs.get("label_floor", "MISSING"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ph, "evidence", spy)
    ev.run(db, _evalset(tmp_path), split="all", split_path=split_file)
    assert seen, "p001 is a hand-graded phenomenon prompt, so evidence() must have been read"
    assert set(seen) == {ph.configured_label_floor()}
