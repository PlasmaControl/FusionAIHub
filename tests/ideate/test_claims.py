"""`text_claims`: what the operators *said*, as a claim -- never as a label.

Plan section 2: "Text is never a label by itself." So a row of `text_claims.parquet` records who the
sentence is about (`scope`), whether it asserts or denies (`polarity`), and whether it reports,
plans or remembers (`temporality`), and keeps the sentence itself. Appendix C item 12 is the scope
rule: 2,302 of 22,950 shots have no shot table row of their own and fall back to the session's
text, and that text is a claim about the run, not about the shot.

The polarity half is not new logic: it is the negation/mitigation parsing `shotdb.text` already
uses for `verdict()`, measured on the real 53,179-record logbook, reused here per match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ideate.labels import claims

from .conftest import text_bundle

LEXICON = {
    "phenomena": {
        "eho": {"aliases": ["eho", "edge harmonic oscillation"]},
        "qh": {"aliases": ["qh", "qh-mode", "quiescent h-mode"]},
        "tearing": {"aliases": ["tearing", "tearing mode", "ntm"]},
    }
}


@pytest.fixture
def lexicon_path(tmp_path) -> Path:
    p = tmp_path / "lexicons.yaml"
    p.write_text(yaml.safe_dump(LEXICON))
    return p


@pytest.fixture
def text_root(tmp_path) -> Path:
    root = tmp_path / "text"
    (root / "sql").mkdir(parents=True)
    (root / "shotsummary" / "processed" / "per_shot_txt").mkdir(parents=True)
    return root


def write_log(text_root: Path, records: dict[int, str]) -> None:
    """`sql/logs.jsonl` with one CHIEF_OPERATOR entry per shot."""
    lines = [
        json.dumps({
            "shot": shot,
            "log_text": f"### [CHIEF_OPERATOR] byrnep 2021-04-13 12:00:00\n{body}\n",
        })
        for shot, body in records.items()
    ]
    (text_root / "sql" / "logs.jsonl").write_text("".join(ln + "\n" for ln in lines))


def write_bundle(text_root: Path, shot: int, **kw) -> None:
    path = text_root / "shotsummary" / "processed" / "per_shot_txt" / f"shot_{shot}.txt"
    path.write_text(text_bundle(shot, **kw))


def claim_rows(df: pd.DataFrame, phenomenon: str) -> list[tuple[str, str, str]]:
    sel = df[df["phenomenon"] == phenomenon]
    return list(zip(sel["polarity"], sel["temporality"], sel["scope"], strict=True))


# ------------------------------------------------------------------- polarity and temporality


@pytest.mark.parametrize(
    ("sentence", "phenomenon", "polarity", "temporality"),
    [
        ("No EHO observed.", "eho", "neg", "observed"),
        ("Strong EHO through the flat top.", "eho", "pos", "observed"),
        ("Plan to get QH next shot.", "qh", "pos", "planned"),
        ("We will try to reach QH at low torque.", "qh", "pos", "planned"),
        ("Like shot 190090 the EHO is clean.", "eho", "pos", "historical"),
        ("Repeat of the previous tearing case.", "tearing", "pos", "historical"),
        ("Maybe a tearing mode near 3 s, hard to tell.", "tearing", "uncertain", "observed"),
        ("Never saw a tearing mode.", "tearing", "neg", "observed"),
    ],
)
def test_one_sentence_one_claim(text_root, lexicon_path, sentence, phenomenon, polarity,
                                temporality):
    write_log(text_root, {900001: sentence})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, phenomenon) == [(polarity, temporality, "shot")]
    assert df["snippet"].iloc[0] == sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "Implement locked mode dud trip to avoid the tearing mode.",   # purpose first
        "Watch the tearing mode and dud trip to avoid it.",            # purpose last
    ],
)
def test_a_planned_mitigation_is_a_planned_absence_either_way_round(text_root, lexicon_path,
                                                                    sentence):
    """A fault named as the thing a protection exists to prevent has not happened. `shotdb.text`
    refuses to grade it as a fault; a claim table must not record it as an observation -- and the
    two orders the corpus writes it in have to agree with each other."""
    write_log(text_root, {900001: sentence})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, "tearing") == [("neg", "planned", "shot")]


def test_a_negation_does_not_reach_across_a_clause(text_root, lexicon_path):
    """The guard `verdict()` uses, per match: "no gas, EHO fine" must not negate the EHO."""
    write_log(text_root, {900001: "No gas, EHO fine."})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, "eho") == [("pos", "observed", "shot")]


def test_a_sentence_can_carry_two_phenomena(text_root, lexicon_path):
    write_log(text_root, {900001: "QH with no tearing."})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, "qh") == [("pos", "observed", "shot")]
    assert claim_rows(df, "tearing") == [("neg", "observed", "shot")]


# ------------------------------------------------------------- text about a different shot


def test_a_last_shot_block_is_historical_and_the_rest_of_the_entry_is_not(text_root, lexicon_path):
    """A "Last shot:" heading opens a span about a *previous* discharge, and the corpus writes it
    as a heading on its own line with the prose under it, terminated by a blank line. Every
    sentence of that span is a claim about the other shot; nothing in it may be dated `observed`,
    which would make this shot's row report what the previous one did. `shotdb.text._SHOT_SCOPE`
    is what draws the span, so the boundary here is the one `verdict()` already grades by."""
    write_log(text_root, {900001: (
        "Last shot: strong EHO all through the flat top.\n"
        "The tearing mode locked at 3 s.\n"
        "\n"
        "QH sustained here from 2 s."
    )})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, "eho") == [("pos", "historical", "shot")]
    assert claim_rows(df, "tearing") == [("pos", "historical", "shot")]
    assert claim_rows(df, "qh") == [("pos", "observed", "shot")]


def test_a_next_shot_block_is_planned_not_observed(text_root, lexicon_path):
    """The same machinery, the other direction: what is written under "Next shot:" has not
    happened either, and calling it `historical` would be as wrong as calling it `observed`."""
    write_log(text_root, {900001: (
        "EHO clean here.\n"
        "Next shot: raise the torque.\n"
        "Give the tearing mode room to grow.\n"
    )})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, "eho") == [("pos", "observed", "shot")]
    assert claim_rows(df, "tearing") == [("pos", "planned", "shot")]


def test_a_current_shot_heading_is_still_this_shots(text_root, lexicon_path):
    """`_SHOT_SCOPE` matches "Current shot:"/"This shot:" too, and those are not back-references:
    the text under them is exactly what the entry is reporting. Only the four markers
    `shotdb.text._OTHER_SHOT` calls another discharge's may move a claim off this shot."""
    write_log(text_root, {900001: "Current shot: EHO through the flat top."})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, "eho") == [("pos", "observed", "shot")]


# ---------------------------------------------------------------------------------- scope


def test_session_text_before_the_shot_marker_is_run_scope(text_root, lexicon_path):
    write_bundle(text_root, 900001, row={"SHOT_TYPE": "plasma"}, title="Tearing mode avoidance")

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, "tearing")
    assert set(df["scope"]) == {"run"}


def test_a_session_fallback_bundle_is_all_run_scope(text_root, lexicon_path):
    """`row=None` is the session-fallback case: the summary page had no row for this shot, so
    nothing in the bundle is the shot's (Appendix C item 12)."""
    write_bundle(text_root, 900001, row=None, title="Tearing mode avoidance")

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert set(df["scope"]) == {"run"}


def test_the_mini_proposal_is_planned_whatever_verbs_it_uses(text_root, lexicon_path):
    """The bundle's mini-proposal section is a statement of intent written before the run day, so
    "Hypothesis to be tested: the EHO survives" is a plan even though it reads as a report. This
    is the section a consumer must never count as evidence that the run saw the phenomenon."""
    write_bundle(text_root, 900001, row={"SHOT_TYPE": "plasma"}, title="EHO at low torque")

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    # The same title is in both blocks and gets a different date from each, which is the point:
    # the session metadata records what the run day was called, the mini-proposal proposes it.
    assert set(claim_rows(df, "eho")) == {("pos", "observed", "run"), ("pos", "planned", "run")}
    planned = df[(df["phenomenon"] == "eho") & (df["temporality"] == "planned")]
    assert planned["snippet"].str.contains("Subject").any()


def test_the_shot_block_of_a_real_bundle_is_shot_scope(text_root, lexicon_path):
    write_bundle(text_root, 900001, row={"SHOT_TYPE": "plasma"}, title="Density scan",
                 pre_blocks="PCS: EHO feedback armed.")

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert claim_rows(df, "eho") == [("pos", "observed", "shot")]


# ------------------------------------------------------------------------- shape and absence


def test_columns_and_dtypes_are_the_contract(text_root, lexicon_path):
    write_log(text_root, {900001: "Strong EHO."})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)
    empty = claims.text_claims([], text_root=text_root, lexicon_path=lexicon_path)

    assert list(df.columns) == list(claims.CLAIMS_COLUMNS)
    assert list(empty.columns) == list(claims.CLAIMS_COLUMNS)
    for name, dtype in claims.CLAIMS_DTYPES.items():
        assert df[name].dtype == dtype, name
        assert empty[name].dtype == dtype, name
    assert set(df["polarity"]) <= set(claims.POLARITIES)
    assert set(df["temporality"]) <= set(claims.TEMPORALITIES)
    assert set(df["scope"]) <= set(claims.SCOPES)


def test_a_shot_with_no_text_at_all_makes_no_claims(text_root, lexicon_path):
    write_log(text_root, {900001: "Strong EHO."})

    df = claims.text_claims([900001, 900002], text_root=text_root, lexicon_path=lexicon_path)

    assert set(df["shot"]) == {900001}


def test_an_absent_text_root_is_no_claims_not_an_error(tmp_path, lexicon_path):
    df = claims.text_claims([900001], text_root=tmp_path / "nope", lexicon_path=lexicon_path)
    assert df.empty
    assert list(df.columns) == list(claims.CLAIMS_COLUMNS)


def test_the_same_sentence_is_not_claimed_twice(text_root, lexicon_path):
    write_log(text_root, {900001: "Strong EHO. Strong EHO."})

    df = claims.text_claims([900001], text_root=text_root, lexicon_path=lexicon_path)

    assert len(df) == 1


# -------------------------------------------------------------------------------- lexicon


def test_the_lexicon_reads_labelmakers_shape(lexicon_path):
    lex = claims.load_lexicon(lexicon_path)
    assert lex.source == lexicon_path
    assert lex.aliases["eho"] == ("edge harmonic oscillation", "eho")  # longest first


def test_the_lexicon_reads_a_bare_alias_list(tmp_path):
    p = tmp_path / "lex.yaml"
    p.write_text(yaml.safe_dump({"eho": ["eho", "edge harmonic oscillation"]}))
    assert claims.load_lexicon(p).aliases["eho"] == ("edge harmonic oscillation", "eho")


def config_dir():
    from ideate import config

    return config.CONFIG_DIR


def test_the_lexicon_falls_back_to_ideates_own_themes(tmp_path):
    """`labelmaker/events/lexicons.yaml` lands from the labelmaker workstream. Until it does, the
    themes of `configs/ideate/labels.yaml` are the alias lists."""
    from ideate import config

    lex = claims.load_lexicon(config.CONFIG_DIR / "labels.yaml")

    assert "qh_mode" in lex.aliases
    assert "eho" in lex.aliases["qh_mode"]


def test_the_default_lexicon_resolves_to_a_file_that_exists_and_loads():
    """Whichever of the two it is today, `join()` must be able to read it without being told."""
    path = claims.default_lexicon_path()
    assert path in (claims.labelmaker_lexicon_path(), config_dir() / "labels.yaml")
    assert path.exists()
    assert claims.load_lexicon(path).aliases


def test_an_excluded_alias_vetoes_the_sentence(text_root, tmp_path):
    p = tmp_path / "lex.yaml"
    p.write_text(yaml.safe_dump(
        {"phenomena": {"eho": {"aliases": ["eho"], "exclude": ["echo"]}}}
    ))
    write_log(text_root, {900001: "EHO seen in the echo chamber.", 900002: "EHO seen."})

    df = claims.text_claims([900001, 900002], text_root=text_root, lexicon_path=p)

    assert set(df["shot"]) == {900002}
