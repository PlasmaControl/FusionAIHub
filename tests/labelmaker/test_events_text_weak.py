"""The phenomenon lexicon, and what the operators' text is allowed to claim.

Every fixture here is a per-shot bundle written into `tmp_path` in the real
corpus's layout - the header, `## General session context`, the marker line,
then the shot's own text and its `SHOT TABLE ROW` block - so that a test
asserts what was PUT in the file. Nothing reads
`/scratch/gpfs/EKOLEMEN/big_d3d_data`; the real-data numbers live in the
task report.
"""
from __future__ import annotations

import json
import math

import pytest

from labelmaker.events import text_weak as tw

MARKER = "## Shot-specific context (from summary.html)"

TABLE = {
    "SHOT": "198658",
    "SHOT_TYPE": "plasma",
    "TIME-OF-SHOT": "10:08",
    "PULSE-LENGTH": "6.12",
    "IP-(MA)": "-1.00",
}


def _write(root, shot, *, session="", shot_text="", table=TABLE, marker=True):
    """One synthetic per-shot bundle; returns its path."""
    parts = [
        "# DIII-D per-shot text bundle",
        "",
        "RUN_ID: 20240517",
        "",
        f"SHOT: {shot}",
        "",
        "## General session context",
        session,
    ]
    if marker:
        parts.append(MARKER)
        if shot_text or table is not None:
            parts += [f"SHOT: {shot}", "", shot_text]
        if table is not None:
            parts.append("SHOT TABLE ROW (name -> value)")
            parts += [f"- {k}: {v}" for k, v in table.items()]
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"shot_{shot}.txt"
    path.write_text("\n".join(parts) + "\n")
    return path


@pytest.fixture
def lex():
    return tw.load_lexicon()


# ------------------------------------------------------------- the lexicon

def test_the_shipped_lexicon_carries_the_twelve_round_one_ids(lex):
    # Plan 5.6's round-1 ids. This file is the SINGLE source of them: ideate
    # reads it too, so an id renamed here is renamed there.
    assert lex.version == 1
    assert set(lex.ids) == set(tw.PHENOMENON_IDS)
    assert len(lex.ids) == 12
    for p in lex.phenomena:
        assert p.title and p.aliases and p.weight > 0.0
        assert all(a == a.lower() and a.strip() for a in p.aliases)
        assert all(n == n.lower() and n.strip() for n in p.negatives)
    assert "edge harmonic oscillation" in lex["eho"].aliases
    assert "sawteeth" in lex["sawtooth"].aliases
    assert "elm-free" in lex["elm"].negatives


def test_an_uppercase_alias_is_a_lexicon_error(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text(
        "version: 1\nphenomena:\n  elm:\n    title: ELM\n    aliases: [ELM]\n"
    )
    with pytest.raises(tw.LexiconError, match="lowercase"):
        tw.load_lexicon(path)


def test_an_unknown_phenomenon_id_is_a_lexicon_error(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text(
        "version: 1\nphenomena:\n  wobble:\n    title: W\n    aliases: [wobble]\n"
    )
    with pytest.raises(tw.LexiconError, match="wobble"):
        tw.load_lexicon(path)


def test_a_phenomenon_with_no_aliases_is_a_lexicon_error(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text("version: 1\nphenomena:\n  elm:\n    title: ELM\n    aliases: []\n")
    with pytest.raises(tw.LexiconError, match="aliases"):
        tw.load_lexicon(path)


def test_a_weight_defaults_to_one_and_is_read_when_given(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text(
        "version: 1\nphenomena:\n"
        "  elm:\n    title: ELM\n    aliases: [elm]\n"
        "  rwm:\n    title: RWM\n    aliases: [rwm]\n    weight: 0.4\n"
    )
    lex = tw.load_lexicon(path)
    assert lex["elm"].weight == 1.0
    assert lex["rwm"].weight == 0.4
    # It is the CONSUMER's knob: `text_events` weighs every mention alike.
    _write(tmp_path, 30, shot_text="rwm grew")
    assert tw.text_events(30, lex, root=tmp_path)[0].confidence \
        == pytest.approx(tw.TEXT_ONLY_CEILING)


def test_a_weight_that_is_not_positive_is_a_lexicon_error(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text(
        "version: 1\nphenomena:\n  elm:\n    title: ELM\n"
        "    aliases: [elm]\n    weight: 0\n"
    )
    with pytest.raises(tw.LexiconError, match="weight"):
        tw.load_lexicon(path)


# ------------------------------------------------------- reading one bundle

def test_the_marker_splits_the_session_text_from_the_shots_own(tmp_path):
    _write(tmp_path, 198658, session="sawtooth crashes all day",
           shot_text="strong eho on this one")
    assert "eho" in tw.shot_block(198658, root=tmp_path)
    assert "sawtooth" not in tw.shot_block(198658, root=tmp_path)
    assert "sawtooth" in tw.run_context(198658, root=tmp_path)
    assert "eho" not in tw.run_context(198658, root=tmp_path)


def test_a_bundle_with_nothing_after_the_marker_has_an_empty_shot_block(tmp_path):
    # The session-fallback case: the summary carried nothing for this shot,
    # so the file is the run's text and the marker, and the shot's own block
    # is empty. The run context is still there and still readable.
    _write(tmp_path, 1, session="elmy h-mode", shot_text="", table=None)
    assert tw.shot_block(1, root=tmp_path) == ""
    assert "elmy h-mode" in tw.run_context(1, root=tmp_path)


def test_a_bundle_whose_summary_had_no_table_reads_as_no_table(tmp_path):
    # The other half of the fallback in the real corpus: the marker and the
    # shot's number are there, and the scraper found no table under them.
    path = tmp_path / "shot_3.txt"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "SHOT: 3\n\n## General session context\nelmy\n"
        f"{MARKER}\nSHOT: 3\n\nSHOT TABLE ROW (name -> value)\n"
        "(Shot table key/value mapping not found.)\n"
    )
    assert tw.shot_table_row(path.read_text()) == {}
    assert tw.hits(tw.shot_block(3, root=tmp_path), tw.load_lexicon()) == {}


def test_a_bundle_with_no_marker_at_all_is_all_run_context(tmp_path):
    _write(tmp_path, 2, session="locked mode", marker=False)
    assert tw.shot_block(2, root=tmp_path) == ""
    assert "locked mode" in tw.run_context(2, root=tmp_path)


def test_a_missing_file_reads_as_empty_and_not_as_an_error(tmp_path):
    assert tw.shot_block(999999, root=tmp_path) == ""
    assert tw.run_context(999999, root=tmp_path) == ""


def test_the_shot_table_row_reads_as_a_mapping(tmp_path):
    text = _write(tmp_path, 198658).read_text()
    row = tw.shot_table_row(text)
    assert row["SHOT_TYPE"] == "plasma"
    assert row["PULSE-LENGTH"] == "6.12"
    assert row["IP-(MA)"] == "-1.00"
    # A value with a colon in it is a value, not a second key.
    assert row["TIME-OF-SHOT"] == "10:08"
    assert tw.shot_table_row("nothing here") == {}


# ---------------------------------------------------------------- matching

def test_sentences_split_on_terminators_and_newlines_and_lowercase():
    got = tw.sentences("Big ELM. Then\n  a  SAWTOOTH crash;  eho?  ")
    assert got == ["big elm", "then", "a sawtooth crash", "eho"]
    assert tw.sentences("") == []


def test_a_word_is_matched_whole_and_not_inside_another(lex):
    # The spec's own example: " nt " is not "want". A lexicon that matched
    # substrings would claim a tearing mode on every "want".
    assert tw.hits("we want more current", lex) == {}
    assert tw.hits("helms deep", lex) == {}
    got = tw.hits("elms are back", lex)
    assert [h.alias for h in got["elm"]] == ["elms"]


def test_a_two_letter_alias_does_not_match_inside_a_word(tmp_path):
    # Spec A7's named acceptance, spelled with its own example: `" nt "` is
    # not "want". The shipped lexicon has no two-letter alias that lands
    # inside a common word, so the guard is tested where it bites - on a
    # lexicon that does.
    path = tmp_path / "lex.yaml"
    path.write_text(
        "version: 1\nphenomena:\n  tearing:\n    title: T\n    aliases: [nt]\n"
    )
    lex = tw.load_lexicon(path)
    assert tw.hits("we want more current", lex) == {}
    assert tw.hits("an nt grew at 2 s", lex)["tearing"][0].alias == "nt"


def test_a_mode_number_is_one_token(lex):
    got = tw.hits("2/1 grew and locked", lex)
    assert {h.alias for h in got["tearing"]} == {"2/1"}
    # "21" is not "2/1", and neither is the "1" of "2/1" on its own.
    assert tw.hits("21 kHz", lex) == {}


def test_a_hyphenated_word_is_one_token(lex):
    # "elm-free" is not an ELM: the hyphen is INSIDE the token, so the alias
    # "elm" cannot match it.
    assert [h.alias for h in tw.hits("elm-free h-mode", lex)["elm"]] == ["elm-free"]


def test_a_name_inside_an_identifier_is_not_a_mention(lex):
    # The session text carries the PCS parameter dump, where `RWM_GAINMULT`
    # is the name of a knob and not a claim that a resistive wall mode
    # happened. The underscore is kept inside the token so that it stays one
    # word; spelled out with a space it is a mention like any other.
    assert tw.hits("change parameter data: RWM_GAINMULT/ShotStart", lex) == {}
    assert "rwm" in tw.hits("rwm feedback was on", lex)


def test_a_negated_sentence_yields_only_negative_hits(lex):
    got = tw.hits("elm-free h-mode all shot", lex)
    assert [h.polarity for h in got["elm"]] == ["neg"]
    assert [h.polarity for h in tw.hits("no sawteeth after 2 s", lex)["sawtooth"]] \
        == ["neg"]
    # The negation is per SENTENCE and per phenomenon: the next sentence is
    # a positive, and a negated ELM says nothing about the tearing mode.
    got = tw.hits("no elms. sawtooth crashes throughout", lex)
    assert [h.polarity for h in got["elm"]] == ["neg"]
    assert [h.polarity for h in got["sawtooth"]] == ["pos"]


def test_two_phenomena_in_one_sentence_are_two_hits(lex):
    got = tw.hits("eho and a 3/2 tearing mode together", lex)
    assert set(got) == {"eho", "tearing"}
    assert all(h.sentence == "eho and a 3/2 tearing mode together"
               for hs in got.values() for h in hs)


def test_overlapping_aliases_count_a_sentence_once(lex):
    # "edge harmonic oscillation" contains "edge harmonic", and one mention
    # is one mention: counting both would double the confidence of a
    # phenomenon whose lexicon happens to spell out its own acronym.
    got = tw.hits("a strong edge harmonic oscillation", lex)
    assert [h.alias for h in got["eho"]] == ["edge harmonic oscillation"]


# ------------------------------------------------------------- weak labels

def test_weak_labels_columns_and_dtypes(tmp_path):
    lex = tw.load_lexicon()
    _write(tmp_path, 10, session="quiet run",
           shot_text="sawtooth crashes; no elms at all")
    _write(tmp_path, 11, session="quiet run", shot_text="nothing to say")
    df = tw.weak_labels([10, 11], lex, root=tmp_path)
    assert list(df.columns) == list(tw.COLUMNS)
    assert {c: str(t) for c, t in df.dtypes.items()} == tw.DTYPES
    assert list(df["shot"]) == [10, 10]
    assert list(df["phenomenon"]) == ["elm", "sawtooth"]
    assert list(df["n_pos"]) == [0, 1]
    assert list(df["n_neg"]) == [1, 0]
    assert set(df["scope"]) == {"shot"}
    assert df.loc[df["phenomenon"] == "sawtooth", "snippet"].iloc[0] \
        == "sawtooth crashes"
    assert df.loc[df["phenomenon"] == "elm", "snippet"].iloc[0] == ""


def test_weak_labels_of_nothing_is_an_empty_frame_with_the_columns(tmp_path):
    lex = tw.load_lexicon()
    df = tw.weak_labels([], lex, root=tmp_path)
    assert df.empty
    assert list(df.columns) == list(tw.COLUMNS)
    assert {c: str(t) for c, t in df.dtypes.items()} == tw.DTYPES


def test_run_scope_reads_the_session_text_and_says_so(tmp_path):
    lex = tw.load_lexicon()
    _write(tmp_path, 12, session="lots of sawteeth today", shot_text="")
    shot = tw.weak_labels([12], lex, root=tmp_path, scope="shot")
    assert shot.empty
    run = tw.weak_labels([12], lex, root=tmp_path, scope="run")
    assert list(run["phenomenon"]) == ["sawtooth"]
    assert list(run["scope"]) == ["run"]
    assert list(run["n_pos"]) == [1]


def test_an_unknown_scope_is_refused(tmp_path):
    lex = tw.load_lexicon()
    with pytest.raises(ValueError, match="scope"):
        tw.weak_labels([1], lex, root=tmp_path, scope="session")


def test_a_long_sentence_is_clipped_to_the_snippet_length(tmp_path):
    lex = tw.load_lexicon()
    long = "sawtooth " + "and again " * 40
    _write(tmp_path, 13, shot_text=long)
    df = tw.weak_labels([13], lex, root=tmp_path)
    assert len(df["snippet"].iloc[0]) == tw.SNIPPET_CHARS


# ------------------------------------------------------------------ events

def test_text_events_are_weak_evidence_over_the_whole_shot(tmp_path):
    lex = tw.load_lexicon()
    _write(tmp_path, 198658, session="ignore me",
           shot_text="sawtooth crashes throughout")
    got = tw.text_events(198658, lex, root=tmp_path)
    assert len(got) == 1
    one = got[0]
    assert (one.source, one.evidence_kind) == ("text", "text")
    assert one.phenomenon == "sawtooth"
    assert one.shot == 198658
    # The whole-shot span, from the table row's PULSE-LENGTH.
    assert (one.t0_s, one.t1_s) == (0.0, 6.12)
    assert (one.t_cov0_s, one.t_cov1_s) == (0.0, 6.12)
    assert one.confidence == pytest.approx(0.25)
    assert math.isnan(one.f0_khz) and math.isnan(one.f1_khz)
    assert one.attrs["scope"] == "shot"
    assert one.attrs["n_pos"] == 1 and one.attrs["n_neg"] == 0
    assert one.attrs["snippet"] == "sawtooth crashes throughout"
    assert json.loads(json.dumps(one.attrs, allow_nan=False)) == dict(one.attrs)


def test_the_confidence_is_a_quarter_a_mention_up_to_one(tmp_path):
    lex = tw.load_lexicon()
    _write(tmp_path, 20, shot_text="sawtooth. sawteeth. sawtooth crash")
    got = tw.text_events(20, lex, root=tmp_path)
    assert got[0].attrs["n_pos"] == 3
    assert got[0].confidence == pytest.approx(0.75)
    _write(tmp_path, 21, shot_text=". ".join(["sawtooth"] * 9))
    assert tw.text_events(21, lex, root=tmp_path)[0].confidence == 1.0
    # A single mention is the TEXT_ONLY ceiling and nothing more: text is
    # never a label by itself.
    assert tw.TEXT_ONLY_CEILING == 0.25


def test_a_negated_mention_is_no_event(tmp_path):
    lex = tw.load_lexicon()
    _write(tmp_path, 22, shot_text="no sawteeth in this one")
    assert tw.text_events(22, lex, root=tmp_path) == []
    df = tw.weak_labels([22], lex, root=tmp_path)
    assert list(df["n_neg"]) == [1]


def test_without_a_pulse_length_the_event_is_a_point_at_zero(tmp_path):
    lex = tw.load_lexicon()
    _write(tmp_path, 23, shot_text="sawtooth crash", table={"SHOT_TYPE": "plasma"})
    one = tw.text_events(23, lex, root=tmp_path)[0]
    assert (one.t0_s, one.t1_s) == (0.0, 0.0)
    _write(tmp_path, 24, shot_text="sawtooth crash",
           table={"PULSE-LENGTH": "(none)"})
    assert tw.text_events(24, lex, root=tmp_path)[0].t1_s == 0.0


def test_the_run_context_is_never_a_shot_event(tmp_path):
    # Appendix C item 12: the session text is shared by every shot of the
    # run, so a sawtooth mentioned there is run-scope evidence and claiming
    # it for this shot would claim it for twenty others too.
    lex = tw.load_lexicon()
    _write(tmp_path, 25, session="sawtooth crashes all afternoon", shot_text="")
    assert tw.text_events(25, lex, root=tmp_path) == []


def test_text_events_of_a_missing_shot_are_no_events(tmp_path):
    lex = tw.load_lexicon()
    assert tw.text_events(999999, tw.load_lexicon(), root=tmp_path) == []
    assert tw.weak_labels([999999], lex, root=tmp_path).empty
