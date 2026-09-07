"""The phenomenon lexicon and its matcher, on text and nothing else.

Split out of `test_events_text_weak.py` with the module it tests: this half
is what ideate reads and what a phrase has to survive to become evidence,
and it never touches a corpus. Every lexicon a test needs is written into
`tmp_path`, except the shipped one, which is the point of the first few.
"""
from __future__ import annotations

import pytest

from labelmaker.events import lexicon as lx


@pytest.fixture
def lex():
    return lx.load_lexicon()


def _lexicon(tmp_path, body: str, name: str = "lex.yaml"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------- the shipped file

def test_the_shipped_lexicon_carries_the_twelve_round_one_ids(lex):
    # Plan 5.6's round-1 ids. This file is the SINGLE source of them: ideate
    # reads it too, so an id renamed here is renamed there.
    assert lex.version == 1
    assert set(lex.ids) == set(lx.PHENOMENON_IDS)
    assert len(lex.ids) == 12
    for p in lex.phenomena:
        assert p.title and p.aliases and p.weight > 0.0
        assert all(a == a.lower() and a.strip() for a in p.aliases)
        assert all(n == n.lower() and n.strip() for n in p.negatives)
    assert "edge harmonic oscillation" in lex["eho"].aliases
    assert "sawteeth" in lex["sawtooth"].aliases
    assert "elm-free" in lex["elm"].negatives


def test_the_shipped_lexicon_is_read_as_utf_8_whatever_the_locale(lex):
    # `alfvén` is the first non-ASCII byte the package reads, and a bare
    # `read_text()` would take it from the process locale.
    assert "alfvén" in lex["ae"].aliases
    assert lx.DEFAULT_LEXICON.name == "lexicons.yaml"


def test_every_negative_in_the_shipped_lexicon_can_actually_fire(lex):
    # A negative only marks a sentence when an ALIAS matches it too, so a
    # negative no alias can co-occur with is dead text. Checked here on the
    # shipped file and refused by `load_lexicon` in general.
    for p in lex.phenomena:
        for negative in p.negatives:
            found = lx.hits(negative, lex)
            assert p.id in found, f"{p.id}: {negative!r} fires nothing"
            assert [h.polarity for h in found[p.id]] == ["neg"]


# ----------------------------------------------------------- what it refuses

def test_an_uppercase_alias_is_a_lexicon_error(tmp_path):
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  elm:\n    title: ELM\n    aliases: [ELM]\n",
    )
    with pytest.raises(lx.LexiconError, match="lowercase"):
        lx.load_lexicon(path)


def test_an_unknown_phenomenon_id_is_a_lexicon_error(tmp_path):
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  wobble:\n    title: W\n    aliases: [wobble]\n",
    )
    with pytest.raises(lx.LexiconError, match="wobble"):
        lx.load_lexicon(path)


def test_a_phenomenon_with_no_aliases_is_a_lexicon_error(tmp_path):
    path = _lexicon(
        tmp_path, "version: 1\nphenomena:\n  elm:\n    title: ELM\n    aliases: []\n"
    )
    with pytest.raises(lx.LexiconError, match="aliases"):
        lx.load_lexicon(path)


def test_a_lexicon_that_is_not_a_mapping_is_a_lexicon_error(tmp_path):
    path = _lexicon(tmp_path, "- elm\n- sawtooth\n")
    with pytest.raises(lx.LexiconError, match="mapping"):
        lx.load_lexicon(path)


@pytest.mark.parametrize("version", ["", "version: 0\n", "version: two\n",
                                     "version: true\n"])
def test_a_missing_or_unusable_version_is_a_lexicon_error(tmp_path, version):
    path = _lexicon(
        tmp_path,
        version + "phenomena:\n  elm:\n    title: ELM\n    aliases: [elm]\n",
    )
    with pytest.raises(lx.LexiconError, match="version"):
        lx.load_lexicon(path)


@pytest.mark.parametrize("title", ["", "    title: ''\n", "    title: 42\n"])
def test_a_missing_or_empty_title_is_a_lexicon_error(tmp_path, title):
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  elm:\n" + title + "    aliases: [elm]\n",
    )
    with pytest.raises(lx.LexiconError, match="title"):
        lx.load_lexicon(path)


def test_a_repeated_phrase_is_a_lexicon_error(tmp_path):
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  elm:\n    title: ELM\n"
        "    aliases: [elm, elms, elm]\n",
    )
    with pytest.raises(lx.LexiconError, match="repeats"):
        lx.load_lexicon(path)


def test_a_phrase_that_is_all_punctuation_is_a_lexicon_error(tmp_path):
    # It would glue to the empty string, and `"" in anything` is True: one
    # such phrase makes its phenomenon hit every sentence in the corpus.
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  elm:\n    title: ELM\n    aliases: ['elm', '...']\n",
    )
    with pytest.raises(lx.LexiconError, match="punctuation"):
        lx.load_lexicon(path)


def test_a_phrase_list_that_is_not_a_list_is_a_lexicon_error(tmp_path):
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  elm:\n    title: ELM\n    aliases: elm\n",
    )
    with pytest.raises(lx.LexiconError, match="list of phrases"):
        lx.load_lexicon(path)


def test_a_phenomenon_body_that_is_not_a_mapping_is_a_lexicon_error(tmp_path):
    path = _lexicon(tmp_path, "version: 1\nphenomena:\n  elm: [elm, elms]\n")
    with pytest.raises(lx.LexiconError, match="mapping"):
        lx.load_lexicon(path)


def test_a_negative_no_alias_can_fire_with_is_a_lexicon_error(tmp_path):
    # `sawtooth-free` is ONE token, so " sawtooth " is not inside it: the
    # negative can never mark a sentence unless the hyphenated form is an
    # alias too. That is exactly how `elm-free` is spelled in the shipped
    # file, and the other three were not - so the check is here rather than
    # in a reviewer's head.
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  sawtooth:\n    title: Sawtooth\n"
        "    aliases: [sawtooth]\n    negatives: ['sawtooth-free']\n",
    )
    with pytest.raises(lx.LexiconError, match="never fire"):
        lx.load_lexicon(path)


def test_a_file_that_is_not_there_is_a_lexicon_error(tmp_path):
    with pytest.raises(lx.LexiconError, match="not readable"):
        lx.load_lexicon(tmp_path / "nope.yaml")
    with pytest.raises(lx.LexiconError, match="not readable"):
        lx.load_lexicon(tmp_path)          # a directory is not a lexicon


def test_bytes_that_are_not_utf_8_are_a_lexicon_error(tmp_path):
    # `alfvén` in latin-1. Read as UTF-8, which is what the file is, this
    # raises `UnicodeDecodeError` - and it has to leave `load_lexicon` as a
    # `LexiconError` like every other unusable file.
    path = tmp_path / "lex.yaml"
    path.write_bytes(
        b"version: 1\nphenomena:\n  ae:\n    title: AE\n    aliases: ['alfv\xe9n']\n"
    )
    with pytest.raises(lx.LexiconError, match="not readable"):
        lx.load_lexicon(path)


def test_a_weight_defaults_to_one_and_is_read_when_given(tmp_path):
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n"
        "  elm:\n    title: ELM\n    aliases: [elm]\n"
        "  rwm:\n    title: RWM\n    aliases: [rwm]\n    weight: 0.4\n",
    )
    lex = lx.load_lexicon(path)
    assert lex["elm"].weight == 1.0
    assert lex["rwm"].weight == 0.4
    with pytest.raises(KeyError):
        lex["sawtooth"]


@pytest.mark.parametrize("weight", ["0", "-1", "null", ".nan", "true"])
def test_a_weight_that_is_not_positive_is_a_lexicon_error(tmp_path, weight):
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  elm:\n    title: ELM\n    aliases: [elm]\n"
        f"    weight: {weight}\n",
    )
    with pytest.raises(lx.LexiconError, match="weight"):
        lx.load_lexicon(path)


# ---------------------------------------------------------------- matching

def test_sentences_split_on_terminators_and_newlines_and_lowercase():
    got = lx.sentences("Sawtooth crash!  Then\nELMs; and   more?")
    assert got == ["sawtooth crash", "then", "elms", "and more"]


def test_a_phrase_that_spans_a_wrapped_line_is_missed():
    # The cost of splitting on newlines, stated: a logbook line wraps
    # mid-phrase and the long spelling is gone. It degrades to the short
    # one where there is one, which is why `edge harmonic` is an alias.
    lex = lx.load_lexicon()
    assert "rwm" not in lx.hits("resistive wall\nmode seen", lex)
    assert "eho" in lx.hits("edge harmonic\noscillation seen", lex)


def test_a_word_is_matched_whole_and_not_inside_another(lex):
    assert "elm" in lx.hits("elms all through the shot", lex)
    assert "elm" not in lx.hits("the helms of the ship", lex)


def test_a_two_letter_alias_does_not_match_inside_a_word(tmp_path):
    # A7's own example: `" nt "` must not match `"want"`.
    path = _lexicon(
        tmp_path,
        "version: 1\nphenomena:\n  tearing:\n    title: T\n    aliases: [nt]\n",
    )
    lex = lx.load_lexicon(path)
    assert "tearing" not in lx.hits("we want more current", lex)
    assert "tearing" in lx.hits("nt mode at 3 s", lex)


def test_a_mode_number_is_one_token(lex):
    assert "tearing" in lx.hits("2/1 locked at 2.5 s", lex)
    assert "tearing" not in lx.hits("12/1 is not it", lex)


def test_a_hyphenated_word_is_one_token(lex):
    got = lx.hits("(elm-free) h-mode", lex)
    assert [h.polarity for h in got["elm"]] == ["neg"]


def test_a_name_inside_an_identifier_is_not_a_mention(lex):
    # The PCS dump is full of `RWM_GAINMULT`, which is a parameter name and
    # not somebody saying a resistive wall mode happened.
    assert "rwm" not in lx.hits("set RWM_GAINMULT to 1.0", lex)
    assert "rwm" in lx.hits("rwm feedback on", lex)


def test_a_negated_sentence_yields_only_negative_hits(lex):
    got = lx.hits("no sawteeth here, but a tearing mode", lex)
    assert [h.polarity for h in got["sawtooth"]] == ["neg"]
    assert [h.polarity for h in got["tearing"]] == ["pos"]


def test_two_phenomena_in_one_sentence_are_two_hits(lex):
    got = lx.hits("elms and fishbones together", lex)
    assert set(got) == {"elm", "fishbone"}


def test_overlapping_aliases_count_a_sentence_once(lex):
    got = lx.hits("a strong edge harmonic oscillation", lex)
    assert [h.alias for h in got["eho"]] == ["edge harmonic oscillation"]


def test_the_matchers_are_built_once_per_lexicon(lex):
    # `hits` is called once per shot over 22,950 bundles in the worst case,
    # and the glued alias lists do not change between calls. A frozen
    # `Lexicon` is hashable, so the cache can be keyed on the lexicon
    # itself rather than on an identity nobody controls.
    assert hash(lex) == hash(lx.load_lexicon())
    first = lx._matchers(lex)
    assert lx._matchers(lx.load_lexicon()) is first


def test_a_decimal_number_is_one_token(lex):
    # The L7-fix review's finding: `.` was a separator everywhere, so
    # "beams 1.3/2.4 MW" broke into "beams 1", "3/2", "4 mw" and the middle
    # piece IS the tearing alias. A beam power pair is not a mode number.
    assert "tearing" not in lx.hits("beams 1.3/2.4 MW", lex)
    assert "tearing" not in lx.hits("pinj 2.3/2.5 MW", lex)
    assert "tearing" not in lx.hits("q95=3.2 at 2.11 s", lex)
    # A `.` anywhere else still ends a sentence, and a real mode number is
    # still one token.
    assert lx.sentences("sawtooth. Then") == ["sawtooth", "then"]
    assert "sawtooth" in lx.hits("sawtooth. Then", lex)
    assert [h.polarity for h in lx.hits("3/2 mode locked", lex)["tearing"]] == [
        "pos"
    ]
