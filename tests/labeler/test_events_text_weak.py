"""Where the operators' text comes from, and what it is allowed to claim.

Two sources and they mean different things, which is most of what these
tests are about: a shot's OWN logbook entries come from `sql/logs.jsonl`
(shot scope), the session context and the shot's length come from the
per-shot bundle (run scope, and the span). Every fixture is written into
`tmp_path` in the real layout of each - a JSON record per line whose first
bytes are `{"shot": N,`, and a bundle with its marker line and its
`SHOT TABLE ROW` block - so a test asserts what was PUT in the file.
Nothing here reads `/scratch/gpfs/EKOLEMEN`; the real-data numbers live in
the task report.
"""
from __future__ import annotations

import builtins
import json
import math
import os

import pytest

from labeler.config import Paths
from labeler.events import lexicon as lx
from labeler.events import text_weak as tw

MARKER = "## Shot-specific context (from summary.html)"

TABLE = {
    "SHOT": "198658",
    "SHOT_TYPE": "plasma",
    "TIME-OF-SHOT": "10:08",
    "PULSE-LENGTH": "6.12",
    "IP-(MA)": "-1.00",
}


@pytest.fixture
def paths(tmp_path):
    """A `Paths` whose every text input and output is under `tmp_path`."""
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "bundles",
        logs_jsonl=tmp_path / "logs.jsonl",
    )


@pytest.fixture
def lex():
    return lx.load_lexicon()


def _entry(role, author, text, ts="2024-05-17 13:12:07"):
    return f"### [{role}] {author} {ts}\n{text}\n"


#: One shot's `log_text` as the corpus writes it: three entries, one of them
#: the PCS machine dump, an `<a href>` mini-proposal link and a `<b>`, and a
#: literal `<ne>=4e13` that must survive the tag stripping.
LOG_TEXT = (
    _entry("PHYSICS_OPERATOR", "smithj",
           "fishbones through the current ramp. <b>note</b> <ne>=4e13")
    + _entry("PCS", "pcsuser",
             "PCS CHANGES: set vertices RWM_GAINMULT 1.0; sawtooth pre-empt",
             ts="2024-05-17 13:12:09")
    + _entry("CHIEF_OPERATOR", "opsx",
             'see <a href="http://d3d/mp/1234">the mini-proposal</a> '
             "for the fishbones",
             ts="2024-05-17 13:13:00")
)


def _write_logs(paths, records):
    """A synthetic `logs.jsonl`: one record per line, shot key first."""
    paths.logs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for rec in records:
        ordered = {"shot": rec["shot"],
                   **{k: v for k, v in rec.items() if k != "shot"}}
        lines.append(json.dumps(ordered))
    paths.logs_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths.logs_jsonl


def _write_bundle(paths, shot, *, session="", shot_text="", table=TABLE,
                  marker=True):
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
    paths.text_root.mkdir(parents=True, exist_ok=True)
    path = paths.text_file(shot)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def corpus(paths):
    """Three logbook records and the bundles that go with two of them."""
    _write_logs(paths, [
        {"shot": 10, "run": "20240517", "log_text": LOG_TEXT,
         "run_log_text": "session leader: sawteeth all afternoon",
         "topics": ["PCS", "PHYSICS_OPERATOR"], "keywords": []},
        {"shot": 11, "run": "20240517",
         "log_text": _entry("SESSION_LEADER", "leadr", "no elms at all")},
        {"shot": 12, "run": "20240517",
         "log_text": _entry("DIAGNOSTICS", "diag",
                            "detachment late in the shot")},
    ])
    _write_bundle(paths, 10, session="sawtooth crashes all afternoon")
    _write_bundle(paths, 11, session="quiet run")
    return paths


# ------------------------------------------------------------- the subset

def test_the_subset_takes_only_the_lines_it_was_asked_for(corpus, paths):
    assert tw.build_logs_subset([10, 12], paths=paths) == 2
    lines = paths.logs_subset.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["shot"] for ln in lines] == [10, 12]
    # The rename is what makes the write atomic; nothing may be left over.
    assert list(paths.text_cache.iterdir()) == [paths.logs_subset]


def test_building_the_subset_again_adds_nothing(corpus, paths):
    assert tw.build_logs_subset([10, 12], paths=paths) == 2
    before = paths.logs_subset.read_bytes()
    assert tw.build_logs_subset([10, 12], paths=paths) == 0
    assert paths.logs_subset.read_bytes() == before
    # And a shot that is new is the only one fetched.
    assert tw.build_logs_subset([10, 11, 12], paths=paths) == 1
    assert {tw.load_log_record(s, paths=paths)["shot"]
            for s in (10, 11, 12)} == {10, 11, 12}


def test_a_shot_the_logbook_has_no_record_of_is_not_an_error(corpus, paths):
    assert tw.build_logs_subset([10, 999999], paths=paths) == 1
    assert tw.load_log_record(999999, paths=paths) is None
    assert tw.shot_prose(999999, paths=paths) == ""


def test_a_shot_the_logbook_has_no_record_of_is_remembered(corpus, paths,
                                                          monkeypatch):
    # The L7-fix review's Important #1. A shot with no record can never
    # enter the subset, so it stayed in `wanted` forever and every later
    # call re-streamed the 616 MB source - one full pass per record-less
    # shot per pass over the shot list, silently, and `text_events` builds
    # for one shot at a time. The misses are remembered in a sidecar.
    assert tw.build_logs_subset([10, 999], paths=paths) == 1
    assert paths.logs_subset_missing.read_text(encoding="utf-8").split() == [
        "999"
    ]
    before = paths.logs_subset.read_bytes()
    stat = paths.logs_subset.stat()
    opened: list[str] = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    assert tw.build_logs_subset([10, 999], paths=paths) == 0
    assert opened == []
    # And a no-op build leaves the cache alone, so the parsed-subset memo
    # is not invalidated either.
    assert paths.logs_subset.read_bytes() == before
    assert paths.logs_subset.stat().st_mtime_ns == stat.st_mtime_ns


def test_the_remembered_miss_is_refreshed_when_the_logbook_catches_up(corpus,
                                                                      paths):
    # A miss is a fact about the source AT THE TIME, not forever: the
    # logbook gains records for today's shots. `refresh_missing=True` is
    # the one call that pays a pass to find out.
    tw.build_logs_subset([10, 999], paths=paths)
    _write_logs(paths, [
        {"shot": 999, "log_text": _entry("PHYSICS_OPERATOR", "a", "big elms")},
    ])
    assert tw.build_logs_subset([999], paths=paths) == 0
    assert tw.load_log_record(999, paths=paths) is None
    assert tw.build_logs_subset([999], paths=paths, refresh_missing=True) == 1
    assert tw.load_log_record(999, paths=paths)["shot"] == 999
    assert paths.logs_subset_missing.read_text(encoding="utf-8").split() == []


def test_a_build_that_finds_nothing_does_not_rewrite_the_subset(corpus, paths):
    tw.build_logs_subset([10], paths=paths)
    before = paths.logs_subset.read_bytes()
    stat = paths.logs_subset.stat()
    assert tw.build_logs_subset([998], paths=paths) == 0
    assert paths.logs_subset.read_bytes() == before
    assert paths.logs_subset.stat().st_mtime_ns == stat.st_mtime_ns
    assert sorted(p.name for p in paths.text_cache.iterdir()) == [
        "logs_subset.jsonl", "logs_subset.missing"
    ]


def test_the_temporary_file_carries_the_pid(corpus, paths, monkeypatch):
    # Two SLURM array tasks sharing a `LABELER_ROOT` build this cache at
    # once. A fixed sibling `.tmp` lets them interleave their writes into
    # one file and rename the result into place.
    seen: list[str] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert tw.build_logs_subset([10, 999], paths=paths) == 1
    assert seen
    assert all(s.endswith(f".{os.getpid()}.tmp") for s in seen)


def test_a_bad_interior_subset_line_is_skipped_with_one_warning(paths):
    # The subset is a file on disk that a person can edit and that two
    # array tasks can race on. One damaged line must not make every later
    # read raise until somebody deletes the cache by hand - the same
    # policy the torn trailing line already had.
    paths.text_cache.mkdir(parents=True, exist_ok=True)
    paths.logs_subset.write_text(
        '{"shot": 10, "log_text": "fishbones everywhere"}\n'
        "{ this line is not json at all\n"
        '{"shot": 12, "log_text": "detachment late in the shot"}\n',
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="unusable") as record:
        assert tw.load_log_record(10, paths=paths)["shot"] == 10
    assert len(record) == 1
    assert tw.load_log_record(12, paths=paths)["shot"] == 12


def test_a_subset_record_with_no_shot_key_is_skipped_and_not_a_key_error(
    paths,
):
    paths.text_cache.mkdir(parents=True, exist_ok=True)
    paths.logs_subset.write_text(
        '{"log_text": "somebody hand-edited this one"}\n'
        '{"shot": 12, "log_text": "detachment late in the shot"}\n',
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="unusable"):
        assert tw.load_log_record(12, paths=paths)["shot"] == 12


def test_a_torn_trailing_line_is_tolerated_and_the_shot_refetched(corpus,
                                                                  paths):
    tw.build_logs_subset([10], paths=paths)
    # What a build killed mid-write leaves: a last line with no newline and
    # no closing brace. Reading it must not break every later read.
    with paths.logs_subset.open("a", encoding="utf-8") as fh:
        fh.write('{"shot": 12, "log_text": "detach')
    with pytest.warns(UserWarning, match="trailing line"):
        assert tw.load_log_record(10, paths=paths)["shot"] == 10
    # The torn line is not carried over, so shot 12 is fetched again. No
    # second warning: the parse is memoised on the file's own size and
    # mtime, so one damaged file is one complaint and not one per shot.
    assert tw.build_logs_subset([12], paths=paths) == 1
    assert tw.load_log_record(12, paths=paths)["shot"] == 12


def test_a_record_is_read_from_the_subset_and_never_from_the_source(corpus,
                                                                    paths):
    # The source is 616 MB. A lookup that could fall back to scanning it
    # would do so once per shot, so the 500-shot loop that looks fine in a
    # test would take an hour on the real corpus.
    tw.build_logs_subset([10], paths=paths)
    paths.logs_jsonl.unlink()
    assert tw.load_log_record(10, paths=paths)["shot"] == 10
    assert tw.load_log_record(12, paths=paths) is None


def test_the_record_a_caller_gets_is_its_own_copy(corpus, paths):
    tw.build_logs_subset([10], paths=paths)
    got = tw.load_log_record(10, paths=paths)
    got["topics"].append("MINE")
    assert tw.load_log_record(10, paths=paths)["topics"] == [
        "PCS", "PHYSICS_OPERATOR"
    ]


def test_a_logbook_that_is_not_there_is_an_error_and_not_silence(paths):
    # A missing source is a misconfigured `Paths`, not a shot with no text,
    # and the failure it would otherwise make - every shot silently
    # textless - is the one this module has been burned by once.
    with pytest.raises(OSError):
        tw.build_logs_subset([10], paths=paths)


# -------------------------------------------------------------- the entries

def test_log_text_splits_on_the_entry_headers():
    got = tw.log_entries(LOG_TEXT)
    assert [e.role for e in got] == [
        "PHYSICS_OPERATOR", "PCS", "CHIEF_OPERATOR"
    ]
    assert [e.author for e in got] == ["smithj", "pcsuser", "opsx"]
    assert got[0].time == "2024-05-17 13:12:07"
    assert got[2].time == "2024-05-17 13:13:00"
    assert got[0].text.startswith("fishbones through the current ramp")
    assert "###" not in got[0].text
    assert tw.log_entries("") == []
    assert tw.log_entries("no header here") == []


def test_the_five_html_tags_go_and_a_measurement_stays():
    got = tw.log_entries(LOG_TEXT)
    assert got[0].text == "fishbones through the current ramp. note <ne>=4e13"
    assert got[2].text == "see the mini-proposal for the fishbones"


def test_the_machine_dump_is_not_something_somebody_wrote(corpus, paths):
    tw.build_logs_subset([10], paths=paths)
    prose = tw.shot_prose(10, paths=paths)
    assert "fishbones" in prose
    assert "RWM_GAINMULT" not in prose        # the PCS entry, left out
    assert "sawtooth pre-empt" not in prose
    assert [e.role for e in tw.shot_entries(10, paths=paths)] == [
        "PHYSICS_OPERATOR", "CHIEF_OPERATOR"
    ]
    # Asked for it, it is there: the exclusion is a default and not a law.
    assert "RWM_GAINMULT" in tw.shot_prose(10, paths=paths, exclude_roles=())


def test_the_prose_joins_entries_so_no_phrase_spans_two_of_them(paths, lex):
    # One person's last word and the next person's first are not a phrase
    # anybody wrote, and `sentences` breaks on the newline that joins them.
    _write_logs(paths, [{
        "shot": 30,
        "log_text": (_entry("PHYSICS_OPERATOR", "a", "watching the edge")
                     + _entry("DIAGNOSTICS", "b", "harmonic oscillation gone")),
    }])
    tw.build_logs_subset([30], paths=paths)
    assert "eho" not in lx.hits(tw.shot_prose(30, paths=paths), lex)


# ------------------------------------------------------------ the bundle

def test_the_marker_splits_the_session_text_from_the_shots_own(paths):
    _write_bundle(paths, 1, session="ELMs all run", shot_text="quiet shot")
    assert tw.run_context(1, paths=paths).endswith("ELMs all run")
    assert tw.shot_block(1, paths=paths).startswith("SHOT: 1")
    assert "quiet shot" in tw.shot_block(1, paths=paths)


def test_a_bundle_with_nothing_after_the_marker_has_an_empty_block(paths):
    _write_bundle(paths, 2, session="ELMs all run", table=None)
    assert tw.shot_block(2, paths=paths) == ""
    assert tw.shot_span_s(2, paths=paths) == (0.0, 0.0)


def test_a_bundle_with_no_marker_at_all_is_all_run_context(paths):
    _write_bundle(paths, 3, session="ELMs all run", marker=False)
    assert "ELMs all run" in tw.run_context(3, paths=paths)
    assert tw.shot_block(3, paths=paths) == ""


def test_a_missing_bundle_reads_as_empty_and_not_as_an_error(paths):
    assert tw.run_context(999999, paths=paths) == ""
    assert tw.shot_block(999999, paths=paths) == ""
    assert tw.shot_span_s(999999, paths=paths) == (0.0, 0.0)


def test_the_shot_table_row_reads_as_a_mapping(paths):
    _write_bundle(paths, 4, shot_text="whatever")
    got = tw.shot_table_row(tw.shot_block(4, paths=paths))
    assert got == TABLE
    # Split on the FIRST colon, so a time of day is one value.
    assert got["TIME-OF-SHOT"] == "10:08"
    assert tw.shot_table_row("nothing here") == {}


def test_the_span_is_the_pulse_length_and_zero_when_it_is_not_a_number(paths):
    _write_bundle(paths, 5, table={"PULSE-LENGTH": "6.12"})
    assert tw.shot_span_s(5, paths=paths) == (0.0, 6.12)
    for bad in ("(none)", "", "-1", "nan"):
        _write_bundle(paths, 6, table={"PULSE-LENGTH": bad})
        assert tw.shot_span_s(6, paths=paths) == (0.0, 0.0)


# ------------------------------------------------------------- weak labels

def test_the_shot_scope_text_is_the_log_and_not_the_bundle(corpus, paths, lex):
    # The fixture says "sawtooth" ONLY in the session text and "fishbones"
    # only in the logbook, so which source each scope read is not a matter
    # of opinion.
    shot = tw.weak_labels([10], lex, paths=paths, scope="shot")
    assert list(shot["phenomenon"]) == ["fishbone"]
    assert list(shot["n_pos"]) == [2]
    assert list(shot["scope"]) == ["shot"]
    run = tw.weak_labels([10], lex, paths=paths, scope="run")
    assert list(run["phenomenon"]) == ["sawtooth"]
    assert list(run["scope"]) == ["run"]


def test_run_scope_never_reads_the_logbook_at_all(paths, lex):
    # `scope="run"` reads the bundle and nothing else, so it must not pay a
    # pass over the 616 MB source - nor fail on a machine that cannot see
    # it. The subset is built for the shot scope, which is the scope that
    # uses it.
    _write_bundle(paths, 40, session="sawtooth crashes all afternoon")
    assert not paths.logs_jsonl.exists()
    df = tw.weak_labels([40], lex, paths=paths, scope="run")
    assert list(df["phenomenon"]) == ["sawtooth"]
    assert not paths.logs_subset.exists()


def test_weak_labels_columns_and_dtypes(corpus, paths, lex):
    df = tw.weak_labels([10, 11, 12], lex, paths=paths)
    assert list(df.columns) == list(tw.COLUMNS)
    assert {c: str(t) for c, t in df.dtypes.items()} == tw.DTYPES
    assert list(df["shot"]) == [10, 11, 12]
    assert list(df["phenomenon"]) == ["fishbone", "elm", "detachment"]
    assert list(df["n_pos"]) == [2, 0, 1]
    assert list(df["n_neg"]) == [0, 1, 0]
    assert df["snippet"].iloc[0] == "fishbones through the current ramp"
    assert df["snippet"].iloc[1] == ""          # a negated mention, no snippet


def test_weak_labels_of_nothing_is_an_empty_frame_with_the_columns(corpus,
                                                                   paths, lex):
    df = tw.weak_labels([], lex, paths=paths)
    assert df.empty
    assert list(df.columns) == list(tw.COLUMNS)
    assert {c: str(t) for c, t in df.dtypes.items()} == tw.DTYPES


def test_an_unknown_scope_is_refused(corpus, paths, lex):
    with pytest.raises(ValueError, match="scope"):
        tw.weak_labels([10], lex, paths=paths, scope="session")


def test_a_long_sentence_is_clipped_to_the_snippet_length(paths, lex):
    _write_logs(paths, [{
        "shot": 13,
        "log_text": _entry("PHYSICS_OPERATOR", "a",
                           "sawtooth " + "and again " * 40),
    }])
    df = tw.weak_labels([13], lex, paths=paths)
    assert len(df["snippet"].iloc[0]) == tw.SNIPPET_CHARS


def test_weak_labels_reads_text_only_through_the_accessors(corpus, paths, lex,
                                                           monkeypatch):
    # The seam the L7 review asked for: swapping a text source is one
    # function, and a consumer that inlined the read would not have moved.
    monkeypatch.setattr(tw, "shot_prose",
                        lambda shot, **kw: "a clear qh-mode phase")
    monkeypatch.setattr(tw, "run_context",
                        lambda shot, **kw: "locked mode at 2.1 s")
    assert list(tw.weak_labels([10], lex, paths=paths)["phenomenon"]) == ["qh"]
    assert list(
        tw.weak_labels([10], lex, paths=paths, scope="run")["phenomenon"]
    ) == ["tearing"]


# ------------------------------------------------------------------ events

def test_a_text_event_takes_its_text_from_the_log_and_its_span_from_the_row(
    corpus, paths, lex,
):
    # The two sources, and the reason they are two accessors: `logs.jsonl`
    # has no `SHOT TABLE ROW`, so a span read from the same place as the
    # text would be `[0, 0]` on every row and nothing would say so.
    got = tw.text_events(10, lex, paths=paths)
    assert len(got) == 1
    one = got[0]
    assert (one.source, one.evidence_kind) == ("text", "text")
    assert one.phenomenon == "fishbone"         # the log
    assert (one.t0_s, one.t1_s) == (0.0, 6.12)  # the bundle's table row
    assert (one.t_cov0_s, one.t_cov1_s) == (0.0, 6.12)
    assert one.shot == 10
    assert one.confidence == pytest.approx(0.5)
    assert math.isnan(one.f0_khz) and math.isnan(one.f1_khz)


def test_a_text_event_says_where_it_came_from(corpus, paths, lex):
    one = tw.text_events(10, lex, paths=paths)[0]
    assert one.attrs["scope"] == "shot"
    assert one.attrs["text_source"] == "logs.jsonl"
    assert one.attrs["n_pos"] == 2 and one.attrs["n_neg"] == 0
    assert one.attrs["n_entries"] == 2          # the PCS dump is not one
    assert one.attrs["roles"] == ["CHIEF_OPERATOR", "PHYSICS_OPERATOR"]
    assert one.attrs["snippet"] == "fishbones through the current ramp"
    assert json.loads(json.dumps(one.attrs, allow_nan=False)) == dict(one.attrs)


def test_the_confidence_is_a_quarter_a_mention_up_to_one(paths, lex):
    _write_logs(paths, [
        {"shot": 20, "log_text": _entry("PHYSICS_OPERATOR", "a",
                                        "sawtooth. sawteeth. sawtooth crash")},
        {"shot": 21, "log_text": _entry("PHYSICS_OPERATOR", "a",
                                        ". ".join(["sawtooth"] * 9))},
    ])
    got = tw.text_events(20, lex, paths=paths)
    assert got[0].attrs["n_pos"] == 3
    assert got[0].confidence == pytest.approx(0.75)
    assert tw.text_events(21, lex, paths=paths)[0].confidence == 1.0
    # A single mention is the TEXT_ONLY ceiling and nothing more: text is
    # never a label by itself.
    assert lx.TEXT_ONLY_CEILING == 0.25


def test_a_negated_mention_is_no_event(corpus, paths, lex):
    assert tw.text_events(11, lex, paths=paths) == []
    df = tw.weak_labels([11], lex, paths=paths)
    assert list(df["n_neg"]) == [1] and list(df["n_pos"]) == [0]


def test_without_a_pulse_length_the_event_is_a_point_at_zero(corpus, paths,
                                                             lex):
    # Shot 12 has a logbook record and no bundle at all.
    one = tw.text_events(12, lex, paths=paths)[0]
    assert one.phenomenon == "detachment"
    assert (one.t0_s, one.t1_s) == (0.0, 0.0)


def test_the_run_context_is_never_a_shot_event(corpus, paths, lex):
    # Appendix C item 12: the session text is shared by every shot of the
    # run, so a sawtooth mentioned there is run-scope evidence and claiming
    # it for this shot would claim it for twenty others too. Shot 10's
    # bundle says "sawtooth crashes all afternoon" and its events do not.
    assert [e.phenomenon for e in tw.text_events(10, lex, paths=paths)] == [
        "fishbone"
    ]


def test_a_shot_with_no_text_at_all_is_no_events(corpus, paths, lex):
    assert tw.text_events(999999, lex, paths=paths) == []
    assert tw.weak_labels([999999], lex, paths=paths).empty


def test_text_events_reads_text_and_span_only_through_the_accessors(
    corpus, paths, lex, monkeypatch,
):
    # ONE shot-scope seam, `shot_entries`, and swapping it alone moves the
    # hits, the counts, the snippet and the roles together. `shot_prose` is
    # that accessor flattened, not a second source, so a swap that changed
    # only it would leave `roles` and `n_entries` behind.
    monkeypatch.setattr(tw, "shot_entries", lambda shot, **kw: (
        tw.LogEntry(role="ANALYSIS", author="a",
                    time="2024-05-17 13:12:07", text="big elms"),
    ))
    monkeypatch.setattr(tw, "shot_span_s", lambda shot, **kw: (0.0, 4.5))
    one = tw.text_events(10, lex, paths=paths)[0]
    assert one.phenomenon == "elm"
    assert (one.t0_s, one.t1_s) == (0.0, 4.5)
    assert one.attrs["roles"] == ["ANALYSIS"]
    assert one.attrs["n_entries"] == 1
    assert one.attrs["snippet"] == "big elms"


def test_the_entries_are_matched_once_and_not_twice(corpus, paths, lex,
                                                    monkeypatch):
    # The counts, the snippet and the roles come from ONE pass over the
    # entries. Matching per entry for the roles and again over the joined
    # prose for the counts agrees only by construction, and is two places
    # that have to go on agreeing.
    real = lx.hits
    calls: list[str] = []

    def counting(text, lexicon):
        calls.append(text)
        return real(text, lexicon)

    monkeypatch.setattr(tw, "hits", counting)
    got = tw.text_events(10, lex, paths=paths)
    assert [e.phenomenon for e in got] == ["fishbone"]
    assert len(calls) == len(tw.shot_entries(10, paths=paths)) == 2


def test_a_shot_that_reached_the_subset_is_pruned_from_the_missing_sidecar(
    corpus, paths,
):
    # L7-fix-2 review, Minor #3. `now_missing` was pruned against this
    # call's `found` alone, so a shot that reached the subset by another
    # route - a refresh in another process, a hand-restored cache - stayed
    # listed as missing forever. A shot in the subset can never be a miss.
    assert tw.build_logs_subset([10, 999], paths=paths) == 1
    paths.logs_subset_missing.write_text("10\n999\n", encoding="utf-8")
    assert tw.build_logs_subset([11], paths=paths) == 1
    assert paths.logs_subset_missing.read_text(encoding="utf-8").split() == [
        "999"
    ]


def test_a_torn_trailing_line_is_repaired_even_when_nothing_is_found(
    corpus, paths,
):
    # L7-fix-2 review, Minor #4. The repair was conditional on `n >= 1`, so
    # a build whose shots the logbook has no record of left the torn line in
    # place - and its warning was re-emitted on every new parse of that
    # version of the file. One rewrite, then clean, whatever was found.
    tw.build_logs_subset([10], paths=paths)
    with paths.logs_subset.open("a", encoding="utf-8") as fh:
        fh.write('{"shot": 12, "log_text": "detach')
    with pytest.warns(UserWarning, match="trailing line"):
        assert tw.load_log_record(10, paths=paths)["shot"] == 10
    # 999 has no record: nothing is found, and the repair may not wait for
    # a successful fetch.
    assert tw.build_logs_subset([999], paths=paths) == 0
    text = paths.logs_subset.read_text(encoding="utf-8")
    assert "detach" not in text
    assert text.endswith("\n")
    # And the reparsed file is clean: no warning, so this line does not
    # raise under `-W error`.
    assert tw.load_log_record(10, paths=paths)["shot"] == 10


def test_weak_labels_reads_the_shot_text_only_through_the_accessor(
    corpus, paths, lex, monkeypatch,
):
    # L7-fix-2 review, Minor #6: the `shot_entries` seam was pinned for
    # `text_events` and asserted for `weak_labels` only by construction.
    # `shot_prose` is that accessor flattened, so swapping the accessor
    # moves the row.
    monkeypatch.setattr(tw, "shot_entries", lambda shot, **kw: (
        tw.LogEntry(role="ANALYSIS", author="a",
                    time="2024-05-17 13:12:07", text="big elms"),
    ))
    df = tw.weak_labels([10], lex, paths=paths)
    assert list(df["phenomenon"]) == ["elm"]
    assert list(df["n_pos"]) == [1]
