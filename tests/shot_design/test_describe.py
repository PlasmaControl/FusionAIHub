"""The description template, the fact check, and the LLM hook that must not reach the network.

The quote tests are the ones that matter most. This repo has already shipped two fabricated
quotes and one mislabelled shot, all from reading the concatenated `log_text` instead of the
parsed entries, so "a quote is a verbatim prefix of exactly one entry" is asserted directly --
inline, and again over all 105 real records.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from shot_design import schema
from shot_design.retrieval import describe as D
from shot_design.retrieval import rank

DB_DIR = Path("/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/db")
WS = re.compile(r"\s+")

# Verbatim from the real logbook entry by hyatt at 16:43 on shot 161172, checked against
# HumanTier.log_entries in the built database. Nothing here is invented or joined.
REAL_ENTRY_161172 = (
    "Remove the triangularity scan, and put in a SqBotOut of -0.1 from 2 seconds to\n"
    "5 seconds.\n\nResult: Well, not so good. We get a LM at 3 seconds."
)


def record(**over) -> schema.ShotRecord:
    base = dict(  # noqa: C408 — retain the source fixture layout
        shot=161172,
        shot_date=date(2015, 1, 13),
        campaign="2014_2015",
        segments=[
            schema.Segment(
                name="flat_top",
                t0_ms=529.5,
                t1_ms=5199.0,
                raw={
                    "ip_mean": 985281.3,
                    "bt_mean": 1.94962,
                    "ne_line_mean": 9.8174e13,
                    "pnbi_total_mean": 8207485.5,
                    "pnbi_15L_mean": 1.6e6,
                    "pnbi_30L_mean": 1.4e6,
                    "pnbi_21L_mean": 0.0,
                    "pech_total_on_frac": 0.0,
                },
                derived={"betan_mean": 2.2799, "kappa_mean": 1.8106, "q95_mean": 3.61},
            ),
            schema.Segment(name="full", t0_ms=14.75, t1_ms=5911.75),
        ],
        derived_provenance={
            "betan": schema.Provenance(tool="EFIT", version="01", tree="efit01", assumed=True)
        },
        human=schema.HumanTier(
            mpid="2014-21-19",
            mp_title="Optimizing pedestal height in hybrid regimes",
            run_id="20150113A",
        ),
        labels=schema.Labels(regime="H", regime_source="text", operational={"disruption_free"}),
        outcome=schema.Outcome(
            ip_target_hit=True, ip_target_err=-0.009, end_reason="programmed_rampdown"
        ),
        built_at=datetime(2026, 9, 4, 20, 11, tzinfo=UTC),
        builder_sha="test",
    )
    base.update(over)
    return schema.ShotRecord(**base)


def line(text: str, prefix: str) -> str | None:
    return next((line for line in text.splitlines() if line.startswith(prefix)), None)


# ---------------------------------------------------------------------------------- template


def test_the_header_carries_date_run_and_mini_proposal():
    assert D.describe(record()).splitlines()[0] == (
        "Shot 161172 (2015-01-13, run 20150113A, MP 2014-21-19 "
        '"Optimizing pedestal height in hybrid regimes").'
    )


def test_the_segment_line_prints_every_number_through_rank_display():
    """One renderer. `shot_design show` prints this same line as its headline, and the units and k/M
    prefixes are `rank.display`'s -- read from configs/shot_design/, chosen from the magnitude -- not a
    hardcoded `/1e6 -> MA` table. (Two such tables, with different precisions, used to make
    `show` and `query` print the same shot differently: `q95 3.38` vs `q95 3.4`.)"""
    got = line(D.describe(record()), "Flat top")
    assert got == (
        "Flat top 0.53-5.20 s: Ip 985 kA, Bt 1.95 T, PNBI 8.21 MW (15L 30L), PECH off, "
        "q95 3.61, betaN 2.28 (EFIT01, assumed), kappa 1.81, ne_line 9.82e+13 m/cm3."
    )
    assert f"Ip {rank.display('ip_mean', 985281.3)}" in got
    assert f"PNBI {rank.display('pnbi_total_mean', 8207485.5)}" in got
    assert D.segment_line(record(), "flat_top") == got
    assert D.segment_line(record(), "ramp_up") is None  # no such segment: no line, no swap


def test_an_idle_system_says_off_and_an_unrecorded_one_says_nothing():
    """features.py averages an actuator over its positive samples, so `pech_total_mean` is None
    both when the gyrotrons idled and when nothing was recorded. Only on_frac separates them."""
    seg = record().segments[0]
    assert "PECH off" in D.describe(record())
    seg.raw.pop("pech_total_on_frac")
    assert "PECH" not in D.describe(record(segments=[seg]))


def test_a_missing_number_is_an_omitted_clause_never_the_word_none():
    seg = schema.Segment(name="flat_top", t0_ms=0.0, t1_ms=1000.0, raw={"ip_mean": 1.0e6})
    text = D.describe(record(segments=[seg], derived_provenance={}, human=schema.HumanTier()))
    assert "Ip 1 MA" in text
    assert not re.search(r"\bNone\b|\bnan\b", text)
    assert "betaN" not in text and "Bt" not in text


def test_a_unit_prints_exactly_as_the_registry_declares_it():
    """ne_line's unit was confirmed on 2026-09-05 as `m/cm3` (path length in metres times density
    in cm^-3, the BCI leaf's MDSplus units attribute) and is printed verbatim; `m^-3` would be
    inventing a conversion the description never made. A unit still `[?]` in the registry prints
    as `[?]` -- see test_retrieval.test_an_unverified_unit_never_gets_a_prefix_invented_for_it."""
    text = D.describe(record())
    assert "ne_line 9.82e+13 m/cm3" in text
    assert "m^-3" not in text


def test_the_named_segment_is_never_silently_swapped_for_another():
    text = D.describe(record(), segment="ramp_up")
    assert "Flat top" not in text and "Ramp up" not in text
    assert text.startswith("Shot 161172")


def test_provenance_marks_an_assumed_efit_run():
    assert "betaN 2.28 (EFIT01, assumed)" in D.describe(record())
    prov = {"betan": schema.Provenance(tool="EFIT", version="01", assumed=False)}
    assert "betaN 2.28 (EFIT01)" in D.describe(record(derived_provenance=prov))


def test_labels_and_outcome_read_as_english():
    text = D.describe(record())
    assert line(text, "Labels:") == "Labels: H-mode (text), disruption_free."
    assert line(text, "Outcome:") == "Outcome: Ip target hit (-1 %); programmed ramp-down at 5.9 s."


def test_a_missed_target_and_a_fault_both_show():
    out = schema.Outcome(
        ip_target_hit=False,
        ip_target_err=0.31,
        end_reason="fast_current_quench",
        fault_strings=["pcs first fault"],
    )
    got = line(D.describe(record(outcome=out)), "Outcome:")
    assert got == (
        "Outcome: Ip missed target (+31 %); fast current quench at 5.9 s; faults: pcs first fault."
    )


# ------------------------------------------------------------------------------------- quotes


def entries(*specs) -> schema.HumanTier:
    return schema.HumanTier(
        log_entries=[schema.LogEntry(role=r, author=a, time=t, text=x) for r, a, t, x in specs]
    )


def test_a_quote_is_a_verbatim_prefix_of_exactly_one_entry():
    human = entries(
        ("PHYSICS_OPERATOR", "hyatt", "16:43", REAL_ENTRY_161172),
        ("CHIEF_OPERATOR", "leer", "16:55", "Plasma shot; ok"),
    )
    got = line(D.describe(record(human=human)), "Operator:")
    quoted = re.fullmatch(r'Operator: "(.*)" \(PHYSICS_OPERATOR hyatt 16:43\)\.', got).group(1)
    assert WS.sub(" ", REAL_ENTRY_161172).startswith(quoted)
    # and nothing from the other entry leaked in
    assert "Plasma shot" not in got


def test_two_entries_are_never_spliced_into_one_quotation():
    human = entries(
        ("PHYSICS_OPERATOR", "a", "10:00", "First author says the shot was fine."),
        ("PHYSICS_OPERATOR", "b", "10:05", "Second author says it disrupted."),
    )
    got = line(D.describe(record(human=human)), "Operator:")
    assert got == 'Operator: "First author says the shot was fine." (PHYSICS_OPERATOR a 10:00).'
    assert "Second author" not in got


def test_a_postshot_note_wins_over_the_preshot_form():
    human = entries(
        (
            "SESSION_LEADER",
            "moyer",
            "11:07",
            "Preshot:\nreduce the ECH step\n- - - -\nrequested ip: 0 MA, btor: 0 T",
        ),
        ("SESSION_LEADER", "moyer", "11:15", "Postshot: lost 1 gyrotron early."),
    )
    got = line(D.describe(record(human=human)), "Operator:")
    assert got == 'Operator: "Postshot: lost 1 gyrotron early." (SESSION_LEADER moyer 11:15).'


def test_status_tables_are_not_quoted():
    """Every RF and DIAGNOSTICS entry in the real corpus is a tab-separated status table. Cutting
    one to 140 characters would produce something that reads like a sentence and is not one."""
    human = entries(
        ("RF", "lohr", "16:45", "System\tDelay\tRequest\tActual\nLeia\t1900\t3100\t1884"),
        ("DIAGNOSTICS", "x", "09:14", "IRTV 60 degree 464 x 4 pixels Xoff=112 Yoff=260"),
    )
    assert line(D.describe(record(human=human)), "Operator:") is None


def test_a_long_entry_is_cut_at_a_sentence_boundary_not_mid_clause():
    text = (
        "The plasma reached a good pedestal and held it for the whole flat top. "
        "Then the density climbed and we lost the H-mode about two hundred milliseconds "
        "before the programmed ramp-down started."
    )
    human = entries(("PHYSICS_OPERATOR", "a", "10:00", text))
    got = line(D.describe(record(human=human)), "Operator:")
    quoted = got[len('Operator: "') : got.index('" (')]
    assert text.startswith(quoted)
    assert quoted.endswith("flat top.")


def test_a_long_entry_with_no_sentence_boundary_is_word_cut_and_marked():
    text = "word " * 60
    human = entries(("PHYSICS_OPERATOR", "a", "10:00", text))
    got = line(D.describe(record(human=human)), "Operator:")
    quoted = got[len('Operator: "') : got.index('" (')]
    assert quoted.endswith(" ...")
    assert WS.sub(" ", text).strip().startswith(quoted[:-4])


def test_no_quotable_entry_means_no_operator_line():
    assert line(D.describe(record(human=schema.HumanTier())), "Operator:") is None


def test_quotable_is_the_one_filter_for_every_quote_on_screen():
    """The Operator line here and the query's `log` highlight (rank._highlight) both ask this
    function, so neither can quote a settings dump the other refuses, and both decode entities
    the same way. A quotable text is a whitespace-collapsed copy of the one entry, nothing else."""
    entry = schema.LogEntry
    assert D.quotable(entry(role="PCS", text="PCS CHANGES: ECH: change parameter data")) is None
    assert D.quotable(entry(role="RF", text="System\tDelay\tRequest\nLeia\t1900\t3100")) is None
    assert (
        D.quotable(entry(role="SESSION_LEADER", text="Preshot:\nx\n- - - -\nrequested ip: 1 MA"))
        is None
    )
    assert D.quotable(entry(role="SESSION_LEADER", text="   ")) is None
    assert (
        D.quotable(entry(role="SESSION_LEADER", text="didn&#39;t  get\n ECH")) == "didn't get ECH"
    )
    assert D.quotable(entry(role="PHYSICS_OPERATOR", text=REAL_ENTRY_161172)) == WS.sub(
        " ", REAL_ENTRY_161172
    )


# -------------------------------------------------------------------------------- fact check


def test_extract_facts_separates_numbers_from_shot_references():
    nums, shots = D.extract_facts("Shot 161172 ran at Ip 1.21 MA and 4.2e19 m^-3, ref -0.1.")
    assert shots == {"161172"}
    assert {"1.21", "4.2e19", "-0.1"} <= nums


def test_shot_references_cover_the_whole_database_band():
    """The database spans 160904-204988. The pattern used to be `1\\d{5}`, so every shot at or
    above 200000 (24 of the 201 in poc_v1) was invisible to the fact check and to the chat's
    unverified-shot marker."""
    _, shots = D.extract_facts("shot 203407 and 161172")
    assert shots == {"203407", "161172"}
    assert D.SHOT_REF is D._SHOT_REF  # one definition; the old private name still works


def test_check_facts_is_a_multiset_comparison_in_both_directions():
    t = "Ip 1.20 MA, Bt 1.20 T on shot 161172."
    assert D.check_facts(t, "On 161172 the field was 1.20 T at 1.20 MA of current.")
    assert not D.check_facts(t, "Ip 1.2 MA, Bt 1.20 T on shot 161172.")  # rounded
    assert not D.check_facts(t, "Ip and Bt were both 1.20 on shot 161172.")  # one 1.20 dropped
    assert not D.check_facts(t, "Ip 1.20 MA, Bt 1.20 T, q95 3.5 on shot 161172.")  # invented
    assert not D.check_facts(t, "Ip 1.20 MA, Bt 1.20 T on shot 161173.")  # wrong shot


def test_a_real_description_checks_against_itself():
    text = D.describe(record())
    assert D.check_facts(text, text)


# ----------------------------------------------------------------------------------- polish


def test_polish_opens_no_socket_when_the_provider_is_off(monkeypatch):
    """`provider: off` has to mean no socket at all: on this node an outbound HTTPS call does not
    fail fast, it hangs for minutes inside connect."""
    import socket

    from shot_design import config

    real = config.load_yaml
    monkeypatch.setattr(
        config,
        "load_yaml",
        lambda name: {"provider": "off"} if name == "llm.yaml" else real(name),
    )

    def forbidden(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("polish() must not touch the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    assert D._client_for_polish() is None
    text = D.describe(record())
    assert D.polish(text) == (text, False)


def test_polish_returns_the_template_when_no_endpoint_is_published(paths):
    """The shipped config names a provider, but nothing is running: the template comes back
    unchanged, with no warning and no network call (the client stats one absent file)."""
    text = D.describe(record())
    assert D.polish(text) == (text, False)


# -------------------------------------------------------------------------------- real data


@pytest.mark.real_data
@pytest.mark.skipif(not DB_DIR.exists(), reason=f"{DB_DIR} not mounted")
def test_every_real_quote_is_a_verbatim_prefix_of_one_attributed_entry():
    from shot_design.shotdb import store

    db = store.ShotDB.load(DB_DIR)
    pattern = re.compile(r'^Operator: "(.*)" \((.+)\)\.$')
    quoted = 0
    quotable_shots = 0
    for shot in db.shots.index:
        rec = db.get(int(shot))
        quotable_shots += any(D.quotable(e) for e in rec.human.log_entries)
        text = D.describe(rec)
        # Word boundaries: real operator prose contains "resonant", which merely spells "nan".
        assert not re.search(r"\bNone\b|\bnan\b", text), shot
        got = line(text, "Operator: ")
        if got is None:
            continue
        quoted += 1
        body, who = pattern.match(got).groups()
        body = body.removesuffix(" ...")
        # html.unescape on the ENTRY side, matching what _quote does for display: the logbook is
        # scraped from HTML and 5 of the 105 built shots carry an entity, so shot 161584 renders
        # "didn't get ECH" where the stored entry holds "didn&#39;t get ECH". Decoding both sides
        # keeps the guarantee this test exists for -- the quote is still a prefix of exactly ONE
        # attributed entry and can never splice two authors together -- while allowing the one
        # transformation the display layer is permitted to make. Entities are decoded here and
        # never in text.parse_log_entries, because "n&#39;t" -> "n't" would feed verdict()'s
        # negation vocabulary and silently rescore shots.
        assert any(
            " ".join(b for b in (e.role, e.author, e.time) if b) == who
            and WS.sub(" ", html.unescape(e.text)).strip().startswith(body)
            for e in rec.human.log_entries
        ), f"shot {shot}: quote is not a prefix of a single entry by {who}"
    # Every shot with a quotable entry gets a quote, and no shot without one does: in the 201-shot
    # database two shots (191868, 204449) have log entries only from roles the quote never uses.
    assert quoted == quotable_shots
    assert quotable_shots >= 0.95 * len(db.shots)
