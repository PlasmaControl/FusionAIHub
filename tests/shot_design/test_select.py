"""`shot_design corpus select`: the 500-shot `recommender_v1` development universe (plan 5.7).

Everything iterations 0-3 is developed on comes out of this rule, so what these tests pin is
that each clause of it can reject on its own and say which one did -- a shot that fell out for
the wrong reason is a bias in every model trained on the list, and a silent one.

All synthetic: bundles are built to the real per-shot text layout (`conftest.text_bundle`) and
census frames to `census.COLUMNS`. Nothing here reads the corpus.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import yaml

from shot_design import cli
from shot_design.shotdb import census, select, text

from .conftest import census_frame, text_bundle

THEMES = [
    {"id": "startup_checkout", "keywords": ["startup", "checkout"]},
    {"id": "rmp_elm", "keywords": ["rmp", "elm suppression"]},
    {"id": "qh_mode", "keywords": ["qh-mode", "qh mode"]},
    {"id": "tearing_mhd", "keywords": ["tearing", "ntm"]},
]

# The shot table row of a shot that satisfies rule (a) outright, with the eleven columns a real
# 2021-2025 plasma row carries -- so the shot-specific block it produces is 238 characters, which
# is the 25th percentile of the real ones and lets MIN_SHOT_CHARS be exercised for real.
GOOD_ROW = {
    "SHOT_TYPE": "plasma",
    "BTOR": "-2.13",
    "A": "1.89",
    "R": "1.17",
    "TIME-OF-SHOT": "10:02",
    "PULSE-LENGTH": "4.09",
    "IP-(MA)": "1.45",
    "PBEAM-MAX-(MW)": "8.46",
    "PECH-MAX-(MW)": "0.57",
    "PLH-MAX-(MW)": "1.78",
}
# The census groups rule (c) requires, each with more than 2 s of coverage.
GOOD_GROUPS = {"mhr": 5.0, "ece": 5.0, "filterscopes": 5.0}


def facts(shot: int = 190000, **over) -> select.ShotFacts:
    """One shot's facts from a bundle built on GOOD_ROW. `row=None` is the session-fallback
    case; `row={...}` overrides individual columns of the good row."""
    given = over.pop("row", {})
    row = None if given is None else {**GOOD_ROW, **given}
    return select.parse_facts(shot, text_bundle(shot, row=row, **over))


# ------------------------------------------------------------------ the bundle parser (text.py)


def test_shot_table_row_reads_only_the_shot_specific_block():
    """The general session header carries `- Run:` and `- Session Leader:` bullets of its own.

    They are the SESSION's, not the shot's, and a parser that swept the whole bundle would mix
    a run-level field into a per-shot fact without any sign that it had.
    """
    b = text_bundle(190000, row=GOOD_ROW)
    assert "- Session Leader:" in b
    kv = text.shot_table_row(b)
    assert kv["SHOT_TYPE"] == "plasma" and kv["IP-(MA)"] == "1.45"
    assert "Session Leader" not in kv and "Run" not in kv


def test_shot_table_row_is_empty_for_the_session_fallback_case():
    assert text.shot_table_row(text_bundle(190000, row=None)) == {}


def test_shot_block_is_everything_after_the_marker():
    b = text_bundle(190000, row=GOOD_ROW)
    block = text.shot_block(b)
    assert block.startswith("SHOT: 190000")
    # 238 characters: a real plasma row's block runs 138-302 (median 252), and this fixture sits
    # inside that range so MIN_SHOT_CHARS is exercised against a realistic length.
    assert 138 <= len(block) <= 302
    assert "Mini-proposal" not in block and text.SHOT_BLOCK_MARKER not in block


def test_session_metadata_is_the_selected_metadata_json():
    meta = text.session_metadata(text_bundle(190000, row=GOOD_ROW, title="Tearing mode avoidance"))
    assert meta["title"] == "Tearing mode avoidance"
    assert meta["run_id"] == "20220301"


# ------------------------------------------------------------------------------ rule (a)


def test_a_shot_that_satisfies_every_clause_is_eligible():
    ok, reasons = select.eligible(facts(), GOOD_GROUPS)
    assert ok is True and reasons == ()


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ({"SHOT_TYPE": "power supply test"}, "shot_type"),
        ({"IP-(MA)": "0.49"}, "ip"),
        ({"PULSE-LENGTH": "1.99"}, "pulse_length"),
        ({"PBEAM-MAX-(MW)": "0.99", "PECH-MAX-(MW)": "0.00"}, "heating"),
    ],
)
def test_rule_a_rejects_each_clause_on_its_own(row, reason):
    ok, reasons = select.eligible(facts(row=row), GOOD_GROUPS)
    assert ok is False and reason in reasons


@pytest.mark.parametrize("ip", ["1.10", "-1.10"])
def test_rule_a_reads_the_magnitude_of_ip_and_not_its_sign(ip):
    """DIII-D logs a reversed-polarity discharge with a NEGATIVE `IP-(MA)`.

    `IP-(MA) >= 0.5` on the signed number is not "at least half a mega-amp", it is "at least half
    a mega-amp AND in the forward direction" -- a physics filter nobody wrote down. It rejected
    all 663 reversed-Ip plasma shots of the pool, including all 254 whose run title matches the
    `qh_mode` lexicon, over 13 run days -- which is why `recommender_v1` v1 had zero `qh_mode`
    shots in a list whose stated purpose includes EHO/QH.
    """
    ok, reasons = select.eligible(facts(row={"IP-(MA)": ip}), GOOD_GROUPS)
    assert ok is True and reasons == ()


@pytest.mark.parametrize("ip", ["0.3", "-0.3"])
def test_a_current_under_the_floor_is_rejected_at_either_polarity(ip):
    """The magnitude is the rule, so it has to bite in both directions -- `abs()` must not turn
    the clause into "any plasma at all"."""
    ok, reasons = select.eligible(facts(row={"IP-(MA)": ip}), GOOD_GROUPS)
    assert ok is False and "ip" in reasons


def test_the_polarity_is_recorded_even_though_the_rule_ignores_it():
    """A counter-Ip discharge is a different plasma, so the list has to say which shots are one --
    the rule admits them, the reader decides what to do about them."""
    assert facts(row={"IP-(MA)": "-1.10"}).ip_sign == -1
    assert facts(row={"IP-(MA)": "1.10"}).ip_sign == 1
    assert facts(row={"IP-(MA)": ""}).ip_sign is None


def test_a_missing_shot_table_field_is_a_rejection_not_a_pass():
    """`IP-(MA)` absent is not `IP-(MA)` large. The 2021-2025 corpus has plasma shots whose row
    carries only SHOT/SHOT_TYPE/BTOR/TIME/PULSE-LENGTH/IP -- a rule that read a missing field as
    unconstrained would admit them on evidence that is not there."""
    b = text_bundle(190000, row={"SHOT_TYPE": "plasma", "IP-(MA)": "1.4", "PULSE-LENGTH": "3.0"})
    ok, reasons = select.eligible(select.parse_facts(190000, b), GOOD_GROUPS)
    assert ok is False and "heating" in reasons


def test_either_beam_or_ech_satisfies_the_heating_clause():
    beam = facts(row={"PBEAM-MAX-(MW)": "1.0", "PECH-MAX-(MW)": "0.00"})
    ech = facts(row={"PBEAM-MAX-(MW)": "0.00", "PECH-MAX-(MW)": "0.01"})
    assert select.eligible(beam, GOOD_GROUPS)[0] is True
    assert select.eligible(ech, GOOD_GROUPS)[0] is True


def test_a_shot_table_value_with_a_stray_space_is_still_a_number():
    """Real rows carry `KAPPA: 1.9 E15` -- the HTML-to-text conversion splits some exponents."""
    assert select.parse_facts(190000, text_bundle(190000, row={**GOOD_ROW, "IP-(MA)": "1.4 5"}))


# ------------------------------------------------------------------------------ rule (b)


def test_the_session_fallback_case_is_rejected_as_such():
    ok, reasons = select.eligible(facts(row=None), GOOD_GROUPS)
    assert ok is False and "session_fallback" in reasons


def test_a_thin_shot_specific_block_is_rejected():
    f = facts()
    assert select.eligible(f, GOOD_GROUPS)[0] is True
    thin = select.eligible(f, GOOD_GROUPS, min_shot_chars=10_000)
    assert thin[0] is False and "shot_text" in thin[1]


@pytest.mark.parametrize("title", ["PS test", "Bcoil check", "Mag Cal", "abort study", "TEST ps"])
def test_a_title_naming_a_machine_activity_is_rejected(title):
    ok, reasons = select.eligible(facts(title=title), GOOD_GROUPS)
    assert ok is False and "title" in reasons


@pytest.mark.parametrize(
    "title",
    [
        "QH-mode with counter-NBI",
        "Pedestal structure at high triangularity",
        "Steps toward a stationary hybrid",  # 'ps' inside 'Steps' is not the word 'PS'
    ],
)
def test_a_physics_title_is_accepted(title):
    assert select.eligible(facts(title=title), GOOD_GROUPS) == (True, ())


def test_a_shot_with_no_title_at_all_is_not_rejected_by_the_title_rule():
    """181 of the pool's bundles have no `title` in their session metadata. Absent is not
    excluded -- the rule names activities to keep out, and it cannot see one that is not there."""
    assert select.eligible(facts(title=None), GOOD_GROUPS) == (True, ())


# ------------------------------------------------------------------------------ rule (c)


@pytest.mark.parametrize("group", ["mhr", "ece", "filterscopes"])
def test_rule_c_rejects_a_missing_required_group(group):
    spans = {k: v for k, v in GOOD_GROUPS.items() if k != group}
    ok, reasons = select.eligible(facts(), spans)
    assert ok is False and f"census_{group}" in reasons


@pytest.mark.parametrize("group", ["mhr", "ece", "filterscopes"])
def test_rule_c_rejects_a_group_that_recorded_for_under_two_seconds(group):
    ok, reasons = select.eligible(facts(), {**GOOD_GROUPS, group: 1.99})
    assert ok is False and f"census_{group}" in reasons


def test_spans_are_read_off_the_census_as_present_groups_only():
    df = census_frame(
        [
            (190000, "mhr", 0.0, 5.0, True),
            (190000, "co2", np.nan, np.nan, False),  # the (C, 1) placeholder
            (190001, "ece", -1.0, 4.0, True),
        ]
    )
    got = select.spans(df)
    assert got[190000] == {"mhr": pytest.approx(5.0)}
    assert got[190001] == {"ece": pytest.approx(5.0)}


def test_an_unopenable_file_has_no_spans_and_so_fails_rule_c():
    df = census_frame([(190002, "", np.nan, np.nan, False)], openable=False)
    assert select.spans(df).get(190002, {}) == {}
    ok, reasons = select.eligible(facts(190002), {})
    assert ok is False and "census_mhr" in reasons


# ------------------------------------------------------------------------------ rule (d)


def test_flattop_is_the_longest_run_of_ip_above_the_plateau_fraction():
    t = np.arange(0, 6.0, 0.025)
    ip = np.where((t >= 1.0) & (t <= 4.0), 1.0e6, 1.0e5)
    assert select.flattop_from_ip(t, ip) == pytest.approx(3.0, abs=0.05)


def test_flattop_ignores_the_sign_of_ip_and_any_nan_samples():
    t = np.arange(0, 6.0, 0.025)
    ip = np.where((t >= 1.0) & (t <= 4.0), -1.0e6, -1.0e5)
    ip[0] = np.nan
    assert select.flattop_from_ip(t, ip) == pytest.approx(3.0, abs=0.05)


def test_flattop_of_a_trace_with_nothing_in_it_is_not_a_number():
    assert np.isnan(select.flattop_from_ip(np.arange(3.0), np.zeros(3)))


def test_the_pulse_length_proxy_subtracts_the_measured_ramp_time():
    assert select.flattop_proxy(4.09) == pytest.approx(4.09 - select.PROXY_RAMP_S)
    assert select.flattop_proxy(0.5) == 0.0  # never negative
    assert np.isnan(select.flattop_proxy(None))


def test_rule_d_rejects_a_flattop_under_a_second():
    short = select.parse_facts(
        190000, text_bundle(190000, row={**GOOD_ROW, "PULSE-LENGTH": "2.1"})
    )
    assert short.flattop_s < 1.0 and short.flattop_source == "pulse_length_proxy"
    ok, reasons = select.eligible(short, GOOD_GROUPS)
    assert ok is False and "flattop" in reasons


def test_a_measured_flattop_replaces_the_proxy_and_says_so():
    f = select.with_flattop(facts(), 2.5, "features_ip")
    assert f.flattop_s == 2.5 and f.flattop_source == "features_ip"
    assert select.eligible(f, GOOD_GROUPS) == (True, ())


# ------------------------------------------------------------------------------ themes


@pytest.mark.parametrize(
    ("title", "theme"),
    [
        ("RMP ELM suppression at low q95", "rmp_elm"),
        ("QH-mode access", "qh_mode"),
        ("NTM control", "tearing_mhd"),
        ("Systems checkout", "startup_checkout"),
        ("Something with no keyword in it", None),
        (None, None),
    ],
)
def test_assign_theme_takes_the_first_matching_lexicon_entry(title, theme):
    assert select.assign_theme(title, THEMES) == theme


def test_the_real_lexicon_is_the_default_and_has_fourteen_themes():
    assert select.assign_theme("RMP ELM suppression") == "rmp_elm"
    assert len(select.lexicon_themes()) == 14


def test_a_physics_theme_wins_over_startup_checkout():
    """`labels.yaml` lists `startup_checkout` first and the assignment used to be first-match, so
    any run day whose title also said "checkout" or "calibration" was filed as machine time.

    Measured on the eligible pool: 521 of the 1,394 checkout-titled shots also match a physics
    theme, and `startup_checkout` is the one theme §5.7 excludes from the quotas -- so the
    shadowing did not just mislabel them, it took them out of the quota that was reaching for
    them. The title below is a real one (run 20220906, shots 189889 ff).
    """
    got = select.assign_theme("Divertor diagnostic checkout for QH/WPQH-Mode", THEMES)
    assert got == "qh_mode"


def test_startup_checkout_is_still_assigned_when_no_physics_theme_matches():
    """It is a fallback, not a deletion: a run day that really is only machine time keeps the
    label, so the summary can still count how much of the pool is checkout time."""
    assert select.assign_theme("Systems checkout and diagnostic cal", THEMES) == "startup_checkout"


def test_the_mini_proposal_subject_is_matched_when_the_title_has_no_keyword():
    """`labels.yaml:2` promises "run title + MP title" and only the title was ever read. The
    subject is the experiment's own words and the run-day title is the session leader's."""
    assert select.assign_theme("Tuesday day session", THEMES) is None
    assert select.assign_theme("Tuesday day session", THEMES, subject="NTM control") == "tearing_mhd"


def test_a_bundle_carries_every_subject_line_of_its_mini_proposal_block():
    """Two `Subject:` lines is the common real shape -- the PDF extraction's, whose glyphs the
    font mapping often mangles, and the markdown export's, which is clean. Both are kept, because
    which of the two is readable is a property of that run day's PDF and not of the physics.
    """
    b = text_bundle(
        190000, row=GOOD_ROW, title="Tuesday day session",
        subjects=("QH-mode acce’’ at low to‘que", "QH-mode access at low torque"),
    )
    assert text.mp_subjects(b) == ("QH-mode acce’’ at low to‘que", "QH-mode access at low torque")
    f = select.parse_facts(190000, b)
    assert f.mp_subject and "QH-mode access at low torque" in f.mp_subject
    assert select.assign_theme(f.title, THEMES, subject=f.mp_subject) == "qh_mode"


def test_the_subject_lines_come_from_the_planned_block_and_nowhere_else():
    b = text_bundle(190000, row=GOOD_ROW, title="Tearing mode avoidance")
    assert text.mp_subjects(b) == ("Tearing mode avoidance",)
    assert text.mp_subjects("no mini-proposal here at all") == ()


# ------------------------------------------------------------------------------ diversify


def candidates(
    n: int,
    *,
    start: int = 190000,
    per_run: int = 1,
    theme: str | None = "rmp_elm",
    year: int = 2022,
    preferred: bool = False,
    groups: frozenset[str] = frozenset(),
) -> list[select.Candidate]:
    out = []
    for i in range(n):
        shot = start + i
        out.append(
            select.Candidate(
                shot=shot,
                run_id=f"{year}{(i // per_run):04d}",
                mpid=f"{year}-{(i // per_run):02d}-01",
                year=year,
                theme=theme,
                has_co2="co2" in groups,
                has_bes="bes" in groups,
                has_tangtv="tangtv" in groups,
                flattop_s=3.0,
                flattop_source="pulse_length_proxy",
                preferred=preferred,
            )
        )
    return out


def quotas(**over) -> select.Quotas:
    base = {
        "n": 10,
        "per_run": 3,
        "per_mpid": 5,
        "per_theme": 0,
        "max_year_frac": 1.0,
        "group_min": {},
        "preferred_cap": 150,
    }
    return select.Quotas(**{**base, **over})


def test_no_more_than_three_shots_come_from_one_run():
    # Four run days of ten shots each: twelve are selectable under the cap, ten are asked for.
    got = select.diversify(candidates(40, per_run=10), quotas(n=10), seed=1)
    assert len(got) == 10
    assert max(Counter(c.run_id for c in got).values()) <= 3


def test_no_more_than_five_shots_come_from_one_mini_proposal():
    pool = candidates(60, per_run=1)
    # One mpid over twenty runs: the run cap cannot stand in for the mpid cap.
    pool = [select.replace(c, mpid="2022-11-05") for c in pool]
    got = select.diversify(pool, quotas(n=20), seed=1)
    assert len(got) == 5


def test_a_candidate_with_no_mpid_is_not_capped_against_the_others():
    """mpid is absent from the per-shot bundles; when the logbook cannot supply one either, the
    cap must not collapse every such shot into a single bucket."""
    pool = [select.replace(c, mpid=None) for c in candidates(30, per_run=10)]
    assert len(select.diversify(pool, quotas(n=10, per_run=1000), seed=1)) == 10


def test_each_theme_gets_its_quota_when_the_pool_can_supply_it():
    pool = (
        candidates(40, start=190000, per_run=20, theme="rmp_elm")
        + candidates(40, start=191000, per_run=20, theme="qh_mode")
        + candidates(400, start=192000, per_run=200, theme=None)
    )
    got = select.diversify(pool, quotas(n=60, per_theme=20, per_run=1000, per_mpid=1000), seed=7)
    by_theme = Counter(c.theme for c in got)
    assert by_theme["rmp_elm"] >= 20 and by_theme["qh_mode"] >= 20


def test_startup_checkout_is_excluded_from_the_theme_quotas():
    pool = candidates(40, per_run=20, theme="startup_checkout") + candidates(
        40, start=191000, per_run=20, theme="rmp_elm"
    )
    got = select.diversify(pool, quotas(n=25, per_theme=20, per_run=1000, per_mpid=1000), seed=7)
    reasons = {c.shot: c.reason for c in got}
    assert not any(r == "theme:startup_checkout" for r in reasons.values())
    assert sum(1 for c in got if c.theme == "rmp_elm") >= 20


def test_an_unmet_theme_quota_takes_what_there_is_and_is_reported():
    """A quota that cannot be met is reported with what the pool actually had.

    `got` alone cannot distinguish "there were only five such shots" from "there were 128 and the
    run-day cap ran out first" -- and on the real pool both happen -- so `available` is the count
    of eligible candidates carrying the theme, not a count of anything selected.
    """
    pool = candidates(5, per_run=5, theme="qh_mode") + candidates(
        100, start=191000, per_run=50, theme=None
    )
    q = quotas(n=30, per_theme=20, per_run=1000, per_mpid=1000)
    got = select.diversify(pool, q, seed=3)
    summary = select.summarize(
        n_candidates=len(pool), reasons=Counter(), selected=got, quotas=q, candidates=pool
    )
    assert summary["theme_quota_unmet"]["qh_mode"] == {"want": 20, "got": 5, "available": 5}


def test_a_theme_quota_missed_because_of_the_run_cap_says_how_many_were_available():
    """Twenty shots of one theme, all from a single run day: the <= 3 cap admits three of them.

    This is the interaction that costs three real themes their quota, and the summary has to
    make it legible -- 3 of 20 with 20 available is a different finding from 3 of 20 with 3.
    """
    pool = candidates(20, per_run=20, theme="qh_mode") + candidates(
        100, start=191000, per_run=50, theme=None
    )
    q = quotas(n=30, per_theme=20, per_run=3, per_mpid=1000)
    got = select.diversify(pool, q, seed=3)
    summary = select.summarize(
        n_candidates=len(pool), reasons=Counter(), selected=got, quotas=q, candidates=pool
    )
    assert summary["theme_quota_unmet"]["qh_mode"] == {"want": 20, "got": 3, "available": 20}


def test_the_group_minimums_are_met_from_the_candidates_that_carry_the_group():
    pool = (
        candidates(80, start=190000, per_run=40, groups=frozenset({"co2"}))
        + candidates(80, start=191000, per_run=40, groups=frozenset({"bes"}))
        + candidates(80, start=192000, per_run=40, groups=frozenset({"tangtv"}))
        + candidates(400, start=193000, per_run=200)
    )
    q = quotas(n=200, per_run=1000, per_mpid=1000, group_min={"co2": 50, "bes": 50, "tangtv": 30})
    got = select.diversify(pool, q, seed=11)
    assert sum(c.has_co2 for c in got) >= 50
    assert sum(c.has_bes for c in got) >= 50
    assert sum(c.has_tangtv for c in got) >= 30


def test_preferred_shots_are_taken_first_and_capped():
    pool = candidates(300, start=190000, per_run=150, preferred=True) + candidates(
        300, start=191000, per_run=150
    )
    got = select.diversify(pool, quotas(n=200, per_run=1000, per_mpid=1000, preferred_cap=150), seed=5)
    assert sum(c.preferred for c in got) == 150
    assert {c.reason for c in got if c.preferred} == {"preferred"}


def test_no_single_year_may_take_more_than_the_year_fraction():
    pool = [
        c
        for i, y in enumerate((2022, 2023, 2024))
        for c in candidates(400, start=190000 + 5000 * i, per_run=200, year=y)
    ]
    q = quotas(n=100, per_run=1000, per_mpid=1000, max_year_frac=0.4)
    got = select.diversify(pool, q, seed=2)
    assert len(got) == 100
    assert max(Counter(c.year for c in got).values()) <= 40


def test_the_fill_spreads_across_the_years_the_pool_has():
    pool = [
        c
        for i, y in enumerate(range(2021, 2026))
        for c in candidates(200, start=190000 + 1000 * i, per_run=100, year=y)
    ]
    got = select.diversify(pool, quotas(n=100, per_run=1000, per_mpid=1000), seed=2)
    assert set(Counter(c.year for c in got)) == {2021, 2022, 2023, 2024, 2025}
    assert min(Counter(c.year for c in got).values()) >= 15


def test_the_same_seed_selects_the_same_list_and_another_seed_does_not():
    pool = candidates(400, per_run=200)
    a = select.diversify(pool, quotas(n=50), seed=20260907)
    again = select.diversify(pool, quotas(n=50), seed=20260907)
    b = select.diversify(pool, quotas(n=50), seed=1)
    assert [c.shot for c in a] == [c.shot for c in again]
    assert {c.shot for c in a} != {c.shot for c in b}


def test_the_selected_list_comes_back_in_shot_order():
    got = select.diversify(candidates(400, per_run=200), quotas(n=50), seed=4)
    assert [c.shot for c in got] == sorted(c.shot for c in got)


def test_a_pool_smaller_than_n_returns_what_there_is():
    got = select.diversify(candidates(7, per_run=1), quotas(n=50), seed=4)
    assert len(got) == 7


# ------------------------------------------------- rule (d), pass two: measured flat-top


def test_a_selected_shot_with_no_feature_file_is_kept_and_marked_pending():
    """`pulse_length_proxy` on a selected shot is an estimate that was never checked. Saying
    `pending` instead is the difference between "measured >= 1 s" and "nobody has looked" -- and
    it is what puts the shot on the list the features stage has to be run for."""
    pool = candidates(20, per_run=1)
    sel = select.diversify(pool, quotas(n=5), seed=3)
    out, repl = select.verify_flattop(sel, pool, quotas(n=5), seed=3, measure=lambda s: None)
    assert [c.shot for c in out] == [c.shot for c in sel]
    assert {c.flattop_source for c in out} == {"pending"}
    assert [c.flattop_s for c in out] == [c.flattop_s for c in sel]  # the proxy is kept as the estimate
    assert repl == []


def test_a_measured_flattop_replaces_the_proxy_on_every_verified_shot():
    pool = candidates(20, per_run=1)
    sel = select.diversify(pool, quotas(n=5), seed=3)
    out, repl = select.verify_flattop(sel, pool, quotas(n=5), seed=3, measure=lambda s: 2.4)
    assert {c.flattop_source for c in out} == {"features_ip"}
    assert {c.flattop_s for c in out} == {2.4} and repl == []


def test_a_shot_whose_measured_flattop_is_short_is_dropped_and_replaced():
    """This is rule (d) biting for real. The proxy said the shot had a flat-top; the Ip trace says
    it did not, and a list of 500 has to still be 500."""
    pool = candidates(20, per_run=1)
    sel = select.diversify(pool, quotas(n=5), seed=3)
    doomed = sel[0].shot
    out, repl = select.verify_flattop(
        sel, pool, quotas(n=5), seed=3, measure=lambda s: 0.4 if s == doomed else None
    )
    shots = [c.shot for c in out]
    assert len(out) == 5 and doomed not in shots
    assert len(repl) == 1
    assert repl[0]["dropped"] == doomed and repl[0]["dropped_flattop_s"] == 0.4
    assert repl[0]["replacement"] in shots and repl[0]["replacement"] not in [c.shot for c in sel]


def test_a_replacement_carries_the_theme_of_the_shot_it_replaces():
    """A quota that was met before the verification has to be met after it. Replacing a
    `tearing_mhd` shot with whatever came next would quietly undo the theme floor."""
    pool = candidates(10, theme="rmp_elm") + candidates(10, start=191000, theme="tearing_mhd")
    sel = select.diversify(pool, quotas(n=6, per_theme=3), seed=5)
    doomed = next(c for c in sel if c.theme == "tearing_mhd")
    out, repl = select.verify_flattop(
        sel, pool, quotas(n=6, per_theme=3), seed=5,
        measure=lambda s: 0.4 if s == doomed.shot else None,
    )
    got = next(c for c in out if c.shot == repl[0]["replacement"])
    assert got.theme == "tearing_mhd"
    assert Counter(c.theme for c in out) == Counter(c.theme for c in sel)


def test_a_candidate_whose_own_measured_flattop_is_short_is_not_used_as_a_replacement():
    pool = candidates(20, per_run=1)
    sel = select.diversify(pool, quotas(n=5), seed=3)
    doomed = sel[0].shot
    chosen = {c.shot for c in sel}
    first_spare = next(
        c.shot for c in select._by_year([c for c in pool if c.shot not in chosen], 3)
    )
    out, _repl = select.verify_flattop(
        sel, pool, quotas(n=5), seed=3,
        measure=lambda s: 0.4 if s in (doomed, first_spare) else None,
    )
    shots = [c.shot for c in out]
    assert doomed not in shots and first_spare not in shots and len(out) == 5


def test_a_replacement_still_obeys_the_caps():
    """The freed slot is the dropped shot's, so its run day gets it back -- but no other run day
    may go over three because of a replacement."""
    pool = candidates(40, per_run=10)
    sel = select.diversify(pool, quotas(n=10), seed=7)
    doomed = {c.shot for c in sel[:3]}
    out, _ = select.verify_flattop(
        sel, pool, quotas(n=10), seed=7, measure=lambda s: 0.4 if s in doomed else None
    )
    assert len(out) == 10
    assert max(Counter(c.run_id for c in out).values()) <= 3


def test_the_verification_is_deterministic_and_comes_back_in_shot_order():
    pool = candidates(40, per_run=10)
    sel = select.diversify(pool, quotas(n=10), seed=7)
    doomed = {sel[0].shot}
    runs = [
        select.verify_flattop(
            sel, pool, quotas(n=10), seed=7, measure=lambda s: 0.4 if s in doomed else None
        )
        for _ in range(2)
    ]
    assert [c.shot for c in runs[0][0]] == [c.shot for c in runs[1][0]]
    assert [c.shot for c in runs[0][0]] == sorted(c.shot for c in runs[0][0])
    assert runs[0][1] == runs[1][1]


def test_a_shot_with_no_replacement_available_is_reported_as_such():
    """Seven candidates, seven selected, one fails the measured rule: the list is short by one and
    the summary has to say so rather than quietly returning 6 of 7."""
    pool = candidates(7, per_run=1)
    sel = select.diversify(pool, quotas(n=7), seed=3)
    out, repl = select.verify_flattop(
        sel, pool, quotas(n=7), seed=3, measure=lambda s: 0.4 if s == sel[0].shot else None
    )
    assert len(out) == 6 and repl[0]["replacement"] is None



# ------------------------------- rule (d), pass three: re-verifying a COMMITTED list


def listed(cands, *, reason: str = "fill") -> list[select.Candidate]:
    """`cands` as a committed list comes back: written into the YAML rows and read out again.

    Round-tripped rather than passed straight through, because that is what `--from-list` does
    and the row is the only thing the second invocation has -- a field the document does not
    carry is a field the re-verification cannot use.
    """
    doc = select.document(
        [select.replace(c, reason=reason) for c in cands],
        select.summarize(
            n_candidates=len(cands), reasons=Counter(), selected=cands, quotas=quotas()
        ),
        name="x",
        seed=3,
        n=len(cands),
    )
    return select.candidates_from_rows(doc["shots"])


def test_a_committed_list_survives_the_round_trip_through_its_own_rows():
    pool = candidates(4, per_run=1, groups=frozenset({"co2", "tangtv"}))
    back = listed(pool, reason="theme:rmp_elm")
    assert [c.shot for c in back] == [c.shot for c in pool]
    for a, b in zip(pool, back, strict=True):
        assert (b.run_id, b.mpid, b.year, b.theme) == (a.run_id, a.mpid, a.year, a.theme)
        assert (b.has_co2, b.has_bes, b.has_tangtv) == (a.has_co2, a.has_bes, a.has_tangtv)
        assert (b.flattop_s, b.flattop_source, b.ip_sign) == (a.flattop_s, a.flattop_source, a.ip_sign)
        assert b.reason == "theme:rmp_elm"


def test_the_committed_list_is_re_verified_and_never_re_selected():
    """The whole point of `--from-list`. The pool the second invocation sees is not the pool the
    first one saw -- the features stage has run since, so more shots are `preferred` and more have
    a measured flat-top -- and re-running the selection on it returns a different 500. These five
    shots are not in the pool at all, and they all come back."""
    pool = candidates(40, per_run=10)
    committed = listed(candidates(5, start=195000, per_run=1))
    out, repl, pending = select.reverify_flattop(
        committed, pool, quotas(n=5), seed=3, measure=lambda s: 2.4
    )
    assert [c.shot for c in out] == [c.shot for c in committed]
    assert repl == [] and pending == []
    assert {c.flattop_source for c in out} == {"features_ip"}
    assert {c.flattop_s for c in out} == {2.4}


def test_only_the_listed_shots_the_measurement_rejects_are_replaced():
    pool = candidates(20, per_run=1)
    committed = listed(pool[:5])
    doomed = committed[0].shot
    out, repl, pending = select.reverify_flattop(
        committed, pool, quotas(n=5), seed=3,
        measure=lambda s: 0.4 if s == doomed else 2.4,
    )
    shots = [c.shot for c in out]
    assert len(out) == 5 and doomed not in shots and pending == []
    # Every other listed shot is still there, and only one row is new.
    assert {c.shot for c in committed[1:]} < set(shots)
    assert len(repl) == 1 and repl[0]["dropped"] == doomed
    assert repl[0]["replacement"] in shots and repl[0]["replacement"] not in {c.shot for c in committed}
    assert repl[0]["replacement_flattop_source"] == "features_ip"


def test_a_listed_shot_that_still_has_no_features_is_kept_pending_and_named():
    """The refusal the CLI turns into exit 1: the caller has to be told WHICH shots, because the
    pending file it rewrites is the work order for them."""
    pool = candidates(20, per_run=1)
    committed = listed(pool[:5])
    waiting = committed[2].shot
    out, repl, pending = select.reverify_flattop(
        committed, pool, quotas(n=5), seed=3,
        measure=lambda s: None if s == waiting else 2.4,
    )
    assert pending == [waiting]
    assert next(c for c in out if c.shot == waiting).flattop_source == "pending"
    assert repl == []


def test_a_kept_shot_is_kept_even_where_it_would_now_break_a_cap():
    """The committed list IS the eligibility snapshot. Re-imposing the caps on shots that are
    already in it would let a re-verification quietly shrink the list -- and the caps were
    satisfied by the run that made it, once, on the pool it saw."""
    pool = candidates(40, per_run=10)          # ten shots share every run day
    committed = listed(pool[:5])               # five of one run day, past the per_run=3 cap
    out, repl, pending = select.reverify_flattop(
        committed, pool, quotas(n=5, per_run=3), seed=3, measure=lambda s: 2.4
    )
    assert [c.shot for c in out] == [c.shot for c in committed] and repl == [] and pending == []


def test_a_replacement_for_a_committed_shot_still_obeys_the_caps():
    pool = candidates(40, per_run=10)
    committed = listed(pool[0:3] + pool[10:13])   # three from run A, three from run B
    doomed = pool[12].shot
    out, repl, _pending = select.reverify_flattop(
        committed, pool, quotas(n=6, per_run=3), seed=3,
        measure=lambda s: 0.4 if s == doomed else 2.4,
    )
    assert len(out) == 6
    assert max(Counter(c.run_id for c in out).values()) <= 3
    # Run A is full at three and stays full; the freed slot is run B's, so run B may have it back.
    assert next(c for c in out if c.shot == repl[0]["replacement"]).run_id != pool[0].run_id


def test_a_replacement_on_a_committed_list_must_itself_be_measured():
    """`--finalize` means every row has been measured, replacements included. On the first pass a
    replacement with no feature file is admitted as `pending` (there is nothing better and the
    list is not final yet); on a re-verification that would put a shot on the list that the run
    finalizing it has never measured, and would keep the loop going for one more round."""
    pool = candidates(20, per_run=1)
    committed = listed(pool[:5])
    doomed = committed[0].shot
    chosen = {c.shot for c in committed}
    first_spare, second_spare = (
        c.shot for c in list(select._by_year([c for c in pool if c.shot not in chosen], 3))[:2]
    )
    out, repl, pending = select.reverify_flattop(
        committed, pool, quotas(n=5), seed=3,
        measure=lambda s: 0.4 if s == doomed else (None if s == first_spare else 2.4),
    )
    assert repl[0]["replacement"] == second_spare
    assert first_spare not in {c.shot for c in out} and pending == []
    assert {c.flattop_source for c in out} == {"features_ip"}


def test_a_committed_list_with_no_measured_replacement_available_keeps_the_hole():
    """Six candidates, five listed, one rejected and the only spare unmeasurable: the list comes
    back short by one, with the hole named, rather than silently gaining an unverified row."""
    pool = candidates(6, per_run=1)
    committed = listed(pool[:5])
    doomed = committed[0].shot
    out, repl, _pending = select.reverify_flattop(
        committed, pool, quotas(n=5), seed=3,
        measure=lambda s: 0.4 if s == doomed else (None if s == pool[5].shot else 2.4),
    )
    assert len(out) == 4 and repl[0]["replacement"] is None


def test_the_re_verification_is_deterministic_and_comes_back_in_shot_order():
    pool = candidates(40, per_run=10)
    committed = listed(pool[:10])
    doomed = {committed[0].shot, committed[4].shot}
    runs = [
        select.reverify_flattop(
            committed, pool, quotas(n=10), seed=7,
            measure=lambda s: 0.4 if s in doomed else 2.4,
        )
        for _ in range(2)
    ]
    assert [c.shot for c in runs[0][0]] == [c.shot for c in runs[1][0]]
    assert [c.shot for c in runs[0][0]] == sorted(c.shot for c in runs[0][0])
    assert runs[0][1] == runs[1][1]


def test_the_summary_says_whether_the_list_is_final_and_which_store_verified_it():
    """Two invocations of the same command produce the same file with different meanings. The
    summary has to carry which one this was, and against which feature store -- the number of
    files and the newest mtime, so a list can be told apart from one verified an hour later."""
    got = candidates(5, per_run=1)
    by_dir = {"/data/ideate/frame_codes": 500, "/models/IGNITE/frame_codes": 10}
    store = {"n_featured": 878, "n_frame_codes": 507, "max_mtime": "2026-09-07T11:28:00+00:00",
             "frame_codes_by_dir": by_dir}
    s = select.summarize(
        n_candidates=9, reasons=Counter(), selected=got, quotas=quotas(n=5),
        replacements=[{"dropped": 190001, "dropped_flattop_s": 0.4, "theme": "rmp_elm",
                       "replacement": 190009, "replacement_theme": "rmp_elm",
                       "replacement_flattop_source": "features_ip"}],
        finalized=True, n_verified=4, store=store,
    )
    assert s["finalized"] is True and s["n_verified"] == 4 and s["n_dropped"] == 1
    assert s["n_featured"] == 878 and s["n_frame_codes"] == 507
    assert s["frame_codes_by_dir"] == by_dir
    assert s["feature_store"] == {"n_files": 878, "max_mtime": "2026-09-07T11:28:00+00:00"}
    text = select.format_summary(s)
    assert "finalized" in text
    # The 507 is a UNION over two directories (production store + shipped bundle), not the
    # production count; the line says so and shows the split, so nobody reads it as 507 encoded
    # production shots.
    assert "507" in text and "counted once" in text
    assert "500  /data/ideate/frame_codes" in text and "10  /models/IGNITE/frame_codes" in text
    # And what is and is not deterministic about the file two runs write: the rows, byte for
    # byte; not the header, which carries the clock and the store's fingerprint.
    assert "byte-identical" in text and "regenerated" in text
    assert "created" in text and "from_list" in text and "feature_store" in text


def test_an_unfinalized_summary_still_carries_the_keys_with_nothing_in_them():
    got = candidates(5, per_run=1)
    s = select.summarize(n_candidates=9, reasons=Counter(), selected=got, quotas=quotas(n=5))
    assert s["finalized"] is False and s["n_dropped"] == 0
    assert s["n_featured"] is None and s["feature_store"] == {"n_files": None, "max_mtime": None}
    assert s["n_frame_codes"] is None and s["frame_codes_by_dir"] is None
    # `n_verified` counts the measured rows when the caller does not say.
    assert s["n_verified"] == 0


def test_the_store_fingerprint_counts_the_files_and_takes_the_newest_mtime(tmp_path):
    import os

    feats, codes = tmp_path / "features", tmp_path / "frame_codes"
    _write_features(feats, 190001, 3.0)
    _write_features(feats, 190002, 3.0)
    codes.mkdir()
    (codes / "190001.pt").write_bytes(b"")
    os.utime(feats / "190001_features.h5", (1.0e9, 1.0e9))
    os.utime(feats / "190002_features.h5", (2.0e9, 2.0e9))
    got = select.store_fingerprint(feats, codes)
    assert got["n_featured"] == 2 and got["n_frame_codes"] == 1
    assert got["frame_codes_by_dir"] == {str(codes): 1}
    # The NEWER of the two, in UTC: 2e9 seconds after the epoch.
    assert got["max_mtime"].startswith("2033-05-18")


def test_the_fingerprint_of_a_store_that_is_not_there_is_empty(tmp_path):
    assert select.store_fingerprint(tmp_path / "nope", None) == {
        "n_featured": 0, "n_frame_codes": 0, "max_mtime": None, "frame_codes_by_dir": {}
    }

# ------------------------------------------------------------------------------ the document


def test_the_yaml_document_carries_every_key_the_spec_names():
    pool = candidates(60, per_run=20, groups=frozenset({"co2"}))
    got = select.diversify(pool, quotas(n=10), seed=99)
    summary = select.summarize(
        n_candidates=len(pool), reasons=Counter({"ip": 3}), selected=got, quotas=quotas(n=10)
    )
    doc = select.document(got, summary, name="recommender_v1", seed=99, n=10)
    assert doc["name"] == "recommender_v1"
    # The rule string names its own amendments, because the plan text is frozen and a reader of
    # the YAML has no other way to know which reading of §5.7 produced these 500 shots.
    assert doc["rule"].startswith("plan §5.7") and "abs(Ip)" in doc["rule"]
    assert doc["seed"] == 99 and doc["n"] == 10 and doc["created"]
    assert doc["hand_review"] == {"drop": [], "add": []}
    assert set(doc["shots"][0]) == {
        "shot",
        "run_id",
        "mpid",
        "year",
        "theme",
        "reason",
        "has_co2",
        "has_bes",
        "has_tangtv",
        "ip_sign",
        "flattop_s",
        "flattop_source",
    }
    assert doc["summary"]["n_eligible"] == 60
    assert doc["summary"]["failed"] == {"ip": 3}
    assert set(doc["summary"]) >= {"n_eligible", "failed", "theme", "year", "groups", "preferred"}


def test_the_document_round_trips_through_the_project_shot_list_loader(tmp_path):
    from shot_design import config

    got = select.diversify(candidates(60, per_run=20), quotas(n=10), seed=99)
    doc = select.document(got, select.summarize(n_candidates=60, reasons=Counter(), selected=got, quotas=quotas(n=10)), name="x", seed=99, n=10)
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    assert config.load_shot_list(path=p) == sorted(c.shot for c in got)


# ------------------------------------------------------------------------------ the CLI


@pytest.fixture
def selection_inputs(tmp_path):
    """A miniature corpus: 60 shots of text over five campaign years, and a census over them.

    Five years because the 40 % year ceiling is a real constraint on a small list -- a fixture in
    one year could only ever yield 0.4 n shots, which is the fixture's fault and not the rule's.
    """
    txt = tmp_path / "per_shot_txt"
    txt.mkdir()
    rows = []
    for i in range(60):
        shot = 190000 + i
        title = ["RMP ELM suppression", "QH-mode access", "NTM control", "Pedestal scan"][i % 4]
        run_id = f"{2021 + i % 5}{i // 4:04d}"
        txt.joinpath(f"shot_{shot}.txt").write_text(
            text_bundle(shot, row=GOOD_ROW, title=title, run_id=run_id), encoding="utf-8"
        )
        for g, span in (("mhr", 5.0), ("ece", 5.0), ("filterscopes", 5.0), ("co2", 5.0)):
            rows.append((shot, g, 0.0, span, True))
    parquet = tmp_path / "census.parquet"
    census_frame(rows).to_parquet(parquet, index=False)
    return txt, parquet


def select_argv(txt_dir, parquet, tmp_path, **over) -> list[str]:
    """The CLI arguments for one selection run, with every real-data path pointed at tmp_path so
    nothing here reads $LABELMAKER_ROOT, the corpus or the 616 MB logbook."""
    args = {
        "--n": "12",
        "--census": str(parquet),
        "--text-dir": str(txt_dir),
        "--features": str(tmp_path / "no-features"),
        "--frame-codes": str(tmp_path / "no-frame-codes"),
        "--logs": str(tmp_path / "no-logs.jsonl"),
        "--seed": "20260907",
        **over,
    }
    return ["corpus", "select", *[x for kv in args.items() for x in kv]]


def test_cli_select_writes_the_yaml_the_txt_list_and_prints_the_summary(
    selection_inputs, tmp_path, capsys
):
    txt_dir, parquet = selection_inputs
    out = tmp_path / "recommender_v1.yaml"
    txt_out = tmp_path / "recommender_v1.txt"
    argv = select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--txt-out": str(txt_out)})
    assert cli.main(argv) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["n"] == 12 and len(doc["shots"]) == 12
    assert txt_out.read_text(encoding="utf-8").split() == [
        str(e["shot"]) for e in doc["shots"]
    ]
    printed = capsys.readouterr().out
    assert "eligible" in printed and "theme" in printed


def test_cli_select_is_deterministic_across_two_runs(selection_inputs, tmp_path, capsys):
    """The claim the summary prints, held: the `shots:` rows two runs write are byte-identical -
    not merely the same shots in the same order - while the header (`created`, the store
    fingerprint) is regenerated. The critic measured exactly this on the finalized list."""
    txt_dir, parquet = selection_inputs
    outs, texts = [], []
    for i in (1, 2):
        out = tmp_path / f"v{i}.yaml"
        assert cli.main(select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out)})) == 0
        texts.append(out.read_text(encoding="utf-8"))
        outs.append(yaml.safe_load(texts[-1]))
    assert [e["shot"] for e in outs[0]["shots"]] == [e["shot"] for e in outs[1]["shots"]]
    rows = [t[t.index("shots:\n"):] for t in texts]
    assert rows[0] == rows[1]
    printed = capsys.readouterr().out
    assert "byte-identical" in printed and "regenerated" in printed


def test_cli_select_refuses_a_census_that_is_not_there(tmp_path, capsys):
    rc = cli.main(select_argv(tmp_path, tmp_path / "nope.parquet", tmp_path))
    assert rc == 1 and "nope.parquet" in capsys.readouterr().err


def test_cli_select_says_when_the_mpid_cap_could_not_be_applied(
    selection_inputs, tmp_path, capsys
):
    """A skipped cap has to be visible. Without the logbook there is no mpid for any shot, and a
    run that quietly dropped the <= 5 per mini-proposal rule would look exactly like one that
    applied it."""
    txt_dir, parquet = selection_inputs
    assert cli.main(select_argv(txt_dir, parquet, tmp_path, **{"--out": str(tmp_path / "v.yaml")})) == 0
    assert "per mpid cap is skipped" in capsys.readouterr().err


def test_cli_select_reads_mpid_out_of_a_logbook_when_there_is_one(
    selection_inputs, tmp_path, capsys
):
    txt_dir, parquet = selection_inputs
    logs = tmp_path / "logs.jsonl"
    logs.write_text(
        "".join(
            json.dumps({"shot": 190000 + i, "run": "r", "mpid": "2022-11-05"}) + "\n"
            for i in range(60)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "v.yaml"
    argv = select_argv(txt_dir, parquet, tmp_path, **{"--logs": str(logs), "--out": str(out)})
    assert cli.main(argv) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    # One mini-proposal for all 60 shots, so the cap alone decides the size of the list.
    assert len(doc["shots"]) == 5
    assert {e["mpid"] for e in doc["shots"]} == {"2022-11-05"}


def _write_features(dirpath, shot: int, flattop_s: float) -> None:
    """A labeler-shaped `<shot>_features.h5` whose `ip` group has a flat-top of `flattop_s`."""
    import h5py

    dirpath.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, flattop_s + 2.0, 0.025)
    ip = np.where((t >= 1.0) & (t <= 1.0 + flattop_s), 1.0e6, 1.0e5)
    with h5py.File(dirpath / f"{shot}_features.h5", "w") as f:
        g = f.create_group("ip")
        g["xdata"] = t
        g["ydata"] = ip


def test_cli_select_verify_flattop_marks_the_unmeasured_rows_and_lists_them(
    selection_inputs, tmp_path, capsys
):
    """The pending list is the point of the pass: it is the exact input to labeler's features
    stage, so the second invocation can measure what the first could only estimate."""
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    # A `pedestal` shot: the theme floors run alphabetically and pedestal is the first theme the
    # fixture carries, so this shot is selected -- and a shot with a feature file is `preferred`,
    # which puts it first inside its own year bucket. The point of the test is the two SOURCES.
    _write_features(feats, 190003, 3.0)
    out, pending = tmp_path / "v.yaml", tmp_path / "pending.txt"
    argv = select_argv(
        txt_dir, parquet, tmp_path,
        **{"--out": str(out), "--features": str(feats), "--pending-out": str(pending)},
    ) + ["--verify-flattop", "--allow-pending"]
    assert cli.main(argv) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    sources = Counter(e["flattop_source"] for e in doc["shots"])
    assert sources["pending"] == 11 and sources["features_ip"] == 1
    assert doc["summary"]["flattop_source"]["pending"] == 11
    assert doc["summary"]["replacements"] == []
    assert pending.read_text(encoding="utf-8").split() == [
        str(e["shot"]) for e in doc["shots"] if e["flattop_source"] == "pending"
    ]


def test_cli_select_refuses_a_list_with_pending_rows_unless_it_is_allowed(
    selection_inputs, tmp_path, capsys
):
    """A list whose rule (d) was never measured is a list that does not satisfy §5.7, and writing
    it under the same name as one that does is how an unverified list becomes the record."""
    txt_dir, parquet = selection_inputs
    out, pending = tmp_path / "v.yaml", tmp_path / "pending.txt"
    argv = select_argv(
        txt_dir, parquet, tmp_path, **{"--out": str(out), "--pending-out": str(pending)}
    ) + ["--verify-flattop"]
    assert cli.main(argv) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "12 selected shot(s) have no measured flat-top" in err and str(pending) in err
    # The pending list IS written: it is the work order that clears the refusal.
    assert len(pending.read_text(encoding="utf-8").split()) == 12


def test_cli_select_finalize_verifies_and_will_not_take_allow_pending(
    selection_inputs, tmp_path, capsys
):
    """`--finalize` is the second invocation: verify, and accept nothing less than a measured
    flat-top for every row."""
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    for i in range(60):
        _write_features(feats, 190000 + i, 3.0)
    out = tmp_path / "v.yaml"
    base = select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)})
    assert cli.main([*base, "--finalize"]) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert {e["flattop_source"] for e in doc["shots"]} == {"features_ip"}
    assert cli.main([*base, "--finalize", "--allow-pending"]) == 2
    assert "--finalize and --allow-pending" in capsys.readouterr().err


def test_cli_select_records_the_polarity_of_every_selected_shot(selection_inputs, tmp_path):
    txt_dir, parquet = selection_inputs
    out = tmp_path / "v.yaml"
    assert cli.main(select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out)})) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert {e["ip_sign"] for e in doc["shots"]} == {1}
    assert doc["summary"]["ip_sign"] == {"+1": 12}


# ------------------------------------------- the CLI, second invocation (`--finalize --from-list`)


def _write_list(argv, out) -> dict:
    """One selection run that writes `out`, and the document it wrote."""
    assert cli.main(argv) == 0
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def test_cli_select_from_list_re_verifies_the_committed_list_instead_of_re_selecting(
    selection_inputs, tmp_path
):
    """The finding this fix is for. Between the two invocations the feature store grows -- that is
    what the first one asked for -- and the enlarged store changes both `eligible` (a measured
    flat-top replaces the proxy) and the `preferred` tie-break, so a plain re-run selects a
    DIFFERENT list and the loop never converges. `--from-list` re-verifies the 500 that were
    committed."""
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    _write_features(feats, 190003, 3.0)
    committed_p = tmp_path / "v.yaml"
    base = select_argv(
        txt_dir, parquet, tmp_path, **{"--out": str(committed_p), "--features": str(feats)}
    )
    committed = _write_list(
        [*base, "--verify-flattop", "--allow-pending", "--pending-out", str(tmp_path / "p.txt")],
        committed_p,
    )
    for i in range(60):  # the features stage has now run over the pending list
        _write_features(feats, 190000 + i, 3.0)

    out = tmp_path / "final.yaml"
    doc = _write_list(
        [*select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)}),
         "--finalize", "--from-list", str(committed_p)],
        out,
    )
    assert [e["shot"] for e in doc["shots"]] == [e["shot"] for e in committed["shots"]]
    assert {e["flattop_source"] for e in doc["shots"]} == {"features_ip"}
    assert doc["summary"]["finalized"] is True
    assert doc["summary"]["n_verified"] == 12 and doc["summary"]["n_dropped"] == 0
    assert doc["summary"]["n_featured"] == 60
    # The store the second invocation saw is the enlarged one, and it is recorded as such: 60
    # files against the 1 the committed list was drawn with. That the enlarged store would have
    # produced a different SELECTION is what `test_the_committed_list_is_re_verified_and_never_
    # re_selected` pins (its five listed shots are not in the pool at all and all come back); on
    # this fixture every shot is interchangeable, so a re-selection happens to agree.
    assert committed["summary"]["n_featured"] == 1


def test_cli_select_from_list_refuses_while_a_listed_shot_has_no_features(
    selection_inputs, tmp_path, capsys
):
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    committed_p, pending = tmp_path / "v.yaml", tmp_path / "p.txt"
    base = select_argv(
        txt_dir, parquet, tmp_path, **{"--out": str(committed_p), "--features": str(feats)}
    )
    committed = _write_list(
        [*base, "--verify-flattop", "--allow-pending", "--pending-out", str(pending)], committed_p
    )
    shots = [e["shot"] for e in committed["shots"]]
    for shot in shots[:-1]:  # every listed shot but one now has a feature file
        _write_features(feats, shot, 3.0)

    out = tmp_path / "final.yaml"
    argv = [
        *select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)}),
        "--finalize", "--from-list", str(committed_p), "--pending-out", str(pending),
    ]
    assert cli.main(argv) == 1
    assert not out.exists()
    assert "1 listed shot(s) have no measured flat-top" in capsys.readouterr().err
    # The pending file is rewritten for exactly the shots still waiting, not for the original 352.
    assert pending.read_text(encoding="utf-8").split() == [str(shots[-1])]


def test_cli_select_from_list_replaces_a_listed_shot_the_measurement_rejects(
    selection_inputs, tmp_path
):
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    committed_p = tmp_path / "v.yaml"
    base = select_argv(
        txt_dir, parquet, tmp_path, **{"--out": str(committed_p), "--features": str(feats)}
    )
    committed = _write_list(
        [*base, "--verify-flattop", "--allow-pending", "--pending-out", str(tmp_path / "p.txt")],
        committed_p,
    )
    doomed = committed["shots"][0]["shot"]
    for i in range(60):
        _write_features(feats, 190000 + i, 0.4 if 190000 + i == doomed else 3.0)

    out = tmp_path / "final.yaml"
    doc = _write_list(
        [*select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)}),
         "--finalize", "--from-list", str(committed_p)],
        out,
    )
    shots = [e["shot"] for e in doc["shots"]]
    assert len(shots) == 12 and doomed not in shots
    repl = doc["summary"]["replacements"]
    assert len(repl) == 1 and repl[0]["dropped"] == doomed and repl[0]["replacement"] in shots
    assert doc["summary"]["n_dropped"] == 1 and doc["summary"]["finalized"] is True


def test_cli_select_from_list_carries_the_hand_review_block_forward(selection_inputs, tmp_path):
    """`hand_review` is the human gate on the list. A second invocation that reset it to empty
    would throw away a reviewer's decisions without saying so."""
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    for i in range(60):
        _write_features(feats, 190000 + i, 3.0)
    committed_p = tmp_path / "v.yaml"
    base = select_argv(
        txt_dir, parquet, tmp_path, **{"--out": str(committed_p), "--features": str(feats)}
    )
    committed = _write_list([*base, "--finalize"], committed_p)
    committed["hand_review"] = {"drop": [190003], "add": [190999]}
    committed_p.write_text(yaml.safe_dump(committed, sort_keys=False), encoding="utf-8")

    out = tmp_path / "final.yaml"
    doc = _write_list(
        [*select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)}),
         "--finalize", "--from-list", str(committed_p)],
        out,
    )
    assert doc["hand_review"] == {"drop": [190003], "add": [190999]}


def test_cli_select_from_list_takes_the_seed_the_list_recorded(selection_inputs, tmp_path):
    """The replacements have to be reproducible from the list itself, so the seed comes off the
    document unless the caller overrides it."""
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    for i in range(60):
        _write_features(feats, 190000 + i, 3.0)
    committed_p = tmp_path / "v.yaml"
    committed = _write_list(
        [*select_argv(txt_dir, parquet, tmp_path,
                      **{"--out": str(committed_p), "--features": str(feats), "--seed": "4242"}),
         "--finalize"],
        committed_p,
    )
    assert committed["seed"] == 4242
    out = tmp_path / "final.yaml"
    argv = ["corpus", "select", "--n", "12", "--census", str(parquet), "--text-dir", str(txt_dir),
            "--features", str(feats), "--frame-codes", str(tmp_path / "no-frame-codes"),
            "--logs", str(tmp_path / "no-logs.jsonl"), "--out", str(out),
            "--finalize", "--from-list", str(committed_p)]
    doc = _write_list(argv, out)
    assert doc["seed"] == 4242


def test_cli_select_records_a_reversed_current_shot_end_to_end(tmp_path):
    """`abs(Ip)` all the way through the CLI: a run day recorded at negative `IP-(MA)` is
    selected, and every row of it says so."""
    txt = tmp_path / "per_shot_txt"
    txt.mkdir()
    rows = []
    for i in range(20):
        shot = 190000 + i
        row = {**GOOD_ROW, "IP-(MA)": "-1.10" if i % 2 else "1.45"}
        txt.joinpath(f"shot_{shot}.txt").write_text(
            text_bundle(shot, row=row, title="QH-mode access", run_id=f"{2021 + i % 5}{i // 4:04d}"),
            encoding="utf-8",
        )
        for g, span in (("mhr", 5.0), ("ece", 5.0), ("filterscopes", 5.0)):
            rows.append((shot, g, 0.0, span, True))
    parquet = tmp_path / "census.parquet"
    census_frame(rows).to_parquet(parquet, index=False)
    out = tmp_path / "v.yaml"
    doc = _write_list(select_argv(txt, parquet, tmp_path, **{"--out": str(out), "--n": "10"}), out)
    signs = Counter(e["ip_sign"] for e in doc["shots"])
    assert signs[-1] > 0 and signs[1] > 0
    assert doc["summary"]["ip_sign"]["-1"] == signs[-1]


def test_the_census_columns_this_module_reads_are_the_ones_the_census_writes():
    assert {"shot", "group", "present", "t0_s", "t1_s"} <= set(census.COLUMNS)
    assert isinstance(census_frame([(1, "mhr", 0.0, 1.0, True)]), pd.DataFrame)


def test_the_two_commands_ask_one_definition_where_frame_codes_live(paths):
    """`build` counted <data_root>/frame_codes AND <models_dir>/IGNITE/frame_codes; `corpus
    select` was handed only the second, so the two commands could disagree about whether the same
    shot has codes. One function names the locations and both call sites take it."""
    from shot_design.shotdb import build

    (paths.data_root / "frame_codes").mkdir(parents=True, exist_ok=True)
    (paths.data_root / "frame_codes" / "190123.pt").write_bytes(b"")
    dirs = build.frame_codes_dirs(paths)
    assert build.frame_codes_path(190123, paths) is not None
    assert select.preferred_shots(features_dir=None, frame_codes_dirs=dirs) == {190123}
    assert select.store_fingerprint(None, dirs)["n_frame_codes"] == 1


# ------------------------------ re-verification: the preferred cap must not close the door


def preferred_list(cands) -> list[select.Candidate]:
    """`cands` as a committed list comes back AFTER the features stage has run over it.

    The difference from `listed()` is the one fact the document does not carry: by the time a
    list is finalized every listed shot has a feature file, so `preferred` is true of all of
    them. That is the state `--finalize --from-list` actually runs in.
    """
    doc = select.document(
        [select.replace(c, reason="fill") for c in cands],
        select.summarize(
            n_candidates=len(cands), reasons=Counter(), selected=cands, quotas=quotas()
        ),
        name="x",
        seed=3,
        n=len(cands),
    )
    return select.candidates_from_rows(doc["shots"], preferred=[c.shot for c in cands])


def test_a_replacement_is_reachable_when_every_listed_shot_is_preferred():
    """The latent Important the I5 re-review left for I7.

    On a re-verification the 500 kept rows are forced in, and forcing counted each of them
    against the preferred cap (150) -- so `_preferred` stood at 500 before a single replacement
    was considered. Every candidate a re-verification may draw is `require_measured`, and a
    measured shot is one with a feature file, which is exactly what makes it `preferred`: the
    cap therefore refused every possible replacement. A dropped shot yielded a 499-row list
    that still called itself finalized, and exited 0.
    """
    pool = candidates(501, per_run=10, preferred=True)
    committed = preferred_list(pool[:500])
    doomed = committed[0].shot
    out, repl, pending = select.reverify_flattop(
        committed, pool, quotas(n=500, per_run=10, preferred_cap=150), seed=3,
        measure=lambda s: 0.4 if s == doomed else 2.4,
    )
    assert pending == []
    assert len(repl) == 1 and repl[0]["dropped"] == doomed
    assert repl[0]["replacement"] == pool[500].shot  # the one shot not on the list
    assert len(out) == 500 and doomed not in {c.shot for c in out}


def test_a_fresh_selection_still_caps_the_preferred_shots():
    """The cap is only vacuous where "measured" already implies "featured". Pass one -- a fresh
    `diversify` over a pool where most shots have no features -- still enforces it, and this
    test is here so the fix above cannot quietly turn it off everywhere."""
    pool = candidates(100, per_run=10, preferred=True) + candidates(
        100, start=195000, per_run=10, preferred=False
    )
    got = select.diversify(pool, quotas(n=50, per_run=10, preferred_cap=5), seed=3)
    assert sum(c.preferred for c in got) == 5


def test_the_summary_records_which_list_a_re_verification_verified():
    """`finalized: true` is a claim about a specific document. Which one it was is not derivable
    from anything else in the file, and a summary that says a list is final without saying what
    it was re-measured from cannot be checked."""
    got = candidates(5, per_run=1)
    s = select.summarize(
        n_candidates=9, reasons=Counter(), selected=got, quotas=quotas(n=5),
        finalized=True, from_list="configs/shot_design/shot_lists/recommender_v1.yaml",
    )
    assert s["from_list"] == "configs/shot_design/shot_lists/recommender_v1.yaml"
    assert "recommender_v1.yaml" in select.format_summary(s)


def test_a_summary_that_verified_no_list_says_so_with_none():
    s = select.summarize(n_candidates=9, reasons=Counter(), selected=[], quotas=quotas(n=5))
    assert s["from_list"] is None and s["finalized"] is False


# ------------------------- the CLI: what may call itself finalized, and how short it may be


def test_cli_select_finalize_without_from_list_does_not_call_the_list_final(
    selection_inputs, tmp_path
):
    """`--finalize` alone re-SELECTS: it runs `eligible()` and `diversify()` over today's store,
    which is the non-convergence `--from-list` exists to stop. Such a run may still write a
    list -- every row of it is measured -- but it is a first invocation, and stamping it
    `finalized: true` would make the un-re-verified list indistinguishable from the verified one.
    """
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    for i in range(60):
        _write_features(feats, 190000 + i, 3.0)
    out = tmp_path / "v.yaml"
    argv = select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)})
    assert cli.main([*argv, "--finalize"]) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["summary"]["finalized"] is False and doc["summary"]["from_list"] is None
    assert {e["flattop_source"] for e in doc["shots"]} == {"features_ip"}


def test_cli_select_finalize_refuses_to_re_select_over_an_existing_list(
    selection_inputs, tmp_path, capsys
):
    """The dangerous case: the file it would overwrite is the committed list. Re-selecting over
    it silently replaces the eligibility snapshot with a different 500."""
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    for i in range(60):
        _write_features(feats, 190000 + i, 3.0)
    out = tmp_path / "v.yaml"
    argv = select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)})
    assert cli.main([*argv, "--finalize"]) == 0
    before = out.read_text(encoding="utf-8")
    assert cli.main([*argv, "--finalize"]) == 2
    assert "--from-list" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == before  # not touched


def test_cli_select_from_list_re_verifies_a_list_that_predates_the_from_list_key(
    selection_inputs, tmp_path
):
    """The committed `recommender_v1.yaml` was stamped `finalized: true` before the invariant
    that only a `--from-list` run may say so, so it carries no `from_list` key. Re-verifying it
    must still work: what `--from-list` reads out of a document is its `shots` (and its `seed`),
    and the summary is the PREVIOUS run's provenance -- an output of that run, never an input to
    this one. A re-verification that demanded the key would refuse every list written so far.
    """
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    for i in range(60):
        _write_features(feats, 190000 + i, 3.0)
    committed_p = tmp_path / "legacy.yaml"
    base = select_argv(
        txt_dir, parquet, tmp_path, **{"--out": str(committed_p), "--features": str(feats)}
    )
    committed = _write_list([*base, "--finalize"], committed_p)
    doc = yaml.safe_load(committed_p.read_text(encoding="utf-8"))
    doc["summary"].pop("from_list")  # what a list written before this task looks like
    doc["summary"]["finalized"] = True
    committed_p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    out = tmp_path / "final.yaml"
    argv = [
        *select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)}),
        "--finalize", "--from-list", str(committed_p),
    ]
    assert cli.main(argv) == 0
    new = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert [e["shot"] for e in new["shots"]] == [e["shot"] for e in committed["shots"]]
    # And the re-verified copy is the one that carries the provenance the legacy list lacked.
    assert new["summary"]["finalized"] is True
    assert new["summary"]["from_list"] == str(committed_p)


def test_cli_select_from_list_refuses_a_finalized_list_shorter_than_it_asked_for(
    selection_inputs, tmp_path, capsys
):
    """A hole nothing measured can fill leaves the list short. Writing it anyway published a
    499-row `recommender_v1` stamped `finalized: true` and exited 0, and every downstream count
    would have been read as 500."""
    txt_dir, parquet = selection_inputs
    feats = tmp_path / "features"
    committed_p = tmp_path / "v.yaml"
    base = select_argv(
        txt_dir, parquet, tmp_path, **{"--out": str(committed_p), "--features": str(feats)}
    )
    for i in range(60):
        _write_features(feats, 190000 + i, 3.0)
    committed = _write_list([*base, "--finalize"], committed_p)
    doomed = committed["shots"][0]["shot"]
    # Only the listed shots keep a feature file, and the doomed one now measures short: there is
    # nothing measured left to replace it with.
    kept = {e["shot"] for e in committed["shots"]}
    for i in range(60):
        shot = 190000 + i
        if shot not in kept:
            (feats / f"{shot}_features.h5").unlink()
    _write_features(feats, doomed, 0.4)

    out = tmp_path / "final.yaml"
    argv = [
        *select_argv(txt_dir, parquet, tmp_path, **{"--out": str(out), "--features": str(feats)}),
        "--finalize", "--from-list", str(committed_p),
    ]
    assert cli.main(argv) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "11 of 12" in err


def test_finalize_txt_out_applies_review_and_names_the_effective_changes(
    selection_inputs, tmp_path, capsys,
):
    from shot_design import config

    txt_dir, parquet = selection_inputs
    features = tmp_path / "features"
    for shot in range(190000, 190060):
        _write_features(features, shot, 3.0)
    original = tmp_path / "original.yaml"
    doc = _write_list([
        *select_argv(txt_dir, parquet, tmp_path,
                     **{"--out": str(original), "--features": str(features)}),
        "--finalize",
    ], original)
    drop, readd = (row["shot"] for row in doc["shots"][:2])
    doc["hand_review"] = {
        "drop": [{"shot": drop}, {"shot": readd}],
        "add": [{"shot": 190999}, {"shot": readd}, {"shot": 190998}],
    }
    original.write_text(yaml.safe_dump(doc))
    out, txt = tmp_path / "final.yaml", tmp_path / "effective.txt"
    capsys.readouterr()
    assert cli.main([
        *select_argv(txt_dir, parquet, tmp_path, **{
            "--out": str(out), "--txt-out": str(txt), "--features": str(features),
        }), "--finalize", "--from-list", str(original),
    ]) == 0
    emitted = yaml.safe_load(out.read_text())
    assert emitted["shots"] == doc["shots"]
    assert emitted["hand_review"] == doc["hand_review"]
    effective = [int(line) for line in txt.read_text().splitlines()]
    assert effective == config.load_shot_list("unused", path=out)
    assert drop not in effective and readd in effective
    assert effective == sorted(set(effective))
    assert f"hand_review: +2 (190998 190999), -1 ({drop})" in capsys.readouterr().out
