"""Ported from shot-recommender-system (shotrec) @565d548."""

import json
import logging

import numpy as np
import pytest

from ideate.schema import HumanTier, LogEntry
from ideate.shotdb import text

LOG = """### [SESSION_LEADER] lizeyu 2024-08-29 10:54:43
Preshot:
Step 2, Ip 1.3MA
- - - - - - - - - -
requested ip: 1.28 MA, btor: 2.05 T, pnbi: 1 MW, pech: 0 MW, density: 2.6e13

### [PCS] pcsops 2024-08-29 10:59:53
PCS CHANGES:
timcon: changes made

### [CHIEF_OPERATOR] byrnep 2024-08-29 11:05:31
200443 Plasma good. IPMin FF.
"""


def test_parse_log_entries_roles_authors_times():
    entries = text.parse_log_entries(LOG)
    assert [e.role for e in entries] == ["SESSION_LEADER", "PCS", "CHIEF_OPERATOR"]
    assert entries[0].author == "lizeyu" and entries[0].time == "10:54"
    assert entries[2].text.startswith("200443 Plasma good")
    assert text.parse_log_entries(None) == []


def test_parse_requested():
    assert text.parse_requested(LOG) == {
        "ip_MA": 1.28,
        "bt_T": 2.05,
        "pnbi_MW": 1.0,
        "pech_MW": 0.0,
        "density_cm3": 2.6e13,
    }
    assert (
        text.parse_requested("requested ip: 0 MA, btor: 0 T, pnbi: 0 MW, pech: 0 MW") == {}
    )  # unfilled template
    assert text.parse_requested("no template here") == {}


def test_fault_strings_ignore_normal_ip_min_termination():
    assert text.fault_strings("13:37 150066 Plasma Shot - Dud Trip @ 1.9 sec") == ["dud trip"]
    assert text.fault_strings("14:10 150069 Plasma Shot - IP min") == []
    assert text.fault_strings("Locked mode at 2.1 s, disrupted", "PCS first fault") == [
        "locked mode",
        "disrupted",
        "pcs first fault",
    ]


def test_verdict_polarity():
    good = text.parse_log_entries(LOG)
    assert text.verdict(good) == "good"
    bad = text.parse_log_entries(
        "### [SESSION_LEADER] x 2024-08-29 10:00:00\nBummer, lost it vertically at 1.2 s.\n"
    )
    assert text.verdict(bad) == "bad"
    rampdown = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] x 2024-08-29 10:00:00\nNice shot, disrupts in rampdown as usual.\n"
    )
    assert text.verdict(rampdown) == "good"  # bare "disrupt" is routine ramp-down language
    assert (
        text.verdict(
            text.parse_log_entries("### [PCS] pcsops 2024-08-29 10:00:00\nperiodic action\n")
        )
        == "unknown"
    )
    assert text.verdict([], extra="no disruption, ran through") == "good"


def test_parse_shot_range():
    assert text.parse_shot_range("150041 - 150080") == (150041, 150080)
    assert text.parse_shot_range("161172-161200") == (161172, 161200)
    assert text.parse_shot_range("") is None


# Real logs.jsonl corpus check (task-8-report.md), each pinned to the exact real shot that showed
# the pattern miss. These are verbatim excerpts (a real substring, or a single real entry cut at
# its natural "### [ROLE] ..." boundary), not invented text.


def test_parse_requested_handles_real_newline_separated_variant():
    # Shot 152721: a run of shots sharing one mini-proposal template repeats "requested" on its
    # own line per field with no comma, instead of the usual one comma-joined line.
    real = (
        "requested ip: 1.1 MA\nrequested btor: 2.0 T\nrequested pnbi: 1 MW\nrequested pech: 1 MW\n"
    )
    assert text.parse_requested(real) == {
        "ip_MA": 1.1,
        "bt_T": 2.0,
        "pnbi_MW": 1.0,
        "pech_MW": 1.0,
    }


def test_verdict_real_corpus_tripped_conjugation():
    # Shot 150555's chief operator: "tripped" (past tense) is missed by a weak-neg list that only
    # had bare "trip", so this shot's only concrete negative signal fell through to "unknown".
    entries = text.parse_log_entries(
        "### [CHIEF_OPERATOR] taylorpl 2012-08-21 09:26:56\n"
        "9:24 Plasma shot short; B module 2 tripped early at ~-1.5s due to delta-com\nfailure.\n"
    )
    assert text.verdict(entries) == "bad"


def test_verdict_real_corpus_reversed_bad_shot():
    # Shot 160309's chief operator writes "shot: Bad" (reversed from the lexicon's "bad shot"),
    # the project's one free outcome label for this shot -- it must not read as "unknown".
    entries = text.parse_log_entries(
        "### [CHIEF_OPERATOR] holtrop 2014-11-18 17:01:34\n"
        "1640 Plasma shot: Bad. D2 opened up its breaker.\n"
    )
    assert text.verdict(entries) == "bad"


def test_verdict_clean_maintenance_shot_stays_unknown():
    # CLAUDE.md: "mode" and "clean" were already removed once from a different keyword list for
    # false-matching plasma vocabulary. Confirmed the same failure mode still live here: shot
    # 154319's session leader calls a zero-plasma wall-conditioning shot a "clean up shot", which
    # must not read as a positive physics verdict just because it contains the word "clean".
    entries = text.parse_log_entries(
        "### [SESSION_LEADER] x 2013-08-13 10:00:00\n"
        "Preshot:\nThird plasma clean up shot\n"
        "requested ip: 0 MA, btor: 0 T, pnbi: 0 MW, pech: 0 MW\n"
    )
    assert text.verdict(entries) == "unknown"


def test_verdict_h_mode_is_not_a_fault():
    # The other removed keyword, "mode": shot 150430's physics operator mentions H-mode routinely
    # (not the *locked*-mode fault), which must not push the verdict toward "bad".
    entries = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] x 2014-01-01 10:00:00\n"
        "Increase FF power to 5.5 MW to get H-mode earlier.\n"
    )
    assert text.verdict(entries) == "unknown"


def test_fault_strings_and_verdict_real_corpus_first_fault_prefix():
    # Shot 153002's chief operator: "IPmin first fault" -- the dominant real prefix -- was missed
    # by a fault list hardcoded to "pcs first fault" alone (seen once in a dense sample versus 70+
    # non-"pcs" prefixes). Also exercises "shot: Bad" and "tripped" together on one real string.
    co_text = "1422 Plasma shot: Bad. BPS tripped early due to low MG2 RPM. IPmin first fault."
    assert text.fault_strings(co_text) == ["ipmin first fault"]
    entries = text.parse_log_entries(
        f"### [CHIEF_OPERATOR] holtrop 2013-05-08 17:52:22\n{co_text}\n"
    )
    assert text.verdict(entries) == "bad"


# Fix wave 1 (task-8-report.md, "Fix wave 1" section): the negation guard applied to only two of
# the five lexicon patterns feeding verdict(). Each test below pins one of the four fixes to the
# real sentence(s) the fix-wave brief quoted, plus two more real sentences found while measuring
# the fix on the full corpus that a naive "guard everything" implementation gets wrong.


def test_verdict_real_corpus_negated_bad_shot_is_not_bad():
    # Fix 1: "bad shot"/"shot bad" were matched unguarded. Shot 194237's physics operator "Not a
    # bad shot, considering." and shot 133446's session leader "Not bad shot." both read as "bad"
    # before this fix despite the operator's own explicit assessment. Neither sentence has any
    # other pos/neg cue, so a correctly-cancelled match must land on "unknown" -- not "good" (the
    # cancelled-negative contract: no signal, not the opposite signal).
    considering = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] hyatt 2023-01-31 12:43:24\nNot a bad shot, considering.\n"
    )
    assert text.verdict(considering) == "unknown"
    terser = text.parse_log_entries(
        "### [SESSION_LEADER] buttery 2008-06-17 09:51:29\nNot bad shot.\n"
    )
    assert text.verdict(terser) == "unknown"


def test_verdict_real_corpus_negated_positive_yields_no_signal_not_good():
    # Fix 2, the dangerous direction: _POS had no guard at all, so a negated positive scored as a
    # point *for* "good". Shot 155244's session leader "Didnt help, Not ok." and shot 164109's
    # physics operator "SPA3 is not OK, and hasn't been." both read as "good" before this fix, from
    # the cancelled "ok"/"OK" alone. This is also the required proof that a cancelled negative
    # yields no signal rather than the opposite signal: with nothing else in either sentence, the
    # correct result is "unknown", never "good".
    didnt_help = text.parse_log_entries(
        "### [SESSION_LEADER] podestam 2013-10-04 13:55:22\nDidnt help, Not ok.\n"
    )
    assert text.verdict(didnt_help) == "unknown"
    spa3 = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] hyatt 2015-11-19 15:38:11\nSPA3 is not OK, and hasn't been.\n"
    )
    assert text.verdict(spa3) == "unknown"


def test_verdict_real_corpus_absent_trip_is_not_bad():
    # Fix 3: _WEAK_NEG had no guard, so an explicitly *absent* fault counted as evidence of one.
    # Real phrasings quoted verbatim in the fix-wave brief. "No trip, good 5s shot" also exercises
    # the guard's clause boundary: the comma must let "No" cancel "trip" without also cancelling
    # the independent "good" verdict right after it (see test_verdict_real_corpus_dash_separates_
    # clauses below for the same shape without a comma).
    assert (
        text.verdict(
            text.parse_log_entries("### [CHIEF_OPERATOR] x 2024-01-01 10:00:00\nNo trips.\n")
        )
        == "unknown"
    )
    assert (
        text.verdict(
            text.parse_log_entries(
                "### [CHIEF_OPERATOR] x 2024-01-01 10:00:00\nNo trip, good 5s shot\n"
            )
        )
        == "good"
    )
    assert (
        text.verdict(
            text.parse_log_entries("### [SESSION_LEADER] x 2024-01-01 10:00:00\ndid not trip\n")
        )
        == "unknown"
    )
    assert (
        text.verdict(
            text.parse_log_entries("### [SESSION_LEADER] x 2024-01-01 10:00:00\nnever tripped\n")
        )
        == "unknown"
    )
    assert (
        text.verdict(
            text.parse_log_entries(
                "### [PHYSICS_OPERATOR] x 2024-01-01 10:00:00\nto avoid tripping\n"
            )
        )
        == "unknown"
    )


def test_fault_strings_table_header_is_not_a_fault():
    # Fix 4: generalizing the literal "pcs first fault" to "\w+ first fault" also matched a routine
    # gyrotron status table's adjacent column headers. Real excerpt from shot 198785's RF entry
    # (tab-separated, as the real table is) -- "Blocks" and "First Fault" are column names, not a
    # system name followed by a real fault report, and must not produce a fault string.
    table_header = (
        "Gyrotron\tPulse Delay [ms]\tPulse Request [ms]\tActual Pulse [ms]\tBlocks\tFirst Fault\n"
        "HAN\t1800\t2200\t106\tPCS Mod, Fwd RF, Body I\tBody OI"
    )
    assert text.fault_strings(table_header) == []
    # The real, single-space-separated "first fault" phrasing this generalization exists for (shot
    # 150082) must keep matching.
    assert text.fault_strings("17:40  Plasma Shot - PCS first fault @ 4.8 sec") == [
        "pcs first fault"
    ]


def test_verdict_real_corpus_causal_no_plasma_stays_unguarded():
    # Guarding a self-negating phrase like "no plasma" against a *preceding* negation is the wrong
    # fix: chief operators routinely write causal "no CAUSE, no EFFECT" chains, and shot 140183's
    # chief operator "No Gas = No Plasma" is a genuinely zero-plasma shot -- the first "No" (about
    # gas) has nothing to do with the "No Plasma" that follows and must not cancel it. A version of
    # this fix that guarded _STRONG_NEG_UNCONDITIONAL's alternatives too got this wrong for 8 of 342
    # real "no plasma" matches during this fix wave's own verification.
    entries = text.parse_log_entries(
        "### [CHIEF_OPERATOR] chamberl 2009-11-18 08:42:03\n"
        "08:40  Reference Shot Attempt - No Gas = No Plasma\n"
    )
    assert text.verdict(entries) == "bad"


def test_verdict_real_corpus_dash_separates_clauses():
    # Fix 3 (task-8-report.md, "Fix wave 2" section): this test previously quoted shot 163826's
    # physics operator (hansonjm) as writing "I-coils do not trip - good Plasma; ok." -- invented.
    # The real entry ends exactly at "...I-coils do not trip - good"; "Plasma" is not in the record
    # at all, and "ok." belongs to the *next* entry, a different author's (chief operator leer's
    # "Plasma; ok.  no ECH due to..."). Corrected to the real, verbatim text, which still exercises
    # the same mechanism: a bare dash joins two short clauses the same way a comma does in this
    # corpus's operator shorthand, so "not" must cancel "trip" (guarded, same clause) without
    # reaching across the dash to also cancel the independent "good" -- without the dash in
    # _CLAUSE_BREAK, both "trip" and "good" are cancelled and this reads as "unknown", not "good".
    entries = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] hansonjm 2015-09-15 11:16:30\nI-coils do not trip - good\n"
    )
    assert text.verdict(entries) == "good"


def test_verdict_real_corpus_line_wrap_is_not_a_clause_break():
    # A bare newline is just a line wrap in this corpus's raw text, not a new clause. Shot 164892's
    # session leader writes "the shot does not\ndevelop a locked mode" wrapped mid-sentence;
    # treating "\n" as a clause boundary stops "not" from reaching "locked mode", wrongly
    # un-cancelling a fault that was genuinely and explicitly negated (confirmed live during this
    # fix wave's own verification, before "\n" was excluded from _CLAUSE_BREAK).
    entries = text.parse_log_entries(
        "### [SESSION_LEADER] burrell 2016-01-12 08:52:16\n"
        "however, the shot does not\ndevelop a locked mode.\n"
    )
    assert text.verdict(entries) == "unknown"


# Fix wave 2 (task-8-report.md, "Fix wave 2" section): three lexicon alternatives were left in the
# unconditional groups despite not spelling their own negation, _guarded_disrupt() never got fix
# wave 1's clause boundary, and fault_strings() had no negation guard at all.


def test_verdict_real_corpus_negated_lost_vertically_is_not_bad():
    # Item 1: "lost it/the shot/vertically" sat in the unconditional strong-negative group on the
    # theory that it spells its own negation. It does not -- it is a bare assertion, and the corpus
    # negates it directly. Shot 204672's physics operator writes "Not lost vertically." while the
    # session leader calls the same shot "very good shot"; before this fix the phrase forced "bad".
    lost = "### [PHYSICS_OPERATOR] hyatt 2025-05-14 15:24:12\nNot lost vertically.\n"
    assert text.verdict(text.parse_log_entries(lost)) == "unknown"
    both = "### [SESSION_LEADER] x 2025-05-14 15:20:00\nPost:  very good shot.\n\n" + lost
    assert text.verdict(text.parse_log_entries(both)) == "good"


def test_verdict_real_corpus_avoided_overcurrent_and_interlock_are_not_faults():
    # Item 2: "overcurrent" and "interlock" are bare nouns, and the corpus's dominant use of both is
    # a *plan to avoid* the fault, not a report of one (35 negated "overcurrent" and 7 negated
    # "interlock" occurrences). Shot 159328's "to avoid overcurrent" was one phantom negative
    # against the chief operator's real "Plasma shot ok", forcing "mixed" instead of "good".
    entries = text.parse_log_entries(
        "### [SESSION_LEADER] x 2014-10-14 08:55:00\n"
        "Reduce AA9 amplitude by 10% to avoid overcurrent\n\n"
        "### [CHIEF_OPERATOR] x 2014-10-14 09:01:00\n9:01 Plasma shot ok.\n"
    )
    assert text.verdict(entries) == "good"
    # Shot 198591: the only negative cue in the record is an explicitly *absent* interlock.
    interlock = text.parse_log_entries(
        "### [SESSION_LEADER] x 2024-05-01 10:00:00\n"
        "Repeat 1MW ECH preionization shot. Making sure ECH has no interlock to prevent\n"
        "early injection.\n"
    )
    assert text.verdict(interlock) == "unknown"


def test_verdict_real_corpus_negation_does_not_cross_a_clause_into_disrupt():
    # Item 3: _guarded_disrupt() kept the old unbounded 16-char backward window after fix wave 1
    # gave _guarded() a clause boundary, so a negation about a *different* subject cancelled a real
    # disruption. Shot 150744's "No H-mode, disrupts." is the shape: the "No" is about H-mode.
    entries = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] x 2012-08-27 16:10:00\nNo H-mode, disrupts.\n"
    )
    assert text.verdict(entries) == "bad"


def test_verdict_real_corpus_disruption_avoided_is_still_neutralized():
    # The counterweight to the test above: clause-bounding the backward scan alone would make shot
    # 159422's "Disruption avoided for the entire discharge" a false positive, since its
    # neutralizing cue *follows* the word. A narrow passive "avoided/prevented/averted" follower
    # (_DISRUPT_AVERTED) handles it -- deliberately narrow, not the full negation vocabulary,
    # because "disrupt" is routinely followed by a negation about something else. Shot 192196's
    # "disrupts without catching" is a real disruption and must still count.
    avoided = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] x 2014-10-01 12:00:00\n"
        "Post: LM suppressed. H-mode recovered or never lost. Disruption avoided for the\n"
        "entire discharge.\n"
    )
    assert text.verdict(avoided) == "unknown"
    without = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] x 2022-09-01 12:00:00\n- Earlier pellet, disrupts without catching\n"
    )
    assert text.verdict(without) == "bad"


def test_fault_strings_negated_fault_is_not_reported():
    # Item 4: fault_strings() had no negation guard, so 541 corpus records reported a fault whose
    # every mention in the record was explicitly *absent*. Task 10 uses this as the primary failure
    # detector for curation, so each of these would have been a wrongly-labelled "failed" shot.
    assert (
        text.fault_strings("Outcome: no beams at all. No locked mode. ECCD raises the betaN") == []
    )
    assert text.fault_strings("Large n=1 at 3.6. No DUD trip.") == []
    assert text.fault_strings("Plasma shot ok, pellet no runaways") == []
    assert text.fault_strings("Last shot appears to have not generated a locked mode.") == []
    # ... but "no plasma" spells its own negation, so a preceding "No" in a causal chain must not
    # cancel it (shot 140183, the same carve-out the verdict lexicon makes).
    assert text.fault_strings("08:40  Reference Shot Attempt - No Gas = No Plasma") == ["no plasma"]


def test_verdict_real_corpus_disruption_across_a_sentence_boundary_still_counts():
    # Fix 1 (task-8-report.md, "Fix wave 2" section): _guarded_disrupt() picked up fix wave 1's
    # clause-bounded window (test_verdict_real_corpus_negation_does_not_cross_a_clause_into_disrupt
    # above already pins the comma-joined shape, shot 150744). This pins the period-joined shape
    # from the brief's third named shot, isolated to just the sentence that carries it so the
    # assertion actually exercises the guard rather than being carried by an unrelated fault
    # elsewhere in the shot (158292's PHYSICS_OPERATOR entry separately mentions "VDE", which alone
    # forces "bad" regardless of this mechanism -- left out here on purpose). Shot 158292's session
    # leader writes "...the scopes in the control room did not trigger. Disruption did take place
    # at 2000ms.": the period must stop "not" from reaching the real, separate "Disruption".
    entries = text.parse_log_entries(
        "### [SESSION_LEADER] degrassi 2014-07-11 10:50:57\n"
        "Shot ran, but the scopes in the control room did not trigger. Disruption did take place "
        "at 2000ms.\n"
    )
    assert text.verdict(entries) == "bad"


def test_verdict_real_corpus_trip_window_widened_with_causal_override():
    # Fix 2 (task-8-report.md, "Fix wave 2" section): bare "trip"'s negation guard used the same
    # 16-char cap as every other lexicon, which misses a same-clause negation further back than
    # that. Shot 199933's entire qualifying text is "CHange back to timeOnly / No longer get the V1
    # trip" (21 chars from "No" to "trip", real typo capitalization kept verbatim) and shot 207440's
    # is "never got the first n=1 mode amplitude trip" (39 chars) -- both read "bad" with only the
    # 16-char cap and must read "unknown" once the same-clause window is widened.
    v1_trip = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] barrj 2024-08-08 15:03:28\nCHange back to timeOnly\n\n"
        "No longer get the V1 trip\n"
    )
    assert text.verdict(v1_trip) == "unknown"
    mode_amplitude_trip = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] choiwilkie 2026-05-28 17:06:15\nSetup:\n"
        "add in 2MW of beams in L-mode, hope to keep it alive\n\n"
        "Result:\nnever got the first n=1 mode amplitude trip\n"
    )
    assert text.verdict(mode_amplitude_trip) == "unknown"
    # But widening the window naively also cancels a *causal reassertion*, where the trip is real
    # and is what caused the earlier clause: shot 157369's session leader writes "Still no lower
    # coils due to power supply trip." -- the "due to" between "no" and "trip" means the trip
    # happened and caused the lack of lower coils, so it must still read "bad", not "unknown".
    causal_trip = text.parse_log_entries(
        "### [SESSION_LEADER] smithsp 2014-05-16 12:39:42\n"
        "Put in I-coil ramp to verify that commands working correctly.\n"
        "Result:\nStill no lower coils due to power supply trip.\n"
    )
    assert text.verdict(causal_trip) == "bad"
    # Known, measured residual (documented on _guarded_trip(), not solved here): shot 164995's
    # session leader writes "Not enough to avoid the power supply trip." -- a same-clause *double*
    # negative (failed to avoid it = it happened) with no "due to"/"because of" for _TRIP_CAUSAL to
    # find, so it still reads "unknown" though the trip is real. Not asserted as correct behavior;
    # left unasserted here deliberately so this test never enshrines the residual as intended.


# ---------------------------------------------------------------- data access, composition, embeddings


def test_logs_subset_and_records(paths, text_fixtures):
    n = text.build_logs_subset(paths, {900001, 900003})
    assert n == 2 and text.subset_path(paths).exists()
    assert text.build_logs_subset(paths, {900001}) == 0  # idempotent
    rec = text.load_log_record(900003, paths)
    assert rec["mpid"] == "2014-21-20" and text.load_log_record(555, paths) is None
    assert text.load_shot_index(paths)[900002] == "20150120"


def test_mp_text_prefers_bundle_then_md_header(paths, text_fixtures):
    title, purpose, source = text.mp_text("20150120", 900001, paths)
    assert source == "bundle" and title == "QH-mode access at low NBI torque"
    assert "wide-pedestal" in purpose
    title3, purpose3, source3 = text.mp_text("20150120", 900003, paths)
    assert source3 == "md_header" and title3 == "QH-mode access at low NBI torque"
    assert purpose3 is None
    assert text.mp_text("29990101", 1, paths) == (None, None, "none")


def test_mp_text_rejects_a_bundle_from_another_run(paths, text_fixtures):
    # Correction C1 (task-9-report.md): shotsummary/processed/per_shot_txt/shot_<N>.txt is written
    # per shot number *mentioned* in a run's text, so 1,876 of the 22,950 real bundles carry a
    # different run's mini-proposal than sql/index.json gives for that shot -- including all 7
    # staged shots that have one. Real example: shot_161405.txt (a 2015 shot, index.json run
    # 20150122B) has header "RUN_ID: 20240530" and a body about run 20240530's "RT torbeam
    # commissioning and KSTAR shape development", shot_range 198801 - 198817. Using it would attach
    # a 2024 experiment's purpose to a 2015 shot, so a bundle whose RUN_ID header disagrees with
    # the resolved run must be skipped entirely rather than trusted.
    wrong_run = (
        "# DIII-D per-shot text bundle\n\nRUN_ID: 20240530\n\nSHOT: 900003\n\n"
        "## Planned context (mini-proposal)\nSubject: RT torbeam commissioning\n"
        "1. Purpose of Experiment\n" + "Commission the real-time torbeam calculation. " * 8 + "\n"
    )
    (paths.per_shot_txt_dir / "shot_900003.txt").write_text(wrong_run)
    title, purpose, source = text.mp_text("20150120", 900003, paths)
    assert source == "md_header" and title == "QH-mode access at low NBI torque"
    assert purpose is None
    # A bundle with no RUN_ID header at all is unverifiable, so it is skipped the same way.
    (paths.per_shot_txt_dir / "shot_900003.txt").write_text(
        wrong_run.replace("RUN_ID: 20240530", "(no run id)")
    )
    assert text.mp_text("20150120", 900003, paths)[2] == "md_header"
    # ... and so is a matching bundle when the caller could not resolve a run id to check against.
    assert text.mp_text(None, 900001, paths) == (None, None, "none")


def test_human_tier_and_compose(paths, text_fixtures):
    text.build_logs_subset(paths, {900001, 900003})
    h1 = text.human_tier(900001, paths)
    assert h1.mpid == "2014-21-20" and h1.run_id == "20150120" and h1.verdict == "good"
    assert h1.requested["pnbi_MW"] == 4.5 and h1.session_leaders == ["luce", "burrell"]
    assert h1.chief_operator_status.startswith("900001 Plasma good")
    h3 = text.human_tier(900003, paths)
    assert h3.verdict == "bad" and h3.refshot == 900001 and h3.mp_step == "1B"
    text_mp, text_log = text.compose_texts(h1)
    assert text_mp.startswith("QH-mode access at low NBI torque")
    # text_log opens with the chief-operator line, not with shot_brief: the brief is "autoload
    # system <timestamp>" in all 52,871 non-null records of the corpus and was dropped in fix
    # wave 1 (item 5). Its own [CHIEF_OPERATOR] entry is not repeated after it.
    assert text_log.startswith("900001 Plasma good. IP min.")
    assert "autoload system" not in text_log and "[CHIEF_OPERATOR]" not in text_log
    assert "requested Ip 1.2 MA" in text_log and h1.shot_brief == "autoload system"
    empty = text.human_tier(900002, paths)  # no logbook record, run known
    assert empty.run_id == "20150120" and empty.verdict == "unknown" and empty.log_entries == []


def test_embed_texts_uses_model_and_zero_for_empty(monkeypatch):
    class Stub:
        def encode(self, texts, **kw):
            return np.full((len(texts), 384), 1 / np.sqrt(384), dtype=np.float32)

    monkeypatch.setattr(text, "_load_model", lambda name=None: Stub())
    e = text.embed_texts(["a", "", "b"])
    assert e.shape == (3, 384) and e.dtype == np.float32
    assert abs(np.linalg.norm(e[0]) - 1.0) < 1e-5 and np.all(e[1] == 0)


# ---------------------------------------------------------------- fix wave 1 (review of 4044957)

PURPOSE_LINES = [
    "Subject: QH-mode access at low NBI torque",
    "1. Purpose of Experiment",
    "The purpose of this experiment is to access the wide-pedestal QH-mode regime with",
    "co-current neutral beams at low torque, and to measure how the pedestal structure",
    "responds to a scan of the injected torque at fixed heating power.",
]


def _run_dir(paths, run_id: str):
    d = paths.shotsummary_raw_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "miniproposal.md").write_text("# Miniproposal\n**Subject**: Fallback md title\n")
    return d


def test_mp_text_pdf_tier_end_to_end(paths, text_fixtures, write_mp_pdf):
    # The pdf tier had no test at all: `raise AssertionError` as the first statement of _pdf_text
    # left all 97 tests green, while on the real corpus it is 435 of the 447 mini-proposal-text
    # hits on this project's staged shots and the only tier that ever yields an mp_purpose.
    d = _run_dir(paths, "20990101")
    write_mp_pdf(d / "miniproposal.pdf", PURPOSE_LINES)
    before = text._pdf_text.cache_info()
    title, purpose, source = text.mp_text("20990101", 999001, paths)
    assert source == "pdf"
    assert title == "QH-mode access at low NBI torque"  # the PDF's Subject line beats the md one
    assert purpose.startswith("Purpose of Experiment")
    assert "wide-pedestal QH-mode regime" in purpose
    # A mini-proposal is per run but mp_text is per shot, which is why _pdf_text is memoized
    # (commit 4044957: 44 s -> 2.7 s over the 936 staged shots). A second shot of the same run must
    # hit the cache rather than re-extract.
    assert text.mp_text("20990101", 999002, paths)[2] == "pdf"
    after = text._pdf_text.cache_info()
    assert after.misses == before.misses + 1 and after.hits == before.hits + 1


def test_pdf_text_replaces_the_unpaired_surrogates_pypdf_returns(tmp_path, monkeypatch):
    """A symbol-font glyph a PDF maps outside Unicode comes out of pypdf as a lone surrogate, and
    a lone surrogate cannot be encoded -- `ShotRecord.model_dump_json()` raises on it, in
    `records_to_tables`, after every shot of the build has already been read. Measured on
    configs/ideate/shot_lists/poc_v1.yaml: 8 of its 200 shots, from 4 run days, carry them in
    mp_purpose.
    """
    import json

    import pypdf

    class _Page:
        def extract_text(self):
            return "scans of the collisionality (\udf08!) and the temperature gradient"

    class _Reader:
        def __init__(self, path):
            self.pages = [_Page()]

    monkeypatch.setattr(pypdf, "PdfReader", _Reader)
    out = text._pdf_text(str(tmp_path / "unique.pdf"), (0, 0))
    assert "\udf08" not in out and "collisionality (\ufffd!)" in out
    json.dumps(out)  # the failure this guards against: unencodable, so unserializable
    out.encode("utf-8")


@pytest.mark.parametrize(
    ("case", "lines"),
    [
        # Shot 194802's run: pypdf reads the Acrobat "no font" placeholder page, and the only gate
        # was len(txt) > 200, so mp_text returned it as this experiment's purpose. 43 real runs.
        (
            "acrobat placeholder",
            [
                "This is a test PDF document.",
                "If you can read this, you have Adobe Acrobat Reader installed on your computer.",
                "5. Resources * Machine Setup",
                "-- Diagnostic Coordinator -- Physics Operator -- Session Leader --",
                "Neutral beams, electron cyclotron heating, gas injection, magnetic diagnostics.",
            ],
        ),
        # 112 real runs whose PDF extracts as the "Resources / Machine Setup" form only, plus 56
        # with no Purpose/Goal heading anywhere (shot 178661's starts at "3. Experimental Method").
        # With no heading to anchor on, _purpose() returns the top of page 1 -- the cover sheet.
        (
            "form dump, no purpose heading",
            [
                "MP test v1",
                "5. Resources * Machine Setup",
                "-- Diagnostic Coordinator: needed for this session, see the run coordinator.",
                "3. Experimental Method: restore the reference shot and step the torque down.",
                "-- Session Leader and Physics Operator to be assigned before the run day.",
            ],
        ),
        # Shot 200057's run: every 's' came out as an apostrophe, every 't' as a left quote. 58 real
        # runs are corrupted this way (shot 162163's is symbol soup instead); the text is confidently
        # wrong rather than absent, so length alone can never catch it.
        (
            "broken font encoding",
            [
                "1. Purpose of Experiment",
                "goal': Thi' expe'imen\" add'e''e' \"he 'eq\"i'emen\"' de'c'ibed in \"he DIII-D",
                "5 Yea' Plan 'ec\"ion 2.3.2 on Con\"'olling \"he pede'\"al wi\"h RMP coil'",
                'and "o e\'"abli\'h a \'ob"\'" ope\'a"ing poin" fo\' "he nex" campaign.',
            ],
        ),
    ],
)
def test_mp_text_demotes_an_unusable_pdf(paths, text_fixtures, write_mp_pdf, case, lines):
    # Item 1 of the fix-wave-1 review. Classifying all 1,661 real miniproposal.pdf files found 270
    # (16%) whose extracted text is not this experiment's purpose, every one of which mp_text used
    # to return as source="pdf". Demoting them to the md header (or to "none") is the same
    # correction C1 makes for the per-shot bundles: never label text as this shot's purpose unless
    # it is.
    d = _run_dir(paths, "20990102")
    write_mp_pdf(d / "miniproposal.pdf", lines)
    title, purpose, source = text.mp_text("20990102", 999003, paths)
    assert source == "md_header", case
    assert title == "Fallback md title" and purpose is None
    # ... and with no md to fall back to, the answer is "none", not a wrong purpose.
    (d / "miniproposal.md").unlink()
    assert text.mp_text("20990102", 999003, paths) == (None, None, "none")


def test_mp_text_keeps_a_purpose_that_merely_lacks_a_subject_line(
    paths, text_fixtures, write_mp_pdf
):
    # The review suggested `_subject() is None` as a rejection signal. Measured, it is true for 332
    # of the 1,661 real PDFs but for only 270 unusable ones: as a rule it would additionally throw
    # away 107 good purposes that simply have no "Subject:" line (run 20170621A, "Goals: As this
    # experiment is part of the DIII-D National Campaign...", is one). So the gate does not use it,
    # and human_tier falls back to run_title for the title of such a run.
    d = _run_dir(paths, "20990103")
    write_mp_pdf(
        d / "miniproposal.pdf",
        [
            "Goals:",
            "As this experiment is part of the DIII-D National Campaign, it is designed to have",
            "relevance to other devices, and will document the pedestal response to the applied",
            "n=3 perturbation over a scan of the edge safety factor at fixed shape and power.",
        ],
    )
    title, purpose, source = text.mp_text("20990103", 999004, paths)
    assert source == "pdf" and title is None
    assert purpose.startswith("Goals:") and "National Campaign" in purpose


def test_load_log_record_cannot_poison_the_cache(paths, text_fixtures):
    # Item 2. _read_subset is memoized (commit 4044957) and load_log_record used to hand back a
    # shallow copy, with a comment claiming "the cached record itself stays read-only". Real
    # records carry two list-valued fields -- `topics` and `keywords` -- and both were shared by
    # reference, so on shot 163112 of the real corpus
    # `load_log_record(163112, p)["keywords"].append("POISON")` made every later read of that shot
    # return ['POISON'], process-wide, for the life of the process.
    text.build_logs_subset(paths, {900001, 900003})
    rec = text.load_log_record(900001, paths)
    rec["keywords"].append("POISON")
    rec["topics"].append("POISON")
    rec["mpid"] = "clobbered"
    again = text.load_log_record(900001, paths)
    assert again["keywords"] == [] and "POISON" not in again["topics"]
    assert again["mpid"] == "2014-21-20"


def test_load_shot_index_cannot_be_mutated(paths, text_fixtures):
    # Item 3. load_shot_index returned the memoized dict itself with no copy at all, so a caller's
    # .pop() corrupted every later human_tier() in the process. The real index has 53,176 entries
    # and human_tier does one lookup per shot, so it is handed out read-only rather than copied.
    idx = text.load_shot_index(paths)
    assert idx[900002] == "20150120"
    with pytest.raises((TypeError, AttributeError)):  # mappingproxy has neither pop nor __setitem__
        idx.pop(900002)
    with pytest.raises(TypeError):
        idx[900002] = "clobbered"
    assert text.load_shot_index(paths)[900002] == "20150120"
    assert text.human_tier(900002, paths).run_id == "20150120"


def test_logs_subset_cache_sees_an_append(paths, text_fixtures):
    # Item 6, and the only test of commit 4044957's memoization. _subset_records is keyed on
    # (path, size, mtime_ns); without the stat key in the key, reading a shot back would prime the
    # cache and the records appended by the *next* build_logs_subset call would be invisible --
    # which is exactly the build_logs_subset -> human_tier order the database build uses.
    assert text.build_logs_subset(paths, {900001}) == 1
    assert text.load_log_record(900001, paths)["mp_step"] == "1A"
    assert text.build_logs_subset(paths, {900003}) == 1
    assert text.load_log_record(900003, paths) is not None


def test_logs_subset_tolerates_and_heals_a_torn_trailing_line(paths, text_fixtures, caplog):
    """A build killed while appending left a cut-off last record; the bare json.loads per line
    then raised on every later build and add until the file was deleted by hand. The cut line is
    skipped with a warning, its shot reads as absent, and the next build rewrites the file whole
    -- with that shot re-extracted and the tail gone."""
    assert text.build_logs_subset(paths, {900001}) == 1
    p = text.subset_path(paths)
    good = p.read_bytes()
    full = next(ln for ln in paths.logs_jsonl.read_bytes().split(b"\n") if b'"shot": 900003' in ln)
    p.write_bytes(good + full[: len(full) // 2])  # SIGKILL halfway through the second record
    with caplog.at_level(logging.WARNING):
        assert text.load_log_record(900001, paths)["mp_step"] == "1A"
        assert text.load_log_record(900003, paths) is None
    assert "undecodable trailing line" in caplog.text
    assert text.build_logs_subset(paths, {900003}) == 1
    lines = p.read_bytes().split(b"\n")
    assert lines[-1] == b"" and len(lines) == 3 and not p.with_name(p.name + ".tmp").exists()
    assert [json.loads(ln)["shot"] for ln in lines[:2]] == [900001, 900003]
    assert text.load_log_record(900003, paths)["mp_step"] == "1B"


def test_logs_subset_mid_file_corruption_is_still_an_error(paths, text_fixtures):
    text.build_logs_subset(paths, {900001, 900003})
    p = text.subset_path(paths)
    first, rest = p.read_bytes().split(b"\n", 1)
    p.write_bytes(first[:-5] + b"\n" + rest)  # not a torn tail: line 1 is damaged
    text._subset_records.cache_clear()
    with pytest.raises(ValueError, match="line 1"):
        text.load_log_record(900001, paths)


def test_logs_subset_write_is_atomic(paths, text_fixtures, monkeypatch):
    """The new content reaches the cache path only through os.replace of a finished temp file,
    so a failure at any point leaves the previous file byte-for-byte intact and no temp behind."""
    text.build_logs_subset(paths, {900001})
    p = text.subset_path(paths)
    before = p.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(text.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        text.build_logs_subset(paths, {900003})
    assert p.read_bytes() == before
    assert not p.with_name(p.name + ".tmp").exists()
    monkeypatch.undo()
    assert text.build_logs_subset(paths, {900003}) == 1
    assert text.load_log_record(900003, paths) is not None


def test_compose_texts_puts_verdict_roles_first(paths, text_fixtures):
    # Item 6. The VERDICT_ROLES-first sort in compose_texts was untested: removing it left all 97
    # tests green while changing the entry order for 612 of the 936 real staged shots. It is what
    # keeps the session-leader/operator assessment inside MiniLM's 256-token window when a run's
    # DIAGNOSTICS and PCS entries are long.
    h = HumanTier(
        log_entries=[
            LogEntry(role="DIAGNOSTICS", author="x", time="10:00", text="CER timing checked"),
            LogEntry(role="PCS", author="x", time="10:01", text="iptipp waveform reloaded"),
            LogEntry(role="SESSION_LEADER", author="luce", time="10:02", text="Good QH shot"),
        ]
    )
    _, log = text.compose_texts(h)
    assert log.index("[SESSION_LEADER]") < log.index("[DIAGNOSTICS]") < log.index("[PCS]")


def test_compose_texts_drops_a_title_repeated_as_the_run_title(paths, text_fixtures):
    # Item 6. mp_title and run_title are the same string for 227 of the 936 real staged shots
    # (the mini-proposal subject is the run title), so without dict.fromkeys text_mp opens by
    # saying it twice. Removing the dedup changes text_mp for those 227 shots and broke no test.
    h = HumanTier(
        mp_title="QH-mode access", run_title="QH-mode access", mp_purpose="Access the WPQH"
    )
    assert text.compose_texts(h)[0] == "QH-mode access . Access the WPQH"
    h2 = HumanTier(mp_title="QH-mode access", run_title="Low torque QH", mp_purpose=None)
    assert text.compose_texts(h2)[0] == "QH-mode access . Low torque QH"


def test_parse_log_entries_strips_the_logbook_html_but_not_physics_brackets():
    # Item 5. The autologger's mini-proposal link is 53 MiniLM tokens -- 21 % of the 256-token
    # window -- is identical for every shot of a run, and landed inside the window for 229 of the
    # 626 real staged shots with text (shot 161392 is one). Real anchor text, from shot 152721.
    entries = text.parse_log_entries(
        "### [SESSION_LEADER] x 2013-05-01 09:00:00\n"
        "requested pech: 1 MW  Step in Miniproposal "
        '<a href="https://diii-d.gat.com/DIII-D/physics/miniprop/mp_list.php?mpid=2013-99-99" '
        'target="_blank" >2013-99-99</a> experimental plan: 1.1a\n\n'
        "### [PHYSICS_OPERATOR] y 2013-05-01 09:05:00\n"
        "<b>Pre pulse comment:</b> First plasma attempt again.\n"
    )
    assert "href" not in entries[0].text and "2013-99-99" in entries[0].text
    assert entries[1].text.startswith("Pre pulse comment:")
    # ... but operators write physics in angle brackets, and a general <[^>]*> strip eats it:
    # <ne> is a line-averaged density and "0 <t < 200ms" is a time window (shots 167471, 128011).
    physics = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] z 2007-02-16 15:02:50\n"
        "density trip point set to <ne>=4.2e13; cut early gas (0 <t < 200ms) to 3V. <TINJ>~2.3\n"
    )
    assert "<ne>=4.2e13" in physics[0].text and "0 <t < 200ms" in physics[0].text
    assert "<TINJ>~2.3" in physics[0].text


def test_compose_texts_does_not_repeat_the_chief_operator_line():
    # Item 5. chief_operator_status is chief[:200] of the CHIEF_OPERATOR entry, and the entry was
    # then emitted again verbatim -- true for 571 of the 626 real staged shots with text. The head
    # copy is the one that is guaranteed to be inside the encoder window, so the entry goes.
    short = LogEntry(role="CHIEF_OPERATOR", author="byrnep", time="11:12", text="Plasma good.")
    h = HumanTier(chief_operator_status="Plasma good.", log_entries=[short])
    assert text.compose_texts(h)[1] == "Plasma good."
    # A chief entry longer than the 200-char head copy keeps its full text: deduplicating on a
    # prefix would throw away everything past character 200.
    long_text = "Plasma good. " + "Ip flat-top held to 5.0 s with no beam faults. " * 6
    h2 = HumanTier(
        chief_operator_status=long_text[:200],
        log_entries=[
            LogEntry(role="CHIEF_OPERATOR", author="byrnep", time="11:12", text=long_text)
        ],
    )
    log = text.compose_texts(h2)[1]
    assert log.startswith(long_text[:200]) and "[CHIEF_OPERATOR] " + long_text in log


def test_compose_texts_renders_the_requested_density(paths, text_fixtures):
    # Item 8. parse_requested extracts an optional fifth field, "requested density: ...", and
    # compose_texts rendered only the other four. No staged shot has it, but 64 of the 4,015 corpus
    # records with a requested block do (shots 171042-171045 among them).
    h = HumanTier(requested={"ip_MA": 1.4, "bt_T": -2.1, "pnbi_MW": 5.0, "pech_MW": 0.0})
    assert text.compose_texts(h)[1].endswith("PECH 0 MW")
    h.requested["density_cm3"] = 0.7
    assert text.compose_texts(h)[1].endswith("PECH 0 MW, ne 0.7e13 cm^-3")


def test_embed_texts_width_follows_the_model(monkeypatch):
    # Item 8. embed_texts allocated a hardcoded (n, 384) while EMBED_MODEL is swappable, so a
    # different encoder would have raised on the assignment or silently disagreed with the caller.
    class Stub:
        def encode(self, texts, **kw):
            return np.full((len(texts), 768), 1 / np.sqrt(768), dtype=np.float32)

    monkeypatch.setattr(text, "_load_model", lambda name=None: Stub())
    assert text.embed_texts(["a", "", "b"]).shape == (3, 768)
    # ... with nothing to embed the model is never loaded, so the width falls back to EMBED_DIM.
    assert text.embed_texts(["", "  "]).shape == (2, text.EMBED_DIM)


# --------------------------------------------------------- task-10 fix wave 1: scope guards
# Every string below was read back from parse_log_entries() over the real sql/logs.jsonl and is
# reproduced character for character.


def test_fault_strings_and_verdict_ignore_a_planned_mitigation_and_a_back_reference():
    """Shot 175676 (run 20180227) shipped into configs/ideate/shot_lists/poc_v1.yaml as `verdict:
    bad, fault: locked mode;dud trip`. Its chief operator wrote "10:32 Plasma shot ok." and its
    session leader wrote "Postshot: Density increased a little too high, but should still have
    usable data." -- nothing in the record says the shot failed. Every fault word came from the
    physics operator's pre-shot plan: "kills us" grades the *previous* shot under a "Last shot:"
    heading, and "locked mode dud trip" is a protection being installed *to avoid* a disruption."""
    entries = text.parse_log_entries(
        "### [SESSION_LEADER] x 2018-02-27 10:24:00\n"
        "Preshot:\nELMy plasma. RMPs off at 2 s. Edge CER to view C and Al in edge. Decrease beam "
        "power from 2 s.\n\n"
        "### [PHYSICS_OPERATOR] y 2018-02-27 10:29:00\n"
        "Last shot: Successful enough - late tearing mode kills us, but get 2/3 LBOs\ngood.\n\n"
        "This shot:\n- Back to full 1 cm gap change program.\n"
        "- Increase L-mode density back by 0.1 E 19.\n"
        "- Implement locked mode dud trip to avoid disruption.\n- Turn off I-coil at 2s.\n\n"
        "### [CHIEF_OPERATOR] z 2018-02-27 10:32:00\n10:32 Plasma shot ok.\n\n"
        "### [SESSION_LEADER] x 2018-02-27 10:39:00\n"
        "Postshot: Density increased a little too high, but should still have usable data.\n"
    )
    plan = entries[1].text
    assert text.fault_strings(plan) == []
    assert text.verdict(entries) == "good"


def test_own_shot_text_drops_only_the_other_shots_span():
    """A back-reference span ends at the next shot heading, at a blank line, or at the logbook's
    own dashed divider -- and a comparison with no colon ("better than last shot") is not a
    heading at all."""
    kept = text._own_shot_text(
        "Last shot: no breakdown again\n\nThis shot:\nGlow, 10 minutes, try agian."
    )
    assert "no breakdown" not in kept
    assert "Glow, 10 minutes, try agian." in kept
    # shot 191977: "Previous shot:" ... "Next shot:" -- both are other shots, nothing survives
    both = text._own_shot_text(
        "Previous shot: Issue with radiation feedback control (see Davids comment). Density "
        "feedback control on point to keep pedestal at 2.2e19.\n\nNext shot: Fixed radiation "
        "feedback control and repeat."
    )
    assert "radiation feedback" not in both
    # no colon, so this is one operator's own comparison and must survive intact
    same = "Very good shot, ITB weaks less than last shot after N2 seeding."
    assert text._own_shot_text(same) == same
    assert text._own_shot_text(None) == ""


def test_own_shot_text_keeps_a_repeat_header_and_stops_at_an_outcome_heading():
    """Two shapes that look like a back-reference but introduce *this* shot.

    "Repeat last shot:" names what this shot copies (shot 145809's physics operator, and 4 more
    in the corpus), and an outcome heading inside a span always reports the shot the entry
    belongs to (shot 158368's session leader, and 4 more).
    """
    repeat = "REPEAT LAST SHOT: Gyros now also in 'profctl' for 2.4 to 3.4 s"
    assert text._own_shot_text(repeat) == repeat
    kept = text._own_shot_text(
        "Next Shot: Higher density scan. Target: 3-6e13 cm-3 at the strike point using div TS. "
        "Result: Density limit for setting of 6."
    )
    assert "Higher density scan" not in kept
    assert "Result: Density limit for setting of 6." in kept


def test_verdict_a_bad_previous_shot_does_not_condemn_this_one():
    """Shot 178572's whole physics-operator entry is a back-reference plus a plan; the chief
    operator is the only one who speaks about the shot itself."""
    entries = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] x 2019-06-13 09:00:00\n"
        "Last shot: Bummer. Very early LM.\n\nThis shot: Add gas in the early phase, repeat....\n"
    )
    assert text.verdict(entries) == "unknown"


def test_verdict_a_good_previous_shot_does_not_excuse_this_one():
    """The guard cuts both ways, and on the corpus it moves 50 verdicts *towards* bad. Shot
    174796's physics operator praises the previous shot and reports a disruption on this one."""
    entries = text.parse_log_entries(
        "### [PHYSICS_OPERATOR] x 2018-08-01 09:00:00\n"
        "Last shot: good shot\n\nThis shot: static drsep=0. Add n=3 solid. turn off gas at 3s.\n\n"
        "### [PHYSICS_OPERATOR] x 2018-08-01 09:30:00\n"
        "Restore DRSEP from 173828, then IP and mod IP to get to 1.6 MA at 2.2 sec and\nhold.\n\n"
        "Result: OOPS. The IP ramp starts before the DRSEP fast ramp starts; we get Q95\n"
        "to approach 3, and a LM disruption at 1.6 seconds.\n"
    )
    assert text.verdict(entries) == "bad"


def test_mitigation_guard_is_forward_and_clause_bounded():
    # the shape it exists for: the fault is named first, the purpose last
    assert (
        text.fault_strings(
            "Switchh Dud trip LM test to 60 at 4 seconds to prevent tripping in rampdown."
        )
        == []
    )
    # "to stop" is not a purpose clause in this corpus -- shot 126933's real trip must survive
    assert text._guarded_trip("180RM1V, E-coil volt sec trip caused plasma to stop at 3 sec") == 1
    # a purpose clause in the *next* clause does not reach back
    assert text.fault_strings("Got a locked mode. Raise the level to avoid it next time.") == [
        "locked mode"
    ]


def test_verdict_does_not_splice_a_phrase_out_of_two_operators_entries():
    """Shot 178493: the physics operator's entry ends "doesn't look half-bad" and the chief
    operator's begins "Plasma shot  ok." -- joined on a bare space that reads as the fault phrase
    "bad plasma", which neither of them wrote. Entries are joined on a clause break instead."""
    entries = text.parse_log_entries(
        "### [SESSION_LEADER] x 2019-06-14 16:00:00\n"
        "Result:\n - Shot runs and doesn't look half-bad\n\n"
        "### [CHIEF_OPERATOR] y 2019-06-14 16:12:00\nPlasma shot  ok.  AAs were not turned on "
        "though.\n"
    )
    assert text.verdict(entries) == "good"


def test_verdict_a_negation_does_not_reach_into_the_next_operators_entry():
    """Shot 149143: the chief's "IP Dot - No 30L" was cancelling the session leader's own "bad
    shot; multiple beam failures." two entries later."""
    entries = text.parse_log_entries(
        "### [CHIEF_OPERATOR] x 2012-06-15 11:08:00\n11:08  Plasma Shot - IP Dot - No 30L\n\n"
        "### [SESSION_LEADER] y 2012-06-15 11:09:00\n"
        "bad shot; multiple beam failures.  Dropping 210R to 65 keV swapping in for 210L.\n"
    )
    assert text.verdict(entries) == "bad"


def test_bundle_blocks_splits_the_three_sections_and_loses_nothing():
    """The three sections of a per-shot bundle, verbatim and concatenating back to the whole.

    Which section a sentence is in is what it means -- a record of the run, an intention, or a
    fact about this discharge -- so the split has to be exact and it has to be lossless.
    """
    from .conftest import text_bundle

    bundle = text_bundle(900001, row={"SHOT_TYPE": "plasma"}, title="Tearing mode avoidance")
    session, planned, specific = text.bundle_blocks(bundle)

    assert session + planned + specific == bundle
    assert session.startswith("# DIII-D per-shot text bundle")
    assert "GENERAL SESSION INFO" in session
    assert planned.startswith("## Planned context (mini-proposal)")
    assert "Hypothesis to be tested" in planned
    assert specific.startswith(text.SHOT_BLOCK_MARKER)
    assert specific.strip().endswith(text.shot_block(bundle))


def test_bundle_blocks_tolerates_a_bundle_with_no_mini_proposal():
    bundle = f"# DIII-D per-shot text bundle\n\nrun stuff\n\n{text.SHOT_BLOCK_MARKER}\nSHOT: 1\n"
    session, planned, specific = text.bundle_blocks(bundle)
    assert planned == ""
    assert session + specific == bundle
