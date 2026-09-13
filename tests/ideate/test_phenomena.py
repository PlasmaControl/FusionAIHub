"""The phenomenon registry, the evidence classes, and what `ideate phenomenon` may claim.

Everything here is about ONE distinction repeated at four scales: an observation, a model's
opinion about the present, a model's opinion about the future, and a sentence somebody typed are
four different claims, and this module's job is to never let one be reported as another.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ideate import cli
from ideate.retrieval import phenomena as ph
from ideate.shotdb import store

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------------- the fixture
#
# Four shots, one phenomenon each way. Their flat top is 1.0-5.0 s (conftest's `shot_record`), so
# an event at 2.0 s is inside it and one at 0.5 s is not.
#
#   100  a detector saw a tearing mode (and an ELM)      -> OBSERVED
#   101  only a forecast, and a label with no valid rows -> FORECAST, label None (never 0)
#   200  only the operators' word                        -> TEXT ONLY
#   201  nothing at all                                  -> not a hit

OBSERVED_SHOT, FORECAST_SHOT, TEXT_SHOT, SILENT_SHOT = 100, 101, 200, 201


def _event(shot: int, event_id: str, **over) -> dict:
    from labelmaker.events import schema as events_schema

    row = {
        "shot": shot,
        "event_id": event_id,
        "source": "tokeye_track",
        "evidence_kind": "detector",
        "phenomenon": "coherent_mode",
        "t0_s": 2.0,
        "t1_s": 2.5,
        "f0_khz": np.nan,
        "f1_khz": np.nan,
        "confidence": 0.9,
        "horizon_s": np.nan,
        "diag": "mhr",
        "channel": 0,
        "pass_name": "wide",
        "attrs": {"f_centroid_khz": 10.0},
        "t_cov0_s": 0.0,
        "t_cov1_s": 6.0,
        "run_id": "r1",
        "git_sha": "abc",
        "written_at": "2026-09-07T00:00:00+00:00",
    }
    row.update(over)
    row["attrs"] = json.dumps(row["attrs"], sort_keys=True)
    assert set(row) == set(events_schema.COLUMNS)
    return row


def _label_row(shot: int, label: str, **over) -> dict:
    row = {
        "shot": shot,
        "slug": "d3d_tearing_onset_cnn1d",
        "label": label,
        "n_valid": 100,
        "valid_frac": 1.0,
        "max_valid": 0.8,
        "mean_valid": 0.4,
        "p95_valid": 0.7,
        "thr": np.nan,
        "thr_source": "",
        "frac_above": np.nan,
        "first_above_t_s": np.nan,
        "n_intervals": None,
        "longest_interval_s": np.nan,
        "artifact_sha256": "sha",
    }
    row.update(over)
    return row


def _claim(shot: int, phenomenon: str, **over) -> dict:
    row = {
        "shot": shot,
        "phenomenon": phenomenon,
        "polarity": "pos",
        "temporality": "observed",
        "snippet": "clear tearing mode at 2 s",
        "scope": "shot",
    }
    row.update(over)
    return row


def _write_tables(db_dir: Path, events: list[dict], labels: list[dict], claims: list[dict]) -> None:
    from ideate.labels.claims import CLAIMS_DTYPES
    from ideate.labels.join import LABELS_WIDE_DTYPES
    from labelmaker.events import schema as events_schema

    pd.DataFrame(events, columns=list(events_schema.COLUMNS)).astype(
        events_schema.DTYPES
    ).to_parquet(db_dir / "events.parquet", index=False)
    pd.DataFrame(labels, columns=list(LABELS_WIDE_DTYPES)).astype(
        LABELS_WIDE_DTYPES
    ).to_parquet(db_dir / "labels_wide.parquet", index=False)
    pd.DataFrame(claims, columns=list(CLAIMS_DTYPES)).astype(CLAIMS_DTYPES).to_parquet(
        db_dir / "text_claims.parquet", index=False
    )


@pytest.fixture
def phen_db(ideate_db: Path) -> store.ShotDB:
    """`ideate_db`'s four shots, plus the three label tables written the way the join writes them."""
    events = [
        # 100: three tracks in the tearing band inside the flat top, one of them barely scored,
        # plus an ELM a detector saw. The ELM row is what `--avoid phenomenon:elm` must find.
        _event(OBSERVED_SHOT, "100-tokeye_track-00000"),
        _event(OBSERVED_SHOT, "100-tokeye_track-00001", t0_s=3.0, t1_s=3.4),
        _event(OBSERVED_SHOT, "100-tokeye_track-00002", t0_s=4.0, t1_s=4.2),
        _event(OBSERVED_SHOT, "100-tokeye_track-00003", t0_s=4.5, t1_s=4.6, confidence=0.3),
        # Out of band (150 kHz is an AE, not a tearing mode) and out of window (0.2 s is ramp-up).
        _event(OBSERVED_SHOT, "100-tokeye_track-00004", attrs={"f_centroid_khz": 150.0}),
        _event(OBSERVED_SHOT, "100-tokeye_track-00005", t0_s=0.2, t1_s=0.3),
        _event(
            OBSERVED_SHOT, "100-tokeye_transient-00000", source="tokeye_transient",
            phenomenon="elm", t0_s=2.2, t1_s=2.2, confidence=np.nan, attrs={"col": 3},
        ),
        # 101: two forecast rows and nothing else. No detector wrote a thing, so no coverage.
        _event(
            FORECAST_SHOT, "101-label_forecast-00000", source="label_forecast",
            evidence_kind="forecast", phenomenon="tearing", horizon_s=1.0, diag="",
            confidence=0.35, attrs={"label": "tm_risk_1s", "thr": 0.2},
        ),
        _event(
            FORECAST_SHOT, "101-label_forecast-00001", source="label_forecast",
            evidence_kind="forecast", phenomenon="tearing", t0_s=3.0, t1_s=3.1,
            horizon_s=0.5, diag="", confidence=0.22, attrs={"label": "tm_risk_500ms"},
        ),
        # 200: the ELM detector ran on the mhr data and found nothing to report but a quiet
        # stretch. That row is the only thing that says the shot was LOOKED AT for ELMs.
        _event(
            TEXT_SHOT, "200-elm_clock-00000", source="elm_clock", evidence_kind="heuristic",
            phenomenon="elm_free", t0_s=1.0, t1_s=4.0, confidence=np.nan,
            attrs={"n_elms_inside": 0},
        ),
    ]
    labels = [
        _label_row(OBSERVED_SHOT, "tm_prob"),
        # Every sample invalid: the model emitted nothing it stood behind.
        _label_row(FORECAST_SHOT, "tm_prob", n_valid=0, max_valid=np.nan, mean_valid=np.nan),
    ]
    claims = [
        _claim(OBSERVED_SHOT, "tearing"),
        _claim(TEXT_SHOT, "tearing", snippet="looked like a 2/1 tearing mode"),
        _claim(TEXT_SHOT, "tearing", snippet="the tearing mode locked"),
    ]
    _write_tables(ideate_db / "db", events, labels, claims)
    db = store.ShotDB.load(ideate_db / "db")
    # Shot 100's logbook says what its detector saw. That is what makes it the hit that lacks
    # NOTHING -- no missing evidence class, and a quote that is about the phenomenon rather than
    # the shot's best sentence on some other subject.
    from ideate.schema import LogEntry

    rec = db.get(OBSERVED_SHOT)
    rec.human.log_entries.append(
        LogEntry(role="PHYSICS_OPERATOR", text="Postshot: clear 2/1 tearing mode from 2 s.")
    )
    db.shots.loc[OBSERVED_SHOT, "record_json"] = rec.model_dump_json()
    return db


# -------------------------------------------------------------------------- registry validation


def test_the_registry_names_exactly_labelmakers_round_one_phenomena():
    from labelmaker.events.lexicon import PHENOMENON_IDS

    reg = ph.registry()
    assert set(reg) == set(PHENOMENON_IDS)
    # And the names come from labelmaker's file, not from a second copy of them here.
    assert reg["eho"].aliases == tuple(ph._lexicon()["eho"].aliases)
    assert reg["elm"].exclude == tuple(ph._lexicon()["elm"].negatives)


def _registry_file(tmp_path: Path, mutate) -> Path:
    doc = yaml.safe_load((REPO / "configs" / "ideate" / "phenomena.yaml").read_text())
    mutate(doc)
    path = tmp_path / "phenomena.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_an_unknown_id_is_an_error_not_a_phenomenon_that_joins_to_nothing(tmp_path):
    def add(doc):
        doc["phenomena"]["sawtoth"] = {"title": "typo"}

    with pytest.raises(ph.PhenomenaError, match="unknown"):
        ph.registry(_registry_file(tmp_path, add))


def test_a_missing_id_is_an_error(tmp_path):
    with pytest.raises(ph.PhenomenaError, match="missing"):
        ph.registry(_registry_file(tmp_path, lambda doc: doc["phenomena"].pop("rwm")))


def test_restating_the_alias_list_here_is_refused(tmp_path):
    """One file, two readers (plan §5.6). A second alias list is a second vocabulary."""

    def add(doc):
        doc["phenomena"]["eho"]["aliases"] = ["eho", "wide pedestal"]

    with pytest.raises(ph.PhenomenaError, match="aliases"):
        ph.registry(_registry_file(tmp_path, add))


def test_the_shipped_registry_states_no_aliases_and_no_excludes():
    doc = yaml.safe_load((REPO / "configs" / "ideate" / "phenomena.yaml").read_text())
    for pid, body in doc["phenomena"].items():
        assert not ({"aliases", "exclude", "negatives"} & set(body)), pid


def test_every_configured_label_exists_in_the_built_labels_wide_or_in_labels_yaml():
    """`labels:` may not name a series nobody produces: it would score as permanently
    unavailable, which reads as "the model says no"."""
    from ideate.labels.join import forecast_rules

    # The five slugs the real join produced, and the risk labels labels.yaml names.
    produced = {
        "d3d_ae_activity_seldnet/ae_active",
        "d3d_ae_activity_seldnet/ae_frequency",
        "d3d_tearing_onset_cnn1d/tm_prob",
        "d3d_tearing_onset_cnn1d/betan",
    } | {r.key for r in forecast_rules()}
    for entry in ph.registry().values():
        for ref in entry.labels:
            assert ref.key in produced, f"{entry.id} names {ref.key}, which nothing produces"


def test_requires_group_only_names_corpus_groups_a_shot_can_actually_have(tmp_path):
    def add(doc):
        doc["phenomena"]["eho"]["requires_group"] = ["mhr", "not_a_group"]

    with pytest.raises(ph.PhenomenaError, match="not_a_group"):
        ph.registry(_registry_file(tmp_path, add))


def test_elm_reads_the_elm_detector_and_not_the_elm_free_one():
    """`elm_clock` writes `elm_free` INTERVALS -- stretches with no ELM in them. Counting an
    absence as evidence of the thing it denies is the error plan §2 is about."""
    rules = ph.registry()["elm"].events
    assert [(r.source, r.phenomenon) for r in rules] == [("tokeye_transient", "elm")]


# --------------------------------------------------------------------------------------- resolve


def test_the_longest_alias_wins_and_names_the_phenomenon_it_spells_out():
    assert ph.resolve("edge harmonic oscillation on the flat top")[0] == ("eho", 1.0)


def test_a_second_phenomenon_named_alongside_ranks_below_the_first():
    got = ph.resolve("edge harmonic oscillation with no elms")
    assert got[0][0] == "eho"
    # "no elms" is a denial, so elm does not come along as a weaker match.
    assert [pid for pid, _ in got] == ["eho"]


def test_a_denial_vetoes_rather_than_resolving_to_the_thing_denied():
    assert ph.resolve("eho-free discharge") == []
    assert ph.resolve("no eho this shot") == []


def test_matching_is_on_word_boundaries_so_eho_is_not_inside_method():
    assert ph.resolve("the method we used") == []
    assert ph.resolve("we saw an eho") == [("eho", 1.0)]


def test_text_that_names_nothing_resolves_to_nothing_which_is_not_an_empty_result():
    assert ph.resolve("") == []
    assert ph.resolve(None) == []
    assert ph.resolve("a nice quiet discharge") == []


def test_secondary_matches_get_decreasing_weight():
    got = dict(ph.resolve("quiescent h-mode with a tearing mode"))
    assert got["qh"] == 1.0
    assert 0.0 < got["tearing"] < 1.0


# -------------------------------------------------------------------------------------- evidence


def test_an_observation_and_a_forecast_never_land_in_the_same_list(phen_db):
    seen = ph.evidence(OBSERVED_SHOT, "tearing", phen_db)
    forecast = ph.evidence(FORECAST_SHOT, "tearing", phen_db)
    assert [iv.event_id for iv in seen.intervals] == [
        "100-tokeye_track-00000", "100-tokeye_track-00001",
        "100-tokeye_track-00002", "100-tokeye_track-00003",
    ]
    assert all(iv.evidence_kind == "detector" for iv in seen.intervals)
    assert seen.forecasts == ()
    assert forecast.intervals == ()
    assert [iv.evidence_kind for iv in forecast.forecasts] == ["forecast", "forecast"]


def test_an_out_of_band_or_out_of_window_event_is_not_this_phenomenons(phen_db):
    ids = {iv.event_id for iv in ph.evidence(OBSERVED_SHOT, "tearing", phen_db).intervals}
    assert "100-tokeye_track-00004" not in ids  # 150 kHz: an AE band, not a tearing mode
    assert "100-tokeye_track-00005" not in ids  # 0.2 s: the ramp-up, not the flat top


def test_a_label_with_no_valid_samples_is_unavailable_and_never_zero(phen_db):
    got = ph.evidence(FORECAST_SHOT, "tearing", phen_db)
    assert got.label_evidence["d3d_tearing_onset_cnn1d/tm_prob.max_valid"] is None
    assert got.max_label_p is None
    assert ph.LABEL_UNAVAILABLE in got.caveats
    scored = ph.evidence(OBSERVED_SHOT, "tearing", phen_db)
    assert scored.label_evidence["d3d_tearing_onset_cnn1d/tm_prob.max_valid"] == pytest.approx(0.8)


def test_a_phenomenon_no_model_labels_says_so_rather_than_reporting_nothing(phen_db):
    assert ph.NO_LABEL_MODEL in ph.evidence(OBSERVED_SHOT, "eho", phen_db).caveats


def test_a_negative_claim_becomes_a_caveat_in_the_operators_own_frame(ideate_db):
    _write_tables(
        ideate_db / "db", [],
        [], [_claim(OBSERVED_SHOT, "eho", polarity="neg", snippet="no EHO this shot")],
    )
    db = store.ShotDB.load(ideate_db / "db")
    got = ph.evidence(OBSERVED_SHOT, "eho", db)
    assert "operator log says NOT Edge harmonic oscillation" in got.caveats
    assert got.text_hits == 0  # a denial is not a weaker positive


def test_no_coverage_is_a_caveat_because_absence_is_not_evidence(phen_db):
    # Nothing tearing-shaped ran on shot 101: its detectors exist, and not on this shot.
    got = ph.evidence(FORECAST_SHOT, "tearing", phen_db)
    assert got.coverage is None and got.coverage_state == "unprocessed"
    assert ph.COVERAGE_UNPROCESSED.format(title="Tearing mode") in got.caveats
    # Shot 100 has tokeye_track rows covering 0-6 s, clipped to the window that was searched.
    assert ph.evidence(OBSERVED_SHOT, "tearing", phen_db).coverage == (1.0, 5.0)


def test_a_detector_that_found_nothing_still_records_that_it_looked(phen_db):
    """Shot 200's only ELM row is an `elm_free` interval off the mhr data. That is what separates
    "the detector ran and saw no ELM" from "nobody looked"."""
    quiet = ph.evidence(TEXT_SHOT, "elm", phen_db)
    assert quiet.intervals == ()
    assert quiet.coverage == (1.0, 5.0)  # the 1.0-4.0 row's own t_cov is 0-6 s
    assert quiet.coverage_state == "observed" and quiet.coverage_partial is False
    assert ph.NO_COVERAGE not in quiet.caveats


def test_a_phenomenon_nothing_detects_never_claims_coverage(phen_db):
    """No detector writes `rwm` or `detachment`. Their coverage must stay unknown however many
    rows off however many diagnostics the shot has: "we looked and it was not there" is a claim
    nothing in this database is entitled to make about them."""
    for pid in ("rwm", "detachment"):
        assert ph.registry()[pid].covering_sources == ()
        for shot in (OBSERVED_SHOT, TEXT_SHOT):
            got = ph.evidence(shot, pid, phen_db)
            assert got.coverage is None, (pid, shot)
            assert ph.NO_COVERAGE in got.caveats


def test_min_confidence_drops_the_weakly_scored_events(phen_db):
    got = ph.evidence(OBSERVED_SHOT, "tearing", phen_db, min_confidence=0.5)
    assert [iv.event_id for iv in got.intervals] == [
        "100-tokeye_track-00000", "100-tokeye_track-00001", "100-tokeye_track-00002",
    ]


def test_an_unscored_event_cannot_be_shown_to_clear_a_bar_and_says_so(phen_db):
    """Shot 100's ELM row carries no confidence -- `tokeye_transient` scores a burst and leaves a
    lone ELM unscored. It is kept at the default bar and dropped above it, and the drop is said
    out loud rather than looking like "there was no ELM"."""
    kept = ph.evidence(OBSERVED_SHOT, "elm", phen_db)
    assert [iv.event_id for iv in kept.intervals] == ["100-tokeye_transient-00000"]
    got = ph.evidence(OBSERVED_SHOT, "elm", phen_db, min_confidence=0.5)
    assert got.intervals == ()
    assert any("no confidence" in c for c in got.caveats)


def test_the_event_refs_point_back_at_the_rows_they_summarise(phen_db):
    refs = ph.evidence(OBSERVED_SHOT, "tearing", phen_db).refs
    assert refs[0].event_id == "100-tokeye_track-00000"
    # The SOURCE's own word, not ideate's id: the detector never said "tearing".
    assert refs[0].phenomenon == "coherent_mode"
    assert refs[0].shot == OBSERVED_SHOT


# ---------------------------------------------------------------------------------------- locate


def test_observed_outranks_forecast_only_which_outranks_text_only(phen_db):
    hits = ph.locate("tearing", phen_db, 10)
    assert [h.shot for h in hits] == [OBSERVED_SHOT, FORECAST_SHOT, TEXT_SHOT]
    assert SILENT_SHOT not in [h.shot for h in hits]  # nothing said anything: not a hit


def test_a_text_only_hit_is_capped_and_labelled_as_one(phen_db):
    from labelmaker.events.lexicon import TEXT_ONLY_CEILING

    hit = next(h for h in ph.locate("tearing", phen_db, 10) if h.shot == TEXT_SHOT)
    assert hit.score <= TEXT_ONLY_CEILING == 0.25
    assert ph.TEXT_ONLY in hit.caveats
    assert hit.intervals == [] and hit.forecasts == []


def test_a_forecast_only_hit_keeps_its_forecasts_in_their_own_field(phen_db):
    hit = next(h for h in ph.locate("tearing", phen_db, 10) if h.shot == FORECAST_SHOT)
    assert hit.intervals == []
    assert len(hit.forecasts) == 2
    assert ph.FORECAST_ONLY in hit.caveats


def test_avoid_drops_the_shots_a_detector_saw_the_avoided_phenomenon_on(phen_db):
    kept = [h.shot for h in ph.locate("tearing", phen_db, 10, avoid=["phenomenon:elm"])]
    assert OBSERVED_SHOT not in kept  # a detector saw an ELM at 2.2 s
    assert FORECAST_SHOT in kept and TEXT_SHOT in kept


def test_avoid_keeps_a_shot_nothing_looked_at_and_says_why(phen_db):
    hits = {h.shot: h for h in ph.locate("tearing", phen_db, 10, avoid=["phenomenon:elm"])}
    unknown = hits[FORECAST_SHOT]
    assert any("--avoid phenomenon:elm" in c for c in unknown.caveats)
    # Shot 200's ELM detector ran and found none: a real negative, and no caveat about it.
    assert not any("--avoid" in c for c in hits[TEXT_SHOT].caveats)


def test_avoid_rejects_a_token_that_is_not_a_phenomenon(phen_db):
    with pytest.raises(ph.PhenomenaError, match="not a phenomenon"):
        ph.locate("tearing", phen_db, 5, avoid=["phenomenon:banana"])


def test_min_confidence_reaches_the_ranking(phen_db):
    hit = next(h for h in ph.locate("tearing", phen_db, 10, min_confidence=0.5)
               if h.shot == OBSERVED_SHOT)
    assert len(hit.intervals) == 3


def test_constraints_narrow_the_candidates(phen_db):
    from ideate.schema import Range

    hits = ph.locate("tearing", phen_db, 10, constraints={"ip_mean": Range(lo=1.3e6)})
    assert [h.shot for h in hits] == [TEXT_SHOT]  # 100 and 101 are under 1.3 MA


def test_the_quote_is_one_logbook_entry_and_is_never_spliced(phen_db):
    hit = next(h for h in ph.locate("tearing", phen_db, 10) if h.shot == OBSERVED_SHOT)
    rec = phen_db.get(OBSERVED_SHOT)
    assert hit.quote_role in {e.role for e in rec.human.log_entries}
    assert any(hit.quote in e.text for e in rec.human.log_entries)


def test_a_shot_with_no_quotable_entry_falls_back_to_a_flagged_claim_snippet(ideate_db):
    _write_tables(ideate_db / "db", [], [], [_claim(TEXT_SHOT, "tearing", snippet="tearing at 3 s")])
    db = store.ShotDB.load(ideate_db / "db")
    # `write_db`'s entries are SESSION_LEADER, which IS quotable, so blank the record's log.
    rec = db.get(TEXT_SHOT)
    rec.human.log_entries = []
    db.shots.loc[TEXT_SHOT, "record_json"] = rec.model_dump_json()
    hit = next(h for h in ph.locate("tearing", db, 10) if h.shot == TEXT_SHOT)
    assert hit.quote_role == "claim_snippet"
    assert hit.quote == "tearing at 3 s"


def test_the_actuator_field_reports_the_segments_own_columns(phen_db):
    hit = next(h for h in ph.locate("tearing", phen_db, 10) if h.shot == OBSERVED_SHOT)
    assert set(hit.actuators_at_onset) == set(ph.ACTUATOR_COLUMNS)
    assert hit.actuators_at_onset["pnbi_total_mean"] == pytest.approx(5.0e6)
    # The fixture records no I-coil current at all: unavailable, and not 0.0.
    assert hit.actuators_at_onset["irmp_total_peak"] is None


def test_every_hit_names_the_evidence_classes_it_lacks_and_only_those(phen_db):
    """Caveats are informative, not boilerplate. Shot 100 has an observation, a label and the
    operators' word, all inside a recorded coverage window, so it carries none -- which is what
    makes the ones on the other two hits mean something."""
    hits = {h.shot: h for h in ph.locate("tearing", phen_db, 10)}
    assert hits[OBSERVED_SHOT].caveats == []
    for shot, hit in hits.items():
        if not hit.intervals:
            assert ph.NO_OBSERVED in hit.caveats, shot
        if not hit.text_snippets:
            assert ph.NO_TEXT in hit.caveats, shot
        if hit.coverage is None:
            assert hit.coverage_state != "observed", shot
            assert any(
                c.endswith("absence is not evidence") or "unmeasured" in c for c in hit.caveats
            ), shot


def test_the_hit_carries_the_shots_identity_for_a_reader_to_follow_up(phen_db):
    hit = next(h for h in ph.locate("tearing", phen_db, 10) if h.shot == OBSERVED_SHOT)
    assert hit.run_id == "r1"
    assert hit.mp_title == "a mini proposal"
    assert hit.total_duration_s == pytest.approx(0.5 + 0.4 + 0.2 + 0.1)


# ------------------------------------------------------------------------------------- the store


def test_the_store_loads_the_three_label_tables_when_they_are_there(phen_db):
    assert len(phen_db.events) == 10
    assert len(phen_db.labels_wide) == 2
    assert len(phen_db.text_claims) == 3


def test_the_store_tolerates_their_absence_with_empty_typed_frames(ideate_db):
    from labelmaker.events import schema as events_schema

    db = store.ShotDB.load(ideate_db / "db")
    assert db.events.empty and list(db.events.columns) == list(events_schema.COLUMNS)
    assert db.labels_wide.empty and "max_valid" in db.labels_wide.columns
    assert db.text_claims.empty and "polarity" in db.text_claims.columns
    # And an empty table is queryable without a "is there a table?" check first.
    assert ph.evidence(OBSERVED_SHOT, "tearing", db).intervals == ()


# --------------------------------------------------------------------------------------- the CLI


def test_the_cli_prints_the_hits_as_json(phen_db, ideate_db, capsys):
    code = cli.main(["phenomenon", "tearing mode", "--n", "5", "--json"])
    assert code == 0
    doc = json.loads(capsys.readouterr().out)
    assert [h["shot"] for h in doc] == [OBSERVED_SHOT, FORECAST_SHOT, TEXT_SHOT]
    first = doc[0]
    assert {"shot", "score", "intervals", "forecasts", "caveats", "label_evidence"} <= set(first)
    assert first["intervals"][0]["evidence_kind"] == "detector"
    assert doc[1]["forecasts"][0]["evidence_kind"] == "forecast"


def test_the_cli_says_nothing_resolved_rather_than_printing_an_empty_table(phen_db, capsys):
    code = cli.main(["phenomenon", "a quiet ohmic discharge"])
    assert code == 2
    err = capsys.readouterr().err
    assert "no phenomenon resolved" in err
    assert "eho (Edge harmonic oscillation)" in err


def test_the_cli_table_names_the_evidence_class_of_every_row(phen_db, capsys):
    assert cli.main(["phenomenon", "tearing mode", "--n", "5"]) == 0
    out = capsys.readouterr().out
    assert "resolved: tearing (Tearing mode) 1.00" in out
    assert "obs x4" in out and "forecast x2" in out and "text" in out
    assert ph.TEXT_ONLY in out


def test_the_cli_lists_the_registry(capsys):
    assert cli.main(["phenomenon", "--list"]) == 0
    out = capsys.readouterr().out
    assert "eho          Edge harmonic oscillation" in out
    assert "edge harmonic oscillation" in out  # the aliases, from labelmaker's file


def test_the_cli_passes_avoid_through(phen_db, capsys):
    assert cli.main(
        ["phenomenon", "tearing mode", "--json", "--avoid", "phenomenon:elm"]
    ) == 0
    doc = json.loads(capsys.readouterr().out)
    assert OBSERVED_SHOT not in [h["shot"] for h in doc]


# ===================================================================== fix loop 1 (review I9a)
#
# One section per numbered finding of `.superpowers/sdd/task-I9a-review.md`. They are all the
# same shape as the tests above: a claim this module must not make, and the fixture that would
# have let it make one.


def _db_with(ideate_db: Path, events: list[dict], labels=(), claims=()) -> store.ShotDB:
    _write_tables(ideate_db / "db", events, list(labels), list(claims))
    return store.ShotDB.load(ideate_db / "db")


def _elm_clock(shot: int, event_id: str, cov: tuple[float, float]) -> dict:
    """One `elm_free` row: the ELM clock saying it READ the mhr data over `cov`, and found no
    ELM in it. The row's own span and its coverage span are the same stretch."""
    return _event(
        shot, event_id, source="elm_clock", evidence_kind="heuristic", phenomenon="elm_free",
        t0_s=cov[0], t1_s=cov[1], confidence=np.nan, attrs={"n_elms_inside": 0},
        t_cov0_s=cov[0], t_cov1_s=cov[1],
    )


# ------------------------------------------------------- finding 1: coverage and the window


def test_coverage_that_misses_the_window_is_not_coverage_of_the_window(ideate_db):
    """The ELM clock read 0.0-0.9 s. The question is about the flat top, 1.0-5.0 s. Nobody
    looked at the flat top for ELMs, and a coverage number that says otherwise turns a gap in
    the data into "we looked and there were none"."""
    db = _db_with(ideate_db, [_elm_clock(TEXT_SHOT, "200-elm_clock-00000", (0.0, 0.9))])
    got = ph.evidence(TEXT_SHOT, "elm", db)
    assert got.coverage is None
    assert got.coverage_state == "uncovered"
    assert ph.COVERAGE_OUTSIDE_WINDOW.format(
        title="Edge localised mode", segment="flat_top"
    ) in got.caveats


def test_avoid_keeps_and_caveats_a_shot_whose_coverage_missed_the_window(ideate_db):
    db = _db_with(
        ideate_db,
        [_elm_clock(TEXT_SHOT, "200-elm_clock-00000", (0.0, 0.9))],
        claims=[_claim(TEXT_SHOT, "tearing")],
    )
    hit = next(h for h in ph.locate("tearing", db, 10, avoid=["phenomenon:elm"])
               if h.shot == TEXT_SHOT)
    assert ph.AVOID_UNCOVERED.format(
        token="phenomenon:elm", title="Edge localised mode"
    ) in hit.caveats


def test_partial_coverage_of_the_window_is_reported_as_partial(ideate_db):
    db = _db_with(
        ideate_db,
        [_elm_clock(TEXT_SHOT, "200-elm_clock-00000", (1.0, 3.0))],
        claims=[_claim(TEXT_SHOT, "tearing")],
    )
    got = ph.evidence(TEXT_SHOT, "elm", db)
    assert got.coverage == (1.0, 3.0)
    assert got.coverage_state == "observed"
    assert got.coverage_partial is True
    hit = next(h for h in ph.locate("tearing", db, 10, avoid=["phenomenon:elm"])
               if h.shot == TEXT_SHOT)
    assert any("--avoid phenomenon:elm" in c for c in hit.caveats)


def test_coverage_of_the_whole_window_is_a_real_negative_and_says_nothing(ideate_db):
    db = _db_with(
        ideate_db,
        [_elm_clock(TEXT_SHOT, "200-elm_clock-00000", (0.0, 6.0))],
        claims=[_claim(TEXT_SHOT, "tearing")],
    )
    got = ph.evidence(TEXT_SHOT, "elm", db)
    assert got.coverage == (1.0, 5.0)  # clipped to the window the question is about
    assert got.coverage_state == "observed" and got.coverage_partial is False
    hit = next(h for h in ph.locate("tearing", db, 10, avoid=["phenomenon:elm"])
               if h.shot == TEXT_SHOT)
    assert not any("--avoid" in c for c in hit.caveats)


def test_the_four_coverage_states_are_distinguished(ideate_db):
    db = _db_with(ideate_db, [_elm_clock(TEXT_SHOT, "200-elm_clock-00000", (0.0, 0.9))])
    assert ph.COVERAGE_STATES == ("unindexed", "unprocessed", "uncovered", "observed")
    # nothing in the pipeline looks for an rwm at all
    assert ph.evidence(TEXT_SHOT, "rwm", db).coverage_state == "unindexed"
    # the elm detectors exist but never ran on this shot
    assert ph.evidence(OBSERVED_SHOT, "elm", db).coverage_state == "unprocessed"
    assert ph.evidence(TEXT_SHOT, "elm", db).coverage_state == "uncovered"


# ------------------------------------------------------- finding 7: the union, not the hull


def test_two_disjoint_coverage_stretches_are_not_one_long_one(ideate_db):
    db = _db_with(ideate_db, [
        _elm_clock(TEXT_SHOT, "200-elm_clock-00000", (0.0, 1.5)),
        _elm_clock(TEXT_SHOT, "200-elm_clock-00001", (4.0, 6.0)),
    ])
    got = ph.evidence(TEXT_SHOT, "elm", db)
    assert got.coverage_windows == ((1.0, 1.5), (4.0, 5.0))
    assert got.coverage == (1.0, 5.0)  # the hull, and it is labelled as one
    assert ph.COVERAGE_GAPS.format(
        title="Edge localised mode", segment="flat_top", n=1
    ) in got.caveats
    assert got.coverage_partial is True


# --------------------------- finding 1 / cross-workstream: db/event_sources.parquet


def test_coverage_comes_from_the_source_table_when_the_database_has_one(ideate_db):
    """`db/event_sources.parquet` (branch `recommender-fix`) records what RAN, including the
    detectors that emitted nothing. When it is there it is the answer; a `skipped` row is not
    coverage, it is a detector that was not run."""
    db = _db_with(ideate_db, [])
    db.event_sources = pd.DataFrame(
        [
            {"shot": TEXT_SHOT, "source": "elm_clock", "status": "ok",
             "t_cov0_s": 0.0, "t_cov1_s": 6.0},
            {"shot": OBSERVED_SHOT, "source": "elm_clock", "status": "skipped",
             "t_cov0_s": np.nan, "t_cov1_s": np.nan},
        ]
    )
    assert ph.evidence(TEXT_SHOT, "elm", db).coverage == (1.0, 5.0)
    assert ph.evidence(OBSERVED_SHOT, "elm", db).coverage_state == "unprocessed"


def test_the_hit_carries_the_coverage_state(phen_db):
    hits = {h.shot: h for h in ph.locate("tearing", phen_db, 10)}
    assert hits[OBSERVED_SHOT].coverage_state == "observed"
    assert hits[FORECAST_SHOT].coverage_state == "unprocessed"


# ---------------------------------- finding 2: a transient detector is not an ELM detector


def test_a_transient_detection_is_reported_as_a_transient_and_not_as_an_elm(phen_db):
    """`tokeye_transient` is a burst detector. labelmaker's own module says a sawtooth crash and
    a disruption precursor are transient too, so its `phenomenon="elm"` is a claim to weigh, not
    an ELM sighting to repeat."""
    seen = ph.evidence(OBSERVED_SHOT, "elm", phen_db)
    assert [iv.source for iv in seen.intervals] == ["tokeye_transient"]
    assert ph.TRANSIENT_NOT_CLASSIFIED in seen.caveats
    # A class-specific detector says nothing of the kind.
    assert ph.TRANSIENT_NOT_CLASSIFIED not in ph.evidence(
        OBSERVED_SHOT, "tearing", phen_db
    ).caveats


def test_the_transient_rule_weighs_less_than_a_class_specific_detector():
    reg = ph.registry()
    (elm_rule,) = reg["elm"].events
    (saw_rule,) = reg["sawtooth"].events
    assert elm_rule.source == "tokeye_transient"
    assert elm_rule.weight < saw_rule.weight == 1.0
    assert elm_rule.caveat == ph.TRANSIENT_NOT_CLASSIFIED


def test_the_event_term_counts_the_rules_weight_not_the_row_count(phen_db):
    ev = ph.evidence(OBSERVED_SHOT, "elm", phen_db)
    (rule,) = ph.registry()["elm"].events
    assert len(ev.intervals) == 1
    assert ev.event_weight == pytest.approx(rule.weight)
    assert ph.score(ev, ph.DEFAULT_WEIGHTS, 3.0) == pytest.approx(
        1.0 - math.exp(-rule.weight / 3.0)
    )


def test_an_avoid_drop_says_what_the_evidence_it_dropped_on_was(phen_db):
    notes: list[str] = []
    kept = ph.locate("tearing", phen_db, 10, avoid=["phenomenon:elm"], notes=notes)
    assert OBSERVED_SHOT not in [h.shot for h in kept]
    assert any("dropped 1 shot" in n for n in notes)
    assert any(ph.TRANSIENT_NOT_CLASSIFIED in n for n in notes)


def test_the_registry_and_the_docs_say_what_a_classified_elm_would_need():
    """ideate cannot fix this: labelmaker has to publish an `elm` point family from the ELM
    clock's own peaks. Naming the requirement where the rule is, and in the docs, is the part
    that is ideate's."""
    yaml_text = (REPO / "configs" / "ideate" / "phenomena.yaml").read_text()
    docs = (REPO / "docs" / "IDEATE.md").read_text()
    for text in (yaml_text, docs):
        assert "point-event family" in text.lower()
        assert "elm_clock" in text


# ----------------------------------------------------- finding 3: the saturation is pinned


def test_the_saturation_curve_is_one_minus_exp_and_not_a_step():
    assert ph.sat(0) == 0.0
    assert ph.sat(1) == pytest.approx(1.0 - math.exp(-1.0 / 3.0))
    assert ph.sat(3) == pytest.approx(1.0 - math.exp(-1.0))
    assert ph.sat(6) == pytest.approx(1.0 - math.exp(-2.0))
    assert ph.sat(1) < ph.sat(2) < ph.sat(3) < ph.sat(30) < 1.0
    assert ph.sat(2, 1.0) == pytest.approx(1.0 - math.exp(-2.0))


def test_the_score_is_the_four_terms_the_config_documents(phen_db):
    """Written out in `math.exp`, not in `ph.sat`, so a stubbed saturation fails it."""
    ev = ph.evidence(OBSERVED_SHOT, "tearing", phen_db)
    assert ev.max_label_p == pytest.approx(0.8)
    assert ev.event_weight == pytest.approx(4.0)
    assert ev.text_hits == 1
    expected = 0.8 + (1.0 - math.exp(-4.0 / 3.0)) + 0.5 * math.tanh(0.5)
    assert ph.score(ev, ph.DEFAULT_WEIGHTS, 3.0) == pytest.approx(expected)


# ------------------------------------ findings 4 and 5: what a label has to say to be evidence


def test_a_label_the_model_scored_below_the_floor_is_reported_and_not_counted(ideate_db):
    db = _db_with(ideate_db, [], labels=[_label_row(OBSERVED_SHOT, "tm_prob", max_valid=0.0104)])
    ev = ph.evidence(OBSERVED_SHOT, "tearing", db)
    assert ev.label_evidence["d3d_tearing_onset_cnn1d/tm_prob.max_valid"] == pytest.approx(0.0104)
    assert ev.max_label_p is None  # the model ran and said no; that is not evidence of a yes
    assert ph.LABEL_BELOW_FLOOR.format(
        key="d3d_tearing_onset_cnn1d/tm_prob", p=0.0104, floor=ph.DEFAULT_LABEL_FLOOR
    ) in ev.caveats


def test_the_model_saying_no_does_not_outrank_a_shot_the_operators_described(ideate_db):
    db = _db_with(
        ideate_db,
        [],
        labels=[_label_row(FORECAST_SHOT, "tm_prob", max_valid=0.0104)],
        claims=[_claim(TEXT_SHOT, "tearing")],
    )
    assert [h.shot for h in ph.locate("tearing", db, 10)] == [TEXT_SHOT]


def test_a_published_operating_point_is_that_labels_floor(ideate_db, tmp_path):
    def raise_thr(doc):
        doc["phenomena"]["tearing"]["labels"][0]["thr"] = 0.9

    entry = ph.registry(_registry_file(tmp_path, raise_thr))["tearing"]
    db = _db_with(ideate_db, [], labels=[_label_row(OBSERVED_SHOT, "tm_prob", max_valid=0.8)])
    assert ph.evidence(OBSERVED_SHOT, entry, db).max_label_p is None
    # the shipped registry publishes no operating point, so the declared floor decides
    assert ph.evidence(OBSERVED_SHOT, "tearing", db).max_label_p == pytest.approx(0.8)


def test_the_label_only_caveat_says_what_the_model_actually_scored(ideate_db):
    db = _db_with(ideate_db, [], labels=[_label_row(FORECAST_SHOT, "tm_prob", max_valid=0.62)])
    hit = next(h for h in ph.locate("tearing", db, 10) if h.shot == FORECAST_SHOT)
    assert ph.LABEL_ONLY.format(p=0.62) in hit.caveats
    assert any("0.620" in c for c in hit.caveats)  # the number, not just "a model's score"


# ------------------------------------- finding 6: the config states the order the code sorts by


def test_the_config_and_the_docs_state_the_ranking_the_code_implements():
    assert ph.OBSERVED > ph.LABELLED > ph.FORECAST > ph.DATABASE > ph.TEXTUAL
    for path in (REPO / "configs" / "ideate" / "retrieval.yaml", REPO / "docs" / "IDEATE.md"):
        text = " ".join(path.read_text().replace("#", " ").split())
        assert ph.RANKING_SENTENCE in text, path


# -------------------------------------- finding 8: "observed" is an allow-list, not a default


def test_a_row_of_an_unrecognised_evidence_kind_is_neither_observed_nor_forecast(ideate_db):
    db = _db_with(
        ideate_db, [_event(OBSERVED_SHOT, "100-tokeye_track-09999", evidence_kind="model")]
    )
    ev = ph.evidence(OBSERVED_SHOT, "tearing", db)
    assert ph.OBSERVED_KINDS == frozenset({"detector", "heuristic"})
    assert ev.intervals == () and ev.forecasts == ()
    assert ph.UNCLASSIFIED_KIND.format(n=1, kinds="model") in ev.caveats
    assert ev.coverage_state == "unprocessed"  # and it donates no coverage either


# ---------------------------------------------- finding 9: a broadband smear is not a mode


def test_a_broadband_smear_is_not_a_mode_in_the_band(ideate_db):
    db = _db_with(ideate_db, [
        _event(OBSERVED_SHOT, "100-tokeye_track-00000",
               attrs={"f_centroid_khz": 10.0, "bandwidth_khz": 2.0}),
        _event(OBSERVED_SHOT, "100-tokeye_track-00001", t0_s=3.0, t1_s=3.2,
               attrs={"f_centroid_khz": 10.0, "bandwidth_khz": 60.0}),
    ])
    ids = [iv.event_id for iv in ph.evidence(OBSERVED_SHOT, "tearing", db).intervals]
    assert ids == ["100-tokeye_track-00000"]
    assert ph.registry()["tearing"].events[0].max_bandwidth_khz == 30.0


def test_a_track_that_records_no_bandwidth_is_not_rejected_for_not_recording_one(ideate_db):
    db = _db_with(ideate_db, [_event(OBSERVED_SHOT, "100-tokeye_track-00000")])
    assert len(ph.evidence(OBSERVED_SHOT, "tearing", db).intervals) == 1


# ------------------------------------ finding 10: a forecast is not present-tense label evidence


def test_a_forecast_label_may_not_be_listed_as_label_evidence(tmp_path):
    def add(doc):
        doc["phenomena"]["tearing"]["labels"].append(
            {"slug": "d3d_tearing_time_to_event_dsm", "label": "tm_risk_1s",
             "thr": None, "weight": 1.0}
        )

    with pytest.raises(ph.PhenomenaError, match="forecast"):
        ph.registry(_registry_file(tmp_path, add))


# ---------------------------------------------- finding 11: what a quote beside a hit claims


def _with_log(db: store.ShotDB, shot: int, texts: list[str]) -> None:
    from ideate.schema import LogEntry

    rec = db.get(shot)
    rec.human.log_entries = [LogEntry(role="PHYSICS_OPERATOR", text=t) for t in texts]
    db.shots.loc[shot, "record_json"] = rec.model_dump_json()


def test_the_quote_prefers_a_logbook_entry_that_names_the_phenomenon(ideate_db):
    db = _db_with(ideate_db, [], claims=[_claim(TEXT_SHOT, "tearing")])
    _with_log(db, TEXT_SHOT, [
        "Restore 184833 Result: Dud trip on LM at appx 2.5 sec.",
        "clear 2/1 tearing mode from 3 s onwards, locked by 4",
    ])
    hit = next(h for h in ph.locate("tearing", db, 10) if h.shot == TEXT_SHOT)
    assert "tearing mode" in hit.quote
    assert not any("does not mention" in c for c in hit.caveats)


def test_a_quote_that_does_not_mention_the_phenomenon_says_so(ideate_db):
    db = _db_with(ideate_db, [], claims=[_claim(TEXT_SHOT, "tearing")])
    _with_log(db, TEXT_SHOT, ["Restore 184833 Result: Dud trip on LM at appx 2.5 sec."])
    hit = next(h for h in ph.locate("tearing", db, 10) if h.shot == TEXT_SHOT)
    assert hit.quote.startswith("Restore 184833")
    assert ph.QUOTE_UNRELATED.format(title="Tearing mode") in hit.caveats


# ------------------------------------------------------------------------------- the nits


def test_shorten_takes_a_width_so_the_table_does_not_cut_mid_word():
    from ideate.retrieval import describe as describe_mod

    text = ("Alex restores all GPU feedback settings from 186533 Error field correction "
            "enabled at 2 s")
    got = describe_mod.shorten(text, 60)
    kept = got.removesuffix(" ...")
    assert len(got) <= 64
    assert text.split()[: len(kept.split())] == kept.split()  # whole words, verbatim


def test_the_table_says_when_every_event_row_in_the_database_is_a_forecast(phen_db, capsys):
    assert cli.main(["phenomenon", "tearing mode", "--n", "5"]) == 0
    out = capsys.readouterr().out
    # phen_db's events are not all forecasts, so the line must NOT be there
    assert ph.ALL_FORECASTS.format(n=10) not in out


def test_a_database_of_nothing_but_forecasts_says_so_on_screen(ideate_db, capsys):
    _write_tables(ideate_db / "db", [
        _event(FORECAST_SHOT, "101-label_forecast-00000", source="label_forecast",
               evidence_kind="forecast", phenomenon="tearing", horizon_s=1.0, diag="",
               confidence=0.35, attrs={"label": "tm_risk_1s"}),
    ], [], [])
    assert cli.main(["phenomenon", "tearing mode", "--n", "5"]) == 0
    assert ph.ALL_FORECASTS.format(n=1) in capsys.readouterr().out


def test_the_same_phenomenon_named_twice_is_one_constraint(phen_db, monkeypatch):
    seen: list[tuple[int, str]] = []
    real = ph.evidence

    def counted(shot, entry, db, segment="flat_top", **kw):
        seen.append((int(shot), entry if isinstance(entry, str) else entry.id))
        return real(shot, entry, db, segment, **kw)

    monkeypatch.setattr(ph, "evidence", counted)
    kept = ph.locate("tearing", phen_db, 10, avoid=["elm", "phenomenon:elm"])
    assert OBSERVED_SHOT not in [h.shot for h in kept]
    assert len(seen) == len(set(seen))  # each (shot, phenomenon) read once, not once per token
    assert sum(1 for c in kept[0].caveats if "--avoid phenomenon:elm" in c) == 1


# ------------------------- L-D1: the curated RWM tables reach the registry as MEMBERSHIP only


def _label_tables(root: Path, shots: list[int], stem: str = "rwm_onsets_2017") -> Path:
    """A `data/labels`-shaped fixture: one manifest, one CSV, the stems the registry names."""
    (root / "resistive_wall_mode/format").mkdir(parents=True, exist_ok=True)
    (root / "tables.yaml").write_text(yaml.safe_dump({
        "version": 1,
        "tables": [{
            "stem": stem,
            "raw_file": f"{stem}.csv", "format_stem": stem, "converter": "csv",
            "made_at": "2026-09-13T00:00:00Z",
            "dir": "resistive_wall_mode",
            "phenomenon": "rwm",
            "kind": "point",
            "shot_col": "SHOT",
            "t_col": "ONSET_TIME",
            "t_units": "ms",
            "attr_cols": ["NTOR"],
            "attr_types": {"NTOR": "int"},
            "provenance": "a fixture",
        }],
    }), encoding="utf-8")
    import csv

    with (root / "resistive_wall_mode/format" / f"{stem}.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["shot", "t0_s", "t1_s", "phenomenon", "evidence_kind",
                         "source", "confidence", "attrs"])
        for shot, ms in [(s, 2000) for s in shots]:
            writer.writerow([shot, ms / 1000, ms / 1000, "rwm", "database",
                             f"database:{stem}", "", json.dumps({"NTOR": 1,
                                                                "table": stem})])
    return root


@pytest.fixture
def rwm_tables(tmp_path, monkeypatch):
    """The registry's `tables:` resolution pointed at a fixture, caches cleared both ways."""
    def install(shots: list[int]) -> Path:
        root = _label_tables(tmp_path / "labels", shots)
        monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
        ph._table_shots.cache_clear()
        return root

    yield install
    ph._table_shots.cache_clear()


def test_the_shipped_registry_names_both_rwm_tables_and_the_manifest_knows_them():
    """The registry entry and `data/labels/tables.yaml` are two files that have to agree, and
    nothing else checks that they do: a stem typo here is a silent empty set, not an error."""
    from labelmaker.events import databases as label_tables

    entry = ph.registry()["rwm"].database
    assert entry is not None
    assert list(entry["tables"]) == ["rwm_onsets_2017", "rwm_onsets_2024"]
    stems = {spec.stem for spec in label_tables.load_manifest()}
    assert set(entry["tables"]) <= stems


def test_a_shot_a_curated_table_names_is_in_the_database_and_one_it_omits_is_not(rwm_tables):
    rwm_tables([TEXT_SHOT])
    rwm = ph.registry()["rwm"]
    assert ph._in_database(rwm, TEXT_SHOT) is True
    assert ph._in_database(rwm, SILENT_SHOT) is False


def test_a_database_only_shot_ranks_below_a_forecast_and_never_as_observed(
    ideate_db, rwm_tables,
):
    """The whole point of the fourth evidence class. A curated table names shot 200 and nothing
    else says a word about RWM on it: it must come back, because a human list IS evidence, and
    it must come back in the DATABASE class carrying the caveat that says so -- not in the
    observed class, where a reader would take it for a measurement."""
    rwm_tables([TEXT_SHOT])
    db = _db_with(ideate_db, [])
    hit = next(h for h in ph.locate("rwm", db, 10) if h.shot == TEXT_SHOT)
    assert ph.DATABASE_ONLY in hit.caveats
    assert hit.intervals == []
    # Membership is not coverage: nobody recorded which interval was examined.
    assert hit.coverage is None
    assert hit.coverage_state != "observed"
    ev = ph.evidence(TEXT_SHOT, "rwm", db)
    assert ev.in_database is True
    assert ph._tier(ev) == ph.DATABASE < ph.FORECAST


def test_a_stem_no_manifest_declares_is_an_empty_set_and_not_a_crash(rwm_tables):
    """Curated tables are optional data. A database built on a machine without them must still
    rank, and a stem that has been renamed must not take the whole query down with it."""
    rwm_tables([TEXT_SHOT], )
    assert ph._table_shots(("no_such_table",)) == frozenset()
    assert ph._table_shots(()) == frozenset()
