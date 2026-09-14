"""The phenomenon registry, the evidence classes, and what `ideate phenomenon` may claim.

Everything here is about ONE distinction repeated at four scales: an observation, a model's
opinion about the present, a model's opinion about the future, and a sentence somebody typed are
four different claims, and this module's job is to never let one be reported as another.
"""

from __future__ import annotations

import json
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
    return store.ShotDB.load(ideate_db / "db")


# -------------------------------------------------------------------------- registry validation


def test_the_registry_names_exactly_labelmakers_round_one_phenomena():
    from labelmaker.events.lexicon import PHENOMENON_IDS

    reg = ph.registry()
    assert set(reg) == set(PHENOMENON_IDS)
    # And the names come from labelmaker's file, not from a second copy of them here.
    assert reg["eho"].aliases == tuple(ph._lexicon()["eho"].aliases)
    assert reg["elm"].exclude == tuple(ph._lexicon()["elm"].negatives)


def test_fast_ion_is_a_text_only_topic_with_no_detector_and_no_coverage():
    """The id the I11 review asked for instead of aliasing fast-ion words onto `ae`.

    Everything that could let it be read as a detector observation is empty, and this test is
    what keeps it that way: an event rule added here later would make `locate("fast_ion")` claim
    a diagnostic saw something, and no diagnostic looks for "fast-ion physics".
    """
    fast = ph.registry()["fast_ion"]
    assert fast.labels == () and fast.events == () and fast.forecasts == ()
    assert fast.coverage_sources == () and fast.band_khz is None
    assert fast.diags == () and fast.requires_group == ()
    assert "fida" in fast.aliases and "beam ion" in fast.aliases
    # And the mode keeps its own words, so an `ae` hit still means an Alfven eigenmode.
    assert "fida" not in ph.registry()["ae"].aliases


def test_resolve_separates_the_fast_ion_topic_from_the_alfven_mode():
    assert [pid for pid, _ in ph.resolve("beam ion losses measured with FIDA")] == ["fast_ion"]
    assert [pid for pid, _ in ph.resolve("TAE bursts during the current ramp")] == ["ae"]
    both = {pid for pid, _ in ph.resolve("fast ion drive of a TAE")}
    assert both == {"ae", "fast_ion"}


def test_every_registry_id_resolves_to_itself():
    """`ideate phenomenon <query>` matches TEXT, and its refusal prints the registry's ids.

    So an id that does not appear in its own alias list is advertised by the error message and
    then rejected when it is typed -- which is what `fast_ion` did, and what `lh` did before it
    (the I11 re-review found the first; this test found the second). The rule is the cheap one:
    every id the "try one of" line can print is a phrase `resolve` accepts.
    """
    reg = ph.registry()
    for pid in reg:
        resolved = [p for p, _ in ph.resolve(pid)]
        assert resolved and resolved[0] == pid, f"{pid!r} resolves to {resolved}"


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
    assert ph.NO_COVERAGE in ph.evidence(FORECAST_SHOT, "tearing", phen_db).caveats
    # Shot 100 has tokeye_track rows, so the tearing detector's window is on the record.
    assert ph.evidence(OBSERVED_SHOT, "tearing", phen_db).coverage == (0.0, 6.0)


def test_a_detector_that_found_nothing_still_records_that_it_looked(phen_db):
    """Shot 200's only ELM row is an `elm_free` interval off the mhr data. That is what separates
    "the detector ran and saw no ELM" from "nobody looked"."""
    quiet = ph.evidence(TEXT_SHOT, "elm", phen_db)
    assert quiet.intervals == ()
    assert quiet.coverage == (0.0, 6.0)
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
            assert ph.NO_COVERAGE in hit.caveats, shot


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
