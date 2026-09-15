"""Retrieval: channels, RRF fusion, reranking, explanations, and the `shot_design query` command.

The synthetic database here is built by the production path -- `build.records_to_tables`,
`fit_scalar_embedding`, `project_scalar` -- from hand-written ShotRecords, so a change to how a
segment row or an embedding is laid out breaks these tests rather than sliding past them. Text
embeddings are hand-written unit vectors, not MiniLM: what is under test is max(mp, log), the
negative filter and the dedup threshold, none of which are statements about the encoder.

`test_query_*` at the bottom run against the real database when it is mounted, because the
things most likely to be wrong -- a column that does not exist, a unit printed wrong, a channel
that fires on nothing -- are invisible against a fixture built to agree with the code.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from shot_design import cli
from shot_design.retrieval import channels, rank
from shot_design.schema import (
    HumanTier,
    Labels,
    LogEntry,
    Outcome,
    QueryState,
    Range,
    Segment,
    ShotRecord,
)
from shot_design.shotdb import build, text
from shot_design.shotdb.store import ShotDB

REAL_STAGED = Path("/scratch/gpfs/EKOLEMEN/d3d_fusion_data")
REAL_SHOTS = [161172, 161164, 160904, 161173]


def _record(shot: int, run: str, ip: float, pnbi: float, q95: float, text: str, **kw) -> ShotRecord:
    """One shot with a flat top and a ramp-up, so `segment` actually has to select."""
    flat = Segment(
        name="flat_top",
        t0_ms=1000.0,
        t1_ms=5000.0,
        # both stats of the system total and one member, so an actuator key has a column to
        # resolve to whichever stat the registry names for it
        raw={
            "ip_mean": ip,
            "pnbi_total_mean": pnbi,
            "pnbi_total_peak": pnbi * 1.1,
            "pnbi_15L_peak": pnbi * 0.4,
            "bt_mean": 2.0,
        },
        derived={"q95_mean": q95, "betan_mean": kw.pop("betan", 1.8)},
    )
    ramp = Segment(
        name="ramp_up",
        t0_ms=0.0,
        t1_ms=1000.0,
        raw={"ip_mean": ip / 2, "pnbi_total_mean": 0.0, "bt_mean": 2.0},
        derived={"q95_mean": q95 * 2, "betan_mean": 0.3},
    )
    return ShotRecord(
        shot=shot,
        shot_date=dt.date(2015, 1, 13),
        campaign="2014_2015",
        segments=[flat, ramp],
        human=HumanTier(
            run_id=run,
            mp_title=kw.pop("mp_title", "a mini proposal"),
            log_entries=[LogEntry(role="SESSION_LEADER", text=text)],
        ),
        labels=Labels(regime=kw.pop("regime", "H"), operational=kw.pop("operational", set())),
        outcome=Outcome(ip_target_hit=kw.pop("ip_target_hit", True)),
        built_at=dt.datetime(2026, 9, 4, tzinfo=dt.UTC),
        builder_sha="test",
    )


def _db(
    records: list[ShotRecord], text_vecs: dict[int, tuple[list[float], list[float]]] | None = None
) -> ShotDB:
    shapes = {r.shot: {s.name: np.zeros(0, np.float32) for s in r.segments} for r in records}
    shots_df, segments_df, shape_mat = build.records_to_tables(records, shapes)
    X, cols = build._scalar_matrix(segments_df, shape_mat)
    pca = build.fit_scalar_embedding(X, 4)
    pca["feature_cols"], pca["n_shape"] = cols, 0
    dim = 3
    log = np.zeros((len(shots_df), dim), np.float32)
    mp = np.zeros((len(shots_df), dim), np.float32)
    for i, shot in enumerate(shots_df.index):
        # Distinct default vectors: adjacent shots sit 0.5 rad apart (cos 0.88), below the 0.97
        # dedup threshold, so a test about ranking is not silently a test about deduplication.
        spread = [float(np.cos(0.5 * i)), float(np.sin(0.5 * i)), 0.0]
        lv, mv = (text_vecs or {}).get(int(shot), (spread, spread))
        log[i], mp[i] = lv, mv
    emb = {"scalar": build.project_scalar(X, pca), "text_log": log, "text_mp": mp}
    return ShotDB(Path("/nonexistent"), shots_df, segments_df, emb, pca, {}, shape_mat)


@pytest.fixture
def db() -> ShotDB:
    """Six shots over three run days. 100/101 and 200/201 are run mates; 300 stands alone."""
    return _db(
        [
            _record(100, "r1", 1.0e6, 5.0e6, 4.0, "QH-mode at low torque, n=2 rmp"),
            _record(101, "r1", 1.05e6, 5.2e6, 4.1, "repeat of 100, still QH-mode"),
            _record(200, "r2", 1.4e6, 2.0e6, 3.0, "L-mode ohmic reference", regime="L"),
            _record(201, "r2", 1.42e6, 2.1e6, 3.1, "another ohmic shot", regime="L"),
            _record(
                300,
                "r3",
                0.8e6,
                8.0e6,
                6.0,
                "disruptive hybrid",
                regime="H",
                operational={"dud"},
                ip_target_hit=False,
            ),
            _record(301, "r3", 0.81e6, 8.1e6, 6.1, "hybrid again", regime="H"),
        ]
    )


# ------------------------------------------------------------------------------- hard filtering


def test_hard_filter_selects_one_segment(db):
    m = channels.hard_filter(QueryState(segment="flat_top"), db)
    assert set(db.segments.loc[m, "segment"]) == {"flat_top"}
    assert int(m.sum()) == 6


def test_hard_filter_applies_ranges_labels_and_exclusions(db):
    q = QueryState(constraints={"ip_mean": Range(lo=0.9e6, hi=1.1e6)})
    assert sorted(db.segments.loc[channels.hard_filter(q, db), "shot"]) == [100, 101]
    q = QueryState(require_labels={"L"})
    assert sorted(db.segments.loc[channels.hard_filter(q, db), "shot"]) == [200, 201]
    q = QueryState(avoid_labels={"dud"}, exclude_shots={100}, exclude_runs={"r2"})
    assert sorted(db.segments.loc[channels.hard_filter(q, db), "shot"]) == [101, 301]


def test_a_nan_in_a_constrained_column_excludes_the_row_and_is_counted(db):
    db.segments.loc["101:flat_top", "q95_mean"] = np.nan
    q = QueryState(constraints={"q95_mean": Range(lo=0.0, hi=99.0)})
    kept = sorted(db.segments.loc[channels.hard_filter(q, db), "shot"])
    assert 101 not in kept  # "not recorded" is not "within tolerance"
    assert channels.nan_excluded(q, db) == {"q95_mean": 1}


def test_nan_count_is_reported_even_when_it_empties_the_result(db):
    db.segments.loc[db.segments["segment"] == "flat_top", "q95_mean"] = np.nan
    q = QueryState(constraints={"q95_mean": Range(lo=0.0, hi=99.0)})
    assert not channels.hard_filter(q, db).any()
    assert channels.nan_excluded(q, db) == {"q95_mean": 6}


def test_an_unknown_constraint_column_is_an_error_not_an_empty_result(db):
    with pytest.raises(KeyError, match="unknown column"):
        channels.hard_filter(QueryState(constraints={"nope_mean": Range(lo=1.0)}), db)


# ------------------------------------------------------------------------------------ channels


def test_scalar_knn_from_a_reference_shot_puts_the_shot_itself_first(db):
    hits = channels.scalar_knn(QueryState(ref_shot=100), db)
    assert hits[0][0] == "100:flat_top"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[1][0] == "101:flat_top"  # its run mate is the nearest other shot


def test_scalar_knn_from_constraint_midpoints_needs_no_reference_shot(db):
    q = QueryState(constraints={"ip_mean": Range(lo=1.35e6, hi=1.45e6)})
    hits = channels.scalar_knn(q, db)
    assert {s for s, _ in hits} == {"200:flat_top", "201:flat_top"}  # the filter admits only these


def test_scalar_knn_is_silent_when_the_query_names_nothing_it_can_use(db):
    assert channels.scalar_knn(QueryState(), db) == []
    assert channels.scalar_knn(QueryState(text="words only"), db) == []


def test_query_z_is_zero_off_the_dimensions_the_query_named(db):
    z, named = channels.query_z(QueryState(constraints={"ip_mean": Range(lo=1.0e6, hi=1.0e6)}), db)
    cols = db.pca["feature_cols"]
    assert named == ["ip_mean"]
    assert z[cols.index("ip_mean")] != 0.0
    assert not np.any(np.delete(z, cols.index("ip_mean")))


def test_scalar_knn_is_empty_not_an_indexerror_on_an_unfitted_pca(db):
    """`build.build` fits no PCA on <= 2 rows but still assigns `feature_cols` onto `empty_pca()`,
    so a specs query on a two-shot database reached `mean[i]` on an empty array."""
    db.pca = {**build.empty_pca(), "feature_cols": db.pca["feature_cols"], "n_shape": 0}
    db.emb["scalar"] = np.zeros((len(db.segments), 1), np.float32)  # what project_scalar writes
    q = QueryState(constraints={"ip_mean": Range(lo=1.0e6, hi=1.4e6)})
    assert channels.query_z(q, db) is None
    assert channels.scalar_knn(q, db) == []


def test_an_actuator_name_resolves_through_the_registry(db):
    assert channels._actuator_column("nbi.total", db) == "pnbi_total_peak"
    assert channels._actuator_column("pnbi_total_mean", db) == "pnbi_total_mean"  # as given
    assert channels._actuator_column("nbi.nosuchbeam", db) is None
    assert channels._actuator_column("nbi.30L", db) is None  # known key, column not in this db


def test_the_scalar_channel_and_the_flag_rules_resolve_an_actuator_to_one_column(db):
    """`--actuator nbi.15L=2e6` used to mean pnbi_15L_mean to scalar_knn and pnbi_15L_peak to
    evaluate_flags -- one flag, two quantities. Both read flags.rules.actuator_columns now."""
    from shot_design.flags import rules

    col = channels._actuator_column("nbi.15L", db)
    assert col == rules.actuator_columns()["nbi.15L"] == "pnbi_15L_peak"
    assert rules.load_rules()["actuator_keys"]["nbi.15L"] == [col]
    assert channels.target_values(QueryState(actuators={"nbi.15L": 2e6}), db) == {col: 2e6}


def test_search_checks_the_proposal_itself_against_the_operating_limits(db):
    """QueryState.actuators was never handed to evaluate_flags: the rules ran on the RESULT rows
    only, so a 50 MW beam request -- 2.5x configs/shot_design/flags.yaml's nbi_total_max -- raised
    nothing."""
    found = rank.search(QueryState(actuators={"nbi.total": 5e7}, n=3), db)
    errors = [f for f in found.proposal_flags if f.severity == "error"]
    assert [f.rule_id for f in errors] == ["nbi_total_max"]
    assert errors[0].value == 5e7 and "pnbi_total_peak = 5e+07" in errors[0].message
    assert all(f.severity == "info" for f in found.proposal_flags[1:])  # the rest had no input
    fine = rank.search(QueryState(actuators={"nbi.total": 5e6}, n=3), db)
    assert [f for f in fine.proposal_flags if f.severity != "info"] == []
    assert rank.search(QueryState(ref_shot=100, n=3), db).proposal_flags == []  # nothing proposed


def test_text_knn_scores_a_shot_by_its_better_matching_text():
    recs = [
        _record(1, "r", 1e6, 1e6, 4.0, "x"),
        _record(2, "r", 1e6, 1e6, 4.0, "y"),
        _record(3, "r", 1e6, 1e6, 4.0, "z"),
    ]
    # shot 1 matches only through its mini proposal, shot 2 only through its log, shot 3 neither.
    db = _db(
        recs, {1: ([0, 1, 0], [1, 0, 0]), 2: ([1, 0, 0], [0, 1, 0]), 3: ([0, 1, 0], [0, 1, 0])}
    )
    q = QueryState(ref_shot=None)
    monkey = np.array([[1.0, 0.0, 0.0]], np.float32)
    hits = _text_knn_with(db, q, monkey)
    assert [s for s, _ in hits][:2] == ["1:flat_top", "2:flat_top"]
    assert dict(hits)["3:flat_top"] == pytest.approx(0.0)


def _text_knn_with(db, q, qvec):
    """text_knn with the encoder replaced by a fixed vector -- MiniLM is not under test here."""
    import shot_design.shotdb.text as text_mod

    old = text_mod.embed_texts
    text_mod.embed_texts = lambda texts: qvec
    try:
        return channels.text_knn(QueryState(**{**q.model_dump(), "text": "anything"}), db)
    finally:
        text_mod.embed_texts = old


def test_text_knn_drops_a_shot_that_matches_a_negative(db):
    q = QueryState(text="plasma", negatives=["ohmic"])
    hits = _text_knn_with(db, q, np.array([[1.0, 0.0, 0.0]], np.float32))
    assert {int(s.split(":")[0]) for s, _ in hits}.isdisjoint({200, 201})


def test_text_knn_skips_a_shot_with_no_text_at_all():
    recs = [_record(1, "r", 1e6, 1e6, 4.0, "x"), _record(2, "r", 1e6, 1e6, 4.0, "y")]
    db = _db(recs, {1: ([1, 0, 0], [1, 0, 0]), 2: ([0, 0, 0], [0, 0, 0])})
    hits = _text_knn_with(db, QueryState(), np.array([[1.0, 0.0, 0.0]], np.float32))
    assert [s for s, _ in hits] == ["1:flat_top"]  # a zero row is no evidence, not a weak match


def test_bm25_finds_the_shot_that_used_the_word(db):
    hits = channels.bm25(QueryState(text="ohmic"), db)
    assert {int(s.split(":")[0]) for s, _ in hits} == {200, 201}


def test_bm25_is_silent_without_query_text(db):
    assert channels.bm25(QueryState(ref_shot=100), db) == []


def test_the_tokenizer_keeps_physics_tokens_whole():
    assert channels.tokenize("QH-mode with n=2 RMP, q95<4") == [
        "qh-mode",
        "with",
        "n=2",
        "rmp",
        "q95",
        "4",
    ]


def test_bm25_scores_a_rare_term_above_a_common_one():
    corpus = [["elm", "suppression"], ["elm", "control"], ["elm", "suppression"], ["helicon"]]
    common = channels.bm25_scores(corpus, ["elm"])
    rare = channels.bm25_scores(corpus, ["helicon"])
    assert rare[3] > common.max()
    assert common[3] == 0.0


def test_split_negatives_reads_an_exclusion_out_of_the_sentence():
    positive, negs = channels.split_negatives("ELM suppression with RMP but no QH-mode shots")
    assert positive == "ELM suppression with RMP"
    assert negs == ["qh mode"]  # punctuation becomes a space, as it does in the haystack
    assert channels.split_negatives("plain query") == ("plain query", [])


def test_text_matches_negative_ignores_punctuation_and_case():
    assert channels.text_matches_negative("a QH-Mode discharge", ["qh mode"])
    assert not channels.text_matches_negative("an L-mode discharge", ["qh mode"])


def test_a_negative_matches_whole_words_not_substrings():
    """`--negative H-mode` used to drop every QH-mode shot, and a negated "elm" dropped "helmet":
    the match was a plain substring test on the normalised text."""
    assert not channels.text_matches_negative("a QH-mode discharge", ["h mode"])
    assert not channels.text_matches_negative("wearing a helmet", ["elm"])
    assert channels.text_matches_negative("an H-mode discharge", ["h mode"])
    assert channels.text_matches_negative("ELM suppression with RMP", ["elm"])
    assert channels.text_matches_negative("H-mode, then L-mode", ["l mode"])  # at the end


def test_ignite_knn_is_registered_and_empty_until_the_build_wrote_its_matrix(db):
    """Registered in CHANNELS so rank.py picks it up unchanged; silent until `shot_design build` has
    written emb_ignite_seg.npy AND the manifest block that says how its columns are laid out."""
    assert "ignite_knn" in channels.CHANNELS
    assert channels.ignite_knn(QueryState(ref_shot=100), db) == []
    db.emb["ignite_seg"] = np.eye(len(db.segments), dtype=np.float32)
    assert channels.ignite_knn(QueryState(ref_shot=100), db) == [], "a matrix without its manifest"
    db.manifest["ignite"] = {"status": "ok", "dims": [len(db.segments)], "modalities": ["a"]}
    hits = channels.ignite_knn(QueryState(ref_shot=100), db)
    assert hits and "100:flat_top" not in {h[0] for h in hits}  # the reference is not an answer


# -------------------------------------------------------------------------------------- fusion


def test_rrf_rewards_agreement_between_channels():
    """From shotsearch/tests/test_fusion.py (same author, MIT)."""
    fused = dict(
        rank.rrf_fuse(
            {"a": [("7", 0.9), ("8", 0.8), ("9", 0.7)], "b": [("7", 5.0), ("5", 4.0), ("6", 3.0)]}
        )
    )
    assert max(fused, key=fused.get) == "7"
    assert set(fused) == {"5", "6", "7", "8", "9"}


def test_rrf_is_scale_invariant():
    """Channel b's scores are 1000x channel a's and it changes nothing: only ranks vote."""
    a = [("x", 0.01), ("y", 0.009)]
    b = [("y", 900.0), ("x", 100.0)]
    assert dict(rank.rrf_fuse({"a": a, "b": b}))["x"] == pytest.approx(
        dict(rank.rrf_fuse({"a": a, "b": b}))["y"]
    )


def test_rrf_weights_a_channel_and_skips_an_empty_one():
    weighted = dict(rank.rrf_fuse({"a": [("x", 1.0)], "b": [("y", 1.0)]}, {"a": 2.0, "b": 1.0}))
    assert weighted["x"] == pytest.approx(2 * weighted["y"])
    assert dict(rank.rrf_fuse({"a": [("x", 1.0)], "b": []})) == {"x": pytest.approx(1 / 61)}


def test_channel_ranks_are_one_based():
    assert rank.channel_ranks({"a": [("x", 1.0), ("y", 0.5)]}) == {"x": {"a": 1}, "y": {"a": 2}}


# ------------------------------------------------------------------------------------ reranking


def test_rerank_drops_a_near_duplicate_and_keeps_a_vectorless_one():
    """After shotsearch/query/ranker.py::dedup_near_duplicates (same author, MIT)."""
    recs = [_record(s, f"r{s}", 1e6, 1e6, 4.0, "t") for s in (1, 2, 3)]
    near = [1.0, 0.02, 0.0]
    near = list(np.asarray(near) / np.linalg.norm(near))
    db = _db(recs, {1: ([1, 0, 0], [1, 0, 0]), 2: (near, near), 3: ([0, 0, 0], [0, 0, 0])})
    items = [("1:flat_top", 1.0), ("2:flat_top", 0.9), ("3:flat_top", 0.8)]
    kept = [s for s, _ in rank.rerank(items, db, dedup_threshold=0.97, decay=1.0)]
    assert kept == ["1:flat_top", "3:flat_top"]  # 2 is a duplicate of 1; 3 has no vector at all


def test_rerank_spreads_results_across_run_days(db):
    items = [("100:flat_top", 1.0), ("101:flat_top", 0.99), ("200:flat_top", 0.9)]
    out = dict(rank.rerank(items, db, dedup_threshold=1.1, decay=0.7))
    assert out["100:flat_top"] == pytest.approx(1.0)
    assert out["101:flat_top"] == pytest.approx(0.99 * 0.7)  # second hit from run r1
    assert out["200:flat_top"] == pytest.approx(0.9)  # a different run day is not penalised
    assert [s for s, _ in rank.rerank(items, db, 1.1, 0.7)][1] == "200:flat_top"


def test_the_run_day_decay_has_one_source_the_yaml(db, monkeypatch):
    """configs/shot_design/retrieval.yaml says 0.9 (tuned, with its reasoning in a comment); load_cfg's
    fallback and rerank's default both still said 0.7, so a direct rerank() call silently used a
    number the config had abandoned."""
    import inspect

    import yaml

    from shot_design import config

    yaml_value = yaml.safe_load((config.CONFIG_DIR / "retrieval.yaml").read_text())["retrieval"][
        "run_diversity_decay"
    ]
    assert rank.load_cfg()["run_diversity_decay"] == yaml_value
    params = inspect.signature(rank.rerank).parameters
    assert params["decay"].default is None and params["dedup_threshold"].default is None
    items = [("100:flat_top", 1.0), ("101:flat_top", 0.99)]
    out = dict(rank.rerank(items, db))  # no decay given: the YAML's, not a literal in rank.py
    assert out["101:flat_top"] == pytest.approx(0.99 * yaml_value)
    # a config that lacks the knob is an error naming it, not a second value
    monkeypatch.setattr(config, "load_yaml", lambda name: {"retrieval": {"k0": 60}})
    with pytest.raises(
        KeyError, match=r"configs/shot_design/retrieval\.yaml: retrieval\.dedup_threshold is missing"
    ):
        rank.load_cfg()


def test_prefer_outcome_demotes_a_shot_that_missed_its_target(db):
    items = [("300:flat_top", 1.0), ("301:flat_top", 0.9)]
    out = dict(rank.rerank(items, db, 1.1, 1.0, outcome_penalty=0.5))
    assert out["300:flat_top"] == pytest.approx(0.5)  # ip_target_hit is False on 300
    assert out["301:flat_top"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------------- explaining


def test_values_print_in_the_units_the_registry_declares():
    assert rank.display("ip_mean", 1.21e6) == "1.21 MA"
    assert rank.display("pnbi_total_mean", 7.34e6) == "7.34 MW"
    assert rank.display("bt_mean", 1.95) == "1.95 T"
    assert rank.display("q95_mean", 3.42) == "3.42"  # dimensionless: no unit invented
    assert rank.display("pnbi_15L_on_frac", 0.7) == "0.7 fraction of segment"
    assert rank.display("ip_slope", 2.0e6) == "2 MA/s"
    assert rank.display("wmhd_mean", 5.73e5) == "573 kJ"


def test_the_si_prefix_comes_from_the_magnitude_not_the_unit_name():
    """configs/shot_design/actuators.yaml declares the RMP coils in amps, the same unit as Ip. A blanket
    /1e6 printed a 14.6 A coil current as "1.46e-05 MA" -- right number, useless label."""
    assert rank.display("irmp_C19_peak", 14.6) == "14.6 A"
    assert rank.display("irmp_IL210_peak", 958.25) == "958 A"
    assert rank.display("irmp_total_peak", 17400.0) == "17.4 kA"
    assert rank.display("ip_mean", 985281.0) == "985 kA"


def test_the_prefix_is_chosen_after_rounding_to_the_printed_precision():
    """999949 A prints to three figures as 1.00e6 A, so it is "1 MA" -- not "1e+03 kA", which is
    what choosing the prefix from the raw magnitude and rounding afterwards produced."""
    assert rank.display("ip_mean", 999949.0) == "1 MA"
    assert rank.display("irmp_C19_peak", 999.9) == "1 kA"
    assert rank.display("irmp_C19_peak", 999.4) == "999 A"  # rounds to 999, stays in A
    assert rank.display_pair("ip_mean", 999949.0, 985281.0) == "1 vs 0.985 MA"
    line = rank.constraint_line("ip_mean", Range(lo=9.5e5, hi=999949.0), 999949.0)
    assert line == "ip_mean 1 MA within [0.95, 1] MA"


def test_an_unverified_unit_never_gets_a_prefix_invented_for_it():
    """neutrons' unit is honestly "[?]" in configs/shot_design/signals.yaml. 1.11e14 of an unknown
    thing is not 111 T-of-that-thing. (ne_line used to be the example until its unit was
    confirmed.)"""
    assert rank.display("neutrons_mean", 1.11e14).startswith("1.11e+14 [?]")
    assert rank.display("gas_GASA_peak", 1.88) == "1.88 V (raw valve command)"


def test_a_compared_pair_prints_in_one_unit():
    assert rank.display_pair("irmp_IL210_peak", 958.25, 6.87) == "958 vs 6.87 A"
    assert rank.display_pair("ip_mean", 1.21e6, 985281.0) == "1.21 vs 0.985 MA"


def test_a_matched_constraint_is_restated_in_physical_units(db):
    q = QueryState(constraints={"ip_mean": Range(lo=1.0e6, hi=1.4e6)})
    e = rank.explain(q, db, "200:flat_top", {})
    assert e.matched_constraints == ["ip_mean 1.4 MA within [1, 1.4] MA"]


def test_a_one_sided_constraint_prints_as_an_inequality(db):
    q = QueryState(constraints={"q95_mean": Range(hi=4.5)})
    assert rank.explain(q, db, "100:flat_top", {}).matched_constraints == ["q95_mean 4 <= 4.5"]


def test_a_constrained_column_is_not_also_reported_as_a_difference(db):
    q = QueryState(constraints={"ip_mean": Range(lo=1.0e6, hi=1.4e6)})
    e = rank.explain(q, db, "200:flat_top", {})
    assert "ip_mean" not in [c for c, _, _ in e.top_similar + e.top_different]


def test_the_explanation_ranks_features_by_z_distance(db):
    e = rank.explain(QueryState(ref_shot=100), db, "101:flat_top", {"scalar_knn": 2})
    assert e.channel_ranks == {"scalar_knn": 2}
    sims = [d for _, d, _ in e.top_similar]
    diffs = [d for _, d, _ in e.top_different]
    assert sims == sorted(sims) and diffs == sorted(diffs, reverse=True)
    assert not sims or not diffs or max(sims) <= min(diffs)


def test_feature_scales_drop_a_column_more_than_half_the_candidates_share(db):
    db.segments.loc[db.segments["segment"] == "flat_top", "bt_mean"] = 2.0
    scales = rank.feature_scales(db, channels.hard_filter(QueryState(), db))
    assert "bt_mean" not in scales  # every candidate reads 2.0 T: it cannot separate them
    assert "ip_mean" in scales


def test_a_highlight_is_quoted_from_a_parsed_log_entry(db):
    e = rank.explain(QueryState(text="ohmic reference"), db, "200:flat_top", {})
    assert e.text_highlight == "[SESSION_LEADER] L-mode ohmic reference"
    assert rank.explain(QueryState(ref_shot=100), db, "200:flat_top", {}).text_highlight is None


def test_a_highlight_skips_settings_dumps_and_decodes_entities_like_the_operator_quote():
    """_highlight scanned every entry with no role filter and no decoding: on the real database
    a query for 'ECH gyrotron power' highlighted a `[PCS] PCS CHANGES: ...` dump on 91 of 105
    shots, and shot 161584 printed `didn&#39;t` on the same screen where the Operator quote
    printed "didn't". Both decisions are describe.quotable's now."""
    rec = _record(1, "r", 1e6, 1e6, 4.0, "Postshot: didn&#39;t get ECH,\n  gyrotron trip")
    rec.human.log_entries.insert(
        0, LogEntry(role="PCS", text="PCS CHANGES: ECH: ECH Control Param ECH gyrotron ECH")
    )
    rec.human.log_entries.append(LogEntry(role="RF", text="System\tECH gyrotron\tECH\tActual"))
    db = _db([rec, _record(2, "r", 1e6, 1e6, 4.0, "nothing here")])
    e = rank.explain(QueryState(text="ECH gyrotron"), db, "1:flat_top", {})
    assert e.text_highlight == "[SESSION_LEADER] Postshot: didn't get ECH, gyrotron trip"


def test_a_highlight_is_steered_by_the_positive_text_not_the_negation_clause():
    """'ELM suppression but no QH-mode shots' tokenised whole gave 'but', 'no', 'qh-mode' and
    'shots' as query terms, so the entry about the excluded thing won the highlight."""
    rec = _record(1, "r", 1e6, 1e6, 4.0, "ELM suppression held")
    rec.human.log_entries.append(
        LogEntry(role="SESSION_LEADER", text="no QH-mode shots today, but we tried")
    )
    db = _db([rec, _record(2, "r", 1e6, 1e6, 4.0, "nothing here")])
    q = QueryState(text="ELM suppression but no QH-mode shots")
    assert channels.positive_and_negatives(q) == ("ELM suppression", ["qh mode"])
    assert (
        rank.explain(q, db, "1:flat_top", {}).text_highlight
        == "[SESSION_LEADER] ELM suppression held"
    )
    # a query that is nothing but a negation clause has no positive terms and highlights nothing
    assert rank.explain(QueryState(negatives=["elm"]), db, "1:flat_top", {}).text_highlight is None


# ------------------------------------------------------------------------------------- search


def test_search_returns_n_explained_results_and_never_the_reference_itself(db):
    results = rank.search(QueryState(ref_shot=100, n=3), db).items
    assert [r.id for r in results] and 100 not in [r.shot for r in results]
    assert len(results) == 3
    first = results[0]
    assert first.segment == "flat_top" and first.run_id
    assert first.description and first.explanation.channel_ranks
    assert first.score == max(r.score for r in results)
    # the one renderer: a result's description is describe.describe for its own segment
    from shot_design.retrieval import describe

    assert first.description == describe.describe(db.get(first.shot), "flat_top", db=db)
    assert (
        f"Ip {rank.display('ip_mean', db.segments.loc[first.id, 'ip_mean'])}" in first.description
    )


def test_search_respects_the_hard_filter(db):
    results = rank.search(
        QueryState(ref_shot=100, constraints={"ip_mean": Range(lo=1.3e6)}, n=5), db
    ).items
    assert sorted(r.shot for r in results) == [200, 201]


def test_search_answers_a_constraint_only_query(db):
    q = QueryState(constraints={"q95_mean": Range(lo=5.5, hi=6.5)}, n=5)
    assert sorted(r.shot for r in rank.search(q, db).items) == [300, 301]


def test_search_returns_nothing_when_the_filter_admits_nothing(db):
    q = QueryState(ref_shot=100, constraints={"ip_mean": Range(lo=9e9)}, n=5)
    assert rank.search(q, db).items == []


def test_search_reports_each_channels_ranking_from_the_same_pass(db):
    """The CLI's "channels scalar_knn 6, bm25 0" line reads these; it used to run every channel a
    second time to count them, which for a --text query meant embedding the text twice."""
    found = rank.search(QueryState(ref_shot=100, n=3), db)
    assert set(found.rankings) == set(channels.CHANNELS)
    assert found.rankings["scalar_knn"] and found.rankings["bm25"] == []
    assert found.rankings["scalar_knn"][0][0] == "100:flat_top"  # raw channel output, pre-fusion
    assert found.items[0].explanation.channel_ranks["scalar_knn"] >= 1


def test_search_loads_the_rules_and_the_query_values_once_not_per_result(db, monkeypatch):
    """load_rules() re-reads two YAML files (55 ms measured on the 105-shot database) and was
    called inside _enrich for every result; query_values was recomputed inside explain likewise."""
    calls = {"rules": 0, "qvals": 0}
    real_rules, real_qvals = rank.load_rules, rank.query_values

    def counting_rules(*a, **k):
        calls["rules"] += 1
        return real_rules(*a, **k)

    def counting_qvals(*a, **k):
        calls["qvals"] += 1
        return real_qvals(*a, **k)

    monkeypatch.setattr(rank, "load_rules", counting_rules)
    monkeypatch.setattr(rank, "query_values", counting_qvals)
    found = rank.search(QueryState(ref_shot=100, n=4), db)
    assert len(found.items) == 4
    assert calls == {"rules": 1, "qvals": 1}


def test_channels_is_a_plain_dict_a_reader_can_extend():
    assert isinstance(channels.CHANNELS, dict)
    assert set(channels.CHANNELS) == {"scalar_knn", "text_knn", "bm25", "ignite_knn", "phenomenon"}
    for fn in channels.CHANNELS.values():
        assert callable(fn)


def test_a_new_channel_needs_only_a_registry_entry(db):
    channels.CHANNELS["always_300"] = lambda q, d: [("300:flat_top", 1.0)]
    try:
        top = rank.search(QueryState(ref_shot=100, n=1), db).items[0]
        assert top.shot == 300 and "always_300" in top.explanation.channel_ranks
    finally:
        del channels.CHANNELS["always_300"]


# ---------------------------------------------------------------------- the real database, if any

pytestmark_real = pytest.mark.skipif(
    not all((REAL_STAGED / f"{s}.h5").exists() for s in REAL_SHOTS),
    reason=f"{REAL_STAGED} test shots not mounted"
)


@pytest.fixture(scope="module")
def real_query_db(tmp_path_factory):
    """Build the real retrieval slice under pytest's temporary root, never in a shared store."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("IDEATE_DATA_ROOT", str(tmp_path_factory.mktemp("ideate-query")))
        mp.delenv("IDEATE_PATHS", raising=False)
        mp.setattr(text, "embed_texts", lambda texts: np.zeros((len(texts), 384), np.float32))
        from shot_design.config import load_paths

        paths = load_paths()
        report = build.build(REAL_SHOTS, paths, build.load_build_cfg(), workers=1, encode=False)
        assert sorted(report.shots) == sorted(REAL_SHOTS)
        yield paths


@pytest.mark.real_data
@pytestmark_real
@pytest.mark.usefixtures("real_query_db")
def test_query_by_reference_shot_finds_its_run_mates(capsys):
    assert cli.main(["query", "--ref", "161172", "--n", "5"]) == 0
    out = capsys.readouterr().out
    assert "161172" not in out.split("\n")[3]  # the reference is not its own first result
    assert "161164" in out  # a shot from the same run day, 20150113A


@pytest.mark.real_data
@pytestmark_real
@pytest.mark.usefixtures("real_query_db")
def test_query_by_constraint_reports_the_candidate_count(capsys):
    assert cli.main(["query", "--where", "ip_mean:1.0e6:1.4e6", "--n", "3"]) == 0
    out = capsys.readouterr().out
    assert "candidates" in out and "flat_top rows" in out
    assert "within [1, 1.4] MA" in out


@pytest.mark.real_data
@pytestmark_real
@pytest.mark.usefixtures("real_query_db")
def test_query_json_carries_the_results_and_the_proposal_flags(capsys):
    assert cli.main(["query", "--ref", "161172", "--json", "-n", "3"]) == 0
    doc = json.loads(capsys.readouterr().out)
    rows = doc["results"]
    assert doc["proposal_flags"] == [] and len(rows) == 3
    assert {"id", "shot", "score", "description", "explanation", "flags"} <= set(rows[0])
    assert rows[0]["score"] >= rows[-1]["score"]
    assert cli.main(["query", "--actuator", "nbi.total=5e7", "--json", "-n", "1"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [f["rule_id"] for f in doc["proposal_flags"] if f["severity"] == "error"] == [
        "nbi_total_max"
    ]


@pytest.mark.real_data
@pytestmark_real
@pytest.mark.usefixtures("real_query_db")
def test_query_refuses_a_column_the_database_does_not_have(capsys):
    assert cli.main(["query", "--where", "no_such_mean:1:2"]) == 2
    assert "no_such_mean" in capsys.readouterr().err


@pytest.mark.real_data
@pytestmark_real
@pytest.mark.usefixtures("real_query_db")
def test_query_with_nothing_to_go_on_says_so(capsys):
    assert cli.main(["query"]) == 2
    assert "no channel had anything to search on" in capsys.readouterr().err


# ------------------------------------------------------------------------- ignite channel


def _with_ignite(db: ShotDB, rows: dict[str, list[float]], dims: list[int]) -> ShotDB:
    """Attach a hand-written IGNITE segment matrix (NaN = modality absent) to a synthetic db."""
    width = sum(dims)
    mat = np.full((len(db.segments), width), np.nan, np.float32)
    for sid, vec in rows.items():
        mat[db.segments.index.get_loc(sid)] = vec
    db.emb["ignite_seg"] = mat
    db.manifest["ignite"] = {
        "status": "ok",
        "dims": dims,
        "modalities": ["a", "b"],
        "window_ms": 250,
    }
    return db


def test_modality_cosine_averages_only_over_shared_modalities():
    dims = [2, 2]
    M = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],  # both modalities
            [1.0, 0.0, np.nan, np.nan],  # only the first
            [np.nan, np.nan, 0.0, 1.0],  # only the second
            [np.nan, np.nan, np.nan, np.nan],  # nothing
            [0.0, 1.0, 1.0, 0.0],  # both, but different
        ],
        np.float32,
    )
    q = np.array([1.0, 0.0, 0.0, 1.0], np.float32)
    s = channels.modality_cosine(M, q, dims)
    assert np.isnan(s[3]), "a row sharing no modality is not scored"
    assert s[0] > s[4], "agreeing on both modalities beats disagreeing on both"
    # centring uses the finite rows of each block, so a partial row scores like the matching
    # half of the full row rather than being penalised for what it lacks
    assert s[1] == pytest.approx(s[2])


def test_ignite_knn_is_empty_without_the_matrix_and_ranks_by_shared_modalities(db):
    q = QueryState(ref_shot=100, segment="flat_top")
    assert channels.ignite_knn(q, db) == []
    db = _with_ignite(
        db,
        {
            "100:flat_top": [1.0, 0.0, 0.0, 1.0],
            "101:flat_top": [0.9, 0.1, 0.1, 0.9],  # near repeat, both modalities
            "200:flat_top": [0.0, 1.0, np.nan, np.nan],  # far, one modality
            "201:flat_top": [np.nan, np.nan, np.nan, np.nan],  # never encoded
            "300:flat_top": [np.nan, np.nan, 0.2, 1.0],  # close on the one modality it has
        },
        [2, 2],
    )
    hits = channels.ignite_knn(q, db)
    ids = [h[0] for h in hits]
    assert ids[0] == "101:flat_top" and "100:flat_top" not in ids and "201:flat_top" not in ids
    assert ids.index("300:flat_top") < ids.index("200:flat_top")
    assert channels.ignite_knn(QueryState(ref_shot=999), db) == [], "unknown reference"


def test_ignite_knn_fragment_query_pools_the_reference_windows(db):
    import pandas as pd

    db = _with_ignite(
        db, {"101:flat_top": [1.0, 0.0, 0.0, 1.0], "200:flat_top": [0.0, 1.0, 1.0, 0.0]}, [2, 2]
    )
    db.windows = pd.DataFrame(
        {
            "id": ["100:0", "100:250", "100:500"],
            "shot": [100, 100, 100],
            "t0_ms": [0.0, 250.0, 500.0],
            "t1_ms": [250.0, 500.0, 750.0],
        }
    ).set_index("id")
    db.emb["ignite_win"] = np.array(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], np.float16
    )
    late = channels.ignite_knn(QueryState(ref_shot=100, ref_window_ms=(300.0, 700.0)), db)
    early = channels.ignite_knn(QueryState(ref_shot=100, ref_window_ms=(0.0, 200.0)), db)
    assert late[0][0] == "101:flat_top" and early[0][0] == "200:flat_top"
    assert channels.ignite_knn(QueryState(ref_shot=100, ref_window_ms=(5000.0, 6000.0)), db) == []
