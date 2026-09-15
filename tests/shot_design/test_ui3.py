"""UI3 HTTP contracts; fixtures and modified configuration stay in tmp_path."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by parameter name

import inspect
import json

import pytest
import yaml

from labeler.events.lexicon import TEXT_ONLY_CEILING
from shot_design import config
from shot_design.mcp import tools
from shot_design.retrieval import channels

from .test_phenomena import _claim, _db_with, _event
from .test_ui import client, local_auxiliary_paths  # noqa: F401


@pytest.fixture
def phenomenon_db(ideate_db):
    config.load_paths().qh_database_csv.write_text("shot\n100\n")
    _db_with(ideate_db, [
        _event(100, "early", t0_s=.1, t1_s=.2, confidence=.875),
        _event(100, "high", confidence=.875),
        _event(100, "low", t0_s=3, t1_s=4, confidence=.2),
        _event(100, "unknown", t0_s=4, t1_s=4, confidence=None),
        _event(100, "elm", source="elm_clock", phenomenon="elm",
               evidence_kind="heuristic", diag="filterscopes", t0_s=2, t1_s=2,
               confidence=.6),
        _event(100, "forecast", source="label_forecast", phenomenon="tearing",
               evidence_kind="forecast", t0_s=2, t1_s=3, confidence=.8),
    ], claims=[_claim(100, "rwm", snippet="RWM mentioned")])
    tools.reset_cache()


def test_event_phenomena_share_description_rows_and_keep_all_intervals(
    client, phenomenon_db,
):
    # No full segment: describe(full) and the event API both search the whole shot.
    parts = client.get("/api/shot/100?segment=full").json()["describe_parts"]
    data = client.get("/api/shot/100/events").json()
    assert [p["id"] for p in data["phenomena"]] == [p["id"] for p in parts["phenomena"]]
    for row, described in zip(data["phenomena"], parts["phenomena"], strict=True):
        assert row["intervals"] == described["intervals"]
        assert row["n_forecast"] == described["n_forecast"]
        assert len(row["forecast_intervals"]) == row["n_forecast"]
        assert {"title", "coverage_note", "coverage_windows", "caveats"} <= row.keys()
    tearing = next(p for p in data["phenomena"] if p["id"] == "tearing")
    assert [iv["event_id"] for iv in tearing["intervals"]] == [
        "early", "high", "low", "unknown",
    ]
    assert [iv["event_id"] for iv in tearing["forecast_intervals"]] == ["forecast"]
    assert {"rwm", "qh"} <= {p["id"] for p in data["phenomena"]}
    assert not any("No None segment" in c for c in tearing["caveats"])


@pytest.mark.parametrize("params,observed,forecasts", [
    ({"t0_s": 2.5, "t1_s": 3, "min_confidence": .5}, ["high"], ["forecast"]),
    ({"t1_s": .2}, ["early"], []),
    ({"t0_s": 4}, ["low", "unknown"], []),
    ({"min_confidence": .875}, ["early", "high"], []),
    # The literal name is coherent_mode, although the registry resolves tearing.
    ({"phenomenon": "tearing"}, [], ["forecast"]),
    ({"phenomenon": "elm", "min_confidence": .5}, ["elm"], []),
])
def test_event_phenomena_follow_literal_window_and_confidence_filters(
    client, phenomenon_db, params, observed, forecasts,
):
    data = client.get("/api/shot/100/events", params=params).json()
    assert [iv["event_id"] for iv in data["events"]] == observed
    assert [iv["event_id"] for iv in data["forecasts"]] == forecasts
    assert data["n"] == len(observed) and data["n_forecasts"] == len(forecasts)
    for row in data["phenomena"]:
        assert {iv["event_id"] for iv in row["intervals"]} <= set(observed)
        assert {iv["event_id"] for iv in row["forecast_intervals"]} <= set(forecasts)
        assert row["n_forecast"] == len(row["forecast_intervals"])
        for lo, hi in row["coverage_windows"]:
            assert lo >= params.get("t0_s", float("-inf"))
            assert hi <= params.get("t1_s", float("inf"))
    if "phenomenon" in params:
        assert all(p["id"] == params["phenomenon"] for p in data["phenomena"])
    else:
        tearing = next(p for p in data["phenomena"] if p["id"] == "tearing")
        assert [iv["event_id"] for iv in tearing["intervals"]] == observed


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1", "wrong"])
def test_events_reject_invalid_confidence(client, value):
    response = client.get("/api/shot/100/events", params={"min_confidence": value})
    assert response.status_code == 422 or "error" in response.json()


def test_confidence_filter_is_http_only_and_does_not_mutate_cached_evidence(
    client, phenomenon_db,
):
    before = client.get("/api/shot/100/events").json()
    filtered = client.get("/api/shot/100/events?min_confidence=.5").json()
    assert filtered["n"] < before["n"]
    assert any("confidence" in c for c in filtered["caveats"])
    assert client.get("/api/shot/100/events").json() == before
    assert "min_confidence" not in inspect.signature(tools.get_events).parameters
    assert "phenomena" not in tools.get_events(100)


def test_untimed_registry_events_preserve_transport_without_inventing_intervals(
    client, ideate_db,
):
    _db_with(ideate_db, [_event(100, "untimed", t0_s=None, t1_s=None)])
    tools.reset_cache()
    data = client.get("/api/shot/100/events").json()
    assert "error" not in data
    assert data["n"] == 1 and data["events"][0]["event_id"] == "untimed"
    assert not any(p["intervals"] for p in data["phenomena"])
    windowed = client.get("/api/shot/100/events?t0_s=0").json()
    assert windowed["n"] == 0
    assert any("no recorded time" in c for c in windowed["caveats"])


@pytest.mark.parametrize("params,window", [
    ({"t0_s": 2}, [2, 6]), ({"t1_s": 3}, [0, 3]),
])
def test_one_sided_window_does_not_invent_partial_phenomenon_coverage(
    client, phenomenon_db, params, window,
):
    data = client.get("/api/shot/100/events", params=params).json()
    tearing = next(p for p in data["phenomena"] if p["id"] == "tearing")
    assert tearing["coverage_windows"] == [window]
    assert not tearing["coverage_partial"]
    assert not any("covered only" in c for c in tearing["caveats"])


def test_scoring_uses_loaded_yaml_channel_registry_and_meta(client):
    response = client.get("/api/scoring")
    assert response.status_code == 200
    data = response.json()
    cfg = yaml.safe_load((config.CONFIG_DIR / "retrieval.yaml").read_text())
    for key in ("k0", "dedup_threshold", "run_diversity_decay", "outcome_penalty"):
        assert data[key] == cfg["retrieval"][key]
    assert data["method"] == "Weighted reciprocal rank fusion"
    assert "rank_c" in data["formula"]
    assert [c["name"] for c in data["channels"]] == list(channels.CHANNELS)
    for channel in data["channels"]:
        assert channel["weight"] == cfg["retrieval"]["weights"].get(channel["name"], 1.0)
        assert channel["compares"]
    ph = data["phenomenon"]
    assert ph["weights"] == cfg["phenomenon"]["weights"]
    assert ph["saturation_n"] == cfg["phenomenon"]["saturation_n"]
    assert ph["text_only_ceiling"] == TEXT_ONLY_CEILING
    assert ph["class_order"] == ["OBSERVED", "LABELLED", "FORECAST", "DATABASE", "TEXTUAL"]
    assert "max_p" in ph["formula"] and "tanh" in ph["formula"]
    assert data["hard_filters"] == ["constraints", "segment", "require labels", "avoid labels"]
    meta = client.get("/api/meta").json()["db"]
    assert data["db"] == {"n_shots": meta["n_shots"], "shot_range": meta["shot_range"],
                          "git_sha": meta["git_sha"], "built": meta["built_at"]}


def test_scoring_follows_changed_config_and_new_default_channel(client, tmp_path, monkeypatch):
    cfg = yaml.safe_load((config.CONFIG_DIR / "retrieval.yaml").read_text())
    cfg["retrieval"].update(k0=41, run_diversity_decay=.72, dedup_threshold=.82,
                            outcome_penalty=.31)
    cfg["retrieval"]["weights"]["bm25"] = 2.75
    cfg["phenomenon"]["weights"]["event"] = 1.7
    cfg["phenomenon"]["saturation_n"] = 7.5
    (tmp_path / "retrieval.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setitem(channels.CHANNELS, "new_channel", lambda q, db: [])
    data = client.get("/api/scoring").json()
    assert data.get("k0") == cfg["retrieval"]["k0"]
    for key in ("run_diversity_decay", "dedup_threshold", "outcome_penalty"):
        assert data[key] == cfg["retrieval"][key]
    assert data["phenomenon"]["weights"] == cfg["phenomenon"]["weights"]
    assert data["phenomenon"]["saturation_n"] == cfg["phenomenon"]["saturation_n"]
    weights = {c["name"]: c for c in data["channels"]}
    assert weights["bm25"]["weight"] == cfg["retrieval"]["weights"]["bm25"]
    assert weights["new_channel"]["weight"] == 1.0
    assert weights["new_channel"]["weight_source"] == "default"
    assert json.dumps(data, allow_nan=False)


def test_scoring_matches_search_settings_after_snapshot_is_warm(client, tmp_path, monkeypatch):
    from shot_design.shotdb.store import ShotDB

    original = yaml.safe_load((config.CONFIG_DIR / "retrieval.yaml").read_text())
    db, error = tools._db()
    assert error is None and isinstance(db, ShotDB)
    # The Search phenomenon channel reads these snapshot settings.
    loaded_weights, loaded_sat, _ = db._phenomenon_config
    assert loaded_weights == original["phenomenon"]["weights"]
    changed = yaml.safe_load((config.CONFIG_DIR / "retrieval.yaml").read_text())
    changed["phenomenon"]["weights"]["event"] += 1
    changed["phenomenon"]["saturation_n"] += 2
    (tmp_path / "retrieval.yaml").write_text(yaml.safe_dump(changed))
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    data = client.get("/api/scoring").json()
    assert data["phenomenon"]["weights"] == loaded_weights
    assert data["phenomenon"]["saturation_n"] == loaded_sat
