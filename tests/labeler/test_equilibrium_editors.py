"""Signal-derived regimes and conservative RWM review drafts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from labeler.events import panels, spans, suggestions
from labeler.events.review import cohort_rosters, labels
from labeler.features import namespace
from labeler.features.store import FeatureArray, write_features

from . import editor_tree as tree

SHOT = 198658


def features(paths, **signals):
    arrays = {
        name: FeatureArray(np.asarray(t) / 1000, np.atleast_2d(y))
        for name, (t, y) in signals.items()
    }
    write_features(paths.features_file(SHOT), SHOT, arrays, {})


def draft(paths, event, window):
    assert event in spans.METHODS, f"no review method for {event}"
    rows, reason, info = spans.suggest(spans.METHODS[event], SHOT, window, paths)
    assert reason is None, reason
    return [row[1:4] for row in rows], info


def test_qmin_editor_labels_all_four_bands_and_keeps_ramps_unknown(tmp_path):
    p = tree.paths(tmp_path)
    t = np.arange(0, 5001, 25)
    q = np.select([t < 1500, t < 2500, t < 3500], [0.95, 1.5, 2.0], 2.1)
    features(p, qmin=(t, q), ip=(t, np.where((t >= 500) & (t <= 4500), -1e6, 0)))
    rows, info = draft(p, "minimum_safety_factor", (0, 5000))
    assert rows == [
        [6, 0, 500], [1, 500, 1475], [5, 1475, 1500],
        [2, 1500, 2475], [5, 2475, 2500],
        [3, 2500, 3475], [5, 3475, 3500],
        [4, 3500, 4500], [6, 4500, 5000],
    ]
    assert info["equilibrium"] == "efit01"
    assert labels.categories("minimum_safety_factor")[3] == "elevated"
    assert 6 not in labels.categories("minimum_safety_factor")


def test_qmin_editor_does_not_bridge_missing_samples_or_short_excursions(tmp_path):
    p = tree.paths(tmp_path)
    t = np.arange(0, 2001, 25)
    q = np.full(t.size, 1.2)
    q[(t >= 700) & (t < 1000)] = np.nan
    q[(t >= 1300) & (t < 1400)] = 2.2
    features(p, qmin=(t, q), ip=(t, np.full(t.size, 1e6)))
    rows, _ = draft(p, "minimum_safety_factor", (0, 2000))
    assert rows == [[2, 0, 675], [6, 675, 1000], [5, 1000, 1400], [2, 1400, 2000]]


def test_missing_qmin_is_not_an_elevated_regime(tmp_path):
    p = tree.paths(tmp_path)
    features(p, ip=([0, 1000], [1e6, 1e6]))
    assert "minimum_safety_factor" in spans.METHODS
    rows, reason, _ = spans.suggest(
        spans.METHODS["minimum_safety_factor"], SHOT, (0, 1000), p
    )
    assert reason and "qmin" in reason
    assert rows == [[SHOT, 6, 0, 1000, ""]]


def test_betap_editor_uses_poloidal_beta_and_splits_gaps(tmp_path):
    p = tree.paths(tmp_path)
    t = np.arange(0, 3001, 25)
    beta = np.where(t >= 500, 1.5, 1.0)
    beta[(t >= 1250) & (t < 1500)] = np.nan
    beta[(t >= 2500) & (t < 2600)] = 2.0
    beta[t >= 2600] = 1.0
    features(p, betap=(t, beta), betan=(t, np.full(t.size, 9.0)),
             ip=(t, np.full(t.size, 1e6)))
    rows, _ = draft(p, "poloidal_beta", (0, 3000))
    assert rows == [[0, 0, 475], [2, 475, 500], [1, 500, 1225], [3, 1225, 1500],
                    [1, 1500, 2575], [2, 2575, 2600], [0, 2600, 3000]]
    assert labels.categories("poloidal_beta") == {1: "present", 2: "uncertain"}


def test_betap_short_high_beta_stretch_is_uncertain(tmp_path):
    p = tree.paths(tmp_path)
    t = np.arange(0, 1001, 25)
    features(p, betap=(t, np.where((t >= 200) & (t <= 400), 1.1, 0.8)),
             ip=(t, np.full(t.size, 1e6)))
    rows, _ = draft(p, "poloidal_beta", (0, 1000))
    assert rows == [[0, 0, 175], [2, 175, 425], [0, 425, 1000]]


def test_betap_can_be_resolved_by_the_existing_fdp_scalar_path(monkeypatch):
    from labeler.features import resolve_fdp

    spec = namespace.by_name("betap")
    monkeypatch.setattr(resolve_fdp, "_import_diagnosis", lambda: None)

    def fetch(node, tree, shot, dims=()):
        assert node == r"\efit01::top.results.aeqdsk:betap"
        return {"data": np.array([0.5, 1.0, 2.0]),
                "times": np.array([1000, 1020, 1040]),
                "units": {"data": " ", "times": "ms"}}

    monkeypatch.setattr(resolve_fdp, "_fetch_mds", fetch)
    got, missing = resolve_fdp.resolve(SHOT, ["betap"])
    assert not missing
    assert spec.kind == "scalar"
    np.testing.assert_allclose(got["betap"].x, [1.0, 1.02, 1.04])
    np.testing.assert_allclose(got["betap"].y, [[0.5, 1.0, 2.0]])


def test_scalar_panels_survive_missing_q_profile_and_ip(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    features(p, qmin=([1000, 1025, 1050], [1.2, 1.3, 1.4]),
             betap=([1000, 1025, 1050], [0.5, 1.0, 1.5]))
    for event, title, thresholds in [
        ("minimum_safety_factor", "qmin (EFIT01 aeqdsk)", [0.95, 1.5, 2.0]),
        ("poloidal_beta", "beta_p (EFIT01 aeqdsk)", [1.0]),
    ]:
        built = panels.build(event, SHOT, paths=p, t_range=(1020, 1050))
        assert len(built) == 1
        assert built[0].title == title
        assert list(built[0].hlines) == thresholds
        assert built[0].x.tolist() == [1025, 1050]


def rwm_database(p):
    directory = p.label_tables / "resistive_wall_mode" / "format"
    directory.mkdir(parents=True)
    pd.DataFrame([[SHOT, 1, 900, 900, ""]], columns=suggestions.COLUMNS).to_csv(
        directory / "rwm_format_test.csv", index=False
    )


def test_rwm_database_onsets_have_no_invented_duration_or_negatives(tmp_path):
    p = tree.paths(tmp_path)
    rwm_database(p)
    rows, info = draft(p, "resistive_wall_mode", (0, 1500))
    assert rows == [[3, 0, 1500]]
    assert info["onsets"] == [{"t_ms": 900.0, "source": "rwm_format_test.csv"}]
    assert "resistive_wall_mode" in spans.METHODS
    found = spans.METHODS["resistive_wall_mode"].detect(SHOT + 1, p, (0, 1500))
    assert found.spans == () and found.measured == ()


def test_rwm_screen_is_uncertain_and_requires_high_beta(tmp_path):
    p = tree.paths(tmp_path)
    t = np.arange(0, 2001, 5)
    amplitude = 0.1 + 0.001 * np.sin(t)
    amplitude[(t >= 900) & (t <= 1000)] += np.linspace(0.0, 5.0, 21)
    features(p, n1rms=(t, amplitude), betan=(t, np.full(t.size, 5.0)),
             li=(t, np.ones(t.size)), ip=(t, np.full(t.size, 1e6)))
    rows, info = draft(p, "resistive_wall_mode", (0, 2000))
    assert any(state == 2 and start >= 900 and stop <= 1000
               for state, start, stop in rows)
    assert all(state != 1 for state, _, _ in rows)
    assert info["screen"] == "n1rms_high_beta"
    features(p, betan=(t, np.ones(t.size)))
    rows, _ = draft(p, "resistive_wall_mode", (0, 2000))
    assert rows == [[4, 0, 2000]], "a non-candidate is not a confirmed negative"


def test_rwm_database_only_shot_still_has_a_review_row(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    rwm_database(p)
    built = panels.build("resistive_wall_mode", SHOT, paths=p)
    assert len(built) == 1 and "onset" in built[0].title.lower()
    assert built[0].x[np.argmax(built[0].y[0])] == 900


def test_rwm_onset_survives_the_review_rows_grid(tmp_path, monkeypatch):
    from labeler.events.review import panel_rows

    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    rwm_database(p)
    grid, built, _ = panel_rows.build("resistive_wall_mode", SHOT, p)
    times = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    row = built[0].values[1, 0]
    assert times[np.nanargmax(row)] == pytest.approx(900, abs=1)
    assert np.nanmax(row) == 1


def test_rwm_queue_includes_curated_shots_beyond_the_cohort(tmp_path):
    p = tree.paths(tmp_path)
    rwm_database(p)
    tree.cohort(p, [tree.queue_row(SHOT + 1, 0)])
    assert "resistive_wall_mode" in spans.METHODS
    targets = spans.targets("resistive_wall_mode", p, "cohort")
    assert targets.shot.tolist() == [SHOT + 1, SHOT]
    assert targets.iloc[1].window_end_ms == 1400
    method = spans.METHODS["resistive_wall_mode"]
    spans.run(method, targets, p, windows="cohort")
    cohort_rosters.build(method.event, p, point=True)
    queue = labels.queue(p.label_tables / method.event,
                         pd.read_csv(p.label_tables / method.event / "shots.csv"))
    assert [row["shot"] for row in queue["shots"]] == [SHOT + 1, SHOT]


def test_missing_time_samples_split_a_regime(tmp_path):
    p = tree.paths(tmp_path)
    t = np.r_[np.arange(0, 1001, 25), np.arange(1500, 2501, 25)]
    ip_t = np.arange(0, 2501, 25)
    features(p, qmin=(t, np.full(t.size, 2.2)),
             ip=(ip_t, np.full(ip_t.size, 1e6)))
    rows, _ = draft(p, "minimum_safety_factor", (0, 2500))
    assert rows == [[4, 0, 1000], [6, 1000, 1500], [4, 1500, 2500]]


def test_equilibrium_prefetch_populates_the_same_cache_drafts_read(tmp_path, monkeypatch):
    from labeler.events import equilibrium
    from labeler.features import resolve_archive, resolve_fdp

    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    tree.cohort(p, [tree.queue_row(SHOT, 0)])
    features(p, ip=([0, 1000], [1e6, 1e6]))
    monkeypatch.setattr(resolve_archive, "resolve", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(resolve_fdp, "_import_diagnosis", lambda: None)
    monkeypatch.setattr(resolve_fdp, "_fetch_mds", lambda *a, **k: {
        "data": np.full(41, 1.5), "times": np.arange(0, 1001, 25),
        "units": {"data": " ", "times": "ms"},
    })
    assert hasattr(equilibrium, "main"), "no prefetch CLI for scalar editor inputs"
    assert equilibrium.main(["--event", "poloidal_beta", "--workers", "1"]) == 0
    rows, _ = draft(p, "poloidal_beta", (0, 1000))
    assert rows == [[1, 0, 1000]]


def test_archive_clock_is_aligned_and_cache_retains_source_metadata(tmp_path, monkeypatch):
    from labeler.events import equilibrium
    from labeler.features import resolve_archive

    p = tree.paths(tmp_path)
    archive = FeatureArray(np.array([0.025, 0.05, 0.075]),
                           np.array([[0.5, 1.0, 1.5]]),
                           {"resolver": "archive", "locator": "betap_EFIT01"})
    monkeypatch.setattr(resolve_archive, "resolve", lambda *a, **k:
                        ({"betap": archive}, {}))
    array = equilibrium.signal(SHOT, "betap", p, fetch=True)
    np.testing.assert_allclose(array.x, [0, 0.025, 0.05])
    cached = equilibrium.signal(SHOT, "betap", p)
    np.testing.assert_allclose(cached.x, array.x)
    assert cached.attrs["resolver"] == "archive"
    assert cached.attrs["locator"] == "betap_EFIT01"
    assert cached.attrs["archive_lag_corrected_s"] == "0.025"


def test_invalid_equilibrium_values_are_not_physical_regimes(tmp_path):
    p = tree.paths(tmp_path)
    t = np.arange(0, 2001, 25)
    q = np.where(t < 1000, -1, 2.2)
    beta = np.where(t < 1000, -1, 1.5)
    features(p, qmin=(t, q), betap=(t, beta), ip=(t, np.full(t.size, 1e6)))
    assert draft(p, "minimum_safety_factor", (0, 2000))[0] == [
        [6, 0, 1000], [4, 1000, 2000]]
    assert draft(p, "poloidal_beta", (0, 2000))[0] == [
        [3, 0, 1000], [1, 1000, 2000]]


def test_a_single_high_betap_sample_cannot_be_labelled_absent(tmp_path):
    p = tree.paths(tmp_path)
    t = np.arange(0, 1001, 25)
    beta = np.full(t.size, 0.8)
    beta[t == 500] = 1.5
    features(p, betap=(t, beta), ip=(t, np.full(t.size, 1e6)))
    rows, _ = draft(p, "poloidal_beta", (0, 1000))
    assert any(state == 2 and a <= 500 < b for state, a, b in rows)


def test_the_full_ip_record_defines_the_gate_even_for_a_short_view(tmp_path):
    p = tree.paths(tmp_path)
    t = np.arange(0, 10001, 25)
    features(p, betap=(t, np.full(t.size, 1.5)), ip=(t, t * 100))
    assert draft(p, "poloidal_beta", (0, 6000))[0] == [[3, 0, 6000]]


def test_ip_timestamp_gaps_split_the_gate_like_missing_values(tmp_path):
    p = tree.paths(tmp_path)
    t = np.r_[np.arange(0, 501, 25), np.arange(2000, 3001, 25)]
    features(p, qmin=(t, np.full(t.size, 2.2)), ip=(t, np.full(t.size, 1e6)))
    assert draft(p, "minimum_safety_factor", (0, 3000))[0] == [
        [6, 0, 2000], [4, 2000, 3000]]


@pytest.mark.parametrize("event,name,value,state", [
    ("poloidal_beta", "betap", 1.4, 1),
    ("minimum_safety_factor", "qmin", 1.7, 3),
])
def test_exact_minimum_duration_survives_legacy_float32_cache_clocks(
    tmp_path, event, name, value, state,
):
    from labeler.events import raw

    p = tree.paths(tmp_path)
    t = np.arange(200, 701, 25)
    path = raw.cache_path(SHOT, paths=p)
    raw.write_group(path, name, t, np.full((1, len(t)), value))
    raw.write_group(path, "ip", t, np.full((1, len(t)), 1e6))
    # These are native-clock records; unknown archive-capable clocks must
    # be refreshed rather than accepted merely because they have samples.
    import h5py

    with h5py.File(path, "a") as f:
        for signal_name in (name, "ip"):
            f[signal_name].attrs["resolver"] = "fdp"
    rows, _ = draft(p, event, (0, 1000))
    assert any(category == state and abs(a - 200) < 1e-3
               and abs(b - 700) < 1e-3 for category, a, b in rows)


@pytest.mark.parametrize("event,name,value", [
    ("poloidal_beta", "betap", 1.4),
    ("minimum_safety_factor", "qmin", 1.7),
])
def test_missing_scalar_times_remain_blank_on_a_finer_review_grid(
    tmp_path, monkeypatch, event, name, value,
):
    from labeler.events.review import panel_rows

    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    t = np.r_[np.arange(0, 1001, 25), np.arange(1500, 2501, 25)]
    ip_t = np.arange(0, 2501)
    features(p, **{name: (t, np.full(t.size, value)),
                  "ip": (ip_t, np.full(ip_t.size, 1e6))})
    grid, built, _ = panel_rows.build(event, SHOT, p)
    middle = round((1200 - grid.t0_ms) / grid.dt_ms - 0.5)
    assert np.isnan(built[0].values[:, 0, middle]).all()


def test_bad_optional_q_profile_does_not_hide_qmin(tmp_path, monkeypatch):
    import h5py

    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    features(p, qmin=([0, 1000], [1.5, 1.5]))
    with h5py.File(p.features_file(SHOT), "a") as f:
        g = f.create_group("qpsi")
        g.create_dataset("xdata", data=[0, 1])
        g.create_dataset("ydata", data=np.zeros((65, 3)))
    built = panels.build("minimum_safety_factor", SHOT, paths=p)
    assert len(built) == 1 and built[0].title == "qmin (EFIT01 aeqdsk)"


def test_qmin_panel_uses_the_same_valid_fallback_as_the_draft(tmp_path, monkeypatch):
    from labeler.events import raw

    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    t = np.arange(0, 1001, 25)
    features(p, qmin=(t, np.zeros(t.size)), ip=(t, np.full(t.size, 1e6)))
    raw.write_group(raw.cache_path(SHOT, paths=p), "qmin", t,
                    np.full((1, len(t)), 1.7))
    assert draft(p, "minimum_safety_factor", (0, 1000))[0] == [[3, 0, 1000]]
    built = panels.build("minimum_safety_factor", SHOT, paths=p)
    np.testing.assert_allclose(built[0].y, 1.7, rtol=1e-6)


def test_malformed_q_profile_in_raw_fallback_is_optional(tmp_path, monkeypatch):
    import h5py
    from labeler.events import raw

    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    features(p, qmin=([0, 1000], [1.7, 1.7]))
    path = raw.cache_path(SHOT, paths=p)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("qpsi")
        g.create_dataset("xdata", data=[0, 1])
        g.create_dataset("ydata", data=np.zeros((65, 3)))
    built = panels.build("minimum_safety_factor", SHOT, paths=p)
    assert len(built) == 1 and built[0].title == "qmin (EFIT01 aeqdsk)"


@pytest.mark.parametrize("name", ["betap", "betan", "ip", "qpsi"])
def test_ambiguous_legacy_archive_clocks_require_refresh(tmp_path, monkeypatch, name):
    from labeler.events import equilibrium, raw
    from labeler.events.verify import NoDataError

    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    raw.write_group(raw.cache_path(SHOT, paths=p), name, [25, 50, 75],
                    np.ones((1, 3)))
    for fetch in (False, True):
        with pytest.raises(NoDataError, match=rf"{name}.*ambiguous legacy clock"):
            equilibrium.signal(SHOT, name, p, fetch=fetch)


def test_preparation_reports_a_malformed_ip_scalar(tmp_path, monkeypatch):
    from labeler.events import equilibrium

    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    features(p, ip=([0, 1000], [[1e6, 1e6], [1e6, 1e6]]))
    _, missing = equilibrium._prepare((SHOT, ("ip",), p))
    assert "ip" in missing and "scalar" in missing["ip"]


def test_v2_drafts_can_be_selected_as_the_review_source(tmp_path):
    p = tree.paths(tmp_path)
    tree.cohort(p, [tree.queue_row(SHOT, 0)])
    event = "poloidal_beta"
    method = spans.METHODS[event]
    table = suggestions.table_path(p, event, method.name, "v2")
    suggestions.write_table(table, [[SHOT, 1, 0, 1000, ""]], {})
    summary = cohort_rosters.build(event, p, point=True, version="v2")
    assert summary["pointer"]["version"] == "v2"
    assert labels.source_path(p.label_tables / event) == table.resolve()


def test_rwm_screen_errors_keep_database_evidence(tmp_path):
    from labeler.events import equilibrium

    p = tree.paths(tmp_path)
    rwm_database(p)
    features(p, n1rms=([0, 100, 50], [1, 2, 3]),
             ip=([0, 1000], [1e6, 1e6]))
    _, missing = equilibrium._prepare((SHOT, ("n1rms",), p))
    assert "n1rms" in missing
    rows, info = draft(p, "resistive_wall_mode", (0, 1500))
    assert rows == [[3, 0, 1500]]
    assert info["onsets"][0]["t_ms"] == 900
    assert "not_run" in info


def test_rwm_onsets_keep_mode_number_and_original_type(tmp_path):
    from labeler.events import rwm

    p = tree.paths(tmp_path)
    rwm_database(p)
    raw_dir = p.label_tables / "resistive_wall_mode" / "raw"
    raw_dir.mkdir()
    (raw_dir / "onsets.csv").write_text(
        f"SHOT,ONSET_TIME,NTOR,MODE_TYPE\n{SHOT},900,2,n2rwm\n")
    meta = p.label_tables / "resistive_wall_mode" / "format" / "rwm_format_test.meta.json"
    meta.write_text(json.dumps({"made_from": [{
        "raw_file": "resistive_wall_mode/raw/onsets.csv", "source": "test_database",
    }]}))
    assert rwm.onsets(SHOT, p) == [{"t_ms": 900.0, "source": "rwm_format_test.csv",
                                  "ntor": 2, "mode_type": "n2rwm",
                                  "raw_source": "test_database"}]


def test_qmin_binary_scoring_is_refused_with_a_clear_message(tmp_path):
    from labeler.events.review import agreement

    p = tree.paths(tmp_path)
    method = spans.METHODS["minimum_safety_factor"]
    with pytest.raises(ValueError, match="multiclass"):
        spans.gold(method, p)
    with pytest.raises(ValueError, match="multiclass"):
        agreement.agreement(p.label_tables / method.event)


def test_rwm_shot_api_exposes_exact_database_onsets(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from labeler.events.review import build
    from labeler.events.ui.app import COOKIE, create_app

    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    p = tree.paths(tmp_path)
    rwm_database(p)
    event = "resistive_wall_mode"
    directory = p.label_tables / event
    (directory / "shots.csv").write_text(
        f"shot,tier,holdout,reviewers,verified_on,notes\n"
        f"{SHOT},unverified,false,,,\n")
    build.build(event, SHOT, p)
    client = TestClient(create_app(p, token="test"))
    client.cookies.set(COOKIE, "test")
    response = client.get("/api/shot", params={"event": event, "shot": SHOT})
    assert response.status_code == 200
    assert response.json()["onsets"] == [{"t_ms": 900.0, "source": "rwm_format_test.csv"}]


@pytest.mark.parametrize("event,name,value,state,unobserved", [
    ("minimum_safety_factor", "qmin", 1.7, 3, 6),
    ("poloidal_beta", "betap", 1.4, 1, 3),
])
def test_scalar_methods_write_class_metadata_and_open_in_the_review_queue(
    tmp_path, event, name, value, state, unobserved,
):
    p = tree.paths(tmp_path)
    t = np.arange(0, 1001, 25)
    features(p, **{name: (t, np.full(t.size, value)),
                  "ip": (t, np.full(t.size, 1e6))})
    tree.cohort(p, [tree.queue_row(SHOT, 0)])
    method = spans.METHODS[event]
    result = spans.run(method, spans.queue(p), p, windows="cohort")
    meta = json.loads(suggestions.table_path(p, method.event, method.name, "v1")
                      .with_suffix(".meta.json").read_text())
    assert meta["categories"][str(state)] == labels.categories(event)[state]
    assert meta["categories"][str(unobserved)] == "not_observable"
    assert result["shots"] == 1
    cohort_rosters.build(method.event, p, point=True)
    source = labels.read_source(p.label_tables / method.event)[SHOT]
    assert source.intervals == ((0, 1000, state),)
