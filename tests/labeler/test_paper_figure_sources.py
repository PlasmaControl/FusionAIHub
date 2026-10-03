"""Figure sources respect curated coverage, validity and causal AE bins."""

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.paper import figure_sources as fs


def _paths(tmp_path, monkeypatch):
    p = Paths(root=tmp_path, label_tables=tmp_path / "data/events")
    run = tmp_path / "regimes"
    run.mkdir()
    monkeypatch.setenv("CONFINEMENT_RUN_DIR", str(run))
    (run / "merged_intervals.csv").write_text(
        "shot,t_start,t_end,regimes\n42,100,200,L\n42,200,500,H\n"
    )
    fallback = fs.dalpha_file(p)
    fallback.parent.mkdir(parents=True)
    fallback.write_text(
        "shot,category,t_start,t_end,confidence\n42,1,0,500,\n43,0,0,1799,\n"
        "43,1,1799,4470,\n43,5,4470,5346,\n"
    )
    return p


def test_curated_regimes_win_over_an_early_dalpha_interval(tmp_path, monkeypatch):
    p = _paths(tmp_path, monkeypatch)
    track = fs.confinement_track(p, 42)
    assert [(r.t_start, r.category) for r in track.rows] == [(100, 2), (200, 1)]
    assert track.spec.title == "regime"


def test_dalpha_fallback_names_hmode_without_inventing_lmode(tmp_path, monkeypatch):
    p = _paths(tmp_path, monkeypatch)
    track = fs.confinement_track(p, 43)
    assert track.spec.title == "H-mode"
    assert [(r.t_start, r.category) for r in track.rows] == [
        (0, 0),
        (1799, 1),
        (4470, 5),
    ]


def test_missing_confinement_source_fails_loudly(tmp_path, monkeypatch):
    p = _paths(tmp_path, monkeypatch)
    monkeypatch.setenv("CONFINEMENT_RUN_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="CONFINEMENT_RUN_DIR"):
        fs.confinement_track(p, 43)


def test_stored_ae_uses_valid_causal_windows_and_paper_threshold(tmp_path):
    p = Paths(root=tmp_path)
    p.labels.mkdir()
    with h5py.File(p.labels_file(42), "w") as h:
        for name, y in [
            ("ae_active", [0.1, 0.6, 0.8, 0.5]),
            ("ae_active_valid", [1, 1, 0, 1]),
        ]:
            g = h.create_group(f"d3d_ae_activity_seldnet/{name}")
            g.create_dataset("xdata", data=[0.025, 0.050, 0.075, 0.100])
            g.create_dataset("ydata", data=np.array([y]))
            g.attrs["time_step_ms"] = 25.0
    track = fs.stored_ae_track(p, 42)
    assert track.source.what.startswith("ae-ours")
    assert [(r.t_start, r.t_end, r.category) for r in track.rows] == [
        (0, 25, 0),
        (25, 50, 1),
        (50, 75, 3),
        (75, 100, 1),
    ]
    assert fs.stored_ae_track(p, 43) is None


def test_unverified_crash_override_cannot_supply_ticks(tmp_path, monkeypatch):
    path = tmp_path / "crashes.csv"
    path.write_text("shot,t_ms\n42,9\n42,10\n42,19\n42,20\n43,15\n")
    monkeypatch.setattr(fs, "local_ece_crashes", lambda *a: ([], {}))
    times, record = fs.crash_times(
        Paths(root=tmp_path), 42, [(10, 20)], path, elm_times=[]
    )
    assert times.tolist() == []
    assert record["source"] == str(path)


def test_reviewed_ae_is_available_without_fallback_model(tmp_path):
    p = Paths(root=tmp_path, label_tables=tmp_path / "data/events")
    path = p.label_tables / "alfven_eigenmode/review/labels.csv"
    path.parent.mkdir(parents=True)
    path.write_text("shot,category,t_start,t_end,confidence\n42,1,100,500,\n")
    track = fs.reviewed_or_stored_ae(p, 42)
    assert track.file == path
    assert track.source.tier == "silver: expert review"
    assert [(r.t_start, r.t_end) for r in track.rows] == [(100, 500)]
    assert fs.reviewed_or_stored_ae(p, 43) is None


def test_expert_ae_takes_precedence_when_paper_predictions_also_exist(
    tmp_path,
):
    p = Paths(root=tmp_path, label_tables=tmp_path / "data/events")
    path = p.label_tables / "alfven_eigenmode/review/labels.csv"
    path.parent.mkdir(parents=True)
    path.write_text("shot,category,t_start,t_end,confidence\n42,1,100,500,\n")
    p.labels.mkdir()
    with h5py.File(p.labels_file(42), "w") as h:
        for name, y in [("ae_active", [0.8, 0.2]), ("ae_active_valid", [1, 1])]:
            g = h.create_group(f"d3d_ae_activity_seldnet/{name}")
            g.create_dataset("xdata", data=[0.025, 0.050])
            g.create_dataset("ydata", data=np.array([y]))
    track = fs.reviewed_or_stored_ae(p, 42)
    assert track.source.tier == fs.lf.SILVER
    assert track.file == path


def test_physics_crashes_require_ece_evidence_and_reject_elm_neighbours(tmp_path):
    import json

    attrs = {
        "core_moves": True,
        "central_relative_drop": 0.08,
        "inversion_channel": 20.5,
        "drop_start": 18,
        "drop_stop": 21,
        "rise_start": 21,
        "rise_stop": 24,
        "state": "present",
    }
    path = tmp_path / "42.json"
    events = [
        {"time_s": t / 1000, "confidence": 1, "attrs": attrs.copy()}
        for t in (10, 15, 21, 30, 40, 50)
    ]
    events[3]["attrs"]["core_moves"] = False
    events[4]["attrs"]["state"] = "uncertain"
    events[5]["attrs"]["dalpha_coincident"] = True
    path.write_text(json.dumps({"shot": 42, "crashes": events, "states": []}))
    times, record = fs.crash_times(Paths(root=tmp_path), 42, None, path, elm_times=[15])
    assert times.tolist() == [21]
    assert record["rejected_elm_times_ms"] == [10, 15, 50]
    assert record["verified"] is True
    # Missing D-alpha is missing rejection evidence, not ELM-free evidence.
    times, _ = fs.crash_times(Paths(root=tmp_path), 42, None, path)
    assert not times.size


def test_physics_interval_categories_are_preserved_independently_of_ticks(tmp_path):
    import json

    path = tmp_path / "42.json"
    path.write_text(
        json.dumps(
            {
                "shot": 42,
                "crashes": [],
                "states": [
                    {"start_s": 0.01, "end_s": 0.02, "state": "present"},
                    {"start_s": 0.02, "end_s": 0.03, "state": "uncertain"},
                ],
            }
        )
    )
    track = fs.sawtooth_track(Paths(root=tmp_path), 42, path, [])
    assert [r.category for r in track.rows] == [fs.PRESENT, fs.UNCERTAIN]
    assert track.spec.title == "sawtooth"


def test_independent_projection_audit_detects_an_end_boundary_leak(monkeypatch):
    from labeler.paper import mode_tags as mt

    monkeypatch.setattr(mt, "present_columns", lambda t, spans: np.ones(len(t), bool))
    mask = np.ones((2, 3), bool)
    audit = fs.projection_audit(mask, [9, 10, 20], [59, 60], [(10, 20)], (60, 250))
    assert audit["outside_present"] == 4
    assert audit["outside_band"] == 3


@pytest.mark.parametrize("tier", [fs.lf.SILVER, fs.lf.LEGACY, fs.lf.GENERATED])
def test_caption_follows_sources_and_actual_acceptance_bars(tier):
    records = {
        key: {"tier": tier, "what": "source", "title": key, "primary_bars": None}
        for key in (
            fs.mt.AE,
            fs.mt.NTM,
            fs.mt.SAWTOOTH,
            "confinement",
            "edge_localized_mode",
        )
    }
    records[fs.mt.AE]["what"] = "ae-ours"
    records["confinement"]["title"] = "regime"
    if tier == fs.lf.GENERATED:
        records[fs.mt.NTM]["primary_bars"] = {"N1": False, "all": False}
    text = fs.caption(
        42,
        records,
        {
            "sawtooth_strip_shown": False,
            "harmonic_support": {
                "support_ms": 100,
                "minimum_support_ms": 50,
                "n1_median_khz": 7.5,
                "n2_median_khz": 15,
            },
        },
    )
    assert len(text.split()) <= 150
    assert "regime:" in text
    assert "Ticks" not in text
    assert ("unverified" in text) == (tier == fs.lf.GENERATED)
    assert "consistent with a second harmonic" in text
    assert "near 15 kHz" in text
    for internal in ("ntm_frames", "dalpha_lh", "MPI66M", "N1", "S1", "PRESENT"):
        assert internal not in text


def test_expert_crowd_lane_does_not_promote_uncertain_category():
    assert fs.elm_category(fs.lf.Row(100, 200, fs.UNCERTAIN, crowd=1)) == fs.UNCERTAIN
    assert fs.elm_category(fs.lf.Row(100, 200, fs.PRESENT, crowd=1)) == fs.PRESENT
    assert fs.elm_category(fs.lf.Row(200, 300, fs.UNCERTAIN)) == fs.UNCERTAIN


def test_cohort_csv_can_use_full_physics_json_evidence(tmp_path):
    import json

    evidence = tmp_path / "42.json"
    evidence.write_text(
        json.dumps(
            {
                "shot": 42,
                "crashes": [
                    {
                        "time_s": 0.1,
                        "attrs": {
                            "core_moves": True,
                            "central_relative_drop": 0.08,
                            "inversion_channel": 20,
                            "drop_start": 18,
                            "drop_stop": 21,
                            "rise_start": 21,
                            "rise_stop": 24,
                            "state": "present",
                        },
                    }
                ],
                "states": [],
            }
        )
    )
    labels = tmp_path / "cohort-000.csv"
    labels.write_text("shot,category,t_start,t_end,confidence\n42,1,100,100,1\n")
    times, record = fs.crash_times(
        Paths(root=tmp_path), 42, source=labels, evidence=evidence, elm_times=[]
    )
    assert times.tolist() == [100]
    assert {f["path"] for f in record["files"]} == {str(labels), str(evidence)}
    direct, _ = fs.crash_times(Paths(root=tmp_path), 42, source=evidence, elm_times=[])
    assert direct.tolist() == times.tolist()


def test_partial_uncertain_physics_event_cannot_use_local_fallback(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    path = tmp_path / "42.json"
    path.write_text(
        json.dumps(
            {
                "shot": 42,
                "crashes": [
                    {
                        "time_s": 0.01,
                        "confidence": 1,
                        "attrs": {"state": "uncertain"},
                    }
                ],
            }
        )
    )
    local = SimpleNamespace(
        t0_s=0.01,
        attrs={
            "fall": 0.08,
            "pulse_channel_lo": 21,
            "pulse_channel_stop": 24,
        },
    )
    monkeypatch.setattr(fs, "local_ece_crashes", lambda *a: ([local], {}))
    times, _ = fs.crash_times(Paths(root=tmp_path), 42, source=path, elm_times=[])
    assert not times.size


def test_harmonic_requires_coincident_frequency_ratio_not_just_n_numbers():
    n = np.array([[1, 1], [2, 2]])
    mask = np.ones((2, 2), bool)
    good = fs.harmonic_support(n, mask, [0, 100], [7.5, 15])
    assert good["support_ms"] >= 50
    bad = fs.harmonic_support(n, mask, [0, 100], [4, 15])
    assert bad["support_ms"] == 0
    mask[0, 1] = False
    mask[1, 0] = False
    assert fs.harmonic_support(n, mask, [0, 100], [7.5, 15])["support_ms"] == 0


def test_harmonic_support_follows_ridges_outside_the_primary_shots_bands():
    n = np.array([[1, 1], [2, 2]])
    got = fs.harmonic_support(n, np.ones((2, 2), bool), [0, 100], [11, 22])
    assert got["support_ms"] == 200
    assert got["n1_median_khz"] == 11
    assert got["n2_median_khz"] == 22
    text = fs.caption(42, {}, {"harmonic_support": got, "sawtooth_strip_shown": False})
    assert "near 22 kHz" in text
    assert "near 15 kHz" not in text


@pytest.mark.parametrize(
    "frequencies,expected_ms", [([3.42, 6.16], 0), ([20, 41.8], 200)]
)
def test_harmonic_support_uses_relative_ratio_error(frequencies, expected_ms):
    n = np.array([[1, 1], [2, 2]])
    got = fs.harmonic_support(n, np.ones((2, 2), bool), [0, 100], frequencies)
    assert got["support_ms"] == expected_ms


def test_harmonic_caption_requires_minimum_sampled_support():
    got = fs.harmonic_support(
        np.array([[1, 1], [2, 2]]), np.ones((2, 2), bool), [0, 10], [8, 16]
    )
    assert "harmonic" not in fs.caption(42, {}, {"harmonic_support": got})


def test_sawtooth_caption_uses_window_states_and_qualified_proxy():
    record = {
        "state_intervals_ms": [
            {"start_ms": 1500, "end_ms": 2291, "state": "absent"},
            {"start_ms": 2291, "end_ms": 2301, "state": "uncertain"},
            {"start_ms": 2301, "end_ms": 2309, "state": "absent"},
            {"start_ms": 2309, "end_ms": 3300, "state": "unassessed"},
        ],
        "density_guard": {"cutoff_proxy": True, "status": "fixed_bt_missing"},
    }
    text = fs.sawtooth_caption(record)
    assert "absent to 2.29 s" in text
    assert "uncertain" in text
    assert "unassessed from 2.31 s" in text
    assert "conservative density proxy" in text
    assert "no Bt available" in text
    assert "is cut off" not in text
    record["state_intervals_ms"] = [
        {"start_ms": 1500, "end_ms": 3300, "state": "absent"}
    ]
    assert fs.sawtooth_caption(record) == "Sawtooth absent throughout."


def test_sawtooth_summary_keeps_present_intervals_between_uncertain_and_blank():
    record = {
        "state_intervals_ms": [
            {"start_ms": 1500, "end_ms": 2000, "state": "uncertain"},
            {"start_ms": 2000, "end_ms": 2020, "state": "present"},
            {"start_ms": 2020, "end_ms": 3300, "state": "unassessed"},
        ],
        "density_guard": {"cutoff_proxy": True, "status": "density_and_local_bt"},
    }
    text = fs.sawtooth_caption(record)
    assert "present intervals" in text
    assert "unassessed from 2.02 s" in text
    assert "no Bt" not in text
