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
    assert len(text.split()) <= fs.CAPTION_MAX_WORDS
    sources = fs.appendix_notes(42, records, {})
    assert "Regime:" in sources
    assert "Ticks" not in text
    assert ("below its bar" in text) == (tier == fs.lf.GENERATED)
    assert "the n=2 ridge is consistent with a harmonic of the n=1 mode" in text
    assert ("the NTM detector (" in text) == (tier == fs.lf.GENERATED)
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
    assert "the n=2 ridge is consistent with a harmonic of the n=1 mode" in text


@pytest.mark.parametrize(
    "frequencies,expected_ms", [([3.42, 6.16], 0), ([20, 41.8], 200)]
)
def test_harmonic_support_uses_relative_ratio_error(frequencies, expected_ms):
    n = np.array([[1, 1], [2, 2]])
    got = fs.harmonic_support(n, np.ones((2, 2), bool), [0, 100], frequencies)
    assert got["support_ms"] == expected_ms


def test_harmonic_fraction_uses_all_columns_with_both_measured_ridges():
    n = np.array([[1, 1, 1, 1], [2, 0, 0, 0], [0, 2, 2, 0]])
    got = fs.harmonic_support(n, n > 0, [0, 100, 200, 300], [8, 16, 14])
    assert got["joint_columns"] == 3
    assert got["passing_columns"] == 1
    assert got["joint_support_ms"] == 300
    assert got["passing_fraction"] == pytest.approx(1 / 3)


def test_sawtooth_expert_review_precedes_physics_states(tmp_path):
    import json

    p = Paths(root=tmp_path, label_tables=tmp_path / "data/events")
    review = p.label_tables / "sawtooth_oscillation/review/labels.csv"
    review.parent.mkdir(parents=True)
    review.write_text("shot,category,t_start,t_end,confidence\n42,1,100,500,\n")
    physics = tmp_path / "42.json"
    physics.write_text(
        json.dumps(
            {
                "shot": 42,
                "crashes": [],
                "states": [{"start_s": 0.1, "end_s": 0.5, "state": "uncertain"}],
            }
        )
    )
    track = fs.sawtooth_track(p, 42, physics, [])
    assert track.file == review
    assert track.source.tier == fs.lf.SILVER
    assert [(r.t_start, r.t_end, r.category) for r in track.rows] == [(100, 500, 1)]


def test_tokeye_fingerprints_change_with_waveform_or_inference_code(tmp_path):
    p = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    p.corpus_file(42).parent.mkdir(parents=True)
    with h5py.File(p.corpus_file(42), "w") as h:
        g = h.create_group("mirnov")
        g.create_dataset("xdata", data=[0, 0.001, 0.002])
        g.create_dataset("ydata", data=[[1, 2, 3]])
    first = fs.tokeye_fingerprints(p, 42, "mirnov", 0, "inference code")
    assert first == fs.tokeye_fingerprints(p, 42, "mirnov", 0, "inference code")
    with h5py.File(p.corpus_file(42), "r+") as h:
        h["mirnov/ydata"][0, 1] = 9
    changed = fs.tokeye_fingerprints(p, 42, "mirnov", 0, "inference code")
    assert first["waveform"] != changed["waveform"]
    assert first["preprocessing"] == changed["preprocessing"]
    code = fs.tokeye_fingerprints(p, 42, "mirnov", 0, "changed inference")
    assert code["preprocessing"] != changed["preprocessing"]


def test_sawtooth_states_and_preinterval_spike_are_disclosed_in_appendix():
    text = fs.appendix_notes(
        42,
        {
            fs.mt.SAWTOOTH: {
                "what": "physics sawtooth states",
                "display_intervals_ms": [],
                "state_intervals_ms": [
                    {"start_ms": 1500, "end_ms": 2332, "state": "uncertain"},
                    {"start_ms": 2332, "end_ms": 3300, "state": "unassessed"},
                ],
                "density_guard": {"cutoff_proxy": True},
            }
        },
        {
            "sawtooth_track_shown": False,
            "first_large_peak_before_expert_ms": 11.5,
            "largest_dalpha_peak_ms": 2296.5,
            "expert_elm_start_ms": 2308,
        },
    )
    assert "Sawtooth: uncertain 832 ms, unassessed 968 ms" in text
    assert "physics labels" in text and "ECE density proxy" in text
    assert "row omitted" not in text and "no present time" not in text
    assert "The largest D-alpha spike (2297 ms) precedes the expert span" in text
    assert "(from 2308 ms)" in text


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
    assert "absent 799 ms, uncertain 10 ms, unassessed 991 ms" in text
    assert "ECE density proxy" in text
    assert "Bt not in local corpus" in text
    assert "is cut off" not in text
    record["state_intervals_ms"] = [
        {"start_ms": 1500, "end_ms": 3300, "state": "absent"}
    ]
    assert fs.sawtooth_caption(record).startswith("Sawtooth: absent 1800 ms")


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
    assert "present 20 ms" in text
    assert "uncertain 500 ms, unassessed 1280 ms" in text
    assert "no Bt" not in text


def test_confinement_intervals_keep_categories_and_regime_names(tmp_path, monkeypatch):
    p = _paths(tmp_path, monkeypatch)
    rows = fs.state_intervals(fs.confinement_track(p, 42), (150, 300))
    assert rows == [
        {"start_ms": 150, "end_ms": 200, "category": 2, "state": "L-mode"},
        {"start_ms": 200, "end_ms": 300, "category": 1, "state": "H-mode"},
    ]
    rows = fs.state_intervals(fs.confinement_track(p, 43), (4400, 5000))
    assert rows[-1]["category"] == 5
    assert rows[-1]["state"] == "uncertain"


def test_sawtooth_display_preserves_short_uncertain_intervals():
    spec = next(s for s in fs.lf.TRACKS if s.key == fs.mt.SAWTOOTH)
    original = (
        fs.lf.Row(0, 100, fs.PRESENT),
        fs.lf.Row(100, 105.5, fs.UNCERTAIN),
        fs.lf.Row(105.5, 200, fs.PRESENT),
        fs.lf.Row(200, 210, fs.UNCERTAIN),
        fs.lf.Row(210, 216.5, fs.UNCERTAIN),
        fs.lf.Row(216.5, 300, fs.NOT_OBSERVABLE),
    )
    track = fs.lf.Track(spec, None, None, original)
    display, changes = fs.sawtooth_display(track, (0, 300))
    assert display.rows == original
    assert changes == []
    assert track.rows == original
    assert fs.has_present_time(track, (20, 40))
    assert not fs.has_present_time(track, (200, 300))


def test_present_time_requires_positive_duration_overlap():
    spec = next(s for s in fs.lf.TRACKS if s.key == fs.mt.SAWTOOTH)
    track = fs.lf.Track(
        spec,
        rows=(
            fs.lf.Row(0, 10, fs.PRESENT),
            fs.lf.Row(10, 100, fs.UNCERTAIN),
            fs.lf.Row(100, 100, fs.PRESENT),
        ),
    )
    assert not fs.has_present_time(track, (10, 110))
    assert fs.has_present_time(track, (9, 11))


def test_sawtooth_display_does_not_bridge_unassessed_gaps():
    spec = next(s for s in fs.lf.TRACKS if s.key == fs.mt.SAWTOOTH)
    track = fs.lf.Track(
        spec,
        rows=(
            fs.lf.Row(0, 50, fs.PRESENT),
            fs.lf.Row(60, 65, fs.UNCERTAIN),
        ),
    )
    display, changes = fs.sawtooth_display(track, (0, 65))
    assert display.rows == track.rows
    assert changes == []


def test_late_line_band_includes_all_late_untagged_pixels():
    t = np.arange(0, 501, 10)
    f = np.array([90.0, 100.0, 110.0, 120.0])
    mask = np.zeros((4, len(t)), bool)
    mask[1:3, 10:31] = True  # 200 ms at 100–110 kHz
    mask[3, 45:] = True  # shorter late line must also enter the reported band
    got = fs.late_untagged_lines(mask, t, f, t >= 100)
    assert got["band_khz"] == [100, 120]
    assert got["first_time_ms"] == 100
    assert got["last_time_ms"] == 500
    assert fs.late_untagged_lines(mask, t, f, t >= 450)["band_khz"] == [120, 120]
    assert fs.late_untagged_lines(mask, t, f, t > 500) is None


def test_caption_omits_absent_highlights_and_expert_elm_claims():
    records = {
        k: {"tier": fs.lf.GENERATED, "what": "source", "title": k}
        for k in (fs.mt.AE, fs.mt.NTM, "edge_localized_mode")
    }
    text = fs.caption(
        42,
        records,
        {
            "blobs": {"tagged": {fs.mt.AE: 0, fs.mt.NTM: 0}},
            "sawtooth_track_shown": False,
            "elm_peaks_in_label": 2,
            "elm_crowd_spans_ms": [],
        },
    )
    assert "AE:" not in text and "NTM:" not in text
    assert "Pink" not in text and "NTM outlines" not in text
    assert "ELMs: detector" in fs.appendix_notes(42, records, {})
    assert "circles" not in text and "expert ELM" not in text
    assert "Triangles:" not in text
    assert "frame model not shown" not in text


def test_appendix_discloses_ae_bins_and_data_derived_late_band():
    record = {"tier": fs.lf.GENERATED, "what": "ae-ours", "title": "AE"}
    text = fs.appendix_notes(
        42,
        {fs.mt.AE: record},
        {
            "blobs": {"tagged": {fs.mt.AE: 1, fs.mt.NTM: 0}},
            "late_untagged_high_frequency": {"band_khz": [105, 125]},
        },
    )
    assert "detector band ≥80 kHz" in text
    assert "25 ms bins" in text
    assert "105–125 kHz" in text
    primary = fs.appendix_notes(
        201978,
        {fs.mt.AE: record},
        {"late_untagged_high_frequency": {"band_khz": [80, 250]}},
    )
    assert "Magnetic lines at 80–250 kHz remain visible" in primary
    assert "stay untagged because the CO2 AE detector is negative" in primary
    assert "magnetics-only" not in primary
    assert "170–250" not in text


def test_caption_uses_the_fallback_detectors_recorded_bin_duration():
    record = {
        "tier": fs.lf.GENERATED,
        "what": "frame detector",
        "title": "AE",
        "temporal_bin_ms": 10,
    }
    text = fs.appendix_notes(
        42,
        {fs.mt.AE: record},
        {
            "blobs": {"tagged": {fs.mt.AE: 1, fs.mt.NTM: 0}},
        },
    )
    assert "10 ms bins" in text
    assert "25 ms bins" not in text


def test_caption_discloses_elm_hmode_conflicts_and_inferred_lmode():
    text = fs.appendix_notes(
        201973,
        {},
        {
            "elm_hmode_conflicts_ms": [[3013, 3045], [3077, 3124]],
            "lmode_inferred": True,
        },
    )
    assert "Expert ELM intervals overlap H-mode-detector absent time" in text
    assert "L-mode (inferred)" in text


def _primary_records(ae="ae-ours"):
    return {
        fs.mt.AE: {
            "tier": fs.lf.GENERATED,
            "what": ae,
            "title": "AE",
            "temporal_bin_ms": 25 if ae == "ae-ours" else 10,
        },
        fs.mt.NTM: {
            "tier": fs.lf.GENERATED,
            "what": "detector",
            "title": "NTM",
            "performance": {
                "f1": 0.457472,
                "shots": 761,
                "bar_criteria": {"N1": [["f1", ">=", 0.7], ["recall", ">=", 0.6]]},
            },
            "primary_bars": {"N1": False},
        },
        "confinement": {"tier": fs.lf.GENERATED, "what": "D-alpha", "title": "H-mode"},
        "edge_localized_mode": {"tier": fs.lf.SILVER, "what": "expert"},
        fs.mt.SAWTOOTH: {"tier": fs.lf.GENERATED, "what": "physics states"},
    }


_HARMONIC = {"support_ms": 326.0, "minimum_support_ms": 50.0}


def test_primary_caption_describes_the_figure_and_its_highlights():
    drawn = {
        "harmonic_support": {**_HARMONIC, "n1_median_khz": 9.1, "n2_median_khz": 18.5},
        "harmonic3_support": {**_HARMONIC, "n3_median_khz": 23.0},
        "first_large_peak_before_expert_ms": 11,
        "largest_dalpha_peak_ms": 2297,
        "late_untagged_high_frequency": {"first_time_ms": 2800},
    }
    text = fs.caption(201978, _primary_records(), drawn)
    assert text == (
        "DIII-D shot 201978. Top: raw Mirnov spectrogram (axis split at 30 and 55 "
        "kHz; bands normalised separately), D-alpha, NBI power. Middle: TokEye "
        "coherent-mode mask after small-object removal; below 30 kHz coloured by "
        "toroidal mode number n (Mirnov array). Pink: mask pixels ≥80 kHz while "
        "the CO2 AE detector is positive (25 ms bins). Orange outlines: n=1/2 "
        "pixels while the NTM detector (held-out F1 0.46, below our 0.7 bar) is "
        "positive. Highlights mark time/band coincidence only; n=2 and n=3 "
        "ridges are consistent with harmonics of the n=1 mode. Bottom: label "
        "tracks with sources."
    )
    for shorthand in ("below bar", "circularity", "four-state", "first ELM"):
        assert shorthand not in text


def test_caption_has_no_shot_specific_branches():
    drawn = {
        "first_large_peak_before_expert_ms": 11,
        "largest_dalpha_peak_ms": 2297,
        "expert_elm_start_ms": 2308,
    }
    records = _primary_records()
    assert fs.caption(201978, records, drawn).replace("201978", "7") == fs.caption(
        7, records, drawn
    )
    assert fs.appendix_notes(201978, records, drawn).replace(
        "201978", "7"
    ) == fs.appendix_notes(7, records, drawn)


def test_harmonic_clause_follows_the_recorded_support_of_each_ridge():
    one = {"harmonic_support": _HARMONIC}
    both = {**one, "harmonic3_support": _HARMONIC}
    weak = {"support_ms": 20.0, "minimum_support_ms": 50.0}
    assert "the n=2 ridge is consistent with a harmonic" in fs.caption(1, {}, one)
    assert "n=2 and n=3 ridges are consistent with harmonics" in fs.caption(1, {}, both)
    assert "harmonic" not in fs.caption(1, {}, {"harmonic_support": weak})
    third = {"harmonic_support": weak, "harmonic3_support": _HARMONIC}
    assert "the n=3 ridge is consistent with a harmonic" in fs.caption(1, {}, third)


def test_harmonic_support_accepts_a_three_to_one_ridge():
    n = np.array([[1, 1], [3, 3]])
    got = fs.harmonic_support(n, np.ones((2, 2), bool), [0, 100], [8, 24], order=3)
    assert got["support_ms"] == 200
    assert got["n3_median_khz"] == 24
    off = fs.harmonic_support(n, np.ones((2, 2), bool), [0, 100], [8, 20], order=3)
    assert off["support_ms"] == 0


def test_alternate_captions_carry_their_recorded_qualifications():
    drawn = {
        "elm_hmode_conflicts_ms": [[3013, 3045]],
        "ae_physical_review_caveat": "Persistent pink lines may be pickup.",
    }
    text = fs.caption(186636, _primary_records(), drawn)
    assert "Expert ELM intervals and the H-mode detector disagree" in text
    assert text.endswith("Persistent pink lines may be pickup.")
    assert "disagree" not in fs.caption(186636, _primary_records(), {})


def test_ae_text_follows_the_ae_source():
    ours = fs.appendix_notes(1, _primary_records(), {})
    frame = fs.appendix_notes(1, _primary_records("the AE frame model"), {})
    assert "AE targets used TokEye's mask" in ours
    assert "AE targets used TokEye's mask" not in frame
    assert "matches the ae-ours input band" in ours
    assert "ae-ours" not in frame
    assert "frame detector's input band" in frame
    assert "trained on the owner's reviewed AE labels" in frame
    assert "up-weights its MHD-absent frames" in frame
    assert "CO2 neural detector" in ours and "CO2 frame detector" in frame
    assert "10 ms bins" in frame


def test_thresholds_are_listed_only_for_detector_tracks():
    records = _primary_records()
    assert (
        "Operating probability thresholds: TokEye 0.2; AE 0.5; NTM 0.63."
        in fs.appendix_notes(1, records, {})
    )
    records[fs.mt.NTM] = {"tier": fs.lf.LEGACY, "what": "archive", "title": "NTM"}
    records[fs.mt.AE]["tier"] = fs.lf.SILVER
    text = fs.appendix_notes(1, records, {})
    assert "Operating probability thresholds: TokEye 0.2." in text
    assert "NTM 0.63" not in text and "AE 0.5" not in text
    assert "NTM detector" not in text and "AE highlights" not in text


def test_appendix_states_the_persistent_row_step_and_its_effect():
    quiet = fs.appendix_notes(1, {}, {"persistent_line_rows": {"wide": 0, "zoom": 0}})
    assert "persistent-row step" in quiet and "not an identified pickup line" in quiet
    assert "No row reached the persistent share" in quiet
    busy = fs.appendix_notes(1, {}, {"persistent_line_rows": {"wide": 3, "zoom": 1}})
    assert "3 wide-pass and 1 zoom-pass rows" in busy
    assert "persistent-row step" in fs.appendix_notes(1, {}, {})


def test_appendix_states_the_ntm_outline_display_rule_and_harmonic_numbers():
    drawn = {
        "ntm_outline_display": {"min_px": 100, "omitted_fragments": 9},
        "harmonic_support": {
            **_HARMONIC,
            "n1_median_khz": 9.1,
            "n2_median_khz": 18.5,
        },
        "harmonic3_support": {
            **_HARMONIC,
            "n1_median_khz": 7.8,
            "n3_median_khz": 23.0,
        },
    }
    text = fs.appendix_notes(1, _primary_records(), drawn)
    assert "fewer than 100 print pixels are not drawn (9 fragments omitted" in text
    assert "n=1 9.1 kHz, n=2 18.5 kHz" in text
    assert "n=1 7.8 kHz, n=3 23.0 kHz" in text
    assert "near three times the n=1 frequency" in text
    assert "not separate islands" in text


def test_ntm_source_text_names_the_bar_from_the_evaluation():
    record = _primary_records()[fs.mt.NTM]
    assert fs.ntm_qualification(record) == "held-out F1 0.46, below our 0.7 bar"
    assert fs.ntm_description(record) == (
        "detector (suggestions; held-out F1 0.46 on 761 shots, below our 0.7 bar)"
    )
    assert fs.ntm_description({}) == "detector (suggestions)"
    record["primary_bars"] = {"N1": True}
    assert fs.ntm_qualification(record).endswith("meets our 0.7 bar")


def test_sawtooth_row_source_names_why_it_is_blank():
    rows = [{"state": "uncertain"}, {"state": "unassessed"}]
    guard = {"cutoff_proxy": True}
    assert fs.sawtooth_row_source(rows, guard) == (
        "physics labels\nblank: not assessable\n(ECE cut-off)"
    )
    assert fs.sawtooth_row_source(rows, None) == "physics labels"
    assert fs.sawtooth_row_source([{"state": "absent"}], guard) == "physics labels"


def test_sawtooth_appendix_says_when_no_sawtooth_is_present():
    records = _primary_records()
    records[fs.mt.SAWTOOTH]["what"] = "physics sawtooth states"
    records[fs.mt.SAWTOOTH]["state_intervals_ms"] = [
        {"start_ms": 0, "end_ms": 10, "state": "uncertain"}
    ]
    text = fs.appendix_notes(1, records, {})
    assert "No sawtooth is labelled present in this window" in text
    assert "four-state" not in text
    records[fs.mt.SAWTOOTH]["state_intervals_ms"].append(
        {"start_ms": 10, "end_ms": 20, "state": "present"}
    )
    assert "No sawtooth is labelled present" not in fs.appendix_notes(1, records, {})


def test_raster_ae_audit_detects_leaks_in_final_pixels():
    rgb = np.zeros((20, 20, 3))
    rgb[4, 5] = [0.87, 0.66, 0.78]
    rgb[4, 16] = [0.87, 0.66, 0.78]  # Outside PRESENT time.
    rgb[18, 5] = [0.87, 0.66, 0.78]  # Below the AE band.
    got = fs.raster_ae_audit(rgb, [0, 0, 1, 1], [0, 200], [55, 250], [(40, 80)])
    assert got["pink_pixels"] == 3
    assert got["outside_present"] == 1
    assert got["below_detector_band"] == 1


def test_raster_ae_audit_accepts_no_ae_tint():
    got = fs.raster_ae_audit(
        np.zeros((20, 20, 3)), [0, 0, 1, 1], [0, 200], [55, 250], []
    )
    assert (
        got["pink_pixels"] == got["outside_present"] == got["below_detector_band"] == 0
    )
