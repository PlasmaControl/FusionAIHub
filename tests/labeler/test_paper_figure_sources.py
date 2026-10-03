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


def test_stored_ae_uses_valid_causal_windows_and_half_threshold(tmp_path):
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


def test_crash_override_is_clipped_to_present_spans(tmp_path):
    path = tmp_path / "crashes.csv"
    path.write_text("shot,t_ms\n42,9\n42,10\n42,19\n42,20\n43,15\n")
    times, record = fs.crash_times(Paths(root=tmp_path), 42, [(10, 20)], path)
    assert times.tolist() == [10, 19]
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


def test_paper_ae_predictions_take_precedence_when_expert_intervals_also_exist(
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
    assert track.source.what.startswith("ae-ours")
    assert track.file == p.labels_file(42)
