import h5py
import numpy as np
import pandas as pd

from labeler.confinement.labels import (
    balanced_weights,
    bins_from_intervals,
    build,
    merge_intervals,
    read_hdf,
    split_shots,
)


def _intervals(rows):
    return pd.DataFrame(rows, columns=["shot", "t_start", "t_end", "regime", "source"])


def test_hdf_unknown_gaps_and_half_open_bounds(tmp_path):
    p = tmp_path / "bes_signals_123at0.hdf5"
    with h5py.File(p, "w") as f:
        f["time"] = [0.0, 1.0, 2.0, 10.0, 11.0]
        f["labels"] = [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    rows, audit = read_hdf(p)
    assert rows[["t_start", "t_end", "regime"]].values.tolist() == [
        [0.0, 2.0, "L"],
        [10.0, 12.0, "H"],
    ]
    assert audit["unknown_samples"] == 1


def test_conflicts_quarantined_and_subtypes_remain_high():
    rows = _intervals(
        [
            (1, 0, 100, "L", "jalal"),
            (1, 50, 150, "H", "kevin"),
            (2, 0, 100, "H", "jalal"),
            (2, 0, 100, "QH", "kevin"),
        ]
    )
    merged = merge_intervals(rows)
    one = merged.loc[merged.shot == 1]
    assert one.label.tolist() == [0, -1, 1]
    two = merged.loc[merged.shot == 2].iloc[0]
    assert two.label == 1 and two.status == "subtype_conflict"
    assert two.regimes == "H|QH"


def test_bins_require_complete_unambiguous_coverage():
    rows = _intervals(
        [
            (1, 5, 50, "L", "j"),
            (1, 50, 100, "L", "j"),
            (1, 100, 120, "L", "j"),
            (1, 120, 150, "H", "j"),
            (1, 150, 200, "H", "j"),
        ]
    )
    bins = bins_from_intervals(merge_intervals(rows), bin_ms=50)
    assert bins.label.tolist() == [-1, 0, -1, 1]


def test_balancing_equal_class_mass_and_equal_shots_per_class():
    labels = np.array([0, 0, 0, 1, 1, 1])
    shots = np.array([1, 1, 2, 1, 2, 2])
    weights = balanced_weights(labels, shots)
    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())
    assert np.isclose(weights[:2].sum(), weights[2])
    assert np.isclose(weights[3], weights[4:].sum())


def test_test_only_is_shot_wide_and_splits_disjoint():
    targets = pd.DataFrame(
        {"shot": np.repeat(np.arange(20), 2), "label": np.tile([0, 1], 20)}
    )
    split = split_shots(targets, {2, 7}, seed=12)
    assert split.loc[split.shot.isin([2, 7]), "split"].eq("test").all()
    assert split.shot.is_unique
    assert set(split.split) == {"train", "val", "test"}


def test_empty_and_truncated_signal_copy_do_not_destroy_time_labels(tmp_path):
    raw = tmp_path / "raw"
    (raw / "confinement_labels_time").mkdir(parents=True)
    (raw / "confinement_data").mkdir()
    pd.DataFrame(
        {
            "Shot": [123, 123],
            "Confinement Start Time (ms)": [0, 50],
            "Confinement Stop Time (ms)": [50, 100],
            "L": [1, 0],
            "H": [0, 1],
            "QH": [0, 0],
            "WP": [0, 0],
        }
    ).to_csv(raw / "Jalal_28042024_confinement_regime_shotlist.csv", index=False)
    name = "bes_signals_123at0.hdf5"
    with h5py.File(raw / "confinement_labels_time" / name, "w") as f:
        f["time"] = np.arange(100.0)
        f["labels"] = np.eye(4)[np.r_[np.zeros(50, int), np.ones(50, int)]]
    (raw / "confinement_data" / name).write_bytes(b"truncated")
    with h5py.File(raw / "confinement_labels_time/bes_signals_124at0.hdf5", "w") as f:
        f["time"] = np.empty(0)
        f["labels"] = np.empty((0, 4))
    report = build(raw, tmp_path / "out")
    assert report["known_bins"] == 2
    assert report["duplicate_signal_clips"][0]["readable"] is False
    assert any(s.get("empty") for s in report["sources"])


def test_confusion_preserves_source_swaps_at_same_regime_union():
    from labeler.confinement.labels import comparison

    rows = _intervals(
        [
            (1, 0, 50, "H", "a"),
            (1, 50, 100, "QH", "a"),
            (1, 0, 50, "QH", "b"),
            (1, 50, 100, "H", "b"),
        ]
    )
    report, _ = comparison(rows)
    matrix = np.asarray(report["a vs b"]["duration_confusion_ms"])
    assert matrix[1, 2] == matrix[2, 1] == 50
