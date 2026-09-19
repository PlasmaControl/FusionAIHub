"""Public interval tables use milliseconds and a category column."""

import json

import pandas as pd
import pytest

from labeler.events.interval_tables import (
    INTERVAL_COLUMNS,
    project_intervals,
    validate_intervals,
    write_interval_table,
)


def test_projection_converts_seconds_and_keeps_unknown_confidence():
    original = pd.DataFrame(
        {
            "shot": [170815, 170815],
            "t0_s": [0.5, 1.25],
            "t1_s": [1, 1.25],
            "confidence": [0.8, float("nan")],
            "attrs": ["{}", "{}"],
            "source": ["database:one", "database:two"],
        }
    )
    result = project_intervals(original)
    assert tuple(result.columns) == INTERVAL_COLUMNS
    assert result[["shot", "t_start", "t_end"]].values.tolist() == [
        [170815, 500, 1000],
        [170815, 1250, 1250],
    ]
    assert result.confidence.iloc[0] == 0.8
    assert pd.isna(result.confidence.iloc[1])
    assert original.t0_s.tolist() == [0.5, 1.25]


def test_writer_sorts_intervals_and_preserves_point_events(tmp_path):
    frame = pd.DataFrame(
        [[2, 1, 30, 30, None], [1, 1, 10, 20, 0.9]],
        columns=INTERVAL_COLUMNS,
    )
    path = tmp_path / "format.csv"
    write_interval_table(frame, path, {"made_from": ["source"]})
    result = pd.read_csv(path)
    assert result.shot.tolist() == [1, 2]
    assert result.t_end.tolist() == [20, 30]
    assert tuple(result.columns) == INTERVAL_COLUMNS
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["time_units"] == "ms"
    assert meta["n_rows"] == meta["n_shots"] == 2


@pytest.mark.parametrize(
    "over",
    [
        {"category": "tae"},
        {"category": -1},
        {"category": 1.5},
        {"category": None},
        {"shot": 1.5},
        {"t_start": float("inf")},
        {"t_end": -1},
        {"confidence": "unknown"},
        {"confidence": 1.1},
    ],
)
def test_interval_validation_rejects_corrupt_values(over):
    row = {
        "shot": 1,
        "category": 1,
        "t_start": 0,
        "t_end": 1,
        "confidence": None,
    }
    row.update(over)
    with pytest.raises(ValueError):
        validate_intervals(pd.DataFrame([row]))


def test_sparse_grid_roundtrip_preserves_integer_classes_zeros_and_unknown(tmp_path):
    import numpy as np

    from labeler.events.interval_tables import read_label_grid, write_label_grid

    labels = np.zeros((3, 20))
    labels[0, 4] = 1
    labels[1, 4] = 3
    labels[2, 4] = np.nan
    path = tmp_path / "170815.npz"
    write_label_grid(path, [0, 25, 50], labels, rho_edges=np.linspace(0, 1, 21))
    result = read_label_grid(path)
    np.testing.assert_equal(result["label"], labels)
    assert result["time_ms"].tolist() == [0, 25, 50]
    assert result["rho_edges"].shape == (21,)
    with np.load(path, allow_pickle=False) as sparse:
        assert sparse["values"].tolist() == [1, 3]
        assert sparse["indices"].tolist() == [[0, 4], [1, 4]]


def test_scalar_labels_broadcast_across_all_twenty_rho_bins(tmp_path):
    import numpy as np

    from labeler.events.interval_tables import read_label_grid, write_label_grid

    path = tmp_path / "1.npz"
    write_label_grid(path, [100, 125, 150, 175], [0, 1, 1, 0])
    result = read_label_grid(path)
    np.testing.assert_equal(
        result["label"], np.repeat([[0], [1], [1], [0]], 20, axis=1)
    )
    assert result["rho_edges"].size == 21
    with pytest.raises(ValueError):
        write_label_grid(path, [0], np.ones((1, 19)))


def test_sampled_event_values_use_classes_and_preserve_point_timestamps():
    from labeler.events.interval_tables import sample_event_labels

    events = pd.DataFrame(
        {
            "shot": [1, 1],
            "t0_s": [0.010, 0.075],
            "t1_s": [0.050, 0.075],
            "phenomenon": ["qmin_hybrid", "qmin_high"],
        }
    )
    result = sample_event_labels(
        events, [0, 50, 100], class_ids={"qmin_hybrid": 2, "qmin_high": 4}
    )
    assert result["time_ms"].tolist() == [0, 50, 100]
    import numpy as np

    np.testing.assert_equal(result["label"], [2, 4, np.nan])


def test_event_loader_reads_public_millisecond_tables(tmp_path):
    from labeler.events.databases import TableSpec, events_for_shot

    path = tmp_path / "resistive_wall_mode/format/example.csv"
    path.parent.mkdir(parents=True)
    path.write_text("shot,t_start,t_end,confidence\n1,1000,1500,0.8\n")
    spec = TableSpec(
        stem="example",
        dir="resistive_wall_mode",
        phenomenon="rwm",
        kind="interval",
        shot_col="shot",
        t_units="ms",
        provenance="test",
        format_stem="example",
    )
    events, _records = events_for_shot(1, [spec], root=tmp_path)
    assert len(events) == 1
    assert (events[0].t0_s, events[0].t1_s) == (1, 1.5)
    assert events[0].source == "database:example"
    assert events[0].confidence == 0.8


def test_event_manifest_loads_implemented_outputs_and_skips_pending(tmp_path):
    from labeler.events.databases import load_manifest

    (tmp_path / "events.yaml").write_text("""version: 1
format_datasets:
  - name: neoclassical_tearing_mode
    raw_path: tearing_v1.csv
    date: "2026-09-14T00:00:00Z"
    abbreviation: ntm
    sources: [tm_labels_h5]
  - name: edge_localized_mode
    raw_path: elm_v1.csv
    date:
    abbreviation: elm
    sources: [elm_labels]
raw_datasets:
  - stem: tm_labels_h5
    path: neoclassical_tearing_mode/raw/tm_labels.h5
    provenance: source
""")
    specs = load_manifest(tmp_path)
    assert len(specs) == 1
    assert specs[0].phenomenon == "tearing"
    assert specs[0].format_stem == "tearing_v1"


def test_projection_uses_integer_qmin_and_binary_ae_categories():
    events = pd.DataFrame(
        {
            "shot": [1, 1, 1, 1, 2],
            "t0_s": [0, 1, 2, 3, 1],
            "t1_s": [0.5, 1.5, 2.5, 3.5, 2],
            "confidence": [None] * 5,
            "phenomenon": [
                "qmin_low",
                "qmin_hybrid",
                "qmin_elevated",
                "qmin_high",
                "ae",
            ],
            "attrs": ["{}", "{}", "{}", "{}", '{"class": "tae"}'],
        }
    )
    result = project_intervals(events)
    assert result.category.tolist() == [1, 2, 3, 4, 1]


def test_categorical_sampling_does_not_label_unclassified_times_as_low():
    import numpy as np

    from labeler.events.interval_tables import sample_event_labels

    events = pd.DataFrame({"t0_s": [0.025], "t1_s": [0.05], "phenomenon": ["qmin_low"]})
    result = sample_event_labels(events, [0, 50, 100], class_ids={"qmin_low": 1})
    np.testing.assert_equal(result["label"], [1, np.nan, np.nan])


def test_zero_categories_are_not_loaded_as_positive_events(tmp_path):
    from labeler.events.databases import TableSpec, events_for_shot

    path = tmp_path / "alfven_eigenmode/format/example.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "shot,category,t_start,t_end,confidence\n1,0,0,50,\n1,1,50,100,\n2,0,0,100,\n"
    )
    spec = TableSpec(
        stem="example",
        dir="alfven_eigenmode",
        phenomenon="ae",
        kind="interval",
        shot_col="shot",
        t_units="ms",
        provenance="test",
        format_stem="example",
    )
    events, records = events_for_shot(1, [spec], root=tmp_path)
    assert len(events) == records[0]["n_events"] == 1
    assert events[0].t0_s == 0.05
    events, records = events_for_shot(2, [spec], root=tmp_path)
    assert events == [] and records[0]["n_events"] == 0


def test_binary_downsampling_counts_onsets_and_preserves_unknown_bins(tmp_path):
    import numpy as np

    from labeler.events.interval_tables import (
        bin_binary_samples,
        grid_intervals,
        read_label_grid,
        write_label_grid,
    )

    grid = bin_binary_samples([1, 2, 49, 50, 151], [0, 1, 1, 1, 0])
    assert grid["time_ms"].tolist() == [0, 50, 100, 150]
    assert grid["event_count"].tolist() == [2, 1, 0, 0]
    np.testing.assert_equal(grid["label"], [1, 1, np.nan, 0])
    rows = grid_intervals(1, grid)
    assert rows[["category", "t_start", "t_end"]].values.tolist() == [
        [1, 0, 100],
        [0, 150, 200],
    ]
    path = tmp_path / "1.npz"
    write_label_grid(
        path, grid["time_ms"], grid["label"], event_count=grid["event_count"]
    )
    restored = read_label_grid(path)
    assert restored["event_count"].tolist() == [2, 1, 0, 0]


def test_categorical_bins_use_duration_then_larger_id_for_ties():
    from labeler.events.interval_tables import sample_event_labels

    events = pd.DataFrame(
        {
            "t0_s": [0, 0.01, 0.05, 0.075],
            "t1_s": [0.01, 0.05, 0.075, 0.1],
            "phenomenon": ["low", "high", "low", "high"],
        }
    )
    grid = sample_event_labels(events, [0, 50], class_ids={"low": 1, "high": 4})
    assert grid["label"].tolist() == [4, 4]


def test_old_time_column_names_can_be_projected_into_the_new_schema():
    frame = pd.DataFrame(
        {
            "shot": [1],
            "category": [1],
            "t_start_ms": [50],
            "t_end_ms": [100],
            "confidence": [None],
        }
    )
    projected = project_intervals(frame)
    assert list(projected) == ["shot", "category", "t_start", "t_end", "confidence"]
    assert projected.t_start.tolist() == [50]
    assert projected.t_end.tolist() == [100]


def test_format_shot_grids_preserve_zeros_unknowns_and_point_events(tmp_path):
    import numpy as np

    from labeler.events.interval_tables import (
        formatted_binary_grids,
        read_label_grid,
        write_label_grid,
    )

    frame = pd.DataFrame(
        [[1, 0, 0, 50, None], [1, 1, 70, 70, None], [1, 1, 6000, 6000, None]],
        columns=INTERVAL_COLUMNS,
    )
    grid = dict(formatted_binary_grids(frame))[1]
    np.testing.assert_equal(grid["label"][:3], [0, 1, np.nan])
    assert grid["time_ms"][-1] == 6000 and grid["label"][-1] == 1
    path = tmp_path / "1.npz"
    write_label_grid(
        path, grid["time_ms"], grid["label"], categories={"0": "absent", "1": "present"}
    )
    loaded = read_label_grid(path)
    assert loaded["categories"] == {"0": "absent", "1": "present"}
    np.testing.assert_equal(loaded["label"][:, 0], grid["label"])
