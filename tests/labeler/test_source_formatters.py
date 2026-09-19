"""Regression checks for source-specific event formatting."""

import io
import json
import pickle
import tarfile

import h5py
import numpy as np
import pytest

from labeler.events.source_formatters import (
    format_ae,
    format_rwm,
    format_tm,
)


def test_rwm_preserves_onsets_duplicates_and_mode_number(tmp_path):
    path = tmp_path / "rwm_onsets_2017.csv"
    path.write_text(
        "SHOT,ONSET_TIME,NTOR,MODE_TYPE\n156787,1616.9,2,n2rwm\n156787,1616.9,2,n2rwm\n"
    )
    frame = format_rwm(path)
    assert len(frame) == 2
    assert frame.t0_s.tolist() == [1.6169, 1.6169]
    assert frame.t1_s.tolist() == frame.t0_s.tolist()
    assert json.loads(frame["attrs"].iloc[0])["NTOR"] == 2


def test_ae_excludes_lfm_and_combines_overlapping_classes(tmp_path):
    path = tmp_path / "ae.pkl"
    labels = np.array(
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]], float
    )
    with path.open("wb") as f:
        pickle.dump([[170815], [{}], [labels], [], [], []], f)
    frame = format_ae(path, start_s=0, stop_s=2)
    assert frame[["t0_s", "t1_s"]].values.tolist() == [
        [0.0, 0.5],
        [0.5, 2.0],
    ]
    assert [json.loads(v)["category"] for v in frame["attrs"]] == [0, 1]
    assert set(frame.phenomenon) == {"ae"}
    assert frame.confidence.isna().all()


def test_tm_preserves_separate_sources_and_uses_last_sample_boundary(tmp_path):
    h5path = tmp_path / "tm_labels.h5"
    with h5py.File(h5path, "w") as f:
        g = f.create_group("140004")
        g["time"] = [0, 20, 40, 60, 80]
        g["label"] = [0, 1, 1, 0, 1]
        g["has_tm"] = True
        quiet = f.create_group("140005")
        quiet["time"] = [0, 20]
        quiet["label"] = [0, 0]
        quiet["has_tm"] = False
    tarpath = tmp_path / "tm_labels.tar"
    with tarfile.open(tarpath, "w") as archive:
        buffer = io.BytesIO()
        np.save(buffer, np.array([[0, 20, 40], [0, 1, 0]]))
        member = tarfile.TarInfo("tm_labels/140004_ntm.npy")
        member.size = buffer.tell()
        buffer.seek(0)
        archive.addfile(member, buffer)
    frame = format_tm(h5path, tarpath)
    assert frame[["t0_s", "t1_s"]].values.tolist() == [
        [0.02, 0.04],
        [0.08, 0.08],
        [0.02, 0.02],
    ]
    assert set(frame.source) == {"database:tm_labels_h5", "database:tm_labels_archive"}
    assert set(frame.shot) == {140004}
    assert set(frame.phenomenon) == {"tearing"}
    assert not (tmp_path / "tm_labels").exists()


@pytest.mark.parametrize(
    "times,labels", [([0, 0], [0, 1]), ([0, 20], [0, float("nan")]), ([0, 20], [0, -1])]
)
def test_tm_rejects_invalid_traces(tmp_path, times, labels):
    path = tmp_path / "tm_labels.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("140004")
        group["time"] = times
        group["label"] = labels
    with pytest.raises(ValueError):
        format_tm(path)


def test_tm_does_not_join_positive_samples_across_a_large_gap():
    from labeler.events.source_formatters import format_tm_traces

    traces = [(140004, [0, 20, 40, 1000, 1020], [1, 1, 1, 1, 1], {})]
    frame = format_tm_traces(traces, "tm_labels_h5")
    assert frame[["t0_s", "t1_s"]].values.tolist() == [[0, 0.04], [1, 1.02]]


def test_category_writer_respects_manifest_and_is_reproducible(tmp_path):
    import hashlib

    import pandas as pd
    import yaml

    from labeler.events.source_formatters import convert_category

    raw = tmp_path / "resistive_wall_mode/raw/rwm_onsets_2017.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("SHOT,ONSET_TIME,NTOR,MODE_TYPE\n158015,2613,2,n2rwm\n")
    before = raw.read_bytes()
    (tmp_path / "events.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "format_datasets": [
                    {
                        "name": "resistive_wall_mode",
                        "raw_path": "example_v1.csv",
                        "date": "2026-09-14T00:00:00Z",
                        "sources": ["rwm_onsets_2017"],
                    }
                ],
                "raw_datasets": [
                    {
                        "stem": "rwm_onsets_2017",
                        "path": raw.relative_to(tmp_path).as_posix(),
                        "provenance": "fixture",
                    }
                ],
            }
        )
    )
    out = convert_category("resistive_wall_mode", tmp_path)
    assert out == tmp_path / "resistive_wall_mode/format/example_v1.csv"
    first = out.read_bytes(), out.with_suffix(".meta.json").read_bytes()
    convert_category("resistive_wall_mode", tmp_path)
    assert first == (out.read_bytes(), out.with_suffix(".meta.json").read_bytes())
    meta = json.loads(first[1])
    assert meta["made_from"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert meta["n_rows"] == meta["n_shots"] == 1
    assert pd.read_csv(out).t_start.tolist() == [2613.0]
    with pytest.raises(ValueError, match="raw"):
        convert_category("resistive_wall_mode", tmp_path, output=raw)
    assert raw.read_bytes() == before


def test_elm_deduplicates_wpqh_slices_and_counts_in_50ms_bins(tmp_path):
    from labeler.events.interval_tables import project_intervals
    from labeler.events.source_formatters import format_elm, read_elm

    base, subset = tmp_path / "full.pkl", tmp_path / "wpqh.pkl"
    times, labels = np.arange(100), np.zeros(100)
    labels[[2, 49]] = 1
    base.write_bytes(pickle.dumps({123: {"times": times, "labels": labels}}))
    subset.write_bytes(
        pickle.dumps({"123_0": {"times": times[1:60], "labels": labels[1:60]}})
    )
    traces = list(read_elm(base, subset))
    assert len(traces) == 1 and len(traces[0][1]) == 100
    table = project_intervals(format_elm(base, subset))
    assert table.category.tolist() == [1, 0]
    assert table.t_start.tolist() == [0, 50]
    assert table.t_end.tolist() == [50, 100]
    labels[2] = 0
    subset.write_bytes(pickle.dumps({"123_0": {"times": times, "labels": labels}}))
    with pytest.raises(ValueError, match="conflicting"):
        list(read_elm(base, subset))


def test_confinement_preserves_unknown_and_uses_explicit_flags(tmp_path):
    from labeler.events.interval_tables import project_intervals
    from labeler.events.source_formatters import format_confinement

    path = tmp_path / "regimes.csv"
    path.write_text(
        "Shot,Confinement Start Time (ms),Confinement Stop Time (ms),L,H,QH,WP\n"
        "1,60,110,1,,,\n1,110,180,,1,,\n1,250,280,,,1,\n"
        "1,300,350,,,,1\n1,350,400,,,,\n2,,,,,,\n"
        "1,60,110,1,,,\n"
    )
    high, hgrids, report = format_confinement(path, "high_confinement_mode")
    low, lgrids, _ = format_confinement(path, "low_confinement_mode")
    assert report["unlabelled_rows"] == 2
    assert report["duplicate_intervals_removed"] == 1
    assert report["unlabelled_rows_without_valid_bounds"] == 1
    np.testing.assert_equal(
        hgrids[1]["label"][:8], [np.nan, 0, 1, 1, np.nan, 1, 1, np.nan]
    )
    np.testing.assert_equal(
        lgrids[1]["label"][:8], [np.nan, 1, 1, 0, np.nan, 0, 0, np.nan]
    )
    assert np.isnan(hgrids[2]["label"]).all()
    assert set(project_intervals(high).category) == {0, 1}
    assert set(project_intervals(low).category) == {0, 1}
    _, h_only, _ = format_confinement(path, "high_confinement_mode", high_regimes=["H"])
    assert h_only[1]["label"][5] == h_only[1]["label"][6] == 0


@pytest.mark.parametrize(
    "rows",
    [
        "1,10,0,1,,,\n",
        "1,0,100,1,1,,\n",
        "1,0,100,1,,,\n1,50,150,,1,,\n",
    ],
)
def test_confinement_rejects_invalid_or_conflicting_annotations(tmp_path, rows):
    from labeler.events.source_formatters import read_confinement

    path = tmp_path / "regimes.csv"
    path.write_text(
        "Shot,Confinement Start Time (ms),Confinement Stop Time (ms),L,H,QH,WP\n" + rows
    )
    with pytest.raises(ValueError):
        read_confinement(path)


def test_tm_grids_merge_sources_preserve_negatives_and_unseen_bins(tmp_path):
    from labeler.events.source_formatters import tm_label_grids

    h5path = tmp_path / "labels.h5"
    with h5py.File(h5path, "w") as store:
        for shot, labels in [("1", [0, 0]), ("2", [0, 0])]:
            group = store.create_group(shot)
            group["time"], group["label"] = [0, 50], labels
    tarpath = tmp_path / "labels.tar"
    with tarfile.open(tarpath, "w") as archive:
        buffer = io.BytesIO()
        np.save(buffer, np.array([[10, 160], [1, 0]]))
        member = tarfile.TarInfo("tm_labels/1_ntm.npy")
        member.size = buffer.tell()
        buffer.seek(0)
        archive.addfile(member, buffer)
    grids = tm_label_grids(h5path, tarpath)
    np.testing.assert_equal(grids[1]["label"][:5], [1, 0, np.nan, 0, np.nan])
    np.testing.assert_equal(grids[2]["label"][:3], [0, 0, np.nan])
