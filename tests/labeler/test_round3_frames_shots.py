"""Round three, Task 2.7: each frame model's eligible shots, split and owner saves."""

from __future__ import annotations

import hashlib
import json
import logging

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler import frames
from labeler.events import raw, spans
from labeler.events.review import labels
from labeler.frames import shots as frames_shots
from labeler.frames import targets

from . import editor_tree, frames_tree
from .frames_tree import (
    BLIND,
    ELM,
    IP_SHOT,
    IP_WINDOW,
    LABELS_SHOT,
    LABELS_SHOT_HULL,
    LEGACY,
    LEGACY_HULL,
    NTM,
    SAWTOOTH,
    SHOTS,
    SPARE,
    WINDOW,
)

SPLITS = ("train", "val", "test", "owner")
META_KEYS = {
    "method",
    "labelled_shots",
    "positive_shots",
    "seed",
    "counts",
    "owner_snapshot",
    "left_out",
    "window_from",
    "labels_window",
}
#: Each method's (labelled, positive) target shots: the ELM grids but SPARE's,
#: LABELS_SHOT's absent throughout; two of every other target.
LABELLED = {
    "elm_frames": (5, 4),
    "hmode_frames": (2, 2),
    "ntm_frames": (2, 2),
    "sawtooth_frames": (2, 2),
}


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return frames_tree.build(tmp_path_factory.mktemp("frames"))


@pytest.fixture(scope="module")
def made(tree):
    out = {}
    for method in frames.SPECS:
        shots = frames_shots.make(tree, method)
        meta = json.loads(frames.shots_meta_file(tree, method).read_text())
        out[method] = shots, meta
    return out


def _rows(frame) -> list[tuple]:
    return [tuple(row) for row in frame.itertuples(index=False)]


def test_the_left_out_shots_each_with_its_reason(tree, monkeypatch):
    tried = editor_tree.no_fetch(monkeypatch)
    # A missing group is record_tier's None, not a stub; IP_SHOT's is cached.
    assert raw.record_tier(LEGACY, "filterscopes", paths=tree) is None
    assert raw.record_tier(IP_SHOT, "filterscopes", paths=tree) == "cache"
    assert raw.record_tier(SHOTS[ELM], "filterscopes", paths=tree) == "corpus"
    frame, left = frames_shots.eligible(tree, frames.SPECS["elm_frames"])
    assert left == {
        "blind": [BLIND],
        "no filterscopes": [LEGACY],
        "no labelled bin": [SPARE],
    }
    assert list(frame.columns) == list(frames_shots.ELIGIBLE_COLUMNS)
    assert _rows(frame) == [
        (SHOTS[ELM], 1, 1, 1, *WINDOW, "catalog"),  # the owner's save
        (IP_SHOT, 0, 1, 0, *IP_WINDOW, "ip"),
        (LABELS_SHOT, 0, 0, 0, *LABELS_SHOT_HULL, "labels"),
    ]
    for method in ("hmode_frames", "ntm_frames", "sawtooth_frames"):
        spec = frames.SPECS[method]
        frame, left = frames_shots.eligible(tree, spec)
        assert left == {"blind": [BLIND]}
        assert _rows(frame) == [(SHOTS[spec.store_event], 1, 1, 1, *WINDOW, "catalog")]
    assert tried == []


def test_the_catalog_windows_are_read_once(tree, monkeypatch):
    calls = []
    for name in ("queue", "population"):
        real = getattr(spans, name)

        def counted(paths, real=real, name=name):
            calls.append(name)
            return real(paths)

        monkeypatch.setattr(spans, name, counted)

    def refuse(*args, **kwargs):
        raise AssertionError("plasma_window re-reads the tables for every shot")

    monkeypatch.setattr(targets, "plasma_window", refuse)
    frame, _ = frames_shots.eligible(tree, frames.SPECS["elm_frames"])
    assert len(frame) == 3
    assert sorted(calls) == ["population", "queue"]


def test_an_unreadable_file_is_its_own_reason(tmp_path, monkeypatch, caplog):
    p = frames_tree.build(tmp_path)
    editor_tree.no_fetch(monkeypatch)
    # LEGACY has no group anywhere; an unreadable corpus file is not "no group".
    p.corpus.mkdir(parents=True, exist_ok=True)
    p.corpus_file(LEGACY).write_bytes(b"not an hdf5 file")
    # LABELS_SHOT's filterscopes are cached, so it stays; its Ip read fails.
    p.corpus_file(LABELS_SHOT).write_bytes(b"not an hdf5 file")
    caplog.set_level(logging.WARNING, logger="labeler.frames.targets")
    frame, left = frames_shots.eligible(p, frames.SPECS["elm_frames"])
    assert left == {
        "blind": [BLIND],
        "no labelled bin": [SPARE],
        "unreadable corpus file": [LEGACY],
    }
    row = frame.set_index("shot").loc[LABELS_SHOT]
    assert row.window_from == "labels"
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(LABELS_SHOT) in r.getMessage() for r in warned)


def test_a_review_store_with_no_signal_is_left_out(tmp_path, monkeypatch):
    p = frames_tree.build(tmp_path)
    editor_tree.no_fetch(monkeypatch)
    spec = frames.SPECS["sawtooth_frames"]
    shot = SHOTS[SAWTOOTH]
    with h5py.File(frames.store_path(p, spec, shot), "r+") as f:
        for name in f["rows"]:
            for level in f["rows"][name]:
                f["rows"][name][level][...] = np.nan  # the ECE of 187154
    frame, left = frames_shots.eligible(p, spec)
    assert left == {"blind": [BLIND], "owner: no signal in the window": [shot]}
    assert frame.empty
    # A roster shot with no review store, or one without a required row.
    elm = frames.SPECS["elm_frames"]
    frames.store_path(p, elm, SHOTS[ELM]).unlink()
    _, left = frames_shots.eligible(p, elm)
    assert left["owner: no review store"] == [SHOTS[ELM]]
    ntm = frames.SPECS["ntm_frames"]
    with h5py.File(frames.store_path(p, ntm, SHOTS[NTM]), "r+") as f:
        meta = json.loads(f["rows"]["p0"].attrs["meta"])
        f["rows"]["p0"].attrs["meta"] = json.dumps(meta | {"title": "another row"})
    _, left = frames_shots.eligible(p, ntm)
    assert left["owner: unusable review store"] == [SHOTS[NTM]]


def test_the_ip_window_ends_at_a_restrike(tmp_path, monkeypatch):
    p = frames_tree.build(tmp_path)
    editor_tree.no_fetch(monkeypatch)
    t = np.arange(0.0, 3000.0, 0.5)
    ip = np.select(
        [(t >= 200) & (t <= 1500), (t > 1500) & (t < 1700), (t >= 1700) & (t <= 2200)],
        [1e6, 1e5, 8e5],
        0.0,
    )
    editor_tree.write(raw.cache_path(LEGACY, paths=p), {"ip": (t, ip)})
    (lo, hi), found = targets.window_for(LEGACY, p, LEGACY_HULL, catalog={})
    assert (lo, found) == (200, "ip")
    assert 1500 < hi < 1700  # the dip before the second plasma, not 2200 ms
    # A plateau with no restrike keeps its assessed window.
    assert targets.window_for(IP_SHOT, p, None, catalog={}) == (IP_WINDOW, "ip")


def test_the_saved_shots_are_the_owner_split(tree, made):
    for method, spec in frames.SPECS.items():
        shots, meta = made[method]
        source = labels.labels_path(tree.label_tables / spec.event)
        saved = set(labels.read_labels(source))
        assert saved == {SHOTS[spec.store_event]}
        assert set(shots.shot[shots.split == "owner"]) == saved
        assert not saved & set(shots.shot[shots.split != "owner"])
        copy = frames.owner_file(tree, method)
        assert copy.read_bytes() == source.read_bytes()
        snapshot = meta["owner_snapshot"]
        assert snapshot["path"] == str(copy)
        assert snapshot["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    shots, _ = made["elm_frames"]
    assert _rows(shots[shots.split == "owner"]) == [(SHOTS[ELM], "owner", 1, 1)]


def test_without_saves_there_is_no_owner_split(tmp_path, monkeypatch):
    p = frames_tree.build(tmp_path)
    editor_tree.no_fetch(monkeypatch)
    spec = frames.SPECS["ntm_frames"]
    labels.labels_path(p.label_tables / spec.event).unlink()
    shots = frames_shots.make(p, "ntm_frames")
    meta = json.loads(frames.shots_meta_file(p, "ntm_frames").read_text())
    assert (meta["owner_snapshot"]["path"], meta["owner_snapshot"]["sha256"]) == (
        None,
        None,
    )
    assert not frames.owner_file(p, "ntm_frames").exists()
    # The shot the owner had saved is an ordinary target shot now.
    assert _rows(shots) == [(SHOTS[spec.store_event], "train", 1, 1)]


def test_the_split_is_reproducible_and_stratified():
    strata = {(1, 1): 40, (1, 0): 20, (0, 1): 7, (0, 0): 3}
    rows = [
        {"shot": 1000 + 100 * i + k, "positive": positive, "roster": roster}
        for i, ((positive, roster), n) in enumerate(strata.items())
        for k in range(n)
    ]
    frame = pd.DataFrame(rows)
    got = frames_shots.split(frame)
    assert sorted(got) == sorted(frame.shot)
    assert got == frames_shots.split(frame.sample(frac=1.0, random_state=1))
    assert got == frames_shots.split(frame, seed=frames.SEED)
    assert got != frames_shots.split(frame, seed=1)
    # Round-half-up 15 % test and val within each stratum, the rest train.
    want = {40: (28, 6, 6), 20: (14, 3, 3), 7: (5, 1, 1), 3: (3, 0, 0)}
    for (positive, roster), n in strata.items():
        stratum = frame[(frame.positive == positive) & (frame.roster == roster)]
        splits = [got[shot] for shot in stratum.shot]
        assert tuple(splits.count(s) for s in ("train", "val", "test")) == want[n]
    assert frames_shots.SPLIT_FRACTIONS == (0.70, 0.15, 0.15)


def test_the_csv_columns_and_the_json_keys(tree, made):
    for method in frames.SPECS:
        shots, meta = made[method]
        path = frames.shots_file(tree, method)
        assert path.read_text().splitlines()[0] == "shot,split,positive,roster"
        read = pd.read_csv(path)
        pd.testing.assert_frame_equal(read, shots.reset_index(drop=True))
        assert all(read[c].dtype.kind == "i" for c in ("shot", "positive", "roster"))
        assert set(read.positive) <= {0, 1} and set(read.roster) <= {0, 1}
        assert read.shot.is_monotonic_increasing
        assert META_KEYS <= set(meta)
        assert {"path", "sha256"} <= set(meta["owner_snapshot"])
        assert (meta["method"], meta["seed"]) == (method, frames.SEED)
        assert meta["counts"] == {s: int((read.split == s).sum()) for s in SPLITS}
    shots, meta = made["elm_frames"]
    assert _rows(shots) == [
        (IP_SHOT, "train", 1, 0),
        (LABELS_SHOT, "train", 0, 0),
        (SHOTS[ELM], "owner", 1, 1),
    ]
    assert meta["left_out"] == {"blind": 1, "no filterscopes": 1, "no labelled bin": 1}
    assert meta["left_out_shots"]["no filterscopes"] == [LEGACY]
    assert meta["window_from"] == {"catalog": 1, "ip": 1, "labels": 1}
    assert meta["windows"][str(IP_SHOT)] == [*IP_WINDOW, "ip"]
    # LABELS_SHOT's hull, 0-3000 ms, holds 60 labelled bins, and 20 of them lie
    # outside its window in the catalog's Ip log, 400-2400 ms.
    assert meta["labels_window"] == {
        "shots": 1,
        "with_ip_window": 1,
        "labelled_bins": 60,
        "outside_ip_window": 20,
    }


def test_labelled_shots_counts_the_grids_with_a_labelled_bin(made):
    for method, (labelled, positive) in LABELLED.items():
        _, meta = made[method]
        assert (meta["labelled_shots"], meta["positive_shots"]) == (labelled, positive)


def test_force_is_needed_to_overwrite(tmp_path, monkeypatch, capsys):
    p = frames_tree.build(tmp_path)
    editor_tree.use_env(monkeypatch, p)
    tried = editor_tree.no_fetch(monkeypatch)
    assert frames_shots.main(["--method", "elm_frames"]) == 0
    (line,) = capsys.readouterr().out.splitlines()
    out = json.loads(line)
    assert out["method"] == "elm_frames"
    assert out["counts"] == {"train": 2, "val": 0, "test": 0, "owner": 1}
    path = frames.shots_file(p, "elm_frames")
    first = path.read_bytes()
    with pytest.raises(FileExistsError, match="--force"):
        frames_shots.make(p, "elm_frames")
    with pytest.raises(SystemExit) as refused:
        frames_shots.main(["--method", "elm_frames"])
    assert refused.value.code not in (0, None)
    assert path.read_bytes() == first
    assert frames_shots.main(["--method", "elm_frames", "--force"]) == 0
    assert path.read_bytes() == first  # the same split again
    # Once a model is trained on the split, not even --force remakes it.
    model = frames.model_dir(p, "elm_frames") / "model.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"")
    with pytest.raises(FileExistsError, match="model"):
        frames_shots.make(p, "elm_frames", force=True)
    assert tried == []
