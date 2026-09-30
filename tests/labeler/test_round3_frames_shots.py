"""Round three, Task 2.7 and F2: each frame model's eligible shots, split and
owner saves; v2's target is the original's with the owner's labels over it."""

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
from labeler.events.catalog.states import PRESENT
from labeler.events.review import labels
from labeler.frames import shots as frames_shots
from labeler.frames import targets

from . import editor_tree, frames_tree
from .frames_tree import (
    BLIND,
    ELM,
    EVENT_MS,
    HMODE,
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

SPLITS = ("train", "val", "test")
META_KEYS = {
    "method",
    "version",
    "labelled_shots",
    "positive_shots",
    "legacy_labelled_shots",
    "present_s",
    "owner",
    "seed",
    "counts",
    "owner_snapshot",
    "left_out",
    "window_from",
    "labels_window",
}
#: Each method's (labelled, positive) target shots: the ELM grids but SPARE's,
#: LABELS_SHOT's absent throughout; two of every other target. The owner's saves
#: (one shot each, present over EVENT_MS) change none of these counts.
LABELLED = {
    "elm_frames": (5, 4),
    "hmode_frames": (2, 2),
    "ntm_frames": (2, 2),
    "sawtooth_frames": (2, 2),
}
#: Each method's merged present time, s (F4): its present bins times bin_ms.
#: ELM: 16 50 ms bins on each of four grids; NTM: 16 on each of two. H-mode:
#: BLIND's 15 H-only bins (the 600 ms bin is H and L, uncertain) and the saved
#: shot's 16, the owner's present over the uncertain bin; sawteeth: BLIND's 75
#: 10 ms bins (1000-1050 ms uncertain) and the saved shot's 80, likewise.
PRESENT_S = {
    "elm_frames": 3.2,
    "hmode_frames": 1.55,
    "ntm_frames": 1.6,
    "sawtooth_frames": 1.55,
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
        (IP_SHOT, 0, 1, 0, *IP_WINDOW, "ip"),
        (LABELS_SHOT, 0, 0, 0, *LABELS_SHOT_HULL, "labels"),
        (SHOTS[ELM], 1, 1, 1, *WINDOW, "catalog"),  # the owner saved it
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
    assert left == {"blind": [BLIND], "no signal in the window": [shot]}
    assert frame.empty
    # A roster shot with no review store, or one without a required row.
    elm = frames.SPECS["elm_frames"]
    frames.store_path(p, elm, SHOTS[ELM]).unlink()
    _, left = frames_shots.eligible(p, elm)
    assert left["no review store"] == [SHOTS[ELM]]
    ntm = frames.SPECS["ntm_frames"]
    with h5py.File(frames.store_path(p, ntm, SHOTS[NTM]), "r+") as f:
        meta = json.loads(f["rows"]["p0"].attrs["meta"])
        f["rows"]["p0"].attrs["meta"] = json.dumps(meta | {"title": "another row"})
    _, left = frames_shots.eligible(p, ntm)
    assert left["unusable review store"] == [SHOTS[NTM]]


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


def test_the_saved_shots_are_frozen_and_split_with_the_rest(tree, made):
    # F2: no owner split; the owner's shots are split as any other, flagged.
    for method, spec in frames.SPECS.items():
        shots, meta = made[method]
        source = labels.labels_path(tree.label_tables / spec.event)
        saved = set(labels.read_labels(source))
        assert saved == {SHOTS[spec.store_event]}
        assert set(shots.split) <= set(SPLITS)
        assert set(shots.shot[shots.owner == 1]) == saved
        copy = frames.owner_file(tree, method)
        assert copy.parent == frames.shots_file(tree, method).parent
        assert copy.read_bytes() == source.read_bytes()
        snapshot = meta["owner_snapshot"]
        assert snapshot["path"] == str(copy)
        assert snapshot["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert meta["owner"] == {"saved": 1, "overriding": 1, "added": 0}
        assert set(meta["counts"]) == set(SPLITS)
    shots, _ = made["elm_frames"]
    assert _rows(shots[shots.owner == 1]) == [(SHOTS[ELM], "train", 1, 1, 1)]


def test_a_saved_shot_the_original_lacks_is_added(tmp_path, monkeypatch):
    p = frames_tree.build(tmp_path)
    editor_tree.no_fetch(monkeypatch)
    # The H-mode shot is in no ELM grid; its filterscopes are in the corpus.
    event = p.label_tables / ELM
    labels.save(
        event,
        SHOTS[HMODE],
        labels.normalise(WINDOW, [(*EVENT_MS, PRESENT)]),
        source=None,
    )
    spec = frames.SPECS["elm_frames"]
    assert SHOTS[HMODE] not in targets.target_shots(p, spec)
    found = frames_shots.read_targets(p, spec)
    merged = frames_shots.merge_targets(found, labels.read_saved(event), spec.bin_ms)
    starts, states = merged[SHOTS[HMODE]]
    alone = targets.label_bins(labels.read_saved(event)[SHOTS[HMODE]], spec.bin_ms)
    assert (starts.tolist(), states.tolist()) == (alone[0].tolist(), alone[1].tolist())
    shots = frames_shots.make(p, "elm_frames")
    meta = json.loads(frames.shots_meta_file(p, "elm_frames").read_text())
    assert meta["owner"] == {"saved": 2, "overriding": 1, "added": 1}
    row = shots.set_index("shot").loc[SHOTS[HMODE]]
    assert (row.split, row.positive, row.roster, row.owner) == ("train", 1, 0, 1)
    # Counted with the labelled shots, not with the original's.
    assert (meta["labelled_shots"], meta["positive_shots"]) == (6, 5)
    assert meta["legacy_labelled_shots"] == 5
    assert meta["present_s"] == 4.0


def test_without_saves_there_is_no_owner_shot(tmp_path, monkeypatch):
    p = frames_tree.build(tmp_path)
    editor_tree.no_fetch(monkeypatch)
    spec = frames.SPECS["ntm_frames"]
    frames_shots.make(p, "ntm_frames")
    assert frames.owner_file(p, "ntm_frames").is_file()
    # F16: the saves gone, a split made again leaves no stale frozen copy.
    labels.labels_path(p.label_tables / spec.event).unlink()
    shots = frames_shots.make(p, "ntm_frames", force=True)
    meta = json.loads(frames.shots_meta_file(p, "ntm_frames").read_text())
    assert (meta["owner_snapshot"]["path"], meta["owner_snapshot"]["sha256"]) == (
        None,
        None,
    )
    assert meta["owner"] == {"saved": 0, "overriding": 0, "added": 0}
    assert not frames.owner_file(p, "ntm_frames").exists()
    assert _rows(shots) == [(SHOTS[spec.store_event], "train", 1, 1, 0)]


def test_the_meta_pins_the_original_s_one_file(tree, made):
    # F16: the sawteeth's table, one file, is pinned by its sha256; a grid
    # target is a file per shot, and pins nothing.
    _, meta = made["sawtooth_frames"]
    table = frames_tree.sawtooth_table(tree)
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    assert meta["original"] == {"path": str(table), "sha256": sha}
    for method in ("elm_frames", "hmode_frames", "ntm_frames"):
        pin = made[method][1]["original"]
        assert (pin["path"], pin["sha256"]) == (None, None)
        assert pin["grids"] and "no single file" in pin["note"]


def test_the_meta_counts_the_owner_s_flips(made):
    # F14: what the owner's saves changed in the owner's shots' targets, over
    # their windows. The tree's saves say what the originals say, but for
    # H-mode's 600 ms bin (H and L, uncertain; the owner says H) and the
    # sawteeth's 1000-1050 ms (uncertain; the owner says present) and
    # 1850-1900 ms (not observable; the owner says absent).
    changed = {
        "elm_frames": {},
        "hmode_frames": {"uncertain_to_present": 1},
        "ntm_frames": {},
        "sawtooth_frames": {"uncertain_to_present": 5, "unknown_to_absent": 5},
    }
    for method, want in changed.items():
        shots, meta = made[method]
        found = meta["owner_flips"]
        assert set(found) == {*SPLITS, "all"}
        (which,) = shots.split[shots.owner == 1]
        assert found[which] == found["all"] and found["all"]["shots"] == 1
        spec = frames.SPECS[method]
        window = meta["windows"][str(SHOTS[spec.store_event])][:2]
        k0, k1 = targets.bin_range(window, spec.bin_ms)
        assert found["all"]["bins"] == found["all"]["owned"] == k1 - k0
        flips = {k: v for k, v in found["all"].items() if k in targets.FLIP_KEYS}
        assert {k: v for k, v in flips.items() if v} == want


def test_flips_count_the_owner_s_changes_by_class():
    U, A, P, C = targets.UNKNOWN, targets.ABSENT, targets.PRESENT_T, 2
    original = [A, A, P, P, P, U, U, C, A, P]
    owner = [A, P, A, P, C, P, A, P, U, U]
    found = targets.flips(original, owner)
    assert (found["bins"], found["owned"]) == (10, 8)
    nonzero = {k: v for k, v in found.items() if k in targets.FLIP_KEYS and v}
    assert nonzero == {
        "absent_to_present": 1,
        "present_to_absent": 1,
        "present_to_uncertain": 1,
        "unknown_to_present": 1,
        "unknown_to_absent": 1,
        "uncertain_to_present": 1,
    }
    mask = [True] * 5 + [False] * 5
    masked = targets.flips(original, owner, mask)
    assert (masked["bins"], masked["owned"], masked["unknown_to_present"]) == (5, 5, 0)
    total = targets.flip_total([found, masked])
    assert (total["shots"], total["bins"], total["absent_to_present"]) == (2, 15, 2)
    assert targets.flip_total([]) == {"shots": 0, "bins": 0, "owned": 0} | {
        k: 0 for k in targets.FLIP_KEYS
    }


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
        assert path == tree.root / "frames/shots/v2" / f"{method}.csv"
        assert path.read_text().splitlines()[0] == "shot,split,positive,roster,owner"
        read = pd.read_csv(path)
        pd.testing.assert_frame_equal(read, shots.reset_index(drop=True))
        ints = ("shot", "positive", "roster", "owner")
        assert all(read[c].dtype.kind == "i" for c in ints)
        assert set(read.positive) <= {0, 1} and set(read.roster) <= {0, 1}
        assert set(read.owner) <= {0, 1}
        assert read.shot.is_monotonic_increasing
        assert META_KEYS <= set(meta)
        assert {"path", "sha256"} <= set(meta["owner_snapshot"])
        assert (meta["method"], meta["seed"]) == (method, frames.SEED)
        assert meta["version"] == frames.VERSION
        assert meta["counts"] == {s: int((read.split == s).sum()) for s in SPLITS}
    shots, meta = made["elm_frames"]
    assert _rows(shots) == [
        (IP_SHOT, "train", 1, 0, 0),
        (LABELS_SHOT, "train", 0, 0, 0),
        (SHOTS[ELM], "train", 1, 1, 1),
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
        assert meta["legacy_labelled_shots"] == labelled
        assert meta["present_s"] == PRESENT_S[method]


def test_force_is_needed_to_overwrite(tmp_path, monkeypatch, capsys):
    p = frames_tree.build(tmp_path)
    editor_tree.use_env(monkeypatch, p)
    tried = editor_tree.no_fetch(monkeypatch)
    assert frames_shots.main(["--method", "elm_frames"]) == 0
    (line,) = capsys.readouterr().out.splitlines()
    out = json.loads(line)
    assert out["method"] == "elm_frames"
    assert out["counts"] == {"train": 3, "val": 0, "test": 0}
    assert (out["version"], out["owner"]["saved"]) == (frames.VERSION, 1)
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
