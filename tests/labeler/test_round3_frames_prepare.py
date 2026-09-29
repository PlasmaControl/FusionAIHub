"""Round three, Task 2.8: the legacy stores outside `spectrograms/`, and the
features each shot's model input is read from."""

from __future__ import annotations

import dataclasses
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler import frames
from labeler.events.review import build as review_build
from labeler.events.review import rows
from labeler.frames import prepare
from labeler.frames import shots as frames_shots
from labeler.frames.features import STALE_TITLE, StaleStore, features
from labeler.frames.targets import ABSENT, PRESENT_T, UNKNOWN

from . import editor_tree, frames_tree
from .frames_tree import ELM, EVENT_MS, IP_SHOT, LABELS_SHOT, NTM, SHOTS, WINDOW

KEYS = {"x", "observed", "bins", "states"}
#: A legacy tearing-mode shot in no roster, for a store under frames/stores/.
LEGACY_NTM = 170001


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The tree, each event's builder replaced by the tree's rows."""
    paths = frames_tree.build(tmp_path / "tree")
    for event in frames_tree.STORE_ROWS:
        monkeypatch.setitem(review_build.BUILDERS, event, frames_tree.builder)
    return paths


def _files(folder) -> list:
    return sorted(p.relative_to(folder) for p in folder.rglob("*"))


def test_build_with_out_writes_there_and_nothing_under_spectrograms(tree, tmp_path):
    before = _files(tree.spectrograms)
    out = tmp_path / "elsewhere"
    path = review_build.build(ELM, IP_SHOT, tree, out=out)
    assert path == out / f"{IP_SHOT}.h5" and path.is_file()
    assert _files(tree.spectrograms) == before
    with h5py.File(path, "r") as f:
        assert f.attrs["event"] == ELM and f.attrs["shot"] == IP_SHOT
    # Kept unless forced; never into spectrograms/, however it is named.
    stamp = path.stat().st_mtime_ns
    assert review_build.build(ELM, IP_SHOT, tree, out=out) == path
    assert path.stat().st_mtime_ns == stamp
    for inside in (tree.spectrograms, tree.spectrograms / ELM):
        with pytest.raises(ValueError, match="spectrograms"):
            review_build.build(ELM, IP_SHOT, tree, out=inside)
    assert _files(tree.spectrograms) == before


def test_build_stores_builds_the_legacy_shots_under_frames_stores(tree):
    before = _files(tree.spectrograms)
    spec = frames.SPECS["elm_frames"]
    shots = [SHOTS[ELM], IP_SHOT, LABELS_SHOT]
    record = prepare.build_stores(tree, "elm_frames", shots)
    assert record["roster"] == [SHOTS[ELM]]  # the review's own store, untouched
    assert record["built"] == sorted([IP_SHOT, LABELS_SHOT])
    assert record["failed"] == {} and record["rebuilt"] == {}
    for shot in (IP_SHOT, LABELS_SHOT):
        path = frames.store_path(tree, spec, shot)
        assert path == tree.root / "frames" / "stores" / ELM / f"{shot}.h5"
        assert path.is_file()
    assert _files(tree.spectrograms) == before
    again = prepare.build_stores(tree, "elm_frames", shots)
    assert again["kept"] == sorted([IP_SHOT, LABELS_SHOT]) and again["built"] == []
    forced = prepare.build_stores(tree, "elm_frames", shots, force=True)
    assert forced["rebuilt"] == {str(IP_SHOT): "forced", str(LABELS_SHOT): "forced"}


def test_prepare_writes_one_npz_a_shot_with_the_keys(tree):
    frames_shots.make(tree, "elm_frames")
    shots = list(pd.read_csv(frames.shots_file(tree, "elm_frames")).shot)
    assert sorted(shots) == sorted([SHOTS[ELM], IP_SHOT, LABELS_SHOT])
    prepare.build_stores(tree, "elm_frames", shots)
    record = prepare.prepare(tree, "elm_frames", shots)
    assert record["written"] == sorted(shots) and record["dropped"] == {}
    folder = frames.features_dir(tree, "elm_frames")
    assert sorted(int(p.stem) for p in folder.glob("*.npz")) == sorted(shots)
    spec = frames.SPECS["elm_frames"]
    meta = json.loads(frames.shots_meta_file(tree, "elm_frames").read_text())
    for shot in shots:
        with np.load(folder / f"{shot}.npz") as z:
            assert KEYS <= set(z.files)
            x, observed, bins, states = (
                z[k] for k in ("x", "observed", "bins", "states")
            )
            first, window = int(z["first"]), tuple(z["window"])
        lo, hi = meta["windows"][str(shot)][:2]
        assert window == (lo, hi)
        # The whole 50 ms bins inside the window, five 10 ms frames each.
        assert bins[0] >= lo and bins[-1] + spec.bin_ms <= hi
        assert bins[0] - spec.bin_ms < lo and bins[-1] + 2 * spec.bin_ms > hi
        assert np.all(np.diff(bins) == spec.bin_ms)
        assert first * 10 == bins[0]
        assert observed.shape == (5 * len(bins),)
        assert x.shape == (2, 5 * len(observed)) and x.dtype == np.float16
        assert np.all((x >= 0) & (x <= 1))
        assert states.dtype == np.int8 and states.shape == bins.shape
    # The owner shot's states are the owner's save, present over EVENT_MS.
    with np.load(folder / f"{SHOTS[ELM]}.npz") as z:
        bins, states = z["bins"], z["states"]
    during = (bins >= EVENT_MS[0]) & (bins < EVENT_MS[1])
    assert np.all(states[during] == PRESENT_T)
    assert np.all(states[~during & (bins >= WINDOW[0])] == ABSENT)
    # LABELS_SHOT's grid is absent in every cell; its store ends at 2100 ms.
    with np.load(folder / f"{LABELS_SHOT}.npz") as z:
        assert set(np.unique(z["states"])) == {ABSENT}
        assert z["observed"].any() and not z["observed"].all()


def test_a_shot_with_no_observed_frame_is_dropped_and_recorded(tree):
    frames_shots.make(tree, "elm_frames")
    prepare.build_stores(tree, "elm_frames", [IP_SHOT])
    spec = frames.SPECS["elm_frames"]
    folder = frames.features_dir(tree, "elm_frames")
    assert prepare.prepare(tree, "elm_frames", [IP_SHOT])["written"] == [IP_SHOT]
    assert (folder / f"{IP_SHOT}.npz").is_file()
    with h5py.File(frames.store_path(tree, spec, IP_SHOT), "r+") as f:
        for name in f["rows"]:
            for level in f["rows"][name].values():
                if level.dtype.kind == "f":  # the traces: ECE, as sawtooth 187154's
                    level[...] = np.nan
    record = prepare.prepare(tree, "elm_frames", [IP_SHOT, LABELS_SHOT])
    assert record["written"] == []
    assert record["dropped"][str(IP_SHOT)] == "no observed frame"
    assert record["dropped"][str(LABELS_SHOT)] == "no store"
    assert not (folder / f"{IP_SHOT}.npz").exists()  # never a stale npz
    dropped = json.loads((folder / f"{IP_SHOT}.dropped.json").read_text())
    assert dropped["reason"] == "no observed frame"
    assert prepare.dropped(tree, "elm_frames") == {
        IP_SHOT: "no observed frame",
        LABELS_SHOT: "no store",
    }


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_the_shards_cover_each_shot_once(count):
    shots = [190005, 160003, 190001, 160001, 170002, 160003]
    got = [s for i in range(count) for s in prepare.shard(shots, i, count)]
    assert sorted(got) == sorted(set(shots))
    assert len(got) == len(set(got))
    with pytest.raises(ValueError):
        prepare.shard(shots, count, count)


def test_the_command_takes_its_shard_from_the_array(tree, monkeypatch, capsys):
    frames_shots.make(tree, "elm_frames")
    editor_tree.use_env(monkeypatch, tree)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "2")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)  # one process
    all_shots = sorted([SHOTS[ELM], IP_SHOT, LABELS_SHOT])
    assert prepare.main(["--method", "elm_frames", "--stores"]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["shots"] == 1 and line["built"] == 1
    records = tree.root / "frames" / "stores" / ELM / "records"
    record = json.loads((records / "elm_frames-1-of-2.json").read_text())
    assert (record["index"], record["count"]) == (1, 2)
    assert record["shots"] == all_shots[1::2] == record["built"]
    assert prepare.main(["--method", "elm_frames", "--index", "0", "--count", "1"]) == 0
    capsys.readouterr()
    folder = frames.features_dir(tree, "elm_frames")
    # Shard 0 of 2 built no store, so its legacy shot is dropped, not guessed.
    assert prepare.dropped(tree, "elm_frames") == {all_shots[0]: "no store"}
    assert (folder / "records" / "0-of-1.json").is_file()


def test_a_stale_tearing_mode_store_is_rebuilt_not_read(tree):
    spec = frames.SPECS["ntm_frames"]
    path = frames.store_path(tree, spec, LEGACY_NTM)
    assert path.parent == tree.root / "frames" / "stores" / NTM
    grid, built, _ = frames_tree.builder(NTM, LEGACY_NTM)
    old = [built[0], dataclasses.replace(built[1], title=STALE_TITLE, modes=None)]
    rows.write(path, grid, old, event=NTM, shot=LEGACY_NTM)
    with pytest.raises(StaleStore):
        features(path, spec, WINDOW)
    record = prepare.build_stores(tree, "ntm_frames", [LEGACY_NTM])
    assert record["rebuilt"] == {str(LEGACY_NTM): "stale"}
    _, observed = features(path, spec, WINDOW)
    assert observed.any()
    assert prepare.build_stores(tree, "ntm_frames", [LEGACY_NTM])["kept"] == [
        LEGACY_NTM
    ]
    # A stale roster store is the review's, never rebuilt here: it is dropped.
    frames_shots.make(tree, "ntm_frames")
    roster = frames.store_path(tree, spec, SHOTS[NTM])
    rows.write(roster, grid, old, event=NTM, shot=SHOTS[NTM])
    record = prepare.prepare(tree, "ntm_frames", [SHOTS[NTM]])
    assert record["dropped"][str(SHOTS[NTM])].startswith("stale store")
    assert prepare.build_stores(tree, "ntm_frames", [SHOTS[NTM]])["roster"] == [
        SHOTS[NTM]
    ]
    with pytest.raises(StaleStore):
        features(roster, spec, WINDOW)


def test_unknown_bins_stay_unknown(tree):
    frames_shots.make(tree, "hmode_frames")
    shot = SHOTS[frames.SPECS["hmode_frames"].store_event]
    record = prepare.prepare(tree, "hmode_frames", [shot])
    assert record["written"] == [shot]
    with np.load(frames.features_dir(tree, "hmode_frames") / f"{shot}.npz") as z:
        assert z["x"].shape[0] == 5 and set(np.unique(z["states"])) <= {
            UNKNOWN,
            ABSENT,
            PRESENT_T,
            2,
        }
