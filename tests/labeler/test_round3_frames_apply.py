"""Round three, Task 2.11: the frame models' application and galleries."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import torch

from labeler import frames
from labeler.ae.xpower.extend import SUMMARY_COLUMNS
from labeler.events import spans, suggestions
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT
from labeler.events.review import build as review_build
from labeler.frames import apply, evaluate, gallery, prepare
from labeler.frames import shots as frames_shots
from labeler.frames.features import features

from . import editor_tree, frames_tree
from .frames_tree import ELM, HMODE, LMODE, POPULATION_ONLY, SHOTS, WINDOW
from .test_round3_frames_evaluate import TEST, _resplit, _save_model, _truth_onsets


def _files(folder) -> list:
    return sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())


@pytest.fixture
def tree(tmp_path, monkeypatch):
    paths = frames_tree.build(tmp_path / "tree")
    for event in frames_tree.STORE_ROWS:
        monkeypatch.setitem(review_build.BUILDERS, event, frames_tree.builder)
    monkeypatch.setattr(spans, "elm_onsets", _truth_onsets(paths), raising=False)
    editor_tree.use_env(monkeypatch, paths)
    for name in ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_COUNT"):
        monkeypatch.delenv(name, raising=False)
    return paths


def _scored(paths, method, shots, test) -> None:
    """The method's split, features, a random model, and its test scored."""
    frames_shots.make(paths, method)
    prepare.build_stores(paths, method, shots)
    assert prepare.prepare(paths, method, shots)["written"] == sorted(shots)
    _resplit(paths, method, dict.fromkeys(test, "test"))
    _save_model(paths, method)
    evaluate.evaluate(paths, method)


@pytest.fixture
def elm(tree):
    _scored(tree, "elm_frames", [SHOTS[ELM], *TEST], TEST)
    return tree


@pytest.fixture
def hmode(tree):
    _scored(tree, "hmode_frames", [SHOTS[HMODE]], [SHOTS[HMODE]])
    return tree


def _tiles(rows, shot) -> list:
    return [r for r in rows if r[0] == shot]


def test_apply_labels_the_roster_and_an_in_memory_population_shot(elm, monkeypatch):
    built = []

    def counted(event, shot, paths=None):
        built.append(shot)
        return frames_tree.builder(event, shot, paths)

    monkeypatch.setitem(review_build.BUILDERS, ELM, counted)
    before = _files(elm.root)
    # SHOTS[HMODE] is in the population, outside the ELM roster, with filterscopes.
    rows = apply.apply(elm, "elm_frames", [SHOTS[ELM], SHOTS[HMODE], POPULATION_ONLY])
    assert _files(elm.root) == before  # no store, no file at all
    assert built == [SHOTS[HMODE]]  # the roster shot is read from its store
    assert {r[0] for r in rows} == {SHOTS[ELM], SHOTS[HMODE]}
    windows = apply.windows(elm)
    for shot in (SHOTS[ELM], SHOTS[HMODE]):
        tiles = _tiles(rows, shot)
        lo, hi = windows[shot]
        assert tiles[0][2] == -(-lo // 10) * 10 and tiles[-1][3] == hi // 10 * 10
        assert all(a[3] == b[2] for a, b in pairwise(tiles))
        assert {r[1] for r in tiles} <= {PRESENT, ABSENT, NOT_OBSERVABLE}
        for r in tiles:
            assert (r[4] == "") == (r[1] == NOT_OBSERVABLE)
    # The shot without the groups is skipped, with why.
    [(shot, kind, why)] = apply.label_all(elm, "elm_frames", [POPULATION_ONLY])
    assert (shot, kind, why) == (POPULATION_ONLY, "skipped", "no filterscopes on disk")
    # Two shards cover the shots once.
    halves = [
        apply.apply(elm, "elm_frames", [SHOTS[ELM], SHOTS[HMODE]], index=i, count=2)
        for i in range(2)
    ]
    assert sorted(halves[0] + halves[1]) == sorted(rows)


def test_frame_probs_leave_a_partial_bin_at_either_window_edge_unobserved():
    logits = [0.0, 0.0, 4.0, -4.0, -4.0, -4.0, -4.0, 9.0, 9.0, 9.0, 9.0]

    class Fixed:
        def eval(self):
            return self

        def __call__(self, x):
            return torch.tensor([logits])

    spec = frames.SPECS["hmode_frames"]  # 50 ms bins of 5 frames, mean
    x = np.zeros((5, len(logits)), np.float32)
    observed = np.ones(len(logits), bool)
    # F7: frames 3..13, a window starting and ending mid-bin. Bin 0 holds
    # frames 3-4 and bin 2 frames 10-13, too few for a bin: not observed, so
    # they are not trained, scored or labelled on part of a bin's frames. Bin 1,
    # frames 5-9, is whole: its logits pooled as in training.
    prob = apply.frame_probs(Fixed(), spec, x, observed, 3)
    assert np.all(np.isnan(prob[:2])) and np.all(np.isnan(prob[7:]))
    assert np.allclose(prob[2:7], 1 / (1 + np.exp(4.0 * 3 / 5)))
    # A whole bin with a frame not observed is not observed either.
    observed[6] = False
    assert np.all(np.isnan(apply.frame_probs(Fixed(), spec, x, observed, 3)))
    # Starting on a bin's edge, the first bin is whole.
    observed[6] = True
    prob = apply.frame_probs(Fixed(), spec, x, observed, 0)
    assert np.allclose(prob[:5], 1 / (1 + np.exp(4.0 / 5)))
    assert np.allclose(prob[5:10], 1 / (1 + np.exp(-19.0 / 5)))
    assert np.isnan(prob[10])


def _run_all(method, counts=(("--roster", 1), ("--population", 2))) -> None:
    for which, count in counts:
        for index in range(count):
            argv = ["--method", method, which, "--index", str(index)]
            assert apply.main([*argv, "--count", str(count)]) == 0


def test_merge_writes_the_table_meta_and_summary(elm, capsys):
    # Before the shards, and a pilot's, the merge is refused.
    assert apply.main(["--method", "elm_frames", "--population", "--limit", "1"]) == 0
    spec = frames.SPECS["elm_frames"]
    assert list(apply.shards_dir(elm, spec, pilot=True).glob("*.json"))
    with pytest.raises(SystemExit):
        apply.main(["--method", "elm_frames", "--merge"])
    _run_all("elm_frames")
    capsys.readouterr()
    assert apply.main(["--method", "elm_frames", "--merge"]) == 0
    said = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    table = suggestions.table_path(elm, ELM, "elm_frames", frames.VERSION)
    assert said["tables"] == {"elm_frames": str(table)}
    meta = json.loads(table.with_suffix(".meta.json").read_text())
    model = frames.model_dir(elm, "elm_frames") / "model.pt"
    scored = json.loads((model.parent / "evaluation.json").read_text())
    assert meta["tier"] == "suggestions" and meta["bar"] == scored["bar"]
    assert table.parent == elm.root / "suggestions" / "elm_frames" / "v2"
    # F6: the model's flag goes beside its bar.
    assert meta["effectively_always"] == scored["effectively_always"]
    assert isinstance(meta["effectively_always"], bool)
    assert meta["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert meta["sets"]["roster"]["done"] == 1
    assert meta["sets"]["population"]["done"] == 1
    assert meta["sets"]["population"]["skipped"] == {"no filterscopes on disk": 4}
    assert meta["sets"]["population"]["failed"] == 0
    assert meta["failed_file"] == str(table.parent / "failed.jsonl")
    assert (table.parent / "failed.jsonl").read_text() == ""
    summary_path = frames.summary_file(elm, "elm_frames")
    assert summary_path.parent == table.parent
    summary = pd.read_csv(summary_path, keep_default_na=False)
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert list(summary.shot) == [SHOTS[ELM], SHOTS[HMODE]]  # one row a shot
    assert set(summary.year.astype(str)) == {""}  # the tree's population has none
    assert list(summary.window_start_ms) == [WINDOW[0], WINDOW[0]]
    rows = pd.read_csv(table)
    for r in summary.itertuples(index=False):
        mine = rows[rows.shot == r.shot]
        present = mine[mine.category == PRESENT]
        assert r.present_frames == ((present.t_end - present.t_start) / 10).sum()
        assert r.present_runs == len(present)
        assert r.frames == ((mine.t_end - mine.t_start) / 10).sum()
    # A shard gone, the merge is refused.
    next(apply.shards_dir(elm, spec).glob("population-1-of-2.json")).unlink()
    with pytest.raises(FileNotFoundError):
        apply.merge(elm, "elm_frames")


def test_lmode_frames_gets_its_own_table(hmode, capsys):
    _run_all("hmode_frames")
    # F15: say the H model is effectively always, calling L on no bin.
    scored = frames.model_dir(hmode, "hmode_frames") / "evaluation.json"
    record = json.loads(scored.read_text())
    record["effectively_always"] = True
    record["scores"]["lmode_frames"]["cells"] = [0, 0, 3, 7]
    scored.write_text(json.dumps(record))
    assert apply.main(["--method", "hmode_frames", "--merge"]) == 0
    h = suggestions.table_path(hmode, HMODE, "hmode_frames", frames.VERSION)
    l_ = suggestions.table_path(hmode, LMODE, "lmode_frames", frames.VERSION)
    assert l_.parent == hmode.root / "suggestions" / "lmode_frames" / "v2"
    meta = json.loads(l_.with_suffix(".meta.json").read_text())
    assert meta["method"] == "lmode_frames" and meta["event"] == LMODE
    assert meta["derived_from"] == "hmode_frames" and meta["tier"] == "suggestions"
    h_meta = json.loads(h.with_suffix(".meta.json").read_text())
    assert h_meta["effectively_always"] is True and "effectively_never" not in h_meta
    # L's flags are its own: always H is never L, and L is not always.
    assert meta["effectively_never"] is True
    assert meta["effectively_always"] is False
    assert meta["effectively_from"]["l_agreement"] == 0.0
    assert meta["failed_file"] == str(l_.parent / "failed.jsonl")
    assert (l_.parent / "failed.jsonl").is_file()
    h_rows, l_rows = pd.read_csv(h), pd.read_csv(l_)
    swap = {PRESENT: ABSENT, ABSENT: PRESENT, NOT_OBSERVABLE: NOT_OBSERVABLE}
    assert list(l_rows.category) == [swap[c] for c in h_rows.category]
    assert l_rows[["shot", "t_start", "t_end"]].equals(
        h_rows[["shot", "t_start", "t_end"]]
    )
    h_sum = pd.read_csv(frames.summary_file(hmode, "hmode_frames"))
    l_sum = pd.read_csv(frames.summary_file(hmode, "lmode_frames"))
    assert tuple(l_sum.columns) == SUMMARY_COLUMNS
    assert list(l_sum.shot) == list(h_sum.shot) == sorted([SHOTS[HMODE], SHOTS[ELM]])
    total = h_sum.present_frames + l_sum.present_frames + l_sum.not_observable_frames
    assert list(total) == list(l_sum.frames)


def test_lmode_s_flags_are_the_h_model_s_calls_read_as_l():
    def flags(cells, always):
        record = {"scores": {"lmode_frames": {"cells": cells}}}
        return apply.lmode_flags(
            record | {"effectively_always": always}, "lmode_frames"
        )

    # L on 995 of the 1000 scored bins: effectively always L, and not never.
    found = flags([990, 5, 5, 0], False)
    assert (found["effectively_always"], found["effectively_never"]) == (True, False)
    assert found["effectively_from"]["l_agreement"] == pytest.approx(0.995)
    # H's flag is L's "never", whatever L's own agreement.
    found = flags([400, 100, 100, 400], True)
    assert (found["effectively_always"], found["effectively_never"]) == (False, True)
    # No scored bin: no agreement, so not always.
    found = flags([0, 0, 0, 0], False)
    assert found["effectively_always"] is False
    assert found["effectively_from"]["l_agreement"] is None


def test_the_gallery_draws_one_picture_a_shot(elm, capsys):
    assert gallery.main(["--method", "elm_frames"]) == 0
    folder = frames.gallery_dir(elm, frames.SPECS["elm_frames"])
    assert folder == elm.root / "gallery" / ELM / "elm_frames-v2"
    test = sorted(int(p.stem) for p in (folder / "test").glob("*.jpg"))
    roster = sorted(int(p.stem) for p in (folder / "roster").glob("*.jpg"))
    assert test == TEST and roster == [SHOTS[ELM]]
    for path in folder.rglob("*.jpg"):
        assert path.read_bytes()[:2] == b"\xff\xd8"  # JPEG
    said = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (said["test"], said["roster"]) == (2, 1)
    for path in folder.rglob("*.jpg"):
        path.unlink()
    assert gallery.main(["--method", "elm_frames", "--limit", "1"]) == 0
    assert len(list(folder.rglob("*.jpg"))) == 2
    # A roster shot in the split whose features were dropped says why it has no
    # target; one outside the split says that.
    split = prepare.split_shots(elm, "elm_frames")
    gone = prepare.dropped(elm, "elm_frames")
    target, why = gallery._target(elm, "elm_frames", SHOTS[ELM], split, gone)
    assert target is not None and why == ""
    features = frames.features_dir(elm, "elm_frames")
    (features / f"{SHOTS[ELM]}.npz").unlink()
    prepare._drop(features, SHOTS[ELM], "stale store: before d9fb57d")
    gone = prepare.dropped(elm, "elm_frames")
    assert gallery._target(elm, "elm_frames", SHOTS[ELM], split, gone) == (
        None,
        "features dropped: stale store: before d9fb57d",
    )
    assert gallery._target(elm, "elm_frames", 1, split, gone) == (
        None,
        "not in the split",
    )
    assert gallery.main(["--method", "elm_frames"]) == 0
    assert (folder / "roster" / f"{SHOTS[ELM]}.jpg").is_file()


def test_a_shot_that_raises_is_failed_and_counted_not_fatal(elm, monkeypatch):
    def broken(store, spec, window):
        if isinstance(store, tuple):  # a population shot's rows, in memory
            raise IndexError("a row too short")
        return features(store, spec, window)

    monkeypatch.setattr(apply, "features", broken)
    _run_all("elm_frames")  # the shard goes on past the shot that raises
    shards = apply.shards_dir(elm, frames.SPECS["elm_frames"])
    manifests = [json.loads(p.read_text()) for p in shards.glob("population-*.json")]
    assert sorted(s for m in manifests for s in m["failed"]) == [SHOTS[HMODE]]
    said = apply.merge(elm, "elm_frames")
    assert said["failed"] == 1
    table = suggestions.table_path(elm, ELM, "elm_frames", frames.VERSION)
    meta = json.loads(table.with_suffix(".meta.json").read_text())
    assert meta["sets"]["population"]["failed"] == 1
    assert meta["sets"]["population"]["done"] == 0
    [row] = [
        json.loads(x) for x in (table.parent / "failed.jsonl").read_text().splitlines()
    ]
    assert row == {"shot": SHOTS[HMODE], "error": "IndexError: a row too short"}
    summary = pd.read_csv(frames.summary_file(elm, "elm_frames"))
    assert list(summary.shot) == [SHOTS[ELM]]
