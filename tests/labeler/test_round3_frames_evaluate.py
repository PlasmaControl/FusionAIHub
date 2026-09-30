"""Round three, Task 2.10: the one test, the baselines and the bars."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler import frames
from labeler.events import spans
from labeler.events.catalog.population import POOL_COLUMNS
from labeler.events.catalog.states import PRESENT, UNCERTAIN
from labeler.events.review import build as review_build
from labeler.events.review import labels
from labeler.frames import evaluate, prepare, targets, train
from labeler.frames import shots as frames_shots
from labeler.frames.model import RowsCNN
from labeler.frames.targets import ABSENT, PRESENT_T, UNCERTAIN_T, UNKNOWN
from labeler.scoring import stats

from . import frames_tree
from .frames_tree import (
    ELM,
    EVENT_MS,
    HMODE,
    IP_SHOT,
    LABELS_SHOT,
    POPULATION_ONLY,
    SHOTS,
    WINDOW,
)

WIDTHS = {"elm_frames": 2, "hmode_frames": 5, "ntm_frames": 42, "sawtooth_frames": 41}
#: The ELM method's test shots here; its owner shot, SHOTS[ELM], is in train.
TEST = sorted([IP_SHOT, LABELS_SHOT])


def _npz(paths, method, shot):
    with np.load(frames.features_dir(paths, method) / f"{shot}.npz") as z:
        return {k: z[k] for k in z.files}


def _save_model(
    paths, method, out=None, threshold=0.5, train_shots=(), val_shots=(), model=None
):
    """A random model (or `model`) on the method's split, as `train.fit` saves
    one, trained (so its record says) on `train_shots` and stopped on
    `val_shots`."""
    spec = frames.SPECS[method]
    out = frames.model_dir(paths, method) if out is None else out
    subs = round(10 / spec.sub_ms)
    split = frames.shots_file(paths, method).read_bytes()
    record = {
        "method": method,
        "threshold": threshold,
        "threshold_rule": spec.threshold_rule,
        "channels": WIDTHS[method],
        "subs": subs,
        "width": 32,
        "split_sha256": hashlib.sha256(split).hexdigest(),
        "shots": {
            "train": [int(s) for s in train_shots],
            "val": [int(s) for s in val_shots],
        },
    }
    model = RowsCNN(WIDTHS[method], subs).eval() if model is None else model
    train.save(out, model, record)
    return out


def rechoose(out, threshold: float):
    """A `threshold.json` beside `out`'s model.pt at `threshold`, holding what
    `train.load` reads of the one `rethreshold` writes."""
    sha = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    path = out / train.THRESHOLD_FILE
    path.write_text(json.dumps({"threshold": threshold, "model": {"sha256": sha}}))
    return path


def _resplit(paths, method, splits: dict) -> None:
    path = frames.shots_file(paths, method)
    frame = pd.read_csv(path)
    frame["split"] = [splits.get(s, v) for s, v in zip(frame.shot, frame.split)]
    frame.to_csv(path, index=False)


def _truth_onsets(paths):
    """A stub `spans.elm_onsets`: an onset in each present bin of the shot's
    features, the clock's channel measured over the whole store."""

    def elm_onsets(shot, paths_, window=None):
        z = _npz(paths, "elm_frames", shot)
        present = z["bins"][z["states"] == PRESENT_T]
        spans_ = tuple((b, b + 50.0, PRESENT) for b in present)
        found = spans.Found(spans_, ((-100.0, 2100.0),))
        return SimpleNamespace(times_ms=tuple(present + 25.0), found=found)

    return elm_onsets


@pytest.fixture
def tree(tmp_path, monkeypatch):
    paths = frames_tree.build(tmp_path / "tree")
    for event in frames_tree.STORE_ROWS:
        monkeypatch.setitem(review_build.BUILDERS, event, frames_tree.builder)
    return paths


@pytest.fixture
def elm(tree, monkeypatch):
    """The ELM method with two test shots and its owner shot, a random model,
    and `elm_onsets` stubbed to the truth."""
    frames_shots.make(tree, "elm_frames")
    shots = [SHOTS[ELM], *TEST]
    prepare.build_stores(tree, "elm_frames", shots)
    assert prepare.prepare(tree, "elm_frames", shots)["written"] == sorted(shots)
    _resplit(tree, "elm_frames", dict.fromkeys(TEST, "test"))
    _save_model(tree, "elm_frames")
    monkeypatch.setattr(spans, "elm_onsets", _truth_onsets(tree), raising=False)
    return tree


def _perfect(model, spec, shot):
    prob = np.where(shot.states == PRESENT_T, 0.9, 0.1)
    return np.where(shot.observed, prob, np.nan)


def _always(model, spec, shot):
    return np.where(shot.observed, 0.9, np.nan)


def test_score_on_a_hand_made_case():
    prob = np.array([0.9, 0.2, 0.7, 0.4, 0.8, 0.9, 0.9, np.nan])
    states = np.array(
        [PRESENT_T, PRESENT_T, ABSENT, ABSENT, UNKNOWN, UNCERTAIN_T, PRESENT_T, ABSENT],
        np.int8,
    )
    observed = np.array([1, 1, 1, 1, 1, 1, 0, 1], bool)
    said = evaluate.score(prob, states, 0.5, observed)
    assert said["cells"] == [1, 1, 1, 1]
    assert said["precision"] == said["recall"] == said["f1"] == 0.5
    low = evaluate.score(prob, states, 0.1, observed)
    assert low["cells"] == [2, 2, 0, 0]
    assert low["precision"] == 0.5 and low["recall"] == 1.0
    assert low["f1"] == pytest.approx(2 / 3)
    none = evaluate.score(prob[4:6], states[4:6], 0.5)
    assert none["cells"] == [0, 0, 0, 0] and none["f1"] is None
    # L's cells are H's with present and absent swapped.
    assert evaluate.swapped([1, 2, 3, 4]).tolist() == [4, 3, 2, 1]


def test_the_bootstrap_is_reproducible_at_the_seed():
    cells = np.random.default_rng(0).integers(0, 20, size=(12, 4))
    first = evaluate.interval(cells, stats.f1)
    assert first == evaluate.interval(cells, stats.f1)
    assert first["seed"] == 20260923 and first["replicates"] == 2000
    assert first["level"] == 0.95 and first["low"] <= first["value"] <= first["high"]
    other = evaluate.interval(cells, stats.f1, seed=1)
    assert (other["low"], other["high"]) != (first["low"], first["high"])
    shifted = np.roll(cells, 1, axis=1)
    diff = evaluate.paired(cells, shifted, stats.f1)
    assert diff == evaluate.paired(cells, shifted, stats.f1)
    assert evaluate.interval(np.zeros((0, 4)), stats.f1) is None


def test_always_scores_as_expected(elm):
    spec = frames.SPECS["elm_frames"]
    found, left = evaluate.baselines(elm, spec, TEST)
    assert set(found) == {"always", "elm_onsets", "elm_clock"}
    assert left == {"always": {}, "elm_onsets": {}, "elm_clock": {}}
    for shot in TEST:
        z = _npz(elm, "elm_frames", shot)
        always = found["always"][shot]
        assert always.shape == z["bins"].shape and np.all(always == 1)
        observed = z["observed"].reshape(-1, 5).all(axis=1)
        said = evaluate.score(always, z["states"], 0.5, observed)
        scored = observed & np.isin(z["states"], (ABSENT, PRESENT_T))
        present = int((scored & (z["states"] == PRESENT_T)).sum())
        absent = int((scored & (z["states"] == ABSENT)).sum())
        assert said["cells"] == [present, absent, 0, 0]
        # The truth's onsets are a perfect baseline.
        onsets = evaluate.score(found["elm_onsets"][shot], z["states"], 0.5, observed)
        assert onsets["cells"][1:3] == [0, 0]
    record = evaluate.evaluate(elm, "elm_frames")
    always = record["scores"]["always"]
    assert always["recall"] == 1.0
    tp, fp, fn, tn = always["cells"]
    assert (fn, tn) == (0, 0) and always["precision"] == pytest.approx(tp / (tp + fp))
    assert record["intervals"]["always"]["recall"]["value"] == 1.0


def test_the_elm_bar_with_stubbed_onsets(elm, monkeypatch):
    monkeypatch.setattr(evaluate, "model_probs", _perfect)
    record = evaluate.evaluate(elm, "elm_frames")
    assert record["scores"]["elm_frames"]["f1"] == 1.0
    assert record["scores"]["elm_onsets"]["f1"] == 1.0
    assert record["paired"]["f1 - elm_onsets"]["low"] == 0.0
    assert record["paired"]["f1 - always"]["low"] > 0
    assert record["bar"] == {"E1": True, "E2": True, "E3": True, "all": True}
    assert record["effectively_always"] is False
    # A model that says present everywhere is `always`: no better than it, and
    # worse than the onsets.
    monkeypatch.setattr(evaluate, "model_probs", _always)
    written = frames.model_dir(elm, "elm_frames") / "evaluation.json"
    written.unlink()
    record = evaluate.evaluate(elm, "elm_frames")
    assert record["paired"]["f1 - always"]["value"] == 0.0
    assert record["paired"]["f1 - elm_onsets"]["low"] < -0.03
    assert not record["bar"]["E2"] and not record["bar"]["E3"]
    assert not record["bar"]["all"]
    # F6: its calls are always-present on every scored bin, and the paired
    # interval is zero: both tests hold.
    assert record["effectively_always"] is True
    assert record["always_like"]["held"] == {"agreement": True, "interval": True}
    assert record["always_like"]["agreement"] == 1.0
    md = (frames.model_dir(elm, "elm_frames") / "evaluation.md").read_text()
    assert evaluate.always_line(record["always_like"]) in md
    assert "Effectively always: yes, its calls equal always-present on 100.0%" in md
    # Without round-three-b's elm_onsets the ELM test is refused, not guessed.
    written.unlink()
    monkeypatch.delattr(spans, "elm_onsets")
    with pytest.raises(RuntimeError, match="round-three-b"):
        evaluate.evaluate(elm, "elm_frames")


def test_the_owner_s_shots_in_test_are_scored_with_the_rest(elm):
    record = evaluate.evaluate(elm, "elm_frames")
    assert record["test"]["shots"]["elm_frames"] == TEST
    owner = record["owner"]
    assert owner["test_shots"] == []  # the owner's shot is in train
    assert owner["saved"] == {"saved": 1, "overriding": 1, "added": 0}
    assert "ELMy time, not onsets" in owner["note"]
    assert "E1" in owner["note"] and "E2" in owner["note"]
    assert owner["snapshot"]["sha256"]
    assert set(record["scores"]) == {"elm_frames", "elm_onsets", "elm_clock", "always"}
    # The test's cells are the test shots' alone, by their merged target.
    cells = np.zeros(4)
    for shot in TEST:
        z = _npz(elm, "elm_frames", shot)
        observed = z["observed"].reshape(-1, 5).all(axis=1)
        cells += evaluate.score(z["states"] >= 0, z["states"], 0.5, observed)["cells"]
    assert sum(record["scores"]["always"]["cells"]) == cells.sum()
    # F2: no owner split; the owner's shot in test is one more test shot.
    (frames.model_dir(elm, "elm_frames") / "evaluation.json").unlink()
    _resplit(elm, "elm_frames", {SHOTS[ELM]: "test"})
    _save_model(elm, "elm_frames")
    record = evaluate.evaluate(elm, "elm_frames")
    assert record["test"]["shots"]["elm_frames"] == sorted([SHOTS[ELM], *TEST])
    assert record["owner"]["test_shots"] == [SHOTS[ELM]]
    assert "owner" not in record["split_years"]
    md = (frames.model_dir(elm, "elm_frames") / "evaluation.md").read_text()
    line = (
        "1 of the 3 scored test shots carry an owner label, 0 of them the "
        "owner's alone (added, not overriding). "
    )
    assert line + evaluate.ELM_OWNER_NOTE in md
    assert md.count("## The owner's saves") == 1


def test_the_owner_s_flips_and_added_shots_are_counted(tree, monkeypatch):
    # F14: the owner's ELM saves are spans of ELMy time over Hiro's onsets. On
    # SHOTS[ELM] they add 1400-1500 ms (0 to 1), leave 1300-1400 out (1 to 0)
    # and call 1000-1050 uncertain; the H-mode shot is in no ELM grid, so its
    # save is its target alone (added).
    event = tree.label_tables / ELM
    spans_ = [
        (600, 1000, PRESENT),
        (1000, 1050, UNCERTAIN),
        (1050, 1300, PRESENT),
        (1400, 1500, PRESENT),
    ]
    labels.save(event, SHOTS[ELM], labels.normalise(WINDOW, spans_), source=None)
    alone = labels.normalise(WINDOW, [(*EVENT_MS, PRESENT)])
    labels.save(event, SHOTS[HMODE], alone, source=None)
    frames_shots.make(tree, "elm_frames")
    meta = json.loads(frames.shots_meta_file(tree, "elm_frames").read_text())
    owned = [SHOTS[ELM], SHOTS[HMODE]]
    prepare.build_stores(tree, "elm_frames", [*owned, *TEST])
    written = prepare.prepare(tree, "elm_frames", [*owned, *TEST])["written"]
    assert written == sorted([*owned, *TEST])
    _resplit(tree, "elm_frames", dict.fromkeys([*owned, *TEST], "test"))
    _save_model(tree, "elm_frames")
    monkeypatch.setattr(spans, "elm_onsets", _truth_onsets(tree), raising=False)
    record = evaluate.evaluate(tree, "elm_frames")
    owner = record["owner"]
    assert owner["test_shots"] == sorted(owned)
    assert owner["added_test_shots"] == [SHOTS[HMODE]]
    # Every bin of both windows (100-1900 ms, 36 bins each) is observed and the
    # owner's; the added shot's 16 present and 20 absent bins had no original.
    changed = {
        "absent_to_present": 2,
        "present_to_absent": 2,
        "present_to_uncertain": 1,
        "unknown_to_present": 16,
        "unknown_to_absent": 20,
    }
    for scope in ("test", "all"):
        found = owner["flips"][scope]
        assert (found["shots"], found["bins"], found["owned"]) == (2, 72, 72)
        flips = {k: v for k, v in found.items() if k in targets.FLIP_KEYS and v}
        assert flips == changed
    # The shots meta's whole-split counts, made before the resplit, agree.
    assert meta["owner_flips"]["all"] == owner["flips"]["all"]
    md = (frames.model_dir(tree, "elm_frames") / "evaluation.md").read_text()
    assert "4 scored test shots carry an owner label, 1 of them the owner's" in md
    assert evaluate.flips_line(owner["flips"]) in md
    assert "0→1 2, 1→0 2, 0 or 1 to uncertain 1, and unknown to 0 or 1 36" in md
    assert evaluate.ELM_OWNER_NOTE in md


def test_effectively_always_by_either_test():
    def record(cells, high):
        paired = {} if high is None else {"f1 - always": {"high": high}}
        return {"scores": {"ntm_frames": {"cells": cells}}, "paired": paired}

    # 995 of 1000 scored bins called present: the agreement test holds.
    found = evaluate.always_like("ntm_frames", record([500, 495, 0, 5], 0.2))
    assert found["value"] and found["held"] == {"agreement": True, "interval": False}
    assert found["agreement"] == pytest.approx(0.995)
    line = evaluate.always_line(found)
    assert line.startswith("Effectively always: yes, its calls equal always-present")
    assert "interval" not in line
    # Half called present, but F1 no better than always's: the interval holds.
    found = evaluate.always_like("ntm_frames", record([400, 100, 100, 400], 0.005))
    assert found["value"] and found["held"] == {"agreement": False, "interval": True}
    line = evaluate.always_line(found)
    assert line == (
        "Effectively always: yes, the paired f1 - always interval's upper end is "
        "0.005 (effectively always below 0.01)."
    )
    # Neither, and an undefined interval holds nothing.
    found = evaluate.always_like("ntm_frames", record([400, 100, 100, 400], None))
    assert not found["value"] and found["high"] is None
    line = evaluate.always_line(found)
    assert line.startswith("Effectively always: no; its calls equal always-present")
    assert "upper end is undefined" in line
    # H-mode's is F1(H)'s.
    hmode = {"scores": {"hmode_frames": {"cells": [0, 0, 0, 0]}}, "paired": {}}
    found = evaluate.always_like("hmode_frames", hmode)
    assert found["key"] == "f1(H) - always" and found["agreement"] is None
    assert not found["value"]


@pytest.mark.parametrize("method", sorted(frames.SPECS))
def test_the_json_bar_keys_per_method(tree, monkeypatch, method):
    spec = frames.SPECS[method]
    shot = SHOTS[spec.store_event]
    frames_shots.make(tree, method)
    assert prepare.prepare(tree, method, [shot])["written"] == [shot]
    _resplit(tree, method, {shot: "test"})
    _save_model(tree, method)
    monkeypatch.setattr(spans, "elm_onsets", _truth_onsets(tree), raising=False)
    record = evaluate.evaluate(tree, method)
    assert set(record["bar"]) == set(spec.bar) | {"all"}
    assert all(isinstance(v, bool) for v in record["bar"].values())
    assert set(record["scores"]) >= {method, *spec.baselines}
    assert record["tier"] == "suggestions"
    assert isinstance(record["effectively_always"], bool)
    assert record["effectively_always"] == record["always_like"]["value"]
    assert record["model"]["threshold_rule"] == spec.threshold_rule
    md = (frames.model_dir(tree, method) / "evaluation.md").read_text()
    assert md.count("Effectively always: ") == 1
    assert "Campaign years" in md and "| legacy (outside the roster) |" in md
    assert "2013-2019" not in md and "2024-2025" not in md
    assert set(record["split_years"]) == {
        "train",
        "val",
        "test",
        "legacy",
        "roster",
    }
    if method == "hmode_frames":
        assert {"f1(H)", "f1(L)"} <= set(record["scores"][method])
        assert (
            record["scores"]["lmode_frames"]["cells"]
            == evaluate.swapped(record["scores"][method]["cells"]).tolist()
        )
        assert "f1(H) - always" in record["paired"]
        assert record["always_like"]["key"] == "f1(H) - always"
        assert "D44" in md and "Jalal Butt" in md and "cannot rank" in md
        # dalpha_lh needs CO2, which the tree's H-mode shot lacks: left out.
        assert str(shot) in record["test"]["baseline_left_out"]["dalpha_lh"]
        assert "dalpha_lh (reported, never gated)" in md
    if method == "sawtooth_frames":
        assert "distillation" in md


def test_reading_evaluation_json_twice_gives_the_same_bytes(elm):
    evaluate.evaluate(elm, "elm_frames")
    path = frames.model_dir(elm, "elm_frames") / "evaluation.json"
    first = path.read_bytes()
    assert path.read_bytes() == first
    # The test is scored once.
    with pytest.raises(FileExistsError):
        evaluate.evaluate(elm, "elm_frames")
    path.unlink()
    evaluate.evaluate(elm, "elm_frames")
    assert path.read_bytes() == first
    record = json.loads(first)
    assert "made_at" not in record
    assert set(record) >= {"bar", "scores", "intervals", "paired", "owner"}
    assert set(record) >= {"effectively_always", "always_like"}
    # A pilot's may be scored again, to the same bytes.
    pilot = _save_model(elm, "elm_frames", out=train.pilot_dir(elm, "elm_frames"))
    evaluate.evaluate(elm, "elm_frames", out=pilot)
    again = (pilot / "evaluation.json").read_bytes()
    evaluate.evaluate(elm, "elm_frames", out=pilot)
    assert (pilot / "evaluation.json").read_bytes() == again


def test_a_changed_split_or_a_trained_test_shot_is_refused(elm):
    blob_dir = frames.model_dir(elm, "elm_frames")
    _resplit(elm, "elm_frames", {IP_SHOT: "val"})
    with pytest.raises(ValueError, match="split changed"):
        evaluate.evaluate(elm, "elm_frames")
    _resplit(elm, "elm_frames", {IP_SHOT: "test"})
    _save_model(elm, "elm_frames", out=blob_dir, train_shots=[IP_SHOT])
    with pytest.raises(ValueError, match="trained on"):
        evaluate.evaluate(elm, "elm_frames")
    assert not (blob_dir / "evaluation.json").exists()


def test_the_labels_window_is_not_available_rather_than_zero(elm):
    record = evaluate.evaluate(elm, "elm_frames")
    # The tree's catalog Ip log has LABELS_SHOT, whose window is its grid's hull.
    assert record["labels_window"]["with_ip_window"] == 1
    none = evaluate.labels_window({"labels_window": {"shots": 4, "with_ip_window": 0}})
    assert none["outside_ip_window"] == "not available" and none["shots"] == 4
    assert "ip.jsonl" in none["reason"]


def _pool(paths, years: dict) -> None:
    """A `pool.csv` of `years` (shot -> year or None), every other field blank."""
    blank = dict.fromkeys(POOL_COLUMNS, "")
    rows = [
        blank | {"shot": s, "year": "" if y is None else y} for s, y in years.items()
    ]
    pd.DataFrame(rows, columns=list(POOL_COLUMNS)).to_csv(
        paths.catalog / "pool.csv", index=False
    )


def test_the_years_are_the_campaigns_not_fixed_text(elm):
    population = elm.catalog / "population.csv"
    frame = pd.read_csv(population)
    frame["year"] = [2024 if s == POPULATION_ONLY else 2023 for s in frame.shot]
    frame.to_csv(population, index=False)
    # The pool dates IP_SHOT itself and LABELS_SHOT by the calendar only.
    _pool(elm, {IP_SHOT: 2021, LABELS_SHOT: None, SHOTS[ELM]: 2023})
    years = evaluate.shot_years(elm)
    assert years.starts == {2021: IP_SHOT, 2023: SHOTS[ELM]}
    assert (years.of(IP_SHOT), years.of(LABELS_SHOT)) == (2021, 2021)
    assert years.of(POPULATION_ONLY) == 2024  # the population's own year
    assert years.of(IP_SHOT - 1) is None  # before the calendar
    assert evaluate.year_counts([IP_SHOT - 1, SHOTS[ELM], LABELS_SHOT], years) == {
        "2021": 1,
        "2023": 1,
        "unknown": 1,
    }
    record = evaluate.evaluate(elm, "elm_frames")
    assert record["test"]["years"] == {"2021": 2}
    assert record["population_years"] == {"2023": 5, "2024": 1}  # BLIND left out
    assert set(record["split_years"]["test"]) == {"2021"}
    assert record["split_years"]["roster"] == {"2023": 1}
    assert set(record["split_years"]["legacy"]) <= {"2021", "2023"}
    md = (frames.model_dir(elm, "elm_frames") / "evaluation.md").read_text()
    assert "The scored test shots: 2021 (2021: 2)." in md
    row = "| the non-blind population it is applied to | 2023-2024 | 2023: 5, 2024: 1 |"
    assert row in md
    assert "No scored test shot is from 2023, 2024, which hold 6 of the" in md


def test_the_md_counts_the_test_shots_without_features(elm):
    folder = frames.features_dir(elm, "elm_frames")
    (folder / f"{LABELS_SHOT}.npz").unlink()
    prepare._drop(folder, LABELS_SHOT, "no observed frame")
    record = evaluate.evaluate(elm, "elm_frames")
    assert record["test"]["left_out"] == {str(LABELS_SHOT): "no observed frame"}
    md = (frames.model_dir(elm, "elm_frames") / "evaluation.md").read_text()
    assert "- 1 test shots left out for missing features: no observed frame (1)" in md


def _h_below_half(model, spec, shot):
    """P(H) 0.4 on the H bins and 0.2 on the L bins: all L at 0.5, right at 0.3."""
    prob = np.where(shot.states == PRESENT_T, 0.4, 0.2)
    return np.where(shot.observed, prob, np.nan)


def test_rethreshold_scores_the_test_again_and_keeps_the_old_record(tree, monkeypatch):
    method, shot = "hmode_frames", SHOTS[HMODE]
    frames_shots.make(tree, method)
    assert prepare.prepare(tree, method, [shot])["written"] == [shot]
    _resplit(tree, method, {shot: "test"})
    out = _save_model(tree, method)
    monkeypatch.setattr(evaluate, "model_probs", _h_below_half)
    monkeypatch.setattr(evaluate, "git_dirty", lambda: False)  # committed code
    with pytest.raises(ValueError, match="threshold.json"):
        evaluate.evaluate(tree, method, rethreshold=True)
    first = evaluate.evaluate(tree, method)
    assert first["model"]["threshold_source"] == "training"
    assert first["model"]["threshold_sha256"] is None
    old = {s: (out / f"evaluation{s}").read_bytes() for s in (".json", ".md")}
    # Without a threshold.json the test is scored once, and at the threshold
    # it scored there is nothing to score again.
    with pytest.raises(ValueError, match="threshold.json"):
        evaluate.evaluate(tree, method, rethreshold=True)
    rechoose(out, 0.5)
    with pytest.raises(ValueError, match="already scored"):
        evaluate.evaluate(tree, method, rethreshold=True)
    path = rechoose(out, 0.3)
    with pytest.raises(FileExistsError, match="once"):
        evaluate.evaluate(tree, method)
    assert {s: (out / f"evaluation{s}").read_bytes() for s in old} == old
    record = evaluate.evaluate(tree, method, rethreshold=True)
    kept = out / f"{evaluate.TRAINED_STEM}.json"
    assert kept.read_bytes() == old[".json"]
    assert (out / f"{evaluate.TRAINED_STEM}.md").read_bytes() == old[".md"]
    assert json.loads((out / "evaluation.json").read_text()) == record
    model = record["model"]
    assert (model["threshold"], model["trained_threshold"]) == (0.3, 0.5)
    assert model["threshold_source"] == "threshold.json"
    assert model["threshold_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["previous_evaluation"] == {
        "path": str(kept),
        "sha256": hashlib.sha256(old[".json"]).hexdigest(),
        "threshold": 0.5,
    }
    # The scores, the bar and effectively_always are the new threshold's.
    assert first["scores"][method]["f1(H)"] == 0.0 and first["effectively_always"]
    assert record["scores"][method]["f1(H)"] == record["scores"][method]["f1(L)"] == 1
    assert record["bar"]["H1"] and not record["effectively_always"]
    assert record["scores"]["always"] == first["scores"]["always"]
    md = (out / "evaluation.md").read_text()
    assert "Threshold 0.3, re-chosen on the val shots (threshold.json)" in md
    assert f"{evaluate.TRAINED_STEM}.json" in md
    # The trained threshold's record is never overwritten.
    rechoose(out, 0.35)
    with pytest.raises(FileExistsError, match="trained-threshold"):
        evaluate.evaluate(tree, method, rethreshold=True)
    assert kept.read_bytes() == old[".json"]


@pytest.mark.parametrize("dirty", [True, None])
def test_rethreshold_refuses_uncommitted_code(tree, monkeypatch, dirty):
    """`--rethreshold` replaces a record that names its commit, so it is refused
    first while the tree differs from the commit or git cannot say."""
    method, shot = "hmode_frames", SHOTS[HMODE]
    frames_shots.make(tree, method)
    assert prepare.prepare(tree, method, [shot])["written"] == [shot]
    _resplit(tree, method, {shot: "test"})
    out = _save_model(tree, method)
    monkeypatch.setattr(evaluate, "model_probs", _h_below_half)
    evaluate.evaluate(tree, method)
    rechoose(out, 0.3)
    old = {s: (out / f"evaluation{s}").read_bytes() for s in (".json", ".md")}
    monkeypatch.setattr(evaluate, "git_dirty", lambda: dirty)
    monkeypatch.setattr(evaluate, "read_shots", lambda *a, **k: pytest.fail("read"))
    with pytest.raises(RuntimeError, match="commit first"):
        evaluate.evaluate(tree, method, rethreshold=True)
    assert {s: (out / f"evaluation{s}").read_bytes() for s in old} == old
    assert not list(out.glob(f"{evaluate.TRAINED_STEM}.*"))
