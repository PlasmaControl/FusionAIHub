"""The tokenizer command. Its functions are tested with the harness replaced; runs on data are trials."""

import collections
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import harness, paths, tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, fit

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mast" / "run_tokenizer.py"
IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)
PROV = {"tokenizer_code_sha256": "a" * 64, "assignment_sha256": "b" * 64, "fit_sha256": "c" * 64,
        "stats_sha256": "d" * 64}


def load():
    spec = importlib.util.spec_from_file_location("run_tokenizer_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ip_fit():
    st = transform.ChannelStats(compress="none", scale=np.ones(1), mean=np.zeros(1), std=np.ones(1))
    shot = np.random.default_rng(0).standard_normal((1, 400)).astype(np.float32)
    return fit.fit_signal(IP, st, [shot], kind="frame_dct")


def fake_score(calls):
    def score(task, predict_fn, out_dir, *, split, shots=None, history_s=None, **kw):
        calls.append({"split": split, "shots": shots, "history_s": history_s, **kw})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return harness.ScoreResult(task=task.name, task_nrmse=0.25,
                                   per_signal=pd.DataFrame({"feature_name": ["summary-ip"], "NRMSE": [0.25]}),
                                   per_shot=pd.DataFrame({"shot_id": [1, 2], "NRMSE": [0.2, 0.3]}),
                                   n_windows=8, out_dir=Path(out_dir))
    return score


def test_names_of_predictors_and_splits():
    rt = load()
    assert rt.predictor_name("floor", "v1", "ranged") == "tok.v1.ranged"
    assert rt.predictor_name("passthrough", "v1", "unquantized", "lr1e4") == "pass.v1.unquantized.lr1e4"
    assert rt.predictor_name("inputs", "v1", "nearest", "base") == "inputs.v1.nearest.base"
    assert rt.predictor_name("raw", None, None, "base") == "raw.base"
    assert rt.split_label("val_report", None) == "val_report" and rt.split_label("val_select", 100) == "val_select-n100"


def test_a_score_is_stored_with_its_provenance_and_kept_while_nothing_changed(tmp_path, monkeypatch, toy_task):
    rt = load()
    calls = []
    monkeypatch.setattr(rt.harness, "score", fake_score(calls))

    def predict(batch):
        return {}

    predict.history_s = 0.05
    out = tmp_path / "scores" / "task_toy" / "tok.v1.ranged" / "val_select-n100"
    kw = dict(predictor="tok.v1.ranged", out_dir=out, split="val_select", label="val_select-n100", shots=[1, 2],
              provenance=PROV, records={"mode": "ranged"})
    record = rt.score_one(toy_task, predict, **kw)
    assert record["nrmse"] == 0.25 and record["split"] == "val_select-n100" and record["predictor"] == "tok.v1.ranged"
    assert record["assignment_sha256"] == "b" * 64 and record["mode"] == "ranged"
    assert calls[0]["split"] == "val_select" and calls[0]["shots"] == [1, 2] and calls[0]["history_s"] == 0.05
    assert json.loads((out / "score.json").read_text()) == record
    assert pd.read_csv(out / "signals.csv").to_dict("records") == [{"feature_name": "summary-ip", "NRMSE": 0.25}]
    assert rt.score_one(toy_task, predict, **kw) == record and len(calls) == 1               # kept
    rt.score_one(toy_task, predict, **{**kw, "provenance": {**PROV, "assignment_sha256": "e" * 64}})
    assert len(calls) == 2                                                                   # another assignment
    rt.score_one(toy_task, predict, force=True, **kw)
    assert len(calls) == 3


def test_provenance_names_the_code_the_assignment_the_fit_and_the_data(tmp_path):
    rt = load()
    cache_root, fit_dir = tmp_path / "cache", tmp_path / "fit"
    cache_root.mkdir()
    fit_dir.mkdir()
    (cache_root / "stats.npz").write_bytes(b"statistics")
    sha = transform.stats_sha256(cache_root / "stats.npz")
    (cache_root / "manifest.json").write_text(json.dumps({"stats_sha256": sha, "tokamark_revision": "146b0a0f8738"}))
    assign.save_assignment({"summary-ip": assign.Setting("frame_dct", (4,))}, tmp_path / "v1.json")
    fit.save_fit({"summary-ip": ip_fit()}, fit_dir, {"stats_sha256": sha})
    p = rt.provenance_of(tmp_path / "v1.json", fit_dir, cache_root)
    assert list(p) == ["tokenizer_code_sha256", "assignment_sha256", "fit_sha256", "stats_sha256", "tokamark_revision",
                       "data_root"]
    assert p["stats_sha256"] == sha and p["tokamark_revision"] == "146b0a0f8738"
    assert p["data_root"] == str(paths.tokamark_zarr_root()) and p["fit_sha256"] == fit.fit_sha256(fit_dir)
    assert list(rt.provenance_of(None, None, cache_root, tmp_path / "v1.json")) == [
        "tokamark_revision", "data_root", "checkpoint_sha256"]                  # `raw`: any file stands for a checkpoint
    (cache_root / "stats.npz").write_bytes(b"other statistics")
    with pytest.raises(ValueError, match="statistics"):
        rt.provenance_of(tmp_path / "v1.json", fit_dir, cache_root)


def test_the_test_split_and_a_wrong_use_of_the_gpu_are_refused(tmp_path, monkeypatch, capsys):
    rt = load()
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    base = ["score", "--task", "task_1-1", "--assignment", str(tmp_path / "v1.json"), "--mode", "ranged"]
    with pytest.raises(SystemExit):
        rt.main([*base, "--split", "test"])
    assert "invalid choice" in capsys.readouterr().err
    assert rt.main([*base, "--split", "val_report", "--instrument", "passthrough"]) == 2
    assert "--gpu" in capsys.readouterr().err
    assert rt.main([*base, "--split", "val_report", "--gpu", "0"]) == 2          # the floor uses no GPU
    assert "--gpu" in capsys.readouterr().err
    assert not (tmp_path / "stage1").exists()                                    # refused before anything is written


def test_fit_writes_once_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    rt = load()
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    cache_root = tmp_path / "cache" / "v1"
    cache_root.mkdir(parents=True)
    (cache_root / "manifest.json").write_text(json.dumps({"stats_sha256": "d" * 64, "tokamark_revision": "146b0a0f8738"}))
    monkeypatch.setattr(rt.tasks, "split_shots", lambda split: list(range(30000, 30040)))
    seen = {}

    def fit_all(root, shots, *, workers, names=None):
        seen.update(root=Path(root), shots=list(shots), workers=workers)
        return {"summary-ip": ip_fit()}

    monkeypatch.setattr(rt.fit, "fit_all", fit_all)
    assert rt.main(["fit", "--n-shots", "4", "--workers", "2"]) == 0
    assert seen == {"root": cache_root, "shots": [30000, 30013, 30026, 30039], "workers": 2}
    fits, meta = fit.load_fit(tmp_path / "stage1" / "fit" / "v1")
    assert set(fits) == {"summary-ip"} and meta["n_shots"] == 4 and meta["stats_sha256"] == "d" * 64
    assert meta["shots"] == [30000, 30013, 30026, 30039] and len(meta["tokenizer_code_sha256"]) == 64
    assert rt.main(["fit", "--n-shots", "4", "--workers", "2"]) == 2
    assert "--force" in capsys.readouterr().err


def test_a_fit_refused_for_a_missing_manifest_leaves_no_directory(tmp_path, monkeypatch, capsys):
    rt = load()
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    (tmp_path / "cache" / "v1").mkdir(parents=True)                              # no manifest.json in it
    monkeypatch.setattr(rt.tasks, "split_shots", lambda split: list(range(30000, 30040)))
    monkeypatch.setattr(rt.fit, "fit_all", lambda *a, **k: pytest.fail("fit_all must not run"))
    assert rt.main(["fit", "--n-shots", "4", "--workers", "2"]) == 2
    assert "manifest" in capsys.readouterr().err
    assert not (tmp_path / "stage1" / "fit" / "v1").exists()


# ------------------------------------------------------------ assign: ladders of other code are refused

CURRENT = {"tokenizer_code_sha256": "a" * 64, "fit_sha256": "c" * 64, "stats_sha256": "d" * 64}


class _Compress:
    compress = "none"


def assign_setup(tmp_path, monkeypatch, rt, *, changed=None, only=None):
    """A stage-1 root with a fit and the seven ladder files, all made with `CURRENT`; `changed` maps a
    file name (without .json) to the keys of its JSON that are replaced (None removes a key). The provenance,
    the statistics and the choice of the rules are replaced: only the checks of `assign` are under test."""
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    stage1 = tmp_path / "stage1"
    (stage1 / "fit" / "v1").mkdir(parents=True)
    (stage1 / "fit" / "v1" / "fit.npz").write_bytes(b"")
    (stage1 / "ladders").mkdir()
    for name in (*rt.ladder.LADDER_TASKS, "inputs"):
        pd.DataFrame({"signal": [f"signal-of-{name}"], "domain": ["linear"], "k": [1]}).to_csv(
            stage1 / "ladders" / f"{name}.csv", index=False)
        record = {**CURRENT, **(changed or {}).get(name, {})}
        (stage1 / "ladders" / f"{name}.json").write_text(
            json.dumps({k: v for k, v in record.items() if v is not None}))
    monkeypatch.setattr(rt, "_fit_provenance", lambda fit_dir, cache_root: dict(CURRENT))
    monkeypatch.setattr(rt.transform, "load_stats", lambda path: collections.defaultdict(_Compress))
    chosen = {"summary-ip": assign.Setting("frame_dct", (4,))}
    monkeypatch.setattr(rt.ladder, "build_assignment", lambda *a, **k: (chosen, {"summary-ip": "toy"}))
    return stage1


def test_assign_accepts_ladders_of_this_code_and_those_that_predate_the_scoring_hash(tmp_path, monkeypatch, capsys):
    rt = load()
    stage1 = assign_setup(tmp_path, monkeypatch, rt)                      # no ladder has `scoring_code_sha256`
    assert rt.main(["assign", "--name", "v1"]) == 0
    assert assign.load_assignment(stage1 / "assignments" / "v1.json") == {"summary-ip": assign.Setting("frame_dct", (4,))}
    now = rt.baselines.scoring_code_sha256()
    same = {n: {"scoring_code_sha256": now} for n in (*rt.ladder.LADDER_TASKS, "inputs")}
    stage1 = assign_setup(tmp_path / "again", monkeypatch, rt, changed=same)       # every ladder has it, and it matches
    assert rt.main(["assign", "--name", "v1"]) == 0 and (stage1 / "assignments" / "v1.json").is_file()


@pytest.mark.parametrize("key, named", [("fit_sha256", "another fit"), ("tokenizer_code_sha256", "tokenizer_code_sha256"),
                                        ("stats_sha256", "stats_sha256"), ("scoring_code_sha256", "scoring_code_sha256")])
def test_assign_refuses_a_ladder_that_differs_in_one_key_and_names_the_file_and_the_key(
        tmp_path, monkeypatch, capsys, key, named):
    rt = load()
    stage1 = assign_setup(tmp_path, monkeypatch, rt, changed={"task_2-1": {key: "9" * 64}})
    assert rt.main(["assign", "--name", "v1"]) == 2
    err = capsys.readouterr().err
    assert "task_2-1.json" in err and named in err
    assert "task_1-2.json" not in err                                    # the other six files are not blamed
    assert not (stage1 / "assignments").exists()                         # refused before anything is written


def test_assign_refuses_a_ladder_with_no_tokenizer_code_hash(tmp_path, monkeypatch, capsys):
    rt = load()
    assign_setup(tmp_path, monkeypatch, rt, changed={"inputs": {"tokenizer_code_sha256": None}})
    assert rt.main(["assign", "--name", "v1"]) == 2
    assert "inputs.json" in capsys.readouterr().err


def test_ladder_records_the_scoring_code_hash_from_now_on():
    rt = load()
    args = type("Args", (), {"n_shots": 100})()
    record = rt._ladder_record([1, 2], args, CURRENT, task="task_1-2")
    assert record["scoring_code_sha256"] == rt.baselines.scoring_code_sha256()
    assert record["tokenizer_code_sha256"] == CURRENT["tokenizer_code_sha256"] and record["task"] == "task_1-2"


def test_assign_refuses_names_options_and_files_that_do_not_fit(tmp_path, monkeypatch, capsys):
    rt = load()
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    for argv, text in [(["--name", "../x"], "not a file name"),
                       (["--name", "v2", "--raise-inputs"], "go together"),
                       (["--name", "v2", "--from", "v1"], "go together"),
                       (["--name", "v2", "--reference", "--from", "v1", "--raise-inputs"], "takes no --from"),
                       (["--name", "v2", "--from", "v1", "--raise-inputs"], "there is no assignment"),
                       (["--name", "v2"], "there is no fit")]:
        assert rt.main(["assign", *argv]) == 2
        assert text in capsys.readouterr().err
    (tmp_path / "stage1" / "assignments").mkdir(parents=True)
    (tmp_path / "stage1" / "assignments" / "v1.json").write_text("{}")
    assert rt.main(["assign", "--name", "v1"]) == 2
    assert "is not overwritten" in capsys.readouterr().err
    assert (tmp_path / "stage1" / "assignments" / "v1.json").read_text() == "{}"


def test_assign_refuses_missing_ladders_and_a_signal_scored_twice_and_the_rules_that_refuse(
        tmp_path, monkeypatch, capsys):
    rt = load()
    stage1 = assign_setup(tmp_path, monkeypatch, rt)
    (stage1 / "ladders" / "task_3-1.csv").unlink()
    (stage1 / "ladders" / "inputs.json").unlink()
    assert rt.main(["assign", "--name", "v1"]) == 2
    err = capsys.readouterr().err
    assert "task_3-1.csv" in err and "inputs.json" in err and "missing" in err
    (stage1 / "ladders" / "task_3-1.csv").write_text("signal,domain,k\nsignal-of-task_1-2,linear,1\n")
    (stage1 / "ladders" / "inputs.json").write_text(json.dumps(CURRENT))
    assert rt.main(["assign", "--name", "v1"]) == 2
    assert "signal-of-task_1-2 more than once" in capsys.readouterr().err
    (stage1 / "ladders" / "task_3-1.csv").write_text("signal,domain,k\nsignal-of-task_3-1,linear,1\n")

    def refuses(*a, **k):
        raise ValueError("no rung of x meets its target")

    monkeypatch.setattr(rt.ladder, "build_assignment", refuses)
    assert rt.main(["assign", "--name", "v1"]) == 2
    assert "no rung of x" in capsys.readouterr().err
    assert not (stage1 / "assignments").exists()


# --------------------------------------------------------------------------- `_raise_inputs`, rule 4

def test_raise_inputs_raises_each_signal_no_task_predicts_by_one_rung():
    rt = load()
    specs = {s.name: s for s in rt.registry.all_signals()}
    old = rt.assign.reference_assignment(list(specs.values()))
    new, notes = rt._raise_inputs(old, {"x": "kept"}, "ref", specs)
    inputs = sorted(n for n, by in rt.assign.predicted_by().items() if not by)
    assert len(inputs) == 11
    for name in specs:
        if name in inputs:
            rungs = rt.assign.rungs(specs[name], old[name].domain)
            assert old[name] in rungs and new[name] == rungs[rungs.index(old[name]) + 1]
            assert notes[name].startswith("rule 4: raised from K ") and "(first assignment ref)" in notes[name]
        else:
            assert new[name] == old[name] and name not in notes                # a signal a task predicts stays
    assert notes["x"] == "kept"


def test_raise_inputs_keeps_the_largest_rung_and_refuses_a_setting_that_is_no_rung():
    rt = load()
    specs = {s.name: s for s in rt.registry.all_signals()}
    name = sorted(n for n, by in rt.assign.predicted_by().items() if not by)[0]
    old = rt.assign.reference_assignment(list(specs.values()))
    top = rt.assign.rungs(specs[name], old[name].domain)[-1]
    new, notes = rt._raise_inputs({**old, name: top}, {}, "v1", specs)
    assert new[name] == top and "already the largest rung" in notes[name] and "(first assignment v1)" in notes[name]
    odd = assign.Setting("frame_dct", (5,), old[name].domain)                  # K = 5 is not a rung
    with pytest.raises(rt.Refused, match=f"v1: the setting of {name}"):
        rt._raise_inputs({**old, name: odd}, {}, "v1", specs)


# ---------------------------------------------- the per-signal file is written before the score is kept

def _result(task, out_dir):
    return harness.ScoreResult(task=task.name, task_nrmse=0.25,
                               per_signal=pd.DataFrame({"feature_name": ["summary-ip"], "NRMSE": [0.25]}),
                               per_shot=pd.DataFrame({"shot_id": [1, 2], "NRMSE": [0.2, 0.3]}),
                               n_windows=8, out_dir=Path(out_dir))


def _disk_full(path, frame):
    raise OSError("no space left on device")


def test_score_one_writes_signals_before_score_json_and_a_failed_write_leaves_no_new_score(
        tmp_path, monkeypatch, toy_task):
    rt = load()
    monkeypatch.setattr(rt.harness, "score", fake_score([]))
    seen = []
    write_score_json = rt.baselines.write_score_json

    def watching(result, predictor, label):
        seen.append((Path(result.out_dir) / "signals.csv").is_file())
        return write_score_json(result, predictor, label)

    monkeypatch.setattr(rt.baselines, "write_score_json", watching)

    def predict(batch):
        return {}

    kw = dict(predictor="tok.v1.ranged", split="val_select", label="val_select-n100", shots=[1, 2], provenance=PROV,
              records={})
    rt.score_one(toy_task, predict, out_dir=tmp_path / "ok", **kw)
    assert seen == [True]                                              # `signals.csv` was there when score.json was made
    monkeypatch.setattr(rt, "_write_csv_atomic", _disk_full)
    with pytest.raises(OSError, match="no space"):
        rt.score_one(toy_task, predict, out_dir=tmp_path / "new", **kw)
    assert not (tmp_path / "new" / "score.json").exists() and not (tmp_path / "new" / "signals.csv").exists()
    assert (tmp_path / "ok" / "score.json").is_file() and (tmp_path / "ok" / "signals.csv").is_file()
    with pytest.raises(OSError, match="no space"):
        rt.score_one(toy_task, predict, out_dir=tmp_path / "ok", force=True, **kw)
    # the old pair is being replaced: its score.json is gone, so the old signals.csv is not kept as current
    assert not (tmp_path / "ok" / "score.json").exists()


def test_score_windows_writes_signals_before_score_json_and_a_failed_write_leaves_no_new_score(
        tmp_path, monkeypatch, toy_task):
    rt = load()
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))

    def score_batches(name, batches, predict_fn, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return _result(toy_task, out_dir)

    monkeypatch.setattr(rt.harness, "score_batches", score_batches)
    kw = dict(predictor="tok.x.unquantized", label="val_select-n100", provenance=PROV, records={"instrument": "floor"})
    out = tmp_path / "stage1" / "scores" / "task_toy" / "tok.x.unquantized" / "val_select-n100"
    assert rt._score_windows(toy_task, [], lambda b: {}, **kw) == {"summary-ip": 0.25}
    assert json.loads((out / "score.json").read_text())["instrument"] == "floor"
    assert pd.read_csv(out / "signals.csv").to_dict("records") == [{"feature_name": "summary-ip", "NRMSE": 0.25}]
    (out / "score.json").unlink()
    monkeypatch.setattr(rt, "_write_csv_atomic", _disk_full)
    with pytest.raises(OSError, match="no space"):
        rt._score_windows(toy_task, [], lambda b: {}, **kw)
    assert not (out / "score.json").exists()


def test_score_windows_replacing_a_stored_score_leaves_no_score_json_when_the_new_signals_fail(
        tmp_path, monkeypatch, toy_task):
    rt = load()
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))

    def score_batches(name, batches, predict_fn, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return _result(toy_task, out_dir)

    monkeypatch.setattr(rt.harness, "score_batches", score_batches)
    kw = dict(predictor="tok.x.unquantized", label="val_select-n100", provenance=PROV, records={})
    out = tmp_path / "stage1" / "scores" / "task_toy" / "tok.x.unquantized" / "val_select-n100"
    rt._score_windows(toy_task, [], lambda b: {}, **kw)
    assert (out / "score.json").is_file() and (out / "signals.csv").is_file()   # a stored pair
    monkeypatch.setattr(rt, "_write_csv_atomic", _disk_full)
    with pytest.raises(OSError, match="no space"):
        rt._score_windows(toy_task, [], lambda b: {}, **kw)
    assert not (out / "score.json").exists()                  # not kept: scored again, never half old and half new


# ------------------------------------------------------------------------------ gate: rule 6, stage 0

def stage0_score(root, task, predictor, nrmse, rt):
    out = root / "scores" / task / predictor / "val_report"
    out.mkdir(parents=True)
    (out / "score.json").write_text(json.dumps({
        "task": task, "predictor": predictor, "split": "val_report", "nrmse": nrmse, "n_shots": 2, "n_windows": 8,
        "git_revision": "abc", "date": "2026-11-23", "code_sha256": rt.baselines.scoring_code_sha256()}))


def gate_setup(tmp_path, monkeypatch, rt, toy_task, ranged):
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    stage1 = tmp_path / "stage1"
    (stage1 / "fit" / "v1").mkdir(parents=True)
    (stage1 / "fit" / "v1" / "fit.npz").write_bytes(b"")
    assign.save_assignment({"summary-ip": assign.Setting("frame_dct", (4,))}, tmp_path / "v1.json")
    (tmp_path / "cache" / "v1" / "shots").mkdir(parents=True)
    for shot in (1, 2):
        (tmp_path / "cache" / "v1" / "shots" / f"{shot}.h5").write_bytes(b"")
    floors = {"unquantized": 0.1, "ranged": ranged, "nearest": 0.2}
    monkeypatch.setattr(rt.tasks, "ALL_TASKS", ("task_toy",))
    monkeypatch.setattr(rt.tasks, "load_task", lambda name: toy_task)
    monkeypatch.setattr(rt.tasks, "best_published", lambda name: 0.5)
    monkeypatch.setattr(rt.tasks, "split_shots", lambda split: [1, 2])
    monkeypatch.setattr(rt, "provenance_of", lambda *a, **k: dict(CURRENT))
    monkeypatch.setattr(rt, "_stored_floors", lambda stage1, t, who, mode, label, prov: (
        {"summary-ip": floors[mode]}, 0.3, ""))
    monkeypatch.setattr(rt.transform, "map_shots", lambda job, shots, args, workers: [{"summary-ip": None} for _ in shots])
    monkeypatch.setattr(rt.transform, "load_stats", lambda path: {})
    monkeypatch.setattr(rt.fit, "load_fit", lambda path: ({}, {}))
    return ["gate", "--assignment", str(tmp_path / "v1.json"), "--n-shots", "2", "--workers", "1",
            "--stage0-dir", str(tmp_path / "stage0")]


def test_gate_reports_a_range_that_is_too_narrow_and_a_stage0_score_that_is_not_a_number(
        tmp_path, monkeypatch, capsys, toy_task):
    rt = load()
    argv = gate_setup(tmp_path, monkeypatch, rt, toy_task, ranged=0.15)           # 50% above the unquantized floor
    stage0_score(tmp_path / "stage0", "task_toy", "mean", None, rt)                # a record with no number
    stage0_score(tmp_path / "stage0", "task_toy", "floor", 0.2, rt)
    assert rt.main(argv) == 0                                                      # reported: no verdict, no exit code
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith("range too narrow")]
    assert len(line) == 1 and "task_toy summary-ip" in line[0] and "50.0% above" in line[0]
    assert "ranged 0.1500" in line[0] and "unquantized 0.1000" in line[0] and "not fixed here" in line[0]
    assert "stage 0 on val_report: mean no score, floor 0.2000" in out
    verdict = pd.read_csv(tmp_path / "stage1" / "gate" / "v1.csv")["verdict"].tolist()
    assert len(verdict) == 1 and not verdict[0].startswith("blocks")


def test_gate_is_silent_about_a_range_within_one_percent(tmp_path, monkeypatch, capsys, toy_task):
    rt = load()
    argv = gate_setup(tmp_path, monkeypatch, rt, toy_task, ranged=0.1009)         # 0.9% above
    assert rt.main(argv) == 0
    out = capsys.readouterr().out
    assert "range too narrow" not in out and "mean no score, floor no score" in out


def test_the_narrow_range_rule_skips_missing_floors_and_a_zero_floor():
    rt = load()
    nan = float("nan")
    table_ = pd.DataFrame({"task": ["t"] * 4, "signal": ["a", "b", "c", "d"], "unquantized": [nan, 0.1, 0.0, 0.1],
                           "ranged": [0.2, nan, 0.1, 0.101]})
    lines = rt._narrow_range_lines(table_)
    assert len(lines) == 1 and lines[0].startswith("range too narrow") and " t c: ranged 0.1000 is above " in lines[0]
    assert [rt._nrmse_text(r) for r in (None, {}, {"nrmse": None}, {"nrmse": "x"}, {"nrmse": float("nan")},
                                        {"nrmse": 0.12345})] == ["no score"] * 5 + ["0.1235"]


# --------------------------------------------------------------- score: the floor and the GPU wiring

class FakeHeadroom:
    """What `run_score` takes from `run_headroom.py`: the GPU lookup and checks, and the staleness reason."""

    def __init__(self, stale=""):
        self.stale, self.asked = stale, []

    def _cuda_initialized(self):
        return False

    def _look_up_gpu(self, index):
        self.asked.append(index)
        return f"GPU-uuid-of-{index}", [4242]

    def _refuse_busy_gpu(self, index, pids, share):
        return False

    def stale_reason(self, stage0, task, predictor, split):
        self.asked.append((task, predictor, split))
        return self.stale


def score_setup(tmp_path, monkeypatch, rt, toy_task, rh, *, checkpoint=True):
    """Everything `run_score` reads, made or replaced; returns the argv base and the calls of `score_one`."""
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")                  # so that the teardown restores what `run_score` sets
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(rt.tasks, "ALL_TASKS", ("task_toy",))
    monkeypatch.setattr(rt.tasks, "load_task", lambda name: toy_task)
    (tmp_path / "stage1" / "fit" / "v1").mkdir(parents=True)
    (tmp_path / "stage1" / "fit" / "v1" / "fit.npz").write_bytes(b"")
    assign.save_assignment({"summary-ip": assign.Setting("frame_dct", (4,))}, tmp_path / "v1.json")
    if checkpoint:
        best = rt.table.direct_run_dir(tmp_path / "stage0", "task_toy", rt.table.BASE_VARIANT) / "best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"weights")
    monkeypatch.setattr(rt, "_run_headroom", lambda: rh)
    monkeypatch.setattr(rt, "provenance_of", lambda assignment, fit_dir, cache_root, checkpoint=None: {
        **CURRENT, **({"checkpoint_sha256": "e" * 64} if checkpoint is not None else {})})
    monkeypatch.setattr(rt, "_build_predictor", lambda *a, **k: (lambda batch: {}))
    calls = []

    def score_one(task, predict_fn, **kw):
        calls.append(kw)
        return {"nrmse": 0.25, "n_windows": 8}

    monkeypatch.setattr(rt, "score_one", score_one)
    return ["score", "--task", "task_toy", "--split", "val_report", "--assignment", str(tmp_path / "v1.json"),
            "--mode", "ranged", "--stage0-dir", str(tmp_path / "stage0")], calls


def test_score_with_a_gpu_selects_it_and_records_why_the_direct_score_is_stale(tmp_path, monkeypatch, capsys, toy_task):
    rt = load()
    rh = FakeHeadroom(stale="the checkpoint is newer than its score")
    base, calls = score_setup(tmp_path, monkeypatch, rt, toy_task, rh)
    assert rt.main([*base, "--instrument", "passthrough", "--gpu", "2"]) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-uuid-of-2"                    # the uuid `_look_up_gpu` gave
    assert rh.asked == [2, ("task_toy", rt.table.predictor_name("direct", "base"), "val_report")]
    (kw,) = calls
    assert kw["predictor"] == "pass.v1.ranged.base" and kw["label"] == "val_report"
    assert kw["records"] == {"instrument": "passthrough", "mode": "ranged", "assignment": "v1.json",
                             "variant": "base", "n_shots_asked": None,
                             "direct_stale_reason": "the checkpoint is newer than its score", "scored_blind": True}
    assert kw["provenance"]["checkpoint_sha256"] == "e" * 64
    assert "stale (the checkpoint is newer than its score)" in capsys.readouterr().err
    assert rt.main([*base, "--instrument", "passthrough", "--gpu", "1", "--variant", "lr1e4"]) == 2
    assert "no checkpoint" in capsys.readouterr().err                               # none for the other variant
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-uuid-of-1"                    # the GPU is set before that is seen


def test_score_refuses_a_missing_checkpoint_after_the_gpu_is_selected(tmp_path, monkeypatch, capsys, toy_task):
    rt = load()
    rh = FakeHeadroom()
    base, calls = score_setup(tmp_path, monkeypatch, rt, toy_task, rh, checkpoint=False)
    assert rt.main([*base, "--instrument", "passthrough", "--gpu", "3"]) == 2
    err = capsys.readouterr().err
    assert "there is no checkpoint at" in err and "direct/task_toy/base/best.pt" in err and "run_headroom.py" in err
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-uuid-of-3" and calls == []   # nothing was scored


def test_score_of_the_floor_uses_no_gpu_and_records_no_direct_model(tmp_path, monkeypatch, toy_task):
    rt = load()
    base, calls = score_setup(tmp_path, monkeypatch, rt, toy_task, rh=None)
    monkeypatch.setattr(rt, "_run_headroom", lambda: pytest.fail("the floor does not load run_headroom.py"))
    assert rt.main(base) == 0
    (kw,) = calls
    assert kw["predictor"] == "tok.v1.ranged" and kw["records"]["instrument"] == "floor"
    assert kw["records"]["variant"] is None and "direct_stale_reason" not in kw["records"]
    assert "checkpoint_sha256" not in kw["provenance"] and "CUDA_VISIBLE_DEVICES" not in os.environ
