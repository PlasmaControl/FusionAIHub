"""The tokenizer command. Its functions are tested with the harness replaced; runs on data are trials."""

import importlib.util
import json
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
