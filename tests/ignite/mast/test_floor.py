"""TokaMind's output-representation floor and the headroom command."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tokamak_foundation_model.ignite.mast import harness, tasks
from tokamak_foundation_model.ignite.mast.headroom import FITTED, floor

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mast" / "run_headroom.py"


def _run_headroom():
    """The command as a module (it lives in scripts/, which is not a package)."""
    spec = importlib.util.spec_from_file_location("run_headroom_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_coefficients_reproduce_the_input():
    x = np.random.default_rng(0).standard_normal((4, 18, 50))
    np.testing.assert_allclose(floor.truncate_dct(x, np.arange(18 * 1 * 50), (18, 1, 50)), x, atol=1e-10)


def test_dc_only_gives_the_window_mean():
    x = np.random.default_rng(0).standard_normal((4, 18, 50))
    y = floor.truncate_dct(x, np.array([0]), (18, 1, 50))
    np.testing.assert_allclose(y, np.broadcast_to(x.mean(axis=(1, 2), keepdims=True), x.shape), atol=1e-10)


def test_matches_tokaminds_codec():
    pytest.importorskip("mmt")
    import torch
    from mmt.data.embeddings.dct3d import DCT3DCodec, DCT3DTorchDecoder

    rng = np.random.default_rng(1)
    x = rng.standard_normal((18, 50)).astype(np.float32)
    indices = np.sort(rng.choice(18 * 50, size=40, replace=False))
    codec = DCT3DCodec(keep_h=1, keep_w=1, keep_t=1, selection_mode="rank",
                       coeff_indices=indices, coeff_shape=(18, 1, 50))
    theirs = DCT3DTorchDecoder(codec, (18, 50))(torch.from_numpy(codec.encode(x))[None])[0].numpy()
    ours = floor.truncate_dct(x[None].astype(np.float64), indices, (18, 1, 50))[0]
    np.testing.assert_allclose(ours, theirs, atol=1e-5)


@pytest.mark.real_data
def test_task_1_2_indices():
    idx = floor.tokamind_output_indices("task_1-2")
    assert set(idx) == {"equilibrium-lcfs_r", "equilibrium-lcfs_z"}
    assert len(idx["equilibrium-lcfs_r"][0]) == 10 and idx["equilibrium-lcfs_r"][1] == (170, 1, 1)


def test_a_three_dimensional_signal_round_trips():
    x = np.random.default_rng(2).standard_normal((2, 3, 8, 5))
    y = floor.truncate_dct(x, np.arange(3 * 8 * 5), (3, 8, 5))
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_a_scalar_signal_is_a_one_by_one_by_t_view():
    x = np.random.default_rng(3).standard_normal((3, 1, 20))
    y = floor.truncate_dct(x, np.array([0, 1, 2]), (1, 1, 20))
    assert y.shape == x.shape
    assert not np.allclose(y, x)


def test_truncation_rejects_a_shape_that_is_not_the_coefficient_shape():
    x = np.zeros((2, 18, 50))
    with pytest.raises(ValueError, match="coeff_shape"):
        floor.truncate_dct(x, np.array([0]), (18, 1, 49))
    with pytest.raises(ValueError, match="out of range"):
        floor.truncate_dct(x, np.array([18 * 50]), (18, 1, 50))


def _write_run(tmp_path: Path, name: str, indices: np.ndarray, coeff_shape) -> Path:
    embeddings = tmp_path / f"ft-{name}-ws-tokamind-base-v2-repl" / "embeddings"
    (embeddings / "dct3d_indices").mkdir(parents=True)
    np.save(embeddings / "dct3d_indices" / "output_summary-ip.npy", indices.astype(np.int32))
    kwargs = {"selection_mode": "rank", "coeff_shape": list(coeff_shape), "num_coeffs": len(indices),
              "coeff_indices_path": "dct3d_indices/output_summary-ip.npy"}
    config = {"embeddings": {"per_signal_overrides": {"output": {
        "summary-ip": {"encoder_name": "dct3d", "encoder_kwargs": kwargs}}}}}
    (embeddings / "dct3d.yaml").write_text(yaml.safe_dump(config))
    return embeddings


def test_indices_are_read_from_the_run(toy_task, tmp_path):
    embeddings = _write_run(tmp_path, "task_toy", np.array([0, 3, 7]), (1, 1, 100))
    out = floor.output_indices_from_run(embeddings, toy_task)
    assert set(out) == {"summary-ip"}
    indices, shape = out["summary-ip"]
    assert indices.tolist() == [0, 3, 7] and shape == (1, 1, 100)


def test_indices_of_a_changed_chunking_raise(toy_task, tmp_path):
    embeddings = _write_run(tmp_path, "task_toy", np.array([0, 3, 7]), (1, 1, 50))
    with pytest.raises(ValueError, match="coeff_shape"):
        floor.output_indices_from_run(embeddings, toy_task)


def test_a_missing_output_signal_raises(toy_task, tmp_path):
    embeddings = _write_run(tmp_path, "task_toy", np.array([0]), (1, 1, 100))
    other = tasks.SignalSpec("summary-q", 0.00025, (1,), 0.0, 1.0)
    two = tasks.TaskSpec(**{**toy_task.__dict__, "outputs": (*toy_task.outputs, other),
                            "steps": {**toy_task.steps, "output": {"summary-ip": 100, "summary-q": 100}}})
    with pytest.raises(ValueError, match="summary-q"):
        floor.output_indices_from_run(embeddings, two)


def test_the_floor_with_every_coefficient_returns_the_targets(toy_task, toy_batch):
    batch = toy_batch(6)
    indices = {"summary-ip": (np.arange(100), (1, 1, 100))}
    out = floor.floor_predictor(toy_task, indices)(batch)
    assert out["summary-ip"].dtype == np.float32
    np.testing.assert_allclose(out["summary-ip"], batch.targets["summary-ip"], rtol=1e-5, atol=1.0)


def test_the_floor_is_finite_where_targets_are_missing_and_does_not_mutate_them(toy_task, toy_batch):
    batch = toy_batch(6)
    batch.targets["summary-ip"][2, 0, 10:20] = np.nan
    before = batch.targets["summary-ip"].copy()
    out = floor.floor_predictor(toy_task, {"summary-ip": (np.arange(10), (1, 1, 100))})(batch)
    assert np.isfinite(out["summary-ip"]).all()
    np.testing.assert_array_equal(batch.targets["summary-ip"], before)


def test_the_floor_standardizes_with_the_signals_statistics(toy_task, toy_batch):
    batch = toy_batch(4)
    spec = toy_task.outputs[0]
    # DC only: the standardized window mean, then back to physical units
    out = floor.floor_predictor(toy_task, {"summary-ip": (np.array([0]), (1, 1, 100))})(batch)
    expected = batch.targets["summary-ip"].mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(out["summary-ip"], np.broadcast_to(expected, out["summary-ip"].shape),
                               rtol=1e-4, atol=spec.std * 1e-4)


def test_fitted_predictors_are_never_scored_on_test():
    assert FITTED == ("shrunk", "ridge", "direct")
    run_headroom = _run_headroom()
    for name in FITTED:
        assert run_headroom.main(["--task", "task_1-2", "--predictor", name, "--split", "test"]) == 2


def test_a_fitted_predictor_without_an_implementation_exits_with_2(capsys):
    run_headroom = _run_headroom()
    assert run_headroom.main(["--task", "task_1-2", "--predictor", "ridge", "--split", "val_report"]) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_the_refusal_names_the_rule_and_opens_no_data(capsys, monkeypatch):
    run_headroom = _run_headroom()

    def forbidden(*a, **k):
        raise AssertionError("data opened")

    monkeypatch.setattr(run_headroom.harness, "score", forbidden)
    monkeypatch.setattr(run_headroom.tasks, "load_task", forbidden)
    assert run_headroom.main(["--task", "task_1-2", "--predictor", "direct", "--split", "test"]) == 2
    assert "validation" in capsys.readouterr().err


@pytest.fixture
def stub_command(monkeypatch, tmp_path, toy_task):
    """The command with scoring replaced by a stub that counts calls; returns (module, calls, out_dir)."""
    run_headroom = _run_headroom()
    root = tmp_path / "stage0"
    monkeypatch.setattr(run_headroom.paths, "stage0_dir", lambda: root)
    monkeypatch.setattr(run_headroom.paths, "prepare_write_root", lambda p: Path(p))
    monkeypatch.setattr(run_headroom.tasks, "load_task", lambda name: dataclasses.replace(toy_task, name=name))
    monkeypatch.setattr(run_headroom.floor, "tokamind_output_indices",
                        lambda name: {"summary-ip": (np.arange(5), (1, 1, 100))})
    calls = []

    def fake_score(task, predict_fn, out_dir, *, split, **kwargs):
        calls.append(split)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return harness.ScoreResult(
            task=task.name, task_nrmse=0.25,
            per_signal=pd.DataFrame({"feature_name": ["summary-ip"], "NRMSE": [0.25]}),
            per_shot=pd.DataFrame({"shot_id": [1], "NRMSE": [0.25]}), n_windows=7, out_dir=Path(out_dir))

    monkeypatch.setattr(run_headroom.harness, "score", fake_score)
    return run_headroom, calls, root / "scores" / "task_1-2" / "floor" / "val_report"


def test_a_stored_floor_score_is_kept_until_the_predictor_changes(stub_command, monkeypatch, capsys):
    run_headroom, calls, out_dir = stub_command
    argv = ["--task", "task_1-2", "--predictor", "floor", "--split", "val_report"]
    assert run_headroom.main(argv) == 0
    stored = json.loads((out_dir / "score.json").read_text())
    assert stored["predictor_sha256"] == floor.PREDICTOR_SHA256 and stored["nrmse"] == 0.25
    assert calls == ["val_report"]

    assert run_headroom.main(argv) == 0
    assert calls == ["val_report"]
    assert "kept task_1-2 floor val_report" in capsys.readouterr().out

    monkeypatch.setattr(run_headroom.floor, "PREDICTOR_SHA256", "0" * 64)
    assert run_headroom.main(argv) == 0
    assert calls == ["val_report", "val_report"]
    assert "re-scoring task_1-2 floor val_report: predictor code changed" in capsys.readouterr().out

    assert run_headroom.main(argv) == 0 and calls == ["val_report", "val_report"]
    assert run_headroom.main([*argv, "--force"]) == 0
    assert calls == ["val_report", "val_report", "val_report"]


def test_a_stored_score_without_a_predictor_hash_is_scored_again(stub_command, capsys):
    run_headroom, calls, out_dir = stub_command
    argv = ["--task", "task_1-2", "--predictor", "floor", "--split", "val_report"]
    run_headroom.main(argv)
    record = json.loads((out_dir / "score.json").read_text())
    del record["predictor_sha256"]
    (out_dir / "score.json").write_text(json.dumps(record))
    run_headroom.main(argv)
    assert len(calls) == 2
    assert "re-scoring task_1-2 floor val_report: predictor code changed" in capsys.readouterr().out
