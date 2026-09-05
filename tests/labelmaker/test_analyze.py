"""`analyze`: one shot, the labels a config names, a JSON summary and a figure."""
import json

import numpy as np
import pytest
import yaml

from labelmaker import analyze, run
from labelmaker.labels.schema import LabelSpec
from labelmaker.labels.store import write_labels
from labelmaker.models.base import Decoded

from .test_run import SLUG


def _cfg(tmp_path, **over):
    body = {"labels": [f"{SLUG}/score"], "context": ["ip"], "threshold": 0.5}
    body.update(over)
    p = tmp_path / "labels.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def _argv(wired, cfg, out, *extra, shots=("190000",)):
    return [
        "analyze", "--shots", *shots, "--config", str(cfg), "--out", str(out),
        "--root", str(wired["root"]), "--corpus-dir", str(wired["corpus"]),
        "--archive", str(wired["archive"]), "--workers", "1", *extra,
    ]


def test_load_config_reads_labels_context_and_threshold(wired, tmp_path):
    cfg = analyze.load_config(_cfg(tmp_path, threshold=0.7))
    assert cfg.labels == (f"{SLUG}/score",)
    assert cfg.context == ("ip",)
    assert cfg.threshold == 0.7
    assert cfg.slugs == (SLUG,)


def test_load_config_rejects_a_label_the_model_does_not_produce(wired, tmp_path):
    with pytest.raises(analyze.ConfigError, match="nope"):
        analyze.load_config(_cfg(tmp_path, labels=[f"{SLUG}/nope"]))


def test_load_config_rejects_an_unknown_model_and_a_malformed_label(tmp_path):
    with pytest.raises(analyze.ConfigError, match="no_such_model"):
        analyze.load_config(_cfg(tmp_path, labels=["no_such_model/x"]))
    with pytest.raises(analyze.ConfigError, match="slug/name"):
        analyze.load_config(_cfg(tmp_path, labels=["tm_prob"]))


def test_load_config_rejects_a_context_name_that_is_not_a_canonical_feature(wired, tmp_path):
    with pytest.raises(analyze.ConfigError, match="not a canonical feature"):
        analyze.load_config(_cfg(tmp_path, context=["n1rms"]))


def test_default_config_names_labels_the_roster_produces():
    cfg = analyze.load_config(analyze.DEFAULT_CONFIG)
    assert "d3d_tearing_onset_cnn1d/tm_prob" in cfg.labels
    assert cfg.slugs == ("d3d_tearing_onset_cnn1d",)


def _labels_file(tmp_path, task, values, valid):
    t = 0.025 * np.arange(len(values))
    p = np.asarray(values, dtype=float)
    spec = LabelSpec(
        name="lab", task=task, activation="sigmoid" if task == "binary" else "none",
        units="", classes=("no", "yes") if task == "binary" else (), slug="m",
        card_id="x/m", time_step_ms=25.0, ensemble_n=2, artifact_sha256="abc",
    )
    path = tmp_path / "1_labels.h5"
    write_labels(
        path, 1, t, {"lab": Decoded(mean=p, lo=p - 0.05, hi=p + 0.05)}, (spec,),
        np.asarray(valid, bool), run_id="r", features_sha256="f" * 64,
    )
    return path


def test_summarize_label_reports_peak_first_crossing_and_validity(tmp_path):
    path = _labels_file(tmp_path, "binary", [0.1, 0.2, 0.6, 0.9, 0.7, 0.3], [1, 1, 1, 1, 0, 0])
    row = {"invalid_reasons": {"kappa value": 2}, "resolvers": {"kappa": "fdp"},
           "missing_inputs": ["ech_rho"]}
    got = analyze.summarize_label(path, "m", "lab", threshold=0.5, infer_row=row)
    assert got["card_id"] == "x/m" and got["task"] == "binary"
    assert got["time_step_ms"] == 25.0 and got["artifact_sha256"] == "abc"
    assert got["n_rows"] == 6 and got["n_valid"] == 4
    assert got["valid_fraction"] == pytest.approx(4 / 6)
    assert got["max"] == pytest.approx(0.9) and got["t_at_max"] == pytest.approx(0.075)
    assert got["first_above_threshold"] == pytest.approx(0.05) and got["threshold"] == 0.5
    assert got["invalid_reasons"] == {"kappa value": 2}
    assert got["resolvers"] == {"kappa": "fdp"} and got["missing_inputs"] == ["ech_rho"]


def test_summarize_label_has_no_threshold_for_a_regression(tmp_path):
    path = _labels_file(tmp_path, "regression", [1.0, 2.0, 3.0], [1, 1, 1])
    got = analyze.summarize_label(path, "m", "lab", threshold=0.5, infer_row={})
    assert got["max"] == pytest.approx(3.0)
    assert got["threshold"] is None and got["first_above_threshold"] is None
    assert got["invalid_reasons"] == {} and got["resolvers"] == {}


def test_analyze_stage_writes_a_summary_and_a_figure_per_shot(wired, tmp_path):
    out = tmp_path / "analysis"
    argv = _argv(wired, _cfg(tmp_path), out, shots=("190000", "190001"))
    assert run.main(argv) == 0
    for shot in (190000, 190001):
        summary = json.loads((out / str(shot) / f"{shot}_analysis.json").read_text())
        assert summary["shot"] == shot
        assert summary["config"]["labels"] == [f"{SLUG}/score"]
        lab = summary["labels"][f"{SLUG}/score"]
        assert lab["n_rows"] == 240 and 0.0 <= lab["valid_fraction"] <= 1.0
        assert summary["labels_file"].endswith(f"{shot}_labels.h5")
        assert (wired["root"] / "labels" / f"{shot}_labels.h5").exists()
        png = out / str(shot) / f"{shot}_labels.png"
        assert png.exists() and png.stat().st_size > 10_000
    # the run is recorded like every other stage
    assert any(d.name.startswith("analyze") for d in (wired["root"] / "runs").iterdir())


def test_analyze_does_not_take_models_and_other_stages_require_it(wired, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run.main(_argv(wired, _cfg(tmp_path), tmp_path, "--models", SLUG))
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run.main(["features", "--shots", "190000", "--root", str(wired["root"])])
    assert exc.value.code == 2
