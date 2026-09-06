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
    assert "d3d_tearing_time_to_event_dsm/tm_risk_1s" in cfg.labels
    assert cfg.labels[2] == "d3d_tearing_time_to_event_dsm/tm_time_p50"
    assert "d3d_tearing_time_to_event_dsm_continued/tm_risk_1s" in cfg.labels
    assert cfg.labels[-3:] == (
        "d3d_ae_activity_seldnet/ae_active",
        "d3d_ae_activity_seldnet/ae_frequency",
        "d3d_elm_time_to_event_dsm/elm_risk_20ms",
    )
    assert cfg.slugs == (
        "d3d_tearing_onset_cnn1d",
        "d3d_tearing_time_to_event_dsm",
        "d3d_tearing_time_to_event_dsm_continued",
        "d3d_ae_activity_seldnet",
        "d3d_elm_time_to_event_dsm",
    )


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


def test_config_thresholds_may_be_set_per_label(wired, tmp_path):
    """The survival model's 1 s risk peaks near 0.3 on shots where the CNN's
    probability reaches 0.9: one threshold does not fit two labels. A
    `thresholds` mapping overrides the global value per label."""
    cfg = analyze.load_config(_cfg(tmp_path, threshold=0.5, thresholds={f"{SLUG}/score": 0.2}))
    assert cfg.threshold_for(f"{SLUG}/score") == 0.2
    assert cfg.threshold_for("other/label") == 0.5
    assert cfg.as_dict()["thresholds"] == {f"{SLUG}/score": 0.2}
    with pytest.raises(analyze.ConfigError, match="thresholds"):
        analyze.load_config(_cfg(tmp_path, thresholds={f"{SLUG}/nope": 0.2}))
    with pytest.raises(analyze.ConfigError, match="thresholds"):
        analyze.load_config(_cfg(tmp_path, thresholds={f"{SLUG}/score": 1.5}))


def test_analyze_summary_and_figure_use_the_per_label_threshold(wired, tmp_path):
    out = tmp_path / "analysis"
    cfg = _cfg(tmp_path, thresholds={f"{SLUG}/score": 0.05})
    assert run.main(_argv(wired, cfg, out)) == 0
    summary = json.loads((out / "190000" / "190000_analysis.json").read_text())
    assert summary["labels"][f"{SLUG}/score"]["threshold"] == 0.05


def test_summary_carries_a_truth_block_per_label(wired, tmp_path):
    """A shot with no archived truth still says so, per label, rather than
    leaving the reader to wonder whether it was scored."""
    out = tmp_path / "analysis"
    assert run.main(_argv(wired, _cfg(tmp_path), out)) == 0
    summary = json.loads((out / "190000" / "190000_analysis.json").read_text())
    assert summary["truth"]["available"] is False
    got = summary["labels"][f"{SLUG}/score"]["truth"]
    assert got["scored"] is False and got["reason"]


def test_panels_carry_the_archived_truth_when_there_is_some(tmp_path):
    path = _labels_file(tmp_path, "binary", [0.1, 0.6, 0.9, 0.2], [1, 1, 1, 1])
    # built directly: "m" is a synthetic label file, not a registered model,
    # and load_config would rightly refuse it
    cfg = analyze.AnalysisConfig(labels=("m/lab",), context=(), threshold=0.5)
    truth = {"available": True, "index": np.array([0, 2, 3]),
             "t": np.array([0.0, 0.05, 0.075]),
             "tm_label": np.array([False, True, True]),
             "betan": np.array([1.0, 2.0, 3.0]), "onset_s": 0.05, "n_rows": 3}
    panels = analyze.panels_for(tmp_path / "missing_features.h5", path, cfg, truth=truth)
    panel = panels[0]
    np.testing.assert_allclose(panel["truth_t"], [0.0, 0.05, 0.075])
    np.testing.assert_array_equal(panel["truth_mask"], [False, True, True])
    assert panel["onset_s"] == 0.05
    png = analyze.plot_shot(1, panels, tmp_path / "p.png", title_ids=["x/m"])
    assert png.exists() and png.stat().st_size > 5_000


@pytest.fixture
def band_labels(wired):
    from labelmaker.config import Paths

    paths = Paths(root=wired["root"], corpus=wired["corpus"])
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    values = {"time_p10": np.full(5, 100.0), "time_p50": np.arange(1, 6) * 1000.0,
              "time_p90": np.full(5, 10000.0)}
    specs = tuple(LabelSpec(name=name, task="regression", activation="none", units="ms",
                            classes=(), slug=SLUG, card_id="x/m", time_step_ms=500.0,
                            ensemble_n=1, artifact_sha256="abc") for name in values)
    write_labels(paths.labels_file(190000), 190000, t,
                 {n: Decoded(mean=y, lo=y, hi=y) for n, y in values.items()},
                 specs, np.array([1, 1, 0, 1, 1], bool), run_id="test", features_sha256="abc")
    return paths.labels_file(190000)


@pytest.mark.parametrize("onset", [1.6, None])
def test_band_panel_uses_quantile_siblings_and_pre_onset_truth(
    wired, band_labels, tmp_path, monkeypatch, onset,
):
    cfg = analyze.AnalysisConfig(labels=(f"{SLUG}/time_p50",), context=(), threshold=0.5)
    truth = {"available": True, "onset_s": onset, "t": np.array([0.0, 1.0, 2.0]),
             "tm_label": np.array([0, 0, 1], bool)}
    panel, = analyze.panels_for(tmp_path / "missing.h5", band_labels, cfg, truth=truth)
    assert panel["kind"] == "band" and panel["units"] == "ms"
    np.testing.assert_allclose(panel["lo"], 100.0)
    np.testing.assert_allclose(panel["hi"], 10000.0)
    np.testing.assert_allclose(panel["y"], np.arange(1, 6) * 1000.0)
    np.testing.assert_array_equal(panel["valid"], [1, 1, 0, 1, 1])
    if onset is None:
        assert panel["truth_y"] is None
    else:
        np.testing.assert_allclose(panel["truth_t"], [0.0, 0.5, 1.0, 1.5])
        np.testing.assert_allclose(panel["truth_y"], [1600.0, 1100.0, 600.0, 100.0])
    from matplotlib.figure import Figure

    savefig = Figure.savefig
    drawn = []

    def capture(fig, *args, **kwargs):
        drawn.append(fig.axes[0])
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    png = analyze.plot_shot(190000, [panel], tmp_path / "band.png", title_ids=["x/m"])
    ax, = drawn
    assert ax.get_yscale() == "log" and ax.get_ylabel() == "time_p50 (ms)"
    assert "p10..p90" in ax.get_legend_handles_labels()[1]
    truth_lines = [line for line in ax.lines if line.get_label() == "archived truth"]
    assert len(truth_lines) == (onset is not None)
    if truth_lines:
        assert truth_lines[0].get_color() == analyze._TRUTH
    assert png.exists() and png.stat().st_size > 5000


@pytest.mark.parametrize("onset, expected", [(1.6, 2000.0), (2.0, None), (None, None)])
def test_p50_summary_uses_nearest_grid_row_and_respects_validity(band_labels, onset, expected):
    got = analyze.summarize_label(band_labels, SLUG, "time_p50", threshold=0.5,
                                  infer_row={}, truth={"available": True, "onset_s": onset})
    assert got["p50_at_onset_minus_1s"] == expected


def test_analyze_stage_passes_truth_to_p50_summary(wired, band_labels, tmp_path, monkeypatch):
    from dataclasses import replace

    from labelmaker import validate
    from labelmaker.models import registry
    from labelmaker.models.base import OutputField, OutputSpec

    adapter = registry.load_adapter(SLUG)
    adapter = replace(adapter, output_spec=OutputSpec(fields=tuple(
        OutputField(f"time_p{q}", "regression", column=i, units="ms")
        for i, q in enumerate((10, 50, 90)))))
    monkeypatch.setattr(registry, "load_adapter", lambda slug: adapter)
    monkeypatch.setattr(validate, "archived_truth", lambda *a, **kw: {
        "available": True, "onset_s": 1.6, "t": np.array([0.0, 1.0, 2.0]),
        "tm_label": np.array([0, 0, 1], bool), "n_rows": 3})
    out = tmp_path / "analysis"
    cfg = _cfg(tmp_path, labels=[f"{SLUG}/time_p50"])
    assert run.main(_argv(wired, cfg, out)) == 0
    summary = json.loads((out / "190000" / "190000_analysis.json").read_text())
    assert summary["labels"][f"{SLUG}/time_p50"]["p50_at_onset_minus_1s"] == 2000.0


def test_p50_without_quantile_siblings_remains_an_ordinary_panel(band_labels, tmp_path):
    import h5py

    with h5py.File(band_labels, "a") as f:
        del f[f"{SLUG}/time_p10"]
    cfg = analyze.AnalysisConfig(labels=(f"{SLUG}/time_p50",), context=(), threshold=0.5)
    panel, = analyze.panels_for(tmp_path / "missing.h5", band_labels, cfg)
    assert panel["kind"] == "label"
    got = analyze.summarize_label(band_labels, SLUG, "time_p50", threshold=0.5, infer_row={})
    assert got["p50_at_onset_minus_1s"] is None
