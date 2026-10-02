"""Leakage, weighting, abstention and evaluation regression tests."""

import numpy as np
import pandas as pd
import pytest

from labeler.confinement import train


def test_recompute_balancing_after_feature_exclusions():
    frame = pd.DataFrame(
        {
            "shot": [1, 1, 2, 3, 3, 4],
            "label": [0, 0, 0, 1, 1, 1],
            "dalpha__mean": [1.0, np.nan, 2.0, 3.0, 3.0, np.nan],
            "dalpha__present": [1, 0, 1, 1, 1, 0],
        }
    )
    used, weights = train.training_rows(frame, ["dalpha__mean", "dalpha__present"])
    assert used.index.tolist() == [0, 2, 3, 4]
    assert weights[used.label == 0].sum() == pytest.approx(2.0)
    assert weights[used.label == 1].sum() == pytest.approx(2.0)
    assert weights[used.shot == 1].sum() == weights[used.shot == 2].sum()


def test_all_raw_test_only_reservations_checked_even_outside_split():
    split = pd.DataFrame({"shot": [1, 2, 3], "split": ["train", "val", "test"]})
    train.validate_split(split, {3, 99})
    with pytest.raises(ValueError, match="test.only"):
        train.validate_split(split, {1, 99})


def test_threshold_selected_using_validation_only():
    selected = train.select_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.4, 0.9])
    )
    assert 0.3 < selected <= 0.4


def test_threshold_search_supports_confident_but_miscalibrated_validation():
    threshold = train.select_threshold(np.array([0, 1]), np.array([0.96, 0.98]))
    assert 0.96 < threshold <= 0.98


def test_four_class_shot_bootstrap_keeps_all_class_metrics():
    frame = pd.DataFrame(
        {"shot": np.repeat([1, 2], 4), "regime_label": np.tile(np.arange(4), 2)}
    )
    result = train._multiclass_bootstrap(frame, np.tile(np.eye(4), (2, 1)), 20)
    assert result["metric_95ci"]["f1_WP"] == [1.0, 1.0]


def test_one_class_test_has_null_auroc_and_preserves_denominator():
    result = train.binary_metrics(np.ones(3, dtype=int), np.ones(3), 0.5)
    assert result["auroc"] is None
    assert result["support"] == {"L": 0, "H": 3}
    assert result["per_class"]["L"]["recall"] is None
    assert result["per_class"]["L"]["f1"] is None
    assert result["macro_f1"] is None
    assert result["balanced_accuracy"] is None


def test_forced_bes_absence_clears_values_and_presence():
    frame = pd.DataFrame(
        {"bes__mean": [2.0], "bes__present": [1.0], "dalpha__mean": [3.0]}
    )
    dropped = train.without_bes(frame)
    assert np.isnan(dropped.loc[0, "bes__mean"])
    assert dropped.loc[0, "bes__present"] == 0
    assert dropped.loc[0, "dalpha__mean"] == 3


def test_inference_abstains_on_missing_signals_and_uses_frozen_threshold():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from labeler.confinement import apply

    model = HistGradientBoostingClassifier(
        min_samples_leaf=1, max_iter=2, early_stopping=False, random_state=1
    )
    model.fit(pd.DataFrame({"dalpha__mean": [0.0, 0.0, 1.0, 1.0]}), [0, 0, 1, 1])
    frame = pd.DataFrame(
        {
            "shot": [3, 3],
            "t_start": [0.0, 50.0],
            "t_end": [50.0, 100.0],
            "dalpha__mean": [1.0, np.nan],
        }
    )
    bundle = {
        "model": model,
        "features": ["dalpha__mean"],
        "threshold": 0.99,
        "task": "binary",
    }
    prediction = apply.predict_features(frame, bundle)
    assert prediction.category.tolist() == [0, 3]
    assert np.isnan(prediction.loc[1, "confidence"])


def test_paired_bootstrap_returns_per_class_intervals():
    frame = pd.DataFrame({"shot": [1, 1, 2, 2], "label": [0, 1, 0, 1]})
    result = train._bootstrap(
        frame, {"full": np.array([0.1, 0.9, 0.1, 0.9])}, {"full": 0.5}, repetitions=20
    )
    assert result["metric_95ci"]["full"]["f1_L"] == [1.0, 1.0]
    assert result["metric_95ci"]["full"]["auroc"] == [1.0, 1.0]


def test_evaluation_preserves_completely_unobservable_test_cohort(tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=1, min_samples_leaf=1)
    model.fit(pd.DataFrame({"dalpha__mean": [0.0, 1.0]}), [0, 1])
    frame = pd.DataFrame(
        {
            "shot": [3],
            "split": ["test"],
            "label": [0],
            "regime_label": [0],
            "t_start": [0.0],
            "t_end": [50.0],
            "reason": ["raw_test_only"],
            "dalpha__mean": [np.nan],
        }
    )
    (tmp_path / "training.json").write_text("{}")
    bundle = {
        "model": model,
        "features": ["dalpha__mean"],
        "threshold": 0.5,
        "task": "binary",
    }
    result = train.evaluate(frame, {"full": bundle}, tmp_path, repetitions=20)
    assert result["models"]["full"]["support"]["bins"] == 0
    assert result["models"]["full"]["excluded_missing_support"]["bins"] == 1


def test_fitting_does_not_read_test_features_or_targets(tmp_path, monkeypatch):
    from labeler.confinement.data import feature_columns

    monkeypatch.setattr(
        train,
        "GRID",
        [
            {
                "max_iter": 2,
                "min_samples_leaf": 1,
                "early_stopping": False,
                "random_state": 9,
            }
        ],
    )
    frame = pd.DataFrame(
        {
            "shot": np.repeat([1, 2, 3], 8),
            "split": np.repeat(["train", "val", "test"], 8),
            "label": np.tile([0, 1, 1, 1], 6),
            "regime_label": np.tile(np.arange(4), 6),
            "t_start": np.tile(np.arange(8) * 50.0, 3),
            "t_end": np.tile(np.arange(1, 9) * 50.0, 3),
        }
    )
    features = pd.DataFrame(
        {
            c: np.zeros(24) if c.endswith("__present") else np.full(24, np.nan)
            for c in feature_columns()
        }
    )
    frame = pd.concat([frame, features], axis=1)
    frame["dalpha__mean"] = frame.regime_label.astype(float)
    frame["dalpha__present"] = 1.0
    _, first = train.fit_models(frame, tmp_path / "first", {})
    frame.loc[frame.split == "test", "dalpha__mean"] = 1e30
    frame.loc[frame.split == "test", "label"] = 1
    _, second = train.fit_models(frame, tmp_path / "second", {})
    assert not first["test_opened"]
    assert not (tmp_path / "first" / "predictions.csv").exists()
    for name in ("full", "no_bes", "dalpha_nbi", "four_class"):
        assert (
            first["models"][name]["model_sha256"]
            == second["models"][name]["model_sha256"]
        )
    weights = pd.read_csv(tmp_path / "first" / "model" / "full_train_weights.csv")
    mass = weights.groupby("label").effective_weight.sum()
    assert mass.loc[0] == pytest.approx(mass.loc[1])
    assert first["models"]["full"]["train_feature_finite_counts"]["dalpha__mean"] == 8


def test_training_refuses_outputs_inside_raw_store(tmp_path, monkeypatch):
    from labeler.config import Paths

    paths = Paths(root=tmp_path / "products", corpus=tmp_path / "corpus")
    monkeypatch.setattr(train.Paths, "from_env", lambda: paths)
    with pytest.raises(ValueError, match="raw diagnostic stores"):
        train.run(tmp_path / "missing_inputs", paths.corpus / "trained")
