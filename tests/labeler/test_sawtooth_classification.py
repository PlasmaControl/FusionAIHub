"""Three-regime diagnostics retain class collapse and shot-level uncertainty."""

import numpy as np
import pytest

from labeler.sawtooth import metrics


def test_classification_exposes_zero_recall_and_fixed_class_macro_f1():
    cells = np.array([[8, 1, 1], [2, 0, 0], [1, 1, 6]])
    actual = metrics.classification_metrics(cells)
    assert actual["confusion"] == cells.tolist()
    assert actual["per_class_recall"] == [0.8, 0.0, 0.75]
    assert actual["window_accuracy"] == 0.7
    assert actual["macro_f1"] == pytest.approx((16 / 21 + 0 + 0.8) / 3)
    assert actual["class_support"] == [10, 2, 8]
    assert actual["predicted_class_counts"] == [11, 2, 7]
    assert actual["per_class_precision"] == [
        pytest.approx(8 / 11),
        0.0,
        pytest.approx(6 / 7),
    ]
    assert actual["class_shares"] == [0.5, 0.1, 0.4]
    assert actual["predicted_class_shares"] == [
        pytest.approx(11 / 20),
        pytest.approx(2 / 20),
        pytest.approx(7 / 20),
    ]


def test_classification_never_predicted_class_has_undefined_precision():
    result = metrics.classification_metrics(np.array([[3, 0, 0], [1, 0, 0], [2, 0, 0]]))
    assert result["per_class_precision"] == [0.5, None, None]
    assert result["predicted_class_shares"] == [1.0, 0.0, 0.0]
    assert result["class_shares"] == [0.5, 1 / 6, 1 / 3]


def test_classification_bootstraps_whole_shots_and_fitted_majority():
    cells = np.array([[8, 1, 1], [2, 0, 0], [1, 1, 6]])
    majority = np.array([[10, 0, 0], [2, 0, 0], [8, 0, 0]])
    rows = [
        {"shot": shot, "cells": cells, "majority_cells": majority} for shot in (11, 22)
    ]
    actual = metrics.bootstrap_classification(rows, replicates=20, seed=3)
    assert actual["shot_ids"] == [11, 22]
    assert actual["ci95"]["window_accuracy"] == [0.7, 0.7]
    assert actual["ci95"]["per_class_precision"][0] == [
        pytest.approx(8 / 11),
        pytest.approx(8 / 11),
    ]
    assert actual["ci95"]["per_class_precision"][1] == [0.0, 0.0]
    assert actual["majority_baseline"]["window_accuracy"] == 0.5
    assert actual["majority_baseline"]["macro_f1"] == pytest.approx(2 / 9)
    assert actual["majority_baseline"]["ci95"]["window_accuracy"] == [0.5, 0.5]


def test_empty_classification_remains_undefined():
    result = metrics.classification_metrics(np.zeros((3, 3), dtype=int))
    assert result["window_accuracy"] is None
    assert result["macro_f1"] is None
    assert result["per_class_recall"] == [None, None, None]


def test_presence_only_baseline_has_no_crash_score_or_interval():
    histogram = metrics.score_histogram([True, False], [1, 1])
    result = metrics.aggregate(
        [
            {
                "shot": 12,
                "cells": None,
                "histogram": histogram,
                "presence_cells": [1, 1, 0, 0],
            }
        ],
        replicates=10,
    )
    assert result["crash"] is None
    assert result["crash_cells"] is None
    assert result["ci95"]["crash_f1"] is None
    assert result["presence"]["f1"] == pytest.approx(2 / 3)


def test_paired_bootstrap_matches_by_shot_and_preserves_equal_models():
    histogram = metrics.score_histogram([True, False], [0.9, 0.1])
    rows = [
        {
            "shot": shot,
            "cells": [1, 0, 0],
            "histogram": histogram,
            "presence_cells": [1, 0, 0, 1],
        }
        for shot in (12, 13)
    ]
    result = metrics.paired_bootstrap(rows, rows[::-1], replicates=20, seed=3)
    assert result["difference"]["crash_f1"] == 0
    assert result["ci95"]["crash_f1"] == [0, 0]
    assert result["ci95"]["presence_auroc"] == [0, 0]
    assert result["shot_ids"] == [12, 13]


def test_paired_bootstrap_rejects_different_shot_populations():
    with pytest.raises(ValueError, match="same unique shots"):
        metrics.paired_bootstrap([{"shot": 1}], [{"shot": 2}], replicates=10)


def test_paired_bootstrap_names_its_direction():
    rows = [
        {
            "shot": 1,
            "cells": [1, 1, 0],
            "histogram": np.zeros((2, 512)),
            "presence_cells": [1, 1, 0, 0],
        }
    ]
    default = metrics.paired_bootstrap(rows, rows, replicates=5)
    assert default["direction"] == "model minus derivative-only baseline"
    named = metrics.paired_bootstrap(rows, rows, replicates=5, direction="a minus b")
    assert named["direction"] == "a minus b"


def test_presence_only_paired_comparison_keeps_crash_undefined():
    always = [
        {
            "shot": 12,
            "cells": None,
            "histogram": metrics.score_histogram([True, False], [1, 1]),
            "presence_cells": [1, 1, 0, 0],
        }
    ]
    derivative = [
        {
            "shot": 12,
            "cells": [1, 0, 0],
            "histogram": metrics.score_histogram([True, False], [1, 0]),
            "presence_cells": [1, 0, 0, 1],
        }
    ]
    result = metrics.paired_bootstrap(always, derivative, replicates=10)
    assert result["difference"]["crash_f1"] is None
    assert result["ci95"]["crash_f1"] is None
    assert result["difference"]["presence_f1"] == pytest.approx(-1 / 3)
    assert result["ci95"]["presence_auroc"] == [-0.5, -0.5]


def test_three_class_abstentions_count_as_errors_on_original_windows():
    cells = np.diag([2, 1, 1])
    result = metrics.classification_metrics(cells, unclassified=[0, 1, 1])
    assert result["windows"] == 6
    assert result["class_support"] == [2, 2, 2]
    assert result["unclassified_by_true_class"] == [0, 1, 1]
    assert result["per_class_recall"] == [1, 0.5, 0.5]
    assert result["window_accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(7 / 9)


def test_three_class_abstentions_survive_shot_bootstrap():
    rows = [
        {
            "shot": 12,
            "cells": np.diag([2, 1, 1]),
            "unclassified": [0, 1, 1],
            "majority_cells": np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0]]),
        }
    ]
    result = metrics.bootstrap_classification(rows, replicates=10)
    assert result["windows"] == 6
    assert result["ci95"]["window_accuracy"] == [2 / 3, 2 / 3]
