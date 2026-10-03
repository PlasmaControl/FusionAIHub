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


def test_classification_bootstraps_whole_shots_and_fitted_majority():
    cells = np.array([[8, 1, 1], [2, 0, 0], [1, 1, 6]])
    majority = np.array([[10, 0, 0], [2, 0, 0], [8, 0, 0]])
    rows = [
        {"shot": shot, "cells": cells, "majority_cells": majority} for shot in (11, 22)
    ]
    actual = metrics.bootstrap_classification(rows, replicates=20, seed=3)
    assert actual["shot_ids"] == [11, 22]
    assert actual["ci95"]["window_accuracy"] == [0.7, 0.7]
    assert actual["majority_baseline"]["window_accuracy"] == 0.5
    assert actual["majority_baseline"]["macro_f1"] == pytest.approx(2 / 9)
    assert actual["majority_baseline"]["ci95"]["window_accuracy"] == [0.5, 0.5]


def test_empty_classification_remains_undefined():
    result = metrics.classification_metrics(np.zeros((3, 3), dtype=int))
    assert result["window_accuracy"] is None
    assert result["macro_f1"] is None
    assert result["per_class_recall"] == [None, None, None]
