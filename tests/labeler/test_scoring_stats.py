"""Rates, kappas, weights and the stratified bootstrap, against hand values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from labeler.scoring.frames import PRESENT, Assessment, frame_counts
from labeler.scoring.stats import (
    cohen_cells,
    cohen_kappa,
    difference,
    estimate,
    f1,
    fleiss_cells,
    fleiss_kappa,
    median_estimate,
    precision,
    recall,
    replicate_weights,
    stratum_weights,
    weighted_median,
)


def test_rates_by_hand_and_undefined_as_nan():
    cells = [2, 2, 2, 2]
    assert (precision(cells), recall(cells), f1(cells)) == (0.5, 0.5, 0.5)
    assert math.isnan(precision([0, 0, 3]))
    assert recall([0, 0, 3]) == 0.0
    assert np.allclose(f1([[1, 0, 0], [1, 1, 0]]), [1.0, 2 / 3])


def test_cohen_kappa_by_hand():
    # po 0.85; A present 0.5, B present 0.55; chance 0.5; kappa 0.7
    assert cohen_kappa([40, 10, 5, 45]) == pytest.approx(0.7)
    frames = np.array([[0, 0], [0, 1], [1, 1], [1, 1]])
    assert cohen_cells(frames).tolist() == [1, 1, 0, 2]


def test_fleiss_kappa_by_hand():
    frames = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1]])
    cells = fleiss_cells(frames)
    assert cells.tolist() == [4, 16, 6]
    # observed 16/24; p = 6/12; chance 0.5; kappa 1/3
    assert fleiss_kappa(cells, 3) == pytest.approx(1 / 3)


def test_perfect_agreement_on_one_class_is_undefined():
    assert math.isnan(cohen_kappa([10, 0, 0, 0]))


def test_stratum_weights_are_population_over_sample():
    assert stratum_weights(["a", "a", "b"], {"a": 10, "b": 5}).tolist() == [5, 5, 5]
    with pytest.raises(ValueError, match="no population count"):
        stratum_weights(["a", "c"], {"a": 1})


def test_replicates_resample_within_each_stratum():
    strata = ["a", "a", "b", "b", "b"]
    draws = replicate_weights(strata, np.ones(5), n=50, seed=1)
    assert draws.shape == (50, 5)
    assert (draws[:, :2].sum(axis=1) == 2).all()
    assert (draws[:, 2:].sum(axis=1) == 3).all()
    assert np.array_equal(draws, replicate_weights(strata, np.ones(5), n=50, seed=1))


def test_tuple_strata_are_one_stratum_each():
    strata = [("L", 2023), ("L", 2023), ("G", 2024)]
    draws = replicate_weights(strata, [1.0, 1.0, 4.0], n=10)
    assert (draws[:, 2] == 4.0).all()


def test_weighted_estimate_by_hand():
    cells = [[1, 0, 0], [0, 1, 0]]  # one true positive, one false positive
    got = estimate(cells, ["a", "b"], [3.0, 1.0], precision)
    assert got.value == pytest.approx(0.75)
    assert got.low == got.high == pytest.approx(0.75)  # one shot per stratum
    assert got.undefined == 0


def test_the_interval_brackets_the_value_when_shots_vary():
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 20, size=(40, 3))
    got = estimate(cells, ["a"] * 40, np.ones(40), f1, n=500)
    assert got.low <= got.value <= got.high and got.low < got.high


def test_paired_difference_of_a_method_with_itself_is_zero():
    cells = np.random.default_rng(2).integers(0, 9, size=(30, 3))
    got = difference(cells, cells, ["a"] * 15 + ["b"] * 15, np.ones(30), f1, n=200)
    assert got.value == got.low == got.high == 0.0


def test_weighted_median_is_the_lower_median():
    assert weighted_median([1, 2, 3], [1, 1, 5]) == 3
    assert weighted_median([4, 1, 3, 2], [1, 1, 1, 1]) == 2
    assert math.isnan(weighted_median([], []))


def test_median_estimate_follows_the_owners_weights():
    got = median_estimate([1.0, 9.0], [0, 1], ["a", "b"], [10.0, 1.0], n=100)
    assert got.value == 1.0 and got.low == got.high == 1.0


def test_from_frames_to_a_weighted_precision():
    reference = Assessment((0, 100), ((0, 50, PRESENT),))
    shots = [
        Assessment((0, 100), ((0, 50, PRESENT),)),  # 5 tp
        Assessment((0, 100), ((0, 100, PRESENT),)),  # 5 tp, 5 fp
    ]
    cells = np.stack([frame_counts(reference, s).cells() for s in shots])
    weights = stratum_weights(["a", "b"], {"a": 2, "b": 6})
    got = estimate(cells, ["a", "b"], weights, precision)
    assert got.value == pytest.approx((2 * 5 + 6 * 5) / (2 * 5 + 6 * 10))
    assert got.as_json()["value"] == pytest.approx(got.value)
