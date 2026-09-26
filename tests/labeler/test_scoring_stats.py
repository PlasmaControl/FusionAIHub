"""Rates, kappas, weights and the stratified bootstrap, against hand values."""

from __future__ import annotations

import json
import math
from itertools import product
from time import perf_counter

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


@pytest.mark.parametrize(
    "strata, weights",
    [
        (["a"], [1, 1]),
        (["a", "b"], [1]),
        (["a", "b"], [2, -1]),
        (["a", "b"], [1, np.nan]),
        (["a", "b"], [1, np.inf]),
        (["a", "b"], [[1], [1]]),
    ],
)
def test_replicate_weights_refuses_misalignment_and_invalid_weights(strata, weights):
    with pytest.raises(ValueError):
        replicate_weights(strata, weights, n=10)


@pytest.mark.parametrize("n", [0, -1, 1.5, np.nan, np.inf, "2", True])
def test_replicate_weights_requires_a_positive_whole_count(n):
    with pytest.raises(ValueError):
        replicate_weights(["a"], [1], n=n)


def test_replicate_weights_accepts_whole_float_counts_and_zero_weights():
    assert replicate_weights(["a"], [0], n=2.0).tolist() == [[0], [0]]


@pytest.mark.parametrize(
    "cells, strata, weights",
    [
        ([[1, 0, 0], [0, 1, 0]], ["a"], [1, 1]),
        ([[1, 0, 0], [0, 1, 0]], ["a", "b"], [2, -1]),
        ([[1, 0, 0], [0, 1, 0]], ["a", "b"], [1]),
        ([1, 0, 0], ["a"], [1]),
        ([[[1, 0, 0]]], ["a"], [1]),
    ],
)
def test_estimate_refuses_cells_not_aligned_with_shots(cells, strata, weights):
    with pytest.raises(ValueError):
        estimate(cells, strata, weights, precision, n=100)


@pytest.mark.parametrize("level", [0, 1, -0.1, 1.1, np.nan, np.inf])
@pytest.mark.parametrize("kind", ["estimate", "difference", "median"])
def test_all_intervals_require_a_level_strictly_between_zero_and_one(level, kind):
    with pytest.raises(ValueError):
        if kind == "estimate":
            estimate([[1, 0, 0]], ["a"], [1], f1, n=10, level=level)
        elif kind == "difference":
            difference([[1, 0, 0]], [[1, 1, 0]], ["a"], [1], f1, n=10, level=level)
        else:
            median_estimate([1], [0], ["a"], [1], n=10, level=level)


@pytest.mark.parametrize(
    "a, b",
    [
        ([[1, 0, 0]], [[1, 0, 0, 0]]),
        ([[1, 0, 0]], [[1, 0, 0], [0, 1, 0]]),
        ([1, 0, 0], [1, 0, 0]),
    ],
)
def test_difference_requires_two_dimensional_cells_of_identical_shape(a, b):
    with pytest.raises(ValueError):
        difference(a, b, ["a"], [1], f1, n=10)


@pytest.mark.parametrize(
    "values, owners",
    [
        ([1, 9], [0]),
        ([1], [0, 1]),
        ([1, 9], [0.9, 1.9]),
        ([1], [-1]),
        ([1], [2]),
        ([1], [np.nan]),
        ([1], [np.inf]),
        ([[1, 9]], [[0, 1]]),
    ],
)
def test_median_estimate_requires_aligned_values_and_integral_shot_owners(
    values, owners
):
    with pytest.raises(ValueError):
        median_estimate(values, owners, ["a", "b"], [10, 1], n=10)


def test_median_estimate_checks_shot_alignment():
    with pytest.raises(ValueError):
        median_estimate([1, 9], [0, 1], ["a"], [10, 1], n=10)


def test_stratum_weights_names_every_unscored_population_stratum():
    with pytest.raises(ValueError) as exc:
        stratum_weights(["L"] * 20 + ["R"] * 20, {"L": 231, "G": 573, "R": 4081})
    assert "G" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        stratum_weights(["L"], {"L": 231, "G": 573, "R": 4081})
    assert "G" in str(exc.value) and "R" in str(exc.value)


@pytest.mark.parametrize("population", [{"a": -1}, {"a": np.nan}, {"a": np.inf}])
def test_stratum_weights_refuses_negative_or_nonfinite_population(population):
    with pytest.raises(ValueError, match="a"):
        stratum_weights(["a"], population)


def test_an_empty_zero_population_stratum_is_allowed():
    assert stratum_weights(["L2025"], {"L2025": 18, "G2025": 0}).tolist() == [18]


@pytest.mark.parametrize("kind", ["estimate", "difference", "median"])
def test_estimate_records_settings_in_strict_json(kind):
    options = {"n": 37, "seed": 19, "level": 0.8}
    if kind == "estimate":
        got = estimate([[1, 0, 0]], ["a"], [1], f1, **options)
    elif kind == "difference":
        got = difference([[1, 0, 0]], [[1, 1, 0]], ["a"], [1], f1, **options)
    else:
        got = median_estimate([2], [0], ["a"], [1], **options)
    assert (got.replicates, got.seed, got.level) == (37, 19, 0.8)
    data = json.loads(json.dumps(got.as_json(), allow_nan=False))
    assert data == {
        "value": got.value,
        "low": got.low,
        "high": got.high,
        "undefined_replicates": 0,
        "replicates": 37,
        "seed": 19,
        "level": 0.8,
    }


@pytest.mark.parametrize("readers", [2, 3])
def test_random_frame_kappas_match_textbook_agreement_and_marginals(readers):
    rng = np.random.default_rng(701)
    frames = (
        rng.random((400, readers)) < np.array([0.18, 0.61, 0.83])[:readers]
    ).astype(int)
    if readers == 2:
        observed = np.mean(frames[:, 0] == frames[:, 1])
        p_a, p_b = frames.mean(axis=0)
        chance = p_a * p_b + (1 - p_a) * (1 - p_b)
        expected = (observed - chance) / (1 - chance)
        assert cohen_kappa(cohen_cells(frames)) == pytest.approx(
            expected, rel=0, abs=1e-12
        )
    present = frames.sum(axis=1)
    absent = readers - present
    per_frame = (present * (present - 1) + absent * (absent - 1)) / (
        readers * (readers - 1)
    )
    prevalence = frames.mean()
    chance = prevalence**2 + (1 - prevalence) ** 2
    expected = (per_frame.mean() - chance) / (1 - chance)
    assert fleiss_kappa(fleiss_cells(frames), readers) == pytest.approx(
        expected, rel=0, abs=1e-12
    )


def _manual_f1(totals):
    tp, fp, fn = totals[:3]
    return 2 * tp / (2 * tp + fp + fn)


def _manual_median(values, weights):
    halfway = sum(weights) / 2
    running = 0
    for value, weight in sorted(zip(values, weights)):
        running += weight
        if running >= halfway:
            return value
    raise AssertionError("the enumerable cohort always has positive event weight")


@pytest.mark.parametrize("kind", ["f1", "difference", "median"])
def test_bootstrap_endpoints_equal_the_extremes_of_all_16_ordered_draws(kind):
    strata, weights = ["a", "a", "b", "b"], np.array([3.0, 3.0, 1.0, 1.0])
    a = np.array([[4, 0, 0], [0, 2, 1], [1, 0, 2], [0, 1, 0]])
    b = np.array([[2, 1, 2], [1, 1, 0], [0, 1, 3], [1, 0, 0]])
    values, owners = [1, 2, 6, 9, 12], [0, 0, 1, 2, 3]

    def answer(w):
        if kind == "f1":
            return _manual_f1(w @ a)
        if kind == "difference":
            return _manual_f1(w @ a) - _manual_f1(w @ b)
        return _manual_median(values, [w[i] for i in owners])

    exact = []
    for first, second in product(product((0, 1), repeat=2), product((2, 3), repeat=2)):
        counts = [sum(pick == i for pick in first + second) for i in range(4)]
        exact.append(answer(weights * counts))
    # Each ordered draw has mass 1/16 > 2.5%; both tails reach the extremes.
    assert len(exact) == 16 and min(exact) < max(exact)
    options = {"n": 4000, "seed": 78}
    if kind == "f1":
        got = estimate(a, strata, weights, f1, **options)
    elif kind == "difference":
        got = difference(a, b, strata, weights, f1, **options)
    else:
        got = median_estimate(values, owners, strata, weights, **options)
    assert [got.value, got.low, got.high] == pytest.approx(
        [answer(weights), min(exact), max(exact)], rel=0, abs=1e-12
    )
    assert got.undefined == 0


def test_pairing_detects_a_real_loss_of_true_positives_with_a_narrower_interval():
    rng = np.random.default_rng(402)
    a = rng.integers(1, 100, size=(60, 3))
    b = a.copy()
    moved = np.maximum(1, rng.binomial(a[:, 0], 0.1))
    b[:, 0] -= moved
    b[:, 2] += moved
    strata = ["a"] * 20 + ["b"] * 40
    weights = np.array([2.0] * 20 + [8.0] * 40)
    paired = difference(a, b, strata, weights, f1, n=1000, seed=81)
    own_a = estimate(a, strata, weights, f1, n=1000, seed=81)
    own_b = estimate(b, strata, weights, f1, n=1000, seed=82)
    independent_point = _manual_f1(weights @ a) - _manual_f1(weights @ b)
    assert paired.value == pytest.approx(independent_point, rel=0, abs=1e-12)
    assert 0 < paired.low < paired.high
    assert paired.high - paired.low < math.hypot(
        own_a.high - own_a.low, own_b.high - own_b.low
    )


def test_undefined_replicates_are_counted_from_the_draws_and_serialize_as_null():
    cells = np.array([[1, 0, 0], [0, 0, 0]])
    strata, weights, n, seed = ["a", "a"], [1, 1], 1000, 12
    draws = replicate_weights(strata, weights, n=n, seed=seed)
    denominators = draws @ (cells[:, 0] + cells[:, 1])
    expected = int(np.count_nonzero(denominators == 0))
    got = estimate(cells, strata, weights, precision, n=n, seed=seed)
    assert 0 < expected < n
    assert got.undefined == expected
    assert got.value == got.low == got.high == 1.0
    empty = estimate(np.zeros((2, 3)), strata, weights, precision, n=n, seed=seed)
    data = json.loads(json.dumps(empty.as_json(), allow_nan=False))
    assert data["value"] is data["low"] is data["high"] is None
    assert data["undefined_replicates"] == n


def test_cohort_design_coverage_and_bias(capsys):
    started = perf_counter()
    rng = np.random.default_rng(2026)
    design = {
        "G2021": (120, 21),
        "G2022": (340, 59),
        "G2023": (112, 19),
        "G2024": (1, 1),
        "L2021": (29, 25),
        "L2022": (73, 63),
        "L2023": (59, 51),
        "L2024": (52, 45),
        "L2025": (18, 16),
        "R2021": (332, 16),
        "R2022": (1154, 57),
        "R2023": (680, 33),
        "R2024": (1065, 52),
        "R2025": (850, 42),
    }
    populations = {}
    for h, (size, _) in design.items():
        prevalence, quality = {"L": (0.35, 0.85), "G": (0.25, 0.75), "R": (0.1, 0.55)}[
            h[0]
        ]
        rows = []
        for _ in range(size):
            frames = rng.integers(200, 800)
            positive = rng.binomial(frames, rng.beta(2, 2 / prevalence - 2))
            q = rng.beta(8 * quality, 8 * (1 - quality))
            tp = rng.binomial(positive, q)
            fp = rng.binomial(frames - positive, (1 - q) * 0.2)
            rows.append((tp, fp, positive - tp))
        populations[h] = np.array(rows, dtype=float)
    totals = np.concatenate(list(populations.values())).sum(axis=0)
    tp, fp, fn = totals
    truth = np.array([tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)])
    strata = [h for h, (_, size) in design.items() for _ in range(size)]
    weights = stratum_weights(strata, {h: size for h, (size, _) in design.items()})
    hits, errors = np.zeros(3, dtype=int), np.zeros((200, 3))
    for sample in range(200):
        cells = np.concatenate(
            [
                populations[h][rng.choice(size, n, replace=False)]
                for h, (size, n) in design.items()
            ]
        )
        for j, metric in enumerate((precision, recall, f1)):
            got = estimate(
                cells, strata, weights, metric, n=1000, seed=int(rng.integers(1 << 30))
            )
            hits[j] += got.low <= truth[j] <= got.high
            errors[sample, j] = got.value - truth[j]
    coverage, bias = hits / 200, errors.mean(axis=0)
    elapsed = perf_counter() - started
    with capsys.disabled():
        print(
            f"\nCohort P/R/F1: coverage={coverage.tolist()}, "
            f"bias={bias.tolist()}, runtime={elapsed:.3f}s"
        )
    assert np.all((0.92 <= coverage) & (coverage <= 0.98))
    assert np.all(np.abs(bias) < 0.01)
    assert elapsed < 20


def test_two_stage_frozen_g_weights_and_group_resampling():
    from labeler.scoring.stats import two_stage_weights

    first = [1, 124 / 21] + [345 / 59] * 4 + [113 / 19] * 4 + [2, 4]
    groups = ["G"] * 10 + ["L"] * 2
    weights = two_stage_weights(first, groups, {"G": 100, "L": 4})
    assert weights[:2] == pytest.approx([10, 124 / 21 * 10])
    draws = replicate_weights(groups, weights, n=50, seed=7) / weights
    assert (draws[:, :10].sum(axis=1) == 10).all()
    assert (draws[:, 10:].sum(axis=1) == 2).all()


def test_two_stage_exact_unbiased_total_by_enumeration():
    from fractions import Fraction
    from itertools import combinations

    from labeler.scoring.stats import two_stage_weights

    # Values differ between cells, so group-only weighting is biased.
    y = [1, 2, 3, 4, 5, 10, 20, 30]
    first = [Fraction(5, 3)] * 5 + [Fraction(3)] * 3
    totals, group_totals = [], []
    for a, b in product(combinations(range(5), 3), combinations(range(5, 8), 1)):
        for blind in combinations(a + b, 2):
            exact = [first[i] * Fraction(4, 2) for i in blind]
            actual = two_stage_weights(
                [float(first[i]) for i in blind], ["g"] * 2, {"g": 4}
            )
            assert actual == pytest.approx([float(w) for w in exact])
            totals.append(sum(w * y[i] for w, i in zip(exact, blind)))
            group_totals.append(sum(Fraction(8, 2) * y[i] for i in blind))
    assert len(totals) == 180
    assert sum(totals) / len(totals) == sum(y)
    assert sum(group_totals) / len(group_totals) != sum(y)


@pytest.mark.parametrize(
    "first, counts",
    [
        ([1, 1], {}),
        ([0, 1], {"g": 4}),
        ([-1, 1], {"g": 4}),
        ([np.nan, 1], {"g": 4}),
        ([np.inf, 1], {"g": 4}),
        ([1, 1], {"g": 2.5}),
        ([1, 1], {"g": np.nan}),
        ([1, 1], {"g": np.inf}),
        ([1, 1], {"g": True}),
        ([1, 1], {"g": 1}),
    ],
)
def test_two_stage_refuses_invalid_group_inputs(first, counts):
    from labeler.scoring.stats import two_stage_weights

    with pytest.raises(ValueError, match="g"):
        two_stage_weights(first, ["g", "g"], counts)
