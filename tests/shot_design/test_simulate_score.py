"""Ensemble scores: fair CRPS, spread against error, an edit against the noise floor."""

import itertools

import numpy as np
import pytest

from shot_design.simulate import score


def test_crps_is_the_pairwise_formula():
    rng = np.random.default_rng(0)
    members, obs = rng.normal(size=(5, 3, 2)), rng.normal(size=(3, 2))
    m = members.shape[0]
    pairs = sum(np.abs(members[i] - members[j]) for i, j in itertools.combinations(range(m), 2))
    expected = np.abs(members - obs).mean(axis=0) - pairs / (m * (m - 1))
    np.testing.assert_allclose(score.crps(members, obs), expected)


def test_crps_of_one_member_is_its_absolute_error():
    np.testing.assert_allclose(score.crps(np.array([[1.0, -2.0]]), np.zeros(2)), [1.0, 2.0])


def test_crps_is_unbiased_for_the_ensembles_distribution():
    # For X ~ N(0, 1) and y = 0 the CRPS is (sqrt(2) - 1) / sqrt(pi), whatever the ensemble
    # size; the plain ensemble CRPS would be biased up for small M.
    rng = np.random.default_rng(1)
    exact = (np.sqrt(2) - 1) / np.sqrt(np.pi)
    for m in (2, 4, 16):
        got = score.crps(rng.normal(size=(m, 200_000)), np.zeros(200_000)).mean()
        assert got == pytest.approx(exact, abs=0.005)


def test_a_reliable_ensemble_has_spread_equal_to_error():
    rng = np.random.default_rng(2)
    centre = rng.normal(size=(50, 40))
    members = centre + rng.normal(size=(8, 50, 40))
    obs = centre + rng.normal(size=(50, 40))  # the truth is one more draw
    assert score.spread_error(members, obs) == pytest.approx(1.0, abs=0.05)
    assert score.spread_error(centre + 0.3 * (members - centre), obs) < 0.5


def test_a_perfect_ensemble_mean_has_no_spread_error_ratio():
    members = np.stack([np.ones((4, 2)), -np.ones((4, 2))])
    assert score.spread_error(members, np.zeros((4, 2))) is None


def _arms(gt, rng, edit=0.0, noise=0.1, m=6):
    def arm(offset=0.0):
        return gt + offset + rng.normal(0.0, noise, (m, *gt.shape))

    real = arm()
    return {"real": real, "proposed": real + edit, "null": arm()}


def test_modality_scores_compare_the_ensemble_with_persistence():
    rng = np.random.default_rng(3)
    gt = np.repeat(np.linspace(0.0, 2.0, 30)[:, None], 2, axis=1)
    s = score.modality_scores(gt, _arms(gt, rng), k0=10)
    assert s["skill"] > 0.5  # a close ensemble beats holding frame 9 over a ramp
    assert s["nrmse"]["real"] < s["nrmse"]["persistence"]
    assert s["crps"]["real"] < s["crps"]["persistence"]
    assert s["nrmse"]["seed_mean"] > s["nrmse"]["persistence"]


def test_an_edit_is_resolved_only_well_beyond_the_null_arms_movement():
    rng = np.random.default_rng(4)
    gt = np.zeros((20, 3))
    big = score.modality_scores(gt, _arms(gt, rng, edit=1.0), k0=5)
    assert big["resolved"] and big["effect_to_noise"] > score.RESOLVED
    small = score.modality_scores(gt, _arms(gt, rng, edit=0.01), k0=5)
    assert not small["resolved"]
    assert small["effect"] == pytest.approx(0.01)


def test_without_proposed_and_null_arms_there_is_no_edit_to_score():
    rng = np.random.default_rng(5)
    gt = rng.normal(size=(12, 2))
    s = score.modality_scores(gt, {"real": _arms(gt, rng)["real"]}, k0=4)
    assert "effect" not in s and "resolved" not in s


def test_one_member_has_no_spread():
    gt = np.linspace(0.0, 1.0, 12)[:, None]
    s = score.modality_scores(gt, {"real": gt[None] + 0.1}, k0=4)
    assert s["spread_error"] is None and s["crps"]["real"] == pytest.approx(0.1)
