"""The label model recovers the accuracies of simulated labelling functions."""

import numpy as np
import pytest

from labeler.events.detachment import core
from labeler.events.detachment import label_model as lm

ALLOWED = {"afrac": (1, 2), "prad": (1, 2), "tangtv": (1, 2, 3)}
NAMES = ("afrac", "prad", "tangtv")


def simulate(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    state = rng.choice([1, 2, 3], size=n, p=[0.45, 0.4, 0.15])
    accuracy = {"afrac": 0.8, "prad": 0.65, "tangtv": 0.9}
    propensity = {"afrac": 0.7, "prad": 0.8, "tangtv": 0.6}
    votes = np.full((n, 3), core.ABSTAIN)
    for j, name in enumerate(NAMES):
        allowed = ALLOWED[name]
        speaks = rng.random(n) < propensity[name]
        right = rng.random(n) < accuracy[name]
        for i in np.flatnonzero(speaks):
            truth = state[i] if state[i] in allowed else 2  # marfe reads as detached
            wrong = [s for s in allowed if s != truth]
            votes[i, j] = truth if right[i] else rng.choice(wrong)
    return state, votes


def test_fit_orders_the_labelling_functions_by_accuracy():
    _, votes = simulate()
    model = lm.LabelModel().fit(votes)
    acc = model.accuracies()
    assert acc["tangtv"]["acc_weight"] > acc["afrac"]["acc_weight"]
    assert acc["afrac"]["acc_weight"] > acc["prad"]["acc_weight"] > 0
    assert acc["afrac"]["implied_accuracy"] == pytest.approx(0.8, abs=0.06)
    assert acc["prad"]["implied_accuracy"] == pytest.approx(0.65, abs=0.06)


def test_posterior_beats_every_single_labelling_function_where_all_speak():
    state, votes = simulate(n=12000)
    model = lm.LabelModel().fit(votes)
    post = model.posterior(votes)
    assert post.shape == (len(votes), 3) and np.allclose(post.sum(axis=1), 1.0)
    everyone = (votes != core.ABSTAIN).all(axis=1)
    got = post.argmax(axis=1) + 1
    combined = (got[everyone] == state[everyone]).mean()
    for j in range(3):
        single = (votes[everyone, j] == state[everyone]).mean()
        assert combined >= single - 0.01
    assert combined > (votes[everyone, 0] == state[everyone]).mean() + 0.02


def test_marfe_can_only_come_from_tangtv():
    _, votes = simulate()
    model = lm.LabelModel().fit(votes)
    post = model.posterior(np.array([[1, 1, -1], [2, 2, -1], [-1, -1, 3], [2, 1, -1]]))
    assert (post[:3, 2] < 0.5).all() or post[2, 2] > post[0, 2]
    assert post[2].argmax() == 2
    assert post[0].argmax() == 0 and post[1].argmax() == 1


def test_gradient_matches_finite_differences():
    _, votes = simulate(n=1500)
    model = lm.LabelModel(corr=(("prad", "tangtv"),))
    counts = model.config_counts(votes)
    rng = np.random.default_rng(1)
    theta = rng.uniform(0.1, 1.0, model.n_params)
    _, grad = model.neg_loglik(theta, counts)
    for k in range(model.n_params):
        step = np.zeros(model.n_params)
        step[k] = 1e-5
        numeric = (
            model.neg_loglik(theta + step, counts)[0]
            - model.neg_loglik(theta - step, counts)[0]
        ) / 2e-5
        assert grad[k] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_rule_agreement_conflict_and_absence():
    votes = np.array(
        [[1, 1, -1], [1, 2, -1], [-1, -1, -1], [-1, -1, -1], [3, -1, -1], [-1, 2, 2]]
    )
    valid = np.array(
        [[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=bool
    )
    out = lm.rule(votes, valid)
    assert out.tolist() == [1, 4, 4, 0, 3, 2]


def test_decide_threshold_and_absence():
    post = np.array(
        [[0.9, 0.05, 0.05], [0.5, 0.45, 0.05], [0.3, 0.3, 0.4], [0.9, 0.05, 0.05]]
    )
    assessed = np.array([True, True, False, True])
    has_vote = np.array([True, True, True, False])
    state = lm.decide(post, assessed, has_vote, threshold=0.7)
    assert state.tolist() == [1, 4, 0, 4]


def test_invalid_indicators_are_not_read_as_abstentions():
    # TangTV is invalid on most bins; the fit must still rate it by where it speaks.
    _, votes = simulate(n=9000, seed=3)
    valid = np.ones(votes.shape, dtype=bool)
    valid[:6000, 2] = False
    votes = votes.copy()
    votes[:6000, 2] = core.ABSTAIN
    acc = lm.LabelModel().fit(votes, valid).accuracies()
    assert acc["tangtv"]["acc_weight"] > acc["afrac"]["acc_weight"]


def test_despeckle_only_bridges_same_neighbours():
    s = np.array([1, 1, 2, 1, 1, 3, 4, 4])
    assert lm.despeckle(s).tolist() == [1, 1, 1, 1, 1, 3, 4, 4]
    t = np.array([1, 1, 2, 3, 3])
    assert lm.despeckle(t).tolist() == t.tolist()
