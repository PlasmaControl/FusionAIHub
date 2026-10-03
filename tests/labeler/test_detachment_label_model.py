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


def test_marfe_mass_is_pooled_where_nothing_resolves_it():
    # detached and marfe split the not-attached mass 0.55 / 0.40: neither reaches 0.7
    # alone, but where only Afrac/Prad spoke the bin is "detached" (0.95); where
    # TangTV voted the model's own split stands.
    post = np.array([[0.05, 0.55, 0.40], [0.05, 0.55, 0.40]])
    assessed = np.array([True, True])
    has_vote = np.array([True, True])
    resolves = np.array([False, True])
    state = lm.decide(post, assessed, has_vote, 0.7, resolves_marfe=resolves)
    assert state.tolist() == [2, 4]
    pooled = lm.pool_marfe(post, resolves)
    assert pooled[0].tolist() == pytest.approx([0.05, 0.95, 0.0])
    assert pooled[1].tolist() == pytest.approx([0.05, 0.55, 0.40])


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


def simulate_two_populations(seed=3):
    """TangTV shots (all three LFs, balanced states) and plain shots (two LFs, one state).

    In the plain shots the state is constant, so Afrac and Prad do not agree more than
    their accuracies imply and the all-bin fit cannot tell which of them is wrong.
    """
    rng = np.random.default_rng(seed)
    accuracy = {"afrac": 0.65, "prad": 0.7, "tangtv": 0.95}
    n_tv, n_plain = 3000, 12000
    state = np.r_[
        rng.choice([1, 2, 3], size=n_tv, p=[0.4, 0.35, 0.25]), np.ones(n_plain, int)
    ]
    votes = np.full((len(state), 3), core.ABSTAIN)
    valid = np.ones((len(state), 3), dtype=bool)
    valid[n_tv:, 2] = False
    for j, name in enumerate(NAMES):
        for i in np.flatnonzero(valid[:, j]):
            truth = state[i] if state[i] in ALLOWED[name] else 2
            wrong = [s for s in ALLOWED[name] if s != truth]
            right = rng.random() < accuracy[name]
            votes[i, j] = truth if right else rng.choice(wrong)
    return votes, valid


def test_anchored_fit_recovers_accuracies_that_pairs_cannot_identify():
    votes, valid = simulate_two_populations()
    model = lm.LabelModel().fit_anchored(votes, valid)
    acc = model.accuracies()
    assert model.anchor_bins == 3000
    assert acc["afrac"]["implied_accuracy"] == pytest.approx(0.65, abs=0.07)
    assert acc["prad"]["implied_accuracy"] == pytest.approx(0.7, abs=0.07)
    assert acc["tangtv"]["implied_accuracy"] == pytest.approx(0.95, abs=0.05)
    assert model.theta[:3].tolist() == pytest.approx(lm.PRIOR_LOGIT.tolist())


def test_anchored_fit_falls_back_to_the_plain_fit_without_enough_anchor_bins():
    votes, valid = simulate_two_populations()
    valid = valid.copy()
    valid[:, 2] = False  # no bin has all three LFs
    votes = votes.copy()
    votes[:, 2] = core.ABSTAIN
    model = lm.LabelModel(names=("afrac", "prad")).fit_anchored(
        votes[:, :2], valid[:, :2]
    )
    assert model.anchor_bins == 3000 + 12000  # every bin has both of these LFs
    thin = lm.LabelModel().fit_anchored(votes, valid)
    assert thin.anchor_bins == 0 and thin.theta is not None


def test_a_lone_vote_is_judged_as_hard_for_detached_as_for_attached():
    # Afrac alone, valid, nothing else: the two votes carry the same posterior once the
    # unresolved detached/marfe mass is pooled (a uniform 3-state prior would favour
    # "detached" 2:1 before any vote).
    votes, valid = simulate_two_populations()
    model = lm.LabelModel().fit_anchored(votes, valid)
    lone = np.array([[1, core.ABSTAIN, core.ABSTAIN], [2, core.ABSTAIN, core.ABSTAIN]])
    post = lm.pool_marfe(model.posterior(lone), np.array([False, False]))
    assert post[0, 0] == pytest.approx(post[1, 1], abs=0.02)
