"""Combine the three indicators into one state per bin: a label model and a rule.

**Label model.** Snorkel-style data programming (Ratner et al. 2017): each
indicator is a labelling function (LF) that votes attached / detached / marfe or
abstains; a generative model of the votes with the unknown true state as a latent
variable learns how accurate each LF is from how often they agree, with no labelled
data. The joint is log-linear,

    p(y, v) = exp( b_y + sum_j [ lab_j * 1(v_j != abstain) + acc_j * 1(v_j ~ y) ]
                   + sum_(j,k) corr_jk * 1(v_j = v_k != abstain) ) / Z,

with the propensity factor `lab_j`, the accuracy factor `acc_j` (a vote is accurate
when the state is one it is consistent with, see `COMPATIBLE`) and, optionally, a
pairwise correlation factor. Three states times at most 4 votes per LF is a few
dozen configurations, so `Z`, the marginal likelihood of the observed votes and its
gradient are computed EXACTLY by enumeration (no sampling, no SGD). `acc` and `corr`
are constrained to [0, WEIGHT_MAX] (an LF is never worse than chance on average by
construction of its thresholds). Each LF has its own allowed vote set: Afrac and
Prad never vote marfe (no position information), so a marfe state can only come from
TangTV, and a bin where only they speak cannot be marfe.

**Rule.** The transparent fallback: collect the non-abstaining votes of the valid
indicators; if they all agree that is the state; if they disagree, or every valid
indicator abstains (a transition band), the state is `uncertain`; if no indicator
is valid the bin has no label.

**Validity.** An indicator that is invalid on a bin (no data, wrong geometry) is not
an abstention in the transition band: it cannot vote at all. The likelihood is
therefore conditional on the validity pattern of each bin: the normaliser runs only
over vote configurations in which every invalid indicator abstains, so a pattern
that rarely lets TangTV speak (most shots have no inversion) does not teach the model
that TangTV is silent by choice.

**Identification.** The model accuracy parameter can only be identified under its conditional
independence assumptions; it is not identified physical accuracy. Agreement
needs three voters: from two the data fix only the product of the two
accuracies. Most bins carry two voters (TangTV is valid on few shots) and in them the
state is nearly constant (attached), so a fit on all bins can place the whole
disagreement on one LF and give the other an accuracy near 1. `fit_anchored` therefore
learns the accuracy and correlation weights from the bins where every LF is valid,
fixes the class balance (nothing in the data identifies it across shots of different
regimes) and fits only the propensities on all bins. The class balance is the
maximum-entropy one for the two-level decision the indicators make: attached against
not attached equally likely, the not-attached half split between detached and marfe
(`PRIOR_LOGIT`). A uniform prior over the three states would count the pooled
"detached or marfe" of the bins where only Afrac and Prad,div speak twice as likely
as attached before any vote is cast, so a lone detached vote would label a bin while
a lone attached vote of the same accuracy would not.

Neither labeler uses the cohort test split: the model is fitted on the bins it is given.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .core import ABSENT, ABSTAIN, UNCERTAIN, VOTE_STATES

LF_NAMES = ("afrac", "prad", "tangtv")
ALLOWED = {"afrac": (1, 2), "prad": (1, 2), "tangtv": (1, 2, 3)}
#: The states a vote is consistent with. A MARFE is the last stage of detachment, so
#: the "detached" vote of a source that cannot see the X-point (Afrac, Prad) is
#: right when the state is detached OR marfe; only TangTV separates the two.
COMPATIBLE = {
    "afrac": {1: (1,), 2: (2, 3)},
    "prad": {1: (1,), 2: (2, 3)},
    "tangtv": {1: (1,), 2: (2,), 3: (3,)},
}
#: Largest accuracy / correlation weight the fit may use. Without labelled data an LF
#: that never disagrees on a finite sample would run to infinity; 4 caps the implied
#: accuracy of a vote at 0.96 (three allowed votes) to 0.98 (two), a numerical bound, not evidence of physical accuracy or calibrated confidence.
WEIGHT_MAX = 4.0
#: Class balance of `fit_anchored`: attached 1/2, detached 1/4, marfe 1/4 (log).
PRIOR_LOGIT = np.log(np.array([0.5, 0.25, 0.25]))
#: L2 penalty on the propensity and prior terms, per observation, for stability.
RIDGE = 1e-6
#: Fewest all-LF-valid bins `fit_anchored` will learn the accuracies from.
MIN_ANCHOR_BINS = 300


@dataclass
class LabelModel:
    """A fitted model: LF names, allowed votes, correlated pairs and parameters."""

    names: tuple[str, ...] = LF_NAMES
    allowed: dict = None
    corr: tuple = ()
    theta: np.ndarray | None = None
    loglik: float | None = None
    n_obs: int = 0
    anchor_bins: int = 0

    def __post_init__(self) -> None:
        if self.allowed is None:
            self.allowed = {n: ALLOWED[n] for n in self.names}
        self._configs = list(
            itertools.product(*[(ABSTAIN, *self.allowed[n]) for n in self.names])
        )
        self._index = {c: i for i, c in enumerate(self._configs)}
        self._phi = self._features()
        self.n_params = self._phi.shape[2]

    # ---- parameter layout: prior (3), propensity (J), accuracy (J), corr (P) ----
    def _features(self) -> np.ndarray:
        n_lf, n_corr = len(self.names), len(self.corr)
        phi = np.zeros((len(self._configs), len(VOTE_STATES), 3 + 2 * n_lf + n_corr))
        for c, votes in enumerate(self._configs):
            for k, state in enumerate(VOTE_STATES):
                phi[c, k, k] = 1.0
                for j, vote in enumerate(votes):
                    if vote != ABSTAIN:
                        phi[c, k, 3 + j] = 1.0
                        ok = state in COMPATIBLE[self.names[j]][vote]
                        phi[c, k, 3 + n_lf + j] = float(ok)
                for p, (a, b) in enumerate(self.corr):
                    ia, ib = self.names.index(a), self.names.index(b)
                    if votes[ia] != ABSTAIN and votes[ia] == votes[ib]:
                        phi[c, k, 3 + 2 * n_lf + p] = 1.0
        return phi

    def _bounds(self):
        n_lf = len(self.names)
        free = (-20.0, 20.0)
        pos = (0.0, WEIGHT_MAX)
        return [free] * (3 + n_lf) + [pos] * (n_lf + len(self.corr))

    def _patterns(self, valid: np.ndarray):
        """Unique validity patterns, each bin's pattern index and the config masks."""
        valid = np.asarray(valid, dtype=bool)
        patterns, inverse = np.unique(valid, axis=0, return_inverse=True)
        cfg = np.array(self._configs)
        masks = np.array(
            [[bool(np.all(row[~pat] == ABSTAIN)) for row in cfg] for pat in patterns]
        )
        return patterns, np.asarray(inverse).reshape(-1), masks

    def config_counts(self, votes: np.ndarray, valid: np.ndarray | None = None):
        """`(counts, masks)`: vote-tuple histograms per validity pattern and the
        configurations each pattern allows. `valid=None` means every LF is valid."""
        votes = np.asarray(votes)
        if valid is None:
            valid = np.ones(votes.shape, dtype=bool)
        patterns, inverse, masks = self._patterns(valid)
        counts = np.zeros((len(patterns), len(self._configs)))
        for row, p in zip(votes, inverse, strict=True):
            counts[p, self._index[tuple(int(v) for v in row)]] += 1
        return counts, masks

    def _scores(self, theta: np.ndarray) -> np.ndarray:
        return self._phi @ theta

    def neg_loglik(self, theta: np.ndarray, data):
        """Negative marginal log-likelihood of the votes and its gradient.

        `data` is the `(counts, masks)` pair of `config_counts`: the votes of each
        validity pattern are scored against a normaliser summed over only the
        configurations that pattern allows.
        """
        counts, masks = data
        s = self._scores(theta)
        log_marg = logsumexp(s, axis=1)
        post = np.exp(s - log_marg[:, None])
        total = counts.sum()
        ll = 0.0
        grad = np.zeros(len(theta))
        for n_c, mask in zip(counts, masks, strict=True):
            n_p = n_c.sum()
            if n_p == 0:
                continue
            log_z = logsumexp(s[mask])
            ll += float(n_c @ log_marg - n_p * log_z)
            joint = np.where(mask[:, None], np.exp(s - log_z), 0.0)
            grad += np.einsum("c,ck,ckd->d", n_c, post, self._phi)
            grad -= n_p * np.einsum("ck,ckd->d", joint, self._phi)
        reg = RIDGE * total
        n_free = 3 + len(self.names)
        ll -= 0.5 * reg * float(theta[:n_free] @ theta[:n_free])
        grad[:n_free] -= reg * theta[:n_free]
        return -ll, -grad

    def fit(self, votes: np.ndarray, valid: np.ndarray | None = None) -> LabelModel:
        """Maximise the marginal likelihood of the observed votes (L-BFGS-B)."""
        data = self.config_counts(votes, valid)
        theta0 = np.zeros(self.n_params)
        theta0[3 + len(self.names) : 3 + 2 * len(self.names)] = 1.0
        res = minimize(
            self.neg_loglik,
            theta0,
            args=(data,),
            jac=True,
            method="L-BFGS-B",
            bounds=self._bounds(),
        )
        self.theta, self.loglik, self.n_obs = res.x, -res.fun, int(data[0].sum())
        return self

    def fit_anchored(
        self,
        votes: np.ndarray,
        valid: np.ndarray,
        anchor_mask: np.ndarray | None = None,
    ) -> LabelModel:
        """Fit accuracies where all LFs are valid, then the propensities on every bin.

        See "Identification" in the module docstring. The class balance is fixed to
        `PRIOR_LOGIT` after the accuracy fit (the accuracies are fitted with the class
        balance free: on the anchor bins it is identified, but those shots are not a
        sample of the corpus). With fewer than `MIN_ANCHOR_BINS` bins on which every
        LF is valid the accuracies are not identified and the plain `fit` is used instead
        (`self.anchor_bins` is then 0).
        """
        votes, valid = np.asarray(votes), np.asarray(valid, dtype=bool)
        full = valid.all(axis=1)
        if anchor_mask is not None:
            full &= np.asarray(anchor_mask, bool)
        self.anchor_bins = int(full.sum())
        if self.anchor_bins < MIN_ANCHOR_BINS:
            self.anchor_bins = 0
            return self.fit(votes, valid)
        self.fit(votes[full], valid[full])
        base = self.theta.copy()
        base[:3] = PRIOR_LOGIT
        data = self.config_counts(votes, valid)
        free = np.arange(3, 3 + len(self.names))  # the propensity weights

        def objective(x):
            theta = base.copy()
            theta[free] = x
            value, grad = self.neg_loglik(theta, data)
            return value, grad[free]

        res = minimize(
            objective,
            base[free],
            jac=True,
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0)] * len(free),
        )
        base[free] = res.x
        self.theta, self.loglik, self.n_obs = base, -res.fun, int(data[0].sum())
        return self

    def loglik_per_obs(self, votes: np.ndarray, valid: np.ndarray | None = None):
        """Mean marginal log-likelihood of `votes` under the fitted parameters."""
        data = self.config_counts(votes, valid)
        return float(-self.neg_loglik(self.theta, data)[0] / data[0].sum())

    def posterior(self, votes: np.ndarray) -> np.ndarray:
        """`(bins, 3)` posterior over attached / detached / marfe for each bin."""
        s = self._scores(self.theta)
        table = np.exp(s - logsumexp(s, axis=1, keepdims=True))
        index = [self._index[tuple(int(v) for v in row)] for row in np.asarray(votes)]
        return table[index]

    def accuracies(self) -> dict:
        """Learned accuracy weight and the implied P(vote = state | vote cast)."""
        n_lf = len(self.names)
        out = {}
        for j, name in enumerate(self.names):
            w = float(self.theta[3 + n_lf + j])
            n_allowed = len(self.allowed[name])
            out[name] = {
                "acc_weight": w,
                "at_weight_bound": bool(np.isclose(w, WEIGHT_MAX)),
                "physical_accuracy_identified": False,
                "calibrated": False,
                "propensity_weight": float(self.theta[3 + j]),
                "implied_accuracy": float(np.exp(w) / (np.exp(w) + n_allowed - 1)),
            }
        return out


def pool_marfe(posterior: np.ndarray, resolves_marfe: np.ndarray) -> np.ndarray:
    """Posterior with the MARFE mass added to "detached" where no indicator that can
    vote MARFE has voted (the two are then indistinguishable; see `decide`)."""
    posterior = np.array(posterior, dtype=float)
    pooled = ~np.asarray(resolves_marfe, dtype=bool)
    posterior[pooled, 1] += posterior[pooled, 2]
    posterior[pooled, 2] = 0.0
    return posterior


def decide(
    posterior: np.ndarray,
    assessed: np.ndarray,
    has_vote: np.ndarray,
    threshold: float = 0.7,
    resolves_marfe: np.ndarray | None = None,
) -> np.ndarray:
    """State per bin from the posterior.

    `assessed` is True where the bin has enough valid indicators to be labelled at
    all (else ABSENT); `has_vote` is True where at least one indicator cast a vote.
    An assessed bin with no vote (every valid indicator in its transition band) or
    a posterior argmax below `threshold` is UNCERTAIN; otherwise the argmax.

    `resolves_marfe` is True where an indicator that can vote MARFE (TangTV) cast a
    vote. Elsewhere "detached" and "marfe" have the same likelihood, the posterior
    only splits by the prior, so their masses are pooled into "detached" (read: not
    attached, a MARFE not excluded) and a MARFE is never decided there. None means
    every bin resolves it.
    """
    if resolves_marfe is not None:
        posterior = pool_marfe(posterior, resolves_marfe)
    best = posterior.argmax(axis=1)
    top = posterior.max(axis=1)
    sure = (top >= threshold) & np.asarray(has_vote, dtype=bool)
    state = np.where(sure, np.array(VOTE_STATES)[best], UNCERTAIN)
    return np.where(assessed, state, ABSENT).astype(np.int8)


def rule(votes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Intersect COMPATIBLE sets, ignoring every invalid or abstaining vote."""
    out = np.full(len(votes), ABSENT, dtype=np.int8)
    for i, (row, ok) in enumerate(zip(votes, valid, strict=True)):
        if not np.any(ok):
            continue
        cast = [
            (name, int(v))
            for name, v, good in zip(LF_NAMES, row, ok, strict=True)
            if good and v > 0
        ]
        compatible = set(VOTE_STATES)
        for name, vote in cast:
            compatible &= set(COMPATIBLE[name].get(vote, (vote,)))
        if not cast or not compatible:
            out[i] = UNCERTAIN
        else:
            out[i] = min(compatible)
    return out


def redundant_decide(posterior, votes, valid, threshold=0.7):
    """Apply redundancy after the uncalibrated posterior; return state and tier.

    Pair-only proxy agreement is exported as uncertain. A certain state needs a
    valid TangTV vote and another compatible vote, no conflicting vote, and its
    own posterior >= threshold. MARFE therefore requires a valid MARFE vote.
    """
    votes = np.where(valid, votes, ABSTAIN)
    assessed = np.asarray(valid).sum(axis=1) >= 2
    fallback = rule(votes, valid)
    state = decide(posterior, assessed, (votes > 0).any(axis=1), threshold)
    tier = np.full(len(state), "low_posterior", dtype=object)
    tv = votes[:, 2]
    support = (votes > 0).sum(axis=1) >= 2
    conflict = (fallback == UNCERTAIN) & (votes > 0).any(axis=1)
    pair = (tv <= 0) & support & ~conflict
    permitted = (tv > 0) & support & ~conflict & (state == fallback)
    tier[permitted & np.isin(state, VOTE_STATES)] = "certain"
    state[~permitted & assessed] = UNCERTAIN
    tier[pair] = "low_confidence_pair"
    tier[conflict] = "conflict"
    tier[~(votes > 0).any(axis=1)] = "no_vote"
    tier[~assessed] = "not_assessed"
    return state, tier


def despeckle(state: np.ndarray, min_run: int = 2) -> np.ndarray:
    """Give a run shorter than `min_run` bins the state of its neighbours when both
    sides agree (a one-bin dither between two bins of the same state is not an event).
    """
    state = np.asarray(state).copy()
    edges = np.flatnonzero(np.r_[True, state[1:] != state[:-1], True])
    runs = list(itertools.pairwise(edges))
    for k in range(1, len(runs) - 1):
        s, e = runs[k]
        if e - s < min_run and state[runs[k - 1][0]] == state[runs[k + 1][0]]:
            state[s:e] = state[runs[k - 1][0]]
    return state
