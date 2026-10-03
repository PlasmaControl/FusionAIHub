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
are constrained non-negative (an LF is never worse than chance on average by
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

Neither uses the cohort test split: the model is fitted on the bins it is given.
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
#: Largest accuracy / correlation weight the fit may use (an LF that never errs would
#: otherwise run to infinity on a finite sample).
WEIGHT_MAX = 8.0
#: L2 penalty on the propensity and prior terms, per observation, for stability.
RIDGE = 1e-6


@dataclass
class LabelModel:
    """A fitted model: LF names, allowed votes, correlated pairs and parameters."""

    names: tuple[str, ...] = LF_NAMES
    allowed: dict = None
    corr: tuple = ()
    theta: np.ndarray | None = None
    loglik: float | None = None
    n_obs: int = 0

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
                "propensity_weight": float(self.theta[3 + j]),
                "implied_accuracy": float(np.exp(w) / (np.exp(w) + n_allowed - 1)),
            }
        return out


def decide(
    posterior: np.ndarray,
    assessed: np.ndarray,
    has_vote: np.ndarray,
    threshold: float = 0.7,
) -> np.ndarray:
    """State per bin from the posterior.

    `assessed` is True where the bin has enough valid indicators to be labelled at
    all (else ABSENT); `has_vote` is True where at least one indicator cast a vote.
    An assessed bin with no vote (every valid indicator in its transition band) or
    a posterior argmax below `threshold` is UNCERTAIN; otherwise the argmax.
    """
    best = posterior.argmax(axis=1)
    top = posterior.max(axis=1)
    sure = (top >= threshold) & np.asarray(has_vote, dtype=bool)
    state = np.where(sure, np.array(VOTE_STATES)[best], UNCERTAIN)
    return np.where(assessed, state, ABSENT).astype(np.int8)


def rule(votes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The fallback: agreement of the votes, uncertain on conflict or no vote.

    `votes` and `valid` are `(bins, LFs)`. No valid indicator: ABSENT. Valid
    indicators but no vote, or two different votes: UNCERTAIN. Otherwise the vote.
    """
    votes = np.asarray(votes)
    out = np.full(len(votes), ABSENT, dtype=np.int8)
    for i, (row, ok) in enumerate(zip(votes, np.asarray(valid), strict=True)):
        if not ok.any():
            continue
        cast = {int(v) for v in row if v != ABSTAIN}
        out[i] = cast.pop() if len(cast) == 1 else UNCERTAIN
    return out


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
