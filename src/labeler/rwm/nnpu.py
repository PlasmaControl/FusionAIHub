"""Non-negative positive-unlabelled risk (Kiryo et al. 2017) on a small MLP.

Excluded development code: neither this module nor evaluate.Nnpu contributes to
any reported RWM result; outer-fold-informed development invalidated comparisons.

Only positives are labelled. With the class prior `pi` known, the risk of a classifier
`g` under the sigmoid loss is estimated from positives `P` and unlabelled `U` as

    R = pi E_P[l(g, +1)] + max(0, E_U[l(g, -1)] - pi E_P[l(g, -1)])

and the bracket, an estimate of the negative-class risk, cannot be negative. A
mini-batch whose bracket falls below `-beta` takes a gradient ascent step on the
bracket alone, scaled by `gamma`, which is what stops a flexible model from
overfitting by driving it negative. The loss is `sigmoid(-y z)`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _sigmoid_loss(z, y):
    return torch.sigmoid(-y * z)


def _logistic_loss(z, y):
    return nn.functional.softplus(-y * z)


LOSSES = {"sigmoid": _sigmoid_loss, "logistic": _logistic_loss}


class NnPU:
    """MLP trained with the nnPU risk; `predict_proba` is the sigmoid of the logit."""

    def __init__(
        self,
        prior,
        hidden=(64, 64),
        epochs=40,
        batch_p=64,
        batch_u=512,
        lr=1e-3,
        weight_decay=1e-4,
        beta=0.0,
        gamma=1.0,
        loss="sigmoid",
        seed=0,
    ):
        if not 0.0 < prior < 1.0:
            raise ValueError("the class prior must be in (0, 1)")
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {sorted(LOSSES)}")
        self.loss = LOSSES[loss]
        self.prior, self.hidden, self.epochs = prior, tuple(hidden), epochs
        self.batch_p, self.batch_u, self.lr = batch_p, batch_u, lr
        self.weight_decay, self.beta, self.gamma, self.seed = (
            weight_decay,
            beta,
            gamma,
            seed,
        )

    def _network(self, n_in):
        layers, width = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def fit(self, x, labelled):
        """`labelled` is 1 for a labelled positive and 0 for an unlabelled row."""
        x = np.asarray(x, dtype=np.float32)
        labelled = np.asarray(labelled)
        positive, unlabelled = (
            np.flatnonzero(labelled == 1),
            np.flatnonzero(labelled == 0),
        )
        if not len(positive) or not len(unlabelled):
            raise ValueError("nnPU needs labelled positives and unlabelled rows")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mean_ = x.mean(axis=0)
        self.scale_ = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
        data = torch.from_numpy((x - self.mean_) / self.scale_)
        self.net = self._network(x.shape[1])
        optimiser = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        steps = max(1, len(unlabelled) // self.batch_u)
        for _ in range(self.epochs):
            for _ in range(steps):
                p = torch.from_numpy(rng.choice(positive, self.batch_p))
                u = torch.from_numpy(rng.choice(unlabelled, self.batch_u))
                zp, zu = self.net(data[p]).squeeze(1), self.net(data[u]).squeeze(1)
                positive_risk = self.prior * self.loss(zp, 1.0).mean()
                negative_risk = (
                    self.loss(zu, -1.0).mean() - self.prior * self.loss(zp, -1.0).mean()
                )
                optimiser.zero_grad()
                if negative_risk.item() >= -self.beta:
                    (positive_risk + negative_risk).backward()
                else:
                    (-self.gamma * negative_risk).backward()
                optimiser.step()
        return self

    def predict_proba(self, x):
        data = torch.from_numpy(
            (np.asarray(x, dtype=np.float32) - self.mean_) / self.scale_
        )
        with torch.no_grad():
            return torch.sigmoid(self.net(data).squeeze(1)).numpy().astype(float)
