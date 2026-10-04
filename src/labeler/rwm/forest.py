"""A balanced random forest in numpy.

The labelmaker environment has no scikit-learn or imbalanced-learn, so this is the
estimator of Chen et al. 2004 as Piccione et al. 2022 use it (imblearn's
`BalancedRandomForestClassifier`), written from that description: every tree is
grown on its own balanced sample, a bootstrap of the minority class plus an equal
number of majority samples drawn at random, so the ensemble still sees all the
majority data while no tree faces the 16:1 or worse imbalance. Splits are Gini on a
random subset of the features; a prediction is the mean of the leaves' positive
fractions.
"""

from __future__ import annotations

import numpy as np


def _best_split(x, y, features, min_leaf):
    """`(gain, feature, threshold)` of the best Gini split over `features`, or None."""
    n = len(y)
    positives = y.sum()
    parent = n * 2.0 * (positives / n) * (1 - positives / n)
    best = None
    for f in features:
        order = np.argsort(x[:, f], kind="stable")
        values = x[order, f]
        left_pos = np.cumsum(y[order])[:-1]
        left_n = np.arange(1, n)
        right_n = n - left_n
        valid = (
            (values[1:] > values[:-1]) & (left_n >= min_leaf) & (right_n >= min_leaf)
        )
        if not valid.any():
            continue
        right_pos = positives - left_pos
        with np.errstate(invalid="ignore", divide="ignore"):
            pl, pr = left_pos / left_n, right_pos / right_n
        impurity = left_n * 2.0 * pl * (1 - pl) + right_n * 2.0 * pr * (1 - pr)
        impurity = np.where(valid, impurity, np.inf)
        i = int(np.argmin(impurity))
        gain = parent - impurity[i]
        if gain > 1e-12 and (best is None or gain > best[0]):
            best = (gain, int(f), 0.5 * (values[i] + values[i + 1]))
    return best


class Tree:
    """One CART tree on flat arrays: `feature < 0` marks a leaf holding `value`."""

    def __init__(self, max_depth, min_leaf, max_features):
        self.max_depth, self.min_leaf, self.max_features = (
            max_depth,
            min_leaf,
            max_features,
        )

    def fit(self, x, y, rng):
        feature, threshold, left, right, value = [], [], [], [], []

        def new_node():
            for column in (feature, threshold, left, right, value):
                column.append(0)
            return len(feature) - 1

        root = new_node()
        stack = [(root, np.arange(len(y)), 0)]
        n_features = x.shape[1]
        while stack:
            node, rows, depth = stack.pop()
            yy = y[rows]
            value[node] = float(yy.mean())
            feature[node], left[node], right[node] = -1, -1, -1
            if (
                depth >= self.max_depth
                or len(rows) < 2 * self.min_leaf
                or yy.min() == yy.max()
            ):
                continue
            chosen = rng.choice(n_features, size=self.max_features, replace=False)
            split = _best_split(x[rows], yy, chosen, self.min_leaf)
            if split is None:
                continue
            _, f, cut = split
            go_left = x[rows, f] <= cut
            feature[node], threshold[node] = f, cut
            left[node], right[node] = new_node(), new_node()
            stack.append((left[node], rows[go_left], depth + 1))
            stack.append((right[node], rows[~go_left], depth + 1))
        self.feature = np.asarray(feature)
        self.threshold = np.asarray(threshold, dtype=float)
        self.left, self.right = np.asarray(left), np.asarray(right)
        self.value = np.asarray(value, dtype=float)
        return self

    def predict(self, x):
        node = np.zeros(len(x), dtype=np.int64)
        for _ in range(self.max_depth + 1):
            f = self.feature[node]
            active = f >= 0
            if not active.any():
                break
            go_left = (
                x[np.arange(len(x)), np.where(active, f, 0)] <= self.threshold[node]
            )
            node = np.where(
                active, np.where(go_left, self.left[node], self.right[node]), node
            )
        return self.value[node]


class BalancedForest:
    """Balanced random forest for a binary label (1 minority)."""

    def __init__(
        self, n_estimators=300, max_depth=8, min_leaf=5, max_features="sqrt", seed=0
    ):
        self.n_estimators, self.max_depth, self.min_leaf = (
            n_estimators,
            max_depth,
            min_leaf,
        )
        self.max_features, self.seed = max_features, seed

    def _n_features(self, p):
        if self.max_features == "sqrt":
            return max(1, round(np.sqrt(p)))
        return min(p, int(self.max_features))

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y).astype(float)
        positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if not len(positive) or not len(negative):
            raise ValueError("both classes are needed to fit a balanced forest")
        rng = np.random.default_rng(self.seed)
        k = self._n_features(x.shape[1])
        self.trees = []
        for _ in range(self.n_estimators):
            minority = rng.choice(positive, size=len(positive), replace=True)
            majority = rng.choice(
                negative, size=len(positive), replace=len(negative) < len(positive)
            )
            rows = np.concatenate([minority, majority])
            self.trees.append(
                Tree(self.max_depth, self.min_leaf, k).fit(x[rows], y[rows], rng)
            )
        return self

    def predict_proba(self, x):
        """Mean positive-class fraction of the leaves reached, in [0, 1]."""
        x = np.asarray(x, dtype=float)
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)
