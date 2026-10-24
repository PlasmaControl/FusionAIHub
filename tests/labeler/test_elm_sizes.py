"""`labeler.elm.sizes`: a detected ELM's size measures and the logistic models."""

import numpy as np
import pytest

from labeler.elm import inputs, inst, sizes


def test_features_measure_rise_contrast_and_density_drop():
    n = 400 * inputs.CELLS_PER_MS
    x = np.zeros((inputs.N_CHANNELS, n), np.float32)
    x[inputs.VALID] = 1
    t0 = inputs.GRID0_MS + 100.05
    c = int(inst.cell_of(t0))
    x[list(inputs.FS_LEVEL), c : c + 6] = 1.0 / inputs.FS_SCALE  # one decade up
    x[list(inputs.FS_CONTRAST), c : c + 6] = 0.8 / inputs.FS_SCALE
    x[inputs.DENSITY_HP[0], c + 10 : c + 30] = -5.0  # 0.5 of the unit, high-passed
    # the other chord was screened to zero: it is left out, not averaged in
    t1 = inputs.GRID0_MS + 300.05
    c1 = int(inst.cell_of(t1))
    x[:, c1 - 30 : c1 + 60] = 0  # an invalid stretch
    out = sizes.features(x, [t0, t1, inputs.GRID0_MS - 10])
    np.testing.assert_allclose(out[0], [1.0, 0.8, 0.5], atol=1e-6)
    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()


def test_fit_separates_two_classes_and_fills_missing_measures():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(1.0, 0.2, (200, 3)), rng.normal(0.3, 0.2, (800, 3))])
    y = np.r_[np.ones(200), np.zeros(800)].astype(bool)
    X[0, 1] = np.nan
    model = sizes.fit(X, y, weight=sizes.balanced(y))
    p = sizes.prob(model, X)
    assert np.isfinite(p).all()
    assert sizes.auroc(p, y) > 0.99
    # balanced weights: the two classes count alike, so the middle is near one half
    assert sizes.prob(model, np.full((1, 3), 0.65))[0] == pytest.approx(0.5, abs=0.1)
    assert sizes.balanced(y)[y].sum() == pytest.approx(sizes.balanced(y)[~y].sum())


def test_auroc_counts_ties_half():
    assert sizes.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert sizes.auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert np.isnan(sizes.auroc([0.5], [1]))
