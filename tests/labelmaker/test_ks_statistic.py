"""`validate._ks_statistic` must equal scipy's oracle (I2).

`validate.py` loads torch at module scope (needed by `adapter_fidelity`),
and torch's bundled `libstdc++` shadows the newer system one that scipy's
compiled `_ckdtree` extension needs - so `import scipy.stats` after
`import labelmaker.validate` fails outside pytest with `ImportError:
version 'GLIBCXX_3.4.29' not found`. No import-order fix inside
`validate.py` is sufficient (a later task's `--stage all` loads torch via
the `infer` stage before `validate` is imported at all), so `_ks_statistic`
replaces the one thing `validate.py` used scipy for
(`scipy.stats.ks_2samp(...).statistic`) with four lines of numpy, and scipy
is no longer imported anywhere at runtime.

This file is the reason that replacement can be trusted: it imports scipy
itself, as an independent oracle, and is only able to because some
test-collection plugin loads a compatible `libstdc++` before torch does -
an environment quirk that holds under pytest and nowhere else, which is
exactly why `_ks_statistic` needs an oracle here rather than a scipy import
at runtime.
"""
import numpy as np
import pytest

from labelmaker.validate import _ks_statistic

scipy_stats = pytest.importorskip("scipy.stats", reason="scipy not importable here")


@pytest.mark.parametrize("seed", range(8))
def test_ks_statistic_matches_scipy_oracle_with_ties(seed):
    """Differing sample sizes and a small integer range, so ties are common."""
    rng = np.random.default_rng(seed)
    n_a = int(rng.integers(5, 60))
    n_b = int(rng.integers(5, 60))
    a = rng.integers(0, 8, size=n_a).astype(np.float64)
    b = rng.integers(0, 8, size=n_b).astype(np.float64)
    got = _ks_statistic(a, b)
    want = scipy_stats.ks_2samp(a, b).statistic
    assert got == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_ks_statistic_matches_scipy_oracle_continuous(seed):
    rng = np.random.default_rng(100 + seed)
    a = rng.normal(size=137)
    b = rng.normal(loc=0.3, scale=1.4, size=211)
    got = _ks_statistic(a, b)
    want = scipy_stats.ks_2samp(a, b).statistic
    assert got == pytest.approx(want, abs=1e-12)


def test_ks_statistic_is_zero_for_identical_samples():
    a = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    assert _ks_statistic(a, a.copy()) == 0.0


def test_ks_statistic_is_one_for_disjoint_supports():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([10.0, 11.0, 12.0])
    assert _ks_statistic(a, b) == 1.0
