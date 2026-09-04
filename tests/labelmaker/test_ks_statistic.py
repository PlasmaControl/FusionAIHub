"""`validate._ks_statistic` must equal scipy's oracle (I2).

`validate.py` loads torch at module scope (needed by `adapter_fidelity`),
and `import torch` binds the SYSTEM `/lib64/libstdc++.so.6` - which lacks
`GLIBCXX_3.4.29` - ahead of the pixi env's own newer copy, unless something
puts the latter first on the loader's path. Unpatched, that made
`import scipy.stats` after `import labelmaker.validate` fail with
`ImportError: version 'GLIBCXX_3.4.29' not found`. This is NOT a defect
unique to scipy: the identical loader-ordering problem is what silently
disabled labelmaker's entire fdp scaling path (Task 16b - see
`features/resolve_fdp.py`'s module docstring), whose fix,
`pyproject.toml`'s `tool.pixi.feature.fdp` activation table, now puts the
pixi env's own `libstdc++` first via `LD_LIBRARY_PATH` for the whole
`labelmaker` environment - which also fixes this scipy import, in any
context, not just under pytest. No import-order fix inside `validate.py`
alone would have been sufficient regardless (a later task's `--stage all`
loads torch via the `infer` stage before `validate` is imported at all), so
`_ks_statistic` replaces the one thing `validate.py` used scipy for
(`scipy.stats.ks_2samp(...).statistic`) with four lines of numpy, and scipy
is no longer imported anywhere at runtime - kept even after the activation
fix, for the independent reduced-dependency benefit.

This file is the reason that replacement can be trusted: it imports scipy
itself, as an independent oracle, guarded with `importorskip` so it
degrades gracefully in an environment where the activation fix does not
apply, rather than asserting anything about *why* scipy is or is not
importable here.
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
