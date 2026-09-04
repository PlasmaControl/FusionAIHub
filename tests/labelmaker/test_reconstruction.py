"""Recovering the archived row mapping, and pricing the substitutions."""
from pathlib import Path

import numpy as np
import pytest

from labelmaker import validate
from labelmaker.catalog import TM_ARCHIVE
from labelmaker.config import Paths
from labelmaker.models.base import BuiltInputs

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def _built(n=240, seed=0):
    rng = np.random.default_rng(seed)
    scalars = np.zeros((n, 11))
    scalars[:, 0] = 2.0 + 0.001 * np.arange(n)            # bt, monotone
    scalars[:, 1] = 1e6 + 1e3 * np.arange(n)              # ip
    scalars[:, 6] = 0.4 + 1e-4 * np.arange(n)             # tritop
    scalars[:, 7] = 0.3 + 1e-4 * np.arange(n)             # tribot
    scalars[:, 8] = 0.05 + 1e-5 * np.arange(n)            # gapin
    scalars[:, 2] = rng.normal(size=n)
    profiles = rng.normal(size=(n, 33, 5))
    return BuiltInputs(
        t=0.025 * np.arange(n), scalars=scalars, profiles=profiles,
        valid=np.ones(n, bool), missing=(), resolvers={},
    )


def test_match_rows_recovers_a_known_subset():
    built = _built()
    take = np.array([41, 58, 59, 100, 199])
    archived = built.scalars[take]
    got = validate.match_rows(archived, built)
    np.testing.assert_array_equal(got["index"], take)
    assert got["median_distance"] < 1e-9
    assert got["monotonic"] is True
    assert got["n_matched"] == 5


def test_match_rows_reports_a_bad_match_rather_than_hiding_it():
    built = _built()
    archived = built.scalars[[10, 20]] + 5.0        # nothing like the real rows
    got = validate.match_rows(archived, built)
    assert got["median_distance"] > 1.0
    assert got["passed"] is False


def test_match_rows_needs_the_match_columns_to_vary():
    built = _built()
    built = BuiltInputs(
        t=built.t, scalars=np.zeros_like(built.scalars), profiles=built.profiles,
        valid=built.valid, missing=(), resolvers={},
    )
    with pytest.raises(ValueError, match="constant"):
        validate.match_rows(np.zeros((3, 11)), built)


def test_match_rows_skips_an_archived_row_with_no_finite_distance():
    """A row missing one of the five match columns must not abort the run.

    Addendum defect 4: `np.nanargmin` raises `ValueError: All-NaN slice
    encountered` when an archived row has no finite distance to any of our
    timesteps. That is reachable per-shot (a column `resolve_archive` did
    not carry for that shot), so it must be guarded rather than allowed to
    abort the whole validation run.
    """
    built = _built()
    archived = built.scalars[[41, 58, 100]].copy()
    archived[1, :] = np.nan          # this archived row cannot match anything
    got = validate.match_rows(archived, built)
    assert got["n_matched"] == 3
    assert not np.isfinite(got["distance"][1])
    assert got["index"][1] == -1
    # the two good rows still match exactly
    assert got["distance"][0] < 1e-9
    assert got["distance"][2] < 1e-9


@pytest.mark.skipif(not TM_ARCHIVE.exists(), reason="tm archive not available")
def test_archive_rows_reads_one_shot():
    got = validate.archive_rows(185945)
    assert got["x0"].shape[1] == 11
    assert got["x1"].shape[1:] == (33, 5)
    assert got["y"].shape[1] == 2
    assert got["x0"].shape[0] == got["y"].shape[0] > 50
    assert validate.archive_rows(1) is None            # not in the archive


@pytest.mark.skipif(
    not (TM_ARCHIVE.exists() and CORPUS.exists()),
    reason="archive or corpus not available",
)
@pytest.mark.skipif(
    not (Paths.from_env().features / "185945_features.h5").exists(),
    reason="run `features` on shot 185945 first",
)
def test_reconstruction_report_on_one_real_shot():
    report = validate.reconstruction_fidelity(
        "d3d_tearing_onset_cnn1d", [185945], Paths.from_env()
    )
    per = report["per_feature"]
    # the five match columns must be near-exact, or the mapping is wrong
    for name in ("bt", "ip", "tritop_EFIT01", "tribot_EFIT01", "gapin_EFIT01"):
        assert per[name]["median_rel"] < 1e-2, (name, per[name])
    # the substituted profiles are the priced ones
    assert per["thomson_density_mtanh_1d"]["corr"] > 0.9
    assert report["match"]["185945"]["median_distance"] < 1e-3


def test_reconstruction_fidelity_counts_shots_before_the_generator_is_consumed():
    """Addendum defect 1: `len(list(shots))` after the loop reports 0 for a
    generator. `shots_requested` must reflect what was actually asked for."""

    def shots():
        yield 999999999    # not in the archive; the loop must still consume it

    report = validate.reconstruction_fidelity(
        "d3d_tearing_onset_cnn1d", shots(), Paths.from_env()
    )
    assert report["n_shots_requested"] == 1
    assert report["skipped"]["999999999"] == "no archived rows"
