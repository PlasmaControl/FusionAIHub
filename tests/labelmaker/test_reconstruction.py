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
    assert got["n_archived_rows"] == 5
    assert got["n_matched"] == 5
    assert got["n_unique_matched"] == 5
    assert got["fail_reason"] is None


def test_match_rows_reports_a_bad_match_rather_than_hiding_it():
    built = _built()
    archived = built.scalars[[10, 20]] + 5.0        # nothing like the real rows
    got = validate.match_rows(archived, built)
    assert got["median_distance"] > 1.0
    assert got["passed"] is False
    # Both offset rows snap to the same out-of-range nearest timestep here,
    # so the failure this particular input produces is a collision, not a
    # loose median - either way `fail_reason` must say something, not hide
    # it behind a bare median (I4).
    assert got["fail_reason"]


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

    I4: `n_archived_rows` (3, how many rows were presented) and `n_matched`
    (2, how many actually found a finite-distance match) are different
    numbers - the pre-fix code called the former `n_matched`, which is wrong
    whenever a row goes unmatched.
    """
    built = _built()
    archived = built.scalars[[41, 58, 100]].copy()
    archived[1, :] = np.nan          # this archived row cannot match anything
    got = validate.match_rows(archived, built)
    assert got["n_archived_rows"] == 3
    assert got["n_matched"] == 2
    assert got["n_unique_matched"] == 2
    assert got["passed"] is False
    assert "unmatched" in got["fail_reason"]
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


def test_match_rows_reports_a_loose_median_when_that_is_the_only_problem():
    """I4's third `fail_reason` branch: every row matched, no collision, but
    the median distance itself exceeds `tol` - a small, uniform perturbation
    that does not change which row is nearest.
    """
    built = _built()
    ours = built.scalars[:, list(validate.MATCH_COLUMNS)]
    scale = np.std(ours, axis=0)
    archived = built.scalars[[100, 150]].copy()
    archived[:, list(validate.MATCH_COLUMNS)] += 0.002 * scale
    got = validate.match_rows(archived, built)
    np.testing.assert_array_equal(got["index"], [100, 150])
    assert got["n_matched"] == 2
    assert got["n_unique_matched"] == 2
    assert got["passed"] is False
    assert "exceeds tol" in got["fail_reason"]


def test_match_rows_reports_a_collision_rather_than_a_bare_median():
    """I4: a collision (two archived rows matching the same timestep) is a
    different failure from an unmatched row or a loose median, and the skip
    reason must say which one happened rather than always naming the
    median - the pre-fix message was "match rejected (median 0)" even for a
    pure collision.
    """
    built = _built()
    archived = built.scalars[[41, 41]]           # both rows claim timestep 41
    got = validate.match_rows(archived, built)
    assert got["n_archived_rows"] == 2
    assert got["n_matched"] == 2
    assert got["n_unique_matched"] == 1
    assert got["passed"] is False
    assert "collided" in got["fail_reason"]


def test_reconstruction_fidelity_isolates_a_per_shot_crash(tmp_path, monkeypatch):
    """C1: `nan_policy='zero'` turns an absent feature into an all-zero
    column. A shot missing every one of the five `MATCH_COLUMNS` features
    (an archive coverage gap - `ip`/`bt` are absent on 2,192 of 5,000 archive
    shots - or a `features` run without the `fdp run` wrapper) makes
    `match_rows`' variance guard raise `ValueError` on a constant column.
    That is reachable per-shot; before this fix it aborted the whole run
    instead of costing only that shot, with no partial JSON to show for it.
    """
    import h5py

    n = 3
    fake_archive = {
        "x0": np.zeros((n, 11)),
        "x1": np.zeros((n, 33, 5)),
        "y": np.zeros((n, 2)),
        "rows": np.arange(n),
    }

    def fake_archive_rows(shot, archive=TM_ARCHIVE):
        return fake_archive if shot == 111 else None

    monkeypatch.setattr(validate, "archive_rows", fake_archive_rows)

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    with h5py.File(features_dir / "111_features.h5", "w"):
        pass  # a valid, but empty, feature file: every field is absent

    report = validate.reconstruction_fidelity(
        "d3d_tearing_onset_cnn1d", [111, 222], Paths(root=tmp_path)
    )
    assert report["n_shots_requested"] == 2
    assert report["n_shots_used"] == 0
    # 222 never reaches match_rows at all - proof the loop kept going past
    # 111's exception rather than aborting.
    assert report["skipped"]["222"] == "no archived rows"
    assert "ValueError" in report["skipped"]["111"]
    assert "constant" in report["skipped"]["111"]
    # I3 (task-16 review): the diagnosis at skip time, not thrown away.
    assert "111" in report["skip_diagnosis"] and "222" in report["skip_diagnosis"]
    assert report["skip_reasons"]["histogram"]  # non-empty: something to count


def test_reconstruction_fidelity_records_incomplete_features_for_skipped_shots(
    tmp_path, monkeypatch
):
    """I3 (related): `incomplete_features` used to be recorded only after
    the `match_rows` `passed` check, so a skipped shot's feature misses
    never appeared in this report OR in `skipped` - the census that
    diagnosed this task's own C2 defect had to come from a fresh
    investigation instead of from this JSON.
    """
    import h5py

    n = 3
    fake_archive = {
        "x0": np.zeros((n, 11)), "x1": np.zeros((n, 33, 5)),
        "y": np.zeros((n, 2)), "rows": np.arange(n),
    }
    monkeypatch.setattr(
        validate, "archive_rows",
        lambda shot, archive=TM_ARCHIVE: fake_archive if shot == 111 else None,
    )
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    with h5py.File(features_dir / "111_features.h5", "w") as f:
        f.attrs["missing"] = '{"bt": "fdp:ToksearchUnavailable"}'

    report = validate.reconstruction_fidelity(
        "d3d_tearing_onset_cnn1d", [111], Paths(root=tmp_path)
    )
    assert report["n_shots_used"] == 0  # 111 fails match_rows (all-zero cols)
    assert report["incomplete_features"]["111"] == {"bt": "fdp:ToksearchUnavailable"}


def test_reconstruction_fidelity_asserts_the_match_column_mapping():
    """I10: MATCH_COLUMNS indexes d3d_tearing_onset_cnn1d's scalar order
    specifically. If it ever silently drifted from that (a spec reorder, or
    this function pointed at a different slug), the row alignment would
    compare the wrong columns and report a plausible-looking distance rather
    than failing. The check must run before any per-shot work, so it is a
    loud `ValueError`, not an entry in `skipped`.
    """
    import labelmaker.validate as validate_mod

    original = validate_mod.MATCH_COLUMNS
    validate_mod.MATCH_COLUMNS = (1, 0, 6, 7, 8)  # bt/ip swapped
    try:
        with pytest.raises(ValueError, match="MATCH_COLUMNS"):
            validate.reconstruction_fidelity(
                "d3d_tearing_onset_cnn1d", [185945], Paths.from_env()
            )
    finally:
        validate_mod.MATCH_COLUMNS = original


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
