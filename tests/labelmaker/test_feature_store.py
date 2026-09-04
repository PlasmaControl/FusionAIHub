"""Feature files round-trip, merge, and record what was missing."""
import h5py
import numpy as np
import pytest

from labelmaker.features.store import (
    FeatureArray,
    is_complete,
    missing_names,
    present,
    read_feature,
    write_features,
)


def _scalar(n=10):
    x = 0.001 * np.arange(n)
    return FeatureArray(
        x=x, y=np.arange(n, dtype=float)[None, :],
        attrs={"resolver": "corpus"},
    )


def _profile(n=4):
    x = 0.025 * np.arange(n)
    y = np.tile(np.linspace(1.0, 0.0, 33)[:, None], (1, n))
    return FeatureArray(x=x, y=y, attrs={"resolver": "archive"})


def test_round_trip_scalar_and_profile(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar(), "ne_zipfit": _profile()}, {})
    got = read_feature(p, "ip")
    np.testing.assert_allclose(got.x, _scalar().x)
    np.testing.assert_allclose(got.y, _scalar().y)
    assert got.attrs["resolver"] == "corpus"
    assert got.attrs["units"] == "A"          # filled in from the namespace
    assert read_feature(p, "ne_zipfit").y.shape == (33, 4)
    assert present(p) == {"ip", "ne_zipfit"}


def test_stored_in_corpus_layout_with_provenance(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    with h5py.File(p, "r") as f:
        assert f["ip"]["xdata"].dtype == np.float64
        assert f["ip"]["ydata"].dtype == np.float32
        assert f["ip"]["ydata"].shape == (1, 10)
        assert f.attrs["shot"] == 190000
        assert f.attrs["labelmaker_version"] and f.attrs["git_sha"]
        assert f["ip"].attrs["complete"] == 1
        assert f["ip"].attrs["fetched_at"]


def test_profile_groups_carry_the_rho_grid(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ne_zipfit": _profile()}, {})
    with h5py.File(p, "r") as f:
        assert f["ne_zipfit"]["rho"].shape == (33,)


def test_missing_is_recorded_not_raised(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {}, {"bt": "KeyError", "pres": "TimeoutError"})
    assert missing_names(p) == {"bt": "KeyError", "pres": "TimeoutError"}
    assert present(p) == set()


def test_merge_keeps_earlier_groups_and_clears_resolved_misses(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {"bt": "KeyError"})
    write_features(p, 190000, {"bt": _scalar()}, {})
    assert present(p) == {"ip", "bt"}
    assert missing_names(p) == {}


def test_merge_false_replaces_the_file(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    write_features(p, 190000, {"bt": _scalar()}, {}, merge=False)
    assert present(p) == {"bt"}


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    assert list(tmp_path.iterdir()) == [p]


def test_is_complete_needs_every_name_present_or_missing(tmp_path):
    p = tmp_path / "190000_features.h5"
    assert not is_complete(p, ["ip"])          # no file yet
    write_features(p, 190000, {"ip": _scalar()}, {"bt": "KeyError"})
    assert is_complete(p, ["ip"])
    assert is_complete(p, ["ip", "bt"])        # a recorded miss counts
    assert not is_complete(p, ["ip", "pres"])


def test_mismatched_shapes_are_rejected_at_construction():
    with pytest.raises(ValueError):
        FeatureArray(x=np.zeros(3), y=np.zeros((1, 4)))
    with pytest.raises(ValueError):
        FeatureArray(x=np.zeros(3), y=np.zeros(3))     # must be (C, T)


def test_read_feature_raises_for_absent_group(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    with pytest.raises(KeyError):
        read_feature(p, "bt")
