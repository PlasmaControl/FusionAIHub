"""Feature files round-trip, merge, and record what was missing."""
import h5py
import numpy as np
import pytest

from labelmaker.features.store import (
    FeatureArray,
    is_complete,
    is_transient,
    missing_names,
    permanent_names,
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


def test_merge_preserves_existing_dataset_bytes_and_metadata(tmp_path):
    # Reconstructing old arrays through float64/float32 changes NaN payloads,
    # custom datasets, storage filters, and attribute types. Copy them intact.
    p = tmp_path / "190000_features.h5"
    bits = np.array([0x3F800000, 0x7F800001, 0x80000000], dtype=np.uint32)
    with h5py.File(p, "w") as f:
        f.attrs["custom"] = 42
        g = f.create_group("ip")
        g.attrs["complete"] = np.int32(1)
        g.attrs["calibration"] = np.array([1., 2.])
        g.create_dataset("xdata", data=np.array([0., 1., 2.], dtype=np.float32))
        g.create_dataset("ydata", data=bits.view(np.float32)[None], compression="gzip")
        g.create_dataset("quality", data=np.array([1, 0, 1], dtype=np.int8))
    write_features(p, 190000, {"bt": _scalar()}, {})
    with h5py.File(p, "r") as f:
        assert f.attrs["custom"] == 42
        assert f["ip/xdata"].dtype == np.float32
        assert f["ip/ydata"][:].tobytes() == bits.tobytes()
        assert f["ip/ydata"].compression == "gzip"
        assert f["ip/quality"][:].tolist() == [1, 0, 1]
        np.testing.assert_array_equal(f["ip"].attrs["calibration"], [1., 2.])


def test_failed_merge_leaves_original_file_byte_identical(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    before = p.read_bytes()
    bad = FeatureArray(x=np.arange(3), y=np.zeros((1, 3)), attrs={"bad": {}})
    with pytest.raises(TypeError):
        write_features(p, 190000, {"bt": bad}, {})
    assert p.read_bytes() == before
    assert list(tmp_path.iterdir()) == [p]


def test_merge_does_not_mark_a_retained_feature_missing(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    write_features(p, 190000, {}, {"ip": "TimeoutError", "bt": "KeyError"})
    assert present(p) == {"ip"}
    assert missing_names(p) == {"bt": "KeyError"}


def test_merge_false_replaces_the_file(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    write_features(p, 190000, {"bt": _scalar()}, {}, merge=False)
    assert present(p) == {"bt"}


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    assert list(tmp_path.iterdir()) == [p]


def test_a_failed_write_leaves_no_temp_file(tmp_path):
    # An attrs value h5py cannot serialise raises after the temp file is open,
    # which is the only case that matters: nothing else would ever remove it,
    # and over 16,909 shots that is a slow leak of files nobody recognises.
    p = tmp_path / "190000_features.h5"
    bad = FeatureArray(x=np.zeros(3), y=np.zeros((1, 3)), attrs={"nested": {"a": 1}})
    with pytest.raises(TypeError):
        write_features(p, 190000, {"ip": bad}, {})
    assert list(tmp_path.iterdir()) == []


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


def test_a_one_sample_feature_is_demoted_to_a_miss(tmp_path):
    # The corpus layout reads ydata.shape[-1] < 2 as "signal absent", so a
    # resolved one-sample group would be silently misread downstream. It is
    # recorded as a miss instead - and NOT raised on, which would cost this
    # shot the other feature resolved in the same call.
    p = tmp_path / "190000_features.h5"
    one = FeatureArray(x=np.zeros(1), y=np.zeros((1, 1)), attrs={"resolver": "corpus"})
    write_features(p, 190000, {"ip": one, "bt": _scalar()}, {})
    assert present(p) == {"bt"}
    assert "OneSampleAmbiguous" in missing_names(p)["ip"]


def test_write_features_leaves_the_callers_dicts_alone(tmp_path):
    p = tmp_path / "190000_features.h5"
    arrays = {"ip": FeatureArray(x=np.zeros(1), y=np.zeros((1, 1)))}
    missing: dict[str, str] = {}
    write_features(p, 190000, arrays, missing)
    assert set(arrays) == {"ip"} and missing == {}


def test_read_feature_raises_for_absent_group(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    with pytest.raises(KeyError):
        read_feature(p, "bt")


def test_a_transient_miss_does_not_count_as_complete(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {"bt": "fdp:TimeoutError"})
    assert not is_complete(p, ["ip", "bt"])                    # retry the timeout
    assert is_complete(p, ["ip", "bt"], retry_transient=False)  # unless told not to


def test_a_permanent_miss_counts_as_complete(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {"bt": "archive:KeyError"})
    assert is_complete(p, ["ip", "bt"])


def test_a_demoted_one_sample_feature_is_permanent(tmp_path):
    # A one-sample record is a property of the data, not of the attempt, so
    # it is not worth retrying - unlike a timeout.
    p = tmp_path / "190000_features.h5"
    one = FeatureArray(x=np.zeros(1), y=np.zeros((1, 1)))
    write_features(p, 190000, {"ip": one}, {})
    assert is_complete(p, ["ip"])


def test_a_cause_joined_across_sources_is_transient_if_any_source_is(tmp_path):
    # `features_for_shot` comma-joins one feature's causes across the sources
    # it tried, so a single cause string can mix a permanent miss with a
    # transient one. It counts as transient: the archive will not grow a
    # column it does not have, but the fdp fetch that timed out is worth one
    # more attempt, and the alternative is losing the feature for good.
    assert is_transient("archive:KeyError,fdp:TimeoutError")
    assert not is_transient("archive:KeyError,corpus:SignalAbsent")
    p = tmp_path / "190000_features.h5"
    write_features(
        p, 190000, {"ip": _scalar()},
        {"bt": "archive:KeyError,fdp:TimeoutError",
         "te_zipfit": "archive:KeyError,fdp:MdsException"},
    )
    assert permanent_names(p) == {"te_zipfit"}
    assert not is_complete(p, ["ip", "bt"])
    assert is_complete(p, ["ip", "te_zipfit"])


def test_the_causes_a_run_launched_without_fdp_run_records_are_transient(tmp_path):
    # MEASURED, and the reason this list is longer than the obvious one.
    # `resolve_fdp.available()` only checks that toksearch imports, which it
    # does here, so a run launched without the `fdp run` wrapper does NOT
    # report ToksearchUnavailable: it reaches the fetch and fails per signal.
    # Shot 189382 without the wrapper gives TreeFOPENR for kappa and
    # ne_zipfit and PtDataError for ip; with the wrapper all three fetch. If
    # those counted as permanent, one mis-launched bulk run would settle
    # "no fdp source" corpus-wide.
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {}, {
        "bt": "archive:KeyError,fdp:PtDataError",
        "kappa": "archive:KeyError,fdp:TreeFOPENR",
        "te_zipfit": "fdp:ToksearchUnavailable",
    })
    assert permanent_names(p) == set()
    assert not is_complete(p, ["bt", "kappa", "te_zipfit"])
    # A node that genuinely does not exist stays permanent, or nothing would
    # ever be settled.
    assert not is_transient("fdp:TreeNNF")


def _waveform(n=1_500_000, channels=4):
    """`co2`-shaped: over `CHUNK_THRESHOLD` samples, so it is stored chunked."""
    x = -1.45 + np.arange(n, dtype=np.float64) / 5.0e5
    y = np.tile(np.arange(channels, dtype=np.float32)[:, None], (1, n))
    return FeatureArray(x=x, y=y, attrs={"resolver": "corpus", "native_rate": "1"})


def test_a_large_waveform_round_trips_chunked_and_in_float32(tmp_path):
    path = tmp_path / "199000_features.h5"
    arr = _waveform()
    write_features(path, 199000, {"co2": arr}, {})
    with h5py.File(path, "r") as f:
        dset = f["co2"]["ydata"]
        assert dset.shape == (4, 1_500_000)
        assert dset.dtype == np.float32
        # Chunked, not contiguous: h5py reports `chunks` only when it is.
        assert dset.chunks is not None
        assert dset.chunks[0] == 1
        assert f["co2"].attrs["units"] == "cm^-2"
    got = read_feature(path, "co2")
    # float32 on the way back out, unlike every scalar and profile: a
    # (4, 4.5e6) waveform in float64 is 144 MB per shot for no measured gain.
    assert got.y.dtype == np.float32
    np.testing.assert_array_equal(got.y, arr.y)
    np.testing.assert_allclose(got.x, arr.x)


def test_a_merge_keeps_a_waveform_and_a_scalar_side_by_side(tmp_path):
    path = tmp_path / "199000_features.h5"
    write_features(path, 199000, {"co2": _waveform(n=1_100_000)}, {})
    write_features(path, 199000, {"ip": _scalar()}, {}, merge=True)
    assert present(path) == {"co2", "ip"}
    assert read_feature(path, "co2").y.shape == (4, 1_100_000)
    assert read_feature(path, "ip").y.dtype == np.float64


def test_a_small_array_is_still_stored_contiguously(tmp_path):
    path = tmp_path / "1_features.h5"
    write_features(path, 1, {"ip": _scalar()}, {})
    with h5py.File(path, "r") as f:
        assert f["ip"]["ydata"].chunks is None
