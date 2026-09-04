"""The archive resolver: 25 ms columns straight onto the canonical grid."""
import h5py
import numpy as np
import pytest

from labelmaker.features import namespace as ns
from labelmaker.features import resolve_archive as ra

REAL = ra.ARCHIVE_FILES[0]


def _fake_archive(tmp_path, n=240):
    p = tmp_path / "archive.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("190000")
        g.create_dataset("bt", data=np.full(n, 2.0))
        g.create_dataset("ip", data=np.linspace(0.0, 1e6, n))
        g.create_dataset("ech_pwr", data=np.full((1, n), 3.0))       # stored (1, N)
        g.create_dataset("pres_EFIT01", data=np.full((n, 33), 5.0e4))
        g.create_dataset("zipfit_edensfit_rho", data=np.tile(
            np.linspace(4.0, 1.0, 33), (n, 1)))
        f.create_group("190001").create_dataset("bt", data=np.full(n, 1.9))
    return p


def test_shot_index_maps_numeric_groups_to_their_file(tmp_path):
    p = _fake_archive(tmp_path)
    idx = ra.shot_index((p,))
    assert idx == {190000: p, 190001: p}


def test_scalars_and_profiles_land_on_the_canonical_grid(tmp_path):
    p = _fake_archive(tmp_path)
    got, missing = ra.resolve(190000, ["bt", "pres", "ne_zipfit"], files=(p,))
    assert missing == {}
    np.testing.assert_allclose(got["bt"].x, ns.GRID_S)
    assert got["bt"].y.shape == (1, 240)
    np.testing.assert_allclose(got["bt"].y[0], 2.0)
    assert got["pres"].y.shape == (33, 240)
    np.testing.assert_allclose(got["pres"].y, 5.0e4)
    assert got["ne_zipfit"].y.shape == (33, 240)
    np.testing.assert_allclose(got["ne_zipfit"].y[:, 0], np.linspace(4.0, 1.0, 33))
    assert got["bt"].attrs["resolver"] == "archive"
    assert got["bt"].attrs["locator"] == "bt"


def test_a_column_stored_with_a_leading_axis_is_squeezed(tmp_path):
    p = _fake_archive(tmp_path)
    got, _ = ra.resolve(190000, ["ech_power_total"], files=(p,))
    assert got["ech_power_total"].y.shape == (1, 240)
    np.testing.assert_allclose(got["ech_power_total"].y[0], 3.0)


def test_a_missing_column_is_recorded_per_feature(tmp_path):
    p = _fake_archive(tmp_path)
    got, missing = ra.resolve(190000, ["bt", "kappa"], files=(p,))
    assert set(got) == {"bt"}
    assert missing == {"kappa": "KeyError"}


def test_a_shot_outside_the_archive_misses_everything(tmp_path):
    p = _fake_archive(tmp_path)
    got, missing = ra.resolve(999999, ["bt", "pres"], files=(p,))
    assert got == {}
    assert missing == {"bt": "ShotNotInArchive", "pres": "ShotNotInArchive"}


def test_an_unknown_feature_name_is_refused(tmp_path):
    # Every Phase 1 feature has an archive source, so the refusal this
    # resolver can actually raise is for a name outside the namespace.
    p = _fake_archive(tmp_path)
    with pytest.raises(KeyError, match="no_such_feature"):
        ra.resolve(190000, ["no_such_feature"], files=(p,))


def test_a_profile_with_the_wrong_radial_width_is_refused(tmp_path):
    # The store's convention is (T, n_rho). A time-first column would
    # transpose into an array whose x and y lengths still agree, so it would
    # pass every check here and fail much later inside InputSpec.build.
    p = tmp_path / "archive.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("190000")
        g.create_dataset("pres_EFIT01", data=np.zeros((33, 240)))   # transposed
    got, missing = ra.resolve(190000, ["pres"], files=(p,))
    assert got == {}
    assert "RadialAxisMismatch" in missing["pres"]


def test_a_shorter_record_keeps_its_own_grid(tmp_path):
    p = _fake_archive(tmp_path, n=120)
    got, _ = ra.resolve(190000, ["bt"], files=(p,))
    assert got["bt"].y.shape == (1, 120)
    np.testing.assert_allclose(got["bt"].x[-1], 0.025 * 119)
    assert got["bt"].attrs["n_rows"] == "120"


@pytest.mark.skipif(not REAL.exists(), reason=f"archive not available: {REAL}")
def test_real_archive_resolves_every_archive_feature_for_an_overlap_shot():
    names = [f.name for f in ns.by_source("archive")]
    got, missing = ra.resolve(185945, names)
    assert missing == {}, missing
    for name in names:
        arr = got[name]
        assert arr.x.size == 240
        want_c = 33 if ns.by_name(name).kind == "profile" else 1
        assert arr.y.shape == (want_c, 240), name
    # pres is bit-identical to the model's training column; check the raw read
    with h5py.File(REAL, "r") as f:
        raw = np.asarray(f["185945"]["pres_EFIT01"])
    np.testing.assert_array_equal(got["pres"].y, raw.T)
