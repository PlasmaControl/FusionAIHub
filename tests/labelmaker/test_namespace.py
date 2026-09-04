"""The canonical feature namespace is the package's contract with models."""
import numpy as np
import pytest

from labelmaker.features import namespace as ns


def test_grids_match_the_upstream_reference_file():
    # test_shots.h5 stores spatial_coordinates = 0, 1/32, ..., 1 and
    # times = 0, 25, ..., 5975 ms. The archive store is on the same grid.
    assert ns.RHO_GRID.shape == (33,)
    assert ns.RHO_GRID[0] == 0.0 and ns.RHO_GRID[-1] == 1.0
    np.testing.assert_allclose(np.diff(ns.RHO_GRID), 1.0 / 32.0)
    assert ns.GRID_S.shape == (240,)
    np.testing.assert_allclose(ns.GRID_S[[0, 1, -1]], [0.0, 0.025, 5.975])
    assert ns.STEP_S == 0.025


def test_specs_are_unique_and_well_formed():
    names = [f.name for f in ns.FEATURES]
    assert len(names) == len(set(names))
    for f in ns.FEATURES:
        assert f.name == f.name.lower()
        assert f.kind in ns.KINDS
        assert f.sources, f"{f.name} has no source"
        assert len(f.sources) == len(f.locators)
        assert set(f.sources) <= set(ns.SOURCES)
        assert len(set(f.sources)) == len(f.sources)


def test_every_feature_the_phase1_model_needs_exists():
    required = {
        "bt", "ip", "pinj_total", "tinj_total", "ech_power_total", "ech_rho",
        "r0", "kappa", "tritop", "tribot", "gapin", "betan",
        "qpsi", "pres", "ne_zipfit", "te_zipfit", "rot_zipfit",
    }
    assert required <= {f.name for f in ns.FEATURES}


def test_profiles_and_scalars_are_labelled_correctly():
    for name in ("qpsi", "pres", "ne_zipfit", "te_zipfit", "rot_zipfit"):
        assert ns.by_name(name).kind == "profile"
    for name in ("ip", "bt", "kappa", "ech_rho"):
        assert ns.by_name(name).kind == "scalar"


def test_lookup_helpers_and_locators():
    assert ns.by_name("ip").sources[0] == "archive"
    assert ns.by_name("pres").locator_for("archive") == "pres_EFIT01"
    assert ns.by_name("pinj_total").locator_for("corpus") == "pinj"
    assert all("archive" in f.sources for f in ns.by_source("archive"))
    assert {f.name for f in ns.by_source("corpus")} == {
        "pinj_total", "tinj_total", "ech_power_total"
    }
    assert "ip" in {f.name for f in ns.by_source("fdp")}
    assert "ech_rho" not in {f.name for f in ns.by_source("fdp")}
    with pytest.raises(KeyError):
        ns.by_name("no_such_feature")
    # ech_rho is the only Phase 1 feature with no second source, so it is the
    # only one whose locator_for("fdp") can raise.
    with pytest.raises(KeyError):
        ns.by_name("ech_rho").locator_for("fdp")


def test_source_preference_and_the_one_archive_only_feature():
    # The archive store comes first everywhere: for twelve features it is
    # bit-identical to what the Phase 1 model was trained on. ZIPFIT is
    # reachable through fdp too, which is what lets the pipeline leave the
    # archive's 21% of the corpus behind.
    for f in ns.FEATURES:
        assert f.sources[0] == "archive" or "archive" not in f.sources
    for name in ("ne_zipfit", "te_zipfit", "rot_zipfit"):
        assert ns.by_name(name).sources == ("archive", "fdp")
    # ech_rho is the one feature with no second source: TORBEAM deposition
    # locations exist only as the archived column.
    assert ns.by_name("ech_rho").sources == ("archive",)
