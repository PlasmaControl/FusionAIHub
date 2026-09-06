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
        "pinj_total", "tinj_total", "ech_power_total", "co2",
        "gas", "ece", "co2_r0", "co2_v1", "co2_v2", "co2_v3",
    }
    # The ELM model's inputs: one channel of a multi-valve group, every
    # channel of a 48-channel group, and one chord of the CO2 record.
    assert ns.by_name("gas").locator_for("corpus") == "gas_raw#0"
    assert ns.by_name("ece").kind == "profile"
    assert ns.by_name("ece").locator_for("corpus") == "ece"
    assert [ns.by_name(f"co2_{c}").locator_for("corpus")
            for c in ("r0", "v1", "v2", "v3")] == ["co2#0", "co2#1", "co2#2", "co2#3"]
    # The one waveform: 4 chords, corpus only, kept at the native rate.
    assert ns.by_name("co2").kind == "waveform"
    assert ns.by_name("co2").sources == ("corpus",)
    assert ns.by_name("co2").locator_for("corpus") == "co2"
    assert ns.by_name("co2").step == 0.0
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


def test_every_source_has_a_declared_sampling_convention():
    """`SAMPLING_BY_SOURCE` must be total over `ns.SOURCES`.

    Before this, `models.base.InputSpec.build` tested `resolver in
    ("corpus", "fdp")` directly - an opt-in against an open set, the same
    shape commit 38b137b fixed for the correction/fill split. A fourth
    source added to `SOURCES` without a matching entry here would silently
    fall through `sample_by_resolver` to "nearest", which is wrong for any
    future high-rate source and would misalign a shot by a full grid step
    with no warning.
    """
    for source in ns.SOURCES:
        assert source in ns.SAMPLING_BY_SOURCE, source
        assert ns.SAMPLING_BY_SOURCE[source] in ("nearest", "window")


def test_every_feature_lists_its_sources_in_global_preference_order():
    """`run.features_for_shot` walks `ns.SOURCES` in order for every feature
    at once, so a spec's own `sources` tuple is honoured only when it is
    ordered the same way. Asserted here so the two cannot silently disagree
    the day a feature prefers fdp over the archive.
    """
    rank = {s: i for i, s in enumerate(ns.SOURCES)}
    for spec in ns.FEATURES:
        ranks = [rank[s] for s in spec.sources]
        assert ranks == sorted(ranks), (spec.name, spec.sources)
