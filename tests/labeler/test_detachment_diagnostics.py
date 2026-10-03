"""Producer gates, provenance and honest detachment context availability."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.panels import detachment


def _paths(tmp_path):
    return Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")


def _bins(paths, **values):
    path = paths.root / "round4/detach/bins/170815.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, start_ms=[0.0, 50.0, 100.0], **values)
    return path


def _corpus(paths, groups):
    path = paths.corpus_file(170815)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as source:
        for name, (x, y) in groups.items():
            group = source.create_group(name)
            group["xdata"] = x
            group["ydata"] = y
    return path


def test_gas_subtracts_finite_preplasma_channel_means_without_a_threshold(tmp_path):
    paths = _paths(tmp_path)
    _corpus(
        paths,
        {
            "gas_flow": (
                [-0.002, -0.001, 0, 0.001, 0.002],
                [[150, 164, 157, 157, 157], [10, 14, 12, 32, 12]],
            )
        },
    )
    panel = detachment.panels(170815, paths=paths)[0]
    np.testing.assert_allclose(panel.y, [[0, 0, 0], [0, 20, 0]])
    assert panel.metadata["preplasma_baseline"] == [157, 12]


def test_gas_without_preplasma_samples_is_flagged_as_uncorrected(tmp_path):
    paths = _paths(tmp_path)
    _corpus(paths, {"gas_flow": ([0, 0.001], [[157, 157]])})
    panel = detachment.panels(170815, paths=paths)[0]
    assert "offset uncorrected" in panel.legend[0]
    assert panel.metadata["preplasma_baseline"] == [None]


def test_default_bins_keep_invalid_votes_and_afrac_guidance(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    paths = _paths(tmp_path)
    path = _bins(
        paths,
        afrac_value=[0.4, 0.6, 9.0],
        afrac_valid=[True, True, False],
        afrac_vote=[2, -1, -1],
        afrac_reason=["", "", "ramp"],
        prad_value=[np.nan, np.nan, np.nan],
        prad_valid=[False, False, False],
        prad_vote=[-1, -1, -1],
        prad_reason=["no_power", "no_power", "no_power"],
    )
    built = detachment.indicator_panels(170815, paths, t_range=(25.0, 120.0))
    assert len(built) == 2  # Invalidity is still useful when there is no trace.
    afrac, prad = built
    np.testing.assert_allclose(afrac.y, [[0.4, 0.6, np.nan]], equal_nan=True)
    assert list(afrac.hlines) == []  # No thresholds invented without a recipe.
    assert "uncalibrated" in afrac.metadata["caveat"].lower()
    assert "recipe" in afrac.metadata["interpretation"].lower()
    assert afrac.metadata["source"] == str(path)
    assert afrac.metadata["indicator"] == "afrac"
    assert afrac.metadata["valid"] == [True, True, False]
    assert afrac.metadata["vote"] == [2, -1, -1]
    assert afrac.metadata["reason"] == ["", "", "ramp"]
    assert afrac.metadata["bin_start_ms"] == [25.0, 50.0, 100.0]
    assert afrac.metadata["bin_end_ms"] == [50.0, 100.0, 120.0]
    assert prad.metadata["valid"] == [False, False, False]
    assert np.isnan(prad.y).all()


def test_default_legacy_csv_is_used_when_bin_file_is_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    paths = _paths(tmp_path)
    path = paths.root / "round4/detach/indicators/170815.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("t_ms,afrac,afrac_valid\n0,0.4,1\n50,0.9,1\n")
    panel = detachment.indicator_panels(170815, paths)[0]
    assert panel.metadata["source"] == str(path)
    np.testing.assert_allclose(panel.y, [[0.4, 0.9]])


@pytest.mark.parametrize(
    ("sources", "title_terms"),
    [
        (["surrogate"] * 3, ("model", "surrogate", "regression")),
        (["inversion"] * 3, ("inversion",)),
        (
            ["inversion", "surrogate", "none"],
            ("inversion", "model", "surrogate", "regression"),
        ),
    ],
)
def test_tangtv_title_and_bin_provenance_follow_source(
    tmp_path, monkeypatch, sources, title_terms
):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    paths = _paths(tmp_path)
    _bins(
        paths,
        tangtv_value=[0.2, 0.6, np.nan],
        tangtv_valid=[True, True, False],
        tangtv_vote=[1, 2, -1],
        tangtv_reason=["", "", "no_video"],
        tangtv_source=sources,
    )
    panel = detachment.indicator_panels(170815, paths)[0]
    assert all(term in panel.title.lower() for term in title_terms)
    assert panel.metadata["tangtv_source"] == sources
    assert panel.metadata["indicator"] == "tangtv"
    assert panel.metadata["valid"] == [True, True, False]


def test_tangtv_unknown_source_does_not_claim_an_inversion(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    paths = _paths(tmp_path)
    _bins(paths, tangtv_value=[0.2, 0.6, 0.7], tangtv_valid=[True] * 3)
    panel = detachment.indicator_panels(170815, paths)[0]
    assert "source not recorded" in panel.title.lower()
    assert panel.metadata["tangtv_source"] == ["unknown"] * 3


def test_aux_density_precedes_other_sources_and_te_is_independent(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    paths = _paths(tmp_path)
    path = _bins(paths, aux_ne=[1e14, 0.0, 3e14], aux_te_div=[25.0, 0.0, 11.0])
    _corpus(
        paths,
        {
            "co2": ([0.0, 0.05, 0.1], np.ones((4, 3))),
            "ts_core_density": ([0.0, 0.05, 0.1], np.ones((2, 3))),
        },
    )
    built = detachment.panels(170815, paths=paths, t_range=(0.0, 150.0))
    density = [p for p in built if "density" in p.title.lower()]
    assert len(density) == 1
    assert "producer aux_ne" in density[0].title
    assert density[0].ylabel == "native units (unverified)"
    assert density[0].metadata["source"] == str(path)
    assert density[0].metadata["measurement"] == "CO2 line-integrated density proxy"
    assert "V2" in density[0].metadata["source_selection"]
    np.testing.assert_allclose(density[0].y, [[1e14, np.nan, 3e14]], equal_nan=True)
    te = [p for p in built if "Thomson Te" in p.title]
    assert len(te) == 1 and "independent check" in te[0].title.lower()
    assert te[0].ylabel == "eV"
    assert te[0].metadata["independent_check"] is True
    assert "indicator" not in te[0].metadata
    np.testing.assert_allclose(te[0].y, [[25.0, np.nan, 11.0]], equal_nan=True)


def test_cache_r0_density_precedes_local_corpus_fallbacks(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    monkeypatch.delenv("LABELER_DETACHMENT_CACHE_ROOT", raising=False)
    paths = _paths(tmp_path)
    cache = paths.root / "round4/detach/cache/170815.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, denr0uf__t=[0.0, 50.0, 100.0], denr0uf__y=[1e14, 0, 2e14])
    _corpus(
        paths,
        {
            "co2": ([0.0, 0.05, 0.1], np.ones((4, 3))),
            "ts_core_density": ([0.0, 0.05, 0.1], np.ones((2, 3))),
        },
    )
    built = detachment.panels(170815, paths=paths, t_range=(0.0, 150.0))
    density = [p for p in built if "density" in p.title.lower()]
    assert len(density) == 1 and "DENR0UF" in density[0].title
    assert density[0].metadata["source"] == str(cache)
    np.testing.assert_allclose(density[0].y, [[1e14, np.nan, 2e14]], equal_nan=True)


def test_thomson_masks_nonpositive_native_samples_before_means(tmp_path):
    paths = _paths(tmp_path)
    _corpus(
        paths,
        {
            "ts_core_density": (
                np.arange(8) * 0.00025,
                [[0, 2e19, -1e19, 4e19, 0, 0, 0, 0]],
            )
        },
    )
    panel = detachment.panels(170815, paths=paths, t_range=(0.0, 2.0))[0]
    np.testing.assert_allclose(panel.y, [[3e19, np.nan]], equal_nan=True)
    assert panel.metadata["valid_fraction"] == [0.25]


def test_corpus_co2_masks_nonpositive_samples_and_discloses_native_units(tmp_path):
    paths = _paths(tmp_path)
    path = _corpus(
        paths,
        {
            "co2": (
                np.arange(8) * 0.00025,
                [[0, 2e14, -1e14, 4e14, 0, 0, 0, 0]],
            )
        },
    )
    panel = detachment.panels(170815, paths=paths, t_range=(0.0, 2.0))[0]
    np.testing.assert_allclose(panel.y, [[3e14, np.nan]], equal_nan=True)
    assert "R0 density proxy" in panel.title
    assert panel.ylabel == "native units (unverified)"
    assert panel.metadata["source"] == str(path)
    assert "line integral" in panel.metadata["caveat"]


def test_thomson_prefers_channels_with_most_positive_native_samples(tmp_path):
    paths = _paths(tmp_path)
    y = np.ones((10, 10)) * 1e19
    y[0, :9] = 0
    y[1, :8] = -1
    _corpus(paths, {"ts_core_density": (np.arange(10) * 0.001, y)})
    panel = detachment.panels(170815, paths=paths, t_range=(0.0, 10.0))[0]
    assert panel.legend == [f"core channel {i} (m^-3)" for i in range(2, 10)]
    assert panel.metadata["valid_fraction"] == [1.0] * 8
    assert "valid fraction" in panel.metadata["channel_selection"]


def test_filterscopes_reject_nonpositive_and_offset_chords(tmp_path):
    paths = _paths(tmp_path)
    y = np.ones((8, 10))
    y[0] = -5.0
    y[1] = 0.0
    y[2] = [-5.0] * 9 + [1.0]  # One spike does not make a live positive chord.
    path = _corpus(paths, {"filterscopes": (np.arange(10) * 0.001, y)})
    built = detachment.panels(170815, paths=paths, t_range=(0.0, 10.0))
    assert [p.legend[0] for p in built] == [
        f"FS{i:02d} (ph/(sr cm2 s))" for i in range(4, 9)
    ]
    for panel in built:
        assert "positive median" in panel.metadata["channel_policy"]
        assert "signals.yaml" in panel.metadata["units_source"]
        assert panel.metadata["source"] == str(path)
