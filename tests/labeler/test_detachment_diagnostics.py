"""Producer gates, provenance and honest detachment context availability."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.panels import detachment
from labeler.events.review import panel_rows
from labeler.events.review.rows import Grid


@pytest.mark.parametrize(
    "values,valid,warn",
    [
        ([0.2, 0.54], [True, True], False),
        ([0.2, 1.88], [True, True], True),
        ([0.2, 1.88], [True, False], False),
    ],
)
def test_radiation_export_note_only_warns_for_valid_ratios_above_one(
    values, valid, warn
):
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/labeler/detachment_review_export.py"
    )
    spec = importlib.util.spec_from_file_location("review_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    note = module.radiation_note(
        {"prad_value": np.array(values), "prad_valid": np.array(valid)}
    )
    assert ("orange points >1" in note) is warn


pytestmark = pytest.mark.usefixtures("detachment_inputs")


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


def test_gas_subtracts_preplasma_median_despite_prefill_outliers(tmp_path):
    paths = _paths(tmp_path)
    _corpus(
        paths,
        {
            "gas_flow": (
                [-0.004, -0.003, -0.002, -0.001, 0, 0.001, 0.002],
                [[157, 157, np.nan, 900, 157, 157, 157], [12, 12, 12, 100, 12, 32, 12]],
            )
        },
    )
    _, panel = detachment.panels(170815, paths=paths)  # after the empty Afrac row
    assert panel.title == "Gas total (valve sum)"
    np.testing.assert_allclose(panel.y, [[0, 20, 0]])  # GASA + GASB
    assert panel.metadata["valves"] == ["GASA", "GASB"]
    assert panel.metadata["preplasma_baseline"] == [157, 12]
    assert "GASTOTAL" in panel.metadata["caveat"]
    assert "median over t < 0" in panel.metadata["baseline_policy"]


def test_gas_without_preplasma_samples_is_flagged_as_uncorrected(tmp_path):
    paths = _paths(tmp_path)
    _corpus(paths, {"gas_flow": ([0, 0.001], [[157, 157]])})
    panel = detachment.panels(170815, paths=paths)[1]  # after the empty Afrac
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
    # Valid bins as before; the invalid bin's ungated line is missing here (no
    # positioned probes), and the caveat says so.
    np.testing.assert_allclose(
        afrac.y, [[0.4, 0.6, np.nan], [np.nan] * 3], equal_nan=True
    )
    assert afrac.legend == ["Afrac lower (dimensionless)", "ungated, invalid bins"]
    assert "No ungated values: no positioned Langmuir probes" in afrac.metadata["caveat"]
    assert afrac.metadata["ungated"] == [False, False, False]
    # The readout still names the invalid bin's reason.
    assert afrac.metadata["afrac_method"][2].startswith("invalid (ramp); ungated: no value")
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


def test_legacy_csv_indicators_have_ratio_units(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    paths = _paths(tmp_path)
    path = paths.root / "round4/detach/indicators/170815.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "t_ms,prad_div,prad_div_valid,tangtv_front_height,tangtv_front_height_valid\n"
        "0,0.2,1,0.4,1\n50,1.2,1,0.8,1\n"
    )
    prad, tangtv = detachment.indicator_panels(170815, paths)
    assert prad.ylabel == tangtv.ylabel == "dimensionless"
    assert prad.hlines == [1.0]
    assert "denominator" in prad.metadata["caveat"]


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
def test_tangtv_caveat_and_bin_provenance_follow_source(
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
    assert panel.title == "TangTV front DZ"
    assert all(term in panel.metadata["caveat"].lower() for term in title_terms)
    assert panel.metadata["tangtv_source"] == sources
    assert panel.metadata["indicator"] == "tangtv"
    assert panel.metadata["valid"] == [True, True, False]


def test_tangtv_unknown_source_does_not_claim_an_inversion(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    paths = _paths(tmp_path)
    _bins(paths, tangtv_value=[0.2, 0.6, 0.7], tangtv_valid=[True] * 3)
    panel = detachment.indicator_panels(170815, paths)[0]
    assert panel.title == "TangTV front DZ"
    assert "source not recorded" in panel.metadata["caveat"].lower()
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
    built = detachment.indicator_panels(170815, paths, t_range=(0.0, 150.0))
    density = [p for p in built if "density" in p.title.lower()]
    assert len(density) == 1
    assert density[0].title == "CO2 density (aux_ne)"
    assert density[0].ylabel == "m·cm⁻³"
    assert density[0].metadata["source"] == str(path)
    assert density[0].metadata["measurement"] == "CO2 line-integrated density proxy"
    assert "V2" in density[0].metadata["source_selection"]
    np.testing.assert_allclose(density[0].y, [[1e14, np.nan, 3e14]], equal_nan=True)
    te = [p for p in built if "Thomson Te" in p.title]
    assert len(te) == 1 and te[0].title == "Divertor Thomson Te"
    assert "independent temperature check" in te[0].metadata["caveat"].lower()
    assert te[0].ylabel == "eV"
    assert te[0].metadata["independent_check"] is True
    assert "indicator" not in te[0].metadata
    np.testing.assert_allclose(te[0].y, [[25.0, np.nan, 11.0]], equal_nan=True)
    # Plan D.B: density and Thomson context left the default rows; the Afrac row
    # stays, empty, and says why.
    [afrac] = detachment.panels(170815, paths=paths, t_range=(0.0, 150.0))
    assert afrac.title == "Afrac lower" and np.isnan(afrac.y).all()
    assert "No value on this shot: the producer bins hold no Afrac" in (
        afrac.metadata["caveat"]
    )


def _prad(path, scale=1.0, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        prad_divl__t=[0.0, 50.0, 100.0, 200.0],
        prad_divl__y=np.array([1e6, np.nan, 3e6, 9e6]) * scale,
        prad_divu__t=[0.0, 50.0, 100.0, 200.0],
        prad_divu__y=np.array([2e5, 4e5, 6e5, 8e5]) * scale,
        **extra,
    )


def test_prad_rows_show_megawatts_from_the_cache_then_the_default_fetch(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LABELER_DETACHMENT_CACHE_ROOT", raising=False)
    paths = _paths(tmp_path)
    cache = paths.root / "round4/detach/cache/170815.npz"
    fetched = paths.root / "detach_default/170815.npz"
    _prad(fetched, scale=2.0)
    np.savez(cache, denr0uf__t=[0.0, 50.0], denr0uf__y=[1e14, 2e14])
    afrac, lower, upper = detachment.panels(170815, paths=paths, t_range=(0.0, 150.0))
    assert afrac.title == "Afrac lower" and np.isnan(afrac.y).all()  # no bins
    assert (lower.title, upper.title) == ("Prad,div lower", "Prad,div upper")
    assert lower.ylabel == "MW" and lower.legend == ["PRAD_DIVL (MW)"]
    assert lower.metadata["node"] == r"\BOLOM::PRAD_DIVL"
    assert lower.metadata["source"] == str(fetched)  # the cache lacks Prad
    np.testing.assert_allclose(lower.y, [[2, np.nan, 6]], equal_nan=True)
    assert "f_div" in lower.metadata["caveat"] and "W" in lower.metadata["units_note"]
    _prad(cache)
    _, lower, upper = detachment.panels(170815, paths=paths, t_range=(0.0, 150.0))
    assert lower.metadata["source"] == str(cache)  # the producer cache wins
    np.testing.assert_allclose(upper.y, [[0.2, 0.4, 0.6]])
    np.testing.assert_allclose(upper.x, [0, 50, 100])


def test_prad_without_a_plasma_window_starts_at_the_shot(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_CACHE_ROOT", raising=False)
    paths = _paths(tmp_path)
    path = paths.root / "detach_default/170815.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, prad_divl__t=[-4963.0, -10.0, 0.0, 50.0], prad_divl__y=[5e5] * 4)
    [lower] = detachment.prad_panels(170815, paths)
    np.testing.assert_allclose(lower.x, [0, 50])  # BOLOM's pre-shot clock is cut


def test_default_rows_follow_plan_db_and_leave_the_chords_out(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    monkeypatch.delenv("LABELER_DETACHMENT_CACHE_ROOT", raising=False)
    paths = _paths(tmp_path)
    _bins(
        paths,
        afrac_value=[0.4, 0.6, 0.9],
        afrac_valid=[True, True, True],
        prad_value=[0.2, 0.5, 1.2],
        prad_valid=[True, True, True],
        tangtv_value=[0.1, 0.2, 0.3],
        tangtv_valid=[True, True, True],
        aux_ne=[1e14, 2e14, 3e14],
    )
    _prad(paths.root / "round4/detach/cache/170815.npz")
    t = np.arange(10) * 0.01
    _corpus(
        paths,
        {
            "bolo": (t, np.ones((48, 10))),
            "filterscopes": (t, np.ones((8, 10))),
            "co2": (t, np.ones((4, 10))),
            "ts_core_density": (t, np.ones((2, 10))),
            "gas_flow": (t, np.ones((11, 10))),
        },
    )
    built = detachment.panels(170815, paths=paths, t_range=(0.0, 150.0))
    assert [p.title for p in built] == [
        "Afrac lower",
        "Prad,div lower",
        "Prad,div upper",
        "Gas total (valve sum)",
    ]
    assert all(p.kind != "heatmap" for p in built)
    assert not any("bolo" in p.title.lower() or "D-alpha" in p.title for p in built)
    assert built[0].metadata["indicator"] == "afrac"
    assert not any(p.metadata.get("indicator") == "prad" for p in built)


def _flux_map(paths, shot=170815):
    """psiN = ((R - 1.7) / 0.5)^2 + (Z / 0.9)^2 on a close EFIT02 clock."""
    r = np.linspace(1.0, 2.4, 57)
    z = np.linspace(-1.4, 1.4, 57)
    psin = ((r[None, :] - 1.7) / 0.5) ** 2 + (z[:, None] / 0.9) ** 2
    gtime = np.arange(0.0, 201.0, 25.0)
    path = paths.root / "round4/detach/efit" / f"{shot}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        gtime_ms=gtime,
        r=r,
        z=z,
        psirz=np.broadcast_to(psin, (len(gtime), *psin.shape)),
        ssimag=np.zeros(len(gtime)),
        ssibry=np.ones(len(gtime)),
        source=np.asarray("EFIT02"),
    )


def _ungated_inputs(paths, monkeypatch, *, reference=True):
    """Four 50 ms bins; probe 1 sits on the separatrix with Jsat 4, 2, 1, 3 per bin
    and probe 2 well outside it. ne = 1 and P_SOL = 1 MW, so a bin's ratio is
    Jsat * 1e6^(3/7)."""
    from labeler.events.detachment import signals

    monkeypatch.setattr(signals, "CORPUS", paths.corpus)  # never the real corpus
    scale = 1e6 ** (3 / 7)
    values = {
        "afrac_value": [0.8, np.nan, np.nan, np.nan],
        "afrac_valid": [True, False, False, False],
        "afrac_vote": [2, -1, -1, -1],
        "afrac_reason": ["", "short_reference", "ramp", "probe_off_separatrix"],
        "afrac_method": ["per_probe_reference"] * 4,
        "aux_jsat_selected_probe": np.array([1, 1, 1, 2], dtype=np.int16),
    }
    if reference:
        values["aux_jsat_reference"] = [5 * scale, np.nan, np.nan, np.nan]
    path = paths.root / "round4/detach/bins/170815.npz"
    np.savez(path, start_ms=[0.0, 50.0, 100.0, 150.0], **values)
    t = np.arange(5.0, 200.0, 10.0)
    probes = paths.root / "round4/detach/processed_probes/170815.npz"
    probes.parent.mkdir(parents=True)
    np.savez(
        probes,
        P1_t_ms=t, P1_jsat=np.repeat([4.0, 2.0, 1.0, 3.0], 5), P1_rz=[2.2, 0.0],
        P2_t_ms=t, P2_jsat=np.full(20, 9.0), P2_rz=[2.3, 0.0],
    )
    clock = np.arange(-40.0, 241.0, 20.0)
    np.savez(
        paths.root / "round4/detach/cache/170815.npz",
        denv2uf__t=t, denv2uf__y=np.ones(20),
        poh__t=clock, poh__y=np.full(len(clock), 1e6),
        wmhd__t=clock, wmhd__y=np.full(len(clock), 3e5),
        pinj_total__t=np.arange(-60.0, 261.0), pinj_total__y=np.zeros(321),
    )
    _flux_map(paths)
    return scale


def test_invalid_bins_show_the_ungated_ratio_with_their_reason(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    monkeypatch.delenv("LABELER_DETACHMENT_CACHE_ROOT", raising=False)
    paths = _paths(tmp_path)
    _ungated_inputs(paths, monkeypatch)
    [afrac] = detachment.panels(170815, paths=paths, t_range=(0.0, 200.0))
    # Valid bins as the producer drew them; invalid bins on the second line, read
    # on probe 1 (nearest the separatrix) against the producer's own reference.
    np.testing.assert_allclose(
        afrac.y, [[0.8, np.nan, np.nan, np.nan], [np.nan, 0.4, 0.2, 0.6]],
        equal_nan=True, rtol=1e-5,
    )
    assert afrac.metadata["ungated"] == [False, True, True, True]
    assert afrac.metadata["valid"] == [True, False, False, False]
    methods = afrac.metadata["afrac_method"]
    assert methods[0] == "per_probe_reference"
    assert methods[1].startswith("invalid (short_reference); ungated: P1, |psiN-1| 0.000")
    assert "producer reference" in methods[1] and "EFIT02" in methods[1]
    assert methods[2].startswith("invalid (ramp)") and "no ELM mask" in methods[2]
    assert "UNGATED" in afrac.metadata["caveat"]
    assert "No ungated values" not in afrac.metadata["caveat"]


def test_without_a_producer_reference_the_probe_sets_its_own(tmp_path, monkeypatch):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    monkeypatch.delenv("LABELER_DETACHMENT_CACHE_ROOT", raising=False)
    paths = _paths(tmp_path)
    _ungated_inputs(paths, monkeypatch, reference=False)
    [afrac] = detachment.panels(170815, paths=paths, t_range=(0.0, 200.0))
    reference = np.quantile([4.0, 2.0, 1.0, 3.0], 0.9)  # ungated, near the separatrix
    np.testing.assert_allclose(
        afrac.y[1], [np.nan, 2 / reference, 1 / reference, 3 / reference],
        equal_nan=True, rtol=1e-5,
    )
    assert "ungated near-separatrix reference" in afrac.metadata["afrac_method"][1]


def test_a_shot_without_bins_keeps_an_empty_afrac_row_with_its_reason(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    paths = _paths(tmp_path)
    [afrac] = detachment.panels(170815, paths=paths, t_range=(100.0, 900.0))
    assert afrac.title == "Afrac lower" and afrac.metadata["indicator"] == "afrac"
    np.testing.assert_allclose(afrac.x, [100.0, 900.0])
    assert np.isnan(afrac.y).all()
    assert "No value on this shot: the producer has no bins" in afrac.metadata["caveat"]
    assert detachment.panels(170815, paths=paths) == []  # no window, nowhere to draw
    _prad(paths.root / "detach_default/170815.npz")
    afrac, *rest = detachment.panels(170815, paths=paths)
    assert [p.title for p in rest] == ["Prad,div lower", "Prad,div upper"]
    np.testing.assert_allclose(afrac.x, [0.0, 200.0])  # spans the other rows
    assert afrac.metadata["caveat"].endswith("(not in its roster).")


def test_the_gas_total_needs_every_listed_valve(tmp_path):
    paths = _paths(tmp_path)
    _corpus(
        paths,
        {"gas_flow": ([0.0, 0.001, 0.002], [[1.0, 1.0, 1.0], [2.0, np.nan, 2.0]])},
    )
    _, panel = detachment.panels(170815, paths=paths)  # after the empty Afrac row
    np.testing.assert_allclose(panel.y, [[3.0, np.nan, 3.0]], equal_nan=True)
    assert "any of them" in panel.metadata["sum_policy"]


def _dts(paths, *, z=(0.99, -1.24, -1.22, -1.18), dead=1, profiles=None):
    """The blessed divertor Te: a profile every 20 ms from t = 10 ms, chord k at
    10 (k + 1) eV times the profile number; chord `dead` has no fit (0)."""
    t = 10.0 + 20.0 * np.arange(profiles or 5)
    temp = 10.0 * (np.arange(len(z))[:, None] + 1) * (np.arange(len(t)) + 1)
    temp[dead] = 0.0
    path = paths.root / "detach_default/dts_blessed/170815.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, temp=temp.astype(np.float32), t_ms=t.astype(np.float32),
             r_m=np.full(len(z), 1.485, dtype=np.float32),
             z_m=np.asarray(z, dtype=np.float32), phi_deg=np.full(len(z), 120.0),
             units=np.asarray("eV"))
    return temp


def test_divertor_te_is_a_z_by_time_map_of_the_blessed_chords(tmp_path):
    paths = _paths(tmp_path)
    temp = _dts(paths)
    panel = detachment.dts_panel(170815, paths, t_range=(0.0, 95.0))
    assert panel.kind == "heatmap" and panel.title == "Divertor Te (blessed)"
    assert panel.ylabel == "Z (m)" and (panel.zmin, panel.zmax) == (0.5, 100.0)
    np.testing.assert_allclose(panel.x, [10, 30, 50, 70, 90])
    np.testing.assert_allclose(np.diff(panel.y), 0.005)  # uniform for the image row
    assert panel.z.shape == (len(panel.y), 5)
    assert panel.metadata["z_scale"] == "log" and panel.metadata["units"] == "eV"
    assert panel.metadata["z_units"] == "eV"
    # The dead chord (-1.24) is left out; the others keep their order in Z.
    assert panel.metadata["chords"] == 3
    assert panel.metadata["chord_z_m"] == pytest.approx([-1.22, -1.18, 0.99])

    def at(z):
        return panel.z[int(np.argmin(np.abs(panel.y - z)))]

    np.testing.assert_allclose(at(-1.22), temp[2])
    np.testing.assert_allclose(at(-1.18), temp[3])
    np.testing.assert_allclose(at(0.99), temp[0])
    # A chord reaches 1.5 cm: -1.21 is the bins of -1.22, -1.17 those of -1.18; the
    # 4 cm between them (-1.20) and the 2 m between the arrays stay empty.
    np.testing.assert_allclose(at(-1.21), temp[2])
    np.testing.assert_allclose(at(-1.17), temp[3])
    assert np.isnan(at(-1.20)).all() and np.isnan(at(-1.0)).all()
    assert np.isnan(at(0.5)).all()
    assert panel.y[0] - 0.0025 <= -1.22 - 0.015 + 1e-9  # the first bin holds the reach
    assert panel.y[-1] + 0.0025 >= 0.99 + 0.015 - 1e-9
    caveat = panel.metadata["caveat"]
    assert caveat.startswith("Blessed divertor Thomson Te") and "0.5 to 100 eV" in caveat
    # The lone upper chord (a minor array) lies outside the displayed band, which
    # covers the lower array's bins; its values stay in z.
    [(low, high)] = panel.bands
    assert panel.metadata["band_z_m"] == [low, high]
    assert low <= -1.22 and high >= -1.18 and high < 0.5
    assert panel.metadata["chords_outside_band_z_m"] == pytest.approx([0.99])
    assert "1 of the 3 chords (Z +0.99 m, a minor array)" in caveat


def test_the_band_follows_the_main_array_and_two_big_arrays_show_everything(tmp_path):
    lower = list(np.linspace(-1.24, -1.01, 14))
    upper = list(np.linspace(1.0, 1.1, 6))
    # An upper-divertor shot: five live upper chords (the first has no fit) and
    # three lower ones left over.
    paths = _paths(tmp_path / "usn")
    _dts(paths, z=[*upper, -1.05, -1.03, -1.01], dead=0, profiles=4)
    [(low, high)] = detachment.dts_panel(170815, paths, (0.0, 95.0)).bands
    assert low > 0.9 and high >= 1.1
    # Two arrays of at least five chords: no band, the whole axis.
    both = _paths(tmp_path / "both")
    _dts(both, z=[*lower[:6], *upper], dead=0, profiles=4)
    panel = detachment.dts_panel(170815, both, (0.0, 95.0))
    assert len(panel.bands) == 0 and panel.metadata["band_z_m"] is None
    assert panel.metadata["chords_outside_band_z_m"] == []
    # One array (no gap above 0.3 m): no band either.
    single = _paths(tmp_path / "single")
    _dts(single, z=lower, dead=0, profiles=4)
    assert len(detachment.dts_panel(170815, single, (0.0, 95.0)).bands) == 0


def test_a_zero_fit_is_empty_and_a_window_without_two_profiles_has_no_row(tmp_path):
    paths = _paths(tmp_path)
    temp = _dts(paths, z=(-1.2, -1.15), dead=0)
    temp[1, 2] = 0.0  # chord 1 has no fit at the third profile
    np.savez(
        paths.root / "detach_default/dts_blessed/170815.npz",
        temp=temp.astype(np.float32), t_ms=np.array([10.0, 30, 50, 70, 90], np.float32),
        r_m=np.full(2, 1.485), z_m=np.array([-1.2, -1.15]), phi_deg=np.full(2, 120.0),
    )
    panel = detachment.dts_panel(170815, paths)
    assert panel.metadata["units"] == "eV"  # the default when the file has none
    column = panel.z[int(np.argmin(np.abs(panel.y + 1.15)))]
    np.testing.assert_allclose(column, [20, 40, np.nan, 80, 100], equal_nan=True)
    assert detachment.dts_panel(170815, paths, t_range=(10.0, 20.0)) is None  # one profile
    assert detachment.dts_panel(170815, paths, t_range=(200.0, 300.0)) is None
    assert detachment.dts_panel(170815, _paths(tmp_path / "none")) is None


def test_a_blessed_file_whose_shapes_disagree_is_refused(tmp_path):
    paths = _paths(tmp_path)
    _dts(paths)
    path = paths.root / "detach_default/dts_blessed/170815.npz"
    np.savez(path, temp=np.ones((2, 5)), t_ms=np.arange(5.0), r_m=np.ones(3),
             z_m=np.ones(3), phi_deg=np.ones(3))
    with pytest.raises(ValueError, match="clock and chords disagree"):
        detachment.dts_panel(170815, paths)


def _irtv(paths, quantity="heat_flux", offset=0.0):
    """10 pixels up the centre post at R = 104.7 cm, then 20 along the floor; the
    heat flux is R - 100 W/cm^2 and the temperature R + 200 deg C (`offset` shifts
    either)."""
    r = np.r_[np.full(10, 104.7), np.linspace(105.1, 114.6, 20)]
    z = np.r_[np.linspace(-80.0, -110.0, 10), np.full(20, -125.0)]
    t = np.arange(0.5, 100.0, 1.0) + 0.04  # block mean times, off the centre
    heat = quantity == "heat_flux"
    value = (r - 100.0) if heat else (r + 200.0)
    data = np.broadcast_to(value + offset, (len(t), len(r))).astype(np.float16)
    path = paths.root / "detach_default" / ("irtv" if heat else "irtv_temp") / "170815.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{quantity: data}, t_ms=t.astype(np.float32), r_cm=r, z_cm=z,
             node=np.asarray(r"\IRTVQ2DR2" if heat else r"\IRTVAT2DR2"),
             system=np.asarray("DIGITAL_CAM"),
             units=np.asarray("W/cm^2" if heat else "Degrees C"))


def test_irtv_heat_flux_is_an_r_by_time_map_without_the_centre_post(tmp_path):
    paths = _paths(tmp_path)
    _irtv(paths)
    panel = detachment.irtv_panel(170815, paths, t_range=(10.0, 60.0))
    assert panel.kind == "heatmap" and panel.title == "IRTV heat flux (W/cm²)"
    assert panel.ylabel == "R (cm)" and panel.zmin == 0.0
    np.testing.assert_allclose(panel.x, np.arange(10.5, 60.0, 1.0))
    np.testing.assert_allclose(np.diff(panel.y), 0.5)  # uniform for the image row
    assert panel.y[0] == pytest.approx(104.75)
    assert panel.z.shape == (len(panel.y), len(panel.x))
    assert np.isnan(panel.z[0]).all()  # the centre post, 30 cm tall in one bin
    assert panel.metadata["wall_left_out"] == [
        {"r_cm": [104.5, 105.0], "z_cm": [-110.0, -80.0], "pixels": 10}
    ]
    finite = np.isfinite(panel.z)
    row = np.where(finite, panel.z, 0).sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
    row[~finite.any(axis=1)] = np.nan  # bins without pixels stay empty
    assert np.isfinite(row[1:]).sum() >= 15 and np.nanmin(row[1:]) > 5
    assert panel.metadata["node"] == r"\IRTVQ2DR2" and panel.metadata["units"] == "W/cm²"
    assert panel.metadata["z_units"] == "W/cm²" and "z_scale" not in panel.metadata


def test_irtv_temperature_is_the_heat_flux_maps_twin(tmp_path):
    paths = _paths(tmp_path)
    _irtv(paths)
    _irtv(paths, "temperature")
    flux = detachment.irtv_panel(170815, paths, t_range=(10.0, 60.0))
    panel = detachment.irtv_panel(170815, paths, (10.0, 60.0), "temperature")
    assert panel.kind == "heatmap" and panel.title == "IRTV temperature (°C)"
    assert panel.ylabel == "R (cm)" and panel.zmin is None and panel.zmax is None
    # The same time blocks, radial bins and left-out wall as the heat flux.
    np.testing.assert_array_equal(panel.x, flux.x)
    np.testing.assert_array_equal(panel.y, flux.y)
    assert panel.metadata["wall_left_out"] == flux.metadata["wall_left_out"]
    np.testing.assert_array_equal(np.isnan(panel.z), np.isnan(flux.z))
    # R + 200 against R - 100: the same bin means 300 apart (float16 steps of 0.1).
    both = np.isfinite(panel.z) & np.isfinite(flux.z)
    assert both.sum() > 300
    np.testing.assert_allclose((panel.z - flux.z)[both], 300.0, atol=0.3)
    assert panel.metadata["node"] == r"\IRTVAT2DR2"
    assert panel.metadata["units"] == "°C"
    assert panel.metadata["z_units"] == "°C" and "z_scale" not in panel.metadata
    assert panel.metadata["quantity"] == "absolute surface temperature"
    assert "1st to 99.5th percentile" in panel.metadata["caveat"]
    assert detachment.irtv_panel(170815, _paths(tmp_path / "none"), None, "temperature") is None


def test_default_rows_put_the_blessed_te_then_both_irtv_maps_after_the_gas_total(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LABELER_DETACHMENT_INDICATORS", raising=False)
    monkeypatch.delenv("LABELER_DETACHMENT_CACHE_ROOT", raising=False)
    paths = _paths(tmp_path)
    _bins(paths, afrac_value=[0.4, 0.6, 0.9], afrac_valid=[True, True, True])
    _prad(paths.root / "round4/detach/cache/170815.npz")
    _corpus(paths, {"gas_flow": (np.arange(10) * 0.01, np.ones((11, 10)))})
    _dts(paths)
    _irtv(paths)
    _irtv(paths, "temperature")
    built = detachment.panels(170815, paths=paths, t_range=(0.0, 95.0))
    assert [p.title for p in built] == [
        "Afrac lower", "Prad,div lower", "Prad,div upper", "Gas total (valve sum)",
        "Divertor Te (blessed)", "IRTV heat flux (W/cm²)", "IRTV temperature (°C)",
    ]
    # The three maps' image rows say what the colour means and how it is scaled.
    rows = {
        p.title: panel_rows._image(f"p{i}", p, np.asarray(p.x), Grid(0.0, 10.0, 10))
        for i, p in enumerate(built) if p.kind == "heatmap"
    }
    assert {t: (r.z_units, r.z_scale) for t, r in rows.items()} == {
        "Divertor Te (blessed)": ("eV", "log"),
        "IRTV heat flux (W/cm²)": ("W/cm²", "linear"),
        "IRTV temperature (°C)": ("°C", "linear"),
    }
    te = rows["Divertor Te (blessed)"]
    assert (te.z_lo, te.z_hi) == pytest.approx((0.5, 100.0))  # linear values, as pinned
    # The retired real-time tssdivte files, if present, add nothing.
    old = paths.root / "detach_default/dts/170815.npz"
    old.parent.mkdir(parents=True)
    np.savez(old, tssdivte01__t=[0.0, 20.0], tssdivte01__y=[1.0, 2.0])
    assert [p.title for p in detachment.panels(170815, paths=paths, t_range=(0.0, 95.0))] == [
        p.title for p in built
    ]


def _fetch_script():
    path = (
        Path(__file__).resolve().parents[2] / "scripts/labeler/detach_default_fetch.py"
    )
    spec = importlib.util.spec_from_file_location("detach_default_fetch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "text,auth",
    [
        ("TreeNNF: node not found on shot 140123", False),
        ("PtDataError: 4013 bytes read", False),
        ("HTTP 401 Unauthorized", True),
        ("SciToken expired: token rejected", True),
        ("Authentication failed", True),
        ("getservbyname failed for task 'PTSERVER'", False),
    ],
)
def test_the_fetch_stops_only_on_whole_auth_words(text, auth):
    module = _fetch_script()
    if auth:
        with pytest.raises(module.AuthError):
            module.checked(RuntimeError(text))
    else:
        assert text in module.checked(RuntimeError(text))


def test_the_fetch_log_resumes_by_part(tmp_path):
    module = _fetch_script()
    log = tmp_path / "fetch_log.jsonl"
    log.write_text(
        '{"shot": 1, "done": true}\n{"shot": 2, "done": false}\n'
        '{"shot": 1, "done_parts": ["dts"], "done": false}\n'
    )
    assert module.logged_parts(log) == {1: {"equilibrium", "prad", "dts"}, 2: set()}


def test_the_default_fetch_parts_leave_the_real_time_dts_out():
    module = _fetch_script()
    assert module.DEFAULT_PARTS == ("equilibrium", "prad", "irtv", "dts_blessed", "irtv_temp")
    assert "dts" in module.PARTS and "dts" not in module.DEFAULT_PARTS
    assert module.IRTV_TEMPERATURE[0] == ("DIGITAL_CAM", r"\IRTVAT2DR2")


def test_irtv_block_means_skip_nonfinite_samples():
    module = _fetch_script()
    t = np.array([0.1, 0.5, 0.9, 1.2, 1.4, 3.1])
    data = np.array([[1.0], [np.nan], [3.0], [4.0], [6.0], [8.0]], dtype=np.float32)
    centres, means = module.block_means(t, data, 1.0)
    np.testing.assert_allclose(centres, [0.5, 1.3, 3.1])
    np.testing.assert_allclose(means[:, 0], [2.0, 5.0, 8.0])
