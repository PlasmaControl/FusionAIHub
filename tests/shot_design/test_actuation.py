"""Actuator waveforms: RDP keeps endpoints and every sample within tolerance, the vertex cap is met
by raising the tolerance, the median seed ignores NaN and records its sources, evaluate flags a
peak above the NBI cap. Numbers come from the conftest fixtures (900001: beams 15L 2.0 MW + 30L
1.5 MW + 33L 1.0 MW on 1000-4500 ms; 900002: 0.9 MA, beams differ).

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from pydantic import ValidationError

from shot_design import config
from shot_design.flags import rules
from shot_design.retrieval import actuation
from shot_design.schema import ActuationSet, ActuatorWaveform, Vertex

CFG = {"rdp_tol_frac": 0.03, "max_vertices": 12, "grid_ms": 50}


def test_ui_yaml_has_the_actuation_block():
    assert config.load_yaml("ui.yaml")["actuation"] == CFG


def test_vertices_are_sorted_unique_and_finite():
    w = ActuatorWaveform(key="nbi.total", vertices=[Vertex(t_s=2, y=1), Vertex(t_s=0, y=0)])
    assert [v.t_s for v in w.vertices] == [0, 2]
    with pytest.raises(ValidationError, match="two vertices at t = 1.0 s"):
        ActuatorWaveform(key="k", vertices=[Vertex(t_s=1, y=0), Vertex(t_s=1, y=2)])
    with pytest.raises(ValidationError, match="at least one vertex"):
        ActuatorWaveform(key="k", vertices=[])
    with pytest.raises(ValidationError, match="finite"):
        ActuatorWaveform(key="k", vertices=[Vertex(t_s=float("nan"), y=0)])
    with pytest.raises(ValidationError, match="says key"):
        ActuationSet(
            waveforms={
                "nbi.total": ActuatorWaveform(key="ech.total", vertices=[Vertex(t_s=0, y=0)])
            }
        )


def _trapezoid(n=2000):
    t = np.linspace(0.0, 6.0, n)
    y = np.interp(t, [0, 1, 1.5, 4.5, 5, 6], [0, 0, 4.5e6, 4.5e6, 0, 0])
    return t, y


def _two_step(n=1200):
    t = np.linspace(0.0, 6.0, n)
    y = np.where(t < 1, 0.0, np.where(t < 3, 2.0e6, np.where(t < 5, 3.5e6, 0.0)))
    return t, y


@pytest.mark.parametrize("make", [_trapezoid, _two_step])
def test_simplify_keeps_endpoints_and_every_sample_within_tolerance(make):
    t, y = make()
    verts = actuation.simplify(t, y, CFG)
    assert (verts[0].t_s, verts[-1].t_s) == (t[0], t[-1])
    # The trapezoid has four corners and the two-step three edges: a seed that carried fewer than
    # four vertices would have flattened the shape the operator is looking at.
    assert 4 <= len(verts) <= CFG["max_vertices"]
    tol = CFG["rdp_tol_frac"] * (y.max() - y.min())
    back = actuation.values_at(verts, t)
    assert np.max(np.abs(back - y)) <= tol + 1e-9
    assert all(a.t_s < b.t_s for a, b in pairwise(verts))


def test_vertex_cap_raises_tolerance_rather_than_dropping_endpoints():
    """A 3-period sine: six interior extrema plus the two endpoints, so eight vertices carry the
    shape and twelve are comfortable. Doubling the tolerance alone lands on 8 (the 13-vertex rung
    is over the cap and the next doubling drops to 8); the bisection must find the rung that
    actually spends the budget."""
    t = np.linspace(0.0, 6.0, 600)
    y = np.sin(np.pi * t) * 1e6  # 3 % tolerance alone would keep far more than 12 points
    assert len(actuation.rdp(t, y, 0.03 * 2e6)) > 12
    verts = actuation.simplify(t, y, CFG)
    assert len(verts) <= 12 and (verts[0].t_s, verts[-1].t_s) == (0.0, 6.0)
    assert len(verts) >= 8  # the six extrema and the endpoints, at least
    # And the budget is actually spent: doubling alone stopped at 8 vertices / 3.07e5 max error,
    # the bisection lands on 12 / 2.50e5 (measured).
    assert len(verts) >= 10
    assert np.max(np.abs(actuation.values_at(verts, t) - y)) < 0.5e6  # well under half the swing


def test_vertex_cap_falls_back_to_the_chord_when_the_budget_cannot_hold_the_extrema():
    """Six periods in the same window: twelve interior extrema, so eleven segments cannot separate
    them (pigeonhole) and NO 12-vertex piecewise-linear fit beats the amplitude. RDP's rungs here
    are 14 vertices (all the extrema) and then 2 -- nothing in between -- so the cap is met by the
    endpoint chord. Measured, not assumed: the cap is still respected and the endpoints kept."""
    t = np.linspace(0.0, 6.0, 600)
    y = np.sin(2 * np.pi * t) * 1e6
    verts = actuation.simplify(t, y, CFG)
    assert len(verts) <= 12 and (verts[0].t_s, verts[-1].t_s) == (0.0, 6.0)
    assert {(v.t_s, v.y) for v in verts} <= set(zip(t.tolist(), y.tolist(), strict=True))


def test_simplify_handles_empty_flat_and_nan_traces():
    empty = actuation.simplify(np.array([]), np.array([]), CFG)
    assert [(v.t_s, v.y) for v in empty] == [(0.0, 0.0), (1.0, 0.0)]
    t = np.linspace(0, 5, 50)
    flat = actuation.simplify(t, np.full(50, 2.0), CFG)
    assert [(v.t_s, v.y) for v in flat] == [(0.0, 2.0), (5.0, 2.0)]
    y = np.linspace(0, 1, 50)
    y[10:20] = np.nan
    gap = actuation.simplify(t, y, CFG)
    assert np.isfinite([v.y for v in gap]).all() and len(gap) >= 2


def test_simplify_of_a_single_sample_is_one_vertex():
    assert actuation.simplify(np.array([2.0]), np.array([5.0]), CFG) == [Vertex(t_s=2.0, y=5.0)]


def test_values_at_holds_the_last_value():
    verts = [Vertex(t_s=0, y=0), Vertex(t_s=1, y=2)]
    assert actuation.values_at(verts, np.array([-1, 0.5, 1, 9])).tolist() == [0, 1, 2, 2]


def test_seed_reference_from_the_fixture_beams(paths, staged_shot_a):
    w = actuation.seed_reference(staged_shot_a, "nbi.total", paths, CFG)
    assert w.key == "nbi.total" and w.units == "W" and w.seed == "reference"
    assert w.source_shots == [staged_shot_a] and 2 <= len(w.vertices) <= 12
    assert actuation.peak(w) == pytest.approx(4.5e6, rel=1e-6)
    # Times are seconds: the flat top sits at 1.0-4.5 s, the trace starts at -0.5 s.
    assert w.vertices[0].t_s == pytest.approx(-0.5, abs=0.01)
    on = actuation.values_at(w.vertices, np.array([2.0, 3.0, 4.0]))
    assert np.allclose(on, 4.5e6, rtol=0.03)
    assert actuation.values_at(w.vertices, np.array([0.0, 5.5]))[0] == pytest.approx(0.0, abs=1e5)
    assert w.description == actuation.descriptions().get("nbi.total", "")


def test_seed_reference_of_a_member_key(paths, staged_shot_a):
    w = actuation.seed_reference(staged_shot_a, "nbi.15L", paths, CFG)
    assert actuation.peak(w) == pytest.approx(2.0e6, rel=1e-6)
    assert w.units == "W" and 2 <= len(w.vertices) <= 12
    on = actuation.values_at(w.vertices, np.array([2.0, 3.0, 4.0]))
    assert np.allclose(on, 2.0e6, rtol=0.03)


def test_seed_reference_of_a_present_all_zero_member_still_names_its_shot(paths, staged_shot_a):
    """900001 writes a column for every beam and fires only 15L/30L/33L, so 21L is PRESENT and flat
    at zero. The seed must still name the shot it was read from: the page reads "no data behind
    this seed" off `source_shots`, and a measured zero is not an absent trace."""
    w = actuation.seed_reference(staged_shot_a, "nbi.21L", paths, CFG)
    assert w.source_shots == [staged_shot_a]
    assert [v.y for v in w.vertices] == [0.0, 0.0]


def test_seed_reference_of_an_absent_actuator_is_two_zero_vertices(paths, staged_shot_a):
    w = actuation.seed_reference(staged_shot_a, "ech.total", paths, CFG)  # no ech group in 900001
    assert [(v.t_s, v.y) for v in w.vertices] == [(0.0, 0.0), (1.0, 0.0)]
    assert w.units == "W"
    assert w.source_shots == []  # nothing was read: distinguishable from a measured zero


def test_median_grid_ignores_nan_and_shots_without_data_at_a_time():
    """Three series on different extents; the middle one has a NaN hole. At each grid point the
    median is over the finite values only, and a time nobody covers is NaN (dropped by simplify)."""
    t1, y1 = np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])
    t2, y2 = np.array([0.0, 1.0, 2.0, 3.0]), np.array([3.0, np.nan, 3.0, 3.0])
    t3, y3 = np.array([0.0, 2.0]), np.array([5.0, 5.0])
    grid, med = actuation.median_grid([(t1, y1), (t2, y2), (t3, y3)], step_s=1.0)
    assert grid.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert med[0] == 3.0  # median of 1, 3, 5
    assert med[1] == 3.0  # NaN in series 2 ignored: median of 1 and 5
    assert med[2] == 3.0
    assert med[3] == 3.0  # only series 2 reaches 3 s
    assert not np.isnan(med).any()
    _, gap = actuation.median_grid([(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))], step_s=1.0)
    assert np.isnan(gap).all()
    with pytest.raises(ValueError, match="at least one series"):
        actuation.median_grid([], step_s=1.0)


def test_median_grid_does_not_trust_the_order_of_a_series():
    """A trace whose samples arrive out of order must give the same grid and median as the sorted
    one: np.interp needs increasing x, and the extent comes from min/max, not from t[0]/t[-1]."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 2.0, 4.0, 6.0])
    order = np.array([2, 0, 3, 1])
    g_ok, m_ok = actuation.median_grid([(t, y)], step_s=0.5)
    g_mix, m_mix = actuation.median_grid([(t[order], y[order])], step_s=0.5)
    assert g_mix.tolist() == g_ok.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert m_mix.tolist() == m_ok.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_seed_median_records_only_the_shots_with_data(paths, staged_shot_a, staged_shot_b):
    """900002 has no p_inj group and 999999 no file: only 900001 contributes to nbi.total."""
    w = actuation.seed_median([staged_shot_a, staged_shot_b, 999999], "nbi.total", paths, CFG)
    assert w.seed == "median" and w.source_shots == [staged_shot_a] and w.units == "W"
    assert 2 <= len(w.vertices) <= 12
    assert np.allclose(actuation.values_at(w.vertices, np.array([2.0, 3.0, 4.0])), 4.5e6, rtol=0.03)


def test_seed_median_of_nothing_present(paths, staged_shot_a):
    w = actuation.seed_median([staged_shot_a], "ech.total", paths, CFG)
    assert w.source_shots == [] and [(v.t_s, v.y) for v in w.vertices] == [(0.0, 0.0), (1.0, 0.0)]


def test_evaluate_flags_a_peak_above_the_nbi_cap_and_nothing_else():
    cfg = rules.load_rules()
    hot = ActuatorWaveform(key="nbi.total", vertices=[Vertex(t_s=0, y=0), Vertex(t_s=1, y=3e7)])
    flags = actuation.evaluate(hot, "nbi.total", cfg)
    assert [f.rule_id for f in flags] == ["nbi_total_max"] and flags[0].severity == "error"
    assert flags[0].value == 3e7
    cool = ActuatorWaveform(key="nbi.total", vertices=[Vertex(t_s=0, y=0), Vertex(t_s=1, y=5e6)])
    assert actuation.evaluate(cool, "nbi.total", cfg) == []  # no "skipped: missing" noise
    out = actuation.check({"nbi.total": hot, "ech.total": cool}, cfg)
    assert set(out) == {"nbi.total", "ech.total"} and out["ech.total"] == []


def test_descriptions_come_from_actuators_yaml():
    d = actuation.descriptions()
    assert set(d) <= set(rules.actuator_columns())
    assert d["nbi.total"] and d["nbi.15L"] and d["coil_rmp.total"]


def test_gas_species_is_a_selection_from_the_registry_vocabulary():
    species = actuation.gas_species()
    assert species[0] == "D2" and {"D2", "H2", "He", "N2", "Ne", "Ar"} <= set(species)
    actuation.check_gas_species({"GASA": "D2", "GASC": "N2"})  # valid: no exception
    actuation.check_gas_species({})
    with pytest.raises(ValueError, match="unknown gas species 'beer'"):
        actuation.check_gas_species({"GASA": "beer"})
    with pytest.raises(ValueError, match="not a gas valve"):
        actuation.check_gas_species({"15L": "D2"})
    assert ActuationSet().gas_species == {}


def _set(shot, **kw):
    return ActuationSet(
        source_shot=shot,
        waveforms={
            "nbi.total": ActuatorWaveform(
                key="nbi.total", units="W",
                vertices=[Vertex(t_s=0, y=0), Vertex(t_s=1, y=3e7), Vertex(t_s=4, y=0)],
            )
        },
        flags=[],
        notes="too hot on purpose",
        **kw,
    )  # fmt: skip


def test_actuations_dir_is_under_the_data_root(paths):
    assert paths.actuations_dir == paths.data_root / "actuations"
    cfg = config.load_yaml("paths.yaml")
    assert cfg["actuations_dir"] == "${data_root}/actuations"


def test_save_list_load_roundtrip_with_the_servers_flags(paths):
    stored = actuation.save(_set(900001), paths)
    assert actuation.ID_RE.fullmatch(stored.id) and stored.id.endswith("-900001")
    assert stored.created is not None
    assert [f.rule_id for f in stored.flags] == ["nbi_total_max"]  # recomputed, not the page's []
    p = paths.actuations_dir / f"{stored.id}.json"
    assert p.exists() and not p.with_suffix(".json.part").exists()
    (item,) = actuation.list_sets(paths)
    assert item["id"] == stored.id and item["source_shot"] == 900001 and item["n_waveforms"] == 1
    assert item["created"] and item["created"].startswith(stored.created.strftime("%Y-%m-%dT%H:%M"))
    back = actuation.load(stored.id, paths)
    assert back == stored
    assert actuation.load(actuation.new_id(1), paths) is None
    with pytest.raises(ValueError):
        actuation.load("../etc/passwd", paths)
    second = actuation.save(_set(900002), paths)
    assert [i["id"] for i in actuation.list_sets(paths)] == [second.id, stored.id]  # newest first


def test_save_rejects_a_waveform_key_that_is_not_an_actuator(paths):
    """The seed route already refuses an unknown key; save must not accept one either, and it
    must refuse before anything is written."""
    aset = _set(900001)
    aset.waveforms["bogus.total"] = aset.waveforms["nbi.total"].model_copy(
        update={"key": "bogus.total"}
    )
    with pytest.raises(ValueError, match="unknown actuator key"):
        actuation.save(aset, paths)
    assert actuation.list_sets(paths) == []


def test_save_rejects_an_unknown_gas_species_and_round_trips_a_valid_mapping(paths):
    """The species check runs before anything is written: a bad set leaves no file behind."""
    with pytest.raises(ValueError, match="unknown gas species"):
        actuation.save(_set(900001, gas_species={"GASA": "beer"}), paths)
    assert actuation.list_sets(paths) == []
    stored = actuation.save(_set(900001, gas_species={"GASA": "D2", "GASC": "N2"}), paths)
    assert actuation.load(stored.id, paths).gas_species == {"GASA": "D2", "GASC": "N2"}
