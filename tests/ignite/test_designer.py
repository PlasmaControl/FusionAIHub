"""Shot designer: plans reproduce IGNITE's actuator modes, couplings hold, the database
check bites, and no machine fact is written in the code.

Needs the act_global caches and the designer index (``designer.index``); skips without.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tokamak_foundation_model.ignite.designer import machine as mc

pytestmark = pytest.mark.skipif(
    not (mc.ACT_STATS.exists() and (mc.INDEX_DIR / "machine_limits.json").exists()),
    reason="needs act_global stats and the designer index",
)

F, K0 = mc.N_FRAMES, mc.K0
CACHE = str(mc.MODELS / "ignite_prod_v4" / "frame_codes")
BAL = dict(op="balance_torque", t=3.5, off=["NB15L", "NB15R", "NB30R", "NB33R"],
           counter=["NB21L", "NB21R"], torque_per_MW=-0.979, kv=72, t_end="end")


@pytest.fixture(autouse=True)
def _act_global(monkeypatch):
    monkeypatch.setenv("IGNITE_ACT_GLOBAL", str(mc.ACT_STATS.parent))


def _reference(shot, mode):
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev
    from tokamak_foundation_model.ignite.designer import shots

    act = torch.as_tensor(shots.load(shot).act_norm[:F])
    return ev.apply_actuator_mode(act, mode, K0, cache_dir=CACHE, F=F).numpy()


@pytest.mark.parametrize("plan, mode", [
    ({"shot": 204011, "ops": [BAL]},
     "balance:off=0,1,5,7;counter=2,3;nm_per_mw=-0.979;kv=72@50"),
    ({"shot": 204011, "ops": [{**BAL, "ramp_s": 1.0}]},
     "linear:20:balance:off=0,1,5,7;counter=2,3;nm_per_mw=-0.979;kv=72@50"),
    ({"shot": 201240, "ops": [{"op": "scale", "channels": "ech_power", "factor": 0, "t": 3.5,
                               "ramp_s": 1.0, "t_end": "end"}]}, "linear:20:scale:ech_power=0@50"),
    ({"shot": 201240, "ops": [{"op": "scale", "channels": "ech_power", "factor": 0.5, "t": 3.5,
                               "ramp_s": 1.0, "t_end": "end"}]}, "linear:20:scale:ech_power=0.5@50"),
    ({"shot": 201081, "ops": [{"op": "copy_from", "shot": 201079, "t": 2.0}]}, "donor:201079"),
    ({"shot": 201081, "ops": [{"op": "copy_from", "shot": 201083, "t": 2.0}]}, "donor:201083"),
    ({"shot": 201081, "ops": [{"op": "copy_from", "shot": 201084, "t": 2.0}]}, "donor:201084"),
])
def test_plan_reproduces_eval_mode(plan, mode):
    from tokamak_foundation_model.ignite.designer import plan as pl

    got = pl.build(plan).normalised()[:F]
    np.testing.assert_allclose(got, _reference(plan["shot"], mode), atol=1e-5)


def test_file_mode_round_trip(tmp_path):
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev
    from tokamak_foundation_model.ignite.designer import plan as pl, shots

    b = pl.build({"shot": 204011, "ops": [BAL]})
    path = tmp_path / "plan.npz"
    np.savez(path, act_norm=b.normalised())
    act = torch.as_tensor(shots.load(204011).act_norm[:F])
    got = ev.apply_actuator_mode(act, f"file:{path}", K0, F=F).numpy()
    np.testing.assert_allclose(got, b.normalised()[:F], atol=1e-6)
    other = torch.as_tensor(shots.load(201240).act_norm[:F])
    with pytest.raises(ValueError, match="another shot"):
        ev.apply_actuator_mode(other, f"file:{path}", K0, F=F)


def test_torque_follows_beam_power():
    from tokamak_foundation_model.ignite.designer import plan as pl

    b = pl.build({"shot": 204011, "ops": [
        {"op": "scale", "channels": "pinj:NB15L", "factor": 0.5, "t": 3.0}]})
    P, T = mc.columns("pinj"), mc.columns("tinj")
    k = slice(mc.frame_of(3.0), mc.frame_of(5.0))
    np.testing.assert_allclose(b.act[k, T.start], 0.5 * b.measured[k, T.start], rtol=1e-5)
    np.testing.assert_allclose(b.act[k, P.start], 0.5 * b.measured[k, P.start], rtol=1e-5)
    with pytest.raises(ValueError, match="cannot be edited on their own"):
        pl.build({"shot": 204011, "ops": [{"op": "scale", "channels": "tinj", "factor": 0}]})


def test_channel_names_come_from_the_data_config():
    assert mc.BEAMS == ("NB15L", "NB15R", "NB21L", "NB21R", "NB30L", "NB30R", "NB33L", "NB33R")
    assert "LEIA" in mc.CHANNEL_NAMES["ech_power"] and "IU30" in mc.CHANNEL_NAMES["i_coil"]
    for name in ("HW_LIMITS", "HW_SINCE", "CER_BEAMS", "beam_power_at", "beam_kv_for"):
        assert not hasattr(mc, name)


def test_zero_torque_in_204011_against_the_database():
    """Zero torque from 3.5 s with both NB21 sources counter-current: with NB30L/NB33L at
    full power the counter sources need more than any shot delivered, so the database
    check fails; with those two at 55 % (the agent's plan D) it passes, the NB21 sources
    above the campaign's maximum but within the database (information)."""
    from tokamak_foundation_model.ignite.designer import plan as pl, validate

    keep = ["NB30L", "NB33L"]
    opts = pl.beam_options(204011, 3.5, counter=["NB21L", "NB21R"], keep_on=keep)
    assert len(opts) == 16 and not any(o["feasible"] for o in opts)
    best = next(o for o in opts if o["off"] == ["NB15L", "NB15R", "NB30R", "NB33R"])
    assert 0.5 < best["keep_on_max_fraction"] < 1.0
    a = pl.build({"shot": 204011, "ops": [{**BAL, "torque_per_MW": None, "kv": None,
                                           "off": ["NB15L", "NB15R", "NB30R"], "split": "fill"}]})
    assert not validate.ok(validate.check(a))
    d = pl.build({"shot": 204011, "ops": [
        {"op": "scale", "channels": "pinj:NB30L,NB33L", "factor": 0.55, "t": 3.5},
        {"op": "balance_torque", "t": 3.5, "off": ["NB15L", "NB15R", "NB30R"],
         "counter": ["NB21L", "NB21R"], "target_Nm": 0.0}]})
    found = validate.check(d)
    assert validate.ok(found)
    assert any(f["check"] == "above_era" and f["level"] == "info" for f in found)
    k = slice(mc.frame_of(4.0), mc.frame_of(5.5))
    assert abs(d.act[k, mc.columns("tinj")].sum(1)).max() < 1e-3
    with pytest.raises(ValueError, match="counter"):
        pl.beam_options(204011, 3.5)


def test_gas_paths_for_201081():
    from tokamak_foundation_model.ignite.designer import plan as pl, validate

    G, R = mc.columns("gas_flow"), mc.columns("gas_raw")
    for donor in (201079, 201083, 201084):
        b = pl.build({"shot": 201081, "ops": [
            {"op": "copy_from", "shot": donor, "groups": ["gas_flow"], "t": 2.0}]})
        assert validate.ok(validate.check(b))
        assert not b.changed[:, mc.columns("pinj")].any()
        assert b.changed[:, R].any() or not b.changed[:, G].any()
    # A designed path: hold the gas at its 2 s level until 2.5 s, then the measured rise.
    b = pl.build({"shot": 201081, "ops": [
        {"op": "shift", "channels": "gas_flow", "dt": 0.5, "t": 2.0}]})
    assert validate.ok(validate.check(b))


def test_precedents_find_low_torque_shots():
    from tokamak_foundation_model.ignite.designer import precedents as pr

    res = pr.search(["torque_Nm < 1", "pnbi_MW > 4"], near=204011, exclude=[204011], limit=5)
    assert res["n_matches"] > 0
    for m in res["matches"]:
        assert m["means"]["torque_Nm"] < 1 and m["means"]["pnbi_MW"] > 4


def test_logbook_shot_and_search():
    from tokamak_foundation_model.ignite.designer import logbook as lb

    if not lb.DB.exists():
        pytest.skip("logbook database not built")
    r = lb.shot(204011)
    assert r["found"] and "radiator" in r["run_title"].lower()
    assert 204011 in r["run_shots"]
    res = lb.search('"counter NBI" OR "counter beams"', tokenized_only=True, limit=5)
    assert res["n_matches"] > 0 and all(m["tokenized"] for m in res["matches"])
    sub = lb.search("torque", shots=[204011, 200411], limit=5)
    assert {m["shot"] for m in sub["matches"]} <= {204011, 200411}
    assert lb._fts_query('detach* "X-point radiator"') == '"detach"* "X-point radiator"'


def test_fill_respects_limit_kind():
    """``fill`` caps the first counter source at the limit the caller chose."""
    from tokamak_foundation_model.ignite.designer import plan as pl

    era = mc.era_limits(204011)["beams"]
    for kind, key in (("era", "power_MW_max"), ("db", "power_MW_db")):
        op = dict(BAL, split="fill", target_Nm=0.5, limit_kind=kind)
        op.pop("torque_per_MW"), op.pop("kv")  # the measured rates: NB21R's is larger
        b = pl.build({"shot": 204011, "ops": [op]})
        first = "NB21R"
        p = b.act[:, mc.columns("pinj").start + mc.BEAMS.index(first)] * 1e-6
        assert p.max() <= era[first][key] + 1e-6


def test_switched_on_beam_follows_the_kept_beams():
    """A counter source switched on by ``beams`` ramps down with the kept beams, so the
    net torque never reverses at the beam shutdown."""
    from tokamak_foundation_model.ignite.designer import plan as pl, validate

    op = {"op": "beams", "t": 3.5, "off": ["NB15L", "NB15R", "NB30R", "NB33R"],
          "power_MW": {"NB21L": 1.6, "NB21R": 1.6},
          "direction": {"NB21L": "counter", "NB21R": "counter"}}
    b = pl.build({"shot": 204011, "ops": [op]})
    tq = b.act[:, mc.columns("tinj")].sum(1)[mc.frame_of(3.5):]
    assert tq.min() > -0.1
    assert not any(f["check"] == "torque_reversal" for f in validate.check(b))
    held = pl.build({"shot": 204011, "ops": [dict(op, hold="window")]})
    assert any(f["check"] == "torque_reversal" for f in validate.check(held))


def test_check_plan_reports_era_margin():
    from tokamak_foundation_model.ignite.designer import tools

    r = tools.check_plan({"shot": 204011, "ops": [dict(BAL, target_Nm=0.9)]})
    assert r["ok"] and "within_era" in r and "NB21L" in r["beam_margins"]
    m = r["beam_margins"]["NB21L"]
    assert r["within_era"] == all(v["within_era"] for v in r["beam_margins"].values())
    assert m["within_db"] and m["plan_max_MW"] <= m["db_max_MW"] * 1.02
