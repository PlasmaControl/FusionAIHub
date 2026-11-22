"""Tests of the channel registry. Real-data tests need TokaMark's store; the others need neither it nor `tokamark`."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import paths, registry, tasks, transform


def test_family_rules():
    pytest.importorskip("tokamark")
    fam = {s.name: registry.family_of(s) for s in registry.all_signals()}
    assert fam["equilibrium-psi"] == "map" and fam["equilibrium-lcfs_r"] == "profile"
    assert fam["thomson_scattering-t_e"] == "profile" and fam["soft_x_rays-horizontal_cam_upper"] == "high_rate"
    assert fam["magnetics-b_field_tor_probe_cc_field"] == "high_rate" and fam["summary-ip"] == "slow"
    assert fam["equilibrium-x_point_r"] == "slow" and len(fam) == 39


def test_family_rules_on_hand_built_specs():
    """The rules of the brief, without tokamark."""
    def sp(name, dt, shape):
        return tasks.SignalSpec(name, dt, shape, 0.0, 1.0)
    assert registry.family_of(sp("equilibrium-psi", 0.005, (65, 65))) == "map"
    assert registry.family_of(sp("equilibrium-lcfs_z", 0.005, (170,))) == "profile"
    assert registry.family_of(sp("thomson_scattering-n_e", 0.005, (120,))) == "profile"
    assert registry.family_of(sp("magnetics-b_field_pol_probe_omv_voltage", 0.001024, (3, 256))) == "high_rate"
    assert registry.family_of(sp("magnetics-b_field_tor_probe_saddle_voltage", 2e-5, (12,))) == "high_rate"
    assert registry.family_of(sp("magnetics-flux_loop_flux", 0.0002, (15,))) == "slow"
    assert registry.family_of(sp("equilibrium-x_point_z", 0.005, (2,))) == "slow"


def test_all_signals_raises_on_conflicting_specs(monkeypatch, toy_task):
    """The same name with another statistic in a second task is an error, not a pick."""
    other = dataclasses.replace(toy_task.inputs[0], std=toy_task.inputs[0].std * 2)
    variant = dataclasses.replace(toy_task, name="task_toy2", inputs=(other, toy_task.inputs[1]))
    monkeypatch.setattr(registry, "ALL_TASKS", ("task_toy", "task_toy2"))
    monkeypatch.setattr(registry, "load_task", lambda n: {"task_toy": toy_task, "task_toy2": variant}[n])
    with pytest.raises(ValueError, match="summary-ip"):
        registry.all_signals()
    monkeypatch.setattr(registry, "ALL_TASKS", ("task_toy",))
    got = registry.all_signals()
    assert [s.name for s in got] == sorted(s.name for s in got) and len(got) == 3


def _frame():
    rows = []
    for name, n in (("a-x", 1), ("b-y", 3)):
        for c in range(n):
            rows.append(dict.fromkeys(registry.REGISTRY_COLUMNS, "") | {"signal": name, "channel": c})
    df = pd.DataFrame(rows, columns=list(registry.REGISTRY_COLUMNS))
    for col in ("dt", "r", "z", "tm_mean", "tm_std", "sat_abs", "valid_fraction", "dead_fraction", "scale", "mean", "std"):
        df[col] = np.nan
    return df


def test_attach_stats_matches_arrays():
    rng = np.random.default_rng(0)
    stats = {
        "a-x": transform.ChannelStats("none", np.ones((1,)), np.array([0.5]), np.array([2.0])),
        "b-y": transform.ChannelStats("asinh", rng.uniform(1, 2, (3,)), rng.standard_normal(3),
                                      rng.uniform(1, 2, (3,))),
    }
    out = registry.attach_stats(_frame(), stats)
    for name, st in stats.items():
        sub = out[out.signal == name].sort_values("channel")
        np.testing.assert_array_equal(sub["mean"].to_numpy(), st.mean.ravel())
        np.testing.assert_array_equal(sub["std"].to_numpy(), st.std.ravel())
        np.testing.assert_array_equal(sub["scale"].to_numpy(), st.scale.ravel())
        assert (sub["compress"] == st.compress).all()


def test_attach_stats_rejects_a_shape_mismatch():
    stats = {"b-y": transform.ChannelStats("none", np.ones((2,)), np.zeros((2,)), np.ones((2,)))}
    with pytest.raises(ValueError, match="b-y"):
        registry.attach_stats(_frame(), stats)


def test_save_load_registry_roundtrip(tmp_path):
    df = registry.attach_stats(_frame(), {"a-x": transform.ChannelStats("none", np.ones((1,)), np.array([0.5]),
                                                                       np.array([2.0]))})
    df.loc[0, "units"] = "A"
    path = tmp_path / "reg.csv"
    registry.save_registry(df, path)
    back = registry.load_registry(path)
    assert tuple(back.columns) == registry.REGISTRY_COLUMNS
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    assert back.loc[1, "units"] == "" and np.isnan(back.loc[1, "mean"])   # empty means not found


def _fake_zarr_store(root, shots):
    zarr = pytest.importorskip("zarr")
    for shot in shots:
        g = zarr.open_group(str(root / f"{shot}.zarr"), mode="w", zarr_format=2)
        s = g.create_group("summary")
        a = s.create_array("ip", data=np.zeros(5), chunks=(5,))
        a.attrs.update({"units": "A", "imas": "summary.global_quantities.ip", "label": "Plasma Current",
                        "_ARRAY_DIMENSIONS": ["time"]})
        if shot == shots[1]:      # only the second shot has the profile source
            t = g.create_group("thomson_scattering")
            t.create_array("major_radius", data=np.array([0.3, 0.4, 0.5]))
            a = t.create_array("t_e", data=np.zeros((3, 5)), chunks=(3, 5))
            a.attrs.update({"units": "eV", "imas": "thomson_scattering.channel[:].t_e",
                            "_ARRAY_DIMENSIONS": ["major_radius", "time"]})
            e = g.create_group("equilibrium")
            e.create_array("z", data=np.array([-1.0, 1.0]))
            e.create_array("major_radius", data=np.array([1.0, 2.0, 3.0]))
            a = e.create_array("psi", data=np.zeros((2, 3, 5)), chunks=(2, 3, 5))
            a.attrs.update({"units": "Wb / rad", "imas": "equilibrium.psi",
                            "_ARRAY_DIMENSIONS": ["z", "major_radius", "time"]})


def test_build_registry_from_a_small_store(tmp_path, monkeypatch):
    shots = [101, 102, 103]
    _fake_zarr_store(tmp_path, shots)
    specs = [
        tasks.SignalSpec("equilibrium-psi", 0.005, (2, 3), 0.0, 1.0),
        tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.0, 4.0),
        tasks.SignalSpec("thomson_scattering-t_e", 0.005, (3,), 5.0, 6.0),
    ]
    monkeypatch.setattr(registry, "all_signals", lambda: specs)
    monkeypatch.setattr(registry, "_roles", lambda: {"summary-ip": "inp+out", "equilibrium-psi": "out",
                                                     "thomson_scattering-t_e": "inp+out"})

    def load(shot, signals):          # 101: Thomson missing; 102: one dead and one NaN channel; 103: all fine
        rng = np.random.default_rng(shot)
        out = {"summary-ip": {"time": np.arange(10.0), "values": rng.standard_normal((1, 10))},
               "equilibrium-psi": {"time": np.arange(4.0), "values": rng.standard_normal((2, 3, 4))},
               "thomson_scattering-t_e": {"time": np.arange(4.0), "values": rng.standard_normal((3, 4))}}
        if shot == 101:
            out["thomson_scattering-t_e"] = {"time": np.zeros((0,)), "values": np.zeros((0,))}
        if shot == 102:
            out["thomson_scattering-t_e"]["values"][0] = 5.0
            out["thomson_scattering-t_e"]["values"][1, :2] = np.nan
        return out
    monkeypatch.setattr(transform, "load_shot", load)
    df = registry.build_registry(tmp_path, shots)
    assert tuple(df.columns) == registry.REGISTRY_COLUMNS
    assert len(df) == 6 + 1 + 3 and (df.device == "MAST").all()
    psi = df[df.signal == "equilibrium-psi"].sort_values("channel")
    assert list(psi.channel_name) == ["0,0", "0,1", "0,2", "1,0", "1,1", "1,2"]
    assert list(psi.z) == [-1.0] * 3 + [1.0] * 3 and list(psi.r) == [1.0, 2.0, 3.0] * 2
    assert (psi.family == "map").all() and (psi.units == "Wb / rad").all()
    ip = df[df.signal == "summary-ip"].iloc[0]
    assert (ip.units, ip.imas_path, ip.roles, ip.family, ip["dt"]) == ("A", "summary.global_quantities.ip", "inp+out",
                                                                  "slow", 0.00025)
    assert (ip.tm_mean, ip.tm_std) == (3.0, 4.0) and ip.valid_fraction == 1.0 and ip.dead_fraction == 0.0
    te = df[df.signal == "thomson_scattering-t_e"].sort_values("channel")
    assert list(te.r) == [0.3, 0.4, 0.5] and te.z.isna().all() and (te.units == "eV").all()
    # shot 101 has no Thomson (dead: no finite sample), shot 102 channel 0 is constant: 2 of 3 shots dead
    assert te.dead_fraction.iloc[0] == pytest.approx(2 / 3) and te.dead_fraction.iloc[2] == pytest.approx(1 / 3)
    # valid fraction: shot-weighted finite fraction, a missing signal counting 0
    assert te.valid_fraction.iloc[1] == pytest.approx((0 + 0.5 + 1.0) / 3)
    assert (df.compress == "").all() and df[["scale", "mean", "std"]].isna().all(axis=None)   # not fitted yet
    assert df.sat_abs.isna().all()


@pytest.mark.real_data
def test_registry_rows_and_columns():
    df = registry.build_registry(paths.tokamark_zarr_root(), tasks.split_shots("train")[:5], workers=4)
    assert tuple(df.columns) == registry.REGISTRY_COLUMNS
    n = {s.name: int(np.prod(s.values_shape)) for s in registry.all_signals()}
    assert len(df) == sum(n.values()) and (df.groupby("signal").size() == pd.Series(n)).all()
    assert df.loc[df.signal == "summary-ip", "roles"].iloc[0] == "inp+out"
    dalpha = df[df.signal == "spectrometer_visible-filter_spectrometer_dalpha_voltage"]
    assert (dalpha.sat_abs == 4.99).all() and df.sat_abs.notna().sum() == len(dalpha)
    assert df.loc[df.signal == "summary-ip", "units"].iloc[0] == "A"
    assert df.valid_fraction.between(0, 1).all() and df.dead_fraction.between(0, 1).all()
