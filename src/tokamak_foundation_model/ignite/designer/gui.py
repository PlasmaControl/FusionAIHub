"""Notebook GUI for the shot designer (ipywidgets): ``gui.app()``.

Tabs follow the designer's steps: look at a shot, find precedents, enumerate beam sets,
write and check a plan (or load one an agent wrote), export it, simulate it with IGNITE
and plot the result. The plotting functions also work on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import machine as mc
from . import plan as pl
from . import precedents as pr
from . import shots
from . import simulate as sim
from . import tools

MEASURED, REPLAY, PLAN = "#1f5c9e", "#e07b1a", "#7b3294"
_P, _T = mc.columns("pinj"), mc.columns("tinj")
_E, _G = mc.columns("ech_power"), mc.columns("gas_flow")


def _totals(a):
    return {"NBI power (MW)": (a[:, _P] * 1e-6).sum(1), "beam torque (N m)": a[:, _T].sum(1),
            "ECH power (MW)": (a[:, _E] * 1e-6).sum(1), "gas (Torr L/s)": a[:, _G].sum(1)}


def plot_actuators(b: pl.Built, t_max: float = 6.0):
    """Measured vs planned totals, and every beam source the plan changed."""
    t = b.shot.times
    keep = t < t_max
    tot_m, tot_p = _totals(b.measured), _totals(b.act)
    beams = [i for i in range(8) if b.changed[:, _P.start + i].any()]
    n = len(tot_m) + (1 if beams else 0)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.7 * n), sharex=True)
    for ax, key in zip(axes, tot_m):
        ax.plot(t[keep], tot_m[key][keep], color=MEASURED, lw=2, label="measured")
        ax.plot(t[keep], tot_p[key][keep], color=PLAN, lw=1.3, ls="--", label="plan")
        ax.set_ylabel(key, fontsize=8)
    if beams:
        ax = axes[-1]
        for i in beams:
            line, = ax.plot(t[keep], b.act[keep, _P.start + i] * 1e-6, lw=1.3, ls="--",
                            label=f"{mc.BEAMS[i]} plan")
            ax.plot(t[keep], b.measured[keep, _P.start + i] * 1e-6, lw=0.8,
                    color=line.get_color(), alpha=0.6)
        ax.set_ylabel("changed sources (MW)", fontsize=8)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    for ax in axes:
        ax.axvspan(mc.T0, mc.time_of(mc.K0), color="0.92", zorder=0)
        if b.shot.t_end:
            ax.axvline(b.shot.t_end, color="0.5", lw=0.8, ls=":")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)  (grey: measured initial state; dotted: plasma end)")
    fig.suptitle(f"{b.shot.shot}: {b.plan.get('goal', '')}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_shot(shot: int):
    """Measured actuator totals and core plasma quantities of a shot (index series)."""
    s = pr.series(shot)
    keys = ["pnbi_MW", "torque_Nm", "ech_MW", "gas_TorrL_s", "rot_core_kms", "ti_core_keV",
            "te_core_keV", "ne_core_1e19"]
    fig, axes = plt.subplots(4, 2, figsize=(9, 6), sharex=True)
    for ax, k in zip(axes.T.ravel(), keys):
        ax.plot(s["t_s"], s[k], color=MEASURED)
        ax.set_ylabel(k, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"{shot} (measured; plasma quantities decoded from its tokens)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_results(run_dir):
    """Measured, replay and plan (seed mean and range) of the core plasma quantities."""
    r = sim.results(run_dir)
    plan = json.loads((Path(run_dir) / "plan.json").read_text())
    t = r["t_s"]
    keys = [k for k in ("rot_core_kms", "ti_core_keV", "te_core_keV", "ne_core_1e19")
            if k in r["measured"]]
    tot_m, tot_p = _totals(r["act_measured"]), _totals(r["act_plan"])
    fig, axes = plt.subplots(len(keys) + 2, 1, figsize=(8, 1.6 * (len(keys) + 2)), sharex=True)
    for ax, key in zip(axes[:2], ("NBI power (MW)", "beam torque (N m)")):
        ax.plot(t, tot_m[key], color=MEASURED, lw=2)
        ax.plot(t, tot_p[key], color=PLAN, ls="--")
        ax.set_ylabel(key, fontsize=8)
    for ax, k in zip(axes[2:], keys):
        ax.plot(t, r["measured"][k], color=MEASURED, lw=2, label="measured")
        for arm, color in (("replay", REPLAY), ("plan", PLAN)):
            x = r[arm].get(k)
            if x is None:
                continue
            ax.plot(t[r["k0"] - 1:], np.nanmean(x, 0)[r["k0"] - 1:], color=color,
                    label=f"IGNITE, {'measured actuators' if arm == 'replay' else 'plan'}")
            ax.fill_between(t[r["k0"]:], np.nanmin(x, 0)[r["k0"]:], np.nanmax(x, 0)[r["k0"]:],
                            color=color, alpha=0.15, lw=0)
        ax.set_ylabel(k, fontsize=8)
    for ax in axes:
        ax.axvspan(mc.T0, mc.time_of(r["k0"]), color="0.92", zorder=0)
    axes[2].legend(fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{plan['shot']}: {plan.get('goal', '')} (seeds {r['seeds']})", fontsize=9)
    fig.tight_layout()
    return fig


def app(shot: int = 204011):
    """The designer GUI."""
    import ipywidgets as w
    from IPython.display import display

    out = w.Output()
    shot_w = w.IntText(value=shot, description="shot")

    def show(obj):
        with out:
            out.clear_output()
            if isinstance(obj, plt.Figure):
                display(obj)
                plt.close(obj)
            else:
                print(tools.to_json(obj))

    def guard(fn):
        def run(_):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - show it in the GUI
                show({"error": f"{type(e).__name__}: {e}"})
        return run

    # 1. shot
    b_desc = w.Button(description="describe")
    b_plot = w.Button(description="plot shot")
    b_desc.on_click(guard(lambda: show(tools.describe_shot(shot_w.value))))
    b_plot.on_click(guard(lambda: show(plot_shot(shot_w.value))))
    tab_shot = w.VBox([w.HBox([b_desc, b_plot])])

    # 2. precedents
    cond_w = w.Textarea(value="torque_Nm < 1\npnbi_MW > 4", description="conditions",
                        layout=w.Layout(width="60%", height="90px"))
    win_w = w.FloatRangeSlider(value=(2.0, 6.0), min=1.0, max=8.0, step=0.05, description="window")
    dur_w = w.FloatText(value=0.5, description="min dur (s)")
    near_w = w.Checkbox(value=True, description="rank by similarity to the shot")
    b_search = w.Button(description="search")

    def search():
        res = tools.search_precedents(
            [c for c in cond_w.value.splitlines() if c.strip()], win_w.value, dur_w.value,
            shot_w.value if near_w.value else None, exclude=[shot_w.value], limit=15)
        rows = [f"{m['shot']}  {m['match_s']}  d={m.get('distance')}  " +
                "  ".join(f"{k}={v}" for k, v in m["means"].items()) for m in res["matches"]]
        show({"n_matches": res["n_matches"], "matches": rows, "caveat": res["caveat"]})

    b_search.on_click(guard(search))
    tab_prec = w.VBox([cond_w, win_w, w.HBox([dur_w, near_w]), b_search])

    # 3. beam options
    t_w = w.FloatText(value=3.5, description="t (s)")
    tq_w = w.FloatText(value=0.0, description="torque (N m)")
    counter_w = w.Text(value="", description="counter")
    keep_w = w.Text(value="", description="keep on")
    b_opts = w.Button(description="beam options")
    use_w = w.IntText(value=0, description="option #")
    b_use = w.Button(description="use as plan")
    state = {}

    def opts():
        res = tools.beam_options(shot_w.value, t_w.value,
                                 [x.strip() for x in counter_w.value.split(",") if x.strip()],
                                 tq_w.value,
                                 keep_on=[x.strip() for x in keep_w.value.split(",") if x.strip()])
        state["opts"] = res["options"]
        show({"n_feasible": res["n_feasible"], "options": [
            f"#{i} off {o['off']} counter MW {o['counter_MW_mean']} (max {o['counter_MW_max']}, "
            f"limit {o['counter_limit_MW']}) total {o['total_power_MW']} MW torque "
            f"{o['net_torque_Nm']} feasible {o['feasible']}"
            + ("" if o["feasible"] or not o.get("keep_on_max_fraction") else
               f" | feasible with {o['keeps']} at {o['keep_on_max_fraction']} of their power: "
               f"total {o['at_max_fraction']['total_power_MW']} MW")
            for i, o in enumerate(res["options"])]})

    def use():
        o = state["opts"][use_w.value]
        ops = o["ops"] if o["feasible"] or not o.get("at_max_fraction") else o["at_max_fraction"]["ops"]
        plan_w.value = json.dumps({"shot": shot_w.value, "goal": "", "rationale": "",
                                   "ops": ops}, indent=1)

    b_opts.on_click(guard(opts))
    b_use.on_click(guard(use))
    tab_beams = w.VBox([w.HBox([t_w, tq_w]), w.HBox([counter_w, keep_w]), b_opts, w.HBox([use_w, b_use])])

    # 4. plan
    plan_w = w.Textarea(value=json.dumps({"shot": shot, "goal": "", "rationale": "", "ops": []},
                                         indent=1),
                        layout=w.Layout(width="95%", height="220px"))
    file_w = w.Text(placeholder="path/to/plan.json (e.g. written by an agent)",
                    layout=w.Layout(width="60%"))
    b_load = w.Button(description="load file")
    b_check = w.Button(description="check")
    b_pplot = w.Button(description="plot plan")

    def plan():
        return json.loads(plan_w.value)

    b_load.on_click(guard(lambda: setattr(plan_w, "value", Path(file_w.value).read_text())))
    b_check.on_click(guard(lambda: show(tools.check_plan(plan()))))
    b_pplot.on_click(guard(lambda: show(plot_actuators(pl.build(plan())))))
    tab_plan = w.VBox([plan_w, w.HBox([file_w, b_load]), w.HBox([b_check, b_pplot])])

    # 5. simulate
    name_w = w.Text(placeholder="run name (optional)", description="name")
    seeds_w = w.Text(value="1,2,3,4", description="seeds")
    confirm_w = w.Checkbox(value=False, description="I approve the GPU job")
    b_export = w.Button(description="export")
    b_sim = w.Button(description="simulate", button_style="warning")
    run_w = w.Text(description="run dir", layout=w.Layout(width="80%"))
    b_status = w.Button(description="status")
    b_res = w.Button(description="results")
    b_rplot = w.Button(description="plot results")

    def seeds():
        return [int(x) for x in seeds_w.value.split(",") if x.strip()]

    def export():
        res = tools.export_plan(plan(), name=name_w.value or None, seeds=seeds())
        run_w.value = res["run_dir"]
        show(res)

    def simulate():
        if not run_w.value:
            export()
        show(tools.simulate(run_w.value, seeds(), confirm=confirm_w.value))

    b_export.on_click(guard(export))
    b_sim.on_click(guard(simulate))
    b_status.on_click(guard(lambda: show(tools.simulation_status(run_w.value))))
    b_res.on_click(guard(lambda: show(tools.simulation_results(run_w.value))))
    b_rplot.on_click(guard(lambda: show(plot_results(run_w.value))))
    tab_sim = w.VBox([w.HBox([name_w, seeds_w]), w.HBox([b_export, confirm_w, b_sim]), run_w,
                      w.HBox([b_status, b_res, b_rplot])])

    tabs = w.Tab(children=[tab_shot, tab_prec, tab_beams, tab_plan, tab_sim])
    for i, title in enumerate(["1 shot", "2 precedents", "3 beam options", "4 plan", "5 simulate"]):
        tabs.set_title(i, title)
    display(w.VBox([shot_w, tabs, out]))
