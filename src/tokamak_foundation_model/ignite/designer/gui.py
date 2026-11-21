"""Notebook GUI for the shot designer (ipywidgets): ``gui.app()``.

Tabs follow the designer's steps: look at a shot, find precedents, enumerate beam sets,
write and check a plan (or load one an agent wrote), export it, simulate it with IGNITE
and plot the result. The plotting functions also work on their own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------- agent
# Claude gets the user's prompt and the fixed instructions in PROMPT_ONLY.md, nothing
# else: no tools, no MCP server, no files, run from an empty directory outside the repo.
# Every DIII-D lookup (database limits, precedents, logbook) runs here, on the plan
# Claude returns, and its results are never sent back.
AGENT_LOGS = sim.RUNS_DIR / "_agent_logs"
PROMPT_SPEC = Path(__file__).with_name("PROMPT_ONLY.md")
# Empty working directory: keeps the repository, its CLAUDE.md and memory out of the call.
_CLAUDE_CWD = Path.home() / ".cache" / "ignite_designer_claude"
DEFAULT_PROMPT = ("Reproduce shot 204011 but with reactor-like rotation after 3.5 s and "
                  "keeping CER.")
ANSWER_SCHEMA = {
    "type": "object",
    "required": ["plan", "expected", "precedent_conditions", "logbook_query"],
    "properties": {
        "plan": {"type": "object", "required": ["shot", "goal", "rationale", "ops"],
                 "properties": {"shot": {"type": "integer"}, "goal": {"type": "string"},
                                "rationale": {"type": "string"},
                                "ops": {"type": "array", "items": {"type": "object"}}}},
        "expected": {"type": "string"},
        "precedent_conditions": {"type": "array", "items": {"type": "string"}},
        "logbook_query": {"type": "string"},
    },
}


def system_prompt() -> str:
    """The instructions sent with every prompt: PROMPT_ONLY.md below its ``---`` line."""
    return PROMPT_SPEC.read_text().split("\n---\n", 1)[1].strip()


def claude_command(prompt: str, model: str = "") -> list:
    """The ``claude`` call: prompt and instructions only, every tool and data source off."""
    import shutil

    claude = shutil.which("claude")
    if not claude:
        raise FileNotFoundError("claude (Claude Code) not found on this machine")
    cmd = [claude, "-p", prompt, "--system-prompt", system_prompt(),
           "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--disable-slash-commands", "--no-session-persistence",
           "--output-format", "stream-json", "--verbose",
           "--json-schema", json.dumps(ANSWER_SCHEMA)]
    return cmd + (["--model", model] if model else [])


def cross_check(answer: dict) -> dict:
    """The local checks of Claude's answer (nothing leaves the cluster): the plan against
    the database, precedents for its targets, and the logbook search it proposed."""
    plan = answer["plan"]
    out = {"check": tools.check_plan(plan)}
    conds = list(answer.get("precedent_conditions") or [])
    try:
        out["precedents"] = tools.search_precedents(conds, near=int(plan["shot"]), limit=10)
    except Exception as e:  # noqa: BLE001 - a bad condition is a finding, not a crash
        out["precedents"] = {"error": f"{type(e).__name__}: {e}"}
    q = answer.get("logbook_query") or ""
    if q:
        try:
            out["logbook"] = tools.search_logbook(q, conditions=conds, limit=10)
        except Exception as e:  # noqa: BLE001
            out["logbook"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _short(x, n=3000):
    s = x if isinstance(x, str) else json.dumps(x, indent=1, default=str)
    return s if len(s) <= n else s[:n] + f"\n... ({len(s) - n} more characters)"


def _card(title, body, color, note=""):
    import html

    from IPython.display import HTML, display
    display(HTML(
        f'<div style="border-left:4px solid {color};padding:4px 10px;margin:6px 0;'
        f'background:#fafafa"><b style="color:{color}">{html.escape(title)}</b>'
        f'<span style="color:#777;font-size:11px"> {html.escape(note)}</span>'
        f'<pre style="white-space:pre-wrap;font-size:12px;margin:4px 0 0 0">'
        f'{html.escape(body)}</pre></div>'))


def design(prompt: str, previous: Optional[dict] = None, model: str = "",
           status=None) -> Optional[dict]:
    """Send ``prompt`` (and, if given, Claude's own ``previous`` answer to revise) to
    Claude, show its reasoning and plan, then run the local cross-check and show it.

    Only the card "Sent to Claude" leaves the cluster, together with PROMPT_ONLY.md.
    Returns Claude's answer (``plan``, ``expected``, ``precedent_conditions``,
    ``logbook_query``), or None if it returned no plan. Works in any notebook front end
    (no widgets needed).
    """
    import subprocess
    import time

    from IPython.display import Markdown, display

    def say(text):
        if status is not None:
            status.value = text

    msg = prompt.strip()
    if previous:
        msg += "\n\nYour previous plan, to revise:\n" + json.dumps(previous["plan"], indent=1)
    cmd = claude_command(msg, model)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    AGENT_LOGS.mkdir(parents=True, exist_ok=True)
    _CLAUDE_CWD.mkdir(parents=True, exist_ok=True)
    log = AGENT_LOGS / f"agent_{stamp}.jsonl"
    t0 = time.time()
    answer, cost = None, None
    display(Markdown("### 1. Claude (sees only what is in the first card)"))
    _card("Sent to Claude", msg, "#333",
          f"+ the fixed instructions {PROMPT_SPEC.name}; no tools, no data")
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=_CLAUDE_CWD, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line)
            say(f"Claude is working ... {time.time() - t0:.0f} s")
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "system" and e.get("subtype") == "init":
                # Proof of isolation, from Claude Code itself. StructuredOutput is the
                # answer format of --json-schema, not a data source.
                if set(e.get("tools") or []) - {"StructuredOutput"} or e.get("mcp_servers"):
                    p.kill()
                    _card("Stopped", "Claude was given tools or MCP servers", "#b00")
                    break
            elif e.get("type") == "assistant":
                for c in e["message"].get("content", []):
                    if c.get("type") == "thinking" and (c.get("thinking") or "").strip():
                        _card("Claude's reasoning", c["thinking"], "#888")
                    elif c.get("type") == "text" and c["text"].strip():
                        display(Markdown(c["text"]))
            elif e.get("type") == "result":
                answer, cost = e.get("structured_output"), e.get("total_cost_usd")
                if e.get("is_error"):
                    _card("Claude returned an error", str(e.get("result")), "#b00")
        p.wait()
    took = f"{time.time() - t0:.0f} s" + (f", ${cost:.2f}" if cost else "")
    if not answer:
        _card("No plan returned", f"{took}; transcript {log}", "#b00")
        say("no plan returned")
        return None
    _card("Claude's plan", json.dumps(answer["plan"], indent=1), PLAN)
    _card("Claude expects", answer.get("expected", ""), PLAN)

    display(Markdown("### 2. Local cross-check (on the cluster; not sent to Claude)"))
    say(f"Claude answered in {took}; checking locally ...")
    try:
        res = cross_check(answer)
    except Exception as e:  # noqa: BLE001 - e.g. an op the builder rejects
        _card("The plan does not build", f"{type(e).__name__}: {e}", "#b00",
              "edit your prompt and send again")
        say(f"Claude answered in {took}; the plan does not build")
        return answer
    chk = res["check"]
    verdict = "within the database" if chk["ok"] else "violates the database"
    _card(f"Database check: {verdict}",
          _short({"findings": chk["findings"], "beam_margins": chk["beam_margins"]}),
          "#2a7" if chk["ok"] else "#b00")
    _card("Measured vs plan", _short(chk["summary"]), MEASURED)
    plot_actuators(pl.build(answer["plan"]))
    plt.show()
    _card("Precedents for Claude's conditions", _short(res["precedents"], 2500), MEASURED,
          ", ".join(answer.get("precedent_conditions") or []))
    if "logbook" in res:
        _card("Logbook search", _short(res["logbook"], 2500), MEASURED,
              answer.get("logbook_query", ""))
    say(f"Claude answered in {took}; checked locally.")
    answer["_check_ok"] = bool(chk["ok"])
    return answer


def export(answer: dict) -> str:
    """Write the plan's IGNITE input (and Claude's answer) to a run directory; returns it."""
    r = tools.export_plan(answer["plan"])
    run = Path(r["run_dir"])
    (run / "claude_answer.json").write_text(
        json.dumps({k: v for k, v in answer.items() if not k.startswith("_")}, indent=1))
    _card("Plan exported", _short(r, 1500), "#2a7")
    return str(run)


def simulate(run_dir: str, seeds=(1, 2, 3, 4)) -> dict:
    """Submit the IGNITE rollouts of an exported plan (one GPU job)."""
    from IPython.display import Markdown, display
    r = tools.simulate(run_dir, seeds=list(seeds), confirm=True)
    _card("Simulation submitted", _short(r), "#2a7")
    display(Markdown(f"When the job is done: `gui.plot_results('{run_dir}')`."))
    return r


def agent(prompt: str = DEFAULT_PROMPT, model: str = ""):
    """The same as :func:`design`, :func:`export` and :func:`simulate` behind a prompt box
    and buttons (needs a front end with ipywidgets; the box appears when the cell runs)."""
    import ipywidgets as w
    from IPython.display import display

    box = w.Textarea(value=prompt, layout=w.Layout(width="100%", height="80px"))
    prev_w = w.Checkbox(value=False, description="include Claude's previous plan (its own "
                        "output, no data) to revise it", indent=False,
                        layout=w.Layout(width="auto"))
    send_b = w.Button(description="Send to Claude", button_style="primary", icon="paper-plane")
    exp_b = w.Button(description="Export plan", icon="save", disabled=True)
    sim_b = w.Button(description="Simulate plan (GPU)", button_style="warning",
                     icon="rocket", disabled=True)
    status = w.HTML()
    steps = w.Output()
    state = {"answer": None, "run_dir": None}

    def on_send(_):
        steps.clear_output()
        send_b.disabled = exp_b.disabled = sim_b.disabled = True
        with steps:
            try:
                ans = design(box.value, state["answer"] if prev_w.value else None, model,
                             status)
            except FileNotFoundError as e:
                status.value = f"<b style='color:#b00'>{e}</b>"
                ans = None
        if ans:
            state["answer"] = ans
            exp_b.disabled = False
        send_b.disabled = False

    def on_export(_):
        exp_b.disabled = True
        with steps:
            state["run_dir"] = export(state["answer"])
        sim_b.disabled = False

    def on_sim(_):
        sim_b.disabled = True
        with steps:
            simulate(state["run_dir"])

    send_b.on_click(on_send)
    exp_b.on_click(on_export)
    sim_b.on_click(on_sim)
    display(w.VBox([
        w.HTML("<b>Your prompt</b> <span style='color:#777'>(only this text, plus the fixed "
               f"instructions in {PROMPT_SPEC.name}, is sent to Claude)</span>"),
        box, prev_w, w.HBox([send_b, exp_b, sim_b, status]), steps]))
