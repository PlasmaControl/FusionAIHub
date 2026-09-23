// Saved revision status and results. Editor changes only affect submission eligibility.
(function (scope) {
  "use strict";
  const element = scope.ShotDesignDOM;
  const node = id => document.querySelector(`#design-${id}`);
  const number = value => typeof value === "number" && Number.isFinite(value) ?
    String(Number(value.toPrecision(4))) : "—";
  const active = run => ["queued", "running"].includes(run?.status.state);

  function metricsView(metrics, id) {
    if (metrics.schema !== "shot-design-simulation-metrics-v1") {
      throw Error("This simulation uses an unsupported metrics format.");
    }
    const root = element("div", {}, element("p", { class: "small muted" },
      `${metrics.members} runs per arm. Error columns are scaled by measured variation. Skill compares the ensemble with the hold-last-frame baseline.`));
    const held = new Set(metrics.held);
    for (const [name, scores] of Object.entries(metrics.modalities)) if (scores.held) held.add(name);
    for (const name of held) root.append(element("p", {}, `${name}: not simulated (diagnostic absent)`));
    for (const [name, scores] of Object.entries(metrics.modalities)) {
      if (held.has(name)) continue;
      const columns = [
        ["Error vs measured", number(scores.nrmse.real)],
        ["Hold-last-frame error", number(scores.nrmse.persistence)],
        ["Skill vs holding the last frame", `${number(scores.skill)}${scores.skill < 0 ? " — worse than the hold-last-frame baseline" : ""}`],
        ["Edit effect", number(scores.effect)],
        ["Run-to-run noise", number(scores.noise)],
        ["Effect resolved?", scores.resolved === true ? "Yes" : scores.resolved === false ? "No — unresolved" : "Not available"],
      ];
      root.append(element("section", { class: "simulation-modality" },
        element("h4", {}, `${name} · ${scores.feature}`),
        element("div", { class: "table-wrap" }, element("table", {},
          element("thead", {}, element("tr", {}, columns.map(([label]) => element("th", { scope: "col" }, label)))),
          element("tbody", {}, element("tr", {}, columns.map(([, value]) => element("td", {}, value)))))),
        element("details", {}, element("summary", {}, "More scores"), element("dl", {}, [
          ["Ensemble error vs measured", scores.crps.real],
          ["Hold-last-frame absolute error", scores.crps.persistence],
          ["Seed-mean error", scores.nrmse.seed_mean],
          ["Spread / error", scores.spread_error],
          ["Edit effect / noise (2 or more is resolved)", scores.effect_to_noise],
        ].flatMap(([label, value]) => [element("dt", {}, label), element("dd", {}, number(value))]))),
        element("img", { src: `/api/design/${encodeURIComponent(id)}/simulate/panels/${encodeURIComponent(name)}.png`,
          alt: `${name}: measured signal and simulated real and proposed actuation over shot time`, loading: "lazy" })));
    }
    return root;
  }

  function create({ api, pollMs = 15000 }) {
    const runs = new Map();
    let id = null, pollTimer = null, clockTimer = null;
    let editor = { savedId: null, dirty: false, canExport: false };
    let rendered = [];

    function stopTimers() {
      clearTimeout(pollTimer); clearTimeout(clockTimer);
      pollTimer = clockTimer = null;
    }

    function render() {
      const run = runs.get(id);
      const busy = run?.checking || run?.submitting || !run?.statusKnown || active(run);
      node("simulate").disabled = !id || editor.savedId !== id || editor.dirty || !editor.canExport || busy;
      node("simulate").textContent = run?.result ? "Run again" : "Simulate";
      const status = run?.status || {};
      let message = "";
      if (run?.submitting) message = "Submitting…";
      else if (run?.checking) message = "Checking simulation…";
      else if (active(run)) {
        const started = Date.parse(status.started || status.submitted);
        const seconds = Math.max(0, Math.floor((Date.now() - started) / 1000));
        message = `${status.state === "running" ? "Running" : "Queued"}${Number.isFinite(seconds) ? ` · ${Math.floor(seconds / 60)}m ${seconds % 60}s elapsed` : ""}`;
      } else if (status.state === "complete") message = "Simulation complete.";
      else if (status.state === "failed") message = `Simulation failed: ${status.error}`;
      if (run?.error) message = run.error;
      node("simulate-status").textContent = message;
      const report = node("simulate-report");
      report.hidden = !run?.result;
      if (run?.result) report.setAttribute("href", `/api/design/${encodeURIComponent(id)}/simulate/report`);
      else report.removeAttribute("href");
      const key = [id, run?.content, run?.metricsError, run?.result, editor.dirty, active(run)];
      if (key.every((value, i) => value === rendered[i])) return;
      rendered = key;
      const results = node("simulation-results");
      results.hidden = !run?.result;
      results.replaceChildren();
      if (!run?.result) return;
      results.append(element("h3", {}, "Simulation results"), element("p", {},
        `${active(run) ? "Previous result for" : "Result for"} revision ${id}.${editor.dirty ? " Unsaved edits are not included." : ""}`));
      if (run.metricsError) results.append(element("p", { class: "error" }, run.metricsError));
      else results.append(run.content || element("p", {}, "Loading simulation metrics…"));
    }

    function schedule() {
      stopTimers();
      if (!active(runs.get(id))) return;
      const selected = id;
      pollTimer = setTimeout(() => poll(selected), pollMs);
      const tick = () => {
        if (selected !== id || !active(runs.get(id))) return;
        render();
        clockTimer = setTimeout(tick, 1000);
      };
      clockTimer = setTimeout(tick, 1000);
    }

    async function poll(selected) {
      const run = runs.get(selected), request = ++run.request;
      try {
        const { data } = await api(`/api/design/${encodeURIComponent(selected)}/simulate`);
        if (request !== run.request) return;
        run.status = data;
        run.statusKnown = true;
        run.error = null;
        run.result ||= data.state === "complete" || data.has_result === true;
        if (run.result && (!run.content || data.state === "complete")) {
          try {
            const { data: metrics } = await api(`/api/design/${encodeURIComponent(selected)}/simulate/metrics`);
            if (request !== run.request) return;
            run.content = metricsView(metrics, selected);
            run.metricsError = null;
          } catch (error) {
            if (request !== run.request) return;
            run.metricsError = error.message;
          }
        }
      } catch (error) {
        if (request !== run.request) return;
        run.error = `Status check failed: ${error.message}`;
      }
      if (request !== run.request) return;
      run.checking = false;
      if (id === selected) { render(); schedule(); }
    }

    async function select(selected) {
      stopTimers();
      id = selected;
      if (!id) { render(); return; }
      if (!runs.has(id)) runs.set(id, { status: { state: "not_started" }, request: 0 });
      runs.get(id).checking = true;
      render();
      await poll(id);
    }

    async function submit() {
      const run = runs.get(id);
      if (node("simulate").disabled) return;
      if (run.result && !scope.confirm(`Run simulation again for revision ${id}?`)) return;
      const selected = id;
      stopTimers();
      run.request += 1;
      run.submitting = true;
      run.error = null;
      render();
      try {
        await api(`/api/design/${encodeURIComponent(selected)}/simulate`, { method: "POST" });
        run.status = { state: "queued", submitted: new Date().toISOString() };
        run.submitting = false;
        if (selected === id) render();
        await poll(selected);
      } catch (error) {
        run.submitting = false;
        run.error = `Submit failed: ${error.message}`;
        if (selected === id) render();
      }
    }

    return { select, submit, update: value => { editor = value; render(); } };
  }

  scope.ShotDesignSimulation = { create };
})(globalThis);
