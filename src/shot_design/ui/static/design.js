// Plain-JavaScript waveform editor. The server remains authoritative for channel
// availability, units, validation, source identity, and IGNITE export.
(function (scope) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const CHART = { width: 760, height: 320, left: 66, right: 22, top: 22, bottom: 48 };
  const state = {
    api: null,
    initialized: false,
    program: null,
    response: null,
    selectedKey: null,
    selectedIndex: null,
    baseline: new Map(),
    dirty: false,
    draft: 0,
    structureDirty: false,
    request: 0,
    timer: null,
    drag: null,
    preparing: false,
    suppressChartClick: false,
  };

  const node = (id) => document.querySelector(`#${id}`);
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const copyVertices = (vertices) => (vertices || []).map(({ t_s, y }) => ({ t_s, y }));
  const numberText = (value) => finite(value) ? String(Number(value.toPrecision(8))) : "";

  // Keep waveform corners within 5% of its scale. This only changes the handles;
  // the original samples remain authoritative until the user makes an edit.
  function compactVertices(vertices) {
    const points = copyVertices(vertices);
    if (points.length < 3) return points;
    const values = points.map((point) => point.y);
    const tolerance = Math.max(...values.map(Math.abs), 1e-12) * 0.05;
    const keep = new Set([0, points.length - 1]);
    // Preserve off/on boundaries, including small pulses: a total has no member
    // split at zero, so a simplified ramp must not turn those frames on.
    for (let i = 0; i < points.length - 1; i += 1) {
      if ((points[i].y === 0) !== (points[i + 1].y === 0)) {
        keep.add(i); keep.add(i + 1);
      }
    }
    const anchors = [...keep].sort((a, b) => a - b);
    const segments = anchors.slice(1).map((last, i) => [anchors[i], last]);
    while (segments.length) {
      const [first, last] = segments.pop();
      const a = points[first], b = points[last];
      let worst = tolerance, corner = null;
      for (let i = first + 1; i < last; i += 1) {
        const estimate = a.y + (b.y - a.y) * (points[i].t_s - a.t_s) / (b.t_s - a.t_s);
        const error = Math.abs(points[i].y - estimate);
        if (error > worst) { worst = error; corner = i; }
      }
      if (corner !== null) {
        keep.add(corner);
        segments.push([first, corner], [corner, last]);
      }
    }
    return [...keep].sort((a, b) => a - b).map((i) => points[i]);
  }

  function element(tag, attrs = {}, ...children) {
    const result = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (name.startsWith("on")) result.addEventListener(name.slice(2), value);
      else result.setAttribute(name, value === true ? "" : String(value));
    }
    result.append(...children.flat().filter((child) => child !== null && child !== undefined));
    return result;
  }

  function svgElement(tag, attrs = {}) {
    const result = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attrs)) result.setAttribute(name, String(value));
    return result;
  }

  function setMessage(target, items, className) {
    target.replaceChildren();
    if (!items?.length) return;
    target.append(element("ul", { class: className }, items.map((item) => element("li", {}, item))));
  }

  function emptyProgram(referenceShot) {
    return {
      schema_version: "shot-design/1",
      id: null,
      created: null,
      reference_shot: referenceShot,
      comparison_shots: [],
      start_s: 1,
      end_s: 5,
      notes: "",
      reference_digest: null,
      edits: {},
      units: {},
    };
  }

  function channel() {
    return state.response?.channels?.find((item) => item.key === state.selectedKey) || null;
  }

  function referenceBaseline(item) {
    if (!item || !state.program) return [];
    const tolerance = (state.response?.frame_s || 0.05) * 1e-4;
    return compactVertices((item.baseline_vertices || item.reference || []).filter((point) =>
      finite(point.t_s) && finite(point.y) &&
      point.t_s >= state.program.start_s - tolerance &&
      point.t_s < state.program.end_s - tolerance));
  }

  function baselineFor(item) {
    if (!item) return null;
    const cached = state.baseline.get(item.key);
    if (cached?.length) return cached;
    const baseline = referenceBaseline(item);
    if (baseline.length) state.baseline.set(item.key, baseline);
    return baseline.length ? baseline : null;
  }

  function invalidateDownloads() {
    for (const id of ["design-json", "design-ignite"]) {
      const link = node(id);
      link.hidden = true;
      link.removeAttribute("href");
    }
  }

  function markDirty() {
    if (!state.program) return;
    if (state.timer !== null) clearTimeout(state.timer);
    state.timer = null;
    state.draft += 1;
    state.dirty = true;
    invalidateDownloads();
    node("design-dirty").textContent = "Unsaved changes";
  }

  function markStructureDirty() {
    markDirty();
    state.structureDirty = true;
    state.selectedIndex = null;
    node("design-prepare").disabled = true;
    node("design-feedback").textContent = "Click Check preview to load the updated shot and time window.";
    renderChannel();
  }

  function updateDownloads() {
    invalidateDownloads();
    const id = state.program?.id;
    if (!id || state.dirty) return;
    const json = node("design-json");
    json.setAttribute("href", `/api/design/${encodeURIComponent(id)}/json`);
    json.hidden = false;
    if (state.response?.validation?.can_export) {
      const ignite = node("design-ignite");
      ignite.setAttribute("href", `/api/design/${encodeURIComponent(id)}/ignite`);
      ignite.hidden = false;
    }
  }

  function renderValidation() {
    const validation = state.response?.validation || {};
    const errors = validation.errors || [];
    const warnings = validation.warnings || [];
    node("design-save").disabled = validation.can_save === false;
    node("design-prepare").hidden = !validation.needs_seed;
    node("design-prepare").disabled = state.preparing || state.structureDirty;
    node("design-feedback").textContent = validation.needs_seed ?
      "Waveforms loaded. You can edit and save now. Prepare IGNITE input when you want to export for simulation." :
      validation.can_export ? "Waveforms and IGNITE input are ready." :
      errors.join(" ") || "Enter a reference shot and click Check preview.";
    setMessage(node("design-errors"), errors, "design-error-list");
    setMessage(node("design-warnings"), warnings, "design-warning-list");
    const status = errors.length ? `${errors.length} check${errors.length === 1 ? "" : "s"} block export` :
      validation.can_export ? "Ready to export" : "Export unavailable";
    node("design-status").textContent = status;
    node("design-status").setAttribute("data-state", errors.length ? "error" :
      validation.can_export ? "ready" : "pending");
  }

  function updateProgramForm() {
    if (!state.program) return;
    node("design-reference").value = [state.program.reference_shot,
      ...(state.program.comparison_shots || [])].join(", ");
    node("design-start").value = numberText(state.program.start_s);
    node("design-end").value = numberText(state.program.end_s);
    node("design-notes").value = state.program.notes || "";
  }

  function readInteger(raw, label) {
    const text = String(raw ?? "").trim();
    const value = Number(text);
    if (!/^[0-9]+$/.test(text) || !Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`${label} must be a whole shot number`);
    }
    return value;
  }

  function readNumber(raw, label) {
    const text = String(raw ?? "").trim();
    const value = Number(text);
    if (!text || !Number.isFinite(value)) throw new Error(`${label} must be a finite number`);
    return value;
  }

  function syncProgramForm() {
    const shots = [...new Set(String(node("design-reference").value || "")
      .split(/[\s,]+/).filter(Boolean).map((value) => readInteger(value, "Reference")))];
    if (!shots.length) throw new Error("Enter at least one reference shot");
    if (shots.length > 6) throw new Error("Choose at most six reference shots");
    const [reference, ...unique] = shots;
    const start = readNumber(node("design-start").value, "Prediction start");
    const end = readNumber(node("design-end").value, "Prediction end");
    if (end <= start) throw new Error("Prediction end must be after start");
    if (!state.program) state.program = emptyProgram(reference);
    const referencesChanged = reference !== state.program.reference_shot ||
      JSON.stringify(unique) !== JSON.stringify(state.program.comparison_shots || []);
    if (state.program.proposal && (referencesChanged || start !== state.program.start_s ||
        end !== state.program.end_s)) {
      state.program.proposal = null;
      state.program.edits = {};
      state.baseline.clear();
      node("design-merge-status").textContent =
        "References or window changed. The previous average and its edits were cleared; preview now starts from the first reference.";
    }
    if (reference !== state.program.reference_shot || start !== state.program.start_s ||
        end !== state.program.end_s) {
      state.baseline.clear();
    }
    if (reference !== state.program.reference_shot) {
      state.program.reference_digest = null;
      state.program.units = {};
      state.program.edits = {};
      state.program.proposal = null;
    }
    state.program.id = null;
    state.program.created = null;
    state.program.reference_shot = reference;
    state.program.comparison_shots = unique;
    state.program.start_s = start;
    state.program.end_s = end;
    state.program.notes = node("design-notes").value;
  }

  function renderChannelOptions() {
    const select = node("design-channel");
    const channels = state.response?.channels || [];
    if (!channels.some((item) => item.key === state.selectedKey)) {
      state.selectedKey = channels.find((item) => item.key === "nbi.total" && item.editable)?.key ||
        channels.find((item) => item.key === "ech.total" && item.editable)?.key ||
        channels.find((item) => item.editable)?.key || channels[0]?.key || null;
      state.selectedIndex = null;
    }
    select.replaceChildren(...channels.map((item) => element("option", { value: item.key },
      `${item.label} (${item.key})${item.units ? ` — ${item.units}` : ""}${item.editable ? "" : " — unavailable"}`)));
    select.value = state.selectedKey || "";
  }

  function chartDomain(item) {
    const points = [
      ...(item?.reference || []),
      ...(item?.vertices || []),
      ...(item?.comparisons || []).flatMap((comparison) => comparison.vertices || []),
    ].filter((point) => finite(point.t_s) && finite(point.y));
    const t0 = finite(state.response?.context_start_s) ? state.response.context_start_s :
      Math.min(...points.map((point) => point.t_s), state.program.start_s);
    const t1 = Math.max(state.program.end_s,
      ...points.map((point) => point.t_s), state.program.start_s + (state.response?.frame_s || 0.05));
    let y0 = Math.min(...points.map((point) => point.y), 0);
    let y1 = Math.max(...points.map((point) => point.y), 1);
    if (!points.length) [y0, y1] = [0, 1];
    const padding = y0 === y1 ? Math.max(Math.abs(y0) * 0.1, 1) : (y1 - y0) * 0.1;
    return { t0, t1, y0: y0 - padding, y1: y1 + padding };
  }

  function geometry(domain) {
    const plotWidth = CHART.width - CHART.left - CHART.right;
    const plotHeight = CHART.height - CHART.top - CHART.bottom;
    return {
      x: (time) => CHART.left + (time - domain.t0) * plotWidth / (domain.t1 - domain.t0),
      y: (value) => CHART.top + (domain.y1 - value) * plotHeight / (domain.y1 - domain.y0),
      time: (x) => domain.t0 + (x - CHART.left) * (domain.t1 - domain.t0) / plotWidth,
      value: (y) => domain.y1 - (y - CHART.top) * (domain.y1 - domain.y0) / plotHeight,
    };
  }

  function pathData(points, scale) {
    return (points || []).filter((point) => finite(point.t_s) && finite(point.y))
      .map((point, index) => `${index ? "L" : "M"}${scale.x(point.t_s)} ${scale.y(point.y)}`).join(" ");
  }

  function addTrace(svg, points, className, scale, label) {
    const path = svgElement("path", { d: pathData(points, scale), class: className,
      fill: "none", "aria-label": label });
    svg.append(path);
  }

  function axisNumber(value) {
    if (!finite(value)) return "—";
    const magnitude = Math.abs(value);
    if (magnitude >= 10000 || (magnitude > 0 && magnitude < 0.001)) {
      return value.toExponential(2).replace(/\.0+(?=e)|(?<=\.[0-9])0+(?=e)/, "");
    }
    return String(Number(value.toPrecision(4)));
  }

  function niceStep(value) {
    const power = 10 ** Math.floor(Math.log10(value));
    const fraction = value / power;
    return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * power;
  }

  function addAxes(svg, domain, scale) {
    const plotBottom = CHART.height - CHART.bottom;
    const xValues = [domain.t0];
    const step = niceStep((state.program.end_s - state.program.start_s) / 4);
    for (let value = state.program.start_s;
         value <= state.program.end_s + step * 0.01; value += step) {
      xValues.push(Number(value.toPrecision(12)));
    }
    if (!xValues.some((value) => Math.abs(value - state.program.end_s) < 1e-9)) {
      xValues.push(state.program.end_s);
    }
    for (const value of [...new Set(xValues)]) {
      const x = scale.x(value);
      svg.append(svgElement("line", { x1: x, x2: x, y1: CHART.top, y2: plotBottom,
        class: "design-grid-line design-x-grid" }));
      const label = svgElement("text", { x, y: plotBottom + 22,
        class: "design-tick-label design-x-tick-label", "text-anchor": "middle" });
      label.textContent = `${axisNumber(value)} s`;
      svg.append(label);
    }
    for (let index = 0; index <= 4; index += 1) {
      const value = domain.y0 + (domain.y1 - domain.y0) * index / 4;
      const y = scale.y(value);
      svg.append(svgElement("line", { x1: CHART.left, x2: CHART.width - CHART.right,
        y1: y, y2: y, class: "design-grid-line design-y-grid" }));
      const label = svgElement("text", { x: CHART.left - 8, y: y + 4,
        class: "design-tick-label design-y-tick-label", "text-anchor": "end" });
      label.textContent = axisNumber(value);
      svg.append(label);
    }
  }

  function renderLegend(item) {
    const legend = node("design-legend");
    const entry = (className, label) => element("span", {},
      element("i", { class: className }), label);
    legend.replaceChildren(
      entry("legend-line reference", `Reference ${state.program.reference_shot} (seed)`),
      ...(item?.comparisons || []).map((comparison) =>
        entry("legend-line comparison", `Reference ${comparison.shot}`)),
      entry("legend-line edit", "Editable program"),
      entry("legend-history", "Locked history"),
    );
  }

  function selectPoint(index) {
    const item = channel();
    const point = item?.vertices?.[index];
    state.selectedIndex = point ? index : null;
    const editable = Boolean(point && item?.editable && !state.structureDirty);
    node("design-point-time").value = point ? numberText(point.t_s) : "";
    node("design-point-value").value = point ? numberText(point.y) : "";
    const anchor = index === 0 || index === item?.vertices?.length - 1;
    node("design-point-time").disabled = !editable || anchor;
    node("design-point-value").disabled = !editable;
    node("design-apply-point").disabled = !editable;
    node("design-delete-point").disabled = !editable || anchor;
    node("design-selection").textContent = point ?
      `Joint ${index + 1} of ${item.vertices.length}; value in ${item.units || "native units"}.${anchor ? " Endpoint time is fixed." : " Delete removes this joint."}` :
      "Click the waveform area to add a joint. Select a joint to edit or delete it.";
    renderChart();
  }

  function renderChart() {
    const svg = node("design-svg");
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${CHART.width} ${CHART.height}`);
    svg.setAttribute("role", "img");
    const item = channel();
    if (!item) {
      svg.setAttribute("aria-label", "No channel selected");
      return;
    }
    svg.setAttribute("aria-label", `${item.label} waveform in ${item.units || "native units"}`);
    const domain = chartDomain(item);
    const scale = geometry(domain);
    const predictionX = scale.x(state.program.start_s);
    svg.append(svgElement("rect", { x: CHART.left, y: CHART.top,
      width: Math.max(0, predictionX - CHART.left),
      height: CHART.height - CHART.top - CHART.bottom,
      class: "design-history", "aria-label": "Locked reference history" }));
    addAxes(svg, domain, scale);
    addTrace(svg, item.reference, "design-trace design-reference-trace", scale, "Reference waveform");
    for (const comparison of item.comparisons || []) {
      addTrace(svg, comparison.vertices, "design-trace design-comparison-trace", scale,
        `Reference ${comparison.shot}`);
    }
    addTrace(svg, item.vertices, "design-trace design-edit-trace", scale, "Editable prediction waveform");
    if (item.editable && !state.structureDirty) item.vertices.forEach((point, index) => {
      const circle = svgElement("circle", { cx: scale.x(point.t_s), cy: scale.y(point.y), r: 6,
        class: `design-point${index === state.selectedIndex ? " selected" : ""}`,
        tabindex: 0, role: "button",
        "aria-label": `${item.label}, ${numberText(point.t_s)} seconds, ${numberText(point.y)} ${item.units || "native units"}` });
      circle.addEventListener("click", (event) => { event.stopPropagation(); selectPoint(index); });
      circle.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectPoint(index); }
        if (event.key === "Delete" || event.key === "Backspace") {
          event.preventDefault(); state.selectedIndex = index; deleteSelected();
        }
      });
      circle.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        state.suppressChartClick = true;
        if (svg.setPointerCapture) svg.setPointerCapture(event.pointerId);
        state.drag = { index, pointerId: event.pointerId, domain, capture: svg };
        selectPoint(index);
      });
      svg.append(circle);
    });
    const axisLabel = svgElement("text", { x: CHART.left, y: CHART.height - 12,
      class: "design-axis-label" });
    axisLabel.textContent = `${numberText(domain.t0)} s  ·  prediction starts ${numberText(state.program.start_s)} s`;
    svg.append(axisLabel);
    const unitLabel = svgElement("text", { x: CHART.left, y: 15, class: "design-axis-label" });
    unitLabel.textContent = item.units || "native units";
    svg.append(unitLabel);
    renderLegend(item);
  }

  function renderChannel() {
    const item = channel();
    const controlsDisabled = !item?.editable || state.structureDirty;
    const canReset = Boolean(state.program?.edits?.[state.selectedKey] && baselineFor(item));
    const anchor = state.selectedIndex === 0 || state.selectedIndex === item?.vertices?.length - 1;
    node("design-point-time").disabled = controlsDisabled || state.selectedIndex === null || anchor;
    node("design-point-value").disabled = controlsDisabled || state.selectedIndex === null;
    node("design-apply-point").disabled = controlsDisabled || state.selectedIndex === null;
    node("design-reset").disabled = controlsDisabled || !canReset;
    node("design-delete-point").disabled = controlsDisabled || state.selectedIndex === null || anchor;
    node("design-simplify").disabled = controlsDisabled || item.vertices.length < 3;
    const availability = item?.editable ? "Editable" : item?.reason || "Unavailable";
    node("design-channel-details").textContent = item ?
      `${item.label} · ${item.key} · ${item.units || "native units"} · ${availability}` :
      "No channels returned by the server.";
    if (state.selectedIndex === null) {
      node("design-point-time").value = "";
      node("design-point-value").value = "";
      node("design-selection").textContent =
        "Click the waveform area to add a joint. Select a joint to edit or delete it.";
    }
    if (state.structureDirty) {
      node("design-selection").textContent =
        "Check preview after changing the reference shots or prediction window.";
    }
    if (!item?.editable) {
      state.selectedIndex = null;
      node("design-point-time").value = "";
      node("design-point-value").value = "";
      node("design-selection").textContent = item?.reason || "This channel cannot be edited.";
    }
    if (!item) node("design-legend").replaceChildren();
    renderChart();
  }

  function render() {
    updateProgramForm();
    renderChannelOptions();
    renderValidation();
    renderChannel();
    updateDownloads();
  }

  function acceptPreview(data, { saved = false, resetBaseline = false } = {}) {
    if (state.structureDirty) state.selectedIndex = null;
    state.structureDirty = false;
    state.response = data;
    state.program = data.program;
    if (resetBaseline) state.baseline.clear();
    for (const item of data.channels || []) {
      if (!state.program.edits?.[item.key]) item.vertices = compactVertices(item.vertices);
      const baseline = referenceBaseline(item);
      if (baseline.length) state.baseline.set(item.key, baseline);
      else if (!state.program.edits?.[item.key]) {
        state.baseline.set(item.key, copyVertices(item.vertices));
      } else state.baseline.delete(item.key);
    }
    if (saved) {
      state.dirty = false;
      node("design-dirty").textContent = `Saved revision ${state.program.id}`;
    } else if (!state.dirty) {
      node("design-dirty").textContent = state.program.id ?
        `Saved revision ${state.program.id}` : "New unsaved draft";
    }
    render();
  }

  async function previewNow({ resetBaseline = false } = {}) {
    if (!state.program) return;
    const request = ++state.request;
    const draft = state.draft;
    node("design-status").textContent = "Checking…";
    node("design-feedback").textContent = `Loading actuator data for shot ${state.program.reference_shot}…`;
    try {
      const { data } = await state.api("/api/design/preview", {
        method: "POST",
        body: JSON.stringify(state.program),
      });
      if (request !== state.request || draft !== state.draft) return;
      acceptPreview(data, { resetBaseline });
    } catch (error) {
      if (request !== state.request || draft !== state.draft) return;
      node("design-status").textContent = "Check failed";
      node("design-feedback").textContent = error.message;
      setMessage(node("design-errors"), [error.message], "design-error-list");
    }
  }

  function schedulePreview() {
    if (state.timer !== null) clearTimeout(state.timer);
    const draft = state.draft;
    state.timer = setTimeout(() => {
      state.timer = null;
      if (draft !== state.draft) return;
      previewNow();
    }, 250);
  }

  function cancelPendingPreview() {
    state.request += 1;
    if (state.timer !== null) clearTimeout(state.timer);
    state.timer = null;
  }

  function setEditedVertices(item, { alreadyDirty = false } = {}) {
    state.program.id = null;
    state.program.created = null;
    state.program.edits[item.key] = copyVertices(item.vertices);
    if (!alreadyDirty) markDirty();
    node("design-reset").disabled = false;
  }

  function applySelected() {
    if (state.structureDirty) {
      throw new Error("Check preview after changing the reference shots or prediction window");
    }
    syncProgramForm();
    const item = channel();
    const index = state.selectedIndex;
    if (!item?.editable || index === null || !item.vertices[index]) return;
    const anchor = index === 0 || index === item.vertices.length - 1;
    const time = anchor ? item.vertices[index].t_s : readNumber(node("design-point-time").value, "Point time");
    const value = readNumber(node("design-point-value").value, "Point value");
    const lower = index ? item.vertices[index - 1].t_s : state.program.start_s;
    const upper = index + 1 < item.vertices.length ? item.vertices[index + 1].t_s :
      state.program.end_s - (state.response?.frame_s || 0.05);
    if (time < lower || time > upper || (index && time === lower) ||
        (index + 1 < item.vertices.length && time === upper)) {
      throw new Error(`Point time must stay ordered between ${numberText(lower)} and ${numberText(upper)} s`);
    }
    item.vertices[index] = { t_s: time, y: value };
    setEditedVertices(item);
    renderChannel();
    schedulePreview();
  }

  function resetSelected() {
    if (state.structureDirty) return;
    syncProgramForm();
    const item = channel();
    const baseline = baselineFor(item);
    if (!item?.editable || !baseline) return;
    delete state.program.edits[state.selectedKey];
    item.vertices = copyVertices(baseline);
    state.program.id = null;
    state.program.created = null;
    state.selectedIndex = null;
    markDirty();
    renderChannel();
    return previewNow().then(() => {
      node("design-selection").textContent = state.program.proposal ?
        "Averaged waveform restored." : "Reference waveform restored.";
    });
  }

  function chartPosition(event) {
    const svg = node("design-svg");
    const inverse = svg.getScreenCTM?.()?.inverse();
    if (inverse) return {
      x: inverse.a * event.clientX + inverse.c * event.clientY + inverse.e,
      y: inverse.b * event.clientX + inverse.d * event.clientY + inverse.f,
    };
    const rect = svg.getBoundingClientRect();
    return { x: (event.clientX - rect.left) * CHART.width / rect.width,
      y: (event.clientY - rect.top) * CHART.height / rect.height };
  }

  function addJoint(event) {
    if (state.suppressChartClick) { state.suppressChartClick = false; return; }
    const item = channel();
    if (state.structureDirty || !item?.editable || !item.vertices.length) return;
    const { x, y } = chartPosition(event);
    if (y < CHART.top || y > CHART.height - CHART.bottom) return;
    const scale = geometry(chartDomain(item));
    const time = Number(scale.time(x).toFixed(3));
    const points = item.vertices;
    if (time <= points[0].t_s || time >= points[points.length - 1].t_s) return;
    const existing = points.findIndex((point) => Math.abs(point.t_s - time) < 0.001);
    if (existing >= 0) { selectPoint(existing); return; }
    syncProgramForm();
    const index = points.findIndex((point) => point.t_s > time);
    points.splice(index, 0, { t_s: time, y: scale.value(y) });
    setEditedVertices(item);
    selectPoint(index);
    schedulePreview();
  }

  function deleteSelected() {
    const item = channel(), index = state.selectedIndex;
    if (state.structureDirty || !item?.editable || index === null || index <= 0 ||
        index >= item.vertices.length - 1) return;
    syncProgramForm();
    item.vertices.splice(index, 1);
    state.selectedIndex = null;
    setEditedVertices(item);
    renderChannel();
    schedulePreview();
  }

  function simplifySelected() {
    const item = channel();
    if (state.structureDirty || !item?.editable) return;
    const simplified = compactVertices(item.vertices);
    if (simplified.length === item.vertices.length) return;
    syncProgramForm();
    item.vertices = simplified;
    state.selectedIndex = null;
    setEditedVertices(item);
    renderChannel();
    schedulePreview();
  }

  async function mergeReferences() {
    cancelPendingPreview();
    const button = node("design-merge");
    button.disabled = true;
    try {
      syncProgramForm();
      if (!state.program.comparison_shots.length) throw new Error("Enter at least two reference shots to average.");
      markDirty();
      const request = ++state.request, draft = state.draft;
      node("design-merge-status").textContent = "Averaging compatible reference actuators…";
      const { data } = await state.api("/api/design/merge", {
        method: "POST", body: JSON.stringify({ ...state.program, edits: {}, proposal: null }),
      });
      if (request !== state.request || draft !== state.draft) return;
      state.selectedIndex = null;
      acceptPreview(data, { resetBaseline: true });
      const skipped = Object.keys(data.program.proposal?.skipped_channels || {}).length;
      node("design-merge-status").textContent = `Equal average loaded across compatible channels.${skipped ? ` ${skipped} unavailable channels retain the first reference.` : ""}`;
    } catch (error) {
      node("design-merge-status").textContent = error.message;
    } finally { button.disabled = false; }
  }

  async function saveDesign() {
    cancelPendingPreview();
    const button = node("design-save");
    button.disabled = true;
    try {
      syncProgramForm();
      const draft = state.draft;
      const { data } = await state.api("/api/design", {
        method: "POST",
        body: JSON.stringify(state.program),
      });
      if (draft !== state.draft) return;
      acceptPreview(data, { saved: true });
      await loadRevisions(state.program.id);
    } catch (error) {
      node("design-status").textContent = "Save failed";
      node("design-feedback").textContent = error.message;
      setMessage(node("design-errors"), [error.message], "design-error-list");
    } finally {
      button.disabled = state.response?.validation?.can_save === false && !state.structureDirty;
    }
  }

  async function prepareIgnite() {
    if (state.preparing || state.structureDirty) return;
    cancelPendingPreview();
    const request = ++state.request;
    const draft = state.draft;
    const button = node("design-prepare");
    const shot = state.program?.reference_shot;
    const progress = node("design-prepare-status");
    state.preparing = true;
    button.disabled = true;
    button.textContent = "Preparing IGNITE input…";
    progress.textContent = `Preparing shot ${shot} for IGNITE. This may take a few minutes; keep this page open.`;
    try {
      if (state.dirty || !state.program) syncProgramForm();
      const { data } = await state.api("/api/design/prepare", {
        method: "POST", body: JSON.stringify(state.program),
      });
      progress.textContent = `IGNITE preparation finished for shot ${shot}.`;
      if (request !== state.request || draft !== state.draft) {
        progress.textContent += " Click Check preview to refresh your current draft.";
        return;
      }
      state.preparing = false;
      acceptPreview(data);
    } catch (error) {
      progress.textContent = `IGNITE preparation for shot ${shot} failed: ${error.message}`;
      if (request === state.request && draft === state.draft) {
        node("design-feedback").textContent = error.message;
        setMessage(node("design-errors"), [error.message], "design-error-list");
      }
    } finally {
      state.preparing = false;
      button.textContent = "Prepare IGNITE input";
      button.disabled = state.structureDirty;
    }
  }

  async function loadRevisions(selected = null) {
    const { data } = await state.api("/api/design");
    const select = node("design-revisions");
    const revisions = Array.isArray(data) ? data : [];
    select.replaceChildren(element("option", { value: "" }, revisions.length ?
      "Choose a saved revision" : "No saved revisions"), ...revisions.map((item) =>
      element("option", { value: item.id },
        `Shot ${item.reference_shot} · ${item.start_s}–${item.end_s} s · ${item.id.slice(0, 8)}` +
        `${item.created ? ` · ${String(item.created).replace("T", " ").slice(0, 16)}` : ""}`)));
    select.value = selected || "";
  }

  async function reopenDesign(id = node("design-revisions").value) {
    if (!id) return;
    cancelPendingPreview();
    state.draft += 1;
    const draft = state.draft;
    const request = ++state.request;
    const { data } = await state.api(`/api/design/${encodeURIComponent(id)}`);
    if (request !== state.request || draft !== state.draft) return;
    state.dirty = false;
    state.selectedKey = null;
    state.selectedIndex = null;
    acceptPreview(data, { saved: true, resetBaseline: true });
  }

  async function refreshFromForm() {
    try {
      syncProgramForm();
      markDirty();
      await previewNow({ resetBaseline: true });
    } catch (error) {
      node("design-status").textContent = "Check input";
      node("design-feedback").textContent = error.message;
      setMessage(node("design-errors"), [error.message], "design-error-list");
    }
  }

  function handleDrag(event) {
    if (!state.drag || event.pointerId !== state.drag.pointerId) return;
    event.preventDefault();
    if (state.structureDirty) return;
    const item = channel();
    if (!item?.editable) return;
    if (!state.drag.changed) {
      try { syncProgramForm(); }
      catch (error) {
        setMessage(node("design-errors"), [error.message], "design-error-list");
        return;
      }
      state.drag.changed = true;
      markDirty();
    }
    const { x, y } = chartPosition(event);
    const scale = geometry(state.drag.domain);
    const index = state.drag.index;
    const previous = index ? item.vertices[index - 1].t_s + 0.001 : state.program.start_s;
    const next = index + 1 < item.vertices.length ? item.vertices[index + 1].t_s - 0.001 :
      state.program.end_s - (state.response?.frame_s || 0.05);
    const anchor = index === 0 || index === item.vertices.length - 1;
    const time = anchor ? item.vertices[index].t_s : Math.min(next, Math.max(previous, scale.time(x)));
    item.vertices[index] = { t_s: Number(time.toFixed(3)), y: scale.value(y) };
    state.selectedIndex = index;
    node("design-point-time").value = numberText(item.vertices[index].t_s);
    node("design-point-value").value = numberText(item.vertices[index].y);
    renderChart();
  }

  function endDrag(event) {
    if (!state.drag || event.pointerId !== state.drag.pointerId) return;
    const capture = state.drag.capture;
    const item = channel();
    const changed = state.drag.changed;
    state.drag = null;
    if (capture?.hasPointerCapture?.(event.pointerId)) {
      capture.releasePointerCapture(event.pointerId);
    }
    if (!item || !changed) return;
    setEditedVertices(item, { alreadyDirty: true });
    renderChannel();
    return previewNow();
  }

  async function initDesign(options = {}) {
    if (options.api) state.api = options.api;
    if (!state.api) throw new Error("ShotDesign.initDesign requires an API function");
    if (!state.initialized) {
      node("design-preview").addEventListener("click", (event) => {
        event.preventDefault();
        return refreshFromForm();
      });
      node("design-save").addEventListener("click", (event) => {
        event.preventDefault();
        return saveDesign();
      });
      node("design-prepare").addEventListener("click", (event) => {
        event.preventDefault();
        return prepareIgnite();
      });
      node("design-reload").addEventListener("click", (event) => {
        event.preventDefault();
        return reopenDesign();
      });
      node("design-channel").addEventListener("change", () => {
        state.selectedKey = node("design-channel").value;
        state.selectedIndex = null;
        renderChannel();
      });
      node("design-apply-point").addEventListener("click", (event) => {
        event.preventDefault();
        try { return applySelected(); }
        catch (error) { setMessage(node("design-errors"), [error.message], "design-error-list"); }
      });
      node("design-reset").addEventListener("click", (event) => {
        event.preventDefault();
        return resetSelected();
      });
      for (const [id, action] of [["design-delete-point", deleteSelected],
        ["design-simplify", simplifySelected], ["design-merge", mergeReferences]]) {
        node(id).addEventListener("click", (event) => {
          event.preventDefault();
          try { return action(); }
          catch (error) { setMessage(node("design-errors"), [error.message], "design-error-list"); }
        });
      }
      for (const id of ["design-reference", "design-start", "design-end"]) {
        node(id).addEventListener("input", markStructureDirty);
      }
      node("design-notes").addEventListener("input", markDirty);
      node("design-svg").addEventListener("pointermove", handleDrag);
      node("design-svg").addEventListener("pointerdown", () => { state.suppressChartClick = false; });
      node("design-svg").addEventListener("click", addJoint);
      node("design-svg").addEventListener("pointerup", endDrag);
      node("design-svg").addEventListener("pointercancel", endDrag);
      state.initialized = true;
    }
    try { await loadRevisions(); }
    catch (error) { setMessage(node("design-errors"), [error.message], "design-error-list"); }
  }

  async function openDesign(referenceShot) {
    const shot = Number(referenceShot);
    if (!Number.isSafeInteger(shot) || shot < 0) throw new Error("Reference must be a whole shot number");
    cancelPendingPreview();
    state.draft += 1;
    state.program = emptyProgram(shot);
    state.response = null;
    state.selectedKey = null;
    state.selectedIndex = null;
    state.baseline.clear();
    state.dirty = false;
    state.structureDirty = false;
    updateProgramForm();
    invalidateDownloads();
    node("design-dirty").textContent = "New unsaved draft";
    await previewNow({ resetBaseline: true });
  }

  scope.ShotDesign = { initDesign, openDesign, openSavedDesign: reopenDesign };
})(globalThis);
