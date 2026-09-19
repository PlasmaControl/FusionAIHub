"use strict";

// Conventions follow src/shot_design/ui/static/app.js: no framework, one
// module-level state object, every refusal read off the endpoint's own
// {"error": ...} body and shown to the person rather than swallowed.

// State the page keeps: the marks not yet saved, and which window is drawn.
const marks = [];
let currentEvent = null;
let currentShot = null;
// The request in flight, so a newer one can cancel it. Serialising instead -
// holding a new request until the running one finishes - wedges the whole
// page behind a single slow render: an AE shot that is not yet cached is a
// multi-minute PTDATA fetch, and until this was a cancellation the reviewer
// could not so much as change event while one was running.
let inFlight = null;
// A relayout the page caused by drawing is not a relayout the reviewer asked
// for. Without this, every render re-triggers the handler that asked for it.
let redrawing = false;
// Only the newest request may paint. A slow whole-shot answer that lands
// after the reviewer has already zoomed must not replace the zoomed figure.
let generation = 0;

const $ = (id) => document.getElementById(id);

// `level` is `true`/falsey for the error/plain pair every call site started
// with, plus "note": something the render wants said - NO LABEL ROW is the
// one there is - which is information about the data, not a failure, and
// must not be dressed in the red a failure gets.
function say(text, level) {
  const status = $("status");
  status.textContent = text;
  status.classList.toggle("error", level === true || level === "error");
  status.classList.toggle("note", level === "note");
}

async function getJSON(url, signal) {
  const response = await fetch(url, signal ? { signal } : undefined);
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    // Every refusal this app makes is {"error": ...}; a body that is not
    // JSON at all means something upstream of the app answered, and the
    // status line is the only place the reviewer would ever see that.
    throw new Error(`${response.status} ${response.statusText} from ${url}`);
  }
  if (!response.ok) throw new Error((body && body.error) || response.statusText);
  return body;
}

// ---------------------------------------------------------------- drawing

const AXIS_GAP = 0.05;

function axisNames(row) {
  // Plotly's first subplot is "x"/"y" with no index, the rest are numbered.
  return row === 1
    ? { x: "x", y: "y", xkey: "xaxis", ykey: "yaxis", legend: "legend" }
    : {
        x: `x${row}`, y: `y${row}`,
        xkey: `xaxis${row}`, ykey: `yaxis${row}`, legend: `legend${row}`,
      };
}

function traceFor(panel, row) {
  const name = axisNames(row);
  if (panel.kind === "heatmap") {
    // x, y and z go over untouched. Plotly reads N coordinates against N
    // columns as cell CENTRES and N+1 as cell EDGES, and `Panel` already
    // guarantees the arrays are one of those two; reindexing them here is
    // how every panel ends up half a cell out.
    return [{
      type: "heatmap", x: panel.x, y: panel.y, z: panel.z,
      zmin: panel.zmin, zmax: panel.zmax,
      showscale: false, showlegend: false,
      hovertemplate: "%{x:.1f} ms<br>%{y:.3g}<br>%{z:.3g}<extra></extra>",
      xaxis: name.x, yaxis: name.y,
    }];
  }
  return panel.y.map((values, i) => ({
    type: "scattergl", mode: "lines", x: panel.x, y: values,
    name: (panel.legend || [])[i] || `ch ${i}`, legend: name.legend,
    xaxis: name.x, yaxis: name.y,
  }));
}

function markShapes() {
  // Drawn over the whole figure height, so a mark made on one panel is
  // visible against every other one. Without this a reviewer marking four
  // intervals has nothing on screen but a count.
  return marks.map((mark) => ({
    type: "rect", xref: "x", yref: "paper",
    x0: mark.t_start, x1: mark.t_end, y0: 0, y1: 1,
    fillcolor: mark.category ? "#1b7837" : "#777777",
    opacity: 0.12, line: { width: 0 }, layer: "below",
  }));
}

function draw(payload) {
  const panels = payload.panels;
  const traces = [];
  const shapes = markShapes();
  const annotations = [];
  const layout = {
    title: {
      text: `${payload.event} - shot ${payload.shot}` +
            (payload.note ? ` - ${payload.note}` : ""),
      font: { size: 14 },
    },
    height: 220 * panels.length + 130,
    dragmode: "select",
    selectdirection: "h",
    showlegend: true,
    // Room on the right only when something is going to be drawn there.
    margin: {
      l: 64, t: 56, b: 48,
      r: panels.some((panel) => panel.kind !== "heatmap") ? 150 : 24,
    },
    shapes,
    annotations,
  };
  // Domains are set here rather than with `layout.grid` so each row's legend
  // can be pinned beside its own row; a grid leaves every legend stacked at
  // the default position, on top of one another.
  const height = (1 - AXIS_GAP * (panels.length - 1)) / panels.length;
  panels.forEach((panel, index) => {
    const row = index + 1;
    const name = axisNames(row);
    const top = 1 - index * (height + AXIS_GAP);
    const domain = [Math.max(0, top - height), top];
    traces.push(...traceFor(panel, row));
    layout[name.ykey] = {
      domain, anchor: name.x, title: { text: panel.ylabel },
      automargin: true,
    };
    layout[name.xkey] = {
      domain: [0, 1], anchor: name.y,
      title: { text: row === panels.length ? "Time (ms)" : "" },
      showticklabels: row === panels.length,
    };
    // Row 1's axis is the one the others follow; `matches: "x"` on "x"
    // itself is a self-reference plotly drops with a console warning.
    if (row > 1) layout[name.xkey].matches = "x";
    // Only a row that has traces in it gets a legend object; a heatmap row
    // would leave an unreferenced `legendN` in the layout.
    if (panel.kind !== "heatmap") {
      layout[name.legend] = {
        yref: "paper", y: domain[1], yanchor: "top",
        xref: "paper", x: 1.005, xanchor: "left", font: { size: 10 },
      };
    }
    annotations.push({
      text: panel.title, showarrow: false, xref: "paper",
      yref: `${name.y} domain`, x: 0, y: 1, yanchor: "bottom",
      xanchor: "left", font: { size: 12 },
    });
    // A filled rect over a spectrogram washes the very image under review;
    // darker reads as less power on every sequential colormap, biasing the
    // reviewer toward under-calling the mode. Boundary lines mark the band
    // without touching what is inside it.
    panel.bands.forEach(([low, high]) => {
      [low, high].forEach((edge) => {
        if (edge === null) return;
        shapes.push({
          type: "line", xref: "paper", x0: 0, x1: 1,
          yref: name.y, y0: edge, y1: edge,
          line: { width: 1, dash: "dot", color: "black" },
        });
      });
    });
    panel.hlines.forEach((level) => {
      if (level === null) return;
      shapes.push({
        type: "line", xref: "paper", x0: 0, x1: 1,
        yref: name.y, y0: level, y1: level,
        line: { width: 1, dash: "dash", color: "#b2182b" },
      });
    });
  });

  const figure = $("figure");
  redrawing = true;
  Plotly.react(figure, traces, layout, {
    displaylogo: false,
    responsive: true,
    // The reviewer asked to be able to scroll through a long spectrogram;
    // the wheel is how they will try to do it.
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso2d", "toggleSpikelines"],
  }).then(() => {
    // Plotly emits its own relayout while laying the new figure out. Clear
    // the flag only once that has drained, or the first real pan is eaten.
    setTimeout(() => { redrawing = false; }, 0);
  });
  figure.removeAllListeners?.("plotly_relayout");
  figure.on("plotly_relayout", onRelayout);
}

// ------------------------------------------------------------ server tiles

// Panning and zooming re-render the visible window server-side, so a window
// is as sharp as the screen can show however wide it is. Debounced, because
// a drag emits relayout continuously and each one is a server render.
let debounce = null;

function windowFrom(event) {
  // A zoom on any row reports that row's axis, and every row after the
  // first matches "x", so the keys that arrive depend on where the reviewer
  // dragged. Take whichever range came.
  for (const key of Object.keys(event)) {
    const match = /^xaxis\d*\.range\[0\]$/.exec(key);
    if (!match) continue;
    const other = key.replace("range[0]", "range[1]");
    if (event[other] === undefined) continue;
    return [Number(event[key]), Number(event[other])];
  }
  for (const key of Object.keys(event)) {
    // Double-click resets the axis. That is a request for the whole shot,
    // and answering it from the coarse whole-shot render is the only way
    // back out of a zoom.
    if (/^xaxis\d*\.autorange$/.test(key) && event[key] === true) return [];
  }
  return null;
}

function onRelayout(event) {
  if (redrawing) return;
  const asked = windowFrom(event);
  if (asked === null) return;
  if (asked.length === 2 && !(asked[1] > asked[0])) return;
  clearTimeout(debounce);
  debounce = setTimeout(() => loadPanels(...asked), 250);
}

let ticker = null;

function waiting(text) {
  const started = Date.now();
  say(text);
  clearInterval(ticker);
  ticker = setInterval(() => {
    say(`${text} - ${Math.round((Date.now() - started) / 1000)}s`);
  }, 1000);
}

async function loadPanels(t0, t1) {
  // Cancel and invalidate BEFORE the guard, not after it. This call is how
  // the page says the previous window is no longer wanted, and the case
  // where it cannot draw a new one - an event whose roster is empty or
  // unreadable - is exactly the case where the old request must not be
  // left to land and paint the event the reviewer has already left, under
  // a status line and a ticker still describing it.
  if (inFlight) inFlight.abort();
  ++generation;
  clearInterval(ticker);
  if (currentEvent === null || !Number.isFinite(currentShot)) {
    Plotly.purge($("figure"));
    return;
  }
  const controller = new AbortController();
  inFlight = controller;
  const mine = generation;
  // Both or neither: `t0` alone is silently the whole shot, and the only
  // sign of it is a null `t_range` in the answer.
  const whole = t0 === undefined || t1 === undefined;
  const query = whole ? "" : `&t0=${t0}&t1=${t1}`;
  waiting(whole
    ? "loading the whole shot - a shot not yet cached is fetched over PTDATA "
      + "and that takes minutes"
    : `rendering ${Math.round(t0)}-${Math.round(t1)} ms`);
  try {
    const payload = await getJSON(
      `/api/panels?event=${encodeURIComponent(currentEvent)}` +
      `&shot=${currentShot}${query}`, controller.signal);
    if (mine !== generation) return;
    draw(payload);
    clearInterval(ticker);
    const span = payload.t_range
      ? `${Math.round(payload.t_range[0])}-${Math.round(payload.t_range[1])} ms`
      : "whole shot";
    say([span, payload.note].filter(Boolean).join(" - "),
        payload.note ? "note" : undefined);
  } catch (error) {
    // A request this page cancelled to make room for a newer one is not a
    // failure to report; the newer one owns the status line now.
    if (error.name === "AbortError" || mine !== generation) return;
    clearInterval(ticker);
    say(String(error.message || error), true);
  } finally {
    if (inFlight === controller) inFlight = null;
  }
}

// ------------------------------------------------------------------- marks

function selectedRange() {
  const figure = $("figure");
  const selections = figure.layout && figure.layout.selections;
  if (!selections || !selections.length) {
    throw new Error("drag a time range on the figure first");
  }
  // The modebar can still be put into Lasso Select. A lasso writes
  // {type: "path"} with x0/x1 both null, so reject anything that is not a
  // plain rectangle before touching x0/x1.
  const box = selections[selections.length - 1];
  if (box.type !== "rect" || box.x0 === null || box.x0 === undefined) {
    throw new Error("use the Box Select tool, not Lasso");
  }
  return [Math.min(box.x0, box.x1), Math.max(box.x0, box.x1)];
}

function mark(category) {
  return () => {
    let range;
    try { range = selectedRange(); }
    catch (error) { say(error.message, true); return; }
    if (!(range[1] > range[0])) {
      // `/api/save` refuses a zero-width interval; say so here rather than
      // after the reviewer has pressed Save on a batch of them.
      say("that selection has no width; drag across a range", true);
      return;
    }
    marks.push({ t_start: range[0], t_end: range[1], category });
    // Clear the drag so a second click cannot silently record the same
    // interval twice, and repaint so the new mark is visible.
    Plotly.relayout($("figure"), { selections: [], shapes: currentShapes() });
    showMarks();
  };
}

function currentShapes() {
  // The band and threshold lines belong to the panels and must survive a
  // marks-only repaint; only the leading rects are ours to replace.
  const existing = ($("figure").layout || {}).shapes || [];
  return markShapes().concat(existing.filter((s) => s.type !== "rect"));
}

function showMarks() {
  const holder = $("marks");
  holder.textContent = "";
  holder.className = "mark-list";
  if (!marks.length) {
    holder.textContent = "no corrections yet";
  } else {
    marks.forEach((mark) => {
      const span = document.createElement("span");
      span.className = `mark ${mark.category ? "present" : "absent"}`;
      span.textContent =
        `${mark.category ? "present" : "absent"} ` +
        `${Math.round(mark.t_start)}-${Math.round(mark.t_end)} ms`;
      holder.appendChild(span);
    });
  }
  $("save").disabled = marks.length === 0;
  $("clear").disabled = marks.length === 0;
}

function forgetMarks() {
  marks.length = 0;
  showMarks();
}

async function save() {
  const button = $("save");
  button.disabled = true;
  try {
    const response = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Numbers, not strings: `/api/save` refuses a "178642" with a 422.
      body: JSON.stringify({
        event: currentEvent,
        shot: Number(currentShot),
        marks: marks.map((mark) => ({
          t_start: Number(mark.t_start),
          t_end: Number(mark.t_end),
          category: Number(mark.category),
        })),
      }),
    });
    let body = null;
    try { body = await response.json(); } catch (error) { body = null; }
    if (!response.ok) {
      say((body && body.error) || `save failed: ${response.status}`, true);
      return;
    }
    const written = body.n;
    forgetMarks();
    const saved = `saved ${written} correction(s) to ${body.written}`;
    say(saved);
    try {
      await loadShots(currentEvent, currentShot);
      // The figure's mark rects came from `marks`, which is now empty.
      Plotly.relayout($("figure"), { shapes: currentShapes() });
    } catch (error) {
      // The file IS on disk; only the refresh after it failed. Reporting
      // this as "save failed" would send the reviewer back to redo a
      // correction that is already written, and corrections are
      // append-only, so redoing one writes it twice.
      say(`${saved}, but refreshing the view failed: ` +
          `${error.message || error}`, true);
    }
  } catch (error) {
    say(`save failed: ${error.message || error}`, true);
  } finally {
    showMarks();
  }
}

// ------------------------------------------------------------ event / shot

async function loadShots(event, keepShot) {
  const select = $("shot");
  const rows = $("roster").querySelector("tbody");
  rows.textContent = "";
  let body;
  try {
    body = await getJSON(`/api/shots?event=${encodeURIComponent(event)}`);
  } catch (error) {
    // A roster the page could not read is why `/api/events` reported an
    // error for this event. Say which, and leave the shot list empty rather
    // than showing a stale one from the previous event.
    select.textContent = "";
    currentShot = NaN;
    // Like the empty-roster branch below: a tier and a holdout left standing
    // from the previous event read as this event's curation calls.
    $("tier").textContent = "";
    $("holdout").textContent = "";
    $("roster-error").textContent =
      `${event}: could not read shots.csv - ${error.message || error}`;
    return;
  }
  select.textContent = "";
  const header = document.createElement("tr");
  ["shot", "tier", "holdout", "reviewers", "corrections on disk"]
    .forEach((text) => {
      const th = document.createElement("th");
      th.textContent = text;
      header.appendChild(th);
    });
  rows.appendChild(header);
  body.shots.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.shot;
    option.textContent = `${row.shot}${row.tier ? ` (${row.tier})` : ""}`;
    select.appendChild(option);
    const tr = document.createElement("tr");
    [
      String(row.shot),
      row.tier || "-",
      row.holdout ? "holdout" : "-",
      (row.reviewers || []).join("; ") || "-",
      String(row.n_corrections),
    ].forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    });
    if (Number(row.shot) === Number(keepShot)) tr.className = "current";
    rows.appendChild(tr);
  });
  if (!body.shots.length) {
    currentShot = NaN;
    $("tier").textContent = "";
    $("holdout").textContent = "";
    say(`${event} has no shots in its roster`, true);
    return;
  }
  if (keepShot !== undefined && keepShot !== null) {
    select.value = String(keepShot);
  }
  currentShot = Number(select.value);
  const record = body.shots.find((r) => Number(r.shot) === currentShot) || {};
  $("tier").textContent = record.tier || "";
  $("holdout").textContent = record.holdout ? "holdout" : "";
}

async function start() {
  const body = await getJSON("/api/events");
  const select = $("event");
  if (!body.events.length) {
    say("no event under label_tables has a shots.csv", true);
    return;
  }
  body.events.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.event;
    // A roster that would not parse still gets an entry - it is the only
    // place the reviewer learns which file to go and fix - and it says so.
    option.textContent = row.error
      ? `${row.event} (roster unreadable)`
      : `${row.event} (${row.n_shots})`;
    option.dataset.guidance = row.guidance;
    option.dataset.builder = row.builder;
    if (row.error) option.dataset.error = row.error;
    select.appendChild(option);
  });

  async function pickEvent() {
    currentEvent = select.value;
    const option = select.selectedOptions[0];
    // `guidance` is HTML by contract - `panels.guidance` returns a builder's
    // GUIDANCE string, which is this app's own prose, not anything posted.
    $("guidance").innerHTML = option.dataset.guidance || "";
    $("builder").textContent = option.dataset.builder === "generic"
      ? "generic panels" : "panels for this event";
    $("roster-error").textContent = option.dataset.error
      ? `${currentEvent}: ${option.dataset.error}`
      : "";
    // Whatever the status line last said was about the event being left.
    say("");
    forgetMarks();
    Plotly.purge($("figure"));
    await loadShots(currentEvent);
    loadPanels();
  }

  select.addEventListener("change", pickEvent);
  $("shot").addEventListener("change", () => {
    currentShot = Number($("shot").value);
    say("");
    forgetMarks();
    loadPanels();
  });
  $("present").addEventListener("click", mark(1));
  $("absent").addEventListener("click", mark(0));
  $("save").addEventListener("click", save);
  $("clear").addEventListener("click", () => {
    forgetMarks();
    Plotly.relayout($("figure"), { shapes: currentShapes() });
  });
  showMarks();
  await pickEvent();
}

start().catch((error) => say(String(error.message || error), true));
