// Ported from shot-recommender-system's shotrec/ui/static/app.js.
// Keep shotrec's DOM helpers and API-driven presentation. No retrieval or physics here.
const $ = (selector) => document.querySelector(selector);
const S = { shot: null, shotRequest: 0, eventRequest: 0 };
const FORECAST_TITLE = "Forecasts (model estimates)";
// Categorical colours identify the supplied evidence_kind, never quality or severity.
const COLOURS = { detector: "#4b74a8", heuristic: "#688591", forecast: "#827299", database: "#87817a" };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

// Measurements: 4 significant digits, no trailing zeros. Use scientific notation
// for |x| >= 1e5 or 0 < |x| < 1e-3 (thresholds use the original magnitude).
// Times are seconds with 3 decimals; confidence also has 3 decimals. Identifiers,
// dates, years and counts bypass rounding. Missing/non-finite values are always —.
// Eight examples: 7.13e14 -> 7.13e14; 892400 -> 8.924e5; .00012 -> 1.2e-4;
// 2.82 -> 2.82; .945678 -> 0.9457; 892 -> 892; 100000 -> 1e5; .001 -> 0.001.
function formatNumber(value, kind = "measurement") {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (kind === "identifier") return String(value);
  if (kind === "time" || kind === "confidence") return value.toFixed(3);
  const magnitude = Math.abs(value);
  if (magnitude >= 1e5 || (magnitude > 0 && magnitude < 1e-3)) {
    const [mantissa, exponent] = value.toExponential(3).split("e");
    return `${Number(mantissa)}e${Number(exponent)}`;
  }
  return String(Number(value.toPrecision(4)));
}

const identifier = (key) => /^(?:shot|refshot|ref_shot|source_shots|shot_range|year|campaign|run_id|mpid|mp_step|channel|cluster|schema_version|prompt_version|torch_threads|n|n_.*|.*_count)$/.test(key);
const timeSpan = (a, b) => `${formatNumber(a, "time")}–${formatNumber(b, "time")} s`;

// A missing measurement never becomes a zero, including nested records and invalid floats.
function display(value, key = "") {
  if (value === null || value === undefined || value === "" ||
      (typeof value === "number" && !Number.isFinite(value)) ||
      (typeof value === "string" && /^(?:nan|[+-]?infinity)$/i.test(value))) return "—";
  if (typeof value === "object") {
    if (Array.isArray(value)) return value.length ? value.map((v) => display(v, key)).join(", ") : "—";
    return Object.entries(value).map(([k, v]) => `${fieldName(k)}: ${display(v, k)}`).join("; ") || "—";
  }
  if (typeof value === "number") {
    if (identifier(key)) return formatNumber(value, "identifier");
    if (key.endsWith("_ms")) return formatNumber(value / 1000, "time");
    if (key.endsWith("_s")) return formatNumber(value, "time");
    return formatNumber(value, key === "confidence" ? "confidence" : "measurement");
  }
  return String(value);
}

function fieldName(key) { return key.endsWith("_ms") ? key.slice(0, -3) + " (s)" : key; }

// Disclosure limits: whole table cells get 320 characters / 5 CSS lines;
// shot-page blocks get 600 characters / 8 CSS lines. See .text-content in CSS.
const CELL_TEXT_LIMIT = 320;
const BLOCK_TEXT_LIMIT = 600;

function formatFlag(flag) {
  const message = String(flag.message ?? flag);
  const rule = message.match(/^(.+?) = \S+ ([<>=!]+) \S+: (.*)$/);
  if (rule) return `${rule[1]} = ${formatNumber(flag.value)} ${rule[2]} ${formatNumber(flag.limit)}: ${rule[3]}`;
  const envelope = message.match(/^(.+?) = \S+ is (below|above) the observed ([\d.e+\-]+|\?)-([\d.e+\-]+|\?) range(?: of (\d+) shots)? in the database -- .*$/);
  if (envelope) {
    const [, name, side, low, high, count] = envelope;
    const lo = side === "below" ? flag.limit : Number(low);
    const hi = side === "above" ? flag.limit : Number(high);
    return `${name} = ${formatNumber(flag.value)}; ${side} observed range ${formatNumber(lo)}–${formatNumber(hi)} (${count ? `${count} shots` : "database"}); not an operating limit`;
  }
  return caption(message);
}

// Browser-only wording. Unknown caveats pass through unchanged; MCP strings and
// status semantics are untouched. Never apply these rewrites to operator quotations.
function caption(value) {
  let text = String(value ?? "—");
  const dropped = text.match(/^(\d+) of those drops rest on evidence that carries: (.*)$/);
  if (dropped) return `${dropped[1]} excluded shots: ${caption(dropped[2])}`;
  const rules = [
    [/^text evidence is run scope:.*$/, "Run-level text, not shot-specific"],
    [/^no diagnostic coverage recorded; absence is not evidence$/, "Coverage unrecorded; absence unmeasured"],
    [/^no detector registered for (.*); text\/database evidence only$/, "$1: no detector; text/database evidence only"],
    [/^no detector for (.*) has run on this shot; absence is not evidence$/, "$1: detectors not run; absence unmeasured"],
    [/^the (.*) detectors ran on this shot but not over the (.*) window;.*$/, "$1: no coverage of $2; absence unmeasured"],
    [/^the (.*) detectors covered only (.*) of the (.*) window;.*$/, "$1: covered $2 of $3 only; outside unmeasured"],
    [/^the (.*) coverage of the (.*) window has (\d+) gap\(s\):.*$/, "$1: $3 gaps in $2 coverage; see covered intervals"],
    [/^observed via tokeye_transient, a class-agnostic transient detector:.*$/, "Class-agnostic transient; may be ELM-like, a sawtooth or a disruption precursor"],
    [/^(\d+) row\(s\) of evidence_kind (.*) match this phenomenon's rules.*$/, "$1 $2 rows; neither observations nor forecasts"],
    [/^(.*) scored ([\d.]+), below the ([\d.]+) evidence floor:.*$/, (_m, key, p, floor) => `${key}: ${formatNumber(Number(p), "confidence")}, below evidence floor ${formatNumber(Number(floor), "confidence")}`],
    [/^the quote is this shot's most informative logbook entry and does not mention (.*)$/, "Shot logbook quote; does not mention $1"],
    [/^this database holds (\d+) event row\(s\), all of them forecasts:.*$/, "$1 indexed rows, all forecasts; no observations"],
    [/^TEXT ONLY$/, "Text only"],
    [/^ranked on forecasts:.*$/, "Ranked on forecasts (model estimates)"],
    [/^ranked on model labels:.*score \(([^)]+)\).*$/, "Ranked on model labels ($1); no diagnostic evidence"],
    [/^ranked on a curated human list:.*$/, "Curated list only; no detector, model or logbook evidence"],
    [/^no observed evidence:.*$/, "No detector evidence for this phenomenon"],
    [/^no label evidence:.*$/, "No label model for this phenomenon"],
    [/^label not run on this shot, or no valid samples:.*$/, "Label unavailable: not run or no valid samples"],
    [/^no operator text names this phenomenon on this shot$/, "No shot-specific operator mention"],
    [/^operator log says NOT (.*)$/, "Operator reports no $1"],
    [/^no (.*) segment on this shot; the whole record was searched$/, "No $1 segment; searched whole record"],
    [/^(\d+) forecast row\(s\) are in `forecasts`.*$/, "$1 forecasts (model estimates) shown separately"],
    [/^(\d+) row\(s\) are in `database_intervals`.*$/, "$1 database intervals (curated lists); coverage unknown; absence unmeasured"],
    [/^(\d+) row\(s\) are in `text_mentions`.*$/, "$1 logbook word matches; not observations"],
    [/^(\d+) row\(s\) have no recorded time.*$/, "$1 rows excluded: time unknown, not outside window"],
    [/^\d+ source\(s\) ran over shot \d+ and recorded NO coverage -- (.*): ran; coverage unknown --.*$/, "$1: coverage unknown"],
    [/^no observed-event product for shot (\d+):.*$/, "Shot $1: no completed covering detector; unprocessed, not quiet"],
    [/^no source with recorded coverage ran over shot (\d+)(.*): the sources that completed.*$/, "Shot $1: coverage unknown$2; absence unmeasured"],
    [/^(.*?) source\(s\) ran over shot (\d+) and reported 0 detections inside their coverage(.*?)\. This IS.*$/, "Shot $2: $1 sources, 0 detections within coverage$3"],
    [/^(\d+) source\(s\) FAILED on shot (\d+):.*$/, "Shot $2: $1 sources failed; evidence missing"],
    [/coverage recorded as a hull by an older writer; interior gaps unknown/g, "Legacy coverage hull; interior gaps unknown"],
    [/: ran; coverage unknown; absence is not evidence$/, ": coverage unknown; absence unmeasured"],
    [/^(\d+) event\(s\) not shown: the source recorded no confidence, so they cannot be shown to reach min_confidence (.*)$/, (_m, n, limit) => `${n} events excluded: confidence unrecorded; minimum ${formatNumber(Number(limit), "confidence")}`],
    [/^(\d+) (events|forecasts) excluded: confidence below (\S+) or unrecorded$/, (_m, n, kind, limit) => `${n} ${kind} excluded: confidence below ${formatNumber(Number(limit), "confidence")} or unrecorded`],
    [/^kept despite --avoid (.*): nothing looked for (.*) on this shot,.*$/, "Kept with avoid $1: $2 unexamined; absence unmeasured"],
    [/^kept despite --avoid (.*): no detector for (.*) has run on this shot,.*$/, "Kept with avoid $1: $2 detectors not run; absence unmeasured"],
    [/^kept despite --avoid (.*): the (.*) detectors ran on this shot but not over the window searched,.*$/, "Kept with avoid $1: $2 coverage outside window; absence unmeasured"],
    [/^kept despite --avoid (.*): the (.*) detectors covered only part of the window searched,.*$/, "Kept with avoid $1: $2 coverage partial; outside unmeasured"],
    [/^--avoid (.*): dropped (\d+) shot\(s\) with observed (.*) evidence$/, "Avoid $1: excluded $2 shots with observed $3"],
    [/^--avoid (.*): excluded (\d+) (.*) segment\(s\) with (.*) coverage; absence is not evidence$/, "Avoid $1: excluded $2 $3 segments; $4 coverage, absence unmeasured"],
    [/^--avoid (.*): detectors covered ([\d.]+)% of the (.*) window; absence outside that coverage is unmeasured$/, (_m, token, percent, segment) => `Avoid ${token}: covered ${formatNumber(Number(percent))}% of ${segment}; outside unmeasured`],
    [/^the window (.*?) is outside every source's coverage of shot (\d+), whose display hull is (.*?) to (.*?) s; covered intervals: (.*?) --.*$/, (_m, window, shot, a, b, intervals) => `Shot ${shot}: ${window} outside coverage; hull ${timeSpan(Number(a), Number(b))}; covered intervals: ${intervals}; requested window unmeasured`],
    [/^the frame-code cache for shot (\d+) has no provenance sidecar:.*$/, "Shot $1: frame-code device and thread count unrecorded; codes vary with both"],
    [/^shot (\d+) has no (.*) segment; the description falls back to `full`$/, "Shot $1: no $2 segment scalars"],
    [/^no channel had anything to search on -- give ref_shot, text, constraints or actuators$/, "Enter a reference shot, text, constraints or actuators"],
    [/^excluded for having no recorded value: (.*)$/, "Missing values excluded: $1"],
  ];
  for (const [pattern, replacement] of rules) text = text.replace(pattern, replacement);
  return text.replace(/\[([\d.e+\-]+|None), ([\d.e+\-]+|None)\] s/g,
    (_m, a, b) => timeSpan(a === "None" ? null : Number(a), b === "None" ? null : Number(b)));
}

let disclosureId = 0;
// Recheck after layout and viewport changes, including text loaded asynchronously.
// No observers retain detached result rows when another search replaces them.
const disclosures = new Set();
let disclosureFramePending = false;
function scheduleDisclosures() {
  if (typeof requestAnimationFrame !== "function" || disclosureFramePending) return;
  disclosureFramePending = true;
  requestAnimationFrame(() => { disclosureFramePending = false; updateDisclosures(); });
}
function watchDisclosure(root, update) {
  if (typeof requestAnimationFrame !== "function") return;
  disclosures.add({ root, update });
  scheduleDisclosures();
}
function updateDisclosures() {
  for (const item of disclosures) {
    if (!item.root.isConnected) disclosures.delete(item);
    else item.update();
  }
}

function inlineText(text) {
  return text.split(/(\b[Ss]hot \d+\b|\b[12]\d{5}\b|[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?:–[+-]?\d+(?:\.\d+)? s)?)/g).map((part, i) =>
    i % 2 ? el("span", { class: /^(?:[Ss]hot |[12]\d{5}$)/.test(part) ? "shot-number" : "numeric" }, part) : part);
}

function longText(value, { limit = BLOCK_TEXT_LIMIT, cell = false } = {}) {
  const text = display(value);
  const cut = text.slice(0, limit + 1).search(/\s+\S*$/);
  const preview = text.length > limit ? text.slice(0, cut > 0 ? cut : limit).trimEnd() + "…" : text;
  const content = el("span", { class: "text-content", id: `text-${++disclosureId}` }, inlineText(preview));
  let expanded = false;
  const button = el("button", { type: "button", class: "text-toggle", "aria-expanded": "false",
    "aria-controls": content.id, onclick: (event) => {
      event.stopPropagation();
      expanded = !expanded;
      content.replaceChildren(...inlineText(expanded ? text : preview));
      root.classList.toggle("expanded", expanded);
      button.textContent = expanded ? "less" : "more";
      button.setAttribute("aria-expanded", String(expanded));
      updateDisclosures();
    } }, "more");
  button.hidden = text.length <= limit;
  const root = el("div", { class: `long-text${cell ? " cell-text" : ""}` }, content, button);
  watchDisclosure(root, () => {
    button.hidden = !expanded && text.length <= limit && content.scrollHeight <= content.clientHeight + 1;
  });
  return root;
}

function cellText(items, { limit = CELL_TEXT_LIMIT } = {}) {
  // One disclosure owns all items, including titles, intervals and late API notes.
  return longText(items.filter((item) => item !== null && item !== undefined && item !== "")
    .map((item) => display(item)).join("\n") || "—", { limit, cell: true });
}

function collapsible(content) {
  const body = el("div", { class: "collapse-body", id: `section-${++disclosureId}` }, content);
  let expanded = false;
  const button = el("button", { type: "button", class: "section-toggle", "aria-expanded": "false",
    "aria-controls": body.id, onclick: (event) => {
      event.stopPropagation();
      expanded = !expanded;
      root.classList.toggle("expanded", expanded);
      button.textContent = expanded ? "Show less" : "Show all";
      button.setAttribute("aria-expanded", String(expanded));
      updateDisclosures();
    } }, "Show all");
  const root = el("div", { class: "collapsible" }, body, button);
  watchDisclosure(root, () => {
    const overflow = body.scrollHeight > 260;
    button.hidden = !overflow;
    root.classList.toggle("overflowing", overflow);
    // Hidden content must not receive keyboard focus until expanded.
    for (const node of body.querySelectorAll("button, a, [tabindex]")) {
      const clipped = !expanded && node.getBoundingClientRect().bottom > body.getBoundingClientRect().bottom;
      if (clipped) node.setAttribute("tabindex", "-1");
      else node.removeAttribute("tabindex");
    }
  });
  return root;
}

function parseWire(text) {
  // json.dumps(default=str) may emit bare non-finite floats. Preserve quoted text,
  // replacing only these numeric tokens with null for the browser's missing-value display.
  return JSON.parse(text.replace(/"(?:[^"\\]|\\.)*"|-?Infinity|NaN/g,
    (token) => token.startsWith('"') ? token : "null"));
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "content-type": "application/json" }, ...options });
  const data = parseWire(await response.text());
  if (!response.ok) throw new Error(response.status === 401 ?
    "401: reopen the token link printed by shot_design serve." : display(data.error ?? data.detail));
  return { data, notes: JSON.parse(response.headers.get("X-Ideate-Caveats") || "[]") };
}

function caveats(items) {
  return items?.length ? el("div", { class: "caveats" },
    longText([...new Set(items)].map(caption).join("\n"))) : null;
}

function notes(target, data = {}, extra = []) {
  target.replaceChildren();
  if (data.error) target.append(el("div", { class: "error" }, longText(data.error)));
  const list = caveats([...(data.caveats || []), ...extra]);
  if (list) target.append(list);
}

function fields(record, units = {}) {
  return el("dl", {}, Object.entries(record || {}).flatMap(([k, v]) => [
    el("dt", {}, fieldName(k)), el("dd", { class: identifier(k) ? "identifier" : typeof v === "number" ? "numeric" : "" },
      typeof v === "number" ? `${display(v, k)}${units[k] && finite(v) ? ` ${units[k]}` : ""}` : longText(display(v, k)))]));
}

function showView(view) {
  for (const node of document.querySelectorAll(".view")) node.hidden = node.id !== `view-${view}`;
  for (const button of document.querySelectorAll("[data-view]")) {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  scheduleDisclosures();
}

function bindForm(id, target, action) {
  $(id).addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.target.querySelector('button[type="submit"]');
    const errorNote = event.target.querySelector('.form-error');
    errorNote.textContent = '';
    button.disabled = true;
    try { await action(new FormData(event.target)); }
    catch (error) { errorNote.textContent = error.message; notes($(target), { error: error.message }); }
    finally { button.disabled = false; }
  });
}

const tokens = (value) => String(value || "").split(",").map((v) => v.trim()).filter(Boolean);
function parseNumberField(value, label, { required = false, integer = false, identifier = false, min, max } = {}) {
  const raw = String(value ?? '').trim();
  if (!raw) {
    if (required) throw new Error(`${label} is required`);
    return null;
  }
  const number = Number(raw);
  if (identifier && (!/^[0-9]+$/.test(raw) || !Number.isSafeInteger(number))) {
    throw new Error(`${label} must be a whole number`);
  }
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw) || !Number.isFinite(number)) {
    throw new Error(`${label} must be a number`);
  }
  if (integer && !Number.isSafeInteger(number)) throw new Error(`${label} must be a whole number`);
  if ((min !== undefined && number < min) || (max !== undefined && number > max)) {
    throw new Error(min !== undefined && max !== undefined ? `${label} must be between ${min} and ${max}` :
      min !== undefined ? `${label} must be at least ${min}` : `${label} must be at most ${max}`);
  }
  return number;
}

function eventParams(form) {
  const params = new URLSearchParams();
  if (form.get('phenomenon')) params.set('phenomenon', form.get('phenomenon'));
  const t0 = parseNumberField(form.get('t0_s'), 'Start');
  const t1 = parseNumberField(form.get('t1_s'), 'End');
  if (t0 !== null && t1 !== null && t0 >= t1) throw new Error('End must be after start');
  const confidence = parseNumberField(form.get('min_confidence'), 'Minimum confidence', {min:0, max:1});
  for (const [key, value] of [['t0_s', t0], ['t1_s', t1], ['min_confidence', confidence]]) {
    if (value !== null) params.set(key, value);
  }
  return params;
}
function shotLink(shot, phenomenon = "", segment = "flat_top") {
  return `#shot/${shot}?${new URLSearchParams({ phenomenon, segment })}`;
}

function blurbText(row, { cell = false } = {}) {
  const text = row.blurb?.trim() ? row.blurb : null;
  if (!text) return el("div", { class: "blurb-text" }, "—");
  return el("div", { class: "blurb-text" }, cell ? cellText([text]) : longText(text),
    row.blurb_source === "template"
      ? el("span", { class: "blurb-auto small muted",
        title: "Deterministic header + outcome; no model summary yet" }, "auto")
      : row.blurb_source === "human"
        ? el("span", { class: "blurb-auto small muted",
          title: "Summary written by hand from the shot's own text; no model" }, "hand") : null);
}

function resultsTable(rows, segment) {
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, ["Shot", "Score", "Run / mini-proposal", "Summary", "Caveats"].map((t) => el("th", {}, t)))),
    el("tbody", {}, rows.map((row) => {
      const open = () => { location.hash = shotLink(row.shot, "", row.segment || segment); };
      const title = el("td", { class: "prose-cell" }, cellText([display(row.run_id, "run_id"), "Loading title…"]));
      const noteItems = [...(row.caveats || []).map(caption), ...(row.flags || []).map(formatFlag)];
      const rowNotes = el("td", { class: "prose-cell" }, cellText(noteItems));
      const refreshNotes = () => rowNotes.replaceChildren(cellText([...new Set(noteItems)]));
      // Titles are absent from search's ResultItem. Read the existing describe route;
      // its caveats/errors remain visible in this same row, and scores stay untouched.
      api(`/api/shot/${row.shot}?${new URLSearchParams({ segment: row.segment || segment })}`)
        .then(({ data }) => {
          const human = data.record?.human;
          title.replaceChildren(cellText([display(row.run_id, "run_id"), human?.run_title, human?.mp_title]));
          if (data.error) noteItems.push(data.error);
          noteItems.push(...(data.caveats || []).map(caption));
          refreshNotes();
        }).catch((error) => {
          title.replaceChildren(cellText([display(row.run_id, "run_id"), "—"]));
          noteItems.push(error.message);
          refreshNotes();
        });
      return el("tr", { class: "clickable", onclick: open },
        el("td", { class: "shot-number" }, el("a", { href: shotLink(row.shot, "", row.segment || segment) }, display(row.shot, "shot"))),
        el("td", { class: "numeric" }, display(row.score)),
        title,
        el("td", { class: "summary-cell prose-cell" }, blurbText(row, { cell: true })),
        rowNotes);
    }))));
}

// Only finite recorded times can place a mark. Axis interpolation is display geometry.
const finite = (n) => typeof n === "number" && Number.isFinite(n);
function colour(kind) { return COLOURS[kind] || "#77869b"; }
function timelineDomain(data) {
  // Events and Locate use the same server-derived full/other-segment domain.
  // Missing replies use the documented default; coverage never widens it.
  return finite(data.domain?.t0_s) && finite(data.domain?.t1_s) && data.domain.t1_s > data.domain.t0_s ?
    [data.domain.t0_s, data.domain.t1_s] : [-2, 8];
}

function timeAxis(domain) {
  const ticks = [];
  for (let time = Math.ceil(domain[0]); time <= domain[1]; time++) {
    ticks.push(el("span", { class: `axis-tick${time === 0 ? " zero" : ""}`,
      style: `left:${100 * (time - domain[0]) / (domain[1] - domain[0])}%` },
    el("span", {}, `${time} s`)));
  }
  return el("div", { class: "axis", "aria-label": "Time (seconds)" }, ticks);
}

function timeline(rows, domain = [-2, 8], coverage = false, phenomena = [], forecast = false) {
  const root = el("div", { class: "timeline" });
  // The top axis is outside the collapsible body, including for empty lanes.
  root.append(timeAxis(domain));
  if (!Array.isArray(rows)) { root.append(el("p", { class: "muted" }, "—")); return root; }
  const phenomenonGroups = phenomena.filter(p => !forecast || p.forecast_intervals?.length);
  if (!rows.length && !phenomenonGroups.length) { root.append(el("p", { class: "muted" }, "No indexed intervals")); return root; }
  const lanes = el("div", { class: "timeline-lanes" });
  const groups = new Map();
  for (const row of rows) {
    const source = display(row.source);
    if (!groups.has(source)) groups.set(source, []);
    groups.get(source).push(row);
  }
  const phenomenonLanes = el('div', {class:'phenomenon-lanes'}, el('h4', {}, 'Phenomena'));
  const sourceLanes = el('div', {class:'source-lanes'},
    phenomenonGroups.length && rows.length ? el('h4', {}, 'Sources') : null);
  const allRows = [...rows, ...phenomenonGroups.flatMap(p => forecast ? p.forecast_intervals || [] : p.intervals || [])];
  if (!coverage) lanes.append(el("div", { class: "legend" },
    [...new Set(allRows.map((r) => r.evidence_kind))].map((kind) => el("span", {},
      el("i", { class: "swatch", style: `--evidence:${colour(kind)}` }), display(kind)))));
  const laneGroups = [
    ...phenomenonGroups.map(p => ({source:p.title, phenomenon:p,
      items:(forecast ? p.forecast_intervals || [] : p.intervals || []).map(iv => ({
        ...iv, phenomenon:p.title, caveats:[...(p.caveats || []), ...(iv.caveats || [])],
      }))})),
    ...[...groups].map(([source, items]) => ({source, items})),
  ];
  for (const {source, items, phenomenon} of laneGroups) {
    const track = el("div", { class: "track", "aria-label": `${source}, seconds` });
    const details = [];
    for (const item of items) {
      const values = coverage ?
        { status: item.status, reason: item.reason, diag: item.diag, channel: item.channel, pass_name: item.pass_name, n_events: item.n_events, min_gap_s: item.min_gap_s } :
        { phenomenon: item.phenomenon, source: item.source, evidence_kind: item.evidence_kind, confidence: item.confidence };
      let tooltip = `${coverage ? "coverage " : ""}${timeSpan(item.t0_s, item.t1_s)}`;
      const drawable = finite(item.t0_s) && finite(item.t1_s) && item.t1_s >= item.t0_s;
      const clip = (time) => Math.max(domain[0], Math.min(domain[1], time));
      const clippedLeft = drawable && item.t0_s < domain[0];
      const clippedRight = drawable && item.t1_s > domain[1];
      if (clippedLeft || clippedRight) tooltip += `, drawn ${display(clip(item.t0_s))}–${display(clip(item.t1_s))} s`;
      tooltip += ` · ${display(values)}${item.caveats?.length ? ` · ${item.caveats.map(caption).join("; ")}` : ""}`;
      if (finite(item.f0_khz) || finite(item.f1_khz)) tooltip += ` · ${display(item.f0_khz)}–${display(item.f1_khz)} kHz`;
      if (drawable) {
        const span = domain[1] - domain[0];
        const left = 100 * (clip(item.t0_s) - domain[0]) / span;
        const width = 100 * (clip(item.t1_s) - clip(item.t0_s)) / span;
        track.append(el("span", {
          class: `mark${item.t0_s === item.t1_s ? " point" : ""}${clippedLeft ? " clipped-left" : ""}${clippedRight ? " clipped-right" : ""}${left === 100 ? " at-right" : ""}`,
          title: tooltip, "aria-label": tooltip,
          style: `left:${left}%;width:${width}%;--evidence:${coverage ? "#8995a5" : colour(item.evidence_kind)}`,
        }));
      } else if (!coverage) {
        details.push(longText(tooltip));
      }
      if (coverage) details.push(tooltip);
    }
    // Coverage rows with no intervals still show status/reason; no invented bars.
    (phenomenon ? phenomenonLanes : sourceLanes).append(el("div", { class: "lane" },
      el("div", { class: "lane-name" }, longText(source),
        phenomenon ? el('div', {class:'muted'}, phenomenon.id) : null), track,
      coverage ? el("div", { class: "coverage-detail" }, longText([...new Set(details)].join("; "))) : details));
  }
  if (phenomenonGroups.length) {
    lanes.append(phenomenonLanes);
  }
  lanes.append(sourceLanes);
  lanes.append(timeAxis(domain));
  root.append(collapsible(lanes));
  return root;
}

function renderEvents(data, prefix = "") {
  const target = (id) => $(`#${prefix}${id}`);
  target("event-status").textContent = display(data.status);
  // The upstream tool sometimes gives no explanatory caveat (e.g. observed detections).
  // Show — then; never manufacture a status explanation or a status absent from the reply.
  const patterns = {
    unprocessed: /^no observed-event product/,
    uncovered: /^(?:no source with recorded coverage|the window)/,
    observed: /source\(s\) ran over shot .*reported 0 detections/,
    unindexed: /not in the database/,
  };
  const meaning = (data.caveats || []).find((c) => patterns[data.status]?.test(c));
  target("event-meaning").replaceChildren(longText(caption(meaning)));
  notes(target("event-notes"), data);
  const coverage = (data.coverage?.sources || []).flatMap((s) => {
    const diagnostic = s.source !== "text" && s.source !== "database" && !s.source?.startsWith("database:");
    const legacy = !Array.isArray(s.intervals) || s.legacy_hull;
    const intervals = s.status === "ok" && diagnostic ?
      (Array.isArray(s.intervals) ? s.intervals :
        (finite(s.t_cov0_s) && finite(s.t_cov1_s) ? [[s.t_cov0_s, s.t_cov1_s]] : [])) : [];
    const row = { ...s, caveats: legacy && intervals.length ?
      ["coverage recorded as a hull by an older writer; interior gaps unknown"] : [] };
    // Keep skipped/unknown rows visible in the details, without drawing a bar.
    return intervals.length ? intervals.map(([t0_s, t1_s]) => ({ ...row, t0_s, t1_s })) :
      [{ ...row, t0_s: null, t1_s: null }];
  });
  const domain = timelineDomain(data);
  target("event-lanes").replaceChildren(timeline(data.events, domain, false, data.phenomena));
  target("forecast-lanes").replaceChildren(timeline(data.forecasts, domain, false, data.phenomena, true));
  target("database-lanes").replaceChildren(timeline(data.database_intervals, domain));
  target("coverage-lanes").replaceChildren(timeline(coverage, domain, true));
  target("text-mentions").replaceChildren(collapsible(el("ul", {},
    (data.text_mentions || []).map((mention) => el("li", {}, fields(mention))))));
}

async function loadEvents() {
  if (S.shot === null) return;
  const params = eventParams(new FormData($("#events-form")));
  const request = ++S.eventRequest;
  const shot = S.shot;
  const filtered = params.has("phenomenon");
  $("#event-context").hidden = !filtered;
  $("#event-filter-note").hidden = !filtered;
  renderEvents({});
  renderEvents({}, "context-");
  const contextParams = new URLSearchParams(params);
  contextParams.delete("phenomenon");
  await Promise.all([
    api(`/api/shot/${shot}/events?${params}`).then(({ data }) => {
      if (request === S.eventRequest) renderEvents(data);
    }),
    filtered ? api(`/api/shot/${shot}/events?${contextParams}`).then(({ data }) => {
      if (request === S.eventRequest) renderEvents(data, "context-");
    }) : Promise.resolve(),
  ]);
}

function scalarGrid(scalars) {
  return el("div", { class: "scalar-grid" }, scalars.map(({ name, value, units }) =>
    el("div", { class: "scalar" }, el("span", { class: "scalar-name" }, name),
      el("span", { class: "scalar-value" }, `${formatNumber(value)}${units && finite(value) ? ` ${units}` : ""}`))));
}

function attributedQuote(entry) {
  return el("figure", { class: "quote" }, el("blockquote", {}, longText(entry.text)),
    el("figcaption", { class: "muted small" }, [entry.role, entry.author, entry.time].filter(Boolean).join(" · ") || "—"));
}

function outcomeFields(outcome) {
  const target = (value) => value === null || value === undefined ? "—" : value ? "hit" : "missed";
  const percent = (value) => finite(value) ? `${formatNumber(value * 100)} %` : "—";
  return fields({
    "Ip target": target(outcome.ip_target_hit), "Ip error": percent(outcome.ip_target_err),
    "NBI target": target(outcome.nbi_target_hit), "NBI error": percent(outcome.nbi_target_err),
    "NBI target units": outcome.nbi_target_unit,
    "Flat top": finite(outcome.flat_top_ms) ? `${formatNumber(outcome.flat_top_ms / 1000, "time")} s` : null,
    "Ended early": outcome.ended_early, "Fast quench": outcome.fast_quench,
    "End reason": outcome.end_reason?.replaceAll("_", " "),
    "End time": finite(outcome.end_time_s) ? `${formatNumber(outcome.end_time_s, "time")} s` : null,
    "Faults": outcome.fault_strings,
  });
}

function phenomenaTable(rows) {
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, ["Phenomenon", "Observed", "Forecasts", "Coverage"].map((t) => el("th", {}, t)))),
    el("tbody", {}, rows.map((row) => el("tr", {},
      el("td", { class: "prose-cell" }, cellText([row.title, row.id])),
      el("td", { class: "numeric" }, display(row.n_observed, "n")),
      el("td", { class: "numeric" }, display(row.n_forecast, "n")),
      el("td", { class: "prose-cell" }, cellText([row.coverage_note,
        ...(row.coverage_windows || []).map(([a, b]) => timeSpan(a, b)),
        row.coverage_partial ? "Partial coverage; outside unmeasured" : null,
        ...(row.caveats || []).map(caption)])),
    )))));
}

function renderShot(data) {
  notes($("#shot-notes"), data);
  const root = $("#shot-record");
  root.replaceChildren();
  if (!data.record) return;
  const record = data.record;
  const parts = data.describe_parts;
  if (record.blurb?.trim()) root.append(el("div", { class: "summary-block" },
    el("h3", {}, "Summary"), blurbText(record)));
  if (parts) {
    root.append(longText(parts.header));
    if (parts.segment) root.append(el("h3", {}, display(parts.segment.name)),
      el("p", { class: "muted" }, timeSpan(parts.segment.t0_s, parts.segment.t1_s)));
    root.append(collapsible(scalarGrid(parts.scalars)), el("h3", {}, "Labels"), fields(parts.labels),
      el("h3", {}, "Outcome"), collapsible(outcomeFields(parts.outcome)));
    if (parts.operator_quote) root.append(el("h3", {}, "Operator quote"), attributedQuote(parts.operator_quote));
    root.append(el("h3", {}, "Phenomena"), collapsible(phenomenaTable(parts.phenomena)), caveats(parts.caveats));
  }
  root.append(el("h3", {}, "Run / mini-proposal"),
    collapsible(fields({ shot_date: record.shot_date, campaign: record.campaign,
      run_id: record.human?.run_id, run_title: record.human?.run_title, mpid: record.human?.mpid,
      mp_title: record.human?.mp_title })));
  root.append(el("h3", {}, "Flags and groups"), collapsible(fields(Object.fromEntries(
    Object.entries(record).filter(([key]) => key.startsWith("has_") || key === "raw_groups")))));
  root.append(el("h3", {}, "Scalars per segment"), collapsible(el("div", { class: "scalars" },
    (record.segments || []).map((seg) => el("div", { class: "card" },
      el("h3", {}, display(seg.name)), el("p", { class: "muted" },
        timeSpan(finite(seg.t0_ms) ? seg.t0_ms / 1000 : null, finite(seg.t1_ms) ? seg.t1_ms / 1000 : null)),
      collapsible(fields({ ...seg.raw, ...seg.derived }, data.units)))))));
  root.append(el("h3", {}, "Logbook"), collapsible(el("div", {},
    (record.human?.log_entries || []).map(attributedQuote))));
  root.append(el("h3", {}, "Frame codes"), collapsible(fields(data.frame_codes)),
    el("details", { ontoggle: updateDisclosures }, el("summary", {}, "Complete stored record"), collapsible(fields(record))));
}

async function openShot(shot, phenomenon = "", segment = "flat_top") {
  S.shot = shot;
  const request = ++S.shotRequest;
  $("#shot-form").elements.shot.value = shot;
  $("#shot-form").elements.segment.value = segment;
  $("#events-form").reset();
  $("#events-form").elements.phenomenon.value = phenomenon;
  $("#shot-title").textContent = `Shot ${shot}`;
  $("#shot-record").replaceChildren(el("p", {}, "Loading…"));
  notes($("#shot-notes"));
  showView("shot");
  await Promise.all([
    api(`/api/shot/${shot}?${new URLSearchParams({ segment })}`).then(({ data }) => {
      if (request === S.shotRequest) renderShot(data);
    }),
    loadEvents(),
  ]);
}

function renderHit(hit, segment) {
  const domain = timelineDomain(hit);
  const card = el("article", { class: "card" }, el("div", { class: "hit-head" },
    el("a", { class: "shot-number", href: shotLink(hit.shot, hit.phenomenon, segment) }, `Shot ${display(hit.shot, "shot")}`),
    el("span", { class: "numeric" }, `score ${display(hit.score)}`), el("span", { class: "identifier" }, `run ${display(hit.run_id, "run_id")}`)),
    longText(hit.mp_title), el("h3", {}, "Summary"), blurbText(hit), caveats(hit.caveats),
    fields({ total_duration_s: hit.total_duration_s, coverage_state: hit.coverage_state }),
    el("h3", {}, "Observed intervals"), timeline((hit.intervals || []).map((iv) => ({ ...iv, phenomenon: hit.phenomenon })), domain));
  if (hit.forecasts?.length) card.append(el("h3", {}, FORECAST_TITLE),
    timeline(hit.forecasts.map((iv) => ({ ...iv, phenomenon: hit.phenomenon })), domain));
  if (hit.text_snippets?.length) card.append(el("h3", {}, "Text mentions"),
    collapsible(el("div", {}, hit.text_snippets.map((text) => el("blockquote", {}, longText(text))))));
  if (Object.keys(hit.label_evidence || {}).length) card.append(el("h3", {}, "Label evidence"), collapsible(fields(hit.label_evidence)));
  return card;
}

function renderScoring(data) {
  notes($('#info-notes'), data);
  const root = $('#info-content');
  root.replaceChildren();
  if (data.error) return;
  const ph = data.phenomenon;
  const range = (data.score_range || []).map(n => formatNumber(n)).join('–');
  root.append(el('section', {}, el('h2', {}, 'Search score'),
    el('p', {}, `Each channel ranks every candidate; the score is a ${data.method.toLowerCase()}: ${data.formula}, k0 = ${display(data.k0)}. Scores are small${range ? ` (about ${range})` : ''} and only their order matters.`),
    el('div', {class:'table-wrap'}, el('table', {},
      el('thead', {}, el('tr', {}, ['Channel','Weight','What it compares'].map(t => el('th', {}, t)))),
      el('tbody', {}, data.channels.map(c => el('tr', {},
        el('td', {class:'identifier'}, c.name),
        el('td', {class:'numeric'}, display(c.weight), c.weight_source === 'default' ? el('span', {class:'small muted'}, ' (default)') : null),
        el('td', {class:'prose-cell'}, cellText([c.compares]))))))),
    el('p', {}, `Hard filters (${data.hard_filters.join(', ')}) apply before fusion.`),
    el('p', {}, `Duplicate shots above cosine ${display(data.dedup_threshold)} in logbook-text space collapse.`),
    el('p', {}, `The m-th hit from one run day is multiplied by ${display(data.run_diversity_decay)}^(m−1).`),
    el('p', {}, `With “prefer successful outcomes”, a failed verdict or missed Ip target is multiplied by ${display(data.outcome_penalty)}.`)),
  el('section', {}, el('h2', {}, 'Phenomenon score'),
    el('p', {class:'prose'}, ph.formula),
    fields({...ph.weights, saturation_n:ph.saturation_n}),
    ph.terms ? el('p', {}, ph.terms) : null,
    el('p', {}, `Evidence class comes first: ${ph.class_order.join(' > ')}. Score orders hits within each class.`),
    el('p', {}, `Text-only ceiling: ${display(ph.text_only_ceiling)}.`)),
  el('section', {}, el('h2', {}, 'Database'),
    fields({'Shots':display(data.db.n_shots, 'n'),
      'Shot range':data.db.shot_range?.map(s => display(s, 'shot')).join('–') || '—',
      'Build SHA':data.db.git_sha, 'Built':data.db.built})));
}

async function route() {
  const hash = location.hash.slice(1) || "search";
  if (hash.startsWith("shot/")) {
    const [path, query] = hash.split("?");
    const params = new URLSearchParams(query);
    const shot = Number(path.split("/")[1]);
    if (Number.isInteger(shot)) await openShot(shot, params.get("phenomenon") || "", params.get("segment") || "flat_top");
  } else if (hash === 'info') {
    showView('info');
    $('#info-content').replaceChildren(el('p', {}, 'Loading…'));
    const {data} = await api('/api/scoring');
    renderScoring(data);
  } else showView(["search", "shot", "locate"].includes(hash) ? hash : "search");
}

async function init() {
  for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => { location.hash = button.dataset.view; });
  const [{ data: meta }, { data: registry }] = await Promise.all([api("/api/meta"), api("/api/phenomena")]);
  notes($("#global-notes"), meta);
  if (meta.db) $("#dbinfo").textContent = `${display(meta.db.n_shots, "n")} shots · ${meta.db.shot_range?.map((s) => display(s, "shot")).join("–") || "—"}`;
  else $("#dbinfo").textContent = "Database unavailable";
  for (const select of document.querySelectorAll(".segment")) {
    select.replaceChildren(...(meta.segments || []).map((s) => el("option", { value: s }, s)));
    select.value = "flat_top";
  }
  if (!Array.isArray(registry)) throw new Error(display(registry.error));
  for (const select of document.querySelectorAll(".phenomenon")) {
    select.replaceChildren(...(select.dataset.all ? [el("option", { value: "" }, "All phenomena")] : []),
      ...registry.map((p) => el("option", { value: p.id }, `${p.title} (${p.id})${p.covering_sources.length ? "" : " — no detector"}`)));
  }
  bindForm("#search-form", "#search-notes", async (form) => {
    const constraints = form.get("constraints").trim();
    const body = { text: form.get("text"), ref_shot: parseNumberField(form.get("ref_shot"), 'Reference shot', {identifier:true}),
      segment: form.get("segment"), n: parseNumberField(form.get("n"), 'Results', {required:true, integer:true, min:1}),
      constraints: constraints ? JSON.parse(constraints) : null,
      require_labels: tokens(form.get("require_labels")), avoid_labels: tokens(form.get("avoid_labels")) };
    $("#search-results").replaceChildren(el("p", {}, "Searching…"));
    const { data } = await api("/api/search", { method: "POST", body: JSON.stringify(body) });
    notes($("#search-notes"), data);
    $("#search-results").replaceChildren(data.error ? el("p") : resultsTable(data.results || [], body.segment));
  });
  bindForm("#shot-form", "#shot-notes", async (form) => {
    const hash = shotLink(parseNumberField(form.get("shot"), 'Shot', {required:true, identifier:true}), $("#events-form").elements.phenomenon.value, form.get("segment"));
    if (location.hash === hash) await route(); else location.hash = hash;
  });
  bindForm("#events-form", "#event-notes", loadEvents);
  $("#events-form").addEventListener("change", () => {
    const errorNote = $('#events-form').querySelector('.form-error');
    errorNote.textContent = '';
    loadEvents().catch(e => { errorNote.textContent = e.message; notes($('#event-notes'), {error:e.message}); });
  });
  bindForm("#locate-form", "#locate-notes", async (form) => {
    const params = new URLSearchParams(form);
    params.set('n', parseNumberField(form.get('n'), 'Hits', {required:true, integer:true, min:1}));
    params.set('min_confidence', parseNumberField(form.get('min_confidence'), 'Minimum confidence', {required:true, min:0, max:1}));
    params.delete("avoid");
    for (const token of tokens(form.get("avoid"))) params.append("avoid", token);
    $("#locate-results").replaceChildren(el("p", {}, "Locating…"));
    const { data, notes: extra } = await api(`/api/locate?${params}`);
    notes($("#locate-notes"), data, extra);
    $("#locate-results").replaceChildren(...(Array.isArray(data) ? data.map((hit) => renderHit(hit, form.get("segment"))) : []));
  });
  window.addEventListener("hashchange", () => route().catch((e) => notes($("#global-notes"), { error: e.message })));
  window.addEventListener("resize", scheduleDisclosures);
  await route();
}

init().catch((error) => notes($("#global-notes"), { error: error.message }));
