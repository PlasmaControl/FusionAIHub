// Ported from shot-recommender-system's shotrec/ui/static/app.js.
// Keep shotrec's DOM helpers and API-driven presentation. No retrieval or physics here.
const $ = (selector) => document.querySelector(selector);
const S = { shot: null, shotRequest: 0, eventRequest: 0 };
const FORECAST_TITLE = "forecasts — a model's risk estimate, not an observation";
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

// A missing measurement never becomes a zero, including nested records and invalid floats.
function display(value) {
  if (value === null || value === undefined || value === "" ||
      (typeof value === "number" && !Number.isFinite(value)) ||
      (typeof value === "string" && /^(?:nan|[+-]?infinity)$/i.test(value))) return "—";
  if (typeof value === "object") {
    if (Array.isArray(value)) return value.length ? value.map(display).join(", ") : "—";
    return Object.entries(value).map(([k, v]) => `${k}: ${display(v)}`).join("; ") || "—";
  }
  return String(value);
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
    "401: reopen the token link printed by ideate serve." : display(data.error ?? data.detail));
  return { data, notes: JSON.parse(response.headers.get("X-Ideate-Caveats") || "[]") };
}

function caveats(items) {
  return items?.length ? el("ul", { class: "caveats" }, items.map((c) => el("li", {}, display(c)))) : null;
}

function notes(target, data = {}, extra = []) {
  target.replaceChildren();
  if (data.error) target.append(el("p", { class: "error" }, display(data.error)));
  const list = caveats([...(data.caveats || []), ...extra]);
  if (list) target.append(list);
}

function fields(record) {
  return el("dl", {}, Object.entries(record || {}).flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, display(v))]));
}

function showView(view) {
  for (const node of document.querySelectorAll(".view")) node.hidden = node.id !== `view-${view}`;
  for (const button of document.querySelectorAll("[data-view]")) {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
}

function bindForm(id, target, action) {
  $(id).addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.target.querySelector('button[type="submit"]');
    button.disabled = true;
    try { await action(new FormData(event.target)); }
    catch (error) { notes($(target), { error: error.message }); }
    finally { button.disabled = false; }
  });
}

const tokens = (value) => String(value || "").split(",").map((v) => v.trim()).filter(Boolean);
const optionalNumber = (value) => value === "" || value === null ? null : Number(value);
function shotLink(shot, phenomenon = "", segment = "flat_top") {
  return `#shot/${shot}?${new URLSearchParams({ phenomenon, segment })}`;
}

function resultsTable(rows, segment) {
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, ["Shot", "Score", "Run / mini-proposal", "Quote", "Caveats"].map((t) => el("th", {}, t)))),
    el("tbody", {}, rows.map((row) => {
      const open = () => { location.hash = shotLink(row.shot, "", row.segment || segment); };
      const title = el("td", {}, display(row.run_id), el("p", {}, "Loading title…"));
      const rowNotes = el("td", {}, caveats(row.caveats),
        ...(row.flags || []).map((flag) => el("p", {}, display(flag.message ?? flag))));
      // Titles are absent from search's ResultItem. Read the existing describe route;
      // its caveats/errors remain visible in this same row, and scores stay untouched.
      api(`/api/shot/${row.shot}?${new URLSearchParams({ segment: row.segment || segment })}`)
        .then(({ data }) => {
          const human = data.record?.human;
          title.replaceChildren(display(row.run_id), el("p", {}, display(human?.run_title)),
            el("p", {}, display(human?.mp_title)));
          if (data.error) rowNotes.append(el("p", { class: "error" }, display(data.error)));
          const extra = caveats(data.caveats);
          if (extra) rowNotes.append(extra);
        }).catch((error) => {
          title.replaceChildren(display(row.run_id), el("p", {}, "—"));
          rowNotes.append(el("p", { class: "error" }, error.message));
        });
      return el("tr", { class: "clickable", onclick: open },
        el("td", {}, el("a", { href: shotLink(row.shot, "", row.segment || segment) }, display(row.shot))),
        el("td", {}, display(row.score)),
        title,
        el("td", {}, el("blockquote", {}, display(row.explanation?.text_highlight))),
        rowNotes);
    }))));
}

// Only finite recorded times can place a mark. Axis interpolation is display geometry.
const finite = (n) => typeof n === "number" && Number.isFinite(n);
function bounds(rows) {
  const times = rows.flatMap((r) => [r.t0_s, r.t1_s]).filter(finite);
  return times.length ? [Math.min(...times), Math.max(...times)] : null;
}
function colour(kind) { return COLOURS[kind] || "#77869b"; }
function timeline(rows, domain = bounds(rows || []), coverage = false) {
  const root = el("div", { class: "timeline" });
  if (!Array.isArray(rows)) { root.append(el("p", { class: "muted" }, "—")); return root; }
  if (!rows.length) { root.append(el("p", { class: "muted" }, "No rows returned.")); return root; }
  const groups = new Map();
  for (const row of rows) {
    const source = display(row.source);
    if (!groups.has(source)) groups.set(source, []);
    groups.get(source).push(row);
  }
  if (!coverage) root.append(el("div", { class: "legend" },
    [...new Set(rows.map((r) => r.evidence_kind))].map((kind) => el("span", {},
      el("i", { class: "swatch", style: `--evidence:${colour(kind)}` }), display(kind)))));
  for (const [source, items] of groups) {
    const track = el("div", { class: "track", "aria-label": `${source}, seconds` });
    const detail = el("ul", { class: "lane-details" });
    for (const item of items) {
      if (domain && finite(item.t0_s) && finite(item.t1_s) && item.t1_s >= item.t0_s) {
        const span = domain[1] - domain[0];
        const left = span ? 100 * (item.t0_s - domain[0]) / span : 50;
        const width = span ? 100 * (item.t1_s - item.t0_s) / span : 0;
        track.append(el("span", {
          class: `mark${item.t0_s === item.t1_s ? " point" : ""}`,
          style: `left:${left}%;width:${width}%;--evidence:${coverage ? "#8995a5" : colour(item.evidence_kind)}`,
        }));
      }
      const values = coverage ?
        { status: item.status, reason: item.reason, diag: item.diag, channel: item.channel, pass_name: item.pass_name, n_events: item.n_events, min_gap_s: item.min_gap_s } :
        { phenomenon: item.phenomenon, evidence_kind: item.evidence_kind, confidence: item.confidence };
      detail.append(el("li", {}, `${display(item.t0_s)} – ${display(item.t1_s)} s · ${display(values)}`, caveats(item.caveats)));
    }
    root.append(el("div", { class: "lane" }, el("div", { class: "lane-name" }, source), track, detail));
  }
  if (domain) {
    const labels = domain[0] === domain[1] ? [domain[0]] : [domain[0], (domain[0] + domain[1]) / 2, domain[1]];
    root.append(el("div", { class: `axis${labels.length === 1 ? " single" : ""}` },
      labels.map((time) => el("span", {}, `${Number(time.toPrecision(6))} s`))));
  } else root.append(el("p", { class: "muted" }, "Recorded time: —"));
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
  target("event-meaning").textContent = display(meaning);
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
  const domain = bounds([...(data.events || []), ...(data.forecasts || []), ...(data.database_intervals || []), ...coverage]);
  target("event-lanes").replaceChildren(timeline(data.events, domain));
  target("forecast-lanes").replaceChildren(timeline(data.forecasts, domain));
  target("database-lanes").replaceChildren(timeline(data.database_intervals, domain));
  target("coverage-lanes").replaceChildren(timeline(coverage, domain, true));
  target("text-mentions").replaceChildren(el("ul", {},
    (data.text_mentions || []).map((mention) => el("li", {}, fields(mention)))));
}

async function loadEvents() {
  if (S.shot === null) return;
  const request = ++S.eventRequest;
  const params = new URLSearchParams();
  for (const [key, value] of new FormData($("#events-form"))) if (value !== "") params.set(key, value);
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

function renderShot(data) {
  notes($("#shot-notes"), data);
  const root = $("#shot-record");
  root.replaceChildren();
  if (!data.record) return;
  const record = data.record;
  root.append(el("p", { class: "prose" }, display(data.description)),
    fields({ shot_date: record.shot_date, campaign: record.campaign,
      run_id: record.human?.run_id, run_title: record.human?.run_title, mp_title: record.human?.mp_title }));
  root.append(el("h3", {}, "Stored flags and groups"), fields(Object.fromEntries(
    Object.entries(record).filter(([key]) => key.startsWith("has_") || key === "raw_groups"))));
  root.append(el("h3", {}, "Scalars per segment"), el("div", { class: "scalars" },
    (record.segments || []).map((seg) => el("div", { class: "card" },
      el("h3", {}, display(seg.name)), el("p", { class: "muted" }, `${display(seg.t0_ms)} – ${display(seg.t1_ms)} ms`),
      fields({ ...seg.raw, ...seg.derived })))));
  root.append(el("h3", {}, "Logbook quotes"), ...(record.human?.log_entries || []).map((entry) =>
    el("div", {}, el("p", { class: "small muted" }, `${display(entry.role)} · ${display(entry.author)} · ${display(entry.time)}`),
      el("blockquote", {}, display(entry.text)))));
  root.append(el("h3", {}, "Frame codes"), fields(data.frame_codes),
    el("details", {}, el("summary", {}, "Complete stored record"), fields(record)));
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
  const card = el("article", { class: "card" }, el("div", { class: "hit-head" },
    el("a", { href: shotLink(hit.shot, hit.phenomenon, segment) }, `Shot ${display(hit.shot)}`),
    el("span", {}, `score ${display(hit.score)}`), el("span", {}, `run ${display(hit.run_id)}`)),
    el("p", {}, display(hit.mp_title)), caveats(hit.caveats),
    fields({ total_duration_s: hit.total_duration_s, coverage_state: hit.coverage_state }),
    el("blockquote", {}, display(hit.quote)), el("p", { class: "muted small" }, display(hit.quote_role)),
    el("h3", {}, "Observed intervals"), timeline(hit.intervals || []));
  if (hit.forecasts?.length) card.append(el("h3", {}, FORECAST_TITLE), timeline(hit.forecasts));
  if (hit.text_snippets?.length) card.append(el("h3", {}, "Text mentions"),
    ...hit.text_snippets.map((text) => el("blockquote", {}, display(text))));
  if (Object.keys(hit.label_evidence || {}).length) card.append(el("h3", {}, "Label evidence"), fields(hit.label_evidence));
  return card;
}

async function route() {
  const hash = location.hash.slice(1) || "search";
  if (hash.startsWith("shot/")) {
    const [path, query] = hash.split("?");
    const params = new URLSearchParams(query);
    const shot = Number(path.split("/")[1]);
    if (Number.isInteger(shot)) await openShot(shot, params.get("phenomenon") || "", params.get("segment") || "flat_top");
  } else showView(["search", "shot", "locate"].includes(hash) ? hash : "search");
}

async function init() {
  for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => { location.hash = button.dataset.view; });
  const [{ data: meta }, { data: registry }] = await Promise.all([api("/api/meta"), api("/api/phenomena")]);
  notes($("#global-notes"), meta);
  if (meta.db) $("#dbinfo").textContent = `${display(meta.db.n_shots)} shots · ${display(meta.db.shot_range)} · build ${display(meta.db.git_sha)}`;
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
    const body = { text: form.get("text"), ref_shot: optionalNumber(form.get("ref_shot")),
      segment: form.get("segment"), n: Number(form.get("n")),
      constraints: constraints ? JSON.parse(constraints) : null,
      require_labels: tokens(form.get("require_labels")), avoid_labels: tokens(form.get("avoid_labels")) };
    $("#search-results").replaceChildren(el("p", {}, "Searching…"));
    const { data } = await api("/api/search", { method: "POST", body: JSON.stringify(body) });
    notes($("#search-notes"), data);
    $("#search-results").replaceChildren(data.error ? el("p") : resultsTable(data.results || [], body.segment));
  });
  bindForm("#shot-form", "#shot-notes", async (form) => {
    const hash = shotLink(Number(form.get("shot")), $("#events-form").elements.phenomenon.value, form.get("segment"));
    if (location.hash === hash) await route(); else location.hash = hash;
  });
  bindForm("#events-form", "#event-notes", loadEvents);
  $("#events-form").addEventListener("change", () => loadEvents().catch((e) => notes($("#event-notes"), { error: e.message })));
  bindForm("#locate-form", "#locate-notes", async (form) => {
    const params = new URLSearchParams(form);
    params.delete("avoid");
    for (const token of tokens(form.get("avoid"))) params.append("avoid", token);
    $("#locate-results").replaceChildren(el("p", {}, "Locating…"));
    const { data, notes: extra } = await api(`/api/locate?${params}`);
    notes($("#locate-notes"), data, extra);
    $("#locate-results").replaceChildren(...(Array.isArray(data) ? data.map((hit) => renderHit(hit, form.get("segment"))) : []));
  });
  window.addEventListener("hashchange", () => route().catch((e) => notes($("#global-notes"), { error: e.message })));
  await route();
}

init().catch((error) => notes($("#global-notes"), { error: error.message }));
