// The bar's activity line (plan D.4): what the page is waiting for and for how long, the notes
// that outlast a request, the last failures, and the copy for row chips. A classic script loaded
// before app.js: it only defines names, and app.js calls them.
"use strict";

/** Requests and builds in flight, the notes beside them and the last failures. `api()` starts
 * an item per request and ends it; an item shows after `SHOW_MS` and its elapsed time after
 * `CLOCK_MS`; a quiet one (a prefetch, a poll) is tracked but never shown. Nothing shows until
 * boot turns `enabled` on (api 11). */
const Activity = {
  SHOW_MS: 300,
  CLOCK_MS: 2000,
  TICK_MS: 250,
  STUCK_S: 120, // a build whose stage has not moved for this long may be stuck
  KEEP: 10, // failures kept for the detail list
  ABSENT: new Set(["masks", "tokeye", "version", "shots/search"]), // kinds whose 404 means "nothing here", not a failure
  STAGES: { queued: "queued", reading: "reading from disk", fetching: "fetching signals",
    building: "building rows", writing: "writing the store" }, // a /api/shot 202 `stage`, as the bar says it
  PLURALS: { signal: (n) => `Fetching ${n} signals`, profile: (n) => `Loading ${n} profiles`,
    frame: (n) => `Loading ${n} camera frames`, slice: (n) => `Loading ${n} slices` },
  enabled: false,
  items: new Map(), // id -> {id, kind, what, started, quiet, text?, fraction?, stop?, shot?}
  notes: new Map(), // key -> {text, tone, at}: the fdp login, a list fallback, a failed prefetch...
  errors: [], // the last failures, newest first: {what, message, at}
  serial: 0,
  timer: 0,
  detail: false, // the detail list is open
  parts: null, // the elements inside #activity, made on first use
  now: () => Date.now(),

  /** What a request to `path` is doing, as the bar says it. */
  describe(path, method = "GET") {
    const [route, query = ""] = String(path).split("?");
    const params = new URLSearchParams(query);
    const post = String(method).toUpperCase() === "POST";
    switch (route.replace(/^\/api\//, "")) {
      case "rows": return "Loading rows";
      case "queue": return "Loading the queue";
      case "shot": return `Opening ${params.get("shot") || "the shot"}`;
      case "overview": return params.has("shot") ? "Loading the shot overview" : "Loading the list";
      case "shotlist": return "Loading the list";
      case "shotlists": return "Loading the shot lists";
      case "silver": return "Loading the silver set";
      case "silver/shot": return "Loading the shot overview";
      case "gold/shots": return "Loading the gold set";
      case "events": return "Loading the events";
      case "version": return "Reading the server version";
      case "label": return "Saving";
      case "masks": return post ? "Saving the mask decisions" : "Loading the AE mask";
      case "tokeye": return "Loading TokEye";
      case "overlays": return "Reading TokEye states";
      case "overlays/request": return "Asking for TokEye";
      case "history": return "Loading saved versions";
      case "names": return post ? "Adding the name" : "Loading names";
      case "frame": return "Loading a camera frame";
      case "signal": return "Fetching a signal";
      case "profile": return "Loading a profile";
      case "catalog": return "Loading the signal list";
      case "views": return post ? "Saving the view" : "Loading views";
      case "fetch/status": return "Checking the fdp login";
      case "slices": return "Reading what the shot has";
      case "slice": return "Loading a slice";
      case "shots/search": return "Searching shots";
      default: return "Waiting for the server";
    }
  },

  /** Elapsed seconds as the bar shows them: "0.8 s", "14 s", "2 min", "2 min 5 s". */
  clock(seconds) {
    if (!(seconds >= 0)) return "";
    if (seconds < 1) return `${(Math.floor(seconds * 10) / 10).toFixed(1)} s`;
    if (seconds < 60) return `${Math.floor(seconds)} s`;
    const [minutes, rest] = [Math.floor(seconds / 60), Math.floor(seconds % 60)];
    return rest ? `${minutes} min ${rest} s` : `${minutes} min`;
  },

  /** 1st, 2nd, 3rd, 4th … 11th … 21st. */
  ordinal(n) {
    const tens = n % 100, ones = n % 10;
    const suffix = tens >= 11 && tens <= 13 ? "th" : ["th", "st", "nd", "rd"][ones] || "th";
    return `${n}${suffix}`;
  },

  /** A server message as a sentence: capitalised, with a full stop. */
  sentence(text) {
    const words = String(text).trim();
    return `${words.charAt(0).toUpperCase()}${words.slice(1)}${/[.!?]$/.test(words) ? "" : "."}`;
  },

  /** The bar's line for `items` at `now`: a build's own text; else one phrase, a count of one
   * kind ("Fetching 2 signals") or how many are loading and the longest, with the longest's
   * elapsed time past `CLOCK_MS`; "" while nothing has been waited on for `SHOW_MS`. */
  line(items, now) {
    const shown = items.filter((item) => !item.quiet && (item.text || now - item.started >= Activity.SHOW_MS));
    const own = shown.find((item) => item.text);
    if (own) return own.text;
    if (!shown.length) return "";
    const longest = shown.reduce((a, b) => (b.started < a.started ? b : a));
    const age = now - longest.started;
    const since = age >= Activity.CLOCK_MS ? ` · ${Activity.clock(age / 1000)}` : "";
    const kinds = new Set(shown.map((item) => item.kind));
    const phrases = new Set(shown.map((item) => item.what));
    if (shown.length > 1 && kinds.size === 1 && Activity.PLURALS[longest.kind]) {
      return `${Activity.PLURALS[longest.kind](shown.length)}${since}`;
    }
    if (phrases.size === 1) return `${longest.what}${since}`;
    return `${shown.length} loading · longest: ${longest.what}${since}`;
  },

  /** A store build's text from a /api/shot 202 `body` (stage, note, progress, elapsed_s,
   * queue_position, prebuilt). The server's short `note` ("Fetching pinj_total") says what it
   * does when it has one. `stalled` is how long, in seconds, its stage has not moved. */
  buildText(shot, name, body = {}, stalled = 0) {
    if (stalled >= Activity.STUCK_S) return ERROR_COPY.stuck();
    const p = body.progress;
    const note = typeof body.note === "string" ? body.note.trim().replace(/\.$/, "") : "";
    // "Fetching pinj_total" reads "fetching pinj_total" after the colon; "EFIT01 …" stays.
    const said = note && /^[A-Z][a-z]/.test(note) ? `${note.charAt(0).toLowerCase()}${note.slice(1)}` : note;
    const doing = p?.stage || said || Activity.STAGES[body.stage] || "building";
    const count = p?.total > 0 ? ` (${p.done} of ${p.total})` : "";
    const since = body.elapsed_s >= Activity.CLOCK_MS / 1000 ? ` · ${Activity.clock(body.elapsed_s)}` : "";
    // queue_position is the place among the waiting builds (0 while it runs), as a chip says it
    const place = body.queue_position > 0 && !/queue|ahead|in line/i.test(note) ? ` · ${Activity.ordinal(body.queue_position)} in line` : "";
    return `Building ${shot}${name ? ` for ${name}` : ""}: ${doing}${count}${since}${place}`;
  },

  /** What the rows say after Stop waiting on a build of `shot`. */
  stopped: (shot) => `${shot} is still building on the server. Open it again later.`,

  /** The fdp login's note from /api/version's `fdp` block, {text, tone}, or null: nothing to
   * say, or a server without the block. */
  fdpNote(fdp, now = Activity.now()) {
    if (!fdp) return null;
    if (fdp.env === false) return { text: ERROR_COPY.fdp_unavailable(), tone: "" };
    const at = Date.parse(fdp.expires_at ?? "");
    if (!Number.isFinite(at)) return null;
    const left = (at - now) / 1000;
    if (left <= 0) return { text: ERROR_COPY.fdp_expired(), tone: "error" };
    if (left > 24 * 3600) return null;
    const hours = Math.floor(left / 3600);
    return { text: `fdp login expires in ${hours ? `${hours} h` : `${Math.max(1, Math.floor(left / 60))} min`}`, tone: "" };
  },

  /** #stale when /api/version failed: null for a 404, which only means an older server. */
  versionNote: (error) => (error?.status === 404 ? null : ERROR_COPY.version({ message: error?.message || String(error) })),

  /** #stale for a server older than this page, or null for api 11. */
  staleText(api) {
    return api < 2 ? "Restart the server for names and history"
      : api < 5 ? "Restart the server for the list of names"
      : api < 7 ? "Restart the server for individual/span annotations"
      : api < 11 ? "Restart the server for shot lists, TokEye status and views"
      : null;
  },

  /** Register a request to `path`; returns its id for `end`. */
  start(path, { quiet = false, method = "GET" } = {}) {
    const id = ++Activity.serial;
    const kind = String(path).split("?")[0].replace(/^\/api\//, "");
    Activity.items.set(id, { id, kind, what: Activity.describe(path, method), started: Activity.now(), quiet: Boolean(quiet) });
    if (!Activity.timer && Activity.enabled && typeof document !== "undefined") {
      Activity.timer = setTimeout(renderActivity, Activity.TICK_MS);
    }
    return id;
  },

  /** The item `id` is over: `error` when it failed. A 404 of a kind that answers "nothing here"
   * (no mask, no TokEye layer, an older server's version) is not kept as a failure. */
  end(id, error = null) {
    const item = Activity.items.get(id);
    if (!item) return;
    Activity.items.delete(id);
    if (error && !(error.status === 404 && Activity.ABSENT.has(item.kind))) {
      Activity.fail(item.what, error.message || String(error));
    }
    renderActivity();
  },

  /** Keep a failure for the detail list. */
  fail(what, message) {
    Activity.errors.unshift({ what, message, at: Activity.now() });
    Activity.errors.length = Math.min(Activity.errors.length, Activity.KEEP);
  },

  /** Say `text` in the bar until it is cleared (a null `text` clears it); tone "error" draws it
   * in --error and keeps it among the failures. Notes show while nothing is loading. */
  note(key, text, tone = "") {
    if (text) {
      Activity.notes.set(key, { text, tone, at: Activity.now() });
      if (tone === "error") Activity.fail("", text);
    } else Activity.notes.delete(key);
    renderActivity();
  },

  clear(...keys) {
    for (const key of keys) Activity.notes.delete(key);
    renderActivity();
  },

  /** Start (no `id`) or update a long wait the reviewer may stop: `fields` holds its `text`, its
   * `fraction` done (or null) and `stop`, which Stop waiting calls. Returns its id. */
  hold(id, fields) {
    let item = Activity.items.get(id);
    if (!item) {
      id = ++Activity.serial;
      item = { id, kind: "wait", what: "", started: Activity.now(), quiet: false };
      Activity.items.set(id, item);
    }
    Object.assign(item, fields);
    renderActivity();
    return id;
  },

  /** Start or update the wait for a store build of `shot` from a /api/shot 202 `body`; `name`
   * is the event as the page shows it. Stop waiting ends openShot's poll (the build carries on
   * on the server) and leaves the shot with `stopped`'s note, unless `stop` says otherwise. */
  build(id, { shot, name, body, stop }) {
    const now = Activity.now();
    const item = Activity.items.get(id);
    const p = body?.progress;
    const key = JSON.stringify([body?.stage, body?.note, p?.stage, p?.done, body?.queue_position]);
    const moved = !item || item.key !== key ? now : item.moved;
    return Activity.hold(id, {
      kind: "build", what: `Building ${shot}`, shot, key, moved,
      text: Activity.buildText(shot, name, body || {}, (now - moved) / 1000),
      fraction: p?.total > 0 && p.done >= 0 ? Math.min(1, p.done / p.total) : null,
      stop: stop || item?.stop || (() => {
        S.ticket++; // openShot's loop sees a newer ticket and stops polling
        showNothing(shot, Activity.stopped(shot));
      }),
    });
  },

  /** The bar's note, {text, tone}, for a /api/fetch/status body, or null when fdp is "ok". The
   * server's `reason` (its 503 text: fetching off, the token expired, no fdp wrapper) wins over
   * the page's own copy; an expired token is drawn as an error. */
  fetchNote(status) {
    if (!status || status.fdp === "ok") return null;
    const expired = status.fdp === "expired";
    const text = status.reason ? Activity.sentence(status.reason)
      : expired ? ERROR_COPY.fdp_expired() : ERROR_COPY.fdp_unavailable();
    return { text, tone: expired ? "error" : "" };
  },

  /** Retry on an auth chip: true when GET /api/fetch/status says the fdp login works; otherwise
   * the bar says why. */
  async fetchReady() {
    try {
      const status = await (await api("/api/fetch/status")).json();
      const note = Activity.fetchNote(status);
      if (!note) {
        Activity.clear("fdp");
        return true;
      }
      Activity.note("fdp", note.text, note.tone);
    } catch (error) {
      Activity.note("fdp", "Could not check the fdp login. Try again in a minute.", "error");
    }
    return false;
  },

  /** What the bar shows now: {text, tone, busy, stop, fraction}, or null when idle. */
  summary(now) {
    const items = [...Activity.items.values()];
    const text = Activity.line(items, now);
    if (text) {
      const own = items.find((item) => !item.quiet && item.text);
      return { text, tone: "", busy: true, stop: Boolean(own?.stop), fraction: own?.fraction ?? null };
    }
    const notes = [...Activity.notes.values()].sort((a, b) => b.at - a.at);
    const note = notes.find((found) => found.tone === "error") || notes[0];
    return note ? { text: note.text, tone: note.tone, busy: false, stop: false, fraction: null } : null;
  },

  /** The room #activity needs on the bar's line before it shows there: its text and Stop
   * waiting, or this many px of them; with less it shows as a strip under the bar. */
  ROOM_PX: 260,

  /** The elements inside #activity, made once: the line (a button that opens the detail list)
   * and Stop waiting in one body, and the detail list after #activity. */
  wire() {
    if (Activity.parts) return Activity.parts;
    const box = document.getElementById("activity");
    if (!box) return null;
    const text = Object.assign(document.createElement("button"), { type: "button", className: "activity-text" });
    text.setAttribute("aria-expanded", "false");
    text.setAttribute("aria-controls", "activity-detail");
    text.title = "What is loading, and the last failures";
    text.addEventListener("click", () => Activity.toggle());
    const stop = Object.assign(document.createElement("button"), {
      type: "button", className: "activity-stop", textContent: "Stop waiting", hidden: true,
    });
    stop.addEventListener("click", () => stopWaiting());
    const detail = Object.assign(document.createElement("div"), { id: "activity-detail", hidden: true });
    detail.setAttribute("aria-label", "Activity");
    const body = Object.assign(document.createElement("span"), { className: "activity-body" });
    body.append(text, stop);
    box.replaceChildren(body);
    box.after(detail);
    Activity.parts = { box, body, text, stop, detail };
    return Activity.parts;
  },

  /** Where #activity shows: on the bar's line when the room left there holds its text and Stop
   * waiting (or `ROOM_PX` of them), else as a strip just under the bar, at the slot's left and
   * kept inside the bar's width, left of an open #dock (its head stays uncovered). It never
   * makes the bar taller and nothing covers Stop.
   * Every read comes before the writes, and a write that changes nothing is not made (R2_CHIPS):
   * a read after a write lays the page out again, which took seconds a call with 28,000 chips
   * drawn. Only the switch to the strip reads once more, for its text at the strip's size. */
  place() {
    const { box, body, text, stop, detail } = Activity.parts;
    if (box.hidden) {
      box.classList.remove("under");
      return;
    }
    const gap = 6, bar = box.parentElement, dock = document.getElementById("dock");
    const was = box.classList.contains("under");
    const extra = stop.hidden ? 0 : stop.offsetWidth + gap;
    let wide = text.scrollWidth, below = was ? body.offsetTop + body.offsetHeight : 0;
    const under = box.clientWidth - 14 < Math.min(Activity.ROOM_PX, wide + extra); // 14: the text's margin, the bar's gap taken back
    const slot = box.offsetLeft;
    const end = dock && !dock.hidden
      ? Math.min(bar.clientWidth, dock.getBoundingClientRect().left - bar.getBoundingClientRect().left) : bar.clientWidth;
    const set = (element, values) => {
      for (const [name, value] of Object.entries(values)) if (element.style[name] !== value) element.style[name] = value;
    };
    box.classList.toggle("under", under);
    let left = slot;
    if (under) {
      if (!was) [wide, below] = [text.scrollWidth, body.offsetTop + body.offsetHeight];
      const style = getComputedStyle(body), px = (name) => parseFloat(style[name]) || 0;
      const most = Math.max(120, end - 24);
      const own = Math.min(most, wide + extra + px("paddingLeft") + px("paddingRight") + px("borderLeftWidth") + px("borderRightWidth"));
      left = Math.round(Math.max(12, Math.min(slot + 14, end - own - 12)));
      set(body, { left: `${left}px`, maxWidth: `${Math.round(most)}px` });
    } else set(body, { left: "", maxWidth: "" });
    set(detail, { top: under ? `${Math.round(below)}px` : "", ...(Activity.detail ? { left: `${left}px` } : {}) });
  },

  /** Open or close the detail list; a press outside it closes it. */
  toggle(open = !Activity.detail) {
    const parts = Activity.wire();
    if (!parts) return;
    Activity.detail = open;
    parts.detail.hidden = !open;
    parts.text.setAttribute("aria-expanded", String(open));
    if (open) {
      document.addEventListener("pointerdown", Activity.outside, true);
    } else document.removeEventListener("pointerdown", Activity.outside, true);
    renderActivity();
  },

  outside(event) {
    if (!event.target.closest?.("#activity, #activity-detail")) Activity.toggle(false);
  },

  /** The detail list: every item in flight with its elapsed time, the notes, the last failures. */
  renderDetail(now) {
    const list = (title, lines) => {
      const heading = Object.assign(document.createElement("p"), { className: "activity-heading", textContent: title });
      const items = document.createElement("ul");
      items.append(...lines.map((line) => Object.assign(document.createElement("li"), { textContent: line })));
      return [heading, items];
    };
    const time = (at) => new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const loading = [...Activity.items.values()].map((item) => item.text ||
      `${item.what} · ${Activity.clock((now - item.started) / 1000)}${item.quiet ? " (in the background)" : ""}`);
    const notes = [...Activity.notes.values()].map((note) => note.text);
    const errors = Activity.errors.map((error) => `${time(error.at)} ${error.what ? `${error.what}: ` : ""}${error.message}`);
    Activity.parts.detail.replaceChildren(
      ...list("Loading", loading.length ? loading : ["Nothing is loading."]),
      ...(notes.length ? list("Notes", notes) : []),
      ...(errors.length ? list(`Last ${errors.length === 1 ? "failure" : `${errors.length} failures`}`, errors) : []),
    );
  },
};

/** What the page says when something fails, by kind: the kinds of /api/signal and
 * /api/profile errors (an added row's chip) and the page's own (the bar and #stale). Each
 * takes the facts it names. */
const ERROR_COPY = {
  auth: () => "fdp login expired. Run fdp login on the server, then Retry.",
  missing: ({ name, shot }) => `No ${name} for ${shot}.`,
  upstream: ({ message }) => `The data server failed: ${String(message).replace(/\.$/, "")}. Try again soon.`,
  timeout: ({ limit_s = 30 } = {}) => `Stopped after ${Activity.clock(limit_s)}. ${limit_s < 120
    ? "A formula node can take 2 min." : "Try again soon."}`,
  invalid: ({ name, source = "ptdata", message } = {}) => (message ? Activity.sentence(message)
    : source === "mds" ? `${name} is not a valid MDS node: letters, digits and _, each part starting with a letter.`
      : `${name} is not a valid PTDATA name: a letter, then letters, digits or _, up to 12.`),
  version: ({ message }) => `Could not read the server version (${message}). Reload the page.`,
  mask: () => "Could not load the AE mask. Press M to retry.",
  prefetch: ({ shot, message }) => `Next shot ${shot} failed to build: ${message}`,
  fdp_expired: () => "fdp login expired: no new signals. Run fdp login on the server, then Retry.",
  fdp_unavailable: () => "This server cannot fetch new signals. Restart it with pixi run -e labeler labeler-verify.",
  stuck: () => `Still building after ${Activity.clock(Activity.STUCK_S)}. The server may be stuck.`,
};

/** An added row's chip (D2): {text, button, tone, collapse?} for `kind` ("queued", "loading" or
 * an error kind of ERROR_COPY) from the server's facts: name, shot, source ("ptdata", "mds",
 * "disk"), tree, elapsed_s, queue_position (0 = running now), message. Null for an unknown
 * kind without a message. */
function rowNote(kind, facts = {}) {
  const since = facts.elapsed_s >= Activity.CLOCK_MS / 1000 ? ` · ${Activity.clock(facts.elapsed_s)}` : "";
  switch (kind) {
    case "queued": {
      const place = facts.queue_position > 0 ? `, ${Activity.ordinal(facts.queue_position)} in line` : "";
      return { text: `Queued${place}${since}`, button: "Stop waiting", tone: "queued" };
    }
    case "loading": {
      const from = facts.source === "disk" ? "Reading from disk"
        : facts.source === "mds" ? `Fetching from MDS${facts.tree ? ` (${facts.tree})` : ""}`
        : "Fetching from PTDATA";
      return { text: `${from}${since}`, button: "Stop waiting", tone: "queued" };
    }
    case "missing":
      return { text: ERROR_COPY.missing(facts), button: "Remove from view", tone: "", collapse: true };
    case "invalid":
      return { text: ERROR_COPY.invalid(facts), button: "Edit", tone: "error" };
    case "auth":
    case "upstream":
    case "timeout":
      return { text: ERROR_COPY[kind](facts), button: "Retry", tone: "error" };
    default:
      return facts.message ? { text: Activity.sentence(facts.message), button: "Retry", tone: "error" } : null;
  }
}

/** Draw #activity, its Stop waiting button, #activity-line and, when open, the detail list;
 * while anything is in flight it draws again every `TICK_MS`. */
function renderActivity() {
  if (typeof document === "undefined") return;
  clearTimeout(Activity.timer);
  Activity.timer = 0;
  const parts = Activity.wire();
  if (!parts) return;
  const now = Activity.now();
  const shown = Activity.enabled ? Activity.summary(now) : null;
  const text = shown?.text || "";
  if (parts.text.textContent !== text) parts.text.textContent = text;
  parts.box.hidden = !text;
  parts.box.classList.toggle("error", shown?.tone === "error");
  parts.stop.hidden = !shown?.stop;
  const line = document.getElementById("activity-line");
  if (line) {
    line.hidden = !shown?.busy;
    line.classList.toggle("indeterminate", Boolean(shown?.busy) && shown.fraction == null);
    line.style.width = shown?.fraction == null ? "" : `${Math.round(shown.fraction * 100)}%`;
  }
  if (Activity.detail) Activity.renderDetail(now);
  Activity.place();
  if (Activity.enabled && (Activity.items.size || Activity.detail)) {
    Activity.timer = setTimeout(renderActivity, Activity.TICK_MS);
  }
}

/** Stop waiting: end the wait `id` (by default the one the bar offers). Its poll loop stops;
 * the server's work carries on. False when there is none. */
function stopWaiting(id = null) {
  const item = id == null ? [...Activity.items.values()].find((found) => found.stop) : Activity.items.get(id);
  if (!item?.stop) return false;
  Activity.items.delete(item.id);
  item.stop();
  renderActivity();
  return true;
}

if (typeof module !== "undefined") {
  module.exports = { Activity, ERROR_COPY, rowNote, renderActivity, stopWaiting };
}
