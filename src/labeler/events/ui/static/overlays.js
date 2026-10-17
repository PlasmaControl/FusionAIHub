// TokEye on every spectrogram (plan D.4): each image row's overlay state from the shot's meta,
// its chip in the gutter (#row-ui), the layers of events other than AE, and the request and
// poll for layers still to compute. A classic script loaded before app.js: it only defines
// names, and app.js calls them (api 11).
"use strict";

/** The open shot's TokEye states and layers. `states` maps an image row's name to the
 * server's overlay object ({state, reason, elapsed_s?, queue_position?, updated?}) plus
 * `seen`, when the page read it, and `local` on a failure of the page's own. AE keeps its
 * one whole-shot layer in S.tokeye as before; other events' layers are in `layers`. */
const Overlays = {
  POLL_MS: 5000,
  WAITING: new Set(["queued", "computing"]),
  ASK: new Set(["stale", "not_computed", "failed"]), // the states Compute, Recompute and Retry ask the server about
  REASON_CHARS: 40, // a failure's reason on the chip; the title has all of it
  states: {},
  layers: {},
  chips: new Map(), // row name -> its chip in #row-ui
  timer: 0,
  ticket: 0, // the openShot the states belong to
  at: null, // "<event>/<shot>" the layers belong to: a reopen of the same shot keeps them
  posting: false, // a request for the shot is on its way
  now: () => Date.now(),

  /** The image rows' states in `meta` (an open shot's /api/shot body), stamped `now`. */
  statesFrom(meta, now) {
    const found = {};
    for (const row of meta?.rows || []) {
      if (row.kind === "image" && row.overlay?.state) found[row.name] = { ...row.overlay, seen: now };
    }
    return found;
  },

  /** The row names whose state is one of `states` (a Set or a state). */
  named(states, which = Overlays.states) {
    const wanted = typeof states === "string" ? new Set([states]) : states;
    return Object.keys(which).filter((name) => wanted.has(which[name].state));
  },

  /** Rows still being computed or queued: the poll runs while there are some. */
  waiting: (which = Overlays.states) => Overlays.named(Overlays.WAITING, which).length > 0,

  /** The open event is AE or one of its subcategories: its layer is S.tokeye, as today. */
  ae: () => (S.meta?.parent?.event ?? S.event) === MASK_EVENT,

  /** A layer from /api/tokeye as drawTokeye takes it (`y0_khz`, `dy_khz` from the encoding's
   * `y0` and `dy`), or null for anything else, such as a {state, reason} answer. */
  layer(raw) {
    if (!raw || typeof raw !== "object" || typeof raw.bits !== "string") return null;
    const y0 = raw.y0_khz ?? raw.y0, dy = raw.dy_khz ?? raw.dy, grid = raw.grid;
    const usable = grid && grid.n > 0 && Number.isFinite(grid.t0_ms) && grid.dt_ms > 0 && raw.n_y > 0
      && Number.isFinite(y0) && dy > 0;
    return usable ? { ...raw, y0_khz: y0, dy_khz: dy } : null;
  },

  /** The usable layers of a /api/tokeye {rows: {name: layer}} answer, by row name. */
  normalise(body) {
    const found = {};
    for (const [name, raw] of Object.entries(body?.rows || {})) {
      const layer = Overlays.layer(raw);
      if (layer) found[name] = layer;
    }
    return found;
  },

  /** `row`'s own layer, or null (drawTokeye then falls back to S.tokeye). */
  layerFor: (row) => Overlays.layers[row?.name] || null,

  /** Some row of the open shot has a layer of its own. */
  any: () => Object.keys(Overlays.layers).length > 0,

  /** An image row's canvas title when TokEye does not apply to it, else "". */
  title(row) {
    return row?.kind === "image" && Overlays.states[row.name]?.state === "not_applicable" ? "No TokEye: not a spectrogram" : "";
  },

  /** Start on a shot (or on none): its states from `meta`, no poll, and no layers unless the
   * same shot was open (a reload after a save keeps them). openShot calls this once S.shot
   * and S.meta are the new shot's, before buildRows; showNothing calls it with null. */
  open(meta, ticket = S.ticket) {
    clearTimeout(Overlays.timer);
    const at = meta ? `${S.event}/${S.shot}` : null;
    Object.assign(Overlays, {
      timer: 0, ticket, posting: false, layers: at && at === Overlays.at ? Overlays.layers : {}, at,
      states: Overlays.statesFrom(meta, Overlays.now()),
    });
    Overlays.renderChips();
  },

  /** Set `fields` on the rows `names`, stamped now. */
  mark(names, fields) {
    const now = Overlays.now();
    for (const name of names) Overlays.states[name] = { ...fields, seen: now };
  },

  /** Done rows the page has no layer for become failures of the page's own: Retry reads the
   * layers again. A failure of the page's own whose layer has since arrived (M retried the
   * mask and TokEye) is done again: only done rows ever become one. */
  check() {
    const has = (name) => Boolean(Overlays.ae() ? S.tokeye : Overlays.layers[name]);
    Overlays.mark(Object.keys(Overlays.states).filter((name) => Overlays.states[name].local && has(name)), { state: "done" });
    const missing = Overlays.named("done").filter((name) => !has(name));
    Overlays.mark(missing, { state: "failed", reason: "the layer was not found", local: true });
    Overlays.renderChips();
  },

  /** One /api/tokeye for an event other than AE, when a row is done. A 404 (nothing served)
   * and any other failure turn the done rows into failures of the page's own. */
  async fetchLayers(ticket = Overlays.ticket) {
    const event = S.event, shot = S.shot;
    try {
      const body = await (await api(`/api/tokeye?event=${encodeURIComponent(event)}&shot=${shot}`)).json();
      if (ticket !== S.ticket) return;
      Overlays.layers = Overlays.normalise(body);
    } catch (error) {
      if (ticket !== S.ticket) return;
      Overlays.layers = {};
      if (error.status !== 404) {
        Overlays.mark(Overlays.named("done"), { state: "failed", reason: `the layer could not be loaded: ${error.message}`, local: true });
      }
    }
    Overlays.check();
    render();
  },

  /** Read the layers again after rows became done (or on Retry of a page failure): AE's one
   * layer through loadTokeye, the others' through fetchLayers. */
  async refresh(ticket = Overlays.ticket) {
    if (!Overlays.ae()) return Overlays.fetchLayers(ticket);
    const layer = await loadTokeye(S.event, S.shot);
    if (ticket !== S.ticket) return;
    S.tokeye = Overlays.layer(layer) && layer;
    showMasks();
    Overlays.check();
    render();
  },

  /** Take a poll's {name: state} answer: new states, chips, and the layers of rows that became
   * done. Rows the shot's meta did not list as image rows are left out. */
  async take(rows, ticket = Overlays.ticket) {
    const now = Overlays.now();
    let done = false;
    for (const [name, state] of Object.entries(rows || {})) {
      const before = Overlays.states[name];
      if (!before || !state?.state) continue;
      done ||= state.state === "done" && before.state !== "done";
      Overlays.states[name] = { ...state, seen: now };
    }
    Overlays.renderChips();
    if (done) await Overlays.refresh(ticket);
  },

  /** A chip's button: Compute, Recompute and Retry ask the server; Retry on a failure of the
   * page's own reads the layers again. Returns when that is done. */
  async act(name) {
    const state = Overlays.states[name];
    if (!state) return;
    if (state.local) {
      Overlays.mark(Overlays.named("failed").filter((found) => Overlays.states[found].local), { state: "done" });
      Overlays.renderChips();
      await Overlays.refresh().catch((error) => Activity.fail("Loading TokEye", error.message));
    } else if (Overlays.ASK.has(state.state)) await requestOverlay();
  },

  /** Draw each image row's chip in #row-ui from its state, keeping the elements of rows that
   * stay; D2's chips there are left alone. A new chip stays invisible until `slot` places it. */
  renderChips() {
    if (typeof document === "undefined") return;
    const ui = document.getElementById("row-ui");
    if (!ui) return;
    const keep = new Set();
    for (const row of S.meta?.rows || []) {
      const state = Overlays.states[row.name];
      const note = row.kind === "image" && state ? overlayNote(state, { hidden: !S.showMasks, now: Overlays.now() }) : null;
      if (!note) continue;
      keep.add(row.name);
      let chip = Overlays.chips.get(row.name);
      if (!chip) {
        chip = Object.assign(document.createElement("div"), { className: "row-chip" });
        chip.dataset.overlay = row.name;
        chip.style.visibility = "hidden";
        const text = document.createElement("span");
        const button = Object.assign(document.createElement("button"), { type: "button" });
        button.addEventListener("click", () => Overlays.act(chip.dataset.overlay));
        chip.append(text, button);
        ui.append(chip);
        Overlays.chips.set(row.name, chip);
      }
      const [text, button] = chip.children;
      chip.className = `row-chip${note.tone ? ` ${note.tone}` : ""}`;
      chip.title = note.title || "";
      if (text.textContent !== note.text) text.textContent = note.text;
      button.hidden = !note.button;
      button.textContent = note.button || "";
    }
    for (const [name, chip] of Overlays.chips) {
      if (keep.has(name)) continue;
      chip.remove();
      Overlays.chips.delete(name);
    }
  },

  /** Place `row`'s chip in its gutter at `top` px down `canvas` (drawGutter passes the bottom
   * of its text), and return the [right, top, bottom] rectangle drawGutter keeps its y ticks
   * clear of; null when the row has no chip on screen. */
  slot(row, top, canvas) {
    const chip = Overlays.chips.get(row?.name);
    const ui = chip?.parentElement;
    if (!chip || !canvas || !ui || !chip.offsetParent) return null;
    const offset = (element) => {
      let [x, y] = [0, 0];
      for (let at = element; at; at = at.offsetParent) [x, y] = [x + at.offsetLeft, y + at.offsetTop];
      return [x, y];
    };
    const [[cx, cy], [ux, uy]] = [offset(canvas), offset(ui)];
    // On a row too short for it under the text, the chip's bottom stays inside the row.
    const [height, room] = [chip.offsetHeight, canvas.clientHeight - 2];
    const at = top + height > room ? Math.max(2, room - height) : top;
    const [left, place] = [`${cx - ux + 8}px`, `${cy - uy + at}px`];
    if (chip.style.left !== left) chip.style.left = left;
    if (chip.style.top !== place) chip.style.top = place;
    chip.style.visibility = "";
    return [8 + chip.offsetWidth, at, at + height];
  },
};

/** A TokEye chip, {text, button, tone, title}, for a row's overlay `state`; null for no chip
 * (not_applicable, an unknown state). `hidden`: M has hidden the layers. `now` brings a
 * computing row's elapsed time forward from when the page read it. */
function overlayNote(state, { hidden = false, now = Date.now() } = {}) {
  const q = state?.queue_position;
  switch (state?.state) {
    case "done":
      return { text: hidden ? "TokEye hidden (M)" : "TokEye", button: null, tone: "", title: "" };
    case "stale":
      return { text: "TokEye from an older model", button: "Recompute", tone: "", title: "" };
    case "queued":
      return { text: q > 0 ? `TokEye queued, ${Activity.ordinal(q)} in line` : "TokEye queued", button: null, tone: "queued", title: "" };
    case "computing": {
      const elapsed = state.elapsed_s >= 0 ? state.elapsed_s + (state.seen ? Math.max(0, now - state.seen) / 1000 : 0) : null;
      const since = elapsed >= Activity.CLOCK_MS / 1000 ? ` · ${Activity.clock(elapsed)}` : "";
      return { text: `TokEye computing${since}`, button: null, tone: "queued", title: "" };
    }
    case "failed": {
      const reason = String(state.reason || "").trim();
      const cut = reason.length > Overlays.REASON_CHARS ? `${reason.slice(0, Overlays.REASON_CHARS - 1).trimEnd()}…` : reason;
      return {
        text: reason ? `TokEye failed: ${cut}` : "TokEye failed", button: "Retry", tone: "error",
        title: reason ? `TokEye failed: ${reason}` : "",
      };
    }
    case "not_computed":
      return { text: "No TokEye yet", button: "Compute", tone: "", title: "" };
    default:
      return null;
  }
}

/** After the shot's rows are built: the chips, the layers of an event other than AE when a done
 * row has none yet (AE's come with loadMasks), and the poll while rows wait. A default open whose rows
 * neither wait nor are done asks for nothing. */
async function loadOverlays(ticket = Overlays.ticket) {
  if (!(S.api >= 11) || ticket !== S.ticket || !S.meta) return;
  Overlays.ticket = ticket;
  Overlays.renderChips();
  render();
  if (!Overlays.ae() && Overlays.named("done").some((name) => !Overlays.layers[name])) await Overlays.fetchLayers(ticket);
  pollOverlays(ticket);
}

/** Compute (or Recompute, or Retry) TokEye for the open shot: one POST /api/overlays/request,
 * the asking rows shown queued at once, then the server's answer (its layers when it says
 * done) and the poll. A failure puts it on those rows' chips. */
async function requestOverlay() {
  if (Overlays.posting || !S.meta || !(S.api >= 11)) return;
  const ticket = Overlays.ticket, event = S.event, shot = S.shot;
  const asking = Overlays.named(Overlays.ASK);
  if (!asking.length) return;
  Overlays.posting = true;
  Overlays.mark(asking, { state: "queued" });
  Overlays.renderChips();
  try {
    const response = await api("/api/overlays/request", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ event, shot }),
    });
    const body = await response.json();
    if (ticket !== S.ticket) return;
    if (body?.state) Overlays.mark(asking, { state: body.state, queue_position: body.queue_position });
  } catch (error) {
    if (ticket !== S.ticket) return;
    Overlays.mark(asking, { state: "failed", reason: `the request was refused: ${error.message}` });
  } finally {
    if (ticket === Overlays.ticket) Overlays.posting = false;
  }
  Overlays.renderChips();
  if (Overlays.named("done").some((name) => asking.includes(name))) await Overlays.refresh(ticket);
  pollOverlays(ticket);
}

/** Every POLL_MS while a row is queued or computing, read the shot's states (quietly) until
 * none waits or another shot opens. */
function pollOverlays(ticket = Overlays.ticket) {
  clearTimeout(Overlays.timer);
  Overlays.timer = 0;
  if (ticket !== S.ticket || !Overlays.waiting()) return;
  Overlays.timer = setTimeout(async () => {
    Overlays.timer = 0;
    if (ticket !== S.ticket) return;
    try {
      const response = await api(`/api/overlays?event=${encodeURIComponent(S.event)}&shot=${S.shot}`, { quiet: true });
      const body = await response.json();
      if (ticket !== S.ticket) return;
      await Overlays.take(body?.rows, ticket);
    } catch (error) {
      if (ticket !== S.ticket) return;
      Activity.fail("Reading TokEye states", error.message);
    }
    if (!Overlays.timer) pollOverlays(ticket);
  }, Overlays.POLL_MS);
}

if (typeof module !== "undefined") {
  module.exports = { Overlays, overlayNote, loadOverlays, requestOverlay, pollOverlays };
}
