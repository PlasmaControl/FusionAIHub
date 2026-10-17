// Signals and profiles beside the store's rows (plan D.2, D.3): /api/signal and /api/profile
// with their 202 polling, one cache per shot for the last six shots. A classic script loaded
// before app.js: it only defines names, and views.js and dock.js call them (api 11, workspace).
"use strict";

/** The index of the time in ascending `times` nearest `t`, by binary search; -1 when `times`
 * is empty or `t` is not a number. A tie goes to the earlier time. */
function sliceIndex(times, t) {
  const n = times?.length || 0;
  if (!n || !Number.isFinite(t)) return -1;
  let [lo, hi] = [0, n - 1];
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  return lo > 0 && t - times[lo - 1] <= times[lo] - t ? lo - 1 : lo;
}

/** A trace's bytes from /api/signal: float32 (2, C, n), each channel's minima then maxima per
 * column. `values` is a copy (aligned); `channels` is C. */
function unpackTrace(buffer, n) {
  const values = new Float32Array(buffer.slice(0));
  return { values, channels: n > 0 ? Math.floor(values.length / (2 * n)) : 0 };
}

/** Added traces by shot and id, and the fetch loop that profiles share. An entry is
 * {kind, shot, id, facts, state, want, asked, data, signal, error, stage, elapsed_s,
 * queue_position, hold, onChange}: `state` is idle, queued, loading, done, error or stopped;
 * `data` the last window that arrived ({t0, t1, n, values, channels}) and `signal` its
 * X-Signal record ({label, units, legend, source}). */
const Signals = {
  KEEP_SHOTS: 6, // the shots whose signals and profiles stay cached
  POLL_MS: 1000, // a 202 is asked again after this long
  KINDS: { 400: "invalid", 422: "invalid", 404: "missing", 503: "auth", 502: "upstream", 504: "timeout" },
  shots: new Map(), // shot -> Map(id -> entry), oldest shot first

  /** The error kind of a failed request: the server's `kind`, else one by the status code. */
  kindOf(error) {
    return error?.kind || Signals.KINDS[error?.status] || "upstream";
  },

  /** The /api/signal query for a window. */
  query(shot, id, t0, t1, cols) {
    return `shot=${shot}&id=${encodeURIComponent(id)}&t0=${t0}&t1=${t1}&cols=${cols}`;
  },

  /** The cache of `shot` in `shots`, made the newest; the oldest shots past `KEEP_SHOTS` go
   * (their loops see `gone` and stop). */
  forShot(shots, shot) {
    let found = shots.get(shot);
    if (found) shots.delete(shot);
    else found = new Map();
    shots.set(shot, found);
    while (shots.size > Signals.KEEP_SHOTS) {
      const [oldest, entries] = shots.entries().next().value;
      for (const entry of entries.values()) entry.gone = true;
      shots.delete(oldest);
    }
    return found;
  },

  /** The entry of `id` on `shot` in `shots`, made idle when new. */
  entry(shots, kind, shot, id) {
    const entries = Signals.forShot(shots, shot);
    let entry = entries.get(id);
    if (!entry) {
      entry = { kind, shot, id, facts: {}, state: "idle", want: null, asked: null, data: null, signal: null,
        error: null, stage: "", elapsed_s: 0, queue_position: 0, hold: null, onChange: null, running: false };
      entries.set(id, entry);
    }
    return entry;
  },

  /** The cached entry of `id` on `shot`, or null; never makes one. */
  peek(shot, id) {
    return Signals.shots.get(shot)?.get(id) || null;
  },

  /** Say the entry changed: its owner redraws its row or panel and chip. */
  changed(entry) {
    try {
      entry.onChange?.(entry);
    } catch (error) {
      console.error(error);
    }
  },

  /** Start the bar's item for the entry while it is fetched (one per entry, not per poll). */
  holdOn(entry) {
    if (typeof Activity === "undefined" || entry.hold) return;
    const signal = entry.kind === "signal";
    entry.hold = Activity.hold(null, {
      kind: entry.kind, what: signal ? "Fetching a signal" : "Loading a profile",
      shot: entry.shot, quiet: false,
    });
  },

  holdOff(entry, error = null) {
    if (typeof Activity === "undefined" || !entry.hold) return;
    Activity.end(entry.hold, error);
    entry.hold = null;
  },

  /** The open shot, or undefined outside the page (node tests). */
  openShot: () => (typeof S === "undefined" ? undefined : S.shot),

  wait: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),

  /** Ask for `entry.want` until it arrives, fails, is stopped or the shot changes; a newer
   * `want` set meanwhile is asked next (the latest wins). One loop per entry. */
  async run(entry) {
    if (entry.running) return;
    entry.running = true;
    let failure = null;
    try {
      while (entry.want && entry.want !== entry.asked && !entry.gone && entry.state !== "stopped") {
        if (entry.shot !== Signals.openShot()) {
          entry.state = entry.data ? "done" : "idle";
          break;
        }
        const query = entry.want;
        const path = `/api/${entry.kind}?${query}`;
        Signals.holdOn(entry);
        let response;
        try {
          response = await api(path, { quiet: true });
        } catch (error) {
          if (entry.gone || entry.state === "stopped") break;
          if (query !== entry.want) continue; // the view moved on: ask for the new window
          entry.state = "error";
          entry.error = { kind: Signals.kindOf(error), message: error.message, status: error.status };
          entry.asked = query;
          failure = error;
          Signals.changed(entry);
          break;
        }
        if (response.status === 202) {
          const body = await response.json().catch(() => ({}));
          if (entry.gone || entry.state === "stopped") break;
          entry.state = body.state === "queued" ? "queued" : "loading";
          Object.assign(entry, { stage: body.stage || "", elapsed_s: Number(body.elapsed_s) || 0,
            queue_position: Number(body.queue_position) || 0 });
          Signals.changed(entry);
          await Signals.wait(Signals.POLL_MS);
          continue;
        }
        try {
          if (entry.kind === "signal") Signals.take(entry, response, await response.arrayBuffer());
          else Profiles.take(entry, response, await response.arrayBuffer());
        } catch (error) {
          entry.state = "error";
          entry.error = { kind: "upstream", message: error.message };
          entry.asked = query;
          failure = error;
          Signals.changed(entry);
          break;
        }
        if (entry.gone) break;
        entry.asked = query;
        entry.state = "done";
        entry.error = null;
        Signals.changed(entry);
      }
    } finally {
      entry.running = false;
      Signals.holdOff(entry, failure);
    }
  },

  /** A 200 from /api/signal: the window's values and its X-Signal record. */
  take(entry, response, buffer) {
    const grid = JSON.parse(response.headers.get("X-Grid") || "null");
    if (!grid || !(grid.n > 0)) throw new Error("the signal came without its time grid");
    const { values, channels } = unpackTrace(buffer, grid.n);
    if (!channels) throw new Error("the signal came without values");
    entry.signal = JSON.parse(response.headers.get("X-Signal") || "{}");
    entry.data = { t0: grid.t0, t1: grid.t1, n: grid.n, values, channels };
  },

  /** Stop waiting for the entry: its loop ends, the server keeps fetching, and Retry asks
   * again. */
  stop(entry) {
    if (!entry || !["queued", "loading"].includes(entry.state)) return false;
    entry.state = "stopped";
    Signals.holdOff(entry);
    Signals.changed(entry);
    return true;
  },

  /** Ask again for the entry's last window (Retry on a chip). */
  retry(entry) {
    if (!entry) return;
    Object.assign(entry, { state: "idle", asked: null, error: null });
    Signals.changed(entry);
    Signals.run(entry);
  },
};

/** Load the window `t0`..`t1` at `cols` columns of the added signal `id` on `shot`. `facts`
 * ({name, source, tree}) word its chip; `onChange(entry)` hears every change. Returns the
 * entry at once; its data arrive later. An entry already showing this window asks nothing. */
function loadSignal(shot, id, t0, t1, cols, facts = {}, onChange = null) {
  const entry = Signals.entry(Signals.shots, "signal", shot, id);
  entry.facts = facts;
  if (onChange) entry.onChange = onChange;
  if (entry.state === "stopped" || entry.state === "error") return entry; // Retry asks again
  entry.want = Signals.query(shot, id, t0, t1, cols);
  if (entry.want !== entry.asked) {
    if (!entry.running && entry.state !== "done") entry.state = "loading";
    Signals.run(entry);
  }
  return entry;
}

/** Profiles (x against t) by shot and id; one blob per (shot, id), sliced in the page. */
const Profiles = {
  KINDS: ["psi_n", "rho", "R_m", "channel"],
  shots: new Map(),

  /** Split /api/profile's body by its X-Profile `header`: float32 t (n_t), then x (n_x, or
   * n_t * n_x when x_per_time), then y (n_t * n_x, C order). Throws when the body is short. */
  parse(header, buffer) {
    const n = Math.max(0, Number(header?.n_t) || 0), m = Math.max(0, Number(header?.n_x) || 0);
    const values = new Float32Array(buffer.slice(0));
    const xs = header.x_per_time ? n * m : m;
    if (values.length < n + xs + n * m) throw new Error("the profile is shorter than its header says");
    return { ...header, n_t: n, n_x: m, t: values.subarray(0, n), x: values.subarray(n, n + xs),
      y: values.subarray(n + xs, n + xs + n * m) };
  },

  take(entry, response, buffer) {
    entry.data = Profiles.parse(JSON.parse(response.headers.get("X-Profile") || "{}"), buffer);
    entry.signal = { label: entry.data.label, units: entry.data.units, source: entry.data.source };
  },

  /** The `lo` and `hi` quantiles of the finite `values` (unsorted): [q_lo, q_hi], or null. */
  span(values, lo = 0.01, hi = 0.99) {
    const finite = Float32Array.from(Array.prototype.filter.call(values || [], Number.isFinite)).sort();
    if (!finite.length) return null;
    const at = (q) => finite[Math.min(finite.length - 1, Math.max(0, Math.round(q * (finite.length - 1))))];
    return [at(lo), at(hi)];
  },

  /** A profile's y range over all its times (1st to 99th percentile) and its x range, cached
   * on the profile. */
  ranges(profile) {
    if (!profile.ranges) {
      const y = Profiles.span(profile.y) || [0, 1];
      const x = Profiles.span(profile.x, 0, 1) || [0, 1];
      if (!(y[1] > y[0])) y[1] = y[0] + 1;
      if (!(x[1] > x[0])) x[1] = x[0] + 1;
      profile.ranges = { x, y };
    }
    return profile.ranges;
  },
};

/** Load the profile `id` of `shot` (once per shot; the entry is cached). */
function loadProfile(shot, id, facts = {}, onChange = null) {
  const entry = Signals.entry(Profiles.shots, "profile", shot, id);
  entry.facts = facts;
  if (onChange) entry.onChange = onChange;
  if (entry.state === "done" || entry.state === "stopped" || entry.state === "error") return entry;
  entry.want = `shot=${shot}&id=${encodeURIComponent(id)}`;
  if (!entry.running) entry.state = "loading";
  Signals.run(entry);
  return entry;
}

/** The slice of `profile` nearest `t`: {i, t, x, y} (x and y are views of length n_x), or
 * null when the profile has no times. */
function profileAt(profile, t) {
  const i = sliceIndex(profile?.t, t);
  if (i < 0) return null;
  const m = profile.n_x;
  return { i, t: profile.t[i], x: profile.x_per_time ? profile.x.subarray(i * m, (i + 1) * m) : profile.x,
    y: profile.y.subarray(i * m, (i + 1) * m) };
}

if (typeof module !== "undefined") {
  module.exports = { Signals, Profiles, loadSignal, loadProfile, profileAt, unpackTrace, sliceIndex };
}
