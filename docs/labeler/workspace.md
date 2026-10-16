# Signal workspace (server side)

Status: **added 2026-10-16 (API 11; not committed).** The review page can add any
catalog signal, PTDATA pointname or MDSplus node as a row, open profiles in the dock
at the cursor, and save layouts as named views. Code: `src/labeler/events/review/`
(`catalog.py`, `signals.py`, `profiles.py`, `fetch_worker.py`, `views.py`). Routes:
`src/labeler/events/ui/workspace_routes.py`. Plan: section B of
`.claude/superpowers/plans/2026-10-15-labeler-workspace.md`.

## The catalog: `configs/labeler/signals.yaml`

```yaml
signals:
  - id: density                  # [a-z0-9][a-z0-9_.-]{0,63}, unique
    label: "Line-integrated density, CO2 R0"
    group: Plasma                # the picker's heading
    units: "cm^-2"
    kind: trace                  # trace | profile
    events: [confinement, ...]   # suggested for these events (and their children)
    legend: [..]                 # optional, one name per channel
    draw: lines                  # optional, lines | map (a channel x time image)
    axis: psi_n                  # a profile's x: psi_n | rho | R_m | channel
    sources:                     # tried in order; the first that has the shot wins
      - {corpus: co2, rows: 0}
      - {raw: co2, rows: 0}    # optional: scale, total, tag
      - {feature: co2_r0}
      - {ptdata: DENR0UF}
```

A source names exactly one of `corpus`, `raw`, `feature` (the features store), `park`
(`a_q95`, `a_qmin`), `recipe` (a profile reader), `ptdata` or `mds` (with `tree`).
Optional: `rows` (an index, `"a-b"` or a list; required for corpus and raw), `total`
(sum the rows), `scale`, `tag` (shown after the label). A local source whose rows are
all NaN for the shot falls through to the next. The recipes are `park_q`, `efit01_q`
(fetched), `thomson_te`, `thomson_ne`, `zipfit_rotation` and `ece_te`; see the module
docstring of `profiles.py` for each x axis.

## Ids a request may name

- a catalog id;
- `custom:<NAME>`, a PTDATA pointname, `[A-Za-z][A-Za-z0-9_]{0,11}`;
- `mds:<TREE>:<node>`, with TREE one of EFIT01, EFIT02, EFIT02ER, MHD, SPECTROSCOPY,
  ELECTRONS, IONS, D3D, ZIPFIT01, ZIPFIT02, ECE, BOLOM, and the node a plain path in
  that tree (`\TREE::A.B:C`, at most 96 characters; no parentheses, quotes, commas,
  spaces, `$`, or a segment starting with `_` or a digit). The node must be a time
  trace; a 2-D node is asked again for both of its dimensions.

PTDATA and MDSplus names are case-insensitive: a free id's source is upper-cased, so
`custom:denr0uf` and `custom:DENR0UF` share one fetch and one cache entry.

Anything else is 422 `invalid` before any fetch.

## Routes

| Route | Answer |
| --- | --- |
| `GET /api/catalog?event&shot` | `{entries: [{id, label, group, units, kind, channels, axis, suggested, source, where, local, draw?}], trees, ptdata_pattern, node_pattern, node_max}`; `local` is disk, fetch or missing for the shot |
| `GET /api/signal?shot&id&t0&t1&cols` | float32 LE `(2, C, cols)` (minima, then maxima) with `X-Grid {t0, t1, n}` and `X-Signal {label, units, legend, source, where, id}`; 202 `{state, stage, elapsed_s, queue_position}` while it is fetched |
| `GET /api/profile?shot&id` | float32 LE `t`, `x`, `y` with `X-Profile {n_t, n_x, x_kind, x_per_time, label, units, x_units, source, id}`; the same 202 |
| `GET /api/fetch/status` | `{fdp: ok\|expired\|unavailable\|off, reason?, queue, running: {id, shot, elapsed_s}\|null}`; `off` while `LABELER_NO_FETCH` is set; `reason` whenever nothing can be fetched |
| `GET /api/version` | adds `fdp: {env, expires_at}` and `workspace: true` |
| `GET /api/views?event&reviewer` | `{views: [{name, layout, etag, preset?}]}`: the reviewer's, then the presets |
| `POST /api/views` | `{op: save\|rename\|delete, event, reviewer, name, etag, layout?, new_name?}` |

Errors are `{error, kind}`: 422 invalid, 404 missing, 503 auth (no fdp wrapper, an
expired token, or `LABELER_NO_FETCH`), 502 upstream, 504 timeout. A catalog blind shot
is 404. Views answer `{error}`: 400 for a bad name (`Default` is the page's own), a
missing reviewer, or a layout over 64 KB; 409 when the etag does not match ("This
view changed in another tab..."); 404 for a missing view; 500 for an unreadable
store, which is left as it is.

## Fetching

One spawn-context child process fetches, one job at a time; jobs are deduplicated per
`(shot, source)`. A PTDATA point or a catalog node has 30 s, a free node or a recipe
120 s, and a new child 90 s to import toksearch. A timeout kills the child; the next
job starts another, and a new fdp token (after `fdp login`) starts a new one too. The
token is read like `fdp.auth`: `$BEARER_TOKEN` unless it is an expired JWT, then
`~/.fdp/cache/*.token`, then `~/.fdp/token`; it is never logged or sent to the page.
The server must run under `fdp run` (`pixi run -e labeler labeler-verify`) to fetch.
A record's size limits (512 channels, 64M samples) are checked after toksearch has
read it into the child, so a huge free node can hold login-node memory until it
answers or its 120 s run out; the parent never receives it.

## The cache

`$LABELER_ROOT/signals/<shot>.h5`, one group per source (`xdata` seconds, `ydata`
`(C, n)` float32; a profile holds `t_ms`, `x`, `y`). A "no such data" answer is
remembered for 24 h; auth, upstream and timeout failures are never cached. Files are
rewritten whole and renamed into place; the folder is kept under 20 GB by last use.
Long local records are drawn from min/max pyramids (64 and 4096 samples per block)
built once per record and kept in memory (1 GiB). When the blocks are coarser than the
columns (many channels over a long window), each column shows the minimum and maximum
of the block that covers it, a step envelope, never values interpolated between
blocks.

## Views

`$LABELER_ROOT/review_views/views.json` (`{reviewer: {event: {name: {layout, etag,
saved_at}}}}`), rewritten atomically under a lock. Presets come from
`configs/labeler/views.yaml` (`presets: {event: [{name, layout}]}`) and cannot be
overwritten.
