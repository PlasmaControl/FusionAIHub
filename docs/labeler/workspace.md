# Signal workspace (server side)

Status: **added 2026-10-16 (API 11; not committed).** The review page can add any
catalog signal, PTDATA pointname or MDSplus node as a row, open profiles in the dock
at the cursor, and save layouts as named views. Code: `src/labeler/events/review/`
(`catalog.py`, `signals.py`, `profiles.py`, `fetch_worker.py`, `views.py`). Routes:
`src/labeler/events/ui/workspace_routes.py`. Plan: section B of
`.claude/superpowers/plans/2026-10-15-labeler-workspace.md`.

The label tables live in an `events` folder outside the repository (`/scratch/gpfs/EKOLEMEN/nc1514/events` by default); the setting `LABELER_LABEL_TABLES` (`Paths.label_tables`) names it.

## The catalog: `configs/labeler/signals.yaml`

```yaml
signals:
  - id: density                  # [a-z0-9][a-z0-9_.-]{0,63}, unique
    label: "Line-integrated density, CO2 R0"
    group: Plasma                # the picker's heading
    units: "cm^-2"
    kind: trace                  # trace | profile
    description: "CO2 interferometer R0 chord, line-integrated density."  # one line, <= 240
    avail: both                  # both | early | late: probed on 170659 / 200958 (below)
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
(sum the rows), `scale`, `tag` (shown after the label), and for an `mds` source
`free: true` (the 120 s limit and the 2-D retry of a typed node, for a slow node such
as `TECEF20`, about a minute, or a record with several channels; a trace entry fed by
a 2-D record also lists `channels`). A local source whose rows are all NaN for the
shot falls through to the next; of the fetched ones only a `missing` answer does.
The local recipes are `park_q`, `thomson_te`, `thomson_ne`, `zipfit_rotation` and
`ece_te`. The fetched ones (through the fetch worker, cached with the signals; see
`profile_fetch.py`) are `efit01_q`, `efit0N_{pres,pprime,ffprim,fpol,q}` against
psi_N and `zipfit0N_{ne,te,ti,rot}` against rho (N is 1 or 2); see the module
docstring of `profiles.py` for each x axis. A fetched profile entry lists them in
the order they should be tried, e.g. `pressure_efit` is `efit02_pres` then
`efit01_pres`, each with `scale: 1.0e-3` (Pa to kPa).

`description` (one line, at most 240 characters) is what the page searches next to
the id, label, group and units; `/api/catalog` also sends `points`, the PTDATA names
and MDS nodes an entry fetches, so a typed name matches the entry that holds it.

As of 2026-10-17 the file lists 423 entries in 16 groups: Magnetics 85, Equilibrium
68, Heating 56, Core 36, Edge 28, Radiation 26, Fuelling 24, Ions 21, Profiles 18,
Divertor 17, Plasma 12, Fluctuations 12, Control 9, Neutrons 5, Density 4 and
Disruption 2. Their `avail` is both on 334, late on 39, early on 17, and 33 have no
fetched source (local only, the Langmuir probes among them). Every PTDATA point and
MDS node in it was tried on shots 170659 and 200958; the ones that answered nothing
on either were left out.

### Adding a pointname to the catalog

A pointname does not need an entry to be fetched: type `custom:<NAME>` (PTDATA) or
`mds:<TREE>:<node>` in the page's "Add signal" box. An entry makes it searchable by a
label and a description, and gives it its units, scale and the events it is suggested
for. To add one:

1. Check it answers: `fdp run python` with `labeler.events.review.fetch_worker.run(
   {"via": "ptdata", "name": "IP"}, shot)` (or `{"via": "mds", "node": ..., "tree":
   ...}`) on a shot from each era; a name that does not exist raises `PtDataError`
   (`getservbyname failed for task 'PTSERVER'`) or `TreeFOPENR`.
2. Append the entry to the group's section of `configs/labeler/signals.yaml`: a unique
   `id`, `label`, `group`, `units` (and a `scale` on the source that takes the archive's
   into them), `kind: trace`, `description`, and sources, local ones first, then
   `{ptdata: NAME}` or `{mds: '\TREE::A.B:C', tree: TREE}`. The tree must be on
   `catalog.TREES`; a new one is added there, and to the list below.
3. Set `avail` to what step 1 found: `both`, `early` or `late`. Leave an entry that has
   no fetched source and no probed archive point without it. An entry that answers on
   neither shot is not listed (an original entry with local sources stays, unmarked).

`tests/labeler/test_review_catalog_r2.py` loads the file and checks each entry's
description, `avail` and that every PTDATA name and node also resolves when typed.

### The availability column

`avail` records which of two reference shots answered when each fetched source of the
entry was tried through the fetch worker's own code (`fetch_worker.run`, 2026-10-17):
**170659** (`early`, 2017) and **200958** (`late`, 2020). `both` means the entry's
data are there in both eras; `early` or `late` means one of them lacked it (a point
introduced later, or retired); an entry listed without `avail` has only local
sources, none of whose archive points was probed. A wide corpus-backed entry (the
48 CER channels, the 12 gyrotrons, the 24 VB chords) takes `avail` from the archive
points of two or three of its channels, which the on-demand build would fetch. PTDATA is read through a JSON index that stops at about shot 201299, so a
PTDATA-only entry is not available for the shots after it. `/api/catalog` sends the
word as `avail`, next to the per-shot `local` (`disk`, `fetch`, `missing`), which
says what this machine has for the shot being reviewed; `avail` does not change with
the shot. The page may show it as a column or a badge.

PTDATA and MDS answer a point the shot lacks with errors that look like an outage
(`getservbyname failed for task 'PTSERVER'`, `TreeFOPENR`). The server's fetch worker
(`profile_fetch.run`) asks for `ip` of the same shot straight afterwards: if that
works the point is `missing` (404, remembered for 24 h, the next source is tried);
if not, the error stays 502 upstream.

## Ids a request may name

- a catalog id;
- `custom:<NAME>`, a PTDATA pointname, `[A-Za-z][A-Za-z0-9_]{0,11}`;
- `mds:<TREE>:<node>`, with TREE one of EFIT01, EFIT02, EFIT02ER, MHD, SPECTROSCOPY,
  ELECTRONS, IONS, D3D, ZIPFIT01, ZIPFIT02, ECE, BOLOM, RF, NB, TRANSPORT,
  OPERATIONS, and the node a plain path in that tree (`\TREE::A.B:C`, at most 96
  characters; no parentheses, quotes, commas, spaces, `$`, or a segment starting with
  `_` or a digit). The node must be a time trace; a 2-D node is asked again for both
  of its dimensions.

PTDATA and MDSplus names are case-insensitive: a free id's source is upper-cased, so
`custom:denr0uf` and `custom:DENR0UF` share one fetch and one cache entry.

Anything else is 422 `invalid` before any fetch.

## Routes

| Route | Answer |
| --- | --- |
| `GET /api/catalog?event&shot` | `{entries: [{id, label, group, units, kind, channels, axis, suggested, source, where, local, description, points, draw?, avail?}], trees, ptdata_pattern, node_pattern, node_max}`; `local` is disk, fetch or missing for the shot; `points` are the PTDATA names and MDS nodes the entry fetches; `avail` is both, early or late (below) |
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

The build of a shot (not a workspace row) also fetches on demand: a feature the shot's
feature file lacks, such as locked mode's `pinj_total`, is fetched and the build is
retried once. It does not use the child process above: it calls toksearch in the server
process (`events.ondemand`), one call at a time under a lock with a 120 s timeout, and
a token fdp refused stops it until the token changes. See docs/labeler/on_demand.md.
That cache is separate from this one.

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
