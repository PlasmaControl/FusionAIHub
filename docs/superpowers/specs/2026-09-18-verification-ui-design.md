# Browser verification surface for event labels

Replace the sixteen per-event `verification.ipynb` notebooks with one
browser application, served the way `shot_design serve` is served, and route
every raw-signal read through the corpus.

`alfven_eigenmode` is the acceptance target: it must work end to end on a
real shot before anything here is called done. The other fifteen events are
registered against a generic builder and are expected to render, but are not
validated by this work.

## Why

The notebooks are three separate problems wearing one coat.

The panel code that decides which traces settle a phenomenon lives in a
notebook cell, so it cannot be called from anywhere else, cannot be tested,
and exists in sixteen copies that drift. Twelve of those copies are the same
placeholder (`ip`, `betan`, `pinj_total`) with a paragraph asking the reader
to replace it.

The window is fixed. `alfven_eigenmode`'s notebook hard-codes `t_range =
(0.0, 2000.0)` and block-averages to a thousand time bins, because a
full-rate spectrogram of even that window is ~3300 columns and three of them
as plotly heatmaps wedges the tab. A reviewer cannot see the rest of the
shot, and cannot zoom into a feature to decide where its boundary sits --
which is the one thing the surface exists for.

Raw data comes from three unrelated places: the corpus, a per-event npz
cache under `data/events/<event>/review/_cache/`, and, in
`example.ipynb`, a hard-coded `/scratch/gpfs/nc1514/aemodes/data/.cache/`
path belonging to a different project entirely.

## What does not change

The write path. `write_corrections`, `correction_path`, `corrections_for`
and `record_review` in `src/labeler/events/verify.py` are untouched, and so
is `ReviewSession.save()`. Corrections stay append-only -- every save writes
its own `review/<shot>__<reviewer>__<stamp>.csv` and nothing there is ever
overwritten -- and `shots.csv` stays the one file edited in place, gaining
only a reviewer and a date. `tier` and `holdout` remain hand-set curation
calls that nothing derives.

The existing `tests/labeler` coverage of that path stays green without
edits. That is the check that it really did not move.

`Panel` and `label_panel` also stay in `verify.py`. The registry composes
them; it does not redefine them.

## Components

### `src/labeler/events/raw.py` -- one way to get a signal

```python
raw_signal(shot, group, *, channels=None, t_range=None) -> FeatureArray
```

Three tiers, in order:

1. `/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5`
2. `.../foundation_model/verify_cache/<shot>_processed.h5`
3. fetch over fdp/toksearch, write tier 2, return

Contract matches `corpus_signal`: `x` in milliseconds, `y` as `(C, T)`
float32, one row per channel in the order asked for. A `Panel` cannot tell
which tier its array came from.

The overlay file is written in the corpus's own layout -- flat groups
holding `xdata` and `ydata`, as described at
`src/labeler/features/resolve_corpus.py:3` -- so a shot later fetched in
full is promoted into the corpus with `mv` and nothing else changes.

Writes are temp-file-plus-rename, and additive: fetching `ece` for a shot
whose overlay already holds `co2` adds a group rather than replacing the
file. The server will be doing this concurrently with itself.

`verify_cache/` is a subdirectory rather than the corpus root because
training loaders glob `*_processed.h5` there --
`src/tokamak_foundation_model/ignite/spike.py:220`,
`fastts_train.py:214`, `train_codec.py:633`. A verification fetch
materializes the one group a panel asked for, not all thirty-two, so a
partial shot in the root is a file those globs hand to training with
thirty-one groups missing. Keeping it one level down means the globs never
see it.

Cost: about 240 MB per AE shot, four CO2 chords at 1.667 MHz as float32.
There is no eviction and no cap, exactly as the npz cache it replaces had
none. The README says so; no reaper is built.

`fdp_signal` survives as the fetch primitive but loses its `cache=`
argument. `raw.py` owns caching now. The npz cache directory and its
gitignore entry are deleted, as are `TS_CACHE` and the feather-reading
`read_co2` in `data/events/alfven_eigenmode/example.ipynb`.

Live fetches only work under the fdp wrapper. `FDP_RESTART_COMMAND` in
`verify.py:125` records what happens without it: PTDATA fails with
`getservbyname failed for task 'PTSERVER'`, MDSplus with `TREE-E-FOPENR`.

### `src/labeler/events/panels/` -- what to plot

```
__init__.py              build(event, shot, *, t_range) -> list[Panel]
                         BUILDERS: dict[str, Builder]
_generic.py              ip / betan / pinj_total
alfven_eigenmode.py      CO2 crosspower over three chord pairs
fishbone.py
sawtooth_oscillation.py
minimum_safety_factor.py
```

Each builder is `(shot, *, t_range) -> list[Panel]`, lifted from the
notebook cell it replaces. An event with no entry falls back to `_generic`,
which is what the twelve placeholder notebooks already were.

A builder also carries a `guidance` string -- the prose the notebook's
markdown cell held about what to look for. The page shows it beside the
panels, which is where a reviewer is actually looking.

`alfven_eigenmode.py` keeps `crosspower()` verbatim, including its two
load-bearing comments: the sample rate is derived from the span and never
from a median of successive differences, because a float32 time vector
quantises its spacing at t ~ 3 s; and the block average is taken over the
power with the log applied after, because averaging the log is a geometric
mean that is pulled down by quiet bins and so suppresses exactly the short
bursts the panel exists to show.

Its band is 80-250 kHz over chord pairs R0xV1, R0xV2, R0xV3. A feature on a
single pair is more likely chord-specific noise than a real mode.

### `src/labeler/events/ui/` -- the surface

Ported from `src/shot_design/ui/`, structure for structure:

```
app.py            FastAPI, token gate, JSON endpoints
serve.py          launcher; binds 127.0.0.1, prints token link + ssh line
static/index.html
static/app.js     vanilla JS, plotly from the browser
static/style.css
```

Endpoints:

```
GET  /?token=...            the page; sets the cookie
GET  /api/events            registered events, with roster counts
GET  /api/shots?event=      rows from shots.csv, plus shots present in
                            format/ that the roster does not list yet
GET  /api/panels?event=&shot=&t0=&t1=
                            panels over [t0, t1] at 1000 time bins
POST /api/save              {event, shot, marks[], verify, notes}
```

`/api/shots` unions the roster with `format/`'s shots because
`alfven_eigenmode/shots.csv` currently holds three example rows and no real
shot, while the annotated set spans 170659-178879. A picker fed by the
roster alone would be empty.

Resolution: the page opens on the whole shot at a thousand bins, and
re-requests the visible window at a thousand bins on pan or zoom, debounced
about 200 ms, driven by plotly's `relayout`. A window is always as sharp as
the screen can show, whatever its width. The raw record sits in a
server-side LRU of two or three shots, so a re-render is a spectrogram and
not a refetch -- warm round trip in the low hundreds of milliseconds, against
several minutes for the fetch.

Marking keeps the notebook's semantics exactly. Box-select a range; *Mark
present* and *Mark absent* accumulate in memory; *Verify* sets the flag;
*Save* posts, and the server calls the existing `ReviewSession.save()`. Two
guards carry over into JS: a lasso writes `{type: "path"}` with `x0`/`x1`
both `None` and is rejected in favour of box select, and the selection is
cleared after a mark so a second click cannot silently record the same
interval twice.

The page also carries a fixed block saying what *Save* writes -- a new file
under `review/`, never an overwrite; a reviewer and a date into `shots.csv`;
nothing promoted. That text is currently in every notebook's closing
markdown cell.

### Launch

```
pixi run -e labelmaker fdp run python -m labeler.events.ui
```

`serve.py` refuses any host but `127.0.0.1`, mints a token, prints the link
and the `ssh -L` line, exactly as `shot_design/ui/serve.py:27` does. It also
checks for the fdp wrapper at startup and prints `FDP_RESTART_COMMAND`'s
equivalent if it is missing, rather than letting the first fetch fail with a
`PTSERVER` error several minutes in.

`fastapi` and `uvicorn` are added to `[tool.pixi.feature.labelmaker.dependencies]`.
That feature already declares `plotly` but neither of those.

## Removals

The sixteen `data/events/*/verification.ipynb` files are deleted. Each
event's `README.md` gains the launch line in their place.

## Testing

`tests/labeler/test_events_raw.py`

- the three-tier lookup returns from each tier in order
- an overlay write is additive: a second group joins the file
- an overlay write is atomic: an interrupted write leaves no partial file
- the overlay honours `channels` and `t_range` the way `corpus_signal` does
- a shot moved from `verify_cache/` to the corpus root reads identically

`tests/labeler/test_events_panels.py`

- every registered event builds panels against a synthetic shot
- an unregistered event falls back to `_generic`
- `crosspower` derives its rate from the span: a float32 time vector whose
  successive differences quantise still yields the right frequency axis

`tests/labeler/test_events_ui.py`

- no token, or a wrong one, is refused
- `/api/panels` honours `t0`/`t1` and returns a thousand bins at any width
- `/api/shots` unions the roster with `format/`
- `/api/save` round-trips through the existing append-only path, and a
  second save writes a second file rather than overwriting the first

Existing `verify.py` tests run unchanged.

## Acceptance

Against `alfven_eigenmode`, on a real annotated shot, over the full shot
rather than 0-2000 ms:

1. `pixi run -e labelmaker fdp run python -m labeler.events.ui` prints a
   token link; the link opens the page over an SSH forward.
2. A shot not in the corpus fetches, writes
   `foundation_model/verify_cache/<shot>_processed.h5`, and draws three
   crosspower panels plus the label row over the whole discharge.
3. Reopening that shot is instant and touches no network.
4. Panning and zooming re-renders at the visible window's resolution.
5. Box-select, *Mark present*, *Verify*, *Save* writes
   `review/<shot>__<reviewer>__<stamp>.csv` and a `shots.csv` row; a second
   save writes a second file and destroys nothing.

The remaining fifteen events are registered and expected to render their
generic panels. They are not part of acceptance.

## Out of scope

Eviction for `verify_cache/`. Any change to what `tier` or `holdout` mean.
Merging `review/` rows into `format/`, which stays a manual step. Public
network exposure -- the server binds the loopback and is reached by SSH
forward, as `shot_design` is.
