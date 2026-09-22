# Label review rebuild, IGNITE v4 only, pixi tidy

2026-09-22. Supersedes `2026-09-18-verification-ui-design.md` for the review surface.
Amended the same day while planning (`plans/2026-09-22-label-review-v4-only-pixi.md`): store
dataset names and attrs, the wheel, the source glob, the Stellar Simulate script.
Three independent parts, each shippable alone: **0** (IGNITE v4 only), **A** (label review),
**E** (pixi). IGNITE retraining, the shot_design flow and the paper get their own specs once the
open decisions in the hand-off message are made.

Decisions already taken by the owner (2026-09-22): checkpoint commit first (done, 88177e1);
re-encode production with v4 and move the v2 codes aside (in progress); label store starts
fresh, the eight earlier saves move to `review/legacy/`; the stale pixi environments `ideate`,
`ideate-cpu` and `rocm` are deleted (done).

---

## Part 0 -- IGNITE v4 is the only generation

### Facts

- v4 bundle on Stellar: `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE_v4`
  (sha256-pinned manifest; Hub `nc1/IGNITE-v4` @ `d2f12b82`). 15 codecs x 1000 codes,
  1209 tokens per frame, frame 0 at 1.0 s, 219 frames per shot.
- Production frame codes: the v2 caches are in `ideate/frame_codes_v2_retired/` (500 shots).
  `ideate/frame_codes/` is being refilled with v4 CPU encodes: pilot 2938821 (4 shots, 58 s/shot)
  and array 2938824 (the 500 `recommender_v1` shots).
- The only reason v2 survived is `SHOT_DESIGN_IGNITE_GENERATION=v2`, exported by
  `scripts/shot_design/_stellar_common.sh` because Stellar was thought to lack v4. It does not.

### Changes

1. `configs/shot_design/ignite_modalities.yaml`: delete `model_generations`; `model:` is the only
   block. Drop the v2 comparisons from its comments.
2. `shotdb/ignite.py`: `model_cfg()` returns `model:`; delete `_select_model_cfg` and the
   environment variable. Every `.get("generation", "v2")` and `.get("t0_start_s", 0.0)` becomes a
   plain `cfg[...]` read. The v2-only `download_bundle` branch goes; `--download` keeps following
   `repo_id` for v4.
3. Same purge in `cli.py`, `simulate/cli.py`, `shotdb/build.py`, `design/seed.py`,
   `design/program_reference.py`, `design/assistant.py`. `simulation.h5` keeps writing
   `frame_origin_s`; readers require it instead of inferring an origin from `codec_generation`.
4. Design defaults follow v4: the earliest window start is 2.0 s (20 history frames), and
   `n_predict` is derived from the seed frames and the 100-row `frame_embed` (at most 80).
5. Stellar Simulate: `_stellar_common.sh` stops exporting a generation. The UI's Simulate button
   submits a new single-design `scripts/shot_design/simulate.sbatch <ident>` on Stellar (one A100,
   the Stellar twin of `scripts/slurm_frontier/shot_design_simulate.sh`), through a per-cluster
   submit command kept beside `paths.yaml` / `paths.frontier.yaml`. Not `simulate_batch.sbatch`:
   that one sources `_stellar_common.sh`, which refuses the production root the UI runs against.
6. `scripts/shot_design/g_enc.py`: keep the v4 gate; delete the v2 historical record.
7. Tests: rewrite the fixtures in the v4 frame (origin 1.0 s) and delete the conftest pin that
   forces `t0_start_s = 0.0`; delete `test_ignite_generation_override.py`; the shipped-cache test
   in `test_seed.py` compares against `model.frame_codes_cache` (v4 ships no `frame_codes/`).
   These two are today's only shot_design failures (1834 passed, 2 failed).
8. Docs and paper: remove the v2 instructions from `docs/clusters/{stellar,frontier}.md`,
   `docs/shot-design/simulation.md`, `docs/models/ignite.md`, `docs/data/corpus.md`,
   `dev/paper/README.md` and `dev/paper/figures/make_figures.py`.
9. Production data still built from v2 (found by the 2026-09-22 audit), refreshed once the
   re-encode finishes: `db/shots.parquet`'s `has_frame_codes` (built 2026-09-14 from the v2
   codes), `ignite_inputs/reference_cache/199597.*` (re-made from the v4 bundle), and the v2
   seeds in `ignite_inputs/designs/{9c615cc0,9cf68ecd,b92784a6}*.pt` and
   `outputs/demo_examples/*.pt` (moved to a `_v2_retired/` sibling).
10. Decode fix (same audit, bug B9): the spectrogram reduction takes `np.abs` of signed,
    standardised log-power, which folds BES, CO2 and Mirnov values below the mean onto values
    above it. Average the signed values in `simulate/decode.py:88`,
    `dev/paper/figures/decode_all_v4.py:31` and `dev/paper/figures/make_figures.py:161`; relabel
    report.py's "band-power" as mean z; and apply the `frac_static` filter to both arms, not just
    the proposed one.
11. Data, last and only after the paper figures read v4 roots: rename the v2 experiment roots
    (`experiments/showcase4`, `experiments/stellar_1k`) and `models/IGNITE` to `*_v2_retired`.

### Done when

`git grep -nE 'model_generations|IGNITE_GENERATION|"v2"'` over `src/shot_design
scripts/shot_design configs/shot_design tests/shot_design docs dev/paper` finds nothing; both
suites pass under `-W error`; one Stellar Simulate from the UI finishes on one A100 with v4, and
its `simulation.h5` carries `frame_origin_s = 1.0`.

---

## Part A -- label review

### What the reviewer gets

- One label per shot. Saving replaces it, and it stays on screen, marked as saved.
- Reopening the page lands on the first unreviewed shot after the last one saved.
- A split screen: spectrogram rows on top (scrolls), the label editor and the queue below.
- AE shots open in under a second: their spectrograms are computed ahead of time.
- Almost no prose on the page. `?` shows the keys.

### Label store

`data/events/<event>/review/`:

| file | contents | written |
|---|---|---|
| `labels.csv` | current label of every reviewed shot, in the format schema: `shot, category, t_start, t_end, confidence` (ms). A shot's rows tile its reviewed window: each span with its category, the gaps as category 0; `confidence` blank. | rewritten atomically on each save; only that shot's rows change |
| `history.jsonl` | one line per save: `shot, reviewer, saved_at, window, intervals, source` | appended |
| `legacy/` | the eight per-save CSVs from 2026-09-22, untouched | once, by hand |

- **Source label**: the newest `format/*_format_*.csv` by name (AE: Heidbrink 2026_v1, 0-2000 ms;
  RWM's table is `rwm_format_2026_v1.csv`, hence no event prefix in the glob). Rows shorter than
  1 ms are ignored: a point event has no span to edit. A shot with no saved label opens on its
  source label; an event with no format CSV opens on an empty label over the whole record.
- **States**: *unreviewed* (no rows in `labels.csv`), *confirmed* (saved spans and window equal
  the source's), *changed* (they differ).
- **Spans carry a category.** Binary events have one (present). `minimum_safety_factor` has four
  (low, hybrid, elevated, high; `interval_tables.category_labels`), and number keys set the
  selected span's category.
- **Rules**: span edges snap to whole ms; overlapping or touching spans merge; spans are clipped
  to the window. The window starts as the source window, grows to cover any span drawn outside
  it, and its edges can be dragged. Outside the window is unknown, not absent.
- **Resume**: the first unreviewed shot, in roster order, after the shot of the newest
  `history.jsonl` line (wrapping).
- **Reviewer**: the user the server runs as. No reviewer field in the page.
- `shots.csv` stays hand-curated (`tier`, `holdout`). The server reads it for roster order and
  tier, never writes it.
- `labels.csv` passes `interval_tables.validate_intervals` like any format table, so code that
  reads format tables reads reviewed labels without a new reader.

### Spectrogram store

`$LABELER_ROOT/spectrograms/<event>/<shot>.h5`, `LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`.

- A file holds rows on one time grid. An **image** row is `rows/<name>/{1,8,64}` (the dataset
  name is the pooling factor): uint8 `(n_y, ceil(n/level))`, the time axis max-pooled per level,
  chunked `(n_y, 512)`, gzip level 1. A **trace** row stores float32 min and max per level
  instead, `(2, n_channels, ceil(n/level))`.
- File attrs: the grid `t0_ms` (left edge of column 0), `dt_ms`, `n`; `rows` (JSON list, the
  display order); `event, shot, builder, params` (JSON), `source` (JSON: tier, path, size,
  mtime_ns), `made_at, git_sha`.
- Row attr `meta` (JSON): `kind`, `title`, and for images `n_y, y0, dy, y_units, z_lo, z_hi,
  z_units` plus an optional `band` (80-250 kHz for AE); for traces `n_channels, y_units, legend,
  hlines`.

**AE builder** -- all 180 roster shots, prebuilt by one CPU job on `pppl` (jobstats-gated):

- Input: `$LABELER_ROOT/raw/<shot>_processed.h5`, group `co2` (4 chords R0, V1, V2, V3;
  1.667 MHz; -1.5 to 7.5 s). All 180 are on disk (56 GB, group `kolemen`).
- `resample_poly(3, 10)` to 500 kHz; Hann STFT, `nperseg=512`, hop 128: columns 0.256 ms apart,
  257 bins over 0-250 kHz at 0.98 kHz. This is the Heidbrink recipe.
- Rows 1-4, power per chord: dB minus its per-frequency median over 0-6 s, mapped from
  [-3, +27] dB to 0-255.
- Rows 5-7, cross-power R0xV1, R0xV2, R0xV3: `|moving complex mean over 8 columns (2 ms) of
  S_R0 * conj(S_Vk)|`, then the same flattening and mapping. A mode coherent across chords
  survives the complex mean; noise local to one chord averages away.
- Measured on 170790: 12 s to compute, 38 MB with the pyramid, 15 ms to read a 4,000-column
  window plus all seven overviews. 180 shots is about 7 GB.

**Other events** (sawtooth, fishbone, q_min, the generic twelve): the server builds the file on
first open from the event's existing panel builder and caches it. Heatmap panels become image
rows (quantised between their 1st and 99.5th percentile unless the panel pins `zmin`/`zmax`);
line panels become trace rows, resampled onto a uniform grid at their median spacing.

`raw.DEFAULT_RAW_CACHE` becomes `$LABELER_ROOT/raw`, so a live fetch lands next to the 180 AE
shots instead of in the repository's `.cache/raw`; `raw.promote` and `raw.clean` go with it.

### Server

FastAPI on the loopback behind the existing token and cookie gate; `serve.py` is kept.

| route | returns |
|---|---|
| `GET /api/events` | `[{event, n_shots, n_reviewed, categories}]` (a bad roster answers `{event, error}`) |
| `GET /api/queue?event` | `{shots: [{shot, tier, state, saved_at}], resume}` |
| `GET /api/shot?event&shot` | row metadata, data time range, window, source spans, saved spans, last save; `202` while a non-AE file is being built |
| `GET /api/rows?event&shot&t0&t1&cols` | binary: image rows as uint8 `n_y x n`, trace rows as float32 `2 x C x n`, in row order; header `X-Grid: {"t0","t1","n"}` |
| `POST /api/label` | body `{event, shot, window, intervals}`; normalises, writes, returns `{row, saved, last_save}` (`row` is the queue row) |

`/api/rows` takes the coarsest level that still has at least `cols` columns over `[t0, t1]`,
slices it and pools it down to at most `cols` (max for images, min/max for traces). When zoomed
in past level 1 it returns fewer than `cols` columns and the page scales them up without
smoothing.

Removed from the current app: `/api/panels` (plotly JSON), `/api/save` (per-save CSVs),
`/api/progress`, `/vendor/plotly.min.js`, and the prose `GUIDANCE` strings.

### Page

```
+-------------------------------------------------------------------+
| 170790   unverified   changed, saved 14:02                    ?   |  one-line header
+-------------------------------------------------------------------+
| R0      [canvas: spectrogram, label spans overlaid]               |  top pane,
| V1                                                                |  scrolls
| V2                                                                |
| V3                                                                |
| R0xV1                                                             |
| R0xV2                                                             |
| R0xV3                                                             |
| time (ms) axis, sticky                                            |
+-------------------------------------------------------------------+
| source   [===]        [==]                                        |  bottom pane
| label    [=====]      [==]            |<- window ->|              |
| queue    chips for all 180 shots, coloured by state               |
|          Save and next (Enter)   Revert (R)                       |
+-------------------------------------------------------------------+
```

- One shared time axis. A plain wheel scrolls the rows pane; Ctrl/⌘ + wheel (or any wheel over
  the axis or the bottom tracks) zooms at the cursor; a horizontal wheel pans; drag pans;
  double-click fits the window.
- Label track: drag on empty space adds a span, drag an edge resizes, drag a span moves it,
  Delete removes the selected span. Shift-drag on any spectrogram row also adds a span.
- Keys: Enter save and next, S save, R revert to source, J/K previous/next shot, U next
  unreviewed, `[` `]` contrast, `?` keys.
- Unsaved edits show a dot on the header and the shot's chip, and survive a reload through
  `localStorage` until saved or reverted.
- Colour: one perceptually uniform 256-entry map applied in the page; contrast moves the map's
  limits without refetching.
- Canvas rendering throughout; no plotting library.
- Words on the page: row titles, axis units, state names, button labels and the key legend.

### Tests

- Label rules: snapping, merging, clipping, window growth; `labels.csv` replaces one shot's rows
  atomically and validates as a format table; `history.jsonl` appends; queue states and resume.
- Store: level choice, pooling at chunk and window edges, zoom past level 0, trace min/max.
- AE builder on a synthetic four-chord record: a 120 kHz mode on every chord shows in all seven
  rows; noise on one chord only shows in that chord's power row and not in the cross-power rows.
- API: every route behind the gate, the binary layout against `X-Grid`, the `202` build path.

### Done when

On a laptop through the SSH forward: the page opens on the resume shot; an AE shot paints within
a second; Save and next leaves the saved label visible on the chip and on return; reopening the
page restores the same state; `labels.csv` validates.

### Left as it is (question for the owner)

The sixteen `data/events/*/verification.ipynb` notebooks and the notebook API in `verify.py` are
now a second, unused way to write labels. Removing them touches `data/events/`, so it waits for
the owner's word. Until then they keep working, and `plotly`/`anywidget` stay in `labelmaker`
for them.

---

## Part E -- pixi

1. Activation roots become defaults, `SHOT_DESIGN_DATA_ROOT = "${SHOT_DESIGN_DATA_ROOT:-...}"`
   and the same for `LABELER_ROOT`, `SHOT_DESIGN_CORPUS` and the Frontier set, so a caller's
   root wins over the manifest. This removes the 2026-09-14 incident class (a scratch write sent
   to production). Checked against pixi 0.76.1 before it lands.
2. Tasks with descriptions: `labeler-test`, `shot_design-test` (kept, gains the suite flags),
   `labeler-spectrograms` (build the store), and `labeler-verify`, which runs under `fdp run`
   itself so the one task works for every shot.
3. Each manifest block keeps a one-line comment. The measured rationale (libstdc++, the protobuf
   pin, the sentence-transformers source, activation order on Frontier) moves verbatim to
   `docs/reference/pixi-environments.md`.
4. No environment renames and no re-solve; `pixi.lock` does not change.
