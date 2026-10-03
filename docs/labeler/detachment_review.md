# Detachment camera review

The `detachment` editor offers spans **1 attached, 2 detached, 3 marfe,
4 uncertain**. Unmarked time is unassessed. Labels, saves, history and the
timeline lanes work as in the other editors.

Above the timeline, each available camera (`bolo`, `tangtv`, `irtv`) has a
grayscale preview and a channel selector. Drag **Time (ms)**, or click a time
on the diagnostic rows or axis, to seek the nearest stored frame. The cursor
stays at that time across all timeline panels and label tracks. **Play/Pause**
steps through native frame times; playback stops on navigation. The frame's
actual timestamp is shown separately from the requested time. Outside a
camera's coverage, the nearest endpoint remains visible with an explicit
coverage note. A missing camera, one-sample stub, or all-NaN channel does not
prevent reviewing the other diagnostics.

Corpus movie arrays must be `(channels, times, height, width)` or
`(times, height, width)`, with `xdata` in seconds. Real image arrays are stored
as previews no faster than 20 fps of shot time, with one `uint8` HDF5 chunk
per frame. Bolometer previews are bounded to 80×120, TangTV to 240×720 and
IRTV to 256×320. Each channel uses a fixed per-shot grayscale scale: the
1st–99.5th percentiles of deterministic pixel samples from the retained
frames. The page receives only a manifest and frame clocks; authenticated
`GET /api/frame?event=detachment&shot=...&camera=tangtv&channel=0&t_ms=...`
reads and serves one nearest frame as PNG. Ties choose the earlier frame.

The current corpus's `bolo` is generally **raw bolometer traces**, not images
or calibrated radiated power. It appears as raw context; the video card
explains that no image frames are available. The other optional context rows
are the first eight D-alpha filterscopes, CO2 R0 density, gas-flow channels,
and a raw Langmuir-channel median. These are sampled at 1 ms (Langmuir at
10 ms); pre-shot baselines are omitted. They do not infer detachment or
provide calibrated Afrac/Prad. A movie-only shot still has a timeline; a shot
with no local diagnostics opens an unmarked 10 s timeline.

Optional indicator files can add Afrac, divertor radiated power and TangTV
front height. Put `<shot>.csv` under
`$LABELER_ROOT/round4/detach/indicators/`, or set
`LABELER_DETACHMENT_INDICATORS` to its directory. The explicit interchange
schema is `t_ms`, `afrac`, `afrac_valid`, `prad_div`, `prad_div_valid`,
`tangtv_front_height`, `tangtv_front_height_valid`; values are in milliseconds,
dimensionless Afrac, MW and metres respectively. Only indicators with a
validity column are drawn; rows with validity other than 1 remain gaps. This
preserves geometry gates from indicator extraction. Missing files are fine.
Rebuild the shot with `--force` after indicators or previously missing frames
arrive. Camera stores carry a panel recipe version so old scalar-only
detachment stores rebuild on first open.

To build a local-only store and start a new review server on a free port:

```bash
LABELER_NO_FETCH=1 pixi run --frozen -e labelmaker python -m labeler.events.review.build \
    --event detachment --shots 190010
LABELER_NO_FETCH=1 pixi run --frozen -e labelmaker python -m labeler.events.ui --port 8812
```

The shot must be on `detachment/shots.csv` in the configured label tables.
Opening `#detachment/190010` selects it. Camera controls require server API 9;
restart an older server and reload to enable them. For an isolated real-shot
smoke check, without changing production labels or an existing server:

```bash
LABELER_NO_FETCH=1 pixi run --frozen -e labelmaker python \
    scripts/labeler/detachment_review_demo.py --shot 190010 \
    --out "$LABELER_ROOT/round4/detach-ui"
```

This writes a review store, `demo.json` with source shapes, frame counts and
browser checks, and an actual Chromium screenshot. It uses a temporary
loopback server and stops it afterwards; an existing headless Chromium and
Node are required. Set `TMPDIR` to your scratch temporary directory.

