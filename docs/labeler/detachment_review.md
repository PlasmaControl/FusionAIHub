# Detachment camera review

The review page offers **1 attached, 2 detached, 3 MARFE, 4 uncertain** on an
exclusive state track. Unmarked time is unassessed. The producer label model is
an **unverified suggestion**, with its fallback rule and three indicator votes
shown directly below the video. Review edits remain in the usual annotation
lanes; the Source lane is seeded from a nonblind snapshot of the producer label.
Binary agreement scoring refuses this multiclass editor.

## Producer definitions and interpretation

These meanings are copied **verbatim** from the producer's
`docs/labeler/detachment.md`:

| Code | State | Meaning |
| --- | --- | --- |
| 0 | absent | fewer than two indicators were valid: nothing can be said |
| 1 | attached | the strike point carries the full heat and particle flux |
| 2 | detached | the radiating front has left the plate (partial or full) |
| 3 | marfe | the front has moved above the X-point onto the confined plasma |
| 4 | uncertain | indicators disagree, or all valid ones sit in a transition band |

MARFE is physically detached but has a distinct stage in this exclusive track.
Partial and full detachment share code 2. Missing evidence never means attached.
The votes and label states are read directly; the page does not reclassify them.

**Afrac = 1/DOD:** high Afrac **≥0.75 supports attached**, falling Afrac
**≤0.5 supports detached**; 0.5–0.75 is a valid abstention. The trace draws both
threshold lines. This producer decodes uncalibrated swept probes without probe
positions and references the shot to its own 0.90 quantile. A shot detached
throughout can be called attached in its top tail. This is a weak, uncalibrated
review aid that cannot establish a label alone; the caveat is visible beside
its trace and in the video help.

Prad,div is calibrated lower-divertor radiation divided by heating power:
≤0.35 attached, ≥0.50 detached, otherwise abstain. It is a radiation measure
and never votes MARFE. TangTV normalized front DZ is attached below 0.35,
detached from 0.5 through 1.0, and MARFE above 1.0, subject to the producer's
geometry/validity gate. **Tomographic inversion** and **surrogate regression
(model estimate)** are named in the row title and preserved per bin. A
surrogate is never presented as a measurement. Legacy height CSVs without
provenance explicitly lack a recorded source.

The strips show the primary label model, fallback rule, Afrac, Prad,div and
TangTV votes on exact half-open bins. Codes 1–4 have the same colours as the
annotation lane. Blank label bins are unassessed; grey votes are **valid
abstentions** (`valid=True, vote=-1`); hatched votes are **invalid**
(`valid=False, vote=-1`). The reading at the committed camera time gives each
vote's validity/reason and the TangTV source. Published labels are aligned to
the producer NPZ grid, preserving votes where fewer than two indicators were
valid. Missing assessed rows within a published shot, or disagreement with its
vote snapshot, fail explicitly. The producer also requires at least 20 jointly
assessed bins per shot before publishing labels. Shots with no published label
retain their votes, blank primary/rule strips and a **label not published**
readout; their bins are not called unassessed. No interpolation or threshold
fitting is performed.

An inconsistent published CSV/NPZ pair is excluded from the Source snapshot,
with its reason and both hashes recorded. The API also suppresses an obsolete
Source baseline when its store reports that mismatch; saved human reviews
remain available. Camera shots remain queued for independent inspection.

## Video and shared time

Drag **Time (ms)** or click a diagnostic row, strip or axis to seek. Play/Pause
steps through stored previews. Every view, caption, committed clock and strip
cursor updates in one transaction after all selected camera frames decode.
Manual seeking hides old pixels while loading. Playback retains the previous
complete transaction while awaiting new frames. Pause, seek, view changes and
navigation cancel outstanding requests and discard staged frames. Slow delivery
slows playback; at most one active request per camera is allowed. Hover time is
independent of the committed video cursor. The sticky video panel remains
visible while the diagnostic rows scroll.

Corpus clocks are seconds; stores and APIs use milliseconds. TangTV corpus
images are **50 Hz linear resamples blending adjacent exposures**, produced by
`prepare_data.py`'s `interp1d(kind="linear")`; they are not native exposures.
Previews retain exact corpus times, decimated to at most 20 fps inside the
plasma window, with fixed per-shot/channel 1st–99.5th percentile grayscale.
Area reduction uses one spatial stride for both axes and actual finite counts.
Limits are bolo 80×120, TangTV 240×720 and IRTV 256×320. `/api/frame` reads one
PNG; ties choose the earlier frame. Pixels are never embedded in shot JSON.
Flattened IRTV arrays and trace-shaped bolometer arrays receive explicit
unavailable cards; no image geometry is invented.

Camera identities follow active `input_key` order in
`src/tokamak_foundation_model/data/config/modalities/modalities.yaml`.
**Default TangTV channel 2**, the producer's lower-divertor perpendicular view,
then channel 0 if 2 is unavailable. Other available views remain selectable.
Filter/emission line is not recorded; PAR/PERP are node names.

| TangTV channel | View node | Region |
| --- | --- | --- |
| 0 | LODIV_240RM1:PAR:INTENSIFIED | Lower divertor |
| 1 | LODIV_240RM1:PAR:STANDARD | Lower divertor |
| 2 | LODIV_240RM1:PERP:STANDARD | Lower divertor |
| 3 | UPDIV_225RP1:PERP:STANDARD | Upper divertor |
| 4 | UPDIV_0RP1:PERP:STANDARD | Upper divertor |
| 5 | UPDIV_225RP1:PAR:STANDARD | Upper divertor |
| 6 | UPDIV_0RP1:PAR:STANDARD | Upper divertor |

TangTV nodes use prefix `\TANGTV::TOP.TANGTV:` and suffix `:VIDEO_IMAGES`.
IRTV has six active nodes, despite seven declared output slots: BIAS_105RM1,
LOCEN_315RM1, LODIV_165RP2, LODIV_60RP2, UPCEN_300RP1, UPDIV_225RM2. Slot 6 is
unmapped. PERI75R0 is commented out and has no active index; it must not shift
UPCEN/UPDIV. IRTV suffix is `:DIGITAL_CAM:DIGITAL_RAW`.

## Diagnostic context and geometry

Density priority is **producer `aux_ne` → cached `denr0uf` → corpus CO2 →
local Thomson core density**. This provides the producer's CO2 upstream-density
context before the local fallback. The producer's `signals.line_density` calls
its CO2 input a **line integral in arbitrary/native units** and selects V2,
then R0, then V3; bins do not record the chosen chord. No chord-length division
or verified UF unit is available. CO2 rows therefore say **density proxy,
native units (unverified)**; calling these a calibrated line average in cm⁻³
would exceed the source evidence. Local Thomson is explicitly not
line-averaged and retains m⁻³ units. **All ne ≤0 is missing before block means**;
Thomson channels are ranked by positive native-sample valid fraction (up to eight,
stable channel-order ties). No local trace is renamed as a line average.

Divertor Thomson Te (`aux_te_div`, eV, per-bin peak over channel medians) is shown
as an **independent check**, not a fourth voter; zero is a failed fit and is
missing. Filterscopes FS01–FS08 have separate scales and calibrated
**ph/(sr cm² s)** units from the local SPECTROSCOPY source and
`configs/shot_design/signals.yaml` (`dalpha`). Chord locations are not recorded.
Dead/nonpositive-median chords are omitted by an explicit availability screen:
retain finite chords with positive median block mean in the displayed window.
This screen does not assert a noise-floor calibration. Gas channels GASA–GASE,
LOB1–LOB2, PFX1–PFX3 and UOB use Torr L/s. Raw probe-sweep medians and raw
bolometer-voltage medians are omitted.

Context uses contiguous native-sample block means, approximately 1 ms (actual
width is recorded, e.g. 1.024 ms for a 256 µs clock), clipped to the catalog
plasma window. For external producer shots, the cached |Ip| ≥300 kA first/last
samples supply the producer's window. Missing windows never imply attachment.

The shelf gate copies the producer's real RVSOD/ZVSOD/RXPT1/ZXPT1 bounds,
excluding EFIT sentinels -0.89/-9.99/0: ZXPT1 <−0.5 m, RVSOD ≥1.37 m,
|ZVSOD+1.25 m| ≤0.05 m, nearest EFIT within 40 ms. It is **separate from
magnetic topology**. XPT1 is the lower point, XPT2 the upper. Valid DRSEP <−1 cm
names LSN, >+1 cm USN; a balanced ±1 cm interval names DN only with both
opposite X-points. Saturated DRSEP is not balance evidence. Without DRSEP the
page shows only the shelf gate and says topology unavailable. Unknown
configuration is a missing value (`null`), never a physical state.

## Read-only producer integration and queue ownership

**Only `r4-detach` owns `data/events/detachment/shots.csv`.** This branch restores
that file to its pre-stream `21183f0` version and never writes it. The queue
script reads the producer roster and outputs and writes only the delivery
`--out/tables/detachment/shots.csv` overlay, preserving curation and review
metadata. Producer defaults are the sibling worktree's
`data/events/detachment/shots.csv`, `$LABELER_ROOT/round4/detach/labels_bins.csv.gz`,
`labels_rule.csv` and `bins/`. Paths are configurable with `--producer-roster`,
`--producer-labels`, `--producer-root`, `--producer-tables`, `--label-source`.

The camera-first union uses producer roster/label/vote shots plus live cohort
camera shots. **No-video shots are excluded by default**, with their count in
the scan; `--include-no-video` explicitly flags and places them last. Camera
availability is checked before and independently of the EFIT shelf gate.
Missing/unreadable EFIT counts are reported separately. Fixed cohort test shots
and all input `split=test` or `holdout=true` shots are excluded before corpus
reads. Manual delivery `holdout=true` reservations are also excluded and retained
in `tables/detachment/review/holdouts.json` so regeneration cannot re-admit them.
The CSV scanner reads flags even for filename-based `<shot>.csv` and
`<shot>.csv.gz`; NPZ scalar/per-row flags are checked too. The delivery is
training-facing, not a review-only blind-test queue.

The primary Source lane is frozen to
`tables/detachment/review/suggestions.csv`; `review/source.json` records its
pointer, producer table/hash, `state_lm` column and bin width. Later producer
files require rerunning the script to refresh this snapshot. Inputs are read
only. Resume compares source hashes for labels, bins, density cache and EFIT.
The recipe change invalidates old stores even without force. Retained stores
outside the queue remain inaccessible through this server's roster.

Library defaults require no detachment-specific environment settings for the
normal `$LABELER_ROOT`: indicators `round4/detach/bins` (legacy
`round4/detach/indicators` fallback), geometry/density `round4/detach/cache`,
labels `round4/detach/labels_bins.csv.gz`. Overrides are
`LABELER_DETACHMENT_INDICATORS`, `LABELER_DETACHMENT_GEOMETRY_ROOT`,
`LABELER_DETACHMENT_CACHE_ROOT`, and `LABELER_DETACHMENT_LABELS`. An isolated
server root must explicitly name the producer paths below.

## Reproduction and appendix export

From this worktree:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/detach-ui
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_roster.py \
  --out "$LABELER_ROOT/round4/detach-ui" --build --rebuild-existing --workers 4
# Repeat separately for 190010 and 190102; --verify once runs covering checks.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_demo.py --shot 190212 \
  --out "$LABELER_ROOT/round4/detach-ui/browser-190212" --verify
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_export.py \
  --evidence docs/labeler/results/detachment_ui_fix3_export.json
```

The static appendix figure is `detachment_review_190212_paper.pdf` (vector) and
`.png` (150 dpi), under the delivery root: real channel-2 frames, primary label,
three vote strips and Afrac/radiation/DZ traces share marked frame times. Fonts
are ≥7 pt at the intended **6.75-inch two-column width without further scaling**.
It identifies the surrogate regression and uncalibrated Afrac. Frame timestamps
and pixels are checked against the corpus, and JSON records all sources/hashes.
This replaces the earlier crop that omitted diagnostics and had empty labels.

Launch a review on a free port:

```bash
LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui \
LABELER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui/tables \
LABELER_DETACHMENT_INDICATORS=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach/bins \
LABELER_DETACHMENT_GEOMETRY_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach/cache \
LABELER_DETACHMENT_CACHE_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach/cache \
LABELER_DETACHMENT_LABELS=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach/labels_bins.csv.gz \
LABELER_NO_FETCH=1 pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python -m labeler.events.ui --port 8812
```

Open `#detachment/190212` on the printed token URL. Existing owner servers are
untouched. Browser evidence uses the installed Playwright Chromium through the
repository's DevTools driver; each free-port server stops after checking.
Fresh queue, covering tests/lint, three real-shot browser checks and export
records are in `docs/labeler/results/detachment_*fix3*.json` and
`detachment_review_queue.json`; historical contradictory records were removed.
