# Detachment camera review

The review page offers **1 attached, 2 detached, 3 MARFE, 4 uncertain** on an
exclusive state track. Unmarked time is not reviewed. The primary producer
suggestion is **unverified**, with its rule comparison and three indicator votes
shown directly below the video. Review edits remain in the usual annotation
lanes. The default Individual lane starts from the primary producer suggestion.
**Start blank** resets selection and clears only Individual/unspecified spans;
Crowd spans and their flags are preserved, and Undo restores the draft.
**Blind mode** hides Source, producer strips, reading and recipe. It does not
clear the initial suggestion: use Start blank separately for an independent
review. Prior exposure and a suggested baseline can bias human/producer agreement;
agreement collected with the default lane is not an independent validation.
Binary agreement scoring refuses this multiclass editor.

## Human camera review protocol

Use the camera together with diagnostics and geometry to mark what you can decide:

| Code | State | Meaning |
| --- | --- | --- |
| 1 | attached | Emission peaked at the target strike point. |
| 2 | detached | Front lifted toward the X-point, with target ion-flux (Afrac/Jsat) rollover or low target Te. |
| 3 | MARFE | Localized radiation inside the separatrix near/above the X-point (high-field side), persisting over several frames. |
| 4 | uncertain | The reviewer cannot decide: view blocked or evidence conflicts. |
| — | unmarked | Not reviewed. |

MARFE is physically detached but has a distinct stage in this exclusive track.
Partial and full detachment share code 2. Missing evidence never means attached.
D-alpha chord locations are not recorded; do not infer a target view from the
channel number. The shelf gate is a producer DZ requirement, not a human
uncertainty rule. The votes and label states are read directly; the page does
not reclassify them.

## What the machine label means

The page help renders the recipe frozen into that shot's store at build time.
`review_recipe.json` in the producer output directory is preferred (override:
`LABELER_DETACHMENT_RECIPE`). Its object is preserved verbatim, including method,
thresholds, definitions and evidence gates. Without that file, the producer's
exported `labels_rule.meta.json` and method record `docs/labeler/detachment.md`
are frozen, with their paths and hashes; the latter includes the definitions,
methods, numeric thresholds, MARFE persistence, spatial/separatrix and independent
corroboration requirements. `LABELER_DETACHMENT_METHOD` can override that record.
The method record resolves inside this repository, never a sibling worktree.
A missing method record raises FileNotFoundError with its path; restore the file
or use the environment override. The bundled method text is a producer-owned
snapshot; the controller brings its final record across when the producer lands.
No UI-authored numeric threshold is inserted into help. Optional guides use only
a structured recipe's `thresholds.<indicator>` numeric entries; without those,
the traces have no classification guides. The physical f_div=1
reference is always drawn, with orange marks above one and a caveat to check the
heating-power denominator and radiation estimate. Afrac text at the cursor
reports the stored bin afrac_method verbatim, without an invented calibration claim.
Read the machine block and each bin's vote/reason together: high DZ alone does
not establish MARFE. A valid abstention can reflect evidence gates as well as a
transition band; the readout preserves the producer reason and does not infer one.

The roster build writes the generated producer wording and hashes to
`--out/producer_recipe.md`. It never rewrites this committed document.

**f_div = Prad,div / P_in** names the lower-divertor radiation fraction throughout
the page and figure. It measures radiation, rather than directly observing
detachment. Page values above one are marked orange, with a physical reference
at one and a denominator/radiation caveat. **Tomographic inversion** and **surrogate regression
(model estimate)** are named in the row title and preserved per bin. A
surrogate is never presented as a measurement. Legacy height CSVs without
provenance explicitly lack a recorded source.

The strips show the primary producer suggestion, rule comparison, Afrac, f_div and
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

Click a camera frame (or press Enter while it is focused) to enlarge that exact
frame and caption; close or Escape returns to the timeline.

Drag **Time (ms)** or click a diagnostic row, strip or axis to seek. Play/Pause
steps through stored previews. Every view, caption, committed clock and strip
cursor updates in one transaction after all selected camera frames decode.
The slider and cursor snap to the selected TangTV frame time (or the first live
camera when TangTV is unavailable); other camera captions retain their own times.
Manual seeking hides old pixels while loading. Playback retains the previous
complete transaction while awaiting new frames. Pause, seek, view changes and
navigation cancel outstanding requests and discard staged frames. Slow delivery
slows playback; at most one active request per camera is allowed. Hover time is
independent of the committed video cursor. The sticky video panel remains
visible while the diagnostic rows scroll. Its height leaves at least one full
diagnostic row and the time axis available at 1366×768 and 1400×900, with
annotation controls below. Unavailable cameras occupy one line. Camera metadata
and producer details are collapsible; expanded details scroll within a bounded
area.

Corpus clocks are seconds; stores and APIs use milliseconds. TangTV corpus
images are **50 Hz linear resamples blending adjacent exposures**, produced by
`prepare_data.py`'s `interp1d(kind="linear")`; they are not native exposures.
New stores retain exact corpus times, with **40 ms preview spacing (25 fps)** on
the 50 Hz grid so every complete 50 ms label bin has a preview when the camera
covers it and its pixels are finite. Frozen stores from earlier rounds have
60 ms spacing (16.7 fps) until the controller rebuilds. Previews stay inside the
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
The filter/emission-line note applies to TangTV only; PAR/PERP are node names.

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
LOB1–LOB2, PFX1–PFX3 and UOB use Torr L/s, after subtracting each channel's
finite native-sample median over **t < 0**, reducing prefill-puff bias. There is
no tuned flatness threshold.
If pre-plasma samples are absent, the legend says **offset uncorrected** and
metadata records a null baseline. Raw probe-sweep medians and raw
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
metadata. Producer roster defaults resolve inside this repository:
`data/events/detachment/shots.csv`, `$LABELER_ROOT/round4/detach/labels_bins.csv.gz`,
`labels_rule.csv` and `bins/`. Paths are configurable with `--producer-roster`,
`--producer-labels`, `--producer-root`, `--producer-tables`, `--label-source`.

The camera-first union uses producer roster/label/vote shots plus live cohort
camera shots. **No-video shots are excluded by default**, with their count in
the scan; `--include-no-video` explicitly flags and places them last. Camera
availability is checked before and independently of the EFIT shelf gate.
Shots with camera/shelf coverage sort first (87 in the frozen scan), then the
remaining camera shots; shot number breaks ties. The human can review either
group. Missing/unreadable EFIT counts are reported separately. Fixed cohort test shots
and all input `split=test` or `holdout=true` shots are excluded before corpus
reads. Manual delivery `holdout=true` reservations are also excluded and retained
in `tables/detachment/review/holdouts.json` so regeneration cannot re-admit them.
The CSV scanner reads flags even for filename-based `<shot>.csv` and
`<shot>.csv.gz`; NPZ scalar/per-row flags are checked too. The delivery is
training-facing, not a review-only blind-test queue.

The primary Source lane is frozen to
`tables/detachment/review/suggestions.csv`; `review/source.json` records its
pointer, producer table/hash, `state_lm` column and bin width. Delivered spans
are clipped to the synchronized plasma window, with original bin start/end
bounds retained in each row's `attrs` provenance. Later producer
files require rerunning the script to refresh this snapshot. Inputs are read
only. Resume compares source hashes for labels, bins, density cache, EFIT and
all interpretation records, including absent recipe files so their appearance
invalidates an older store.
The recipe change invalidates old stores even without force. Retained stores
outside the queue remain inaccessible through this server's roster.

Library defaults require no detachment-specific environment settings for the
normal `$LABELER_ROOT`: indicators `round4/detach/bins` (legacy
`round4/detach/indicators` fallback), geometry/density `round4/detach/cache`,
labels `round4/detach/labels_bins.csv.gz`. Overrides are
`LABELER_DETACHMENT_INDICATORS`, `LABELER_DETACHMENT_GEOMETRY_ROOT`,
`LABELER_DETACHMENT_CACHE_ROOT`, and `LABELER_DETACHMENT_LABELS`. An isolated
server root needs those overrides only when building missing stores; retained
stores contain frozen producer context. Run the controller build before serving
normal review so this server cannot build against an unfinished producer.

## Reproduction and controller resume

The producer is in its own fix round. No delivery stores were rebuilt for fix 5.
After its code, method document and outputs land, run this one command from the
integrated repository (replace `$PWD` only if using a different checkout):

```bash
TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/detach-ui \
LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker \
LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events \
LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 \
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_roster.py \
  --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui \
  --producer-roster /scratch/gpfs/nc1514/FusionAIHub/data/events/detachment/shots.csv \
  --build --rebuild-existing --resume-build --workers 4
```

Resume preserves saved human reviews and rebuilds stores with old panel recipes
or changed producer fingerprints. It regenerates the Source snapshot, queue order
and `--out/producer_recipe.md`. A method-only metadata refresh keeps the physical
f_div=1 reference. No sibling worktree is needed after integration.

For UI-only verification while the producer changes, copy and serve the existing
frozen example. This prohibits every store build and preserves its bytes:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/detach-ui
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_demo.py --shot 200977 \
  --frozen-from "$LABELER_ROOT/round4/detach-ui/browser-fix4-200977" \
  --out "$LABELER_ROOT/round4/detach-ui/browser-fix5-200977" --verify
```

The raw review-page figure is **excluded from the paper**. The producer's views
figure with EFIT separatrix and X-point overlays is the appendix figure. Keep
`browser-fix5-200977/detachment_review_200977.png` (1366×768) as the sole current
review-tool illustration. It shows unverified suggestions, not independent human
validation. The 190212 figure has been deleted and earlier 200977 exports archived.
The optional diagnostic exporter uses `_diagnostic` filenames and adds its
f_div > 1 footnote only when valid plotted bins actually exceed one.

After the resume build, launch on a free port:

```bash
LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui \
LABELER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui/tables \
LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python -m labeler.events.ui --port 8812
```

Open `#detachment/200977` on the printed token URL. Existing owner servers are
untouched. The installed Chromium/DevTools driver checks the actual browser and
stops its own free-port server. Current evidence is in
`docs/labeler/results/detachment_ui_fix5_*.json`; previous snapshots are in
`docs/labeler/results/archive/`. `HANDOFF.md` and the current-state report clearly
separate frozen delivery counts from the pending controller rebuild.
