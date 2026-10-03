# Detachment camera review

The `detachment` editor offers **1 attached, 2 detached, 3 marfe, 4 uncertain**.
Unmarked time is unassessed. This is an exclusive state track: MARFE takes its
own state even though MARFE and divertor detachment can physically coexist.
There is no partial-detachment state. Do not interpret these labels as a
complete taxonomy of divertor physics; the paper must state these limits.
Detached uses blue and uncertain orange (distinct colour-blind-safe hues).
Binary agreement
scoring refuses both multiclass editors.

## Operational state definitions

The target is the **lower outer divertor leg**, interpreted with EFIT magnetic
configuration and the outer strike point. The page help contains these same
definitions. Upper-single-null (USN), double-null (DN), limited or unknown
geometry can make a lower-camera diagnosis unsuitable; inspect the shown
configuration and gate at the cursor time and use uncertain when needed.

| State | Visual and supporting indicator evidence |
| --- | --- |
| Attached | Emission concentrated near the lower outer target/strike point, without a sustained upstream front. Low valid Afrac and low upstream radiation/front displacement support attachment; target ion saturation current supports it when available. |
| Detached | Sustained emission/radiation front lifts from the target along the lower outer leg, with reduced target emission and, when available, reduced target ion saturation current. Increasing valid Afrac, lower-divertor radiation fraction and normalized TangTV front displacement support detachment. Partial lifting and pronounced lifting toward the X-point share this state; no depth grade is encoded. |
| MARFE | Localized radiating blob near the X-point or inner wall, often associated with high density and a density-limit precursor. Distinguish the localized shape from an extended detached leg and seek corroborating radiation/density evidence. A bright blob alone does not establish a density limit. |
| Uncertain | Missing/saturated images, invalid indicators, unsuitable/unknown geometry, transitions or conflicting evidence prevent a reliable state. Missing evidence does not establish attachment. |

The exclusive track cannot represent coexisting MARFE and detachment. Use
uncertain when choosing one state would conceal material conflicting evidence.
Producer drafts and indicators assist review; these are operational definitions,
not new fitted thresholds or a claim that image brightness alone measures Isat.

The sticky video panel stays visible while scrolling the diagnostics. Drag
**Time (ms)** or click the diagnostic rows/axis to seek. The requested time
pins a cursor after delivery; a separate hover cursor still reads other times.
Manual seeking shows **Loading** and hides old pixels until decoding finishes.
During playback the previous committed views stay visible while all cameras
decode into staging objects. One synchronous update publishes every image,
caption and the shared clock/cursor. **Play/Pause** waits for all selected
cameras with at most one active request per camera. Pausing, seeking, changing
views or navigating aborts pending requests and discards staged pixels/URLs.
Slow delivery slows playback. Navigation cancels requests and stops playback.
Outside coverage, the nearest endpoint has an explicit note; unavailable
cameras never prevent annotation.

## Physical camera identities

Names and source nodes follow the active `input_key` order in
`src/tokamak_foundation_model/data/config/modalities/modalities.yaml`. The
manifest includes unavailable views as well as live channels; selectors name
the view, region and availability. The frame caption/alt text identifies the
delivered view. **Filter/emission line is not recorded in the corpus**; PAR
and PERP are node names, not inferred spectral filters.

TangTV nodes have prefix `\TANGTV::TOP.TANGTV:` and suffix `:VIDEO_IMAGES`:

| Index | View node | Region |
| --- | --- | --- |
| 0 | LODIV_240RM1:PAR:INTENSIFIED | Lower divertor |
| 1 | LODIV_240RM1:PAR:STANDARD | Lower divertor |
| 2 | LODIV_240RM1:PERP:STANDARD | Lower divertor |
| 3 | UPDIV_225RP1:PERP:STANDARD | Upper divertor |
| 4 | UPDIV_0RP1:PERP:STANDARD | Upper divertor |
| 5 | UPDIV_225RP1:PAR:STANDARD | Upper divertor |
| 6 | UPDIV_0RP1:PAR:STANDARD | Upper divertor |

Default to live channel 0, then live channel 2. The review queue requires at
least one of these lower-divertor views. Other shots can still open their
available views, explicitly named; an upper-divertor image is not evidence
about the lower divertor.

IRTV nodes have prefix `\IRTV::TOP.IRTV:` and suffix
`:DIGITAL_CAM:DIGITAL_RAW`:

| Active index | View node | Region |
| --- | --- | --- |
| 0 | BIAS_105RM1 | Bias view |
| 1 | LOCEN_315RM1 | Lower central |
| 2 | LODIV_165RP2 | Lower divertor |
| 3 | LODIV_60RP2 | Lower divertor |
| 4 | UPCEN_300RP1 | Upper central |
| 5 | UPDIV_225RM2 | Upper divertor |
| 6 | Unmapped padded slot | Unknown |
| Commented out | PERI75R0 | Inactive; no index assigned |

The YAML declares seven IRTV output slots but lists only six active nodes.
The corpus preparation code preserves active list order and pads extra slots.
Do not insert the commented PERI75R0 node at index 4 and shift the other names.
The manifest records its full inactive node and the mapping caveat. A future
source with different ordering needs explicit provenance before relabelling.

## Previews and diagnostic context

Corpus image arrays are `(C,T,H,W)` or `(T,H,W)`, with seconds in `xdata`.
TangTV corpus images are **50 Hz linearly resampled frames**, a blend of adjacent
exposures produced by `prepare_data.py`'s `interp1d(kind="linear")`. They are not
individual exposures. Store their corpus sample times in milliseconds and
decimate previews to no more than 20 fps, within the plasma window. One spatial
stride (the larger of the strides needed by either axis) is used for both
axes, with area means before quantization. This preserves IRTV pixel aspect.
Limits are bolo 80×120, TangTV 240×720 and IRTV 256×320. Grayscale uses fixed
per-shot/channel 1st–99.5th percentiles of deterministic pixel samples from
retained frames. Authenticated `/api/frame` reads one PNG; ties choose earlier
frames. Images are never embedded in the shot JSON.

Some real IRTV arrays are flattened `(C,T,N,1)` records. Their two-dimensional
image geometry is unavailable, so they receive an explicit unavailable card;
no height/width is guessed from N. Area reduction uses actual pixels and
finite counts without padding very narrow arrays to enormous squares.

Context traces are clipped to the cohort's plasma window (population window
as fallback). For external producer shots without a catalogued window, use the
first/last cached samples with |Ip| ≥ 300 kA, exactly the producer's fixed
`detach_bins.py` rule. The chosen source and boundaries are recorded in store
metadata. The review grid uses those exact boundaries; diagnostic tails do not
extend the default view. With no valid catalog or cached-current window, use
nonnegative context and available movie coverage. A wholly missing shot gets a
blank 10 s track.
Context uses contiguous block means of every native sample, with approximately
1 ms blocks, timestamped at the mean sample time. Actual block width is stored
in panel metadata: a 256 µs native clock produces **1.024 ms**, not 1 ms.

Legends name FS01–FS08 with calibrated photon-radiance units
**ph/(sr cm² s)**, as recorded by the local raw spectroscopy cache and
`configs/shot_design/signals.yaml`. Every chord has its own row and scale,
so divertor chords are never overlaid with upper/midplane chords. The local
corpus/raw caches do not record a trustworthy FS01–FS08 sightline-location
map; the page says **location not recorded**, rather than guessing a divertor
assignment. An authoritative shot-dependent chord map remains needed to name
those locations. Density uses CO2 R0 DENUF (cm^-3, line-averaged) first, then
Thomson core **local density (not line-averaged)** in m^-3 when CO2 is a stub or
has no finite plasma-window samples. The fallback preserves separate sampled
core channels; no local signal is relabelled as a line average. Gas legends name
GASA, GASB, GASC, GASD, GASE, LOB1,
LOB2, PFX1, PFX2, PFX3, UOB (Torr L/s). No arbitrary channel numbers or
"corpus units" are used. **Raw TPLANG sweeps are omitted**: averaging their
cross-probe median would still not isolate ion saturation current. **Raw
bolometer medians are omitted**: the 48 voltage chords neither share a useful
radiation median nor supply calibrated divertor power. The bolo video card
explains that these corpus arrays are traces, not tomographic images. Use the
producer's validity-gated indicators for calibrated diagnostic evidence.

## Producer indicators and roster

Set `LABELER_DETACHMENT_INDICATORS` to the producer directory. The portable
library default is `$LABELER_ROOT/indicators/detachment`; round paths belong
in run configuration. The detach stream currently emits
`$LABELER_ROOT/round4/detach/bins/<shot>.npz` through `detach_bins.py`. Its
handoff still describes eventual labels/posteriors under `labels/`, pending.
The adapter reads the files that exist now, with `allow_pickle=False`:

| Field | Meaning shown | Unit |
| --- | --- | --- |
| start_ms | Bin start; width inferred from median diff(start_ms), including 20/50/100 ms grids | ms |
| afrac_value | Afrac | dimensionless |
| prad_value | Prad_divL / P_in radiation fraction | dimensionless |
| tangtv_value | Normalized emission-front DZ | dimensionless |
| *_valid | Required producer validity gate; invalid values become gaps | boolean |
| *_reason, *_vote | Preserved in panel metadata, without changing decisions | producer codes |

This schema is also accepted as CSV. It is not the MW/metre CSV schema.
Bins are drawn as steps over their full start/end intervals, clipped to the
plasma window. Invalid bins remain gaps, and isolated valid bins remain visible;
values are never linearly interpolated between indicator bins. A single start
without an explicit second start cannot establish bin width and is rejected.
The original interchange CSV remains a fallback: `t_ms`, `afrac`,
`afrac_valid`, `prad_div` (MW), `prad_div_valid`, `tangtv_front_height` (m),
`tangtv_front_height_valid`. Missing files, missing validity columns, all-invalid
signals and missing cameras are acceptable. Force-rebuild when new producer
files arrive; validity is never inferred from a value or a vote.

`data/events/detachment/shots.csv` is the real review queue, without placeholder
rows. The committed `detachment_review_roster.py` writes the union of live
lower-divertor TangTV train/validation cohort shots passing the producer's EFIT
lower-null/outer-strike-point shelf gate and every producer-labelled shot that
is not in the blind test split. It snapshots the producer's current outputs
under both the detach stream's output root and its worktree tables; rerunning
at integration incorporates newly written labels. Blind test shots are excluded
before corpus reads. Existing review metadata is preserved when rerunning.
The isolated server consumes a copy at `tables/detachment/shots.csv`.

The gate uses real EFIT RVSOD/ZVSOD/RXPT1/ZXPT1, excluding -0.89/-9.99/0
sentinels; primary ZXPT1 < -0.5 m, RVSOD ≥ 1.37 m and
|ZVSOD + 1.25 m| ≤ 0.05 m. EFIT is matched within 40 ms. This reproduces the
producer gate rather than tuning a new detachment threshold. Configuration
is shown separately at the cursor: LSN/USN from an unambiguous primary X-point,
DN only with separatrix-balance evidence. Opposite X-points without DRSEP and
missing points without verified limiter status stay **unknown**; missing points
are not labelled limited. Producer-labelled shots may fail the camera/geometry
gate and remain in the queue for honest uncertain/unavailable review.
Set `LABELER_DETACHMENT_GEOMETRY_ROOT` to the local producer cache; the portable
default is `$LABELER_ROOT/indicators/detachment`. No production store is written.

### Fix-round coverage snapshot

The final local snapshot at **2026-10-03 15:01:35 UTC**, written by the committed
roster script, is recorded in
[`results/detachment_review_queue.json`](results/detachment_review_queue.json).
It scans **450 train/validation** cohort shots and excludes **50 blind test**
shots before reading their corpus. **21** have live lower TangTV at a
producer-gate-valid time. The producer already has **202** nonblind label/vote
shots, including those 21, so the union is **202**: **141 train, 20 validation,
41 producer-external**. Producer-external shots are outside the fixed cohort;
they carry no invented cohort split. The record lists every included/excluded
shot and each snapshotted producer file/hash. The producer is still running;
the controller must rerun the script at integration.

Density now covers **226 of the original 229 camera candidates**: **113 CO2,
113 Thomson, 3 unavailable**. In the current queue, **191 of 202** have density:
**10 CO2, 181 Thomson, 11 unavailable**. All **394** repaired isolated stores
(current queue plus retained earlier nonblind candidates) are audited:
**123 CO2, 257 Thomson, 14 unavailable**. No aux_ne substitution is used.
Every store has a catalog or producer-current plasma window; the audit reports
**zero** frames outside either plasma or store bounds. The detailed per-shot
row/channel/window audit is
`$LABELER_ROOT/round4/detach-ui/roster_build.json`. Counts describe display
coverage, not model performance or a detachment prevalence estimate.

From this worktree, using the shared installed environment and scratch TMPDIR:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/detach-ui
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_roster.py \
  --out "$LABELER_ROOT/round4/detach-ui" --build --rebuild-existing --workers 4
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_demo.py --shot 190212 \
  --capture-time-ms 600 \
  --out "$LABELER_ROOT/round4/detach-ui" --verify
```

The demo uses a neutral `Reviewer` name, delays frame delivery by 180 ms,
asserts decoded-frame consistency and bounded playback, captures an actual
Chromium screenshot of a lower-divertor view, and stops its temporary free-port
server. JSON records include source hashes and command output. Existing owner
servers are untouched. A separate real two-camera browser regression holds
IRTV delivery after TangTV decoding and pauses between them; it verifies that
both images/captions/clock retain their prior time, then advance together on
resume. The demo also exports a cropped live-page PDF and 150-dpi PNG at
3.25-inch column width with readable camera identity, timestamp, cursor and
state controls, retaining the desktop evidence screenshot. To open the scanned
stores on a free port:

```bash
LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui \
LABELER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui/tables \
LABELER_DETACHMENT_INDICATORS=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach/bins \
LABELER_NO_FETCH=1 pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python -m labeler.events.ui --port 8812
```

Open `#detachment/190212` on the printed token URL. Camera controls require
server API 9. Rebuild after indicator delivery; the store recipe version also
invalidates earlier scalar/stride previews.
