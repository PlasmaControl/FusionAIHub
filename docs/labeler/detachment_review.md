# Detachment camera review

The `detachment` editor offers **1 attached, 2 detached, 3 marfe, 4 uncertain**.
Unmarked time is unassessed. This is an exclusive state track: MARFE takes its
own state even though MARFE and divertor detachment can physically coexist.
There is no partial-detachment state. Do not interpret these labels as a
complete taxonomy of divertor physics; the paper must state these limits.
Uncertain uses the same orange as the confinement editor. Binary agreement
scoring refuses both multiclass editors.

The sticky video panel stays visible while scrolling the diagnostics. Drag
**Time (ms)** or click the diagnostic rows/axis to seek. The requested time
pins a cursor; a separate hover cursor still reads other times. Each camera
shows **Loading** and hides old pixels until the requested image decodes.
Pixels, displayed view identity, and the frame's actual timestamp then update
together. **Play/Pause** waits for all selected cameras to deliver each step
before advancing the clock, with at most one active request per camera.
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
Store native frame times in milliseconds at no more than 20 fps. One spatial
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
as fallback). The review grid uses those exact boundaries; diagnostic tails
do not extend the default view. With no known window, use nonnegative context
and available movie coverage. A wholly missing shot gets a blank 10 s track.
Context uses contiguous block means of every native sample, with approximately
1 ms blocks, timestamped at the mean sample time. Actual block width is stored
in panel metadata: a 256 µs native clock produces **1.024 ms**, not 1 ms.

Legends name FS01–FS08 (a.u.; calibration units absent from corpus), CO2 R0
DENUF (cm^-3, line-averaged density), and GASA, GASB, GASC, GASD, GASE, LOB1,
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
| start_ms | Start of fixed 50 ms bin; plotted at start + 25 ms | ms |
| afrac_value | Afrac | dimensionless |
| prad_value | Prad_divL / P_in radiation fraction | dimensionless |
| tangtv_value | Normalized emission-front DZ | dimensionless |
| *_valid | Required producer validity gate; invalid values become gaps | boolean |
| *_reason, *_vote | Preserved in panel metadata, without changing decisions | producer codes |

This schema is also accepted as CSV. It is not the MW/metre CSV schema.
The original interchange CSV remains a fallback: `t_ms`, `afrac`,
`afrac_valid`, `prad_div` (MW), `prad_div_valid`, `tangtv_front_height` (m),
`tangtv_front_height_valid`. Missing files, missing validity columns, all-invalid
signals and missing cameras are acceptable. Force-rebuild when new producer
files arrive; validity is never inferred from a value or a vote.

`shots.csv` retains the shared category convention: curation and review
history, not a data-availability list. It is preserved. The committed
`data/events/detachment/shots_review.csv` is a candidate queue with the same
six-column roster schema, all unverified, no fictitious reviewers. The scan
uses only train/validation cohort shots and records each lower-view liveness
witness in `corpus_scan.json`; blind test shots are excluded before reads.
The isolated server consumes a copy as `tables/detachment/shots.csv`. Nothing
writes to production review stores or label tables.

From this worktree, using the shared installed environment and scratch TMPDIR:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/detach-ui
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_roster.py \
  --out "$LABELER_ROOT/round4/detach-ui" --build --workers 4
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_demo.py --shot 190010 \
  --out "$LABELER_ROOT/round4/detach-ui" --verify
```

The demo uses a neutral `Reviewer` name, delays frame delivery by 180 ms,
asserts decoded-frame consistency and bounded playback, captures an actual
Chromium screenshot of a lower-divertor view, and stops its temporary free-port
server. JSON records include source hashes and command output. Existing owner
servers are untouched. To open the scanned stores on a free port:

```bash
LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui \
LABELER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach-ui/tables \
LABELER_DETACHMENT_INDICATORS=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/detach/bins \
LABELER_NO_FETCH=1 pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python -m labeler.events.ui --port 8812
```

Open `#detachment/190010` on the printed token URL. Camera controls require
server API 9. Rebuild after indicator delivery; the store recipe version also
invalidates earlier scalar/stride previews.
