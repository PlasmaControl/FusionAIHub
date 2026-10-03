# Detachment camera review

The review page offers **1 attached, 2 detached, 3 MARFE, 4 uncertain** on an
exclusive state track. Unmarked time is not reviewed. The producer label model is
an **unverified suggestion**, with its fallback rule and three indicator votes
shown directly below the video. Review edits remain in the usual annotation
lanes. The Individual lane initially copies the producer label: reviewers correct
suggestions. **Start blank** clears the editable lane and supports Undo; Source
stays visible, so this option does not constitute blind review.
Binary agreement scoring refuses this multiclass editor.

## Human camera review protocol

Use the camera together with diagnostics and geometry to mark what you can decide:

| Code | State | Meaning |
| --- | --- | --- |
| 1 | attached | Emission peaked at the target strike point. |
| 2 | detached | Emission peak lifted off the target along the leg toward the X-point, with target D-alpha or Te rollover. |
| 3 | MARFE | Localized bright radiation at or above the X-point, or on the inner wall. |
| 4 | uncertain | The reviewer cannot decide: view blocked, shelf gate invalid, or evidence conflicts. |
| — | unmarked | Not reviewed. |

MARFE is physically detached but has a distinct stage in this exclusive track.
Partial and full detachment share code 2. Missing evidence never means attached.
The votes and label states are read directly; the page does not reclassify them.

## What the machine label means

The page help renders the recipe frozen into that shot's store at build time.
`review_recipe.json` in the producer output directory is preferred (override:
`LABELER_DETACHMENT_RECIPE`). Its object is preserved verbatim, including method,
thresholds, definitions and evidence gates. Without that file, the producer's
exported `labels_rule.meta.json` and method record `docs/labeler/detachment.md`
are frozen, with their paths and hashes; the latter includes the definitions,
methods, numeric thresholds, MARFE persistence, spatial/separatrix and independent
corroboration requirements. `LABELER_DETACHMENT_METHOD` can override that record.
Missing interpretation is explicitly unavailable. No UI-authored numeric threshold
is inserted into help. Optional guides use only a structured recipe's
`thresholds.<indicator>` numeric entries; without those, the traces have no guides.
Read the machine block and each bin's vote/reason together: high DZ alone does
not establish MARFE. A valid abstention can reflect evidence gates as well as a
transition band; the readout preserves the producer reason and does not infer one.

<!-- MACHINE_RECIPE_START -->
<details>
<summary>Producer wording (generated recipe snapshot)</summary>

````text
# Detachment protocol

This is an **unverified diagnostic consensus**, awaiting the detachment review
stream. It is suitable for studying agreement and coverage; its posterior and
model weights are not calibrated physical probabilities or diagnostic accuracy.
The fixed cohort's test shots are excluded from fitting, threshold validation,
surrogate selection and bin-policy analysis.

## Definitions and export

The scalar lower-divertor state is attached (1), detached (2), MARFE (3), or
uncertain (4). Absence (0 internally) means not assessed; missing time is never
interpreted as attached. Attached means the accepted diagnostics support a
low radiation front and an attached target. Detached means the front has lifted
and another compatible indicator supports reduced target current or enhanced
divertor radiation. MARFE means a persistent high front with localized emission
inside the separatrix near/above the X-point and a separate operational cue.
These definitions express the protocol, not independently established truth.

Bins are 50 ms, following the catalog grid and published diagnostic response
scales. A full TangTV camera frame is approximately 30 Hz; the corpus 50 fps
resampling does not create independent frames. Thus a bin holds roughly 1.5
independent camera frames, and a median alone cannot reject ELM contamination.

`data/events/detachment/extend_detach_vote/detach_shots.csv` contains assessed
intervals with the standard shot/category/start/end/confidence fields. Its
`attrs` JSON includes `tier`. The large `labels_bins.csv.gz` has an explicit
`tier` column, indicator value/valid/reason/vote columns, source and geometry
provenance, posterior columns, and a separate `state_temporal_imputation`.
The owned `shots.csv` is an unverified camera-review roster, not a gold table.
The interval table, sparse grids and per-shot indicator CSVs derive from the
same observed bin labels. Time bases are milliseconds in these outputs; corpus
HDF5 time bases are converted from seconds.

## Indicators and validity

### Target current

The processed path reads `LANGMUIR::TOP.PROBE_*:{R,Z,JSAT,TIME}`. Invalid/zero
positions are excluded. It uses the probe nearest the EFIT outer strike point,
with a maximum distance of 2 cm, and inter-ELM bin medians of processed Jsat
(amps/cm²). Following the Eldon 2021 density-only DOD construction, the attached
reference is `C ne²`, fitted separately for explicit L and H confinement labels
in an attached pre-puff window. The window ends at the first corpus total-gas
rise exceeding 10% of its shot excursion above the tenth-percentile floor.
It needs six quality-valid bins with an attached, valid TangTV vote per regime.
Unknown confinement, missing gas timing, bad geometry, low power, ramping or
ELM contamination cannot fit C. This operational pre-puff recipe is explicit
and should be checked by a human before interpreting C as a physical reference.

If that reference is unavailable, the output is **uncalibrated Jsat ratio
(local proxy)**. Positioned processed traces use a nearest-probe density-scaled
whole-shot 90th-percentile reference; missing processed traces use the original
raw swept-probe maximum with density and power scaling. Neither self-reference
establishes a wholly detached shot's attached current. The `afrac_method` field
separates `eldon_pre_puff_LH` from `local_proxy`; benchmark terminology defaults
to the local proxy. It votes attached at ratio ≥0.75, detached at ≤0.5, and
abstains between. It cannot vote MARFE. The nearest-probe quality and geometry
gates still apply to the processed fallback.

### Divertor radiation

`f_div = PRAD_DIVL / P_in` uses calibrated lower-divertor bolometry and heating
power (beams, ECH and ohmic power). It votes attached at ≤0.36 and detached at
≥0.50, abstaining between, and never votes MARFE. Eldon 2019 supplies the
sensor definition, **not universal state thresholds**. These fixed thresholds
come from the local Chen 201081 worked example and are checked on cohort training and external development
shots with true inversions; they are not optimized on that check.
The validation counts and binary kappa are reported below. Validity requires
input power ≥0.5 MW and acceptable samples/ELM share. The processed-target path
uses density-only scaling; the legacy raw proxy uses P_SOL = P_in − dW/dt,
without subtracting core radiation, another limitation of that fallback.

### TangTV front

C-III emissivity inversions give the SSA height ZE and
`DZ = 1 − (ZX − ZE)/(ZX − ZS)`. The **upper shelf** is near Z=−1.25 m with
outer strike R≥1.37 m. The **lower shelf** is near Z=−1.363 m with R<1.37 m.
Lower-shelf inversions use plasma_tv's 2026 window `RX ≤ R < SHELF_WALL_R`
(`r_max=1.37 m`) and the actual lower-shelf strike height, excluding upper-shelf
emission. All local lower-shelf inversion shots are attempted and checked
against the other indicators below; unlocalized Thomson temperatures are not
used as confirmation.

Both sources require lower-single-null geometry, a positive outer leg of at
least 10 cm, a valid shelf strike, finite emission, DZ≥−0.25, and a nearest
EFIT slice within 40 ms. Camera geometry preferentially uses EFIT02, matching
plasma_tv. Missing or single-slice EFIT02 records have an explicit EFIT01
fallback for inversions, with source exported; the surrogate rejects this
fallback. Heating/current calculations retain their original EFIT01 source.
A bin requires at least half its camera frames to pass its frame gates.

The surrogate is a ridge regression of block-averaged raw frames to inversion
ZE. Features subtract the black level, take square-root block means and apply
per-frame mean-zero/unit-standard-deviation normalization (ZE_Norm), plus RX.
Deployment requires leg length, outer strike R and X-point R inside the
accepted inversion training envelope, training-range brightness, less than 1%
saturation, and verified camera/filter provenance. Camera provenance is verified
only on matching SAV/corpus training shots. Predictions on other shots abstain:
there is no warranted exposure/camera transfer claim. Alpha selection is nested
by shot: each outer held-out shot is excluded from three grouped inner folds,
then evaluated after a fit on the outer training shots. Final deployment alpha
is selected on development data only. The reported error is a surrogate-to-
inversion error, not detachment or MARFE accuracy.

TangTV votes attached for DZ<0.35 and detached for 0.5≤DZ<1.2. The transition
band abstains. DZ≥1.2 alone is a **candidate MARFE**. A MARFE vote needs at least
two consecutive bins satisfying high DZ, accepted-frame spatial evidence and
a second cue. Spatial evidence requires the inversion's global emission peak
to have `0<psiN<1`, within 30 cm radially of X and between ZX−2 cm and ZX+30 cm,
using an EFIT02 flux map within 40 ms. Only frames accepted for DZ contribute;
an excluded ELM frame cannot supply spatial evidence. The second cue is a
recorded H–L back-transition within 200 ms, or fG≥0.8 from unit-confirmed line
density and Ip. Cached UF density lacks confirmed units, so it supplies no
Greenwald cue. The implemented future SI-density chord approximation is not
used to invent present fG values. Surrogate fronts lack spatial inversions and
cannot confirm MARFE. The spatial window is an operational candidate test,
not proof of a localized instability or an expert MARFE label.

### ELM mask

Robust divertor D-alpha spikes are widened by 17 ms before and 50 ms after,
covering half a 30 Hz integration plus a conservative post-ELM recovery window.
Accepted TangTV frames exclude this mask; bins with ELM share >50% are invalid
and cannot become certain through TangTV. Target-current and radiation bins
retain their explicit quality masks. The fixed recovery window is a conservative
protocol assumption and merits shot-specific development validation.

## Redundancy, model and tiers

The label model exactly enumerates three states and indicator propensities,
weights and optional pair correlations. Correlation structure uses shot-grouped
development folds and the simplest structure within one standard error of the
best held-out marginal likelihood. The primary weight fit requests true
inversion anchors with all three indicators valid and a fixed 1/2, 1/4, 1/4
class prior. With fewer than 300 anchors it uses an all-bin fallback, explicitly
unidentified. Weight saturation is recorded. In either case TangTV's physical
accuracy is **not identified**, and all posteriors are **uncalibrated**.

After the posterior, a certain state requires at least two valid cast votes,
a valid in-domain TangTV vote agreeing with another indicator, no incompatible
vote, a posterior-selected state compatible with those votes, and that state's
posterior ≥0.7. `COMPATIBLE` treats a detached target-current/radiation vote as
compatible with a TangTV MARFE vote; an attached vote conflicts with either
not-attached state. Invalid votes do not participate. Thus all available cast
votes must have a common compatible state. Afrac plus Prad alone never produce
a certain state, even when both agree or the posterior is high.

| Tier | Exported state | Meaning |
|---|---|---|
| `certain` | 1/2/3 | All certainty and physical gates pass |
| `low_confidence_pair` | 4 | Target-current/radiation agreement without TangTV support |
| `candidate_marfe` | 4 | High front without the full MARFE evidence/certainty chain |
| `conflict` | 4 | Cast votes have no common compatible state |
| `low_posterior` | 4 | Insufficient compatible vote support or posterior below threshold |
| `no_vote` | 4 | No sufficient supported state |
| `not_assessed` | absent | Fewer than two valid indicators |

Candidate MARFE takes precedence in the tier field; indicator votes still reveal
any conflict. The transparent `detach_rule` uses the same compatibility and
physical support conditions without fitted confidence. Primary states are
**not smoothed**. A one-bin neighbour-fill suggestion is exported separately;
it may cross a validity/confidence boundary and must never be promoted to an
observed certain label. Its count is audited separately.
````

Source SHA256:

- `label_metadata`: `ad5645cbb07064c7452180d8645a536cbbfe469ece305b8502a641774ae448ed`
- `method_record`: `92effb514af488b8fd1224e4305243fa81786abce82a4b23af8358d4c02df388`

</details>
<!-- MACHINE_RECIPE_END -->

**f_div = Prad,div / P_in** names the lower-divertor radiation fraction throughout
the page and figure. It measures radiation, rather than directly observing
detachment. Figure points above one are flagged for a heating-power denominator
check. **Tomographic inversion** and **surrogate regression
(model estimate)** are named in the row title and preserved per bin. A
surrogate is never presented as a measurement. Legacy height CSVs without
provenance explicitly lack a recorded source.

The strips show the primary label model, fallback rule, Afrac, f_div and
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
visible while the diagnostic rows scroll. Its height leaves at least one full
diagnostic row and the time axis available at 1366×768 and 1400×900, with
annotation controls below. Unavailable cameras occupy one line. Camera metadata
and producer details are collapsible; expanded details scroll within a bounded
area.

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
finite native-sample mean over **t < 0**. There is no tuned flatness threshold.
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
  --out "$LABELER_ROOT/round4/detach-ui" --build --rebuild-existing --resume-build --workers 4
# The command above is the idempotent controller rebuild after producer changes.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_demo.py --shot 200977 \
  --out "$LABELER_ROOT/round4/detach-ui/browser-fix4-200977" --verify
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python scripts/labeler/detachment_review_export.py \
  --prefer-inversion --evidence docs/labeler/results/detachment_ui_fix4_export.json
```

The current static appendix figure is `detachment_review_200977_paper.pdf` (vector) and
`.png` (150 dpi), under the delivery root: real channel-2 frames, primary label,
three vote strips and Afrac/radiation/DZ traces share marked frame times. Fonts
are ≥7 pt at the intended **6.75-inch two-column width without further scaling**.
It uses a nonblind producer-external development shot with inversion-sourced DZ.
The current producer labels shown are attached, uncertain and detached; the
caption explicitly says the raw view does not visibly separate the states.
The earlier 190212 figure is regenerated with its now-unpublished label lane
blank. Frame timestamps
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
Fresh queue, covering tests/lint, browser checks and export records are in
`docs/labeler/results/detachment_*fix4*.json` and `detachment_review_queue.json`.
The fix3 records describe a superseded producer snapshot.
