# Stream fig1 — Figure 1 fix round

Status: DONE_WITH_CONCERNS. Recommend **201978** as the primary figure; retain
**186636** as the alternate with more expert interval labels. Both original
reviews were read in full. This report supersedes the original round's
descriptions, counts, caption and deviations where they differ.

Code commits: **43d9acf** (pixel projections, sources, drawing, provenance and
regressions), **2725813** (explicit stored ae-ours priority). All final records
identify **2725813** as the code that generated them. The records/captions/docs
commit is listed in the final response. Every commit uses the requested
labeler: prefix and Codex co-author trailer. Nothing was pushed or merged.

## What was built

- scripts/labeler/paper/fig_interpreter_tokeye.py: raw Mirnov spectrogram,
  D-alpha and NBI; a TokEye-processed copy with toroidal n over 0–30 kHz;
  clipped AE tint/dashed boxes and solid NTM inner outlines; a thin strip of
  sawtooth crash ticks; processed D-alpha with ELM intervals/peaks and
  confinement shading; five time-aligned label tracks. Static, no hover.
- src/labeler/paper/mode_tags.py: tag_blobs requires measured dominant n=1/2
  for NTM; tag_mask intersects native component pixels with half-open PRESENT
  times and the fixed event band. sample_n_map decodes the stored n/brightness
  codes without extrapolating. Tied n evidence is ambiguous, not an NTM tag.
- src/labeler/paper/figure_sources.py: best available confinement source,
  valid stored paper AE predictions, and replaceable crash-time inputs.
- tests/labeler/test_paper_mode_tags.py and test_paper_figure_sources.py:
  clipping, absent gaps, frequency boundaries, wide-pass low-band eligibility,
  dominant n/ties, n-map support, persistent physical rows, source precedence,
  validity/causal AE bins, missing confinement sources and crash overrides.
- outputs/labeler/paper/fig_interpreter_tokeye/: six JSON records, six exact
  caption copies, README.md and this fix report. Large figures and TokEye
  caches remain under LABELER_ROOT/round4/fig1/, outside Git.

The figure uses the pinned TokEye U-Net on MPI66M322D, coherent >=0.2 and
transient <0.2, persistent-line candidate bridging at row share >0.8, small
object removal (zoom 40 px / wide 30 px), small-hole filling (10 px), then
scipy.ndimage.label with a 3x3 structure. These values and the checkpoint hash
are in each record's tokeye/filter fields. Temporal persistence alone is not
claimed to identify receiver pickup.

## Data and source provenance

All inputs were read locally and read-only. LABELER_NO_FETCH=1 throughout;
no network, GPU, training, threshold tuning or test-split selection. The main
checkout's data/events and curated confinement run were never edited.

| shot | split / displayed window (ms) | AE | NTM | confinement | ELMs | sawtooth interval |
|---|---|---|---|---|---|---|
| 201978 | train / 1400–4900 | CO2 xpower frame model | ntm_frames | D-alpha L-H detector | expert review | sawtooth_frames |
| 201973 | val / 1600–3350 | CO2 xpower frame model | ntm_frames | D-alpha L-H detector | expert review | sawtooth_frames |
| 203187 | train / 1700–3150 | CO2 xpower frame model | ntm_frames | D-alpha L-H detector | expert review | sawtooth_frames |
| 186636 | val / 1300–3900 | CO2 xpower, raw-cache CO2 | legacy tearing archive | D-alpha: uncertain throughout this window | expert review | expert review |
| 191376 | train / 1500–2900 | CO2 xpower frame model | ntm_frames | Gill/Butt curated regimes | elm_frames | sawtooth_frames |
| 191782 | train / 1800–3700 | CO2 xpower frame model | ntm_frames | Gill/Butt curated regimes | elm_frames | sawtooth_frames |

Sources: the corresponding <shot>.json, tracks, ae_co2 and ae_ours_lookup.
The four recommended candidates have no per-shot label-store file containing
ae-ours predictions. The script now checks valid stored
d3d_ae_activity_seldnet/ae_active first; if present it uses the paper's
SELDnet-style ae-ours, at fixed p>=0.5 with its validity and causal bins.
Otherwise it retains the current expert/xpower source. It does not resolve or
require the old xpower installation when a stored source is available.

Crash ticks are **detector event times**, including on 186636: the local
ece_sawtooth ECE/SXR union, clipped to each shot's sawtooth PRESENT intervals.
An expert-reviewed oscillation interval does not turn a detector crash time
into an expert point annotation. The crash record includes all event times,
retained times, plotted times, diagnostic information and signal hashes.

Confinement source order: saved four-class review in main data/events,
Gill/Butt curated merged_intervals.csv if the shot is covered, then the
D-alpha L-H table. CONFINEMENT_RUN_DIR defaults to runs/labeler/confinement/v1
in the checkout containing LABELER_LABEL_TABLES; with the required environment
that is the main checkout. Missing required files or shot coverage raise
errors. The script docstring and committed README document the default.
No environment override was needed for the final renders.

The primary H-mode interval is **1799–4470 ms**, consistent with the D-alpha
drop visible near its start; 201973 starts at **1793 ms**, 203187 at **2270 ms**.
Source: each record's tracks.confinement.present_spans_ms. Uncertain D-alpha
classification is hatched in the track and is not shaded as H-mode. The
binary track is named H-mode; category 0 never asserts L-mode. Curated regime
examples show the actual L/H names and use a track named regime.

## Fix round — every review finding

| finding | change and verification |
|---|---|
| Opus Important 1: early H-mode shading; Sol Minor: ambiguous confinement absent | Replaced hmode_frames with curated/D-alpha source order; the primary starts at 1799 ms. Checked shading against the rendered D-alpha trace. Named binary tracks H-mode and curated tracks regime. Caption explains H-mode absent does not establish L-mode. |
| Opus Important 2: NTM paints nearly everything; sawtooth is physically misrepresented | NTM requires a unique measured dominant n=1 or 2. Unknown or tied n does not qualify. Sawtooth never tags a component; discrete crash-time ticks occupy a thin strip. NTM keeps its mode outline when a crash coincides. No seeding claim is made. Source replacements are independent --sawtooth-crashes (shot,t_ms CSV) and --sawtooth-labels (standard interval CSV). |
| Opus Important 3 / Sol Important 1: whole outlines over absent intervals | Native tag masks are component AND PRESENT time; absent gaps remain false. Outlines are inner edges, with no dilation or pooling. Compound rendered clip paths also clip half-cell extents and stroke width. Tests cover partial overlap, absent gaps and half-open ends; all final JSON audits have zero outside-present pixels. |
| Sol Important 2: centroid frequency rule and lost 55–60 kHz band | Every mask pixel is tested against AE >=60 / NTM <60. Both passes receive both event rules; the fold is independent of the tag boundary. Wide components retain lower-row n evidence before cropping to display. Tests explicitly preserve synthetic NTM pixels at 55–59.5 kHz while rejecting >=60. In these real renders no wide 55–60 component has qualifying n evidence, so none is invented: the map itself ends near 30 kHz. |
| Opus Important 4 / Sol Important 3: hue collisions, red/green, unkeyed modes, crowded identification | Remapped n=1/2/3 to blue/cyan/blue-green; every other n is grey and keyed as other n. No n hue shares event pink/orange/yellow. NTM uses solid inner outlines, AE dashed boxes, crashes dotted ticks. Added leader annotations to actual AE activity, a measured n=1 mode and detector crashes where present. Moved the full key above the data. |
| Opus Important 5: shared NTM inputs and failed detector bars omitted | All captions explicitly name shared MPI66M322D and MPI66M n-map inputs and partly built-in coincidence. They disclose ntm_frames N1=false and sawtooth_frames S1=false, identify models and expert tracks, and define tags as time/band coincidence, not independent classification. Primary track records include the metadata paths, hashes and bar outcomes. |
| Opus Minor: spurious short AE box | Boxes omit PRESENT spans shorter than 100 ms, including the 2730–2770 ms decision. Tint and the track still show a short detector decision; boxes do not merge across an absent gap. All final boxes were checked against eligible source intervals. |
| Opus Minor: AE lower edge and source | Box lower edges are >=60 kHz and their strokes are clipped there. Valid ae-ours store predictions are preferred; unavailable for the four candidate shots, so captions explicitly name the earlier CO2 xpower frame model and its 80–250 kHz input band. |
| Opus Minor: legend, AE chip / 250 tick, ELM chip / triangles | The key is in the processed header, AE chips were removed in favour of leaders, and ELM chips occupy headroom above the peak triangles. Inspected all six final PNGs and intermediate renders. |
| Opus Minor: misleading fold | Added broken-axis marks on both sides of each folded spectrum and a 55 kHz tick. Wording is exactly 0-55 kHz, finer resolution. |
| Opus Minor: untagged AE-like comb | Caption keeps steady untagged magnetic lines as unclassified coherent activity. The core-CO2 AE label does not establish their identity; no unsupported TAE/EAE or pickup classification is asserted. |
| Opus Minor: confinement named present | H-mode/regime track names and shading labels state what the source actually shows. General present/absent legend remains for binary event tracks, with the H-mode meaning disclosed in the caption. |
| Opus Minor: pickup removes real n=2 rows | Renamed PICKUP_SHARE to PERSISTENT_ROW_SHARE and raised the cutoff to 0.8. The old 0.4 removed rows at the reviewers' measured 0.52–0.68 / 0.44–0.57 shares. Regression retains physical 15 and 30 kHz rows at 65%/55% occupancy; existing pickup-crossing tests still pass. |
| Opus Minor: wrong component backend | Record now names scipy.ndimage.label, 8-connected, 3x3 structure. skimage is named only for object/hole morphology. |
| Opus Minor: elm_chip UnboundLocalError without D-alpha | Initialize to None and call clear_of only for an existing chip. A real-data draw without D-alpha, with ELM spans and confinement shading, completed using draw_without_rendering; no additional PNG was produced. |
| Opus Minor: silent CONFINEMENT_RUN_DIR behaviour | Sensible main-checkout default documented in module, script and README. Missing curated/fallback files are explicit errors, covered by a regression. |
| Opus Minor: caption implies frequency labels | Caption says the label supplies time and the band is a fixed rule. Updated the report caption and generated caption.tex beside every figure; exact copies are committed. |
| Required correction to original report deviation 5 | The original no-hue-collision claim was false: unkeyed n=4/-2/5 collided with yellow/orange/pink. Those hues are now grey. The red/green pair was removed. This correction supersedes the original deviation, not a defence of it. |

An additional fresh read-only code review found eager xpower fallback lookup,
caption branches that misnamed expert/replacement sources, and ambiguous
dominant-n ties. All were fixed. Its final AE-priority finding was also fixed:
stored ae-ours is checked before an expert fallback, with a regression where
both exist. No independent new scores are asserted.

## Results and records

Generator: scripts/labeler/paper/fig_interpreter_tokeye.py at 2725813.
Sources: outputs/labeler/paper/fig_interpreter_tokeye/<shot>.json, drawn.blobs,
drawn.ae_boxes_ms_khz, drawn.sawtooth_crashes.drawn_times_ms,
drawn.elm_peaks_in_label, drawn.n_map and drawn.projection_audit.

| shot | wide / zoom components | AE tags | NTM tags | AE boxes | crash ticks | ELM peaks | n-map kept share |
|---|---|---|---|---|---|---|---|
| 201978 | 897 / 267 | 207 | 21 | 2 | 52 | 86 | 0.2747 |
| 201973 | 297 / 106 | 118 | 14 | 2 | 1 | 67 | 0.1593 |
| 203187 | 361 / 124 | 98 | 8 | 2 | 1 | 6 | 0.1394 |
| 186636 | 929 / 138 | 358 | 5 | 4 | 1 | 27 | 0.0695 |
| 191376 | 323 / 120 | 0 | 0 | 0 | 6 | 2 | 0.1212 |
| 191782 | 339 / 87 | 0 | 0 | 0 | 1 | 0 | 0.0937 |

All six records have **zero AE/NTM pixels outside PRESENT spans and zero
outside their frequency bands**. All NTM tag counts are accounted for by
dominant n=1/2 (drawn.ntm_dominant_n); all sawtooth component-tag counts are
zero because crashes are point ticks. Counts are descriptive figure outputs,
not classification scores. Components retain full-pass connectivity for
dominant-n evidence, so these counts supersede the original display-cropped
component counts.

## Figures and visual inspection

Under /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/:

- Primary 201978: fig_interpreter.pdf, fig_interpreter.png,
  fig_interpreter.json, caption.tex.
- Same four files under alt_201973/, alt_203187/, alt_186636/,
  alt_191376/ and alt_191782/.

Every PNG made during the fix round was opened and inspected; all six final
PNG files were inspected after the final code commit. Full text width,
6.75 x 5.5 inches, fonts >=7 pt, vector PDF with embedded raster spectra and
150-dpi PNG. Confirmed key outside data, visible fold marks/55 tick, no AE
chip/tick collision, ELM chips above triangles, separate crash strip, keyed
n hues, and clipped highlighting.

201978 remains the recommendation: a clear AE cascade, strong n ridges and
a D-alpha-consistent H-mode onset. 201973 also shows a clear cascade, but a
shorter verified H-mode interval. 203187 has good mode structure but a less
distinct AE cascade; its leader is called AE coincidence. 186636 supplies
expert sawtooth/ELM intervals and legacy NTM but weaker high-frequency
structure and no verified H-mode interval in this window; only one retained
detector crash tick. It is not an equally clear replacement. 191376/191782
demonstrate curated L/H shading but lack tagged AE/NTM structures and remain
regime examples, not primary candidates.

## Draft caption

**From raw signals to label–mode coincidence in DIII-D shot 201978.** Top:
the raw Mirnov-probe spectrogram, D-alpha and neutral-beam power. The broken
frequency axis uses 0–55 kHz at finer resolution and 55–250 kHz from the wide
pass. Middle: TokEye's coherent mask, cleaned by removing small objects and
filling small holes, then split into 8-connected components; the toroidal-n
view replaces 0–30 kHz (blue/cyan/green for n=1,2,3; other n grey). Pink dashed
AE boxes and tint are clipped to PRESENT times and frequencies >=60 kHz;
boxes omit intervals shorter than 100 ms. Solid orange NTM outlines require
dominant measured n=1 or 2 and frequencies <60 kHz, clipped to PRESENT times.
Yellow dotted ticks on a thin strip are ece_sawtooth ECE/SXR crash-detector
event times inside sawtooth PRESENT intervals, including where NTM is
outlined; they are not rotating-mode labels or evidence of NTM seeding.
D-alpha shows ELM intervals/peak markers and confinement shading; bottom
tracks share the time axis. AE uses the earlier CO2 xpower frame model
(80–250 kHz input band); no valid ae-ours SELDnet-style predictions were
available for this shot. NTM uses ntm_frames, sawtooth intervals use
sawtooth_frames, confinement uses the D-alpha L-H detector, and ELMs use expert
review. ELM is the only expert-reviewed primary track. The NTM detector reads
the same MPI66M322D Mirnov probe and MPI66M toroidal-n map as TokEye/the
displayed n view, so coincidence is partly built in. NTM and sawtooth
interval detectors fail their primary acceptance bars N1 and S1. A tag means
coincidence of a label in time and a mode in a fixed band, not independent
classification. Steady untagged magnetic lines remain unclassified coherent
activity; the core-CO2 AE label does not establish their identity. Grey on the
H-mode track means H-mode absent, not established L-mode; hatching means
uncertainty, and blank means unassessed.

The generated TeX beside the primary is the authoritative exact copy; each
alternate has a source-specific caption. All identify expert tracks, detector
crash times, input coupling, failed primary bars and the AE fallback.

## Verification

From this worktree, TMPDIR set to the required scratch/…/r4/tmp/fig1 directory
and LABELER_NO_FETCH=1:

- Covering tests only:
  bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_mode_tags.py
  tests/labeler/test_paper_figure_sources.py -q -p no:cacheprovider.
  Latest output: 23 passed in 3.24s.
- Ruff through frozen/no-install main-manifest labelmaker environment, on the
  script, both library modules and both test files: All checks passed!
  ruff format --check: 5 files already formatted.
- Real-data missing-D-alpha draw with ELM spans and confinement:
  Missing-D-alpha path with ELM spans and confinement shading: passed.
- Final record audits on all six: outside-present/outside-band zero;
  NTM count equals n=1/2 accounting; AE boxes inside >=100-ms PRESENT spans
  and >=60 kHz; crash ticks inside PRESENT intervals; worktree/external
  JSON copies equal; caption copies/hashes match; all generator git fields
  equal 2725813.
- Every final render exited successfully; the final sequential render job
  ran tmpsweep.sh: my /tmp use: 622 MB -> 622 MB.
- No full suite was run, per the binding rules. No source data or manuscript
  was modified.

## Decisions, remaining concerns and next step

The 55 kHz fold is a resolution change, independent of the 60 kHz tag rule.
Standalone 55–60 kHz components without measured n evidence remain untagged;
extrapolating the 0–30 kHz n map would manufacture evidence.

The primary has only one expert-reviewed track, and its NTM/sawtooth interval
detectors fail their primary bars. These limits are disclosed, not repaired
by a figure. No ae-ours predictions exist for its label-store path. Improved
sawtooth crash times/intervals can be swapped in using the documented parameters.

Original reviewers' >=8 re-scoring remains pending; this fix round does not
claim a new score. Next: have both original reviewers reassess the regenerated
primary and the source-specific 186636 alternate; keep 201978 unless expert
label provenance is judged more important than the clearer cascade/H-mode
story.
