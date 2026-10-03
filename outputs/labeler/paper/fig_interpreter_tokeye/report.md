# Figure 1 — ninth fix round

Status: **DONE_WITH_CONCERNS**. Primary remains **201978, 1500–3300 ms**.
Read both complete re-reviews: Sol9 (8/8/8/8) and Opus9 (7/8/8/8).
Every Important and Minor finding is addressed. This is the single current
report, mirrored at the dispatch `reports/fig1.md` and this worktree's
`outputs/labeler/paper/fig_interpreter_tokeye/report.md`. Superseded round-7
and round-8 text, captions and counts are archived below **History**.

Commit range: **41ba2067..HEAD**, branch **r4-fig1**. Source renderer commit:
**c7cb0acd**. All six renders began from that clean commit. The following
artifact/report commit also corrects the audit's fallback-threshold assertion.
Every new commit uses `labeler:` and
`Co-Authored-By: Codex gpt-6.1-sol <noreply@openai.com>`.

## Fix round 9

The toroidal mode-number key now sits in the right margin beside the processed
0–30 kHz panel, under the single NTM entry. Its title is **toroidal mode number
n (Mirnov array)**, and the caption identifies the measurement source. The key
has two columns and is removed from the Labels heading. The duplicate NTM
leader is removed; the remaining entry says **NTM candidate suggestions** and
retains the measured n=1/2 and ≤30 kHz rule. **Detector F1 0.46, below bar**
now appears in the caption. The primary's AE annotation is in the right margin,
joined to detector-positive support by a neutral leader anchored at
2775 ms / 180 kHz. Its former black in-panel box no longer covers the cascade.
The anchor is presentation metadata, not an event-label edit.

The scale note reads **0–55 kHz: higher-resolution spectrogram; 0–30 stretched,
30–55 compressed**. A precommit bounds check caught clipped NTM text and then
an overlap with the mode-key title; wrapping and measured spacing fixed both.
The final audit checks the mode key's complete bounds within the processed
low-panel height, its right-margin location, title/legend/text non-overlap,
AE-callout margin placement and the scale-note text on all six renders. The
regime swatches on the two diagnostic views moved into the unused raw-panel
margin to leave the mode key clear; regime labels remain on D-alpha.

The primary caption is **90 words**, under the 95-word limit.
It states that AE tags inherit the CO2 detector's 25 ms timing and that the
same lines stay untagged after 2.8 s when the detector is negative; the first
ELM (2297 ms) precedes the expert span; raw bands are normalised separately;
the 80 kHz AE input-band floor leaves 55–80 kHz white; NTM outlines mark time
coincidence only, including small 3–5 kHz fragments; AE targets used TokEye's
mask (circularity); and the sawtooth row shows only four-state notation in
this window. Source names, n restrictions, the shared magnetic inputs,
inferred L-mode, exact sawtooth states and alternate caveats remain in the
generated appendix. The late-structure sentence now clearly says that
magnetic lines remain visible while detector-negative time stays untagged.

The primary appendix gives **TokEye 0.2 / AE 0.5 / NTM 0.63**. Supplied
`ae-ours` predictions use AE 0.5 on 201978, 201973 and 203187. The three
fallback views preserve their separately recorded frame-detector AE **0.7**
operating point and 10 ms cadence; the audit checks the actual source-specific
threshold rather than imposing 0.5 on those views. No threshold changed.
Source: each `<shot>.json:tracks.alfven_eigenmode.decision_threshold` and its
corresponding generated appendix.

Sawtooth display now clips source intervals without merging or changing any
category, including the primary's 5.5 and 6.5 ms uncertain intervals. Source
and display intervals match exactly. The primary has **832.100 ms uncertain**,
**967.900 ms unassessed**, and **zero present or absent time**; the four-state
key remains, with no smoothing disclosure needed. The appendix explicitly
states there is no state smoothing. Source: `audit.json:renders[shot=201978]`
fields `sawtooth_states`, `sawtooth_display_intervals_ms`,
`sawtooth_state_duration_ms` and `sawtooth_display_merge`.

Time bounds now reject incomplete pairs before reading annotations or sources.
`--start`/`--end` are aliases for `--tmin`/`--tmax`; either both bounds or neither
must be supplied. Reversed and non-finite pairs also fail clearly. The final
six-shot render batch used paired `--start`/`--end`; omitted pairs still use
the existing presets/cohort window. Source: renderer `main`; rejection checks
are recorded in `round4/fig1/fix9-bounds.log`.

## Current data and result evidence

All scientific results come from the committed renderer and
`scripts/labeler/paper/fig1_audit.py`, with their current JSON records here.
The table below is sourced from `audit.json:renders[]`; component and peak
counts are overlap/display descriptors, not classification scores.

| Shot | Split | Window (ms) | AE / NTM components | D-alpha peaks | ECE candidates | Caption words |
|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 / 11 | 49 | 0 | 90 |
| 201973 | val | 1600–3350 | 154 / 14 | 65 | 3 | 58 |
| 203187 | train | 1700–3150 | 262 / 8 | 6 | 0 | 65 |
| 186636 | val | 1300–3900 | 307 / 4 | 0 | 0 | 53 |
| 191376 | train | 1500–2900 | 0 / 0 | 2 | 0 | 28 |
| 191782 | train | 1800–3700 | 0 / 0 | 0 | 0 | 28 |

There are four train and two validation examples and **zero blind-test shots**.
The primary retains **319 AE / 11 NTM** components and **49** D-alpha peaks.
Its saved processed high-band PNG has **6,050** pink pixels,
with **zero** time or band clipping violations. The extra exposed pink pixels
follow removal of the covering annotation box; tag arrays and scientific counts
are unchanged. Source: `audit.renders[shot=201978].raster_ae_audit` and
`tag_counts`. The raster check allows one pixel at clip boundaries.

The new `--baseline-ref 41ba206779910882b9a826fe668970da848f38af` audit comparison
proves the scientific inputs/outputs match round 8: all six shots/windows,
source tracks and exact categories, detector predictions/operating points,
checkpoint/cache/waveform identities, training lists, filters, source stores,
mode counts and measured-n support, ELM peaks/spans, crash evidence, late-line
support and harmonic-support records. Only presentation, captions/appendices,
annotation coordinates and the unsmoothed display states changed. Source:
`audit.json:scientific_data_unchanged_from`, with assertions in the committed
audit. All source hashes are independently rechecked. AE and NTM detector
training still excludes 201978 (`201978.json:detector_training`). The sawtooth
snapshot remains **ad0ca40f** and the source manifest is unchanged.

## Artifacts and verification

All six final native **150-dpi PNGs** were opened and inspected, including the
primary. The primary rebuild and layout previews were also inspected. PDFs
remain **6.75 × 5.5 inches**, with **≥7 pt** fonts, for text-width placement.
Primary: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fig_interpreter.{pdf,png,json}`,
`caption.tex` and `appendix.txt`. Each `alt_<shot>/` folder contains the matching
artifacts. Large figures and caches stay outside git; small JSON, caption,
appendix, audit, README and report files are saved in the worktree.

Covering tests only:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_paper_label_figure.py
  -q -p no:cacheprovider
67 passed in 4.40s
```

The updated short-interval regression failed against the old smoothing code
before the fix (`fix9-tests-red.log`). Final tests: `fix9-tests.log`.
Ruff check of all four changed Python files: **All checks passed!**
(`fix9-lint.log`). Ruff format --check: **4 files already formatted**
(`fix9-format.log`). All six captions compile in a plain LaTeX/T1 document
(`fix9-caption-smoke.log`). `git diff --check` passes.

```text
pixi run --frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary
  --baseline-ref 41ba206779910882b9a826fe668970da848f38af
  --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/audit.json
Audited 6 non-blind renders; zero array/raster AE clipping violations
Primary PDF and PNG rebuild identically byte-for-byte
```

Final audit: `fix9-audit.log` and `audit.json`; per-shot render logs:
`fix9-render_<shot>.log`, all under `round4/fig1/`. No model inference rerun,
fetching, training, source/data/checkpoint edit or manuscript edit was needed.
Commands used the stream TMPDIR, read-only local inputs, LABELER_NO_FETCH=1
and frozen/no-install pixi from this worktree. The mandated tmpsweep ran after
the final render batch and audit/rebuild. Nothing was pushed, merged or rebased.

## Concerns and integration

No implementation or clarity blocker remains. Existing scientific limitations
remain disclosed: NTM suggestions are below acceptance (F1 0.46); physics
sawtooth labels/crash candidates are unvalidated research evidence; AE targets
used TokEye's mask and NTM shares magnetic inputs. Alternate expert-ELM/regime
disagreements, post-NBI AE and possible pickup caveats remain in the appendices.
The controller can integrate the primary with this caption and appendix,
retaining the selected physics source and re-pinning it at integration if
required. Existing catalog source disagreements remain recorded in the source
manifest. There were no deviations from this clarity-only brief.

## Current primary caption

```tex
\caption{DIII-D shot 201978: raw signals $\rightarrow$ TokEye-processed modes $\rightarrow$ event labels; raw bands normalised separately. Toroidal mode number $n$: Mirnov array. Pink AE tags inherit the CO2 detector's 25 ms timing: the same lines stay untagged after 2.8 s, where it is negative. AE floor: 80 kHz input band; 55–80 kHz stays white. The first ELM (2297 ms) precedes the expert span. Orange NTM outlines: time coincidence only (including 3–5 kHz fragments); detector F1 0.46, below bar. AE targets used TokEye's mask (circularity). Sawtooth: four-state notation only in this window.}
\label{fig:interpreter-201978}
```

## History

All text, counts and captions below are **superseded historical records**.
The single current report and caption are above. Round-7 and round-8 observations
are preserved for provenance; they do not describe the current artifacts.

### Figure 1 — seventh fix round

Status: **DONE_WITH_CONCERNS**. Primary remains **201978, 1500–3300 ms**.
Both `fig1-sol7.md` (8/8/8/8) and `fig1-opus7.md` (7/7/6/6) were read in full.
All requested Important and listed Minor findings have implementation fixes or
explicit physical/source caveats below. No implementation blocker remains.

Commit range: **a09fcf00..HEAD**, on **r4-fig1**. Source commits:
**e651654c** (sources/captions/shared frequency rows/history), **24c11029**
(mode-panel minimum height, short NTM leader and alternate caveats),
**990d3ebe** (measured key spacing and text-overlap audit). Final artifacts
and this current report are in the following artifact commit. Each message
uses `labeler:` and the requested Codex gpt-6.1-sol co-author trailer.

This is the single current status, identical at the dispatch `reports/fig1.md`
and `outputs/labeler/paper/fig_interpreter_tokeye/report.md`. Historical
`fix_round*.md` reports/plans were consolidated into
`docs/labeler/fig1_fix_history.md`; superseded audit JSONs remain in git at
`a09fcf00`. The outputs directory now holds only current records/captions,
README, source manifest, report and audit.

#### Result and evidence

All six final renders were made from clean committed **990d3ebe**, using local
read-only inputs and cached model inference. All six native **150-dpi** PNGs
were opened and inspected: **201978, 201973, 203187, 186636, 191376, 191782**.
The byte-identical primary rebuild PNG was also inspected. Raw and processed
views share identical ranges and heights for 0–30, 30–55 and 55–250 kHz,
including a visible 55 kHz tick and matching fold. The 30–55 kHz white-mask
strip restores the structures previously omitted. The n panel retains its
0.8-inch minimum across all renders. The fixed AE callout is removed. NTM's
white dashed key overlays a black swatch; other-n uses the same patch handle
as the measured hues. The longer suggestion leader and inferred-L-mode label
are readable, with source-specific text and preserved source categories.

All scientific numbers below come from the committed generators
`scripts/labeler/paper/fig_interpreter_tokeye.py`, `fig1_refresh_sources.py`
and `fig1_audit.py`, and the JSONs they write. Current aggregate evidence is
`outputs/labeler/paper/fig_interpreter_tokeye/audit.json`, byte-identical to
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/audit.json`.
Per-shot source paths, hashes, exact states, thresholds, timing, drawing
geometry and figure hashes are in `<shot>.json` beside the audit. Counts are
illustrative overlap descriptors, not classification scores.

| Shot | Split | Window (ms) | AE / NTM components | D-alpha peaks | ECE candidates | Caption words |
|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 / 11 | 49 | 0 | 114 |
| 201973 | val | 1600–3350 | 154 / 14 | 65 | 3 | 121 |
| 203187 | train | 1700–3150 | 262 / 8 | 6 | 0 | 117 |
| 186636 | val | 1300–3900 | 307 / 4 | 0 | 0 | 84 |
| 191376 | train | 1500–2900 | 0 / 0 | 2 | 0 | 33 |
| 191782 | train | 1800–3700 | 0 / 0 | 0 | 0 | 29 |

The renders use four train and two validation shots, **zero blind-test shots**.
The primary retains **319 AE / 11 NTM** tagged components and **49** threshold
D-alpha peaks. Model/checkpoint inputs, source tables and selected thresholds
were not changed. The primary caption is **114 words**; all six captions fit
the **150-word** limit. PDFs are **6.75 × 5.5 inches**, with fonts **≥7 pt**,
for inclusion at text width; PNG DPI and PDF dimensions are audited. Sources:
`audit.renders[].tag_counts`, `elm_peak_count`, `caption_words`, `pdf_size_in`,
`layout` and each record's `print_layout`.

The audit checks exact states/source priority, immutable physics hashes,
cache/waveform/code identity, source/evaluation/figure hashes, matching raw and
processed geometry, caption source wording, font/text bounds and non-overlap
of heading/legend text. It reports zero array projection violations and zero
unmeasured/n=3 NTM support violations. A separate saved-PNG audit identifies
pink RGB pixels inside the processed high-frequency axes and checks source
PRESENT times and the detector's ≥80 kHz band, allowing one raster pixel at
clip boundaries for antialiasing/rounding. The primary has **4,880** such pink
pixels and **zero** clipping violations (`raster_ae_audit`). The raster test
also detects deliberate time/band leaks and accepts shots without AE tint.
The primary PDF and PNG rebuild **byte-for-byte identically** (`reproducibility`).
These are rendering/provenance checks, not independent event identification.
NTM's support audit remains array-based; contour strokes do not create new
measured-n evidence.

#### Source choices and review fixes

**Sawtooth.** `figure_sources.sawtooth_caption(record)` now always summarises
exact source-state durations, including an omitted row, rather than reducing
uncertainty to a lack of present time. Source names are plain words: physics
labels, expert intervals or the selected detector. No catalog versions appear
in paper prose. Primary source durations are **832.100 ms uncertain**,
**967.900 ms unassessed**, **0 ms absent** and **0 ms present**, rounded in the
caption to 832/968 ms. The caption states the row is omitted, qualifies the
ECE density proxy and discloses unavailable Bt. Source:
`audit.renders[shot=201978].sawtooth_state_duration_ms` and `sawtooth_density_proxy`.

The immutable snapshot is newly pinned to saw-stream commit
**ad0ca40f1a7e85c22a88ffcf71ba7ee7456c13e2** at
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/saw_source/ad0ca40f-ad0ca40f1a7e/shots`.
The refresh script reads the completion record from that exact git object,
requires the source worktree HEAD and external completion to agree, verifies
the shot rules, and hashes each copied external shot record. The external
shot files themselves are not git objects; the manifest preserves their exact
content hashes. Completion SHA-256:
`fde6ec2ac7eea37cfd524128a1cae241e5e13a44b43f70c0b3460415c6076f28`.
`--sawtooth-source` remains configurable. Expert review still takes precedence,
so 186636 uses the reviewed intervals (1884 ms present, 716 ms absent); physics
crash candidates are separate and use only ECE evidence with a D-alpha veto.
The audit's legacy `verified_crashes` count means evidence-checked research
candidates, not physical validation. Source: `sawtooth_source_manifest.json`
and the corresponding audit source fields.

**Catalog conflict requiring integration choice.** The pinned physics source
is still `unvalidated_research_labels_owner_away`. In the primary's displayed
window, catalog `ece_sawtooth` **v3** contains PRESENT **2291–3300 ms**, while
`sawtooth_frames` **v2** contains PRESENT **2290–2890, 2900–2940 and
2960–3300 ms**. Both conflict with the selected physics source's uncertain/
unassessed states. No source has been silently reconciled or overwritten.
The current source is the user-requested research snapshot; **the controller
pins the paper's shipped sawtooth source at integration** and must rerender
with `--sawtooth-source` if selecting another compatible physics source.
Source paths, hashes and clipped intervals for all six shots are retained in
`sawtooth_source_manifest.catalog_comparison` and `audit.sawtooth_source_manifest`.

**NTM suggestions.** Generated-source captions and leaders say
**“NTM detector suggestions (F1 0.46, below acceptance bar)”**. Evaluation is
read from its recorded source JSON and hashed; its exact F1 is **0.457472**,
precision **0.417698**, recall **0.505618**, over **761** evaluation shots with
**50 ms mean** pooling. It fails the primary acceptance requirement
F1≥0.7 and precision/recall≥0.6, although it is better than always-positive.
Source: `audit.renders[shot=201978].ntm_performance`, including evaluation
path/digest and criteria. Shared magnetic inputs remain disclosed. Imported
NTM on 186636 keeps imported-label wording and does not inherit the generated
source's F1 qualification. Leaders end at rightmost actually tagged, measured
n=1 support; the long diagonal in 203187 is removed.

**Late modes, bands and cadence.** The caption names a **neural detector on
CO2 interferometer data** and states that evenly spaced lines seen only on
magnetics after **2.8 s** are unlabelled. The window remains 1500–3300 ms so
NTM support from 2600 ms stays visible. The primary's **126,086** late coherent
pixels span **80.078–249.512 kHz**, with no component-duration cutoff; source:
`audit.renders[shot=201978].late_untagged_high_frequency`. The **25 ms** pink
cadence stays explicit; the frame fallback uses its recorded **10 ms** cadence.
The **≥80 kHz** floor reflects the AE model's input band. It conflicts with the owner's earlier no-80-kHz-floor
preference, but extending this detector's labels into 60–80 kHz would claim
unsupported band coverage. The floor and limitation remain visible and
reported; no speculative physical AE identity is assigned to untagged lines.
The broad harmonic caption claim remains removed: primary passing support is
**318/684** joint-ridge columns (**46.49%**) for the recorded relative ratio
rule, insufficient to describe all n=2 support as a harmonic. Source:
`audit.renders[shot=201978].harmonic_support`.

**ELM/regime disclosures.** The primary caption now says
**“The largest D-alpha spike (2297 ms) precedes the expert ELM interval
(from 2308 ms)”**. Expert categories stay unchanged. The inferred pre-transition
shading reads **L-mode (inferred)** and is explained in the caption; imported
regime captions start **Regime: imported**. Hatching, Blank, Circles and
Triangles use consistent capitalization. Sources: each audit render's
`largest_dalpha_peak_ms`, `expert_elm_start_ms`, `lmode_inferred`,
`confinement_intervals_ms` and exact current caption.

Both cascade alternate captions explicitly disclose source disagreement:
201973 expert ELM intervals overlap H-mode-detector ABSENT at
**3013–3045, 3077–3124, 3313–3350 ms**; 203187 overlaps at
**2793–2856, 3036–3080 ms**. Source: `audit.renders[].elm_hmode_conflicts_ms`.
203187 also discloses that AE suggestions persist after NBI turns off.
186636 discloses that persistent pink lines may be pickup. Those qualitative
physical-review flags originate in Opus7 and are recorded as
`ae_physical_review_caveat` in the generator, records and audit; no detector
output was reclassified from appearance alone.

**Alternates and maintainability.** 191376 and 191782 are explicitly
**unsuitable Figure 1 alternates** (no AE, n=1 structures with NTM absent),
both in the README and `publication_suitability` records/audit. Their rerenders
remain diagnostic comparisons. 201973 is the strongest visual alternate,
subject to the disclosed source conflict; 203187 and 186636 need the stated
physical review. The long drawing routine now delegates frequency panels and
source-aware legends to `draw_frequency_panels` and `draw_legends`; mode/key
spacing is determined from actual text bounds. The audit's overlap regression
first failed on the old layout and passes the final six renders. No unrelated
model refactor, inference rerun, fetching, dataset/checkpoint edits or
manuscript changes were needed.

#### Artifacts and validation

Primary: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fig_interpreter.{pdf,png,json}`
and `caption.tex`. The five `alt_<shot>/` folders contain the corresponding
PDF/PNG/JSON/caption pairs. Figure and cache hashes are in each record and the
audit. Generated figures and caches remain external, outside commits.

Required environment: stream TMPDIR, local LABELER_ROOT/LABELER_LABEL_TABLES,
PYTHONPATH from this worktree, LABELER_NO_FETCH=1, MPLBACKEND=Agg. Python used
only frozen/no-install pixi through the main manifest.

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_paper_label_figure.py
  -q -p no:cacheprovider
66 passed in 4.70s
```

Ruff check of all five changed Python files: **All checks passed!**
Ruff format --check: **5 files already formatted**. `git diff --check` passed.
Focused caption/state/raster regressions were observed failing before the
fixes. Full logs: `round4/fig1/tests7-red.log`, `tests7.log`, `lint7.log`,
`format7.log`, `overlap7-red.log`, `render7_<shot>.log` and `audit7.log`.
All six captions compiled in a plain LaTeX/T1 smoke document; log:
`round4/fig1/caption7-smoke.log`.
An independent read-only review of the first source-fix commit e651654c found
no substantive issues; final spacing refinements were verified by the saved
rasters and the new non-overlap audit.

```text
pixi run --frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary
  --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/audit.json
Audited 6 non-blind renders; zero array/raster AE clipping violations
Primary PDF and PNG rebuild identically byte-for-byte
```

The mandated tmpsweep ran after render batches and audit/rebuild. All work
stayed on r4-fig1; nothing was pushed, merged, rebased or written in the main
checkout or manuscript.

#### Concerns and next integration step

The controller still needs to pin the shipped sawtooth source, retaining the
catalog conflict as a review item. Physics sawtooth labels/crash candidates
remain unvalidated, NTM suggestions remain below acceptance, and primary/
alternate ELM timing, post-NBI AE and possible pickup flags need owner review.
These limitations are now disclosed; changing physical labels without new
validated evidence would exceed this figure task. Next: integrate the primary
with its exact current caption and source pin, and send the recorded label
conflicts to the owner through the controller's normal review process.

#### Primary caption (superseded)

\caption{DIII-D shot 201978. Raw bands normalised separately; TokEye extracts coherent modes. AE: neural detector on CO2 interferometer data (p$\geq$0.5; targets used TokEye's mask); NTM detector suggestions (F1 0.46, below acceptance bar); H-mode: D-alpha detector; ELMs: expert. Pink: AE overlap in detector band $\geq$80 kHz in 25 ms bins. Dashed outlines: measured and dominant $n=1$ or 2, $\leq$30 kHz (shared inputs). L-mode (inferred): pre-transition H-mode-absent shading. Sawtooth: uncertain 832 ms, unassessed 968 ms (physics labels; ECE density proxy; Bt unavailable); row omitted. Evenly spaced magnetics-only lines after 2.8 s remain unlabelled. The largest D-alpha spike (2297 ms) precedes the expert ELM interval (from 2308 ms). Circles: expert spans containing many ELMs; Triangles: threshold D-alpha peaks.}
\label{fig:interpreter-201978}

Presentation follow-up (renderer f99ae7ba): dropped 40 kHz ticks in both groups, enlarged matching 30–55 kHz strips, and centered/padded regime labels with wrapping where needed; sources, thresholds, windows, states and captions unchanged.
All six PNGs re-rendered and inspected at native 150 dpi; audit.json/audit8.log pass tick spacing, regime padding, clipping and byte-identical primary rebuild; covering tests: 66 passed (tests8.log); Ruff check/format: clean (lint8.log, format8.log); git diff --check passed.


#### Fix round 8

The following current report supersedes the round-seven text above.

### Figure 1 — eighth fix round

Status: **DONE_WITH_CONCERNS**. Primary remains **201978, 1500–3300 ms**.
Read both re-reviews in full: Sol8 (8/8/8/8, accept) and Opus8 (7/7/8/7).
All Important and Minor presentation findings are handled; the controller has
selected the physics sawtooth source and retained NTM as candidate suggestions.

Commit range: **311271c9..HEAD**, branch **r4-fig1**. Source commits:
**958e72ca** (scales, outer contours, AE label, four-state row, concise captions,
appendix text, annotations, training provenance, expanded audit) and
**a04b1c28** (measured tick and marker clearance). All six final renders use
clean renderer **a04b1c28**. Records, captions, appendix files, audit and this
report are saved in the subsequent artifact commit. Every new commit has the
requested `labeler:` prefix and
`Co-Authored-By: Codex gpt-6.1-sol <noreply@openai.com>` trailer.
Historical mixed co-author trailers are preserved under the no-rebase rule.

#### Result and evidence

| Shot | Split | Window (ms) | AE / NTM components | D-alpha peaks | ECE candidates | Caption words |
|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 / 11 | 49 | 0 | 67 |
| 201973 | val | 1600–3350 | 154 / 14 | 65 | 3 | 67 |
| 203187 | train | 1700–3150 | 262 / 8 | 6 | 0 | 67 |
| 186636 | val | 1300–3900 | 307 / 4 | 0 | 0 | 61 |
| 191376 | train | 1500–2900 | 0 / 0 | 2 | 0 | 35 |
| 191782 | train | 1800–3700 | 0 / 0 | 0 | 0 | 35 |

These are overlap descriptors, not classification results. The displayed
cohort remains four train and two validation shots, with **zero blind-test
shots**. Source: `audit.json:renders[]` and `blind_test_shots_used`, generated by
committed `scripts/labeler/paper/fig1_audit.py`. Per-shot source, checkpoint,
waveform, cache, code, PDF/PNG and annotation hashes are in `<shot>.json`,
generated by committed `fig_interpreter_tokeye.py`.

The primary retains **319 AE /
11 NTM** tagged components and
**49** threshold D-alpha peaks. Its caption is
**67 words**. The final saved PNG has **5,273**
pink pixels, **zero** pixels outside detector-positive times and **zero**
pixels below the detector band (one raster pixel of boundary tolerance).
Source: `audit.renders[shot=201978].raster_ae_audit`.
The Sol8 correction is also recorded: the previous renderer **f99ae7ba** had
**4,556**, not 4,880, pink pixels
(`311271c9:outputs/labeler/paper/fig_interpreter_tokeye/audit.json`).
The current count changes with the taller high-frequency panel, opaque pink
and added in-panel label; it is not carried forward from the old raster.

#### Presentation fixes

Both raw and processed panels now mark scale changes at **30 and 55 kHz** with
the same glyph and restore the **40 kHz** tick. Their leader reads
“0–55 kHz: finer frequency bins (30–55 compressed)”; the caption states that
0–30 kHz is vertically expanded. The first audit caught insufficient separation
of the restored 30/40 tick labels. Increasing the middle strip to 0.70 units
resolved it in all six renders. The audit measures actual text bounds and
requires a one-point gap. Source: `drawn.layout.frequency_panels`,
`frequency_scale_breaks_khz`, `lower_frequency_tick_bounds`.

NTM contours now fill the holes in max-pooled display support before drawing a
**solid 0.9-point orange (#E69F00) outer contour**. The legend uses the same
solid stroke and track colour. Measured n=1 or 2 support and source times stay
unchanged; filled interiors and the contour stroke do not create measured-n
evidence. All array support audits remain clean. Source:
`filter.outline_display_rule`, `drawn.ntm_measured_pixel_audit`.

AE pink is fully opaque and keyed at full opacity. Both 55–250 kHz panels use
**1.20 units**, taking height from the low-frequency panels. The primary high
panel is **0.557 inches** tall; its
low-frequency mode panel is **0.627 inches** tall.
The audit's old low-panel minimum was adjusted to 0.6 inches to accommodate
the requested allocation. Frequency content and mode support are preserved.
The one primary in-panel label reads “AE (detector-positive time)” at
**2000 ms / 180 kHz**. Caption highlights mean **time/band coincidence only**,
with no independent event-identity claim. Source: `drawn.layout`.

The sawtooth row is visible even when its source has no PRESENT time. It uses
the existing four-state drawing: colour for present, light grey for absent,
hatching for uncertain, blank for unassessed. On the primary the unchanged
source has **832.100 ms uncertain** and **967.900 ms unassessed**, with no
present or absent time. Display sliver merging remains recorded separately
from exact source intervals. Source:
`audit.renders[shot=201978].sawtooth_state_duration_ms`,
`sawtooth_display_intervals_ms`, `sawtooth_display_merge`.

H-mode and other regime text now clear the ELM box. The audit checks every
regime text rectangle against visible present and uncertain ELM rectangles.
D-alpha peak triangles are below the box top with a measured minimum
one-point clearance. ELM/regime labels are separated without changing expert
boundaries. On 201973 the ECE-supported crash-candidate key is **in its strip**,
beside its ticks, with no right-margin key overlap. The n legend uses four
columns in order **n=1, n=2, n=3, other n**, on a white strip.
Source: `drawn.layout.regime_text_bounds`, `elm_box_bounds`,
`dalpha_peak_box_gap_pt`, `ece_candidate_key_placement`, `legend_labels`.

#### Sources, caption and appendix

The physics snapshot remains pinned at saw-stream
**ad0ca40f1a7e85c22a88ffcf71ba7ee7456c13e2**. Expert sawtooth review still takes
precedence on 186636. The controller will re-pin to the final physics saw head
at integration through **--sawtooth-source**. The audit uses the selected
manifest and immutable shot/completion hashes, rather than hard-coding the
old saw commit. No source-choice ambiguity remains for the paper.
The catalog conflict is retained in `sawtooth_source_manifest.catalog_comparison`
for research provenance; nothing was reconciled by changing labels.

NTM remains **candidate suggestions (detector F1 0.46, below bar)** on 201978,
in the caption, leader and track wording. Exact detector F1 is
**0.457472**, read from the unchanged evaluation JSON; no inset substitution.
Source: `audit.renders[shot=201978].ntm_performance`.

**Both detectors exclude 201978 from training.** AE's checkpoint has
**180** committed training shots; the NTM checkpoint's training JSON has
**3,550** training shots. Neither list contains 201978. The renderer records
the complete lists and hashes; the audit independently reloads both and
checks checkpoint hashes. NTM's checkpoint hash also matches
`model_sha256` in the suggestion metadata.
Source: `201978.json:detector_training` and
`audit.renders[shot=201978].detector_training`;
AE source `src/labeler/models/d3d_ae_activity_seldnet/training_shots.txt`;
NTM source
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/ntm_frames/v2/training.json`.

The caption describes raw → TokEye-processed → labels, colour meanings and
the per-track sources. Detailed timing, cadence, training-target dependence,
shared magnetic inputs, inferred L-mode and state-duration caveats are in
the generated **<shot>.appendix.txt** files and external **appendix.txt**.
The late-structure wording is now
“late magnetic structures without a positive AE-detector label”;
the field caveat is “Bt not in local corpus”.
The expert ELM boundary sentence is in the appendix.
Shot-specific pickup/post-NBI cautions, roles and primary label coordinates
are CLI annotations loaded from **docs/labeler/fig1_annotations.json**, with
the selected annotation and file hash saved in every shot record.
Library code contains no shot-specific appearance judgement.

#### Artifacts and visual inspection

Primary:
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fig_interpreter.{pdf,png,json}`,
`caption.tex`, `appendix.txt`.
The five `alt_<shot>/` folders hold the corresponding files.
Repo sidecars and audit match the external files byte for byte.
Large outputs stay external.

All six final native **150-dpi** PNGs were opened and inspected after the final
rerender; the primary rebuild PNG was inspected too. The primary PNG was also
resampled to **488 pixels** wide (a 3.25-inch column at 150 dpi) and inspected at
`$TMPDIR/fix8-column-primary.png`. Colours, contours and tracks survive this
reduction, but detailed labels become very small. Use the recorded
**6.75 × 5.5 inch text-width** layout to retain **7-point** type.
Column-width inspection is a visual probe, not a claim of seven-point type
after reducing the entire text-width composition.

#### Validation

Covering command (only the covering files):
`bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
/scratch/gpfs/nc1514/FusionAIHub-r4-fig1
tests/labeler/test_paper_figure_sources.py tests/labeler/test_paper_mode_tags.py
tests/labeler/test_paper_label_figure.py -q -p no:cacheprovider`.

Final output: **67 passed** (`round4/fig1/fix8-tests.log`).
Updated library caption/appendix tests first failed against the old contract
(`fix8-tests-red.log`), then passed. The initial audit's 30/40-spacing failure
is retained in `fix8-audit-first.log`; all final geometry checks pass.

Ruff check of the four changed Python files: **All checks passed!**
(`fix8-lint.log`). Ruff format --check: **4 files already formatted**
(`fix8-format.log`). All six captions compiled in a LaTeX/T1 smoke document
(`fix8-caption-smoke.log`); the arrows use LaTeX math commands.
`git diff --check` passes.

Full final audit command:
`pixi run --frozen --no-install --manifest-path
/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
python scripts/labeler/paper/fig1_audit.py --rebuild-primary
--out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/audit.json`.

Output (`fix8-audit.log`):
`Audited 6 non-blind renders; zero array/raster AE clipping violations`.
`Primary PDF and PNG rebuild identically byte-for-byte`.

All commands used the stream TMPDIR and required frozen/no-install pixi
environment, local read-only inputs, cached inference and LABELER_NO_FETCH=1.
The mandated tmpsweep ran after render batches and audit/rebuild. No inference
rerun, fetching, training, checkpoint/data modification, manuscript edit,
push, merge, rebase or history rewrite was needed.

#### Concerns and integration

Physics sawtooth labels remain unvalidated research evidence; NTM suggestions
remain below the acceptance bar. Existing primary/alternate expert-ELM,
post-NBI AE and possible pickup caveats are retained in the appendix. The
controller re-pins to the final physics saw head at integration. Use text-width
placement; the column reduction is too small for full source-label readability.

#### Primary caption (superseded)

\caption{DIII-D shot 201978: raw signals $\rightarrow$ TokEye-processed modes $\rightarrow$ event labels. The 0–30 kHz range is vertically expanded. Pink AE highlights and orange NTM outlines show time/band coincidence only. Outlines require measured $n=1$ or 2 at $\leq$30 kHz. Tracks: AE: CO2 neural detector; NTM: candidate suggestions (detector F1 0.46, below bar); H-mode: D-alpha detector; ELMs: expert; sawtooth: physics labels. Hatching marks uncertain states; blank marks unassessed time.}
\label{fig:interpreter-201978}
