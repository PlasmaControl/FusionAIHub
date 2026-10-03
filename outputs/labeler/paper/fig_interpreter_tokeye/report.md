# Figure 1 — seventh fix round

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

## Result and evidence

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

## Source choices and review fixes

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

## Artifacts and validation

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

## Concerns and next integration step

The controller still needs to pin the shipped sawtooth source, retaining the
catalog conflict as a review item. Physics sawtooth labels/crash candidates
remain unvalidated, NTM suggestions remain below acceptance, and primary/
alternate ELM timing, post-NBI AE and possible pickup flags need owner review.
These limitations are now disclosed; changing physical labels without new
validated evidence would exceed this figure task. Next: integrate the primary
with its exact current caption and source pin, and send the recorded label
conflicts to the owner through the controller's normal review process.

## Current primary caption

\caption{DIII-D shot 201978. Raw bands normalised separately; TokEye extracts coherent modes. AE: neural detector on CO2 interferometer data (p$\geq$0.5; targets used TokEye's mask); NTM detector suggestions (F1 0.46, below acceptance bar); H-mode: D-alpha detector; ELMs: expert. Pink: AE overlap in detector band $\geq$80 kHz in 25 ms bins. Dashed outlines: measured and dominant $n=1$ or 2, $\leq$30 kHz (shared inputs). L-mode (inferred): pre-transition H-mode-absent shading. Sawtooth: uncertain 832 ms, unassessed 968 ms (physics labels; ECE density proxy; Bt unavailable); row omitted. Evenly spaced magnetics-only lines after 2.8 s remain unlabelled. The largest D-alpha spike (2297 ms) precedes the expert ELM interval (from 2308 ms). Circles: expert spans containing many ELMs; Triangles: threshold D-alpha peaks.}
\label{fig:interpreter-201978}
