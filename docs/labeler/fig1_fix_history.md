# Figure 1 fix history appendix

Superseded reports and plans, retained verbatim for provenance. Current status is in
`outputs/labeler/paper/fig_interpreter_tokeye/report.md` and the dispatch
`reports/fig1.md`. Historical audits remain accessible in git at `a09fcf00`.


---

## fix_round.md

# Historical Figure 1 report

This report is superseded by [fix_round5.md](fix_round5.md), the current state,
source revisions, selection rationale, results and verification. Earlier audit
JSONs are historical snapshots; use `fix_round5_audit.json` for current artifacts.


---

## fix_round2.md

# Historical Figure 1 report

This report is superseded by [fix_round5.md](fix_round5.md), the current state,
source revisions, selection rationale, results and verification. Earlier audit
JSONs are historical snapshots; use `fix_round5_audit.json` for current artifacts.


---

## fix_round2_plan.md

# Figure 1 fix round 2

The controller's seven decisions are the specification. Execution is native
in the existing r4-fig1 worktree; no source stores or manuscript are modified.

- [x] Pin expert-first AE priority, p=0.7, ECE core-drop/inversion evidence,
  inclusive +/-5 ms ELM veto, unknown-D-alpha exclusion, same-source sawtooth
  track, and independent raw-interval projection audit with covering tests.
- [x] Infer the model-card ae-ours checkpoint on GPU 0 for local corpus CO2,
  saving isolated predictions under round4/fig1. Record a train/val shortlist.
- [x] Separate expert ELM periods from uncertainty; key figure-threshold peaks.
  Remove unsupported crash strips; retain prediction wording for NTM.
- [x] Pool/thicken outlines at display resolution, separate white masks from
  other n, clear headings/leaders, and generate source-specific <=150-word
  captions from source records and acceptance bars.
- [x] Regenerate primary plus five alternates; inspect all 150-dpi PNGs at
  6.75-inch print width; run covering tests and Ruff; append the findings,
  evidence and concerns to the controller's report; commit with the specified
  labeler prefix and Codex co-author trailer.


---

## fix_round3.md

# Historical Figure 1 report

This report is superseded by [fix_round5.md](fix_round5.md), the current state,
source revisions, selection rationale, results and verification. Earlier audit
JSONs are historical snapshots; use `fix_round5_audit.json` for current artifacts.


---

## fix_round4.md

# Figure 1 — historical fix round 4

This report is historical. Use [fix_round5.md](fix_round5.md) and
`fix_round5_audit.json` for the current figures, records and verification.

Status: **DONE**. Both `fig1-opus4.md` and `fig1-sol4.md` were read in full;
all Important and Minor findings are addressed. This is the current report;
previous report copies and audits are historical, not current measurements.

Commit range: **a2ee67a7..HEAD**, branch **r4-fig1**. Renderer source revision:
**9ce61a15**. The final artifact/report commit follows the renderer and
contains the six records, captions and audit; the records intentionally name
the renderer revision. All six were rendered with `--record` while the
worktree was clean, before the resulting records were copied into the checkout.
Every commit uses `labeler:` and the requested Codex gpt-6.1-sol trailer.
Nothing was pushed, merged, rebased, or edited in the main checkout or manuscript.

## What is shown

Primary **201978**, **1500–3300 ms**: raw Mirnov spectrogram, D-alpha and NBI,
then TokEye's binary coherent mask, measured toroidal n, and five label rows.
The physics **sawtooth** row now preserves its source's interval categories,
including uncertain hatching and blank/unassessed regions. The catalog's
**generated sawtooth frame model is not shown**, explicitly disclosed in each
caption. Crash ticks remain separate independently corroborated evidence.
No shot-number branch supplies a scientific caption sentence, omission reason,
view description or TeX label. Every caption uses `fig:interpreter-<shot>`;
the controller should reference **fig:interpreter-201978** for the primary.

The n view uses constant full colour: blue for 1, bluish-green for 2, yellow
for 3, purple for other n. These hue choices follow the accessible Okabe-Ito
palette for keyed n; there is no brightness scaling. Short white arrowed
leaders sit beside representative structures. The longer source text fits in
the widened margin; the heading is **Labels** and NTM reads **detector
(unverified)**. The processed D-alpha panel has an **L-mode** cue before the
D-alpha L-H transition; the binary H-mode source categories are retained.

AE tint starts at **80 kHz**, approximately the interferometer model's lower
input-band edge, and intersects PRESENT times. Its training targets used
TokEye's mask, so overlap is not independent confirmation. NTM outlines here
require a component's unique dominant measured n=1 or 2; they intersect PRESENT
times and measured-n pixels at ≤30 kHz, within the nominal <60 kHz band. They
share the displayed magnetic inputs and remain unverified detector outputs.
No outline invents measured n. The caption states this as a display rule.

We **retain the full window** and explicitly note the late 170–250 kHz lines,
Their time support is recorded in `late_untagged_high_frequency`; they remain
untagged where the AE detector is absent. Keeping them exposes the limitation of time-overlap
labelling while preserving the H-mode transition, ELM onset and emerging
low-frequency structure. The old claim that this window removed the late
comb is withdrawn. This illustration does not identify those lines physically.

All captions are ≤150 words. They disclose separate normalisation of the raw
bands, AE training-target dependence, the NTM display rule, the unshown
catalog frame model, and source-derived sawtooth observability. Triangles
are **D-alpha peaks from a threshold (not annotated)**, restricted to PRESENT
ELM intervals; a peak before the expert interval is consequently unmarked.
Circles mean **expert ELM interval (one span for many ELMs)**. Hatching and
blank remain uncertainty and unassessed/unobservable, respectively.

## Shot-selection rationale

Source: the committed **fix_round2_audit.json**, `selection_rule` and
`ranked_shortlist`; its old thresholds, windows and counts are historical.
Criteria were clear AE plus persistent low-frequency n=1/n=2 structure,
local CO2, D-alpha and NBI availability, and non-blind cohort membership.
Verified sawtooth evidence was preferred but never fabricated to satisfy
selection. Six candidates were rendered; the ranked shortlist was:

1. **201978 (train)**: clearest AE cascade and persistent n=1/n=2 low-frequency
   mode, with local corpus CO2 supporting paper-model inference and aligned
   D-alpha/NBI. Retained for these structures, without requiring verified crashes.
2. **201973 (val)**: strongest visual alternate, with a clear cascade,
   low-frequency mode and the same local diagnostics/paper-model support.
3. **203187 (train)**: AE and low-frequency structure with local diagnostics,
   but a less distinct cascade than the first two.
4. **186636 (val)**: expert ELM intervals and an imported tearing archive
   improve provenance; weaker AE structure, an earlier interferometer frame
   model and uncertain confinement make it a less clear primary.

**191376** and **191782** (both train) are additional diagnostic/regime
comparisons, outside that four-shot ranked shortlist. The round-two audit
recorded no AE or NTM coincidences for them. The current audit below replaces
historical counts and caption claims. No blind test shot was selected or used.

## Inputs and current results

All inputs were local and read-only. No network, fetching, training, tuning,
production-label writes or changes to the fixed cohort occurred. Four train
and two validation shots are illustrated; counts are descriptive coincidences,
not classification metrics or independent physical identifications.

The **newest complete physics source at final rendering** was
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix2/shots`. Its completed cohort export is pinned in
**sawtooth_source_manifest.json** with its SHA-256 digest. The six complete
shot JSONs were copied byte-for-byte to the immutable figure-owned snapshot
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/saw_source/fix2-79a2c18ed4aa/shots`; that parameter is used by every final render
and rebuild. Original and snapshot paths and full per-shot hashes are in the
manifest. The original saw stream's files were never edited. The snapshot
avoids invalidating reproducibility if the producing stream rewrites its files.

Current primary physics state durations within the displayed window are
**0.000 ms absent**, **832.100 ms uncertain**,
**967.900 ms unassessed**, and
**0.000 ms present**. Source: `fix_round4_audit.json`, primary
`sawtooth_states` and `sawtooth_state_duration_ms`. Its density proxy metadata
is retained exactly in `sawtooth_density_proxy`; the caption qualifies ECE
observability without asserting demonstrated cutoff. This replaces all old
claims that the whole window was unassessed.

AE uses supplied isolated paper-model CO2 predictions for 201978, 201973 and
203187, **p≥0.5**. The other three use the earlier CO2 cross-power frame model
and its separately recorded checkpoint threshold. Confinement prioritises
saved review, Gill's and Jalal Butt's curated regimes, then D-alpha L-H states.
ELMs use expert intervals where available, preserving uncertain categories,
otherwise a frame detector. Exact sources, metadata and digests are in each
record's `tracks`, `drawn.stores` and `ae_ours_lookup`.

Harmonics require **|f2/f1 − 2| ≤ 0.1** and **50 ms** sampled simultaneous
support. Frequencies round to **1 kHz**. Primary support is
**325.632 ms**, with per-column measured-n ridge frequency
medians **9.096 / 18.461 kHz**.
The caption uses the second median rounded to the recorded step. This is
harmonic consistency, not proof of a common island. Source:
`fix_round4_audit.json`, primary `harmonic_support`. The previous absolute
1.2-kHz tolerance and 5-kHz prose rounding are superseded.

Source for every count below: the committed generator and **fix_round4_audit.json**.

| Shot | Split | Window (ms) | AE components | NTM components | D-alpha triangles | Caption words |
|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 | 12 | 49 | 125 |
| 201973 | val | 1600–3350 | 154 | 14 | 65 | 136 |
| 203187 | train | 1700–3150 | 262 | 8 | 6 | 100 |
| 186636 | val | 1300–3900 | 307 | 5 | 0 | 101 |
| 191376 | train | 1500–2900 | 0 | 0 | 2 | 76 |
| 191782 | train | 1800–3700 | 0 | 0 | 0 | 76 |

All six have **zero** projected pixels outside end-exclusive PRESENT times
or event bands and **zero** NTM outline pixels outside measured n or above
the n-map band. No AE boxes are drawn. All source hashes match the consumed
files. The primary PDF and PNG rebuild **byte-for-byte identically**.

## Files and verification

Changed implementation: `src/labeler/paper/figure_sources.py` (relative harmonic
support and source-derived caption/state summary), `mode_tags.py` (80-kHz AE
floor), `scripts/labeler/paper/fig_interpreter_tokeye.py` (physics row, display,
clipped state records, figure/code hashes and uniform TeX labels), and
`fig1_audit.py` (committed-code/figure hashes, exact physics row comparison,
source identity, captions, clipping and rebuild). Covering regression tests
are in `test_paper_figure_sources.py` and `test_paper_mode_tags.py`.

Large products under `$LABELER_ROOT/round4/fig1/`: primary
`fig_interpreter.pdf`, `fig_interpreter.png`, `fig_interpreter.json`,
`caption.tex`; the same four in all five `alt_<shot>/` folders; the immutable
physics snapshot; existing `ae_ours.csv`, metadata and TokEye caches;
`render4_<shot>.log`, `audit4.log`, `tests4.log`, `lint4.log`, `format4.log`.
Small committed outputs: six records/captions, **fix_round4_audit.json**,
**sawtooth_source_manifest.json**, README and **fix_round4.md**. Older report
copies point to the current report. PDFs are 6.75 × 5.5 inches with vector
text/lines over raster spectra; 150-dpi PNGs and ≥7-pt fonts are retained.

Every final PNG was opened at native resolution: primary 201978 and alternates
201973, 203187, 186636, 191376, 191782. Inspection covered the visible sawtooth
states, all source labels, n hues, arrows, ELM uncertainty, L/H cues and legends.
The audit separately checks PDF page size, actual figure hashes, ≤150-word
captions, uniform unique labels, source/code hashes and equality of external
and committed records/captions before and after a scratch rebuild. The scratch
rebuild preserves original records even after the artifact commit changes HEAD.

Required TMPDIR/local data variables and LABELER_NO_FETCH=1 were set. Python
ran only through frozen/no-install main-manifest pixi from this worktree.
Covering tests only, through the mandated wrapper:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_paper_label_figure.py
  -q -p no:cacheprovider
49 passed in 4.10s
```

Fresh Ruff check on all six changed Python files: **All checks passed!**
Ruff format --check: **6 files already formatted**. A TeX caption smoke check also passed.
No full suite was run, per implementer rules. The new regressions were first
observed failing for the old harmonic tolerance/rounding, missing physics
summary and below-band AE pixels; the final covering run passed.

Audit command and output (full commands/hashes in its `reproducibility` record):

```text
pixi run --frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary
  --out outputs/labeler/paper/fig_interpreter_tokeye/fix_round4_audit.json
Audited 6 non-blind renders; zero projection violations
Primary PDF and PNG rebuild identically byte-for-byte
```

The mandated tmpsweep script ran after rendering. No unresolved fix-round
implementation defect remains. Remaining scientific limits are visible:
NTM is unverified, AE targets depend on TokEye, harmonic coincidence does not
identify an island, and physics sawtooth assessment is limited by recorded
observability/evidence. The controller can place the primary at textwidth
with its current caption and updated TeX reference.

## Fix round 4

Draw the physics sawtooth states and qualify the density proxy; disclose the
unshown generated frame model. Use full-colour n hues, short neutral arrows,
80-kHz AE tint, relative harmonic support and 1-kHz rounding. Add L-mode and
plain source/ELM language, retain and disclose late untagged lines, and replace
stale provenance with six clean-commit renders and a passing rebuild audit.


---

## fix_round5.md

# Figure 1 — Fix round 5

Status: **DONE_WITH_CONCERNS**. Every Important and Minor item in
`fig1-opus5.md` and `fig1-sol5.md`, together with the owner's fifth-round
instructions, is addressed. The concern is the pending sawtooth-stream
regeneration; this round consumes the newest complete source currently available.

Commit range for this round: **2ae5b1fa..HEAD**, branch **r4-fig1**. Presentation
and source helpers: **cf046f13**; cadence and eligible-pixel component counts:
**21e72826**; six clean-render records/captions: **2965d609**. The final commit
adds the audit and this report. All commits have `labeler:` subjects and the
requested `Co-Authored-By: Codex gpt-6.1-sol <noreply@openai.com>` trailer.

### Presentation and source changes

`fig_interpreter_tokeye.py` gives **0–30 kHz** its own **1.30-unit** panel.
Its measured height is **0.856495 in** for the primary and **0.815930 in**
for the sawtooth alternate. The processed 30–55 kHz band is omitted and the
raw 0–55 kHz panel remains; the break and caption disclose this choice. Smaller
high-frequency/raw bands and keys in the empty right margin pay for the enlarged
n panel. The overall size stays **6.75 × 5.5 in**, with fonts **≥7 pt**.
Source: `fix_round5_audit.json`, `renders[].layout` and `pdf_size_in`.

NTM uses a **one-print-pixel inner edge, alpha 0.45**. Both the component's
unique dominant n and every marked pixel must be **n∈{1,2}**; the measured view
limits outlines to **≤30 kHz**. Components with only n=3 pixels during the
NTM interval are no longer counted as tagged. This corrects the primary's
NTM count to **11**, and 186636's to **4**. The n colours remain readable.
The keys now read **“NTM (n=1/2, ≤30 kHz)”** and
**“AE (detector band ≥80 kHz)”**; the latter wraps to fit the margin.
The raw resolution note, AE callout and n=1/NTM callout sit outside the data
with neutral grey leaders. Source: each record's `filter`, `drawn.layout`,
`drawn.blobs.tagged` and `drawn.ntm_measured_pixel_audit`.

The sawtooth row is shown only if the source has positive-duration PRESENT
time in the displayed window, for every shot. The primary has none, so its row
and empty crash strip are omitted. Its exact source states are retained in JSON:
**832.099943 ms uncertain**, **967.900057 ms unassessed**, **0 ms present**.
Only **201973** has a displayed row in these six windows, with **66.099998 ms
present** and **3** ECE-verified, ELM-vetoed crash ticks. Sawtooth uses vermillion
**#D55E00**, distinct from the yellow n=3. The bottom “present” chip contains
only event colours actually drawn as present bars. Sources:
`fix_round5_audit.json`, `sawtooth_state_duration_ms`, `verified_crashes`,
`sawtooth_track_shown`, and each `layout.present_chip_colours`.

`figure_sources.sawtooth_display` coalesces touching equal states and merges
state slivers **<10 ms** into the longer touching neighbour, processing shortest
first and preferring the earlier neighbour on ties. It never bridges gaps.
This is a display operation: source intervals/categories and exact crash times
stay unchanged. Each record retains raw `state_intervals_ms`, plus
`display_intervals_ms` and `display_merge` with its rule, threshold and all
changes. The current displayed 201973 row needs no sliver reassignment; the
primary's slivers disappear with the omitted row. Regression tests exercise
an actual short-state merge, source immutability, boundary visibility and gaps.

Late-line bands now come from eligible coherent components after the last
PRESENT AE interval, during ABSENT AE time, in the detector's input band.
Each component must persist **≥150 ms** continuously. Frequency bounds come
from its pixels. The primary's bounds are **109.863286–188.476571 kHz**,
captioned as **110–188 kHz**; 201973's are **177.246087–220.703117 kHz**.
The other four windows have no qualifying late-line description; in particular,
186636's previous window-edge sliver no longer triggers it. Sources:
`fix_round5_audit.json`, `late_untagged_high_frequency` for each render, generated
by `figure_sources.late_untagged_lines`. No 170–250 kHz scientific band remains
hard-coded in the renderer or caption.

All event keys and caption claims follow displayed content and source tier.
191376/191782 have no AE/NTM keys or highlight claims and correctly name ELMs
as detector output; neither has the expert-circle key. The current captions
have **26–143 words**. The primary clause says the AE tint follows the
detector's **25 ms bins**; the earlier frame-model alternate uses its actual
**10 ms** cadence, recorded as `temporal_bin_ms`. “Bt not in the local corpus”
replaces the misleading missing-measurement wording when the displayed sawtooth
assessment needs that qualification. The frame-model jargon sentence is gone
from every caption. Sources: six caption files, records and audit `caption_words`.

Confinement intervals now retain numeric `category` alongside the track's own
state/regime names. Curated category **2** is **L-mode**, category **3** is
**QH-mode**, and category **5** is **uncertain**. The D-alpha fallback retains
its H-mode/binary mapping, including numeric category **5**, without claiming
a curated L-mode label. Thus 191376's L-mode interval is no longer serialized
as “uncertain”. Source: all six `tracks.confinement.states` and
`state_intervals_ms`; audit `confinement_intervals_ms` checks that mapping.

### Data, outputs and final counts

Shot selection, thresholds, model inputs and the reviewed windows are retained.
201978 remains the primary for its clear AE cascade and persistent low-frequency
n structure; 201973 is the strongest visual alternate, 203187 another cascade
example, and 186636 adds imported tearing/expert ELM provenance. 191376/191782
remain diagnostic/regime comparisons, not complete substitutes for the primary.
The cohort is unchanged: **four train and two validation** shots, **zero blind
test** shots. These are descriptive coincidences, not classification scores.
All inputs were local/read-only; no fetching, training or threshold tuning ran.

The newest complete source remains
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix2/shots`, with
**500 requested/500 processed** shots, no errors, and completion SHA-256
`79a2c18ed4aae29ba2f08aa8caaeff0c74c23cf7a24b4b150b6aa943500f10c4`.
The live completion and six shot hashes were rechecked and still match the
immutable figure-owned snapshot recorded in `sawtooth_source_manifest.json`:
`round4/fig1/saw_source/fix2-79a2c18ed4aa/shots`. Every final render and rebuild
uses that snapshot through **--sawtooth-source**. The sawtooth stream is still
regenerating; its newer incomplete output was not substituted. The controller
can rerender with its eventual complete output without changing renderer code.

Source for every count and window below: **fix_round5_audit.json**, generated by
the committed `scripts/labeler/paper/fig1_audit.py`, plus the six underlying
`<shot>.json` records in `outputs/labeler/paper/fig_interpreter_tokeye/`.

| Shot | Split | Window (ms) | AE components | NTM components | D-alpha triangles | Sawtooth row | Caption words |
|---|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 | 11 | 49 | omitted | 104 |
| 201973 | val | 1600–3350 | 154 | 14 | 65 | shown | 143 |
| 203187 | train | 1700–3150 | 262 | 8 | 6 | omitted | 72 |
| 186636 | val | 1300–3900 | 307 | 4 | 0 | omitted | 70 |
| 191376 | train | 1500–2900 | 0 | 0 | 2 | omitted | 32 |
| 191782 | train | 1800–3700 | 0 | 0 | 0 | omitted | 26 |

Large outputs under `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/`:
the primary `fig_interpreter.pdf`, `.png`, `.json`, `caption.tex`, and the same
four files in `alt_201973/`, `alt_203187/`, `alt_186636/`, `alt_191376/`,
`alt_191782/`. PDFs retain vector text/lines; PNGs are **150 dpi**.
Logs: `render5_<shot>.log`, `tests5.log`, `lint5.log`, `format5.log`, `audit5.log`.
The external and committed `fix_round5_audit.json` are identical. Small committed
outputs are the six records/captions, audit, plan, README and `fix_round5.md`.

All six final PNGs were opened and inspected: enlarged n ridges, visible n=3,
translucent NTM edges, off-data labels, keys/source text, ELM provenance and
sawtooth row visibility. All trial PNGs and the rebuilt primary PNG were also
opened. The audit checks every heading/key text bound lies inside the page.

### Draft caption

DIII-D shot 201978. Raw bands normalised separately; TokEye's U-Net extracts
coherent modes. Processed 30–55 kHz omitted. AE: neural interferometer detector
(p≥0.5; training targets used TokEye's mask); NTM: magnetic detector (unverified;
shared inputs); H-mode: D-alpha detector; ELMs: expert. Pink: AE time overlap in
detector band ≥80 kHz; tint follows the detector's 25 ms bins. n measured ≤30 kHz.
NTM outlines require dominant and pixel n=1/2. The n=2 ridge near 18 kHz is
consistent with a second harmonic of n=1. Late 110–188 kHz lines stay untagged
where AE detector is absent. circles: expert ELM interval (one span for many
ELMs); triangles: threshold D-alpha peaks (not annotated).

This is the primary's generated caption, copied from `201978.caption.tex`
without TeX wrapping; the controller can edit it for the manuscript. The source
JSON retains the frequency-ratio support behind the harmonic sentence.

### Deviations from the brief

**Sawtooth crashes are not projected onto rotating modes.** This explicitly
deviates from the owner's original request to project sawtooth detections below
60 kHz and, if present, illustrate sawtooth seeding an NTM. A crash is a point
event; overlap with an n=1 mode does not identify that mode as its precursor.
The available crash evidence does not establish the precursor or a causal
seeding relation. The figure therefore uses independently ECE-verified,
ELM-vetoed crash ticks, while the physics-state row is conditional on actual
PRESENT time. No seeding claim is fabricated for the primary, whose source
window has no PRESENT sawtooth time. `mode_tags.tag_blobs` excludes sawtooth
from mode tags; exact physics intervals and tick evidence remain in the records.

AE projection begins at **80 kHz**, rather than the original brief's 60 kHz,
because 80–250 kHz is the detector's input band. White 60–80 kHz structures are
outside that input band. The processed 30–55 kHz band is omitted to enlarge the
measured-n panel, as explicitly authorized in round five. The raw band remains.
The catalog sawtooth frame model is excluded: its output is not corroborated
crash/precursor evidence; that explanation belongs in this report/text, not in
the caption. The existing complete sawtooth snapshot is retained pending the
new source's completion, as authorized by the current rerender instruction.

### Verification and remaining work

Renderer records were generated externally from clean committed **21e72826**;
all six have `render_started_from_clean_head: true`, matching code hashes,
and the same source revision. After record commit **2965d609**, the final audit
and scratch rebuild ran from another clean committed HEAD. Nothing was pushed,
merged, rebased, written to the main checkout, or changed in the manuscript.

Covering tests, with required TMPDIR and local-data variables, through the
mandated wrapper:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_paper_label_figure.py
  -q -p no:cacheprovider
59 passed in 4.08s
```

Ruff check on the six changed Python files: **All checks passed!**
Ruff format --check: **6 files already formatted**. `git diff --check` passes.
New failure cases were observed before fixing n=3 pixels, regime serialization,
short-state display merging, conditional captions, late-line bounds/duration,
source cadence and components whose eligible interval contains only n=3.
No full suite ran, per the implementer rules. The mandated tmpsweep ran after
rendering and after the audit/rebuild.

```text
pixi run --frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary
  --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fix_round5_audit.json
Audited 6 non-blind renders; zero projection violations
Primary PDF and PNG rebuild identically byte-for-byte
```

Every projection audit has **zero** outside-PRESENT/band pixels; every NTM audit
has **zero** unmeasured, above-30-kHz or measured-n=3 outline pixels. Sources,
committed renderer hashes, figure hashes, regime categories, raw physics states,
display rows, conditional keys, caption lengths and external/committed copies
all match. The primary PDF and PNG hashes are identical before and after rebuild;
full hashes and the rebuild command are in `fix_round5_audit.json.reproducibility`.

No unresolved fifth-round implementation finding remains. The next action is
the controller's rerender after the sawtooth stream completes, followed by the
same audit and PNG inspection. Scientific limitations remain disclosed: NTM
is unverified, AE targets depend on TokEye, harmonic consistency does not identify
an island, and the density proxy limits local ECE assessment without proving
cutoff. A co-occurrence does not establish a precursor or causation.


---

## fix_round5_plan.md

# Figure 1 fix round 5 implementation plan

**Goal:** Address every Important and Minor item in both fifth reviews and the
owner's explicit round-five instructions.

**Architecture:** Keep scientific masks and source states separate from display
choices. The renderer consumes reusable source/display helpers, exports both
source and displayed intervals, and the audit checks the committed renderer.

**Spec:** `fig1-opus5.md`, `fig1-sol5.md` in the dispatch review directory and
the user's fifth-round instructions. Execute in this existing worktree.

**Constraints:** 6.75 × ≤5.5 inches; fonts ≥7 pt; read-only local data; no fetching;
only covering tests; committed code before final renders; requested Codex trailer.

1. Add regressions in `test_paper_figure_sources.py` and
   `test_paper_mode_tags.py`: NTM never covers n=3; confinement retains numeric
   categories and regime names; sawtooth display coalesces sub-10 ms intervals
   without changing source rows; late-line support requires 150 ms and derives
   frequency bounds; captions follow actually drawn elements and ELM provenance.
   Run the covering files to observe missing behavior, then implement helpers in
   `figure_sources.py` and the pixel restriction in `mode_tags.py`.
2. Update `fig_interpreter_tokeye.py`: enlarge the dedicated 0–30 kHz panel,
   omit processed 30–55 kHz, move keys/callouts to the right margin, use vermillion
   sawtooth and translucent one-pixel NTM edges, conditional sawtooth row and
   conditional legends. Export full source states, smoothed display states and
   geometry. Verify the primary PNG and covering tests/lint before source commit.
3. Update `fig1_audit.py` to verify row visibility, source/display separation,
   confinement categories, zero n=3 outlines, geometry and clean renderer HEAD.
   Snapshot newest complete sawtooth source; render all six from clean committed
   HEAD into external outputs and temporary records, inspect every PNG, then copy
   small records/captions and run the audit with byte-rebuild verification.
4. Append “Fix round 5” to the dispatch report, including draft caption,
   deviations, all six results with JSON sources, source freshness, PNG inspection,
   tests/lint and commit range. Update README/current-report pointers and commit
   records/report separately. No push or merge.

Review focus: source states stay exact; display smoothing is disclosed; hidden
sawtooth rows retain provenance; no n=3 NTM marks; detector ELMs have no expert
legend; late-edge slivers have no late-line caption; margin text fits at print size.


---

## fix_round6.md

# Figure 1 — fix round 6

Status: **DONE_WITH_CONCERNS**. Both `fig1-sol6.md` and `fig1-opus6.md` were
read in full. Every Important and Minor finding is addressed. Owner label
flags and the existing evidence limitations below remain; no implementation
finding is open.

Commit range: **d94df795..HEAD**, branch **r4-fig1**. Implementation:
**cd8e07fb**; final print-margin adjustment and renderer revision:
**d3c0519a**. The final artifact commit contains the six records/captions,
audit and this report. All commits use `labeler:` and the requested
`Co-Authored-By: Codex gpt-6.1-sol <noreply@openai.com>` trailer.

This is the single current report. Superseded rounds 4–5 moved to
`reports/fig1-appendix-rounds4-5.md` beside the dispatch report; older committed
`fix_round*.md` and audits are historical.

## Result and evidence

Primary **201978, 1500–3300 ms** retains the raw → TokEye → labels layout.
All six final renders were made externally from clean committed **d3c0519a**,
then their records/captions copied into this worktree together. Every final
PNG was opened at native **150 dpi**: 201978, 201973, 203187, 186636, 191376,
191782. The white dashed NTM contours are readily findable on the primary's
blue and green ridges; the same line style appears in the key. They are opaque,
**1.2 pt / 2.5 print pixels** wide, with a thin black halo. The AE callout sits
in the black region without a leader. The ECE key uses the open right margin
by NBI, clear of the ELM key. Curated L-mode bars/key are visibly darker than
absent grey in both regime alternates.

Scientific numbers below come from the committed generator
`scripts/labeler/paper/fig1_audit.py` and its current JSON:
`outputs/labeler/paper/fig_interpreter_tokeye/fix_round6_audit.json` (identical
external copy: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fix_round6_audit.json`). Per-shot source metadata,
thresholds, intervals, figures and hashes are in `<shot>.json` in the same
committed output directory. These are descriptive coincidences, not
classification scores or independent physical identifications.

| Shot | Split | Window (ms) | AE components | NTM components | D-alpha triangles | ECE-supported candidates | Caption words |
|---|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 | 11 | 49 | 0 | 120 |
| 201973 | val | 1600–3350 | 154 | 14 | 65 | 3 | 148 |
| 203187 | train | 1700–3150 | 262 | 8 | 6 | 0 | 104 |
| 186636 | val | 1300–3900 | 307 | 4 | 0 | 0 | 98 |
| 191376 | train | 1500–2900 | 0 | 0 | 2 | 0 | 44 |
| 191782 | train | 1800–3700 | 0 | 0 | 0 | 0 | 37 |

There are **four train, two validation and zero blind test** shots. The primary
still has **319 AE / 11 NTM** tagged components. Audit `renders[]` records
**zero** time/band projection violations, unmeasured NTM support and n=3
support marked as NTM, plus valid source/code/cache/figure hashes, text bounds,
font sizes, caption labels and external/committed copies. The primary PDF and
PNG rebuild **byte-for-byte identically** (`reproducibility`). Support-pixel
audits describe the region bordered by the contour; the visible stroke is
not interpreted as additional measured n.

The general harmonic caption sentence is **deleted**. For primary NTM time,
**318 / 684** columns with both measured
ridges pass **|f2/f1 − 2| ≤ 0.1**, a fraction of
**46.49%**. Passing support is **325.632 ms**
out of **700.416 ms** simultaneous support. The **50 ms**
floor alone did not justify the removed general claim. Source:
`renders[shot=201978].harmonic_support`; the audit now reports both counts,
the denominator duration and the passing fraction.

Late frequency bounds now include **all** coherent pixels during ABSENT AE
time after its last PRESENT interval in the detector band, with no component
duration cutoff. The primary spans **80.078–249.512 kHz**, captioned as
**80–250 kHz mask pixels**, without physically identifying them as lines.
Source: `renders[shot=201978].late_untagged_high_frequency` (exact bounds,
pixel counts and times). The same rule supplies the conditional alternate
captions.

## Sources and changes

`figure_sources.sawtooth_track` uses the expert review from the first
`lf.TRACKS` source before physics states. `shot_tracks` uses that choice, and
`draw` preserves it. **186636** now displays the expert review from
`data/events/sawtooth_oscillation/review/labels.csv`, with **1884 ms present**
and **716 ms absent** in the window; the imported NTM interval remains inside
the reviewed sawtooth period. No seeding relation is inferred. Interval and
candidate-tick provenance are recorded separately. Other shots retain the
pinned physics snapshot; hidden rows get the caption clause **“Sawtooth: no
present time in this window.”** Source: each audit `sawtooth_interval_source`,
`sawtooth_states`, `sawtooth_state_duration_ms` and `sawtooth_track_shown`.

**201973** retains **3 ECE-supported crash candidates**, described with the
channel-order-only geometry caveat. They are research candidates, not validated
crashes. The audit's legacy `verified_crashes` count names these evidence-checked
candidates; it is not an independent validation claim. Candidate evidence uses
the immutable figure-owned physics snapshot
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/saw_source/fix2-79a2c18ed4aa/shots`, pinned by
`sawtooth_source_manifest.json`. No production label or physics source was edited.

`figure_sources.tokeye_fingerprints` records actual trimmed float32 waveform
samples, timing/rate/channel identity, preprocessing and inference source
hashes, and NumPy/SciPy/PyTorch versions. `tokeye_passes` validates these together
with the checkpoint before either exact or superset cache reuse. Missing or
mismatched metadata regenerates arrays. All six caches were regenerated from
local read-only inputs; each NPZ carries the fingerprints. Their digests and
fingerprints are checked again and included in audit `renders[].tokeye_cache`
and `checked_sources`. Primary cache SHA-256:
`f63e81f5fa115c3b085db16d59d4d058823f04f67eb8aea8a7ef27bcf05653a5`.

Other presentation fixes: **“Raw signals”** heading, explicitly labelled
**Mirnov magnetics** spectrogram, **D-alpha (a.u.)**, **n=1 or 2** throughout
current keys/prose, reader language for event overlap and ELM symbols, and
regime keys with darker L-mode bars. NTM contour support still requires
both unique dominant n and individual measured pixels to be 1 or 2 at
≤30 kHz. AE tint starts at ≥80 kHz and follows the source's recorded cadence.

Changed code: `src/labeler/paper/figure_sources.py`,
`scripts/labeler/paper/fig_interpreter_tokeye.py`,
`scripts/labeler/paper/fig1_audit.py`; covering regressions in
`tests/labeler/test_paper_figure_sources.py`. README contains reproducible
commands. No dependencies, labels, cohort, model thresholds or manuscript changed.

## Owner flags and limits

- **201978:** the first large D-alpha spike is at **2296.54 ms**, **11.46 ms**
  before the expert ELM interval starts at **2308 ms**. Caption flags it;
  triangles remain restricted to expert PRESENT times. Source: audit
  `largest_dalpha_peak_ms`, `first_large_peak_before_expert_ms`, and the render's
  expert ELM intervals. Labels were not altered.
- **201973:** the currently consumed H-mode source is ABSENT from **3013 ms**,
  yet expert ELM PRESENT overlaps **3013–3045**, **3077–3124**, and
  **3313–3350 ms** in this window. This is the requested H-mode/ELM contradiction,
  with exact current-source times rather than the review's approximate times.
  Source: audit `elm_hmode_conflicts_ms` and `confinement_intervals_ms`.
- The same automatic audit also flags **203187**: its largest plotted D-alpha
  spike at **2257.49 ms** precedes the first expert ELM interval by **274.51 ms**;
  ELM/H-mode-ABSENT overlaps are **2793–2856** and **3036–3080 ms**. These are
  owner review flags, not automatic ELM identifications or label edits.

NTM remains unverified and shares magnetic inputs; AE training targets used
TokEye's mask; ECE candidate geometry is not spatially validated. Sawtooth
crashes remain point candidates rather than rotating-mode projections, since
time overlap cannot establish a precursor or NTM seeding. This retains the
scientific deviation from the original brief. AE's ≥80 kHz floor and omission
of processed 30–55 kHz are retained from the reviewed design. No fetching,
training, threshold tuning or blind-test selection occurred. The primary
remains selected for its clear cascade and measured low-frequency structure;
201973/203187 are cascade alternates, 186636 adds reviewed/imported provenance,
and 191376/191782 are regime/diagnostic comparisons. Historical selection
criteria/ranking remain in the committed `fix_round2_audit.json`.

## Outputs and verification

Primary artifacts: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fig_interpreter.{pdf,png,json}` and `caption.tex`.
Each of `alt_201973`, `alt_203187`, `alt_186636`, `alt_191376`, `alt_191782`
contains the same four artifacts. PDFs are **6.75 × 5.5 inches**, vector
text/lines over raster spectra, with **≥7 pt** fonts; PNGs are **150 dpi**.
The six small JSON/caption copies, current audit, README and `fix_round6.md`
are committed. Large figures/caches stay outside git.

Required TMPDIR/local-data variables and `LABELER_NO_FETCH=1` were set. Python
used frozen/no-install pixi through the main manifest from this worktree.
Covering tests only, via the mandated wrapper:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_paper_label_figure.py
  -q -p no:cacheprovider
63 passed in 4.20s
```

Fresh Ruff check of all four changed Python files: **All checks passed!**
Ruff format --check: **4 files already formatted**. `git diff --check` passed.
All six generated captions also compiled with a plain LaTeX/T1 smoke document;
log: `round4/fig1/caption6-smoke.log`.
New regressions were first observed failing for review precedence, waveform/
code fingerprints, the harmonic fraction, all-pixel late bounds and disclosures.
Full logs: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/tests6{,-red}.log`, `lint6.log`, `format6.log`,
`render6_<shot>.log`, `cache6_201978.log`, `audit6.log`.

```text
pixi run --frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary
  --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fix_round6_audit.json
Audited 6 non-blind renders; zero projection violations
Primary PDF and PNG rebuild identically byte-for-byte
```

The mandated tmpsweep ran after the inference/render batches and audit/rebuild.
Nothing was pushed, merged, rebased or edited in the main checkout/manuscript.
Next: the controller can place the primary with its current caption and send
the recorded label discrepancies to the owner for review; no automatic label
correction is warranted.

## Current primary caption

DIII-D shot 201978. Raw bands normalised separately; TokEye's U-Net extracts coherent modes. Processed 30–55 kHz omitted. AE: neural interferometer detector (p≥0.5; training targets used TokEye's mask); NTM: magnetic detector (unverified; shared inputs); H-mode: D-alpha detector; ELMs: expert. Pink: AE time overlap in detector band ≥80 kHz; tint follows the detector's 25 ms bins. Toroidal mode number n is measured ≤30 kHz. Dashed outlines mark NTM time overlap on measured n=1 or 2 ridges whose dominant n is 1 or 2. Sawtooth: no present time in this window. Late 80–250 kHz mask pixels stay untagged where AE is absent. The first large spike precedes the expert interval. Circles mark expert intervals spanning many ELMs; triangles mark threshold D-alpha peaks (not annotations).
