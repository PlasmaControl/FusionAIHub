# Figure 1 — current state after fix round 4

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
