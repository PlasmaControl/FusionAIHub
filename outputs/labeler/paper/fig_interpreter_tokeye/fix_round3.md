# Figure 1 — current state after final fix

Status: **DONE**. Primary **201978**, **1500–3300 ms**. All Important and Minor
findings in `fig1-sol3.md` and `fig1-opus3.md` were read in full and addressed.
This report supersedes the earlier fix reports; only the short appendix
below describes fix history.

Commit range: **4aa6b8c0..HEAD** on **r4-fig1**. Source commits:
**eadf8b88**, **1bcf78ea**, **b7639a4b**. The artifact commit containing this
report completes the range. Every render records **b7639a4b** as its source.
All new commits have the requested `labeler:` prefix and Codex gpt-6.1-sol
co-author trailer. Nothing was pushed, merged or changed in the main checkout
or `dev/label_paper`.

## What the figure shows

The raw Mirnov spectrogram, D-alpha and NBI power lead to the TokEye coherent
mode mask and measured toroidal mode numbers, then aligned AE, NTM, H-mode
and ELM source tracks. The primary window preserves the cascade, H-mode
transition, ELM onset and emerging low-frequency mode with harmonics. The
Mirnov label is in the raw heading, clear of the cascade. Frequency resolution
changes at 55 kHz; this is separate from the 60 kHz tagging rule.

**AE boxes were removed; clipped pixel tint remains.** This avoids bounds
set by stray pixels and empty late boxes. The shorter primary window removes
the late high-frequency comb and its disputed late AE highlights. The AE
track still displays the detector's complete states within the new window.

The processed D-alpha panel has one **ELMs** label on the largest visible
interval, an **ELM intervals** key and separate peak triangles. Uncertain
intervals are hatched. Expert crowd rows retain category: category 2 remains
UNCERTAIN, with its crowd marker above the hatching. In 186636 the uncertain
crowd interval is consequently hatched and supplies no definite triangles.
Blank means **unassessed / unobservable**.

The magnetic NTM detector is an **unverified suggestion**, with **detector**
below its bar; its record/caption disclose the failed acceptance bar and
shared displayed magnetic inputs. Eligibility uses a component's unique
dominant measured n=1 or n=2. Outlines then intersect PRESENT time, the
nominal <60 kHz band and actual measured-n pixels at ≤30 kHz. The pooled
inner edge is intersected again with the native eligible mask, so it cannot
extend into unmeasured pixels. Components dominated by n=3 are unoutlined;
measured n=3 pixels inside a component dominated by n=1/2 may remain outlined.
That distinction and the measured-n3 outline count are recorded explicitly.

Sawtooth is not assessed in the primary because the physics source's density
guard marks ECE cutoff during ELMy H-mode. The current renders omit the row
and crash strip. The read-only source/evidence parameters remain available;
`--show-sawtooth` enables the row for a suitable later shot. Enabling it keeps
the source's PRESENT/UNCERTAIN/ABSENT/unobservable categories unchanged.
Interval state and independently verified crash ticks are separate evidence.

## Sources, data and thresholds

All data were local and read-only. No network, training, threshold tuning,
production-label writes or blind-test-shot selection occurred. The six
records cover four train and two validation shots; the audit's
`blind_test_shots_used` is zero. Figure counts below are descriptive
component coincidences and marker counts, not evaluation scores.

| Shots | AE source | NTM source | Regime source | ELM source |
|---|---|---|---|---|
| 201978, 201973, 203187 | supplied paper neural CO2-interferometer activity | magnetic frame detector, failed acceptance | D-alpha L-H detector | expert review |
| 186636 | earlier CO2 cross-power frame detector, raw-cache CO2 | imported tearing archive | D-alpha detector, uncertain in view | expert review, uncertain crowd category retained |
| 191376, 191782 | earlier CO2 cross-power frame detector | magnetic frame detector, failed acceptance | Gill/Jalal Butt curated regimes | frame detector |

Exact source paths, metadata, hashes, tiers and intervals are in each
`<shot>.json`, `tracks`, `drawn.stores`, `drawn.sawtooth_crashes` and
`ae_ours_lookup`. The physics inputs are the read-only shot JSONs under
`$LABELER_ROOT/round4/saw/fix/shots`. The primary's source JSON records the
cutoff proxy in `density_guard`. All consumed source/evidence hashes checked
by the audit match disk.

`fig1_ae_infer.py` calls the model adapter's `spec.load`, `INPUT_SPEC.build`
and `predict` on CPU, replacing duplicated inference. Isolated probabilities,
validity, local CO2 hashes and the pinned checkpoint digest are in
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/ae_ours.meta.json`; predictions are in `ae_ours.csv` beside it.
`figure_sources.AE_THRESHOLD` is the single figure operating-point constant
imported by inference: **p≥0.5**, from the SELDnet operating point in
`scripts/labeler/ae_baselines_evaluate.py`. Causal 25-ms bins and invalid-bin
states are preserved. The earlier frame fallback is a different model;
its own checkpoint threshold **0.7** is retained and recorded per track,
rather than attributed to the paper model. This is disclosed in alternate
captions and `ae_threshold_scope`.

Other fixed settings: NTM **0.63**, explicit sawtooth confidence **0.6**,
TokEye coherent **≥0.2** with transient **<0.2**, persistent-line candidate
row share **>0.8**, object removal **40 px zoom / 30 px wide**, hole filling
**10 px**, then 8-connected components. These are written by the generator to
`decision_thresholds` and `filter`. They were not tuned in this fix.

Harmonic prose is computed from each record, without the old shot-specific
frequency bands. Per-column measured-n frequency summaries are compared at
2:1 within **1.2 kHz**, requiring
**50 ms** sampled support. Primary support is
**402.432 ms**; frequency medians are
**8.099 / 16.052 kHz**. The same
recorded **5 kHz** prose rounding on
every shot yields “near 15 kHz” in the primary. The sentence says
“is consistent with a second harmonic of the n=1 ridge,” without claiming a
common island. Source: `fix_round3_audit.json`, primary `harmonic_support`,
and `201978.json`, `drawn.harmonic_support`.

## Current outputs and audit

Numbers below come from the committed `fig1_audit.py` and its
`outputs/labeler/paper/fig_interpreter_tokeye/fix_round3_audit.json` output.
Caption word counts exclude TeX wrappers and labels.

| Shot | Split | Window (ms) | AE components | NTM components | D-alpha markers | Caption words |
|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 365 | 12 | 49 | 120 |
| 201973 | val | 1600–3350 | 178 | 14 | 65 | 100 |
| 203187 | train | 1700–3150 | 287 | 8 | 6 | 100 |
| 186636 | val | 1300–3900 | 358 | 5 | 0 | 81 |
| 191376 | train | 1500–2900 | 0 | 0 | 2 | 82 |
| 191782 | train | 1800–3700 | 0 | 0 | 0 | 70 |

All six have **zero** projected pixels outside raw end-exclusive PRESENT
intervals or event bands, **zero** rendered NTM outline pixels without
measured n or above the n-map band, and no AE boxes. The primary rebuilt
**PDF and PNG byte-for-byte**; before/after hashes and exact command are in
`fix_round3_audit.json`, `reproducibility`. The audit also checks source
hashes, unique caption labels, record/caption identity between external and
worktree copies, non-blind splits and actual PDF page size.

Files under `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/`:

- Primary: `fig_interpreter.pdf`, `fig_interpreter.png`,
  `fig_interpreter.json`, `caption.tex`.
- The same four products in `alt_201973/`, `alt_203187/`, `alt_186636/`,
  `alt_191376/`, `alt_191782/`.
- `ae_ours.csv`, `ae_ours.meta.json`, `cache/`, `infer3.log`,
  `render3_<shot>.log`, `audit3.log`.

Small committed outputs: six records and exact caption copies, the current
README, this report copy `fix_round3.md`, and `fix_round3_audit.json`. The old
report copies now point here; historical audit snapshots remain clearly
identified. No large generated plot or data file is committed.

All six final PNGs were opened at native 150-dpi size after regeneration.
ELM labels are clear in 201973 and 191376; 186636 shows uncertainty with a
crowd marker; the primary shows the clipped orange low-frequency outlines
and the pink cascade. Caption length is ≤150 words, with a TokEye definition,
what to follow, and source/tier facts. Only the primary uses
`fig:interpreter`; every alternate has `fig:interpreter-<shot>`.
PDFs are **6.75 × 5.5 inches**, with vector text/lines over raster spectra;
fonts are **≥7 pt**. Include at **textwidth** to preserve that size.

## Implementation and verification

- `src/labeler/paper/figure_sources.py`: paper threshold, category preservation,
  source descriptions, optional physics state track and data-derived caption.
- `src/labeler/paper/mode_tags.py`: measured-pixel NTM projection.
- `scripts/labeler/paper/fig1_ae_infer.py`: adapter-based isolated inference.
- `scripts/labeler/paper/fig_interpreter_tokeye.py`: shortened preset, optional
  sawtooth, tint, measured-pixel inner outline, representative ELM label,
  source tier, unique TeX label and print layout. The primary reuses the pinned
  full-record TokEye cache; a shorter view does not change model inputs.
- `scripts/labeler/paper/fig1_audit.py`: sources, projection support, captions,
  print dimensions and reproducible rebuilding.
- Covering regressions: `tests/labeler/test_paper_figure_sources.py` and
  `tests/labeler/test_paper_mode_tags.py`. The existing adapter tests also run.

Required TMPDIR and local data variables were set, with LABELER_NO_FETCH=1.
All Python ran via frozen/no-install main-manifest pixi in this worktree.
Covering tests only, through the mandated wrapper:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_ae_adapter.py
  -q -p no:cacheprovider
.................................................. [100%]
50 passed in 14.60s
```

The final caption-only spelling refinement then reran its covering source
file: **18 passed in 2.93s**. Ruff check on all seven changed Python files:
**All checks passed!** Ruff format --check: **7 files already formatted**;
the final source refinement separately passed both checks. No full suite
was run, per the implementer rules.

Audit output:

```text
Audited 6 non-blind renders; zero projection violations
Primary PDF and PNG rebuild identically byte-for-byte
```

Long render jobs ran the mandated tmpsweep script. It completed without
losing generated files. A read-only code review found no Critical or
Important issue; its two Minor findings (ELM text overlap and the scope of
n=3 prose) were fixed, re-reviewed with no residual findings, and rerendered.
This does not claim fresh Sol/Opus paper-review scores.

## Limits and controller action

NTM remains an unverified suggestion sharing the shown magnetic inputs;
frequency coincidence does not establish an island. Sawtooth is not assessed
in this primary window. These are visible source limits, not unresolved
implementation work. No open fix-round defect remains. The controller can
place the primary at textwidth with its current caption; a later shot with
verified ECE crashes can use the optional physics source and row.

## Appendix: brief fix history

1. First fixes (`d8e8dc1..e5a56995`): raw time/band clipping, n-map provenance,
   source priority, confinement fallback, audit records and reproducible builds.
2. Second fixes (`e5a56995..4aa6b8c0`): ECE evidence and ELM-vetoed crash ticks,
   source-specific captions and print-visible outlines. Its threshold, crowd
   and sawtooth category choices are superseded by the current source semantics.
3. Final fixes (`4aa6b8c0..HEAD`): paper operating point and adapter, shorter
   primary, tint without boxes, optional sawtooth with unchanged categories,
   uncertain crowd rows, labelled/keyed ELM intervals, unverified NTM tier,
   measured-pixel clipping, computed harmonic wording, unique captions,
   smaller print height, all-PNG inspection and current-state reporting.
