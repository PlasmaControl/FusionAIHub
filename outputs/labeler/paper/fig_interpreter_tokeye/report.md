# Figure 1 — tenth fix round

Status: **DONE_WITH_CONCERNS**. Primary remains **201978, 1500–3300 ms**; data, windows,
detector outputs and the sawtooth source pin (**ad0ca40f**) are unchanged.
I read the brief and both re-reviews, Sol10 (8/8/8/8) and Opus10 (7/8/8/8), and addressed
every Important and Minor finding in both. This is the single current report. It is
mirrored byte-for-byte at the dispatch `reports/fig1.md` and at this worktree's
`outputs/labeler/paper/fig_interpreter_tokeye/report.md`. Earlier rounds are in git;
`docs/labeler/fig1_fix_history.md` is now a short index to them.

Commit range **1fdf3762..HEAD**, branch **r4-fig1**:

| Commit | What |
|---|---|
| `9284525d` | Caption and appendix rewrite, AE-source branching, shot-specific claims out of the library, display rules, tests, audit |
| `10ee50a9` | Measured first-line anchoring of the multi-line sawtooth source text |
| `c89ec51f` | n=1 frequency stated beside the n=3 harmonic in the appendix (render source) |
| artifact commit | Six sidecar sets, `audit.json`, this report, the README, the history index |

All six renders began from the clean commit **c89ec51f** (`render_started_from_clean_head`
is true in each record). Every commit uses `labeler:` and
`Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Fix round 10

**Primary caption (item 1).** The caption is the descriptive text the brief gave, **92
words** (the code cap is now 120; see deviations). Source of the text: the committed
`caption()` in `src/labeler/paper/figure_sources.py`, written to `201978.caption.tex`:

```tex
\caption{DIII-D shot 201978. Top: raw Mirnov spectrogram (axis split at 30 and 55 kHz; bands normalised separately), D-alpha, NBI power. Middle: TokEye coherent-mode mask after small-object removal; below 30 kHz coloured by toroidal mode number $n$ (Mirnov array). Pink: mask pixels $\geq$80 kHz while the CO2 AE detector is positive (25 ms bins). Orange outlines: $n=1/2$ pixels while the NTM detector (held-out F1 0.46, below our 0.7 bar) is positive. Highlights mark time/band coincidence only; $n=2$ and $n=3$ ridges are consistent with harmonics of the $n=1$ mode. Bottom: label tracks with sources.}
\label{fig:interpreter-201978}
```

The "first ELM" sentence is gone from the caption; the appendix now says *The largest
D-alpha spike (2297 ms) precedes the expert span (from 2308 ms)*. The words "below bar",
"circularity", "four-state" and "first ELM" are forbidden by the audit in every caption.
The pink bin width (25 ms) comes from the AE record (10 ms on the frame-detector view of
186636). The NTM sentence takes its F1 and bar from the detector's evaluation record
(`ntm_qualification`). The "n=2 and n=3 ridges" clause is computed from the ridge-ratio
support, not asserted (item 3).

**AE-source branching (item 2).** The appendix branches on the AE source. For `ae-ours`
(201978, 201973, 203187) it says the 80 kHz floor matches the ae-ours input band and that
AE targets used TokEye's mask. For the CO2 frame detector (186636) it says the frame
detector's input band applies, that the model was trained on the owner's reviewed AE
labels, and that TokEye's mask only up-weights its MHD-absent frames. Thresholds are
listed only for detector tracks: TokEye 0.2, plus AE (0.5 for ae-ours, 0.7 for the
frame detector) and NTM 0.63 only where that track is a detector. On 186636 the NTM
track is imported labels, so its appendix lists "TokEye 0.2; AE 0.7" and no NTM
threshold.
The persistent-row (pickup) step is stated in the appendix mask chain, with the counts of
rows that reached the persistent share per pass (201973 6 wide / 0 zoom; 203187 18 / 4;
191376 6 / 17; 191782 0 / 11; 201978 and 186636 none). Alternate captions carry their
qualifications: 201973 and 203187 the ELM/H-mode disagreement, 203187 the post-NBI AE
suggestions, 186636 the possible pickup lines. Sources: `<shot>.appendix.txt`,
`<shot>.json:drawn.persistent_line_rows`.

**No shot-specific claims in the library (item 3).** `figure_sources.py` has no
`shot ==` branch. The AE leader anchor lives in `docs/labeler/fig1_annotations.json`
(201978: 2100 ms, 200 kHz). Everything else is computed from the records: the harmonic
clause from the measured-n ridges (`harmonic_support`, order 2 and order 3), the pickup
caveat from the persistent-row counts, the sawtooth row source from the source states and
the ECE cut-off proxy, the NTM qualification from the detector's evaluation record. A test
(`test_figure_sources_have_no_shot_specific_branches`) fails if a shot number returns.

**Display (item 4).**

- The AE leader is anchored at **2100 ms / 200 kHz** in an early cascade line and runs
  below the dense late white lines (crop inspected). The audit requires the anchor inside
  a detector-positive span and more than 500 ms before the first late untagged time.
- A display-only **minimum-size rule**: NTM outline regions under **100 print pixels**
  after max-pooling and hole-filling are not drawn. The tiny 3–5 kHz fragments (1–67 px)
  are not outlined; the three main regions (169, 2931 and 3166 px) stay. The tags, counts
  and audit masks use native pixels and are unchanged. Counts omitted per shot: 9 (201978),
  9 (201973), 7 (203187), 3 (186636), 0, 0. The rule is recorded in
  `<shot>.json:filter.outline_display_rule`, `drawn.ntm_outline_display` and the appendix.
- Sawtooth row source text, where unassessed time exists and the ECE cut-off proxy is
  recorded: "physics labels / blank: not assessable / (ECE cut-off)". Elsewhere it reads
  "physics labels". The text is anchored on the measured height of its first line and the
  audit checks that no source-text box overlaps another or leaves the figure.
- The NTM source column reads **detector (suggestions)**. The caption and appendix carry
  the F1, shot count and bar.

**Reports (item 5).** This file holds only the current report. The history is in git, and
`docs/labeler/fig1_fix_history.md` is a short index with the commands.

## Results

All numbers come from `audit.json:renders[]` and the per-shot records, from the committed
renderer and `scripts/labeler/paper/fig1_audit.py`. Component and peak counts are
overlap/display descriptors, not classification scores.

| Shot | Split | Window (ms) | AE / NTM components | D-alpha peaks | Caption words | Pink px | NTM fragments omitted |
|---|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 / 11 | 49 | 92 | 6005 | 9 |
| 201973 | val | 1600–3350 | 154 / 14 | 65 | 105 | 3987 | 9 |
| 203187 | train | 1700–3150 | 262 / 8 | 6 | 112 | 6062 | 7 |
| 186636 | val | 1300–3900 | 307 / 4 | 0 | 77 | 2080 | 3 |
| 191376 | train | 1500–2900 | 0 / 0 | 2 | 43 | 0 | 0 |
| 191782 | train | 1800–3700 | 0 / 0 | 0 | 43 | 0 | 0 |

There are four train and two validation shots and **zero blind-test shots**. The counts
equal round 9 (`1fdf3762`) exactly. The primary's NTM detector: F1 0.457 on its 761
held-out shots, precision 0.418, recall 0.506, against the N1 bar (F1 ≥ 0.7, precision ≥ 0.6,
recall ≥ 0.6), so the caption says "below our 0.7 bar". Source:
`201978.json:drawn.ntm_performance`, evaluation
`models/ntm_frames/v2/evaluation.json` (sha256 in the record).

Harmonic support (ridge ratio |f_n/f_1 − n| ≤ 0.05·n, at least 50 ms): on 201978 the n=2
ridge has 326 ms of joint support (n=1 at 9.1 kHz, n=2 at 18.5 kHz) and the n=3 ridge has
232 ms (n=1 at 7.8 kHz, n=3 at 23.0 kHz, over the columns where both are measured).
Source: `201978.json:drawn.harmonic_support` and `harmonic3_support`.

The sawtooth row in the primary has **832.1 ms uncertain**, **967.9 ms unassessed** and no
present or absent time (`audit.json:renders[shot=201978].sawtooth_state_duration_ms`).
No sawtooth is labelled present in this window.

`--baseline-ref 41ba206779910882b9a826fe668970da848f38af` compares the scientific
records with round 8. All six shots and windows, source tracks and categories, detector
predictions and operating points, checkpoint/cache hashes, training lists, filters, mode
counts and measured-n support, ELM peaks and late-line support are unchanged. The audit
allows only the display records (`ntm_outline_display`, `ntm_measured_pixel_audit`,
`filter.outline_display_rule`) and two added records (`harmonic3_support`,
`persistent_line_rows`) to differ. The AE and NTM detectors' training still excludes
201978 (`201978.json:detector_training`).

## Artifacts and verification

All six native 150-dpi PNGs from the final render (clean commit c89ec51f) were opened and
inspected: 201978, 201973, 203187, 186636, 191376 and 191782, plus a crop of the primary's
AE leader. In 191376 and 191782 the AE and NTM rows are detector tracks that are
negative throughout, which is why those appendices still list the detector thresholds.
PDFs are **6.75 × 5.5 in**, fonts ≥ 7 pt.
Primary: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fig_interpreter.{pdf,png,json}`
with `caption.tex` and `appendix.txt`; each `alt_<shot>/` holds the matching files. Large
files stay outside git.

Covering tests only:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_paper_label_figure.py
  -q -p no:cacheprovider
78 passed
```

Ruff check on `figure_sources.py`, `fig_interpreter_tokeye.py`, `fig1_audit.py` and
`test_paper_figure_sources.py`: **All checks passed!** Ruff format --check: **4 files
already formatted**. `git diff --check` passes. All six captions compile in a plain
LaTeX/T1 document (`fix10-caption-smoke.log`).

```text
pixi run --frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary
  --baseline-ref 41ba206779910882b9a826fe668970da848f38af
  --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/audit.json
Audited 6 non-blind renders; zero array/raster AE clipping violations
Primary PDF and PNG rebuild identically byte-for-byte
```

Logs under `round4/fig1/`: `fix10-render.log`, `fix10-audit.log`, `fix10-tests.log`,
`fix10-lint.log`, `fix10-format.log`. The audit also checks the new items: AE anchor
position, source-text bounds and exact strings, the conditional thresholds, the AE-source
branching, the persistent-row recount from the cache and the display-rule record. No
model inference rerun, fetching, training, source or data edit, or manuscript edit was
needed. All commands used the stream TMPDIR, `LABELER_NO_FETCH=1` and frozen/no-install
pixi from this worktree. `tmpsweep.sh` ran after the render batches. Nothing was pushed,
merged or rebased.

## Deviations from the brief

1. **"validation F1 0.46" became "held-out F1 0.46".** The detector's evaluation record
   gives F1 0.457 on the detector's own 761 held-out test shots (not its validation
   shots, and not the cohort's blind split). "Validation" would have misnamed that set,
   so the caption and appendix say "held-out" (and, in the appendix, the 761 shots). The
   0.46 and the 0.7 bar are as in the brief.
2. **The harmonic clause is computed, not fixed text.** The brief's "n=2 and n=3 ridges
   are consistent with harmonics of the n=1 mode" appears on 201978 because the data
   support it (n=2 326 ms, n=3 232 ms, both over the 50 ms floor). 201973 shows both too.
   203187 shows only n=3, so its caption says "the $n=3$ ridge is consistent with a
   harmonic of the $n=1$ mode"; 186636 and the two diagnostics show none and say nothing.
3. **The chain note is unchanged; the persistent-row step is in the appendix only.** The
   figure's in-panel chain note ("mask → remove small objects / fill holes → components →
   time/band tags") stays short for print width; the appendix states the step with its
   counts. The brief allowed either place.
4. **Caption word cap raised from 95 to 120** (`CAPTION_MAX_WORDS`). The brief's caption
   is 92 words with the label and alternates reach 112; the old 95 cap was a round-9
   artefact for the old shorthand caption.
5. **Commit trailer.** The brief names the Codex trailer; the binding rules file names
   `Co-Authored-By: Claude Sonnet 5.5`, which I used.
6. **The leader crosses the panel.** The AE leader must run from the right-margin
   legend to a point at 2100 ms, so it crosses the mask panel. It passes below the dense
   late white lines (checked in a crop) and covers only a thin strip, so I treated this
   as meeting the brief's intent. It cannot avoid the panel without moving the legend.
7. **Sawtooth is not projected onto the modes (physics reason).** Sawtooth is a
   point-event row in the label tracks and never a mode tag. The sawtooth labels come
   from the ECE core-temperature crash (timing and, where the ECE cut-off allows, the
   density proxy), not from the magnetic spectrogram. The m/n = 1/1 precursor and
   postcursor activity that can accompany a crash is ordinary n=1 structure that the
   mode colouring already shows, but these labels do not measure it. Painting mask
   pixels "sawtooth" would therefore assert a magnetic signature that no label
   supports. The sawtooth entry in the tag counts is 0 by construction. Where the physics
   labels cannot be assessed (the ECE cut-off), the row says so.

## Concerns

No blocker remains. Disclosed limits remain: the NTM suggestions are below acceptance
(held-out F1 0.46 against a 0.7 bar); AE targets used TokEye's mask and the NTM detector
shares its magnetic inputs; the physics sawtooth labels are unvalidated research evidence
(`validation_status` in the source manifest); the harmonic wording is a consistency
statement, not a mode identification. The caption on 203187 is the longest at 112 words.
The controller should re-pin the sawtooth source at integration if the saw stream
moves. The README and this report name the same render commit (c89ec51f).

## Current primary caption

See the block above (92 words; `201978.caption.tex`).
