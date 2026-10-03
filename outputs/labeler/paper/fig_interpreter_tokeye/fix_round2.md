## Fix round 2

Status: **DONE_WITH_CONCERNS**. Code commit **b3d1a1e2** on `r4-fig1`,
prefix `labeler:`, with the requested Codex gpt-6.1-sol co-author trailer.
The regenerated records/captions, audit and this report are in the accompanying
artifact commit; the complete range is recorded in the dispatch and below in
this controller report after that commit. No manuscript, production label
store, checkpoint, source data, main checkout or saw-stream file was modified.
No fetching, training, pushing, merging or rebasing occurred.

Both fig1-opus2.md and fig1-sol2.md were read in full. This section supersedes
the previous fix's assertions about verified sawtooth, source priority,
caption specificity, hatch semantics and collision-free layout. No new
original-reviewer scores are asserted.

### Finding → change

| Re-review finding | Change and evidence |
|---|---|
| Critical: SXR/ELM-driven ticks called sawtooth | Replaced the union with full ECE core-drop plus spatial inversion/rise evidence, an accepted physics state, and an inclusive ±5-ms D-alpha spike veto independent of ELM label spans. Recorded physics ELM coincidence also vetoes a point. SXR alone never supplies a tick; missing D-alpha cannot establish rejection. The current physics JSON source supplies no surviving verified crash in any displayed window. Every crash strip is removed; no crash leader remains. Primary removal is recorded as 52 → 0 in fix_round2_audit.json. |
| Sawtooth interval track follows the ELMy phase | The old interval model is no longer loaded for this figure. Track and ticks share the same read-only physics source. Only surviving verified point events can be present; other positive intervals remain candidates. With no survivors the visible title is “sawtooth cand.” The current source's assessed absence/uncertainty/unassessed coverage is retained honestly. |
| Need a rerenderable improved physics source | --sawtooth-source accepts a shot JSON, a JSON directory, cohort CSV or shard directory, and minimal shot,t_ms CSV. --sawtooth-evidence pairs reduced cohort CSV point rows with full JSON evidence from the same saw run. The README documents the actual saw-worktree directory and round4/saw/fix/shots recipe. All consumed source/evidence files are hashed. Only export-rounded point matches within 0.5 ms are joined; without full evidence, reduced/minimal CSVs require local ECE corroboration within 3 ms. |
| Hatching conflates uncertainty and expert ELMing periods | Expert iscrowd=1 rows (including category-2 crowd rows such as 186636) are normalized to ELMing periods and drawn solid with a white line/circle mark. Ordinary category-2 rows remain uncertain. Only uncertainty is hatched; unassessed states are blank. The key separately identifies expert ELMing periods. |
| Expert-uncertain 201978 4486–4599 ms gets definite ELM highlights | That span has a lighter hatched box and no filled triangles. The source rows are retained; definite peak markers are confined to definite/period ELM coverage. Peak times and uncertain spans are recorded and checked independently by fig1_audit.py. |
| Peak triangles mistaken for labels | The figure key and applicable captions explicitly say “D-alpha peaks, figure threshold.” The JSON records the 25-sample median, 4-MAD threshold and source trace; the figure code uses 3-sample peak spacing. They are descriptive figure markers, not catalog labels. |
| AE model is the wrong fallback, and expert priority is reversed | Added fig1_ae_infer.py and ran the pinned paper ae-ours checkpoint, inference only, on GPU 0 using corpus CO2 on 201978, 201973 and 203187. Expert review now outranks supplied predictions, which outrank valid stored paper predictions, followed by the earlier frame fallback. Fixed AE threshold is 0.7. The old fallback is still named honestly on the older/regime alternatives where the corpus CO2 sentinel prevents this card's inference. |
| “n=1 tearing mode” overstates evidence | The primary/other generated-source leader is “n=1 mode; NTM prediction.” The imported archive alternate says “NTM label.” There is no definite tearing-mode or seeding claim. |
| Captions are fixed, long, internally named, and transfer failed bars to unrelated sources | Captions are built from the actual track records, their titles and primary_bars. Imported/expert sources never inherit frame-model failure statements. The generated NTM source's failed acceptance check is disclosed in plain language; shared-input wording is conditional on that generated source. No internal identifiers or acceptance-bar codes appear in captions. The confinement title is H-mode or regime as actually selected; 186636 explicitly says uncertain here. |
| n=2 near 15 kHz is a harmonic, not an independent island | Caption wording is conditional: “Under an island interpretation, outlined n=2 near 15 kHz is a harmonic of the same island.” It appears only with recorded simultaneous measured n=1/n=2 frequency support near a 2:1 ratio, rather than merely two disconnected component labels. This is an interpretation of the plotted frequencies, not independent island confirmation. |
| NBI/heading collisions; shrinking fonts to 0.72 text width | Dropped bottom ticks on NBI/intermediate rows, added heading space, and placed both legends on separate reserved rows. Final figure size is 6.75 × 5.8 inches with a 7-pt minimum; the README and JSON require textwidth. The controller must change the manuscript inclusion width; the implementer rule forbids editing the manuscript. |
| Mask grey is confused with other n | Unmeasured coherent-mask pixels are white and explicitly keyed “TokEye mask, n not measured.” Other n is purple with a diamond key; n=1/2/3 retain distinct blue/cyan/green. No n is extrapolated beyond measured support. |
| NTM native-grid outline disappears in print | Max-pool the native mask to the 150-dpi axes display grid, extract its edge and thicken it by one print pixel. The compound raw time/band clip prevents pooling, half-cell and stroke bleed across label boundaries. The final PNGs show visible orange ridges at print width. |
| Chips hide triangles/modes; saw leaders cross; AE leader hard-coded by shot | Removed the ELM chip and every sawtooth leader. AE/NTM leaders occupy the otherwise sparse upper portions of their panels; text depends on selected source tier, never shot number. Every final PNG was opened at print size, including 186636; no leader crosses another leader or a heading. |
| Thresholds omitted from records | Every record includes AE 0.7, NTM 0.63, sawtooth explicit-confidence 0.6, TokEye 0.2, ELM veto 5 ms, ECE match 3 ms, and the full projection/display rules. Deterministic accepted physics classes without a probability must still pass all ECE evidence; no confidence is invented. |
| projection_audit uses the projection's own present_columns | Independent audit enumerates the nonzero projected coordinates against raw end-exclusive source intervals and raw frequency boundaries. It calls neither present_columns nor interval union. A regression intentionally breaks present_columns and proves the independent audit catches leakage. All six final records have zero violations. |

A read-only source/code review additionally caught CSV/JSON evidence parity,
missing-confidence physics JSONs, explicitly uncertain partial events entering
local corroboration, an overbroad harmonic flag, and an absolute-path baseline
lookup in the audit CLI. Each was corrected; dedicated regressions cover the
source cases. The final source/code review reported no remaining material
regressions. It was not either original paper reviewer and supplied no new
scores.

### Data, inference and ranked shortlist

Source stores were read-only: the main cohort/review tables, existing corpus
and raw-cache signals, review-page n/D-alpha/NBI stores, pinned TokEye cache,
and current improved saw physics JSONs under
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix/shots/`.
The same JSONs are the track and crash source; this is a parameterized choice,
not a hard-coded future source. The cohort CSV directory is supported but the
full JSON evidence was used for these final renders.

AE inference used
`$LABELER_ROOT/envs/phase3/bin/python`, CUDA_VISIBLE_DEVICES=0,
PYTHONPATH=this-worktree/src, and at most eight CPU threads. It reads the
model card's pinned checkpoint and transform, recurrent window/context,
causal aggregation and validity rules. Output is isolated at
`$LABELER_ROOT/round4/fig1/ae_ours.csv` and `ae_ours.meta.json`, with log
`ae_ours.log`; no production label file is written. The metadata records
all three input-file/checkpoint hashes and per-bin activity/validity. Each
inferred shot has 240 valid causal bins, as written by the committed
fig1_ae_infer.py to that metadata. No training or thresholds were chosen here.

Shortlist source: **outputs/labeler/paper/fig_interpreter_tokeye/fix_round2_audit.json**,
produced by committed **scripts/labeler/paper/fig1_audit.py**. Rank reasons are
visual judgments on the fixed train/val candidates, not a fitted score.

1. **201978 (train)** — Clearest AE cascade and persistent n=1/n=2 low-frequency mode; local corpus CO2 supports paper-model inference, D-alpha and NBI. Retain for AE plus low-frequency structure; no verified sawtooth survives.
2. **201973 (val)** — Strongest visual alternate: clear AE cascade, low-frequency mode, D-alpha and NBI with paper-model CO2 inference; current physics export does not supply verified sawtooth crashes in the window.
3. **203187 (train)** — AE and low-frequency structure with local CO2/D-alpha/NBI; cascade is less distinct than the first two, and no verified crash survives.
4. **186636 (val)** — Reviewed ELM intervals and imported NTM archive improve provenance; weaker AE structure, earlier interferometer frame detector, no definite confinement shading or verified sawtooth crash in this window.

Retain 201978 under the controller's AE + low-frequency-mode criterion.
Verified sawtooth is preferred, but the current physics source cannot supply
it on either of the two strongest candidates; manufacturing ticks or keeping
the old ELMy interval phase would misrepresent the evidence. The regime
examples 191376/191782 do not support primary AE/NTM highlights. No test-split
signals were used for selection, and the audit records blind_test_shots_used=0.

### Final descriptive outputs

All numbers in this table come from fix_round2_audit.json and its hashed
per-shot records. Baseline crash counts are read from the tracked records at
**e5a56995** by fig1_audit.py. AE/NTM counts are component coincidences, not
classification scores. Caption counts conservatively include the TeX wrapper
and label; all are below the controller's 150-word ceiling.

| Shot | Split | Window (ms) | AE components | NTM components | Old → verified crash ticks | D-alpha markers | Caption words |
|---|---|---|---|---|---|---|---|
| 201978 | train | 1400–4900 | 389 | 21 | 52 → 0 | 80 | 119 |
| 201973 | val | 1600–3350 | 178 | 14 | 1 → 0 | 65 | 119 |
| 203187 | train | 1700–3150 | 287 | 8 | 1 → 0 | 6 | 96 |
| 186636 | val | 1300–3900 | 358 | 5 | 1 → 0 | 32 | 107 |
| 191376 | train | 1500–2900 | 0 | 0 | 6 → 0 | 2 | 94 |
| 191782 | train | 1800–3700 | 0 | 0 | 1 → 0 | 0 | 86 |

The primary's harmonic-support record independently samples the n view
inside the NTM projection. Its detailed support, frequency medians, bands,
tolerance and minimum duration are in 201978.json, drawn.harmonic_support.
The conditional harmonic sentence appears on 201978, 201973 and 186636,
not on 203187 or the regime examples. 186636's H-mode track remains
uncertain in the shown window and therefore supplies no definite shading;
its caption now states this. All records name the actual model/source and
acceptance bars without transferring claims between sources.

### Figures and print-size inspection

Final source commit in every record: **b3d1a1e2**.
All six final PNGs were opened with view_image after final regeneration,
at their native 150-dpi size (6.75-inch print width). The inspected output
set under `$LABELER_ROOT/round4/fig1/` is:

- Primary: fig_interpreter.pdf, fig_interpreter.png, fig_interpreter.json,
  caption.tex.
- Same four files under alt_201973/, alt_203187/, alt_186636/,
  alt_191376/ and alt_191782/.

The PDFs retain vector text/lines over embedded raster spectra. The records
and exact caption copies in outputs/labeler/paper/fig_interpreter_tokeye are
identical to the external copies; their caption hashes and source hashes were
checked. All final headings are clear of the NBI/D-alpha axes, peak triangles
remain visible, no unsupported crash strip/leader remains, masks/n values
are separately keyed, and orange outlines are visible at print width.

### Verification

Required TMPDIR, LABELER_ROOT, LABELER_LABEL_TABLES, LABELER_NO_FETCH=1 and
worktree PYTHONPATH were set. Covering tests only, via the mandated wrapper:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
/scratch/gpfs/nc1514/FusionAIHub-r4-fig1
 tests/labeler/test_paper_figure_sources.py
 tests/labeler/test_paper_mode_tags.py -q -p no:cacheprovider
................................. [100%]
33 passed in 3.29s
```

Ruff check on figure_sources.py, fig_interpreter_tokeye.py, fig1_ae_infer.py,
fig1_audit.py and test_paper_figure_sources.py, through the frozen/no-install
main-manifest labelmaker environment: **All checks passed!**
Ruff format --check: **5 files already formatted**.
No full suite was run, per the implementer rules.

Committed-script audit: **Audited 6 non-blind renders; zero projection
violations**. Additional final checks confirmed external/worktree record
identity, caption hash/text identity, all consumed saw source hashes, and
every generator git field matching b3d1a1e2. Every render exited successfully.
The final sequential render job ran tmpsweep.sh; its output was
“my /tmp use: 622 MB -> 622 MB”, with one harmless transient find race on an
unrelated disappearing assembly temp file. No generated plot or source was lost.

The final read-only reviewer also opened every final PNG and inspected every
caption/record. It reported **no residual Critical, Important or Minor
render/caption findings**, confirming the heading, leader, mask-key, outline
and ELM uncertainty fixes. This remains a separate implementation review,
not a substitute for the two original paper reviewers.

### Concerns and controller next step

- No ECE-verified sawtooth crash survives in any displayed window with the
  current physics source. The source is still being improved in r4-saw;
  rerender using the documented JSON or cohort-CSV/evidence parameters when
  that run is frozen. The primary is therefore an AE/low-frequency/ELM
  capability example with honest sawtooth candidate coverage.
- NTM remains a model prediction with shared displayed inputs and a failed
  primary acceptance check; the caption/leader qualify its interpretation.
- The controller must include this figure at **textwidth**, not 0.72 textwidth,
  to preserve the minimum print font. The manuscript was deliberately untouched.
- The requested >=8 from both original reviewers is still a re-review outcome,
  not a score this implementer can certify. Both re-review finding lists have
  been addressed; send the regenerated primary, 201973 and provenance-focused
  186636 alternate for reassessment.
