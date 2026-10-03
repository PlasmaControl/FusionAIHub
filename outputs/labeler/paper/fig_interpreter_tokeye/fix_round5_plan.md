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
