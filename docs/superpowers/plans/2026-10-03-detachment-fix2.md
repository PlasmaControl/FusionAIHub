# Detachment review fix round 2 implementation plan

> **For agentic workers:** Use systematic debugging, focused regression tests,
> parallel agents for independent data/queue fixes, and verification before completion.

**Goal:** Resolve every Important and Minor finding in the second re-reviews.

**Architecture:** Camera requests decode into transaction-owned staging objects.
One synchronous publication updates every image/caption and the playback clock.
Data stores use plasma-window previews, scientific context and producer bin masks;
the checked-in queue combines LSN cohort eligibility with producer-labelled shots.

**Tech stack:** Existing Python/HDF5 review builder, vanilla JavaScript, Chromium
DevTools browser tests and matplotlib paper exports; no dependency changes.

**Spec:** User's second fix-round request and the full `detach-ui-opus2.md` and
`detach-ui-sol2.md` reviews under the controller's `r4/reviews/` directory.

## Constraints and review focus

- Work only here; producer/corpus/main checkout remain read-only. No fetching.
- Exclude blind test shots before reading/selecting queue candidates.
- Python uses the prescribed frozen shared pixi environment and task TMPDIR.
- Atomic publication must survive pause, seek, view change and navigation.
- Missing geometry/source units must be explicit; do not fabricate evidence.
- Preserve isolated valid producer bins and gaps; support changed bin widths.
- Rebuild current stores after context changes; inspect PNGs before completion.
- Commit `labeler:` messages with the user's Codex co-author trailer.

## Task 1: Atomic playback and page clarity (root)

Files: `ui/static/app.js`, `index.html`, `style.css`,
`tests/labeler/test_review_detachment_browser.py`, `review_browser.mjs`.

- [x] Add real two-camera unequal-delivery/pause regression and observe failure.
- [x] Stage decoded frames; publish all views/captions/clock together; cancel URLs.
- [x] Observe the video panel and resize rows; add distinct semantic colours.
- [x] Show magnetic configuration and operational definitions in page help.
- [x] Run the covering browser tests and JavaScript syntax checks.

## Task 2: Scientific context and stores (data_panels agent)

Files: `panels/detachment.py`, `review/panel_rows.py`, `review/video.py` and
their focused tests. Interfaces: existing Panel metadata and video manifest.

- [x] Reproduce fallback/bin/trim defects with focused tests.
- [x] Add labelled Thomson/producer density fallback, FS units/location groups,
  bin-width inference and steps over complete bins; trim previews to plasma.
- [x] Correct resampling wording, preserve validity reasons/votes.
- [x] Run covering tests and Ruff; report interfaces and source limitations.

## Task 3: Real queue and geometry (queue agent)

Files: `scripts/labeler/detachment_review_roster.py`,
`data/events/detachment/shots.csv`, geometry helper and focused tests.

- [x] Inspect producer output and strike-point gate; add regression tests.
- [x] Write reproducible union queue excluding test shots; expose geometry.
- [x] Scan live corpus and producer snapshots into JSON; controller can rerun.
- [x] After all sources are ready, root rebuilds isolated stores and records coverage.

## Task 4: Evidence, docs and delivery (root)

Files: `detachment_review_demo.py`, `docs/labeler/detachment_review.md`, result
JSONs, external `r4/reports/detach-ui.md`; figures under round4/detach-ui only.

- [x] Run covering tests, Ruff and real-shot browser checks; retain screenshot.
- [x] Make cropped annotated PDF and 150-dpi PNG with readable paper text.
- [x] Inspect both evidence PNG and paper PNG; record source hashes and outputs.
- [x] Append Fix round 2 mapping every finding to changes and evidence.
- [x] Review integrated diff, commit on r4-detach-ui, verify clean task status.

## Delivery notes

All implementation and evidence tasks are complete; the final commit is recorded
in the external Fix round 2 report. Filterscope sightline locations are absent
from authoritative local inputs: each calibrated chord is isolated and labelled
as location not recorded. This source limitation remains explicit in the report.
The final covering run passes 252 tests, Ruff and JavaScript syntax checks; the
real-shot Chromium run passes 20 checks. The screenshot and column PNG were
visually inspected. An independent source review found no remaining findings,
including the cached-current plasma-window fallback added during the audit.
