# Modernization intent: labeler review page

Date: 2026-10-15. Written for `/code-modernization:modernize`. The owner was not asked the skill's pop-up questions.
Their standing rule is "do the recommended", so the choices below are defaults. Each one is listed under Open items
so it can be changed.

## Goal

**Transform, targeted, same stack.** The labeler's review page keeps FastAPI on the server and plain JavaScript in
the browser, with no framework and no build step. Four pieces of expert feedback drive the work:

1. A ReviewPlus-like customizable workspace. Each event opens on a default view. The reviewer can add any MDSplus or
   PTDATA signal from a list and change the layout. Profiles (such as the q-profile) and video frames show at a time
   slice, and crosshairs can be dropped to compare times.
2. One dropdown of named shot lists replaces the gold checkbox and the separate silver page. Each list is a plain
   file of shot numbers. An event folder names its recommended lists. Evaluation can run on any named list.
3. TokEye overlays are precomputed for every spectrogram the page shows. The page reports loading, progress and
   errors, and it gets faster.
4. The best model of each event set runs on the 450-shot silver list, so its output shows as a source suggestion.

## From

- One FastAPI server, `src/labeler/events/ui/app.py`, and one ~3,100-line `static/app.js`.
- Diagnostic rows are prebuilt per shot.
- The gold checkbox and the silver page are separate.
- TokEye overlays exist for some shots and rows only.

## To

The same stack. New server modules cover named shot lists, on-demand signals and profiles (with a cache), and
TokEye status. New page modules cover the workspace (signal picker, layout, slice dock, crosshairs), the shot-list
dropdown and the status bar.

## What must stay true (defaults)

- Saved labels and their on-disk formats do not change.
- Every existing review behaviour keeps working. The covering tests are `tests/labeler/test_review_*.py` and
  `tests/labeler/test_events_ui.py`.
- Each event's default view is the rows it shows today.
- The server stays loopback-only behind its token. Signal names typed by a user never reach a TDI expression
  unchecked.
- Fetching happens only on the login node, under `fdp run`, while the owner is logged in.

## Road

1. Map (done as a six-reader workflow).
2. Design: the system design and UI plan.
3. Transform: build the new modules.
4. Verify: one Opus review per stream, plus the covering tests and the browser tests.
5. Harden: a security pass over the new fetch endpoint.

The ledger is `.claude/superpowers/plans/2026-10-15-labeler-workspace.ledger.md`.

## Open items (the owner may change these)

- Where the slice dock sits (right side by default, or the bottom) and which profile panels each event opens with.
- The shot-list file layout and its names.
- Whether app.js is split further than the new features need.
