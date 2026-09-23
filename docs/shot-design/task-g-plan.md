# Part G implementation plan

Spec: `.claude/superpowers/specs/2026-09-22-iteration2-ignite-stage1-flow-design.md`.
Implementation stays in the UI, its tests, and UI documentation in this worktree.

1. G1: Test search metadata, selection order and six-shot limit, explanations and
   channel chips, filter validation, hash restoration and search-to-editor notes.
   Add metadata to app.py; build search controls and routing in app.js; extend
   design.js to accept comparisons and notes. Share the DOM factory, remove row
   fetches and normalize non-finite JSON at the transport boundary. Run UI tests,
   Ruff and commit.
2. G2: Test the F6 metrics endpoint and rendered fixture, revision isolation,
   edits retaining results, rerun confirmation, status errors and elapsed time.
   Add the read-only metrics route and a simulation state map in design.js with
   one renderer deriving button/status/results. Fetch status on reopen, serve
   polling configuration in metadata, render tables and images. Run tests,
   Ruff and commit.
3. G3: Test export eligibility, assistant-to-simulation action, shot links,
   restored running/completed/missing jobs and stale replies. Keep job ids in
   the hash, reattach from routing and provide the simulation action using the
   saved editor revision. Run tests, Ruff and commit.
4. Run the full requested suite under -W error, inspect the complete diff and
   ownership boundaries, document results and manual checks in
   `.superpowers/sdd/task-G-report.md`, force-add and commit the report.

Review focus: stale asynchronous replies must not replace another search or
revision; browser history must restore filters; missing metrics remain explicit;
negative skill/unresolved effects remain visible; reloading never resubmits a job.
