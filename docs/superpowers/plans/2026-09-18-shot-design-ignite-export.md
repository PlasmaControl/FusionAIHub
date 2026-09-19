# Shot design IGNITE export implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver a browser waveform editor with saved revisions and directly loadable IGNITE inputs.

**Architecture:** A new design-program service owns validation, reference resolution,
normalization and export. A separate HTTP router and browser module expose the
service without coupling it to retrieval or changing the existing ActuationSet.

**Tech Stack:** Existing Python/NumPy/Pydantic/PyTorch/FastAPI, plain JavaScript/SVG.

**Spec:** `docs/superpowers/specs/2026-09-18-shot-design-ignite-export-design.md`

## Global Constraints

- Do not add dependencies or modify production stores.
- Use 88 channels in ACT_SPEC order, 50 ms frames, 20 history frames, at most 80 prediction frames.
- Statistics come from the complete reference; preserve cached unedited/seed inputs exactly.
- Unknown, missing, overlapping or unrepresentable edits cannot silently export.
- Existing retrieval and ActuationSet behavior must remain compatible.
- All routes use the existing token gate; saved IDs cannot address arbitrary paths.
- Tests use synthetic local data. Real-data verification is read-only except task-owned scratch outputs.

## Task 1: Program service, persistence, transport and numerical regressions

**Files:** Create `src/shot_design/design/program.py`, additional focused design
service modules as needed, `src/shot_design/ui/design_routes.py`,
`tests/shot_design/test_program.py`, `tests/shot_design/test_design_routes.py`.
Modify `src/shot_design/ui/app.py` only to register the router before unknown_api.

**Interfaces:** Implement the exact endpoints and wire schema in the spec.
Expose `DesignProgram`, `preview(program, paths)`, `save(program, paths)`,
`load(id, paths)`, `list_programs(paths)`, `export_ignite(program, paths)` from
the service or documented focused submodules. All call paths share validation.

- [x] Write deterministic fixture tests before implementation. With a reference
  channel alternating raw 100 and 200, a 20% edit must use the original mean
  150/std 50: edited 120 becomes approximately -0.6, edited 240 becomes 1.8.
  The first 20 frames and unedited channels must remain exactly cached values.

```python
assert torch.equal(exported['actuators'][:20], reference['actuators'][:20])
assert exported['actuators'][20, 12].item() == pytest.approx(-0.6, abs=0.001)
assert exported['actuators'][21, 12].item() == pytest.approx(1.8, abs=0.001)
```

- [x] Run the new tests, confirm failure because the service/routes are absent.
- [x] Implement the service using `build_actuators` for complete-reference
  statistics and copied cache tensors for unchanged values. Independently
  validate edited channel coverage and reference-cache consistency. Validate
  frame bounds and source identity before applying interpolation.
- [x] Add tests for total split preserving ratios, no active-member split,
  constant reference normalization, missing data, time/shape errors, changed
  source files, JSON reload and export parity. Do not invent hardware limits.
- [x] Implement atomic storage and router, test save/get/download and invalid
  export, missing IDs, source drift, unauthenticated requests and body errors.
- [x] Run focused tests and existing actuator/UI regression suites. Commit only
  task-owned files with `shot_design: add validated IGNITE program export`.

## Task 2: Browser editor and interaction tests

**Files:** Create `src/shot_design/ui/static/design.js` and focused browser tests;
modify `static/index.html`, `static/style.css`, `static/app.js` for navigation
and Use as reference actions. Do not modify service/router files.

**Interfaces:** Consume the spec's JSON endpoints; module exports
`initDesign()` and `openDesign(referenceShot)` (or an equally small documented
interface). Existing DOM boot and route behavior remain intact.

- [x] Write failing executable JavaScript/browser tests for numeric edits,
  reset, save/reopen and stale-download invalidation using the specified wire
  shape. Keep network fixtures at the HTTP boundary; do not fake editor logic.
- [x] Implement the controls and SVG editor; use reference frame values as the
  initial vertices, retain history unchanged, interpolate only the prediction
  region, and expose selected point's time and value in inputs.
- [x] Add comparison overlays and labels/units. Handle missing data and server
  validation visibly. Mark dirty state and prevent old saved exports from
  masquerading as current edited drafts.
- [x] Connect Design navigation, saved revision listing/loading, reference
  selection from search and shot detail, and download links.
- [x] Run existing Node UI tests and new interactions, then commit task files
  with `shot_design: add actuator waveform editor`.

## Task 3: Integration, documentation and final verification

**Files:** Create `docs/SHOT_DESIGN_PROGRAMS.md`; add integration tests and small
cross-component fixes only where verification finds real gaps.

**Interfaces:** The editor's saved `.pt` is consumed directly by
`ignite_infer.load_shot` and `rollout(model, cfg, shot)`; no conversion required.

- [x] Exercise preview -> edit -> save -> reload -> download on a task-isolated
  app with a real browser. Use read-only real references and scratch outputs;
  check handles, numeric edits, checks and downloads.
- [x] Load the `.pt` using the local bundle's loader and validate it against the
  production contract without requiring a GPU rollout. Compare a no-edit
  export's codes/controls exactly against the corresponding reference slice.
- [x] Document the UI flow, artifact locations, physical units, fixed history,
  model-specific limitations and a working inference example.
- [x] Run the full shot_design suite excluding real_data plus focused lint.
- [x] Review the complete diff, fix findings, and bring the verified change
  into the user's checkout without touching unrelated edits.

Final verification: 1,571 tests passed, 22 real-data tests deselected; focused
Ruff and JavaScript syntax checks passed. A real browser verified numeric and
pointer editing (including release outside the chart), save/reopen/reset,
comparison overlays, downloads, and mobile layout. Actual shot 204346 exports
passed the local production IGNITE loader and validator; reference history and
untouched controls matched exactly. No GPU rollout was performed.
