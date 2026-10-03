# TokEye Figure 1 — fix round 5

Primary: shot **201978**, **1500–3300 ms**. Raw Mirnov, D-alpha and NBI lead to
TokEye's coherent mask, a dedicated **0–30 kHz** measured-n panel and four aligned
label rows: AE, NTM, H-mode and ELMs. The sawtooth row is conditional on positive
source PRESENT duration in the view; only 201973 currently qualifies. Hidden
rows retain their exact source states in JSON. The catalog's generated sawtooth
frame model remains excluded. Keys and callouts occupy the empty right margin;
the processed 30–55 kHz band is omitted to enlarge the n view. The raw band remains.

Include the PDF at **\textwidth**: **6.75 × 5.5 inches**, fonts **≥7 pt**.
PDFs and 150-dpi PNGs live under `$LABELER_ROOT/round4/fig1/`, with five
`alt_<shot>/` directories. The six records and exact caption copies here are
current; `fix_round5_audit.json` and `fix_round5.md` are the current audit and
report. Earlier audits/reports are historical. The `git` in each render is the
committed renderer's source revision, before the records were committed;
`render_code_sha256` is checked against that revision and the current code.

## Reproduce from local read-only inputs

Run from this worktree. Set `SAWTOOTH_SOURCE` to the completed physics export
recorded in `drawn.sawtooth_crashes.files`; it is a parameter, not a shot-specific
caption override. All consumed source files have SHA-256 digests in the records.

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/fig1
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
# Use the pinned, completed export named in the current report.
export SAWTOOTH_SOURCE="/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/saw_source/fix2-79a2c18ed4aa/shots"
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig_interpreter_tokeye.py --shot 201978 \
  --tmin 1500 --tmax 3300 \
  --ae-labels "$LABELER_ROOT/round4/fig1/ae_ours.csv" \
  --sawtooth-source "$SAWTOOTH_SOURCE"
```

Generate all six externally while HEAD is clean, then copy their JSONs/captions
here. `--record` is supported, but writing tracked records between renders makes
later HEADs dirty. The current clean renderer revision is **21e72826**.

The sawtooth stream is still regenerating. This snapshot is the newest completed
source available during this round, with live completion/shot hashes matching
`sawtooth_source_manifest.json`. For the later rerender, pin its new complete
source, update the manifest, pass **--sawtooth-source**, regenerate all six,
inspect their PNGs, and rerun the audit. No renderer change is needed.

The supplied AE predictions use the existing isolated adapter inference and
its fixed paper operating point **p≥0.5**. The three other shots retain the
earlier interferometer frame model's checkpoint threshold, recorded per track.
Expert review takes precedence over supplied or stored paper predictions,
then the earlier frame model. AE training targets used TokEye's coherent mask;
their overlap with the displayed mask is not independent evidence.

## Source and display conventions

- Raw frequency bands are normalised separately (3rd/99.8th percentiles).
  The 55 kHz fold changes frequency resolution, not an event-band threshold.
- AE tint intersects PRESENT times and **≥80 kHz**, the model's input band.
  There are no AE boxes. The caption discloses **25 ms** bins for `ae-ours` and
  **10 ms** bins for the earlier frame fallback. Late untagged frequency bounds
  come from coherent components of **≥150 ms continuous duration**, after the
  last PRESENT AE interval during ABSENT time, within the detector input band.
  Descriptions are conditional; there is no fixed 170–250 kHz scientific band.
- NTM outlines here require a component's unique dominant measured **n=1 or 2**,
  PRESENT times and individually measured **n=1/2 pixels ≤30 kHz**. Its
  one-print-pixel inner edge has **alpha 0.45** and is intersected with those
  pixels again after pooling. No n=3 pixels are marked. The source reads
  **detector (unverified)**; it shares the shown magnetic inputs and has not
  met its validation criteria. A coincidence does not establish an island.
- Measured n uses constant full colour: blue (1), bluish-green (2), yellow (3),
  purple (other). Brightness does not encode n. Unmeasured mask pixels are white.
  Sawtooth uses vermillion **#D55E00**. Structure labels sit in the empty right
  margin with neutral grey leaders.
- Harmonic support compares simultaneous pixel-weighted n=1 and n=2 ridge
  frequencies: **|f2/f1 − 2| ≤ 0.1**, at least **50 ms** sampled support.
  Caption frequencies round to **1 kHz**. This describes harmonic consistency.
- Expert ELM intervals preserve categories: uncertain intervals are hatched.
  Circles appear only for expert crowd intervals (one span for many ELMs).
  Triangles are D-alpha peaks from a threshold, **not annotations**, restricted
  to PRESENT ELM intervals. The peak finder uses a 25-sample running median,
  4 MADs and 3-sample spacing; it cannot mark a spike before the expert span.
- Hatching means uncertain; blank means unassessed/unobservable. A D-alpha
  L-H transition supplies the pre-transition **L-mode** cue; the binary label
  row remains **H-mode**. JSON preserves numeric categories with the track's
  own regime/binary mapping: curated category 2 is L-mode, category 5 uncertain.
- Supplying `--sawtooth-source` reads the physics states automatically. It
  accepts JSONs, a JSON directory or cohort CSVs. `--sawtooth-evidence` pairs
  reduced CSVs with full physics evidence. The caption uses the window's
  states and `density_guard` metadata, explicitly identifying a conservative
  density proxy and **“Bt not in the local corpus”** when needed. It never
  infers confirmed cutoff.
  Independently ECE-verified ticks reject inclusive ±5 ms D-alpha coincidence.
  The row requires positive source PRESENT duration in the displayed window.
  For display only, **<10 ms** state slivers merge into the longer touching
  neighbour, shortest first, with earlier-neighbour ties; gaps are preserved.
  `display_intervals_ms` and `display_merge` retain the rule and every change;
  raw source intervals and exact crash ticks remain untouched.
- Sawtooth crashes are point events, never rotating-mode tags. Overlap with n=1
  does not identify a precursor or establish NTM seeding. The report's
  **Deviations from the brief** explicitly explains this scientific departure
  from the owner's original projection request. Event/crowd keys and captions
  follow actual drawn content and source tiers.

## Alternates and audit

Use explicit **--tmin/--tmax** for all reproductions. The two comparison shots
otherwise default to their full cohort windows. Use the generator with
`--out "$LABELER_ROOT/round4/fig1/alt_<shot>"`, the same
physics source parameter, and `--record .../<shot>.json`:

| Shot | Window (ms) | Supplied AE CSV |
|---|---|---|
| 201973 | 1600–3350 | yes |
| 203187 | 1700–3150 | yes |
| 186636 | 1300–3900 | no |
| 191376 | 1500–2900 | no |
| 191782 | 1800–3700 | no |

Omit `--ae-labels` for the last three. 201973 is the strongest visual alternate;
191376/191782 are diagnostic/regime comparisons with detector ELMs and no AE/NTM
coincidences in these windows. The report includes the primary draft caption.

All captions use `fig:interpreter-<shot>`; use `fig:interpreter-201978` for the
primary. Caption text has no shot-specific scientific branches. The report
records the shot-selection criteria, candidates and ranking from the round-two
selection audit. To validate the current artifacts:

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary \
  --out "$LABELER_ROOT/round4/fig1/fix_round5_audit.json"
```

The audit checks committed renderer hashes, actual figure hashes, physics
state rows and source hashes, display smoothing/visibility, numeric confinement
categories, conditional keys, zero n=3 outlines, text bounds/fonts, pixel
clipping, caption length and unique labels,
external/committed identity before and after a scratch rebuild, PDF size, non-blind splits,
and byte-identical primary PDF/PNG rebuilding.
