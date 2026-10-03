# TokEye Figure 1 — current build

Primary: shot **201978**, **1500–3300 ms**. Raw Mirnov, D-alpha and NBI lead to
TokEye's binary coherent mask, measured toroidal n and five aligned label rows:
AE, NTM, H-mode, ELMs and physics sawtooth states. The catalog's generated
sawtooth frame model is **not shown**. The sawtooth row preserves absent,
uncertain and blank/unassessed intervals; a conservative density proxy limits
ECE observability during part of the view, without establishing actual cutoff.

Include the PDF at **\textwidth**: **6.75 × 5.5 inches**, fonts **≥7 pt**.
PDFs and 150-dpi PNGs live under `$LABELER_ROOT/round4/fig1/`, with five
`alt_<shot>/` directories. The six records and exact caption copies here are
current; `fix_round4_audit.json` and `fix_round4.md` are the current audit and
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
export SAWTOOTH_SOURCE="$LABELER_ROOT/round4/saw/fix/shots"
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig_interpreter_tokeye.py --shot 201978 \
  --ae-labels "$LABELER_ROOT/round4/fig1/ae_ours.csv" \
  --sawtooth-source "$SAWTOOTH_SOURCE" \
  --record outputs/labeler/paper/fig_interpreter_tokeye/201978.json
```

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
  There are no AE boxes. Late 170–250 kHz lines remain in view and stay
  untagged where the AE detector is absent; a time overlap is required.
- NTM outlines here require a component's unique dominant measured **n=1 or 2**,
  PRESENT times and the nominal **<60 kHz** band. The outline is intersected
  with measured-n pixels **≤30 kHz** again after pooling. The source reads
  **detector (unverified)**; it shares the shown magnetic inputs and has not
  met its validation criteria. A coincidence does not establish an island.
- Measured n uses constant full colour: blue (1), bluish-green (2), yellow (3),
  purple (other). Brightness does not encode n. Unmeasured mask pixels are white.
  Structure labels use short white arrowed leaders.
- Harmonic support compares simultaneous pixel-weighted n=1 and n=2 ridge
  frequencies: **|f2/f1 − 2| ≤ 0.1**, at least **50 ms** sampled support.
  Caption frequencies round to **1 kHz**. This describes harmonic consistency.
- Expert ELM intervals preserve categories: uncertain intervals are hatched.
  Circles identify an expert ELM interval (one span for many ELMs).
  Triangles are D-alpha peaks from a threshold, **not annotations**, restricted
  to PRESENT ELM intervals. The peak finder uses a 25-sample running median,
  4 MADs and 3-sample spacing; it cannot mark a spike before the expert span.
- Hatching means uncertain; blank means unassessed/unobservable. A D-alpha
  L-H transition supplies the pre-transition **L-mode** cue; the binary label
  row remains **H-mode**. Curated regimes retain their original classes.
- Supplying `--sawtooth-source` shows the physics states automatically. It
  accepts JSONs, a JSON directory or cohort CSVs. `--sawtooth-evidence` pairs
  reduced CSVs with full physics evidence. The caption uses the window's
  states and `density_guard` metadata, explicitly identifying a conservative
  density proxy and missing Bt when recorded. It never infers confirmed cutoff.
  Independently ECE-verified ticks reject inclusive ±5 ms D-alpha coincidence.
  The row is preserved even if no verified ticks survive.

## Alternates and audit

Use the generator with `--out "$LABELER_ROOT/round4/fig1/alt_<shot>"`, the same
physics source parameter, and `--record .../<shot>.json`:

| Shot | Window (ms) | Supplied AE CSV |
|---|---|---|
| 201973 | 1600–3350 | yes |
| 203187 | 1700–3150 | yes |
| 186636 | 1300–3900 | no |
| 191376 | 1500–2900 | no |
| 191782 | 1800–3700 | no |

All captions use `fig:interpreter-<shot>`; use `fig:interpreter-201978` for the
primary. Caption text has no shot-specific scientific branches. The report
records the shot-selection criteria, candidates and ranking from the round-two
selection audit. To validate the current artifacts:

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary \
  --out outputs/labeler/paper/fig_interpreter_tokeye/fix_round4_audit.json
```

The audit checks committed renderer hashes, actual figure hashes, physics
state rows and source hashes, pixel clipping, caption length and unique labels,
external/committed identity before and after rebuild, PDF size, non-blind splits,
and byte-identical primary PDF/PNG rebuilding.
