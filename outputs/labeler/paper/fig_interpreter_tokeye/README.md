# TokEye Figure 1 — current build

Primary: shot **201978**, **1500–3300 ms**. This view follows the AE cascade,
H-mode transition, ELM onset and the low-frequency mode and harmonics. It has
four aligned source tracks: AE, NTM, H-mode and ELMs. Sawtooth is not assessed
here because ECE is cut off in ELMy H-mode according to the physics source's
density guard. The sawtooth row is optional for later, suitable shots.

Include the PDF at **\textwidth**: **6.75 × 5.5 inches**, fonts **≥7 pt**.
PDFs, 150-dpi PNGs, cached TokEye arrays and isolated AE inference live under
`$LABELER_ROOT/round4/fig1/`. Six small records, matching caption copies and
`fix_round3_audit.json` are committed here. `fix_round3.md` is the current report;
older audit files are historical snapshots, not current measurements.

## Reproduce from local read-only inputs

Run in the `r4-fig1` worktree:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/fig1
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_ae_infer.py --shots 201978 201973 203187 \
  --out "$LABELER_ROOT/round4/fig1/ae_ours.csv"
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig_interpreter_tokeye.py --shot 201978 \
  --ae-labels "$LABELER_ROOT/round4/fig1/ae_ours.csv" \
  --sawtooth-source "$LABELER_ROOT/round4/saw/fix/shots" \
  --record outputs/labeler/paper/fig_interpreter_tokeye/201978.json
```

AE inference calls the model adapter (`spec.load`, `INPUT_SPEC.build`,
`predict`) on CPU. The pinned checkpoint digest, local CO2 input hashes,
probabilities and validity are in `ae_ours.meta.json`. One constant,
`figure_sources.AE_THRESHOLD`, sets the paper operating point **p≥0.5**, as
in `scripts/labeler/ae_baselines_evaluate.py`. This is fixed, not selected
from these shots. The three other renders retain the earlier interferometer
frame fallback's own checkpoint threshold (recorded separately in each track).
Source priority: expert review, supplied paper-model predictions, valid stored
paper-model predictions, then the earlier frame fallback.

## Source and display conventions

- Expert ELM `iscrowd=1` identifies the annotation lane and does not change
  category. Category 1 stays PRESENT; category 2 stays UNCERTAIN, with hatching
  and a crowd marker above it. Only PRESENT intervals receive definite
  D-alpha peak triangles. The processed panel names the box **ELMs** and the
  key **ELM intervals**. Peaks use a 25-sample median, 4 MADs and 3-sample spacing.
- Hatching means uncertainty. Blank means **unassessed / unobservable**.
  Confinement prioritizes saved review, Gill's and Jalal Butt's curated regime
  intervals, then the D-alpha L-H detector. The binary fallback is named
  **H-mode**; its absence does not establish L-mode.
- AE uses **tint only**, with no bounding boxes. Every highlighted pixel
  intersects a raw end-exclusive PRESENT interval and **≥60 kHz**.
- NTM coincidence requires a component's unique dominant measured n=1/2,
  then clips to pixels where n was actually measured, **≤30 kHz**, inside
  PRESENT times and the nominal **<60 kHz** band. Print-grid inner outlines
  are intersected again with the native eligible mask. No n is extrapolated.
  The detector fails its acceptance bar and is displayed as an
  **unverified suggestion**, with **detector** below its bar. Shared magnetic
  inputs mean this is not independent mode identification.
- Measured n=1/2/3 use blue/cyan/blue-green; other n is purple; unmeasured mask
  pixels are white. Components dominated by n=3 are not outlined because they
  do not meet the NTM rule. Measured n=3 pixels within a component dominated
  by n=1/2 can remain inside an outline; their count is recorded separately.
- Harmonic support is computed from coincident measured n=1/2 frequencies,
  using each column's pixel-weighted ridge frequency, without shot-specific
  bands. A 2:1 match must be within 1.2 kHz with ≥50 ms sampled support.
  The caption's approximate frequency is rounded to a recorded 5 kHz step
  on every shot. This establishes consistency with a harmonic, not a common
  magnetic island. Medians, duration, tolerance and rounding are in the record.
- The 55 kHz fold changes spectrogram resolution, not the 60 kHz tag rule.
  The raw magnetics label sits in the heading, clear of the cascade.

## Optional sawtooth source

`--show-sawtooth` enables the row and verified crash strip for a suitable shot.
`--sawtooth-source` accepts read-only shot JSONs, a JSON directory, cohort CSVs
or `shot,t_ms` CSVs. `--sawtooth-evidence` pairs reduced CSVs with full physics
JSONs (point matches within 0.5 ms). Track and ticks share the source, but
**interval categories are preserved** independently of tick verification.

A tick needs an accepted core ECE drop with spatial inversion/heat-pulse
corroboration; explicit confidence must pass 0.6. A reduced point without
full evidence must match local ECE evidence within 3 ms. Ticks reject inclusive
±5 ms D-alpha coincidence and recorded ELM coincidence; missing D-alpha
supplies no verified ticks. SXR alone cannot supply one. All consumed sources
and evidence are hashed. The primary still records the physics source and
rejections while omitting the row/strip.

## Alternates and audit

Use the same generator with `--out "$LABELER_ROOT/round4/fig1/alt_<shot>"` and
`--record outputs/labeler/paper/fig_interpreter_tokeye/<shot>.json`:

| Shot | Window (ms) | Supplied AE CSV |
|---|---|---|
| 201973 | 1600–3350 (preset) | yes |
| 203187 | 1700–3150 (preset) | yes |
| 186636 | 1300–3900 (preset) | no |
| 191376 | `--tmin 1500 --tmax 2900` | no |
| 191782 | `--tmin 1800 --tmax 3700` | no |

Every alternate has its own `fig:interpreter-<shot>` label; only the primary
uses `fig:interpreter`. All six current PNGs were visually inspected. Audit:

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary \
  --out outputs/labeler/paper/fig_interpreter_tokeye/fix_round3_audit.json
```

The audit validates raw time/band clipping, rendered n support, source hashes,
caption lengths and unique labels, external/committed record identity, PDF
size, train/val-only selection and byte-identical primary PDF/PNG rebuilding.
