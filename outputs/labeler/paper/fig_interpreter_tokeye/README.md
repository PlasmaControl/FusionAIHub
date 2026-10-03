# TokEye Figure 1

The generator is `scripts/labeler/paper/fig_interpreter_tokeye.py`. Large PDF,
150-dpi PNG, `fig_interpreter.json` and `caption.tex` files live under
`$LABELER_ROOT/round4/fig1/`; this directory holds the small per-shot records
and exact caption copies. Primary: 201978. Other-shot alternates: 201973,
203187, 186636. Regime examples: 191376 and 191782.

Run from the stream worktree, using only local data:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/fig1
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig_interpreter_tokeye.py \
  --shot 201978 --record outputs/labeler/paper/fig_interpreter_tokeye/201978.json
```

Other-shot renders use `--out "$LABELER_ROOT/round4/fig1/alt_<shot>"` and the
corresponding `--record`. The presets cover the four recommended candidates.
Regime examples use `--shot 191376 --tmin 1500 --tmax 2900` or
`--shot 191782 --tmin 1800 --tmax 3700`. The pinned TokEye cache is reused;
CPU threads are limited to eight. No GPU or fetching is needed.

`CONFINEMENT_RUN_DIR` defaults to `runs/labeler/confinement/v1` in the checkout
containing `LABELER_LABEL_TABLES` (the main checkout in this recipe). Override
it only to point to a different curated source directory. Source order:
saved four-class confinement review, curated `merged_intervals.csv`, then
`$LABELER_ROOT/suggestions/dalpha_lh/v1/confinement_suggest_dalpha_lh_v1.csv`.
Missing required tables or missing shot coverage raise errors. A binary
fallback is named **H-mode**; its absent state never asserts L-mode. Curated
multiregime tracks are named **regime**, with names on the shading.

AE uses valid stored `d3d_ae_activity_seldnet/ae_active` probabilities
(paper **ae-ours**) when available, at fixed p>=0.5, respecting validity and
the causal 25-ms window convention. Otherwise the current expert or CO2
xpower frame source remains. The record names that fallback; the caption
discloses it. None of the four recommended candidate shots has stored
ae-ours predictions. AE boxes omit PRESENT spans shorter than 100 ms;
their tint/track can still show those short detector decisions.

Projection is at native pixel resolution: component AND PRESENT time AND
the event band (AE >=60 kHz, NTM <60 kHz). NTM requires a unique dominant
measured n=1 or n=2. The wide pass includes low tags at 55–60 kHz; a component
with no n-map evidence cannot receive an NTM tag. The n map ends near 30 kHz:
it is never extrapolated. Outlines are inside the mask, without dilation;
compound clip paths also prevent half-cell extents and line strokes crossing
time/band boundaries. AE boxes are dashed, NTM outlines solid.

Sawtooth is never a rotating-mode tag. Dotted crash ticks use the local
`ece_sawtooth` ECE/SXR crash detector's event times, clipped to the sawtooth
track's PRESENT intervals. These point times remain detector estimates even
when the interval track is expert reviewed. To replace them, pass
`--sawtooth-crashes /path/to/crashes.csv` (columns `shot,t_ms`). To replace
the interval track independently, pass `--sawtooth-labels /path/to/labels.csv`
(standard interval-table schema). Both sources are recorded and named in
the generated caption.

`drawn.projection_audit`, `drawn.ntm_dominant_n`, `drawn.ae_boxes_ms_khz` and
`drawn.sawtooth_crashes` record what was drawn. Every coloured n has a key;
all other n are grey. The caption discloses shared NTM inputs, failed N1/S1
primary acceptance bars, expert tracks, and the coincidence interpretation.
No crash-to-NTM seeding claim is made.
