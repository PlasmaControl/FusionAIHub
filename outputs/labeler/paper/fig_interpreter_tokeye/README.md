# TokEye Figure 1 — fix round 2

Generator: `scripts/labeler/paper/fig_interpreter_tokeye.py`. Primary 201978;
alternates 201973, 203187, 186636; regime examples 191376, 191782. Records,
caption copies and `fix_round2_audit.json` are committed here. PDFs, 150-dpi
PNGs, caches and isolated AE predictions live under `$LABELER_ROOT/round4/fig1`.
Use **\\textwidth** (6.75 inches) in the manuscript; all text is at least 7 pt
at that width. The source manuscript is intentionally left for the controller.

Run in this stream worktree using local data only:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/fig1
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig_interpreter_tokeye.py --shot 201978 \
  --ae-labels "$LABELER_ROOT/round4/fig1/ae_ours.csv" \
  --sawtooth-source "$LABELER_ROOT/round4/saw/fix/shots" \
  --record outputs/labeler/paper/fig_interpreter_tokeye/201978.json
```

The physics source is read-only. It accepts a shot JSON, a directory of shot
JSONs (direct or under `shots/`), a cohort CSV, a directory of `cohort-*.csv`
shards, or a minimal `shot,t_ms` CSV. The current cohort export can be paired
with its full ECE evidence when rerendering:

```bash
# Add these options to the generator command, replacing --sawtooth-source above:
--sawtooth-source /scratch/gpfs/nc1514/FusionAIHub-r4-saw/data/events/sawtooth_oscillation/extend_saw_physics \
--sawtooth-evidence "$LABELER_ROOT/round4/saw/fix/shots"
```

JSON/CSV point matches allow at most 0.5 ms of export rounding. Full physics
JSON evidence must establish a core temperature drop and a spatially distinct
rise/inversion, with accepted state. Explicit confidence values must pass 0.6;
a deterministic accepted physics class without a probability still requires
all ECE evidence. Reduced CSV points without paired evidence require local
ECE core-drop/heat-pulse corroboration within 3 ms. SXR alone never supplies a
tick. All ticks are vetoed within inclusive +/-5 ms of D-alpha peaks anywhere
in the window, independent of ELM label coverage; recorded physics ELM
coincidence also vetoes a point. Missing D-alpha means no verified ticks.

The sawtooth track uses the **same source** as the crash ticks, replacing the
old frame-model ELMy phase track. Only surviving verified crashes receive
present point marks; other positive intervals remain candidates. With no
surviving crash the strip and leader disappear and the track is honestly
named **sawtooth cand.** All consumed source/evidence files are hashed in the
record. `--sawtooth-crashes` and `--sawtooth-labels` are aliases; different
independent label/crash sources are rejected.

AE source priority is expert review, supplied ae-ours inference, valid stored
ae-ours, then the existing interferometer frame fallback. Fixed AE probability
threshold 0.7; NTM 0.63; sawtooth explicit confidence 0.6; TokEye 0.2. Infer
only local corpus CO2 using the pinned paper checkpoint, with no training:

```bash
CUDA_VISIBLE_DEVICES=0 \
/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python \
  scripts/labeler/paper/fig1_ae_infer.py --shots 201978 201973 203187 \
  --out "$LABELER_ROOT/round4/fig1/ae_ours.csv"
```

Inference reads the model-card transform, 1024-frame windows with 128-frame
context, causal 25-ms aggregation and validity. It writes only isolated
predictions plus metadata in round4/fig1; production labels are untouched.

Expert `iscrowd=1` ELM rows encode solid, distinctly marked ELMing periods,
including the source's category-2 crowd convention. Ordinary uncertain rows
are lighter/hatched and receive no definite triangles. Triangles are explicitly
keyed **D-alpha peaks, figure threshold** (25-sample median; 4 MAD; 3-sample
peak spacing). Hatching only denotes uncertainty; blank denotes no assessment.

Projection uses raw end-exclusive present intervals, AE >=60 kHz, NTM <60 kHz
with uniquely dominant measured n=1/2. The n map is never extrapolated beyond
its measured band. Unmeasured TokEye-mask pixels are white; other n is purple
with a diamond key. NTM outlines are max-pooled to the print display grid,
then thickened and clipped to time/frequency boundaries. Independent audit
enumerates projected coordinates against **raw spans**, without calling
`present_columns` or interval union. Frequency support for the near-15-kHz
harmonic interpretation is recorded (n=1 at 6–10 kHz, n=2 at 12–18 kHz,
2:1 within 1.2 kHz for at least 50 ms); the caption qualifies the interpretation.

Confinement source order: saved four-class review, curated regime intervals,
then D-alpha transition detector. The curated run directory defaults to
`runs/labeler/confinement/v1` beside the main label tables. Missing sources
raise errors. Regime tracks retain the name **regime**; the binary fallback
is **H-mode**, whose absence does not establish L-mode.

Other-shot renders use `--out "$LABELER_ROOT/round4/fig1/alt_<shot>"` and their
corresponding record paths. Only 201973 and 203187 also use the supplied AE CSV.
Regime windows: 191376 1500–2900 ms, 191782 1800–3700 ms. No blind test shot is
selected. Audit the six records and generate the ranked shortlist with:

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_audit.py \
  --out outputs/labeler/paper/fig_interpreter_tokeye/fix_round2_audit.json
```
