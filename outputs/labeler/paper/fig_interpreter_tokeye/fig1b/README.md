# TokEye Figure 1, owner revision (shot 199563, linear frequency axis)

Supersedes the six-shot set in the folder above (primary 201978, three frequency scales).
The owner asked for the original Figure 1 shot and a linear, not log, frequency scale.

**Shot 199563, 700-5800 ms** (train split; in neither the AE nor the NTM detector's
training set). Vector PDF, 150-dpi PNG, record, caption and appendix are under
`$LABELER_ROOT/round4/fig1b/` (`fig_interpreter.{pdf,png,json}`, `caption.tex`,
`appendix.txt`); the small records and the audit are here. The PDF is 6.75 x 5.6 in with
7-point text at the smallest.

**Axis.** Both spectrograms share one linear 0-250 kHz axis (ticks every 50 kHz), with no
scale break, no stretching and no note. The raw spectrogram is the wide-range pass on one
colour scale. The higher-resolution 0-50 kHz pass is not drawn raw (at about 1.15 kHz per
print pixel it adds nothing visible and would put a normalisation seam at 50 kHz); the mask
keeps it below 50 kHz, with the wide pass above, because its decimation filter rolls off
above about 50 kHz. The toroidal-n view (below 30 kHz) is the bottom 30/250 of the processed
panel (0.19 in): small on a linear axis, but legible. Pink (AE) starts at 60 kHz as before.

**Label rows (present spans, ms; all rows are non-expert).**

| Row | Source | Present |
|---|---|---|
| AE | ae-ours CO2 detector (25 ms bins) | 850-1050, 1075-1200, 1250-1475, 1525-1675, 1700-1725, 2075-2150, 3725-4000, 4050-4200 |
| NTM | detector (suggestions) | 3500-3550, 3650-3800 |
| confinement | released confine-ours roster (model; QH unreviewed) | L 700-1595; uncertain 1595-2153; QH 2153-4537; uncertain 4537-4633; H 4633-5001; blank after 5001 |
| ELMs | ELM frame model | 2100-2150, 2200-2250, 2400-2650, 2800-2850, 2900-3000, 3050-3150, 3300-3750, 4150-4200, 4250-4300, 4500-4550, 4650-4750 |
| sawtooth | physics export (fix5), blank where ECE is cut off | 5004-5257, 5289-5305, 5333-5381, 5550-5589, 5667-5678; mostly uncertain elsewhere, 539 ms unassessed |

The confinement row is the paper's four-class `confinement` event (1 high, 2 low, 3 qh,
4 wpqh, 5 uncertain), read through `figure_sources.confinement_track` from the released
roster `data/events/confinement/extend_confine_ours/roster.csv` (the catalog holds no
curated or reviewed confinement label for this shot). Colours are the catalog's (L green,
H blue, QH teal; WPQH would be navy, not drawn here); class 5 is hatched; time outside the
roster's beam-on segments is blank (after 5001 ms). The roster's tiers for this shot are
`model` (L, H, and the 2055-2153 and 4537-4572 uncertain pieces) and `unreviewed` (QH and
the 1595-2055 and 4572-4633 uncertain pieces, whose predicted class is QH); the shot is
read by the ensemble of the fold models and lies past the last curated shot, so the
network never saw its campaign. The source text at right is "model (unreviewed)". The
caption says the ELM intervals overlap the QH span (2200-4537 ms; nine detector ELM
spans), where QH is ELM-free by definition: the ELM boxes are the detector's and the QH
label the model's, both unreviewed. 12 ECE-supported crash candidates (all after 5000 ms)
sit on the strip.

**Sawtooth source.** The final population export of the saw stream,
`$LABELER_ROOT/round4/saw/fix5/` (`labels/SHA256SUMS` pinned by its sha256 in
`sawtooth_source_manifest.json`). 199563 is in `labels/population-100.csv`; the render reads
the shot's physics record (`shots/199563.json`, which also holds the density guard), and
`fig1_refresh_sources.py` checks its states and crash points equal that shard's rows.
The immutable copies are under `$LABELER_ROOT/round4/fig1b/saw_source/`.

## Reproduce

Start from a clean committed HEAD (the render records it).

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/fig1b
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
PX="pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker"
R=$LABELER_ROOT/round4/fig1b
$PX python scripts/labeler/paper/fig1_refresh_sources.py          # pin the saw export
$PX python scripts/labeler/paper/fig_interpreter_tokeye.py --shot 199563 \
  --tmin 700 --tmax 5800 --out "$R" --ae-labels "$R/ae_ours.csv" \
  --sawtooth-source "$R/saw_source/shots/199563.json" \
  --record outputs/labeler/paper/fig_interpreter_tokeye/fig1b/199563.json
$PX python scripts/labeler/paper/fig1_audit.py --rebuild-primary \
  --records outputs/labeler/paper/fig_interpreter_tokeye/fig1b \
  --out outputs/labeler/paper/fig_interpreter_tokeye/fig1b/audit.json
```

`ae_ours.csv` is the isolated ae-ours inference for 199563 (its `.meta.json` is beside it).
The audit checks the linear axis (ticks, no breaks, equal raw and processed panels), the
n-view height, tick and text clearance, the AE chip and key placement, final-raster AE
clipping above the n view, exact sawtooth states against the pinned record and shard,
training membership, the caption and appendix text, PDF size and a byte-identical rebuild.
