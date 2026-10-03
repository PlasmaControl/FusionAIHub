# TokEye Figure 1

Primary: **201978, 1500–3300 ms**. Final renderer: **a04b1c28**.
Vector PDFs and native 150-dpi PNGs are under
`$LABELER_ROOT/round4/fig1/`; small records/captions/appendix files and the audit
are here. [report.md](report.md) is the current eighth-round report.
The dispatch report has an appended **Fix round 8** section.

Both spectra have matching ranges/heights and scale-break glyphs at **30 and
55 kHz**, with a restored **40 kHz** tick. The 0–30 kHz range is vertically
expanded; the 30–55 kHz strip is compressed. Pink is full-opacity AE time/band
coincidence at ≥80 kHz. Solid orange outer contours enclose measured n=1/2
NTM support at ≤30 kHz after display-hole filling. The n key reads
**n=1, n=2, n=3, other n** in four columns.

The caption explains the three tiers and sources in 67 words on the primary.
Detailed caveats are in [201978.appendix.txt](201978.appendix.txt) and the
other `<shot>.appendix.txt` files: training-target/shared-input dependence,
timing/cadence, inferred L-mode, expert ELM conflicts and research physics
limits. Late structures are described as lacking a positive AE-detector
label; the field caveat says **Bt not in local corpus**.

NTM stays **candidate suggestions (detector F1 0.46, below bar)** on 201978.
Both its 3,550-shot training list and AE's 180-shot training list exclude
201978. The lists, provenance hashes and checkpoint hashes are recorded in
`201978.json:detector_training` and independently checked by the audit.
Cohort membership and detector training membership are distinct.

The paper ships the **physics sawtooth source**. Its primary row now shows
hatched uncertainty and blank unassessed time; all selected sawtooth rows use
four-state rendering. The current immutable snapshot is **ad0ca40f**.
**--sawtooth-source** remains configurable; the controller re-pins to the
final saw head at integration. Expert sawtooth review wins on 186636.
The source manifest retains catalog disagreements and exact content hashes.

Shot-specific roles, primary AE-label coordinates and alternate appearance
caveats live in [fig1_annotations.json](../../../../docs/labeler/fig1_annotations.json),
loaded through **--annotations** and hashed in each record.

| Shot | Split | Window (ms) | Role | Supplied AE CSV |
|---|---|---|---|---|
| 201978 | train | 1500–3300 | Primary | yes |
| 201973 | val | 1600–3350 | Best visual alternate; ELM/H-mode conflict | yes |
| 203187 | train | 1700–3150 | ELM/H-mode and post-NBI review | yes |
| 186636 | val | 1300–3900 | Imported NTM; possible pickup | no |
| 191376 | train | 1500–2900 | Unsuitable diagnostic: no AE, n=1 with NTM absent | no |
| 191782 | train | 1800–3700 | Unsuitable diagnostic: no AE, n=1 with NTM absent | no |

No blind-test shot is used. Model outputs and source intervals are unchanged.
All six final PNGs and the primary rebuild were visually inspected at 150 dpi.
The primary was also inspected at column width: colours/tracks survive but
source text becomes small. Include at **text width (6.75 inches)** for 7-point
type; the column reduction is a visual check only.

## Reproduce

Start from a clean committed HEAD and render all six externally before copying
the sidecars here, so every render records the same clean source commit.

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/fig1
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
export SAWTOOTH_SOURCE="$LABELER_ROOT/round4/fig1/saw_source/ad0ca40f-ad0ca40f1a7e/shots"
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig_interpreter_tokeye.py --shot 201978 \
  --tmin 1500 --tmax 3300 --ae-labels "$LABELER_ROOT/round4/fig1/ae_ours.csv" \
  --sawtooth-source "$SAWTOOTH_SOURCE" \
  --annotations docs/labeler/fig1_annotations.json
```

For alternates use the table's window and
`--out "$LABELER_ROOT/round4/fig1/alt_<shot>"`, supplying the AE CSV only where
listed. Copy the JSON, caption and appendix sidecars after the whole batch.

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary \
  --out "$LABELER_ROOT/round4/fig1/audit.json"
```

The audit checks exact states/source priority, source/code/cache/checkpoint
hashes, training exclusion, frequency scales/tick spacing, regime text versus
ELM boxes, marker clearance, keys/fonts/bounds, native PNG DPI, final-raster AE
clipping, measured NTM support, caption/appendix identity, PDF dimensions and
byte-identical primary rebuild. These are rendering and provenance checks.
