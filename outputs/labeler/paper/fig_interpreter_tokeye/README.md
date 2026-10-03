# TokEye Figure 1

Primary: **201978, 1500–3300 ms**, at text width. Both spectra share identical
frequency rows and heights: 0–30 kHz, a short 30–55 kHz strip, and 55–250 kHz.
The only fold is at 55 kHz. Processed white pixels are coherent TokEye support;
measured mode-number hues replace white only below 30 kHz. The fixed AE callout
has been removed; the legend explains pink time-overlap tint.

Vector PDFs and native 150-dpi PNGs stay in `$LABELER_ROOT/round4/fig1/`.
The current records/captions, `audit.json`, `report.md` and source manifest are
here. Superseded reports/plans are consolidated in
`docs/labeler/fig1_fix_history.md`; historical audits are retained in git.
The dispatch `reports/fig1.md` is the same current report.

## Sources and limits

AE: a neural detector on CO2 interferometer data, with its paper operating
point and temporal bins recorded in each JSON. Pink intersects PRESENT time
and the model's ≥80 kHz band. Training targets used TokEye's mask; this is
not independent physical confirmation. The 80 kHz floor follows the model's
input band; extrapolating the detector to 60–80 kHz is unsupported. The primary
caption explains that evenly spaced magnetics-only lines after 2.8 s remain
unlabelled. Expert review precedes supplied/stored inference and frame fallback.

NTM detector suggestions are labelled **“F1 0.46, below acceptance bar”** in
both caption and leader. The evaluation JSON and SHA-256 are recorded; shared
magnetic inputs are disclosed. Outlines require dominant and pixel n=1 or 2
at ≤30 kHz. Their white dashed key is drawn on a black swatch. Imported NTM
intervals keep their own source wording. “Other n” uses the same square handle
as the measured hues. L-mode shading from pre-transition H-mode-absent time
is explicitly marked **L-mode (inferred)**; the source categories stay exact.

Sawtooth: expert review first, otherwise the physics records pinned from
`FusionAIHub-r4-saw` commit **ad0ca40f**. `--sawtooth-source` remains configurable.
The manifest records the source commit, completion record and each immutable
shot hash. Captions name the source in plain words and always report exact
state durations, including hidden rows; uncertainty/unassessment never imply
physical absence. ECE density guards are conservative proxies, with missing
Bt disclosed. ECE-supported crash ticks are separate research evidence with
channel-order geometry and a ±5 ms D-alpha veto.

On the primary, these research physics states conflict with the catalog's
`ece_sawtooth` v3 and `sawtooth_frames` v2, which both contain PRESENT intervals.
The manifest and current report retain both comparisons. The **controller
pins the paper's shipped sawtooth source at integration**; the research source
still lacks independent physical validation. Source versions never appear in
paper captions or axes.

Expert ELM intervals remain unchanged. The primary caption gives the largest
D-alpha spike and first expert interval timings; 201973 and 203187 captions
disclose ELM intervals overlapping H-mode-detector absent time. The owner must
resolve those source disagreements. 203187 also retains AE after NBI turns off;
186636 has nearly stationary pink structures that may be pickup. These are
physical-review caveats, not corrected event identities.

## Available renders

| Shot | Split | Window (ms) | Role | Supplied AE CSV |
|---|---|---|---|---|
| 201978 | train | 1500–3300 | Primary | yes |
| 201973 | val | 1600–3350 | Best visual alternate; ELM/H-mode conflict disclosed | yes |
| 203187 | train | 1700–3150 | Alternate requiring ELM/H-mode and post-NBI review | yes |
| 186636 | val | 1300–3900 | Imported-NTM alternate; possible pickup | no |
| 191376 | train | 1500–2900 | **Unsuitable alternate**: no AE, n=1 with NTM absent | no |
| 191782 | train | 1800–3700 | **Unsuitable alternate**: no AE, n=1 with NTM absent | no |

The unsuitable shots remain diagnostic renders only. No blind-test shot is used.
All scientific counts, source hashes, intervals and checks are in `audit.json`.

## Reproduce

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
  --sawtooth-source "$SAWTOOTH_SOURCE"
```

For other windows use `--out "$LABELER_ROOT/round4/fig1/alt_<shot>"`,
adding the supplied AE CSV only where specified. Render all from a clean
committed HEAD before copying the six JSON/caption pairs here together.

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary \
  --out "$LABELER_ROOT/round4/fig1/audit.json"
```

The audit checks exact source states and priorities, source/code/cache hashes,
identical frequency mapping, visible keys, bounds/fonts/captions, native PNG
DPI, final-PNG pink clipping (one-pixel boundary tolerance), measured NTM support,
PDF dimensions and byte-identical primary rebuild. These rendering checks do
not measure event identification accuracy. Inspect every PNG at native 150 dpi.
