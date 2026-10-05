# TokEye Figure 1

> **Superseded for Figure 1 by [fig1b/](fig1b/README.md)** (owner revision: shot 199563,
> one linear 0-250 kHz frequency axis). The record below describes the earlier six-shot
> render with three frequency scales; it is kept for history.

Primary: **201978, 1500–3300 ms**. Final renderer commit: **34937d65** (all six renders
began from it, clean). Vector PDFs and native 150-dpi PNGs are under
`$LABELER_ROOT/round4/fig1/`; small records, captions, appendices and the audit are here.
[report.md](report.md) is the current report (eleventh and final fix round), mirrored at
the dispatch report. Earlier rounds are in git; see
[fig1_fix_history.md](../../../../docs/labeler/fig1_fix_history.md).

**Why 201978.** Chosen in round 5 for a clear AE cascade and persistent low-frequency n
structure (n=1 with n=2/n=3 after 2.6 s), as a training-split shot that neither the AE nor
the NTM detector trained on. **Only the ELM row is expert-reviewed**; the AE, NTM, H-mode
and sawtooth rows are generated. 186636 is the alternate with expert and imported
provenance (NTM imported, ELMs and sawtooth expert).

Both spectra have matching ranges and heights and scale-break glyphs at **30 and 60 kHz**:
three frequency scales, 0–30 kHz stretched, 30–60 kHz and 60–250 kHz compressed. The upper
axis is exactly the AE region. The higher-resolution spectrogram covers 0–50 kHz; 50–60 kHz
comes from the wide pass because the higher-resolution pass's decimation filter rolls off
above about 50 kHz. Pink is full-opacity AE time/band coincidence at **≥60 kHz** (the owner's
decision; the detector's input band is 80–250 kHz, so 60–80 kHz pink is coincidence, not
detection). A small **AE** chip at 2100 ms / 200 kHz names it; there is no leader line. A thin
dashed orange outline shows the rest of a tagged NTM component outside the NTM-positive
time. Solid orange outer contours enclose
measured n=1/2 NTM support at ≤30 kHz after display-hole filling; outline regions under
**100 print pixels** (the tiny 3–5 kHz fragments) are not drawn, a display-only rule that
leaves tags, counts and audits unchanged and is recorded in each record. The two-column key
beside the processed 0–30 kHz panel reads **n=1, n=2, n=3, other n**, titled **toroidal
mode number n (Mirnov array)**. The AE chip sits at **2100 ms, 200 kHz**; its anchor is in
`fig1_annotations.json`.

The primary caption (**97 words**) is descriptive: top (with the three scales), middle and
bottom panels, the pink and orange highlights with their conditions, that the AE detector
was trained on TokEye-mask-derived targets (so pink is not independent of TokEye), and the
NTM detector's held-out F1 0.46 against its 0.7 bar. A harmonic clause appears only when a
ridge pair passes the 50 ms floor **and** a passing fraction of 0.6; on 201978 the fractions
are 0.46 and 0.41, so it does not. Alternate captions add only the qualifications that apply
to them (201973 and 203187 the expert-ELM/H-mode disagreement, 203187 the post-NBI AE
suggestions, 186636 the possible pickup lines); the longest is 117 words (cap 120). Every
caption number and clause is computed from the records; no shot-specific text lives in
`src/labeler/paper/figure_sources.py`.

The generated appendix (`<shot>.appendix.txt`) carries what the caption leaves out: the
mask chain including the persistent-row step and its counts; track sources; thresholds
for detector tracks only (TokEye 0.2; AE 0.5 for ae-ours, 0.7 for the frame detector; NTM
0.63); the AE-source branch (ae-ours: input band and TokEye-mask targets; frame detector:
trained on the owner's reviewed labels, TokEye only up-weights MHD-absent frames); NTM
limits; the ridge frequency ratios as "within 5% of n×f(n=1) in X of Y ms where both are
measured (P%)" with the caveat that a ratio cannot separate harmonics from phase-locked
coupled modes (m needs EFIT q or the poloidal array); whether the shot is in each detector's
training set (the frame AE detector's split is read for 186636, which is in neither its
train nor test split; 191782 is in the NTM detector's training set); the sawtooth summary;
and the late-line and D-alpha notes (the largest D-alpha spike, 2297 ms, precedes the expert
span). Boilerplate that does not apply to a shot (open circles, AE sentences) is omitted.

The track source column reads **detector (suggestions)** for NTM, and for the sawtooth row
one line, **physics; blank: ECE cut-off** where that applies, else **physics labels**.
Sawtooth is a point-event row, never a mode tag. The paper ships the **physics sawtooth
source** (snapshot **ad0ca40f**); **--sawtooth-source** stays configurable and the
controller re-pins it at integration. Expert sawtooth review wins on 186636. The source
manifest keeps the catalog disagreements and exact content hashes.

Shot-specific roles, the primary AE anchor and alternate appearance caveats live in
[fig1_annotations.json](../../../../docs/labeler/fig1_annotations.json), loaded through
**--annotations** and hashed in each record.

| Shot | Split | Window (ms) | Role | Supplied AE CSV |
|---|---|---|---|---|
| 201978 | train | 1500–3300 | Primary | yes |
| 201973 | val | 1600–3350 | Best visual alternate; ELM/H-mode conflict | yes |
| 203187 | train | 1700–3150 | ELM/H-mode and post-NBI review | yes |
| 186636 | val | 1300–3900 | Imported NTM; possible pickup | no |
| 191376 | train | 1500–2900 | Unsuitable diagnostic: no AE, n=1 with NTM absent | no |
| 191782 | train | 1800–3700 | Unsuitable diagnostic: no AE, n=1 with NTM absent | no |

No blind-test shot is used. Model outputs and source intervals are unchanged; the audit
compares all scientific records with round-10 commit **3941c5b6**, allowing only what the
60 kHz AE floor and fold move (AE tag counts, the late untagged band, the projection audit,
the harmonic gate) and the added AE entry of `detector_training`. Include the figure at
**text width (6.75 inches)** for 7-point type.

## Reproduce

Start from a clean committed HEAD and render all six externally before copying the
sidecars here, so every render records the same clean source commit.

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/fig1
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
export SAWTOOTH_SOURCE="$LABELER_ROOT/round4/fig1/saw_source/ad0ca40f-ad0ca40f1a7e/shots"
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig_interpreter_tokeye.py --shot 201978 \
  --start 1500 --end 3300 --ae-labels "$LABELER_ROOT/round4/fig1/ae_ours.csv" \
  --sawtooth-source "$SAWTOOTH_SOURCE" \
  --annotations docs/labeler/fig1_annotations.json
```

For alternates use the table's window and
`--out "$LABELER_ROOT/round4/fig1/alt_<shot>"`, supplying the AE CSV only where listed.
Copy each render's JSON, caption and appendix into this folder as `<shot>.json`,
`<shot>.caption.tex` and `<shot>.appendix.txt` after the whole batch, then audit.

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary \
  --baseline-ref 3941c5b6a00216215a6ce5d99b184cd5af20ea82 \
  --out "$LABELER_ROOT/round4/fig1/audit.json"
```

The audit checks exact states and source priority, source/code/cache/checkpoint hashes,
training exclusion, frequency scales and tick spacing, regime text versus ELM boxes, marker
clearance, keys, fonts and bounds (including the one-line source text and the AE chip),
native PNG DPI, final-raster AE clipping, measured NTM support, the caption and appendix
text and their per-source branches, the harmonic gate against the recorded support,
detector training membership recomputed from each training list, PDF dimensions and a
byte-identical primary rebuild.
These are rendering and provenance checks.

Supply both time bounds or neither; **--start/--end** alias **--tmin/--tmax**. Incomplete,
reversed or non-finite pairs fail explicitly before source access.
