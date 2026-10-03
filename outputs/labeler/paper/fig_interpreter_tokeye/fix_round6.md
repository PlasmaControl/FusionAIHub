# Figure 1 — fix round 6

Status: **DONE_WITH_CONCERNS**. Both `fig1-sol6.md` and `fig1-opus6.md` were
read in full. Every Important and Minor finding is addressed. Owner label
flags and the existing evidence limitations below remain; no implementation
finding is open.

Commit range: **d94df795..HEAD**, branch **r4-fig1**. Implementation:
**cd8e07fb**; final print-margin adjustment and renderer revision:
**d3c0519a**. The final artifact commit contains the six records/captions,
audit and this report. All commits use `labeler:` and the requested
`Co-Authored-By: Codex gpt-6.1-sol <noreply@openai.com>` trailer.

This is the single current report. Superseded rounds 4–5 moved to
`reports/fig1-appendix-rounds4-5.md` beside the dispatch report; older committed
`fix_round*.md` and audits are historical.

## Result and evidence

Primary **201978, 1500–3300 ms** retains the raw → TokEye → labels layout.
All six final renders were made externally from clean committed **d3c0519a**,
then their records/captions copied into this worktree together. Every final
PNG was opened at native **150 dpi**: 201978, 201973, 203187, 186636, 191376,
191782. The white dashed NTM contours are readily findable on the primary's
blue and green ridges; the same line style appears in the key. They are opaque,
**1.2 pt / 2.5 print pixels** wide, with a thin black halo. The AE callout sits
in the black region without a leader. The ECE key uses the open right margin
by NBI, clear of the ELM key. Curated L-mode bars/key are visibly darker than
absent grey in both regime alternates.

Scientific numbers below come from the committed generator
`scripts/labeler/paper/fig1_audit.py` and its current JSON:
`outputs/labeler/paper/fig_interpreter_tokeye/fix_round6_audit.json` (identical
external copy: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fix_round6_audit.json`). Per-shot source metadata,
thresholds, intervals, figures and hashes are in `<shot>.json` in the same
committed output directory. These are descriptive coincidences, not
classification scores or independent physical identifications.

| Shot | Split | Window (ms) | AE components | NTM components | D-alpha triangles | ECE-supported candidates | Caption words |
|---|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 | 11 | 49 | 0 | 120 |
| 201973 | val | 1600–3350 | 154 | 14 | 65 | 3 | 148 |
| 203187 | train | 1700–3150 | 262 | 8 | 6 | 0 | 104 |
| 186636 | val | 1300–3900 | 307 | 4 | 0 | 0 | 98 |
| 191376 | train | 1500–2900 | 0 | 0 | 2 | 0 | 44 |
| 191782 | train | 1800–3700 | 0 | 0 | 0 | 0 | 37 |

There are **four train, two validation and zero blind test** shots. The primary
still has **319 AE / 11 NTM** tagged components. Audit `renders[]` records
**zero** time/band projection violations, unmeasured NTM support and n=3
support marked as NTM, plus valid source/code/cache/figure hashes, text bounds,
font sizes, caption labels and external/committed copies. The primary PDF and
PNG rebuild **byte-for-byte identically** (`reproducibility`). Support-pixel
audits describe the region bordered by the contour; the visible stroke is
not interpreted as additional measured n.

The general harmonic caption sentence is **deleted**. For primary NTM time,
**318 / 684** columns with both measured
ridges pass **|f2/f1 − 2| ≤ 0.1**, a fraction of
**46.49%**. Passing support is **325.632 ms**
out of **700.416 ms** simultaneous support. The **50 ms**
floor alone did not justify the removed general claim. Source:
`renders[shot=201978].harmonic_support`; the audit now reports both counts,
the denominator duration and the passing fraction.

Late frequency bounds now include **all** coherent pixels during ABSENT AE
time after its last PRESENT interval in the detector band, with no component
duration cutoff. The primary spans **80.078–249.512 kHz**, captioned as
**80–250 kHz mask pixels**, without physically identifying them as lines.
Source: `renders[shot=201978].late_untagged_high_frequency` (exact bounds,
pixel counts and times). The same rule supplies the conditional alternate
captions.

## Sources and changes

`figure_sources.sawtooth_track` uses the expert review from the first
`lf.TRACKS` source before physics states. `shot_tracks` uses that choice, and
`draw` preserves it. **186636** now displays the expert review from
`data/events/sawtooth_oscillation/review/labels.csv`, with **1884 ms present**
and **716 ms absent** in the window; the imported NTM interval remains inside
the reviewed sawtooth period. No seeding relation is inferred. Interval and
candidate-tick provenance are recorded separately. Other shots retain the
pinned physics snapshot; hidden rows get the caption clause **“Sawtooth: no
present time in this window.”** Source: each audit `sawtooth_interval_source`,
`sawtooth_states`, `sawtooth_state_duration_ms` and `sawtooth_track_shown`.

**201973** retains **3 ECE-supported crash candidates**, described with the
channel-order-only geometry caveat. They are research candidates, not validated
crashes. The audit's legacy `verified_crashes` count names these evidence-checked
candidates; it is not an independent validation claim. Candidate evidence uses
the immutable figure-owned physics snapshot
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/saw_source/fix2-79a2c18ed4aa/shots`, pinned by
`sawtooth_source_manifest.json`. No production label or physics source was edited.

`figure_sources.tokeye_fingerprints` records actual trimmed float32 waveform
samples, timing/rate/channel identity, preprocessing and inference source
hashes, and NumPy/SciPy/PyTorch versions. `tokeye_passes` validates these together
with the checkpoint before either exact or superset cache reuse. Missing or
mismatched metadata regenerates arrays. All six caches were regenerated from
local read-only inputs; each NPZ carries the fingerprints. Their digests and
fingerprints are checked again and included in audit `renders[].tokeye_cache`
and `checked_sources`. Primary cache SHA-256:
`f63e81f5fa115c3b085db16d59d4d058823f04f67eb8aea8a7ef27bcf05653a5`.

Other presentation fixes: **“Raw signals”** heading, explicitly labelled
**Mirnov magnetics** spectrogram, **D-alpha (a.u.)**, **n=1 or 2** throughout
current keys/prose, reader language for event overlap and ELM symbols, and
regime keys with darker L-mode bars. NTM contour support still requires
both unique dominant n and individual measured pixels to be 1 or 2 at
≤30 kHz. AE tint starts at ≥80 kHz and follows the source's recorded cadence.

Changed code: `src/labeler/paper/figure_sources.py`,
`scripts/labeler/paper/fig_interpreter_tokeye.py`,
`scripts/labeler/paper/fig1_audit.py`; covering regressions in
`tests/labeler/test_paper_figure_sources.py`. README contains reproducible
commands. No dependencies, labels, cohort, model thresholds or manuscript changed.

## Owner flags and limits

- **201978:** the first large D-alpha spike is at **2296.54 ms**, **11.46 ms**
  before the expert ELM interval starts at **2308 ms**. Caption flags it;
  triangles remain restricted to expert PRESENT times. Source: audit
  `largest_dalpha_peak_ms`, `first_large_peak_before_expert_ms`, and the render's
  expert ELM intervals. Labels were not altered.
- **201973:** the currently consumed H-mode source is ABSENT from **3013 ms**,
  yet expert ELM PRESENT overlaps **3013–3045**, **3077–3124**, and
  **3313–3350 ms** in this window. This is the requested H-mode/ELM contradiction,
  with exact current-source times rather than the review's approximate times.
  Source: audit `elm_hmode_conflicts_ms` and `confinement_intervals_ms`.
- The same automatic audit also flags **203187**: its largest plotted D-alpha
  spike at **2257.49 ms** precedes the first expert ELM interval by **274.51 ms**;
  ELM/H-mode-ABSENT overlaps are **2793–2856** and **3036–3080 ms**. These are
  owner review flags, not automatic ELM identifications or label edits.

NTM remains unverified and shares magnetic inputs; AE training targets used
TokEye's mask; ECE candidate geometry is not spatially validated. Sawtooth
crashes remain point candidates rather than rotating-mode projections, since
time overlap cannot establish a precursor or NTM seeding. This retains the
scientific deviation from the original brief. AE's ≥80 kHz floor and omission
of processed 30–55 kHz are retained from the reviewed design. No fetching,
training, threshold tuning or blind-test selection occurred. The primary
remains selected for its clear cascade and measured low-frequency structure;
201973/203187 are cascade alternates, 186636 adds reviewed/imported provenance,
and 191376/191782 are regime/diagnostic comparisons. Historical selection
criteria/ranking remain in the committed `fix_round2_audit.json`.

## Outputs and verification

Primary artifacts: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fig_interpreter.{pdf,png,json}` and `caption.tex`.
Each of `alt_201973`, `alt_203187`, `alt_186636`, `alt_191376`, `alt_191782`
contains the same four artifacts. PDFs are **6.75 × 5.5 inches**, vector
text/lines over raster spectra, with **≥7 pt** fonts; PNGs are **150 dpi**.
The six small JSON/caption copies, current audit, README and `fix_round6.md`
are committed. Large figures/caches stay outside git.

Required TMPDIR/local-data variables and `LABELER_NO_FETCH=1` were set. Python
used frozen/no-install pixi through the main manifest from this worktree.
Covering tests only, via the mandated wrapper:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_paper_label_figure.py
  -q -p no:cacheprovider
63 passed in 4.20s
```

Fresh Ruff check of all four changed Python files: **All checks passed!**
Ruff format --check: **4 files already formatted**. `git diff --check` passed.
All six generated captions also compiled with a plain LaTeX/T1 smoke document;
log: `round4/fig1/caption6-smoke.log`.
New regressions were first observed failing for review precedence, waveform/
code fingerprints, the harmonic fraction, all-pixel late bounds and disclosures.
Full logs: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/tests6{,-red}.log`, `lint6.log`, `format6.log`,
`render6_<shot>.log`, `cache6_201978.log`, `audit6.log`.

```text
pixi run --frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary
  --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fix_round6_audit.json
Audited 6 non-blind renders; zero projection violations
Primary PDF and PNG rebuild identically byte-for-byte
```

The mandated tmpsweep ran after the inference/render batches and audit/rebuild.
Nothing was pushed, merged, rebased or edited in the main checkout/manuscript.
Next: the controller can place the primary with its current caption and send
the recorded label discrepancies to the owner for review; no automatic label
correction is warranted.

## Current primary caption

DIII-D shot 201978. Raw bands normalised separately; TokEye's U-Net extracts coherent modes. Processed 30–55 kHz omitted. AE: neural interferometer detector (p≥0.5; training targets used TokEye's mask); NTM: magnetic detector (unverified; shared inputs); H-mode: D-alpha detector; ELMs: expert. Pink: AE time overlap in detector band ≥80 kHz; tint follows the detector's 25 ms bins. Toroidal mode number n is measured ≤30 kHz. Dashed outlines mark NTM time overlap on measured n=1 or 2 ridges whose dominant n is 1 or 2. Sawtooth: no present time in this window. Late 80–250 kHz mask pixels stay untagged where AE is absent. The first large spike precedes the expert interval. Circles mark expert intervals spanning many ELMs; triangles mark threshold D-alpha peaks (not annotations).
