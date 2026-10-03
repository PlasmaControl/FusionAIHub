# TokEye Figure 1 — fix round 6

Primary: **201978, 1500–3300 ms**. Raw Mirnov magnetics, D-alpha (a.u.) and
NBI lead to TokEye's coherent mask, a **0–30 kHz** measured-n panel and aligned
labels. Include at **textwidth**: **6.75 × 5.5 inches**, fonts **≥7 pt**.
Vector PDFs and native **150-dpi** PNGs are under `$LABELER_ROOT/round4/fig1/`,
with five `alt_<shot>/` directories. The six records/captions here,
`fix_round6_audit.json` and `fix_round6.md` are current; earlier rounds are history.

## Sources and display

Expert sawtooth review now precedes physics states, following `lf.TRACKS`.
186636 therefore displays the expert row. Rows require positive source PRESENT
time in the window; hidden rows remain recorded and are explained in captions.
ECE-supported crash candidates remain separate physics evidence, with only
channel-order geometry and an inclusive ±5 ms D-alpha veto. The catalog's
sawtooth frame model is excluded. No rotating-mode precursor or seeding is inferred.
The immutable physics evidence snapshot is pinned by
`sawtooth_source_manifest.json`; labels and production stores remain read-only.

AE tint intersects PRESENT times at **≥80 kHz**, with recorded **25 ms**
(paper ae-ours) or **10 ms** (earlier frame detector) cadence. AE training targets
used TokEye's mask; overlap is not independent confirmation. Expert AE review
precedes supplied/stored predictions and the earlier frame fallback.

NTM requires both a component's unique dominant measured **n=1 or 2** and
individual measured **n=1 or 2** pixels at **≤30 kHz**. Its visible **white dashed
contour is opaque, 1.2 pt (2.5 print pixels)**, with a thin black halo and a
matching key. Contours border measured support; support-pixel audits do not
count the stroke as a new n measurement. The magnetic detector remains
unverified and shares the displayed inputs. Constant full-colour n hues remain.

The harmonic caption sentence is removed. The audit retains the ratio test
**|f2/f1 − 2| ≤ 0.1**, its **50 ms** floor, and the passing fraction among all
columns with both measured ridges. Late untagged frequency bounds now use
**all** eligible coherent pixels after the last PRESENT AE interval during
ABSENT AE time in the detector band; no component-duration cutoff truncates them.

Expert ELM categories remain unchanged; circles mark crowd intervals and
triangles mark threshold D-alpha peaks within PRESENT intervals. The primary
caption flags the first large spike before the expert interval. The audit
records its timing and the alternate 201973 H-mode/ELM contradiction for the owner.
Curated L-mode has a darker bar and regime key, distinct from absent grey.
Raw bands are normalised separately; the 55 kHz fold changes resolution.
Processed 30–55 kHz is omitted to enlarge the n panel.

Cached arrays carry the checkpoint hash, actual trimmed waveform sample/timing
fingerprint, preprocessing/inference code hashes and runtime versions. Missing
or mismatched identity triggers regeneration, including for superset caches.
Each record and audit includes the cache SHA-256 digest.

## Reproduce and audit

Run from this worktree with local read-only inputs:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/fig1
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
export SAWTOOTH_SOURCE="$LABELER_ROOT/round4/fig1/saw_source/fix2-79a2c18ed4aa/shots"
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig_interpreter_tokeye.py --shot 201978 \
  --tmin 1500 --tmax 3300 --ae-labels "$LABELER_ROOT/round4/fig1/ae_ours.csv" \
  --sawtooth-source "$SAWTOOTH_SOURCE"
```

Render all six externally from a clean committed HEAD, then copy JSONs/captions
here together. `--record` is available but would dirty HEAD between renders.
All captions use `fig:interpreter-<shot>`.

| Shot | Split | Window (ms) | Supplied AE CSV |
|---|---|---|---|
| 201978 | train | 1500–3300 | yes |
| 201973 | val | 1600–3350 | yes |
| 203187 | train | 1700–3150 | yes |
| 186636 | val | 1300–3900 | no |
| 191376 | train | 1500–2900 | no |
| 191782 | train | 1800–3700 | no |

Alternates use `--out "$LABELER_ROOT/round4/fig1/alt_<shot>"` and explicit windows.
No blind test shots are used. For the current verification:

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary \
  --out "$LABELER_ROOT/round4/fig1/fix_round6_audit.json"
```

The audit checks cache identity/digests, source priority/hashes, exact source
states, display smoothing/visibility, regime keys/categories, measured NTM
support and clipping, code/figure hashes, fonts/text bounds, caption length,
external/committed copies, PDF dimensions and byte-identical primary rebuilding.
Inspect every native PNG before use. 201973 is the strongest visual alternate;
191376/191782 are diagnostic/regime comparisons without AE/NTM coincidences.
