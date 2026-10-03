# Whole-interval tearing-mode detection

The labels describe the rotating phase of a strong magnetic tearing mode, from
growth through decay or locking. Each observed onset is a point (`iscrowd: 0`);
the whole interval is a span (`iscrowd: 1`). The historical category directory
includes both classical and neoclassical tearing modes. These are offline
magnetic rule labels, not independently reviewed island identifications.

The machine-readable record is
[`tm_benchmark.json`](../../data/events/neoclassical_tearing_mode/benchmark/tm_benchmark.json).
It bundles label counts, agreement, coverage, model rows and their exact source
records in `benchmark/sources/`. Large arrays, predictions, galleries and logs
are under `$LABELER_ROOT/round4/tm/`, where
`LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`.

## Label rule and sources

`labeler.tearing.rule` reads `\MHD::N1RMS` and `N2RMS` in gauss on the
approximately 1 kHz grid. Inside the catalog window after the Ip-derived plasma
start, it applies these frozen choices:

| Choice | Value | Source or status |
|---|---|---|
| n = 1 seed | smoothed RMS > 12 G | Farre-Kaga et al. (2025) |
| n = 2 seed | > 6 G | local extension; no published n = 2 threshold |
| Release | max(1 G, 10% of the seed peak) | 10% onset convention from Farre-Kaga; 1 G floor is local |
| Duration | at least 50 ms above release after merging | interval extension of the published 50 ms criterion |
| Median smoothing | 5 ms | local spike rejection |
| Merge gap | at most 50 ms | local hysteresis choice |
| Seed duty | at least 50% of the joined seed run above seed threshold | local rejection of sparse spike trains |
| n = 2 harmonic rejection | n2 RMS > 0.4 × n1 RMS | local heuristic; does not establish independent islands |
| Onset unobserved | interval starts within 20 ms of the window opening | local censoring convention |
| Lock candidate | frequency ≤ 1 kHz for 20 ms after ≥ 1.5 kHz | local frequency-based proxy |

[Farre-Kaga et al., *Interpreting AI for Fusion*](https://arxiv.org/abs/2502.20294)
specifies n1 RMS above 12 G continuously for 50 ms, with onset at 10% of the
eventual peak, H-mode constraints and flat-top selection. Fu et al., *Physics of
Plasmas* 27, 022501 (2020),
[doi:10.1063/1.5125581](https://doi.org/10.1063/1.5125581), used 10 G and 50 ms,
and selected quiet negatives below 5 G. The interval extension here requires a
50 ms release-level span rather than a continuous 50 ms seed-level crossing;
it also omits the H-mode restriction and covers n = 2. Those differences are
explicit methodological choices, not claims that the literature specified them.
The peak-relative boundary uses future samples and is an offline annotation.

Seo's archive is a 25 ms growth-phase `tm_label`, not an independently supplied
full present span. Its exact historical preprocessing is retained in the published
model adapter. The new rule adopts the documented magnetic growth/onset concept;
it does not assume the archived point or forecast label stays true throughout a
mode's later lifetime. Agreement below uses the archive itself rather than an
invented reconstruction of Seo's threshold.

The lower release threshold extends a seeded event both backwards to growth and
forwards through decay. Short dips are merged. The interval ends at decay, the
catalog plasma-window end, or a nearby locking candidate. `N1FREQ`/`N2FREQ`
supplies that candidate: a drop near the last 150 ms of an interval or within
100 ms after its end can truncate it and set `locked: true`. This is frequency
evidence, not confirmation by a dedicated locked-mode saddle-loop signal.
Born-locked modes and the stationary phase after the flag are not labelled by
this rotating-mode rule. Missing frequency evidence leaves the attribute unset;
unflagged population intervals are not verified negatives for locking. Exact
frequency coverage is in the benchmark's `locking_coverage` record.

Both event types carry toroidal number `n`. Poloidal number `m` requires
`m = n q` at an independently located island radius (for example, ECE flattening).
`labeler.tearing.surface.supported_m` supports that hook but the present pipeline
has no resolved island radii. All `m` values are empty. An EFIT profile admitting
only one rational surface is insufficient radial evidence; the takeover removed
the previous assignment on that basis. Omnimode m is not used.

The remaining observed window is absent. A firing during the excluded ramp-up
is uncertain; missing RMS stretches of at least 50 ms are not observable and
are excluded from scoring. The labels use the processed toroidal magnetic RMS;
they do not automatically require a spectrogram line below 30 kHz. Weak coherent
lines can therefore remain outside this strong-mode label definition.

## Labels and gallery

The committed cohort table is
[`extend_tm_interval/tm_interval.csv`](../../data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv)
with its adjacent metadata. The population table and the detailed amplitude
tables are external:

- `labels/tm_interval_population.csv` and `.meta.json`;
- `labels/tm_intervals_full_{cohort,population}.csv`;
- `labels/plasma_start_{cohort,population}.json`.

| Set | Requested | Labelled | Shots with mode | Intervals n1 / n2 | Onset points | Locked flags | Missing RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohort | 500 | 500 | 193 | 199 / 74 | 271 | 84 | 0 |
| Population | 4,872 | 4,848 | 1,595 | 1,704 / 619 | 2,298 | 84 | 24 |

Source: benchmark `label_counts`, produced by `tm_label.py`. The sets overlap;
their counts must not be added. Population missing-signal shots are listed in
the population metadata. Existing RMS records were used; no additional signals
were fetched during the takeover. A labelled shot can contain unobservable time.

`tm_gallery.py --n 12 --seed 0` draws uniformly from labelled development shots,
with blind cohort test shots excluded. Both galleries were opened and visually
inspected: `$LABELER_ROOT/round4/tm/figures/tm_gallery_mhr.{pdf,png,json}` and
`tm_gallery_mirnov.{pdf,png,json}`. They show 0–50 kHz spectrograms, interval
overlays, n1/n2 RMS and seed/release thresholds. MHR coverage gaps are grey;
the longer Mirnov record provides a second check of the same shots. Strong RMS
growth and decay match the marked intervals. Weak lines, including the line
on shot 187072, need not cross the seed threshold. No visually clear boundary
failure justified changing the rule. The rule is frozen; the review and its
source hash are recorded in `gallery_review`.

## Agreement with historical onsets

`tm_agreement.py` reads survival onsets from `raw/tm_labels.h5`, and matches
Seo's archived feature rows to physical time with `validate.archived_truth`.
Reference coverage is limited: many cohort shots have no usable Seo match.
The primary comparison counts an onset inside a same-n1 interval or within
100 ms of its edges. The archives sample at 20–25 ms. The strict comparison
with zero tolerance is also recorded; the tolerant rate is not literal
containment. Error = reference onset minus interval start, conditional on a
match; positive errors mean the new interval begins earlier.

| Reference / set | Covered shots | Onsets | Match / miss (100 ms) | Strict containment | Median error [Q25,Q75] ms | Mean absolute error ms | Intervals with no onset |
|---|---:|---:|---:|---:|---|---:|---:|
| Seo / development | 80 | 26 | 25 / 1 | 15 / 26 | 73 [-30,217] | 158.4 | 14 / 38 |
| Seo / all cohort | 92 | 29 | 28 / 1 | 18 / 29 | 81.5 [-26.25,212.5] | 152.2 | 14 / 41 |
| Survival / development | 175 | 67 | 59 / 8 | 24 / 67 | -3 [-21.5,16.5] | 95.1 | 32 / 91 |
| Survival / all cohort | 200 | 72 | 64 / 8 | 27 / 72 | -2.5 [-21.25,37.25] | 123.5 | 37 / 101 |

Sources: bundled `agreement_{seo,survival}_cohort_{dev,all}.json`, including
`agreement_strict`. Per-onset and per-interval CSVs remain in external
`agreement/`. Both n1 and any-n summaries are available. All-cohort agreement
is label auditing only, not model selection or blind-test benchmark scoring.

Seo development errors range from -92 to 585 ms: 12%, 36%, 52% and 80% of
matched errors are within 25, 50, 100 and 250 ms, respectively. The lone
100 ms miss is shot 186640: its interval starts later. Of the 14 unmatched
intervals, four are on reference-positive shots and ten on reference-quiet
shots. Some modes begin outside the archive's limited growth-phase rows.
With literal containment, 11 development onsets miss; ten of these are the
near-boundary matches admitted by the 100 ms tolerance. The rule was not
shifted to force those archive timestamps into the spans.

The survival development misses are two later intervals, two short bursts
and four below-seed events. Its large positive error tail reaches 1,825 ms
(2,149 ms on all cohort), consistent with an onset inside an already running
interval; a high tolerant match rate does not imply precise boundaries.
The code does not promise that every historical onset is inside a new span.

## Detection benchmark

Targets are presence at the centres of absolute 10 ms bins. Category 2 and 3
bins, invalid model inputs and non-finite scores are excluded. AUROC and
average precision use pooled score-quantile histograms (1,024 quantile
thresholds); they approximate exact ranking metrics. F1 uses a published
threshold or an inner-validation threshold. All CIs resample whole shots
1,000 times with seed 0 and report percentile 95% intervals. The number of
valid bootstrap replicates is recorded per metric.

For segmental F1, predicted and true positive runs close gaps up to 50 ms
and discard runs shorter than 50 ms. Matching is greedy, one-to-one by
temporal IoU. The table uses IoU 0.5; source records also contain 0.3 and 0.7.
Segmental metrics apply to present-span detection. They are undefined for
legacy growth-phase and onset-within-horizon targets and are shown as dashes.

`tm-onsetcnn` rebuilds the published profile/scalar CNN with fresh weights
and a binary detection objective at t. The original ensemble has ten
members; the detection ensemble averages three seeds. Its pinned scalar
preprocessing reads some values at t + 25 ms, so this is offline detection
with that input lookahead, not a causal alarm. `tm-dsm` retains the published
38-feature preprocessing and 100–1,000-unit embedding, replacing the survival
mixture by a detection logit. It does not retain a survival likelihood.

`tm-ours` uses six Mirnov probes' per-bin spectrogram band power and toroidal
phase-coherence features, with three dilated temporal convolutions and three
seeds. Its primary input omits the processed RMS that defines the labels.
The RMS-input variant is an explicitly circular ablation. The trivial
baseline thresholds the bin maximum of median-smoothed n1 RMS at 12 G;
AUROC/AUPRC sweep that RMS score. A single globally development-tuned threshold
is only exploratory and is not a paper row.

The 450 cohort train/validation shots are assigned to five shot groups with
seed 0. Each outer held group is predicted by fresh networks fitted on the
other groups; a seeded 10% inner validation set controls early stopping and
chooses the F1 threshold. Normalization for the magnetic detector uses its
training shots only. Exact lists are in the bundled records' `explicit_splits`.
Prior-model learning rates were selected on fold 0's inner validation loss,
so this is development cross-validation rather than fully nested
hyperparameter validation. No cohort test shot enters any reconstructed
train/validation/held development group.

Published weights are also scored unchanged against intervals. The CNN
probability is shifted by 25 ms to the time its original output describes;
survival risks retain their forecast horizons and the 0.7 default alarm
threshold. DSM rows below exclude its known original training shots. The
CNN training list is unavailable, so its legacy and published-interval
rows are descriptive, with unknown overlap; they cannot be called held out.

| Model | Setting | Scored shots | AUROC [95% CI] | AUPRC [95% CI] | F1 [95% CI] | Segmental F1 [95% CI] |
|---|---|---:|---|---|---|---|
| tm-onsetcnn | Legacy, overlap unknown | 79 | .897 [.832,.952] | .601 [.385,.799] | .611 [.437,.750] | — |
| tm-dsm (250 ms) | Legacy | 219 | .781 [.713,.847] | .084 [.050,.154] | .000 [.000,.000] | — |
| tm-dsm (500 ms) | Legacy | 219 | .756 [.685,.825] | .133 [.081,.229] | .000 [.000,.000] | — |
| tm-dsm (1 s) | Legacy | 219 | .749 [.677,.823] | .189 [.121,.307] | .000 [.000,.000] | — |
| tm-onsetcnn | Interval, published; overlap unknown | 286 | .930 [.905,.952] | .778 [.683,.855] | .628 [.537,.714] | .468 [.343,.588] |
| tm-dsm (250 ms) | Interval, published; outside training | 252 | .698 [.624,.765] | .353 [.274,.434] | .000 [.000,.000] | .000 [.000,.000] |
| tm-dsm (500 ms) | Interval, published; outside training | 252 | .697 [.624,.765] | .353 [.274,.433] | .000 [.000,.000] | .000 [.000,.000] |
| tm-dsm (1 s) | Interval, published; outside training | 252 | .696 [.621,.763] | .353 [.275,.434] | .000 [.000,.000] | .000 [.000,.000] |
| tm-onsetcnn | Tokamak-SI, retrained | 286 | .926 [.898,.950] | .778 [.694,.852] | .725 [.656,.786] | .564 [.471,.644] |
| tm-dsm | Tokamak-SI, retrained | 341 | .890 [.860,.913] | .681 [.581,.766] | .647 [.579,.703] | .409 [.326,.495] |
| tm-ours | Tokamak-SI, Mirnov only | 450 | .972 [.964,.978] | .852 [.799,.894] | .754 [.707,.797] | .475 [.414,.535] |
| n1 RMS, 12 G | Tokamak-SI, fixed rule | 450 | .941 [.926,.955] | .750 [.682,.803] | .373 [.309,.441] | .282 [.213,.350] |
| tm-ours + RMS | Tokamak-SI, circular ablation | 450 | .978 [.972,.984] | .877 [.835,.913] | .789 [.745,.831] | .593 [.534,.653] |

Each row and exact thresholds/bins/shot IDs are in benchmark `rows` with an
individual JSON source link. Saved out-of-fold probabilities were rescored
against the final label table; point F1 and segmental F1 are unchanged. The
repeated scoring differences are recorded against the immediately prior result;
the final verification reproduces all reported point estimates exactly.
Input coverage differs, so this table alone does not establish a paired
ranking of detectors. The published DSM does not reach its default 0.7
threshold here; a zero F1 does not mean its ranking has no information.

`--final` was run by the previous implementer and wrote external `*_test.json`
records. The takeover did not score those shots again or use their scores
to choose a rule, threshold or model. Those records are historical exposure,
excluded from the paper export; the blind split cannot be described as
previously unexamined. The brief is interpreted conservatively as authorizing
development benchmark scoring. Legacy scoring now explicitly drops the
cohort blind test shots too.

## Paper outputs and reproduction

[`table_tm_benchmark.tex`](table_tm_benchmark.tex) uses human-readable model
names, model × setting rows and a caption shorter than 80 words.
[`figure2_tm.json`](figure2_tm.json) supplies legacy-versus-Tokamak-SI F1
with confidence bounds and coverage. The same-cohort observable coverage is
500 shots / 2,747.69 s for ours, 92 / 169.725 s of usable matched Seo samples,
and 200 / 1,200 s of survival-label samples. Quiet labelled time is included;
these are not positive-event durations or the size of the full legacy archive.
Population coverage, separately identified, is 4,848 labelled shots /
26,621.63 observable seconds. Sources and definitions are in `coverage`.
The comparison changes both target and input availability and is not a
paired performance improvement.

Run Python through the owner's prescribed environment, with `TMPDIR` in the
stream's temporary directory, `LABELER_NO_FETCH=1` and CPU thread limits ≤ 8:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/tm
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python scripts/labeler/tm_benchmark.py --rescore --gallery-reviewed
```

`--gallery-reviewed` records a review already performed; inspect the PNGs
before supplying it. Training entry points are `tm_ours.py` and
`tm_prior_retrain.py`, using the CUDA venv described in the stream rules
and GPU 1 when available. Omit `--final` to preserve the blind split.
Agreement runs use the main labeler root for survival and the private
`round4/tm/lroot` for Seo, each bounded by `timeout 1800` with a log. Only
canonical `_dev` and `_all` output names remain.
