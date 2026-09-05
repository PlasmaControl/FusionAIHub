# labelmaker

Runs the group's trained models over the FAITH shot corpus and writes their
predictions as per-shot label files in the corpus HDF5 layout, with a measured
statement of how much each model's labels can be trusted.

Design: `docs/superpowers/specs/2026-09-03-labelmaker-design.md`.
Phase 1 build log: `docs/superpowers/plans/2026-09-03-labelmaker-phase1.md`.
Code: `src/labelmaker/`. Tests: `tests/labelmaker/`.

## Running it

```bash
pixi install -e labelmaker              # once
pixi run -e labelmaker fdp login        # once per token; only the fdp resolver needs it

pixi run -e labelmaker fdp run python -m labelmaker.run all \
    --models d3d_tearing_onset_cnn1d \
    --overlap --sample 100 --seed 20260903 --workers 8 --timeout 300
```

The `fdp run` wrapper is required whenever a feature may come through fdp, which
for the tearing model is every shot whose archive group lacks `ip`/`bt` (2,192 of
the archive's 5,000) and every shot outside the archive. Without it PTDATA fails
with `getservbyname failed for task 'PTSERVER'` and MDSplus with `TREE-E-FOPENR`,
and those misses are recorded as transient, so the next correctly wrapped run
retries exactly them.

Shot selection is one of `--shots N N`, `--shot-file PATH`, `--corpus`, or
`--overlap` (shots present in both the corpus and the tearing model's training
archive), with `--sample N --seed S` to take a reproducible subset. Stages are
`features`, `infer`, `validate`, `all` and `analyze`; each is independently
rerunnable and skips work that is already complete unless given `--force`. The
100-shot proof-of-concept pool is `$LABELMAKER_ROOT/poc_shots.txt`; the
500-shot pool the reliability numbers below come from is `shots_500.txt` (the
100, 389 more validatable shots, and the tearing-mode shots 199597-199607).

Exit codes: 0 ok; 1 no shots selected; 2 usage; 3 the weights on disk do not
match the card's sha256 (nothing ran); 4 a requested model cannot be loaded; 5
adapter fidelity failed - the evaluator disagrees with the framework it is
supposed to reproduce, and no label from that model should be trusted; 6 a
validation report raised.

## One shot, all the labels you asked for

```bash
pixi run -e labelmaker fdp run python -m labelmaker.run analyze \
    --shots 199597 --config my_labels.yaml --out /some/dir
```

`analyze` takes its models from the config rather than `--models`. The config
is YAML (the default is `src/labelmaker/analyze_default.yaml`):

```yaml
labels:                                  # plotted top to bottom, in this order
  - d3d_tearing_onset_cnn1d/tm_prob
  - d3d_tearing_onset_cnn1d/betan
context:                                 # canonical features drawn above them
  - ip
  - pinj_total
threshold: 0.5                           # drawn on binary labels
thresholds:                              # per-label overrides of that value
  d3d_tearing_time_to_event_dsm/tm_risk_1s: 0.2
```

It runs `features` and `infer` for the models those labels need, then writes
`<out>/<shot>/<shot>_analysis.json` - per label: card id, weights digest, rows,
valid fraction, *why* rows are invalid (per rule), the peak and when the label
first crosses the threshold - and `<shot>_labels.png`, one panel per context
feature and per label, invalid rows washed grey, the ensemble spread as a
band.

When the shot is in the tearing model's training archive, the archived truth is
overlaid and scored: the rows the archive calls a tearing mode are shaded, the
first of them marked as the onset, `betan`'s archived column drawn as a dashed
line, and each label's `truth` block in the JSON carries AUROC, precision,
recall and F1 at the label's threshold, plus `lead_time_s` - how far before the
archived onset the label first crossed. A survival label is scored only on rows
before the onset, against "an onset occurs within the horizon". A shot with no
archived truth says so instead, per label. `--out` defaults to `$LABELMAKER_ROOT/analysis/`. The label file under
`labels/` stays the one canonical output; the analysis directory is a view of
it.

## What it writes

Everything under `$LABELMAKER_ROOT` (default
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`; `$LABELMAKER_CORPUS` overrides the
corpus location):

| path | contents |
|---|---|
| `features/<shot>_features.h5` | canonical features, one group each, `xdata`/`ydata`, with the `resolver` that served it |
| `labels/<shot>_labels.h5` | `<slug>/<label>` plus `_spread` (ensemble min/max) and `_valid` companions |
| `labels_index.parquet` | one row per shot, model and label - the "which shots have labels" query |
| `models/<slug>/` | the weights, copied once; verified against the card's sha256 before every load |
| `runs/<run_id>/` | `manifest.json` (config, git sha, shot list), `log.txt` (one JSON row per shot), `summary.json` |
| `validation/<slug>/` | the three reliability reports (below) |
| `analysis/<shot>/` | `analyze`'s JSON summary and figure for that shot |

Labels are probabilities, never thresholded, on the model's own time step.
`<label>_valid` is 1 where every input was present and inside the model's
training domain; a probability where it is 0 is an extrapolation. A label
stamped at `t` is computed from inputs averaged over `[t - 50 ms, t]`, the
convention the archive the model trained on was built with, uniformly across
sources; for the tearing model the 0-D inputs are read one step ahead, so the
label answers "is a mode present at t + 25 ms".

## Reading a label

```python
import h5py

with h5py.File(".../labels/190000_labels.h5") as f:
    g = f["d3d_tearing_onset_cnn1d/tm_prob"]
    t, p = g["xdata"][:], g["ydata"][0]          # seconds, probability
    valid = f["d3d_tearing_onset_cnn1d/tm_prob_valid/ydata"][0].astype(bool)
    print(g.attrs["card_id"], g.attrs["task"], g.attrs["artifact_sha256"][:12])
```

## How reliable are the labels

`python -m labelmaker.run validate` writes three reports and folds the headline
numbers into the model card's `model-index`, so the card is the one place to
read how a model performed. For `d3d_tearing_onset_cnn1d`, measured on the
500-shot pool (486 aligned to their archived training rows - the 11 tearing-mode
shots of 2024 have no archived truth and 3 shots have archived rows our
reconstruction does not cover; 31,257 published rows of 35,776 matched):

1. **Adapter fidelity** - our torch evaluation of the Keras-2 graphs against
   real-Keras outputs frozen for the same weight files. Passes all four gates;
   the published `tm_prob` differs from TensorFlow's by at most 1.2e-6.
2. **Reconstruction fidelity** - our features against the model's own training
   inputs, feature by feature. Archive-served columns are exact; `bt`/`ip`
   through fdp agree to 1e-5; offline EFIT01 standing in for real-time EFIT
   costs 2.1e-3 (R0), 2.8e-3 (kappa), 3.0e-2 (1/qpsi); ZIPFIT standing in for
   the pipeline's own profile fits costs 6.0e-2 (ne), 1.24e-1 (Te), 1.24e-1
   (rotation) median relative difference.
3. **Label quality** - against the archived labels, scored twice over the same
   rows: with the training inputs (the model's ceiling) and with our
   reconstruction (what labelmaker publishes).

| `tm_prob` | AUROC | F1 at 0.5 | precision | recall | best F1 (at) |
|---|---|---|---|---|---|
| archived inputs | 0.932 | 0.521 | 0.373 | 0.862 | 0.576 (0.76) |
| reconstructed inputs | 0.897 | 0.486 | 0.403 | 0.613 | 0.490 (0.60) |

The gap - AUROC -0.034, best F1 -0.086, `betan` RMSE 0.120 -> 0.157 - is the
reconstruction penalty, and the number that answers "is this reliable". The
ceiling is scored on rows the model trained on, so read it as optimistic; the
honest reading is the gap. Best F1 is reported because the AUROC gap
understates the cost: the reconstruction loses positives the model had placed
confidently, which a rank statistic barely registers.

F1 at 0.5 is low for a reason that is not the reconstruction: the model was
trained oversampled and class-weighted (`mse_bin_os_w`), so at 0.5 it flags
about twice the base rate of rows (reliability bin 0.5-0.6 observes 8%). Its
ranking is fine; the threshold is the operator's choice, and `analyze` takes
one in its config.

## Adding a model

1. `src/labelmaker/models/<device>_<phenomenon>_<predicted>_<arch>/`, with
   `README.md` (HuggingFace card plus the `labelmaker:` block), `spec.py`
   (`ADAPTER`), and an empty `__init__.py`. Copy the closest existing folder.
2. Map each trained-on input name to a canonical feature in
   `features/namespace.py`. Add a `FeatureSpec` only if the quantity is
   genuinely new, and give it every source that can serve it, cheapest first.
3. Put the training-time filter in as `DomainRule`s. Labelmaker flags rows
   outside them instead of dropping them.
4. Load the weights from their upstream location, copy them into
   `<root>/models/<slug>/`, and record the sha256 in the card - inference
   refuses to run against bytes the card does not know.
5. `pixi run -e labelmaker python -m pytest tests/labelmaker -q -W error` - the
   registry tests check that the card and `spec.py` agree - and
   `pixi run -e labelmaker ruff check src/labelmaker tests/labelmaker`.

## Known limits

- The archive resolver covers 3,246 of 16,909 corpus shots (19%). Everything
  else goes through fdp, which is built, measured against the archive on
  random overlap shots, and exercised on 69 of the 100 proof-of-concept shots -
  but has not been run at corpus scale (measured 3.5 s per shot median at 8
  workers, so the corpus is hours, not days).
- The three kinetic profiles are ZIPFIT fits standing in for the tearing model's
  own mtanh/csaps fits - the dominant reconstruction error, priced above. The
  rotation profile is absent on ~22% of shots, and a shot missing any input for
  its whole record has no valid rows: 5 of the 100 proof-of-concept shots.
  Fitting the profiles from raw Thomson and CER needs channel geometry the
  corpus does not carry, and is Phase 2.
- ECH is on hold. The corpus has no deposition-location group, its
  `ech_power` sum under-reports the archive's `EC.PECH` by ~22% at the median,
  and whether fdp can serve a deposition location has not been investigated;
  outside the archive, any row with ECH power flowing is invalid. This is what
  decides the 2024 tearing-mode shots 199597-199607: ECH is on for ~90% of
  their plasma rows, so they come out 4-15% valid, even though `tm_prob` peaks
  above 0.9 on 9 of the 11 (`analyze` shows the members disagree widely on
  them too). Until a deposition-location source exists, their labels are
  extrapolations by the card's own definition.
- Three of the 500 validated shots (190859, 190861, 190863) are rejected by the
  row matcher: 4-12 of their archived rows have EFIT geometry that appears
  nowhere in our EFIT01 series, so the archive covers times our reconstruction
  does not. Refusing the whole shot is the conservative choice; a per-row
  refusal is the planned fix.
- ZIPFIT rotation on some 2024 shots reaches 1,200-2,000 in units the model
  reads as km/s; the `|rotation| < 150` domain rule catches those rows, but
  whether that is a units change or a fit blow-up has not been checked.
- The label-quality numbers come from shots the archive holds, because only
  there is there an archived truth to score against; they are a measurement of
  the reconstruction, not of the model on unseen shots.
- Two models are implemented: the tearing-onset CNN and the tearing
  time-to-event survival model (`d3d_tearing_time_to_event_dsm`, onset risk at
  250 ms, 500 ms and 1 s). The survival model's labels are produced but not yet
  scored by `validate`, whose reports are specific to the CNN's training
  archive; its adapter fidelity is a test against the upstream fork's own
  outputs. The other five roster folders are scaffolds whose cards say what
  blocks each of them; `docs/superpowers/specs/2026-09-05-labelmaker-phase2-design.md`
  records what the upstream archaeology found for each.
- The survival model's calibration depends on which rows a report covers, and
  that is a property of its training population, not a defect: over all aligned
  shots' pre-onset rows (5.5% positive at 1 s) it is calibrated, ECE 0.022;
  restricted to shots that do get an onset (54% positive) it is under-confident,
  ECE 0.448. It was fit on rows that are 85% censored with a median 1.92 s to
  event. Every figure and every report states its row set for this reason.
- `docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md` carries the
  next round: what the tearing-survival, ELM and Alfven-eigenmode training loops
  upstream actually do (measured, with the shipped survival model's
  hyperparameters decoded from its own pickle), the three agreed reliability
  fixes, the uncertainty series to publish from the survival mixture, and the
  requirements for `d3d_ae_activity_seldnet` - the one model labelmaker will
  train itself.
