# Rerun the all-shot ELM labels with elm-inst2 (StellarAI)

Rerun the ELM labels for every shot with a new model, here elm-inst2 (`elm_inst2.md`) in place of
the label-tiers student. Keep the inputs (`inputs/<shot>.npz`), so no shot is fetched again.
The result has two layers, as the labeler does:

- **span**: the ELMy period on a 1 ms grid, from the occupancy head (`spans.parquet`, `grid_1ms/`);
- **individual**: one row per ELM, from the onset head (`individual.parquet`: `shot, t_ms, in_span`).

No large/small typing. Run every step on a StellarAI login node, in the checkout the infer job
uses (`/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub`, the `REPO` default of
`elm_fdp_infer.sbatch`).

```bash
cd /scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub && git checkout nathan_dev && git pull
printenv LABELER_ROOT          # the job's default is $REPO/.cache/labeler; use the same
R=$LABELER_ROOT/elm_fdp_1ms    # the all-shot run root: inputs/, logs/, batches/
M=$LABELER_ROOT/models/elm_inst2
P=/projects/EKOLEMEN/tokamaksi/diiid
```

## 1. Copy the model to scratch

Compute nodes do not mount `/projects`, so the job reads a copy on scratch. The staged folder
holds `fold0-4/{model.pt,fold.json}`, `decoding.json` (the onset decoding), `evaluation.json`,
`run.json`, `SOURCE.txt` and `SHA256SUMS`.

```bash
mkdir -p "$M" && rsync -a "$P/models/elm_inst2/" "$M/" && (cd "$M" && sha256sum -c SHA256SUMS)
```

## 2. Move the old model's work aside

Infer skips a shot that has a work file, so a new model would label only the shots the old
one missed. Infer refuses to start when `work/` holds files from another model. First check
that no infer job is running (`squeue -u $USER -n elm-fdp-infer`), then:

```bash
mv "$R/work" "$R/work_tiers_student_2026-11-04" && mv "$R/results" "$R/results_tiers_student_2026-11-04"
```

Leave `inputs/`, `logs/`, `batches/` and `stage/` in place.

## 3. Pilot one batch

```bash
B=$(ls "$R"/batches/*.csv | head -1)
MODELS=$M LIST=$B sbatch scripts/labeler/elm_fdp_infer.sbatch
```

The log is `$LABELER_ROOT/runs/slurm/elm-fdp-infer_<job>.out`. Check four lines:

- `instance run: presence from the occupancy head; onsets by elm-inst2-sized (key2_record.json):
  heatmap >= 0.5, 3.0 ms apart`
- `<n> shots in <s> s`: decides how many shots one 45-minute job can take;
- `GPU utilisation over the shot loop: mean ... %`: at least 50 % (the project's GPU rule).
  Under 50 %: stop and report, do not run the rest;
- `results/<job>p0.onsets.csv` and `p1` have rows.

## 4. The rest

Submit one job for each remaining batch. Jobs on different lists never share a shot. A job
that times out can be resubmitted with the same `LIST`, because the shots already done are
skipped. If the QOS refuses more submissions, chain the jobs with
`--dependency=afterany:<previous job>`.

```bash
for B in $(ls "$R"/batches/*.csv | tail -n +2); do
  MODELS=$M LIST=$B sbatch --parsable scripts/labeler/elm_fdp_infer.sbatch
done
```

When the batch jobs are done, submit one job with no `LIST`. It picks up any input that is on
no batch list. Run only one no-`LIST` job at a time.

```bash
MODELS=$M sbatch scripts/labeler/elm_fdp_infer.sbatch
```

The old release has 42,406 labeled shots, so expect about as many. Rows with `infer_failed`
in `results/*.shots.csv` are retried by resubmitting.

## 5. Publish (login node)

A label folder holds one model run, and `publish` refuses a folder that holds another run.
Move the old release to `dev/` first:

```bash
mkdir -p "$P/dev/edge_localized_mode"
mv "$P/release/edge_localized_mode/elm_fdp_1ms" "$P/dev/edge_localized_mode/elm_fdp_1ms_tiers_student_2026-11-04"
pixi run --frozen -e labeler python scripts/labeler/elm_fdp_label.py publish --models "$M"
```

The output defaults to `release/edge_localized_mode/elm_fdp_1ms`. It prints the status counts
and the number of spans and ELMs, with how many ELMs fall inside a span. Report those numbers,
and the `GPU utilisation` lines of the jobs, to the owner.

Then refresh what reads the release:

- the coverage listing: `ae_fdp_label.py coverage`;
- a Hugging Face build: `hf_publish.py build --fdp-1ms`. Only when the owner asks; it copies
  `individual.parquet` as well.

## Caveats to pass on

- No one has reviewed the span layer on the same shots as the student's gate. Its out-of-fold
  score is AUROC 0.938 and F1 0.838 on the reviewed bins (`evaluation.json`).
- The onset head fires about 7 times a second in reviewed ELM-free time. An ELM with
  `in_span` false is suspect.
- The confinement ELM gate (`conf_fdp_label.ELM_ROOT`) still reads the `elm_inst2_fdp_1ms`
  root, not this one.
