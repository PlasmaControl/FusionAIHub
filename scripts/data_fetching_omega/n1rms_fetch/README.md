# n1rms / n2rms fetch for the stage-1 eval campaign

Fetches the DIII-D MHD-tree RMS mode amplitudes (`\MHD::N1RMS`, `\MHD::N2RMS`)
for shots used by the evaluation campaign. These signals are **not** in the
processed foundation-model HDF5 files; Study C currently uses Mirnov/MHR band
power as an n=1 proxy, and this fetch replaces the proxy with real labels.

Runs on **omega** (GA network) with the parent directory's batch framework —
see `../README.md` for environment setup (`pip install mdsh5`).

## Files

- `config_n1rms.yaml` — signal list (MHD tree only: `\MHD::N1RMS` +
  `\MHD::N2RMS`, 2 signals/shot; a fetch job is seconds per shot)
- `shots_n1rms_finetune.txt` — **use this one**: 165 curated shots
  (110 train / 55 val; 135 MHD-active ranked by mirnov 1–10 kHz band power
  + 30 quiet controls). Selection detail per shot in
  `shots_n1rms_finetune_meta.csv`.
- `shots_n1rms.txt` — optional full list (all 875 val + 875 matched-train
  campaign shots) if a complete label set is ever wanted

## Run (on omega)

From `scripts/data_fetching_omega/`, edit `submit_read_mds_batches.sh`:

```bash
MODE="list"
SHOT_LIST_FILE="n1rms_fetch/shots_n1rms_finetune.txt"
CONFIG_FILE="n1rms_fetch/config_n1rms.yaml"
OUTPUT_DIR=<somewhere with ~a few GB free>   # e.g. /cscratch/$USER/n1rms
```

Set `ENABLE_GLOBUS=false` in `read_mds.sh` (the output is small enough to
copy by hand), then:

```bash
nohup ./submit_read_mds_batches.sh > submission_n1rms.log 2>&1 &
```

State tracking lives in `.completed_shots` / `.failed_shots`; rerunning the
submit script resumes. Expect some shots to fail (MHD tree not built for
every shot) — that is fine, the ingest step marks them missing.

## Bring back to Frontier

Copy the fetched HDF5 file(s) from `OUTPUT_DIR` to Frontier, e.g.:

```
/lustre/orion/fus187/proj-shared/foundation_model_meta/n1rms_raw/
```

Layout per the framework: `<shot>/MHD/<signal path>/{data, dim0}` with
`dim0` the time axis (DIII-D MDSplus convention: milliseconds).

## Ingest (on Frontier)

```bash
python scripts/evaluation/ingest_n1rms.py \
    /path/to/n1rms_raw_dir_or_file.h5 \
    --labels-dir data/outputs/eval_suite/e2e_stage1_best/labels
```

Writes `labels/n1rms_windows_{val,train}.csv` on the same fixed window grid
as the other labels (t_start = 1.0 + 0.25·k, k = 0..43; join key
`(shot, t_start_s)`), with per-window mean n1rms/n2rms in raw and log10
form for the current and next 50 ms windows. `study_c_probes.py
--n1rms-csv` then swaps the proxy labels for the real ones.
