# All-shot 1 ms labels: where they are made, published and read

The all-shot runs (`scripts/labeler/ae_fdp_label.py`, `elm_fdp_label.py`,
`conf_fdp_label.py`) use three places, and data moves one way through them:

| Place | Holds | Written by | Read by |
|---|---|---|---|
| `$LABELER_ROOT/<run>/` (GPFS scratch, `.cache/labeler`) | the run root: candidates, staged inputs, work files, result CSVs, logs | the SLURM jobs | `publish`, the run's own checks |
| `/projects/EKOLEMEN/tokamaksi/diiid/dev/<event>/<run>/` | development labels: samples, pilots, checks, relabel trials | `publish` of any other root (login node) | development only |
| `/projects/EKOLEMEN/tokamaksi/diiid/release/<event>/<name>/` | the release (the scripts call it the deployment): one all-shot database per event | `publish` of the all-shot root (login node) | everything outside the run |

`/projects/EKOLEMEN/tokamaksi/<machine>/` holds one machine's labels: `diiid` today, MAST
later. Inside it, `release/`, `dev/` and `events/` sit side by side. Until 2026-11-07 the same
folders were in `/projects/EKOLEMEN/fusionaihub_labels`.

The database format is the same in `dev/` and `release/`: `spans.parquet`, `shots.parquet`,
`valid.csv`, `meta.json`, `README.md` and `grid_1ms/`.

## Rules

- Jobs write only to scratch. Compute nodes do not mount `/projects`, which is mounted on the
  login node only, so `publish` runs on the login node.
- `publish` picks the folder (`ae_fdp_label.publish_out`):
  - `--out` when it is given;
  - `release/<event>/<name>` for the all-shot root (`ae_fdp_1ms`, `elm_fdp_1ms` or
    `conf_fdp_1ms` under `$LABELER_ROOT`);
  - `dev/<event>/<root name>` for any other root.

  So a pilot or check run never overwrites the release.
- Readers outside the run read `release/` only. They never read a run root or `dev/`:
  - tokamaksi, with `TOKAMAKSI_FDP_LABELS=/projects/EKOLEMEN/tokamaksi/diiid/release`. Its
    database rule (`<event>/` or `<event>/<name>/` holding `meta.json`) stays inside it.
  - the public coverage listing, `ae_fdp_label.py coverage`. It reads the databases'
    `shots.parquet` and `valid.csv` (`--deployment`, default `release/`), and the run roots
    no longer.
  - the Hugging Face export, `hf_publish.py build --fdp-1ms <folder>`. The folder is
    `release/` or a verbatim copy of it. A build that runs as a SLURM job needs the copy on
    GPFS.
- A job that needs the labels reads a verbatim copy on GPFS. Use `rsync -a` of the database
  folder, and leave out `grid_1ms/` when only the tables are needed.
- To read one development run, use
  `tokamaksi.fdp.FdpLabels("<event>", "/projects/EKOLEMEN/tokamaksi/diiid/dev/<event>/<run>")`.
- The curated review tables are a different system, kept apart from all of this. They live at
  `$LABELER_LABEL_TABLES` (`/scratch/gpfs/EKOLEMEN/D3D/events`).
  `/projects/EKOLEMEN/tokamaksi/diiid/events` is a copy of them from 2026-10-17 that nothing
  reads. [Shared label folder](shared_labels.md) has the plan to move them.

## Status (2026-11-07)

`release/` holds all three all-shot databases:
- `alfven_eigenmode/ae_seldnet_fdp_1ms`
- `edge_localized_mode/elm_fdp_1ms`
- `confinement/confine_ours_fdp_1ms`

`dev/` was created empty on 2026-11-07. The earlier development runs were never published, and
their results are still in their run roots. Those runs are `ae_fdp_pilot`, `ae_fdp_preview`,
`ae_fdp_1ms_check`, `ae_notch`, `ae_fdp_notch`, `ae_tiers`, `elm_tiers` and `round5`. The notch
relabel (`ae_fdp_notch`) is merged into the AE release by `publish --notch-root`.
