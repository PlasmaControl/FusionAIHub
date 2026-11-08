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
The ELM database of an instance run (elm-inst2) adds `individual.parquet`, one row per ELM
beside the ELMy-period spans. To rerun all shots with a new ELM model, see
[elm_fdp_rerun.md](elm_fdp_rerun.md).

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

## MHD mode labels (`mode_fdp_label.py`, proposed)

`scripts/labeler/mode_fdp_label.py` converts omnimode's port-v6 labels into the labeler's
formats. It computes nothing and fetches nothing: it reads one set of files a shot
(`<shot>.json`, `.tracks.parquet`, `.frames.parquet`) and a shot list, and reads no shot that is
not on the list. The code is `labeler.modes`; the tests are `tests/labeler/test_modes_*.py`.

Status: PROPOSED, NOT CHECKED. The rows are model output with the labeller's own confidence.
They are to be superseded by the labeller redesign, which also adds the "n>3" and "n=0" classes
that this version does not have.

- Source: omnimode port-v6. The equilibrium is the shot's own tree in omnimode's info record,
  EFIT02ER first, else EFIT02 or EFIT01.
- Rules (owner, 2026-11-07):
  1. Harmonics are not labelled. A track with `harm` true (a 4/2 locked at 2f of a 2/1) is
     dropped with its frames. An independent 4/2 stays.
  2. The scope is |n| = 1, 2, 3 with their m.
  3. Only labelled modes are written: resonance status "m/n", the class held at P >= 0.9.
  4. Amplitudes are uncalibrated and carry no value.
- Events: `mhd_n1`, `mhd_n2`, `mhd_n3`. Each has an `m<m>` subcategory a class, so a 2/1 mode
  is `mhd_n1` / `m2` and a 4/2 mode is `mhd_n2` / `m4`.
- Not applied: the crash table and the q = 1 gate of the v5 tables are FusionAIHub inputs, not
  omnimode's.

Commands (all take `--labels DIR --shots FILE` first):

| Command | Writes |
|---|---|
| `status` | nothing; counts the listed shots with and without port-v6 files |
| `suggest` | the `mhd_modes` suggestion tables, in v5's format (`shot, category, t_start, t_end, confidence`, 0 absent, 1 present, 2 uncertain, 3 not observable), into `<root>/suggestions/mhd_modes/<version>/`, default version `v6port`. A version is never written into twice. `--events DIR` also points each event's `review/source.json` at its table, with `labels.write_pointer`, and makes the `m<m>` folders |
| `roster` | `shots.csv` of `mhd_n1`, `mhd_n2` and `mhd_n3` replaced by the listed shots, each file backed up first; prints counts only |
| `publish` | the 1 ms databases (`spans.parquet`, `shots.parquet`, `valid.csv`, `meta.json`, `README.md`, `grid_1ms/`), one a parent event, at `<out-root>/<event>/<run>/`. `--out-root` has no default, and nothing here writes `release/` or the Hugging Face export |

The suggestion tables show the v5 semantics: `mhd_n<K>` is present over every non-line track of
that n, labelled or not, and a subcategory is present over its labelled tracks and uncertain over
the abstentions whose 90% credible set holds its class. The `publish` databases show only the
labelled modes: a parent span is some labelled n = K mode, a subcategory span is a labelled m/K
mode, and the grid byte has the parent in bit 0, the subcategories in bits 1 to 6 (`grid_bits` in
`meta.json`) and not observable in bit 7. A listed shot with no labels, no window or a failed run
stays listed in `shots.parquet` with its `status`.

The development run of `publish` has not been made. When it is, give `--out-root` as
`/projects/EKOLEMEN/tokamaksi/diiid/dev` on the login node.
