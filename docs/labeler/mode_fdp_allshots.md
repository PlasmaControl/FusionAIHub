# Label every shot's MHD modes with omnimode (StellarAI)

Label the MHD modes (toroidal n, with m/n) of every DIII-D plasma shot from 185000 to 204999 with
omnimode's redesigned labeller, then publish the 1 ms databases of `fdp_labels.md` to
`/projects/EKOLEMEN/tokamaksi/diiid/dev/`. Trigger: "label all the modes on StellarAI" or "run
the all-shot mode labels".

- **Labeller**: omnimode (`PlasmaControl/integratedmode`, branch `labeller-redesign`, 7205c13 or
  later), variant `ermap-magrot`, TokEye 1.0 front end. The label store is
  `labels/B0.5-v6-a0.01-fzv6-ermap-magrot/`.
- **Inputs**: 14 Mirnov probes (the wall pair, the M3, M5 and M89 poloidal pairs, six toroidal
  probes), Ip, the EFIT02ER, EFIT02 and EFITRT2 trees, and CER rotation (ZIPFIT01 and ZIPFIT02
  TROTFIT). No ECE, no CO2, no CAKE.
- **Route**: FDP's per-shot views. The login node downloads each shot's native files (about
  1 min a shot). A CPU job turns them into omnimode's formats (`labeler.modes.fetch`). A GPU job
  makes the TokEye masks. A CPU array runs `omnimode batch`. Nothing in the labeller is
  reimplemented here.
- **Driver**: `scripts/labeler/mode_all_shots.py`. The jobs are `scripts/labeler/mode_fdp_*.sbatch`,
  and all of them source `mode_fdp_env.sh`.

## Rules

- **Sealed shots.** omnimode's held-out and quarantined lists (`protocol/heldout_v1.txt`,
  `heldout_v2.txt`, `quarantine_v2.txt`) are sealed. `shots` drops them inside the omnimode venv,
  and every command prints only counts. Never print, list or quote a shot from those files.
  Never open a seal access (`omnimode seal-access`); that is the owner's call. `omnimode batch`
  and `tokeye_masks.py` refuse a list that holds a sealed shot. `mode_fdp_env.sh` unsets
  `OMNIMODE_ACCESS`, `OMNIMODE_UNSEALED` and `OMNIMODE_PROTOCOL` in every job.
- **Never push omnimode**, and never edit its checkout. It is a clone used read-only.
- **Never resubmit part of a labels array.** The slices change with the task count. Resubmit a
  whole list (`submit --list ... --start masks`); finished shots are cache hits.
- **Login node** only for `stage` (fdp downloads), `shots`, `pilot`, `status`, `clean`, `redo`,
  `submit`, `compare` (a light read) and the final copy. Everything else runs as a SLURM job.
- **FDP auth.** `stage` stops at the first refused token (exit 3). Run
  `pixi run --frozen -e fdp fdp login`, then rerun `stage`; it resumes.
- **GPU rule.** The masks job's log ends with `GPU utilisation: mean N %`. If the pilot shows
  less than 50 %, stop and report. Do not start the full run.

## 0. Setup (once)

```bash
cd /scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub && git checkout nathan_dev && git pull
printenv LABELER_ROOT       # the jobs default to $REPO/.cache/labeler; use the same
export MODE_FDP_ROOT=$LABELER_ROOT/mode_fdp     # the run root (the default)
export OMNIMODE_REPO=/scratch/gpfs/EKOLEMEN/nc1514/integratedmode   # the default
```

The omnimode clone and its venv. If `uv` is missing, install it first with
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
git clone git@github.com:PlasmaControl/integratedmode.git "$OMNIMODE_REPO"
cd "$OMNIMODE_REPO" && git checkout labeller-redesign && git log --oneline -1   # 7205c13 or later
uv sync --frozen
# warm the TokEye env: the GPU job runs it offline
uv run --no-project --python 3.13 --with tokeye==1.0.0 --with torch==2.9.1 --with-editable . \
    python -I -c "import tokeye, torch; print(torch.__version__)"
cd /scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
```

Copy the TokEye weights to scratch. Compute nodes do not mount `/projects`. omnimode checks the
same sha256 when it loads them.

```bash
mkdir -p "$MODE_FDP_ROOT/tokeye"
rsync -a /projects/EKOLEMEN/tokamaksi/diiid/models/tokeye/ "$MODE_FDP_ROOT/tokeye/"
(cd "$MODE_FDP_ROOT/tokeye" && sha256sum -c SHA256SUMS)
```

## 1. The shot list

```bash
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py shots
```

The source is the AE/ELM candidates (`$LABELER_ROOT/ae_fdp_1ms/candidates.csv`: every shot
with a pulse longer than 0.1 s and |Ip| above 100 kA) from 185000 to 204999, minus the sealed
lists. The command prints two counts and writes `$MODE_FDP_ROOT/shots.txt`.

## 2. Pilot

The pilot takes four listed shots that StellarAI's full corpus (`/scratch/gpfs/EKOLEMEN/D3D/signals`)
also holds, so the fetched Mirnov data can be compared with it.

```bash
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py pilot --n 4
pixi run --frozen -e fdp fdp -D d3d run python scripts/labeler/mode_all_shots.py \
    stage --list "$MODE_FDP_ROOT/batches/pilot.txt" --batch 4 --submit
```

`stage` downloads the four shots and writes `batches/b0000.txt`. It then submits extract (CPU),
then masks (GPU), then labels (CPU array), each step `afterany` the one before. Logs are in
`$LABELER_ROOT/runs/slurm/mode-fdp-*`. When the jobs are done:

```bash
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py status
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py compare --list "$MODE_FDP_ROOT/batches/pilot.txt"
```

Check these before the full run. If any check fails, stop and report:

- **`status`, stage line.** Every shot is `staged`, and every tree (`efit02er`, `efit02`,
  `efitrt2`, `zipfit01`, `zipfit02`) shows up in most shots. A tree that is never staged means
  the views keep it somewhere `wanted()` does not look. Without the EFIT trees no mode gets an m.
- **`status`, extract line.** It reads `ok`, with equilibria read for most shots. A
  `mirnov_rate` status means a probe is not sampled at 500 kHz. Report it; do not resample.
- **`compare`.** The worst correlation is at least 0.999 and the offsets are `[0]`. This is the
  fetched Mirnov data against the corpus on the same shots.
- **masks log.** `GPU utilisation: mean N %` is at least 50, and every task reports
  `0 errors`.
- **`status`, labels line.** Statuses are mostly `ok`. `no TokEye mask` or `read error` can be
  retried with `redo`. If most shots read `no window: ...`, the Ip did not give a window.

Also note the minutes per shot: staging is in the `stage` progress lines, extract in its log, and
the labels in each array task's log (about 25 s a shot on Stellar).

## 3. The full run

```bash
pixi run --frozen -e fdp fdp -D d3d run python scripts/labeler/mode_all_shots.py \
    stage --submit --batch 200 --parallel 6 --budget-gb 4000
```

`stage` works through the list in order. Every 200 staged shots it writes the next
`batches/bNNNN.txt` and submits that batch's three jobs. At each batch it also runs `clean`,
which frees the Mirnov copy (about 0.5 GB a shot) of every shot whose label is final. It pauses
while `stage/` and `corpus/` hold more than `--budget-gb`.

It can be stopped and rerun at any time. It skips a shot that is extracted or labelled, a shot
with no view, no Mirnov or no Ip, and a shot already in a batch whose jobs have not yet
extracted it. A batch whose jobs were lost (cancelled, or failed to submit) is resubmitted whole:

```bash
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py submit --list "$MODE_FDP_ROOT/batches/b0007.txt"
```

Run it inside `tmux` or `screen`: it is a long login-node loop of downloads, with no compute.

## 4. Retries and status

```bash
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py status
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py redo --submit   # no TokEye mask, read errors, model errors
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py clean           # after the last labels job
```

`redo` writes `batches/redo_<time>.txt` and submits masks, then labels with `--force`. A shot
with a transient extract error is staged again by the next `stage` run.

## 5. Publish to dev

```bash
pixi run --frozen -e fdp python scripts/labeler/mode_all_shots.py publish
# when the job is done (log: mode-fdp-publish_<job>.out):
rsync -a "$MODE_FDP_ROOT/publish/" /projects/EKOLEMEN/tokamaksi/diiid/dev/
```

The publish job writes `mhd_n1`, `mhd_n2` and `mhd_n3` under
`publish/<event>/omnimode_ermap_magrot_fdp_1ms/`, with `spans`, `shots`, `valid`, `grid_1ms` and
`meta.json` (`mode_fdp_label.py publish --labeller redesign`). Its log prints each event's
labelled-shot and span counts.

Report to the owner:

- those counts;
- the status counts;
- the pilot's `compare` and GPU lines;
- how many shots had no view, no Mirnov or no Ip.

It goes to `dev/`, never `release/`. Promotion is the owner's call.

## Caveats to pass on

- **Unchecked.** The labels are proposed, not checked. No expert reference set scores this run.
  The `meta.json` source says so.
- **No CAKE.** The er ladder's CAKE rungs are absent. Each track reads EFIT02ER, then EFIT02,
  then EFITRT2. The last two count at half p.
- **Ip windows.** Windows come from the fetched Ip (`omnimode.modes.window`'s corpus fallback),
  not omnimode's catalog, so they can differ from omnimode's own runs on the same shots.
