# One label folder for both clusters

The label tables live in one folder on `/projects`,
`/projects/EKOLEMEN/tokamaksi/diiid/events` (`env.SHARED`). Login and visualization
nodes of Stellar and StellarAI mount `/projects`; compute nodes do not. Batch label
jobs run on StellarAI, now and then, so StellarAI's own folder becomes a read-only
mirror for them, refreshed before each batch. How several accounts
save into one folder is [shared_labels.md](shared_labels.md); this page covers where
the folder is, the mirror, and switching it on.

## Where `LABELER_LABEL_TABLES` resolves

| Node | Before the switch | After the switch |
| --- | --- | --- |
| login or vis node, either cluster | the cluster's folder | `/projects/EKOLEMEN/tokamaksi/diiid/events` |
| compute node | the cluster's folder | the cluster's folder: a mirror on StellarAI, as left at the switch on Stellar |

The cluster's folder is its row in `env.CLUSTERS`: `/scratch/gpfs/EKOLEMEN/nc1514/events`
on Stellar and `/scratch/gpfs/EKOLEMEN/D3D/events` on StellarAI. "Switched on" means
the shared folder holds `.shared`, which `labeler-sync activate` writes.

- An export of `LABELER_LABEL_TABLES` still wins, with one exception. If the export
  names the shared folder and this node does not mount it (a job that inherited a
  login node's environment), the job reads the cluster's mirror and says so once on
  stderr.
- The check runs once per process, so a server running across the switch keeps
  its folder until it restarts.
- With `LABELER_CLUSTER=""` (no cluster, as the tests run) nothing is shared.
- `python src/labeler/env.py LABELER_LABEL_TABLES` prints what this node resolves.

## The mirror

Before submitting a batch of label jobs, run this on a StellarAI login node:

```bash
pixi run --frozen -e labeler labeler-sync mirror            # --dry-run to see first
```

This runs `rsync -rlpt --delete` from the shared folder into the cluster's folder and
writes `.mirror` (source, time, counts). The first run copies everything that
differs; later runs copy only what changed. Use `--from`/`--to` to name the folders.
Jobs read the mirror as of the last refresh, so no cron entry is needed.

Stellar needs no mirror while no Stellar job reads the label tables: its folder is
left as it was at the switch, and its login nodes use the shared folder. Before
running such a job on Stellar, run the same command on a Stellar login node.

Nothing saves into a mirror. A save, the merge lock and a review pointer all raise
`MirrorReadOnly`, a `PermissionError`, and the review page shows its message, which
names the shared folder.

A file written into the mirror anyway (a job's output, an old server) is never lost.
Before rsync replaces or deletes a file that changed since the last refresh, `mirror`
copies it to `<mirror>.replaced/<UTC time>/` and lists it. On the first refresh it
copies every file it replaces. If review files are among them, fold them into the
shared folder:

```bash
labeler-sync import-tree <mirror>.replaced/<time> --events /projects/EKOLEMEN/tokamaksi/diiid/events
```

## Pointers travel with the folder

A review pointer (`review/source.json`) used to name a suggestion table by an absolute
path under the owner's `LABELER_ROOT`, which the other cluster cannot read. Now:

- **Pointing at a table copies it in.** `labels.write_pointer` copies the table and
  its `.meta.json` and `_onsets.csv` into
  `<event>/review/sources/<method>/<version>/`.
- **The pointer names the copy relative to `review/`,** so the folder can move or be
  mirrored. `copied_from` records the original path.
- **Pointing again after a rebuild replaces the copy.**
- **Release names are unchanged.** A copy is published under its suggestion's
  name, `suggestions/<method>/<version>/<file>` (`hf_export.dataset_id`).

`labeler-sync localize` converts old pointers. It copies each table in, checks the
sha256, and rewrites the pointer, keeping its method, version, `set_at` and `set_by`.
`labeler-sync check` treats an absolute pointer as a note and a pointer to a missing
table as a problem.

## The lock across clusters

`labeler-sync probe DIR` takes DIR's merge lock `--rounds` times (default 200). In
each round it reads a counter, adds one, replaces the counter by rename and logs the
total it saw. `probe DIR --verify` then checks that every total was seen exactly once.

Measured on 2026-11-07 in `/projects/EKOLEMEN/tokamaksi/diiid/.lock-probe-2026-11-07`,
from `stellar-vis2` and `stellarai-vis2` at once:

- 400 rounds, 200 from each cluster, with 97 hand-offs between them in 22.7 s;
- no update lost, as both clusters' `--verify` agreed;
- the median wait for the lock was 24 ms on StellarAI and 7 ms on Stellar, the
  longest 1.16 s.

The `mkdir` lock and the rename are atomic through the NFS gateway, as the design
assumes. To repeat the probe, run the same command on both clusters at the same
time, then `--verify`:

```bash
P=.pixi/envs/labeler/bin/python
$P -m labeler.events.review.sync probe /projects/EKOLEMEN/tokamaksi/diiid/.lock-probe-<date>
$P -m labeler.events.review.sync probe /projects/EKOLEMEN/tokamaksi/diiid/.lock-probe-<date> --verify
```

## Switching on (the owner, once)

Use one StellarAI login node. `S` is the shared folder and `LS` is
`pixi run --frozen -e labeler labeler-sync`.

1. **Update both checkouts.** Pull this change into the Stellar and StellarAI
   checkouts of `nathan_dev`.
2. **Stop writers.** Stop every review server on both clusters, and every job that
   writes the label tables (see below).
3. **Fill `S` from StellarAI's folder.** StellarAI's folder is the current one;
   Stellar's review tables were imported into it. `S` holds an older copy from
   2026-10-17, so move that aside and copy:
   ```bash
   mv $S $S.2026-10-17
   rsync -rlpt /scratch/gpfs/EKOLEMEN/D3D/events/ $S/      # 8.1 GB, 32,648 files
   ```
4. **Localize the pointers.** Run `$LS localize --events $S --dry-run`, then without
   `--dry-run`.
5. **Initialize and check.** Run `$LS init --events $S` and `$LS check --events $S`.
6. **Switch on.** Run `$LS activate`. It refuses while `S` is a mirror, while any
   pointer is absolute, or while `check` reports a problem.
7. **Mirror StellarAI.** Run `$LS mirror`. The folder is the one `S` was copied
   from, so little changes. Stellar's folder is left as it is (see the mirror).
8. **Remove the old export.** Delete `export LABELER_LABEL_TABLES=$_E/D3D/events` from
   `~/.bashrc.d/faithaihub.bashrc` on StellarAI; an export wins, and this one names
   what is now the mirror. Then, in a new shell,
   `python src/labeler/env.py LABELER_LABEL_TABLES` prints `S`.
9. **Restart.** Start the review servers again and run `$LS status`; it says
   "shared folder, switched on".

After the switch, run `$LS mirror` on a StellarAI login node before each batch.

## Rolling back

1. Stop the review servers on both clusters.
2. On a StellarAI login node, run `$LS mirror` one last time, so its folder has every
   save made on `/projects`. If anyone reviewed from Stellar after the switch, run it
   on a Stellar login node too.
3. Run `mv $S/.shared $S/.shared.off.<date>`.
4. In each folder `mirror` wrote, run `mv <folder>/.mirror <folder>/.mirror.off.<date>`.
   The folder is writable again.
5. Restart the review servers. Each cluster then uses its own folder again, as
   before the switch.

## Jobs that write the label tables

Every batch script under `scripts/labeler/` resolves `LABELER_LABEL_TABLES` with
`env.py` on its compute node. After the switch, all of them read StellarAI's mirror. A job
that writes into the label tables writes into the mirror, and the next `mirror`
moves its output aside. Such a job should write to scratch, and its results should be
copied into `S` from a login node.

As of 2026-11-07, two of the 33 scripts write into the label tables with the
arguments they pass:

| Script | What it writes (under the label tables) | After the switch |
| --- | --- | --- |
| `silver_confine_base3.sbatch` | `confinement/extend_confine_base3/silver_shots.{csv,meta.json}` (the default `OUT_DIR`) | set `OUT_DIR` to scratch, then copy the result into `S` |
| `silver_tm_saw_rwm.sbatch`, `STAGE=table` only | `extend_tm_ours/`, `extend_saw_ours/`, `extend_rwm_ours/` under their events | run the `table` stage on a login node, or copy its output into `S` |

Three more write there only with arguments the scripts do not pass today:

- `sawtooth_population.sbatch` with `--shards 1`;
- `silver_ae_seldnet.sbatch` without `--no-write`;
- the `elm/serve.py` default.

The others read the label tables and write under `LABELER_ROOT` or a run folder. The
review page and `labeler-sync` write the shared folder from a login node.
