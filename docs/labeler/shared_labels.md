# Shared label folder

Several people can run the review page under their own accounts and save into one
label folder. Each account writes only its own files; a merge, under a lock, folds
them into the files every reader already uses. This page says how that works, what a
contributor does and what the owner runs.

## Status

The shared folder on `/projects` (`/projects/EKOLEMEN/fusionaihub_labels/events`) is
not active. The label folder both clusters use today is still the one
`LABELER_LABEL_TABLES` names (default `/scratch/gpfs/EKOLEMEN/nc1514/events`), and the
code in this change only makes it safe for several accounts to save into one folder.
Before the `/projects` folder is the default, in this order:

1. The cluster defaults point both clusters at it (a separate change).
2. The owner runs `labeler-sync init` on it, once, so the group may write
   `review/` and `review/saves/` (see below).
3. The suggestion tables are published somewhere both clusters read. The page's
   pointer (`review/source.json`) names a table file, and today those files live
   under each cluster's own `LABELER_ROOT`, so a contributor on the other cluster
   would not find the table a pointer names.
4. The mkdir lock and the rename are probed across the two clusters (the design
   relies on both being atomic on the NFS server; that is not yet measured).
5. A release build (`hf_publish.py build`) does not check for unmerged saves yet. The
   export reads only the merged files, so a save still waiting in a writer's file is
   not in a release: the owner runs `labeler-sync merge` and `labeler-sync check`
   before building, or the build learns to refuse.

The review page shows nothing yet when a save is waiting for its merge: the reply of
`POST /api/label` carries `merge.pending`, and showing it is a later change to the page.

## How a save is stored

A save is one line. It goes first into the account's own file, and only then into the
shared files.

- `review/saves/history.<login>.<host>.jsonl` and, for the Alfven eigenmode mask
  clicks, `review/saves/masks.<login>.<host>.jsonl`. The writer name is the server's
  login and the short name of the node it runs on, so one login on two nodes is two
  writers and no file is ever appended to from two NFS clients. The line is written
  with one append and flushed to disk before the page is told it is saved.
- Every line carries a `save_id` (the time to the microsecond and six random
  characters) and the `writer`. A line written before this change has neither; its id
  is computed from its `saved_at` and a hash of shot and reviewer, so it is the same
  after a rename of the reviewer or a regrid.
- The merge takes `review/.lock` (a directory made with `mkdir`, which is atomic on
  the server), reads the writers' files, and folds every line it does not hold yet
  into `review/history.jsonl` (or `masks.jsonl`). A shot's last line, ordered by
  `saved_at` and then `save_id`, decides its label: that label replaces the shot's
  rows in `review/labels.csv`. Both files are replaced by a rename of a uniquely named
  temporary file, so a reader sees the whole old file or the whole new one.
- The merge is the same whoever runs it and in whatever batches: the same set of
  lines always gives the same files. A line is never moved once it is merged; a late
  arrival with an earlier time goes in front of the later saves, so a shot's versions
  renumber from that point (`GET /api/history`).
- The page, `versions`, the silver set, the Hugging Face export and training read the
  merged files exactly as before.

If the lock stays held for 20 seconds, or the merge fails, the save is still on disk
in its writer's file; the reply says `merge.pending` and the next save or
`labeler-sync merge` completes it. A lock older than 120 seconds is taken as left by a
crashed process and is broken. At the end of every merge each shot's last save is
compared with what `labels.csv` holds, and a label that is missing or equals an
earlier save of the shot (an update another process lost) is written again. A label
that matches no save of the shot (the owner edited the file by hand) is left alone and
reported by `labeler-sync check`.

Two saves of one shot by two people: both lines are kept, the later one is the label,
and the earlier one stays in the history. Nothing is shown side by side and no
conflict record is kept. A label the table refuses (a malformed span) is moved to
`review/saves/rejected.jsonl`, once, with the reason, and the other lines still merge.

Files the code creates are mode 0644 and the directories it creates are 2775
(setgid, group-writable), set in code whatever the account's umask. A line in a
writer's file that is torn or not JSON is skipped and reported; the next save by that
writer ends it first.

`LABELER_AUTO_MERGE=0` makes a save append only: nothing is merged until someone runs
`labeler-sync merge`. The owner can use it to read the writers' files before they
count.

## Reviewer names

The name list is `reviewers.txt` next to the events, plus every name found in the
merged logs. An account that cannot write `reviewers.txt` still gets Add Name: the
name is not written anywhere, and it is listed from the account's first save on,
because the saved line carries it.

## Contributor quick start

This is for someone who runs the page under their own account, on Stellar or
StellarAI, once the shared folder is active (see Status). The owner adds you to the
folder's group; `id` shows it.

```bash
git clone <repo> && cd FusionAIHub && git checkout nathan_dev
export LABELER_LABEL_TABLES=<the shared folder the owner names>
pixi run --frozen -e labeler labeler-sync status
pixi run --frozen -e labeler labeler-verify
```

`labeler-verify` prints the page's address and the `ssh -L` line for your laptop. Pick
your name (the first time, Add Name) and save as usual. You run no sync: each save
merges itself, and `labeler-sync status` shows what the folder holds and what is
unmerged.

The page reads each shot's spectrogram rows from your `LABELER_ROOT`, not from the
shared folder. Link the owner's row store read-only (the owner says where) or let the
page fetch missing shots with fdp after `fdp login`. The page writes only your own
`LABELER_ROOT` and the shared folder's `review/saves/`.

A mistake of yours is not undone by deleting a line: open the shot, History, restore
the earlier version as a draft and save. There is no command to withdraw a save.

## What the owner runs

`labeler-sync` is `python -m labeler.events.review.sync`, or
`pixi run -e labeler labeler-sync` in an environment that has the labeler feature.
Every command takes `--events DIR` (default: the label folder of the environment) and
`--event REL` (repeatable, for example `alfven_eigenmode` or
`mhd_n1/subcategories/m1`).

| Command | What it does |
|---|---|
| `status` | Lines held, writers' files, unmerged lines, refused lines and a held lock for each event. Writes nothing. |
| `merge [--dry-run]` | Folds the writers' files into the merged files, for every event that has any. |
| `import-tree SOURCE [--tag T] [--dry-run]` | Adds the review lines of another label folder that this one does not hold (by id) to a writer file `saves/<kind>.import.<tag>.jsonl`, then merges. Run it again any time; it adds only what is new, never overrides a newer save, and skips a line that was withdrawn or refused here. |
| `init [--dry-run]` | Makes `review/` and `review/saves/` and sets mode 2775 on them. Needs the owner's rights; nothing else changes. |
| `check` | Writes nothing and exits 1 when something is wrong: an unreadable merged file, a lock older than 120 seconds, a file not readable by all, or a shot whose label is not what its last save made. It also prints notes (torn lines, directories that are not group-writable, last saves written before ids existed). |
| `repair [--shot N]` | Writes again the label of the shots `check` reports, or of the shots named. |

Scripts that rewrite the merged files take the same lock now:
`scripts/labeler/regrid_saved_labels.py` (it also refuses a table that changed since
it read it), and `scripts/labeler/jalal_wpqh_import.py` (the saved-name rename and the
withdrawal of its own saves; the withdrawal records the lines in
`saves/rejected.jsonl` so no merge or `import-tree` brings them back). Run
`labeler-sync merge` before such a script and do not run it while anyone is saving.

The migration itself (backups, folding the live folder into the shared one, the
permissions, the probe) is not written here, because the shared folder is not active.
`import-tree` and `init` are the two commands it will use.
