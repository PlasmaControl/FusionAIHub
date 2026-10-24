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

### Change-over: every process that saves must run this code

Before this change a save appended to `history.jsonl` and rewrote `labels.csv` in place.
The merge now replaces both files by rename, so a save made by older code in the same
moment is in no writer's file and is lost from the history without any notice (an old
server's rewrite of `labels.csv` can likewise drop merged rows). The first day is the
risk: a review server that was not restarted, or a script from an old checkout that calls
`labels.save`. So, before anyone saves into a folder with this code:

- restart every review server that can reach the folder, on both clusters, from a
  checkout that has this change;
- run no script from an older checkout that saves labels (`labels.save`, the Jalal
  import) against the folder;
- afterwards run `labeler-sync check`: it notes any line without a `save_id` that comes
  after the first line that has one, which is what an older writer leaves behind.

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
  after a rename of the reviewer or a regrid. Old lines of one file that share a
  computed id (a script saved one shot twice in a second) are told apart by their
  order, and `import-tree` writes that number into the copy as `legacy_id`; the line
  stays an old one, not a `save_id` line.
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
conflict record is kept.

A label the table refuses for its own sake (a malformed span, a shot whose rows carry
attrs the page cannot edit) is recorded in `review/saves/rejected.jsonl`, once, with
the reason; the line stays in its writer's file and the other lines still merge. It is
not tried again by itself: `check` and `status` report it, and after the cause is fixed
`labeler-sync merge --retry-rejected` tries it again. A label table that is invalid as
it stands (a bad row on any shot) is nobody's save to refuse: the merge stops with an
error and every line stays pending until the table is fixed.

Files the code creates are mode 0644, set in code whatever the account's umask. A
directory it creates follows its parent: 2775 (setgid, group-writable) when the parent
is group-writable already, which is how `init` leaves a shared folder, and a plain 0755
otherwise, so making a directory never opens a closed folder to its group. A line in a
writer's file that is torn, not JSON, or has no whole-number `shot` is skipped and
reported; the next save by that writer ends a torn one first.

`LABELER_AUTO_MERGE=0` makes the saves of that one process append only: its page saves
wait for a merge. It is not a gate for the folder: AE mask clicks always merge, and any
other account's saving process (its own switch left on) folds every writer's pending
lines into the merged files at its next save. Use it to test, not to hold lines back.

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
page fetch missing shots with fdp after `fdp login`. The page writes your own
`LABELER_ROOT` and, in the shared folder, your files under `review/saves/`. Its merge
also replaces `review/labels.csv`, `history.jsonl`, `masks.jsonl` and `saves/rejected.jsonl`
(by rename, under `review/.lock`), and the first save of a brand-new subcategory makes
that subcategory's folder.

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
| `merge [--dry-run] [--retry-rejected]` | Folds the writers' files into the merged files, for every event that has any. `--retry-rejected` tries the saves the label table refused once more, after the cause is fixed; a line an owner script took back is never retried. |
| `import-tree SOURCE [--tag T] [--dry-run]` | Adds the review lines of another label folder that this one does not hold (by id) to a writer file `saves/<kind>.import.<tag>.jsonl`, then merges. Run it again any time; it adds only what is new, never overrides a newer save, and skips a line that was withdrawn or refused here. |
| `init [--dry-run]` | Applies the permission table, for the owner, once: makes `review/` and `review/saves/` and sets 2775 on them, on an existing `review/legacy/` and on an existing `<event>/subcategories/` (where the first save of a new subcategory makes its folder; an event with none needs the owner to make it); sets 0644 on the files in `review/` and `review/saves/`; sets 0600 on `raw/elm_challenge_key_*.json` (other copies of the key are not touched). Needs the owner's rights; nothing else changes. |
| `check` | Writes nothing and exits 1 when something is wrong: an unreadable merged file, a lock older than 120 seconds, a file not readable by all, or a shot whose label is not what its last save made. It also prints notes: torn lines, saves the table refused (and the retry command), lines without a `save_id` after the first one that has it (an older writer), directories that are not group-writable, and shots that differ from a last save written before ids existed. |
| `repair [--shot N]` | Writes again the last save of the shots `check` reports, or of the shots named. It undoes an edit of `labels.csv` made on purpose (by hand, or by `regrid_saved_labels.py`, which leave the history alone), so use `--shot` only for a shot you mean to put back. |

Scripts that rewrite the merged files take the same lock now:
`scripts/labeler/regrid_saved_labels.py` (it also refuses a table that changed since
it read it), and `scripts/labeler/jalal_wpqh_import.py` (the saved-name rename and the
withdrawal of its own saves; the withdrawal records the lines in
`saves/rejected.jsonl` so no merge or `import-tree` brings them back). Run
`labeler-sync merge` before such a script and do not run it while anyone is saving. A
regrid changes `labels.csv` and not the history, so `check` may then report shots whose
label differs from their last save (a dropped span, a settled midpoint claim); do not
`repair` those.

Known limits: two processes that break the same stale lock at the same moment can both
proceed (`check` is the net). A lock counts as stale by the file server's clock, read
from the mtime of a file just made beside it, so a machine whose own clock is wrong does
not break a live lock.

The migration itself (backups, folding the live folder into the shared one, the
permissions, the probe) is not written here, because the shared folder is not active.
`import-tree` and `init` are the two commands it will use.
