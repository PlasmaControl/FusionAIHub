"""One label folder for every cluster: the `labeler-sync` commands that set it up.

The label tables live on `/projects` (`env.SHARED`), which login and visualization
nodes of both clusters mount and compute nodes do not. So:

    localize   copy each review pointer's table into the folder (`review/sources/`)
               and point at the copy by a relative path, so the folder can be read
               from any machine that has it
    activate   switch the shared folder on: from then on `LABELER_LABEL_TABLES`
               resolves to it wherever it is mounted (`env.default`)
    mirror     refresh this cluster's own folder (`env.local`) from the shared one,
               for the compute nodes; run it on a login node before submitting jobs
    probe      take a folder's merge lock many times and count, to show that two
               clusters writing into one folder at once lose nothing

A mirror holds `saves.MIRROR` and nothing saves into it (`saves.refuse_mirror`). A
file written into a mirror anyway (an older server, a job) is moved aside, never
lost: `mirror` copies every file it is about to replace or delete that changed
since the last refresh into `<mirror>.replaced/<time>/` first and says so.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from ... import env
from . import labels, saves

NAME = "LABELER_LABEL_TABLES"
#: What `mirror` never copies or deletes: the switch and the mirror's own marker,
#: lock directories, and the probe and temporary files of processes at work.
EXCLUDE = (
    f"/{env.ACTIVE}",
    f"/{saves.MIRROR}",
    f"{saves.LOCK}/",
    ".probe.*",
    "*.tmp",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    with saves.replacing(path) as tmp:
        tmp.write_text(json.dumps(data, indent=1) + "\n")


def absolute_pointers(event_dirs) -> list[tuple[Path, Path]]:
    """(event directory, table) for every pointer that names an absolute path."""
    found = []
    for event_dir in event_dirs:
        pointer = labels.pointer_path(event_dir)
        if pointer.is_file():
            table = Path(json.loads(pointer.read_text())["table"])
            if table.is_absolute():
                found.append((Path(event_dir), table))
    return found


# ---- localize ----


def localize(events, event_dirs, *, dry_run: bool) -> int:
    """Copy each absolute pointer's table into its event's `review/sources/` and
    point at the copy (`labels.copy_source`); the pointer keeps its method, version,
    `set_at` and `set_by`, and gains `copied_from` and `copied_at`."""
    code = 0
    for event_dir, table in absolute_pointers(event_dirs):
        rel = event_dir.relative_to(events)
        pointer = labels.pointer_path(event_dir)
        entry = json.loads(pointer.read_text())
        if not table.is_file():
            print(f"{rel}: {table} does not exist; not copied")
            code = 2
            continue
        where = f"review/{labels.SOURCES}/{entry.get('method')}/{entry.get('version')}/"
        print(f"{rel}: {'would copy' if dry_run else 'copied'} {table.name} into {where}")
        if dry_run:
            continue
        try:
            with saves.lock(event_dir / saves.REVIEW):
                relative = labels.copy_source(
                    event_dir, table, method=entry["method"], version=entry["version"]
                )
                copy = labels.pointed_table(event_dir, relative)
                if _sha256(copy) != _sha256(table):
                    raise OSError(f"{copy} differs from {table} after the copy")
                _write_json(pointer, {**entry, "table": relative,
                                      "copied_from": str(table), "copied_at": _now()})
        except (saves.StoreBusy, saves.StoreMissing, OSError, ValueError, KeyError) as error:
            print(f"{rel}: not localized: {error}")
            code = 2
    return code


# ---- activate ----


def activate(events, event_dirs, check, *, dry_run: bool) -> int:
    """Write `env.ACTIVE` into the shared folder, after making sure it can be: it is
    the folder `env.SHARED` names, no mirror, every pointer relative, `check` clean."""
    shared = Path(env.SHARED[NAME])
    if Path(events).resolve() != shared.resolve():
        print(f"only the shared folder {shared} is switched on, not {events}")
        return 2
    if (Path(events) / saves.MIRROR).exists():
        print(f"{events} is a mirror ({saves.MIRROR}); not switched on")
        return 2
    if (Path(events) / env.ACTIVE).is_file():
        print(f"{events} is switched on already")
        return 0
    left = absolute_pointers(event_dirs)
    if left:
        for event_dir, table in left:
            print(f"{event_dir.relative_to(events)}: source.json names {table}")
        print("labeler-sync localize first: other machines cannot open those tables")
        return 2
    if check(events, []) != 0:
        print("labeler-sync check reports problems; not switched on")
        return 2
    if dry_run:
        print(f"would switch on {events}")
        return 0
    _write_json(Path(events) / env.ACTIVE, {
        "activated_at": _now(), "by": getpass.getuser(), "host": saves.short_host(),
        "mirrors": {name: row["env"].get(NAME) for name, row in env.CLUSTERS.items()},
    })
    print(f"switched on {events}. Next: restart every review server; run "
          "`labeler-sync mirror` on a login node of each cluster; drop any "
          f"`export {NAME}=` from shell startup files (docs/labeler/shared_folder.md)")
    return 0


# ---- mirror ----


def _rsync(source: Path, target: Path, *, dry_run: bool) -> list[tuple[str, str]]:
    """Run rsync from `source` into `target` (recursive, links, modes, times, deletes,
    `EXCLUDE` left alone); return its itemized changes as (code, relative path)."""
    command = ["rsync", "-rlpt", "--delete", "--out-format=%i %n"]
    command += [f"--exclude={pattern}" for pattern in EXCLUDE]
    if dry_run:
        command.append("--dry-run")
    command += [f"{source}/", f"{target}/"]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode not in (0, 24):  # 24: a file vanished while it ran
        raise OSError(f"rsync exited {done.returncode}: {done.stderr.strip()[-500:]}")
    changes = []
    for line in done.stdout.splitlines():
        if line.startswith("*deleting"):
            changes.append(("*deleting", line[len("*deleting"):].strip()))
        elif " " in line:
            code, rel = line.split(" ", 1)
            changes.append((code, rel))
    return changes


def written_here(target: Path, changes, since: float | None) -> list[str]:
    """The files `changes` would replace or delete in `target` that changed after
    `since` (every one when None, a first refresh): written into the mirror, not
    copied from the shared folder."""
    found = []
    for code, rel in changes:
        if rel.endswith("/") or not (code == "*deleting" or code.startswith(">f")):
            continue
        path = target / rel
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if path.is_file() and (since is None or info.st_mtime > since):
            found.append(rel)
    return found


def mirror(source=None, target=None, *, dry_run: bool) -> int:
    """Refresh `target` (default: this cluster's own folder) from the shared folder."""
    source = Path(source or env.SHARED[NAME])
    if not (source / env.ACTIVE).is_file():
        print(f"{source} is not mounted here or not switched on (labeler-sync activate); "
              "run this on a login node")
        return 2
    target = Path(target or env.local(NAME) or "")
    if not str(target) or str(target) == ".":
        print("no cluster folder to mirror into here: give --to DIR")
        return 2
    if target.exists() and target.resolve() == source.resolve():
        print(f"{target} is the shared folder itself")
        return 2
    if (target / env.ACTIVE).exists():
        print(f"{target} holds {env.ACTIVE}: a shared folder, not a mirror")
        return 2
    if not target.is_dir():
        if dry_run:
            print(f"would make {target}")
            return 0
        target.mkdir()
    marker = target / saves.MIRROR
    since = marker.stat().st_mtime if marker.is_file() else None
    changes = _rsync(source, target, dry_run=True)
    local = written_here(target, changes, since)
    files = sum(1 for code, rel in changes if code.startswith(">f"))
    gone = sum(1 for code, _ in changes if code == "*deleting")
    if local:
        keep = target.with_name(f"{target.name}.replaced") / datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%SZ")
        first = " (a first refresh keeps every file it replaces)" if since is None else ""
        print(f"{len(local)} files in {target} changed there, not in {source}{first}; "
              f"{'they would be' if dry_run else 'they are'} kept in {keep}:")
        for rel in local[:20]:
            print(f"  {rel}")
        if len(local) > 20:
            print(f"  ... {len(local) - 20} more")
        if any(f"/{saves.REVIEW}/" in f"/{rel}" for rel in local):
            print("  review files among them: saves made into the mirror; look at them, and "
                  "fold the saves into the shared folder (labeler-sync import-tree)")
        if not dry_run:
            for rel in local:
                (keep / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target / rel, keep / rel)
    if dry_run:
        print(f"would copy {files} files and delete {gone} from {target}")
        return 0
    _rsync(source, target, dry_run=False)
    _write_json(marker, {
        "source": str(source), "synced_at": _now(), "by": getpass.getuser(),
        "host": saves.short_host(), "copied": files, "deleted": gone,
        "kept": len(local),
    })
    print(f"{target} mirrors {source}: {files} files copied, {gone} deleted")
    return 0


def describe(events) -> str | None:
    """One line on what `events` is, when it is the shared folder or a mirror."""
    for name, what in ((env.ACTIVE, "shared folder, switched on"),
                       (saves.MIRROR, "read-only mirror")):
        path = Path(events) / name
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                data = {}
            when = data.get("activated_at") or data.get("synced_at") or "?"
            source = f" of {data['source']}" if data.get("source") else ""
            return f"{what}{source}, at {when}"
    return None


# ---- probe ----


COUNTER = "counter.json"


def probe(folder, *, rounds: int, wait: float, pause: float) -> int:
    """Take `folder`'s merge lock `rounds` times; each time, read the counter, add one
    (to the total and to this writer's count) and replace it, and append the total
    this writer saw to its own log. Run it from two clusters at once, then
    `verify`: with a working lock every total is seen exactly once."""
    folder = Path(folder)
    review = folder / saves.REVIEW
    saves.shared_mkdir(review, under=folder, shared=True)
    saves.shared_mkdir(review / saves.SAVES, under=review, shared=True)
    me = saves.writer_id()
    counter, log = review / COUNTER, review / saves.SAVES / f"probe.{me}.jsonl"
    waits, busy = [], 0
    for i in range(rounds):
        started = time.monotonic()
        try:
            with saves.lock(review, wait=wait):
                waited = time.monotonic() - started
                data = (json.loads(counter.read_text()) if counter.is_file()
                        else {"total": 0, "by": {}})
                data["total"] += 1
                data["by"][me] = data["by"].get(me, 0) + 1
                _write_json(counter, data)
                saves.append_text(log, json.dumps(
                    {"round": i, "total": data["total"], "waited_s": round(waited, 3),
                     "at": time.time()}))
        except saves.StoreBusy:
            busy += 1
            continue
        waits.append(waited)
        if pause:
            time.sleep(random.uniform(0, pause))
    if waits:
        print(f"{me}: {len(waits)} rounds under the lock, {busy} gave up after {wait} s; "
              f"waited median {statistics.median(waits):.3f} s, max {max(waits):.3f} s")
    else:
        print(f"{me}: no round took the lock ({busy} gave up after {wait} s)")
    return 1 if busy else 0


def verify(folder) -> int:
    """Whether the probe lost nothing: the counter's total is the number of lines all
    writers logged, each writer's count is its own lines, and every total from 1 to
    the last was seen by exactly one round."""
    review = Path(folder) / saves.REVIEW
    data = json.loads((review / COUNTER).read_text())
    seen: dict[str, list[int]] = {}
    for path in sorted((review / saves.SAVES).glob("probe.*.jsonl")):
        writer = path.name[len("probe."):-len(".jsonl")]
        seen[writer] = [json.loads(line)["total"] for line in path.read_text().splitlines()
                        if line.strip()]
    totals = sorted(t for values in seen.values() for t in values)
    problems = []
    if totals != list(range(1, data["total"] + 1)):
        repeated = len(totals) - len(set(totals))
        problems.append(f"totals seen {len(totals)}, counter {data['total']}, "
                        f"{repeated} seen twice: an update was lost")
    for writer, values in seen.items():
        if data["by"].get(writer) != len(values):
            problems.append(f"{writer}: counter says {data['by'].get(writer)}, "
                            f"its log {len(values)}")
    hosts = sorted(seen)
    print(f"{data['total']} rounds by {len(hosts)} writers: "
          + ", ".join(f"{w} {len(seen[w])}" for w in hosts))
    for problem in problems:
        print(f"  {problem}")
    print("ok: no update lost" if not problems else "FAILED")
    return 1 if problems else 0
