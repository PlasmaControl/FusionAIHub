"""Per-writer saves and the merge that folds them into the canonical review files.

Several people may save labels into one folder, from different accounts and
nodes, even over NFS. Nobody appends to a file somebody else appends to, and the
canonical files are replaced by one process at a time:

    review/saves/history.<writer>.jsonl   every save of one writer, appended once
    review/saves/masks.<writer>.jsonl     the same for AE mask decisions
    review/labels.csv, history.jsonl, masks.jsonl   the canonical files, written
                                          only by `merge`, by rename

A writer is `<login>.<short hostname>`, derived by the server, never taken from the
client: one login on two nodes is two writers, so no two NFS clients append to one
file. A save is one line written with `O_APPEND` and fsynced; after that it is
durable, whatever happens to the merge. The merge takes the directory lock
`review/.lock` (a `mkdir`, atomic on the NFS server; stale after two minutes),
re-lists the directories so a client does not work from a minute-old cache, and
replaces each canonical file by renaming a uniquely named temporary file over it.

A line is known to the merge when its id is in the canonical log. The id is the
line's `save_id`, or, for a line written before ids, `<saved_at>-<hash of shot and
login>` (never the name or the intervals, so renaming a reviewer or regridding
a label keeps ids; old lines of one file that share an id are told apart by their
order in it, see `ids_of`). New lines go into the log in `(saved_at, id)` order and a
shot's label is the last of its lines: last save wins, by time, whoever merges
and in whatever batches. Running the merge again changes nothing. A line the
label table refuses is moved to `saves/rejected.jsonl` and not retried.
"""

from __future__ import annotations

import errno
import getpass
import hashlib
import json
import os
import random
import re
import shutil
import socket
import stat
import time
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REVIEW, SAVES, LOCK = "review", "saves", ".lock"
SUBDIR = "subcategories"  # the folder `events.taxonomy.SUBDIR` names
KINDS = ("history", "masks")
REJECTED = "rejected.jsonl"
#: Modes are set in code, whatever the process's umask: a reader needs the files
#: readable, a writer in the group needs the directories group-writable.
FILE_MODE, DIR_MODE = 0o644, 0o2775
STALE_LOCK_S = 120.0


class StoreBusy(RuntimeError):
    """The merge lock stayed held for the whole wait."""


class StoreMissing(RuntimeError):
    """The folder or the event is not there; nothing is created."""


# ---- identities and ids ----


def short_host() -> str:
    return socket.gethostname().split(".")[0]


def writer_id(login: str | None = None) -> str:
    """`<login>.<short hostname>`: the one process family that appends to a file."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{login or getpass.getuser()}.{short_host()}")


def new_save(now: datetime | None = None) -> tuple[str, str]:
    """`(saved_at, save_id)`: the time to the second, and an id that also orders
    one writer's saves within a second."""
    now = now or datetime.now(UTC)
    return (
        now.isoformat(timespec="seconds"),
        f"{now:%Y%m%d%H%M%S%f}-{uuid.uuid4().hex[:6]}",
    )


def line_id(entry: dict) -> str:
    """The line's `save_id`, or one computed for a line written before ids."""
    given = entry.get("save_id")
    if isinstance(given, str) and given:
        return given
    stamp = re.sub(r"\D", "", str(entry.get("saved_at") or ""))[:14]
    who = f"{entry.get('shot')}|{entry.get('reviewer')}"
    return f"{stamp}-{hashlib.sha1(who.encode()).hexdigest()[:6]}"


def ids_of(entries: Iterable[dict]) -> list[str]:
    """The id of each line of one file, in order.

    Lines written before ids can share one (the same shot and reviewer within one
    second, as when a script saved a label under two names); the later ones of a file
    get `.2`, `.3`, ... so a copy of the file gives the same ids line by line."""
    seen: dict[str, int] = {}
    found = []
    for entry in entries:
        base = line_id(entry)
        if isinstance(entry.get("save_id"), str) and entry["save_id"]:
            found.append(base)
            continue
        seen[base] = seen.get(base, 0) + 1
        found.append(base if seen[base] == 1 else f"{base}.{seen[base]}")
    return found


def order_key(entry: dict) -> tuple[str, str]:
    return str(entry.get("saved_at") or ""), line_id(entry)


# ---- files ----


def retrying(call: Callable, directory=None):
    """`call()`, again (up to twice) when NFS says the file handle went stale
    because another client renamed a new file over the name just read."""
    for attempt in range(3):
        try:
            return call()
        except OSError as error:
            if error.errno != errno.ESTALE or attempt == 2:
                raise
            if directory is not None:
                refresh(directory)
            time.sleep(0.1 * (attempt + 1))


def read_lines(path, *, strict: bool = False, bad: list | None = None) -> list[tuple[dict, str]]:
    """`(entry, original text)` of each JSON object line of the file, in order.

    A canonical log is read `strict`: a line that does not parse raises, because a
    merge must not rewrite a file it cannot reproduce. A writer file is read
    tolerantly: a torn or non-JSON line is skipped and its `(path, line number)`
    is added to `bad`.
    """
    path = Path(path)
    if not path.is_file():
        return []
    text = retrying(lambda: path.read_text(encoding="utf-8"), path.parent)
    found = []
    for number, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError("not a JSON object")  # noqa: TRY004 (caught below)
        except ValueError as error:
            if strict:
                raise ValueError(f"{path}: line {number} is not a JSON object") from error
            if bad is not None:
                bad.append((path, number))
            continue
        found.append((entry, line))
    return found


def _chmod(path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:  # not ours to change: the folder's own mode stands
        pass


def _fsync_dir(directory) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def shared_mkdir(path, *, under) -> None:
    """Make `path`, one level, with the shared directory mode. `under` must exist:
    a missing folder fails here, loudly, instead of a new tree appearing."""
    path, under = Path(path), Path(under)
    if not under.is_dir():
        raise StoreMissing(f"{under} does not exist; nothing was created")
    try:
        os.mkdir(path)
    except FileExistsError:
        return
    _chmod(path, DIR_MODE)


def review_dir(event_dir) -> Path:
    """The event's `review/` folder, made when it is not there yet.

    The event's own folder is never made, except a subcategory's
    (`<parent>/subcategories/<key>`), which its first save makes under the parent."""
    event_dir = Path(event_dir)
    if event_dir.parent.name == SUBDIR and not event_dir.is_dir():
        if not event_dir.parent.is_dir():
            shared_mkdir(event_dir.parent, under=event_dir.parent.parent)
        shared_mkdir(event_dir, under=event_dir.parent)
    shared_mkdir(event_dir / REVIEW, under=event_dir)
    return event_dir / REVIEW


def append_text(path, text: str) -> None:
    """Add `text` and a newline to the file: one `O_APPEND` write, then fsync.
    A final line left without its newline (a torn write) is ended first, so the
    next line is never glued to it. The file is made 0644."""
    path = Path(path)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, FILE_MODE)
    try:
        info = os.fstat(fd)
        if info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) != FILE_MODE:
            os.fchmod(fd, FILE_MODE)
        data = text.encode("utf-8") + b"\n"
        if info.st_size and os.pread(fd, 1, info.st_size - 1) != b"\n":
            data = b"\n" + data
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    if info.st_size == 0:
        _fsync_dir(path.parent)


def append(event_dir, kind: str, entry: dict) -> Path:
    """Write the save into its writer's own file, `saves/<kind>.<writer>.jsonl`."""
    review = review_dir(event_dir)
    shared_mkdir(review / SAVES, under=review)
    path = review / SAVES / f"{kind}.{entry.get('writer') or writer_id()}.jsonl"
    append_text(path, json.dumps(entry))
    return path


@contextmanager
def replacing(path):
    """Yield a uniquely named temporary sibling to write, then rename it over `path`.

    The name carries host, pid and a random part, so two writers never share a
    temporary file; the file is made 0644, flushed to disk and renamed, so a reader
    sees the whole old file or the whole new one. The parent is not created.
    """
    path = Path(path)
    tmp = path.with_name(
        f".{path.name}.{short_host()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    done = False
    try:
        yield tmp
        _chmod(tmp, FILE_MODE)
        with open(tmp, "rb") as written:
            os.fsync(written.fileno())
        os.replace(tmp, path)
        done = True
        _fsync_dir(path.parent)
    finally:
        if not done:
            tmp.unlink(missing_ok=True)


def refresh(directory) -> None:
    """Make this NFS client re-read `directory`: creating and removing an entry
    updates its cached listing and attributes. Writes nothing that stays."""
    probe = Path(directory) / f".probe.{uuid.uuid4().hex[:8]}"
    try:
        os.close(os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        os.unlink(probe)
    except OSError:
        pass


# ---- the lock ----


def lock_age(review) -> float | None:
    """Seconds since the merge lock was taken, or None when it is free."""
    try:
        return time.time() - (Path(review) / LOCK).stat().st_mtime
    except OSError:
        return None


def holder(review) -> str | None:
    """Who holds the lock, and for how long, or None when it is free."""
    age = lock_age(review)
    if age is None:
        return None
    try:
        who = (Path(review) / LOCK / "holder").read_text().strip()
    except OSError:
        who = "unknown"
    return f"{who} for {age:.0f} s"


def _break_stale(path: Path, stale: float) -> bool:
    """Remove a lock older than `stale` seconds (its holder crashed); True when it
    is gone, by this call or another's. Two breakers racing on one stale lock are
    the one accepted risk: `check` is the net."""
    age = lock_age(path.parent)
    if age is None:
        return True
    if age <= stale:
        return False
    gone = path.with_name(f"{LOCK}.stale.{uuid.uuid4().hex[:8]}")
    try:
        os.rename(path, gone)
    except FileNotFoundError:
        return True
    shutil.rmtree(gone, ignore_errors=True)
    return True


@contextmanager
def lock(review, *, wait: float = 20.0, stale: float = STALE_LOCK_S):
    """Hold `review/.lock` (a `mkdir`) for the body; raise `StoreBusy` when it stays
    held for `wait` seconds. After it is taken both directory listings are
    refreshed, so what the body reads is not older than the last holder's writes."""
    review = Path(review)
    path = review / LOCK
    deadline = time.monotonic() + wait
    while True:
        try:
            os.mkdir(path)
            break
        except FileNotFoundError:
            raise StoreMissing(f"{review} does not exist") from None
        except FileExistsError:
            if _break_stale(path, stale):
                continue
            if time.monotonic() >= deadline:
                raise StoreBusy(f"{path} is held by {holder(review)}") from None
            time.sleep(random.uniform(0.05, 0.25))
    try:
        _chmod(path, DIR_MODE)
        try:
            (path / "holder").write_text(
                f"{writer_id()} pid {os.getpid()} at "
                f"{datetime.now(UTC).isoformat(timespec='seconds')}\n"
            )
        except OSError:
            pass
        refresh(review)
        refresh(review / SAVES)
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---- the merge ----


@dataclass
class Merged:
    """What one merge did. Lists hold history lines (dicts)."""

    added: list = field(default_factory=list)       # new to the canonical log
    applied: list = field(default_factory=list)     # changed the label table
    superseded: list = field(default_factory=list)  # added, but a later save of the shot won
    skipped: list = field(default_factory=list)     # (entry, why): refused, now rejected
    repaired: list = field(default_factory=list)    # re-applied: the table had lost them
    bad: list = field(default_factory=list)         # (path, line number) of torn writer lines
    ids: set = field(default_factory=set)           # every id the canonical log holds now

    def outcome(self, save_id: str) -> dict:
        """What the page is told of one save: merged or `pending`, and whether a
        later save of the shot already stands over it."""
        if save_id in self.ids:
            return {
                "pending": False,
                "superseded": any(line_id(e) == save_id for e in self.superseded),
            }
        why = next((w for e, w in self.skipped if line_id(e) == save_id), None)
        return {"pending": True, "why": why or "the save is not in the shared log yet"}


Apply = Callable[[list], list]   # winners -> [(entry, why)] it could not apply
Stale = Callable[[list], list]   # every entry in order -> those whose label the table lost


def rejected_ids(review) -> set[str]:
    return {e["id"] for e, _ in read_lines(Path(review) / SAVES / REJECTED) if "id" in e}


def merge(
    event_dir,
    kind: str,
    apply: Apply | None = None,
    stale: Stale | None = None,
    *,
    extra: Iterable[dict] = (),
    wait: float = 20.0,
    dry_run: bool = False,
) -> Merged:
    """Fold the writers' lines of `kind` into the canonical log (see the module).

    `apply(winners)` changes the label table for the lines that are the last of
    their shot and returns the ones it refused. `stale(entries)` says which shots'
    last save the table does not show (an update another client lost); they are
    applied again. `extra` are lines to consider besides the writers' (a dry run
    of an import). A dry run takes no lock and writes nothing.
    """
    review = Path(event_dir) / REVIEW
    if not review.is_dir() and not extra:  # no save has been made here: nothing to fold
        return Merged()
    if dry_run:
        return _merge(review, kind, apply, stale, extra, dry_run=True)
    with lock(review, wait=wait):
        return _merge(review, kind, apply, stale, extra, dry_run=False)


def _merge(review, kind, apply, stale, extra, *, dry_run) -> Merged:
    result = Merged()
    canonical = review / f"{kind}.jsonl"
    lines = read_lines(canonical, strict=True)
    have = set(ids_of(e for e, _ in lines))
    refused = rejected_ids(review)
    found: dict[str, tuple[dict, str]] = {}
    for path in sorted((review / SAVES).glob(f"{kind}.*.jsonl")):
        pairs = read_lines(path, bad=result.bad)
        for i, pair in zip(ids_of(e for e, _ in pairs), pairs, strict=True):
            found.setdefault(i, pair)
    for entry in extra:
        found.setdefault(line_id(entry), (entry, json.dumps(entry)))
    incoming = sorted(
        (pair for i, pair in found.items() if i not in have and i not in refused),
        key=lambda pair: order_key(pair[0]),
    )
    for pair in incoming:  # into time order; the lines already there never move
        at = len(lines)
        while at and order_key(lines[at - 1][0]) > order_key(pair[0]):
            at -= 1
        lines.insert(at, pair)
    added = [e for e, _ in incoming]
    done: set[int] = set()
    while True:  # the last line of each shot decides its label
        last = {e.get("shot"): e for e, _ in lines}
        winners = [e for e in added if last[e.get("shot")] is e and id(e) not in done]
        if not winners:
            break
        refusals = apply(winners) if (apply and not dry_run) else []
        done.update(id(e) for e in winners)
        if not refusals:
            break
        gone = {id(e) for e, _ in refusals}
        lines = [p for p in lines if id(p[0]) not in gone]
        added = [e for e in added if id(e) not in gone]
        done -= gone
        result.skipped += refusals
    result.added = added
    result.applied = [e for e in added if id(e) in done]
    result.superseded = [e for e in added if id(e) not in done]
    if stale is not None and not dry_run:
        try:  # a repair that cannot be made must not hold the saves back
            again = stale([e for e, _ in lines])
            if again and apply is not None:
                refused_again = {id(e) for e, _ in apply(again)}
                result.repaired = [e for e in again if id(e) not in refused_again]
        except (OSError, ValueError):
            pass
    result.ids = set(ids_of(e for e, _ in lines))
    if not dry_run:
        if added:
            with replacing(canonical) as tmp:
                tmp.write_text("".join(text + "\n" for _, text in lines), encoding="utf-8")
        if result.skipped:
            reject(review, result.skipped)
    return result


def reject(review: Path, skipped: list) -> None:
    """Remember the lines `skipped` (pairs of line and reason) so no merge brings
    them in again: the ones the label table refused, or an owner script took back.
    The caller holds the folder's lock."""
    review = Path(review)
    shared_mkdir(review / SAVES, under=review)
    path = review / SAVES / REJECTED
    kept = [text for _, text in read_lines(path)]
    at = datetime.now(UTC).isoformat(timespec="seconds")
    kept += [
        json.dumps({"id": line_id(e), "shot": e.get("shot"), "why": why, "at": at,
                    "writer": e.get("writer")})
        for e, why in skipped
    ]
    with replacing(path) as tmp:
        tmp.write_text("".join(text + "\n" for text in kept), encoding="utf-8")


def pending(event_dir, kind: str) -> int:
    """How many lines of the writers' files the canonical log does not hold yet."""
    return len(merge(event_dir, kind, dry_run=True).added)
