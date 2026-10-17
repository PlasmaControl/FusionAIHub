"""Shots built when they are opened: the temporary cache, the fetch of what a build
lacks, and plain failure messages (docs/labeler/on_demand.md).

The review server opens any shot of the catalog, about 4,800, on any event, and only
a few hundred of them have a prebuilt store. `ui.app.Builds` builds the rest on
first open, through `build` here:

* A shot is PROTECTED (`protected`) when gold, silver, the event's roster or a list
  marked `protect` holds it, or its store is already on disk: it builds under
  `$LABELER_ROOT` as ever. Any other shot builds into the temporary cache,
  `<root>/cache/ondemand/<shot>/` (`Paths.ondemand`): its review store, its feature
  file and the raw groups fetched for it, through `Paths.overlay` (`overlay_paths`).
  What the data root already holds of the shot (its feature file, its raw record) is
  linked in, read-only in effect: a write replaces the link, never the original.
* The cache holds `LABELER_CACHE_GB` (default 50) GB. After each build `evict` removes
  the least recently opened shots until it fits, never a shot being built or opened in
  the last `GRACE_S` seconds, so the cap is soft by a few shots. Labels are not here,
  so they are never evicted: saving works the same on any shot.
* A build that fails for a feature missing from the shot's feature file (a
  `MissingFeature`, such as locked mode's `pinj_total`) fetches it (`fetch_features`:
  the corpus' route through the raw groups, else fdp), writes it into the feature
  file and builds once more. A raw group a panel needs is fetched by `raw.raw_signal`
  as before. One toksearch call runs at a time in the process (`resolve_fdp.FETCH_LOCK`,
  a call waits at most `resolve_fdp.LOCK_TIMEOUT_S` for it), and nothing is fetched
  when the server is not under fdp, fetching is off or fdp refused the token.
* What a fetch writes into a feature file never includes a miss: a fetch that fails
  (off, refused, timed out) is not a permanent "missing" for the pipelines.
* A failure reaches the page as a `BuildFailure`: a short message without paths or
  exception names, the full text, and a `code`.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config import Paths, store_event
from ..env import getenv
from ..features import namespace as ns
from ..features import resolve_corpus
from ..features.store import MissingFeature, write_features
from . import raw, rosters, shotlists, taxonomy
from .review import build as review_build
from .review import fetch_worker, gold, silver
from .verify import NoDataError

log = logging.getLogger(__name__)

#: The cache's size cap in GB (1e9 bytes), from the environment.
CAP_ENV = "LABELER_CACHE_GB"
DEFAULT_CAP_GB = 50.0
#: A shot opened this recently is never evicted, so a page reading its store does not
#: lose it to a build that began after.
GRACE_S = 300.0
#: The file whose mtime says when a shot was last opened, for the LRU order.
STAMP = ".opened"
#: The shot numbers the review opens off every list: DIII-D's six-digit shots. Anything
#: shorter is no shot of the catalog's era, and a typed number is checked against this
#: before it reaches a build.
SHOT_RANGE = (100_000, 999_999)

CODES = ("missing_signal", "no_data", "fetch_off", "build_failed")
#: The longest `note` of a 202 body, for the activity bar.
NOTE_MAX = 40

#: What the page calls a feature, for a message ("No beam power (pinj_total) for 170659").
LABELS = {
    "ip": "plasma current", "betan": "normalized beta", "betap": "poloidal beta",
    "pinj_total": "beam power", "tinj_total": "beam torque", "bt": "toroidal field",
    "qmin": "minimum q", "q95": "edge q", "li": "internal inductance",
    "wmhd": "stored energy", "ece": "ECE", "co2": "CO2 interferometer",
    "ech_power_total": "ECH power", "gas": "gas puff",
}


# ---- openable shots ----


def openable(paths: Paths, shot: int) -> bool:
    """Whether a typed shot may be opened though no list holds it: a six-digit number
    that is not a blind test shot."""
    shot = int(shot)
    return SHOT_RANGE[0] <= shot <= SHOT_RANGE[1] and shot not in shotlists.blind_set(paths)


def _quiet(check, *args) -> bool:
    """`check(*args)` as a bool; a table that is missing or unreadable holds nothing."""
    try:
        return bool(check(*args))
    except (OSError, ValueError, KeyError):
        return False


def protected(paths: Paths, event: str, shot: int) -> bool:
    """Whether `shot` builds into the data root rather than the temporary cache: its
    store is already there, or gold, silver, the event's roster or a `protect` list
    holds it."""
    shot, tables = int(shot), paths.label_tables
    if (paths.spectrograms / store_event(event) / f"{shot}.h5").exists():
        return True
    if _quiet(gold.is_gold, tables, shot) or _quiet(silver.is_silver, tables, shot):
        return True
    if _quiet(_on_roster, paths, event, shot):
        return True
    try:
        sets = shotlists.members(paths)
        return any(
            shot in sets.get(item.name, ())
            and (item.protect or shotlists.sidecar(item.path).get("protect"))
            for item in shotlists.entries(paths)
        )
    except (OSError, ValueError):
        return False


def has_store(paths: Paths, shot: int, event: str | None = None) -> bool:
    """Whether `shot`'s review store is built: `event`'s, and current, or with no
    event any event's, in the data root or the temporary cache."""
    shot = int(shot)
    if event:
        return bool(review_build.current(
            replace(paths, read_cache=True).spectrogram_file(event, shot), event))
    pattern = f"*/{shot}.h5"
    for where in (paths.spectrograms, shot_dir(paths, shot) / "spectrograms"):
        try:
            if any(path.is_file() for path in Path(where).glob(pattern)):
                return True
        except OSError:
            continue
    return False


def _on_roster(paths: Paths, event: str, shot: int) -> bool:
    parent = taxonomy.load(paths.events_manifest).parent(event)
    roster = rosters.read_roster(rosters.roster_path(parent or event, root=paths.label_tables))
    return bool((roster.shot == shot).any())


# ---- the temporary cache ----


def cap_bytes() -> int:
    """The cache's cap, `LABELER_CACHE_GB` GB (default 50); a value that is not a
    positive number is the default."""
    try:
        gigabytes = float(getenv(CAP_ENV, str(DEFAULT_CAP_GB)))
    except ValueError:
        gigabytes = 0.0
    return int((gigabytes if gigabytes > 0 else DEFAULT_CAP_GB) * 1e9)


def shot_dir(paths: Paths, shot: int) -> Path:
    return paths.ondemand / str(int(shot))


def in_cache(paths: Paths, path) -> bool:
    """Whether `path` lies in the temporary cache."""
    return paths.ondemand in Path(path).parents


def touch(paths: Paths, shot: int) -> None:
    """Stamp the shot as opened now, for `evict`; nothing for a shot not cached."""
    where = shot_dir(paths, shot)
    if where.is_dir():
        try:
            (where / STAMP).touch()
        except OSError:  # a read-only tree still serves the shot
            pass


def _seed(source: Path, target: Path) -> None:
    """Link the data root's file in, where the cache has none: reads see it, and a
    write (`write_group`, `write_features`) copies it and renames over the link."""
    if target.exists() or target.is_symlink() or not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source)
    except OSError:
        pass


def overlay_paths(paths: Paths, shot: int) -> Paths:
    """`paths` with the shot's cache directory as the place to write: its review store
    and features there, its raw groups in `<dir>/raw`; what the data root already holds
    of the shot is linked into it. Stamps the shot opened."""
    shot = int(shot)
    where = shot_dir(paths, shot)
    name = f"{shot}_processed.h5"
    _seed(paths.raw_cache / name, where / "raw" / name)
    features = f"{shot}_features.h5"
    _seed(paths.features / features, where / "features" / features)
    where.mkdir(parents=True, exist_ok=True)
    touch(paths, shot)
    return replace(paths, overlay=where, raw_cache=where / "raw")


def _size(where: Path) -> int:
    """Bytes of the regular files under `where`; links are not counted."""
    total = 0
    for folder, _dirs, files in os.walk(where):
        for name in files:
            try:
                info = os.lstat(os.path.join(folder, name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def cached(paths: Paths) -> list[tuple[int, float, int]]:
    """`(shot, opened, bytes)` for every shot in the cache, `opened` an epoch time."""
    found = []
    try:
        folders = [item for item in os.scandir(paths.ondemand)
                   if item.is_dir(follow_symlinks=False) and item.name.isdigit()]
    except OSError:
        return []
    for item in folders:
        try:
            opened = (Path(item.path) / STAMP).stat().st_mtime
        except OSError:
            try:
                opened = item.stat().st_mtime
            except OSError:
                continue
        found.append((int(item.name), opened, _size(Path(item.path))))
    return found


def usage(paths: Paths) -> int:
    """Bytes the temporary cache holds."""
    return sum(size for _, _, size in cached(paths))


#: Shots being built now, which `evict` leaves alone.
_active: dict[int, int] = {}
_active_lock = threading.Lock()
_evict_lock = threading.Lock()


def evict(paths: Paths, *, keep=(), cap: int | None = None, now: float | None = None,
          grace: float | None = None) -> list[int]:
    """Remove the least recently opened shots from the cache until it fits `cap` bytes
    (the environment's by default); the shots in `keep`, those being built and those
    opened within `grace` seconds stay, so the cache may stay over its cap. Returns
    the shots removed, oldest first."""
    cap = cap_bytes() if cap is None else cap
    now = time.time() if now is None else now
    grace = GRACE_S if grace is None else grace
    with _evict_lock:
        found = cached(paths)
        total = sum(size for _, _, size in found)
        with _active_lock:
            busy = set(keep) | set(_active)
        removed = []
        for shot, opened, size in sorted(found, key=lambda item: item[1]):
            if total <= cap:
                break
            if shot in busy or now - opened < grace:
                continue
            shutil.rmtree(shot_dir(paths, shot), ignore_errors=True)
            total -= size
            removed.append(shot)
        if removed:
            log.info("on-demand cache: evicted %s, now %.1f GB", removed, total / 1e9)
        return removed


def _discard_if_empty(paths: Paths, shot: int) -> None:
    """Drop a shot's cache directory that holds nothing but links and the stamp (a
    build that fetched nothing and failed)."""
    where = shot_dir(paths, shot)
    if where.is_dir() and _size(where) == 0:
        shutil.rmtree(where, ignore_errors=True)


# ---- what is being fetched ----

_fetching: dict[int, list[str]] = {}
_fetching_lock = threading.Lock()


def fetching_signals(shot: int) -> list[str]:
    """The signals being fetched for `shot` now: the features `fetch_features` is on,
    and the raw group `raw` is fetching."""
    with _fetching_lock:
        names = list(_fetching.get(int(shot), ()))
    record = raw.progress_for(shot)
    if record is None or names:  # a feature fetch names the group it is fetching
        return names
    group = record.get("group") or ""
    stage = record.get("stage") or ""
    if not group and stage.startswith("fetching "):
        group = stage[len("fetching "):]
    return [group] if group else []


# ---- an fdp token that was refused ----

#: The digest of the token fdp refused (a 401 or 403), until another is read: every
#: later fetch is refused without asking fdp again, as the fetch worker's would be.
_refused: str | None = None
_refused_lock = threading.Lock()
REFUSED_NOTE = "fdp refused the token: run fdp login, then open the shot again"


def _token_digest() -> str | None:
    try:
        fdp = fetch_worker.WATCH.get()
        return None if fdp is None else fdp.digest
    except Exception:  # noqa: BLE001 - no token to compare: the latch stays
        return None


def latch_auth() -> None:
    """Stop fetching until a token other than the one in use is read."""
    global _refused
    with _refused_lock:
        _refused = _token_digest() or "-"


def auth_refused() -> str | None:
    """`REFUSED_NOTE` while the token fdp refused is still the one in use, else None."""
    global _refused
    with _refused_lock:
        if _refused is None:
            return None
        digest = _token_digest()
        if digest is not None and digest != _refused:  # `fdp login` made a new one
            _refused = None
            return None
        return REFUSED_NOTE


raw.FETCH_GATE = auth_refused  # also gates the raw groups a panel fetches itself


# ---- fetching what a build lacks ----


@dataclass
class FetchReport:
    """What `fetch_features` did: each name fetched with its source, each failure
    with its cause, and why nothing could be tried (`refused`)."""

    fetched: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    refused: str | None = None

    def text(self) -> str:
        parts = [f"fetched {', '.join(f'{n} ({s})' for n, s in self.fetched.items())}"
                 if self.fetched else "fetched nothing"]
        parts += [f"{name}: {cause}" for name, cause in self.failed.items()]
        if self.refused:
            parts.append(f"fetching refused: {self.refused}")
        return "; ".join(parts)


def _learn(text: str, report: FetchReport | None = None) -> str | None:
    """Why fetching must stop, from an fdp failure's text: an auth failure latches
    (`latch_auth`), a lock timeout (`FdpBusy`) does not; None when it is neither."""
    if fetch_worker.AUTH.search(text):
        latch_auth()
        why = "fdp refused the request"
    elif "fdp is busy" in text:
        why = "fdp is busy"
    else:
        return None
    if report is not None:
        report.refused = report.refused or why
    return why


def _from_corpus(shot: int, spec, paths: Paths, allow):
    """`(arrays, missing)` of one feature from the corpus, else from its raw group:
    the raw cache's, or one fetched into it when `allow()` says fetching is on and
    `raw` has a route for the group."""
    name = spec.name
    group, _ = resolve_corpus.split_locator(spec.locator_for("corpus"))
    got, missed = resolve_corpus.resolve(shot, [name], corpus=paths.corpus)
    if name in got or group not in raw.FETCH_SPECS:
        return got, missed
    if not raw._holds_record(shot, group, paths.raw_cache):
        if not allow():
            return {}, {name: "fetching is off"}
        raw._fetch(shot, group, channels=[0], t_range=None, paths=paths)
    return resolve_corpus.resolve(shot, [name], corpus=paths.raw_cache)


def _from_fdp(shot: int, names):
    """`(arrays, missing)` through fdp, one toksearch call at a time (the lock is in
    `resolve_fdp`). A feature's `missing` cause is the exception's class and message
    here, not the class alone, so that a refused token can be told from a gap."""
    from ..features import resolve_fdp  # deferred: toksearch is only under fdp

    errors: dict[str, str] = {}

    def keep(name: str, exc: Exception) -> None:
        errors[name] = f"{type(exc).__name__}: {exc}"[:300]

    arrays, missing = resolve_fdp.resolve(shot, list(names), on_error=keep)
    return arrays, {name: errors.get(name, cause) for name, cause in missing.items()}


def _write_features(shot: int, arrays, paths: Paths) -> None:
    """Merge `arrays` into the shot's feature file, with no miss recorded. Under the
    path's writer lock (as `equilibrium.signal` does) and through a working copy of
    its own, so two builds of one shot cannot clobber each other's."""
    path = paths.features_file(shot)
    with raw._lock_for(path.resolve()):
        write_features(path, shot, arrays, {}, unique_tmp=True)


def fetch_features(shot: int, names, paths: Paths) -> FetchReport:
    """Fetch the named features of one shot and merge them into its feature file
    (`paths.features_file`).

    Each name goes through its corpus route when the raw record has a fetch route for
    its group (`pinj_total`: the eight beam nodes, summed from the raw `pinj` group),
    else through fdp (`ip`, `betan`). The archive is not used: its rows are 25 ms
    late against these (`resolve_archive`). What is on disk (the corpus, the raw cache)
    is read first; fdp calls are serialised and time out (`resolve_fdp.one_call`) and
    do not start where the server cannot fetch (`report.refused` says why: not under
    fdp, fetching off, a token fdp refused, an fdp session held too long).

    Only what was fetched is written, and no miss: a failure here is the server's
    (off, refused, timed out), never a settled "this shot has no such signal" for the
    features pipeline, whose `is_complete` would skip it for good."""
    shot = int(shot)
    names = list(dict.fromkeys(names))
    report = FetchReport()

    def allow() -> bool:
        """Whether this server can fetch now, asked only where data is missing on disk."""
        refusal = fetch_worker.refusal(fetch_worker.WATCH.get())
        why = refusal.message if refusal is not None else auth_refused()
        if why is not None:
            report.refused = report.refused or why
        return why is None

    with _fetching_lock:
        _fetching[shot] = list(names)
    arrays, causes, specs = {}, {}, {}
    try:
        for name in names:
            try:
                specs[name] = ns.by_name(name)
            except KeyError:
                causes[name] = "no such feature"
        for name, spec in specs.items():
            if "corpus" not in spec.sources:
                continue
            try:
                got, missed = _from_corpus(shot, spec, paths, allow)
            except NoDataError as error:
                got, missed = {}, {name: str(error)[:200]}
                if isinstance(error, raw.FetchDisabledError):
                    report.refused = report.refused or str(error)[:200]
                else:
                    _learn(str(error), report)
            except Exception as error:  # noqa: BLE001 - one feature must not cost the rest
                got, missed = {}, {name: type(error).__name__}
            if name in got:
                arrays[name], report.fetched[name] = got[name], "corpus"
            else:
                causes[name] = f"corpus: {missed.get(name, 'no data')}"
        pending = [n for n, spec in specs.items() if n not in arrays and "fdp" in spec.sources]
        if pending:
            try:
                got, missed = _from_fdp(shot, pending) if allow() else (
                    {}, dict.fromkeys(pending, "fetching is off"))
            except Exception as error:  # noqa: BLE001 - recorded as the cause
                got, missed = {}, dict.fromkeys(pending, f"{type(error).__name__}: {error}")
            for name in pending:
                if name in got:
                    arrays[name], report.fetched[name] = got[name], "fdp"
                else:
                    cause = missed.get(name, "no data")
                    _learn(cause, report)
                    causes[name] = f"fdp: {cause}"
        for name in [n for n, a in arrays.items() if a.y.shape[-1] < 2]:
            # `write_features` would record such a feature as a permanent miss.
            del arrays[name]
            causes[name] = f"{report.fetched.pop(name, 'fetch')}: fewer than two samples"
        report.failed = {n: c for n, c in causes.items() if n not in arrays}
        if arrays:
            _write_features(shot, arrays, paths)
    finally:
        with _fetching_lock:
            _fetching.pop(shot, None)
    return report


def note(stage: str, fetching, ahead: int = 0) -> str:
    """The activity bar's line for a 202 body: at most `NOTE_MAX` characters."""
    if stage == "queued":
        text = "Queued, next" if not ahead else f"Queued, {ahead} ahead"
    elif stage == "fetching":
        text = f"Fetching {', '.join(fetching)}" if fetching else "Fetching data"
    else:
        text = "Building the review rows"
    return text if len(text) <= NOTE_MAX else text[:NOTE_MAX - 1] + "\u2026"


# ---- the build ----


def build(event: str, shot: int, paths: Paths, *, protect: bool | None = None):
    """Build `event`'s review store for `shot` as the server does: into the data root
    for a protected shot, else into the temporary cache; a build stopped by missing
    features fetches them and builds once more. Raises what the build raised, with the
    fetch's report as `error.ondemand`; returns the store's path."""
    shot = int(shot)
    protect = protected(paths, event, shot) if protect is None else protect
    # A protected shot builds in the data root even where an older copy sits in the
    # cache: its paths do not read the cache.
    where = replace(paths, read_cache=False) if protect else overlay_paths(paths, shot)
    with _active_lock:
        _active[shot] = _active.get(shot, 0) + 1
    try:
        try:
            return review_build.build(event, shot, where)
        except MissingFeature as missing:
            report = fetch_features(shot, missing.names, where)
            try:
                if not report.fetched:
                    raise
                return review_build.build(event, shot, where)
            except Exception as error:
                error.ondemand = report
                raise
    except Exception as error:
        if isinstance(error, raw.UpstreamError):  # a panel's own fetch: fdp's answer
            _learn(str(error))
        if not protect:
            _discard_if_empty(paths, shot)
        raise
    finally:
        with _active_lock:
            _active[shot] -= 1
            if not _active[shot]:
                del _active[shot]
        if not protect:
            evict(paths, keep=(shot,))


# ---- failures ----


class BuildFailure(str):
    """A failed build as the page gets it: `short` (a plain sentence, no path or
    exception name), `detail` (the full text) and `code` (one of `CODES`). As a string
    it is the detail, so code that held the old message still reads it."""

    short: str
    detail: str
    code: str

    def __new__(cls, short: str, detail: str, code: str):
        text = super().__new__(cls, detail)
        text.short, text.detail, text.code = short, detail, code
        return text

    def body(self) -> dict:
        """The 502 body."""
        return {"error": self.short, "detail": self.detail, "code": self.code}


_GROUP = re.compile(r"has no '([a-z0-9_]+)'")


def _label(name: str) -> str:
    return f"{LABELS[name]} ({name})" if name in LABELS else name


def _refusal() -> str | None:
    """Why this server cannot fetch now (not under fdp, off, token expired or
    refused), else None."""
    try:
        refusal = fetch_worker.refusal(fetch_worker.WATCH.get())
    except Exception:  # noqa: BLE001 - a message must not fail on a bad token file
        return None
    return refusal.message if refusal is not None else auth_refused()


def _reason(text: str) -> str:
    """Why nothing was fetched, in a few words, from a refusal's text."""
    if "busy" in text:
        return "fdp is busy, try again"
    if "expired" in text:
        return "fdp token expired"
    if "wrapper" in text or "not started" in text or "Toksearch" in text:
        return "server not under fdp"
    if "refused" in text:
        return "fdp refused the request, log in again"
    return "fetching is off"


def explain(error: BaseException, event: str, shot: int) -> BuildFailure:
    """`error`, which stopped the build of `event` for `shot`, as a `BuildFailure`."""
    shot = int(shot)
    full = f"{type(error).__name__}: {error}"
    report = getattr(error, "ondemand", None)
    if report is not None:
        full = f"{full}. On-demand fetch: {report.text()}"
    refused = None if report is None else report.refused
    group = _GROUP.search(str(error))
    code = "build_failed"
    if isinstance(error, MissingFeature):
        base, code = f"No {', '.join(_label(n) for n in error.names[:3])} for {shot}", "missing_signal"
    elif isinstance(error, raw.FetchDisabledError):
        base = f"No {group.group(1) if group else 'data'} for {shot}"
        refused = refused or str(error)
    elif isinstance(error, raw.UpstreamError):
        text = str(error)
        base = f"No data for {shot}"
        if fetch_worker.AUTH.search(text):
            refused = "refused"
        elif "fdp is busy" in text:
            refused = "busy"
        elif (why := _refusal()) is not None:  # fdp could not be asked at all
            refused = why
        else:
            return BuildFailure(f"fdp did not answer for {shot}; open it again to retry",
                                full, "build_failed")
    elif isinstance(error, NoDataError):
        base, code = f"No {group.group(1) + ' ' if group else ''}data for shot {shot}", "no_data"
    elif isinstance(error, FileNotFoundError):
        base, code = f"Input data for {shot} is missing", "no_data"
    else:
        base = f"Could not build {event.split('.')[0].replace('_', ' ')} for {shot}"
    if refused:
        return BuildFailure(f"{base}: {_reason(refused)}", full, "fetch_off")
    return BuildFailure(base, full, code)
