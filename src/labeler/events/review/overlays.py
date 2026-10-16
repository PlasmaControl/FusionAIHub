"""TokEye overlays on every spectrogram row of a review store (plan section C).

    python -m labeler.events.review.overlays build --event E (--shots-file F | --list NAME) --device cuda
    python -m labeler.events.review.overlays build --requests F --device cuda   (a queue batch)
    python -m labeler.events.review.overlays status --event E --shots S [S ...]
    python -m labeler.events.review.overlays request --event E --shots S [S ...]
    python -m labeler.events.review.overlays drain
    python -m labeler.events.review.overlays compare [--shots-file F]   (the AE IoU gate)

The producer runs in SLURM GPU jobs only (`scripts/labeler/overlays.sbatch`), never on the
login node. See docs/labeler/overlays.md.

Files, under `$LABELER_ROOT/overlays/tokeye/<model sha8>/<store event>/`:

- `<shot>.npz`: `meta` (JSON: each layer's key, grid, y0, dy and n_y) and one packed bit
  array per layer, `(n_y, ceil(n / 8))` uint8 as `np.packbits(lit, axis=1)`: row j is the
  row's bin j (row 0 the lowest), column k bit `7 - k % 8` of byte `k // 8`, on that row's
  store grid at the pyramid level `level_for` picks. The AE store has one layer, `union`,
  that every AE image row shows.
- `<shot>.json`: the shot's state record: `state` (queued, computing, done or failed), a
  reason, the store's size and mtime when it was read, the model's sha256, each row's own
  state when done, the SLURM job, and times.

The states a row answers with (`row_states`):

- `done`: a layer from the pinned model, read from the store as it is now;
- `stale`: the store was rebuilt after TokEye ran, or only an older model's layer exists;
- `queued`: a request waits for a GPU job, or the job holding it is pending;
- `computing`: the GPU job holding it is running;
- `failed`: the job failed on the shot, or ended without a result;
- `not_computed`: nothing yet;
- `not_applicable`: not a spectrogram (a trace, a mode-number map, a profile).

The rows: every image row on a kHz axis without a modes legend gets the pinned U-Net
(`events/unet.py`) on its own store spectrogram, preprocessed as `events/masks.py`
prepares a waveform's (clip to its own 1st/99th percentiles, z-score, 512-column tiles
overlapping 64 and averaged), lit where either channel reaches `PROB_THRESHOLD`. The AE
store instead keeps `ae.seg.whole.build`'s 2-of-4 CO2 union on its level-8 grid (four CO2
chords through `labeler.ae.gpu`, `clean_mask`, `persist_open`, the notch), on the whole
record: no UCI-v1 refusal, no 0-6 s crop and no probability files.

Requests (`POST /api/overlays/request`, `request`) land as `overlays/queue/<store>__<shot>.req`.
`drain` claims them into `overlays/queue/batches/<stamp>/requests.txt` and submits one small
GPU job for them when none of the queue's jobs is pending or running, at most once a minute.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import fcntl
import getpass
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ...ae import full, seg
from ...ae.labels import PERSIST_FRAMES, PROB_THRESHOLD
from ...ae.seg import pseudo, whole
from ...ae.transform import STD_EPS
from ...ae.xpower.data import CROSS_ROWS, MIN_CHORDS, NOTCH
from ...config import Paths, atomic_path, store_event
from .. import masks, raw, unet
from . import build as review_build
from . import rows as store_rows

log = logging.getLogger(__name__)

#: The pinned TokEye checkpoint; its first eight hex digits name the overlay directory.
MODEL_SHA256 = unet.CHECKPOINT_SHA256
MODEL_SHA8 = MODEL_SHA256[:8]
#: The served layer's name.
LAYER = f"tokeye-{MODEL_SHA8}"
STATES = ("done", "stale", "queued", "computing", "failed", "not_computed", "not_applicable")
#: The coarsest pyramid level whose columns are at most this long is the U-Net's input.
MAX_COLUMN_MS = 1.0
#: The U-Net halves its input four times.
HEIGHT_MULTIPLE = 16
#: The AE union's key in the npz, and the level its grid is on (`whole.build`'s).
AE_KEY = "union"
AE_LEVEL = pseudo.LEVEL
#: AE tiles: frames kept per tile and context either side (`full.tiles`' margin).
AE_KEEP = 2048
AE_MARGIN = 128
#: `ae_dataset.PROB_CODE`: a probability's code `floor(p * 255)` is lit from 51 (p >= 0.2).
PROB_CODE = 51
#: The page's 80-250 kHz band in 0.977 kHz bins (TokEye bins 164-511, `pseudo.tokeye_rows`).
PAGE_BANDS = {"0-250": (0, 257), "80-250": (83, 257)}
#: SLURM job names: a list's run, and the drain's small jobs.
LIST_JOB = "overlays"
QUEUE_JOB = "overlays-queue"
#: A queue job is under the GPU rule's tiny-model branch: one batched job of at most
#: 5 minutes. The drain sends it at most QUEUE_MAX_SHOTS shots (the rest wait for the
#: next job) with these sbatch options ahead of the script's own.
QUEUE_MAX_SHOTS = 8
QUEUE_SBATCH = ("--time=00:05:00", "--cpus-per-task=4", "--mem=8G")
#: The AE gate's verdict (`compare`), beside the AE records; until it says passed, a shot
#: with a tokeye-full layer keeps showing that layer on the AE page.
GATE = "gate.json"
#: The drain submits at most once in this many seconds.
SUBMIT_EVERY_S = 60
#: How long an squeue answer is reused.
SQUEUE_TTL_S = 15
#: A claimed request with no job id after this long failed to submit; a batch older than
#: STUCK_S no longer holds a line.
SUBMIT_GRACE_S = 300
STUCK_S = 6 * 3600
REPO = Path(__file__).resolve().parents[4]
SBATCH = REPO / "scripts" / "labeler" / "overlays.sbatch"
STORE = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?\Z")
#: Every subprocess goes through here (tests replace it).
RUN = subprocess.run


# ------------------------------------------------------------------ files


def overlays_root(paths: Paths) -> Path:
    return Path(paths.root) / "overlays"


def model_dir(paths: Paths, sha8: str = MODEL_SHA8) -> Path:
    return overlays_root(paths) / "tokeye" / sha8


def layer_file(paths: Paths, event: str, shot: int) -> Path:
    return model_dir(paths) / store_event(event) / f"{int(shot)}.npz"


def state_file(paths: Paths, event: str, shot: int) -> Path:
    return model_dir(paths) / store_event(event) / f"{int(shot)}.json"


def gate_file(paths: Paths) -> Path:
    return model_dir(paths) / seg.EVENT / GATE


def gate_passed(paths: Paths) -> bool:
    """Whether `compare` last passed the AE gate for this model over every tokeye-full
    shot. A missing, failed or unreadable gate is not passed."""
    gate = _read_json(gate_file(paths)) or {}
    return gate.get("passed") is True and gate.get("model_sha256") == MODEL_SHA256


def queue_dir(paths: Paths) -> Path:
    return overlays_root(paths) / "queue"


def request_file(paths: Paths, event: str, shot: int) -> Path:
    return queue_dir(paths) / f"{store_event(event)}__{int(shot)}.req"


def batches_dir(paths: Paths) -> Path:
    return queue_dir(paths) / "batches"


def pending(paths: Paths) -> list[Path]:
    """The request files waiting for the drain, oldest first."""
    found = []
    for path in queue_dir(paths).glob("*.req"):
        with contextlib.suppress(OSError):
            found.append((path.stat().st_mtime_ns, path.name, path))
    return [path for *_, path in sorted(found)]


def applicable(row: dict) -> bool:
    """An image row on a kHz axis that is not a modes map: a spectrogram."""
    return row.get("kind") == "image" and row.get("y_units") == "kHz" and "modes" not in row


def level_for(dt_ms: float) -> int:
    """The coarsest pyramid level whose columns are at most MAX_COLUMN_MS long."""
    fit = [level for level in store_rows.LEVELS if dt_ms * level <= MAX_COLUMN_MS + 1e-9]
    return max(fit) if fit else 1


def _stamp(path: Path) -> dict | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _read_json(path: Path) -> dict | None:
    """A JSON object from `path`; None when missing, unreadable or not an object."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        log.warning("unreadable overlay record %s: %s", path, error)
        return None
    if not isinstance(value, dict):
        log.warning("overlay record %s is not a JSON object", path)
        return None
    return value


def _seconds(value, default: float) -> float:
    """A record's epoch time, or `default` when it is missing or not a number."""
    try:
        found = float(value)
    except (TypeError, ValueError):
        return default
    return found if np.isfinite(found) else default


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_record(paths: Paths, store: str, shot: int, state: str, reason: str,
                 **extra) -> dict:
    """Replace the shot's state record; `since` is when it entered `state`."""
    record = {
        "store_event": store, "shot": int(shot), "state": state, "reason": reason,
        "model_sha256": MODEL_SHA256, "updated": _now_iso(), "since": time.time(),
        **extra,
    }
    _write_json(state_file(paths, store, shot), record)
    return record


# ------------------------------------------------------------------ SLURM


_SQUEUE: dict = {"at": 0.0, "jobs": None}
_SQUEUE_LOCK = threading.Lock()


def slurm_jobs(*, fresh: bool = False) -> tuple[float, dict | None]:
    """`(when, {job id: (name, state)})` for this user's SLURM jobs, reused for
    SQUEUE_TTL_S; `None` when squeue does not answer."""
    with _SQUEUE_LOCK:
        if not fresh and time.time() - _SQUEUE["at"] < SQUEUE_TTL_S:
            return _SQUEUE["at"], _SQUEUE["jobs"]
    at = time.time()
    try:
        out = RUN(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j|%T"],
                  capture_output=True, text=True, timeout=30, check=True)
        jobs = {}
        for line in out.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 3:
                jobs[parts[0]] = (parts[1], parts[2])
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        log.warning("squeue did not answer: %s", error)
        jobs = None
    with _SQUEUE_LOCK:
        _SQUEUE.update(at=at, jobs=jobs)
    return at, jobs


def forget_jobs() -> None:
    with _SQUEUE_LOCK:
        _SQUEUE.update(at=0.0, jobs=None)


# ------------------------------------------------------------------ states


def _state(state: str, reason: str, **extra) -> dict:
    return {"state": state, "reason": reason, **extra}


NOT_APPLICABLE = {"state": "not_applicable", "reason": "not a spectrogram"}


def _open_batches(paths: Paths) -> list[Path]:
    """The batches whose job may still run their shots. A batch is closed when it
    has finished.json, is older than STUCK_S, was never submitted (no job.json past
    SUBMIT_GRACE_S), or its job has left squeue (an answer taken over 5 s after
    job.json was written): a job killed at its time limit writes no finished.json,
    and its shots must not hold places in the line."""
    found = []
    now, answer = time.time(), None
    for batch in batches_dir(paths).glob("*/requests.txt"):
        with contextlib.suppress(OSError):
            if (batch.parent / "finished.json").exists():
                continue
            age = now - batch.stat().st_mtime
            if age >= STUCK_S:
                continue
            submitted = batch.parent / "job.json"
            if not submitted.exists():
                if age <= SUBMIT_GRACE_S:  # sbatch may still be answering
                    found.append(batch)
                continue
            job = str((_read_json(submitted) or {}).get("job", ""))
            if answer is None:
                answer = slurm_jobs()
            at, jobs = answer
            if jobs is not None and at > submitted.stat().st_mtime + 5 and job not in jobs:
                continue
            found.append(batch)
    return sorted(found)


def line(paths: Paths) -> list[tuple[str, int]]:
    """Every shot waiting for TokEye, first in line first: the open batches' shots
    still queued, then the requests not yet claimed."""
    found = []
    for batch in _open_batches(paths):
        for store, shot in read_requests(batch):
            record = _read_json(state_file(paths, store, shot)) or {}
            if record.get("state") == "queued":
                found.append((store, shot))
    for path in pending(paths):
        store, _, shot = path.stem.rpartition("__")
        if STORE.match(store) and shot.isdigit():
            found.append((store, int(shot)))
    return found


def _position(paths: Paths, store: str, shot: int) -> int | None:
    try:
        return line(paths).index((store, int(shot))) + 1
    except ValueError:
        return None


def _older_model(paths: Paths, store: str, shot: int) -> bool:
    root = overlays_root(paths) / "tokeye"
    with contextlib.suppress(OSError):
        for other in root.iterdir():
            if other.name != MODEL_SHA8 and (other / store / f"{int(shot)}.json").is_file():
                return True
    return False


def _waiting(paths: Paths, store: str, shot: int, record: dict) -> dict:
    """A queued or computing record as it stands, or failed when its job is gone."""
    now = time.time()
    job, since = record.get("job"), _seconds(record.get("since"), now)
    if not job:
        if record["state"] == "computing":  # a run outside SLURM: no job to ask about
            return _state("computing", "TokEye is running",
                          elapsed_s=round(now - since, 1))
        if now - since > SUBMIT_GRACE_S:
            return _state("failed", "the GPU job for this request was never submitted")
        return _state("queued", "submitting a GPU job", elapsed_s=round(now - since, 1),
                      queue_position=_position(paths, store, shot))
    at, jobs = slurm_jobs()
    if jobs is not None and at > since + 5 and str(job) not in jobs:
        return _state("failed", f"the GPU job {job} ended without a result: see "
                                f"runs/slurm/{job}.out")
    if record["state"] == "computing":
        return _state("computing", f"GPU job {job} is running",
                      elapsed_s=round(now - since, 1))
    requested = _seconds(record.get("requested_at"), since)
    running = jobs is not None and jobs.get(str(job), ("", ""))[1] == "RUNNING"
    reason = (f"GPU job {job} is running; this shot is waiting in it" if running
              else f"GPU job {job} is pending")
    return _state("queued", reason, elapsed_s=round(now - requested, 1),
                  queue_position=_position(paths, store, shot))


def shot_state(paths: Paths, event: str, shot: int) -> tuple[dict, dict]:
    """`(state, record)`: the shot's own state before rows (`row_states` splits a done
    record by row), and its record (`{}` without one)."""
    store, shot = store_event(event), int(shot)
    request = request_file(paths, store, shot)
    record = _read_json(state_file(paths, store, shot)) or {}
    if request.is_file():
        info = _read_json(request) or {}
        requested = _seconds(info.get("requested_at"), time.time())
        return _state("queued", "waiting for a GPU job",
                      elapsed_s=round(time.time() - requested, 1),
                      queue_position=_position(paths, store, shot)), record
    if record.get("state") in ("queued", "computing"):
        return _waiting(paths, store, shot, record), record
    if record.get("state") == "failed":
        return _state("failed", record.get("reason") or "TokEye failed",
                      updated=record.get("updated")), record
    if record.get("state") == "done":
        current = record.get("store_stamp") == _stamp(paths.spectrogram_file(store, shot))
        if not current:
            return _state("stale", "the store was rebuilt after TokEye ran",
                          updated=record.get("updated")), record
        return _state("done", f"TokEye from model {MODEL_SHA8}", layer=LAYER,
                      updated=record.get("updated")), record
    if _older_model(paths, store, shot):
        return _state("stale", "TokEye from an older model"), record
    return _state("not_computed", "No TokEye yet"), record


def _legacy(paths: Paths, store: str, shot: int) -> bool:
    return store == seg.EVENT and whole.layer_file(paths, shot).is_file()


def _shows_legacy(paths: Paths, store: str, shot: int, state: dict) -> bool:
    """Whether the AE page shows the tokeye-full layer: the shot has one, and either the
    AE gate has not passed for this model, or this model's layer is not done and no
    job is making it."""
    if not _legacy(paths, store, shot):
        return False
    if not gate_passed(paths):
        return True
    return state["state"] not in ("done", "queued", "computing")


LEGACY = "TokEye from the tokeye-full layer (0-6 s)"


def row_states(paths: Paths, event: str, shot: int, rows: list[dict]) -> dict[str, dict]:
    """Every row's overlay state, by name (`rows` as `rows.meta` describes them).

    Cheap enough for every `/api/shot`: one small JSON read and a few stats; squeue
    only for a shot in a job, its answer reused for SQUEUE_TTL_S."""
    names = [row["name"] for row in rows if applicable(row)]
    found = {row["name"]: dict(NOT_APPLICABLE) for row in rows if not applicable(row)}
    if not names:
        return found
    store = store_event(event)
    state, record = shot_state(paths, store, shot)
    legacy = _shows_legacy(paths, store, shot, state)
    if legacy:
        state = _state("done", LEGACY, layer=whole.LAYER)
    own_rows = _dict(record.get("rows"))
    for name in names:
        if state["state"] == "done" and not legacy:
            own = own_rows.get(name)
            if not isinstance(own, dict):
                found[name] = _state("not_computed", "No TokEye yet for this row")
            elif own.get("state") != "done":
                found[name] = _state("failed", own.get("reason") or "TokEye failed",
                                     updated=record.get("updated"))
            else:
                found[name] = dict(state)
        else:
            found[name] = dict(state)
    return found


# ------------------------------------------------------------------ layers


def _layer(shot: int, entry: dict, bits: np.ndarray, *, name: str = LAYER) -> dict:
    """One layer in the page's TokEye encoding (`whole.view`'s keys, plus y0 and dy)."""
    return {
        "shot": int(shot), "layer": name,
        "grid": {"t0_ms": float(entry["t0_ms"]), "dt_ms": float(entry["dt_ms"]),
                 "n": int(entry["n"])},
        "y0_khz": float(entry["y0"]), "dy_khz": float(entry["dy"]),
        "y0": float(entry["y0"]), "dy": float(entry["dy"]),
        "n_y": int(bits.shape[0]),
        "bits": base64.b64encode(np.ascontiguousarray(bits).tobytes()).decode(),
    }


def _read_layers(paths: Paths, store: str, shot: int, keys) -> dict[str, dict]:
    path = layer_file(paths, store, shot)
    out = {}
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        entries = {entry["key"]: entry for entry in meta["layers"]}
        for key in set(keys):
            if key in entries:
                out[key] = _layer(shot, entries[key], z[key])
    return out


def layers(paths: Paths, event: str, shot: int, rows: list[dict]) -> dict[str, dict]:
    """`{row name: layer}` for every row whose state is done with this model."""
    store = store_event(event)
    states = row_states(paths, store, shot, rows)
    done = [name for name, s in states.items()
            if s["state"] == "done" and s.get("layer") == LAYER]
    if not done:
        return {}
    record = _read_json(state_file(paths, store, shot)) or {}
    own_rows = _dict(record.get("rows"))
    keys = {name: _dict(own_rows.get(name)).get("key") for name in done}
    keys = {name: key for name, key in keys.items() if isinstance(key, str)}
    try:
        found = _read_layers(paths, store, shot, keys.values())
    except (OSError, ValueError, KeyError) as error:
        log.warning("overlay layers of %s %s unreadable: %s", store, shot, error)
        return {}
    return {name: {**found[key], "row": name}
            for name, key in keys.items() if key in found}


def ae_view(paths: Paths, shot: int) -> dict | None:
    """The AE page's one TokEye layer: this model's whole-record union when it is done
    and current and either the AE gate passed or the shot has no tokeye-full layer;
    else the tokeye-full layer (`whole.view`); else None."""
    state, record = shot_state(paths, seg.EVENT, shot)
    if state["state"] == "done" and record.get("rows") and \
            not _shows_legacy(paths, seg.EVENT, shot, state):
        with contextlib.suppress(OSError, ValueError, KeyError):
            found = _read_layers(paths, seg.EVENT, shot, [AE_KEY])
            if AE_KEY in found:
                return found[AE_KEY]
    return whole.view(paths, shot)


# ------------------------------------------------------------------ requests


def store_rows_meta(paths: Paths, event: str, shot: int) -> list[dict]:
    """The rows the page shows for the shot's store (FileNotFoundError without one)."""
    store = store_event(event)
    path = paths.spectrogram_file(store, shot)
    if not path.is_file():
        raise FileNotFoundError(f"shot {int(shot)} has no {store} review store yet")
    hide = review_build.HIDDEN.get(store, frozenset())
    return store_rows.meta(path, hide)["rows"]


def request(paths: Paths, event: str, shot: int, rows: list[dict] | None = None) -> dict:
    """Queue the shot for TokEye unless it is done or already waiting:
    `{"state", "queue_position"}` as `POST /api/overlays/request` answers."""
    store, shot = store_event(event), int(shot)
    if not STORE.match(store):
        raise ValueError(f"not an event: {store!r}")
    rows = store_rows_meta(paths, store, shot) if rows is None else rows
    live = [s for s in row_states(paths, store, shot, rows).values()
            if s["state"] != "not_applicable"]
    if not live:
        return {"state": "not_applicable", "queue_position": None}
    if all(s["state"] == "done" for s in live):
        return {"state": "done", "queue_position": None}
    for s in live:
        if s["state"] in ("queued", "computing"):
            return {"state": s["state"], "queue_position": s.get("queue_position")}
    _write_json(request_file(paths, store, shot), {
        "event": event, "store_event": store, "shot": shot, "requested_at": time.time(),
    })
    return {"state": "queued", "queue_position": _position(paths, store, shot)}


def read_requests(path: Path) -> list[tuple[str, int]]:
    """A batch's `<store event> <shot>` lines, validated."""
    found = []
    for line_ in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line_.split()
        if len(parts) == 2 and STORE.match(parts[0]) and parts[1].isdigit():
            found.append((parts[0], int(parts[1])))
        elif line_.strip():
            raise ValueError(f"{path}: not a request line: {line_[:80]!r}")
    return found


def drain(paths: Paths) -> dict:
    """Claim up to QUEUE_MAX_SHOTS waiting requests, oldest first, and submit one GPU
    job for them, when none of the queue's jobs is pending or running and none was
    submitted in SUBMIT_EVERY_S. The job's command line carries only this module's
    constants and paths: the shots are validated ints in the batch file."""
    if not pending(paths):
        return {"submitted": None, "reason": "nothing is queued"}
    qdir = queue_dir(paths)
    qdir.mkdir(parents=True, exist_ok=True)
    with open(qdir / ".drain.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"submitted": None, "reason": "another drain is running"}
        last = qdir / "last_submit"
        with contextlib.suppress(OSError):
            wait = SUBMIT_EVERY_S - (time.time() - last.stat().st_mtime)
            if wait > 0:
                return {"submitted": None, "reason": f"next submission in {wait:.0f} s"}
        _, jobs = slurm_jobs(fresh=True)
        if jobs is None:
            return {"submitted": None, "reason": "squeue did not answer"}
        for job, (name, state) in sorted(jobs.items()):
            if name == QUEUE_JOB:
                return {"submitted": None, "reason": f"job {job} is {state.lower()}"}
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        batch = batches_dir(paths) / stamp
        batch.mkdir(parents=True)
        claimed = []
        for path in pending(paths):
            if len(claimed) >= QUEUE_MAX_SHOTS:
                break  # the rest wait for the next job
            info = _read_json(path) or {}
            store, shot = str(info.get("store_event", "")), info.get("shot")
            target = batch / path.name
            try:
                path.rename(target)
            except OSError:
                continue
            if not STORE.match(store) or type(shot) is not int or shot <= 0:
                log.warning("dropped a malformed request %s", path.name)
                with contextlib.suppress(OSError):
                    target.unlink()
                continue
            claimed.append((store, shot, _seconds(info.get("requested_at"), time.time())))
        if not claimed:
            with contextlib.suppress(OSError):
                batch.rmdir()
            return {"submitted": None, "reason": "nothing is queued"}
        requests = batch / "requests.txt"
        requests.write_text("".join(f"{store} {shot}\n" for store, shot, _ in claimed),
                            encoding="utf-8")
        for store, shot, requested in claimed:
            write_record(paths, store, shot, "queued", "submitting a GPU job",
                         job=None, batch=stamp, requested_at=requested)
        last.touch()
        command = ["sbatch", "--parsable", f"--job-name={QUEUE_JOB}", *QUEUE_SBATCH,
                   f"--export=ALL,REPO={REPO}", str(SBATCH), "--requests", str(requests)]
        try:
            out = RUN(command, capture_output=True, text=True, timeout=60, check=True)
            job = out.stdout.strip().split(";")[0]
            if not job.isdigit():
                raise ValueError(f"sbatch answered {out.stdout.strip()[:80]!r}")
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            stderr = getattr(error, "stderr", None) or ""
            reason = f"sbatch failed: {error} {stderr}".strip()
            for store, shot, requested in claimed:
                write_record(paths, store, shot, "failed", reason, job=None, batch=stamp,
                             requested_at=requested)
            _write_json(batch / "finished.json", {"error": reason, "at": _now_iso()})
            return {"submitted": None, "reason": reason}
        _write_json(batch / "job.json", {"job": job, "submitted": _now_iso()})
        for store, shot, requested in claimed:
            write_record(paths, store, shot, "queued", f"GPU job {job} is pending",
                         job=job, batch=stamp, requested_at=requested)
        forget_jobs()
        return {"submitted": job, "shots": len(claimed), "batch": str(batch)}


# ------------------------------------------------------------------ reading (workers)


def read_store(paths: Paths, store: str, shot: int) -> dict:
    """One shot's inputs, in a loader worker: every applicable row's uint8 values at
    its level, or, for the AE store, the four CO2 chords up to the resample."""
    path = paths.spectrogram_file(store, shot)
    stamp = _stamp(path)
    if stamp is None:
        raise FileNotFoundError(f"shot {int(shot)} has no {store} review store")
    hide = review_build.HIDDEN.get(store, frozenset())
    with h5py.File(path, "r") as f:
        t0, dt, n = float(f.attrs["t0_ms"]), float(f.attrs["dt_ms"]), int(f.attrs["n"])
        names = [name for name in json.loads(f.attrs["rows"]) if name not in hide]
        metas = {name: json.loads(f["rows"][name].attrs["meta"]) for name in names}
        live = [name for name in names if applicable(metas[name])]
        item = {"store": store, "shot": int(shot), "stamp": stamp, "names": live,
                "grid": [t0, dt, n]}
        if store == seg.EVENT:
            if not live:
                return {**item, "rows": []}
            cross = metas.get(CROSS_ROWS[0]) or metas[live[0]]
            level = AE_LEVEL
            item["ae"] = {
                "grid": [t0, dt * level, -(-n // level)], "y0": float(cross["y0"]),
                "dy": float(cross["dy"]), "n_y": int(cross["n_y"]),
            }
            return item
        rows = []
        for name in live:
            level = level_for(dt)
            values = f["rows"][name][str(level)][...]
            rows.append({
                "name": name, "values": np.ascontiguousarray(values, dtype=np.uint8),
                "grid": [t0, dt * level, int(values.shape[-1])], "level": level,
                "y0": float(metas[name]["y0"]), "dy": float(metas[name]["dy"]),
            })
        item["rows"] = rows
    return item


def read_item(paths: Paths, store: str, shot: int) -> dict:
    """`read_store`, plus the AE CO2 read; an error comes back as `{"error"}`."""
    try:
        item = read_store(paths, store, shot)
        if "ae" in item:
            from ...ae import gpu

            grid = item["ae"]["grid"]
            item["co2"] = gpu.read_co2(shot, paths=paths, end_ms=grid[0] + grid[1] * grid[2])
        return item
    except Exception as error:  # noqa: BLE001 - one shot's failure is its record's
        return {"store": store, "shot": int(shot),
                "error": f"{type(error).__name__}: {error}"}


class Items(torch.utils.data.Dataset):
    """`(store event, shot)` pairs read by DataLoader workers (`read_item`)."""

    def __init__(self, paths: Paths, items: list[tuple[str, int]]):
        self.paths, self.items = paths, list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        store, shot = self.items[i]
        return read_item(self.paths, store, shot)


# ------------------------------------------------------------------ the GPU side


class Timers:
    """Seconds per stage; `cuda` syncs at a stage's end so the GPU's time is its own."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.device = device
        self.total: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, stage: str, *, sync: bool = True):
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda and sync:
                torch.cuda.synchronize(self.device)
            self.total[stage] = self.total.get(stage, 0.0) + time.perf_counter() - started


def percentiles_u8(codes: torch.Tensor, q=(1.0, 99.0)) -> tuple[float, float]:
    """`np.percentile(codes, q)` (numpy's linear method) of uint8 codes, from a 256-bin
    count: the k-th smallest code is the first whose cumulative count exceeds k."""
    counts = torch.bincount(codes.reshape(-1).to(torch.int64), minlength=256)
    cumulative = torch.cumsum(counts, 0).cpu().numpy()
    n = int(cumulative[-1])
    if n == 0:
        raise ValueError("percentiles of an empty array")

    def kth(k: int) -> float:
        return float(np.searchsorted(cumulative, k, side="right"))

    out = []
    for v in (n - 1) * (np.asarray(q, dtype=np.float64) / 100):
        lo = int(np.floor(v))
        g = float(v - lo)
        a = kth(lo)
        b = kth(min(lo + 1, n - 1))
        out.append(b - (b - a) * (1 - g) if g >= 0.5 else a + (b - a) * g)
    return out[0], out[1]


def prepare_row(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A store row's uint8 `(n_y, n)` as the U-Net's input `(H, n)` float32 (H the next
    multiple of 16), and its columns with data: clipped to its own 1st and 99th
    percentiles and z-scored over the columns with data (`masks.prep`'s order), the
    columns without data at the clip floor, the added rows at zero."""
    n_y, n = values.shape
    valid = (values > 0).any(0)
    height = -(-n_y // HEIGHT_MULTIPLE) * HEIGHT_MULTIPLE
    out = torch.zeros((height, n), dtype=torch.float32, device=values.device)
    if not bool(valid.any()):
        return out, valid
    inside = values[:, valid]
    lo, hi = percentiles_u8(inside)
    x = values.to(torch.float64).clamp_(lo, hi)
    mean = x[:, valid].mean()
    std = x[:, valid].std(unbiased=False)
    z = ((x - mean) / (std + STD_EPS)).to(torch.float32)
    z[:, ~valid] = float((lo - mean) / (std + STD_EPS))
    out[:n_y] = z
    return out, valid


def row_tiles(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(H, n)` -> `(n_tiles, H, TILE)` tiles every STRIDE columns, the last zero-padded
    (`masks.tile`), and the tiles' column indices `(n_tiles * TILE,)` for the stitch
    (`masks.stitch` averages the overlaps)."""
    n = int(x.shape[1])
    count = masks.n_tiles(n)
    width = (count - 1) * masks.STRIDE + masks.TILE
    padded = F.pad(x, (0, width - n))
    tiles = padded.unfold(1, masks.TILE, masks.STRIDE).permute(1, 0, 2).contiguous()
    starts = torch.arange(count, device=x.device) * masks.STRIDE
    columns = (starts[:, None] + torch.arange(masks.TILE, device=x.device)).reshape(-1)
    return tiles, columns


def ae_cuts(n: int) -> list[tuple[int, int, int, int]]:
    """`(lo, hi, keep_lo, keep_hi)` per AE tile: keep runs of AE_KEEP frames, each tile
    AE_KEEP + 2 AE_MARGIN frames wide, slid inside the record at the ends so all are
    one width (one tile of the whole record when it is shorter)."""
    width = AE_KEEP + 2 * AE_MARGIN
    if n <= width:
        return [(0, n, 0, n)]
    cuts = []
    for keep_lo in range(0, n, AE_KEEP):
        keep_hi = min(n, keep_lo + AE_KEEP)
        lo = max(0, min(keep_lo - AE_MARGIN, n - width))
        cuts.append((lo, lo + width, keep_lo, keep_hi))
    return cuts


def persist_open(mask: torch.Tensor, frames: int = PERSIST_FRAMES) -> torch.Tensor:
    """`labels.persist_open` on a device: a pixel survives only inside a run of at
    least `frames` lit frames of its own bin (time last)."""
    if frames <= 1:
        return mask.clone()
    shape, n = mask.shape, mask.shape[-1]
    if n < frames:
        return torch.zeros_like(mask)
    x = mask.reshape(-1, 1, n).to(torch.float32)
    whole_run = -F.max_pool1d(-x, frames, stride=1)  # 1 where frames [w, w + frames) are lit
    grown = F.max_pool1d(F.pad(whole_run, (frames - 1, frames - 1)), frames, stride=1)
    return (grown > 0.5).reshape(shape)


def notch(clean: torch.Tensor, threshold: float = NOTCH) -> torch.Tensor:
    """`apply_notch(clean, notch_bins(bin_active_fraction(clean), threshold))`."""
    share = clean.sum(-1, dtype=torch.int64).to(torch.float64) / clean.shape[-1]
    return clean & ~(share > threshold)[..., None]


def union_on_grid(clean: torch.Tensor, t_ms: np.ndarray, grid: store_rows.Grid,
                  n_y: int) -> torch.Tensor:
    """`whole.build`: `pseudo.tokeye_rows` (2 of 4 chords, onto the page's 257 bins)
    then `pseudo.pool_columns` (OR within each grid column), first `n_y` bins."""
    lit = clean.sum(0) >= MIN_CHORDS
    padded = torch.zeros((514, lit.shape[1]), dtype=torch.bool, device=lit.device)
    padded[2:] = lit
    page = padded.reshape(257, 2, -1).any(1)
    t = np.asarray(t_ms, dtype=np.float64)
    cols = np.floor((t - grid.t0_ms) / grid.dt_ms).astype(np.int64)
    keep = np.flatnonzero((cols >= 0) & (cols < grid.n))
    out = torch.zeros((257, grid.n), dtype=torch.int32, device=lit.device)
    if keep.size:
        index = torch.from_numpy(cols[keep]).to(lit.device)
        chosen = torch.from_numpy(keep).to(lit.device)
        out.index_add_(1, index, page[:, chosen].to(torch.int32))
    return (out > 0)[:n_y]


class Sink:
    """Where one row's (or one AE shot's) tiles come back; `done` runs once all have,
    `fail` when a forward pass over its tiles fails."""

    def __init__(self, count: int, put, done, fail=None):
        self.remaining, self.put_fn, self.done_fn = count, put, done
        self.fail_fn = fail

    def fail(self, error: Exception) -> None:
        if self.fail_fn is None:
            raise error
        self.fail_fn(error)

    def put(self, first: int, probs: torch.Tensor) -> None:
        self.put_fn(first, probs)
        self.remaining -= len(probs)
        if self.remaining == 0:
            self.done_fn()


class Runner:
    """U-Net forward passes over tiles queued from many rows and shots, one tile shape at
    a time, each pass about `batch` 512 x 512 tiles' worth of pixels; fp16 autocast on
    CUDA; out of memory, the pass is split in halves."""

    def __init__(self, model, device: torch.device, batch: int, amp: bool, timers: Timers):
        self.model, self.device, self.batch = model, device, max(1, int(batch))
        self.amp = bool(amp) and device.type == "cuda"
        self.timers = timers
        self.queues: dict[tuple[int, int], list] = {}
        self.passes = 0
        self.tiles = 0

    def capacity(self, shape: tuple[int, int]) -> int:
        return max(1, self.batch * masks.TILE * masks.TILE // (shape[0] * shape[1]))

    def submit(self, tiles: torch.Tensor, sink: Sink) -> None:
        shape = (int(tiles.shape[-2]), int(tiles.shape[-1]))
        queue = self.queues.setdefault(shape, [])
        queue.append([tiles, sink, 0])
        while sum(len(entry[0]) for entry in queue) >= self.capacity(shape):
            self._pass(shape)

    def flush(self) -> None:
        for shape in list(self.queues):
            while self.queues[shape]:
                self._pass(shape)

    def _pass(self, shape: tuple[int, int]) -> None:
        queue, room, taken = self.queues[shape], self.capacity(shape), []
        while queue and room:
            tiles, sink, first = queue[0]
            take = min(room, len(tiles))
            taken.append((tiles[:take], sink, first))
            if take == len(tiles):
                queue.pop(0)
            else:
                queue[0] = [tiles[take:], sink, first + take]
            room -= take
        self._run(taken)

    def _run(self, taken: list) -> None:
        """One forward pass over `taken`'s tiles. When it fails, each sink's tiles are
        retried alone, and a sink whose own pass fails fails its shot; the run goes on."""
        try:
            x = torch.cat([tiles for tiles, _, _ in taken]).unsqueeze(1)
            with self.timers("network"):
                probs = self._forward(x)
        except Exception as error:  # noqa: BLE001 - a bad pass fails its shots, not the run
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if len(taken) == 1:
                taken[0][1].fail(error)
            else:
                for one in taken:
                    self._run([one])
            return
        self.passes += 1
        self.tiles += len(x)
        offset = 0
        with self.timers("post"):
            for tiles, sink, first in taken:
                sink.put(first, probs[offset:offset + len(tiles)])
                offset += len(tiles)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            with torch.autocast("cuda", dtype=torch.float16) if self.amp \
                    else contextlib.nullcontext():
                return unet.probabilities(self.model, x).float()
        except torch.cuda.OutOfMemoryError:
            if len(x) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(x) // 2
            return torch.cat([self._forward(x[:half]), self._forward(x[half:])])


@dataclass
class Shot:
    """One shot's layers as they come back, written once all have."""

    store: str
    shot: int
    stamp: dict
    names: list
    layers: dict
    expected: int
    started: float


class Producer:
    """The build loop: items from the loader in, layers and records out."""

    def __init__(self, paths: Paths, model, device: torch.device, *, batch: int, amp: bool,
                 job: str | None):
        self.paths, self.device, self.job = paths, device, job
        self.timers = Timers(device)
        self.runner = Runner(model, device, batch, amp, self.timers)
        self.counts = {"done": 0, "failed": 0, "not_applicable": 0}
        self.by_event: dict[str, dict[str, int]] = {}
        self.ready: list[Shot] = []

    def count(self, store: str, outcome: str) -> None:
        self.counts[outcome] += 1
        tally = self.by_event.setdefault(store, {})
        tally[outcome] = tally.get(outcome, 0) + 1

    def fail(self, store: str, shot: int, reason: str) -> None:
        write_record(self.paths, store, shot, "failed", reason, job=self.job)
        self.count(store, "failed")
        print(f"shot {store} {shot}: failed: {reason}", flush=True)

    def take(self, item: dict) -> None:
        """Prepare one shot's tiles on the GPU and queue them; write whatever is ready."""
        store, shot = item["store"], int(item["shot"])
        if "error" in item:
            self.fail(store, shot, item["error"])
            return
        if not item["names"]:
            self.count(store, "not_applicable")
            return
        write_record(self.paths, store, shot, "computing", f"GPU job {self.job} is running",
                     job=self.job)
        keys = 1 if "ae" in item else len(item["rows"])
        entry = Shot(store, shot, item["stamp"], list(item["names"]), {}, keys, time.time())
        try:
            with self.timers("prep"):
                work = ([self._ae(item, entry)] if "ae" in item
                        else [self._row(row, entry) for row in item["rows"]])
        except Exception as error:  # noqa: BLE001 - the shot fails, the run goes on
            self._failed(entry, error)
            return
        for tiles, sink in work:
            try:
                self.runner.submit(tiles, sink)
            except Exception as error:  # noqa: BLE001 - the shot fails, the run goes on
                self._failed(entry, error)
        self.write_ready()

    def finish(self) -> None:
        self.runner.flush()
        self.write_ready()

    def _failed(self, entry: Shot, error: Exception) -> None:
        if entry.expected >= 0:
            entry.expected = -1  # its tiles still in the queue come back to nothing
            self.fail(entry.store, entry.shot, f"{type(error).__name__}: {error}")

    def _sink(self, entry: Shot, count: int, put, done) -> Sink:
        """A sink whose failures fail its shot, not the run."""

        def guarded(fn):
            def call(*args):
                if entry.expected < 0:
                    return
                try:
                    fn(*args)
                except Exception as error:  # noqa: BLE001
                    self._failed(entry, error)
            return call

        def fail(error: Exception) -> None:
            self._failed(entry, error)

        return Sink(count, guarded(put), guarded(done), fail)

    def _row(self, row: dict, entry: Shot) -> tuple[torch.Tensor, Sink]:
        values = torch.as_tensor(row["values"]).to(self.device, non_blocking=True)
        n_y, n = int(values.shape[0]), int(values.shape[1])
        x, valid = prepare_row(values)
        tiles, columns = row_tiles(x)
        total = torch.zeros((2, x.shape[0], len(tiles) * masks.STRIDE + masks.OVERLAP),
                            dtype=torch.float32, device=self.device)
        count = torch.bincount(columns).to(torch.float32)
        width = masks.TILE

        def put(first: int, probs: torch.Tensor) -> None:
            index = columns[first * width:(first + len(probs)) * width]
            flat = probs.permute(1, 2, 0, 3).reshape(2, probs.shape[2], -1)
            total.index_add_(2, index, flat)

        def done() -> None:
            mean = total[..., :n] / count[:n]
            lit = ((mean[0] >= PROB_THRESHOLD) | (mean[1] >= PROB_THRESHOLD))[:n_y]
            lit &= valid[None, :]
            grid = row["grid"]
            self._layer(entry, row["name"], lit, {
                "t0_ms": float(grid[0]), "dt_ms": float(grid[1]), "n": int(grid[2]),
                "y0": row["y0"], "dy": row["dy"], "level": int(row["level"]),
            })

        return tiles, self._sink(entry, len(tiles), put, done)

    def _ae(self, item: dict, entry: Shot) -> tuple[torch.Tensor, Sink]:
        from ...ae import gpu

        signals, t0_ms, fs_hz = gpu.input_signals(item["co2"], device=self.device)
        spec = gpu.standardise(gpu.spectrogram(signals))  # (4, 512, frames)
        t_ms = full.frame_times(int(signals.shape[1]), t0_ms, fs_hz)
        del signals
        chords, bins, n = (int(v) for v in spec.shape)
        if len(t_ms) != n:
            raise ValueError(f"{n} frames, {len(t_ms)} frame times")
        cuts = ae_cuts(n)
        width = cuts[0][1] - cuts[0][0]
        padded = -(-width // HEIGHT_MULTIPLE) * HEIGHT_MULTIPLE
        tiles = torch.stack([
            F.pad(spec[c, :, lo:hi], (0, padded - width)) for c in range(chords)
            for lo, hi, _, _ in cuts
        ])
        del spec
        clean = torch.zeros((chords, bins, n), dtype=torch.bool, device=self.device)
        info = item["ae"]

        def put(first: int, probs: torch.Tensor) -> None:
            codes = (probs.clamp(0, 1) * 255).floor()
            mode = (codes[:, 0] >= PROB_CODE) & ~(codes[:, 1] >= PROB_CODE)
            for i in range(len(probs)):
                c, k = divmod(first + i, len(cuts))
                lo, _, keep_lo, keep_hi = cuts[k]
                clean[c, :, keep_lo:keep_hi] = mode[i, :, keep_lo - lo:keep_hi - lo]

        def done() -> None:
            kept = notch(persist_open(clean))
            grid = store_rows.Grid(*info["grid"])
            lit = union_on_grid(kept, t_ms, grid, int(info["n_y"]))
            self._layer(entry, AE_KEY, lit, {
                "t0_ms": float(grid.t0_ms), "dt_ms": float(grid.dt_ms), "n": int(grid.n),
                "y0": float(info["y0"]), "dy": float(info["dy"]), "level": AE_LEVEL,
                "t_range_ms": [float(t_ms[0]), float(t_ms[-1])],
            })

        return tiles, self._sink(entry, len(tiles), put, done)

    def _layer(self, entry: Shot, key: str, lit: torch.Tensor, info: dict) -> None:
        entry.layers[key] = (np.packbits(lit.cpu().numpy(), axis=1), info)
        if len(entry.layers) == entry.expected:
            self.ready.append(entry)

    def write_ready(self) -> None:
        while self.ready:
            entry = self.ready.pop(0)
            try:
                with self.timers("write", sync=False):
                    self.write(entry)
            except Exception as error:  # noqa: BLE001
                self._failed(entry, error)

    def write(self, entry: Shot) -> None:
        keyed = {}
        meta = {"store_event": entry.store, "shot": entry.shot,
                "model_sha256": MODEL_SHA256, "layers": []}
        for i, (key, (bits, info)) in enumerate(sorted(entry.layers.items())):
            name = AE_KEY if key == AE_KEY else f"row{i}"
            keyed[name] = bits
            meta["layers"].append({"key": name, "row": key, "n_y": int(bits.shape[0]),
                                   **info})
        by_row = {layer["row"]: layer["key"] for layer in meta["layers"]}
        path = layer_file(self.paths, entry.store, entry.shot)
        with atomic_path(path) as tmp, open(tmp, "wb") as f:
            np.savez_compressed(f, meta=np.array(json.dumps(meta)), **keyed)
        rows = {name: {"state": "done", "key": by_row[AE_KEY if AE_KEY in by_row else name]}
                for name in entry.names}
        write_record(self.paths, entry.store, entry.shot, "done",
                     f"TokEye from model {MODEL_SHA8}", job=self.job, store_stamp=entry.stamp,
                     rows=rows, elapsed_s=round(time.time() - entry.started, 2))
        self.count(entry.store, "done")
        print(f"shot {entry.store} {entry.shot}: done, {len(entry.layers)} layer(s) "
              f"in {time.time() - entry.started:.1f} s", flush=True)


def stamp_now() -> str:
    """Local wall-clock time as nvidia-smi's csv writes it, for the job to cut its samples."""
    return datetime.now(UTC).astimezone().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def todo(paths: Paths, items: list[tuple[str, int]], *, force: bool,
         by_event: dict | None = None) -> tuple[list, dict]:
    """The items to run, and how many were skipped: no store, or done and current
    (also per store event into `by_event`, when given)."""
    run, skipped = [], {"no_store": 0, "done": 0}

    def skip(store: str, why: str) -> None:
        skipped[why] += 1
        if by_event is not None:
            tally = by_event.setdefault(store, {})
            tally[f"skipped_{why}"] = tally.get(f"skipped_{why}", 0) + 1

    for store, shot in items:
        record = _read_json(state_file(paths, store, shot)) or {}
        if not paths.spectrogram_file(store, shot).is_file():
            skip(store, "no_store")
            if record.get("state") in ("queued", "computing"):  # a request: say why
                write_record(paths, store, shot, "failed",
                             f"shot {shot} has no {store} review store")
            continue
        if not force and record.get("state") == "done" and \
                record.get("store_stamp") == _stamp(paths.spectrogram_file(store, shot)):
            skip(store, "done")
            continue
        run.append((store, shot))
    return run, skipped


def build(paths: Paths, items: list[tuple[str, int]], *, device: str = "cuda",
          workers: int = 8, batch: int = 32, amp: bool = True, force: bool = False,
          model=None, prefetch: int = 4) -> dict:
    """TokEye over `(store event, shot)` items: the stores read by `workers` loader
    processes while the GPU runs the U-Net over tiles of many rows and shots at once.
    Returns the counts and per-stage seconds."""
    device = torch.device(device)
    by_event: dict[str, dict[str, int]] = {}
    run, skipped = todo(paths, items, force=force, by_event=by_event)
    job = os.environ.get("SLURM_JOB_ID")
    print(f"overlays: {len(run)} shot(s) to run, skipped {skipped}, model {MODEL_SHA8}, "
          f"device {device}, {workers} workers, batch {batch}, amp {amp}", flush=True)
    if model is None and run:
        model = unet.load_unet(device=str(device))
    producer = Producer(paths, model, device, batch=batch, amp=amp, job=job)
    producer.by_event = by_event
    loader = torch.utils.data.DataLoader(
        Items(paths, run), batch_size=None, shuffle=False, num_workers=workers,
        prefetch_factor=prefetch if workers else None, persistent_workers=False,
    )
    started = time.perf_counter()
    print("LOOP_START", stamp_now(), flush=True)
    items_iter, warm = iter(loader), False
    with torch.inference_mode():
        while True:
            with producer.timers("read_wait", sync=False):
                item = next(items_iter, None)
            if item is None:
                break
            producer.take(item)
            if not warm and producer.runner.passes:
                warm = True  # CUDA start-up and cuDNN's first kernels are behind us
                print("LOOP_WARM", stamp_now(), flush=True)
        producer.finish()
    print("LOOP_END", stamp_now(), flush=True)
    seconds = time.perf_counter() - started
    stages = {k: round(v, 2) for k, v in producer.timers.total.items()}
    summary = {**producer.counts, "skipped": skipped, "seconds": round(seconds, 1),
               "stages": stages, "passes": producer.runner.passes,
               "tiles": producer.runner.tiles,
               "by_event": {k: producer.by_event[k] for k in sorted(producer.by_event)}}
    print("overlays summary:", json.dumps(summary), flush=True)
    return summary


# ------------------------------------------------------------------ the AE gate


def gate_shots(paths: Paths) -> list[int]:
    """The shots with a tokeye-full layer: the AE gate's set (180 today)."""
    return sorted(int(p.stem) for p in whole.layer_dir(paths).glob("*.npz")
                  if p.stem.isdigit())


def _compare_one(paths: Paths, shot: int) -> tuple[dict | None, str | None]:
    """`({shot, iou}, None)`, or `(None, why the shot cannot be compared)`."""
    old = whole.view(paths, shot)
    if old is None:
        return None, "no tokeye-full layer"
    state, record = shot_state(paths, seg.EVENT, shot)
    if state["state"] != "done" or not record.get("rows"):
        return None, f"the new layer is {state['state']}: {state['reason']}"
    try:
        new = _read_layers(paths, seg.EVENT, shot, [AE_KEY])[AE_KEY]
    except (OSError, ValueError, KeyError) as error:
        return None, f"the new layer is unreadable: {type(error).__name__}: {error}"
    a, b = _bits(old), _bits(new)
    grid = old["grid"]
    if grid != new["grid"] or a.shape != b.shape:
        return None, f"grids differ: tokeye-full {grid}, new {new['grid']}"
    t = grid["t0_ms"] + np.arange(grid["n"]) * grid["dt_ms"]
    cols = (t >= 0) & (t + grid["dt_ms"] <= full.CROP_END_MS)
    return {"shot": shot, "iou": {
        band: full.iou(a[lo:hi, cols], b[lo:hi, cols])
        for band, (lo, hi) in PAGE_BANDS.items()
    }}, None


def compare(paths: Paths, shots: list[int] | None = None) -> dict:
    """The AE gate: the new AE union against the tokeye-full layer on every shot that
    has one, IoU in the page's 0-250 and 80-250 kHz bins over the columns both cover
    (0-6 s), and `full.check_bar`. It passes only when every one of those shots is
    compared (none excluded) and the bar passes, and then writes `gate.json`, which
    lets the AE page show the new layer. `shots` compares a subset for a look and
    writes no gate."""
    gate_set = gate_shots(paths)
    subset = shots is not None and sorted({int(s) for s in shots}) != gate_set
    shots = gate_set if shots is None else sorted({int(s) for s in shots})
    results, excluded = [], []
    for shot in shots:
        found, why = _compare_one(paths, shot)
        if found is None:
            excluded.append({"shot": shot, "reason": why})
        else:
            results.append(found)
    bar = full.check_bar(results) if results else None
    passed = bool(bar and bar["passed"]) and not excluded and bool(shots) and not subset
    verdict = {
        "passed": passed, "model_sha256": MODEL_SHA256, "at": _now_iso(),
        "n": len(results), "n_shots": len(shots), "subset": subset,
        "excluded": excluded, "bar": bar,
        "bar_rule": {"median": full.IOU_MEDIAN, "low": full.IOU_LOW,
                     "low_share": full.IOU_LOW_SHARE, "coverage": "every tokeye-full shot"},
        "shots_compared": [r["shot"] for r in results], "shots": results,
    }
    if not subset:
        _write_json(gate_file(paths), verdict)
    return {**verdict, "gate": None if subset else str(gate_file(paths))}


def _bits(layer: dict) -> np.ndarray:
    packed = np.frombuffer(base64.b64decode(layer["bits"]), dtype=np.uint8)
    packed = packed.reshape(layer["n_y"], -1)
    return np.unpackbits(packed, axis=1, count=layer["grid"]["n"]).astype(bool)


# ------------------------------------------------------------------ the CLI


def shots_from_file(path) -> list[int]:
    """A shot list: a CSV with a `shot` column, or one shot per line."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [line_ for line_ in text.splitlines() if line_.strip()]
    if lines and not lines[0].split(",")[0].strip().isdigit():
        return [int(row["shot"]) for row in csv.DictReader(lines)]
    return [int(line_.split(",")[0]) for line_ in lines]


def _shots(paths: Paths, args) -> list[int]:
    if args.shots_file:
        return shots_from_file(args.shots_file)
    if args.list:
        from .. import shotlists

        return shotlists.shots(paths, args.list)
    return [int(s) for s in args.shots or []]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m labeler.events.review.overlays",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "status", "request", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--event", nargs="+", default=[])
        which = p.add_mutually_exclusive_group()
        which.add_argument("--shots-file")
        which.add_argument("--list")
        which.add_argument("--shots", nargs="+", type=int)
        if name == "build":
            which.add_argument("--requests", help="a queue batch's requests.txt")
            p.add_argument("--device", default="cuda")
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--batch", type=int, default=32,
                           help="512 x 512 tiles' worth of pixels per forward pass")
            p.add_argument("--no-amp", action="store_true")
            p.add_argument("--force", action="store_true")
    sub.add_parser("drain")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault(raw.NO_FETCH_ENV, "1")  # an overlay job never fetches
    paths = Paths.from_env()
    if args.command == "drain":
        print(json.dumps(drain(paths)))
        return 0
    if args.command == "compare":
        shots = _shots(paths, args) or None
        found = compare(paths, shots)
        print(json.dumps({k: v for k, v in found.items()
                          if k not in ("shots", "shots_compared")}, indent=1))
        for item in found["excluded"]:
            print(f"EXCLUDED {item['shot']}: {item['reason']}")
        print(f"AE gate: {'PASSED' if found['passed'] else 'FAILED'} on {found['n']} of "
              f"{found['n_shots']} shots" + (" (a subset: no gate written)"
                                             if found["subset"] else ""))
        return 0 if found["passed"] else 1
    if args.command == "build" and args.requests:
        items = read_requests(args.requests)
    else:
        if not args.event:
            parser.error("--event is required (or --requests for build)")
        shots = _shots(paths, args)
        stores = list(dict.fromkeys(store_event(e) for e in args.event))
        # Shot by shot, each shot's stores together: the loader then reads the light
        # stores while the GPU runs a heavy one (AE), rather than all of one at the end.
        items = [(store, shot) for shot in dict.fromkeys(shots) for store in stores]
    if args.command == "build":
        try:
            summary = build(paths, items, device=args.device, workers=args.workers,
                            batch=args.batch, amp=not args.no_amp, force=args.force)
        finally:
            if args.requests:
                _write_json(Path(args.requests).parent / "finished.json",
                            {"at": _now_iso(), "job": os.environ.get("SLURM_JOB_ID")})
        return 0 if summary["failed"] == 0 else 1
    for store, shot in items:
        try:
            if args.command == "request":
                found = request(paths, store, shot)
            else:
                found = row_states(paths, store, shot, store_rows_meta(paths, store, shot))
        except FileNotFoundError as error:
            found = {"error": str(error)}
        print(store, shot, json.dumps(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
