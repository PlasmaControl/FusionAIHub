"""Many shots on one GPU: the batch driver behind `tokeye_masks.py`.

The CLI pools consecutive tiles across channels, passes and shots. Prep runs
in spawned CPU workers; the parent fills pinned batches, overlaps H2D on a
copy stream, and retains overlap stitching and compaction on the device.
Only packed masks, row/column summaries and sparse coherent probabilities
cross back. With tail workers, description and shot finishing run in that
pool while the GPU processes later shots. Results retain plan and shot order.

The full-probability per-block schedule remains available with
`--probs-on-host` (or `run_shots(pooled=False)`). `pipeline.process_shot` uses
that independent inference reference. Both schedules preserve complete NPZ
bytes and exact events/sources frames except run_id/written_at, including
source coverage intervals and min_gap_s, when no OOM halving occurs or when
it occurs identically. Forward groups match the reference on every device;
CPU cross-block forwards require the explicit --pool-cpu-forwards opt-in.

Memory is bounded by the prep queue, two pinned tile batches, partial device
stitches and the tail backlog. `--prefetch` is also prep concurrency; choosing
fewer slots than workers leaves workers idle. Raw log-power survives until
CPU description, so pending tails cost more than their packed output files.
The network, CUDA context and streams remain exclusively in the parent.

Each task owns its per-shot files. Build the whole list's text subset once,
then use `--text-subset readonly --no-index` for concurrent tasks. Rebuild the
shared index separately after completion. A resolved output root is printed
before any directory is created; pass an explicit --root for scratch runs.

Prep and tail failures are isolated and broken pools are replaced; inferred
tail payloads survive a restart. `PrepPool.close` terminates wedged workers
rather than leaving interpreter shutdown waiting forever. Timeout alarms
bound synchronous stages; future waits have explicit deadlines.

Operating rationale retained from the original driver:

**Job and split units.** Contiguous chunks let array tasks walk different
regions of the corpus directory; strided ranks interleave within one node's
shared page cache. One prep job is one `(channel, pass)`, so wide and zoom
read a channel twice. On shot 185786 this cost ~3-6 % more CPU per channel
and roughly doubled GPFS requests (~260 MB -> ~520 MB per shot), mostly
served from the page cache because the passes are adjacent. This did not
justify splitting the job unit.

**Memory arithmetic.** A prepared block holds two `(512, T)` float32 arrays:
the standardised spectrogram and raw log-power. At 64,000 columns these are
131 MB each; a `--prefetch 4` queue alone is ~1.05 GB. Workers hold another
pair while preparing, and the pooled schedule additionally holds input
buffers, partial device groups and raw arrays awaiting tail description.
Do not size the whole job from the prep queue alone.

**The two files this driver does NOT own.** `masks/<shot>_masks.npz` and
`events/<shot>_events.parquet` are per shot, so sixteen tasks writing
sixteen shots never meet. Two other files belong to the whole root, and
both are whole-file rewrites, which is a lost update rather than a torn
file - the damage a pid-suffixed temporary does not prevent:

* `events_index.parquet` (`labels.store.append_index`, called per shot by
  `pipeline.finish_shot` through a FIXED `.tmp` sibling). Concurrent tasks
  lose each other's rows AND can rename a half-written parquet into place,
  after which the next shot's `read_parquet` raises inside the write guard
  and a healthy shot is recorded `status == "error"`. **`--no-index`** is
  the answer: the per-shot products are written and the shared file is
  left alone, and `--rebuild-index` regenerates it from
  `events/*_events.parquet` in one pass afterwards (it is derivable, so
  the rebuild is exact and is idempotent even if a task died).
* `text/logs_subset.jsonl` (`text_weak.build_logs_subset`), which is worse
  because it changes what is WRITTEN: sixteen tasks read the same old
  subset, each appends its own thirty shots and the last rename wins, so
  fifteen tasks' records are gone before their shots are processed and
  `finish_shot` files `skipped["text"]` for every one of them. **The
  subset is built once, before the array** (`--build-text-subset`), and a
  run with `--world` or `--n-chunks` above 1 reads it and never writes it
  (`--text-subset auto`, or `readonly` to force it). A shot the pre-pass
  did not cover is named in the run record and in its own `skipped["text"]`
  with the command that fixes it. (`text_weak.text_events` re-checks the
  subset per shot, which under `readonly` is always a no-op: it is only
  reached for a shot whose record is already there, and
  `build_logs_subset` returns without writing when it has nothing to add.)
  **The pre-pass is REQUIRED for multi-rank or multi-chunk runs** to match
  `python -m labeler.run events`: without it, `readonly` adds a `text`
  skip and omits `text` from an uncovered shot's declared sources. A
  covered record with no matches still declares `text` with zero events.
  Check that every run JSON's `text_subset_missing` is empty.

**Isolation and SIGALRM.** The legacy schedule places one alarm around a
shot; the pooled schedule arms each synchronous stage from the remaining
shot deadline and attributes a shared forward timeout only to expired shots.
The alarm interrupts waits and Python execution; a native call may deliver
it only when control returns to Python. Inside `pipeline.finish_shot`, a
step's own guard can turn the alarm into that step's skip, after which the
shot finishes without that timer, as in `process_shot`. A parent alarm does
not reach spawned tail workers: their future waits have a separate timeout.
A dead worker or an expired wait replaces the pool and resubmits retained
payloads in their original positions; ordinary task exceptions, including a
task-raised TimeoutError, leave the executor usable. Prep-pool replacement
invalidates the old epoch's queued futures so remaining blocks are resubmitted.
Closing pools terminates and then kills wedged workers: a shutdown sentinel
alone cannot release CPython's interpreter-exit join of the manager thread.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import platform
import resource
import socket
import sys
import time
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..catalog import read_shot_file
from ..config import Paths, git_sha
from ..run import (
    EXIT_NO_SHOTS,
    EXIT_OK,
    StageTimeout,
    time_limit,
    write_events_run,
)
from . import channels, masks, pipeline, schema, text_weak, unet
from .lexicon import load_lexicon
from .unet import CHECKPOINT_SHA256

#: `--plan` values. One for now; the flag exists because round 2 is a
#: different eleven channels and a run has to record which it was.
PLANS: dict[str, tuple[channels.ChannelSpec, ...]] = {
    "round1": channels.ROUND1_PLAN,
}

#: `--tile-batch` when the flag is not given: 96 on a card (spec A5; the
#: measurement behind it is that 8-32 was flat at 197 tiles/s on a V100S, so
#: the number is about filling the card, not about speed) and 8 on a CPU,
#: where a 512-tile batch is real memory and there is no launch cost to
#: amortise.
TILE_BATCH_DEFAULT = {"cuda": 96, "cpu": 8}

#: `--prep-workers` on a card when `SLURM_CPUS_PER_TASK` says nothing.
CUDA_WORKERS_FALLBACK = 4

#: Seconds `PrepPool.close` waits for a worker to die, in each of its two
#: rounds (`terminate`, then `kill`). Short on purpose: this is the path a
#: shot that timed out goes through, and the run is already late.
WORKER_EXIT_GRACE_S = 5.0

#: What `pipeline.finish_shot` records for a shot the subset has no record
#: of, mirrored here so a read-only run can tell "the logbook says nothing
#: about this shot" from "nobody built this shot into the subset".
NO_SUBSET_RECORD = "no logbook record in the subset"

#: The second, and better, reason - only ever recorded under
#: `--text-subset readonly`, where the driver KNOWS the shot is absent
#: because it looked before the run started.
NOT_PREBUILT = (
    "not in the prebuilt logs subset (--text-subset readonly); build it "
    "once before the array: tokeye_masks.py --build-text-subset --shot-file"
)

#: Cores left to this process when the workers are sized from
#: `SLURM_CPUS_PER_TASK`: one for the parent's own describe steps and one
#: for the CUDA driver's spinning.
CORES_FOR_THE_PARENT = 2


# ------------------------------------------------------------- the shot split


def split_shots(
    shots: Iterable[int],
    *,
    chunk: int = 0,
    n_chunks: int = 1,
    rank: int = 0,
    world: int = 1,
    limit: int = 0,
) -> list[int]:
    """The shots THIS array task and THIS rank are to do.

    Sorted and de-duplicated first, because the split has to partition: a
    shot listed twice would otherwise land in two tasks, and two tasks
    writing one `<shot>_masks.npz` is a race over one file.

    The chunk is contiguous and the rank is strided, for the two different
    reasons in the module docstring. `limit` is applied LAST, so
    `--limit 2` means "this task does two shots" and a five-task smoke run
    does five shots per task rather than five in total.

    An empty result is legal - more tasks than shots is what the tail of an
    array looks like - but a chunk or rank outside its range is not: it
    would silently drop the shots nobody else is doing.
    """
    if not 1 <= int(n_chunks) or not 0 <= int(chunk) < int(n_chunks):
        raise ValueError(f"chunk {chunk} of {n_chunks} is not a valid split")
    if not 1 <= int(world) or not 0 <= int(rank) < int(world):
        raise ValueError(f"rank {rank} of {world} is not a valid split")
    ordered = sorted({int(s) for s in shots})
    lo = (len(ordered) * int(chunk)) // int(n_chunks)
    hi = (len(ordered) * (int(chunk) + 1)) // int(n_chunks)
    mine = ordered[lo:hi][int(rank)::int(world)]
    return mine[:int(limit)] if int(limit) > 0 else mine


# ----------------------------------------------------------------- the index


#: What identifies a row of `events_index.parquet` - the key
#: `pipeline.finish_shot` passes `labels.store.append_index` per shot, and
#: the one a rebuild has to pass to get the same file.
INDEX_KEYS = ("shot", "source", "phenomenon")


def rebuild_index(paths: Paths, *, out: Path | None = None, echo=print) -> dict:
    """`events_index.parquet`, regenerated from the per-shot events files.

    The index is derivable: `schema.index_rows(events_file)` is a pure
    function of one shot's parquet, so the whole file is
    `sorted(events/*_events.parquet)` read once. That is what makes
    `--no-index` safe for a SLURM array - the shared file nobody can write
    concurrently is simply not written during the array, and this rebuilds
    it in one pass afterwards (`--rebuild-index`, or an `afterok` job).

    REPLACES rather than merges. `append_index` merges by
    `(shot, source, phenomenon)`, so a row whose events file has since been
    deleted would survive every incremental write for ever; a rebuild is
    the repair for exactly that, and has to describe what is on disk now.
    An events file that cannot be read is named in `unreadable` and costs
    only its own rows: a torn file from an earlier racing array is the
    likeliest reason to be running this at all.
    """
    from ..labels.store import append_index

    files = sorted(Path(paths.events).glob("*_events.parquet"))
    rows: list[dict] = []
    unreadable: dict[str, str] = {}
    for events_file in files:
        try:
            rows.extend(schema.index_rows(events_file))
        except Exception as exc:  # noqa: BLE001 - one bad file is not the run
            unreadable[events_file.name] = pipeline._cause(exc)
            echo(f"tokeye_masks: cannot index {events_file.name} - "
                 f"{unreadable[events_file.name]}")
    out = Path(out) if out is not None else Path(paths.events_index)
    out.unlink(missing_ok=True)
    append_index(out, rows, keys=list(INDEX_KEYS))
    echo(f"tokeye_masks: rebuilt {out} - {len(rows)} rows from "
         f"{len(files) - len(unreadable)} of {len(files)} events files")
    return {"files": len(files), "rows": len(rows), "index": str(out),
            "unreadable": unreadable}


# ------------------------------------------------------------------- the jobs


@dataclass(frozen=True)
class PlanJob:
    """"Which channels can this shot run, and over what window": header reads."""

    corpus_file: Path
    plan: tuple[channels.ChannelSpec, ...]
    norm: str = "record"


@dataclass(frozen=True)
class PrepJob:
    """The CPU half of one `(channel, pass)`, as sent to a prep worker."""

    corpus_file: Path
    spec: channels.ChannelSpec
    pass_name: str
    norm: str = "record"
    window: tuple[float, float] | None = None

    @property
    def key(self) -> str:
        """`"mhr:0:wide"` - how this block is named in `skipped`."""
        return f"{self.spec.key}:{self.pass_name}"


class BlockFailed(Exception):
    """A job that failed, reduced to `(stage, cause)` before it crosses back.

    Two things at once. The **stage** is which of `process_shot`'s skip keys
    this is: `read <diag>:<ch>` for the waveform, `mask <diag>:<ch>:<pass>`
    for the transform, and `plan` for the corpus file itself, which is a
    shot-ending error rather than a skip. The **reduction to strings** is so
    that what comes back over the pipe is always picklable: a worker may
    raise anything at all - an h5py error carrying an open file handle, a
    `MemoryError` raised while there is no memory to build a traceback - and
    a driver that could not unpickle the failure would turn one bad channel
    into a dead run.
    """

    def __init__(self, stage: str, cause: str):
        super().__init__(stage, cause)
        self.stage = str(stage)
        self.cause = str(cause)

    def __str__(self) -> str:
        return f"{self.stage}: {self.cause}"


def plan_one(job: PlanJob):
    """`pipeline.plan_shot`, in whichever process the driver put it in."""
    try:
        return pipeline.plan_shot(job.corpus_file, plan=job.plan, norm=job.norm)
    except Exception as exc:  # noqa: BLE001 - reduced to (stage, cause)
        raise BlockFailed("plan", pipeline._cause(exc)) from None


def prep_one(job: PrepJob) -> pipeline.PreparedBlock:
    """One channel read and transformed - the whole of a prep worker's work.

    The read and the transform are caught separately because
    `process_shot` records them under different keys, and a driver that
    called a failed STFT a failed read would be reporting a different shot
    from the one the sequential path reports.
    """
    started = time.monotonic()
    try:
        y, fs_hz, t0_s, t1_s = masks.read_waveform(
            job.corpus_file, job.spec.diag, job.spec.channel
        )
    except Exception as exc:  # noqa: BLE001 - per-channel isolation
        raise BlockFailed("read", pipeline._cause(exc)) from None
    try:
        prepared = pipeline.prep_block(
            y, fs_hz, t0_s, t1_s, job.spec, job.pass_name,
            norm=job.norm, window=job.window,
        )
        prepared.prep_seconds = time.monotonic() - started
        return prepared
    except Exception as exc:  # noqa: BLE001 - per-pass isolation
        raise BlockFailed("mask", pipeline._cause(exc)) from None


@dataclass
class FinishJob:
    """Everything a shot does AFTER its blocks, as sent to the tail worker.

    `pipeline.finish_shot` and its inputs: the result so far, this shot's
    `_BlockRun`s in plan order, and the two writes' destinations. Tens of
    megabytes of packed masks per shot cross the pipe with it - against the
    ~3.7 s of parent-thread work it takes off the process that holds the
    GPU (measured; see the module docstring).

    `text_missing` is this shot's answer to "was it absent from a read-only
    logs subset", carried on the job because the sentence it changes is
    written in `finish_shot`.
    """

    res: pipeline.ShotResult
    paths: Paths
    corpus_file: Path
    runs: list
    unet_sha256: str = CHECKPOINT_SHA256
    lexicon: Any = None
    run_id: str = "manual"
    write: bool = True
    index: bool = True
    text_missing: bool = False
    descriptions: list = field(default_factory=list)


@dataclass
class FinishDone:
    """What a finished tail brings back: the shot's result and its cost."""

    res: pipeline.ShotResult
    seconds: float = 0.0
    describe_seconds: float = 0.0


def finish_one(job: FinishJob) -> FinishDone:
    """`pipeline.finish_shot`, in whichever process the driver put it in.

    The result is RETURNED rather than mutated in place, because in a
    worker the `res` this runs on is a copy and the parent's is not.
    """
    at = time.monotonic()
    for prepared, compact in job.descriptions:
        try:
            job.runs.append(pipeline.describe_block(
                prepared, compact, unet_sha256=job.unet_sha256))
        except Exception as exc:  # noqa: BLE001 - one descriptor block
            job.res.skipped[f'mask {prepared.key}'] = pipeline._cause(exc)
    describe_seconds = time.monotonic() - at if job.descriptions else 0.0
    if job.descriptions:
        job.res.n_blocks = len(job.runs)
    job.descriptions.clear()
    at = time.monotonic()
    res = pipeline.finish_shot(
        job.res, job.paths, job.corpus_file, job.runs,
        unet_sha256=job.unet_sha256, lexicon=job.lexicon,
        run_id=job.run_id, write=job.write, index=job.index,
    )
    if job.text_missing and res.skipped.get("text") == NO_SUBSET_RECORD:
        # The same fact with the cause in it: under `--text-subset readonly`
        # the driver looked this shot up in the subset before the run and
        # knows it is absent, which is a pre-pass that did not cover it and
        # not a logbook that has nothing to say about the shot.
        res.skipped["text"] = f"{NOT_PREBUILT} <list>"
    return FinishDone(res=res, seconds=time.monotonic() - at,
                      describe_seconds=describe_seconds)


@dataclass(frozen=True)
class WarmJob:
    """Nothing at all, sent to a pool to make it start a worker NOW.

    A spawned worker is a fresh interpreter and a torch import - 5.4 s
    against 2.0 s for the same job in a warm worker, measured - and a
    `ProcessPoolExecutor` starts its workers lazily, at the first submit.
    For the tail pool that would put the whole of it in front of the first
    shot's tail, with the parent waiting; sent at the top of a run it is
    paid while the first shot's blocks are in the network instead.
    """


def run_job(job):
    """Dispatch, so a pool holds one picklable entry point and not three."""
    if isinstance(job, WarmJob):
        return None
    if isinstance(job, PlanJob):
        return plan_one(job)
    if isinstance(job, FinishJob):
        return finish_one(job)
    return prep_one(job)


#: The thread-count variables a prep worker must inherit. Set in the PARENT
#: before the pool starts (`PrepPool._start`), because a spawned child gets
#: its environment at exec and OpenBLAS/MKL size their pools when they are
#: imported - which, with `spawn`, has already happened by the time the
#: initializer below runs.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def _init_worker() -> None:
    """One thread per prep worker - the part of it that still can be set.

    Only `torch.set_num_threads(1)` bites here. It is a runtime call, so it
    takes effect whenever it is made; the four environment variables do not,
    because with `spawn` the child unpickles the worker bootstrap - which
    resolves this function by IMPORTING `labeler.events.driver`, and with
    it numpy, scipy and torch - before this body runs, and by then
    OpenBLAS and MKL have already sized their pools from the environment
    they were exec'd with. They are set in the parent instead
    (`PrepPool._start`), and the sbatch exports them before python starts,
    which is the only moment they are certain to be read. Setting them
    again here costs nothing and covers a worker started some other way.

    Belt and braces either way: the prep work itself - `ShortTimeFFT`,
    `signal.decimate`, a percentile and a z-score - is single-threaded
    numpy and scipy with no BLAS call in it. What it protects against is
    eighteen workers sharing a node with two GPU tasks (spec A5).
    """
    for var in THREAD_VARS:
        os.environ[var] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001, S110 - a thread count is not worth a worker
        pass


def _vmhwm_gib(pid) -> float:
    """A LIVE process's peak resident set, in GiB, from `/proc/<pid>/status`.

    `getrusage(RUSAGE_CHILDREN)` cannot answer this: it counts only children
    that have been WAITED FOR, and it counts the fork-before-exec of a
    spawned child - which inherits this process's resident pages - so on a
    pool that is shut down without waiting it reads either zero or the
    parent's own peak. Neither is the prep worker. `VmHWM` is the kernel's
    own high-water mark for the process, so it can be read at any moment
    while the worker is alive and still reports the worst it ever was.

    Best effort by construction: a worker that has already exited has no
    `/proc` entry, and this is a report field, not a control input.
    """
    try:
        with open(f"/proc/{int(pid)}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


class PrepPool:
    """`--prep-workers` processes doing the CPU half of a block, or none.

    **Spawned, never forked.** By the time a pool starts, this process has
    torch loaded (and on a GPU run a CUDA context alive) and h5py open on
    the corpus; forking that is the classic way to get a child that hangs on
    its first read or a CUDA context that cannot be used in either process.
    `spawn` costs a fresh interpreter and a torch import per worker - a few
    seconds, once, against hours of run - and is the only start method that
    is safe here. `forkserver` is not: its server process is itself forked
    from this one, after torch is live.

    `workers=0` is the degenerate pool: the job runs in this process, on
    submission, and nothing is prepared ahead. It is what a debugger, a
    profiler and the identity test want, and it is the same code path -
    everything below sees a future either way.
    """

    def __init__(self, workers: int):
        self.workers = max(0, int(workers))
        #: The worst `VmHWM` seen in any worker of any of this pool's
        #: incarnations - sampled per shot and again as each pool is closed,
        #: because a worker that is gone cannot be asked.
        self.peak_worker_rss_gib = 0.0
        self.worker_rss_gib: dict[str, float] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._start()

    def _note_worker_rss(self, procs) -> None:
        for proc in procs:
            if proc.pid:
                rss = _vmhwm_gib(proc.pid)
                key = str(proc.pid)
                self.worker_rss_gib[key] = max(self.worker_rss_gib.get(key, 0.0),
                                               rss)
                self.peak_worker_rss_gib = max(self.peak_worker_rss_gib,
                                               rss)

    def sample(self) -> float:
        """Read the live workers' high-water marks; the worst so far.

        Called once per shot by `run_shots`: one `/proc` read per worker against
        hours of run, and the only way the number survives a worker that
        the OOM killer takes before the pool is closed.
        """
        self._note_worker_rss(self.processes())
        return self.peak_worker_rss_gib

    def _start(self) -> None:
        if self.workers > 0:
            # Before the executor, so the children inherit it at exec:
            # OpenBLAS and MKL size their pools when they are imported, and
            # a spawned worker imports them before its initializer runs.
            # This process's own libraries read these at THEIR import and
            # are long past caring, so this sets the workers' threads and
            # not the parent's.
            for var in THREAD_VARS:
                os.environ[var] = "1"
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
            )

    def warm(self) -> None:
        """Start a worker now rather than at the first real job (`WarmJob`)."""
        if self._pool is not None:
            self.submit(WarmJob())

    def submit(self, job) -> Future:
        """A future for `job`, ALWAYS - a dead pool included.

        `ProcessPoolExecutor.submit` raises `BrokenProcessPool` in the
        caller once a worker has died, and a driver that let that out of
        `in_order` would only know that some job could not be submitted,
        not which. Carried on the future instead, the failure arrives at
        the job it belongs to and is recorded against that block.
        """
        future: Future = Future()
        if self._pool is None:
            try:
                future.set_result(run_job(job))
            except BaseException as exc:  # noqa: BLE001 - carried, not raised
                future.set_exception(exc)
            return future
        try:
            return self._pool.submit(run_job, job)
        except BrokenExecutor as exc:
            future.set_exception(exc)
            return future

    def processes(self) -> list:
        """The pool's live worker processes, for `close` and for tests.

        `ProcessPoolExecutor._processes` is private and is read here
        deliberately: it is the only handle on the children, and `close`
        cannot guarantee that a wedged worker dies without one.
        """
        pool = self._pool
        return list(getattr(pool, "_processes", {}).values()) if pool else []

    def restart(self) -> None:
        """Replace a pool a dead worker broke, or one an aborted shot left.

        Every future the old pool held is abandoned. That is the point: a
        shot that hit its `SIGALRM` may have four prepared blocks in flight
        and no reader for them, and a worker that was killed mid-job has
        poisoned the executor for everything after it.
        """
        self.close()
        self._start()

    def close(self) -> None:
        """Shut the pool down and leave NO worker of it alive.

        `shutdown(wait=False)` asks politely - it drops a sentinel on the
        call queue and returns - and that is not enough. A worker wedged in
        an uninterruptible read (the GPFS failure `--timeout` exists for)
        never reads the sentinel, and CPython's own `_python_exit` atexit
        hook joins the executor's manager thread at interpreter shutdown:
        the process then hangs AFTER `main` has returned, which on SLURM is
        a finished task recorded as a wall-clock TIMEOUT still holding its
        GPU. So the children are taken by the handle - `terminate`, a
        bounded `join`, then `kill` - and only then is this pool forgotten.

        Waiting is bounded at `WORKER_EXIT_GRACE_S` per worker in each of
        the two rounds: a process nothing can kill is a hung node, and a
        driver that joined it forever would be one too.
        """
        pool, self._pool = self._pool, None
        if pool is None:
            return
        # Private API, deliberately: `_processes` is the only handle on the
        # children, and without it `close` cannot promise the run can exit.
        procs = list(getattr(pool, "_processes", {}).values())
        self._note_worker_rss(procs)
        pool.shutdown(wait=False, cancel_futures=True)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=WORKER_EXIT_GRACE_S)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=WORKER_EXIT_GRACE_S)


def in_order(jobs, submit, *, prefetch: int = 4):
    """Submit `jobs`, yield `(job, future)` in order, `prefetch` outstanding.

    The whole of the overlap, in nine lines: the consumer gets the jobs back
    in the order it gave them - which is what makes the driver's block order
    `process_shot`'s plan order - while up to `prefetch` of them are being
    prepared ahead of it. The bound is the memory bound (a prepared block is
    262 MB on the widest channel), so it is enforced here, on submission,
    rather than left to a queue that grows to the length of the shot.

    `prefetch=0` means one job in flight, which is what a degenerate
    (in-process) pool wants: preparing ahead of yourself buys nothing.
    """
    pending: deque = deque()
    remaining = iter(jobs)
    while True:
        while len(pending) < max(1, int(prefetch)):
            job = next(remaining, None)
            if job is None:
                break
            pending.append((job, submit(job)))
        if not pending:
            return
        yield pending.popleft()


# --------------------------------------------------------------- the shot loop


@dataclass
class ShotTiming:
    """Where one shot's wall clock went, and how many tiles bought it.

    `prep_wait_s` is the number this driver exists to drive down: the time
    the process holding the GPU spent waiting for a prepared block. Zero
    would mean the workers are always ahead; a `prep_wait_s` close to the
    shot's own elapsed time means they never are and the run wants more
    `--prep-workers`.

    `finish_s` is the shot's TAIL - `pipeline.finish_shot`: the sawtooth
    pass over the whole ECE array, L-H, the actuators, the QH proxy, the
    text lookup and the two writes - wherever it ran. `tail_wait_s` is how
    much of it the GPU-owning thread actually waited for, which is the
    whole point of `--tail-workers`: with one, `finish_s` is paid beside
    the next shot's forward passes and `tail_wait_s` falls to ~0; with
    zero, the two are equal and the card idles for `finish_s` a shot.
    """

    prep_wait_s: float = 0.0
    infer_s: float = 0.0
    describe_s: float = 0.0
    finish_s: float = 0.0
    tail_wait_s: float = 0.0
    n_tiles: int = 0

    def as_row(self) -> dict[str, float | int]:
        return {
            "prep_wait_s": round(self.prep_wait_s, 3),
            "infer_s": round(self.infer_s, 3),
            "describe_s": round(self.describe_s, 3),
            "finish_s": round(self.finish_s, 3),
            "tail_wait_s": round(self.tail_wait_s, 3),
            "n_tiles": int(self.n_tiles),
        }

    def add(self, other: ShotTiming) -> None:
        self.prep_wait_s += other.prep_wait_s
        self.infer_s += other.infer_s
        self.describe_s += other.describe_s
        self.finish_s += other.finish_s
        self.tail_wait_s += other.tail_wait_s
        self.n_tiles += other.n_tiles


def _shot_blocks(
    shot: int,
    *,
    corpus_file: Path,
    model,
    pool: PrepPool,
    plan: Sequence[channels.ChannelSpec],
    passes: Sequence[str],
    prefetch: int,
    device: str,
    tile_batch: int,
    amp: bool,
    norm: str,
    unet_sha256: str = CHECKPOINT_SHA256,
) -> tuple[pipeline.ShotResult, ShotTiming, list | None]:
    """One shot's MASK BLOCKS: prepared in the pool, inferred here.

    The same sequence of decisions as `process_shot` - plan, then a block
    per `(channel, pass)` in plan order - with the CPU half of each block
    moved off this process. Every failure is recorded under the key
    `process_shot` records it under, because the two are required to
    produce the same `ShotResult`.

    Returns the blocks rather than finishing the shot, so that the caller
    can hand the tail (`pipeline.finish_shot`) to another process and get
    on with the next shot's forward passes. `runs is None` means the one
    failure that ends a shot before it starts: a corpus file that could not
    be planned.
    """
    res = pipeline.ShotResult(shot=int(shot))
    timing = ShotTiming()

    try:
        specs, planned, window = pool.submit(
            PlanJob(corpus_file, tuple(plan), norm)
        ).result()
    except BlockFailed as exc:
        # The one run-ending failure: a corpus file that cannot be read.
        res.error = exc.cause
        return res, timing, None
    except BrokenExecutor as exc:
        pool.restart()
        res.error = pipeline._cause(exc)
        return res, timing, None
    res.skipped.update(planned)

    jobs = [
        PrepJob(corpus_file, spec, pass_name, norm, window)
        for spec in specs
        for pass_name in passes
    ]
    runs: list[Any] = []
    done = 0
    while done < len(jobs):
        broken = False
        for job, future in in_order(jobs[done:], pool.submit,
                                    prefetch=prefetch):
            waited = time.monotonic()
            try:
                prepared = future.result()
            except BlockFailed as exc:
                timing.prep_wait_s += time.monotonic() - waited
                done += 1
                key = (f"read {job.spec.key}" if exc.stage == "read"
                       else f"mask {job.key}")
                res.skipped[key] = exc.cause
                continue
            except BrokenExecutor as exc:
                # A worker died - the OOM killer, a segfault - and took
                # this block with it. It is a skip, like any other failed
                # block; the pool is replaced below and the shot's
                # remaining blocks are submitted to the new one.
                timing.prep_wait_s += time.monotonic() - waited
                done += 1
                res.skipped[f"mask {job.key}"] = pipeline._cause(exc)
                broken = True
                break
            timing.prep_wait_s += time.monotonic() - waited
            done += 1
            try:
                at = time.monotonic()
                probs = pipeline.infer_block(
                    prepared, model=model, device=device,
                    tile_batch=tile_batch, amp=amp,
                )
                timing.infer_s += time.monotonic() - at
                timing.n_tiles += prepared.n_tiles
                # 131 MB, and `describe_block` wants `raw`, not this.
                prepared.spectrogram = None
                at = time.monotonic()
                runs.append(pipeline.describe_block(
                    prepared, probs, unet_sha256=unet_sha256
                ))
                timing.describe_s += time.monotonic() - at
            except StageTimeout:
                # The shot's own alarm, not this block's failure: it is the
                # driver's to report as an error row, and a per-block guard
                # that swallowed it would leave the shot running with no
                # timer left.
                raise
            except Exception as exc:  # noqa: BLE001 - per-pass isolation
                res.skipped[f"mask {job.key}"] = pipeline._cause(exc)
            if prepared.norm_note:
                res.skipped[f"norm {prepared.key}"] = prepared.norm_note
        if broken:
            pool.restart()
    res.n_blocks = len(runs)
    return res, timing, runs


@dataclass
class _Unfinished:
    """A shot whose blocks are done and whose tail is still running.

    At most one of these exists at a time: shot i's tail is collected as
    soon as shot i+1's blocks are, which is what bounds the overlap at one
    shot and keeps `rows` in shot order. Keep the job so its inferred
    blocks survive a pool restart while the previous tail is collected.
    """

    shot: int
    timing: ShotTiming
    blocks_s: float
    future: Future
    job: FinishJob


@dataclass
class DriverRun:
    """What a driver invocation produced: the per-shot rows and the totals."""

    rows: list[dict]
    totals: dict


def _rss_gib(who: int) -> float:
    """Peak resident set of this process, in GiB (`ru_maxrss` is in KiB).

    `RUSAGE_SELF` only. The prep workers are measured by `_vmhwm_gib`
    instead: `RUSAGE_CHILDREN` counts only children this process has waited
    for, and a spawned child's fork-before-exec inherits the parent's
    resident pages, so asking it about a pool shut down with `wait=False`
    returns either zero or this process's own peak wearing a worker's name.
    """
    return resource.getrusage(who).ru_maxrss / (1024.0 * 1024.0)


def _cuda_peak_gib(device: str) -> float | None:
    if str(device) != "cuda":
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / float(1 << 30)


@dataclass
class _PooledShot:
    shot: int
    corpus_file: Path
    res: pipeline.ShotResult
    timing: ShotTiming = field(default_factory=ShotTiming)
    runs: list = field(default_factory=list)
    descriptions: list = field(default_factory=list)
    remaining: int = 0
    started: float = field(default_factory=time.monotonic)
    skipped: bool = False


def _run_pooled(shots, **opts):
    """Bounded, ordered prep stream spanning channels, passes and shots."""
    paths = opts["paths"]
    pool = PrepPool(opts["prep_workers"])
    tails = PrepPool(opts["tail_workers"])
    tails.warm()
    states, pending = deque(), deque()
    rows, totals = [], ShotTiming()
    started = time.monotonic()
    epoch = 0

    def remaining(state):
        return opts["timeout_s"] - (time.monotonic() - state.started)

    def guard(states):
        left = min(remaining(state) for state in states)
        if left <= 0:
            raise StageTimeout(f"exceeded {opts['timeout_s']}s for shot blocks")
        return time_limit(max(1, int(left) + 1))

    def jobs():
        nonlocal epoch
        for shot in shots:
            corpus_file = (
                Path(opts["corpus_dir"]) / f"{int(shot)}_processed.h5"
                if opts["corpus_dir"] is not None
                else paths.corpus_file(shot)
            )
            state = _PooledShot(int(shot), corpus_file, pipeline.ShotResult(int(shot)))
            states.append(state)
            if (
                opts["skip_existing"]
                and not opts["force"]
                and paths.masks_file(shot).exists()
                and paths.events_file(shot).exists()
            ):
                state.skipped = True
                continue
            try:
                with guard([state]):
                    specs, planned, window = pool.submit(
                        PlanJob(corpus_file, tuple(opts["plan"]), opts["norm"])
                    ).result(timeout=opts["timeout_s"])
                state.res.skipped.update(planned)
            except Exception as exc:  # noqa: BLE001 - one shot's header
                state.res.error = (
                    exc.cause if isinstance(exc, BlockFailed) else pipeline._cause(exc)
                )
                if isinstance(exc, (BrokenExecutor, TimeoutError, StageTimeout)):
                    pool.restart()
                    epoch += 1
                continue
            state.remaining = len(specs) * len(opts["passes"])
            for spec in specs:
                for pass_name in opts["passes"]:
                    yield (
                        state,
                        PrepJob(corpus_file, spec, pass_name, opts["norm"], window),
                    )

    def submit(item):
        try:
            with guard([item[0]]):
                future = pool.submit(item[1])
        except StageTimeout as exc:
            future = Future()
            future.set_exception(exc)
        future.l14_epoch = epoch
        return future

    def prepared_blocks():
        nonlocal epoch
        for (state, job), future in in_order(jobs(), submit, prefetch=opts["prefetch"]):
            at = time.monotonic()
            try:
                if future.l14_epoch != epoch:
                    future = submit((state, job))
                prepared = future.result(timeout=max(0, remaining(state)))
                if remaining(state) <= 0:
                    raise StageTimeout("shot timed out during preparation")
            except Exception as exc:  # noqa: BLE001 - preserve per-block isolation
                key = (
                    f"read {job.spec.key}"
                    if isinstance(exc, BlockFailed) and exc.stage == "read"
                    else f"mask {job.key}"
                )
                state.res.skipped[key] = (
                    exc.cause if isinstance(exc, BlockFailed) else pipeline._cause(exc)
                )
                state.remaining -= 1
                if isinstance(exc, (TimeoutError, StageTimeout)):
                    state.res.error = pipeline._cause(exc)
                if isinstance(exc, (BrokenExecutor, TimeoutError, StageTimeout)):
                    pool.restart()
                    epoch += 1
                continue
            finally:
                state.timing.prep_wait_s += time.monotonic() - at
            if prepared.norm_note:
                state.res.skipped[f"norm {prepared.key}"] = prepared.norm_note
            yield (state, prepared), prepared.spectrogram
            prepared.spectrogram = None

    def flush():
        state, future, _job = pending.popleft()
        at = time.monotonic()
        if state.skipped:
            row = {
                "shot": state.shot,
                "status": "skipped",
                "error": "",
                "skipped": {},
                "by_source": {},
                "n_blocks": 0,
                "n_events": 0,
            }
        else:
            try:
                if future is not None:
                    done = future.result(timeout=opts["timeout_s"])
                    state.timing.finish_s += done.seconds
                    state.timing.describe_s += done.describe_seconds
                    state.res = done.res
                state.res.elapsed_s = time.monotonic() - state.started
                row = state.res.as_row()
            except Exception as exc:  # noqa: BLE001 - tail failure is one shot
                row = {
                    "shot": state.shot,
                    "status": "error",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:200],
                }
                wait_expired = isinstance(exc, TimeoutError) and not future.done()
                if isinstance(exc, BrokenExecutor) or wait_expired:
                    tails.restart()
                    # Preserve inferred payloads already queued behind this job.
                    for position, (other, old, payload) in enumerate(pending):
                        if old is not None and (
                            not old.done()
                            or old.cancelled()
                            or isinstance(old.exception(), BrokenExecutor)
                        ):
                            pending[position] = (other, tails.submit(payload), payload)
        state.timing.tail_wait_s += time.monotonic() - at
        row["seconds"] = round(time.monotonic() - state.started, 2)
        row.update(state.timing.as_row())
        totals.add(state.timing)
        rows.append(row)
        opts["echo"](
            f"{state.shot}: {row.get('n_blocks', 0)} blocks, "
            f"{row.get('n_events', 0)} events in {row['seconds']}s"
        )

    def dispatch_ready():
        while states and states[0].remaining == 0:
            state = states.popleft()
            state.res.n_blocks = len(state.runs)
            job, future = None, None
            if not state.skipped and not state.res.error:
                job = FinishJob(
                    res=state.res,
                    paths=paths,
                    corpus_file=state.corpus_file,
                    runs=state.runs,
                    unet_sha256=opts["unet_sha256"],
                    lexicon=opts["lexicon"],
                    run_id=opts["run_id"],
                    write=opts["write"],
                    index=opts["index"],
                    text_missing=state.shot in opts["text_missing"],
                    descriptions=state.descriptions,
                )
                with time_limit(opts["timeout_s"]):
                    future = tails.submit(job)
            state.runs = []
            state.descriptions = []
            pending.append((state, future, job))
            # Bound queued payloads while keeping every tail worker fed.
            if len(pending) >= max(2, 2 * opts["tail_workers"]):
                flush()

    try:
        inferred = iter(
            masks.infer_pooled(
                opts["model"],
                prepared_blocks(),
                opts["device"],
                batch=opts["tile_batch"],
                amp=opts["amp"],
                forward_context=lambda tokens: guard([token[0] for token in tokens]),
                remaining=lambda token: remaining(token[0]),
                preserve_batches=False if opts["pool_cpu_forwards"] else None,
            )
        )
        while True:
            at = time.monotonic()
            output = next(inferred, None)
            if output is None:
                break
            (state, prepared), result = output
            state.timing.infer_s += time.monotonic() - at
            state.timing.n_tiles += prepared.n_tiles
            prepared.spectrogram = None
            at = time.monotonic()
            try:
                if isinstance(result, Exception):
                    raise result
                if state.res.error:
                    state.remaining -= 1
                    dispatch_ready()
                    continue
                if opts["tail_workers"]:
                    state.descriptions.append((prepared, result))
                else:
                    state.runs.append(
                        pipeline.describe_block(
                            prepared, result, unet_sha256=opts["unet_sha256"]
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - one descriptor failure
                if isinstance(exc, StageTimeout):
                    state.res.error = pipeline._cause(exc)
                state.res.skipped[f"mask {prepared.key}"] = pipeline._cause(exc)
            state.timing.describe_s += time.monotonic() - at
            state.remaining -= 1
            dispatch_ready()
            pool.sample()
            tails.sample()
        dispatch_ready()
        while pending:
            flush()
    finally:
        pool.close()
        tails.close()
    elapsed = time.monotonic() - started
    summary = pipeline.summarise(rows)
    summary.update(totals.as_row())
    summary.update(
        n_shots=len(rows),
        elapsed_s=round(elapsed, 2),
        seconds_per_shot=round(elapsed / len(rows), 2) if rows else 0.0,
        blocks_per_s=round(summary["n_blocks"] / elapsed, 3),
        tiles_per_s=round(totals.n_tiles / elapsed, 2),
        peak_rss_gib=round(_rss_gib(resource.RUSAGE_SELF), 3),
        peak_worker_rss_gib=round(
            max(pool.peak_worker_rss_gib, tails.peak_worker_rss_gib), 3
        ),
        worker_rss_gib={
            role: workers.worker_rss_gib
            for role, workers in [("prep", pool), ("tail", tails)]
        },
        cuda_max_alloc_gib=_cuda_peak_gib(opts["device"]),
        prep_workers=opts["prep_workers"],
        tail_workers=opts["tail_workers"],
        prefetch=opts["prefetch"],
        tile_batch=opts["tile_batch"],
        device=str(opts["device"]),
        pooled=True,
    )
    return DriverRun(rows, summary)


def run_shots(
    shots: Sequence[int],
    *,
    paths: Paths,
    model,
    device: str = "cpu",
    corpus_dir=None,
    plan: Sequence[channels.ChannelSpec] = channels.ROUND1_PLAN,
    passes: Sequence[str] = masks.PASS_NAMES,
    tile_batch: int = 96,
    amp: bool = False,
    norm: str = "record",
    prep_workers: int = 1,
    prefetch: int = 4,
    timeout_s: int = 300,
    write: bool = True,
    index: bool = True,
    lexicon=None,
    run_id: str = "manual",
    unet_sha256: str = CHECKPOINT_SHA256,
    skip_existing: bool = False,
    force: bool = False,
    tail_workers: int = 0,
    text_missing: Iterable[int] = (),
    pooled: bool = False,
    pool_cpu_forwards: bool = False,
    echo=print,
) -> DriverRun:
    """Every shot in `shots`, one prep pool, one model, one row each.

    The model is loaded and the logbook subset built by the CALLER, once,
    before this is entered - `main` does both - because a driver that loaded
    the network per shot would spend more time on `torch.load` than on the
    shots, and a `build_logs_subset` per shot would stream 616 MB per shot.

    `tail_workers=1` runs each shot's tail - `pipeline.finish_shot` - in a
    second, single-worker pool, so that it is paid beside the NEXT shot's
    forward passes instead of in front of them. The rows are still in shot
    order and every byte written is the same: one shot's tail touches only
    that shot's files and reads only that shot's corpus file, so two shots'
    tails and blocks are order-independent. At most one shot is unfinished
    at a time - shot i's tail is harvested as soon as shot i+1's blocks are
    done - which bounds the extra memory at one shot's `_BlockRun`s.
    `tail_workers=0` finishes each shot in this process before starting the
    next, under the shot's own alarm, which is what `process_shot` does.

    Nothing in here raises for one shot: a shot over its `--timeout` or one
    that failed in a way `process_shot` does not guard is one row with an
    `error`, and the run goes on. A hung or broken tail pool is replaced;
    an ordinary tail exception does not require a restart.
    """
    if str(norm) not in pipeline.NORMS:
        raise ValueError(f"norm must be one of {pipeline.NORMS}; got {norm!r}")
    passes = tuple(passes)
    bad = [p for p in passes if p not in masks.PASS_NAMES]
    if bad:
        raise ValueError(f"passes must be from {masks.PASS_NAMES}; got {bad}")

    if pool_cpu_forwards and (not pooled or str(device) != "cpu"):
        raise ValueError("pool_cpu_forwards requires pooled=True and device=cpu")
    if pooled:
        if tail_workers > 1 and index:
            raise ValueError("multiple tail workers require index=False / --no-index")
        options = {
            "paths": paths, "model": model, "device": device, "corpus_dir": corpus_dir,
            "plan": plan, "passes": passes, "tile_batch": tile_batch, "amp": amp, "norm": norm,
            "prep_workers": prep_workers, "prefetch": prefetch, "timeout_s": timeout_s,
            "write": write, "index": index, "lexicon": lexicon, "run_id": run_id,
            "unet_sha256": unet_sha256, "skip_existing": skip_existing, "force": force,
            "tail_workers": tail_workers, "text_missing": frozenset(text_missing),
            "pool_cpu_forwards": pool_cpu_forwards, "echo": echo,
        }
        return _run_pooled(shots, **options)

    started = time.monotonic()
    totals = ShotTiming()
    rows: list[dict] = []
    absent_text = frozenset(int(s) for s in text_missing)
    pool = PrepPool(prep_workers)
    tails = PrepPool(tail_workers)
    # The tail pool's spawn and torch import happen during the first shot's
    # blocks, not in front of the first shot's tail.
    tails.warm()
    pending: _Unfinished | None = None

    def flush(unfinished: _Unfinished | None) -> None:
        """Wait for one shot's tail, then write its row - in shot order."""
        if unfinished is None:
            return
        waited = time.monotonic()
        timing = unfinished.timing
        try:
            done = unfinished.future.result(timeout=max(1, int(timeout_s)))
            timing.tail_wait_s += time.monotonic() - waited
            timing.finish_s += done.seconds
            res = done.res
            res.elapsed_s = unfinished.blocks_s + done.seconds
            row = res.as_row()
            echo(res.line())
        except Exception as exc:  # noqa: BLE001 - per-shot isolation
            timing.tail_wait_s += time.monotonic() - waited
            row = {"shot": int(unfinished.shot), "status": "error",
                   "error": type(exc).__name__, "detail": str(exc)[:200]}
            echo(f"{unfinished.shot}: ERROR {type(exc).__name__}: "
                 f"{str(exc)[:120]}")
            # A payload or task exception leaves the executor usable. Only
            # a hung tail or a dead worker requires replacing the pool.
            wait_expired = (isinstance(exc, TimeoutError)
                            and not unfinished.future.done())
            if isinstance(exc, BrokenExecutor) or wait_expired:
                tails.restart()
                # The next shot was submitted before this flush. Preserve
                # its GPU work when close() cancels or breaks its future,
                # including a future already marked BrokenProcessPool.
                # A completed result or ordinary task error is still valid;
                # neither that tail nor the failing shot should run again.
                if pending is not None and pending is not unfinished:
                    future = pending.future
                    if (not future.done() or future.cancelled()
                            or isinstance(future.exception(), BrokenExecutor)):
                        pending.future = tails.submit(pending.job)
        row["seconds"] = round(unfinished.blocks_s + timing.finish_s, 2)
        row.update(timing.as_row())
        totals.add(timing)
        rows.append(row)

    try:
        for shot in shots:
            corpus_file = (
                Path(corpus_dir) / f"{int(shot)}_processed.h5"
                if corpus_dir is not None else paths.corpus_file(shot)
            )
            if (skip_existing and not force
                    and paths.masks_file(shot).exists()
                    and paths.events_file(shot).exists()):
                # In shot order: the previous shot's row first, whatever
                # this one turns out to be.
                flush(pending)
                pending = None
                echo(f"{shot}: skipped, masks and events already written")
                rows.append({
                    "shot": int(shot), "status": "skipped", "seconds": 0.0,
                    "skipped": {}, "by_source": {},
                    "error": "", "n_blocks": 0, "n_events": 0,
                })
                continue
            at = time.monotonic()
            handed_off = None
            try:
                with time_limit(timeout_s):
                    res, timing, runs = _shot_blocks(
                        shot, corpus_file=corpus_file,
                        model=model, pool=pool, plan=plan, passes=passes,
                        prefetch=prefetch, device=device,
                        tile_batch=tile_batch, amp=amp, norm=norm,
                        unet_sha256=unet_sha256,
                    )
                    if runs is not None:
                        # Inside the alarm on purpose: with `tail_workers=0`
                        # the submit IS the tail, and it keeps the shot's
                        # own timer, exactly as `process_shot` has it. With
                        # a worker the submit returns at once.
                        submitted = time.monotonic()
                        job = FinishJob(
                            res=res, paths=paths, corpus_file=corpus_file,
                            runs=runs, unet_sha256=unet_sha256,
                            lexicon=lexicon, run_id=run_id, write=write,
                            index=index,
                            text_missing=int(shot) in absent_text,
                        )
                        handed_off = tails.submit(job)
                        timing.tail_wait_s += time.monotonic() - submitted
                        del runs, res
            except Exception as exc:  # noqa: BLE001 - per-shot isolation
                flush(pending)
                pending = None
                timing = ShotTiming()
                row = {"shot": int(shot), "status": "error",
                       "error": type(exc).__name__, "detail": str(exc)[:200]}
                echo(f"{shot}: ERROR {type(exc).__name__}: {str(exc)[:120]}")
                # Whatever this shot left in flight has no reader now.
                pool.restart()
                row["seconds"] = round(time.monotonic() - at, 2)
                row.update(timing.as_row())
                totals.add(timing)
                rows.append(row)
                pool.sample()
                tails.sample()
                continue
            if handed_off is None:
                # The plan failed: one error row, no tail, as `process_shot`.
                flush(pending)
                pending = None
                res.elapsed_s = time.monotonic() - at
                row = res.as_row()
                echo(res.line())
                row["seconds"] = round(res.elapsed_s, 2)
                row.update(timing.as_row())
                totals.add(timing)
                rows.append(row)
            else:
                previous, pending = pending, _Unfinished(
                    shot=int(shot), timing=timing,
                    blocks_s=time.monotonic() - at, future=handed_off, job=job,
                )
                del job  # pending now owns the payload, including its blocks
                # The PREVIOUS shot's tail has been running beside this
                # shot's blocks; collecting it here is what keeps the rows
                # in shot order without ever waiting in front of the GPU.
                flush(previous)
                del previous  # release the completed shot before the next blocks
            # Both pools' high-water marks while their workers are still
            # alive to be asked (see `_vmhwm_gib`).
            pool.sample()
            tails.sample()
        flush(pending)
        pending = None
    finally:
        flush(pending)
        pool.close()
        tails.close()

    elapsed = time.monotonic() - started
    summary = pipeline.summarise(rows)
    summary.update(totals.as_row())
    summary.update(
        n_shots=len(rows),
        elapsed_s=round(elapsed, 2),
        seconds_per_shot=round(elapsed / len(rows), 2) if rows else 0.0,
        blocks_per_s=round(summary["n_blocks"] / elapsed, 3) if elapsed else 0.0,
        tiles_per_s=round(totals.n_tiles / elapsed, 2) if elapsed else 0.0,
        peak_rss_gib=round(_rss_gib(resource.RUSAGE_SELF), 3),
        peak_worker_rss_gib=round(max(pool.peak_worker_rss_gib,
                                     tails.peak_worker_rss_gib), 3),
        worker_rss_gib={
            role: {pid: round(rss, 3) for pid, rss in workers.worker_rss_gib.items()}
            for role, workers in (("prep", pool), ("tail", tails))
        },
        cuda_max_alloc_gib=_cuda_peak_gib(device),
        prep_workers=int(prep_workers),
        tail_workers=int(tail_workers),
        prefetch=int(prefetch),
        tile_batch=int(tile_batch),
        device=str(device),
    )
    return DriverRun(rows=rows, totals=summary)


# ------------------------------------------------------------------- the CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokeye_masks.py",
        description=(
            "Run the pinned TokEye U-Net over a chunk of a shot list, "
            "preparing the next blocks on CPU while the current one infers. "
            "Writes masks/<shot>_masks.npz and events/<shot>_events.parquet, "
            "the same files `python -m labeler.run events` writes."
        ),
    )
    # Not `required=True`: `--rebuild-index` runs over what is already on
    # disk and has no shot list. `main` refuses every other run without one.
    picker = parser.add_mutually_exclusive_group(required=False)
    picker.add_argument("--shot-file", type=Path,
                        help="one shot per line, # starting a comment")
    picker.add_argument("--shots", nargs="+", type=int, metavar="SHOT")
    parser.add_argument("--chunk", type=int, default=0,
                        help="which contiguous chunk of the sorted list this "
                             "array task does (default 0)")
    parser.add_argument("--n-chunks", type=int, default=1,
                        help="how many array tasks share the list (default 1)")
    parser.add_argument("--rank", type=int, default=None,
                        help="stride this rank out of the chunk; default "
                             "$SLURM_PROCID, else 0")
    parser.add_argument("--world", type=int, default=None,
                        help="how many ranks share the chunk; default "
                             "$SLURM_NTASKS, else 1")
    parser.add_argument("--limit", type=int, default=0,
                        help="keep only the first N shots OF THIS RANK's "
                             "share; 0 means all")
    parser.add_argument("--root", type=Path, default=None,
                        help="LABELER_ROOT: where masks/, events/ and "
                             "runs/ are written")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="directory of <shot>_processed.h5")
    parser.add_argument("--plan", default="round1", choices=sorted(PLANS),
                        help="which channel plan to run (default round1)")
    parser.add_argument("--passes", nargs="+", default=list(masks.PASS_NAMES),
                        choices=list(masks.PASS_NAMES),
                        help="wide, zoom, or both (default both)")
    parser.add_argument("--tile-batch", type=int, default=None,
                        help="512-column tiles per forward pass; default "
                             f"{TILE_BATCH_DEFAULT['cuda']} on cuda, "
                             f"{TILE_BATCH_DEFAULT['cpu']} on cpu")
    parser.add_argument("--prep-workers", type=int, default=None,
                        help="processes doing the CPU half of a block; 0 runs "
                             "it in this process. Default 1 on cpu; on cuda "
                             "$SLURM_CPUS_PER_TASK - 2, else "
                             f"{CUDA_WORKERS_FALLBACK}")
    parser.add_argument("--prefetch", type=int, default=4,
                        help="blocks in flight ahead of the network (default "
                             "4; ~262 MB each on the widest channel). Also "
                             "the concurrency: at most min(prefetch, "
                             "prep-workers) workers are ever busy")
    parser.add_argument("--tail-workers", type=int, default=None,
                        help="processes running each shot's TAIL "
                             "(the heuristics, the text and the two "
                             "writes) beside the next shot's forward "
                             "passes; 1 overlaps it, 0 runs it in this "
                             "process before the next shot starts. "
                             "Default 1 when --prep-workers is not 0")
    parser.add_argument("--amp", action="store_true",
                        help="fp16 autocast; CUDA only, ignored on cpu")
    parser.add_argument("--probs-on-host", action="store_true",
                        help="use the reference per-block full-probability "
                             "path instead of compact pooled inference")
    parser.add_argument("--pool-cpu-forwards", action="store_true",
                        help="opt in to cross-block CPU forward batches; exactness "
                             "is only measured with the pinned single-threaded setup")
    parser.add_argument("--norm", default="record",
                        choices=list(pipeline.NORMS),
                        help="record: z-score over the whole record, as the "
                             "AE path; plasma: over the fast groups' "
                             "coverage intersection")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds per shot before it is abandoned")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--unet", type=Path, default=None,
                        help="checkpoint; default "
                             "<root>/models/tokeye/big_tf_unet_251210.pt")
    parser.add_argument("--run-id", default=None,
                        help="names the run's JSON, which is written as "
                             "<run-id>_c<chunk>of<n>_r<rank>.json")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip a shot whose masks AND events files both "
                             "exist; --force overrides")
    parser.add_argument("--force", action="store_true",
                        help="redo shots --skip-existing would leave alone")
    parser.add_argument("--text-subset", default="auto",
                        choices=("auto", "build", "readonly"),
                        help="text/logs_subset.jsonl is one file for the "
                             "whole root and building it is a whole-file "
                             "rewrite, so concurrent tasks lose each "
                             "other's records. auto (default) builds it "
                             "for a single-task run and READS it when "
                             "--world or --n-chunks is more than 1; build "
                             "and readonly force either way")
    parser.add_argument("--build-text-subset", action="store_true",
                        help="the pre-pass: build text/logs_subset.jsonl "
                             "for the WHOLE shot list (not this chunk) and "
                             "exit. REQUIRED before multi-rank/multi-chunk "
                             "runs to preserve declared text sources; check "
                             "text_subset_missing in each run JSON")
    parser.add_argument("--refresh-text", action="store_true",
                        help="re-ask the logbook about the shots recorded in "
                             "text/logs_subset.missing")
    parser.add_argument("--no-index", action="store_true",
                        help="do not touch events_index.parquet: the one "
                             "file a whole root shares, which sixteen array "
                             "tasks cannot rewrite at once. Rebuild it "
                             "afterwards with --rebuild-index")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="regenerate events_index.parquet from "
                             "events/*_events.parquet in one pass and exit; "
                             "takes no shot list")
    parser.add_argument("--index-out", type=Path, default=None,
                        help="output path for --rebuild-index only; default "
                             "<root>/events_index.parquet")
    parser.add_argument("--no-write", action="store_true",
                        help="compute everything and store nothing; for a "
                             "pilot measuring throughput")
    return parser


def settle(args):
    """Fill in the defaults that depend on the device and on SLURM.

    In one place and after parsing, so that `--help` can describe them
    honestly (an `argparse` default computed at import time would print
    whatever the login node's environment happened to say) and a flag always
    wins over the environment.
    """
    if args.rank is None:
        args.rank = int(os.environ.get("SLURM_PROCID", "0"))
    if args.world is None:
        args.world = int(os.environ.get("SLURM_NTASKS", "1"))
    if args.prep_workers is None:
        if args.device != "cuda":
            # One worker: overlap enough to hide a read behind the previous
            # block, without asking a login node or a CPU job for cores it
            # was not given.
            args.prep_workers = 1
        else:
            cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            args.prep_workers = (
                max(1, int(cpus) - CORES_FOR_THE_PARENT) if cpus
                else CUDA_WORKERS_FALLBACK
            )
    if args.text_subset == "auto":
        # One task owns the subset; several share it, and a shared
        # whole-file rewrite is a lost update. See `--text-subset`.
        args.text_subset = ("readonly"
                            if int(args.world) > 1 or int(args.n_chunks) > 1
                            else "build")
    if args.tail_workers is None:
        # The tail is ~3.7 s of parent-thread work per shot against ~0.84 s
        # of A100 forward pass (measured), so not overlapping it caps GPU
        # utilisation near 15 %. Off only where there is no pool at all -
        # `--prep-workers 0` is the debugger's setting, and a second
        # process would be a second thing to step through.
        args.tail_workers = 0 if args.prep_workers == 0 else 1
    if args.tile_batch is None:
        args.tile_batch = TILE_BATCH_DEFAULT[args.device]
    if args.prefetch < args.prep_workers:
        # Not raised behind the caller's back - `--prefetch` is the memory
        # bound and multiplying it multiplies ~262 MB a block - but said
        # out loud, because the pairing in plan section 8 (18 workers,
        # prefetch 4) leaves fourteen workers idle and a pilot that did not
        # notice would measure the wrong cost model.
        print(f"tokeye_masks: {args.prep_workers - args.prefetch} of "
              f"{args.prep_workers} prep workers will be idle - --prefetch "
              f"is the concurrency as well as the queue", file=sys.stderr)
    return args


def run_tag(run_id: str, args) -> str:
    """`<run-id>_c<chunk>of<n>_r<rank>`: one run JSON per (task, rank).

    The chunk and the rank are in the NAME and not only in the payload
    because every task of an array writes into one `runs/events/`
    directory, and a name they shared would be a name they overwrote.
    """
    return f"{run_id}_c{int(args.chunk)}of{int(args.n_chunks)}_r{int(args.rank)}"


def main(argv=None) -> int:
    parser = build_parser()
    args = settle(parser.parse_args(argv))
    if (not args.probs_on_host and args.tail_workers > 1 and not args.no_index
            and not (args.rebuild_index or args.build_text_subset)):
        parser.error("multiple tail workers require --no-index")
    if args.pool_cpu_forwards and (args.device != "cpu" or args.probs_on_host):
        parser.error("--pool-cpu-forwards requires --device cpu and compact inference")
    if args.index_out is not None and not args.rebuild_index:
        parser.error("--index-out requires --rebuild-index")
    if not (args.rebuild_index or args.shot_file or args.shots):
        parser.error("one of --shot-file or --shots is required")
    base = Paths.from_env()
    paths = replace(
        base,
        root=args.root or base.root,
        corpus=args.corpus or base.corpus,
    )
    paths = replace(paths, root=paths.root.resolve())
    print(f"Resolved root: {paths.root}", flush=True)
    paths.mkdirs()

    if args.rebuild_index:
        if args.shot_file or args.shots:
            parser.error("--rebuild-index takes no shot list: it rebuilds "
                         "the index from every events file under --root")
        rebuild_index(paths, out=args.index_out)
        return EXIT_OK

    listed = (read_shot_file(args.shot_file) if args.shot_file
              else list(args.shots or []))
    if not listed:
        print("no shots selected", file=sys.stderr)
        return EXIT_NO_SHOTS

    if args.build_text_subset:
        # The whole list, once, in one process, before any array task
        # starts: after this every task's `wanted = asked - have` is empty
        # and nobody rewrites the file. `text_events` reads the subset per
        # shot and never the 616 MB source, so this is the only writer.
        n_added = text_weak.build_logs_subset(
            listed, paths=paths, refresh_missing=args.refresh_text
        )
        print(f"tokeye_masks: {n_added} logbook records added to "
              f"{paths.logs_subset} for {len(listed)} shots")
        return EXIT_OK
    mine = split_shots(listed, chunk=args.chunk, n_chunks=args.n_chunks,
                       rank=args.rank, world=args.world, limit=args.limit)
    span = f"{mine[0]}..{mine[-1]}" if mine else "nothing to do"
    print(
        f"tokeye_masks: {len(mine)} of {len(listed)} shots "
        f"[chunk {args.chunk}/{args.n_chunks}, rank {args.rank}/{args.world}] "
        f"{span} on {args.device}, {args.prep_workers} prep workers, "
        f"prefetch {args.prefetch}, tile-batch {args.tile_batch}"
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"masks-{stamp}-{os.getpid()}"
    n_added, text_note = 0, ""
    text_missing: list[int] = []
    result = DriverRun(rows=[], totals=pipeline.summarise([]))
    if mine:
        if args.text_subset == "build":
            try:
                n_added = text_weak.build_logs_subset(
                    mine, paths=paths, refresh_missing=args.refresh_text
                )
            except OSError as exc:
                # As `run.py`: the text is one of eight sources and the only
                # one that is not in the corpus. Losing it is not losing the
                # masks.
                text_note = f"{type(exc).__name__}: {exc}"
                print(f"tokeye_masks: no shot-scope text this run - "
                      f"{text_note}", file=sys.stderr)
        else:
            # Read-only: the subset is not written at all, so this task
            # cannot lose another task's records. What it CAN do is say
            # which of its shots the pre-pass did not cover, per shot and
            # in the run record, instead of leaving them to look like
            # shots the logbook has nothing about.
            text_missing = [s for s in mine
                            if text_weak.load_log_record(s, paths=paths) is None]
            if text_missing:
                text_note = (
                    f"read-only text subset: {len(text_missing)} of "
                    f"{len(mine)} shots have no record in "
                    f"{paths.logs_subset}"
                )
                print(f"tokeye_masks: {text_note}", file=sys.stderr)
        model = unet.load_unet(args.unet, device=args.device)
        result = run_shots(
            mine, paths=paths, model=model, device=args.device,
            corpus_dir=paths.corpus, plan=PLANS[args.plan],
            passes=tuple(args.passes), tile_batch=args.tile_batch,
            amp=args.amp, norm=args.norm, prep_workers=args.prep_workers,
            prefetch=args.prefetch, timeout_s=args.timeout,
            write=not args.no_write, index=not args.no_index,
            lexicon=load_lexicon(), run_id=run_id,
            skip_existing=args.skip_existing, force=args.force,
            tail_workers=args.tail_workers, text_missing=text_missing,
            pooled=not args.probs_on_host,
            pool_cpu_forwards=args.pool_cpu_forwards,
        )

    totals = result.totals
    out = write_events_run(paths, run_tag(run_id, args), {
        "run_id": run_id,
        "stage": "events",
        "driver": "tokeye_masks",
        "git_sha": git_sha(),
        "labelmaker_version": __version__,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {
            "chunk": int(args.chunk),
            "n_chunks": int(args.n_chunks),
            "rank": int(args.rank),
            "world": int(args.world),
            "device": args.device,
            "plan": args.plan,
            "passes": list(args.passes),
            "tile_batch": int(args.tile_batch),
            "prep_workers": int(args.prep_workers),
            "tail_workers": int(args.tail_workers),
            "prefetch": int(args.prefetch),
            "amp": bool(args.amp),
            "pool_cpu_forwards": bool(args.pool_cpu_forwards),
            "norm": args.norm,
            "timeout_s": int(args.timeout),
            "limit": int(args.limit),
            "write": not args.no_write,
            "index": "skipped" if args.no_index else "per-shot",
            "skip_existing": bool(args.skip_existing),
            "force": bool(args.force),
            "refresh_text": bool(args.refresh_text),
            "text_subset": args.text_subset,
            "unet": str(args.unet) if args.unet else "",
            "root": str(paths.root),
            "corpus": str(paths.corpus),
        },
        "n_shots": len(mine),
        "shots_selected": mine,
        "n_log_records_added": int(n_added),
        "logs_subset": str(paths.logs_subset),
        "text_subset_missing": text_missing,
        "text_note": text_note,
        "totals": totals,
        "shots": result.rows,
    })
    print(
        f"tokeye_masks: {totals.get('n_shots', 0)} shots, "
        f"{totals.get('n_blocks', 0)} blocks, {totals.get('n_events', 0)} events "
        f"in {totals.get('elapsed_s', 0.0)}s "
        f"({totals.get('seconds_per_shot', 0.0)}s/shot, "
        f"{totals.get('tiles_per_s', 0.0)} tiles/s); "
        f"prep wait {totals.get('prep_wait_s', 0.0)}s, "
        f"infer {totals.get('infer_s', 0.0)}s, "
        f"describe {totals.get('describe_s', 0.0)}s, "
        f"finish {totals.get('finish_s', 0.0)}s "
        f"(waited {totals.get('tail_wait_s', 0.0)}s), "
        f"peak RSS {totals.get('peak_rss_gib', 0.0)} GiB"
    )
    print(f"tokeye_masks: {out}")
    return EXIT_OK


__all__ = [
    "INDEX_KEYS",
    "DriverRun",
    "FinishJob",
    "PrepJob",
    "PrepPool",
    "build_parser",
    "in_order",
    "main",
    "rebuild_index",
    "run_shots",
    "settle",
    "split_shots",
]
