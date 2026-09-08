"""Many shots on one GPU: the batch driver behind `tokeye_masks.py`.

`python -m labelmaker.run events` is a sequential loop - read, transform,
infer, describe, one channel after another - and on a GPU that is the 1 %
utilisation the plan measured (V8): a shot is ~15 core-seconds of HDF5 read
and STFT against ~0.84 s of A100 forward pass. This module is the same work
re-ordered so the card is busy, and nothing else:

* **The list is split twice.** `--chunk K --n-chunks N` cuts a CONTIGUOUS
  slice of the sorted, de-duplicated shot list for one SLURM array task, so
  two tasks walk two regions of the corpus directory rather than competing
  for the same GPFS blocks; `--rank R --world W` then STRIDES that slice
  across the ranks of one `srun` step, whose tasks share a node and its page
  cache. Between them, every shot is done exactly once
  (`test_tokeye_masks.py` pins that over all (chunk, rank) pairs).
* **The CPU part runs somewhere else.** `--prep-workers` spawned processes
  run `pipeline.plan_shot` and `pipeline.prep_block` - the header reads, the
  waveform read, the STFT and the standardisation - while this process runs
  `pipeline.infer_block` on the block before. The parent consumes prepared
  blocks IN ORDER, so what reaches `pipeline.finish_shot` is the plan order
  `process_shot` would have produced.
* **`process_shot` stays the definition.** This module owns scheduling and
  nothing else: the three stage functions, `plan_shot` and `finish_shot` are
  `pipeline.py`'s, and a mask file or an events table written here is
  required to be byte-identical to the one the sequential path writes. The
  one thing deliberately NOT done is pooling tiles from two blocks into one
  forward pass: `masks.infer` already batches a block's tiles to
  `--tile-batch` (a `mirnov` wide pass is 143 of them), and re-cutting the
  batches across blocks would change which batch a tile is run in, which is
  the one thing that could make these outputs differ in the last bits.

**Memory.** At most `--prefetch` prepared blocks are held at once, plus the
one in the network. A prepared block is two `(512, T)` float32 arrays - the
standardised spectrogram and the pre-standardisation log-power - which on
the widest block (`mirnov` wide, 64,000 columns) is 131 MB each, so
`--prefetch 4` bounds the queue at ~1.05 GB and the spectrogram of the block
being inferred is dropped as soon as the network has read it. The prep
workers add one such pair each while they work.

**Isolation.** Per shot: one `SIGALRM` (`--timeout`) and one try/except, as
`run.py`'s stages. The alarm ends the shot wherever the driver itself is
waiting - on a prepared block, in the network, in a describe step. Inside
`pipeline.finish_shot` it behaves as it does under `run.py`: that step's own
guard turns it into that step's skip and the shot finishes without a timer,
which is `process_shot`'s long-standing behaviour and not something a
scheduler may quietly change. A block whose prep raised - a `co2` channel whose
digitiser gapped, a worker that ran out of memory - is a `skipped` entry and
the shot goes on, exactly as in `process_shot`. A prep worker that DIES
(the OOM killer, a segfault) breaks the pool: the block in flight is
recorded as a skip, the pool is restarted, and the shot's remaining blocks
are re-submitted to it. Nothing waits forever for a process that is gone.
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
from dataclasses import dataclass, replace
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
from . import channels, masks, pipeline, text_weak, unet
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
    try:
        y, fs_hz, t0_s, t1_s = masks.read_waveform(
            job.corpus_file, job.spec.diag, job.spec.channel
        )
    except Exception as exc:  # noqa: BLE001 - per-channel isolation
        raise BlockFailed("read", pipeline._cause(exc)) from None
    try:
        return pipeline.prep_block(
            y, fs_hz, t0_s, t1_s, job.spec, job.pass_name,
            norm=job.norm, window=job.window,
        )
    except Exception as exc:  # noqa: BLE001 - per-pass isolation
        raise BlockFailed("mask", pipeline._cause(exc)) from None


def run_job(job):
    """Dispatch, so a pool holds one picklable entry point and not two."""
    return plan_one(job) if isinstance(job, PlanJob) else prep_one(job)


def _init_worker() -> None:
    """One thread per prep worker.

    Eighteen workers share a node with two GPU tasks (spec A5), and a
    prep worker that helped itself to the node's cores would leave the
    other task's workers waiting. Belt and braces rather than the whole
    story: the sbatch exports `OMP_NUM_THREADS=1` before python starts,
    which is the only moment it is guaranteed to be read, and the prep work
    itself - `ShortTimeFFT`, `signal.decimate`, a percentile and a z-score -
    is single-threaded numpy and scipy with no BLAS call in it.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001, S110 - a thread count is not worth a worker
        pass


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
        self._pool: ProcessPoolExecutor | None = None
        self._start()

    def _start(self) -> None:
        if self.workers > 0:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
            )

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
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


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
    """

    prep_wait_s: float = 0.0
    infer_s: float = 0.0
    describe_s: float = 0.0
    n_tiles: int = 0

    def as_row(self) -> dict[str, float | int]:
        return {
            "prep_wait_s": round(self.prep_wait_s, 3),
            "infer_s": round(self.infer_s, 3),
            "describe_s": round(self.describe_s, 3),
            "n_tiles": int(self.n_tiles),
        }

    def add(self, other: ShotTiming) -> None:
        self.prep_wait_s += other.prep_wait_s
        self.infer_s += other.infer_s
        self.describe_s += other.describe_s
        self.n_tiles += other.n_tiles


def _one_shot(
    shot: int,
    *,
    paths: Paths,
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
    write: bool,
    lexicon,
    run_id: str,
    unet_sha256: str,
) -> tuple[pipeline.ShotResult, ShotTiming]:
    """One shot, prepared in the pool and inferred here.

    The same sequence of decisions as `process_shot` - plan, then a block
    per `(channel, pass)` in plan order, then everything that is not a block
    - with the CPU half of each block moved off this process. Every failure
    is recorded under the key `process_shot` records it under, because the
    two are required to produce the same `ShotResult`.
    """
    started = time.monotonic()
    res = pipeline.ShotResult(shot=int(shot))
    timing = ShotTiming()

    try:
        specs, planned, window = pool.submit(
            PlanJob(corpus_file, tuple(plan), norm)
        ).result()
    except BlockFailed as exc:
        # The one run-ending failure: a corpus file that cannot be read.
        res.error = exc.cause
        res.elapsed_s = time.monotonic() - started
        return res, timing
    except BrokenExecutor as exc:
        pool.restart()
        res.error = pipeline._cause(exc)
        res.elapsed_s = time.monotonic() - started
        return res, timing
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

    pipeline.finish_shot(res, paths, corpus_file, runs,
                         unet_sha256=unet_sha256, lexicon=lexicon,
                         run_id=run_id, write=write)
    res.elapsed_s = time.monotonic() - started
    return res, timing


@dataclass
class DriverRun:
    """What a driver invocation produced: the per-shot rows and the totals."""

    rows: list[dict]
    totals: dict


def _rss_gib(who: int) -> float:
    """Peak resident set of this process (or of its children), in GiB.

    `ru_maxrss` is in kibibytes on Linux. For `RUSAGE_CHILDREN` it is the
    largest peak of any ONE waited-for child, which is what sizing
    `--mem-per-cpu` off a prep worker wants - not the sum, which nobody
    ever holds at once.
    """
    return resource.getrusage(who).ru_maxrss / (1024.0 * 1024.0)


def _cuda_peak_gib(device: str) -> float | None:
    if str(device) != "cuda":
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / float(1 << 30)


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
    lexicon=None,
    run_id: str = "manual",
    unet_sha256: str = CHECKPOINT_SHA256,
    skip_existing: bool = False,
    force: bool = False,
    echo=print,
) -> DriverRun:
    """Every shot in `shots`, one prep pool, one model, one row each.

    The model is loaded and the logbook subset built by the CALLER, once,
    before this is entered - `main` does both - because a driver that loaded
    the network per shot would spend more time on `torch.load` than on the
    shots, and a `build_logs_subset` per shot would stream 616 MB per shot.

    Nothing in here raises for one shot: a shot over its `--timeout` or one
    that failed in a way `process_shot` does not guard is one row with an
    `error`, the pool is replaced, and the run goes on.
    """
    if str(norm) not in pipeline.NORMS:
        raise ValueError(f"norm must be one of {pipeline.NORMS}; got {norm!r}")
    passes = tuple(passes)
    bad = [p for p in passes if p not in masks.PASS_NAMES]
    if bad:
        raise ValueError(f"passes must be from {masks.PASS_NAMES}; got {bad}")

    started = time.monotonic()
    totals = ShotTiming()
    rows: list[dict] = []
    pool = PrepPool(prep_workers)
    try:
        for shot in shots:
            corpus_file = (
                Path(corpus_dir) / f"{int(shot)}_processed.h5"
                if corpus_dir is not None else paths.corpus_file(shot)
            )
            if (skip_existing and not force
                    and paths.masks_file(shot).exists()
                    and paths.events_file(shot).exists()):
                echo(f"{shot}: skipped, masks and events already written")
                rows.append({
                    "shot": int(shot), "status": "skipped", "seconds": 0.0,
                    "skipped": {}, "by_source": {},
                    "error": "", "n_blocks": 0, "n_events": 0,
                })
                continue
            at = time.monotonic()
            try:
                with time_limit(timeout_s):
                    res, timing = _one_shot(
                        shot, paths=paths, corpus_file=corpus_file,
                        model=model, pool=pool, plan=plan, passes=passes,
                        prefetch=prefetch, device=device,
                        tile_batch=tile_batch, amp=amp, norm=norm,
                        write=write, lexicon=lexicon, run_id=run_id,
                        unet_sha256=unet_sha256,
                    )
                row = res.as_row()
                echo(res.line())
            except Exception as exc:  # noqa: BLE001 - per-shot isolation
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
    finally:
        pool.close()

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
        peak_worker_rss_gib=round(_rss_gib(resource.RUSAGE_CHILDREN), 3),
        cuda_max_alloc_gib=_cuda_peak_gib(device),
        prep_workers=int(prep_workers),
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
            "the same files `python -m labelmaker.run events` writes."
        ),
    )
    picker = parser.add_mutually_exclusive_group(required=True)
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
                        help="LABELMAKER_ROOT: where masks/, events/ and "
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
                        help="prepared blocks held ahead of the network "
                             "(default 4; ~262 MB each on the widest channel)")
    parser.add_argument("--amp", action="store_true",
                        help="fp16 autocast; CUDA only, ignored on cpu")
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
    parser.add_argument("--refresh-text", action="store_true",
                        help="re-ask the logbook about the shots recorded in "
                             "text/logs_subset.missing")
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
    if args.tile_batch is None:
        args.tile_batch = TILE_BATCH_DEFAULT[args.device]
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
    return args


def run_tag(run_id: str, args) -> str:
    """`<run-id>_c<chunk>of<n>_r<rank>`: one run JSON per (task, rank).

    The chunk and the rank are in the NAME and not only in the payload
    because every task of an array writes into one `runs/events/`
    directory, and a name they shared would be a name they overwrote.
    """
    return f"{run_id}_c{int(args.chunk)}of{int(args.n_chunks)}_r{int(args.rank)}"


def main(argv=None) -> int:
    args = settle(build_parser().parse_args(argv))
    base = Paths.from_env()
    paths = replace(
        base,
        root=args.root or base.root,
        corpus=args.corpus or base.corpus,
    )
    paths.mkdirs()

    listed = (read_shot_file(args.shot_file) if args.shot_file
              else list(args.shots or []))
    if not listed:
        print("no shots selected", file=sys.stderr)
        return EXIT_NO_SHOTS
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
    result = DriverRun(rows=[], totals=pipeline.summarise([]))
    if mine:
        try:
            n_added = text_weak.build_logs_subset(
                mine, paths=paths, refresh_missing=args.refresh_text
            )
        except OSError as exc:
            # As `run.py`: the text is one of eight sources and the only one
            # that is not in the corpus. Losing it is not losing the masks.
            text_note = f"{type(exc).__name__}: {exc}"
            print(f"tokeye_masks: no shot-scope text this run - {text_note}",
                  file=sys.stderr)
        model = unet.load_unet(args.unet, device=args.device)
        result = run_shots(
            mine, paths=paths, model=model, device=args.device,
            corpus_dir=paths.corpus, plan=PLANS[args.plan],
            passes=tuple(args.passes), tile_batch=args.tile_batch,
            amp=args.amp, norm=args.norm, prep_workers=args.prep_workers,
            prefetch=args.prefetch, timeout_s=args.timeout,
            write=not args.no_write, lexicon=load_lexicon(), run_id=run_id,
            skip_existing=args.skip_existing, force=args.force,
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
            "prefetch": int(args.prefetch),
            "amp": bool(args.amp),
            "norm": args.norm,
            "timeout_s": int(args.timeout),
            "limit": int(args.limit),
            "write": not args.no_write,
            "skip_existing": bool(args.skip_existing),
            "force": bool(args.force),
            "refresh_text": bool(args.refresh_text),
            "unet": str(args.unet) if args.unet else "",
            "root": str(paths.root),
            "corpus": str(paths.corpus),
        },
        "n_shots": len(mine),
        "shots_selected": mine,
        "n_log_records_added": int(n_added),
        "logs_subset": str(paths.logs_subset),
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
        f"peak RSS {totals.get('peak_rss_gib', 0.0)} GiB"
    )
    print(f"tokeye_masks: {out}")
    return EXIT_OK


__all__ = [
    "DriverRun",
    "PrepJob",
    "PrepPool",
    "build_parser",
    "in_order",
    "main",
    "run_shots",
    "settle",
    "split_shots",
]
