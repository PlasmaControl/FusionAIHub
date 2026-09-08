"""The batch driver: `events/driver.py` and `scripts/labelmaker/tokeye_masks.py`.

Hermetic. The corpus is `test_events_pipeline.py`'s synthetic shot written into
`tmp_path`, the network is that module's `PaintedNet`, and every assertion here
is about the DRIVER - how a shot list is split across array tasks and ranks, how
the prepared blocks are kept flowing without unbounding memory, what a failure in
a prep worker costs, and above all that the driver's outputs are the ones
`pipeline.process_shot` would have written.

Identity is the point of the file. `process_shot` stays the single definition of
a shot's work; the driver only re-orders WHEN the CPU part of a block happens.
So the test that matters compares the two byte for byte - every array in
`masks/<shot>_masks.npz` with `np.array_equal`, every event row bar the two
columns that record when and under which run id it was written.
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from labelmaker.config import Paths
from labelmaker.events import channels, driver, masks
from labelmaker.events import pipeline as pl

from .test_events_pipeline import FAKE_SHA, SHOT, PaintedNet, _write_corpus

#: A second synthetic shot, so a split has something to split.
SHOTS = [SHOT, SHOT + 1, SHOT + 2]


@pytest.fixture
def paths(tmp_path):
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "bundles",
        logs_jsonl=tmp_path / "logs.jsonl",
    )


@pytest.fixture
def model():
    return PaintedNet().eval()


# ------------------------------------------------------- the stage functions


def test_the_three_stages_compose_back_into_one_block(paths, synth_shot, model):
    """`prep_block` + `infer_block` + `describe_block` IS `_one_block`.

    The driver runs the three on three different schedules - prep in a pool,
    infer on the GPU, describe in the parent - so if the composition ever
    stopped being the whole of `_one_block` the two paths would drift and
    every identity assertion below would be measuring the wrong thing.
    """
    _write_corpus(paths.corpus, SHOT, synth_shot)
    spec = channels.ChannelSpec("mhr", 0, "magnetics")
    y, fs_hz, t0_s, t1_s = masks.read_waveform(paths.corpus_file(SHOT), "mhr", 0)

    skipped: dict[str, str] = {}
    whole = pl._one_block(
        y, fs_hz, t0_s, t1_s, spec, "wide", model=model, device="cpu",
        tile_batch=4, amp=False, norm="record", window=None,
        unet_sha256=FAKE_SHA, skipped=skipped,
    )
    prepared = pl.prep_block(y, fs_hz, t0_s, t1_s, spec, "wide", norm="record",
                             window=None)
    probs = pl.infer_block(prepared, model=model, device="cpu", tile_batch=4,
                           amp=False)
    piecewise = pl.describe_block(prepared, probs, unet_sha256=FAKE_SHA)

    assert skipped == {}
    assert piecewise.prefix == whole.prefix
    assert set(piecewise.arrays) == set(whole.arrays)
    for key, value in whole.arrays.items():
        assert np.array_equal(np.asarray(piecewise.arrays[key]),
                              np.asarray(value)), key
    assert [t.__dict__ for t in piecewise.tracks] == [
        t.__dict__ for t in whole.tracks
    ]
    assert np.array_equal(piecewise.activity, whole.activity)
    assert np.array_equal(piecewise.t_s, whole.t_s)


def test_the_norm_fallback_is_reported_and_not_swallowed(paths, synth_shot,
                                                          model):
    # `_one_block` records the fall back to `record` in `skipped`; the stage
    # function has no `skipped` dict to write into, so it carries the same
    # sentence out on the prepared block and the caller records it.
    _write_corpus(paths.corpus, SHOT, synth_shot)
    spec = channels.ChannelSpec("mhr", 0, "magnetics")
    y, fs_hz, t0_s, t1_s = masks.read_waveform(paths.corpus_file(SHOT), "mhr", 0)
    prepared = pl.prep_block(y, fs_hz, t0_s, t1_s, spec, "wide", norm="plasma",
                             window=None)
    assert "whole record" in prepared.norm_note
    assert prepared.meta["norm"] == "record"

    skipped: dict[str, str] = {}
    pl._one_block(y, fs_hz, t0_s, t1_s, spec, "wide", model=model,
                  device="cpu", tile_batch=4, amp=False, norm="plasma",
                  window=None, unet_sha256=FAKE_SHA, skipped=skipped)
    assert skipped == {"norm mhr:0:wide": prepared.norm_note}


def test_the_tile_count_is_known_without_cutting_the_tiles(paths):
    # The driver reports tiles/s, which means it has to count a block's tiles
    # before the GPU sees them and without building the (n, 512, 512) array.
    for n_cols in (1, 511, 512, 513, 960, 16391, 64000):
        tiles, _ = masks.tile(np.zeros((masks.N_BINS, n_cols), dtype=np.float32))
        assert masks.n_tiles(n_cols) == tiles.shape[0], n_cols


# ------------------------------------------------------- chunk / rank / limit


def test_every_shot_is_done_exactly_once_across_the_chunks_and_ranks():
    # The whole point of the two splits: an array of `n_chunks` tasks, each
    # of `world` ranks, must between them do every shot and no shot twice.
    shots = list(range(180001, 180018))            # 17, a prime
    seen: list[int] = []
    for chunk in range(4):
        for rank in range(3):
            seen.extend(driver.split_shots(shots, chunk=chunk, n_chunks=4,
                                           rank=rank, world=3))
    assert sorted(seen) == shots
    assert len(seen) == len(set(seen))


def test_the_list_is_sorted_and_deduplicated_before_it_is_split():
    # A shot file with a repeat would otherwise put one shot in two chunks,
    # and two tasks writing one `<shot>_masks.npz` is a race.
    shots = [4, 2, 9, 2, 7]
    assert driver.split_shots(shots, chunk=0, n_chunks=1) == [2, 4, 7, 9]
    whole = [
        s
        for chunk in range(2)
        for s in driver.split_shots(shots, chunk=chunk, n_chunks=2)
    ]
    assert whole == [2, 4, 7, 9]


def test_an_empty_subset_is_legal():
    # More tasks than shots is what the tail of an array looks like, and a
    # task with nothing to do is not a failure.
    assert driver.split_shots([1, 2], chunk=0, n_chunks=4) == []
    assert driver.split_shots([1, 2], chunk=0, n_chunks=1, rank=3, world=4) == []
    assert driver.split_shots([], chunk=0, n_chunks=1) == []


def test_the_chunk_is_contiguous_and_the_rank_is_strided():
    # Contiguous chunks so two array tasks walk two regions of the corpus
    # directory (`ideate.design.seed.chunk_of`'s reason); strided ranks so
    # the two ranks of ONE task, which share a node and its page cache,
    # interleave.
    shots = list(range(10))
    assert driver.split_shots(shots, chunk=0, n_chunks=2) == [0, 1, 2, 3, 4]
    assert driver.split_shots(shots, chunk=0, n_chunks=2, rank=1, world=2) == [
        1, 3
    ]


def test_limit_is_applied_after_the_split():
    # `--limit 2` is "this task does two shots", not "the run does two".
    shots = list(range(20))
    assert driver.split_shots(shots, chunk=1, n_chunks=2, limit=3) == [10, 11, 12]
    assert driver.split_shots(shots, chunk=1, n_chunks=2, rank=1, world=2,
                              limit=2) == [11, 13]


@pytest.mark.parametrize("kw", [
    {"chunk": 2, "n_chunks": 2}, {"chunk": -1, "n_chunks": 2},
    {"n_chunks": 0}, {"rank": 2, "world": 2}, {"world": 0},
])
def test_a_split_that_would_lose_shots_is_refused(kw):
    with pytest.raises(ValueError):
        driver.split_shots([1, 2, 3], **kw)


# ----------------------------------------------------------- bounded prefetch


def test_no_more_than_prefetch_blocks_are_ever_in_flight():
    """The memory bound, measured rather than asserted in a docstring.

    A prepared `mirnov` wide block is 262 MB, so "prepare the next one while
    the GPU works" has to mean `--prefetch` of them and not the whole shot's
    eleven channels. The probe counts live jobs: submitted and not yet
    consumed.
    """
    live = 0
    peak = 0
    lock = __import__("threading").Lock()

    def slow(job):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.01)
        return job

    with ThreadPoolExecutor(max_workers=8) as pool:
        got = []
        stream = driver.in_order(range(24), lambda job: pool.submit(slow, job),
                                 prefetch=3)
        for job, future in stream:
            assert future.result() == job
            with lock:
                live -= 1
            got.append(job)
    assert got == list(range(24))                  # in order, every one
    assert peak <= 3


def test_in_order_yields_nothing_for_no_jobs():
    assert list(driver.in_order([], lambda job: None, prefetch=4)) == []


def test_in_order_holds_one_job_when_prefetch_is_zero():
    # `--prep-workers 0` runs prep in this process, where preparing ahead
    # buys nothing and costs a block's 262 MB.
    submitted: list[int] = []

    def submit(job):
        submitted.append(job)
        future = Future()
        future.set_result(job)
        return future

    stream = driver.in_order([1, 2, 3], submit, prefetch=0)
    next(stream)
    assert submitted == [1]


# ------------------------------------------------------------------- the CLI


def test_help_names_every_flag():
    text = driver.build_parser().format_help()
    for flag in (
        "--shot-file", "--shots", "--chunk", "--n-chunks", "--rank", "--world",
        "--limit", "--root", "--corpus", "--plan", "--passes", "--tile-batch",
        "--prep-workers", "--prefetch", "--amp", "--norm", "--timeout",
        "--device", "--unet", "--run-id", "--skip-existing", "--force",
        "--refresh-text", "--no-write",
    ):
        assert flag in text, flag


def test_the_shot_pickers_are_exclusive_and_one_is_required():
    parser = driver.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--root", "/tmp"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--shots", "1", "--shot-file", "/tmp/x"])


def test_the_rank_and_world_default_to_the_srun_environment(monkeypatch):
    # One `srun` step, two tasks, one GPU each: the driver must split itself
    # without the sbatch having to pass `$SLURM_PROCID` twice.
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "20")
    args = driver.settle(driver.build_parser().parse_args(["--shots", "1"]))
    assert (args.rank, args.world) == (1, 2)
    assert args.prep_workers == 1                  # cpu default
    args = driver.settle(driver.build_parser().parse_args(
        ["--shots", "1", "--device", "cuda"]
    ))
    assert args.prep_workers == 18                 # cpus-per-task - 2
    assert args.tile_batch == 96


def test_the_flags_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    args = driver.settle(driver.build_parser().parse_args(
        ["--shots", "1", "--rank", "0", "--world", "1", "--prep-workers", "3"]
    ))
    assert (args.rank, args.world, args.prep_workers) == (0, 1, 3)
