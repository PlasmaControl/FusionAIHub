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

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import h5py
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from labelmaker.config import Paths
from labelmaker.events import channels, driver, masks, schema, text_weak, unet
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


# --------------------------------------------------------------- the identity


def _paths_under(tmp_path, name, corpus):
    out = Paths(root=tmp_path / name, corpus=corpus,
                text_root=tmp_path / "bundles",
                logs_jsonl=tmp_path / "logs.jsonl")
    out.mkdirs()
    return out


def _mask_keys(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def _events(paths, shot):
    """The shot's events, without the two columns that say WHEN it ran."""
    df = schema.read_events(paths.events_file(shot))
    return df.drop(columns=["run_id", "written_at"])


def _silently(*args, **kwargs):
    return None


@pytest.mark.parametrize(("workers", "prefetch"), [(0, 1), (1, 4), (2, 2)])
def test_the_driver_writes_exactly_what_process_shot_writes(
    tmp_path, synth_shot, model, workers, prefetch,
):
    """The requirement the whole task turns on.

    Same shot, same network, same flags, two schedules: `process_shot`'s
    sequential loop and the driver's pool. Every array of the masks file and
    every event row has to be the same, because the driver is a re-ordering
    of when the CPU work happens and nothing else. Run over both passes, so
    the decimated zoom transform is in it, and over three
    (`--prep-workers`, `--prefetch`) settings, including the in-process one.
    """
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    sequential = _paths_under(tmp_path, "sequential", corpus)
    driven = _paths_under(tmp_path, "driven", corpus)

    ref = pl.process_shot(
        SHOT, sequential, model=model, device="cpu", passes=("wide", "zoom"),
        tile_batch=4, run_id="test-run", unet_sha256=FAKE_SHA,
    )
    got = driver.run_shots(
        [SHOT], paths=driven, model=model, device="cpu",
        passes=("wide", "zoom"), tile_batch=4, prep_workers=workers,
        prefetch=prefetch, run_id="test-run", unet_sha256=FAKE_SHA,
        echo=_silently,
    )

    assert ref.error == "" and ref.n_blocks == 14
    row = dict(got.rows[0])
    # The two timings and the wall clock are the only rows that may differ:
    # they are what the driver exists to change.
    for key in ("seconds", "prep_wait_s", "infer_s", "describe_s", "n_tiles"):
        row.pop(key, None)
    assert row == {k: v for k, v in ref.as_row().items() if k != "seconds"}

    mine, theirs = _mask_keys(driven.masks_file(SHOT)), _mask_keys(
        sequential.masks_file(SHOT))
    assert list(mine) == list(theirs)               # the same keys, in order
    for key, value in theirs.items():
        assert np.array_equal(mine[key], value), key

    pd.testing.assert_frame_equal(_events(driven, SHOT),
                                  _events(sequential, SHOT))
    assert len(_events(driven, SHOT)) == ref.n_events


def test_a_block_whose_prep_fails_in_a_worker_is_a_skip_and_not_a_hang(
    tmp_path, synth_shot, model,
):
    """A prep worker that raises costs its block, and only its block.

    `co2` truncated to three samples is a real version of what a gapped
    digitiser does: `plan_for` accepts the group (it has more than the
    absent-signal sentinel's one sample), `read_waveform` returns a
    waveform, and the transform is what raises - in the WORKER, where the
    parent can only see it as a returned failure. Both channels of both
    passes are lost and the other five blocks are not, exactly as in
    `process_shot`, and the run reaches the end rather than waiting for a
    result that is never coming.
    """
    corpus = tmp_path / "corpus"
    path = _write_corpus(corpus, SHOT, synth_shot)
    with h5py.File(path, "a") as f:
        del f["co2"]
        g = f.create_group("co2")
        g.create_dataset("xdata", data=np.arange(3, dtype=np.float32))
        g.create_dataset("ydata", data=np.zeros((4, 3), dtype=np.float32))

    sequential = _paths_under(tmp_path, "sequential", corpus)
    driven = _paths_under(tmp_path, "driven", corpus)
    ref = pl.process_shot(SHOT, sequential, model=model, device="cpu",
                          passes=("wide",), tile_batch=4, run_id="test-run",
                          unet_sha256=FAKE_SHA)
    got = driver.run_shots([SHOT], paths=driven, model=model, device="cpu",
                           passes=("wide",), tile_batch=4, prep_workers=1,
                           prefetch=2, run_id="test-run",
                           unet_sha256=FAKE_SHA, echo=_silently)

    assert ref.n_blocks == 5                        # mhr 0/4, ece 8/20/40
    assert got.rows[0]["n_blocks"] == 5
    assert got.rows[0]["status"] == "ok"
    assert got.rows[0]["skipped"] == ref.as_row()["skipped"]
    assert [k for k in got.rows[0]["skipped"] if k.startswith(("read co2",
                                                               "mask co2"))]
    for key, value in _mask_keys(sequential.masks_file(SHOT)).items():
        assert np.array_equal(_mask_keys(driven.masks_file(SHOT))[key], value)


def test_an_unreadable_corpus_file_is_one_error_row_and_the_run_goes_on(
    tmp_path, synth_shot, model,
):
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / f"{SHOT + 1}_processed.h5").write_bytes(b"not an hdf5 file")
    paths = _paths_under(tmp_path, "root", corpus)

    got = driver.run_shots([SHOT, SHOT + 1], paths=paths, model=model,
                           device="cpu", passes=("wide",), tile_batch=4,
                           prep_workers=0, unet_sha256=FAKE_SHA,
                           echo=_silently)
    rows = {row["shot"]: row for row in got.rows}
    assert rows[SHOT]["status"] == "ok"
    assert rows[SHOT + 1]["status"] == "error"
    assert not paths.masks_file(SHOT + 1).exists()
    assert got.totals["counts"] == {"error": 1, "ok": 1}


def test_a_shot_over_its_timeout_is_recorded_and_the_next_one_runs(
    tmp_path, synth_shot, model, monkeypatch,
):
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    _write_corpus(corpus, SHOT + 1, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    real = pl.infer_block

    def slow(prepared, **kw):
        if prepared.spec.diag == "ece" and prepared.spec.channel == 20:
            while True:
                pass
        return real(prepared, **kw)

    monkeypatch.setattr(pl, "infer_block", slow)
    got = driver.run_shots([SHOT, SHOT + 1], paths=paths, model=model,
                           device="cpu", passes=("wide",), tile_batch=4,
                           prep_workers=0, timeout_s=1, unet_sha256=FAKE_SHA,
                           echo=_silently)
    rows = {row["shot"]: row for row in got.rows}
    assert rows[SHOT]["status"] == "error"
    assert "exceeded" in rows[SHOT]["detail"]
    assert rows[SHOT + 1]["status"] == "error"      # the same hang, twice


# --------------------------------------------------------- skip-existing/force


def test_skip_existing_leaves_a_finished_shot_alone_and_force_redoes_it(
    tmp_path, synth_shot, model,
):
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    common = {"paths": paths, "model": model, "device": "cpu",
              "passes": ("wide",), "tile_batch": 4, "prep_workers": 0,
              "unet_sha256": FAKE_SHA, "echo": _silently}
    driver.run_shots([SHOT], **common)
    stamp = paths.masks_file(SHOT).stat().st_mtime_ns

    again = driver.run_shots([SHOT], skip_existing=True, **common)
    assert again.rows[0]["status"] == "skipped"
    assert again.rows[0]["n_blocks"] == 0
    assert paths.masks_file(SHOT).stat().st_mtime_ns == stamp

    forced = driver.run_shots([SHOT], skip_existing=True, force=True, **common)
    assert forced.rows[0]["status"] == "ok"
    assert forced.rows[0]["n_blocks"] == 7
    assert paths.masks_file(SHOT).stat().st_mtime_ns != stamp


def test_skip_existing_does_not_skip_a_shot_with_masks_but_no_events(
    tmp_path, synth_shot, model,
):
    # Half a shot is not a done shot: the events file is what a consumer
    # reads, and a run killed between the two writes must be redone.
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    driver.run_shots([SHOT], paths=paths, model=model, device="cpu",
                     passes=("wide",), tile_batch=4, prep_workers=0,
                     unet_sha256=FAKE_SHA, echo=_silently)
    paths.events_file(SHOT).unlink()
    again = driver.run_shots([SHOT], paths=paths, model=model, device="cpu",
                             passes=("wide",), tile_batch=4, prep_workers=0,
                             skip_existing=True, unet_sha256=FAKE_SHA,
                             echo=_silently)
    assert again.rows[0]["status"] == "ok"
    assert paths.events_file(SHOT).exists()


def test_no_write_computes_everything_and_stores_nothing(tmp_path, synth_shot,
                                                          model):
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    got = driver.run_shots([SHOT], paths=paths, model=model, device="cpu",
                           passes=("wide",), tile_batch=4, prep_workers=0,
                           write=False, unet_sha256=FAKE_SHA, echo=_silently)
    assert got.rows[0]["n_blocks"] == 7
    assert not paths.masks_file(SHOT).exists()
    assert not paths.events_file(SHOT).exists()


# ------------------------------------------------------------- the run record


def _only_run(paths):
    written = list((paths.runs / "events").glob("*.json"))
    assert len(written) == 1, written
    return written[0]


@pytest.fixture
def staged(tmp_path, monkeypatch, model, synth_shot):
    """The driver's `main`, with the network and the checkpoint stubbed."""
    corpus = tmp_path / "corpus"
    for shot in SHOTS:
        _write_corpus(corpus, shot, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    monkeypatch.setattr(unet, "load_unet",
                        lambda path=None, device="cpu", **kw: model)
    monkeypatch.setattr(text_weak, "build_logs_subset",
                        lambda shots, **kw: 0)
    return paths


def _argv(paths, *extra):
    return ["--root", str(paths.root), "--corpus", str(paths.corpus),
            "--passes", "wide", "--tile-batch", "4", "--prep-workers", "0",
            *extra]


def test_main_writes_one_run_json_per_chunk_and_rank(staged, capsys):
    assert driver.main(_argv(staged, "--shots", *[str(s) for s in SHOTS],
                             "--chunk", "1", "--n-chunks", "2",
                             "--rank", "0", "--world", "1",
                             "--run-id", "smoke")) == 0
    path = _only_run(staged)
    assert path.name == "smoke_c1of2_r0.json"
    payload = json.loads(path.read_text())

    assert payload["run_id"] == "smoke"
    assert payload["driver"] == "tokeye_masks"
    assert payload["shots_selected"] == SHOTS[1:]
    assert [row["shot"] for row in payload["shots"]] == SHOTS[1:]
    settings = payload["settings"]
    assert settings["chunk"] == 1 and settings["n_chunks"] == 2
    assert settings["rank"] == 0 and settings["world"] == 1
    assert settings["device"] == "cpu" and settings["plan"] == "round1"
    assert settings["passes"] == ["wide"]
    assert settings["prep_workers"] == 0 and settings["prefetch"] == 4
    totals = payload["totals"]
    for key in ("n_shots", "n_blocks", "n_events", "elapsed_s",
                "seconds_per_shot", "blocks_per_s", "tiles_per_s", "n_tiles",
                "prep_wait_s", "infer_s", "describe_s", "peak_rss_gib",
                "peak_worker_rss_gib", "cuda_max_alloc_gib", "counts",
                "events_by_source", "shots_skipping"):
        assert key in totals, key
    assert totals["n_shots"] == 2
    assert totals["n_tiles"] > 0
    assert totals["cuda_max_alloc_gib"] is None     # cpu run
    assert payload["logs_subset"] == str(staged.logs_subset)
    out = capsys.readouterr().out
    assert f"{SHOTS[1]}: 7 blocks" in out
    assert "tokeye_masks: 2 shots" in out


def test_main_still_writes_a_run_json_when_the_rank_has_nothing_to_do(staged):
    assert driver.main(_argv(staged, "--shots", "1", "--rank", "3",
                             "--world", "4", "--run-id", "empty")) == 0
    payload = json.loads(_only_run(staged).read_text())
    assert payload["shots_selected"] == []
    assert payload["shots"] == []
    assert payload["totals"]["counts"] == {}


def test_main_refuses_an_empty_shot_list(staged, tmp_path):
    empty = tmp_path / "none.txt"
    empty.write_text("# nothing here\n")
    assert driver.main(_argv(staged, "--shot-file", str(empty))) == 1


def test_main_builds_the_log_subset_once_for_its_own_shots(staged, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        text_weak, "build_logs_subset",
        lambda shots, *, paths=None, refresh_missing=False: (
            calls.append((sorted(shots), refresh_missing)) or 0
        ),
    )
    assert driver.main(_argv(staged, "--shots", *[str(s) for s in SHOTS],
                             "--rank", "0", "--world", "3",
                             "--refresh-text")) == 0
    assert calls == [([SHOTS[0]], True)]


class _DeadPool:
    """A `ProcessPoolExecutor` whose workers are gone: it refuses new work."""

    def submit(self, *args, **kwargs):
        raise BrokenProcessPool("the pool is broken")

    def shutdown(self, **kwargs) -> None:
        return None


def test_a_dead_prep_worker_costs_its_block_and_the_pool_is_replaced(
    tmp_path, synth_shot, model, monkeypatch,
):
    """A worker that DIES is a skip, not a hang, and not a lost shot.

    `BrokenProcessPool` is how a killed worker reaches this process - the
    OOM killer on a `--mem-per-cpu` that was too tight, a segfault in a
    codec - and it arrives two ways: on the future of the job the dead
    worker held, and, once the executor is poisoned, out of `submit` itself.
    Both are faked here because neither can be provoked reliably; what is
    real is what the driver does with them - the block is skipped, the pool
    is replaced, and the blocks that had not been submitted yet are run on
    the new one.
    """
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)

    killed = {"mhr:4:wide": "on the future", "co2:0:wide": "out of submit"}
    seen: list[str] = []
    restarts: list[int] = []
    real_submit = driver.PrepPool.submit
    real_restart = driver.PrepPool.restart

    def flaky(self, job):
        key = getattr(job, "key", "")
        if key in killed and key not in seen:
            seen.append(key)
            if killed[key] == "out of submit":
                # A poisoned executor refuses the work in the CALLER, and
                # `PrepPool.submit`'s own conversion of that into a failed
                # future is what is under test here.
                self._pool = _DeadPool()
                return real_submit(self, job)
            future = Future()
            future.set_exception(BrokenProcessPool("a worker was killed"))
            return future
        return real_submit(self, job)

    monkeypatch.setattr(driver.PrepPool, "submit", flaky)
    monkeypatch.setattr(driver.PrepPool, "restart",
                        lambda self: (restarts.append(1), real_restart(self)))
    got = driver.run_shots([SHOT], paths=paths, model=model, device="cpu",
                           passes=("wide",), tile_batch=4, prep_workers=0,
                           prefetch=2, unet_sha256=FAKE_SHA, echo=_silently)

    row = got.rows[0]
    assert row["status"] == "ok"                    # not an error, and not a hang
    assert row["n_blocks"] == 5                     # seven, less the two killed
    assert sorted(seen) == ["co2:0:wide", "mhr:4:wide"]
    assert restarts == [1, 1]                       # one per death
    for key in ("mask mhr:4:wide", "mask co2:0:wide"):
        assert "BrokenProcessPool" in row["skipped"][key]
    assert masks.list_blocks(paths.masks_file(SHOT)) == [
        "co2_02_wide", "ece_08_wide", "ece_20_wide", "ece_40_wide",
        "mhr_00_wide",
    ]
