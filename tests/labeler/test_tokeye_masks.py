"""The batch driver: `events/driver.py` and `scripts/labeler/tokeye_masks.py`.

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
import os
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import h5py
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from labeler.config import Paths
from labeler.events import channels, driver, masks, schema, text_weak, unet
from labeler.events import pipeline as pl

from .test_events_pipeline import (
    FAKE_SHA,
    SHOT,
    PaintedNet,
    _write_corpus,
    _write_text,
)

#: Three copies of the synthetic shot, so a chunk and a rank have
#: something to split. The corpus files differ only in their names.
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
    # directory (`shot_design.design.seed.chunk_of`'s reason); strided ranks so
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
        "--refresh-text", "--no-write", "--no-index", "--rebuild-index",
        "--text-subset", "--build-text-subset", "--tail-workers",
    ):
        assert flag in text, flag


def test_the_shot_pickers_are_exclusive_and_one_is_required(tmp_path):
    parser = driver.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--shots", "1", "--shot-file", "/tmp/x"])
    # The requirement moved from the argparse group to `main`, because
    # `--rebuild-index` is a run over what is already on disk and has no
    # shot list; everything else still refuses to start without one.
    with pytest.raises(SystemExit):
        driver.main(["--root", str(tmp_path)])


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


def _sources(paths, shot):
    """The shot's per-source record, without the two columns that say WHEN."""
    df = schema.read_sources(paths.sources_file(shot))
    return df.drop(columns=["run_id", "written_at"])


#: The row keys the driver's schedule is allowed to change.
_TIMINGS = ("seconds", "prep_wait_s", "infer_s", "describe_s", "finish_s",
            "tail_wait_s", "n_tiles")


def _silently(*args, **kwargs):
    return None


@pytest.mark.parametrize(("workers", "prefetch", "tail"),
                         [(0, 1, 0), (1, 4, 1), (2, 2, 1)])
def test_the_driver_writes_exactly_what_process_shot_writes(
    tmp_path, synth_shot, model, workers, prefetch, tail,
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
        prefetch=prefetch, tail_workers=tail, run_id="test-run",
        unet_sha256=FAKE_SHA, echo=_silently,
    )

    assert ref.error == "" and ref.n_blocks == 14
    row = dict(got.rows[0])
    # The two timings and the wall clock are the only rows that may differ:
    # they are what the driver exists to change.
    for key in ("seconds", "prep_wait_s", "infer_s", "describe_s",
                "finish_s", "tail_wait_s", "n_tiles"):
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
    # And the per-source completion record: `finish_shot` writes it for
    # both, so which sources ran, over what coverage, and how many rows
    # each produced has to read the same off either schedule.
    pd.testing.assert_frame_equal(_sources(driven, SHOT),
                                  _sources(sequential, SHOT))
    assert not _sources(driven, SHOT).empty


@pytest.mark.parametrize("pooled", [False, True])
def test_a_block_whose_prep_fails_in_a_worker_is_a_skip_and_not_a_hang(
    tmp_path, synth_shot, model, pooled,
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
                           unet_sha256=FAKE_SHA, echo=_silently, pooled=pooled)

    assert ref.n_blocks == 5                        # mhr 0/4, ece 8/20/40
    assert got.rows[0]["n_blocks"] == 5
    assert got.rows[0]["status"] == "ok"
    assert got.rows[0]["skipped"] == ref.as_row()["skipped"]
    assert [k for k in got.rows[0]["skipped"] if k.startswith(("read co2",
                                                               "mask co2"))]
    for key, value in _mask_keys(sequential.masks_file(SHOT)).items():
        assert np.array_equal(_mask_keys(driven.masks_file(SHOT))[key], value)


@pytest.mark.parametrize("pooled", [False, True])
def test_an_unreadable_corpus_file_is_one_error_row_and_the_run_goes_on(
    tmp_path, synth_shot, model, pooled,
):
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / f"{SHOT + 1}_processed.h5").write_bytes(b"not an hdf5 file")
    paths = _paths_under(tmp_path, "root", corpus)

    got = driver.run_shots([SHOT, SHOT + 1], paths=paths, model=model,
                           device="cpu", passes=("wide",), tile_batch=4,
                           prep_workers=0, unet_sha256=FAKE_SHA,
                           echo=_silently, pooled=pooled)
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
    # `--text-subset build` explicitly: `auto` reads a multi-rank run's
    # subset instead of writing it (see the text-subset tests below).
    assert driver.main(_argv(staged, "--shots", *[str(s) for s in SHOTS],
                             "--rank", "0", "--world", "3",
                             "--text-subset", "build", "--refresh-text")) == 0
    assert calls == [([SHOTS[0]], True)]


class _DeadPool:
    """A `ProcessPoolExecutor` whose workers are gone: it refuses new work."""

    def submit(self, *args, **kwargs):
        raise BrokenProcessPool("the pool is broken")

    def shutdown(self, **kwargs) -> None:
        return None


@pytest.mark.parametrize("pooled", [False, True])
def test_a_dead_prep_worker_costs_its_block_and_the_pool_is_replaced(
    tmp_path, synth_shot, model, monkeypatch, pooled,
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
                           prefetch=2, unet_sha256=FAKE_SHA, echo=_silently, pooled=pooled)

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


# ------------------------------------------------------- the wedged worker


def _fifo_corpus(tmp_path, shot=SHOT):
    """A corpus file that never answers: a FIFO with no writer.

    The cheapest faithful stand-in for the failure `--timeout` exists for -
    a GPFS read that has gone away - because `open()` on it blocks in the
    kernel, uninterruptibly as far as the reading process is concerned, and
    a worker in that state never takes the pool's shutdown sentinel.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    os.mkfifo(corpus / f"{int(shot)}_processed.h5")
    return corpus


def _until_dead(procs, seconds: float = 180.0) -> None:
    deadline = time.monotonic() + seconds
    while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
        time.sleep(0.2)


def test_closing_a_pool_kills_a_worker_that_is_wedged(tmp_path):
    """`close()` must not leave a live worker behind - at all.

    `shutdown(wait=False)` asks politely: it drops a sentinel on the call
    queue and returns. A worker blocked in an uninterruptible read never
    reads that sentinel, and CPython's own `_python_exit` atexit hook then
    joins the executor's manager thread at interpreter shutdown - so the
    process hangs AFTER `main()` has returned, which on SLURM is a finished
    job recorded as a wall-clock TIMEOUT still holding its GPU.
    """
    corpus = _fifo_corpus(tmp_path)
    pool = driver.PrepPool(1)
    try:
        spec = channels.ChannelSpec("mhr", 0, "magnetics")
        pool.submit(driver.PrepJob(corpus / f"{SHOT}_processed.h5", spec,
                                   "wide"))
        deadline = time.monotonic() + 60
        while not pool.processes() and time.monotonic() < deadline:
            time.sleep(0.05)
        procs = pool.processes()
        assert procs, "the pool never started a worker"
        at = time.monotonic()
        pool.close()
        assert time.monotonic() - at < 30       # close does not wait forever
        # `close` SENDS the signals and does not join for ever - a child
        # caught in an uninterruptible read (a busy GPFS, which is the whole
        # reason this path exists) is reaped when it leaves that state, and
        # a driver that waited for it would be the hang this fixes. So the
        # assertion is that it dies, not that it dies synchronously.
        _until_dead(procs)
        assert [p for p in procs if p.is_alive()] == []
    finally:
        pool.close()


@pytest.mark.parametrize("tail_workers", [0, 1])
@pytest.mark.parametrize("pooled", [False, True])
def test_a_shot_wedged_in_a_worker_does_not_stop_the_run_exiting(
    tmp_path, monkeypatch, tail_workers, pooled,
):
    """The same thing end to end, through `run_shots`.

    The shot is one error row after its `--timeout`, which the driver
    already did; what is under test is what is left behind afterwards. No
    executor worker may still be alive after `run_shots` closes its pools:
    their manager threads would be joined at interpreter exit. Python's
    multiprocessing.resource_tracker is a separate helper, not a leaked
    worker; check the pools' own handles rather than all OS children.
    """
    corpus = _fifo_corpus(tmp_path)
    paths = _paths_under(tmp_path, "root", corpus)
    closed_workers = []
    real_close = driver.PrepPool.close

    def close(self):
        closed_workers.extend(self.processes())
        real_close(self)

    monkeypatch.setattr(driver.PrepPool, "close", close)

    at = time.monotonic()
    got = driver.run_shots([SHOT], paths=paths, model=None, device="cpu",
                           corpus_dir=corpus, passes=("wide",), tile_batch=4,
                           prep_workers=1, prefetch=2, timeout_s=4,
                           tail_workers=tail_workers,
                           unet_sha256=FAKE_SHA, echo=_silently, pooled=pooled)
    elapsed = time.monotonic() - at

    assert got.rows[0]["status"] == "error"
    assert elapsed < 120
    assert closed_workers, "the pools never started any workers"
    _until_dead(closed_workers)
    assert [p for p in closed_workers if p.is_alive()] == []


@pytest.mark.parametrize(("prep_workers", "tail_workers"),
                         [(1, 0), (0, 1), (1, 1)])
def test_the_worker_peak_rss_is_a_worker_and_not_the_parent(
    tmp_path, synth_shot, model, prep_workers, tail_workers,
):
    """`peak_worker_rss_gib` must include both prep and tail workers.

    It used to be `getrusage(RUSAGE_CHILDREN)`, which on a pool shut down
    with `wait=False` reads either zero (nothing was waited for) or the
    parent's own peak (a spawned child's fork-before-exec inherits its
    resident pages). Both are the wrong number to size `--mem-per-cpu`
    from, and the second is wrong in a way that looks right.
    """
    assert driver._vmhwm_gib(os.getpid()) > 0.0
    assert driver._vmhwm_gib(2 ** 30) == 0.0        # no such process

    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    got = driver.run_shots([SHOT], paths=paths, model=model, device="cpu",
                           passes=("wide",), tile_batch=4,
                           prep_workers=prep_workers, tail_workers=tail_workers,
                           prefetch=2, unet_sha256=FAKE_SHA, echo=_silently)
    worker = got.totals["peak_worker_rss_gib"]
    assert worker > 0.0
    assert worker != got.totals["peak_rss_gib"]
    by_pool = got.totals["worker_rss_gib"]
    for role, n_workers in (("prep", prep_workers), ("tail", tail_workers)):
        assert len(by_pool[role]) == n_workers
        assert all(int(pid) != os.getpid() and rss > 0
                   for pid, rss in by_pool[role].items())
    assert worker == max(rss for group in by_pool.values() for rss in group.values())


# ------------------------------------------------------------- the index


def test_no_index_writes_the_per_shot_products_and_not_the_shared_file(
    tmp_path, synth_shot, model,
):
    """`--no-index` is what an array task runs.

    `events_index.parquet` is ONE file for the whole root and
    `labels.store.append_index` rewrites it whole through a FIXED `.tmp`
    sibling. Sixteen tasks doing that per shot lose each other's rows and
    can rename a torn parquet into place - after which the next shot's
    `read_parquet` raises inside the write guard and a healthy shot is
    recorded `status == "error"`. The per-shot products are untouched by
    the flag; only the derivable file is left for one pass afterwards.
    """
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    got = driver.run_shots([SHOT], paths=paths, model=model, device="cpu",
                           passes=("wide",), tile_batch=4, prep_workers=0,
                           index=False, unet_sha256=FAKE_SHA, echo=_silently)

    assert got.rows[0]["status"] == "ok"
    assert paths.masks_file(SHOT).exists()
    assert paths.events_file(SHOT).exists()
    assert not paths.events_index.exists()


def test_a_rebuilt_index_is_the_one_the_shot_by_shot_writes_would_have_left(
    tmp_path, synth_shot, model,
):
    """The rebuild is not an approximation of the incremental index.

    Same shots, two roots: one written with the per-shot `append_index`,
    one with `--no-index` and then rebuilt in a single pass from
    `events/*_events.parquet`. The two parquet files must agree row for
    row and column for column, bar `written_at` - which records when each
    run wrote, and is the one thing two runs cannot share.
    """
    corpus = tmp_path / "corpus"
    for shot in SHOTS:
        _write_corpus(corpus, shot, synth_shot)
    incremental = _paths_under(tmp_path, "incremental", corpus)
    deferred = _paths_under(tmp_path, "deferred", corpus)
    for paths, index in ((incremental, True), (deferred, False)):
        driver.run_shots(SHOTS, paths=paths, model=model, device="cpu",
                         corpus_dir=corpus, passes=("wide",), tile_batch=4,
                         prep_workers=0, index=index, run_id="test-run",
                         unet_sha256=FAKE_SHA, echo=_silently)
    assert incremental.events_index.exists()
    assert not deferred.events_index.exists()

    report = driver.rebuild_index(deferred)
    assert report["files"] == len(SHOTS)
    assert report["rows"] > 0
    assert report["unreadable"] == {}

    def _index(paths):
        df = pd.read_parquet(paths.events_index).drop(columns=["written_at"])
        return df.sort_values(list(driver.INDEX_KEYS)).reset_index(drop=True)

    pd.testing.assert_frame_equal(_index(deferred), _index(incremental))


def test_the_rebuild_replaces_a_stale_index_rather_than_merging_into_it(
    tmp_path, synth_shot, model,
):
    """A rebuild is derived from what is on disk NOW.

    `append_index` merges, so a row for a shot whose events file has been
    deleted (a re-run with a different plan, a shot withdrawn from the
    list) would survive every incremental write for ever. The rebuild is
    the repair for that, so it must not merge.
    """
    corpus = tmp_path / "corpus"
    for shot in SHOTS:
        _write_corpus(corpus, shot, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    driver.run_shots(SHOTS, paths=paths, model=model, device="cpu",
                     corpus_dir=corpus, passes=("wide",), tile_batch=4,
                     prep_workers=0, run_id="test-run", unet_sha256=FAKE_SHA,
                     echo=_silently)
    paths.events_file(SHOTS[2]).unlink()

    report = driver.rebuild_index(paths, echo=_silently)
    assert report["files"] == len(SHOTS) - 1
    left = pd.read_parquet(paths.events_index)
    assert sorted(set(left["shot"])) == SHOTS[:2]


def test_main_can_rebuild_the_index_without_a_shot_list(staged, capsys):
    """`--rebuild-index` is the one-pass repair task L12 runs after the array."""
    assert driver.main(_argv(staged, "--shots", *[str(s) for s in SHOTS],
                             "--no-index", "--run-id", "arrayish")) == 0
    assert not staged.events_index.exists()
    payload = json.loads(_only_run(staged).read_text())
    assert payload["settings"]["index"] == "skipped"

    assert driver.main(["--root", str(staged.root), "--rebuild-index"]) == 0
    rows = pd.read_parquet(staged.events_index)
    assert sorted(set(rows["shot"])) == SHOTS
    assert "tokeye_masks: rebuilt" in capsys.readouterr().out


def test_rebuild_can_write_inside_events_without_touching_root_index(staged):
    assert driver.main(_argv(staged, "--shots", str(SHOTS[0]),
                             "--no-index", "--run-id", "index-output")) == 0
    staged.events_index.write_bytes(b"leave the canonical index alone")
    out = staged.events / "events_index.parquet"
    assert driver.main(["--root", str(staged.root), "--rebuild-index",
                        "--index-out", str(out)]) == 0
    assert set(pd.read_parquet(out)["shot"]) == {SHOTS[0]}
    assert staged.events_index.read_bytes() == b"leave the canonical index alone"


def test_index_output_requires_rebuild(staged):
    with pytest.raises(SystemExit) as exc:
        driver.main(_argv(staged, "--shots", str(SHOTS[0]),
                          "--index-out", str(staged.events / "index.parquet")))
    assert exc.value.code == 2


# --------------------------------------------------------- the text subset


@pytest.fixture
def staged_text(tmp_path, monkeypatch, model, synth_shot):
    """`staged`, but with the REAL `build_logs_subset`.

    The subset file itself is the assertion here - whether a run left its
    bytes alone - so the builder must be the one that would rewrite it.
    """
    corpus = tmp_path / "corpus"
    for shot in SHOTS:
        _write_corpus(corpus, shot, synth_shot)
    paths = _paths_under(tmp_path, "root", corpus)
    monkeypatch.setattr(unet, "load_unet",
                        lambda path=None, device="cpu", **kw: model)
    return paths


def _subset_state(paths):
    stat = paths.logs_subset.stat()
    return (paths.logs_subset.read_bytes(), stat.st_size, stat.st_mtime_ns,
            sorted(p.name for p in paths.logs_subset.parent.iterdir()))


def test_a_multi_rank_run_reads_the_text_subset_and_never_writes_it(
    staged_text,
):
    """`text/logs_subset.jsonl` is the second file a whole root shares.

    `build_logs_subset` rewrites it WHOLE - the old lines plus this call's
    records - into a pid-suffixed `.tmp` and renames it over the old file.
    The pid stops two tasks interleaving bytes; it does not stop a lost
    update. Sixteen array tasks each add their own thirty shots to the same
    old file and the last rename wins, so fifteen tasks' records vanish
    before their shots are processed and those shots silently lose their
    `text` events. So a run that is one of several does not build it at
    all: it reads what a pre-pass built.
    """
    staged = staged_text
    _write_text(staged, SHOTS[0], "fishbones through the current ramp")
    before = _subset_state(staged)

    assert driver.main(_argv(staged, "--shots", *[str(s) for s in SHOTS],
                             "--rank", "0", "--world", "3")) == 0
    # Byte for byte, mtime included, and no `.tmp` sibling left behind:
    # this rank did not rewrite the file, so it cannot have dropped
    # another rank's records from it.
    assert _subset_state(staged) == before
    payload = json.loads(_only_run(staged).read_text())
    assert payload["settings"]["text_subset"] == "readonly"
    assert payload["text_subset_missing"] == []
    row = payload["shots"][0]
    assert row["shot"] == SHOTS[0]
    assert row["n_text"] == 1                       # the prebuilt record read
    assert "text" not in row["skipped"]


def test_a_shot_missing_from_the_prebuilt_subset_is_told_how_to_fix_it(
    staged_text,
):
    """A missing record must not read as "the logbook has nothing".

    And it must not send this rank to the logbook either: a build here is
    the whole-file rewrite that loses the other ranks' records, and there
    is no `logs.jsonl` under this root for it to read anyway.
    """
    staged = staged_text
    _write_text(staged, SHOTS[2], "fishbones through the current ramp")
    before = _subset_state(staged)

    assert driver.main(_argv(staged, "--shots", *[str(s) for s in SHOTS],
                             "--rank", "0", "--world", "3")) == 0
    assert _subset_state(staged) == before
    payload = json.loads(_only_run(staged).read_text())
    assert payload["text_subset_missing"] == [SHOTS[0]]
    why = payload["shots"][0]["skipped"]["text"]
    assert "prebuilt" in why and "--build-text-subset" in why


def test_the_text_pre_pass_builds_the_whole_list_once_and_runs_nothing(
    staged, monkeypatch,
):
    """The documented one-shot pre-pass, before the array is submitted."""
    calls: list = []
    monkeypatch.setattr(
        text_weak, "build_logs_subset",
        lambda shots, *, paths=None, refresh_missing=False: (
            calls.append((sorted(shots), refresh_missing)) or 7
        ),
    )
    assert driver.main([
        "--root", str(staged.root), "--corpus", str(staged.corpus),
        "--build-text-subset", "--shots", *[str(s) for s in SHOTS],
        "--chunk", "0", "--n-chunks", "3", "--rank", "0", "--world", "4",
    ]) == 0
    # The WHOLE list, not this task's chunk or this rank's stride, and no
    # shot was run: a pre-pass is not a run.
    assert calls == [(SHOTS, False)]
    assert list((staged.runs / "events").glob("*.json")) == []
    assert not staged.masks_file(SHOTS[0]).exists()


# ------------------------------------------------------------- the tail


class _UnpicklableOnce:
    """Only the first shot's payload is bad; later tails receive None."""

    def __init__(self):
        self.first = True

    def __reduce__(self):
        if self.first:
            self.first = False
            raise RuntimeError("the first lexicon cannot be pickled")
        return type(None), ()


_REAL_RUN_JOB = driver.run_job


def _run_with_bad_first_tail(job):
    """Inject one failure inside a real spawned worker; keep other work real."""
    if isinstance(job, driver.FinishJob):
        failure, job.lexicon = job.lexicon, None
        if job.res.shot == SHOT:
            if failure == "raise":
                raise RuntimeError("the first tail raised")
            if failure == "task_timeout":
                raise TimeoutError("the first tail raised its own timeout")
            if failure == "crash":
                os._exit(1)
            if failure == "timeout":
                time.sleep(3600)
    return _REAL_RUN_JOB(job)


def _totals(tmp_path, name, shots, synth_shot, model, **kw):
    corpus = tmp_path / f"corpus_{name}"
    for shot in shots:
        _write_corpus(corpus, shot, synth_shot)
    paths = _paths_under(tmp_path, name, corpus)
    got = driver.run_shots(shots, paths=paths, model=model, device="cpu",
                           corpus_dir=corpus, passes=("wide",), tile_batch=4,
                           prep_workers=0, run_id="test-run",
                           unet_sha256=FAKE_SHA, echo=_silently, **kw)
    return paths, got


@pytest.mark.parametrize("pooled", [False, True])
def test_an_unpicklable_tail_costs_only_its_shot(tmp_path, synth_shot, model, pooled,
):
    """A payload error must not kill the next tail already in the real pool."""
    paths, got = _totals(tmp_path, "unpicklable", SHOTS, synth_shot, model,
                         tail_workers=1, lexicon=_UnpicklableOnce(), pooled=pooled)

    assert [r["shot"] for r in got.rows] == SHOTS
    assert got.rows[0]["error"] == "RuntimeError"
    assert "first lexicon cannot be pickled" in got.rows[0]["detail"]
    assert [r["status"] for r in got.rows] == ["error", "ok", "ok"], got.rows
    assert got.totals["counts"] == {"error": 1, "ok": 2}
    assert not paths.masks_file(SHOT).exists()
    assert not paths.events_file(SHOT).exists()
    for shot in SHOTS[1:]:
        assert masks.list_blocks(paths.masks_file(shot))
        assert not _events(paths, shot).empty


@pytest.mark.parametrize(("failure", "error"), [
    ("raise", "RuntimeError"),
    ("task_timeout", "TimeoutError"),
    ("crash", "BrokenProcessPool"),
    ("timeout", "TimeoutError"),
])
@pytest.mark.parametrize("pooled", [False, True])
def test_a_failed_tail_worker_preserves_the_next_shots_gpu_work(
    tmp_path, synth_shot, model, monkeypatch, failure, error, pooled,
):
    """A restart must recover the queued job even if its future is broken."""
    monkeypatch.setattr(driver, "run_job", _run_with_bad_first_tail)

    def warm(self):
        # Pay imports before the timeout measures the deliberately hung job.
        self.submit(driver.WarmJob()).result(timeout=120)

    monkeypatch.setattr(driver.PrepPool, "warm", warm)
    restarts = []
    real_restart = driver.PrepPool.restart

    def restart(self):
        restarts.append(self.workers)
        real_restart(self)

    monkeypatch.setattr(driver.PrepPool, "restart", restart)
    inferred = []
    real_forward = model.forward

    def forward(x):
        inferred.extend(["mhr:0:wide"] * len(x))
        return real_forward(x)

    monkeypatch.setattr(model, "forward", forward)
    paths, got = _totals(
        tmp_path, failure, SHOTS, synth_shot, model,
        plan=(channels.ChannelSpec("mhr", 0, "magnetics"),),
        tail_workers=1, timeout_s=30, lexicon=failure, pooled=pooled,
    )

    assert [r["shot"] for r in got.rows] == SHOTS
    assert got.rows[0]["error"] == error
    assert [r["status"] for r in got.rows] == ["error", "ok", "ok"], got.rows
    assert got.totals["counts"] == {"error": 1, "ok": 2}
    assert restarts == ([1] if failure in {"crash", "timeout"} else [])
    assert inferred == ["mhr:0:wide"] * 3  # no repeated GPU work on recovery
    assert not paths.masks_file(SHOT).exists()
    assert not paths.events_file(SHOT).exists()
    for shot in SHOTS[1:]:
        assert masks.list_blocks(paths.masks_file(shot)) == ["mhr_00_wide"]
        assert not _events(paths, shot).empty


def test_a_finished_tail_releases_its_blocks_before_the_next_shot(
    tmp_path, synth_shot, model, monkeypatch,
):
    """Keeping a retry payload must not retain an extra completed shot."""
    jobs = {}
    real_submit = driver.PrepPool.submit

    def submit(self, job):
        if isinstance(job, driver.FinishJob):
            jobs[job.res.shot] = weakref.ref(job)
        return real_submit(self, job)

    monkeypatch.setattr(driver.PrepPool, "submit", submit)
    retained = []
    real_blocks = driver._shot_blocks

    def blocks(shot, **kwargs):
        if shot == SHOTS[2]:
            # Shot 1 was flushed after shot 2's blocks. Only shot 2's
            # payload may remain when shot 3 starts allocating its blocks.
            retained.append(jobs[SHOT]() is not None)
        return real_blocks(shot, **kwargs)

    monkeypatch.setattr(driver, "_shot_blocks", blocks)
    _, got = _totals(
        tmp_path, "released", SHOTS, synth_shot, model,
        plan=(channels.ChannelSpec("mhr", 0, "magnetics"),), tail_workers=0,
    )
    assert [r["status"] for r in got.rows] == ["ok", "ok", "ok"]
    assert retained == [False]


def test_the_tail_of_a_shot_is_paid_beside_the_next_shots_blocks(
    tmp_path, synth_shot, model,
):
    """The per-shot tail must not stand in front of the next forward pass.

    `pipeline.finish_shot` - the sawtooth pass over the whole ECE array,
    L-H, the actuators, the QH proxy, the text lookup and the two writes -
    was measured at ~3.72 s per shot of parent-thread work against ~0.84 s
    of A100 forward pass, which caps GPU utilisation near 15 %. It is
    order-independent across shots (one shot's tail touches one shot's
    files), so it is handed to a worker and collected once the next shot's
    blocks are done.

    Measured here as the driver reports it: `finish_s` is what the tail
    cost, `tail_wait_s` is how much of that the thread holding the GPU
    actually waited for. The middle shot is the one to read - the first
    shot's tail is also where the worker's own spawn and torch import land
    if `warm()` has not finished paying for them, and the last shot's tail
    has no next shot to hide behind.
    """
    overlapped = _totals(tmp_path, "overlapped", SHOTS, synth_shot, model,
                         tail_workers=1)[1]
    serial = _totals(tmp_path, "serial", SHOTS, synth_shot, model,
                     tail_workers=0)[1]

    mine, theirs = overlapped.rows[1], serial.rows[1]
    assert mine["finish_s"] > 0.0 and theirs["finish_s"] > 0.0
    # Serial: the tail IS the wait, to within the clock either side of it.
    assert theirs["tail_wait_s"] >= theirs["finish_s"] * 0.9
    # Overlapped: the GPU-owning thread did not wait for what it cost.
    assert mine["tail_wait_s"] < mine["finish_s"]
    assert overlapped.totals["finish_s"] > 0.0


def test_an_overlapped_run_writes_the_same_rows_in_the_same_order(
    tmp_path, synth_shot, model,
):
    """Overlapping the tail may not re-order or lose a shot."""
    later, overlapped = _totals(tmp_path, "overlapped", SHOTS, synth_shot,
                                model, tail_workers=1)
    first, serial = _totals(tmp_path, "serial", SHOTS, synth_shot, model,
                            tail_workers=0)

    assert [r["shot"] for r in overlapped.rows] == SHOTS
    assert [r["shot"] for r in serial.rows] == SHOTS
    for mine, theirs in zip(overlapped.rows, serial.rows, strict=True):
        assert {k: v for k, v in mine.items() if k not in _TIMINGS} == {
            k: v for k, v in theirs.items() if k not in _TIMINGS
        }
    for shot in SHOTS:
        assert _mask_keys(later.masks_file(shot)).keys() == _mask_keys(
            first.masks_file(shot)).keys()
        pd.testing.assert_frame_equal(_events(later, shot),
                                      _events(first, shot))


@pytest.mark.parametrize("pooled", [False, True])
def test_a_tail_that_raises_is_one_error_row_and_the_run_goes_on(
    tmp_path, synth_shot, model, monkeypatch, pooled,
):
    """A failed tail costs its shot and not the run.

    In-process (`tail_workers=0`) so the stand-in can be monkeypatched; the
    path under test is the one that collects the tail, which is shared with
    the worker case - `PrepPool.submit` puts a failure on the future either
    way.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("the tail fell off")

    monkeypatch.setattr(pl, "finish_shot", boom)
    _, got = _totals(tmp_path, "boom", SHOTS[:2], synth_shot, model,
                     tail_workers=0, pooled=pooled)
    assert [r["status"] for r in got.rows] == ["error", "error"]
    assert [r["shot"] for r in got.rows] == SHOTS[:2]
    assert all("the tail fell off" in r["detail"] for r in got.rows)


def test_the_tail_worker_default_follows_the_prep_pool(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    args = driver.settle(driver.build_parser().parse_args(["--shots", "1"]))
    assert args.tail_workers == 1
    args = driver.settle(driver.build_parser().parse_args(
        ["--shots", "1", "--prep-workers", "0"]
    ))
    assert args.tail_workers == 0
    args = driver.settle(driver.build_parser().parse_args(
        ["--shots", "1", "--prep-workers", "0", "--tail-workers", "1"]
    ))
    assert args.tail_workers == 1


def test_two_tail_workers_restart_without_reordering_pending_shots(
    tmp_path, synth_shot, model, monkeypatch,
):
    shots = [SHOT + i for i in range(4)]
    real_submit = driver.PrepPool.submit
    submitted, restarts = [], []
    # Deterministic mixed completion state with two configured tail workers.
    monkeypatch.setattr(driver.PrepPool, "_start", lambda self: None)

    def restart(self):
        restarts.append(self.workers)

    def submit(self, job):
        if isinstance(job, driver.FinishJob):
            submitted.append(job.res.shot)
            if not restarts and job.res.shot in (shots[0], shots[2]):
                future = Future()
                if job.res.shot == shots[0]:
                    future.set_exception(BrokenProcessPool("first tail died"))
                return future
        return real_submit(self, job)

    monkeypatch.setattr(driver.PrepPool, "submit", submit)
    monkeypatch.setattr(driver.PrepPool, "restart", restart)
    paths, got = _totals(
        tmp_path, "mixed", shots, synth_shot, model, pooled=True,
        plan=(channels.ChannelSpec("mhr", 0, "magnetics"),),
        tail_workers=2, index=False,
    )
    assert restarts == [2]
    assert submitted == shots + [shots[2]]
    assert [row["shot"] for row in got.rows] == shots
    assert [row["status"] for row in got.rows] == ["error", "ok", "ok", "ok"]
    for shot in shots[1:]:
        assert masks.list_blocks(paths.masks_file(shot)) == ["mhr_00_wide"]


def test_cli_refuses_multiple_index_writers_before_loading_the_model(staged,
                                                                    monkeypatch,
                                                                    capsys):
    def load(*args, **kwargs):
        pytest.fail("invalid writer configuration reached model loading")

    monkeypatch.setattr(unet, "load_unet", load)
    with pytest.raises(SystemExit) as exc:
        driver.main(_argv(staged, "--shots", str(SHOT), "--tail-workers", "2"))
    assert exc.value.code == 2
    assert "--no-index" in capsys.readouterr().err


def test_pooled_options_have_an_explicit_surface(paths, model, monkeypatch):
    import inspect

    seen = {}

    def run(shots, **opts):
        seen.update(opts)
        return driver.DriverRun([], {})

    monkeypatch.setattr(driver, "_run_pooled", run)
    driver.run_shots([], paths=paths, model=model, pooled=True)
    expected = set(inspect.signature(driver.run_shots).parameters) - {"shots", "pooled"}
    assert set(seen) == expected


@pytest.mark.parametrize("extra", [["--device", "cuda"], ["--probs-on-host"]])
def test_cpu_forward_pooling_opt_in_refuses_other_schedules(staged, extra):
    with pytest.raises(SystemExit) as exc:
        driver.main(_argv(staged, "--shots", str(SHOT), "--pool-cpu-forwards", *extra))
    assert exc.value.code == 2
