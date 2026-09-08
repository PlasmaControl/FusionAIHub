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

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from labelmaker.config import Paths
from labelmaker.events import channels, masks
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
