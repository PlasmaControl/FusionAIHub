"""Exact products across each L14 scheduling step; process_shot is the oracle."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from labeler.events import driver, masks, pipeline, schema

from .test_events_pipeline import FAKE_SHA, SHOT, PaintedNet, _write_corpus
from .test_tokeye_masks import _paths_under


@pytest.mark.parametrize("workers,prefetch", [(0, 1), (1, 4), (2, 2)])
@pytest.mark.parametrize("stage", ["compact", "pooled", "tail", "prep"])
def test_output_identity(tmp_path, synth_shot, monkeypatch, workers, prefetch, stage):
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    ref_paths = _paths_under(tmp_path, "reference", corpus)
    got_paths = _paths_under(tmp_path, stage, corpus)
    model = PaintedNet().eval()
    ref = pipeline.process_shot(
        SHOT,
        ref_paths,
        model=model,
        passes=("wide", "zoom"),
        tile_batch=4,
        unet_sha256=FAKE_SHA,
    )
    assert not ref.error and ref.n_blocks == 14

    def compact_infer(prepared, **kwargs):
        return masks.infer(
            model, prepared.spectrogram, "cpu", batch=4, amp=False, compact=True
        )

    if stage == "compact":
        monkeypatch.setattr(pipeline, "infer_block", compact_infer)
    if stage in {"tail", "prep"}:

        def parent_describe(*args, **kwargs):
            raise AssertionError("describe_block ran in the GPU-owning parent")

        monkeypatch.setattr(pipeline, "describe_block", parent_describe)
    got = driver.run_shots(
        [SHOT],
        paths=got_paths,
        model=model,
        passes=("wide", "zoom"),
        tile_batch=4,
        prep_workers=workers,
        prefetch=prefetch,
        unet_sha256=FAKE_SHA,
        tail_workers=int(stage in {"tail", "prep"}),
        pooled=stage != "compact",
    )
    assert got.rows[0]["status"] == "ok"
    assert got.rows[0]["n_blocks"] == ref.n_blocks
    assert (
        got_paths.masks_file(SHOT).read_bytes()
        == ref_paths.masks_file(SHOT).read_bytes()
    )
    evidence = {
        "stage": stage,
        "workers": workers,
        "prefetch": prefetch,
        "masks_npz": hashlib.sha256(
            ref_paths.masks_file(SHOT).read_bytes()
        ).hexdigest(),
    }
    for name, reader in [
        ("events", schema.read_events),
        ("sources", schema.read_sources),
    ]:
        reference = reader(getattr(ref_paths, name + "_file")(SHOT)).drop(
            columns=["run_id", "written_at"]
        )
        actual = reader(getattr(got_paths, name + "_file")(SHOT)).drop(
            columns=["run_id", "written_at"]
        )
        pd.testing.assert_frame_equal(actual, reference, check_exact=True)
        if name == "sources":
            assert {"intervals", "min_gap_s"} <= set(actual.columns)
        evidence[name] = hashlib.sha256(
            reference.to_json(orient="table", double_precision=15).encode()
        ).hexdigest()
    print("IDENTITY " + json.dumps(evidence, sort_keys=True), flush=True)


@pytest.mark.parametrize("tails", [1, 2])
def test_pool_fills_batches_across_shots(tmp_path, synth_shot, tails):
    corpus = tmp_path / "corpus"
    refs = _paths_under(tmp_path, "reference", corpus)
    pooled = _paths_under(tmp_path, "pooled", corpus)
    shots = [SHOT, SHOT + 1, SHOT + 2]
    for shot in shots:
        _write_corpus(corpus, shot, synth_shot)
        ref = pipeline.process_shot(
            shot, refs, model=PaintedNet().eval(), tile_batch=5, unet_sha256=FAKE_SHA
        )
        assert not ref.error and ref.n_blocks == 14

    class CountedNet(PaintedNet):
        def __init__(self):
            super().__init__()
            self.batches = []

        def forward(self, x):
            self.batches.append(len(x))
            return super().forward(x)

    model = CountedNet().eval()
    result = driver.run_shots(
        shots,
        paths=pooled,
        model=model,
        tile_batch=5,
        pool_cpu_forwards=True,
        prep_workers=2,
        prefetch=2,
        tail_workers=tails,
        index=False,
        pooled=True,
        unet_sha256=FAKE_SHA,
    )
    assert [row["shot"] for row in result.rows] == shots
    assert model.batches == [5] * 8 + [2]
    for shot in shots:
        assert (
            pooled.masks_file(shot).read_bytes() == refs.masks_file(shot).read_bytes()
        )
        for name, reader in [
            ("events", schema.read_events),
            ("sources", schema.read_sources),
        ]:
            pd.testing.assert_frame_equal(
                reader(getattr(pooled, name + "_file")(shot)).drop(
                    columns=["run_id", "written_at"]
                ),
                reader(getattr(refs, name + "_file")(shot)).drop(
                    columns=["run_id", "written_at"]
                ),
                check_exact=True,
            )


def test_device_overlap_and_compaction_are_exact():
    import os

    import numpy as np
    import torch

    device = os.environ.get("L14PERF_TEST_DEVICE", "cpu")
    rng = np.random.default_rng(1401)
    spec = np.zeros((512, 1999), dtype=np.float32)
    _, meta = masks.tile(spec)
    pred = rng.random((meta["n_tiles"], 2, 512, 512), dtype=np.float32)
    threshold = np.float32(masks.PROB_THRESHOLD)
    pred[:, :, :3] = np.array(
        [
            np.nextafter(threshold, np.float32(0)),
            threshold,
            np.nextafter(threshold, np.float32(1)),
        ]
    )[None, None, :, None]
    expected = masks.stitch(pred, meta)
    accumulator = masks.DeviceStitch(1999, device)
    for start in range(0, len(pred), 2):
        accumulator.add(torch.from_numpy(pred[start : start + 2]).to(device), start)
    result = accumulator.finish()
    coh, tra = expected >= threshold
    assert np.array_equal(result.coh_packed, masks.pack(coh))
    assert np.array_equal(result.tra_packed, masks.pack(tra))
    assert np.array_equal(result.row_lit, coh.mean(axis=1).astype(np.float32))
    assert np.array_equal(result.col_act, tra.mean(axis=0).astype(np.float32))
    assert np.array_equal(result.coh_values, expected[0][coh])


def test_pooled_oom_retries_transfer_and_forward(monkeypatch):
    import numpy as np
    import torch

    class Net(torch.nn.Module):
        def forward(self, x):
            if len(x) > 1:
                raise torch.cuda.OutOfMemoryError("test forward budget")
            return (torch.cat([x, -x], dim=1),)

    original = masks._copy_batch
    calls = []

    def copy(host, device, stream):
        calls.append(len(host))
        if len(host) > 1:
            raise torch.cuda.OutOfMemoryError("test transfer budget")
        return original(host, device, stream)

    specs = [
        np.random.default_rng(i).normal(size=(512, 999)).astype(np.float32)
        for i in range(3)
    ]
    expected = [masks.infer(Net(), spec, "cpu", batch=1) for spec in specs]
    monkeypatch.setattr(masks, "_copy_batch", copy)
    actual = list(masks.infer_pooled(Net(), enumerate(specs), "cpu", batch=5,
                           preserve_batches=False))
    assert calls[0] == 5 and 1 in calls
    assert [token for token, _ in actual] == [0, 1, 2]
    for (_, compact), reference in zip(actual, expected, strict=True):
        lit = reference[0] >= masks.PROB_THRESHOLD
        assert np.array_equal(compact.coh_packed, masks.pack(lit))
        assert np.array_equal(compact.coh_values, reference[0][lit])


def test_pooled_model_failure_preserves_later_blocks():
    import numpy as np
    import torch

    class Net(torch.nn.Module):
        def forward(self, x):
            if float(x[0, 0, 0, 0]) == 1:
                raise ValueError("bad block")
            return (torch.cat([x, -x], dim=1),)

    specs = [np.full((512, 20), i, dtype=np.float32) for i in range(3)]
    result = list(masks.infer_pooled(Net(), enumerate(specs), "cpu", batch=1))
    assert [token for token, _ in result] == [0, 1, 2]
    assert isinstance(result[0][1], masks.CompactMask)
    assert isinstance(result[1][1], ValueError)
    assert isinstance(result[2][1], masks.CompactMask)


def test_pooled_sources_keep_disjoint_coverage(tmp_path):
    from .coverage_fixture import gapped_filterscopes

    corpus = tmp_path / "corpus"
    expected_intervals = gapped_filterscopes(corpus)
    reference = _paths_under(tmp_path, "reference", corpus)
    pooled = _paths_under(tmp_path, "pooled", corpus)
    ref = pipeline.process_shot(198658, reference, model=None, passes=("wide",))
    result = driver.run_shots(
        [198658],
        paths=pooled,
        model=None,
        passes=("wide",),
        prep_workers=1,
        prefetch=4,
        tail_workers=1,
        pooled=True,
    )
    assert not ref.error and result.rows[0]["status"] == "ok"
    a = schema.read_sources(reference.sources_file(198658)).drop(
        columns=["run_id", "written_at"]
    )
    b = schema.read_sources(pooled.sources_file(198658)).drop(
        columns=["run_id", "written_at"]
    )
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    clock = b[b.source == "elm_clock"].iloc[0]
    assert json.loads(clock.intervals) == [list(i) for i in expected_intervals]
    assert clock.min_gap_s > 0


def test_copy_stream_pool_matches_full_probability_reference():
    import os

    import numpy as np
    import torch

    class Net(torch.nn.Module):
        def forward(self, x):
            position = torch.linspace(-1, 1, 512, device=x.device)[None, None, None]
            logits = x + position
            return (torch.cat([logits, -logits], dim=1),)

    device = os.environ.get("L14PERF_TEST_DEVICE", "cpu")
    specs = [
        np.random.default_rng(i).normal(size=(512, width)).astype(np.float32)
        for i, width in enumerate([20, 1999, 777])
    ]
    model = Net().to(device).eval()
    reference = [masks.infer(model, spec, device, batch=2, amp=False) for spec in specs]
    actual = list(
        masks.infer_pooled(model, enumerate(specs), device, batch=3, amp=False)
    )
    assert [key for key, _ in actual] == [0, 1, 2]
    for (_, compact), expected in zip(actual, reference, strict=True):
        coh, tra = expected >= masks.PROB_THRESHOLD
        assert np.array_equal(compact.coh_packed, masks.pack(coh))
        assert np.array_equal(compact.tra_packed, masks.pack(tra))
        assert np.array_equal(compact.coh_values, expected[0][coh])
        assert np.array_equal(compact.row_lit, coh.mean(axis=1).astype(np.float32))
        assert np.array_equal(compact.col_act, tra.mean(axis=0).astype(np.float32))


def test_pooled_timeout_is_an_error_and_next_shot_runs(tmp_path, synth_shot):
    import time

    corpus = tmp_path / "corpus"
    paths = _paths_under(tmp_path, "pooled", corpus)
    for shot in [SHOT, SHOT + 1]:
        _write_corpus(corpus, shot, synth_shot)

    class SlowOnce(PaintedNet):
        slow = True

        def forward(self, x):
            if self.slow:
                self.slow = False
                time.sleep(5)
            return super().forward(x)

    result = driver.run_shots(
        [SHOT, SHOT + 1],
        paths=paths,
        model=SlowOnce(),
        prep_workers=0,
        prefetch=1,
        tail_workers=0,
        pooled=True,
        tile_batch=1,
        timeout_s=2,
    )
    assert result.rows[0]["status"] == "error"
    assert result.rows[1]["status"] == "ok"
    assert result.rows[1]["n_blocks"] == 14


def test_full_transfers_preserve_reference_forward_boundaries(monkeypatch):
    import numpy as np
    import torch

    class CountedNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batches = []

        def forward(self, x):
            self.batches.append(len(x))
            return (torch.cat([x, -x], dim=1),)

    copies = []
    original = masks._copy_batch

    def copy(host, device, stream):
        copies.append(len(host))
        return original(host, device, stream)

    monkeypatch.setattr(masks, "_copy_batch", copy)
    specs = [np.zeros((512, width), dtype=np.float32) for width in [960, 1999, 777]]
    model = CountedNet()
    actual = list(
        masks.infer_pooled(
            model, enumerate(specs), "cpu", batch=3
        )
    )
    assert copies == [3, 3, 3]
    assert model.batches == [2, 3, 2, 2]
    assert [token for token, _ in actual] == [0, 1, 2]
    assert all(isinstance(value, masks.CompactMask) for _, value in actual)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_batches_are_the_default_on_every_device(device):
    import torch

    assert masks._preserves_reference_batches(torch.device(device)) is True


@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("widths", [[20, 20, 20], [960, 1999, 777, 20]])
def test_reference_transfer_oom_is_bounded_and_isolated(monkeypatch, failures, widths):
    import numpy as np
    import torch

    original = masks._copy_batch
    calls, clears = [], []

    def copy(host, device, stream):
        calls.append(len(host))
        # Fail the second buffer, including a block with a pending group.
        if 2 <= len(calls) < 2 + failures:
            raise torch.cuda.OutOfMemoryError("transfer failed")
        return original(host, device, stream)

    monkeypatch.setattr(masks, "_copy_batch", copy)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: clears.append(True))
    specs = [np.zeros((512, width), dtype=np.float32) for width in widths]
    batch = 1 if widths[0] == 20 else 3
    actual = list(masks.infer_pooled(
        PaintedNet(), enumerate(specs), "cpu", batch=batch, preserve_batches=True,
    ))
    assert [token for token, _ in actual] == list(range(len(specs)))
    failed = [token for token, value in actual if isinstance(value, Exception)]
    assert failed == ([1] if failures == 2 else [])
    assert isinstance(actual[-1][1], masks.CompactMask)
    assert len(clears) == 1


def test_nonreference_oom_size_resets_at_block_boundary():
    import numpy as np
    import torch

    class OOMOnce(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batches = []

        def forward(self, x):
            self.batches.append(len(x))
            if len(self.batches) == 1:
                raise torch.cuda.OutOfMemoryError("one transient forward OOM")
            return (torch.cat([x, -x], dim=1),)

    # Six tiles: block 1 ends halfway through the second transfer buffer.
    specs = [np.zeros((512, 2600), dtype=np.float32) for _ in range(3)]
    model = OOMOnce()
    actual = list(masks.infer_pooled(
        model, enumerate(specs), "cpu", batch=4, preserve_batches=False,
    ))
    assert all(isinstance(value, masks.CompactMask) for _, value in actual)
    assert model.batches == [4, 2, 2, 2, 2, 4, 4, 2]


@pytest.mark.parametrize("preserve", [False, True])
def test_shared_batch_timeout_only_fails_the_expired_shot(monkeypatch, preserve):
    import time

    import numpy as np
    import torch

    from labeler.run import StageTimeout, time_limit

    class SlowOnce(torch.nn.Module):
        slow = True

        def forward(self, x):
            if self.slow:
                self.slow = False
                time.sleep(2)
            return (torch.cat([x, -x], dim=1),)

    at = time.monotonic()
    deadlines = {0: at + 0.5, 1: at + 10}

    def remaining(token):
        return deadlines[token] - time.monotonic()

    def guard(tokens):
        if min(remaining(token) for token in tokens) <= 0:
            raise StageTimeout("shot expired")
        return time_limit(1)

    specs = [np.zeros((512, 960), dtype=np.float32) for _ in range(2)]
    actual = list(masks.infer_pooled(
        SlowOnce(), enumerate(specs), "cpu", batch=4, preserve_batches=preserve,
        forward_context=guard, remaining=remaining,
    ))
    assert [token for token, _ in actual] == [0, 1]
    assert isinstance(actual[0][1], StageTimeout)
    assert isinstance(actual[1][1], masks.CompactMask)


@pytest.mark.parametrize("broken", ["interleaved", "last", "extra"])
def test_reference_group_invariants_raise_real_exceptions(broken):
    from contextlib import nullcontext

    import torch

    ref = masks._ReferenceBatches(PaintedNet(), torch.device("cpu"), 4, False,
                                  nullcontext)
    state = masks._PooledBlock(0, 960)
    copied = torch.zeros((3, 1, 512, 512))
    if broken == "interleaved":
        ref.consume(copied, [(state, 0, 1, 0, False)])
        spans = [(masks._PooledBlock(1, 20), 0, 1, 0, True)]
    else:
        spans = [(state, 0, 3 if broken == "extra" else 2, 0, False)]
    with pytest.raises(RuntimeError, match="reference"):
        ref.consume(copied, spans)


def test_compact_activity_rejects_a_different_threshold(tmp_path, synth_shot,
                                                       monkeypatch):
    from labeler.events import channels, transients

    corpus = tmp_path / "corpus"
    _write_corpus(corpus, SHOT, synth_shot)
    y, fs, t0, t1 = masks.read_waveform(corpus / f"{SHOT}_processed.h5", "mhr", 0)
    prepared = pipeline.prep_block(y, fs, t0, t1,
                                  channels.ChannelSpec("mhr", 0, "magnetics"),
                                  "wide", norm="record", window=None)
    compact = masks.infer(PaintedNet(), prepared.spectrogram, "cpu", compact=True)
    monkeypatch.setattr(transients, "ACTIVITY_THR", masks.PROB_THRESHOLD + 0.1)
    with pytest.raises(AssertionError):
        pipeline.describe_block(prepared, compact, unet_sha256=FAKE_SHA)


@pytest.mark.parametrize("pool_cpu_forwards", [False, True])
def test_driver_shared_batch_deadlines_are_attributed_per_shot(
    tmp_path, synth_shot, monkeypatch, pool_cpu_forwards,
):
    import time

    from labeler.events import channels

    corpus = tmp_path / "corpus"
    paths = _paths_under(tmp_path, "deadlines", corpus)
    for shot in [SHOT, SHOT + 1]:
        _write_corpus(corpus, shot, synth_shot)
    original = driver._PooledShot

    def staggered(*args, **kwargs):
        state = original(*args, **kwargs)
        if state.shot == SHOT:
            state.started -= 9.5
        return state

    class SlowOnce(PaintedNet):
        slow = True

        def forward(self, x):
            if self.slow:
                self.slow = False
                time.sleep(5)
            return super().forward(x)

    monkeypatch.setattr(driver, "_PooledShot", staggered)
    result = driver.run_shots(
        [SHOT, SHOT + 1], paths=paths, model=SlowOnce(), prep_workers=0,
        prefetch=1, tail_workers=0, pooled=True, pool_cpu_forwards=pool_cpu_forwards,
        plan=(channels.ChannelSpec("mhr", 0, "magnetics"),),
        tile_batch=4, timeout_s=10,
    )
    assert [row["status"] for row in result.rows] == ["error", "ok"]
    assert result.rows[1]["n_blocks"] == 2
    assert not any("StageTimeout" in value
                   for value in result.rows[1]["skipped"].values())
