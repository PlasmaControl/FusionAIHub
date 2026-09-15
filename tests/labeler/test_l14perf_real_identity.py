"""Opt-in byte identity against the original CPU pipeline on a real shot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.events import driver, masks, pipeline, schema, unet


def _digest_masks(path):
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            value = data[key]
            digest.update(key.encode())
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _stable_frame(path, reader):
    return reader(path).drop(columns=["run_id", "written_at"])


def _digest_frame(frame):
    return hashlib.sha256(
        frame.to_json(orient="table", double_precision=15).encode()
    ).hexdigest()


def test_real_shot_cpu_output_identity():
    requested = os.environ.get("L14PERF_REAL_ROOT")
    if not requested:
        pytest.skip("set L14PERF_REAL_ROOT to an l14perf scratch directory")
    root = Path(requested).resolve()
    assert root.is_relative_to("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf")
    print(f"Resolved real identity root: {root}", flush=True)
    production = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
    shot = 185786
    device = os.environ.get("L14PERF_REAL_DEVICE", "cpu")
    batch = int(os.environ.get("L14PERF_REAL_BATCH", "8"))
    amp = os.environ.get("L14PERF_REAL_AMP") == "1"
    model = unet.load_unet(
        production / "models/tokeye/big_tf_unet_251210.pt", device=device
    )
    paths = []
    stages = os.environ.get("L14PERF_STAGES", "compact,pooled,tail,prep").split(",")
    assert set(stages) <= {"compact", "pooled", "tail", "prep"}
    for name in ("legacy", *stages):
        item = Paths(root=root / name)
        item.mkdirs()
        shutil.copyfile(production / "text/logs_subset.jsonl", item.logs_subset)
        paths.append(item)
    reference_root = os.environ.get("L14PERF_REFERENCE_ROOT")
    if reference_root:
        paths[0] = Paths(root=Path(reference_root).resolve())
        assert paths[0].root.is_relative_to(root.parent)
        events = schema.read_events(paths[0].events_file(shot))
        assert set(events.git_sha) == {driver.git_sha()}, "reference HEAD changed"
        ref = SimpleNamespace(
            error="",
            n_blocks=len(masks.list_blocks(paths[0].masks_file(shot))),
            n_events=len(events),
        )
    else:
        ref = pipeline.process_shot(
            shot,
            paths[0],
            model=model,
            device=device,
            passes=("wide", "zoom"),
            tile_batch=batch,
            amp=amp,
            run_id="identity",
        )
    assert not ref.error and ref.n_blocks > 0
    # Each step is compared against the unchanged full-probability reference.
    for stage, item in zip(stages, paths[1:], strict=True):
        if stage == "compact":
            original_infer = pipeline.infer_block

            def compact_infer(prepared, **kwargs):
                return masks.infer(
                    model,
                    prepared.spectrogram,
                    device,
                    batch=batch,
                    amp=amp,
                    compact=True,
                )

            try:
                pipeline.infer_block = compact_infer
                result = pipeline.process_shot(
                    shot,
                    item,
                    model=model,
                    device=device,
                    passes=("wide", "zoom"),
                    tile_batch=batch,
                    amp=amp,
                    run_id="identity",
                )
                assert not result.error and result.n_blocks == ref.n_blocks
            finally:
                pipeline.infer_block = original_infer
        else:
            result = driver.run_shots(
                [shot],
                paths=item,
                model=model,
                device=device,
                passes=("wide", "zoom"),
                tile_batch=batch,
                amp=amp,
                prep_workers=2 if stage == "prep" else 0,
                prefetch=2 if stage == "prep" else 1,
                tail_workers=int(stage in {"tail", "prep"}),
                index=False,
                run_id="identity",
                pooled=True,
                timeout_s=3600,
            )
            assert result.rows[0]["status"] == "ok"
            assert result.rows[0]["n_blocks"] == ref.n_blocks
    evidence = {
        "shot": shot,
        "device": device,
        "tile_batch": batch,
        "amp": amp,
        "checkpoint": unet.CHECKPOINT_SHA256,
        "n_blocks": ref.n_blocks,
        "n_events": ref.n_events,
        "paths": {},
    }
    for item in paths:
        with (
            np.load(paths[0].masks_file(shot), allow_pickle=False) as reference,
            np.load(item.masks_file(shot), allow_pickle=False) as actual,
        ):
            assert actual.files == reference.files
            for key in reference.files:
                assert actual[key].dtype == reference[key].dtype, key
                assert actual[key].tobytes() == reference[key].tobytes(), key
        assert (
            item.masks_file(shot).read_bytes() == paths[0].masks_file(shot).read_bytes()
        )
        hashes = {
            "masks_npz": hashlib.sha256(item.masks_file(shot).read_bytes()).hexdigest(),
            "masks_arrays": _digest_masks(item.masks_file(shot)),
        }
        for name, getter, reader in (
            ("events", Paths.events_file, schema.read_events),
            ("sources", Paths.sources_file, schema.read_sources),
        ):
            actual = _stable_frame(getter(item, shot), reader)
            reference = _stable_frame(getter(paths[0], shot), reader)
            pd.testing.assert_frame_equal(actual, reference, check_exact=True)
            hashes[name] = _digest_frame(actual)
        evidence["paths"][str(item.root)] = hashes
    (root / "identity.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)
