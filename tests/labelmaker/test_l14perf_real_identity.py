"""Opt-in byte identity against the original CPU pipeline on a real shot."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labelmaker.config import Paths
from labelmaker.events import driver, pipeline, schema, unet


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
    return reader(path).drop(columns=['run_id', 'written_at'])


def _digest_frame(frame):
    return hashlib.sha256(frame.to_json(orient='table', double_precision=15)
                          .encode()).hexdigest()


def test_real_shot_cpu_output_identity():
    requested = os.environ.get('L14PERF_REAL_ROOT')
    if not requested:
        pytest.skip('set L14PERF_REAL_ROOT to an l14perf scratch directory')
    root = Path(requested).resolve()
    assert root.is_relative_to(
        '/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf'
    )
    print(f'Resolved real identity root: {root}', flush=True)
    production = Path('/scratch/gpfs/EKOLEMEN/nc1514/labelmaker')
    shot = 185786
    model = unet.load_unet(
        production / 'models/tokeye/big_tf_unet_251210.pt', device='cpu'
    )
    paths = []
    for name in ('legacy', 'compact', 'pooled'):
        item = Paths(root=root / name)
        item.mkdirs()
        shutil.copyfile(production / 'text/logs_subset.jsonl', item.logs_subset)
        paths.append(item)
    ref = pipeline.process_shot(
        shot, paths[0], model=model, device='cpu', passes=('wide', 'zoom'),
        tile_batch=8, amp=False, index=False, run_id='identity',
    )
    assert not ref.error and ref.n_blocks > 0
    # The compact-only path preserves tile grouping; pooled deliberately changes it.
    original_infer = pipeline.infer_block
    def compact_infer(prepared, **kwargs):
        from labelmaker.events import masks
        return masks.infer(model, prepared.spectrogram, 'cpu', batch=8,
                           amp=False, compact=True)
    try:
        pipeline.infer_block = compact_infer
        compact = pipeline.process_shot(
            shot, paths[1], model=model, device='cpu', passes=('wide', 'zoom'),
            tile_batch=8, amp=False, index=False, run_id='identity',
        )
        assert not compact.error and compact.n_blocks == ref.n_blocks
    finally:
        pipeline.infer_block = original_infer
    result = driver.run_shots(
        [shot], paths=paths[2], model=model, device='cpu',
        passes=('wide', 'zoom'), tile_batch=8, amp=False,
        prep_workers=2, prefetch=2, tail_workers=1, index=False,
        run_id='identity', pooled=True,
    )
    assert result.rows[0]['status'] == 'ok'
    evidence = {'shot': shot, 'checkpoint': unet.CHECKPOINT_SHA256,
                'n_blocks': ref.n_blocks, 'n_events': ref.n_events, 'paths': {}}
    for item in paths:
        with np.load(paths[0].masks_file(shot), allow_pickle=False) as reference:
            with np.load(item.masks_file(shot), allow_pickle=False) as actual:
                assert actual.files == reference.files
                for key in reference.files:
                    assert actual[key].dtype == reference[key].dtype, key
                    assert actual[key].tobytes() == reference[key].tobytes(), key
        hashes = {'masks': _digest_masks(item.masks_file(shot))}
        for name, getter, reader in (
            ('events', Paths.events_file, schema.read_events),
            ('sources', Paths.sources_file, schema.read_sources),
        ):
            actual = _stable_frame(getter(item, shot), reader)
            reference = _stable_frame(getter(paths[0], shot), reader)
            pd.testing.assert_frame_equal(actual, reference, check_exact=True)
            hashes[name] = _digest_frame(actual)
        evidence['paths'][str(item.root)] = hashes
    (root / 'identity.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2), flush=True)
