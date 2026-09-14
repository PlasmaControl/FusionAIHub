"""Instrumentation of the unchanged driver; products and traces require --root."""
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import Future
from pathlib import Path

import torch
from labelmaker.events import driver, masks, pipeline, schema


def main():
    args = sys.argv[1:]
    root = Path(args[args.index('--root') + 1]).resolve()
    assert root.is_relative_to('/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf')
    print(f'PROFILE resolved root: {root}', flush=True)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    writes = defaultdict(float)
    active = None
    pending_wait = 0.0
    original_result = Future.result

    def result(self, *a, **kw):
        nonlocal pending_wait
        start = time.perf_counter()
        value = original_result(self, *a, **kw)
        if isinstance(value, pipeline.PreparedBlock):
            pending_wait += time.perf_counter() - start
        return value
    Future.result = result

    def timed(owner, name, phase, *, sync=False):
        original = getattr(owner, name)
        def wrapped(*a, **kw):
            if sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.profiler.record_function('l14perf::' + phase):
                value = original(*a, **kw)
                if sync:
                    torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if active is not None:
                active[phase] = active.get(phase, 0) + seconds
            else:
                writes[phase] += seconds
            return value
        setattr(owner, name, wrapped)
    timed(masks, 'tile', 'tile_s')
    timed(masks, '_to_device', 'h2d_s', sync=True)
    timed(masks, 'probabilities', 'forward_s', sync=True)
    timed(masks, 'stitch', 'stitch_s')
    timed(masks, 'write_masks', 'write_masks_s')
    timed(schema, 'write_events', 'write_events_s')
    timed(schema, 'write_sources', 'write_sources_s')
    original_to = torch.Tensor.to
    def to(self, *a, **kw):
        if active is not None and self.is_cuda and a and str(a[0]) == 'cpu':
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.profiler.record_function('l14perf::d2h_s'):
                value = original_to(self, *a, **kw)
                torch.cuda.synchronize()
            active['d2h_s'] = active.get('d2h_s', 0) + time.perf_counter() - start
            return value
        return original_to(self, *a, **kw)
    torch.Tensor.to = to
    original_infer = pipeline.infer_block
    def infer(prepared, **kw):
        nonlocal active, pending_wait
        active = {'key': prepared.key, 'n_tiles': prepared.n_tiles,
                  'prep_wait_s': pending_wait}
        pending_wait = 0
        rows.append(active)
        start = time.perf_counter()
        value = original_infer(prepared, **kw)
        active['infer_s'] = time.perf_counter() - start
        return value
    pipeline.infer_block = infer
    original_describe = pipeline.describe_block
    def describe(*a, **kw):
        nonlocal active
        start = time.perf_counter()
        value = original_describe(*a, **kw)
        active['describe_s'] = time.perf_counter() - start
        active = None
        return value
    pipeline.describe_block = describe
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        code = driver.main(args)
    prof.export_chrome_trace(str(root / 'trace.json'))
    (root / 'operators.txt').write_text(prof.key_averages().table(
        sort_by='self_cuda_time_total', row_limit=35))
    summary = {key: sum(row.get(key, 0) for row in rows)
               for key in ['prep_wait_s', 'tile_s', 'h2d_s', 'forward_s',
                           'd2h_s', 'stitch_s', 'describe_s', 'infer_s', 'n_tiles']}
    (root / 'profile.json').write_text(json.dumps(
        {'blocks': rows, 'totals': summary, 'write': dict(writes)}, indent=2))
    print(json.dumps({'totals': summary, 'write': dict(writes)}, indent=2), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
