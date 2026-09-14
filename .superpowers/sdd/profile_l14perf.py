"""Profile either driver schedule; all products require an explicit scratch root."""
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
    totals = defaultdict(float)
    rows, batches, prep = [], [], []
    original_result = Future.result

    def result(self, *a, **kw):
        start = time.perf_counter()
        value = original_result(self, *a, **kw)
        if isinstance(value, pipeline.PreparedBlock):
            waited = time.perf_counter() - start
            totals['prep_wait_s'] += waited
            prep.append({'key': value.key, 'wait_s': waited,
                         'work_s': value.prep_seconds})
        elif isinstance(value, tuple) and len(value) == 3 and isinstance(value[1], dict):
            totals['plan_wait_s'] += time.perf_counter() - start
        return value
    Future.result = result

    def timed(owner, name, phase, *, sync=False):
        original = getattr(owner, name)
        def wrapped(*a, **kw):
            if sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            nested_d2h = totals['d2h_s']
            with torch.profiler.record_function('l14perf::' + phase):
                value = original(*a, **kw)
                if sync:
                    torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if phase == 'compact_s':
                seconds -= totals['d2h_s'] - nested_d2h
            totals[phase] += seconds
            return value
        setattr(owner, name, wrapped)

    timed(masks, 'tile', 'tile_s')
    timed(masks, '_fill_tiles', 'tile_s')
    timed(masks, '_to_device', 'h2d_s', sync=True)
    timed(masks, '_copy_batch', 'h2d_s', sync=True)
    original_forward = masks.probabilities
    def forward(model, x):
        value = original_forward(model, x)
        batches.append(len(x))
        return value
    masks.probabilities = forward
    timed(masks, 'probabilities', 'forward_s', sync=True)
    timed(masks, 'stitch', 'stitch_s')
    timed(masks.DeviceStitch, 'add', 'stitch_s', sync=True)
    timed(masks.DeviceStitch, 'finish', 'compact_s', sync=True)
    timed(pipeline, 'describe_block', 'describe_s')
    for owner, name in [(masks, 'write_masks'), (schema, 'write_events'),
                        (schema, 'write_sources')]:
        timed(owner, name, name + '_s')

    original_to = torch.Tensor.to
    def to(self, *a, **kw):
        if self.is_cuda and a and str(a[0]) == 'cpu':
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.profiler.record_function('l14perf::d2h_s'):
                value = original_to(self, *a, **kw)
                torch.cuda.synchronize()
            totals['d2h_s'] += time.perf_counter() - start
            totals['d2h_bytes'] += value.numel() * value.element_size()
            return value
        return original_to(self, *a, **kw)
    torch.Tensor.to = to

    original_infer = masks.infer_pooled
    def pooled(*a, **kw):
        iterator = iter(original_infer(*a, **kw))
        while True:
            start = time.perf_counter()
            waits = totals['prep_wait_s'] + totals['plan_wait_s']
            value = next(iterator, None)
            elapsed = time.perf_counter() - start
            totals['infer_with_input_wait_s'] += elapsed
            totals['infer_s'] += elapsed - (totals['prep_wait_s'] + totals['plan_wait_s'] - waits)
            if value is None:
                return
            (state, prepared), compact = value
            rows.append({'shot': state.shot, 'key': prepared.key,
                         'n_tiles': prepared.n_tiles,
                         'prep_work_s': prepared.prep_seconds,
                         'output_bytes': sum(getattr(compact, k).nbytes for k in
                             ['coh_packed', 'tra_packed', 'row_lit', 'col_act', 'coh_values'])})
            yield value
    masks.infer_pooled = pooled

    original_block = pipeline.infer_block
    def block(prepared, **kw):
        start = time.perf_counter()
        value = original_block(prepared, **kw)
        totals['infer_s'] += time.perf_counter() - start
        rows.append({'key': prepared.key, 'n_tiles': prepared.n_tiles})
        return value
    pipeline.infer_block = block

    start = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        code = driver.main(args)
    elapsed = time.perf_counter() - start
    prof.export_chrome_trace(str(root / 'trace.json'))
    (root / 'operators.txt').write_text(prof.key_averages().table(
        sort_by='self_cuda_time_total', row_limit=35))
    totals['n_tiles'] = sum(row['n_tiles'] for row in rows)
    totals['prep_work_s'] = sum(row.get('work_s', 0) for row in prep)
    payload = {'blocks': rows, 'totals': dict(totals), 'prep': prep,
               'batches': batches, 'wall_s': elapsed,
               'peak_alloc_gib': torch.cuda.max_memory_allocated() / 2**30}
    (root / 'profile.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: v for k, v in payload.items() if k not in ['blocks', 'prep']},
                     indent=2), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
