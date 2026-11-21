#!/usr/bin/env python3
"""Fetch older shots in descending batches without enumerating all history first."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import threading

import bulk_pipeline as bulk
from plan_descending import descending_plan


def status(root, state):
    snapshot = dict(state)
    totals = dict(completed=0, failed=0, planned=0, in_flight=[])
    for batch in sorted((root / 'batches').glob('*')):
        plan = batch / 'plan.json'
        if plan.is_file():
            totals['planned'] += len(json.loads(plan.read_text())['shots'])
        progress = batch / 'status.json'
        if progress.is_file():
            record = json.loads(progress.read_text())
            totals['completed'] += record['completed']
            totals['failed'] += record['failed']
            totals['in_flight'].extend(record['in_flight'])
            if str(batch) == snapshot.get('current_batch'):
                snapshot['batch_status'] = record['status']
    snapshot.update(totals)
    bulk.pilot.save_json(root / 'status.json', snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--start', type=int, default=190000)
    parser.add_argument('--oldest', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--token-dir', type=Path, default=(
        bulk.pilot.MODELS / 'ignite_fdp_below190000_v4/frame_codes'))
    parser.add_argument('--validation-report', type=Path, default=(
        bulk.pilot.REPO / 'scratch/fdp_pilot_207801/report.json'))
    args = parser.parse_args()
    if not 100000 <= args.oldest <= args.start <= 190000:
        parser.error('Require 100000 <= oldest <= start <= 190000')
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    root = args.campaign.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.descending.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'batches').mkdir(exist_ok=True)
    cached_plan = (json.loads((root / 'plan.json').read_text())
                   if (root / 'plan.json').exists() else None)
    state = dict(status='planning', direction='descending', start=args.start,
                 oldest=args.oldest, concurrency=args.concurrency,
                 token_dir=str(args.token_dir.resolve()),
                 batch_size=args.batch_size)
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            status(root, state)
            stop.wait(15)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        upper = args.start
        while upper >= args.oldest:
            lower = max(args.oldest, upper - args.batch_size + 1)
            batch = root / 'batches' / f'{upper}_{lower}'
            state.update(status='planning', current_batch=str(batch))
            if not (batch / 'plan.json').exists():
                options = argparse.Namespace(
                    campaign=batch, start=lower, end=upper,
                    concurrency=args.concurrency, token_dir=args.token_dir,
                    validation_report=args.validation_report,
                )
                if cached_plan is not None:
                    selected = [s for s in cached_plan['shots']
                                if lower <= s <= upper]
                    if not selected:
                        upper = lower - 1
                        continue
                    batch.mkdir(parents=True, exist_ok=True)
                    plan = {**cached_plan, 'shots': selected,
                            'concurrency': args.concurrency}
                    bulk.pilot.save_json(batch / 'plan.json', plan)
                else:
                    try:
                        bulk.make_plan(options)
                    except ValueError as exc:
                        if str(exc) == 'No available unused shots in requested range':
                            upper = lower - 1
                            continue
                        raise
                plan_path = batch / 'plan.json'
                plan = descending_plan(json.loads(plan_path.read_text()), upper)
                source = Path(__file__).resolve()
                plan['source_sha256'][str(source)] = bulk.pilot.digest(source)
                bulk.pilot.save_json(plan_path, plan)
                (batch / 'shots.txt').write_text(
                    ''.join(f'{s}\n' for s in plan['shots']))
            state['status'] = 'running'
            bulk.run_campaign(argparse.Namespace(campaign=batch))
            completed = json.loads((batch / 'status.json').read_text())
            if completed['status'] == 'failure_limit':
                state['status'] = 'failure_limit'
                break
            upper = lower - 1
        else:
            state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop.set()
        thread.join()
        status(root, state)


if __name__ == '__main__':
    main()
