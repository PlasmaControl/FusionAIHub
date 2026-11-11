#!/usr/bin/env python3
"""Plan older FDP shots without changing the running upward campaign's code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bulk_pipeline as bulk


def descending_plan(plan, upper):
    """Order selected, available shots from the requested upper bound downward."""
    shots = sorted((s for s in plan['shots'] if s <= upper), reverse=True)
    if not shots:
        raise ValueError('No older unused FDP shots found')
    plan.update(shots=shots, direction='descending', start=upper, end=min(shots))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--start', type=int, default=190000)
    parser.add_argument('--oldest', type=int, default=100000,
                        help='Lowest six-digit shot supported by this pipeline')
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--token-dir', type=Path, default=(
        bulk.pilot.MODELS / 'ignite_fdp_below190000_v4/frame_codes'))
    parser.add_argument('--validation-report', type=Path, default=(
        bulk.pilot.REPO / 'scratch/fdp_pilot_207801/report.json'))
    args = parser.parse_args()
    if not 100000 <= args.oldest <= args.start <= 190000:
        parser.error('Require 100000 <= oldest <= start <= 190000')
    upper = args.start
    args.start, args.end = args.oldest, upper
    bulk.make_plan(args)
    target = args.campaign.resolve() / 'plan.json'
    plan = descending_plan(json.loads(target.read_text()), upper)
    source = Path(__file__).resolve()
    plan['source_sha256'][str(source)] = bulk.pilot.digest(source)
    bulk.pilot.save_json(target, plan)
    (target.parent / 'shots.txt').write_text(
        ''.join(f'{s}\n' for s in plan['shots']))
    bulk.log(f'Descending plan: {len(plan["shots"])} shots; '
             f'{plan["shots"][0]} down to {plan["shots"][-1]}; '
             f'{plan["concurrency"]} at a time')


if __name__ == '__main__':
    main()
