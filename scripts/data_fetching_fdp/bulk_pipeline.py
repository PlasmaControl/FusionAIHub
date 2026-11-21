#!/usr/bin/env python3
"""Resumable login-host FDP campaign with bounded CPU Slurm stages."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import validate_pipeline as pilot

HERE = Path(__file__).resolve().parent
FDP_PROJECT = Path('/scratch/gpfs/EKOLEMEN/ps9551/fdp-demo')
INDEX_ROOT = '/fdp-d3d/archives/index/json'
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(f'{time.strftime("%F %T")} {message}', flush=True)


def active_shots():
    """Exclude actual training/evaluation/cache shots, not legacy candidate lists."""
    root = pilot.PRODUCTION
    shots = {int(p.stem) for p in (root / 'frame_codes').glob('[0-9]*.pt')}
    splits = sorted((root / 'runs').glob('*/split_shots.json'))
    if not shots or not splits:
        raise ValueError('Active cache or split files unavailable')
    for path in splits:
        doc = json.loads(path.read_text())
        for key in ('train', 'val', 'test', 'pin_train', 'pin_val', 'pin_both'):
            shots.update(int(s) for s in doc.get(key, []))
    return shots


def fdp_ls(path):
    result = subprocess.run(
        ['pixi', 'run', '--frozen', '--manifest-path',
         str(FDP_PROJECT / 'pixi.toml'), 'fdp', 'ls', path],
        capture_output=True, text=True, check=True, timeout=180,
    )
    return [line.strip().rstrip('/').rsplit('/', 1)[-1]
            for line in result.stdout.splitlines() if line.strip()]


def make_plan(args):
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    target = campaign / 'plan.json'
    if target.exists():
        raise ValueError(f'Plan already exists: {target}')
    pilot.check_token()
    snapshots = sorted(s for s in fdp_ls(INDEX_ROOT)
                       if s.startswith('json_indexes_'))
    snapshot = snapshots[-1]
    root = f'{INDEX_ROOT}/{snapshot}'
    # Ignore special calibration/test-shot namespaces (700xxx, 94xxxx, ...).
    prefixes = sorted(p for p in fdp_ls(root)
                      if re.fullmatch(r'\d{4}', p)
                      and args.start // 100 <= int(p) < 3000
                      and (not args.end or int(p) <= args.end // 100))
    available = set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for names in pool.map(fdp_ls, [f'{root}/{p}' for p in prefixes]):
            available.update(int(n[:-5]) for n in names
                             if re.fullmatch(r'\d{6}\.json', n))
    used = active_shots()
    shots = sorted(s for s in available if s >= args.start and s not in used
                   and (not args.end or s <= args.end))
    if not shots:
        raise ValueError('No available unused shots in requested range')
    source_files = [HERE / n for n in (
        'bulk_pipeline.py', 'validate_pipeline.py', 'fdp_stage.py',
        'fetch_shot.py', 'fetch_shot_fast.sh', 'mds_reader.py', 'config_fdp.yaml',
        '_without_index.sh',
    )]
    source_files += [pilot.REPO / 'scripts/data_preparation/prepare_data.py',
                     pilot.REPO / 'src/tokamak_foundation_model/data/config/'
                     'modalities/modalities.yaml']
    validation = json.loads(args.validation_report.read_text())
    if validation['status'] != 'validated' or not validation['fresh_fdp']:
        raise ValueError('Campaign needs a successful fresh FDP validation report')
    plan = dict(
        start=args.start, end=max(shots), shots=shots,
        snapshot=snapshot, concurrency=args.concurrency,
        token_dir=str(args.token_dir.resolve()), archive_dir=str(pilot.ARCHIVE),
        overwrite_raw=True, max_failures=5,
        source_sha256={str(p): pilot.digest(p) for p in source_files},
        validation_report=str(args.validation_report.resolve()),
        codec_sha256=validation['codec_sha256'],
    )
    pilot.save_json(target, plan)
    (campaign / 'shots.txt').write_text(''.join(f'{s}\n' for s in shots))
    log(f'Planned {len(shots)} shots: {shots[0]}–{shots[-1]}; '
        f'{args.concurrency} at a time; tokens={plan["token_dir"]}')


def archive_replace(raw, archive):
    """User-authorized replacement, with verified staging before publication."""
    expected = pilot.digest(raw)
    if archive.exists() and pilot.digest(archive) == expected:
        return expected
    tmp = archive.with_name(f'.{archive.name}.fdp-partial.{os.getpid()}')
    # Each shot is owned by one campaign worker. Stale partials are retained
    # for inspection rather than being reused or overwritten.
    with raw.open('rb') as src, tmp.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if pilot.digest(tmp) != expected:
        raise ValueError(f'Staged archive checksum mismatch: {tmp}')
    os.replace(tmp, archive)
    if pilot.digest(archive) != expected:
        raise ValueError(f'Published archive checksum mismatch: {archive}')
    return expected


def publish_tokens(work, token_dir, report, plan):
    """Publish only checked tokens and a consistent manifest; keep the H5 on error."""
    shot = report['shot']
    source = work / 'frame_codes' / f'{shot}.pt'
    manifest = json.loads((work / 'frame_codes/_codec_manifest.json').read_text())
    if report['status'] != 'validated':
        raise ValueError('Token publication requires validated output')
    if report['codec_sha256'] != plan['codec_sha256']:
        raise ValueError('Frozen codecs changed since the accepted pilot')
    if pilot.digest(source) != report['code_sha256']:
        raise ValueError('Token checksum changed before publication')
    token_dir.mkdir(parents=True, exist_ok=True)
    target = token_dir / f'{shot}.pt'
    with (token_dir / '.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta = token_dir / '_codec_manifest.json'
        if meta.exists() and json.loads(meta.read_text()) != manifest:
            raise ValueError('Token directory uses a different codec manifest')
        if not meta.exists():
            pilot.save_json(meta, manifest)
        if target.exists():
            if pilot.digest(target) != report['code_sha256']:
                raise ValueError(f'Token collision: {target}')
        else:
            tmp = target.with_suffix(f'.pt.tmp.{os.getpid()}')
            shutil.copyfile(source, tmp)
            if pilot.digest(tmp) != report['code_sha256']:
                raise ValueError('Token publication checksum mismatch')
            os.replace(tmp, target)
    return target


def finish_shot(work, token_dir, report, plan):
    """Delete the preprocessed H5 only after archive and token verification."""
    processed = work / 'processed' / f'{report["shot"]}_processed.h5'
    archive = Path(report['archive'])
    if not archive.is_file() or archive.stat().st_size != report['raw_bytes']:
        raise ValueError('Raw archive unavailable or changed size; preserving H5')
    if (archive.stat().st_mtime_ns != report.get('archive_mtime_ns')
            and pilot.digest(archive) != report['raw_sha256']):
        raise ValueError('Raw archive changed; preserving H5')
    if pilot.digest(processed) != report['processed_sha256']:
        raise ValueError('Processed H5 changed; preserving it')
    target = publish_tokens(work, token_dir, report, plan)
    if pilot.digest(target) != report['code_sha256']:
        raise ValueError('Published token verification failed; preserving H5')
    processed.unlink()
    report.update(status='completed', token_path=str(target),
                  preprocessed_h5_deleted=True, local_h5_deleted=True)
    pilot.save_json(work / 'report.json', report)
    return report


def cpu_stage(work, shot, name):
    cmd = ['srun', '--account=kolemen', '--partition=cpu', '--nodes=1',
           '--ntasks=1', '--cpus-per-task=8', '--mem=96G', '--time=1:30:00',
           f'--job-name=fdp-{name}-{shot}', sys.executable,
           str(HERE / 'validate_pipeline.py'), 'validate', '--run-dir', str(work),
           '--shots', str(shot), '--worker', name]
    with (work / f'{name}.log').open('w') as output:
        subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=7200, cwd=pilot.REPO)


def process_shot(shot, campaign, plan):
    work = campaign / 'work' / str(shot)
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / 'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        for folder in ('raw', 'processed'):
            (work / folder).mkdir()
        report = dict(shot=str(shot), source='FDP', status='planned',
                      archive=str(Path(plan['archive_dir']) / f'{shot}.h5'),
                      production_cache_modified=False, local_h5_deleted=False)
        pilot.save_json(report_path, report)
    raw = work / 'raw' / f'{shot}.h5'
    try:
        for path, expected in plan['source_sha256'].items():
            if pilot.digest(path) != expected:
                raise ValueError(f'Pipeline changed after planning: {path}')
        if report['status'] == 'completed':
            target = Path(report['token_path'])
            if pilot.digest(target) != report['code_sha256']:
                raise ValueError('Completed token file changed')
            return report
        if not report.get('fetch_complete'):
            if raw.exists():
                raise ValueError('Interrupted fetch retained; inspect before retry')
            log(f'{shot}: fetch')
            with (work / 'fetch.log').open('w') as output:
                subprocess.run(
                    ['nice', '-n', '19', 'bash', str(HERE / 'fetch_shot_fast.sh'),
                     str(shot), str(raw.parent)], stdout=output,
                    stderr=subprocess.STDOUT, check=True, timeout=7200,
                    cwd=pilot.REPO,
                )
            report['fetch_checks'] = pilot.validate_fetch_log(
                (work / 'fetch.log').read_text())
            report.update(fetch_complete=True, raw_bytes=raw.stat().st_size,
                          status='fetched')
            pilot.save_json(report_path, report)
        if not report.get('preprocess_complete'):
            log(f'{shot}: preprocess (CPU)')
            cpu_stage(work, shot, 'preprocess')
            report.update(preprocess_complete=True, status='preprocessed')
            pilot.save_json(report_path, report)
        if not report.get('archive_complete'):
            log(f'{shot}: overwrite raw archive and verify checksum')
            report['raw_sha256'] = archive_replace(raw, Path(report['archive']))
            report['archive_mtime_ns'] = Path(report['archive']).stat().st_mtime_ns
            report.update(archive_complete=True, status='archived')
            pilot.save_json(report_path, report)
        if raw.exists():
            if pilot.digest(raw) != report['raw_sha256']:
                raise ValueError('Local raw changed; preserving it')
            raw.unlink()  # Verified archive implements the requested raw move.
        if report['status'] != 'validated':
            log(f'{shot}: encode and validate tokens (CPU)')
            cpu_stage(work, shot, 'encode')
            report = json.loads(report_path.read_text())
        report = finish_shot(work, Path(plan['token_dir']), report, plan)
        log(f'{shot}: completed; processed H5 deleted')
        return report
    except Exception as exc:
        report = json.loads(report_path.read_text())
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        pilot.save_json(report_path, report)
        log(f'{shot}: FAILED: {exc}; H5 evidence retained')
        return report


def run_campaign(args):
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / 'plan.json').read_text())
    lock = (campaign / '.driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Refuse silently mixing pipeline revisions across a long-running campaign.
    for path, expected in plan['source_sha256'].items():
        if pilot.digest(path) != expected:
            raise ValueError(f'Pipeline changed after planning: {path}')
    if not os.access(plan['archive_dir'], os.W_OK):
        raise ValueError('TigerData is not writable on this host')
    (campaign / 'reports').mkdir(exist_ok=True)
    Path(plan['token_dir']).mkdir(parents=True, exist_ok=True)
    summary = dict(status='running', total=len(plan['shots']), completed=0,
                   concurrency=plan['concurrency'],
                   failed=0, in_flight=[], token_dir=plan['token_dir'])
    pending = []
    for shot in plan['shots']:
        receipt = campaign / 'reports' / f'{shot}.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['status'] == 'completed':
                if pilot.digest(Path(old['token_path'])) != old['code_sha256']:
                    raise ValueError(f'Completed tokens changed: {shot}')
                summary['completed'] += 1
                continue
            if old['status'] == 'failed':
                summary['failed'] += 1
                continue
        pending.append(shot)
    active, index = {}, 0
    with ThreadPoolExecutor(max_workers=plan['concurrency']) as pool:
        while active or index < len(pending):
            token_ok = True
            try:
                pilot.check_token()
            except ValueError:
                token_ok = False
            failures_ok = summary['failed'] < plan['max_failures']
            while (token_ok and failures_ok and index < len(pending)
                   and len(active) < plan['concurrency']):
                shot = pending[index]
                # Recheck the active dataset immediately before collection.
                if shot in active_shots():
                    raise ValueError(f'Shot newly entered active dataset: {shot}')
                active[pool.submit(process_shot, shot, campaign, plan)] = shot
                index += 1
            summary['in_flight'] = sorted(active.values())
            summary['remaining'] = len(pending) - index
            summary['status'] = ('failure_limit' if not failures_ok else
                                 'running' if token_ok else 'waiting_for_token')
            pilot.save_json(campaign / 'status.json', summary)
            if not active:
                if not failures_ok:
                    break
                if not token_ok:
                    log('Waiting for refreshed FDP token; checking every 60 s')
                    time.sleep(60)
                    continue
                break
            finished, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                shot = active.pop(future)
                report = future.result()
                pilot.save_json(campaign / 'reports' / f'{shot}.json', report)
                summary['completed' if report['status'] == 'completed'
                        else 'failed'] += 1
            if not failures_ok and not active:
                break
    summary.update(in_flight=[], status=(
        'failure_limit' if summary['failed'] >= plan['max_failures'] else
        'completed_with_failures' if summary['failed'] else 'completed'))
    pilot.save_json(campaign / 'status.json', summary)
    log(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--start', type=int, default=205000)
    parser.add_argument('--end', type=int)
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--token-dir', type=Path, default=(
        pilot.MODELS / 'ignite_fdp_205000_v4/frame_codes'))
    parser.add_argument('--validation-report', type=Path, default=(
        pilot.REPO / 'scratch/fdp_pilot_207801/report.json'))
    args = parser.parse_args()
    (make_plan if args.action == 'plan' else run_campaign)(args)


if __name__ == '__main__':
    main()
