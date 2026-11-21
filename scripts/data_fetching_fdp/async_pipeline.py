#!/usr/bin/env python3
"""Shared ascending/descending workers, asynchronous archives, and disk budgets."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

import bulk_pipeline as bulk

TB = 10**12
GB = 10**9
FILE_LIMIT = 64 * 1024**3
RESERVATION = 160 * GB
ROOT = Path('/scratch/gpfs/EKOLEMEN')
# Training-set actuator statistics; new shots are normalised with them, never refit.
ACT_STATS = Path('/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_prod_v4/'
                 'act_global/act_global_stats.npz')


def allocated_bytes(path):
    """Use whole-filesystem allocation as a conservative project usage bound."""
    stat = os.statvfs(path)
    return (stat.f_blocks - stat.f_bfree) * stat.f_frsize


class QuotaProbe:
    """Read the fileset metric used by checkquota and live filesystem allocation."""

    def __init__(self):
        self.updated = 0
        self.used = 0

    def __call__(self, root):
        if time.monotonic() - self.updated > 20:
            metric = ('gpfs_quota_block_usage_bytes{filesetname="EKOLEMEN",'
                      'fs="stellarai.storage",quota_type="FILESET"}')

            def query(expression):
                url = ('http://cluster-stats:8480/api/v1/query?'
                       + urllib.parse.urlencode({'query': expression}))
                with urllib.request.urlopen(url, timeout=10) as response:
                    doc = json.load(response)
                if doc['status'] != 'success' or len(doc['data']['result']) != 1:
                    raise ValueError('EKOLEMEN quota metric unavailable')
                return float(doc['data']['result'][0]['value'][1])

            measured = query(metric)
            timestamp = query(f'timestamp({metric})')
            if time.time() - timestamp > 600:
                raise ValueError('EKOLEMEN quota accounting is more than 10 min old')
            self.used, self.updated = int(measured), time.monotonic()
        return max(self.used, allocated_bytes(root))


class DiskBudget:
    """One reservation ledger shared by both directions in this controller."""

    def __init__(self, root=ROOT, limit=40*TB, headroom=500*GB,
                 reservation=RESERVATION, probe=allocated_bytes):
        self.root = root
        self.limit = limit
        self.headroom = headroom
        self.reservation = reservation
        self.probe = probe
        self.reserved = {}
        self.used, self.error = None, None
        self.refresh()

    def refresh(self):
        try:
            self.used, self.error = self.probe(self.root), None
        except Exception as exc:
            self.error = str(exc)

    def admit(self, key):
        self.refresh()
        if self.error:
            return False
        needed = self.used + sum(self.reserved.values())
        if needed + self.reservation + self.headroom > self.limit:
            return False
        self.reserved[key] = self.reservation
        return True

    def retain(self, key, work):
        """After compute, retain a conservative allowance for queued raw files."""
        self.refresh()
        files = list((work / 'raw').glob('*.h5'))
        files += list((work / 'processed').glob('*.h5'))
        size = sum(p.stat().st_size for p in files if p.exists())
        if size:
            self.reserved[key] = size + GB
        else:
            self.reserved.pop(key, None)

    def snapshot(self):
        self.refresh()
        return dict(path=str(self.root), used_bytes=self.used,
                    limit_bytes=self.limit, headroom_bytes=self.headroom,
                    reserved_bytes=sum(self.reserved.values()),
                    accounting='EKOLEMEN quota metric + live allocation bound',
                    quota_error=self.error,
                    admission_allowed=(not self.error and
                                       self.used + sum(self.reserved.values())
                                       + self.reservation + self.headroom
                                       <= self.limit))


# Subdirectory of the archive for new fetches of shots whose old file belongs to
# another user.
REDIRECT_DIR = 'fdp_refetch'

# Archive copies over the login node's NFS mount of TigerData reach ~7 MB/s per
# stream and ~60 MB/s in all (measured 2026-11-15), about 33 min per 16 GB raw file.
# `--archive-mode globus` moves them DTN to DTN instead (StellarAI scratch DTN ->
# TigerData collection); it needs `globus login` and the collection's data_access
# consent once.
ARCHIVE_MODE = 'nfs'
GLOBUS_CLI = Path.home() / '.pixi/bin/globus'
GLOBUS_SRC = ('0b091761-42ba-4e22-95cb-fe7afcf3f8e0', Path('/scratch/gpfs'))
GLOBUS_DST = ('52d95801-fced-4b1e-951b-63836783b175',
              Path('/tigerdata/EKOLEMEN/d3d_fusion_data'))
GLOBUS_POLL_S, GLOBUS_MAX_FAULTS = 30, 20
# Globus mode: every queued job is handed to Globus at once, GLOBUS_BATCH files per
# task, and the archive workers only finish jobs whose task has ended (size check,
# rename, raw unlink). Waiting on a running copy in a worker held all workers on the
# oldest copies while finished ones queued behind them (~85 MB/s instead of ~180).
GLOBUS_BATCH, GLOBUS_SCAN_S = 8, 60
# At most this many of our Globus tasks run at once (each moves its files over several
# parallel streams). Without a cap, every file became its own task as soon as it was
# fetched: up to 101 tasks at once on 2026-11-17, and Research Computing asked us to
# pause and resume with fewer parallel streams.
GLOBUS_MAX_ACTIVE = 2
GLOBUS_DONE = ('SUCCEEDED', 'FAILED')


def globus(*args):
    out = subprocess.run([str(GLOBUS_CLI), *args, '-F', 'json'], capture_output=True,
                         text=True, timeout=300)
    if out.returncode:
        raise RuntimeError(f'globus {" ".join(args[:2])}: {out.stderr.strip()[-300:]}')
    return json.loads(out.stdout)


def globus_copy(job, job_path, raw, tmp):
    """Transfer ``raw`` to ``tmp`` with Globus and wait for it.

    The task id is kept in the job, so a restarted driver waits for the same task
    instead of starting another. A task that keeps faulting (permissions, an
    expired consent) is cancelled and the copy fails like an NFS one.
    """
    task = job.get('globus_task')
    if not task:
        src = f'{GLOBUS_SRC[0]}:/{raw.resolve().relative_to(GLOBUS_SRC[1])}'
        dst = f'{GLOBUS_DST[0]}:/{tmp.relative_to(GLOBUS_DST[1])}'
        # No --verify-checksum: it re-reads the whole file at both ends, and TigerData
        # reads at ~23 MB/s, longer than the transfer. Globus checks the data in
        # transit; the archive is checked by size, and the raw file can be fetched
        # again from FDP.
        task = globus('transfer', src, dst, '--notify', 'off',
                      '--label', f'archive {job["shot"]}')['task_id']
        job.update(globus_task=task)
        bulk.pilot.save_json(job_path, job)
    while True:
        info = globus('task', 'show', task)
        if info['status'] == 'SUCCEEDED':
            job.pop('globus_task')
            return
        if info['status'] == 'FAILED' or info.get('faults', 0) >= GLOBUS_MAX_FAULTS:
            if info['status'] != 'FAILED':
                globus('task', 'cancel', task)
            job.pop('globus_task')
            raise RuntimeError(f'Globus task {task} {info["status"]}: '
                               f'{info.get("nice_status")}, {info.get("faults")} faults')
        time.sleep(GLOBUS_POLL_S)


def archive_paths(job):
    """The archive target and its temporary file. Another user's archive file cannot
    be replaced (the rename fails with EIO): the new fetch then goes beside it in
    REDIRECT_DIR, and ``job`` records that."""
    target = Path(job['archive'])
    if (job.get('status') != 'verified' and target.exists()
            and target.stat().st_uid != os.getuid()):
        target = target.parent / REDIRECT_DIR / target.name
        target.parent.mkdir(exist_ok=True)
        job.update(archive=str(target), redirected_from=job['archive'])
    return target, target.with_name(f'.{target.name}.async-{job["key"]}.partial')


def globus_submit(job_paths):
    """Start one Globus task for up to GLOBUS_BATCH queued jobs and store its id in
    each job (``archive_job`` then waits for that task instead of starting one)."""
    lines, jobs = [], []
    for path in job_paths:
        job = json.loads(path.read_text())
        raw = Path(job['raw'])
        if job.get('globus_task') or job.get('status') == 'verified' or not raw.exists():
            continue
        _, tmp = archive_paths(job)
        lines.append(f'/{raw.resolve().relative_to(GLOBUS_SRC[1])} '
                     f'/{tmp.relative_to(GLOBUS_DST[1])}')
        jobs.append((path, job))
    if not jobs:
        return None
    out = subprocess.run(
        [str(GLOBUS_CLI), 'transfer', GLOBUS_SRC[0], GLOBUS_DST[0], '--batch', '-',
         '--notify', 'off', '--label', f'archive {jobs[0][1]["shot"]}+{len(jobs) - 1}',
         '-F', 'json'],
        input='\n'.join(lines) + '\n', capture_output=True, text=True, timeout=300)
    if out.returncode:
        raise RuntimeError(f'globus transfer: {out.stderr.strip()[-300:]}')
    task = json.loads(out.stdout)['task_id']
    for path, job in jobs:
        job['globus_task'] = task
        bulk.pilot.save_json(path, job)
    return task, [job['shot'] for _, job in jobs]


def globus_status(tasks):
    """Status of each task id (missing from the answer: not known)."""
    out = {}
    tasks = sorted(tasks)
    for i in range(0, len(tasks), 50):
        args = ['task', 'list', '--limit', '1000']
        for t in tasks[i:i + 50]:
            args += ['--filter-task-id', t]
        out.update({t['task_id']: t['status'] for t in globus(*args)['DATA']})
    return out


def archive_job(job_path):
    """Copy/hash once, verify the temporary file once, then atomically replace."""
    job = json.loads(job_path.read_text())
    raw = Path(job['raw'])
    target, tmp = archive_paths(job)
    if job.get('status') == 'verified':
        if raw.exists():
            if bulk.pilot.digest(raw) != job['sha256']:
                raise ValueError('Raw file changed after successful archive')
            raw.unlink()
        return job
    try:
        job.update(status='copying', started=time.time(),
                   attempts=job.get('attempts', 0) + 1)
        bulk.pilot.save_json(job_path, job)
        source_stat = raw.stat()
        if ARCHIVE_MODE == 'globus':
            globus_copy(job, job_path, raw, tmp)
            if tmp.stat().st_size != source_stat.st_size:
                raise ValueError('TigerData temporary-file size mismatch')
            # The receipt's checksum, of the local file (GPFS, fast).
            expected = bulk.pilot.digest(raw)
            job.update(status='verifying', sha256=expected)
            bulk.pilot.save_json(job_path, job)
        else:
            digest = hashlib.sha256()
            with raw.open('rb') as src, tmp.open('wb') as dst:
                for data in iter(lambda: src.read(8*1024**2), b''):
                    digest.update(data)
                    dst.write(data)
                dst.flush()
                os.fsync(dst.fileno())
            expected = digest.hexdigest()
            job.update(status='verifying', sha256=expected)
            bulk.pilot.save_json(job_path, job)
            if bulk.pilot.digest(tmp) != expected:
                raise ValueError('TigerData temporary-file checksum mismatch')
        verified = tmp.stat()
        os.replace(tmp, target)
        published = target.stat()
        # Rename preserves the already verified bytes. Read again only if the
        # filesystem changed metadata, rather than rereading every large file.
        if ((published.st_size, published.st_mtime_ns)
                != (verified.st_size, verified.st_mtime_ns)
                and bulk.pilot.digest(target) != expected):
            raise ValueError('TigerData publication checksum mismatch')
        current = raw.stat()
        if ((current.st_size, current.st_mtime_ns)
                != (source_stat.st_size, source_stat.st_mtime_ns)):
            raise ValueError('Local raw file changed during archival')
        job.update(status='verified', bytes=published.st_size,
                   archive_mtime_ns=published.st_mtime_ns, finished=time.time())
        bulk.pilot.save_json(job_path, job)  # Durable receipt precedes unlink.
        raw.unlink()
        bulk.log(f'{job["shot"]}: raw archived asynchronously; local raw deleted')
        return job
    except Exception as exc:
        job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        bulk.pilot.save_json(job_path, job)
        bulk.log(f'{job["shot"]}: archive FAILED; local raw retained: {exc}')
        return job


def enqueue_archive(work, direction, shot, plan, notifications):
    path = work / 'archive_job.json'
    if not path.exists():
        raw = work / 'raw' / f'{shot}.h5'
        bulk.pilot.save_json(path, dict(
            key=f'{direction}-{shot}', shot=str(shot), status='queued',
            raw=str(raw), archive=str(Path(plan['archive_dir']) / f'{shot}.h5'),
            bytes=raw.stat().st_size, queued=time.time(),
        ))
    notifications.put((direction, shot, path))


QOS_RETRIES, QOS_WAIT_S = 60, 60


def cpu_stage(work, shot, stage):
    cmd = ['srun', '--account=kolemen', '--partition=cpu', '--nodes=1',
           '--ntasks=1', '--cpus-per-task=8', '--mem=96G', '--time=1:30:00',
           f'--job-name=fdp-{stage}-{shot}', 'prlimit',
           f'--fsize={FILE_LIMIT}:{FILE_LIMIT}', sys.executable,
           str(bulk.HERE / 'validate_pipeline.py'), 'validate', '--run-dir',
           str(work), '--shots', str(shot), '--worker', stage]
    log = work / f'{stage}.log'
    # The per-user job limit is shared with other work (training, backfills); a
    # refused submission is retried for up to an hour instead of failing the shot.
    for _ in range(QOS_RETRIES):
        with log.open('w') as out:
            result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT,
                                    timeout=7200, cwd=bulk.pilot.REPO)
        if result.returncode == 0:
            return
        if 'QOSMaxSubmitJobPerUserLimit' not in log.read_text(errors='replace'):
            break
        time.sleep(QOS_WAIT_S)
    result.check_returncode()


def publish_act_global(work, shot, models_dir, unchecked=()):
    """Write ``<models_dir>/act_global/<shot>.pt`` from the processed H5.

    Frame-mean physical actuators (``train_dynamics._raw_actuators``) and their
    normalisation with the frozen training-set statistics, as in
    ``ignite_prod_v4/act_global``. The raw values must reproduce the per-shot
    z-scored actuators of the shot's token entry, except in the actuator groups
    named in ``unchecked`` (``train_dynamics._ACT_SPEC`` keys), for shots whose
    re-fetch is known to differ there from the data the tokens were built from.
    """
    import numpy as np
    import torch
    from tokamak_foundation_model.ignite import train_dynamics as td

    shot = str(shot)
    stats = np.load(ACT_STATS, allow_pickle=True)
    mu, sd = stats['mu'], stats['sd']
    present = stats['present'].astype(bool)
    entry = torch.load(work / 'frame_codes' / f'{shot}.pt', map_location='cpu',
                       weights_only=False)
    manifest = json.loads((work / 'frame_codes/_codec_manifest.json').read_text())
    n = entry['actuators'].shape[0]
    raw = td._raw_actuators(shot, n, work / 'processed', manifest['t0_start'])
    z = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)
    group = np.concatenate([[k] * c for k, c in td._ACT_SPEC])
    checked = ~np.isin(group, list(unchecked))
    err = float(np.abs(z - entry['actuators'].double().numpy())[:, checked].max())
    if err >= 1e-2:
        raise ValueError(f'Raw actuators do not reproduce the token entry ({err})')
    normed = np.where(present, (raw - mu) / np.where(present, sd, 1.0), 0.0)
    target = models_dir / 'act_global' / f'{shot}.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f'.pt.tmp.{os.getpid()}.{threading.get_ident()}')
    torch.save({'actuators_global': torch.tensor(normed, dtype=torch.float16),
                'actuators_raw': torch.tensor(raw, dtype=torch.float32)}, tmp)
    os.replace(tmp, target)
    return target


def compute_shot(direction, shot, campaign, plan, notifications):
    """Produce tokens independently of the archive queue; keep raw until copied."""
    work = campaign / 'work' / str(shot)
    work.mkdir(parents=True, exist_ok=True)
    record = work / 'report.json'
    if record.exists():
        report = json.loads(record.read_text())
    else:
        for name in ('raw', 'processed'):
            (work / name).mkdir(exist_ok=True)
        report = dict(shot=str(shot), source='FDP', status='planned',
                      archive=str(Path(plan['archive_dir']) / f'{shot}.h5'),
                      production_cache_modified=False, local_h5_deleted=False)
        bulk.pilot.save_json(record, report)
    raw = work / 'raw' / f'{shot}.h5'
    try:
        if report['status'] in ('tokens_ready', 'completed'):
            return report
        for path, expected in plan['source_sha256'].items():
            if bulk.pilot.digest(path) != expected:
                raise ValueError(f'Pipeline changed after planning: {path}')
        if not report.get('fetch_complete'):
            if raw.exists():
                raise ValueError('Interrupted fetch retained for review')
            bulk.log(f'{shot}: fetch ({direction})')
            with (work / 'fetch.log').open('w') as out:
                subprocess.run([
                    'prlimit', f'--fsize={FILE_LIMIT}:{FILE_LIMIT}',
                    'nice', '-n', '19', 'bash',
                    str(bulk.HERE / 'fetch_shot_fast.sh'), str(shot),
                    str(raw.parent),
                ], stdout=out, stderr=subprocess.STDOUT, check=True,
                   timeout=7200, cwd=bulk.pilot.REPO)
            report['fetch_checks'] = bulk.pilot.validate_fetch_log(
                (work / 'fetch.log').read_text())
            report.update(fetch_complete=True, raw_bytes=raw.stat().st_size,
                          status='fetched')
            bulk.pilot.save_json(record, report)
        if not report.get('preprocess_complete'):
            bulk.log(f'{shot}: preprocess (CPU)')
            cpu_stage(work, shot, 'preprocess')
            report.update(preprocess_complete=True, status='preprocessed')
            bulk.pilot.save_json(record, report)
        if not report.get('archive_complete'):
            enqueue_archive(work, direction, shot, plan, notifications)
        if report['status'] != 'validated':
            bulk.log(f'{shot}: encode while raw archival runs independently')
            cpu_stage(work, shot, 'encode')
            report = json.loads(record.read_text())
        processed = work / 'processed' / f'{shot}_processed.h5'
        if bulk.pilot.digest(processed) != report['processed_sha256']:
            raise ValueError('Processed H5 checksum changed; keeping it')
        target = bulk.publish_tokens(work, Path(plan['token_dir']), report, plan)
        if bulk.pilot.digest(target) != report['code_sha256']:
            raise ValueError('Published tokens changed; keeping processed H5')
        act = publish_act_global(work, shot, Path(plan['token_dir']).parent)
        report['act_global_path'] = str(act)
        processed.unlink()
        report.update(status='tokens_ready', token_path=str(target),
                      preprocessed_h5_deleted=True,
                      archive_pending=not report.get('archive_complete', False))
        bulk.pilot.save_json(record, report)
        bulk.log(f'{shot}: tokens published; processed H5 deleted')
        return report
    except Exception as exc:
        report = json.loads(record.read_text())
        freed = drop_failed_h5(work)
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      failed_h5_deleted_bytes=freed)
        bulk.pilot.save_json(record, report)
        bulk.log(f'{shot}: compute FAILED; logs kept, {freed / GB:.1f} GB of H5 '
                 f'deleted: {exc}')
        return report


def drop_failed_h5(work):
    """Delete a failed shot's H5 files, keeping its logs and report.

    A failed shot is refetched from FDP when retried, so its H5s only fill the disk.
    The raw file stays while an archive job holds it; that job deletes it once the
    copy is verified.
    """
    files = list((work / 'processed').glob('*.h5*'))
    if not (work / 'archive_job.json').exists():
        files += list((work / 'raw').glob('*.h5*'))
    freed = 0
    for path in files:
        try:
            size = path.stat().st_size
            path.unlink()
            freed += size
        except FileNotFoundError:
            pass
    return freed


def finalize(campaign, shot):
    """Join independently durable token and archive receipts."""
    work = campaign / 'work' / str(shot)
    path = work / 'report.json'
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report['status'] == 'completed':
        bulk.pilot.save_json(campaign / 'reports' / f'{shot}.json', report)
        return report
    if report['status'] == 'failed':
        bulk.pilot.save_json(campaign / 'reports' / f'{shot}.json', report)
        return report
    if report['status'] != 'tokens_ready':
        return None
    job_path = work / 'archive_job.json'
    job = json.loads(job_path.read_text()) if job_path.exists() else None
    if job and job['status'] == 'verified':
        if Path(job['raw']).exists():
            return report
        report.update(archive_complete=True, raw_sha256=job['sha256'],
                      archive_mtime_ns=job['archive_mtime_ns'],
                      archive=job['archive'])
    if report.get('archive_complete'):
        report.update(status='completed', archive_pending=False,
                      local_h5_deleted=True)
        bulk.pilot.save_json(path, report)
        bulk.pilot.save_json(campaign / 'reports' / f'{shot}.json', report)
        bulk.log(f'{shot}: completed (tokens + verified archive)')
    return report


RECENT_WINDOW, RECENT_FAILURES = 50, 40
FINAL = ('completed', 'tokens_ready', 'failed')


def recent_outcomes(campaign, records):
    """Failed/not of the last RECENT_WINDOW finished shots, oldest first.

    A campaign stops on a run of failures, not on their total: a few percent of
    shots fail for good (no data, FDP client crashes), so a cumulative limit
    stops a long campaign for nothing, while a systematic bug fails nearly all.
    """
    done = [(path.stat().st_mtime, records[shot]['status'] == 'failed')
            for shot in records if records[shot].get('status') in FINAL
            for path in [campaign / 'work' / str(shot) / 'report.json']
            if path.exists()]
    return deque((failed for _, failed in sorted(done)[-RECENT_WINDOW:]),
                 maxlen=RECENT_WINDOW)


def failing(outcomes):
    """True when a run of failures should stop the campaign."""
    return sum(outcomes) >= RECENT_FAILURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ascending', type=Path, required=True)
    parser.add_argument('--descending', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--per-direction', type=int, default=8)
    parser.add_argument('--archive-workers', type=int, default=4)
    parser.add_argument('--archive-mode', choices=('nfs', 'globus'), default='nfs',
                        help='copy raw files to TigerData over NFS or with Globus')
    args = parser.parse_args()
    global ARCHIVE_MODE
    ARCHIVE_MODE = args.archive_mode
    control = args.control.resolve()
    control.mkdir(parents=True, exist_ok=True)
    global_lock = (control / '.controller.lock').open('a')
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bulk.pilot.save_json(control / 'deployment.json', dict(
        started=time.time(), limit_bytes=40*TB, headroom_bytes=500*GB,
        reservation_bytes=RESERVATION, file_limit_bytes=FILE_LIMIT,
        source_sha256=bulk.pilot.digest(Path(__file__).resolve()),
        per_direction=args.per_direction, archive_workers=args.archive_workers,
        archive_mode=args.archive_mode,
    ))
    roots = dict(ascending=args.ascending.resolve(),
                 descending=args.descending.resolve())
    plans, locks = {}, []
    for direction, campaign in roots.items():
        lock = (campaign / '.driver.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
        plans[direction] = json.loads((campaign / 'plan.json').read_text())
        (campaign / 'reports').mkdir(exist_ok=True)
        Path(plans[direction]['token_dir']).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    budget = DiskBudget(probe=QuotaProbe())
    notifications = queue.Queue()
    pending, records, recent, archive_pending = {}, {}, {}, deque()
    for direction, campaign in roots.items():
        known = {}
        for path in (campaign / 'work').glob('*/report.json'):
            report = json.loads(path.read_text())
            shot = int(report['shot'])
            known[shot] = report
            finalized = finalize(campaign, shot)
            if finalized:
                known[shot] = finalized
            job = path.parent / 'archive_job.json'
            if job.exists():
                data = json.loads(job.read_text())
                if data['status'] != 'verified' or Path(data['raw']).exists():
                    archive_pending.append((direction, shot, job))
                    budget.retain((direction, shot), path.parent)
        records[direction] = known
        pending[direction] = deque(s for s in plans[direction]['shots']
                                   if known.get(s, {}).get('status') not in FINAL)
        recent[direction] = recent_outcomes(campaign, known)
    compute, transfers, archive_seen, archive_errors = {}, {}, set(), {}
    busy = {d: 0 for d in roots}
    task_status, next_scan = {}, 0.0
    with ThreadPoolExecutor(max_workers=2*args.per_direction) as workers, \
            ThreadPoolExecutor(max_workers=args.archive_workers) as archivers:
        while (any(pending.values()) or compute or transfers or archive_pending
               or not notifications.empty()):
            while not notifications.empty():
                archive_pending.append(notifications.get_nowait())
            if ARCHIVE_MODE == 'globus' and time.time() >= next_scan:
                next_scan = time.time() + GLOBUS_SCAN_S
                waiting, tasks = [], set()
                for direction, shot, job in archive_pending:
                    if (direction, shot) in archive_seen:
                        continue
                    data = json.loads(job.read_text())
                    if data.get('globus_task'):
                        tasks.add(data['globus_task'])
                    elif (data.get('status') != 'verified'
                          and data.get('retry_after', 0) <= time.time()):
                        waiting.append(job)
                try:
                    task_status = globus_status(tasks) if tasks else {}
                except Exception as exc:
                    bulk.log(f'globus status failed: {exc}')
                # A task whose status is unknown counts as running.
                active = sum(task_status.get(t) not in GLOBUS_DONE for t in tasks)
                for i in range(0, len(waiting), GLOBUS_BATCH):
                    if active >= GLOBUS_MAX_ACTIVE:
                        break
                    try:
                        started = globus_submit(waiting[i:i + GLOBUS_BATCH])
                    except Exception as exc:  # retried at the next scan
                        bulk.log(f'globus submit failed: {exc}')
                        break
                    if started:
                        tasks.add(started[0])
                        active += 1
                        bulk.log(f'globus task {started[0]}: {", ".join(started[1])}')
            for _ in range(len(archive_pending)):
                if len(transfers) >= args.archive_workers:
                    break
                direction, shot, job = archive_pending.popleft()
                key = (direction, shot)
                if key in archive_seen:
                    continue
                data = json.loads(job.read_text())
                if data.get('retry_after', 0) > time.time():
                    archive_pending.append((direction, shot, job))
                    continue
                if (ARCHIVE_MODE == 'globus' and data.get('status') != 'verified'
                        and (not data.get('globus_task') or task_status.get(
                            data['globus_task'], 'ACTIVE') not in GLOBUS_DONE)):
                    # Not yet handed to Globus, or its copy is still running.
                    archive_pending.append((direction, shot, job))
                    continue
                archive_seen.add(key)
                transfers[archivers.submit(archive_job, job)] = key
            for future in list(compute):
                if future.done():
                    direction, shot = compute.pop(future)
                    report = future.result()
                    busy[direction] -= 1
                    records[direction][shot] = report
                    final = finalize(roots[direction], shot)
                    if final:
                        records[direction][shot] = final
                    if records[direction][shot].get('status') in FINAL:
                        recent[direction].append(
                            records[direction][shot]['status'] == 'failed')
                    budget.retain((direction, shot),
                                  roots[direction] / 'work' / str(shot))
            for future in list(transfers):
                if future.done():
                    direction, shot = transfers.pop(future)
                    job = future.result()
                    key = (direction, shot)
                    if job['status'] == 'failed':
                        archive_errors[key] = job['error']
                        if job.get('attempts', 0) < 3:
                            path = roots[direction] / 'work' / str(shot) / 'archive_job.json'
                            job['retry_after'] = time.time() + 30 * job['attempts']
                            bulk.pilot.save_json(path, job)
                            archive_seen.discard(key)
                            archive_pending.append((direction, shot, path))
                    else:
                        archive_errors.pop(key, None)
                    if (direction, shot) not in compute.values():
                        final = finalize(roots[direction], shot)
                        if final:
                            records[direction][shot] = final
                        budget.retain((direction, shot),
                                      roots[direction] / 'work' / str(shot))
            token_ok = True
            try:
                bulk.pilot.check_token()
            except ValueError:
                token_ok = False
            admitted = True
            while (admitted and token_ok and not stop.is_set()
                   and len(archive_errors) < 3):
                admitted = False
                for direction in roots:
                    if (pending[direction] and busy[direction] < args.per_direction
                            and not failing(recent[direction])):
                        shot = pending[direction][0]
                        if not budget.admit((direction, shot)):
                            continue
                        pending[direction].popleft()
                        future = workers.submit(
                            compute_shot, direction, shot, roots[direction],
                            plans[direction], notifications)
                        compute[future] = (direction, shot)
                        busy[direction] += 1
                        admitted = True
            disk = budget.snapshot()
            for direction, campaign in roots.items():
                recs = records[direction].values()
                progress = dict(
                    status=('draining' if stop.is_set() else
                            'failure_limit' if failing(recent[direction]) else
                            'waiting_for_archive_recovery' if len(archive_errors) >= 3 else
                            'waiting_for_quota' if disk['quota_error'] else
                            'waiting_for_token' if not token_ok else
                            'running' if disk['admission_allowed'] else
                            'waiting_for_space'),
                    direction=direction, concurrency=args.per_direction,
                    total=len(plans[direction]['shots']),
                    completed=sum(r['status'] == 'completed' for r in recs),
                    tokens_ready=sum(r['status'] in ('completed', 'tokens_ready')
                                     for r in recs),
                    failed=sum(r['status'] == 'failed' for r in recs),
                    recent_failed=f'{sum(recent[direction])}/{len(recent[direction])}',
                    in_flight=[s for d, s in compute.values() if d == direction],
                    remaining=len(pending[direction]),
                    token_dir=plans[direction]['token_dir'],
                    asynchronous_archive=True, disk_budget=disk,
                )
                bulk.pilot.save_json(campaign / 'status.json', progress)
            bulk.pilot.save_json(control / 'status.json', dict(
                status='draining' if stop.is_set() else 'running',
                compute_in_flight=len(compute),
                archive_in_flight=len(transfers),
                archive_queued=len(archive_pending), disk_budget=disk,
                archive_errors={f'{d}:{s}': error for (d, s), error
                                in archive_errors.items()},
            ))
            # On SIGTERM/SIGINT stop as soon as no fetch/encode is running. Queued
            # archive copies are durable (archive_job.json, raw kept until verified)
            # and the next start resumes them, so a restart must not wait for the
            # whole archive queue (that blocked fetching for ~9 h on 2026-11-11).
            # Copies already running finish when the executor shuts down.
            if stop.is_set() and not compute:
                break
            if (not compute and not transfers and not archive_pending
                    and notifications.empty()):
                if stop.is_set():
                    break
                if not any(pending.values()):
                    break
                if all(not pending[d] or failing(recent[d]) for d in roots):
                    break
            stop.wait(2) if not stop.is_set() else time.sleep(2)


if __name__ == '__main__':
    main()
