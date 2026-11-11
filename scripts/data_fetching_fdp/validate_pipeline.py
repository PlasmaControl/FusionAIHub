#!/usr/bin/env python3
"""Bounded FDP pilot. Keep all local H5 evidence until explicit cleanup.

Uses the existing raw preprocessor and IGNITE's frozen-codec precompute path.
The production cache and split files are read only. This is intentionally not
a bulk submitter: validate one excluded shot before a campaign.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
MODELS = Path('/scratch/gpfs/EKOLEMEN/ps9551/ignite/models')
PRODUCTION = MODELS / 'ignite_prod_v4'
ARCHIVE = Path('/tigerdata/EKOLEMEN/d3d_fusion_data')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def excluded_shots(production=PRODUCTION):
    """Union all run splits, cached shots, and legacy preprocessing shot lists."""
    excluded = {p.stem for p in (production / 'frame_codes').glob('*.pt')}
    splits = sorted((production / 'runs').glob('*/split_shots.json'))
    if not splits or not (production / 'frame_codes/_codec_manifest.json').is_file():
        raise ValueError('Production splits/cache manifest missing; cannot exclude used shots')
    for path in splits:
        doc = json.loads(path.read_text())
        for key in ('train', 'val', 'test'):
            if key not in doc or not isinstance(doc[key], list):
                raise ValueError(f'Malformed {key} split in {path}')
        for value in doc.values():
            if isinstance(value, list):
                excluded.update(str(s) for s in value)
    for path in (REPO / 'src/tokamak_foundation_model/data/config/shot_list').glob('*'):
        excluded.update(re.findall(r'\b\d{6}\b', path.read_text()))
    return excluded, splits


def check_token():
    token = Path(os.environ.get('BEARER_TOKEN_FILE', str(Path.home() / '.fdp/token')))
    try:
        payload = token.read_text().strip().split('.')[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4)))
        if claims['exp'] <= time.time() + 3600:
            raise ValueError('FDP token is expired or has less than one hour remaining; refresh login')
    except (OSError, IndexError, KeyError) as exc:
        raise ValueError(f'Cannot read FDP token at {token}') from exc


def archive_raw(raw, archive):
    """Publish a verified copy without overwriting an existing archive shot."""
    expected = digest(raw)
    if archive.exists():
        if digest(archive) != expected:
            raise ValueError(f'Archive collision: {archive}; neither file was removed')
        return expected
    # TigerData does not support hard-link publication. Exclusive creation
    # prevents overwrites; only a verified readback authorizes local cleanup.
    # An interrupted transfer leaves its destination for diagnosis and will
    # fail the checksum gate on retry, never silently count as a complete shot.
    with raw.open('rb') as src, archive.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if digest(archive) != expected:
        raise ValueError(f'Archive checksum mismatch: {archive}')
    return expected


def validate_fetch_log(log):
    """Reject known silent-quality failures even when all fetch passes exit zero."""
    if re.search(r'Cannot open file: .*?/archives/index/json/', log):
        raise ValueError('Native PTDATA could not open its remote index')
    if re.search(r'PTDATA fixups:.*\bdefect:', log):
        raise ValueError('Fetch wrote unrepaired PTDATA defects')
    snapshots = re.findall(r'PTDATA index snapshot: (\S+)', log)
    if not snapshots:
        raise ValueError('Fetch did not record a pinned PTDATA index snapshot')
    return dict(index_snapshot=snapshots[-1], unrepaired_ptdata_defects=False)


def worker(args):
    """Separate heavy stages so raw arrays do not share memory with the codecs."""
    sys.path.insert(0, str(REPO / 'src'))
    sys.path.insert(0, str(REPO / 'scripts/data_preparation'))
    import h5py
    import numpy as np

    run = args.run_dir.resolve()
    shot = args.shots[0]
    raw = run / 'raw' / f'{shot}.h5'
    processed = run / 'processed' / f'{shot}_processed.h5'
    report = json.loads((run / 'report.json').read_text())
    if args.worker == 'preprocess':
        from omegaconf import OmegaConf
        from prepare_data import process_and_write_shot

        if report['source'] == 'FDP':
            checks = validate_fetch_log((run / 'fetch.log').read_text())
            save_json(run / 'fetch_checks.json', checks)
        with h5py.File(raw, 'r') as h5:
            if shot not in h5:
                raise ValueError('Raw H5 has no matching shot group')
            checked = 0
            for name, group in h5[shot].get('PTDATA', {}).items():
                checked += 1
                if 'units' in group and bytes(group['units'][()]).strip().lower() == b'vo':
                    raise ValueError(f'Uncalibrated PTDATA signal: {name}')
                if 'dim0' in group:
                    times = group['dim0'][:]
                    if (times.ndim == 1 and len(times) > 2
                            and not (np.diff(times.astype('f8')) > 0).all()):
                        raise ValueError(f'Non-monotonic PTDATA timebase: {name}')
            save_json(run / 'raw_checks.json', dict(ptdata_signals_checked=checked))
        cfg = OmegaConf.to_container(OmegaConf.load(
            REPO / 'src/tokamak_foundation_model/data/config/modalities/modalities.yaml'
        ), resolve=True)
        cfg.update(input_data_path=str(raw.parent), output_data_path=str(processed.parent))
        error = process_and_write_shot(int(shot), cfg)
        if error:
            raise RuntimeError(error)
        coverage = {}
        with h5py.File(processed, 'r') as h5:
            for name, group in h5.items():
                x, y = group['xdata'], group['ydata']
                if y.shape[1] != len(x):
                    raise ValueError(f'Invalid time/data alignment: {name}')
                times = x[:]
                if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
                    raise ValueError(f'Invalid processed timebase: {name}')
                # Scan bounded slabs; large diagnostics must not be loaded twice.
                finite = sum(int(np.isfinite(y[:, i:i+10000]).sum())
                             for i in range(0, y.shape[1], 10000))
                coverage[name] = dict(shape=list(y.shape), finite_samples=finite,
                                      total_samples=int(np.prod(y.shape)))
        if not any(v['finite_samples'] > 1 for v in coverage.values()):
            raise ValueError('Preprocessing produced no usable signals')
        save_json(run / 'coverage.json', coverage)
        return

    import torch
    from tokamak_foundation_model.ignite import train_dynamics as td

    baseline = json.loads((PRODUCTION / 'frame_codes/_codec_manifest.json').read_text())
    if args.worker == 'encode':
        for name in baseline['resolved']:
            if not (MODELS / 'ignite_codecs_v4' / name / 'codec_best.pt').is_file():
                raise ValueError(f'Missing frozen v4 codec: {name}')
        td.run_precompute(
            run / 'frame_codes', run / 'processed',
            codec_tmpl=str(MODELS / 'ignite_codecs_v4/{m}/codec_best.pt'),
            t0_start=baseline['t0_start'], shot_timeout_s=3600,
        )
    entry_path = run / 'frame_codes' / f'{shot}.pt'
    entry = torch.load(entry_path, map_location='cpu', weights_only=False)
    # Compare against an actual production entry, including the actuator width.
    reference = next((PRODUCTION / 'frame_codes').glob('[0-9]*.pt'))
    ref = torch.load(reference, map_location='cpu', weights_only=False)
    n = entry['n_frames']
    if n < 8 or set(entry['codes']) != set(ref['codes']):
        raise ValueError('Frame count or modality set differs from production contract')
    for name, codes in entry['codes'].items():
        if (codes.shape != (n, ref['codes'][name].shape[1])
                or codes.dtype != torch.int32
                or entry['vocabs'][name] != ref['vocabs'][name]
                or codes.min() < 0 or codes.max() >= entry['vocabs'][name]):
            raise ValueError(f'Invalid code sequence: {name}')
    act = entry['actuators']
    if (act.shape != (n, ref['actuators'].shape[1])
            or not torch.isfinite(act).all()):
        raise ValueError('Invalid actuator sequence')
    actual = json.loads((run / 'frame_codes/_codec_manifest.json').read_text())
    if actual['t0_start'] != baseline['t0_start'] or actual['frame_tokens'] != baseline['frame_tokens']:
        raise ValueError('Frame layout differs from production')
    # Independently regenerate boundary frames and exercise the frozen decoders.
    # These metrics describe codec input space, not physical-unit fidelity.
    samples = {}
    with torch.inference_mode():
        for name, path in actual['resolved'].items():
            family = next(m.family for m in td.FROZEN_MODALITIES if m.name == name)
            codec, cfg = td._load_codec(family, Path(path))
            dataset = td._single_shot_dataset(
                name, family, cfg, shot, processed.parent, baseline['t0_start']
            )
            samples[name] = []
            for index in (0, n - 1):
                x = td._window(dataset[index]).unsqueeze(0)
                if not torch.isfinite(x).all():
                    raise ValueError(f'Non-finite codec input: {name}, frame {index}')
                fresh = td.encode_flat(codec, x)[0].to(torch.int32)
                if not torch.equal(fresh, entry['codes'][name][index]):
                    raise ValueError(f'Code re-encoding mismatch: {name}, frame {index}')
                out = codec(x)
                target = out.get('x_std', x)
                recon = out['recon']
                if recon.shape != target.shape or not torch.isfinite(recon).all():
                    raise ValueError(f'Invalid codec reconstruction: {name}, frame {index}')
                samples[name].append(dict(
                    frame=index, input_std=float(target.std()),
                    mae=float((recon - target).abs().mean()),
                ))
    save_json(run / 'codec_samples.json', samples)
    coverage = json.loads((run / 'coverage.json').read_text())
    report.update(
        status='validated', n_frames=n, frame_tokens=actual['frame_tokens'],
        code_sha256=digest(entry_path), processed_sha256=digest(processed),
        missing_groups=[k for k, v in coverage.items() if not v['finite_samples']],
        fresh_fdp=not bool(args.raw_source),
        new_shot=not args.reference_shot,
        codec_sha256={name: digest(Path(path))
                      for name, path in actual['resolved'].items()},
        raw_checks=json.loads((run / 'raw_checks.json').read_text()),
    )
    if (run / 'fetch_checks.json').is_file():
        report['fetch_checks'] = json.loads((run / 'fetch_checks.json').read_text())
    save_json(run / 'report.json', report)


def cleanup(run):
    report = json.loads((run / 'report.json').read_text())
    if report['status'] != 'validated':
        raise ValueError('Cleanup requires a successful validation report')
    shot = report['shot']
    archive = Path(report['archive'])
    if digest(archive) != report['raw_sha256']:
        raise ValueError('Archive verification failed; keeping local H5 files')
    if digest(run / 'frame_codes' / f'{shot}.pt') != report['code_sha256']:
        raise ValueError('Code verification failed; keeping local H5 files')
    raw = run / 'raw' / f'{shot}.h5'
    processed = run / 'processed' / f'{shot}_processed.h5'
    if digest(processed) != report['processed_sha256']:
        raise ValueError('Processed H5 changed; keeping local H5 files')
    # Only remove this pilot's working files. The TigerData archive is retained.
    processed.unlink()
    raw.unlink()
    report['local_h5_deleted'] = True
    save_json(run / 'report.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'validate', 'cleanup'])
    p.add_argument('--shots', nargs='+')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--raw-source', type=Path,
                   help='Archived raw H5 for downstream-only validation; no FDP success claim')
    p.add_argument('--reference-shot', action='store_true',
                   help='Downstream regression only: permit an already-used archive shot')
    p.add_argument('--slurm', action='store_true',
                   help='Run heavy stages in CPU Slurm allocations; orchestrate on login node')
    p.add_argument('--worker', choices=['preprocess', 'encode'], help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    run = args.run_dir.resolve()
    if args.action == 'cleanup':
        cleanup(run)
        return
    if not args.shots or len(args.shots) != 1:
        p.error('Pilot must contain exactly one shot; bulk runs are not enabled')
    if any(not re.fullmatch(r'\d{6}', s) for s in args.shots):
        p.error('Shots must be six-digit integers')
    excluded, splits = excluded_shots()
    overlap = set(args.shots) & excluded
    if args.reference_shot and not args.raw_source:
        p.error('--reference-shot requires --raw-source; it cannot fetch or add used shots')
    if overlap and not args.reference_shot:
        p.error(f'Shots already used in existing splits/cache/shot lists: {sorted(overlap)}')
    shot = args.shots[0]
    plan = dict(shot=shot, run_dir=str(run), archive=str(ARCHIVE / f'{shot}.h5'),
                split_files=[str(s) for s in splits], excluded_count=len(excluded),
                source=str(args.raw_source) if args.raw_source else 'FDP',
                reference_shot=args.reference_shot,
                production_cache_modified=False, local_h5_deleted=False,
                preprocessing_config_sha256=digest(REPO / 'src/tokamak_foundation_model/data/config/modalities/modalities.yaml'),
                fetch_config_sha256=digest(Path(__file__).with_name('config_fdp.yaml')))
    if args.action == 'plan':
        print(json.dumps(plan, indent=2))
        return
    if not args.slurm:
        p.error('Use --slurm from the login node: TigerData is not mounted on compute nodes')
    if args.raw_source:
        if args.raw_source.resolve() != Path(plan['archive']).resolve():
            p.error('--raw-source must be the existing TigerData archive for this shot')
    else:
        check_token()
    if not ARCHIVE.is_dir():
        p.error(f'Raw archive is not mounted: {ARCHIVE}; run from the login host')
    # Refuse overwrites, including partial runs. Use a new directory to retry.
    run.mkdir(parents=True, exist_ok=False)
    (run / 'raw').mkdir()
    (run / 'processed').mkdir()
    plan['status'] = 'started'
    save_json(run / 'report.json', plan)
    raw = run / 'raw' / f'{shot}.h5'
    try:
        if args.raw_source:
            # TigerData is mounted on the login host, not on CPU compute nodes.
            # A shared-scratch working copy is needed by the preprocessing allocation.
            shutil.copyfile(args.raw_source, raw)
        else:
            with (run / 'fetch.log').open('w') as log:
                subprocess.run(['bash', str(Path(__file__).with_name('fetch_shot_fast.sh')),
                                shot, str(raw.parent)], stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=7200, cwd=REPO)
        def stage(name):
            with (run / f'{name}.log').open('w') as log:
                cmd = [sys.executable, str(Path(__file__).resolve()), 'validate',
                       '--run-dir', str(run), '--shots', shot, '--worker', name]
                if args.raw_source:
                    cmd += ['--raw-source', str(args.raw_source)]
                if args.reference_shot:
                    cmd += ['--reference-shot']
                cmd = ['srun', '--account=kolemen', '--partition=cpu', '--nodes=1',
                       '--ntasks=1', '--cpus-per-task=8', '--mem=96G',
                       '--time=1:30:00', *cmd]
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=7200, cwd=REPO)
        stage('preprocess')
        plan['raw_sha256'] = archive_raw(raw, Path(plan['archive']))
        plan['status'] = 'archived'
        save_json(run / 'report.json', plan)
        stage('encode')
        print((run / 'report.json').read_text())
    except Exception as exc:
        plan.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_json(run / 'report.json', plan)
        raise


if __name__ == '__main__':
    main()
