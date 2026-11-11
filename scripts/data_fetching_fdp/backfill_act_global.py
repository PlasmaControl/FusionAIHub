#!/usr/bin/env python3
"""Backfill token caches whose processed H5s are gone, from the archived raw H5.

Default mode: build ``act_global`` entries. Re-preprocesses only the actuator
groups and writes ``<models_dir>/act_global/<shot>.pt`` with
``async_pipeline.publish_act_global``, which checks the result against the shot's
token entry. Shots that already have an entry are skipped. Run on a compute node::

    srun --account=kolemen --partition=cpu --mem=96G --cpus-per-task=16 \\
        pixi run --frozen -e stellarai python \\
        scripts/data_fetching_fdp/backfill_act_global.py \\
        /scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_fdp_205000_v4

``--co2-tend`` mode: plasma end times, and CO2 codes for shots whose CO2 is flat.
Until 2026-11-08 the fetch replaced the fast CO2 PTDATA channels (``DEN*UF``) with
an all-zero TDI read, so every shot without the MDSplus BCI path has flat CO2
codes. Per shot without ``<models_dir>/plasma_end/<shot>.json``:

1. Login node (FDP token, /tigerdata): the plasma end from the plasma current
   (:func:`ip_end`). If CO2 is flat, re-fetch the four ``DEN*UF`` pointnames and
   copy the raw CO2 signals into a small file.
2. CPU node (``srun``; compute nodes cannot see /tigerdata): preprocess CO2 and
   re-encode it into the token entry.

Run it on a login node; ``--merge`` then writes ``<models_dir>/tend_ip.json``::

    setsid nohup pixi run --frozen -e stellarai python \\
        scripts/data_fetching_fdp/backfill_act_global.py --co2-tend --workers 12 \\
        <models_dir> [<models_dir> ...] > backfill_co2_tend.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import async_pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/data_preparation'))
sys.path.insert(0, str(REPO / 'src'))
ARCHIVE = Path('/tigerdata/EKOLEMEN/d3d_fusion_data')
ACT_GROUPS = ('ech_power', 'pinj', 'beam_voltage', 'tinj', 'gas_flow', 'gas_raw',
              'rmp', 'i_coil')
IP_POINTNAMES = (('IP', 1.0), ('IPTIPP', 1.0), ('IPSIP', 1e6))  # name, to amperes
CO2_PTDATA = ('DENR0UF', 'DENV1UF', 'DENV2UF', 'DENV3UF')
FDP_ENV = Path('/scratch/gpfs/EKOLEMEN/ps9551/fdp-demo/.pixi/envs/default')
CO2_CODEC = Path('/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_codecs_v4/co2/'
                 'codec_best.pt')


def find_raw(shot, raw_dirs=()):
    """The raw H5 of a shot in FDP layout (``/<shot>/<tree>/...``) this user can read.

    A shot still waiting in the archive queue has its raw only in the campaign's
    work dir, and the archive may then still hold an older file in another layout.
    """
    import h5py

    # Archives owned by another user are unreadable; the campaign rewrites those
    # shots to fdp_refetch/ (async_pipeline.archive_job).
    candidates = [Path(d) / f'{shot}.h5' for d in raw_dirs]
    candidates += sorted(REPO.glob(f'scratch/fdp_bulk_*/work/{shot}/raw/{shot}.h5'))
    candidates += [ARCHIVE / 'fdp_refetch' / f'{shot}.h5', ARCHIVE / f'{shot}.h5']
    for path in candidates:
        if not os.access(path, os.R_OK):
            continue
        try:
            with h5py.File(path, 'r') as f:
                if shot in f:
                    return path
        except OSError:
            continue
    raise FileNotFoundError(f'no readable FDP-layout raw H5 for {shot}')


def preprocess(shot, work, groups):
    """Run ``prepare_data`` on ``work/raw/<shot>.h5`` for ``groups`` only."""
    from omegaconf import OmegaConf
    from prepare_data import process_and_write_shot

    cfg = OmegaConf.to_container(OmegaConf.load(
        REPO / 'src/tokamak_foundation_model/data/config/modalities/modalities.yaml'
    ), resolve=True)
    cfg['signals'] = {k: v for k, v in cfg['signals'].items() if k in groups}
    cfg.update(input_data_path=str(work / 'raw'),
               output_data_path=str(work / 'processed'), num_workers=1)
    error = process_and_write_shot(int(shot), cfg)
    if error:
        raise RuntimeError(error)


def build(shot, models_dir, scratch, raw_dirs=(), unchecked=()):
    """Preprocess one shot's actuators into ``scratch`` and publish its entry."""
    work = scratch / shot
    shutil.rmtree(work, ignore_errors=True)
    (work / 'raw').mkdir(parents=True)
    (work / 'processed').mkdir()
    (work / 'frame_codes').mkdir()
    raw = find_raw(shot, raw_dirs)
    # Absolute, since a relative target would resolve against the link's own dir.
    (work / 'raw' / f'{shot}.h5').symlink_to(raw.resolve())
    tokens = models_dir / 'frame_codes'
    shutil.copyfile(tokens / f'{shot}.pt', work / 'frame_codes' / f'{shot}.pt')
    shutil.copyfile(tokens / '_codec_manifest.json',
                    work / 'frame_codes/_codec_manifest.json')
    preprocess(shot, work, ACT_GROUPS)
    target = async_pipeline.publish_act_global(work, shot, models_dir, unchecked)
    shutil.rmtree(work)
    return target


def co2_flat(token_path):
    """True if a token entry's CO2 codes never change, i.e. CO2 was not recorded."""
    import torch

    entry = torch.load(token_path, map_location='cpu', weights_only=False)
    codes = entry['codes']['co2'][:int(entry['n_frames'])]
    return not bool((codes != codes[:1]).any())


def refetch_co2(shot, dest):
    """Fetch the four fast CO2 PTDATA channels into ``dest/<shot>.h5`` (login node).

    Returns the file, or None when no channel is archived or every one is all zero
    (no CO2 that shot).
    """
    import h5py
    import numpy as np

    dest.mkdir(parents=True, exist_ok=True)
    config = dest / 'co2_ptdata.yaml'
    if not config.exists():
        config.write_text('trees:\n  ptdata:\n'
                          + ''.join(f'    - {p}\n' for p in CO2_PTDATA)
                          + 'prefetch_trees: []\n')
    # One output dir per shot: parallel fetches into a shared dir lose each
    # other's shard files.
    stage, outdir = dest / f'stage_{shot}', dest / shot
    outdir.mkdir(exist_ok=True)
    cmd = [str(FDP_ENV / 'bin/fdp'), 'run', str(FDP_ENV / 'bin/python'), '-u',
           str(Path(__file__).resolve().parent / 'fetch_shot.py'), '--shot', shot,
           '--config', str(config), '--output-dir', str(outdir), '--stage-dir', str(stage),
           '--no-retry', '--overwrite']
    env = dict(os.environ, BEARER_TOKEN_FILE=str(Path.home() / '.fdp/token'))
    # The FDP client crashes intermittently (as in a quarter of campaign fetches).
    for attempt in range(3):
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           check=True, timeout=1800, env=env, text=True)
            break
        except subprocess.CalledProcessError as exc:
            # No CO2 archived for this shot: the fetch exits 1 with every channel
            # missing from FDP's PTDATA index.
            if 'NotInIndexx4' in exc.stdout and 'read: 0 written' in exc.stdout:
                shutil.rmtree(outdir, ignore_errors=True)
                return None
            if attempt == 2:
                raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    out = outdir / f'{shot}.h5'
    with h5py.File(out, 'r') as f:
        group = f[shot].get('PTDATA', {})
        live = [p for p in CO2_PTDATA
                if p in group and np.any(group[p]['data'][::97])]
    if not live:
        shutil.rmtree(outdir)
        return None
    return out


def extract_raw(shot, raw, fetched, target, groups=('co2',)):
    """Copy the raw signals ``groups`` read into ``target`` (login node).

    Compute nodes cannot see /tigerdata, so they get this small file instead of the
    archive. The CO2 PTDATA channels come from ``fetched`` when it is given;
    ``raw`` may be None (no readable archive), leaving ``fetched`` the only source.
    """
    import h5py
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(
        REPO / 'src/tokamak_foundation_model/data/config/modalities/modalities.yaml'
    ), resolve=True)['signals']
    wanted = []
    for name in groups:
        group = cfg[name]
        wanted += [(group['tree'], key) for key in group['input_key']]
        if group.get('fallback'):
            wanted += [(group['fallback']['tree'], key)
                       for key in group['fallback']['input_key']]
    sources = [h5py.File(raw, 'r')] if raw is not None else []
    if fetched is not None:
        sources.insert(0, h5py.File(fetched, 'r'))
    try:
        with h5py.File(target, 'w') as out:
            dest = out.create_group(shot)
            for tree, key in wanted:
                for src in sources:
                    trees = src[shot]
                    name = tree if tree in trees else tree.lower()
                    if name in trees and key in trees[name]:
                        if key not in dest.require_group(name):
                            src.copy(trees[name][key], dest[name], name=key)
                        break
    finally:
        for src in sources:
            src.close()


def ip_end(raw, shot):
    """Plasma end [s] from the plasma current in the raw H5, with its peak [A].

    The end is the first time after the current peak at which |Ip| (pre-shot
    baseline removed) falls below 10% of that peak. Unlike the multi-signal
    noise-floor estimator of ``analysis/shot_duration.py`` (``tend_all.json``),
    which is off by more than 1 s on about a fifth of shots, this follows
    disruptions and long flat-tops alike. ``(None, peak)`` when the peak is below
    100 kA (no plasma) and ``(None, None)`` without a current signal.
    """
    import h5py
    import numpy as np

    with h5py.File(raw, 'r') as f:
        group = f[shot].get('PTDATA', {})
        name, scale = next(((n, k) for n, k in IP_POINTNAMES if n in group),
                           (None, None))
        if name is None:
            return None, None
        t = group[name]['dim0'][:].astype('f8') / 1000.0
        y = group[name]['data'][:].astype('f8') * scale
    pre = (t > -4.0) & (t < -0.5)
    if pre.sum() >= 20:
        y = y - np.median(y[pre])
    y = np.abs(y)
    window = np.flatnonzero((t > 0.0) & (t < 12.0))
    if not window.size:
        return None, None
    i_peak = window[np.nanargmax(y[window])]
    peak = float(y[i_peak])
    if peak < 1e5:
        return None, peak
    below = np.flatnonzero(y[i_peak:] < 0.1 * peak)
    end = t[i_peak + below[0]] if below.size else t[window[-1]]
    return float(end), peak


def reencode_co2(shot, models_dir, scratch):
    """Preprocess CO2 from ``scratch/<shot>/raw`` and re-encode it (CPU node)."""
    import h5py
    import torch

    import extend_frame_cache
    from tokamak_foundation_model.ignite import train_dynamics as td

    work = scratch / shot
    shutil.rmtree(work / 'processed', ignore_errors=True)
    (work / 'processed').mkdir(parents=True)
    preprocess(shot, work, ('co2',))
    processed = work / 'processed' / f'{shot}_processed.h5'
    tokens = models_dir / 'frame_codes'
    token = tokens / f'{shot}.pt'
    entry = torch.load(token, map_location='cpu', weights_only=False)
    t0 = float(json.loads((tokens / '_codec_manifest.json').read_text())['t0_start'])
    codec, cfg = td._load_codec('spectro', CO2_CODEC)
    n = int(entry['n_frames'])
    codes = extend_frame_cache.encode_shot(codec, cfg, 'co2', 'spectro', shot,
                                           str(work / 'processed'), t0, n)
    # The dataset's frame count follows the latest-ending group in the processed
    # file (data_loader._compute_duration), which here is CO2 alone, so it stops
    # at the last 50 ms window that fits inside CO2's data. Windows that run past
    # CO2's last sample hold no full CO2 window in either encoding (the loader
    # re-draws them from a nearby valid chunk), so such a missing tail repeats the
    # last frame. Anything else fails rather than truncating every modality
    # (which would also break act_global).
    if codes is not None and codes.shape[0] < n:
        with h5py.File(processed, 'r') as f:
            co2_end = float(f['co2/xdata'][-1])
        if t0 + (codes.shape[0] + 1) * 0.05 > co2_end:
            codes = torch.cat([codes, codes[-1:].expand(n - codes.shape[0], -1)])
    if codes is None or codes.shape[0] != n:
        raise RuntimeError(f'CO2 gives {None if codes is None else codes.shape[0]} '
                           f'frames, the entry has {n}')
    entry['codes']['co2'] = codes
    tmp = token.with_suffix(f'.pt.tmp.{os.getpid()}')
    torch.save(entry, tmp)
    tmp.replace(token)
    shutil.rmtree(work / 'processed')
    return bool((codes != codes[:1]).any())


def drive_co2_tend(args):
    """Login-node driver: re-fetch CO2 here, preprocess and encode on CPU nodes."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for models_dir in args.models_dirs:
        for pt in sorted((models_dir / 'frame_codes').glob('[0-9]*.pt')):
            if args.shots is not None and pt.stem not in args.shots:
                continue
            record = models_dir / 'plasma_end' / f'{pt.stem}.json'
            # A --tend-only record leaves the CO2 check to a later full pass.
            if not record.exists() or not (
                    args.tend_only or json.loads(record.read_text()).get('co2_checked')):
                jobs.append((pt.stem, models_dir))
    print(f'{len(jobs)} shots to do', flush=True)
    fetch_dir = args.scratch / 'co2_fetch'

    def one(job):
        shot, models_dir = job
        # Without a readable archive (missing or truncated) there is no plasma
        # current, so the end time falls back to tend_all.json, but CO2 can still
        # be re-encoded from the fresh PTDATA fetch alone.
        try:
            raw = find_raw(shot, args.raw_dir)
        except FileNotFoundError:
            if args.tend_only:
                raise
            raw = None
        t_end, peak = ip_end(raw, shot) if raw is not None else (None, None)
        record = dict(shot=shot, t_end=t_end, ip_peak_a=peak, co2_reencoded=False,
                      co2_checked=not args.tend_only)
        fetched = None
        if not args.tend_only and co2_flat(models_dir / 'frame_codes' / f'{shot}.pt'):
            fetched = refetch_co2(shot, fetch_dir)
        if fetched is not None:
            work = args.scratch / shot
            shutil.rmtree(work, ignore_errors=True)
            (work / 'raw').mkdir(parents=True)
            try:
                extract_raw(shot, raw, fetched, work / 'raw' / f'{shot}.h5')
                shutil.rmtree(fetched.parent)
                cmd = ['srun', '--account=kolemen', '--partition=cpu', '--nodes=1',
                       '--ntasks=1', '--cpus-per-task=8', '--mem=64G', '--time=1:00:00',
                       f'--job-name=co2-{shot}', sys.executable, __file__,
                       '--co2-tend-one', shot, '--scratch', str(args.scratch),
                       str(models_dir)]
                # The FDP campaign's own CPU jobs share the per-user QOS limit.
                for _ in range(60):
                    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, timeout=7200, text=True)
                    if 'QOSMaxSubmitJobPerUserLimit' not in result.stderr:
                        break
                    time.sleep(60)
                result.check_returncode()
            finally:
                shutil.rmtree(fetched.parent, ignore_errors=True)
                shutil.rmtree(work, ignore_errors=True)
            record.update(co2_reencoded=True, co2_live=not co2_flat(
                models_dir / 'frame_codes' / f'{shot}.pt'))
        out = models_dir / 'plasma_end' / f'{shot}.json'
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(record))
        return record

    failed = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(one, j): j[0] for j in jobs}
        for k, future in enumerate(as_completed(futures), 1):
            shot = futures[future]
            try:
                r = future.result()
                print(f'{shot} t_end={r["t_end"]} co2_reencoded={r["co2_reencoded"]}'
                      f' co2_live={r.get("co2_live")} ({k}/{len(jobs)})', flush=True)
            except Exception as exc:  # keep going; report every failure at the end
                detail = getattr(exc, 'stderr', '') or ''
                failed[shot] = f'{type(exc).__name__}: {exc} {detail[-300:]}'
                print(f'{shot} FAILED {failed[shot]}', flush=True)
    print(f'done: {len(jobs) - len(failed)} ok, {len(failed)} failed', flush=True)
    merge_tend(args.models_dirs)
    return failed


def merge_tend(models_dirs, fallback=None):
    """Write ``<models_dir>/tend_ip.json`` ({shot: t_end}) from the per-shot records.

    With ``fallback`` (an older {shot: t_end} table such as ``tend_all.json``), also
    write ``tend_ip_fallback.json``: the plasma-current end where there is one, the
    fallback's value for every other shot (no plasma, or no raw file).
    """
    old = {str(k): v for k, v in json.loads(fallback.read_text()).items()} if fallback else {}
    for models_dir in models_dirs:
        table = {}
        for path in sorted((models_dir / 'plasma_end').glob('*.json')):
            record = json.loads(path.read_text())
            if record['t_end'] is not None:
                table[str(record['shot'])] = record['t_end']
        (models_dir / 'tend_ip.json').write_text(json.dumps(table, indent=1))
        print(f'{models_dir}/tend_ip.json: {len(table)} shots', flush=True)
        if fallback:
            merged = {**old, **table}
            (models_dir / 'tend_ip_fallback.json').write_text(json.dumps(merged, indent=1))
            print(f'{models_dir}/tend_ip_fallback.json: {len(merged)} shots '
                  f'({len(set(old) - set(table))} from {fallback.name})', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models_dirs', nargs='+', type=Path,
                   help='Dirs holding frame_codes/; entries go to <dir>/act_global.')
    p.add_argument('--scratch', type=Path, default=REPO / 'scratch/act_global_backfill')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--raw-dir', type=Path, action='append', default=[],
                   help='Extra dirs searched for <shot>.h5 before the archive.')
    p.add_argument('--unchecked', action='append', default=[],
                   help='Actuator group left out of the token-entry check.')
    p.add_argument('--shots', nargs='+', default=None,
                   help='Only these shots (default: every shot without an entry).')
    p.add_argument('--co2-tend', action='store_true',
                   help='Plasma end times and CO2 re-encoding instead of act_global.')
    p.add_argument('--co2-tend-one', default=None, metavar='SHOT',
                   help='Internal: the CPU-node CO2 re-encode of --co2-tend for one shot.')
    p.add_argument('--tend-only', action='store_true',
                   help='--co2-tend without touching any token entry (end times only).')
    p.add_argument('--merge', action='store_true',
                   help='Only rewrite tend_ip.json from plasma_end/.')
    p.add_argument('--fallback', type=Path, default=None,
                   help='With --merge: also write tend_ip_fallback.json, taking this '
                        "table's end time for shots without a plasma-current end.")
    args = p.parse_args()
    if args.merge:
        merge_tend(args.models_dirs, args.fallback)
        return
    if args.co2_tend_one:
        print(reencode_co2(args.co2_tend_one, args.models_dirs[0], args.scratch))
        return
    if args.co2_tend:
        if drive_co2_tend(args):
            sys.exit(1)
        return
    jobs = []
    for models_dir in args.models_dirs:
        for pt in sorted((models_dir / 'frame_codes').glob('[0-9]*.pt')):
            if args.shots is not None and pt.stem not in args.shots:
                continue
            if not (models_dir / 'act_global' / pt.name).exists():
                jobs.append((pt.stem, models_dir))
    print(f'{len(jobs)} shots without act_global', flush=True)
    failed = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(build, s, d, args.scratch, args.raw_dir, args.unchecked): s
                   for s, d in jobs}
        for future in as_completed(futures):
            shot = futures[future]
            try:
                print(f'{shot} -> {future.result()}', flush=True)
            except Exception as exc:  # keep going; report every failure at the end
                failed[shot] = f'{type(exc).__name__}: {exc}'
                print(f'{shot} FAILED {failed[shot]}', flush=True)
    print(f'done: {len(jobs) - len(failed)} built, {len(failed)} failed', flush=True)
    if failed:
        print(json.dumps(failed, indent=1))
        sys.exit(1)


if __name__ == '__main__':
    main()
