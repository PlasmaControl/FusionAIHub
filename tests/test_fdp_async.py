"""Storage admission and independent compute/archive lifecycle checks."""
import json
from pathlib import Path
import queue
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / 'scripts/data_fetching_fdp'))
import async_pipeline as pipeline


def test_shared_reservations_prevent_combined_campaign_overcommit(tmp_path):
    usage = [31 * pipeline.TB]
    budget = pipeline.DiskBudget(
        tmp_path, limit=35 * pipeline.TB, probe=lambda _: usage[0],
        reservation=pipeline.TB, headroom=pipeline.TB,
    )
    assert budget.admit(('ascending', 205000))
    assert budget.admit(('descending', 189999))
    assert budget.admit(('ascending', 205001))
    assert not budget.admit(('descending', 189998))
    usage[0] += pipeline.TB  # Another project job consumes storage.
    assert not budget.admit(('ascending', 205002))
    assert budget.snapshot()['admission_allowed'] is False


def test_unknown_quota_stops_admission(tmp_path):
    def unavailable(_):
        raise OSError('quota service unavailable')

    budget = pipeline.DiskBudget(tmp_path, probe=unavailable)
    assert not budget.admit(('ascending', 205000))
    assert budget.snapshot()['quota_error']


def test_raw_backlog_keeps_a_reservation_until_archived(tmp_path):
    (tmp_path / 'raw').mkdir()
    raw = tmp_path / 'raw/205000.h5'
    raw.write_bytes(b'queued raw')
    budget = pipeline.DiskBudget(tmp_path, probe=lambda _: 30*pipeline.TB)
    key = ('ascending', 205000)
    assert budget.admit(key)
    budget.retain(key, tmp_path)
    assert budget.reserved[key] >= raw.stat().st_size
    raw.unlink()
    budget.retain(key, tmp_path)
    assert key not in budget.reserved


def archive_fixture(tmp_path):
    raw, archive = tmp_path / 'raw.h5', tmp_path / 'archive.h5'
    raw.write_bytes(b'new scientific raw data')
    archive.write_bytes(b'older raw')
    job = tmp_path / 'archive_job.json'
    pipeline.bulk.pilot.save_json(job, dict(
        key='ascending-205000', shot='205000', status='queued',
        raw=str(raw), archive=str(archive), bytes=raw.stat().st_size,
    ))
    return raw, archive, job


def test_async_archive_verifies_replacement_without_reading_older_file(tmp_path,
                                                                     monkeypatch):
    raw, archive, job = archive_fixture(tmp_path)
    read_paths = []
    original = pipeline.bulk.pilot.digest

    def digest(path):
        read_paths.append(Path(path))
        return original(path)

    monkeypatch.setattr(pipeline.bulk.pilot, 'digest', digest)
    result = pipeline.archive_job(job)
    assert result['status'] == 'verified'
    assert archive.read_bytes() == b'new scientific raw data'
    assert not raw.exists()
    assert archive not in read_paths  # Verified temp renamed; no redundant read.


def test_failed_archive_keeps_source_and_older_destination(tmp_path, monkeypatch):
    raw, archive, job = archive_fixture(tmp_path)
    monkeypatch.setattr(pipeline.bulk.pilot, 'digest', lambda _: 'corrupted')
    result = pipeline.archive_job(job)
    assert result['status'] == 'failed'
    assert raw.exists()
    assert archive.read_bytes() == b'older raw'


def test_verified_archive_restart_cleans_raw_only_if_unchanged(tmp_path):
    raw, archive, job = archive_fixture(tmp_path)
    data = json.loads(job.read_text())
    data.update(status='verified', sha256=pipeline.bulk.pilot.digest(raw))
    pipeline.bulk.pilot.save_json(job, data)
    raw.write_bytes(b'changed raw data')
    with pytest.raises(ValueError, match='changed'):
        pipeline.archive_job(job)
    assert raw.exists()


def test_tokens_publish_and_processed_h5_disappears_before_archive_finishes(
        tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    work = campaign / 'work/205000'
    (work / 'raw').mkdir(parents=True)
    (work / 'processed').mkdir()
    (campaign / 'reports').mkdir()
    raw = work / 'raw/205000.h5'
    raw.write_bytes(b'scientific raw')
    processed = work / 'processed/205000_processed.h5'
    processed.write_bytes(b'processed signal')
    archive = tmp_path / 'archive/205000.h5'
    archive.parent.mkdir()
    tokens = tmp_path / 'tokens'
    report = dict(shot='205000', source='FDP', status='preprocessed',
                  fetch_complete=True, preprocess_complete=True,
                  raw_bytes=raw.stat().st_size, archive=str(archive))
    pipeline.bulk.pilot.save_json(work / 'report.json', report)
    plan = dict(source_sha256={}, archive_dir=str(archive.parent),
                token_dir=str(tokens), codec_sha256={'ece': 'frozen'})

    def encode(path, shot, stage):
        assert stage == 'encode'
        frame_dir = path / 'frame_codes'
        frame_dir.mkdir()
        codes = frame_dir / '205000.pt'
        codes.write_bytes(b'validated discrete tokens')
        (frame_dir / '_codec_manifest.json').write_text('{"t0_start": 1.0}')
        record = json.loads((path / 'report.json').read_text())
        record.update(status='validated', codec_sha256=plan['codec_sha256'],
                      code_sha256=pipeline.bulk.pilot.digest(codes),
                      processed_sha256=pipeline.bulk.pilot.digest(processed))
        pipeline.bulk.pilot.save_json(path / 'report.json', record)

    def act_global(path, shot, models_dir):
        assert processed.exists()  # Built from the processed H5 before deletion.
        target = models_dir / 'act_global' / f'{shot}.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'actuators')
        return target

    monkeypatch.setattr(pipeline, 'cpu_stage', encode)
    monkeypatch.setattr(pipeline, 'publish_act_global', act_global)
    notifications = queue.Queue()
    result = pipeline.compute_shot('ascending', 205000, campaign, plan,
                                   notifications)
    assert result['status'] == 'tokens_ready'
    assert (tmp_path / 'act_global/205000.pt').exists()
    assert result['archive_pending']
    assert not processed.exists()
    assert raw.exists()  # The asynchronous transfer has not happened yet.
    assert not archive.exists()
    assert (tokens / '205000.pt').exists()
    assert pipeline.finalize(campaign, 205000)['status'] == 'tokens_ready'
    direction, shot, job_path = notifications.get_nowait()
    pipeline.archive_job(job_path)
    assert pipeline.finalize(campaign, 205000)['status'] == 'completed'
    assert not raw.exists()


def test_failed_shot_drops_h5_but_keeps_logs(tmp_path):
    for name in ('raw', 'processed'):
        (tmp_path / name).mkdir()
    (tmp_path / 'raw/205000.h5').write_bytes(b'r' * 10)
    (tmp_path / 'processed/205000_processed.h5').write_bytes(b'p' * 5)
    (tmp_path / 'fetch.log').write_text('BrokenProcessPool')
    assert pipeline.drop_failed_h5(tmp_path) == 15
    assert not list(tmp_path.glob('*/*.h5'))
    assert (tmp_path / 'fetch.log').exists()


def test_failed_shot_keeps_raw_held_by_an_archive_job(tmp_path):
    for name in ('raw', 'processed'):
        (tmp_path / name).mkdir()
    (tmp_path / 'raw/205000.h5').write_bytes(b'r' * 10)
    (tmp_path / 'processed/205000_processed.h5').write_bytes(b'p' * 5)
    (tmp_path / 'archive_job.json').write_text('{}')
    assert pipeline.drop_failed_h5(tmp_path) == 5
    assert (tmp_path / 'raw/205000.h5').exists()


def test_scattered_failures_never_stop_a_campaign_but_a_run_does(tmp_path):
    import os
    records = {}
    for i in range(400):
        shot = 100000 + i
        work = tmp_path / 'work' / str(shot)
        work.mkdir(parents=True)
        # One shot in four fails: 100 failures in total, but never a run.
        status = 'failed' if i % 4 == 0 else 'completed'
        (work / 'report.json').write_text(json.dumps(dict(shot=shot, status=status)))
        os.utime(work / 'report.json', (i, i))
        records[shot] = dict(status=status)
    outcomes = pipeline.recent_outcomes(tmp_path, records)
    assert len(outcomes) == pipeline.RECENT_WINDOW
    assert not pipeline.failing(outcomes)
    for _ in range(pipeline.RECENT_FAILURES):
        outcomes.append(True)
    assert pipeline.failing(outcomes)


def test_cpu_stage_waits_out_the_job_submit_limit(tmp_path, monkeypatch):
    import subprocess
    calls = []

    def fake_run(cmd, stdout, **_):
        calls.append(cmd)
        if len(calls) < 3:
            stdout.write('srun: error: QOSMaxSubmitJobPerUserLimit\n')
            return subprocess.CompletedProcess(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    monkeypatch.setattr(pipeline, 'QOS_WAIT_S', 0)
    pipeline.cpu_stage(tmp_path, 205000, 'preprocess')
    assert len(calls) == 3


def test_cpu_stage_fails_at_once_on_other_errors(tmp_path, monkeypatch):
    import subprocess
    calls = []

    def fake_run(cmd, stdout, **_):
        calls.append(cmd)
        stdout.write('Traceback: KeyError\n')
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.cpu_stage(tmp_path, 205000, 'preprocess')
    assert len(calls) == 1
