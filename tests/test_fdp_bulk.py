"""Bulk publication must preserve working data on failed verification."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / 'scripts/data_fetching_fdp'))
import bulk_pipeline as bulk
import plan_descending


def test_verified_raw_replacement_overwrites_old_file(tmp_path):
    raw, archive = tmp_path / 'raw.h5', tmp_path / 'archive.h5'
    raw.write_bytes(b'new raw')
    archive.write_bytes(b'old raw')
    expected = bulk.archive_replace(raw, archive)
    assert archive.read_bytes() == b'new raw'
    assert expected == bulk.pilot.digest(raw)


def test_older_campaign_runs_in_descending_order():
    original = dict(shots=[100001, 150000, 189998, 189999, 190001])
    plan = plan_descending.descending_plan(original, 190000)
    assert plan['shots'] == [189999, 189998, 150000, 100001]
    assert plan['start'] == 190000
    assert plan['end'] == 100001
    assert plan['direction'] == 'descending'


def test_failed_staging_verification_preserves_old_archive(tmp_path, monkeypatch):
    raw, archive = tmp_path / 'raw.h5', tmp_path / 'archive.h5'
    raw.write_bytes(b'new raw')
    archive.write_bytes(b'old raw')
    real_digest = bulk.pilot.digest
    monkeypatch.setattr(bulk.pilot, 'digest', lambda p:
                        'corrupt' if 'fdp-partial' in str(p) else real_digest(p))
    with pytest.raises(ValueError, match='Staged archive'):
        bulk.archive_replace(raw, archive)
    assert archive.read_bytes() == b'old raw'
    assert raw.read_bytes() == b'new raw'


def setup_publication(tmp_path):
    work = tmp_path / 'work'
    (work / 'frame_codes').mkdir(parents=True)
    (work / 'processed').mkdir()
    codes = work / 'frame_codes/205000.pt'
    codes.write_bytes(b'validated codes')
    processed = work / 'processed/205000_processed.h5'
    processed.write_bytes(b'processed')
    archive = tmp_path / '205000.h5'
    archive.write_bytes(b'raw archive')
    manifest = dict(t0_start=1.0, frame_tokens=1209, resolved={'ece': '/codec.pt'})
    (work / 'frame_codes/_codec_manifest.json').write_text(json.dumps(manifest))
    report = dict(shot='205000', status='validated', codec_sha256={'ece': 'frozen'},
                  code_sha256=bulk.pilot.digest(codes),
                  processed_sha256=bulk.pilot.digest(processed),
                  raw_sha256=bulk.pilot.digest(archive), archive=str(archive),
                  raw_bytes=archive.stat().st_size)
    return work, processed, archive, report, dict(codec_sha256={'ece': 'frozen'})


def test_bulk_deletes_processed_only_after_validated_token_publication(tmp_path):
    work, processed, archive, report, plan = setup_publication(tmp_path)
    tokens = tmp_path / 'tokens'
    finished = bulk.finish_shot(work, tokens, report, plan)
    assert not processed.exists()
    assert archive.read_bytes() == b'raw archive'
    assert (tokens / '205000.pt').read_bytes() == b'validated codes'
    assert finished['status'] == 'completed'


@pytest.mark.parametrize('corrupt', ['archive', 'tokens', 'manifest', 'codecs'])
def test_publication_failure_keeps_processed_h5(tmp_path, corrupt):
    work, processed, archive, report, plan = setup_publication(tmp_path)
    tokens = tmp_path / 'tokens'
    tokens.mkdir()
    if corrupt == 'archive':
        archive.write_bytes(b'bad archive')  # same length, caught by checksum
    elif corrupt == 'tokens':
        (tokens / '205000.pt').write_bytes(b'other tokens')
    elif corrupt == 'manifest':
        (tokens / '_codec_manifest.json').write_text('{"t0_start": 0.0}')
    else:
        plan['codec_sha256'] = {'ece': 'different codec'}
    with pytest.raises(ValueError):
        bulk.finish_shot(work, tokens, report, plan)
    assert processed.exists()
