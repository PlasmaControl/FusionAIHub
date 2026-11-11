"""Integrity and deletion gates for the FDP pilot (no network or torch)."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    'fdp_pilot', Path(__file__).resolve().parents[1]
    / 'scripts/data_fetching_fdp/validate_pipeline.py',
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_archive_collision_preserves_both_files(tmp_path):
    raw, archive = tmp_path / 'raw.h5', tmp_path / 'archive.h5'
    raw.write_bytes(b'new data')
    archive.write_bytes(b'old data')
    with pytest.raises(ValueError, match='collision'):
        pilot.archive_raw(raw, archive)
    assert raw.read_bytes() == b'new data'
    assert archive.read_bytes() == b'old data'


def test_archive_copy_is_verified_and_repeatable(tmp_path):
    raw, archive = tmp_path / 'raw.h5', tmp_path / 'archive.h5'
    raw.write_bytes(b'shot payload')
    expected = pilot.digest(raw)
    assert pilot.archive_raw(raw, archive) == expected
    assert pilot.archive_raw(raw, archive) == expected
    assert archive.read_bytes() == raw.read_bytes()
    assert not list(tmp_path.glob('*.partial.*'))


def test_interrupted_archive_cannot_authorize_cleanup(tmp_path, monkeypatch):
    raw, archive = tmp_path / 'raw.h5', tmp_path / 'archive.h5'
    raw.write_bytes(b'complete shot')

    def interrupted(src, dst, length):
        dst.write(b'partial')
        raise OSError('transfer interrupted')

    monkeypatch.setattr(pilot.shutil, 'copyfileobj', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        pilot.archive_raw(raw, archive)
    assert raw.read_bytes() == b'complete shot'
    with pytest.raises(ValueError, match='collision'):
        pilot.archive_raw(raw, archive)
    assert archive.read_bytes() == b'partial'


def cleanup_fixture(tmp_path):
    for folder in ('raw', 'processed', 'frame_codes'):
        (tmp_path / folder).mkdir()
    raw = tmp_path / 'raw/170001.h5'
    raw.write_bytes(b'raw')
    archive = tmp_path / 'archive.h5'
    archive.write_bytes(b'raw')
    codes = tmp_path / 'frame_codes/170001.pt'
    codes.write_bytes(b'codes')
    processed = tmp_path / 'processed/170001_processed.h5'
    processed.write_bytes(b'processed')
    report = dict(status='validated', shot='170001', archive=str(archive),
                  raw_sha256=pilot.digest(raw), code_sha256=pilot.digest(codes),
                  processed_sha256=pilot.digest(processed))
    pilot.save_json(tmp_path / 'report.json', report)
    return raw, archive, codes, processed


@pytest.mark.parametrize('corrupt', ['archive', 'codes', 'processed'])
def test_cleanup_refuses_changed_artifact(tmp_path, corrupt):
    raw, archive, codes, processed = cleanup_fixture(tmp_path)
    dict(archive=archive, codes=codes, processed=processed)[corrupt].write_bytes(b'bad')
    with pytest.raises(ValueError, match='keeping local'):
        pilot.cleanup(tmp_path)
    assert raw.exists() and processed.exists()


def test_cleanup_keeps_archive_and_codes(tmp_path):
    raw, archive, codes, processed = cleanup_fixture(tmp_path)
    pilot.cleanup(tmp_path)
    assert archive.read_bytes() == b'raw'
    assert codes.read_bytes() == b'codes'
    assert not raw.exists() and not processed.exists()
    assert json.loads((tmp_path / 'report.json').read_text())['local_h5_deleted']


def test_all_split_and_cached_shots_excluded(tmp_path):
    cache = tmp_path / 'frame_codes'
    cache.mkdir()
    (cache / '_codec_manifest.json').write_text('{}')
    (cache / '170004.pt').touch()
    run = tmp_path / 'runs/old_run'
    run.mkdir(parents=True)
    (run / 'split_shots.json').write_text(json.dumps(
        dict(train=[170001], val=[170002], test=[170003], pin_both=[170005])))
    excluded, splits = pilot.excluded_shots(tmp_path)
    assert {str(s) for s in range(170001, 170006)} <= excluded
    assert len(splits) == 1


@pytest.mark.parametrize('mds_rc,pt_rc,expected', [(0, 0, 0), (1, 0, 1), (0, 124, 1)])
def test_partial_fetch_cannot_report_success(tmp_path, mds_rc, pt_rc, expected):
    project = tmp_path / 'fdp'
    bin_dir = project / '.pixi/envs/default/bin'
    bin_dir.mkdir(parents=True)
    fdp = bin_dir / 'fdp'
    fdp.write_text('''#!/bin/bash
case "$*" in
  *--stage-only*) exit 0 ;;
esac
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --output-dir ]]; then
    printf payload > "$2/210000.h5"
  fi
  if [[ "$1" == --skip-tree ]]; then exit "$TEST_MDS_RC"; fi
  if [[ "$1" == --only-tree ]]; then exit "$TEST_PT_RC"; fi
  shift
done
exit 1
''')
    fdp.chmod(0o755)
    script = Path(pilot.__file__).with_name('fetch_shot_fast.sh')
    result = subprocess.run(
        ['bash', str(script), '210000', str(tmp_path / 'raw'), str(tmp_path / 'stage')],
        env={**os.environ, 'FDP_PROJECT': str(project),
             'TEST_MDS_RC': str(mds_rc), 'TEST_PT_RC': str(pt_rc)},
        capture_output=True, text=True,
    )
    assert (tmp_path / 'raw/210000.h5').is_file()
    assert result.returncode == expected, result.stdout + result.stderr


def load_staging_module():
    stage_spec = importlib.util.spec_from_file_location(
        'fdp_stage', Path(pilot.__file__).with_name('fdp_stage.py'))
    stage = importlib.util.module_from_spec(stage_spec)
    stage_spec.loader.exec_module(stage)
    return stage


def test_native_ptdata_uses_selected_snapshot(tmp_path, monkeypatch):
    stage = load_staging_module()
    monkeypatch.setenv('PTDATA_JSON_INDEX_DIR', 'pelican://example/index')

    def download(urls, dest):
        (dest / '207800.json').write_text(json.dumps({
            'pointname_ext': {'IP': '.PCA'},
            'ext_location': {'.PCA': 'pelican://example/207800.PCA'},
        }))
        (dest / '207800.PCA').write_bytes(b'staged object')
        return []

    monkeypatch.setattr(stage, 'stage_objects', download)
    stage.fetch_shot_index(207800, tmp_path, snapshot='json_indexes_2026-06-23')
    root = stage.pin_staged_index(tmp_path)
    assert root == 'pelican://example/index/json_indexes_2026-06-23'
    assert os.environ['PTDATA_JSON_INDEX_DIR'] == str(tmp_path / '_ptdata_index')
    localized = json.loads((tmp_path / '_ptdata_index/2078/207800.json').read_text())
    assert localized['ext_location']['.PCA'] == str(tmp_path / '207800.PCA')
    original = json.loads((tmp_path / '207800.json').read_text())
    assert original['ext_location']['.PCA'] == 'pelican://example/207800.PCA'


def test_snapshot_pin_preserves_index_free_mds_pass(tmp_path, monkeypatch):
    stage = load_staging_module()
    (tmp_path / '_index_source.json').write_text(json.dumps(
        {'snapshot_root': 'pelican://example/index/json_indexes_2026-06-23'}))
    monkeypatch.delenv('PTDATA_JSON_INDEX_DIR', raising=False)
    assert stage.pin_staged_index(tmp_path) is None
    assert 'PTDATA_JSON_INDEX_DIR' not in os.environ


@pytest.mark.parametrize('failure', [
    'Cannot open file: pelican://host/archives/index/json/2078/207800.json',
    'PTDATA fixups: repaired:timebasex2, defect:uncalibratedx1',
])
def test_fetch_quality_failures_block_preprocessing(failure):
    log = 'PTDATA index snapshot: pelican://host/json_indexes_2026-06-23\n'
    with pytest.raises(ValueError):
        pilot.validate_fetch_log(log + failure)


def _fetch_shot_module():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / 'scripts/data_fetching_fdp'))
    import fetch_shot
    return fetch_shot


@pytest.mark.parametrize('tdi, kept', [(0.0, True), (2.0, False)])
def test_all_zero_tdi_read_never_replaces_bulk_data(monkeypatch, tdi, kept):
    np = pytest.importorskip('numpy')
    fetch_shot = _fetch_shot_module()
    staged = {'data': np.linspace(1e12, 2e14, 50, dtype='f4'),
              'dim0': np.arange(50, dtype='f4'), 'units': b'f'}
    fixed = {'data': np.full(50, tdi, dtype='f4'),
             'dim0': np.arange(50, dtype='f4'), 'units': b' '}
    monkeypatch.setattr(fetch_shot.mds_reader, 'gather_ptdata_tdi',
                        lambda name, shot: fixed)
    payload, repaired = fetch_shot._repair_ptdata(
        'DENV1UF', staged, 'unverifiable_bulk', 188500)
    assert (payload is staged) == kept
    assert repaired == ('staged_kept_over_zero_tdi' if kept else 'unverifiable_bulk')
    # The fetch log note this produces must pass the pipeline's quality gate.
    import validate_pipeline
    validate_pipeline.validate_fetch_log(
        'PTDATA index snapshot: pelican://x/index\n'
        f'  PTDATA fixups: repaired:uncalibratedx6, repaired:{repaired}x4\n')
