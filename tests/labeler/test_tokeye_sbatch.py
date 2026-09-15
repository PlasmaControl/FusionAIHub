"""Pin the L12 scheduler contract without submitting jobs from tests."""

import re
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"


def _script(name):
    path = SCRIPTS / name
    assert path.is_file(), f"missing scheduler companion: {name}"
    subprocess.run(["bash", "-n", str(path)], check=True)
    return path.read_text()


def test_array_reserves_one_gpu_and_enough_cpus_for_both_pools():
    text = _script("tokeye_masks.sbatch")
    for flag in ("--nodes=1", "--ntasks=1", "--gpus-per-task=1",
                 "--partition=gpu", "--qos=gpu-stellar", "--array=0-0%1"):
        assert f"#SBATCH {flag}" in text
    cpus = int(re.search(r"#SBATCH --cpus-per-task=(\d+)", text)[1])
    workers = int(re.search(r'PREP_WORKERS=\"\$\{PREP_WORKERS:-(\d+)\}', text)[1])
    prefetch = int(re.search(r'PREFETCH=\"\$\{PREFETCH:-(\d+)\}', text)[1])
    tail = int(re.search(r'TAIL_WORKERS="\$\{TAIL_WORKERS:-(\d+)\}', text)[1])
    assert cpus == workers + tail + 1
    assert prefetch >= workers
    assert "--mem=" in text and "--time=" in text
    assert "#SBATCH --output=/dev/null" in text
    assert '$ROOT/runs/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out' in text
    assert 'srun --cpu-bind=cores "$PHASE3_PYTHON" -u' in text
    for flag in ('--chunk "$SLURM_ARRAY_TASK_ID"', '--n-chunks "$N_CHUNKS"',
                 "--rank 0 --world 1", "--device cuda", "--plan round1",
                 '--tail-workers "$TAIL_WORKERS"', "--text-subset readonly", "--no-index",
                 "--amp", "--timeout 240"):
        assert flag in text
    for export in ('PYTHONPATH="$REPO/src"', "OMP_NUM_THREADS=1",
                   "MKL_NUM_THREADS=1", "HDF5_USE_FILE_LOCKING=FALSE"):
        assert text.index(export) < text.index("srun --cpu-bind")


def test_prepass_builds_the_whole_list_on_cpu():
    text = _script("tokeye_text_subset.sh")
    assert '--shot-file "$SHOT_FILE"' in text
    assert "--build-text-subset" in text
    assert "--device cpu" in text
    assert "--chunk" not in text and "--limit" not in text
    assert "-e labelmaker python" in text


def test_prepass_stages_the_site_jobstats_client_for_compute_nodes():
    text = _script("tokeye_text_subset.sh")
    assert 'install -d -m 700 "$ROOT/runs/slurm/jobstats-client"' in text
    for name in ("jobstats", "jobstats.py", "config.py", "output_formatters.py"):
        assert name in text
    afterok = _script("tokeye_masks_afterok.sbatch")
    assert 'export PATH="$ROOT/runs/slurm/jobstats-client:$PATH"' in afterok


def test_afterok_rebuilds_before_the_gate_and_preserves_both_captures():
    text = _script("tokeye_masks_afterok.sbatch")
    assert "--dependency=afterok:" in text
    assert text.index("--rebuild-index") < text.index('"$REPO/scripts/labeler/jobstats_check.py"')
    assert '--index-out "$ROOT/events/events_index.parquet"' in text
    assert '"$REPO/scripts/labeler/jobstats_check.py"' in text
    for flag in ('--job-id "$JOBID"', '--preserve-dir "$ROOT/runs/slurm"',
                 '--out "$ROOT/runs/slurm/jobstats.json"', "--wait-for-data 300"):
        assert flag in text
    assert "-e labelmaker python" in text
    assert 'gate_mode+=(--pilot)' in text
    assert "--gpus" not in text
    assert text.count("srun --cpu-bind=cores pixi run") == 2


@pytest.mark.parametrize("workers,prefetch,cpus", [(6, 5, 8), (6, 6, 7)])
def test_array_rejects_inconsistent_worker_reservations(workers, prefetch, cpus):
    import os

    path = SCRIPTS / "tokeye_masks.sbatch"
    assert path.is_file()
    env = dict(os.environ, PREP_WORKERS=str(workers), PREFETCH=str(prefetch),
               SLURM_CPUS_PER_TASK=str(cpus), SLURM_ARRAY_TASK_ID="0",
               N_CHUNKS="1", SHOT_FILE="/tmp/unused-l12-shots.txt")
    done = subprocess.run(["bash", str(path)], env=env, capture_output=True,
                          text=True, check=False)
    assert done.returncode == 2
    assert "must" in done.stderr


def test_scratch_root_separates_outputs_from_readonly_runtime(tmp_path):
    """ROOT must win over an activated production LABELMAKER_ROOT."""
    import json
    import os

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    capture = tmp_path / 'arguments.json'
    for name, body in {
        'nvidia-smi': '#!/bin/sh\nexit 0\n',
        'srun': '#!/usr/bin/env python3\nimport json, os, sys\n'
                'with open(os.environ["CAPTURE"], "w") as handle:\n'
                '    json.dump(sys.argv[1:], handle)\n',
    }.items():
        program = bin_dir / name
        program.write_text(body)
        program.chmod(0o755)
    root = tmp_path / 'scratch-products'
    repo = SCRIPTS.parents[1]
    env = dict(os.environ, ROOT=str(root), LABELMAKER_ROOT='/production',
               REPO=str(repo), CAPTURE=str(capture),
               PATH=f'{bin_dir}:/usr/bin:/bin', SHOT_FILE='/input/shots.txt',
               PHASE3_PYTHON='/readonly/phase3/bin/python',
               UNET='/readonly/checkpoint.pt', PREP_WORKERS='6', PREFETCH='6',
               TAIL_WORKERS='1',
               SLURM_CPUS_PER_TASK='8', SLURM_ARRAY_TASK_ID='0',
               SLURM_ARRAY_JOB_ID='1234', SLURM_JOB_ID='1234', N_CHUNKS='1')
    done = subprocess.run(['bash', str(SCRIPTS / 'tokeye_masks.sbatch')],
                          env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    args = json.loads(capture.read_text())
    assert args[args.index('--root') + 1] == str(root)
    assert args[args.index('--unet') + 1] == '/readonly/checkpoint.pt'
    assert args[1] == '/readonly/phase3/bin/python'
    log = (root / 'runs/slurm/1234_0.out').read_text()
    assert f'Resolved root: {root}' in done.stdout
    assert f'Resolved root: {root}' in log
    assert log.index('Resolved root:') < log.index('job 1234_0')
    assert '/production' not in ' '.join(args)


@pytest.mark.parametrize("name", ["tokeye_text_subset.sh", "tokeye_masks_afterok.sbatch"])
def test_cpu_workflow_uses_existing_frozen_environment(name):
    text = _script(name)
    for line in text.splitlines():
        if "pixi run" in line:
            assert "pixi run --frozen --no-install --manifest-path" in line
    assert "HF_HUB_OFFLINE=1" in text


def test_defaults_record_second_pilot_capacity_and_failed_gpu_gate():
    text = _script("tokeye_masks.sbatch")
    for measured in ("2932066_0", "97.29 tiles/s", "GPU 7.4%", "13559492K"):
        assert measured in text
    for flag in ("--cpus-per-task=12", "--mem=17G", "--time=00:10:00"):
        assert f"#SBATCH {flag}" in text
    for name, value in (("PREP_WORKERS", 7), ("TAIL_WORKERS", 4), ("PREFETCH", 8)):
        assert f'{name}="${{{name}:-{value}}}"' in text
