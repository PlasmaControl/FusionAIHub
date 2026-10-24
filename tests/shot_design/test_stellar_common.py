"""The guard in `scripts/shot_design/_stellar_common.sh`: a batch never runs on a production root."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COMMON = REPO / "scripts" / "shot_design" / "_stellar_common.sh"
STELLAR = "/scratch/gpfs/EKOLEMEN/nc1514/shot_design"
STELLARAI = "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/shot_design"


def _source(root: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "REPO": str(REPO), "SHOT_DESIGN_PY": sys.executable,
           "SHOT_DESIGN_DATA_ROOT": root, "LABELER_CLUSTER": ""}
    return subprocess.run(
        ["bash", "-c", f'source "{COMMON}" && echo "root=$ROOT"'],
        env=env, capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("spelling", [
    STELLAR, STELLAR + "/", STELLAR + "//", STELLAR.replace("/nc1514/", "/nc1514//"),
    STELLAR.replace("/nc1514/", "/nc1514/./"), "/scratch/gpfs/EKOLEMEN/nc1514/x/../shot_design",
    STELLARAI, STELLARAI + "/", STELLARAI.replace("/.cache/", "//.cache/./"),
])
def test_either_production_root_is_refused_however_it_is_spelled(spelling):
    result = _source(spelling)
    assert result.returncode == 2
    assert "refusing to run a batch against the production data root" in result.stderr


def test_a_symlink_to_a_production_root_is_refused(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(STELLAR)
    assert _source(str(link)).returncode == 2


def test_a_batch_root_runs_and_gets_its_log_folder(tmp_path):
    batch = tmp_path / "experiments" / "demo"
    result = _source(str(batch))
    assert result.returncode == 0, result.stderr
    assert f"root={batch}" in result.stdout
    assert (batch / "runs" / "slurm").is_dir()
