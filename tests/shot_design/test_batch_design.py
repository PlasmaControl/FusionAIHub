"""The production-root guard of `scripts/shot_design/batch_design.py`: a batch never designs into a cluster's own root."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "shot_design" / "batch_design.py"
STELLAR = "/scratch/gpfs/EKOLEMEN/nc1514/shot_design"
STELLARAI = "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/shot_design"


@pytest.mark.parametrize("root", [STELLAR, STELLARAI + "/"])
def test_a_production_root_is_refused_however_it_is_spelled(root, tmp_path):
    env = {**os.environ, "SHOT_DESIGN_DATA_ROOT": root, "LABELER_CLUSTER": ""}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--prompts", str(tmp_path / "p.jsonl"),
         "--out", str(tmp_path / "o.jsonl")],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "must name a batch root, not production" in result.stderr
