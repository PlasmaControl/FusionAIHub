"""House-style checks on the Frontier shot_design SLURM wrappers.

A wrapper is only ever exercised by submitting it, so the cheap mistakes -- the
wrong account, a Stellar path or QOS copied over with the body, logs written
into the repository -- are caught here instead of by a job that dies at minute
one of a six-hour allocation.
"""

import re
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[2] / "scripts" / "slurm_frontier"
SCRIPTS = ["shot_design_census.sh", "shot_design_build.sh", "shot_design_encode.sh",
           "shot_design_simulate.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_frontier_shot_design_scripts_follow_house_style(name):
    text = (SLURM / name).read_text()
    assert "#SBATCH -A fus187" in text
    assert re.search(r"#SBATCH -p (batch|extended)", text)
    assert "_shot_design_common.sh" in text
    assert "/scratch/gpfs" not in text, "Stellar path leaked into a Frontier script"
    assert "gpu-stellar" not in text and "pppl" not in text
    assert "runs/slurm/%j" in text or "runs/slurm/%A_%a" in text
