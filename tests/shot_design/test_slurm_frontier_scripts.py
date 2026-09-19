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


def test_the_census_writes_the_parquet_name_every_consumer_defaults_to():
    """`corpus scan --out`, `select` and `coverage` all default to
    `<db_dir>/corpus_coverage.parquet`; a census under any other name is written and
    then never read."""
    text = (SLURM / "shot_design_census.sh").read_text()
    n = text.count('"$ROOT/db/corpus_coverage.parquet"')
    assert n == 2, "scan --out and summary must name the same file"
    assert "db/census.parquet" not in text


def test_the_gpu_sampler_measures_only_the_gcds_the_job_was_given():
    """rocm-smi ignores ROCR_VISIBLE_DEVICES and prints every card on the node, so
    without an explicit `-d` a one-GCD job on an exclusive node is averaged against idle
    siblings -- and the utilisation gate then judges that eighth."""
    text = (SLURM / "_gpu_sampler.sh").read_text()
    assert "ROCR_VISIBLE_DEVICES" in text
    assert "-d " in text
