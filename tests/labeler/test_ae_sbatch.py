"""The AE jobs' scheduler contract, read from the scripts: nothing is submitted."""

from __future__ import annotations

import contextlib
import importlib
import io
import re
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"
#: Each script and the commands it runs, in order.
JOBS = {
    "ae_xpower_train.sbatch": ("labeler.ae.xpower.train",),
    "ae_xpower_evaluate.sbatch": ("labeler.ae.xpower.evaluate",) * 2,
}
EXPORTS = (
    'export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"',
    'export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-',
    "OMP_NUM_THREADS=1",
    "HDF5_USE_FILE_LOCKING=FALSE",
)
OUT = "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/"
FLAG = r"(?<![\w-])--[a-z][\w-]*"


def _script(name: str) -> str:
    path = SCRIPTS / name
    subprocess.run(["bash", "-n", str(path)], check=True)
    return path.read_text()


def _sbatch(text: str, flag: str) -> str | None:
    m = re.search(rf"^#SBATCH --{flag}=(\S+)$", text, re.MULTILINE)
    return m[1] if m else None


def _runs(text: str) -> list[tuple[str, list[str]]]:
    """Each `srun ... -m <module> <flags>` the script runs (not its comments)."""
    found = []
    for line in text.replace("\\\n", " ").splitlines():
        m = re.match(r"srun .*?-m (labeler\.[\w.]+)(.*)$", line)
        if m:
            found.append((m[1], re.findall(FLAG, m[2])))
    return found


def _help(module: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        importlib.import_module(module).main(["--help"])
    return out.getvalue()


@pytest.mark.parametrize("name", sorted(JOBS))
def test_a_cpu_job_on_the_short_queue_with_its_inputs_exported(name):
    text = _script(name)
    queue = (_sbatch(text, "partition"), _sbatch(text, "qos"))
    assert queue == ("pppl", "pppl-short-stellar")
    assert _sbatch(text, "gres") is None and "--gpus" not in text
    assert all(_sbatch(text, f) for f in ("cpus-per-task", "mem", "time"))
    assert _sbatch(text, "output").startswith(OUT)
    first = text.index("\nsrun ")
    for export in EXPORTS:
        assert -1 < text.find(export) < first, export
    assert "python -m labeler.jobstats --job-id" in text, "the gate, in the header"
    assert "sizing (measured" in text


@pytest.mark.parametrize("name", sorted(JOBS))
def test_the_script_runs_its_commands_with_flags_they_take(name):
    runs = _runs(_script(name))
    assert tuple(module for module, _ in runs) == JOBS[name]
    for module, flags in runs:
        text = _help(module)
        for flag in flags:
            assert re.search(rf"(?<![\w-]){flag}\b", text), (module, flag)


def test_the_train_array_has_a_task_per_candidate():
    from labeler.ae.xpower.train import CANDIDATES

    text = _script("ae_xpower_train.sbatch")
    assert _sbatch(text, "array") == f"0-{len(CANDIDATES) - 1}"
