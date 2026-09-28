"""v2's jobs: the folds, the choice and the final model; read, never submitted."""

from __future__ import annotations

import re
import subprocess

import pytest

from labeler.ae.xpower import train

from .test_ae_sbatch import EXPORTS, OUT, SCRIPTS, _help, _runs, _sbatch

JOBS = {
    "ae_xpower_cv.sbatch": ("labeler.ae.xpower.cv",),
    "ae_xpower_final.sbatch": ("labeler.ae.xpower.cv", "labeler.ae.xpower.train"),
}
#: The scripts that already ran v1: they pass VERSION on, v1 by default.
V1_SCRIPTS = (
    "ae_xpower_train.sbatch",
    "ae_xpower_evaluate.sbatch",
    "ae_xpower_gallery.sbatch",
    "ae_xpower_extend.sbatch",
)
TABLES = (
    'export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-'
    '/scratch/gpfs/nc1514/FusionAIHub/data/events}"'
)


def _script(name: str) -> str:
    path = SCRIPTS / name
    subprocess.run(["bash", "-n", str(path)], check=True)
    return path.read_text()


@pytest.mark.parametrize("name", sorted(JOBS))
def test_a_four_thread_cpu_job_with_its_inputs_pilot_and_gate(name):
    text = _script(name)
    assert (_sbatch(text, "partition"), _sbatch(text, "qos")) == (
        "pppl",
        "pppl-short-stellar",
    )
    assert _sbatch(text, "gres") is None and "--gpus" not in text
    assert _sbatch(text, "cpus-per-task") == "4"
    assert _sbatch(text, "output").startswith(OUT)
    first = text.index("\nsrun ")
    for export in (*EXPORTS, TABLES):
        assert -1 < text.find(export) < first, export
    assert 'VERSION="${VERSION:-v2}"' in text
    assert "python -m labeler.jobstats --job-id" in text
    assert "sizing (measured" in text
    assert "pilot:" in text and "exempt from the gate but reported" in text


@pytest.mark.parametrize("name", sorted(JOBS))
def test_the_script_runs_its_commands_with_flags_they_take(name):
    runs = _runs(_script(name))
    assert tuple(module for module, _ in runs) == JOBS[name]
    for module, flags in runs:
        assert "--version" in flags
        text = _help(module)
        for flag in flags:
            assert re.search(rf"(?<![\w-]){flag}\b", text), (module, flag)
    if name == "ae_xpower_final.sbatch":
        assert "--choose" in runs[0][1] and "--from-cv" in runs[1][1]


def test_the_cv_array_is_every_candidate_and_fold_in_its_header_order():
    text = _script("ae_xpower_cv.sbatch")
    names = list(train.candidates("v2"))
    assert _sbatch(text, "array") == f"0-{len(names) * 5 - 1}" == "0-14"
    assert "candidates('$VERSION'))[$K // 5]" in text and "FOLD=$((K % 5))" in text
    stated = "#   0-4 band80-mhd3    5-9 band80-mhd10    10-14 band80-mhd30"
    assert stated in text
    assert [names[k // 5] for k in (0, 5, 10)] == stated.split()[2::2]
    assert "--folds" in text  # the step the array needs first, in the header


@pytest.mark.parametrize("name", V1_SCRIPTS)
def test_the_v1_scripts_pass_the_version_on_v1_by_default(name):
    text = _script(name)
    assert 'VERSION="${VERSION:-v1}"' in text and TABLES in text
    runs = _runs(text)
    assert runs and all("--version" in flags for _, flags in runs)


def test_the_train_array_reads_the_versions_candidates():
    text = _script("ae_xpower_train.sbatch")
    assert "candidates('$VERSION'))[$K]" in text
    evaluate = _script("ae_xpower_evaluate.sbatch")
    assert "runs/ae_xpower/pilot/$VERSION" in evaluate
    assert "--test" in _runs(evaluate)[1][1]
