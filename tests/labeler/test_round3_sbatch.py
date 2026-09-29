"""Round three's jobs, read from the scripts and never submitted: the three new
ones' scheduler contract and the flags they pass, SegNet's version, and no fetch
in any job the round submits."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path

import pytest

from .test_ae_sbatch import EXPORTS, FLAG, OUT, _help, _runs, _sbatch, _script

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"
TOKEYE_PY = "/scratch/gpfs/nc1514/tokeye/.venv/bin/python"
#: The CPU jobs round three adds, and the commands each runs, in order.
NEW_CPU = {
    "ae_masks_full_check.sbatch": ("labeler.ae.full",),
    "ae_below80.sbatch": ("labeler.ae.xpower.below80",),
}
#: Every existing script round three submits.
EXISTING = (
    "ae_seg_poi.sbatch",
    "ae_seg_train.sbatch",
    "ae_xpower_cv.sbatch",
    "ae_xpower_evaluate.sbatch",
    "ae_xpower_extend.sbatch",
    "ae_xpower_final.sbatch",
    "ae_xpower_gallery.sbatch",
    "review_build.sbatch",
    "spans.sbatch",
)


def _first_run(text: str) -> int:
    return text.index("\nsrun ")


def _exported_before_the_run(text: str, name: str) -> bool:
    """`name=1` on an export line above the first srun."""
    head = text[: _first_run(text)]
    return re.search(rf"^export .*(?<![\w-]){name}=1\b", head, re.MULTILINE) is not None


@pytest.mark.parametrize("name", sorted(NEW_CPU))
def test_a_new_cpu_job_on_the_short_queue_that_fetches_nothing(name):
    text = _script(name)
    queue = (_sbatch(text, "partition"), _sbatch(text, "qos"))
    assert queue == ("pppl", "pppl-short-stellar")
    assert _sbatch(text, "gres") is None and "--gpus" not in text
    assert all(_sbatch(text, f) for f in ("cpus-per-task", "mem", "time"))
    assert _sbatch(text, "output").startswith(OUT)
    for export in EXPORTS:
        assert -1 < text.find(export) < _first_run(text), export
    assert _exported_before_the_run(text, "LABELER_NO_FETCH")
    assert "python -m labeler.jobstats --job-id" in text and "--cpu-only" in text
    assert "sizing (measured" in text


@pytest.mark.parametrize("name", sorted(NEW_CPU))
def test_a_new_cpu_job_passes_flags_its_commands_take(name):
    runs = _runs(_script(name))
    assert tuple(module for module, _ in runs) == NEW_CPU[name]
    for module, flags in runs:
        text = _help(module)
        for flag in flags:
            assert re.search(rf"(?<![\w-]){flag}\b", text), (module, flag)


def _runner_help() -> str:
    """`scripts/labeler/ae_masks_full.py --help`, loaded as its test loads it."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "ae_masks_full_help", SCRIPTS / "ae_masks_full.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
            module.main(["--help"])
    finally:
        sys.path.remove(str(SCRIPTS))
    return out.getvalue()


def test_tokeye_over_whole_shots_is_the_one_gpu_job():
    text = _script("ae_masks_full.sbatch")
    assert (_sbatch(text, "partition"), _sbatch(text, "gres")) == ("gpu", "gpu:1")
    assert all(_sbatch(text, f) for f in ("cpus-per-task", "mem", "time"))
    assert _sbatch(text, "output").startswith(OUT)
    first = _first_run(text)
    assert -1 < text.find(f"PY={TOKEYE_PY}") < first, "the tokeye venv"
    assert -1 < text.find('PYTHONPATH="$ROOT/ae/pylibs:$REPO/src"') < first
    assert -1 < text.find("nvidia-smi --query-gpu=name,memory.total") < first
    assert _exported_before_the_run(text, "LABELER_NO_FETCH")
    assert _exported_before_the_run(text, "HF_HUB_OFFLINE")
    assert "python -m labeler.jobstats --job-id" in text and "--gpu" in text
    assert "sizing (measured" in text
    lines = text.replace("\\\n", " ").splitlines()
    [run] = [line for line in lines if line.startswith("srun ")]
    assert run.startswith('srun "$PY" -u "$REPO/scripts/labeler/ae_masks_full.py"')
    flags = re.findall(FLAG, run)
    assert {"--device", "--batch", "--workers", "--tile-ms"} <= set(flags)
    helped = _runner_help()
    for flag in flags:
        assert re.search(rf"(?<![\w-]){flag}\b", helped), flag


@pytest.mark.parametrize("name", EXISTING)
def test_every_job_the_round_submits_fetches_nothing(name):
    assert _exported_before_the_run(_script(name), "LABELER_NO_FETCH"), name


def test_segnet_trains_scores_and_draws_the_version_it_is_given():
    text = _script("ae_seg_train.sbatch")
    assert 'VERSION="${VERSION:-v1}"' in text, "v1 by default, so v1's commands mean v1"
    assert 'MODELS="$ROOT/models/ae_seg/$VERSION"' in text
    for module, flags in _runs(text):
        assert "--version" in flags, module
    assert text.count('--version "$VERSION"') == 2, "both runs take the variable"
    assert 'PILOT_DIR="$PILOT_DIR-$VERSION"' in text, "a v2 pilot is not v1's"
    poi = _script("ae_seg_poi.sbatch")
    assert 'VERSION="${VERSION:-v1}"' in poi
    [(module, flags)] = _runs(poi)
    assert module == "labeler.ae.seg.poi" and "--version" in flags
    assert poi.count('--version "$VERSION"') == 1
