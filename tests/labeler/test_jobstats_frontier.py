"""The Frontier backend: `sacct` plus a `rocm-smi` sample file, no `jobstats`.

OLCF has no `jobstats`, so the utilisation gate has no report to parse there and
every check would read `undetermined` -- which the gate treats as a failure, so
the gate would be useless rather than merely silent. `parse_frontier` builds the
same `JobStats` from the two things Frontier does have: `sacct`, and the GPU
samples `scripts/slurm_frontier/_gpu_sampler.sh` writes beside the job's log.

The field names here are `JobStats`' own (`cpu_util_pct`, ...); the numbers are
the brief's.
"""

import json
import subprocess
from pathlib import Path

from labeler import jobstats

SACCT = """JobID|Elapsed|AllocCPUS|ReqMem|MaxRSS|TotalCPU|State
123|01:00:00|56|100G||56:00:00|COMPLETED
123.0|01:00:00|56||40G|50:24:00|COMPLETED
"""
GPU = "epoch_s,gpu_pct,vram_used_mb,vram_total_mb\n1,80,32000,65536\n2,60,32000,65536\n"


def test_parse_frontier_reads_cpu_and_gpu_utilisation():
    s = jobstats.parse_frontier(SACCT, GPU)
    assert s.job_id == "123"
    assert round(s.cpu_util_pct) == 90       # 50.4 h TotalCPU / (1 h * 56 cores)
    assert round(s.cpu_mem_pct) == 40        # 40G / 100G
    assert round(s.gpu_util_pct) == 70       # mean of samples
    assert round(s.gpu_mem_pct) == 49        # 32000/65536
    assert jobstats.has_utilisation(s)


def test_parse_frontier_without_gpu_samples_is_cpu_only():
    s = jobstats.parse_frontier(SACCT, None)
    assert s.gpu_util_pct is None and s.gpu_mem_pct is None


def test_the_cli_uses_sacct_alone_when_the_cluster_has_no_jobstats(
    tmp_path, monkeypatch
):
    """The auto-detect, end to end: no `jobstats` on PATH means none is run.

    Running it anyway is not harmless -- the "command not found" lands in the
    record's `jobstats_stderr`, and the gate then reports a missing tool as the
    reason the job failed its check.
    """
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path / "sd"))
    (tmp_path / "sd" / "runs" / "slurm").mkdir(parents=True)
    (tmp_path / "sd" / "runs" / "slurm" / "123.gpu.csv").write_text(GPU)
    monkeypatch.setattr(jobstats, "frontier_backend", lambda: True)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        out = SACCT if cmd[0] == "sacct" else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(jobstats.subprocess, "run", fake_run)
    out = tmp_path / "jobstats.json"
    jobstats.main(["--job-id", "123", "--out", str(out), "--quiet"])

    assert "jobstats" not in calls
    record = json.loads(out.read_text())["jobs"]["123"]
    assert record["jobstats_stderr"] == ""
    assert round(record["stats"]["gpu_util_pct"]) == 70


def test_the_sampler_writes_the_header_this_module_parses():
    """The sampler writes the CSV and `parse_gpu_samples` reads it by position.

    The two live in different languages in different directories, and nothing
    else pins the column order, so a column added to one and not the other
    would silently turn VRAM into a utilisation figure.
    """
    sampler = (Path(__file__).resolve().parents[2]
               / "scripts" / "slurm_frontier" / "_gpu_sampler.sh")
    assert f'"{jobstats.GPU_SAMPLE_HEADER}" > "$out"' in sampler.read_text()
