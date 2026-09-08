"""The utilisation gate: parsing real `jobstats`/`sacct` text, and the gate on it.

Every fixture under `data/jobstats/` is a VERBATIM capture of one of our own
Stellar jobs, kept because the numbers a gate refuses a job over have to come
from the format the cluster actually prints and not from a format we imagined:

* `2925387_0` - a healthy CPU-only task (75.5 % CPU, 70 % CPU memory);
* `2925387_1` - the same array, a task whose step was OUT_OF_MEMORY;
* `2925412_5` - the trap this module exists for. `jobstats` prints
  `CPU memory usage (Value was erroneously found to be >100%)` with a
  detail line reading `16.0PB/16GB`, and its notes say the value "could not
  be determined". A parser that reads that as 0 % would fail a job that in
  fact used 65.6 % of its memory; a parser that skips it would pass a job it
  knows nothing about. It must be `None`, and the gate must fall back to
  `sacct`'s `MaxRSS`/`ReqMem`.
* `2923913_ae` - the old AE dataset job, the GPU example the spec names:
  23 % CPU, 6 % CPU memory, 1 % GPU, 97 % GPU memory, which must fail three
  of the four checks and exit 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from labelmaker import jobstats

DATA = Path(__file__).parent / "data" / "jobstats"


def text(name: str) -> str:
    return (DATA / name).read_text()


@pytest.fixture
def out(tmp_path, monkeypatch):
    """A data root of our own, so no test can write to the real one."""
    monkeypatch.setenv("LABELMAKER_ROOT", str(tmp_path / "root"))
    return tmp_path / "jobstats.json"


# --------------------------------------------------------------- parsing

def test_parses_the_header_of_a_cpu_only_task():
    s = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    assert s.job_id == "2925387_0"
    assert s.state == "COMPLETED"
    assert s.nodes == 1
    assert s.cpu_cores == 5
    assert s.cpu_mem_gb == pytest.approx(10.0)
    assert s.gpus == 0
    assert s.run_time_s == 42 * 60 + 15
    assert s.time_limit_s == 70 * 60


def test_healthy_cpu_task_parses_to_its_reported_numbers():
    s = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    # 02:39:25/03:31:15 -> efficiency=75.5 %, and 7.0GB/10GB -> 70 %.
    assert s.cpu_util_pct == pytest.approx(75.5)
    assert s.cpu_mem_pct == pytest.approx(70.0)
    assert s.gpu_util_pct is None
    assert s.gpu_mem_pct is None


def test_per_node_detail_is_kept():
    s = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    cpu = s.per_node["cpu_util"]["stellar-k09n1"]
    assert cpu["used_s"] == 2 * 3600 + 39 * 60 + 25
    assert cpu["run_s"] == 3 * 3600 + 31 * 60 + 15
    mem = s.per_node["cpu_mem"]["stellar-k09n1"]
    assert mem["used_bytes"] == round(7.0 * 1024 ** 3)
    assert mem["alloc_bytes"] == 10 * 1024 ** 3


def test_notes_are_kept_as_lines():
    s = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    assert any("overall CPU utilization of this job is 75.5%" in n for n in s.notes)
    assert any("Have a nice day" in n for n in s.notes)


def test_an_out_of_memory_task_keeps_its_state():
    s = jobstats.parse_jobstats(text("2925387_1.jobstats.txt"))
    assert s.state == "FAILED"
    assert s.cpu_util_pct == pytest.approx(64.6)
    rows = jobstats.parse_sacct(text("2925387_1.sacct.txt"))
    assert [r.state for r in rows if r.job_id.endswith(".0")] == ["OUT_OF_MEMORY"]


def test_an_erroneous_memory_value_is_undetermined_not_zero():
    s = jobstats.parse_jobstats(text("2925412_5.jobstats.txt"))
    assert s.cpu_util_pct == pytest.approx(70.6)
    assert s.cpu_mem_pct is None
    assert any("could not be determined" in n for n in s.notes)
    # The bogus detail line is kept verbatim, so a human can see WHY.
    assert s.per_node["cpu_mem"]["stellar-i01n1"]["used_bytes"] == 16 * 1024 ** 5


def test_the_gpu_job_parses_to_the_numbers_the_spec_names():
    s = jobstats.parse_jobstats(text("2923913_ae.jobstats.txt"))
    assert s.gpus == 1
    # The spec's acceptance for this job is jobstats' own headline bar:
    # "parses old dataset.json -> 23/6/1/97".
    assert s.overall_pct == {"cpu": 23.0, "cpu_mem": 6.0, "gpu": 1.0, "gpu_mem": 97.0}
    # The gated values are the more precise ones from the detail lines.
    assert s.cpu_util_pct == pytest.approx(23.9)
    assert s.cpu_mem_pct == pytest.approx(5.9, abs=0.05)
    assert s.gpu_util_pct == pytest.approx(0.6)
    assert s.gpu_mem_pct == pytest.approx(96.6)


def test_a_blank_report_parses_to_nothing_rather_than_raising():
    s = jobstats.parse_jobstats("")
    assert s.job_id == ""
    assert s.cpu_util_pct is None
    assert s.cpu_mem_pct is None
    assert not jobstats.has_utilisation(s)


# ----------------------------------------------------------------- sacct

def test_sacct_rows_carry_the_steps():
    rows = jobstats.parse_sacct(text("2925412_5.sacct.txt"))
    assert [r.job_id for r in rows] == [
        "2925412_5", "2925412_5.batch", "2925412_5.extern", "2925412_5.0"
    ]
    assert rows[0].alloc_cpus == 4
    assert rows[0].req_mem == "16G"
    assert rows[3].max_rss_bytes == 10999176 * 1024


def test_max_rss_and_req_mem_in_bytes():
    rows = jobstats.parse_sacct(text("2925412_5.sacct.txt"))
    assert jobstats.max_rss_bytes(rows) == 10999176 * 1024
    assert jobstats.req_mem_bytes(rows) == 16 * 1024 ** 3


@pytest.mark.parametrize(
    "req_mem, alloc_cpus, expected",
    [
        ("16G", 4, 16 * 1024 ** 3),
        ("1500M", 1, 1500 * 1024 ** 2),
        ("2800Mn", 8, 2800 * 1024 ** 2),
        ("2800Mc", 8, 8 * 2800 * 1024 ** 2),
        ("4Gc", 5, 5 * 4 * 1024 ** 3),
        ("", 4, None),
    ],
)
def test_req_mem_suffixes(req_mem, alloc_cpus, expected):
    assert jobstats.parse_req_mem(req_mem, alloc_cpus) == expected


# ------------------------------------------------------------------ gate

def test_a_healthy_cpu_task_passes():
    s = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    v = jobstats.gate(s, min_cpu=70, min_cpu_mem=70, min_gpu=70, min_gpu_mem=70,
                      cpu_only=None)
    assert v.passed
    assert set(v.checks) == {"cpu", "cpu_mem"}          # auto-detected CPU-only
    assert v.checks["cpu"]["source"] == "jobstats"
    assert v.reasons == []


def test_cpu_only_is_auto_detected_from_the_absence_of_gpu_rows():
    cpu = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    gpu = jobstats.parse_jobstats(text("2923913_ae.jobstats.txt"))
    assert set(jobstats.gate(cpu, cpu_only=None).checks) == {"cpu", "cpu_mem"}
    assert set(jobstats.gate(gpu, cpu_only=None).checks) == {
        "cpu", "cpu_mem", "gpu", "gpu_mem"
    }
    # ...and the override wins over the auto-detection, both ways.
    assert set(jobstats.gate(gpu, cpu_only=True).checks) == {"cpu", "cpu_mem"}
    assert set(jobstats.gate(cpu, cpu_only=False).checks) == {
        "cpu", "cpu_mem", "gpu", "gpu_mem"
    }


def test_the_gpu_job_fails_three_of_the_four_checks():
    s = jobstats.parse_jobstats(text("2923913_ae.jobstats.txt"))
    v = jobstats.gate(s, cpu_only=None)
    assert not v.passed
    failed = [k for k, c in v.checks.items() if not c["passed"]]
    assert failed == ["cpu", "cpu_mem", "gpu"]
    assert v.checks["gpu_mem"]["passed"]
    assert len(v.reasons) == 3


def test_undetermined_memory_falls_back_to_sacct():
    s = jobstats.parse_jobstats(text("2925412_5.jobstats.txt"))
    rows = jobstats.parse_sacct(text("2925412_5.sacct.txt"))
    v = jobstats.gate(s, cpu_only=None, sacct=rows)
    # 10999176K / 16G = 65.6 %, which is under 70 and so a real failure -
    # but a failure the gate can NAME, not an undetermined one.
    assert v.checks["cpu_mem"]["value"] == pytest.approx(65.6, abs=0.05)
    assert v.checks["cpu_mem"]["source"] == "sacct"
    assert not v.passed
    assert not any("undetermined" in r for r in v.reasons)
    assert v.checks["cpu"]["passed"]


def test_undetermined_without_sacct_fails_rather_than_reading_as_zero():
    s = jobstats.parse_jobstats(text("2925412_5.jobstats.txt"))
    v = jobstats.gate(s, cpu_only=None, sacct=None)
    assert not v.passed
    assert v.checks["cpu_mem"]["value"] is None
    assert v.checks["cpu_mem"]["source"] == "undetermined"
    assert [r for r in v.reasons if "undetermined" in r]


def test_the_sacct_fallback_passes_a_job_that_used_its_memory():
    s = jobstats.parse_jobstats(text("2925412_5.jobstats.txt"))
    rows = jobstats.parse_sacct(text("2925412_5.sacct.txt"))
    v = jobstats.gate(s, min_cpu_mem=60, cpu_only=None, sacct=rows)
    assert v.passed


# ------------------------------------------------------- array expansion

def test_array_expansion_from_a_captured_listing():
    ids = jobstats.expand_array("2925387", text("2925387.array.sacct.txt"))
    assert ids == [f"2925387_{i}" for i in range(10)]


def test_a_task_id_expands_to_itself():
    assert jobstats.expand_array("2925387_3", text("2925387.array.sacct.txt")) == [
        "2925387_3"
    ]


def test_a_pending_array_range_expands():
    assert jobstats.expand_array("99_", "99_[3-5,8%2]\n") == [
        "99_3", "99_4", "99_5", "99_8"
    ]


def test_an_empty_listing_falls_back_to_the_id_itself():
    assert jobstats.expand_array("2925387", "") == ["2925387"]


# ------------------------------------------------------------------- CLI

def test_help_lists_every_flag(capsys):
    with pytest.raises(SystemExit) as e:
        jobstats.main(["--help"])
    assert e.value.code == 0
    text_out = capsys.readouterr().out
    for flag in (
        "--job-id", "--jobstats-file", "--sacct-file", "--min-cpu",
        "--min-cpu-mem", "--min-gpu", "--min-gpu-mem", "--cpu-only", "--gpu",
        "--pilot", "--wait-for-data", "--out", "--preserve-dir", "--quiet",
    ):
        assert flag in text_out


def test_offline_pass_prints_one_line_and_exits_zero(out, capsys):
    code = jobstats.main([
        "--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
        "--sacct-file", str(DATA / "2925387_0.sacct.txt"),
        "--out", str(out),
    ])
    assert code == 0
    line = capsys.readouterr().out.splitlines()[0]
    assert line == "2925387_0  CPU 75.5 %  CPU-mem 70 %  GPU n/a  GPU-mem n/a  PASS"


def test_the_gpu_job_exits_one_through_main(out, capsys):
    code = jobstats.main([
        "--jobstats-file", str(DATA / "2923913_ae.jobstats.txt"),
        "--out", str(out),
    ])
    assert code == 1
    printed = capsys.readouterr().out
    assert "FAIL" in printed
    record = json.loads(out.read_text())["jobs"]["2923913"]
    assert len(record["verdict"]["reasons"]) == 3
    assert record["verdict"]["passed"] is False


def test_pilot_reports_the_failure_but_exits_zero(out, capsys):
    code = jobstats.main([
        "--jobstats-file", str(DATA / "2923913_ae.jobstats.txt"),
        "--out", str(out), "--pilot",
    ])
    assert code == 0
    assert "exempt" in capsys.readouterr().out
    verdict = json.loads(out.read_text())["jobs"]["2923913"]["verdict"]
    assert verdict["exempt"] is True
    assert verdict["passed"] is False
    assert len(verdict["reasons"]) == 3


def test_quiet_prints_nothing(out, capsys):
    jobstats.main([
        "--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
        "--out", str(out), "--quiet",
    ])
    assert capsys.readouterr().out == ""


def test_thresholds_are_configurable(out):
    code = jobstats.main([
        "--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
        "--out", str(out), "--min-cpu", "80", "--quiet",
    ])
    assert code == 1
    record = json.loads(out.read_text())["jobs"]["2925387_0"]
    assert record["thresholds"]["cpu"] == 80.0
    assert record["thresholds"]["cpu_mem"] == 70.0


def test_the_gpu_override_gates_a_cpu_only_report_on_gpu_too(out):
    code = jobstats.main([
        "--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
        "--out", str(out), "--gpu", "--quiet",
    ])
    assert code == 1
    checks = json.loads(out.read_text())["jobs"]["2925387_0"]["verdict"]["checks"]
    assert checks["gpu"]["source"] == "undetermined"


def test_out_json_appends_then_replaces_by_job_id(out):
    jobstats.main(["--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
                   "--out", str(out), "--quiet"])
    jobstats.main(["--jobstats-file", str(DATA / "2925412_5.jobstats.txt"),
                   "--out", str(out), "--quiet"])
    assert sorted(json.loads(out.read_text())["jobs"]) == ["2925387_0", "2925412_5"]

    # Re-running one job replaces its record rather than adding a second.
    jobstats.main(["--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
                   "--out", str(out), "--quiet"])
    doc = json.loads(out.read_text())
    assert sorted(doc["jobs"]) == ["2925387_0", "2925412_5"]
    assert doc["jobs"]["2925387_0"]["verdict"]["passed"] is True


def test_a_record_carries_the_raw_texts_and_the_parsed_values(out):
    jobstats.main([
        "--jobstats-file", str(DATA / "2925412_5.jobstats.txt"),
        "--sacct-file", str(DATA / "2925412_5.sacct.txt"),
        "--out", str(out), "--quiet",
    ])
    record = json.loads(out.read_text())["jobs"]["2925412_5"]
    assert record["jobstats_text"] == text("2925412_5.jobstats.txt")
    assert record["sacct_text"] == text("2925412_5.sacct.txt")
    assert record["stats"]["cpu_util_pct"] == pytest.approx(70.6)
    assert record["stats"]["cpu_mem_pct"] is None
    assert record["verdict"]["checks"]["cpu_mem"]["source"] == "sacct"
    assert record["timestamp"].startswith("20")


def test_out_defaults_under_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELMAKER_ROOT", str(tmp_path / "root"))
    jobstats.main(["--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
                   "--quiet"])
    assert (tmp_path / "root" / "runs" / "slurm" / "jobstats.json").is_file()


def test_preserve_dir_writes_the_controllers_two_files(out, tmp_path):
    keep = tmp_path / "keep"
    jobstats.main([
        "--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
        "--sacct-file", str(DATA / "2925387_0.sacct.txt"),
        "--out", str(out), "--preserve-dir", str(keep), "--quiet",
    ])
    assert (keep / "2925387_0.jobstats.txt").read_text() == text(
        "2925387_0.jobstats.txt"
    )
    assert (keep / "2925387_0.sacct.txt").read_text() == text(
        "2925387_0.sacct.txt"
    )


# ------------------------------------------------- the live path, faked

def test_job_id_runs_jobstats_and_sacct_and_gates_every_array_task(
    out, monkeypatch, capsys
):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "sacct" and "-X" in cmd:
            return _completed(text("2925387.array.sacct.txt"))
        if cmd[0] == "sacct":
            return _completed(text(f"{cmd[2]}.sacct.txt"))
        return _completed(text(f"{cmd[1]}.jobstats.txt"))

    monkeypatch.setattr(jobstats.subprocess, "run", fake_run)
    monkeypatch.setattr(
        jobstats, "expand_array", lambda job_id, listing: ["2925387_0", "2925387_1"]
    )
    code = jobstats.main(["--job-id", "2925387", "--out", str(out)])
    assert code == 1                       # 2925387_1 is the 64.6 % OOM task
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("2925387_0") and lines[0].endswith("PASS")
    assert lines[1].startswith("2925387_1") and lines[1].endswith("FAIL")
    assert sorted(json.loads(out.read_text())["jobs"]) == ["2925387_0", "2925387_1"]
    assert ["jobstats", "2925387_0"] in calls


def test_wait_for_data_retries_until_the_report_is_populated(out, monkeypatch):
    reports = ["", "", text("2925387_0.jobstats.txt")]
    slept = []
    monkeypatch.setattr(jobstats, "_sleep", slept.append)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "sacct":
            return _completed(text("2925387_0.sacct.txt"))
        return _completed(reports.pop(0))

    monkeypatch.setattr(jobstats.subprocess, "run", fake_run)
    code = jobstats.main([
        "--job-id", "2925387_0", "--wait-for-data", "120",
        "--out", str(out), "--quiet",
    ])
    assert code == 0
    assert reports == []
    assert slept == [jobstats.POLL_SECONDS, jobstats.POLL_SECONDS]


def test_wait_for_data_gives_up_and_reports_undetermined(out, monkeypatch, capsys):
    monkeypatch.setattr(jobstats, "_sleep", lambda seconds: None)
    monkeypatch.setattr(
        jobstats.subprocess, "run", lambda cmd, **kw: _completed("")
    )
    code = jobstats.main([
        "--job-id", "2925387_0", "--wait-for-data", "40", "--out", str(out),
    ])
    assert code == 1
    assert "n/a" in capsys.readouterr().out
    checks = json.loads(out.read_text())["jobs"]["2925387_0"]["verdict"]["checks"]
    assert checks["cpu"]["source"] == "undetermined"


class _completed:
    """The two attributes of `CompletedProcess` this module reads."""

    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


# ------------------------------------------------------------ the script

def test_the_script_is_a_thin_wrapper_over_the_library():
    script = (
        Path(__file__).parents[2] / "scripts" / "labelmaker" / "jobstats_check.py"
    )
    body = script.read_text()
    assert "from labelmaker.jobstats import main" in body
    assert "SystemExit(main())" in body
