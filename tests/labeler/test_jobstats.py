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

from labeler import jobstats

DATA = Path(__file__).parent / "data" / "jobstats"


def text(name: str) -> str:
    return (DATA / name).read_text()


@pytest.fixture
def out(tmp_path, monkeypatch):
    """A data root of our own, so no test can write to the real one."""
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
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
    # `state` first, then the auto-detected CPU-only pair.
    assert list(v.checks) == ["state", "cpu", "cpu_mem"]
    assert v.checks["cpu"]["source"] == "jobstats"
    assert v.reasons == []


def test_cpu_only_is_auto_detected_from_the_absence_of_gpu_rows():
    cpu = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    gpu = jobstats.parse_jobstats(text("2923913_ae.jobstats.txt"))
    assert set(jobstats.gate(cpu, cpu_only=None).checks) == {
        "state", "cpu", "cpu_mem"
    }
    assert set(jobstats.gate(gpu, cpu_only=None).checks) == {
        "state", "cpu", "cpu_mem", "gpu", "gpu_mem"
    }
    # ...and the override wins over the auto-detection, both ways.
    assert set(jobstats.gate(gpu, cpu_only=True).checks) == {
        "state", "cpu", "cpu_mem"
    }
    assert set(jobstats.gate(cpu, cpu_only=False).checks) == {
        "state", "cpu", "cpu_mem", "gpu", "gpu_mem"
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
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
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
    """The three attributes of `CompletedProcess` this module reads."""

    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# ------------------------------------------------------------ the script

def test_the_script_is_a_thin_wrapper_over_the_library():
    script = (
        Path(__file__).parents[2] / "scripts" / "labeler" / "jobstats_check.py"
    )
    body = script.read_text()
    assert "from labeler.jobstats import main" in body
    assert "SystemExit(main())" in body


# ================================================================ review fixes
#
# Five findings from the L11 review, each with the case that produced it. The
# theme running through all of them is the same as the module's: a number the
# gate cannot vouch for must read as `undetermined` and FAIL, never as a
# measurement - and least of all as a PASS.

def _sacct(*rows: str) -> str:
    """A synthetic `sacct -P` capture with the format this module asks for."""
    header = ("JobID|State|Elapsed|ExitCode|MaxRSS|TotalCPU|AllocCPUS|ReqMem|"
              "AllocTRES|NodeList")
    return "\n".join([header, *rows]) + "\n"


# --- finding 1: the sacct fallback bypassed the absurd-value guard ----------

def test_an_absurd_sacct_maxrss_is_undetermined_not_a_pass():
    """The sacct-side twin of the `16.0PB` glitch must not sail through.

    17179869184K against a 16G reservation is 102 400 %, and before the fix
    the fallback handed that straight to `value >= threshold` and PASSED.
    """
    s = jobstats.parse_jobstats(text("2925412_5.jobstats.txt"))
    rows = jobstats.parse_sacct(_sacct(
        "2925412_5|COMPLETED|00:50:59|0:0||02:25:41|4|16G|cpu=4,mem=16G|n1",
        "2925412_5.batch|COMPLETED|00:50:59|0:0|6232K|00:00.080|4|||n1",
        "2925412_5.0|COMPLETED|00:50:58|0:0|17179869184K|02:25:40|4|||n1",
    ))
    v = jobstats.gate(s, cpu_only=None, sacct=rows)
    assert v.checks["cpu_mem"]["source"] == "undetermined"
    assert v.checks["cpu_mem"]["value"] is None
    assert not v.passed
    assert [r for r in v.reasons if "undetermined" in r]


# --- finding 5: a MaxRSS that is only the batch shell's ---------------------

def test_a_batch_only_maxrss_is_undetermined_not_a_confident_zero():
    """6 MB of shell on a 16 GB reservation is not "0 % of the memory used".

    It reads like a measurement and would send someone to shrink a
    reservation that may in fact have been nearly full.
    """
    s = jobstats.parse_jobstats(text("2925412_5.jobstats.txt"))
    rows = jobstats.parse_sacct(_sacct(
        "2925412_5|COMPLETED|00:50:59|0:0||02:25:41|4|16G|cpu=4,mem=16G|n1",
        "2925412_5.batch|COMPLETED|00:50:59|0:0|6000K|00:00.080|4|||n1",
        "2925412_5.extern|COMPLETED|00:50:59|0:0||00:00:00|4|||n1",
    ))
    v = jobstats.gate(s, cpu_only=None, sacct=rows)
    assert v.checks["cpu_mem"]["source"] == "undetermined"
    assert v.checks["cpu_mem"]["value"] is None
    assert any("batch shell" in r for r in v.reasons)


def test_a_numbered_step_maxrss_is_still_used():
    """The guard must not swallow the case the fallback exists for."""
    s = jobstats.parse_jobstats(text("2925412_5.jobstats.txt"))
    rows = jobstats.parse_sacct(text("2925412_5.sacct.txt"))
    v = jobstats.gate(s, cpu_only=None, sacct=rows)
    assert v.checks["cpu_mem"]["source"] == "sacct"
    assert v.checks["cpu_mem"]["value"] == pytest.approx(65.6, abs=0.05)


# --- finding 3: the gate has to know the job died --------------------------

def test_an_oom_killed_task_can_never_pass():
    """2925387_1 was killed for exceeding its 10 GB.

    jobstats reports its memory utilisation as 74 % - the SAMPLED mean - and
    before the fix that passed the memory check outright. A gate that drives
    re-sizing must not advise shrinking the memory of a job that OOMed.
    """
    s = jobstats.parse_jobstats(text("2925387_1.jobstats.txt"))
    rows = jobstats.parse_sacct(text("2925387_1.sacct.txt"))
    v = jobstats.gate(s, min_cpu=0, min_cpu_mem=0, cpu_only=None, sacct=rows)
    assert not v.passed
    assert v.checks["state"]["value"] == "OUT_OF_MEMORY"
    assert v.checks["state"]["source"] == "sacct 2925387_1.0"
    assert any("OUT_OF_MEMORY" in r for r in v.reasons)


@pytest.mark.parametrize(
    "state", ["OUT_OF_MEMORY", "FAILED", "CANCELLED by 1234567", "TIMEOUT",
              "NODE_FAIL"]
)
def test_every_fatal_state_fails_however_good_the_numbers(state):
    s = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    rows = jobstats.parse_sacct(_sacct(
        f"2925387_0|{state}|00:42:15|0:0||02:39:35|5|10G|cpu=5,mem=10G|n1",
    ))
    v = jobstats.gate(s, cpu_only=None, sacct=rows)
    assert v.checks["cpu"]["passed"] and v.checks["cpu_mem"]["passed"]
    assert not v.passed                       # ...and yet


def test_a_cancelled_job_is_not_a_sizing_signal():
    """2925203 ran at 93.5 % CPU and 75 % memory, and was CANCELLED."""
    s = jobstats.parse_jobstats(text("2925203.jobstats.txt"))
    assert s.state == "CANCELLED"
    v = jobstats.gate(s, cpu_only=None)
    assert v.checks["cpu"]["passed"] and v.checks["cpu_mem"]["passed"]
    assert not v.passed
    assert any("CANCELLED" in r for r in v.reasons)


def test_the_two_memory_figures_are_kept_apart_and_both_reported():
    """jobstats samples the mean; sacct's MaxRSS is a peak. Different scales.

    On 2925387_0 they are 70.0 % and 99.9 % - the gate uses one of them and
    has to say which, and record the other, or an array's ledger entries are
    not comparable to each other.
    """
    s = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    rows = jobstats.parse_sacct(text("2925387_0.sacct.txt"))
    check = jobstats.gate(s, cpu_only=None, sacct=rows).checks["cpu_mem"]
    assert check["source"] == "jobstats"
    assert check["jobstats_pct"] == pytest.approx(70.0)
    assert check["sacct_pct"] == pytest.approx(99.9, abs=0.1)
    assert "sampled" in check["measurement"]
    assert "MaxRSS" in jobstats.MEASUREMENTS["sacct"]


def test_the_record_carries_both_memory_figures(out):
    jobstats.main([
        "--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
        "--sacct-file", str(DATA / "2925387_0.sacct.txt"),
        "--out", str(out), "--quiet",
    ])
    check = json.loads(out.read_text())["jobs"]["2925387_0"]["verdict"]["checks"]
    assert check["cpu_mem"]["jobstats_pct"] == pytest.approx(70.0)
    assert check["cpu_mem"]["sacct_pct"] == pytest.approx(99.9, abs=0.1)


# --- finding 4: GPU detection cross-checks the allocation record -----------

def test_alloc_tres_is_parsed():
    rows = jobstats.parse_sacct(text("2913583_0.sacct.txt"))
    assert rows[0].alloc_tres == "billing=256,cpu=4,gres/gpu=1,mem=48G,node=1"
    assert jobstats.tres_gpus(rows[0].alloc_tres) == 1
    assert jobstats.tres_gpus("billing=5,cpu=5,mem=10G,node=1") == 0


def test_a_real_gpu_job_gates_on_all_four():
    s = jobstats.parse_jobstats(text("2913583_0.jobstats.txt"))
    rows = jobstats.parse_sacct(text("2913583_0.sacct.txt"))
    v = jobstats.gate(s, cpu_only=None, sacct=rows)
    assert s.gpu_util_pct == pytest.approx(83.1)
    assert s.gpu_mem_pct == pytest.approx(30.9)
    assert set(v.checks) == {"state", "cpu", "cpu_mem", "gpu", "gpu_mem"}


def test_a_gpu_job_whose_report_lost_its_gpu_rows_is_still_gated_on_gpu():
    """The wrong-PASS finding 4 guards against.

    jobstats' GPU rows are not always there; `sacct`'s AllocTRES is the
    allocation record and it is. A GPU job whose report arrives with CPU rows
    only must NOT auto-detect as CPU-only and pass on CPU alone - the GPU
    checks are undetermined, and undetermined fails.
    """
    stripped = "\n".join(
        line for line in text("2913583_0.jobstats.txt").splitlines()
        if "GPU" not in line
    )
    s = jobstats.parse_jobstats(stripped)
    assert s.gpus == 0 and s.gpu_util_pct is None   # jobstats knows nothing
    rows = jobstats.parse_sacct(text("2913583_0.sacct.txt"))

    assert set(jobstats.gate(s, cpu_only=None).checks) == {
        "state", "cpu", "cpu_mem"
    }                                                # ...without sacct
    v = jobstats.gate(s, cpu_only=None, sacct=rows)  # ...with it
    assert v.checks["gpu"]["source"] == "undetermined"
    assert v.checks["gpu_mem"]["source"] == "undetermined"
    assert not v.passed


def test_a_cpu_job_with_alloc_tres_is_still_cpu_only():
    s = jobstats.parse_jobstats(text("2925387_0.jobstats.txt"))
    rows = jobstats.parse_sacct(text("2925387_0.tres.sacct.txt"))
    assert "gres/gpu" not in rows[0].alloc_tres
    assert set(jobstats.gate(s, cpu_only=None, sacct=rows).checks) == {
        "state", "cpu", "cpu_mem"
    }


# --- finding 2: --preserve-dir must not truncate a good capture ------------

def test_preserve_dir_never_truncates_an_existing_capture(out, tmp_path,
                                                          monkeypatch):
    """`--wait-for-data` expiring is the case the flag exists for.

    When it does, `jobstats` has returned nothing, and the run used to write
    that nothing over the capture preserved by the run that worked.
    """
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "2925387_0.jobstats.txt").write_text("GOOD PRESERVED CAPTURE\n")
    (keep / "2925387_0.sacct.txt").write_text("GOOD PRESERVED SACCT\n")
    monkeypatch.setattr(
        jobstats.subprocess, "run", lambda cmd, **kw: _completed("")
    )
    code = jobstats.main([
        "--job-id", "2925387_0", "--preserve-dir", str(keep),
        "--out", str(out), "--quiet",
    ])
    assert code == 1
    assert (keep / "2925387_0.jobstats.txt").read_text() == (
        "GOOD PRESERVED CAPTURE\n"
    )
    assert (keep / "2925387_0.sacct.txt").read_text() == "GOOD PRESERVED SACCT\n"


def test_preserve_dir_leaves_no_temporary_files(out, tmp_path):
    keep = tmp_path / "keep"
    jobstats.main([
        "--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
        "--sacct-file", str(DATA / "2925387_0.sacct.txt"),
        "--out", str(out), "--preserve-dir", str(keep), "--quiet",
    ])
    assert sorted(p.name for p in keep.iterdir()) == [
        "2925387_0.jobstats.txt", "2925387_0.sacct.txt"
    ]
    assert [p.name for p in out.parent.iterdir() if p.name.startswith(".")] == []


# --- housekeeping ----------------------------------------------------------

def test_expand_array_matches_the_id_exactly():
    """`29253870_1` is not a task of `2925387`."""
    listing = "2925387_0\n29253870_1\n2925387_1\n"
    assert jobstats.expand_array("2925387", listing) == ["2925387_0", "2925387_1"]


def test_a_running_job_keeps_its_run_time():
    """`Run Time: 00:10:00 (in progress)` is a run time, not a blank."""
    report = text("2925387_0.jobstats.txt").replace(
        "Run Time: 00:42:15", "Run Time: 00:10:00 (in progress)"
    )
    s = jobstats.parse_jobstats(report)
    assert s.run_time_s == 600.0
    assert s.in_progress is True
    assert jobstats.parse_jobstats(text("2925387_0.jobstats.txt")).in_progress is False


def test_help_has_no_empty_default_noise(capsys):
    with pytest.raises(SystemExit):
        jobstats.main(["--help"])
    printed = capsys.readouterr().out
    assert "(default: )" not in printed
    assert "(default: None)" not in printed
    assert "(default: 70.0)" in printed          # the ones that mean something


def test_a_failed_jobstats_call_puts_its_stderr_in_the_reason(out, monkeypatch,
                                                              capsys):
    """"no such job" and "not populated yet" must not look identical."""
    def fake_run(cmd, **kwargs):
        if cmd[0] == "jobstats":
            return _completed(
                "", "Error: JobID 99999999\nnot found.", 1
            )
        return _completed("")

    monkeypatch.setattr(jobstats.subprocess, "run", fake_run)
    code = jobstats.main(["--job-id", "99999999", "--out", str(out)])
    assert code == 1
    printed = capsys.readouterr().out
    assert "Error: JobID 99999999 not found." in printed
    # One reason, one line: jobstats wraps its own error messages.
    assert all(line.strip() for line in printed.splitlines())
    assert len(printed.splitlines()) == 3       # the line, and two reasons
    record = json.loads(out.read_text())["jobs"]["99999999"]
    assert record["jobstats_stderr"] == "Error: JobID 99999999\nnot found."


def test_the_ledger_is_replaced_atomically(out):
    """A half-written ledger is worse than a stale one."""
    jobstats.main(["--jobstats-file", str(DATA / "2925387_0.jobstats.txt"),
                   "--out", str(out), "--quiet"])
    before = out.read_text()
    jobstats.main(["--jobstats-file", str(DATA / "2925412_5.jobstats.txt"),
                   "--out", str(out), "--quiet"])
    assert json.loads(before)["jobs"].keys() == {"2925387_0"}
    assert sorted(p.name for p in out.parent.iterdir()) == ["jobstats.json"]
