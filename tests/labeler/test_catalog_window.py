"""The assessed window and the flat-top, from Ip traces drawn by hand."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import struct
import subprocess
import sys
import types

import numpy as np
import pytest

from labeler.config import Paths, git_dirty, git_sha
from labeler.events.catalog import window
from labeler.events.catalog.check import CatalogError
from labeler.events.catalog.window import (
    assessed_window,
    fetch,
    flattop_s,
    read_log,
    summarise,
)
from labeler.events.raw import write_group
from labeler.events.verify import NoDataError
from labeler.features.store import FeatureArray
from shot_design.shotdb.select import flattop_from_ip


def _trace(stop=4000.0, peak=1e6):
    """Up over 0-500 ms, flat to 2000 ms, down by 2500 ms; samples at x.2 and x.7."""
    t = np.arange(-1000.3, stop, 0.5)
    return t, peak * np.interp(t, [0, 500, 2000, 2500], [0, 1, 1, 0])


def test_the_window_is_rounded_inward_whatever_the_sign():
    t, ip = _trace()  # 50 kA at 25 ms and 2475 ms; samples 25.2 ... 2474.7
    assert assessed_window(t, ip) == (26, 2474)
    assert assessed_window(t, -ip) == (26, 2474)


def test_a_sample_on_a_whole_ms_counts_whichever_way_its_float_rounds():
    t = np.array([5.0, 6.0, 7.0, 8.0])
    for noise in (-1e-7, 0.0, 1e-7):  # float64 steps, float32 seconds
        assert assessed_window(t + noise, [0.0, 6e4, 6e4, 0.0]) == (6, 7)


def test_pickup_away_from_the_plasma_does_not_move_the_window():
    t, ip = _trace()
    ip[(t >= -900) & (t < -892)] = 60e3  # the pre-magnetisation pulse
    ip[(t >= 3000) & (t < 3005)] = -60e3
    assert assessed_window(t, ip) == (26, 2474)


def test_short_gaps_are_bridged_and_long_ones_split():
    t, ip = _trace()
    short = ip.copy()
    short[(t >= 30) & (t < 35)] = 0.0
    short[(t >= 1000) & (t < 1001)] = np.nan
    assert assessed_window(t, short) == (26, 2474)
    long = ip.copy()
    long[(t >= 100) & (t < 120)] = 0.0
    assert assessed_window(t, long) == (121, 2474)  # the longer stretch


def test_no_plasma_no_window():
    t, ip = _trace(peak=40e3)
    assert assessed_window(t, ip) is None
    assert assessed_window(t, np.full_like(t, np.nan)) is None
    assert assessed_window([], []) is None


@pytest.mark.parametrize("sign", [1, -1])
def test_a_quench_ends_at_the_plasmas_own_sign(sign):
    t, _ = _trace()
    ip = 1e6 * np.interp(t, [0, 500, 2000, 2005, 3000], [0, 1, 1, -0.3, 0])
    assert assessed_window(t, sign * ip) == (26, 2003)


@pytest.mark.parametrize("dip", ["one_sample", "six_ms"])
def test_a_brief_dip_does_not_split_the_flattop(dip):
    t, ip = _trace()
    if dip == "one_sample":
        ip[np.argmin(abs(t - 1000.2))] = 0.79e6
    else:
        ip[(t >= 1000) & (t < 1006)] = 0.78e6
    assert flattop_s(t, ip, (26, 2474)) == pytest.approx(1.6995)


def test_the_noisy_flattop_agrees_across_native_rates():
    rng = np.random.default_rng(0)
    measured = []
    for start, dt in [(-1000.03, 0.05), (-1000.3, 0.5)]:
        t = np.arange(start, 4000.0, dt)
        ip = 1e6 * np.interp(t, [0, 500, 2000, 2500], [0, 1, 1, 0])
        ip += rng.normal(0, 4e3, t.size)
        measured.append(flattop_s(t, ip, assessed_window(t, ip)))
    assert measured[0] == pytest.approx(measured[1], abs=0.002)


def test_the_flattop_is_measured_inside_the_window():
    t, ip = _trace()  # at least 0.8 MA from 400.2 to 2099.7 ms
    assert flattop_s(t, ip, (26, 2474)) == pytest.approx(1.6995)
    far_t, far_ip = _trace(stop=40_000.0)  # the same shot, a longer record
    assert flattop_s(far_t, far_ip, (26, 2474)) == pytest.approx(1.6995)
    assert flattop_from_ip(far_t / 1000, far_ip) > 2.0  # the whole record's


def test_a_summary_line_says_what_was_measured():
    t, ip = _trace()
    assert summarise(7, t, ip) == {
        "shot": 7,
        "version": 3,
        "n": t.size,
        "dt_ms": 0.5,
        "status": "ok",
        "window_start_ms": 26,
        "window_end_ms": 2474,
        "flattop_s": pytest.approx(1.6995),
        "ip_peak_ma": 1.0,
    }
    assert summarise(8, t, ip / 100)["status"] == "no_plasma"


def test_the_spacing_survives_the_raw_caches_float32_seconds():
    t = np.arange(-4000.0, 20_000.0, 0.05)  # 0.05 ms, as from about shot 189,000
    cached = (t / 1000).astype("float32").astype(float) * 1000
    assert np.median(np.diff(cached)) != pytest.approx(0.05, abs=1e-6)
    assert summarise(9, cached, np.full_like(cached, 1e6))["dt_ms"] == 0.05


def _fake_ip(calls):
    t, ip = _trace()

    def raw_signal(shot, group, *, paths):
        assert group == "ip"
        calls.append(shot)
        if shot == 3:
            raise NoDataError("fdp could not answer")
        scale = 0.01 if shot == 2 else 1.0
        return FeatureArray(x=t, y=(ip * scale)[None], attrs={"tier": "fetch"})

    return raw_signal


def test_a_rerun_measures_only_what_is_not_settled(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(window, "raw_signal", _fake_ip(calls))
    log, paths = tmp_path / "catalog" / "ip.jsonl", Paths(root=tmp_path)
    assert fetch([3, 1, 2], log, paths) == {"ok": 1, "no_plasma": 1, "error": 1}
    assert fetch([1, 2, 3], log, paths) == {"error": 1}
    assert calls == [1, 2, 3, 3]
    table = read_log(log)
    assert table[["shot", "status"]].values.tolist() == [
        [1, "ok"],
        [2, "no_plasma"],
        [3, "error"],
    ]
    assert table.loc[0, ["window_start_ms", "window_end_ms"]].tolist() == [26, 2474]
    assert table.loc[2, "error"] == "fdp could not answer"
    assert len(log.read_text().splitlines()) == 4


def test_an_unexpected_failure_is_logged_and_the_run_goes_on(tmp_path, monkeypatch):
    fake = _fake_ip([])

    def raw_signal(shot, group, *, paths):
        if shot == 5:
            raise OSError("No space left on device")
        return fake(shot, group, paths=paths)

    monkeypatch.setattr(window, "raw_signal", raw_signal)
    log = tmp_path / "ip.jsonl"
    assert fetch([5, 1], log, Paths(root=tmp_path)) == {"ok": 1, "error": 1}
    assert read_log(log).loc[1, "error"] == "OSError: No space left on device"


def test_workers_measure_the_same(tmp_path, monkeypatch):
    monkeypatch.setattr(window, "raw_signal", _fake_ip([]))
    logs = []
    for workers in (1, 2):
        log = tmp_path / f"ip-{workers}.jsonl"
        counts = fetch([1, 2, 3, 4], log, Paths(root=tmp_path), workers=workers)
        assert counts == {"ok": 2, "no_plasma": 1, "error": 1}
        assert read_log(log)["status"].tolist() == ["ok", "no_plasma", "error", "ok"]
        lines = [json.loads(text) for text in log.read_text().splitlines()]
        for line in lines:
            del line["written_at"]
            del line["run"]  # each fetch invocation owns a different run
        logs.append(sorted(lines, key=lambda line: line["shot"]))
    assert logs[0] == logs[1]


def test_an_empty_log_is_an_empty_table(tmp_path):
    table = read_log(tmp_path / "none.jsonl")
    assert table.empty and list(table.columns) == list(window.LOG_COLUMNS)


def test_the_command_logs_under_the_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(window, "raw_signal", _fake_ip([]))
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    shots = tmp_path / "shots.txt"
    shots.write_text("2\n1  # a comment\n\n3\n")
    assert window.main(["--shot-file", str(shots), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{\n") :])  # after the progress line
    assert summary["log"] == str(tmp_path / "catalog" / "ip.jsonl")
    assert summary["status"] == {"ok": 1, "no_plasma": 1, "error": 1}
    assert summary["flattop_at_least_1s"] == 1
    assert summary["versions"] == {"3": 3}
    meta = json.loads((tmp_path / "catalog" / "ip.meta.json").read_text())
    assert meta["version"] == 3
    assert re.fullmatch(r"[0-9a-f]{64}", meta["shot_file_sha256"])
    assert meta["shot_file"] == str(shots) and meta["shots"] == 3
    assert meta["this_run"] == {"ok": 1, "no_plasma": 1, "error": 1}
    assert meta["log"] == summary["log"]
    assert re.fullmatch(r"[0-9a-f]{40}", meta["git_sha"]) and meta["written_at"]
    definition = meta["definition"]
    assert definition == window.definition()
    assert definition["log_version"] == 3
    assert definition["ip"] == "PTDATA ip, native rate"
    assert definition["window_ip_a"] == 50e3
    assert definition["bridge_ms"] == 10.0
    assert definition["flattop_mean_ms"] == 25.0
    assert definition["flattop_fraction"] == 0.8
    assert definition["min_flattop_s"] == 1.0


@pytest.mark.live
def test_live_window_of_a_measured_shot(tmp_path):
    line = window.measure(200111, Paths(root=tmp_path))
    assert line["dt_ms"] == 0.05 and line["n"] == 480_256
    assert (line["window_start_ms"], line["window_end_ms"]) == (7, 5348)
    assert line["flattop_s"] == pytest.approx(3.534, abs=0.001)


@pytest.mark.live
def test_live_window_of_a_restrike(tmp_path):
    line = window.measure(204238, Paths(root=tmp_path))
    assert (line["window_start_ms"], line["window_end_ms"]) == (11, 3983)
    assert line["flattop_s"] == pytest.approx(2.903, abs=0.001)
    assert line["ip_peak_ma"] == pytest.approx(0.7356, abs=0.0001)


@pytest.mark.live
def test_live_window_keeps_an_early_dip(tmp_path):
    """200811 is a runaway beam after a triggered disruption; rule 5 drops it."""
    line = window.measure(200811, Paths(root=tmp_path))
    assert (line["window_start_ms"], line["window_end_ms"]) == (8, 2097)
    assert line["flattop_s"] == pytest.approx(1.2486, abs=0.001)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_bad_cached_record_does_not_stop_other_shots(tmp_path, workers):
    paths = Paths(root=tmp_path, corpus=tmp_path / "absent-corpus")
    t = np.arange(0.0, 2000.0, 0.05)
    ip = np.full_like(t, 1e6)
    bad_t = t.copy()
    bad_t[-1] = np.nan
    write_group(paths.raw_cache / "1_processed.h5", "ip", bad_t, ip[None])
    write_group(paths.raw_cache / "2_processed.h5", "ip", t, ip[None])
    log = tmp_path / "ip.jsonl"
    assert fetch([1, 2], log, paths, workers=workers) == {"error": 1, "ok": 1}
    assert read_log(log)["status"].tolist() == ["error", "ok"]
    assert fetch([1, 2], log, paths, workers=workers) == {"error": 1}
    assert len(log.read_text().splitlines()) == 3


@pytest.mark.parametrize(
    ("t", "ip"),
    [
        ([0, 1], [1e6]),
        ([0], [1e6]),
        ([0, np.nan], [1e6, 1e6]),
        ([0, 0], [1e6, 1e6]),
        ([0, 1], [np.nan, np.nan]),
    ],
    ids=["length", "too_short", "nonfinite_time", "unordered", "no_finite_ip"],
)
def test_summarise_refuses_unmeasurable_records(t, ip):
    with pytest.raises(ValueError):
        summarise(1, t, ip)


def test_all_nan_current_is_a_retryable_error(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "absent-corpus")
    t, _ = _trace()
    write_group(
        paths.raw_cache / "1_processed.h5", "ip", t, np.full((1, t.size), np.nan)
    )
    line = window.measure(1, paths)
    assert line["status"] == "error" and line["version"] == 3
    assert line["error"].startswith("ValueError:")


def test_a_missing_tier_is_also_a_per_shot_error(tmp_path, monkeypatch):
    t, ip = _trace()
    monkeypatch.setattr(
        window,
        "raw_signal",
        lambda *args, **kwargs: FeatureArray(x=t, y=ip[None], attrs={}),
    )
    line = window.measure(1, Paths(root=tmp_path))
    assert line["status"] == "error" and "KeyError" in line["error"]


def test_logging_keeps_full_precision_at_the_one_second_cut():
    t = np.arange(-1000.0, 3000.0, 0.05) + 0.01
    end = 1299.97
    ip = 1e6 * np.interp(t, [0, 500, end, end + 500], [0, 1, 1, 0])
    bounds = assessed_window(t, ip)
    assert bounds == (26, 1774)
    measured = flattop_s(t, ip, bounds)
    assert measured == pytest.approx(0.99995)
    logged = summarise(1, t, ip)["flattop_s"]
    assert logged == measured
    assert logged < 1.0


def _ok_line():
    return {
        "shot": 1,
        "status": "ok",
        "window_start_ms": 0,
        "window_end_ms": 2000,
        "flattop_s": 1.5,
        "ip_peak_ma": 1.0,
        "dt_ms": 0.05,
        "version": 3,
        "ip_sha256": "a" * 64,
        "run": "b" * 32,
    }


def _legacy_run(log):
    log.with_name("ip_runs.jsonl").write_text(
        json.dumps({"run": "b" * 32, "this_run": {"ok": 1}}) + "\n"
    )


def test_a_torn_tail_is_set_aside_before_resuming(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(window, "raw_signal", _fake_ip(calls))
    log = tmp_path / "ip.jsonl"
    fragment = '{"shot": 2, "status":'
    complete = json.dumps(_ok_line()) + "\n"
    log.write_text(complete + fragment)
    _legacy_run(log)
    assert read_log(log)["shot"].tolist() == [1]
    assert fetch([1, 2], log, Paths(root=tmp_path)) == {"no_plasma": 1}
    torn = tmp_path / "ip.jsonl.torn"
    assert torn.read_text() == fragment + "\n"
    assert str(torn) in capsys.readouterr().out
    assert calls == [2]
    assert [json.loads(t)["shot"] for t in log.read_text().splitlines()] == [1, 2]
    before = log.read_bytes()
    assert fetch([1, 2], log, Paths(root=tmp_path)) == {}
    assert log.read_bytes() == before and calls == [2]
    assert torn.read_text() == fragment + "\n"


def test_only_newline_terminated_nonblank_lines_are_read(tmp_path):
    log = tmp_path / "ip.jsonl"
    first = json.dumps(_ok_line())
    log.write_text("\n  \n" + first + "\n" + json.dumps(_ok_line() | {"shot": 2}))
    _legacy_run(log)
    assert read_log(log)["shot"].tolist() == [1]


@pytest.mark.parametrize(
    "bad",
    [
        "{broken json}",
        {},
        {"shot": True, "status": "error"},
        {"shot": 1.0, "status": "error"},
        {"shot": 1, "status": "pending"},
        _ok_line() | {"window_start_ms": 2000},
        _ok_line() | {"flattop_s": 2.1},
        _ok_line() | {"ip_peak_ma": float("inf")},
        _ok_line() | {"version": True},
        _ok_line() | {"version": 0},
        _ok_line() | {"version": 1.5},
        _ok_line() | {"window_start_ms": None},
        _ok_line() | {"window_end_ms": 2000.0},
        _ok_line() | {"window_start_ms": False},
        _ok_line() | {"flattop_s": -0.1},
        _ok_line() | {"flattop_s": float("nan")},
        _ok_line() | {"dt_ms": 0},
        _ok_line() | {"dt_ms": float("inf")},
    ],
    ids=[
        "json",
        "no_shot",
        "bool_shot",
        "float_shot",
        "status",
        "window_order",
        "long_flattop",
        "peak",
        "bool_version",
        "zero_version",
        "float_version",
        "null_window",
        "float_window",
        "bool_window",
        "negative_flattop",
        "nan_flattop",
        "zero_dt",
        "infinite_dt",
    ],
)
def test_bad_complete_log_lines_name_the_file_and_line(tmp_path, bad):
    log = tmp_path / "ip.jsonl"
    text = bad if isinstance(bad, str) else json.dumps(bad)
    log.write_text(json.dumps(_ok_line()) + "\n" + text + "\n")
    with pytest.raises(CatalogError, match=r"ip\.jsonl:2:"):
        read_log(log)


def test_old_log_versions_are_remeasured_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "raw_signal", _fake_ip(calls))
    log = tmp_path / "ip.jsonl"
    old = _ok_line()
    del old["version"]
    log.write_text(json.dumps(old) + "\n")
    table = read_log(log)
    assert table["version"].tolist() == [1]
    assert str(table["version"].dtype) == "Int64"
    assert fetch([1], log, Paths(root=tmp_path)) == {"ok": 1}
    assert read_log(log)["version"].tolist() == [3]
    assert len(log.read_text().splitlines()) == 2
    assert fetch([1], log, Paths(root=tmp_path)) == {}
    assert calls == [1]


@pytest.mark.parametrize("module", ["toksearch", "toksearch_d3d"])
def test_toksearch_in_the_parent_refuses_a_worker_pool(tmp_path, monkeypatch, module):
    calls = []
    monkeypatch.setattr(window, "raw_signal", _fake_ip(calls))
    monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    log = tmp_path / "ip.jsonl"
    with pytest.raises(RuntimeError, match="toksearch"):
        fetch([1], log, Paths(root=tmp_path), workers=2)
    assert not log.read_text() and not calls
    assert fetch([1], log, Paths(root=tmp_path), workers=1) == {"ok": 1}
    assert calls == [1]


def test_a_gap_of_exactly_ten_ms_is_bridged():
    t = np.arange(0.0, 3000.0, 0.5)
    clean = 1e6 * np.minimum(np.clip(t / 500, 0, 1), np.clip((2500 - t) / 500, 0, 1))
    for end, want in [(110.0, (25, 2475)), (110.5, (111, 2475))]:
        ip = clean.copy()
        ip[(t > 100) & (t < end)] = 0
        assert assessed_window(t, ip) == want


def test_import_does_not_load_torch_or_toksearch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import labeler.events.catalog.window; "
                "assert not {'torch', 'toksearch', 'toksearch_d3d'} "
                "& sys.modules.keys()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_window_catalog_error_is_a_usage_error(tmp_path, monkeypatch, capsys):
    from labeler.events.catalog.check import CatalogError

    shots = tmp_path / "shots.txt"
    shots.write_text("190001\n")

    def broken(*args, **kwargs):
        raise CatalogError("corrupt log fixture")

    monkeypatch.setattr(window, "fetch", broken)
    with pytest.raises(SystemExit) as exc:
        window.main(["--shot-file", str(shots), "--log", str(tmp_path / "ip.jsonl")])
    assert exc.value.code == 2
    assert "corrupt log fixture" in capsys.readouterr().err


def _restrike_trace(dt, *, dip=80e3, rise=800e3, restrike=True):
    t = np.arange(0, 4000 + dt / 2, dt)
    knots = [0, 500, 3500, 3620, 3760, 3860, 3872, 4000]
    levels = [0, 700e3, 700e3, dip, rise, rise, 0, 0]
    if not restrike:
        knots, levels = [0, 500, 3500, 3620, 3632, 4000], [0, 700e3, 700e3, dip, 0, 0]
    return t, np.interp(t, knots, levels)


@pytest.mark.parametrize("dt", [0.05, 0.5])
def test_d2b_ends_at_post_flattop_dip_and_remeasures_peak(dt):
    t, ip = _restrike_trace(dt)
    original = window.assessed_window(t, ip)
    end = window.restrike_end(t, ip, original)
    assert abs(end - 3620) <= 1
    line = summarise(204238, t, ip)
    assert line["window_end_ms"] == end < original[1]
    assert line["ip_peak_ma"] == 0.7
    t0, ip0 = _restrike_trace(dt, restrike=False)
    baseline = summarise(204238, t0, ip0)
    assert line["flattop_s"] == pytest.approx(baseline["flattop_s"], abs=1e-12)


@pytest.mark.parametrize("dt", [0.05, 0.5])
def test_d2b_keeps_the_early_discharge_dip(dt):
    """200811's runaway beam stays in the Ip window, then rule 5 excludes it."""
    t = np.arange(0, 2400 + dt / 2, dt)
    ip = np.interp(
        t, [0, 400, 470, 900, 2100, 2300, 2400], [0, 600e3, 170e3, 600e3, 600e3, 0, 0]
    )
    original = window.assessed_window(t, ip)
    assert window.restrike_end(t, ip, original) is None
    line = summarise(200811, t, ip)
    assert (line["window_start_ms"], line["window_end_ms"]) == original


@pytest.mark.parametrize("dt", [0.05, 0.5])
@pytest.mark.parametrize("dip, rise", [(0.25, 0.5), (0.35, 0.9)])
def test_d2b_requires_both_dip_and_rise_thresholds(dt, dip, rise):
    t, ip = _restrike_trace(dt, dip=dip * 700e3, rise=rise * 700e3)
    original = window.assessed_window(t, ip)
    assert window.restrike_end(t, ip, original) is None
    assert summarise(1, t, ip)["window_end_ms"] == original[1]


def test_d2b_sampling_rates_agree_within_one_ms():
    ends = [summarise(1, *_restrike_trace(dt))["window_end_ms"] for dt in [0.05, 0.5]]
    assert abs(ends[0] - ends[1]) <= 1
    assert max(ends) <= 3620


def test_v2_log_is_remeasured_at_version_3(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "raw_signal", _fake_ip(calls))
    log = tmp_path / "ip.jsonl"
    log.write_text(json.dumps(_ok_line() | {"version": 2}) + "\n")
    assert window.LOG_VERSION == 3
    assert fetch([1], log, Paths(root=tmp_path)) == {"ok": 1}
    assert calls == [1]
    assert read_log(log).version.tolist() == [3]
    assert fetch([1], log, Paths(root=tmp_path)) == {}
    definition = window.definition()
    assert definition["restrike_dip_fraction"] == 0.3
    assert definition["restrike_rise_fraction"] == 0.6
    assert "restrike" in definition


def test_shot_file_hash_is_of_the_bytes_measured(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    shots = tmp_path / "shots.txt"
    original = b"2\n1 # comment\n"
    shots.write_bytes(original)
    calls = []
    raw = _fake_ip(calls)

    def replace_after_read(*args, **kwargs):
        shots.write_text("99\n")
        return raw(*args, **kwargs)

    monkeypatch.setattr(window, "raw_signal", replace_after_read)
    assert window.main(["--shot-file", str(shots), "--workers", "1"]) == 0
    meta = json.loads((tmp_path / "catalog" / "ip.meta.json").read_text())
    assert meta["shot_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert calls == [1, 2] and meta["shots"] == 2


@pytest.mark.parametrize("dt", [0.05, 0.5])
@pytest.mark.parametrize("sign", [1, -1])
def test_d2d_opposite_single_sample_glitch_keeps_the_plasma(dt, sign):
    t = np.arange(0, 4500, dt)
    clean = sign * np.interp(t, [0, 500, 3500, 4000], [0, 1e6, 1e6, 0])
    ip = clean.copy()
    ip[np.argmin(abs(t - 2000))] = -sign * 2e6
    assert assessed_window(t, ip) == (25, 3975)
    baseline = summarise(1, t, clean)
    measured = summarise(1, t, ip)
    for key in ("window_start_ms", "window_end_ms", "flattop_s", "ip_peak_ma"):
        assert measured[key] == baseline[key], key


@pytest.mark.parametrize("dt", [0.05, 0.5])
def test_d2d_long_opposite_pickup_does_not_choose_the_sign(dt):
    t = np.arange(0, 23000, dt)
    ip = np.interp(t, [0, 500, 1500, 2000, 2500, 22500], [0, 1e6, 1e6, 0, -60e3, -60e3])
    assert assessed_window(t, ip) == (25, 1975)


def test_d2d_short_record_falls_back_to_largest_finite_sample():
    assert assessed_window([0, 1, 2, 3], [np.nan, -60e3, -70e3, 0]) == (1, 2)


@pytest.mark.parametrize("bad", [False, True])
def test_measured_waveform_hash_is_little_endian_time_then_current(tmp_path, bad):
    paths = Paths(root=tmp_path, corpus=tmp_path / "absent")
    t, ip = _trace()
    if bad:
        t[-1] = t[-2]  # read successfully, but summarise refuses duplicate times
    write_group(paths.raw_cache / "1_processed.h5", "ip", t, ip[None])
    record = window.raw_signal(1, "ip", paths=paths)
    expected = hashlib.sha256(
        b"".join(
            struct.pack("<d", float(value))
            for values in (record.x, record.y[0])
            for value in values
        )
    ).hexdigest()
    line = window.measure(1, paths)
    assert line["status"] == ("error" if bad else "ok")
    assert line["ip_sha256"] == expected


def test_unread_waveform_error_has_no_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(window, "raw_signal", _fake_ip([]))
    line = window.measure(3, Paths(root=tmp_path))
    assert line["status"] == "error" and line["ip_sha256"] is None


def test_runs_are_append_only_and_lines_keep_their_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    monkeypatch.setattr(window, "raw_signal", _fake_ip([]))
    shots = tmp_path / "shots.txt"
    args = ["--shot-file", str(shots), "--workers", "1"]
    log = tmp_path / "catalog" / "ip.jsonl"
    runs = log.with_name("ip_runs.jsonl")
    shots.write_text("1\n")
    assert window.main(args) == 0
    first_lines = log.read_bytes()
    assert runs.is_file()
    first_run = runs.read_bytes()
    shots.write_text("1\n2\n")
    assert window.main(args) == 0
    assert runs.read_bytes().startswith(first_run)
    assert log.read_bytes().startswith(first_lines)
    lines_before = log.read_bytes()
    records_before = runs.read_bytes()
    assert window.main(args) == 0
    assert log.read_bytes() == lines_before
    assert runs.read_bytes().startswith(records_before)
    records = [json.loads(row) for row in runs.read_bytes().splitlines()]
    assert [record["type"] for record in records] == ["start", "complete"] * 3
    records = records[1::2]
    ids = [record["run"] for record in records]
    assert len(set(ids)) == 3
    assert all(re.fullmatch(r"[0-9a-f]{32}", run) for run in ids)
    assert [record["this_run"] for record in records] == [
        {"ok": 1},
        {"no_plasma": 1},
        {},
    ]
    lines = [json.loads(row) for row in lines_before.splitlines()]
    assert [line["run"] for line in lines] == ids[:2]
    assert read_log(log)["run"].tolist() == ids[:2]
    assert read_log(log)["ip_sha256"].tolist() == [line["ip_sha256"] for line in lines]
    for record, shot_bytes in zip(records, [b"1\n", b"1\n2\n", b"1\n2\n"]):
        assert record["git_sha"] == git_sha(full=True)
        assert record["git_dirty"] == git_dirty()
        assert record["written_at"] and record["definition"] == window.definition()
        assert record["log"] == str(log) and record["shot_file"] == str(shots)
        assert record["shots"] == len(shot_bytes.splitlines())
        assert record["shot_file_sha256"] == hashlib.sha256(shot_bytes).hexdigest()
        assert shlex.split(record["command"]) == [
            "python",
            "-m",
            "labeler.events.catalog.window",
            *args,
        ]
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["run"] == ids[-1] and meta["git_dirty"] == git_dirty()


@pytest.mark.parametrize("field", ["ip_sha256", "run"])
@pytest.mark.parametrize("value", [None, "", "f", "G" * 64, 123])
def test_v3_ok_line_requires_valid_fingerprints(tmp_path, field, value):
    line = _ok_line() | {"ip_sha256": "a" * 64, "run": "b" * 32}
    if value is None:
        del line[field]
    else:
        line[field] = value
    log = tmp_path / "ip.jsonl"
    log.write_text(json.dumps(line) + "\n")
    with pytest.raises(CatalogError, match=field):
        read_log(log)


@pytest.mark.parametrize("status", ["ok", "no_plasma", "error"])
def test_optional_fingerprints_are_kept_on_any_version(tmp_path, status):
    line = _ok_line() | {
        "status": status,
        "version": 2,
        "ip_sha256": "a" * 64,
        "run": "b" * 32,
    }
    log = tmp_path / "ip.jsonl"
    log.write_text(json.dumps(line) + "\n")
    table = read_log(log)
    assert table.loc[0, "ip_sha256"] == "a" * 64
    assert table.loc[0, "run"] == "b" * 32


def test_interruption_keeps_the_start_record_and_resume_completes(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    shots = tmp_path / "shots.txt"
    shots.write_text("1\n2\n")
    args = ["--shot-file", str(shots), "--workers", "1"]
    raw = _fake_ip([])

    def interrupted(shot, *args, **kwargs):
        if shot == 2:
            raise KeyboardInterrupt
        return raw(shot, *args, **kwargs)

    monkeypatch.setattr(window, "raw_signal", interrupted)
    with pytest.raises(KeyboardInterrupt):
        window.main(args)
    log = tmp_path / "catalog" / "ip.jsonl"
    line = json.loads(log.read_text())
    runs = log.with_name("ip_runs.jsonl")
    assert runs.is_file()
    start = json.loads(runs.read_text())
    assert start["type"] == "start" and start["run"] == line["run"]
    assert start["git_sha"] == git_sha(full=True)
    assert start["shot_file_sha256"] == hashlib.sha256(b"1\n2\n").hexdigest()
    assert read_log(log).shot.tolist() == [1]
    monkeypatch.setattr(window, "raw_signal", raw)
    assert window.main(args) == 0
    records = [json.loads(row) for row in runs.read_text().splitlines()]
    assert [row["type"] for row in records] == ["start", "start", "complete"]
    assert records[-1]["run"] == records[-2]["run"]
    assert records[-1]["this_run"] == {"no_plasma": 1}
    assert read_log(log).shot.tolist() == [1, 2]


@pytest.mark.parametrize("record_type", ["absent", "complete"])
def test_unknown_run_is_refused_by_name(tmp_path, record_type):
    log = tmp_path / "ip.jsonl"
    log.write_text(json.dumps(_ok_line()) + "\n")
    if record_type == "complete":
        log.with_name("ip_runs.jsonl").write_text(
            json.dumps({"type": "complete", "run": "b" * 32}) + "\n"
        )
    with pytest.raises(CatalogError, match=r"ip\.jsonl.*run"):
        read_log(log)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("field", ["ip_sha256", "run"])
def test_no_plasma_requires_fingerprints_from_v3(tmp_path, version, field):
    line = _ok_line() | {"status": "no_plasma", "version": version}
    del line[field]
    log = tmp_path / "ip.jsonl"
    log.write_text(json.dumps(line) + "\n")
    with pytest.raises(CatalogError, match=field):
        read_log(log)


def test_legacy_whole_run_record_is_still_read(tmp_path):
    log = tmp_path / "ip.jsonl"
    log.write_text(json.dumps(_ok_line()) + "\n")
    log.with_name("ip_runs.jsonl").write_text(
        json.dumps({"run": "b" * 32, "this_run": {"ok": 1}}) + "\n"
    )
    assert read_log(log).shot.tolist() == [1]


def test_unnamed_log_stream_refuses_when_runs_cannot_be_located():
    source = io.BytesIO((json.dumps(_ok_line()) + "\n").encode())
    with pytest.raises(CatalogError, match="runs file cannot be located"):
        read_log(source)


def test_unnamed_log_stream_accepts_supplied_runs_bytes():
    source = io.BytesIO((json.dumps(_ok_line()) + "\n").encode())
    runs_data = (json.dumps({"type": "start", "run": "b" * 32}) + "\n").encode()
    assert read_log(source, runs_data=runs_data).shot.tolist() == [1]
